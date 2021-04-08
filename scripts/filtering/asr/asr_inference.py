from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
import torch
from torch.utils.data import DataLoader
from jiwer import wer
from tqdm import tqdm
import argparse
from time import time
from multiprocessing import cpu_count
import json
from pathlib import Path
import sys

sys.path.append("./../fairseq/examples")
sys.path.append("scripts/filtering")

from asr.asr_dataset import AsrDataset, asr_collate_fn

MODEL_NAME = "facebook/wav2vec2-large-960h-lv60-self"


def asr_inference(tsv_path: Path, batch_size: int) -> None:

    # number of cpu and gpu devices
    n_gpu = torch.cuda.device_count()
    n_cpu = cpu_count()
    print(f"Number of cuda devices: {n_gpu} | Number of CPU cores: {n_cpu}")

    # specify main device and all devices (if gpu available)
    device_list = [torch.device(f"cuda:{i}") for i in range(n_gpu)]
    main_device = device_list[0] if n_gpu > 0 else torch.device("cpu")
    print(f"Main device: {main_device}")
    print(f"Parallel devices = {device_list}")

    # load model and tokenizer
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).eval()
    print(f"Loaded model and tokenizer: {MODEL_NAME}")

    if len(device_list) > 1:
        model = torch.nn.DataParallel(model,
            device_ids = device_list, output_device = main_device)

    model.to(main_device)

    # to store results
    wer_path = tsv_path.parent / f"{tsv_path.stem}_wer_results.json"

    # check if there are already predictions for some ids
    if wer_path.is_file():
        with open(wer_path, "r") as file:
            completed_ids = [json.loads(line)["id"] for line in file]
    else:
        completed_ids = []

    # load dataset and initialize dataloader
    dataset = AsrDataset(tsv_path, completed_ids)
    dataloader = DataLoader(dataset,
        batch_size = batch_size,
        collate_fn = asr_collate_fn,
        shuffle = False,
        num_workers = n_cpu // 8,
        drop_last = False)
    print("Loaded dataset and intialized dataloader")

    # for global scores
    start_time = time()
    wer_sum, n_examples = 0, 0

    print(f"Starting inference, results file: {wer_path} is updated every batch")

    # loop through the dataset
    with torch.no_grad():
        for (ids, audio, tgt_text) in tqdm(iter(dataloader),
        miniters = len(dataloader) // 100):

            tokenized_audio = tokenizer(audio, return_tensors = "pt", padding = "longest")
            input_values = tokenized_audio.input_values.to(main_device)
            attention_mask = tokenized_audio.attention_mask.to(main_device)

            # retrieve logits
            logits = model(input_values, attention_mask = attention_mask).logits

            # take argmax and decode
            predicted_ids = torch.argmax(logits, dim = -1)
            transcriptions = [trans.lower()
                for trans in tokenizer.batch_decode(predicted_ids)]

            # record WER per datapoint
            wer_results = [wer(tgt_text[i], transcriptions[i]) for i in range(len(ids))]

            # store results for this batch
            with open(wer_path, "a") as f:
                for i in range(len(ids)):
                    json.dump({"id": ids[i],
                        "WER": wer_results[i],
                        "target text": tgt_text[i],
                        "transcription": transcriptions[i]}, f)
                    f.write("\n")

            # for global scores
            wer_sum += sum(wer_results)
            n_examples += len(wer_results)

    if wer_sum != 0:
        n_data = len(dataset)
        total_time = time() - start_time
        avg_proc_time = total_time / n_data
        macro_wer = wer_sum / n_examples

        print(f"Finished inference for {n_data} datapoints in {total_time / 60} minutes")
        print(f"Average processing time per datapoint = {avg_proc_time} seconds")
        print(f"Macro-averaged WER = {macro_wer}")

    else:
        print("Predictions file was already completed.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv_path", type = str, required = True,
                help = "path of the tsv file")
    parser.add_argument("--batch_size", type = int, default = 24,
                help = "batch size to be used during inference")
    args = parser.parse_args()

    asr_inference(Path(args.tsv_path), args.batch_size)