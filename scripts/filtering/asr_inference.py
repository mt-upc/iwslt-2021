from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
import torch
from torch.utils.data import DataLoader
from torchaudio import load
from jiwer import wer
from tqdm import tqdm
import argparse
from time import time
from multiprocessing import cpu_count
import json

from asr_dataset import AsrDataset


def asr_collate_fn(batch):
	"""
	Combines the examples from the dataset

	Args:
		batch (list[list]): list of args.batch_size examples
			every example is [Id(str), wav_array(1d np.array), tgt_text(str)]
	Returns:
		ids (list[str]): unique ids
		tokenized_audio (2d torch.tensor [args.batch_size x longest]): tokenized
			input values for wav2vec
		tgt_text (list[str]): target texts
	"""

	ids = [example[0] for example in batch]

	tokenized_audio = tokenizer([example[1] for example in batch],
		return_tensors = "pt",
		padding = "longest")

	tgt_text = [example[2] for example in batch]

	return ids, tokenized_audio, tgt_text


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dir_path", type = str, default = "./../../datasets/speech_translation/MUSTC_v1.0_prep_wav/en-de/",
			help = "path of the directory containing the tsv file and its associated wav files")
parser.add_argument("--tsv_file_name", type = str, default = "dev_asr.tsv",
			help = "name of the tsv file in $dir_path")
parser.add_argument("--batch_size", type = int, default = 8,
			help = "batch size to be used during inference")
parser.add_argument("--model_name", type = str, default = "facebook/wav2vec2-large-960h-lv60-self",
			help = "name of the pre-trained model to be used")
args = parser.parse_args()

# cpu and gpu devices
n_gpu = torch.cuda.device_count()
n_cpu = cpu_count()
main_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Number of cuda devices: {n_gpu} | Number of CPU cores: {n_cpu}")

# load model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained(args.model_name)
model = Wav2Vec2ForCTC.from_pretrained(args.model_name).to(main_device)
print(f"Loaded model and tokenizer: {args.model_name}")

# load dataset and initialize dataloader
dataset = AsrDataset(args.dir_path, args.tsv_file_name)
dataloader = DataLoader(dataset,
	args.batch_size,
	collate_fn = asr_collate_fn,
	shuffle = False,
	num_workers = n_cpu // 4,
	drop_last = False)
print("Loaded dataset and intialized dataloader")

# to store results
wer_path = "wer_results.json"

# for global scores
start_time = time()
wer_sum, n_examples = 0, 0

print(f"Starting inference, results file: {wer_path} is updated every batch")

# loop through the dataset
with torch.no_grad():
	for batch in tqdm(iter(dataloader), miniters = 1000):

		# fetch data from batch
		ids = batch[0]
		input_values = batch[1].input_values.to(main_device)
		attention_mask = batch[1].attention_mask.to(main_device)
		tgt_text = batch[2]

		# retrieve logits
		logits = model(input_values, attention_mask = attention_mask).logits

		# take argmax and decode
		predicted_ids = torch.argmax(logits, dim = -1)
		transcriptions = tokenizer.batch_decode(predicted_ids)

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

n_data = len(dataset)
total_time = time() - start_time
avg_proc_time = total_time / n_data
macro_wer = wer_sum	/ n_examples

print(f"Finished inference for {n_data} datapoints in {total_time / 60} minutes")
print(f"Average processing time per datapoint = {avg_proc_time} seconds")
print(f"Macro-averaged WER = {macro_wer}")