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

# define collate function for the dataloader
def asr_collate_fn(batch):

	ids = [example[0] for example in batch]

	input_values = tokenizer([example[1] for example in batch],
							return_tensors = "pt",
							padding = "longest").input_values

	tgt_text = [example[2] for example in batch]

	return ids, input_values, tgt_text


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--tsv_file_path", type = str, default = "C:\\Users\\ioann\\Documents\\datasets\\MUSTC_v1.0_prep_wav\\en-de\\data\\dev_asr_ioannis.tsv",
			help = "Path to the tsv file")
parser.add_argument("--batch_size", type = int, default = 2,
			help = "batch size to be used during inference")
parser.add_argument("--num_workers", type = int, default = 0,
			help = "number of cpu workers for the dataloader")
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
dataset = AsrDataset(args.tsv_file_path)
dataloader = DataLoader(dataset,
						args.batch_size,
						collate_fn = asr_collate_fn,
						shuffle = False,
						num_workers = args.num_workers,
						drop_last = False)
print("Loaded dataset and intialized dataloader")

# to store results
wer_path = "wer_results.json"

print(f"Starting inference, results file: {wer_path} is updated every batch")
start_time = time()

wer_sum, n_examples = 0, 0

# loop through the dataset
with torch.no_grad():
	for batch in tqdm(iter(dataloader), miniters = 1000):

		# fetch data from batch
		ids = batch[0]
		input_values = batch[1]
		tgt_text = batch[2]

		# retrieve logits
		logits = model(input_values).logits

		# take argmax and decode
		predicted_ids = torch.argmax(logits, dim = -1)
		transcriptions = tokenizer.batch_decode(predicted_ids)

		# record wer per datapoint
		wer_results = [wer(tgt_text[i], transcriptions[i]) for i in range(len(ids))]

		# store results for this batch
		with open(wer_path, "a") as f:
			for i in range(len(ids)):
				json.dump({"id": ids[i],
							"WER": wer_results[i],
							"target text": tgt_text[i],
							"transcription": transcriptions[i]}, f)
				f.write("\n")

		# for global score
		wer_sum += sum(wer_results)
		n_examples += len(wer_results)

n_data = len(dataset)
total_time = time() - start_time

print(f"Finished inference for {n_data} datapoints in {total_time / 60} minutes")
print(f"Average processing time per datapoint = {total_time / n_data} seconds")
print(f"Macro-averaged WER = {wer_sum / n_examples}")