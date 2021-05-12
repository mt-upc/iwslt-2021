from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
from torch.utils.data import Dataset, DataLoader
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import List
from multiprocessing import cpu_count
import json
import argparse

MODEL_NAME = "facebook/wav2vec2-large-960h-lv60-self"
MS = 0.02  # tokens per seconds output of wav2vec 2.0


def my_collate_fn(batch: List[np.array]) -> List[np.array]:
    return [example for example in batch]


class TokenPredDataset(Dataset):
    def __init__(self, path_to_wav: Path, extra_step: float,
    loading_step: float) -> None:
        super(TokenPredDataset, self).__init__()
        
        self.extra_step = extra_step
        self.loading_step = loading_step
        self.ms = MS

        # load the whole wav file
        self.wav_array, self.sr = \
            torchaudio.backend._soundfile_backend.load(path_to_wav)
        self.wav_array = self.wav_array[0].numpy()

        self.total_duration = len(self.wav_array) // self.sr
        offset = 0
        self.start, self.end = [], []

        while offset < self.total_duration:
            if offset == 0:
                self.start.append(offset)
            else:
                # load extra_step before for more accuracy
                self.start.append(offset - self.extra_step)
            # also extra step after
            self.end.append(offset + self.loading_step + self.extra_step + self.ms)

            offset += self.loading_step

    def __len__(self) -> int:
        return len(self.start)

    def __getitem__(self, i: int) -> np.array:
        s = round(self.start[i] * self.sr)
        e = round(self.end[i] * self.sr)

        if i == len(self) - 1:
            part = self.wav_array[s:]
        else:
            part = self.wav_array[s: e]

        return part


def get_preds_for_wav(model: Wav2Vec2ForCTC, tokenizer: Wav2Vec2Tokenizer,
device: torch.device, bs: int, path_to_wav: Path, extra_step: float,
loading_step: float) -> List[int]:

    dataset = TokenPredDataset(path_to_wav, extra_step, loading_step)
    dataloader = DataLoader(dataset, batch_size = bs, shuffle = False,
        collate_fn = my_collate_fn, num_workers = min(cpu_count() // 2, 6),
        drop_last = False)

    # for the extra frames loaded before and after each segment
    correction = int(extra_step / MS)

    all_preds = []
    i = 0
    with torch.no_grad():
        for wav_batch in iter(dataloader):

            tokenized_audio = tokenizer(wav_batch,
                return_tensors = "pt", padding = "longest")
            input_values = tokenized_audio.input_values.to(device)
            attention_mask = tokenized_audio.attention_mask.to(device)
            logits = model(input_values, attention_mask = attention_mask).logits
           
            predicted_ids = torch.argmax(logits, dim = -1)

            for j, preds in enumerate(predicted_ids.tolist()):
                true_length = attention_mask[j].cpu().numpy().sum() / dataset.sr / MS

                # apply corrections
                if i == 0:
                    preds = preds[:-correction]
                    true_length -= correction
                elif i == len(dataset) - 1:
                    preds = preds[correction:]
                    true_length -= correction
                else:
                    preds = preds[correction:-correction]
                    true_length -= 2 * correction

                # remove padding
                all_preds.extend(preds[:int(true_length)])

                i += 1

    return all_preds


def get_predictions(test_dir_root: str, bs: int, extra_step: float, loading_step: float) -> None:

    device = torch.device("cuda:0") if torch.cuda.is_available() \
        else torch.device("cpu")

    # load model and tokenizer
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).eval().to(device)
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(MODEL_NAME)

    test_dir_root = Path(test_dir_root)

    # iterate over the files in the correct order
    with open(test_dir_root / "FILE_ORDER", "r") as f:
        wav_file_order = f.read().splitlines()

    token_predictions = {}
    for wf in wav_file_order:
        wf = f"{wf}.wav"
        print(f"Generating token predictions for {wf}")
        path_to_wav = test_dir_root / "wavs" / wf
        token_predictions[wf] = get_preds_for_wav(model, tokenizer, device, bs,
            path_to_wav, extra_step, loading_step)

    test_dir_root.mkdir(parents = True, exist_ok = True)
    path_to_preds = test_dir_root / "token_predictions.json"
    with open(path_to_preds, "w") as f:
        json.dump(token_predictions, f)

    print(f"Wav2Vec predictions saved at {path_to_preds}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir_root", type = str, required = True,
                help = "path to the directory of the test set")
    parser.add_argument("--batch_size", type = int, default = 16,
                help = "batch size to be used during inference")
    parser.add_argument("--extra_step", type = float, default = 1,
                help = "how many extra seconds to load before each part")
    parser.add_argument("--loading_step", type = float, default = 10,
                help = "the duration of each part in seconds")
    args = parser.parse_args()

    get_predictions(args.test_dir_root, args.batch_size, args.extra_step, args.loading_step)