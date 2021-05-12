from transformers import Wav2Vec2Tokenizer
import numpy as np
import json
import re
from typing import List, Union
import yaml
import argparse
from pathlib import Path


MODEL_NAME = "facebook/wav2vec2-large-960h-lv60-self"
MS = 0.02  # tokens per seconds output of wav2vec 2.0
EXTRA_MS = 3  # this number of MS are expanded on the segment before and after


def flatten(x: Union[List, str]) -> List[str]:
    if isinstance(x, list):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


def add_delim(x: List[str], delim: str) -> List[str]:
    x_new = [delim] * (len(x) * 2 - 1)
    x_new[0::2] = x
    return x_new


def is_pause(x: str) -> bool:
    set_x = set(x)
    if set_x == set(" "):
        return True
    elif set_x == set("|"):
        return True
    elif set_x == set(" |"):
        return True
    return False


def split_text_to_segments(tokens: str, max_segm_len: int, min_pause_len: int) -> List[str]:

    segments = [tokens]
    cond = [True]
    n_segm, n_segm_prev = 1, 0

    # second condition to account for the case where segments cannot be further
    # split due to min_pause_len
    while any(cond) and n_segm != n_segm_prev:
        
        for i, current_segm in enumerate(segments):
            
            if cond[i]:
                
                # find pause patterns: combination of spaces and char split symbol (|)
                pauses = re.findall(r"[\s\|]{1,}", current_segm)

                # get max pause
                pauses.sort(key = lambda s: len(s))
                max_pause = pauses[-1]
                
                # avoid splitting on small pauses
                if len(max_pause) < min_pause_len:
                    continue
                
                # split segment and add the pause between
                segments[i] = add_delim(current_segm.split(max_pause), max_pause)

        segments = flatten(segments)
        
        # longer than allowed and not a pause
        cond = [len(segm) > max_segm_len and not is_pause(segm)
            for segm in segments]

        n_segm_prev = n_segm
        n_segm = len(segments)
        
    # delete first and last empty segments
    del segments[0]
    del segments[-1]

    return segments


def segment_dataset(max_segm_len_secs: int, min_pause_len: float, dataset_root: Path,
tokenizer: Wav2Vec2Tokenizer, path_to_original_segmentation: Path) -> None:

    # convert secs to wav2vec prediciton steps
    max_segm_len_steps = int(max_segm_len_secs / MS)
    min_pause_len = int(min_pause_len / MS)

    # load wav2vec token predicitons for all the wavs in the dataset
    with open(dataset_root / "token_predictions.json", "r") as f:
        predictions = json.load(f)

    segm_data = []

    print(f"Segmenting with max segmentation length = {max_segm_len_secs} secs.")

    for wav_file, preds in predictions.items():

        tokens = tokenizer.convert_ids_to_tokens(preds)

        # convert to a long string and replace the pad token with a space (easier to handle)
        tokens_text = "".join(tokens).replace("<pad>", " ")

        # apply segmentation algorithm
        segments = split_text_to_segments(tokens_text, max_segm_len_steps, min_pause_len)

        # duration and offset of each segment in seconds
        duration = np.array([len(segm) for segm in segments]) * MS
        duration_cumsum = np.insert(np.cumsum(duration)[1:], 0, 0)
        offset = duration_cumsum - duration
        offset[0] = 0

        # delete segments that do not contain words
        num_pauses = 0
        for i, segm in reversed(list(enumerate(segments))):
            if is_pause(segm):
                del segments[i]
                offset = np.delete(offset, i)
                duration = np.delete(duration, i)
                num_pauses += 1

        # expand each segment by EXTRA_MS steps before and after
        offset = list(offset - EXTRA_MS * MS)
        duration = list(duration + EXTRA_MS * MS)

        path_to_wav = str(dataset_root / wav_file)

        segm_data.extend([{"wav": path_to_wav, "offset": round(float(off), 2),
                "duration": round(float(dur), 2)}
            for off, dur in zip(offset, duration)])

        print(f"{wav_file} -> {len(segments) + num_pauses} segments ({num_pauses} pauses) ")

    new_segmentation_file = Path(f"_own_{max_segm_len_secs}.".join(
        str(path_to_original_segmentation).rsplit(".", maxsplit = 1))).name
    path_to_new_segmentation = dataset_root / "own_segmentation" / new_segmentation_file
    with open(path_to_new_segmentation, "w") as f:
        yaml.dump(segm_data, f, default_flow_style = True)

    print(f"Segmentation for max_segm_len={max_segm_len_secs} saved at {path_to_new_segmentation}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type = str, required = True,
                help = "Path to dataset to be segmented")
    parser.add_argument("--max_segm_len", type = str, required = True,
                help = "Maximum segment length during segmentation in seconds.\
                Use two number separated by commas to run for each value in this range.")
    parser.add_argument("--min_pause_len", type = float, default = 0.2,
                help = "Minimum allowed pause length in seconds")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)

    tokenizer = Wav2Vec2Tokenizer.from_pretrained(MODEL_NAME)

    # find original segmentation file
    original_segmentation_file = next(dataset_root.glob("*en-de.yaml"))

    # run multiple times
    if "," in args.max_segm_len:
        (dataset_root / "own_segmentation").mkdir(parents = True, exist_ok = True)

        max_segm_len_range = range(*map(int, args.max_segm_len.split(",")))
        for max_segm_len_value in max_segm_len_range:
            segment_dataset(max_segm_len_value, args.min_pause_len, dataset_root,
                tokenizer, original_segmentation_file)
    # run once
    else:
        max_segm_len_value = int(args.max_segm_len)
        segment_dataset(max_segm_len_value, args.min_pause_len, dataset_root,
            tokenizer, original_segmentation_file)