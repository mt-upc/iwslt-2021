import argparse
from pathlib import Path
from typing import Callable
import sys

sys.path.append("./../fairseq/examples")
sys.path.append("scripts/filtering")

from speech_to_text.data_utils import load_df_from_tsv, save_df_to_tsv
from asr.asr_inference import asr_inference
from filtering_utils import (find_noisy_examples, clean_mustc_utterance,
    clean_europarlst_utterance, clean_covost_utterance)

DATASETS = ["MUSTC", "EuroparlST", "CoVoST"]
SPLITS = {
    "MUSTC": ["train", "dev", "tst-COMMON", "tst-HE"],
    "EuroparlST": [],
    "CoVoST": []}
CLEAN_FUNC = {
    "MUSTC": clean_mustc_utterance,
    "EuroparlST": clean_europarlst_utterance,
    "CoVoST": clean_covost_utterance}
TASKS = ["asr", "st"]


def filter(tsv_root: Path, split: str, task: str,
utterance_cleaner_func: Callable, duration_sec_threshold: int, apply_asr_filtering: bool,
asr_batch_size: int, asr_wer_threshold: float) -> None:

    tsv_path = tsv_root / f"{split}_{task}.tsv"
    df = load_df_from_tsv(tsv_path)

    # target text cleaning
    df["tgt_text"] = df.apply(lambda x: utterance_cleaner_func(x["tgt_text"]), axis = 1)

    # removal of empty examples
    empty_examples_bool = df.tgt_text == ""
    df = df.loc[~empty_examples_bool]
    print(f"removed {empty_examples_bool.sum()} empty examples. remaing: {len(df)}")

    # removal of long examples
    long_examples_bool = df.duration_ms > duration_sec_threshold * 1000
    df = df.loc[~long_examples_bool]
    print(f"removed {long_examples_bool.sum()} long examples. remaing: {len(df)}")

    # removal of noisy examples (based on ASR system predictions)
    if apply_asr_filtering and task == "st":

        print(f"Starting ASR inference with Wav2Vev_2.0 on {split} set")
        asr_tsv_file = tsv_root / f"{split}_asr.tsv"
        asr_predictions_file = tsv_root / f"{split}_asr_wer_results.json"
        asr_inference(asr_tsv_file, asr_batch_size)

        noisy_examples_bool = find_noisy_examples(df, asr_predictions_file, asr_wer_threshold)
        df = df.loc[~noisy_examples_bool]
        print(f"removed {noisy_examples_bool.sum()} noisy examples. remaing: {len(df)}")

    filtered_tsv_path = tsv_root / f"{split}_{task}_filtered.tsv"
    save_df_to_tsv(df, filtered_tsv_path)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv_root", type = str, required = True)
    parser.add_argument("--duration_sec_threshold", type = int, default = 30)
    parser.add_argument("--apply_asr_filtering", action = "store_true")
    parser.add_argument("--asr_batch_size", type = int, default = 24)
    parser.add_argument("--asr_wer_threshold", type = float, default = .5)
    args = parser.parse_args()

    tsv_root = Path(args.tsv_root)

    # get dataset name
    for dataset in DATASETS:
        if dataset in args.tsv_root:
            break

    utterance_cleaner_func = CLEAN_FUNC[dataset]

    for task in TASKS:
        for split in SPLITS[dataset]:
            if split.startswith("train"):
                filter(tsv_root, split, task, utterance_cleaner_func,
                    args.duration_sec_threshold, args.apply_asr_filtering,
                    args.asr_batch_size, args.asr_wer_threshold)


if __name__ == "__main__":
    main()
