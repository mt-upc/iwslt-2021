import argparse
from pathlib import Path
from typing import Callable, Union
import pandas as pd
import os
import warnings

import sys
sys.path.append("./../fairseq/examples")
sys.path.append("./../fairseq/fairseq/data")
sys.path.append("scripts/filtering")
from speech_to_text.data_utils import load_df_from_tsv, save_df_to_tsv
from asr.asr_inference import asr_inference
from filtering_utils import (find_noisy_examples, mustc_utterance_cleaner,
    europarlst_utterance_cleaner, covost_utterance_cleaner)


DATASETS = ["MUSTC", "EUROPARLST", "COVOST"]
TRAIN_SPLITS = {
    "MUSTC": ["train"],
    "EUROPARLST": ["train", "dev"],
    "COVOST": ["train", "dev"]}
CLEANER_FUNC = {
    "MUSTC": mustc_utterance_cleaner,
    "EUROPARLST": europarlst_utterance_cleaner,
    "COVOST": covost_utterance_cleaner}
TASKS = ["asr", "st"]


def filter(df: pd.DataFrame, asr_tsv_path: Union[Path, str], task: str,
utterance_cleaner: Callable, asr_batch_size: int, asr_wer_threshold: float) -> pd.DataFrame:

    # target text cleaning
    df["tgt_text"] = df.apply(lambda x: utterance_cleaner(x["tgt_text"]), axis = 1)

    # removal of empty examples after cleaning
    empty_examples_bool = df.tgt_text == ""
    df = df.loc[~empty_examples_bool]
    print(f"removed {empty_examples_bool.sum()} empty examples. remaining: {len(df)}")

    # removal of noisy examples (based on ASR system predictions)
    if task == "st":
        asr_predictions_path = asr_tsv_path.parent / f"{asr_tsv_path.stem}_wer_results.json"

        print("Starting ASR inference with Wav2Vev_2.0 ...")
        asr_inference(asr_tsv_path, asr_batch_size)

        noisy_examples_bool = find_noisy_examples(df, asr_predictions_path, asr_wer_threshold)
        df = df.loc[~noisy_examples_bool]
        print(f"removed {noisy_examples_bool.sum()} noisy examples. remaining: {len(df)}")

    return df


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type = str, required = True, choices = DATASETS)
    parser.add_argument("--dataset_root", type = str, required = True,
        help = "Path for the directory containing the TSV files for this DATASET.")
    parser.add_argument("--asr_batch_size", type = int, default = 24,
        help = "Batch size to be used during the ASR inference with Wav2Vec.")
    parser.add_argument("--asr_wer_threshold", type = float, default = .5,
        help = "Word-Error-Rate above which an example is considered noisy.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    file_list = os.listdir(dataset_root)

    utterance_cleaner = CLEANER_FUNC[args.dataset_name]
    asr_tsv_path = ""

    for task in TASKS:

        for split in TRAIN_SPLITS[args.dataset_name]:

            # find the file for this (task, split) pair
            match = False
            for file_name in file_list:
                file_name = Path(file_name)
                if file_name.suffix == ".tsv":
                    file_attributes = file_name.stem.split("_")
                    if task in file_attributes and split in file_attributes:
                        match = True
                        break

            if not match:
                warnings.warn(f"no file found for dataset = {args.dataset_name}, task = {task}, split = {split}")
                continue

            tsv_path = dataset_root / file_name
            df_split = load_df_from_tsv(tsv_path)

            print(f"Running filtering script for {split} split of {args.dataset_name} from file {tsv_path}")
            df_split_filtered = filter(df_split, asr_tsv_path, task, utterance_cleaner,
                args.asr_batch_size, args.asr_wer_threshold)

            new_tsv_path = dataset_root / Path(tsv_path.stem + "_filtered.tsv")
            save_df_to_tsv(df_split_filtered, new_tsv_path)
            print(f"Saved filtered TSV for: {split} of {task} at: {new_tsv_path}")

            if task == "asr":
                asr_tsv_path = new_tsv_path


if __name__ == "__main__":
    main()
