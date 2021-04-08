#!/usr/bin/env python3
from torch.utils import data
import string
from num2words import num2words
import numpy as np
from typing import List, Tuple
from pathlib import Path
import re

from audio.audio_utils import get_waveform
from speech_to_text.data_utils import load_df_from_tsv


class AsrDataset(data.Dataset):
    """
    Dataset object for loading a datapoint, processing it,
    and passing to the dataloader
    """

    def __init__(self, tsv_root: Path, completed_ids: List[str]) -> None:
        super(AsrDataset, self).__init__()

        self.data = load_df_from_tsv(tsv_root)

        self.data = self.data[~self.data.id.isin(set(completed_ids))]

        # sort on n_frames
        self.data = self.data.sort_values("n_frames", axis = 0)

        # for cleaning target text
        self.my_punctuation = string.punctuation.replace("'", "")
        
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> Tuple[str, np.array, str]:

        # get datapoint
        _id, audio, tgt_text = self.data.iloc[i][["id", "audio", "tgt_text"]]

        # load audio frame into 1-d numpy array
        if audio.count(":") > 1:
            wav_file_path, offset, num_frames = audio.rsplit(":", 2)
            offset, num_frames = int(offset), int(num_frames)
        else:
            wav_file_path, offset, num_frames = audio, 0, -1

        wav_array, _ = get_waveform(
                wav_file_path,
                start = int(offset),
                frames = int(num_frames),
                always_2d = False)

        # apply formatting to target text
        tgt_text = self._fix_format(tgt_text)

        return (_id, wav_array, tgt_text)

    def _fix_format(self, tgt_text: str) -> str:

        # convert different brackets format
        # (1) ‘ sequence of words ’ -> "sequence of words"
        # (2) John  ’ s car -> John's car
        for brackets_text in re.findall(r"‘\s.+\s’", tgt_text):
            tgt_text = tgt_text.replace(brackets_text, f'"{brackets_text[2:-2]}"')
        tgt_text = tgt_text.replace(" ’ ", "'")

        # & symbol
        tgt_text = tgt_text.replace("&", " and ")

        # words connected by -
        tgt_text = tgt_text.replace("-", " ")

        # plus signs
        tgt_text = tgt_text.replace("+", "plus ")

        # remove punctuation, except apostrophes
        tgt_text = tgt_text.translate(str.maketrans("", "", self.my_punctuation))

        # trasform numbers to words
        tgt_text = " ".join([num2words(token).replace("and ", "").replace("-", " ")
            if token.isnumeric() else token
            for token in tgt_text.split(" ")])

        # reduce tabs, newlines and trailing spaces
        tgt_text = re.sub(" +", " ", tgt_text.replace("\t", " ").replace("\n", " "))

        # lowercase
        tgt_text = tgt_text.lower()

        return tgt_text


def asr_collate_fn(batch: List) -> Tuple[List[str], List[np.array], List[str]]:

    ids = [example[0] for example in batch]
    audio = [example[1] for example in batch]
    tgt_text = [example[2] for example in batch]

    return (ids, audio, tgt_text)