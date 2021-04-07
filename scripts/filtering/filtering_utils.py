import json
import pandas as pd
from pathlib import Path
import re
import nltk

nltk.download("punkt")


def clean_speaker_name(text: str) -> str:
    """
    Checks if the text starts with the pattern: [speaker name followed
    by colon] and returns a clean version of it by removing the pattern
    """

    start_text, rest_text = text.split(": ", maxsplit = 1)
    start_tokens = re.sub(" +", " ", text).strip().split(" ")
    num_start_tokens = len(start_tokens)

    # XXX: one word, initials, all caps
    if num_start_tokens == 1:
        if start_tokens[0].isupper():
            return rest_text

    # Xxxx (Zzzz) Yyyy: two or three words, first (middle) last, start of each name is capitalized
    elif num_start_tokens < 4:
        if all([start_tokens[i][0].isupper() for i in range(num_start_tokens)]):
            return rest_text

    return text


def clean_mustc_utterance(text: str) -> str:

    event_pattern = r"\([^()]*\)"

    if ": " in text:
        for event in re.findall(event_pattern, text):

            # check if event contains actual text from a speaker: (XX: utterance) -> utterance
            if ": " in event:
                event_text = event[1:-1]  # (xyz) -> xyz
                event_text_cleaned = clean_speaker_name(event_text)

                # replace event with its cleaned text
                if event_text != event_text_cleaned:
                    text = text.replace(event, event_text_cleaned)

    # remove rest of the events
    text = re.sub(event_pattern, "", text)

    # remove speaker name and colon
    if ": " in text:
        for sentence in nltk.sent_tokenize(text):

            if ": " in sentence:
                sentence_cleaned = clean_speaker_name(sentence)

                if sentence_cleaned != sentence:
                    text = text.replace(sentence, sentence_cleaned)


    # correct "&" symbol
    text = text.replace("& amp;", "&")

    # tabs, newlines and trailing spaces
    text = re.sub(" +", " ", text.replace("\t", " ").replace("\n", " "))

    # empty if \s
    if text == " ":
        text = ""

    return text


def clean_europarlst_utterance(text: str) -> str:

    # Fixes spaces in numbers > 1000 with inclusing a comma (50 000 -> 50,000)
    # For consitency with MuST-C data
    search = True
    while search:
        num = re.search(r"\d\s\d{3}", text)
        if num:
            text = text.replace(num[0], num[0].replace(" ", ","))
        else:
            search = False

    # tabs, newlines and trailing spaces
    text = re.sub(" +", " ", text.replace("\t", " ").replace("\n", " "))

    # empty if \s
    if text == " ":
        text = ""

    return text


def clean_covost_utterance(text: str) -> str:
    return text


def find_noisy_examples(df: pd.DataFrame, asr_predictions_file: Path,
asr_wer_theshold: float) -> pd.Series:

    # to ensure removal of non-existent ids in both asr and st datasets
    df["WER"] = 999.

    ids = set(df.id.tolist())

    # fill-in WER results for each example
    with open(asr_predictions_file, "r") as file:
        for line in file:
            example = json.loads(line)

            if example["id"] in ids:
                df.loc[df.id == example["id"], "WER"] = float(example["WER"])

    noisy_examples_bool = df.WER > asr_wer_theshold

    return noisy_examples_bool