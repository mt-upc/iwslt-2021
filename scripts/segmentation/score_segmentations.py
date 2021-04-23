import argparse
from pathlib import Path
from typing import List
import sacrebleu
from sacremoses import MosesTokenizer, MosesDetokenizer


def load_segm_file(path_to_segmentation_file_i: Path) -> List[str]:
    segm_translation = []
    with open(path_to_segmentation_file_i, "r", encoding = "utf-8") as f:
        lines = f.read().splitlines()
        for i, line in enumerate(lines):
            if line[:4] == "<seg":
                segm_translation.append(lines[i + 1])
    return segm_translation


def score(path_to_segmentation: str, path_to_reference: str) -> None:

    path_to_segmentation = Path(path_to_segmentation)
    path_to_reference = Path(path_to_reference)

    # init tokenizer and detokenizer
    mt, md = MosesTokenizer(lang = "de"), MosesDetokenizer(lang = "de")

    # extract the reference sentences from the xml file
    reference = []
    with open(path_to_reference, "r", encoding = "utf-8") as f:
        for line in f.read().splitlines():
            if line[:4] == "<seg":
                reference.append(line.split(">", maxsplit = 1)[1].split("</seg>")[0])

    scores = {}
    for path_to_segmentation_file_i in path_to_segmentation.glob("own_*.xml"):

        max_segm_len = int(path_to_segmentation_file_i.stem.split("_")[-1])

        # extract generated translations from the xml file
        segm_translation = load_segm_file(path_to_segmentation_file_i)

        # detokenize (have to tokenize first with the python implementation of Moses)
        segm_translation = [md.detokenize(mt.tokenize(s)) for s in segm_translation]

        assert len(reference) == len(segm_translation)

        # get bleu score
        bleu = sacrebleu.corpus_bleu(segm_translation, [reference])
        scores[max_segm_len] = bleu.score

    scores = dict(sorted(scores.items()))

    # do the same process for the original segmentation
    path_to_original_segmentation_file = path_to_segmentation / "original_segm.xml"
    original_segm_translation = load_segm_file(path_to_original_segmentation_file)
    original_segm_translation = [md.detokenize(mt.tokenize(s)) for s in original_segm_translation]
    assert len(reference) == len(original_segm_translation)

    bleu = sacrebleu.corpus_bleu(original_segm_translation, [reference])
    scores["original"] = bleu.score

    for n, s in scores.items():
        print(f"{n}: {s} BLEU")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_segmentation", "-s", required=True, type=str,
        help = "Path to the directory with the segmentation files generated from mwerSegmenter")
    parser.add_argument("--path_to_reference", "-r", required=True, type=str,
        help = "Path to the xml file with the reference translation in the target language (de)")
    args = parser.parse_args()

    score(args.path_to_segmentation, args.path_to_reference)