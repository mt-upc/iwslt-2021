#!/usr/bin/env python3

import yaml
import argparse
import pandas as pd
from pathlib import Path
from examples.speech_to_text.data_utils import save_df_to_tsv


MANIFEST_COLUMNS = ["id", "audio", "duration_ms", "n_frames", "tgt_text", "tgt_lang"]


def process(root: Path, path_to_segm_file: Path, test_id: str,
is_custom: bool = False) -> None:

    with open(path_to_segm_file, "r") as f:
        segments = yaml.load(f, Loader=yaml.BaseLoader)

    print("Generating manifest...")
    manifest = {c: [] for c in MANIFEST_COLUMNS}
    prev_wav_filename, id_suffix = "", 0
    for s in segments:

        # add a suffix to have a unique id for each segment
        wav_filename = root / "wavs" / Path(s['wav']).name
        if prev_wav_filename == wav_filename:
            id_suffix += 1
        else:
            id_suffix = 1
        prev_wav_filename = wav_filename

        offset = int(float(s["offset"]) * 16000)
        n_frames = int(float(s["duration"]) * 16000)
        duration_ms = int(float(s["duration"]) * 1000)
        manifest["id"].append(f"{wav_filename.stem}_{id_suffix}")
        manifest["audio"].append(f"{wav_filename}:{offset}:{n_frames}")
        manifest["duration_ms"].append(duration_ms)
        manifest["n_frames"].append(n_frames)
        manifest["tgt_text"].append("???")
        manifest["tgt_lang"].append("de")

    df = pd.DataFrame.from_dict(manifest)
    if is_custom:
        max_segm_len = path_to_segm_file.stem.rsplit("_", maxsplit = 1)[1]
        out_file = root.parent / f"IWSLT.{test_id}_own_{max_segm_len}.tsv"
    else:
        out_file = root.parent / f"IWSLT.{test_id}.tsv"
    save_df_to_tsv(df, out_file)
    print(f"Saved manifest at: {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir-root", "-d", required=True, type=str,
        help = "path to the directory containing the specific test dataset, eg. IWSLT.tst2019")
    parser.add_argument("--custom-segm-file", "-c", action="store_true",
        help = "whether the script is going to use custom segmentation yamls.")
    args = parser.parse_args()

    root = Path(args.test_dir_root).absolute()
    test_id = root.suffix[1:]

    # run for every custom segmentation
    if args.custom_segm_file:
        print("Preparing test set {test_id} with custom segmentations")
        for path_to_segm_file in (root / "own_segmentation").glob("*own*"):
            process(root, path_to_segm_file, test_id, is_custom = True)
    # run once (for the provided segmentation)
    else:
        path_to_segm_file = root / f"IWSLT.TED.{test_id}.en-de.yaml"
        print("Preparing test set {test_id} with original segmentation")
        process(root, path_to_segm_file, test_id)


if __name__ == "__main__":
    main()
    