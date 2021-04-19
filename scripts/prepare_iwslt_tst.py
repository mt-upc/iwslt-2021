#!/usr/bin/env python3

import yaml
import argparse
import pandas as pd
from pathlib import Path
from examples.speech_to_text.data_utils import save_df_to_tsv


MANIFEST_COLUMNS = ["id", "audio", "duration_ms", "n_frames", "tgt_text", "tgt_lang"]


def process(args):
    root = Path(args.data_root).absolute()
    if not root.is_dir():
        print(f"{root.as_posix()} does not exist.")
    for test_dir in root.glob('tst*/'):
        test_id = test_dir.stem
        print(f"Fetching {test_id}...")
        with open(test_dir / f"IWSLT.TED.{test_id}.en-de.yaml") as f:
            segments = yaml.load(f, Loader=yaml.BaseLoader)
            print("Generating manifest...")
            manifest = {c: [] for c in MANIFEST_COLUMNS}
            for s in segments:
                wav_filename = test_dir / "wavs" / Path(s['wav']).name
                offset = int(float(s["offset"]) * 16000)
                n_frames = int(float(s["duration"]) * 16000)
                duration_ms = int(float(s["duration"]) * 1000)
                manifest["id"].append(wav_filename.stem)
                manifest["audio"].append(
                        f"{wav_filename}:{offset}:{n_frames}"
                    )
                manifest["duration_ms"].append(duration_ms)
                manifest["n_frames"].append(n_frames)
                manifest["tgt_text"].append("???")
                manifest["tgt_lang"].append("de")
        df = pd.DataFrame.from_dict(manifest)
        out_file = root / f"{test_id}.tsv"
        save_df_to_tsv(df, out_file)
        print(f"Saved manifest at: {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
