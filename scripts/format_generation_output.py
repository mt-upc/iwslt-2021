import argparse
from pathlib import Path


def format_gen_output(path_to_generation_file: Path) -> None:

    # parse the txt with the generation output from fairseq-generate
    # generated sentences start with D-$i where $i is the correct order
    # of the sentence in the dataset
    raw_generation, correct_order = [], []
    with open(path_to_generation_file, "r", encoding = "utf8") as f:
        for line in f.read().splitlines():
            if line[:2] == "D-":
                correct_order.append(int(line.split(maxsplit = 1)[0].split("D-")[-1]))
                splits = line.split(maxsplit = 2)
                if len(splits) == 3:
                    raw_generation.append(splits[2])
                else:
                    raw_generation.append("")

    # fix to correct order
    raw_generation = [gen for _, gen in sorted(zip(correct_order, raw_generation))]

    # save clean generation txt file
    clean_generation_path = "_formated.".join(str(path_to_generation_file).rsplit(".", maxsplit = 1))
    with open(clean_generation_path, "w", encoding = "utf8") as f:
        for line in raw_generation:
            f.write(line + "\n")

    print(f"Saved formatted generation at {clean_generation_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_generation", "-p", required=True, type=str,
        help = "Path to a generation file or a directory with generation files.")
    args = parser.parse_args()

    path_to_generation = Path(args.path_to_generation)

    if path_to_generation.is_file():
        format_gen_output(path_to_generation)
    elif path_to_generation.is_dir():
        for path_to_generation_file in path_to_generation.glob("*.txt"):
            if "_formated" not in str(path_to_generation):
                format_gen_output(path_to_generation_file)
    else:
        print("Path does not exist")