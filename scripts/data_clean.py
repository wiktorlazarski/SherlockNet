import argparse
import os
import re
import string

def normalize_text(text):
    """Normalize text.

    Args:
        text (str): in text

    Returns:
        str: normalized text
    """
    text = text.lstrip()

    remove_punctuations = "\"#$%&()*+,-/:;<=>@[\]^_`{|}~’"
    split_punctuations = re.compile(r"(?<!M[rs])(?<!Mrs|etc)(?<!Dr|Ph|St)(?<!D)([!.?])")

    text = text.translate(str.maketrans("", "", remove_punctuations))
    text = split_punctuations.sub(r' \1', text)

    return text


def normalize_file(in_file, out_file):
    """Normalize english text in book.

    Args:
        in_file (str): unprocessed book input path
        out_file (str): processed book output path
    """
    with open(in_file, "r") as book, open(out_file, "w") as out:
        for line in book:
            if line.strip() == "":
                continue

            line = normalize_text(line)

            out.write(line.lstrip())


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--raw_data', type=str, required=True)
    parser.add_argument('--out_data', type=str, required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    for book in os.listdir(args.raw_data):
        book_path = os.path.join(args.raw_data, book)
        out_path = os.path.join(args.out_data, book)

        normalize_file(book_path, out_path)


if __name__ == "__main__":
    main()
