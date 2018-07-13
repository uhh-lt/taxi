import argparse
from random import random
import codecs


def sample(input_fpath, output_fpath, percent):
    with codecs.open(input_fpath, "r", "utf-8") as input, codecs.open(output_fpath, "w", "utf-8") as output:
        for line in input:
            if random() <= percent:
                output.write(line)


def main():
    parser = argparse.ArgumentParser(description="Generates a random sample from an input CSV file.")
    parser.add_argument('input', help='Input file')
    parser.add_argument('output', help='Output file.')
    parser.add_argument('percent', help="Percent.")
    args = parser.parse_args()

    print("Input: ", args.input)
    print("Output: ", args.output)
    print("Percent: ", args.percent)

    sample(args.input, args.output, float(args.percent))


if __name__ == '__main__':
    main()
