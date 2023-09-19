import os
import argparse
import sys
sys.path.append(os.getcwd())
from MuGEN.analysis.dataset_diversity import DatasetDiversity


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--required_column", type=str, required=True)
    parser.add_argument("--n_sample", type=int, default=1000,
                        help="how many sentences to sample to calculate bleu")
    parser.add_argument("--dump_path", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    dataset_diversity = DatasetDiversity(method="selfbleu")
    args = parse_args()
    dataset_diversity.compute_diversity(args.file_path, args.required_column, args.n_sample, args.dump_path)
