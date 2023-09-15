import argparse
from .analysis.dataset_diversity import DatasetDiversity

datadiv = DatasetDiversity(method = "selfbleu")
datadiv.bleu_i()



def parse_args(self, ):
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--n_sample", type=int, default=1000,
                        help="how many sentences to sample to calculate bleu")
    parser.add_argument("--logto", type=str)
    parser.add_argument("--gen_column", type=str, required=True)
    return parser.parse_args()