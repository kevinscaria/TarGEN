import json
import os
import random
import spacy
from functools import partial
from multiprocessing.pool import Pool
from tqdm import tqdm
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import datasets
import pandas as pd


class DatasetDiversity:
    function = "Dataset diversity"  # Static Variable

    def __init__(self, method) -> None:
        # Instance Variables
        self.method = method

    @staticmethod
    def bleu_i(weights, all_sentences, smoothing_function, i):
        return sentence_bleu(
            references=all_sentences[:i] + all_sentences[i + 1:],
            hypothesis=all_sentences[i],
            weights=weights,
            smoothing_function=smoothing_function)

    @staticmethod
    def get_all_sentences(file_path, required_column, n_sample):
        all_sentences = []
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])

        if os.path.isfile(file_path):
            if file_path.endswith("json"):
                with open(file_path, "r") as f_in:
                    for line in f_in:
                        obj = json.loads(line.strip())
                        all_sentences.append(obj[required_column])
            elif file_path.endswith("csv"):
                df = pd.read_csv(file_path)
                all_sentences = df[required_column].to_list()
        else:
            all_sentences = datasets.load_from_disk(file_path)[required_column]

        all_sentences = random.sample(all_sentences, n_sample)
        all_sentences = [[tok.text for tok in nlp(s)] for s in all_sentences]
        return all_sentences

    def compute_diversity(self, file_path=None, required_column=None, n_sample=None, dump_path=None):
        random.seed(0)
        if self.method == "selfbleu":
            print("Method: SelfBLEU")
            all_sentences = self.get_all_sentences(file_path, required_column, n_sample)
            smoothing_function = SmoothingFunction().method1
            pool = Pool(processes=os.cpu_count())
            bleu_scores = []
            for n_gram in range(1, 6):
                if n_gram == 1:
                    weights = (1.0, 0, 0, 0)
                elif n_gram == 2:
                    weights = (0.5, 0.5, 0, 0)
                elif n_gram == 3:
                    weights = (1.0 / 3, 1.0 / 3, 1.0 / 3, 0)
                elif n_gram == 4:
                    weights = (0.25, 0.25, 0.25, 0.25)
                elif n_gram == 5:
                    weights = (0.2, 0.2, 0.2, 0.2, 0.2)
                else:
                    raise ValueError
                bleu_scores.append(
                    list(tqdm(
                        pool.imap_unordered(
                            partial(self.bleu_i, weights, all_sentences, smoothing_function),
                            list(range(len(all_sentences)))),
                        total=n_sample,
                        smoothing=0.0,
                        desc=f"bleu-{n_gram}")))
                print(f"\n\nbleu-{n_gram} = {sum(bleu_scores[n_gram - 1]) / n_sample}")

            for n_gram in range(5):
                print(f"bleu-{n_gram + 1} = {sum(bleu_scores[n_gram]) / n_sample}")

            if dump_path:
                with open(dump_path, 'a') as fout:
                    print(f"{os.path.basename(file_path)}", end='\t', file=fout)
                    for n_gram in range(5):
                        print(f"{sum(bleu_scores[n_gram]) / n_sample}", end='\t', file=fout)
                    print(file=fout)
