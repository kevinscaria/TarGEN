import json
import os
import random
import umap
import pandas as pd
from functools import partial
from multiprocessing.pool import Pool
from tqdm import tqdm

import spacy
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import datasets
from sentence_transformers import SentenceTransformer


class DatasetDiversity:
    function = "Dataset diversity"  # Static Variable

    def __init__(self) -> None:
        pass

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

        if n_sample is not None:
            all_sentences = random.sample(all_sentences, n_sample)
        return all_sentences
    
    @staticmethod
    def get_embeddings(sentence, embedding_model_ckpt='bert-base-uncased'):
        model = SentenceTransformer(embedding_model_ckpt)
        return model.encode(sentence)

    @staticmethod
    def umap_reduce(dataset):
        reducer = umap.UMAP()
        umap_embedding = reducer.fit_transform(dataset)
        return umap_embedding

    @staticmethod
    def compute_diversity(method, file_path, required_column, n_sample=None, model_ckpt = 'sentence-transformers/all-MiniLM-L6-v2'):
        random.seed(0)
        all_sentences = DatasetDiversity.get_all_sentences(file_path, required_column, n_sample)
        nlp =  spacy.load('en_core_web_sm', disable=['parser', 'ner', 'lemmatizer'])

        if method == 'vocabulary':
            print("Method: Vocabulary")
            vocabulary_count=0
            for sentence in tqdm(all_sentences, desc='Counting Vocabulary: '):
                if isinstance(sentence, str):
                    spacy_sentence = nlp(sentence)
                    non_stop_words = [token.text for token in spacy_sentence if not token.is_stop]
                vocabulary_count+=len(set(non_stop_words))
            return vocabulary_count
        
        if method == 'sdra':
            print("Method: Semantic Density Reduction Analysis (SDRA)")

            # Compute SentenceBERT embeddings
            high_dimensional_embeddings = []
            for sentence in tqdm(all_sentences, desc='SDRA'):
                if isinstance(sentence, str):
                    high_dimensional_embeddings.append(DatasetDiversity.get_embeddings(sentence, model_ckpt))
            
            # Compute UMAP 2D embeddings
            low_dimensional_embeddings = (DatasetDiversity.umap_reduce(high_dimensional_embeddings))
            return low_dimensional_embeddings


        if method == "selfbleu":
            print("Method: SelfBLEU")
            all_sentences = [[tok.text for tok in nlp(s)] for s in all_sentences]
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
                            partial(DatasetDiversity.bleu_i, weights, all_sentences, smoothing_function),
                            list(range(len(all_sentences)))),
                        total=n_sample,
                        smoothing=0.0,
                        desc=f"bleu-{n_gram}")))
                print(f"\n\nbleu-{n_gram} = {sum(bleu_scores[n_gram - 1]) / n_sample}")

            for n_gram in range(5):
                print(f"bleu-{n_gram + 1} = {sum(bleu_scores[n_gram]) / n_sample}")
