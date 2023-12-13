import os
import argparse
import pickle
import sys
sys.path.append(os.getcwd())
import pandas as pd
from TarGEN.analysis.dataset_diversity import DatasetDiversity


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_data_folder", type=str, required=True)
    parser.add_argument("--required_column", type=str, required=True)
    parser.add_argument("--n_sample", type=int, help="how many sentences to sample to calculate bleu")
    parser.add_argument("--dump_path", type=str)
    parser.add_argument("--model_ckpt", type=str)
    parser.add_argument("--method", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Vocabulary Count - Dataset Diversity
    dataset_diversity = DatasetDiversity()

    final_vocab_diversity = []
    col_name = ['dataset']
    input_data_folder_list = [i for i in os.listdir(args.input_data_folder) if not i.startswith('.')]

    # Iterate through input data folders
    for idx, data_folder in enumerate(input_data_folder_list[:], start=1):
        data_folder_vocab_diversity = [f'{data_folder}']
        print(f'\n===========DATASET: {data_folder}===========')
        
        # Get path of dataset_type
        dataset_type_path = os.path.join(args.input_data_folder, data_folder)
        dataset_type_list = [i for i in os.listdir(dataset_type_path) if not i.startswith('.')]

        for dataset_type in dataset_type_list:
            
            if idx == len(input_data_folder_list) and 'inst' not in dataset_type:
                col_name.append(dataset_type)
                
            # Get path of dataset_split
            dataset_split_path = os.path.join(dataset_type_path, dataset_type)
            dataset_split_list = [i for i in os.listdir(dataset_split_path) if not i.startswith('.')]

            train_file = None
            for dataset_split in dataset_split_list:
                if "train" in dataset_split:
                    train_file = os.path.join(dataset_split_path, dataset_split)
                    break
            
            if train_file is not None and 'inst' not in train_file:
                if args.method == 'vocabulary':
                    print(f'===========DATASET TYPE: {dataset_type}===========')
                    vocab_size = dataset_diversity.compute_diversity(args.method, train_file, args.required_column)
                    data_folder_vocab_diversity.append(vocab_size)
                if args.method == 'sdra':
                    print(f'===========DATASET TYPE: {dataset_type}===========')
                    if not os.path.exists(os.path.join(args.dump_path, f"sdra_emb_{data_folder}_{dataset_type}.pkl")):
                        density_reduced_embeddings = dataset_diversity.compute_diversity(args.method, train_file, args.required_column, args.n_sample, args.model_ckpt)
                        
                        # Save the embeddings as Pickle
                        with open(os.path.join(args.dump_path, f"sdra_emb_{data_folder}_{dataset_type}.pkl"), 'wb') as f:
                            pickle.dump(density_reduced_embeddings, f)

        if args.method == 'vocabulary':
            final_vocab_diversity.append(data_folder_vocab_diversity)

    if args.dump_path and args.method == 'vocabulary':
        out_df = pd.DataFrame(final_vocab_diversity, columns = col_name)
        out_df.to_csv(args.dump_path, index=False)
