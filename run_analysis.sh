#!/bin/bash

# Dataset Difficulty
## Augment Data
python ./MuGEN/scripts/extract_dataset_difficulty.py \
--input_data_folder_path ./MuGEN/data \
--output_data_folder_path ./MuGEN/analysis_files/dataset_difficulty \
--exclude_folders multirc record \
--augment True

## Train V-model
python ./MuGEN/scripts/extract_dataset_difficulty.py \
--input_data_folder_path ./MuGEN/data \
--output_data_folder_path ./MuGEN/analysis_files/dataset_difficulty \
--model_checkpoints bert-base-uncased \
--include_folders v_boolq \
--model_output_path ./MuGEN/analysis_files/dataset_difficulty \
--trainer_file ./MuGEN/MuGEN/analysis/vusable_information/run_glue_no_trainer.py \
--run_model True

## Compute V-Info
python ./MuGEN/scripts/extract_dataset_difficulty.py \
--input_data_folder_path ./MuGEN/data \
--output_data_folder_path ./MuGEN/analysis_files/dataset_difficulty \
--model_checkpoints bert-base-uncased \
--model_output_path ./MuGEN/analysis_files/dataset_difficulty \
--pvi_output_path ./MuGEN/analysis_files/dataset_difficulty \
--get_v_usable True


# Dataset Diversity
##  Vocabulary Based
python ./MuGEN/scripts/extract_diversity.py \
--input_data_folder ./MuGEN/data/ \
--required_column input \
--method vocabulary \
--dump_path ./MuGEN/analysis_files/diversity/vocabulary_diversity.csv

##  Semantic Density Reduction Analysis
python ./MuGEN/scripts/extract_diversity.py \
--input_data_folder ./MuGEN/data/ \
--required_column 'input' \
--dump_path ./MuGEN/analysis_files/diversity/embeddings \
--model_ckpt sentence-transformers/msmarco-distilbert-base-tas-b \
--method sdra

##  SelfBLEU Based
python ./MuGEN/scripts/extract_diversity.py \
--input_data_folder ./MuGEN/data/ \
--required_column input \
--method vocabulary \
--dump_path ./MuGEN/analysis_files/diversity/vocabulary_diversity.csv