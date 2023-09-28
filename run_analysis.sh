#!/bin/bash

# Dataset Difficulty
## Augment Data
python ./LLMDataGen/scripts/extract_dataset_difficulty.py \
--input_data_folder_path ./LLMDataGen/data \
--output_data_folder_path ./LLMDataGen/analysis_files/dataset_difficulty \
--exclude_folders multirc record \
--augment True

## Train V-model
python ./LLMDataGen/scripts/extract_dataset_difficulty.py \
--input_data_folder_path ./LLMDataGen/data \
--output_data_folder_path ./LLMDataGen/analysis_files/dataset_difficulty \
--model_checkpoints bert-base-uncased \
--include_folders v_boolq \
--model_output_path ./LLMDataGen/analysis_files/dataset_difficulty \
--trainer_file ./LLMDataGen/LLMDataGen/analysis/vusable_information/run_glue_no_trainer.py \
--run_model True

## Compute V-Info
python ./LLMDataGen/scripts/extract_dataset_difficulty.py \
--input_data_folder_path ./LLMDataGen/data \
--output_data_folder_path ./LLMDataGen/analysis_files/dataset_difficulty \
--model_checkpoints bert-base-uncased \
--model_output_path ./LLMDataGen/analysis_files/dataset_difficulty \
--pvi_output_path ./LLMDataGen/analysis_files/dataset_difficulty \
--get_v_usable True


# Dataset Diversity
##  Vocabulary Based
python ./LLMDataGen/scripts/extract_diversity.py \
--input_data_folder ./LLMDataGen/data/ \
--required_column input \
--method vocabulary \
--dump_path ./LLMDataGen/analysis_files/diversity/vocabulary_diversity.csv

##  Semantic Density Reduction Analysis
python ./LLMDataGen/scripts/extract_diversity.py \
--input_data_folder ./LLMDataGen/data/ \
--required_column 'input' \
--dump_path ./LLMDataGen/analysis_files/diversity/embeddings \
--model_ckpt sentence-transformers/msmarco-distilbert-base-tas-b \
--method sdra

##  SelfBLEU Based
python ./LLMDataGen/scripts/extract_diversity.py \
--input_data_folder ./LLMDataGen/data/ \
--required_column input \
--method vocabulary \
--dump_path ./LLMDataGen/analysis_files/diversity/vocabulary_diversity.csv