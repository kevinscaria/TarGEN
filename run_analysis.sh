#!/bin/bash

# # Dataset Difficulty
# ## Augment Data
# python /data/data/hgupta35/MuGEN/scripts/extract_dataset_difficulty.py \
# --input_data_folder_path /data/data/hgupta35/MuGEN/data \
# --output_data_folder_path /data/data/hgupta35/MuGEN/analysis_files/dataset_difficulty \
# --exclude_folders multirc record \
# --augment True

# ## Train V-model
# python /data/data/hgupta35/MuGEN/scripts/extract_dataset_difficulty.py \
# --input_data_folder_path /data/data/hgupta35/MuGEN/data \
# --output_data_folder_path /data/data/hgupta35/MuGEN/analysis_files/dataset_difficulty \
# --model_checkpoints bert-base-uncased \
# --include_folders v_boolq \
# --model_output_path /data/data/hgupta35/MuGEN/analysis_files/dataset_difficulty \
# --trainer_file /data/data/hgupta35/MuGEN/MuGEN/analysis/vusable_information/run_glue_no_trainer.py \
# --run_model True

# ## Compute V-Info
# python /data/data/hgupta35/MuGEN/scripts/extract_dataset_difficulty.py \
# --input_data_folder_path /data/data/hgupta35/MuGEN/data \
# --output_data_folder_path /data/data/hgupta35/MuGEN/analysis_files/dataset_difficulty \
# --model_checkpoints bert-base-uncased \
# --model_output_path /data/data/hgupta35/MuGEN/analysis_files/dataset_difficulty \
# --pvi_output_path /data/data/hgupta35/MuGEN/analysis_files/dataset_difficulty \
# --get_v_usable True