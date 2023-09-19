import os
import sys
import argparse
from tqdm import tqdm

sys.path.append(os.getcwd())
from MuGEN.analysis.dataset_difficulty import DatasetDifficulty


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_data_folder_path", type=str, required=True)
    parser.add_argument("--output_data_folder_path", type=str, required=True)
    parser.add_argument('--model_checkpoints', nargs='+', type=int, help='A list of model checkpoints', required=True)
    parser.add_argument('--model_output_path', nargs='+', type=int, help='Path to dump model weights', required=True)
    parser.add_argument('--augment', type=bool, required=False)
    parser.add_argument('--run_model', type=bool, required=False)
    parser.add_argument('--get_v_usable', type=bool, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_difficulty = DatasetDifficulty()

    # Step 1: Augment data iteratively with Standard and Null transformation
    if args.augment:
        data_folder_path = [i for i in os.listdir(args.input_data_folder_path) if not i.startswith('.')]
        pbar = tqdm(data_folder_path)

        for dataset_folder in pbar:
            pbar.set_description(f"Dataset: {dataset_folder}")

            # Set output and input folder paths
            output_dataset_folder_path = os.path.join(args.output_data_folder_path, 'v_usable_' + dataset_folder)
            input_dataset_folder_path = os.path.join(args.input_data_folder_path, dataset_folder)

            # Get the types of dataset (viz. original, synthetic etc.) for each dataset
            dataset_type_list = [i for i in os.listdir(input_dataset_folder_path) if not i.startswith('.')]

            # Create folder to store v-usable augmented dataset types
            os.makedirs(output_dataset_folder_path, exist_ok=True)

            for dataset_type in dataset_type_list:

                # Set output and input sub folder paths within V-Usable folder
                output_dataset_directory = os.path.join(output_dataset_folder_path, dataset_type)
                input_dataset_directory = os.path.join(input_dataset_folder_path, dataset_type)

                # Create folder to store v-usable augmented datasets
                os.makedirs(output_dataset_directory, exist_ok=True)

                # Get the dataset split for each dataset's data type configuration
                dataset_split_list = [i for i in os.listdir(os.path.join(input_dataset_folder_path, dataset_type)) if
                                      not i.startswith('.')]

                train_path, test_path, val_path = '', '', ''
                for dataset_split in dataset_split_list:
                    if 'train' in dataset_split:
                        train_path = os.path.join(input_dataset_directory, dataset_split)
                    elif 'test' in dataset_split:
                        test_path = os.path.join(input_dataset_directory, dataset_split)
                    elif 'val' in dataset_split:
                        val_path = os.path.join(input_dataset_directory, dataset_split)

                print('DATASET: ', dataset_folder, 'TYPE: ', dataset_type)
                data_difficulty.augment_data(output_dataset_directory, train_path, test_path, val_path, 'standard')
                data_difficulty.augment_data(output_dataset_directory, train_path, test_path, val_path, 'null')

    # Step 2: Run models iteratively using all the augmented datasets
    if args.run_model:

        # Iterate through the output data folder
        output_data_folder_path_list = [i for i in os.listdir(args.output_data_folder_path) if not i.startswith(".")]
        pbar = tqdm(output_data_folder_path_list)

        for dataset_folder in pbar:
            if dataset_folder.startswith('v_usable'):
                dataset_folder_list = [i for i in os.listdir(dataset_folder) if not i.startswith(".")]

                for data_type_folder in dataset_folder_list:
                    data_type_folder_path = os.path.join(args.output_data_folder_path, dataset_folder, data_type_folder)
                    dataset_split_list = [i for i in os.listdir(data_type_folder_path) if not i.startswith(".")]

                    # Create directory for the store model weights for each dataset split
                    model_output_dir = os.path.join(args.model_output_path, dataset_folder, data_type_folder)
                    os.makedirs(model_output_dir, exist_ok=True)

                    null_train_file, null_val_file, std_train_file, std_val_file = None, None, None, None

                    for dataset_split in dataset_split_list:
                        if dataset_split.startswith("null") and "train" in dataset_split:
                            null_train_file = dataset_split
                        if dataset_split.startswith("null") and "test" in dataset_split:
                            null_test_file = dataset_split
                        if dataset_split.startswith("null") and "val" in dataset_split:
                            null_val_file = dataset_split
                        if dataset_split.startswith("std") and "train" in dataset_split:
                            std_train_file = dataset_split
                        if dataset_split.startswith("std") and "test" in dataset_split:
                            std_test_file = dataset_split
                        if dataset_split.startswith("std") and "val" in dataset_split:
                            std_val_file = dataset_split

                    # Run NULL Model
                    if null_train_file is not None and null_val_file is not None:
                        null_arguments = f"""
                            --model_name_or_path {args.model_checkpoint} \
                            --tokenizer_name {args.model_checkpoint} \
                            --output_dir {model_output_dir} \
                            --train_file {null_train_file} \
                            --validation_file {null_val_file} \
                            --per_device_train_batch_size 3 \
                            --per_device_eval_batch_size 3 \
                            --num_train_epochs 2 \
                            --seed 1
                            """
                        data_difficulty.finetune_models(null_arguments)

                    # Run STD model
                    if std_train_file is not None and std_val_file is not None:
                        std_arguments = f"""
                            --model_name_or_path {args.model_checkpoint} \
                            --tokenizer_name {args.model_checkpoint} \
                            --output_dir {args.model_output_path} \
                            --train_file {std_train_file} \
                            --validation_file {std_val_file} \
                            --per_device_train_batch_size 3 \
                            --per_device_eval_batch_size 3 \
                            --num_train_epochs 2 \
                            --seed 1
                            """
                        data_difficulty.finetune_models(std_arguments)


