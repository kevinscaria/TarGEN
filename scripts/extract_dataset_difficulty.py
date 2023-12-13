import os
import sys
import argparse
from tqdm import tqdm

sys.path.append(os.getcwd())
from TarGEN.analysis.dataset_difficulty import DatasetDifficulty


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_data_folder_path", type=str, required=True)
    parser.add_argument("--output_data_folder_path", type=str, required=True)
    parser.add_argument("--exclude_folders", nargs='+', type=str, required=False)
    parser.add_argument("--include_folders", nargs='+', type=str, required=False)
    parser.add_argument('--model_checkpoints', nargs='+', type=str, help='A list of model checkpoints', required=False)
    parser.add_argument('--model_output_path', type=str, help='Path to dump model weights', required=False)
    parser.add_argument('--pvi_output_path', type=str, help='Path to dump model weights', required=False)
    parser.add_argument('--trainer_file', type=str, help='Path to the trainer code', required=False)
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

        # Check inclusion folders
        if args.include_folders is not None:
            data_folder_path = [i for i in os.listdir(data_folder_path) if i in args.include_folders]

        pbar = tqdm(data_folder_path)
        for dataset_folder in pbar:
            
            if dataset_folder in args.exclude_folders:
                print(f" =========== IGNORED {dataset_folder} \n =========== ")
                continue

            pbar.set_description(f"Dataset: {dataset_folder}")

            # Set output and input folder paths
            output_dataset_folder_path = os.path.join(args.output_data_folder_path, "augmented_data", "v_" + dataset_folder)
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

                print('\n\n=============================\nDATASET: ', dataset_folder, 'TYPE: ', dataset_type)
                data_difficulty.augment_data(output_dataset_directory, train_path, test_path, val_path, 'standard')
                data_difficulty.augment_data(output_dataset_directory, train_path, test_path, val_path, 'null')

    # Step 2: Run models iteratively using all the augmented datasets
    if args.run_model:

        # Iterate through the augmented output data folder
        root_path = os.path.join(args.output_data_folder_path, "augmented_data")
        output_data_folder_path_list = [i for i in os.listdir(root_path) if not i.startswith(".")]

        # Check inclusion folders
        if args.include_folders is not None:
            output_data_folder_path_list = [i for i in output_data_folder_path_list if i in args.include_folders]

        pbar = tqdm(output_data_folder_path_list)

        for dataset_folder in pbar:
            if dataset_folder.startswith("v_"):

                if args.exclude_folders is not None and dataset_folder in args.exclude_folders:
                    print(f" ====== IGNORED ======\n\t{dataset_folder}\n=========================\n")
                    continue

                # Get path for dataset folder and get the files
                dataset_folder_path = os.path.join(root_path, dataset_folder)
                dataset_type_list = [i for i in os.listdir(dataset_folder_path) if not i.startswith(".")]

                for dataset_type in dataset_type_list:
                    dataset_type_folder_path = os.path.join(dataset_folder_path, dataset_type)
                    dataset_split_list = [i for i in os.listdir(dataset_type_folder_path) if not i.startswith(".")]

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

                    for model_checkpoint in args.model_checkpoints:

                        # Create directory for the store NULL model weights for each dataset split
                        model_output_dir = os.path.join(args.model_output_path, 'model_weights', 
                                                        dataset_folder, dataset_type, f'{model_checkpoint}-null')
                        os.makedirs(model_output_dir, exist_ok=True)

                        # Run NULL Model
                        if null_train_file is not None:

                            if null_val_file is None:
                                null_val_path = os.path.join(dataset_type_folder_path, null_test_file)

                            null_arguments = f"""
                                --model_name_or_path {model_checkpoint} \
                                --tokenizer_name {model_checkpoint} \
                                --output_dir {model_output_dir} \
                                --train_file {os.path.join(dataset_type_folder_path, null_train_file)} \
                                --validation_file {null_val_path} \
                                --per_device_train_batch_size 3 \
                                --per_device_eval_batch_size 3 \
                                --num_train_epochs 2 \
                                --seed 1
                                """
                            
                            if not os.path.exists(os.path.join(model_output_dir, 'config.json')):
                                print('\n\n=============================\nDATASET: ', dataset_folder)
                                print('\nTYPE: ', dataset_type)
                                print('\nNULL MODEL CHECKPOINT: ', model_checkpoint)
                                print('\n=============================\n')
                                data_difficulty.finetune_models(args.trainer_file,  null_arguments)
                            else:
                                print('\n\n=============================\nDATASET: ', dataset_folder)
                                print('\nTYPE: ', dataset_type)
                                print('\nNULL MODEL CHECKPOINT: ', model_checkpoint)
                                print('\nALREADY TRAINED')

                        # Create directory for the store STD model weights for each dataset split
                        model_output_dir = os.path.join(args.model_output_path, 'model_weights', 
                                                        dataset_folder, dataset_type, f'{model_checkpoint}-std')
                        os.makedirs(model_output_dir, exist_ok=True)

                        # Run STD model
                        if std_train_file is not None:

                            if std_val_file is None:
                                std_val_path = os.path.join(dataset_type_folder_path, std_test_file)

                            std_arguments = f"""
                                --model_name_or_path {model_checkpoint} \
                                --tokenizer_name {model_checkpoint} \
                                --output_dir {model_output_dir} \
                                --train_file {os.path.join(dataset_type_folder_path, std_train_file)} \
                                --validation_file {std_val_path} \
                                --per_device_train_batch_size 3 \
                                --per_device_eval_batch_size 3 \
                                --num_train_epochs 2 \
                                --seed 1
                                """
                            
                            if not os.path.exists(os.path.join(model_output_dir, 'config.json')):
                                print('\n\n=============================\nDATASET: ', dataset_folder)
                                print('\nTYPE: ', dataset_type)
                                print('\nSTD MODEL CHECKPOINT: ', model_checkpoint)
                                print('\n=============================\n')
                                data_difficulty.finetune_models(args.trainer_file, std_arguments)
                            else:
                                print('\n\n=============================\nDATASET: ', dataset_folder)
                                print('\nTYPE: ', dataset_type)
                                print('\nSTD MODEL CHECKPOINT: ', model_checkpoint)
                                print('\nALREADY TRAINED')

    # Step 3: Compute v_info on all the augmented datasets and trained models
    if args.get_v_usable:

        # Iterate through the augmented output data folder
        root_path = os.path.join(args.output_data_folder_path, "augmented_data")
        output_data_folder_path_list = [i for i in os.listdir(root_path) if not i.startswith(".")]

        # Check inclusion folders
        if args.include_folders is not None:
            output_data_folder_path_list = [i for i in output_data_folder_path_list if i in args.include_folders]

        pbar = tqdm(output_data_folder_path_list)

        for dataset_folder in pbar:
            if dataset_folder.startswith("v"):

                if args.exclude_folders is not None and dataset_folder in args.exclude_folders:
                    print(f" ====== IGNORED ======\n\t{dataset_folder}\n=========================\n")
                    continue

                # Get path for dataset folder and get the files
                dataset_folder_path = os.path.join(root_path, dataset_folder)
                dataset_type_list = [i for i in os.listdir(dataset_folder_path) if not i.startswith(".")]

                for dataset_type in dataset_type_list:
                    dataset_type_folder_path = os.path.join(dataset_folder_path, dataset_type)
                    dataset_split_list = [i for i in os.listdir(dataset_type_folder_path) if not i.startswith(".")]

                    std_data_fn, null_data_fn = None, None

                    for dataset_split in dataset_split_list:
                        if dataset_split.startswith("null") and "test" in dataset_split:
                            null_test_file = dataset_split
                        if dataset_split.startswith("std") and "test" in dataset_split:
                            std_test_file = dataset_split
                    
                    # Create directory for the store PVI for each dataset split
                    pvi_output_dir = os.path.join(args.pvi_output_path, 'PVI', 
                                                  dataset_folder, dataset_type)
                    os.makedirs(pvi_output_dir, exist_ok=True)

                    for model_checkpoint in args.model_checkpoints:

                        print('\n\n=============================\nDATASET: ', dataset_folder)
                        print('\nTYPE: ', dataset_type)
                        print('\nMODEL CHECKPOINT: ', model_checkpoint)
                        print('\n=============================\n')

                        # Set output path where PVI csv will be saved
                        out_fn = os.path.join(pvi_output_dir, f"{model_checkpoint}_pvi.csv")
                        
                        # Get STD model path
                        std_model = os.path.join(args.model_output_path, 'model_weights', 
                                                        dataset_folder, dataset_type, 
                                                        f'{model_checkpoint}-std')
                        # Get NULL model path
                        null_model = os.path.join(args.model_output_path, 'model_weights', 
                                                        dataset_folder, dataset_type, 
                                                        f'{model_checkpoint}-null')
                        
                        # Get corresponding augmented data paths
                        std_data_fn = os.path.join(dataset_type_folder_path, std_test_file)
                        null_data_fn = os.path.join(dataset_type_folder_path, null_test_file)
                    
                        if not os.path.exists(out_fn):
                            data_difficulty.compute_v_usable_info(
                                out_fn = out_fn, 
                                std_data_fn = std_data_fn,
                                std_model = std_model,
                                null_data_fn = null_data_fn, 
                                null_model = null_model,
                                model_name = model_checkpoint,
                                input_col = "input",
                                )
