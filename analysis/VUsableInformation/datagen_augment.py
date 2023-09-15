from dataset_difficulty.augment import (
    CustomDataTransformation, CustomDataNullTransformation,
    CustomDataStandardTransformation, CustomDataLengthTransformation
    )

output_directory = './dataset_difficulty/data/project_data/voriginal/boolq'
input_directory = './dataset_difficulty/data/project_data/original/boolq'

cdt = CustomDataTransformation(
    output_dir = output_directory,
    train_file = f'{input_directory}/train.csv',
    test_file = f'{input_directory}/test.csv',
    # val_file = f'{input_directory}/val.csv'
    )
cdt.transform() 