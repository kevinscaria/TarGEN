import os
from TarGEN.analysis.vusable_information.augment import CustomDataNullTransformation, CustomDataStandardTransformation
from TarGEN.analysis.vusable_information.v_info import v_info


class DatasetDifficulty:
    def __init__(self, ):
        pass

    @staticmethod
    def augment_data(output_dataset_directory,
                     train_path='',
                     test_path='',
                     val_path='',
                     transformation_type="standard"
                     ):

        if transformation_type == 'standard':
            print('Transformation Type: ', transformation_type)
            if not os.path.exists(os.path.join(output_dataset_directory, 'std_train.csv')):
                cdt = CustomDataStandardTransformation(
                    output_dir=output_dataset_directory,
                    train_file=train_path if os.path.exists(train_path) else None,
                    val_file=val_path if os.path.exists(val_path) else None,
                    test_file=test_path if os.path.exists(test_path) else None
                )
                cdt.transform()
            else:
                print(f"Transformation for the file already completed")

        if transformation_type == 'null':
            print('Transformation Type: ', transformation_type)
            if not os.path.exists(os.path.join(output_dataset_directory, 'null_train.csv')):
                cdnt = CustomDataNullTransformation(
                    output_dir=output_dataset_directory,
                    train_file=train_path if os.path.exists(train_path) else None,
                    test_file=test_path if os.path.exists(test_path) else None,
                    val_file=val_path if os.path.exists(val_path) else None
                )
                cdnt.transform()
            else:
                print(f"Transformation for the file already completed")

    @staticmethod
    def finetune_models(trainer_file, arguments):
        command = f"python {trainer_file} \{arguments}"
        os.system(command)

    @staticmethod
    def compute_v_usable_info(out_fn, std_data_fn, std_model, 
                              null_data_fn, null_model, model_name,
                              input_col
                              ):
        if not os.path.exists(out_fn):
            v_info(
                data_fn=std_data_fn,
                model=std_model,
                null_data_fn=null_data_fn,
                null_model=null_model,
                tokenizer=model_name,
                out_fn=out_fn,
                input_key=input_col
            )