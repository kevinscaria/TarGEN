# CUDA_VISIBLE_DEVICES="0"

model_name="bert-large-uncased"
# model_name="roberta-large"
# model_name="EleutherAI/pythia-410m"
# model_name="cerebras/Cerebras-GPT-590M"
root_folder="/data/data/hgupta35/VUsable/dataset_difficulty/data/syndatagen"
subfolders=$(find "$root_folder" -maxdepth 1 -mindepth 1 -type d)

# Loop through each subfolder
for subfolder in $subfolders
do
  dataset_type=$(basename "${subfolder}")
  first_char="${dataset_type:0:1}"
  if [ "$first_char" == "v" ]; then
    echo "Dataset type: $dataset_type"
    data_set_names=$(find "$subfolder" -maxdepth 1 -mindepth 1 -type d)

    for data_set_name in $data_set_names
    do
      echo "Dataset name: $data_set_name"
      dataset_basename=$(basename "${data_set_name}")
      data_set_files=$(find "$data_set_name" -type f)

      for data_set_file in $data_set_files
        do
          dataset_file_base_name=$(basename "${data_set_file}")
          if [ "$dataset_file_base_name" == "std_train.csv" ]; then
            std_train_file=$data_set_file
          elif [ "$dataset_file_base_name" == "std_val.csv" ]; then
            std_val_file=$data_set_file
          elif [ "$dataset_file_base_name" == "null_train.csv" ]; then
            null_train_file=$data_set_file
          elif [ "$dataset_file_base_name" == "null_val.csv" ]; then
            null_val_file=$data_set_file
          fi
      done
      null_model_out_path="/data/data/hgupta35/VUsable/Models/"
      null_model_out_path+=$dataset_type
      std_model_out_path="/data/data/hgupta35/VUsable/Models/"
      std_model_out_path+=$dataset_type

      model_path_prefix="/"
      model_path_suffix="/"
      null_model_suffix="$model_name""_null"
      std_model_suffix="$model_name""_std"

      null_model_out_path+=$model_path_prefix
      null_model_out_path+=$dataset_basename
      null_model_out_path+=$model_path_suffix
      null_model_out_path+=$null_model_suffix

      std_model_out_path+=$model_path_prefix
      std_model_out_path+=$dataset_basename
      std_model_out_path+=$model_path_suffix
      std_model_out_path+=$std_model_suffix

      config_file="/config.json"
      echo "Null Model Out Path: $null_model_out_path"
      echo "Null Config File Path: $null_model_out_path$config_file"
      echo "Std Model Out Path: $std_model_out_path"
      echo "Std Config File Path: $std_model_out_path$config_file"
      echo "Train std: $std_train_file"
      echo "Eval std: $std_val_file"
      echo "Train null: $null_train_file"
      echo "Eval null: $null_val_file"

      if [ -e "$null_model_out_path$config_file" ]; then
          echo "----------- MODEL TRAINED ALREADY -----------"
      else
          python /data/data/hgupta35/VUsable/vusable_information/run_glue_no_trainer.py \
          --model_name_or_path $model_name \
          --tokenizer_name $model_name \
          --train_file $null_train_file \
          --validation_file $null_val_file \
          --per_device_train_batch_size 6 \
          --per_device_eval_batch_size 6 \
          --num_train_epochs 2 \
          --seed 1 \
          --output_dir $null_model_out_path
      fi

      if [ -e "$std_model_out_path$config_file" ]; then
          echo "----------- MODEL TRAINED ALREADY -----------"
      else
          python /data/data/hgupta35/VUsable/vusable_information/run_glue_no_trainer.py \
            --model_name_or_path $model_name \
            --tokenizer_name $model_name \
            --train_file $std_train_file \
            --validation_file $std_val_file \
            --per_device_train_batch_size 3 \
            --per_device_eval_batch_size 3 \
            --num_train_epochs 2 \
            --seed 1 \
            --output_dir $std_model_out_path
      fi

    done
    echo ""
  fi
done