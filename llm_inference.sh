#!/usr/bin/env bash
gpu_num="6"
num_of_instructs=20
version="test_project"
model_input_ratio=0.8
test_dir="/home/FullMouth/data/dataset/test_notes"
save_root_dir="/home/FullMouth/data/"

base_model="Qwen2.5-7B-Instruct"

selected_instructs=3
validation_threshold=0.9

  python LLMs_inferences.py \
    --version $version --gpu_num $gpu_num --base_model $base_model \
    --test_dir $test_dir --dst_root $save_root_dir \
    --num_of_instructions $selected_instructs --validation_threshold $validation_threshold \
    --add_stop_token --model_input_limit_ratio $model_input_ratio \
    --revised_training_set --include_description --include_examples \
    --result_type_dir "gold_prompt" 