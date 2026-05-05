#!/usr/bin/env bash
gpu_num="0"
num_of_instructs=3
version="test_project"
model_input_ratio=0.8
validation_threshold=0.8
training_dir="/home/FullMouth/data/dataset/training_notes/"
save_root_dir="/home/FullMouth/data/"
base_model="Qwen2.5-7B-Instruct"

python LLMs_prompt_generation.py \
    --version $version --reset \
    --train_data_path $training_dir --dst_root $save_root_dir \
    --gpu_num $gpu_num --base_model $base_model \
    --num_of_instructions $num_of_instructs --instruction_length 500 --validation_threshold $validation_threshold \
    --add_stop_token --model_input_limit_ratio $model_input_ratio \
    --revised_training_set --include_description --include_examples --error_feedback_in_loop
    