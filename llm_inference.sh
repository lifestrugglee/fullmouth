#!/usr/bin/env bash
gpu_num="5"
num_of_instructs=20
save_root_dir="/home/FullMouth/data/"
version="UTH_2025_pred"
model_input_ratio=0.8
test_dir="/home/FullMouth/data/notes/UTH_2025_notes17578"

base_model="Qwen2.5-14B-Instruct"
model_path="/data_sys/lang_model/4_n20/Qwen2.5-14B-Instruct_revisedTrain_withDesc_withEx_withErrFeed_ep04r16_medium_DPO_ep06"
src_dir_path="/home/FullMouth/data/UTH_2025_pred/Qwen2.5-14B-Instruct_revisedTrain_withDesc_withEx_withErrFeed" # Optional, if not provided, it will be set to dst_root/version/target_dir
selected_instructs=3
validation_threshold=0.9

for i in $(seq 14000 1000 18000); do
# for i in $(seq 10 3 16); do
  python LLMs_inferences.py \
  --version $version --gpu_num $gpu_num --base_model $base_model \
  --model_path $model_path --src_dir_path $src_dir_path \
  --test_dir $test_dir --dst_root $save_root_dir \
  --num_of_instructions $selected_instructs --validation_threshold $validation_threshold \
  --add_stop_token --model_input_limit_ratio $model_input_ratio \
  --revised_training_set --include_description --include_examples --error_feedback_in_loop \
  --result_type_dir "pred_dpo_2025" --start_idx $i --end_idx $(($i+1000))

  python /home/tg_bot/tg_bot_send.py "2025-$i inferences done!"
done