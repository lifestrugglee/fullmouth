import os
import time
from glob import glob
from pathlib import Path
import yaml

from function_util import *
from fullmouth_util import (
    write_json, load_json, SENT, PRED, INSTRUCT_PROMPT
)

def main(model_dir):
    target_dir = get_model_name(args.base_model, args)
    src_model_dir_path = os.path.join(args.dst_root, args.version, target_dir)
    print(f'Source model dir path: {src_model_dir_path!r}')

    instruct_prompt_dict_fp = os.path.join(src_model_dir_path, 'instruct_prompt_dict.json')
    assert os.path.exists(instruct_prompt_dict_fp), f"File not found: {instruct_prompt_dict_fp}"
    instruct_dict = load_json(instruct_prompt_dict_fp)

    # Configure logging
    post_fix = f'{args.num_of_instructions}instructs{str(args.validation_threshold).replace(".", "")}'
    result_type_dir = f'{args.result_type_dir}_{post_fix}'
    result_dir_path = os.path.join(src_model_dir_path, result_type_dir)
    print(f'Results will be saved in {result_dir_path!r}')
    Path(result_dir_path).mkdir(parents=True, exist_ok=True)

    log_fp = os.path.join(src_model_dir_path, f'pred_{result_type_dir}.log')
    print(log_fp)
    setup_logger(log_fp)
    log_msg(args)
    llms_setup(model_dir, args)
    log_msg(f"result dir - {result_dir_path}")
    log_msg("*"*20, print_message=False)

    json_fp_ls = glob(os.path.join(args.test_dir, '*.json'))
    log_msg(f'Gold standard size: {len(json_fp_ls)}')
    if args.go_reverse:
        json_fp_ls = json_fp_ls[::-1]
        log_msg('Processing files in reverse order.')

    run_name = f'v{args.version}\n{target_dir}\n{result_type_dir}'
    print(run_name)

    # instruction prompt preparation
    selected_prompt_fp = os.path.join(src_model_dir_path, f'selected_prompt_{post_fix}.json')

    if not os.path.exists(selected_prompt_fp):
        instruct_dict_selected = instruct_prompt_preparation(args, instruct_dict)
        write_json(instruct_dict_selected, selected_prompt_fp)
    else:
        instruct_dict_selected = load_json(selected_prompt_fp)
        log_msg(f'Loaded selected instruction prompts from {selected_prompt_fp!r}')
    
    total_min = 0
    args.end_idx = args.end_idx if args.end_idx > 0 else len(json_fp_ls)
    json_fp_ls = json_fp_ls[args.start_idx:  args.end_idx]
    total_json = len(json_fp_ls)
    log_msg(f'Processing from idx {args.start_idx} to {args.end_idx}, total {total_json} files.')

    for json_idx, json_fp in enumerate(json_fp_ls):
        json_fn = os.path.basename(json_fp)
        dst_json_fp = os.path.join(result_dir_path, json_fn)
        if os.path.exists(dst_json_fp): continue

        gold_content_ls = load_json(json_fp)

        start_time = time.time()
        target_sent_dict = {idx: section_dict[SENT] for idx, section_dict in enumerate(gold_content_ls)}

        # ------------------------------
        for entity in fm_label.gold_entity_ls:
            instruct_ls = [instruct_dict_selected[entity][idx][INSTRUCT_PROMPT] for idx in instruct_dict_selected[entity]]
            threshold = len(instruct_ls) / 2
            
            # Phase 1: Check instructions for all sentences
            bool_prompt_result_ls = [
                get_bool_batch_result_ls(
                    batch_msg_ls=get_batch_msg_ls_checkInstruction(instruction_prompt, target_sent_dict, entity),
                    model_input_limit_ratio=args.model_input_limit_ratio
                )
                for instruction_prompt in instruct_ls
            ]
            bool_prompt_sum_result_ls = [sum(col) for col in zip(*bool_prompt_result_ls)]
            yes_sentence_dict = { sent_idx: sent 
                for sent_idx, sent in target_sent_dict.items()
                if bool_prompt_sum_result_ls[sent_idx] > threshold
            }
            if not yes_sentence_dict: continue
            # Phase 2: Get batch instruction results
            result_dict = get_batch_instruction_ls_result_ls(instruct_ls, yes_sentence_dict,
                                                            model_input_limit_ratio=args.model_input_limit_ratio,
                                                            add_stop_token=args.add_stop_token)
            
            # Build sentence-to-index mapping for O(1) lookup
            for sent_idx, e_data in result_dict.items():
                if e_data is None or not e_data.entity_ls:
                    continue

                if PRED not in gold_content_ls[sent_idx]:
                    gold_content_ls[sent_idx][PRED] = {}
                gold_content_ls[sent_idx][PRED][entity] = e_data.entity_ls

        # ------------------------------------------------------
        write_json(gold_content_ls, dst_json_fp)
        end_time = time.time()
        execution_time_min = (end_time - start_time)/60
        total_min += execution_time_min

        status_msg = f'{json_idx+1}/{total_json} is done with {execution_time_min:.2f} ({total_min:.2f}) mins!'
        log_msg(status_msg)

    log_msg(f'Gold {run_name} are done with {total_min:.2f} mins!')


if __name__ == "__main__":
    assert os.path.exists("config.yml"), "config.yml not found. Please create a config.yml file with the required configurations."
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)

    model_root = config['model_root']

    # Apply arguments
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    if args.model_name:
        model_dir = os.path.join(model_root, args.version, args.model_name)
    else:
        model_dir = os.path.join(model_root, args.base_model)

    if args.checkpoint_dir:
        model_dir = os.path.join(model_dir, args.checkpoint_dir)
    
    main(model_dir)