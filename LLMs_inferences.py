# %cd /home/FullMouth/code/
import os
import sys
import time
# import shutil
# import random
from glob import glob
from pathlib import Path


from function_util_v4 import *
from fullmouth_util import (
    savePickle, loadPickle, keep_longest_fuzzy, SENT, PRED, INSTRUCT_PROMPT
)

sys.path.append(r'/home/tg_bot')
from tg_bot_send import send_msg_https

root_dir = r'/home/FullMouth'
src_txt_root = r'/home/FullMouth/data/notes/UTH_notes_type_filter'
model_root = r'/data_sys/lang_model'

def main(gold_dir, model_dir):
    target_dir = get_model_name(args.base_model, args)
    src_model_dir_path = os.path.join(root_dir, 'data', f'instruct_{args.version}', target_dir)
    print(f'Source model dir path: {src_model_dir_path!r}')

    instruct_prompt_dict_fp = os.path.join(src_model_dir_path, 'instruct_prompt_dict.pkl')
    assert os.path.exists(instruct_prompt_dict_fp), f"File not found: {instruct_prompt_dict_fp}"
    instruct_dict = loadPickle(instruct_prompt_dict_fp)

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

    pkl_fp_ls = glob(os.path.join(gold_dir, '*.pkl'))
    log_msg(f'Gold standard size: {len(pkl_fp_ls)}')
    if args.go_reverse:
        pkl_fp_ls = pkl_fp_ls[::-1]
        log_msg('Processing files in reverse order.')

    run_name = f'v{args.version}\n{target_dir}\n{result_type_dir}'
    print(run_name)

    # instruction prompt preparation
    selected_prompt_fp = os.path.join(src_model_dir_path, f'selected_prompt_{post_fix}.pkl')
    if not os.path.exists(selected_prompt_fp):
        instruct_dict_selected = instruct_prompt_preparation(args, instruct_dict)
        savePickle(selected_prompt_fp, instruct_dict_selected)
    else:
        instruct_dict_selected = loadPickle(selected_prompt_fp)
        log_msg(f'Loaded selected instruction prompts from {selected_prompt_fp!r}')
    
    total_min = 0
    args.end_idx = args.end_idx if args.end_idx > 0 else len(pkl_fp_ls)
    pkl_fp_ls = pkl_fp_ls[args.start_idx:  args.end_idx]
    total_pkl = len(pkl_fp_ls)
    log_msg(f'Processing from idx {args.start_idx} to {args.end_idx}, total {total_pkl} files.')

    for pkl_idx, pkl_fp in enumerate(pkl_fp_ls):
        pkl_fn = os.path.basename(pkl_fp)
        dst_pkl_fp = os.path.join(result_dir_path, pkl_fn)
        if os.path.exists(dst_pkl_fp): continue

        gold_content_ls = loadPickle(pkl_fp)

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
        savePickle(dst_pkl_fp, gold_content_ls)
        end_time = time.time()
        execution_time_min = (end_time - start_time)/60
        total_min += execution_time_min

        status_msg = f'{pkl_idx+1}/{total_pkl} is done with {execution_time_min:.2f} ({total_min:.2f}) mins!'
        log_msg(status_msg)
        if pkl_idx > 0 and pkl_idx%200 == 0:
            send_msg_https(f'[{run_name}] Gold {status_msg}')

    send_msg_https(f'Gold {run_name} are done with {total_min:.2f} mins!')


if __name__ == "__main__":
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

    if args.run_train:
        gold_dir = os.path.join(root_dir, 'data', 'annotation_v4', 'training_data_swap_revised')
        args.result_type_dir = f'train_{args.result_type_dir}'
    elif args.run_test:
        # gold_dir = os.path.join(root_dir, 'data', 'annotation_v4', 'test_data_swap_revised')
        # gold_dir = os.path.join(root_dir, 'data', 'annotation_v4', 'test_notes_part1n2')
        gold_dir = os.path.join(root_dir, 'data', 'annotation_v4', 'test_notes_1000')
        args.result_type_dir = f'gold_{args.result_type_dir}'
    # elif args.run_extra:
        
    #     args.result_type_dir = f'gold_{args.result_type_dir}'

    main(gold_dir, model_dir)