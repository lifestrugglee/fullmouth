# %cd /home/FullMouth/code/
import os
import sys
import time
import shutil
from glob import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml
from function_util import *

from fullmouth_util import (
    INSTRUCT_PROMPT, EVAL_MTX, TEST_MTX, is_similarity, write_json, load_json
)

########################################################################################
def main():
    target_dir = get_model_name(args.base_model, args)
    dst_dir = os.path.join(args.dst_root, args.version, target_dir)
    #####################################################################
    # Reset everytime
    if args.reset: 
        print('Resetting the directory...')
        shutil.rmtree(dst_dir, ignore_errors=True)
    #####################################################################
    Path(dst_dir).mkdir(parents=True, exist_ok=True)

    run_name = f'{args.version}_{target_dir}'

    # Configure logging
    current_time = time.strftime("%Y%m%d_%H%M")
    log_fp = os.path.join(dst_dir, f'Instruction_{current_time}.log')
    print(log_fp)
    setup_logger(log_fp)
    log_msg(args)
    log_msg(f"MODEL_NAME: {args.model_name}", print_message=False)
    log_msg(args, print_message=False)
    log_msg("*"*20, print_message=False)
    llms_setup(model_dir, args)

    json_fp_ls = glob(os.path.join(args.train_data_path, '*.json'))

    data_ls = []
    for json_fp in json_fp_ls:
        data_ls.extend(load_json(json_fp))
    log_msg(f'Training set size: {len(json_fp_ls)} with {len(data_ls)} sentencess')
    '''
    data_ls example:
    [{'sent': 'D: 16 y/o male presents with father for Recall.',
    'sent_key': '.txt_0_0',
    'entity_ls': ['Age', 'Gender'],
    'entity_dict': {'Age': ['16 y/o'], 'Gender': ['male']}},
    '''
    instruct_prompt_dict_fp = os.path.join(dst_dir, 'instruct_prompt_dict.json')
    instruct_prompt_dict = {}
    if os.path.exists(instruct_prompt_dict_fp): instruct_prompt_dict = load_json(instruct_prompt_dict_fp)

    for key_entity in fm_label.gold_entity_ls[:2]:
        key_entity_desc = fm_label.get_entity_description(key_entity) if args.include_description else None
        entity_data, no_entity_data = get_entity_data(data_ls, key_entity)

        entity_data = simplify_entity_data(entity_data, key_entity)
        no_entity_data = simplify_entity_data(no_entity_data, '')
        log_msg("*"*20)
        log_msg(f"{key_entity} - Entity data size: {len(entity_data)}, No entity data size: {len(no_entity_data)}")

        start_time = time.time()
        if len(entity_data) >= 10:
            # split positives
            pos_train, pos_tmp = train_test_split(entity_data, test_size=0.2, random_state=42)
            pos_val  , pos_test = train_test_split(pos_tmp, test_size=0.5, random_state=42)

            # split negatives
            neg_train, neg_tmp = train_test_split(no_entity_data, test_size=0.2, random_state=42)
            neg_val, neg_test = train_test_split(neg_tmp, test_size=0.5, random_state=42)

            if args.revised_training_set: 
                log_msg('Revising the training set...')
                log_msg(f'[Before revision] len(pos_train) - {len(pos_train)}')
                pos_train.sort(key=lambda x: x.entity_ls)
                new_pos_train = getRevisedTrainingSet(pos_train, add_stop_token=args.add_stop_token, isDebug=args.debug_mode)
                pos_train = [e for e in pos_train if e in new_pos_train]
                log_msg(f'[After  revision] len(pos_train) - {len(pos_train)}')

            log_msg(f"Positive data size : Training set size: {len(pos_train):3d}, Validation set size: {len(pos_val):3d}, Test set size: {len(pos_test):3d}")
            train_data = build_mixed_split(pos_train, neg_train, neg_ratio=args.train_neg_ratio, seed=42)
            val_data = build_mixed_split(pos_val, neg_val, neg_ratio=args.val_neg_ratio, seed=43)
            test_data = build_mixed_split(pos_test, neg_test, neg_ratio=args.test_neg_ratio, seed=44)
            log_msg(f'Mixed neg data size: Training set size: {len(train_data):3d}, Validation set size: {len(val_data):3d}, Test set size: {len(test_data):3d}')

            instruct_prompt_dict[key_entity] = {}
            for num_instruct in range(1, args.num_of_instructions+1):
                log_msg(f'Generating # {num_instruct}/{args.num_of_instructions} instruction for {key_entity}')
                instruct_prompt_dict[key_entity][num_instruct] =  {}
                indent_level = 1

                isPass = False; count = 0            
                best_validation_result_dict = {INSTRUCT_PROMPT: "", EVAL_MTX: dict()}
                revised_list = []
                while count < ROUND_THRES and not isPass:
                    log_msg(f'round - {count+1}', indent_level=indent_level)
                    indent_level = 2
                    if count > 0 and args.error_feedback_in_loop:
                        error_ls = eval_metrix['pos_error_cases'] + eval_metrix['neg_error_cases'] 
                        revised_list = [error['e_data'].get_json_str() for error in error_ls]

                    instruction_prompt = getInstruction(train_data, key_entity, entity_desc=key_entity_desc, 
                                        instruction_length=args.instruction_length, revised_list=revised_list,
                                        include_description=args.include_description,
                                        include_examples=args.include_examples,
                                        add_stop_token=args.add_stop_token
                                        )

                    isGoodPrompt = verifyInstruction(instruction_prompt, key_entity, isDebug=False)
                    if not isGoodPrompt:
                        log_msg(f'{key_entity} Instruction Prompt fail the verification... revising', indent_level=indent_level)
                        instruction_prompt = reviseInstruction(instruction_prompt, key_entity, add_stop_token=args.add_stop_token, 
                                                include_examples=args.include_examples, isDebug=False)
                        
                    try:
                        target_sent_dict = {t_idx: t.sentence for t_idx, t in enumerate(val_data)}
                        bool_output_ls = get_bool_batch_result_ls(batch_msg_ls=get_batch_msg_ls_checkInstruction(instruction_prompt, target_sent_dict, key_entity),
                                                model_input_limit_ratio=args.model_input_limit_ratio)
                        checked_sent_dict = {sent_idx: sent for sent_idx, sent in target_sent_dict.items() if bool_output_ls[sent_idx]}
                        result_dict = get_batch_InstructionResult_ls(instruction_prompt, checked_sent_dict,
                                                                        model_input_limit_ratio=args.model_input_limit_ratio,
                                                                        add_stop_token=args.add_stop_token)

                        eval_metrix = evaluate_mixed_entity_extraction(val_data, result_dict, is_similarity)
                        log_msg(f"Val_data f1: {eval_metrix['f1']:.2f} w/t ratio: {args.validation_threshold}", indent_level=indent_level)
                        log_msg(f"Validation set evaluation: Precision: {eval_metrix['precision']:.2f}, Recall: {eval_metrix['recall']:.2f}, F1: {eval_metrix['f1']:.2f}, Neg Sent Acc: {eval_metrix['neg_acc']:.2f}, Pos Sent Exact Acc: {eval_metrix['pos_sent_exact_acc']:.2f}", indent_level=indent_level)    

                        if eval_metrix['f1'] >= args.validation_threshold:
                            isPass = True

                        if eval_metrix['f1'] > best_validation_result_dict[EVAL_MTX].get('f1', 0.0):
                            best_validation_result_dict[INSTRUCT_PROMPT] = instruction_prompt
                            best_validation_result_dict[EVAL_MTX] = eval_metrix

                    except Exception as e:
                        log_msg( "".join(traceback.format_exception(type(e), e, e.__traceback__)), isError=True, indent_level=indent_level)
                        log_msg("fail", indent_level=indent_level)
                    
                    count += 1

                instruction_prompt = best_validation_result_dict[INSTRUCT_PROMPT]
                instruct_prompt_dict[key_entity][num_instruct][INSTRUCT_PROMPT] = instruction_prompt
                instruct_prompt_dict[key_entity][num_instruct][EVAL_MTX] = best_validation_result_dict[EVAL_MTX]
                
                #######################################################################################
                # test on test set w/t negative samples
                target_sent_dict = {t_idx: t.sentence for t_idx, t in enumerate(test_data)}
                bool_output_ls = get_bool_batch_result_ls(batch_msg_ls=get_batch_msg_ls_checkInstruction(instruction_prompt, target_sent_dict, key_entity),
                                        model_input_limit_ratio=args.model_input_limit_ratio)
                checked_sent_dict = {sent_idx: sent for sent_idx, sent in target_sent_dict.items() if bool_output_ls[sent_idx]}
                result_dict = get_batch_InstructionResult_ls(instruction_prompt, checked_sent_dict,
                                                                model_input_limit_ratio=args.model_input_limit_ratio,
                                                                add_stop_token=args.add_stop_token)

                eval_metrix = evaluate_mixed_entity_extraction(test_data, result_dict, is_similarity)
                instruct_prompt_dict[key_entity][num_instruct][TEST_MTX] = eval_metrix
                log_msg(f"Test set evaluation: Precision: {eval_metrix['precision']:.2f}, Recall: {eval_metrix['recall']:.2f}, F1: {eval_metrix['f1']:.2f}, Neg Sent Acc: {eval_metrix['neg_acc']:.2f}, Pos Sent Exact Acc: {eval_metrix['pos_sent_exact_acc']:.2f}", indent_level=indent_level)

        else:# Not enough entity data
            instruct_prompt_dict[key_entity] = {}
            # no split positives
            pos_train = entity_data.copy()

            # split negatives
            neg_train, neg_tmp = train_test_split(no_entity_data, test_size=0.2, random_state=42)
            neg_val, neg_test = train_test_split(neg_tmp, test_size=0.5, random_state=42)

            log_msg(f"Positive data size : Training set size: {len(pos_train):3d}, Validation set size: {len(pos_val):3d}, Test set size: {len(pos_test):3d}")
            train_data = build_mixed_split(pos_train, neg_train, neg_ratio=args.train_neg_ratio, seed=42)
            test_data = build_mixed_split(pos_train, neg_test, neg_ratio=args.test_neg_ratio, seed=44)
            log_msg(f'Mixed neg data size: Training set size: {len(train_data):3d}, Validation set size: N/A, Test set size: {len(test_data):3d}')

            for num_instruct in range(1, args.num_of_instructions+1):
                log_msg(f'Generating # {num_instruct}/{args.num_of_instructions} instruction for {key_entity}', indent_level=indent_level)
                instruct_prompt_dict[key_entity][num_instruct] = {}
                indent_level = 1
                instruction_prompt = getInstruction(train_data, key_entity, entity_desc=key_entity_desc, 
                                                    instruction_length=args.instruction_length, 
                                                    include_description=args.include_description,
                                                    include_examples=args.include_examples,
                                                    add_stop_token=args.add_stop_token,)


                isGoodPrompt = verifyInstruction(instruction_prompt, key_entity, isDebug=False)
                if not isGoodPrompt:
                    log_msg(f'{key_entity} Instruction Prompt fail the verification... revising')
                    instruction_prompt = reviseInstruction(instruction_prompt, key_entity, add_stop_token=args.add_stop_token, 
                                            include_examples=args.include_examples, isDebug=False)

                instruct_prompt_dict[key_entity][num_instruct][INSTRUCT_PROMPT] = instruction_prompt

                target_sent_dict = {t_idx: t.sentence for t_idx, t in enumerate(test_data)}
                bool_output_ls = get_bool_batch_result_ls(
                    batch_msg_ls=get_batch_msg_ls_checkInstruction(instruction_prompt, 
                                                                   target_sent_dict, 
                                                                   key_entity),
                    model_input_limit_ratio=args.model_input_limit_ratio)
                checked_sent_dict = {sent_idx: sent for sent_idx, sent in target_sent_dict.items() if bool_output_ls[sent_idx]}
                result_dict = get_batch_InstructionResult_ls(instruction_prompt, checked_sent_dict,
                                                             model_input_limit_ratio=args.model_input_limit_ratio,
                                                             add_stop_token=args.add_stop_token)

                eval_metrix = evaluate_mixed_entity_extraction(test_data, result_dict, is_similarity)
                instruct_prompt_dict[key_entity][num_instruct][TEST_MTX] = eval_metrix
                log_msg(f"Test set evaluation: Precision: {eval_metrix['precision']:.2f}, Recall: {eval_metrix['recall']:.2f}, F1: {eval_metrix['f1']:.2f}, Neg Sent Acc: {eval_metrix['neg_acc']:.2f}, Pos Sent Exact Acc: {eval_metrix['pos_sent_exact_acc']:.2f}", indent_level=indent_level)

        end_time = time.time()
        execution_time = end_time - start_time

        log_msg(f"Execution time: {execution_time:.1f} seconds", indent_level=indent_level)
        instruct_prompt_dict[key_entity]['exec_time'] = execution_time
        print(instruct_prompt_dict)
        write_json(instruct_prompt_dict, instruct_prompt_dict_fp)
        printTokensUpdate()

    final_msg = f"{run_name} - Completed"
    log_msg(final_msg)

if __name__ == "__main__":

    assert os.path.exists("config.yml"), "config.yml not found. Please create a config.yml file with the required configurations."
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)

    model_root = config['model_root']
    
    args = parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    if args.model_name:
        model_dir = os.path.join(model_root, args.model_name)
    else:
        model_dir = os.path.join(model_root, args.base_model)

    if args.checkpoint_dir:
        model_dir = os.path.join(model_dir, args.checkpoint_dir)

    main()
