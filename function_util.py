import traceback
import json
import ast
import torch
import logging
from logging.handlers import RotatingFileHandler
import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Union

try:
    from py_files.fullmouth_util import (
    FM_label, ROUND_THRES, ENTITY_LS, ENTITY_DICT, OUTPUT_FORMAT, SENT, STOP, TEST_MTX
)
except ImportError:
    from fullmouth_util import (
        FM_label, ROUND_THRES, ENTITY_LS, ENTITY_DICT, OUTPUT_FORMAT, SENT, STOP, TEST_MTX
    )

fm_label = FM_label()

# ########################################################################################
# Data Classes
@dataclass(slots=True)
class EntityData:
    """Structured container for entity extraction data."""
    sentence: str
    entity_ls: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for LLM prompts."""
        return {SENT: self.sentence, ENTITY_LS: self.entity_ls}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EntityData':
        """Create from dictionary."""
        return cls(
            sentence=data.get(SENT, data.get('sentence', '')),
            entity_ls=data.get(ENTITY_LS, data.get('entity_ls', []))
        )
    
    def __str__(self) -> str:
        # keeps it predictable and consistent with prompt formatting
        return self.get_json_str()

    def get_json_str(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    def update(self, entity_ls: list) -> None:
        """Extend entities and dedupe while preserving first-seen order."""
        if not entity_ls:
            return
        combined = self.entity_ls + entity_ls
        self.entity_ls = list(dict.fromkeys(combined))

    def sentence_equals(self, other: Union["EntityData", str]) -> bool:
        """Return True if this sentence exactly equals the other sentence."""
        other_sentence = other.sentence if isinstance(other, EntityData) else str(other)
        return self.sentence.strip() == other_sentence.strip()
# ########################################################################################
tokenizer = None; model = None; device = None; no_sys_prompt = False
global key_entity_prompts_tokens_count, key_entity_output_tokens_count
key_entity_prompts_tokens_count = 0
key_entity_output_tokens_count = 0
model_max_window = -1

TF_MAX_NEW_TOKENS = 1
OUTPUT_MULTIPLIER = 3

# Functions 1/ 3 ########################################################################################
def setup_logger(log_fp, logger_name='FullMouth', max_bytes=10*1024*1024, backup_count=5):
    """Configure logger with rotation support."""
    app_logger = logging.getLogger(logger_name)
    app_logger.setLevel(logging.INFO)
    app_logger.handlers.clear()  # Remove existing handlers
    
    # Rotating file handler
    file_handler = RotatingFileHandler(log_fp, maxBytes=max_bytes, backupCount=backup_count)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    app_logger.addHandler(file_handler)
    return app_logger

def log_msg(message, print_message=True, isError=False, indent_level=0):
    app_logger = logging.getLogger('FullMouth')

    message = ' ' * (indent_level * 2) + str(message)
    if print_message:
        print(message)
    if isError:
        app_logger.error(message)
    else:
        app_logger.info(message)

def get_model_name(model_name:str, args) -> str:
    target_dir = model_name
    if args.revised_training_set:
        target_dir += '_revisedTrain'
    if args.include_description:
        target_dir += '_withDesc'
    if args.include_examples:
        target_dir += '_withEx'
    if args.error_feedback_in_loop:
        target_dir += '_withErrFeed'
    return target_dir

def auto_transformers(model_dir):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

def mistral_llm_init(model_dir):
    from transformers import Mistral3ForConditionalGeneration, FineGrainedFP8Config, MistralCommonBackend

    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_dir,
        device_map="auto",
        quantization_config=FineGrainedFP8Config(dequantize=True)
    )
    tokenizer = MistralCommonBackend.from_pretrained(model_dir)
    return model, tokenizer


MODEL_DISPATCH = {
    "mistral3": mistral_llm_init,
}

def llms_setup(_model_dir, args):# _key_entity_prompts_tokens_count, _key_entity_output_tokens_count):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    global device, STOP, tokenizer, model, model_max_window, no_sys_prompt
    no_sys_prompt = args.no_system_prompt
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_msg(f'Device - {device}')
    tokenizer = AutoTokenizer.from_pretrained(_model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        _model_dir, dtype="auto", device_map="auto", local_files_only=True
    )
    
    def ensure_eos_and_pad(tokenizer, model, eos_fallback="</s>", padding_side="left", log_fn=print):
        # 1) Ensure EOS exists
        if tokenizer.eos_token is None:
            log_fn(f"Tokenizer has no EOS token. Adding eos_token={eos_fallback!r} and resizing embeddings.")
            tokenizer.add_special_tokens({"eos_token": eos_fallback})

            # Resize if tokenizer size changed vs model embeddings
            n_tok = len(tokenizer)
            n_emb = model.get_input_embeddings().num_embeddings
            if n_tok != n_emb:
                model.resize_token_embeddings(n_tok)

        # 2) Ensure PAD exists (use EOS as PAD for causal LMs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            log_fn(f"Using EOS token as PAD token: pad_token={tokenizer.pad_token!r}")

        # 3) Keep model config in sync (helps silence warnings)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

        # 4) Padding side
        tokenizer.padding_side = padding_side
        return tokenizer, model
    
    tokenizer, model = ensure_eos_and_pad(tokenizer, model, eos_fallback=STOP, log_fn=log_msg)
    
    def get_real_context_window(model, tokenizer=None) -> int:
        """Determine the effective context window size for a model.
        Handles nested configs (e.g., multimodal models with text_config).
        """
        config = model.config

        # For multimodal/composite models, prefer the text sub-config
        text_config = getattr(config, "text_config", config)

        max_pos = getattr(text_config, "max_position_embeddings", None) or \
                getattr(config, "max_position_embeddings", None)

        # Check RoPE scaling — only scale for types that genuinely extend context
        rope_scaling = getattr(text_config, "rope_scaling", None) or \
                    getattr(config, "rope_scaling", None)
        if rope_scaling and max_pos:
            rope_type = rope_scaling.get("type", rope_scaling.get("rope_type", ""))
            factor = rope_scaling.get("factor", 1.0)
            if rope_type in ("dynamic", "linear", "yarn"):
                return int(max_pos * factor)

        if max_pos:
            return max_pos

        # Fallback: check tokenizer model_max_length (filter sentinel values like 1e30)
        if tokenizer is not None:
            tok_max = getattr(tokenizer, "model_max_length", None)
            if tok_max and tok_max < 1e10:
                return tok_max

        return 2048  # last-resort default

    model_max_window = get_real_context_window(model, tokenizer)

def instruct_prompt_preparation(args: argparse.Namespace, instruct_dict: dict, eval_name = 'f1', makeup = False) -> dict[str, dict[int, dict]]:
    instruct_prompt_dict = {}
    target_num = args.num_of_instructions
    for entity in fm_label.gold_entity_ls:
        entity_prompt_dict = {}
        fail_entity_prompt_dict = {}
        for idx, prompt_dict in instruct_dict[entity].items():
            if isinstance(idx, int) or (isinstance(idx, str) and idx.isdigit()):
                idx = int(idx)
                if TEST_MTX not in prompt_dict:
                    # for v3, only
                    raise ValueError(f"{TEST_MTX} not found in prompt_dict for {entity} idx {idx}")
                if prompt_dict[TEST_MTX][eval_name] >= args.validation_threshold:
                    entity_prompt_dict[idx] = prompt_dict
                else:
                    fail_entity_prompt_dict[idx] = prompt_dict

        num_valid = len(entity_prompt_dict)

        if num_valid >= target_num:
            # Pick top valid prompts by score
            valid_prompt_ls = list(entity_prompt_dict.items())
            valid_prompt_ls.sort(key=lambda x: x[1][TEST_MTX][eval_name], reverse=True)
            selected_instruction_prompts = dict(valid_prompt_ls[:target_num])
            log_msg(f'{entity} - Selecting top {target_num} prompts from {num_valid} valid prompts')
        else:
            log_msg(f'{entity} - Num valid prompts selected: {num_valid}')
            selected_instruction_prompts = entity_prompt_dict.copy()

            missing = target_num - num_valid
            fail_entity_prompt_ls = list(fail_entity_prompt_dict.items())
            fail_entity_prompt_ls.sort(key=lambda x: x[1][TEST_MTX][eval_name], reverse=True)
            if missing > 0 and args.makeup:
                log_msg(
                    f'{entity} - Not enough valid instruction prompts, selecting '
                    f'{missing} top prompts from failed prompts to make up {target_num}'
                )
                selected_instruction_prompts.update(dict(fail_entity_prompt_ls[:missing]))
            elif num_valid == 0:
                # No valid prompt, directly select from failed prompts
                log_msg(
                    f'{entity} - No valid instruction prompts, selecting '
                    f'{missing} top prompts from failed prompts'
                )
                selected_instruction_prompts.update(dict(fail_entity_prompt_ls[:missing]))


        for k, v in selected_instruction_prompts.items():
            log_msg(f'{entity} - Selecting prompt {k:2d} with {eval_name} {v[TEST_MTX][eval_name]:.3f}')
            
        instruct_prompt_dict[entity] = selected_instruction_prompts
        log_msg('=='*20)
    return instruct_prompt_dict

def simplify_entity_data(entity_data: List[Dict[str, Any]], key_entity: str) -> List[EntityData]:
    """Simplify raw entity data into EntityData objects.
    
    Args:
        entity_data: List of raw annotation dictionaries
        key_entity: Target entity type to extract (empty string for negative samples)
    
    Returns:
        List of EntityData objects containing sentence and entity list
    """
    simplified_data = []
    for data in entity_data:
        if key_entity == '':
            simplified_data.append(EntityData(
                sentence=data[SENT],
                entity_ls=[]
            ))
        elif key_entity in data[ENTITY_DICT]:
            simplified_data.append(EntityData(
                sentence=data[SENT],
                entity_ls=data[ENTITY_DICT].get(key_entity, [])
            ))
    return simplified_data

def printTokensUpdate():
    # global key_entity_prompts_tokens_count, key_entity_output_tokens_count
    log_msg(f"key_entity_prompts_tokens_count: {key_entity_prompts_tokens_count}")
    log_msg(f"key_entity_output_tokens_count: {key_entity_output_tokens_count}")

def get_entity_data(data_ls: List[Dict[str, Any]], key_entity: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    matched = []
    not_matched = []
    for data in data_ls:
        if key_entity in data[ENTITY_DICT] and len(data[ENTITY_DICT][key_entity]) > 0:
            matched.append(data)
        else:
            not_matched.append(data)
    return matched, not_matched

def sample_negatives(neg_data, k, seed=42):
    if k <= 0:
        return []
    rng = random.Random(seed)
    k = min(k, len(neg_data))
    return rng.sample(neg_data, k)


def build_mixed_split(pos_data, neg_data, neg_ratio=None, seed=42) -> list[EntityData]:
    if neg_ratio is None:
        sampled_neg = list(neg_data)
    else:
        sampled_neg = sample_negatives(neg_data, int(len(pos_data) * neg_ratio), seed)

    mixed = list(pos_data) + sampled_neg
    rng = random.Random(seed)
    rng.shuffle(mixed)
    return mixed

def evaluate_mixed_entity_extraction(x_data, result_dict, is_similarity):
    """
    Returns entity-level precision / recall / F1 on mixed positive+negative sentence data,
    and collects positive / negative error cases.

    Error case format:
    - pos_error_cases: positive samples with FN and/or FP
    - neg_error_cases: negative samples with any predicted entities
    """

    tp = 0
    fp = 0
    fn = 0

    neg_sent_total = 0
    neg_sent_correct = 0

    pos_sent_total = 0
    pos_sent_exact_correct = 0

    pos_error_cases = []
    neg_error_cases = []

    for e_idx, e_data in enumerate(x_data):
        gold_entities = e_data.entity_ls or []

        pred_entities = []
        if e_idx in result_dict and result_dict[e_idx]:
            pred_entities = result_dict[e_idx].entity_ls or []

        # -------------------------
        # negative sentence
        # -------------------------
        if len(gold_entities) == 0:
            neg_sent_total += 1

            if len(pred_entities) == 0:
                neg_sent_correct += 1
            else:
                fp += len(pred_entities)
                neg_error_cases.append({
                    'e_idx': e_idx,
                    'e_data': e_data,
                    'gold_entities': gold_entities,
                    'pred_entities': pred_entities,
                })
            continue

        # -------------------------
        # positive sentence
        # -------------------------
        pos_sent_total += 1

        matched_gold = [False] * len(gold_entities)
        matched_pred = [False] * len(pred_entities)

        # greedy 1-to-1 matching
        for g_idx, gold_entity in enumerate(gold_entities):
            for p_idx, pred_entity in enumerate(pred_entities):
                if matched_pred[p_idx]:
                    continue

                if is_similarity(gold_entity, [pred_entity]):
                    matched_gold[g_idx] = True
                    matched_pred[p_idx] = True
                    tp += 1
                    break

        cur_fn = matched_gold.count(False)
        cur_fp = matched_pred.count(False)

        fn += cur_fn
        fp += cur_fp

        if all(matched_gold) and all(matched_pred):
            pos_sent_exact_correct += 1
        else:
            missed_gold_entities = [
                gold_entities[i]
                for i, is_matched in enumerate(matched_gold)
                if not is_matched
            ]
            extra_pred_entities = [
                pred_entities[i]
                for i, is_matched in enumerate(matched_pred)
                if not is_matched
            ]

            pos_error_cases.append({
                'e_idx': e_idx,
                'e_data': e_data,
                'gold_entities': gold_entities,
                'pred_entities': pred_entities,
                'missed_gold_entities': missed_gold_entities,
                'extra_pred_entities': extra_pred_entities,
            })

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    neg_acc = neg_sent_correct / neg_sent_total if neg_sent_total else 0.0
    pos_sent_exact_acc = (
        pos_sent_exact_correct / pos_sent_total if pos_sent_total else 0.0
    )

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'neg_sent_total': neg_sent_total,
        'neg_sent_correct': neg_sent_correct,
        'neg_acc': neg_acc,
        'pos_sent_total': pos_sent_total,
        'pos_sent_exact_correct': pos_sent_exact_correct,
        'pos_sent_exact_acc': pos_sent_exact_acc,
        # 'pos_error_cases': pos_error_cases,
        # 'neg_error_cases': neg_error_cases,
    }


def replace_msg_ls_to_user(msg_ls: List[dict]) -> List[dict]:
    """
    Some models require user/assistant only
    """
    all_content_ls = [m_dict['content'] for m_dict in msg_ls]
    msg_ls = [{"role": "user", "content":  "\n".join(all_content_ls)}]
    return msg_ls


# def replace_msg_ls_to_user(msg_ls: List[dict]) -> List[dict]:
#     """
#     Some models require user/assistant only
#     """
#     for i, msg in enumerate(msg_ls):
#         if msg['role'] == 'system':
#             msg_ls[i]['role'] = 'user'
    
#     # Combine consecutive messages with the same role into one message
#     combined_msg_ls = []
#     for msg in msg_ls:
#         if not combined_msg_ls:
#             combined_msg_ls.append(msg)
#         else:
#             last_msg = combined_msg_ls[-1]
#             if msg['role'] == last_msg['role']:
#                 # Combine the content of the messages
#                 combined_msg_ls[-1]['content'] += '\n' + msg['content']
#             else:
#                 combined_msg_ls.append(msg)
    
#     return combined_msg_ls


def revised_entity_list(entity_ls, add_stop_token=True, isDebug=False):
    """
    Removes duplicate or highly similar entities from a list of dictionaries 
    using an LLM prompt.
    """
    global key_entity_prompts_tokens_count, key_entity_output_tokens_count
    assert tokenizer is not None and model is not None, "Tokenizer and model must be initialized."
    # --- Build prompt ---
    sys_prompt = (
    "You are an expert in dentistry and natural language processing.\n"
    "\n"
    "Input:\n"
    "- A list of dictionaries. Each dictionary has:\n"
    f"    - '{SENT}': the source sentence (string)\n"
    f"    - '{ENTITY_LS}': the extracted span text (string)\n"
    "\n"
    "Task:\n"
    "Deduplicate entries so that only unique entity mentions remain, using BOTH span text and\n"
    "its sentence context. Treat two entries as duplicates if ANY of the following are true:\n"
    "  1) Exact match on both 'sent' and 'entity' (after normalization), or\n"
    "  2) Same 'sent' (after normalization) AND the 'entity' strings are highly similar, or\n"
    "  3) The 'entity' strings are highly similar AND the sentences are paraphrases of each other\n"
    "     with equivalent dental meaning (e.g., domain synonyms like 'tooth #14' vs 'maxillary left first molar').\n"
    "\n"
    "Normalization rules (apply before comparing):\n"
    "- Trim whitespace.\n"
    "- Case-insensitive comparisons.\n"
    "- Collapse multiple spaces to one.\n"
    "- Remove leading/trailing punctuation around the entity span.\n"
    "- For similarity, consider token overlap/fuzzy matching so minor typos or inflections count as similar.\n"
    "\n"
    "Similarity thresholds (guidance):\n"
    "- For 'entity' similarity, treat as duplicate if token-level Jaccard ≥ 0.8 or a fuzzy ratio ≥ 90.\n"
    "- For sentence paraphrase, require strong overlap in dental terms, tooth numbers, surfaces (MO/DO/OC),\n"
    "  and procedures (e.g., 'RCT', 'root canal'), ignoring stopwords.\n"
    "\n"
    "Tie-breaking (when duplicates are found):\n"
    "- Preserve the earliest occurrence in the original list (stable deduplication).\n"
    "- If spans differ only by length within the same sentence, prefer the longer, more specific span\n"
    "  (e.g., 'distal caries on #19' over 'caries').\n"
    "\n"
    "Output format:\n"
    "- Return a JSON-serializable list of the remaining dictionaries in the original key names and types.\n"
    "- Do not add extra keys or commentary.\n"
    "- If the input list is empty, return an empty list.\n"
    + (f"- After the JSON output, append on a new line: {STOP!r}\n" if add_stop_token else "")
    + "\n"
    # "Think through the comparisons internally, but ONLY output the final JSON followed by the stop token."
    + "Think through the comparisons internally, but ONLY output the final JSON"
    + (" followed by the stop token." if add_stop_token else ".")
    )

    msg_ls = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"{entity_ls}"},
    ]
    if no_sys_prompt: msg_ls = replace_msg_ls_to_user(msg_ls)
    if isDebug: 
        for msg in msg_ls:
            print(msg)
        print('=='*10)


    # --- Tokenization (single call) ---
    text = tokenizer.apply_chat_template(msg_ls, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=False).to(device)

    num_input_tokens = model_inputs.input_ids.shape[1]
    max_new_tokens = min(8192, int(num_input_tokens * 1.5))
    
        # --- Generation ---
    generate_kwargs = dict(
        # do_sample=False, temperature=None, top_p=None,
        do_sample=True, num_beams=3, temperature=0.3,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        # repetition_penalty=1.1,
    )

    with torch.inference_mode():
        generate_ids = model.generate(**model_inputs, **generate_kwargs)
    
    # --- Decode output ---
    output = tokenizer.decode(generate_ids[0][model_inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    key_entity_prompts_tokens_count += num_input_tokens; key_entity_output_tokens_count += len(generate_ids[0])

    return output

# train_set = x_train
# tuple_ls_str = "[ " + ', '.join([e.get_json_str() for e in train_set]) + " ]"
# new_entity_list = revised_entity_list(tuple_ls_str, add_stop_token=True, isDebug=True)
# new_entity_list

def getRevisedTrainingSet(train_set: List[EntityData], add_stop_token=True, isDebug=False) -> List[EntityData]:
    """Deduplicate training set using LLM.
    
    Args:
        train_set: List of EntityData objects
        add_stop_token: Whether to append a stop token after the output
        isDebug: Enable debug output
    
    Returns:
        Deduplicated list of EntityData objects
    """
    final_train_set = []
    # Convert EntityData to dict strings for LLM processing
    tuple_ls_str = "[ " + ', '.join([e.get_json_str() for e in train_set]) + " ]"

    # isCorrect = False; count = 1 
    # while isCorrect is False and count <= ROUND_THRES:
    new_entity_list = revised_entity_list(tuple_ls_str, add_stop_token=add_stop_token, isDebug=isDebug)
    if new_entity_list.endswith("'") and not new_entity_list.startswith("'"):
        new_entity_list = new_entity_list[:-1]
    if isDebug: print(f'new_entity_list -\n{new_entity_list}')
    if add_stop_token and STOP in new_entity_list:
        new_entity_list = new_entity_list.split(STOP)[0].strip()
    
    final_train_set = train_set[:]
    try:
        parsed_list = ast.literal_eval(new_entity_list)
        # Convert parsed dicts back to EntityData objects
        final_train_set = [EntityData.from_dict(d) for d in parsed_list]
        # log_msg(f'len(src_x_train) - {len(train_set)}, len(x_train) - {len(final_train_set)}')
        # isCorrect = True
    except:
        log_msg(f"Fail to revise the entity list... use the original entity list.", isError=True)

        # count += 1

    if len(final_train_set) == 0:
        log_msg(f"len(x_train) - {len(final_train_set)} is zero", isError=True)
        final_train_set = train_set[:]

    return final_train_set

# getRevisedTrainingSet(x_train, add_stop_token=True, isDebug=True)

# Functions 2/ 3 ########################################################################################
def get_TF_output(prompt_text_ls, isDebug=False, return_confidence=False):
    """
    Get True/False output with optional confidence scores.
    
    Args:
        prompt_text_ls: List of prompt texts
        isDebug: Enable debug output
        return_confidence: If True, return List[Tuple[bool, float]], else List[bool]
    
    Returns:
        List of booleans or List of (bool, confidence) tuples
    """
    if not prompt_text_ls:
        return []
    global key_entity_prompts_tokens_count, key_entity_output_tokens_count
    assert tokenizer is not None and model is not None, "Tokenizer and model must be initialized."
    
    model_inputs = tokenizer(
        prompt_text_ls, 
        return_tensors="pt",
        padding=True,
        truncation=False
    ).to(device)

    # batch_size = model_inputs.input_ids.size(0)
    key_entity_prompts_tokens_count += model_inputs.attention_mask.sum().item() 
    
    generate_kwargs = dict(
        do_sample=False,
        # temperature=None,
        # top_p=None,
        # top_k=None,
        max_new_tokens=TF_MAX_NEW_TOKENS,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )
    
    with torch.inference_mode():
        outputs = model.generate(**model_inputs, **generate_kwargs)
    
    generate_ids = outputs.sequences
    scores = outputs.scores  # Tuple of logits for each generated token
    
    # Get probabilities from first token's logits
    first_token_logits = scores[0]  # Shape: (batch_size, vocab_size)
    probs = torch.softmax(first_token_logits, dim=-1)
    
    # Get token IDs for all case variants of "Yes" and "No"
    yes_token_ids = [
        tokenizer.encode("Yes", add_special_tokens=False)[0],
        tokenizer.encode("yes", add_special_tokens=False)[0],
        tokenizer.encode("YES", add_special_tokens=False)[0],
    ]
    no_token_ids = [
        tokenizer.encode("No", add_special_tokens=False)[0],
        tokenizer.encode("no", add_special_tokens=False)[0],
        tokenizer.encode("NO", add_special_tokens=False)[0],
    ]
    
    # Decode outputs for fallback
    input_len = model_inputs.input_ids.size(1)
    new_token_ids = generate_ids[:, input_len:]
    key_entity_output_tokens_count += new_token_ids.numel()
    outputs_decoded = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    if isDebug:
        print(f"[DEBUG] Number of Raw outputs: {len(outputs_decoded)}")

    output_results = []
    for idx, output in enumerate(outputs_decoded):
        # Sum probabilities for all case variants
        yes_prob = sum(probs[idx, tid].item() for tid in yes_token_ids)
        no_prob = sum(probs[idx, tid].item() for tid in no_token_ids)
        
        # Determine Yes/No based on probabilities first, then text fallback
        output_stripped = output.strip().casefold()
        if yes_prob > no_prob:
            is_yes = True
        elif no_prob > yes_prob:
            is_yes = False
        elif "yes" in output_stripped:
            is_yes = True
        elif "no" in output_stripped:
            is_yes = False
        else:
            is_yes = False  # Default fallback
        
        # Calculate normalized confidence
        total_prob = yes_prob + no_prob + 1e-8
        confidence = max(yes_prob, no_prob) / total_prob
        
        if isDebug:
            print(f"[DEBUG] [{idx}] Output: {output!r}, Yes prob: {yes_prob:.4f}, No prob: {no_prob:.4f}, Result: {is_yes}, Confidence: {confidence:.2%}")
        
        if return_confidence:
            output_results.append((is_yes, confidence))
        else:
            output_results.append(is_yes)
            
    return output_results

def get_instruction_sys_prompt(include_description, include_examples) -> str:
    
    # if include_examples :
    constraints_prompt = """Constraints:
- Do not include evaluation metrics, scoring methods, or performance commentary
- Do not copy, paraphrase, or resemble any user-provided example text"""
    if not include_examples :
        constraints_prompt += """
- Do not include evaluation metrics, scoring methods, or performance commentary
- Do not copy, paraphrase, or resemble any user-provided example text"""

    description_prompt = ""
    if include_description :
        description_prompt = """Definition:\n{DESCRIPTION_OF_ENTITY}\n\n"""

    sys_prompt = (
"""Role:
You are an expert in dentistry and clinical natural language processing.

Task:
Generate a clear, self-contained instruction prompt of approximately {INSTRUCTION_LENGTH} words that guides a language model to perform named entity recognition (NER) for the {TARGET_ENTITY} in clinical notes.

Before writing the instruction prompt, carefully examine the provided example data. Use the examples to infer how {TARGET_ENTITY} is expressed in clinical text, including its common linguistic patterns, clinical shorthand, boundary definitions, and variations. The resulting instruction prompt must descriptively encode these observations rather than simply restating the short definition.

The instruction prompt should be written as if it will be given directly to a model that has not seen the example data but must behave consistently with it. The prompt must clearly and thoroughly describe a task focused on identifying {TARGET_ENTITY}-related expressions relevant to clinical documentation.

""" + description_prompt + 

"""Content Requirements:
The instruction prompt must explicitly specify:
- The task objective (extracting {TARGET_ENTITY} mentions from text)
- Clear inclusion rules describing what qualifies as a valid {TARGET_ENTITY} mention
- Clear exclusion rules describing similar or related expressions that should not be extracted
- Guidance on handling abbreviations, clinical shorthand, partial mentions, multiple occurrences, and ambiguous cases, as reflected in the example data
- Several illustrative examples demonstrating correct extraction behavior using realistic clinical language

""" + 
constraints_prompt + 
"""\n\nOutput:
- Return only the finalized instruction prompt"""
    )
    return sys_prompt

def getInstruction(entity_dict_ls: List[EntityData], entity_type: str, entity_desc: Optional[str], 
                   instruction_length: int = 300, revised_list: List[str] = [],
                   include_description: bool = True,
                   include_examples: bool = True,
                   add_stop_token: bool = True, isDebug: bool = False) -> str:
    """Generate an NER instruction prompt using LLM.
    
    Args:
        entity_dict_ls: List of EntityData training examples
        entity_type: Target entity type name
        entity_desc: Optional description of the entity
        instruction_length: Target word count for generated prompt
        revised_list: List of commonly missed entities for feedback
        include_description: Whether to include the entity description in the prompt
        include_examples: Whether to include example data in the prompt
        add_stop_token: Whether to append a stop token at the end of the output
        isDebug: Enable debug output
    
    Returns:
        Generated instruction prompt string
    """
    # --- Globals ---
    global key_entity_prompts_tokens_count, key_entity_output_tokens_count
    assert tokenizer is not None and model is not None, "Tokenizer and model must be initialized."
    # --- Prepare example data string ---
    entity_dict_str = '\n'.join([
        f"Input:\n{e.sentence}\nOutput:\n{str(e.entity_ls)}\n" for e in entity_dict_ls
    ])

    # --- Optional revision message ---
    extra_msg = ""
    if revised_list:
        extra_msg = "\n\nAdditional notes:\nThe following are additional examples are easily missed: \n" + ', '.join(revised_list) + "\n\n"

    # --- Construct messages ---
    msg_ls = [
        {
            "role": "system",
            "content": get_instruction_sys_prompt(include_description, include_examples).format(
                INSTRUCTION_LENGTH=instruction_length,
                TARGET_ENTITY=entity_type,
                DESCRIPTION_OF_ENTITY=entity_desc if entity_desc else "N/A",
            )
             + 
            f"\n- Append the word {STOP} on a new line at the very end of the output{extra_msg}" if add_stop_token else extra_msg
        }
    ]
    
    if entity_dict_ls:
        msg_ls += [
            {"role": "user", "content": f"Data examples:\n{entity_dict_str}\n\n"}
        ]
    
    msg_ls.append({"role": "assistant", "content": "The instruction prompt:"})
    
    if no_sys_prompt: msg_ls = replace_msg_ls_to_user(msg_ls)
    if isDebug:
        print("msg_ls -\n", msg_ls); print("=" * 20)
    # return msg_ls
    # --- Tokenization & Generation ---
    text = tokenizer.apply_chat_template(msg_ls, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=False).to(device)
    num_input_tokens = model_inputs.input_ids.shape[1]

    # Compute capped max tokens for stability
    max_new_tokens = min(2048, int(instruction_length * 1.5))
    generate_kwargs = dict(
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    with torch.inference_mode():
        generate_ids = model.generate(
            **model_inputs,
            **generate_kwargs,
        )

    # --- Extract generated text (excluding input tokens) ---
    output_ids = generate_ids[0][model_inputs.input_ids.size(1):]
    output = tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    
    # --- Trim at STOP marker ---
    if STOP in output:
        output = output.split(STOP)[0].strip()

    num_output_tokens = len(generate_ids[0])

    # --- Debug logging ---
    if isDebug:
        print(f"Input tokens: {num_input_tokens}")
        print(f"Generated tokens: {num_output_tokens}")
        print(f"Sliced output tokens: {len(output_ids)}")

    # --- Update global counters ---
    key_entity_prompts_tokens_count += num_input_tokens
    key_entity_output_tokens_count += num_output_tokens
    return output

# instruction_prompt = getInstruction(x_train, key_entity, entity_desc=key_entity_desc, 
#                                         instruction_length=500, 
#                                         include_description=args.include_description,
#                                         include_examples=args.include_examples, revised_list=revised_list,
#                                         add_stop_token=args.add_stop_token
#                                         )
# print(instruction_prompt)

def get_verify_sys_prompt():
    sys_prompt = """Role:
You are an expert in dentistry and clinical natural language processing.

Task:
Evaluate the following generated instruction prompt and determine whether it fully complies with the required criteria for generating an instruction prompt to perform Named Entity Recognition (NER) for the {TARGET_ENTITY} in dental or clinical notes.
Return TRUE only if the instruction prompt satisfies the following conditions.
Return FALSE If the condition is not met.

Assessment Criteria:
{CRITERIA}

Output Format:
Return only a single token: TRUE or FALSE.
Do not include any explanation, justification, or additional text."""
    return sys_prompt

def verifyInstruction(instruct_prompt: str, target_entity: str, isDebug=False, return_confidence=False) -> bool:
    '''
    Verify if the instruction prompt meets all specified criteria.
    Args:
        instruct_prompt: The instruction prompt to verify
        target_entity: The target entity type
        isDebug: Enable debug output
        return_confidence: If True, return confidence scores along with boolean results
    Returns:
        bool: True if the instruction prompt meets all criteria, False otherwise
    '''
    assert tokenizer is not None and model is not None, "Tokenizer and model must be initialized."
    criteria_ls = [
        "It is clear, concise, and written as a self-contained instruction.",
        "It is complete, explicitly describing what task the model should perform and how it should behave.",
        "It is fully written and not truncated, unfinished, or abruptly cut off.",
        "It is purely descriptive and does not include any examples, sample inputs, or sample outputs.",
        "It does not reference, quote, paraphrase, or imply access to any user-provided example data.",
        "It does not include illustrative Examples.",
        "It avoids duplicated, redundant, or unnecessary content.",
        f"It provides clear guidance on inclusion and exclusion criteria for identifying {target_entity}.",
        f"It is functionally appropriate for guiding a language model to extract {target_entity}-related expressions from clinical or dental notes.",
    ]
    prompt_txt_ls = []
    for criteria in criteria_ls:
        msg_ls = [
            {"role": "system", "content": (
                get_verify_sys_prompt().format(TARGET_ENTITY=target_entity, CRITERIA=criteria)
            )},
            {"role": "user", "content": f"Instruction prompt:\n{instruct_prompt}"},
        ]
        if no_sys_prompt: msg_ls = replace_msg_ls_to_user(msg_ls)
        prompt_txt_ls.append( tokenizer.apply_chat_template(msg_ls, tokenize=False, add_generation_prompt=True) )
    return all(get_TF_output(prompt_txt_ls, isDebug=isDebug, return_confidence=return_confidence))
    # return getTF_output(msg_ls, isDebug=isDebug)

def get_revise_sys_prompt(include_examples: bool = True) -> str:
    no_example_prompt = ""
    if not include_examples:
        no_example_prompt = """
- Maintain a purely descriptive style without including any examples, sample inputs, or sample outputs.
- Do not reference or imply access to user-provided example data."""

    sys_prompt = """Role:
You are an expert in dentistry and clinical natural language processing.

Task:
Revise the following instruction prompt so that it fully complies with the required criteria for generating a clear, self-contained instruction prompt to perform Named Entity Recognition (NER) for the {TARGET_ENTITY} in clinical or dental notes.

The input is an earlier version of an instruction prompt that does not meet the criteria. You must always revise the prompt and return an improved version; do not return the input unchanged, even if it appears mostly correct.

When revising the prompt:
- Improve clarity, conciseness, and logical structure.
- Ensure the task objective is explicit and unambiguous.
- Remove duplicated, redundant, or unnecessary content.
- Ensure the prompt is fully written, complete, and not truncated or unfinished.
- Ensure the focus remains on identifying {TARGET_ENTITY}-related expressions in clinical documentation.
- Include clear descriptive guidance on inclusion and exclusion criteria, clinical shorthand, and ambiguity handling when relevant.
- Avoid excessive or rigid input/output formatting unless it is essential for task performance.""" + no_example_prompt + """\n
The revised instruction prompt must be suitable for direct use by a language model that has not seen any example data but must behave consistently with it.

Output:
Return only the revised, finalized instruction prompt.
Do not include explanations, commentary, comparisons, or additional text.
"""
    return sys_prompt

def reviseInstruction(instruct_prompt: str, entity_type: str, add_stop_token: bool = False, 
                      include_examples: bool = True, isDebug=False) -> str:
    global key_entity_prompts_tokens_count; global key_entity_output_tokens_count
    assert tokenizer is not None and model is not None, "Tokenizer and model must be initialized."
    msg_ls = [
        {"role": "system", "content": (
            get_revise_sys_prompt(include_examples).format(TARGET_ENTITY=entity_type)
            + f'\nAt the end of the output, append the word {STOP!r} on a new line' if add_stop_token else ''
        )},
        {"role": "user", "content": f"Instruction prompt:\n{instruct_prompt}\nRevised instruction prompt:"},
    ]
    if no_sys_prompt: msg_ls = replace_msg_ls_to_user(msg_ls)
    # 1. Combine apply_chat_template and tokenization into one call.
    text = tokenizer.apply_chat_template(msg_ls, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], padding=True, return_tensors="pt").to(device)
    num_input_tokens = model_inputs.input_ids.shape[1]

    # 2. Calculate max_new_tokens length more efficiently using .encode()
    max_new_tokens = len(tokenizer.encode(instruct_prompt)) + 200

    # --- Generation ---
    with torch.inference_mode():
        generate_ids = model.generate(
            **model_inputs,
            temperature=0.7,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # --- Extract generated text (excluding input tokens) ---
    # 3. Use efficient tensor slicing to get *only* the new tokens.
    new_token_ids = generate_ids[:, num_input_tokens:]
    num_output_tokens = new_token_ids.shape[1]
    output = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # --- Trim at STOP marker ---
    output = remove_STOP(output, isDebug=isDebug)

    # --- Debug logging ---
    if isDebug:
        print("Input token count:", num_input_tokens)
        # This now correctly reports the count of *new* tokens
        print("Generated token count:", num_output_tokens) 

    # --- Update global counters ---
    # 5. Ensure the counter *only* adds the number of *newly generated* tokens.
    key_entity_prompts_tokens_count += num_input_tokens
    key_entity_output_tokens_count += num_output_tokens

    return output


# Functions 3/ 3 ########################################################################################
def get_msg_ls_checkInstruction(instruction_prompt:str, sentence:str, key_entity:str, positive_testing:bool = True) -> list[dict]:
    testing_prompt = ""
    if not positive_testing:
        testing_prompt = "not "
    msg_ls = [
                {
                    "role": "system",
                    "content": "You are an expert in dentistry and clinical natural language processing.\n\n"
                            "Task:\n"
                            "Determine whether the specified target entity is{TESTING_PROMPT} present in the target sentence, following the provided entity instruction.\n\n"
                            "Instruction:\n"
                            "{INSTRUCTION_PROMPT}\n\n"
                            "The decision must be based solely on the content of the target sentence and the definition and rules specified in the entity instruction. "
                            "Consider clinical shorthand and abbreviations when explicitly present. "
                            "The target entity must clearly and unambiguously appear in the sentence context.\n\n"
                            "Output Rules:\n"
                            "- Respond with exactly one token: Yes or No.\n"
                            "- Do not include explanations, punctuation, whitespace, or any additional text.\n"
                            "- If the presence of the target entity is ambiguous, indirect, or unclear, respond with No.\n\n"
                            "You must strictly follow the output rules."
                            .format(
                                INSTRUCTION_PROMPT=instruction_prompt,
                                TESTING_PROMPT=testing_prompt
                            )
                },
                {
                    "role": "user",
                    "content": f"Target entity:\n{key_entity}\n\nTarget sentence:\n{sentence}\n\nAnswer:"
                }
            ]
    if no_sys_prompt: msg_ls = replace_msg_ls_to_user(msg_ls)
    return msg_ls

def get_batch_msg_ls_checkInstruction(instruction_prompt:str, sent_dict: dict, key_entity:str, positive_testing:bool = True) -> list[list[dict]]:
    msg_ls = [
        get_msg_ls_checkInstruction(instruction_prompt, sentence, key_entity, positive_testing=positive_testing)
        for _, sentence in sent_dict.items()
    ]
    return msg_ls

def get_bool_batch_result_ls(batch_msg_ls: list[list[dict]],
                             model_input_limit_ratio=0.8, isDebug=False) -> List[bool]:
    """
    Batch version of checkInstruction - processes multiple sentences in one forward pass.

    Args:
        instruction_prompt: The instruction prompt for the model
        sentence_ls: List of sentences to check
        key_entity: The target entity to check for
        isDebug: Enable debug output

    Returns:
        List of booleans indicating whether each sentence contains the target entity
    """
    assert tokenizer is not None and model is not None, "Tokenizer and model must be initialized."

    if isDebug:
        print(f"Processing batch of {len(batch_msg_ls)} sentences")
    # Build message list for each sentence
    
    limit = int(model_max_window*model_input_limit_ratio)
    # try:
    prompt_text_ls = []
    token_counts = 0
    bool_output_ls = []
    for msg_ls in batch_msg_ls:
        prompt_text = tokenizer.apply_chat_template(msg_ls, tokenize=False, add_generation_prompt=True)
        cur_token_counts = len(tokenizer.encode(prompt_text)) + TF_MAX_NEW_TOKENS
        if token_counts + cur_token_counts >= limit:      
            
            chunk_output_ls = get_TF_output(prompt_text_ls)
            bool_output_ls.extend(chunk_output_ls)
            
            prompt_text_ls = [prompt_text]
            token_counts = cur_token_counts
        else:
            prompt_text_ls.append(prompt_text)
            token_counts += cur_token_counts

    if len(prompt_text_ls)> 0:
        chunk_output_ls = get_TF_output(prompt_text_ls)
        bool_output_ls.extend(chunk_output_ls)

    assert len(bool_output_ls) == len(batch_msg_ls), "Output length mismatch"
    # except Exception as e:
    #     log_msg(f"Error in get_bool_batch_result_ls: {e}", isError=True)
    #     bool_output_ls = get_bool_batch_result_ls(batch_msg_ls, model_input_limit_ratio=model_input_limit_ratio*0.5)

    return bool_output_ls

def get_msg_ls_resultsInstruction(instruction_prompt:str, sentence:str, add_stop_token: bool = True) -> list[dict]:
    msg_ls = [
            {
                "role": "system",
                "content": f"You are an expert in dentistry and clinical natural language processing.\n\n"
                        "Task:\n"
                        "Extract all valid mentions of the target entity from the provided sentence, following the rules defined in the entity instruction.\n\n"
                        "Instruction:\n"
                        f"{instruction_prompt}\n\n"
                        "Output Requirements:\n"
                        f"{OUTPUT_FORMAT}\n"
                        "- Return only the final output in the specified list-of-dictionaries format.\n"
                        "- Do not include explanations, comments, or additional text.\n"
                        "- If no valid target entity is present, return an empty list.\n"
                        "- Ensure the output is complete and not truncated."
                        + (f"\n- Append the word {STOP!r} on a new line at the very end of the output" if add_stop_token else "")
            },
            {
                "role": "user",
                "content": f"Target sentence:\n{sentence}\n\nOutput:"
            }
        ]
    if no_sys_prompt: msg_ls = replace_msg_ls_to_user(msg_ls)
    return msg_ls

def get_batch_InstructionResult_ls(instruction_prompt: str, sent_dict: dict[int, str], 
                                   model_input_limit_ratio=0.8, add_stop_token: bool = True, isDebug=False) -> dict[int, EntityData]:
    """
    Batch version of checkInstruction - processes multiple sentences in one forward pass.

    Args:
        instruction_prompt: The instruction prompt for the model
        sent_dict: Dictionary of sent_idx, sentences to check
        isDebug: Enable debug output
        isPromptOnly: If True, return the message list without running inference

    Returns:
        List of EntityData objects with predicted entities
    """
    # instruction_prompt = instruction_prompt
    # sentence_ls = target_sentence_ls
    # model_input_limit_ratio = 0.8
    # isDebug = True
    ###########################################################################################
    assert tokenizer is not None and model is not None, "Tokenizer and model must be initialized."
    if isDebug:
        print(f"Processing batch of {len(sent_dict)} sentences")
        # Build message list for each sentence
    limit = model_max_window*model_input_limit_ratio

    prompt_text_ls = []
    sent_text_ls = []
    token_counts = 0 # calculate for model limitation
    result_output_dict = {}
    max_new_tokens = -2 # calculate the max new tokens for generation
    for sent_idx, sentence in sent_dict.items():
        msg_ls = get_msg_ls_resultsInstruction(instruction_prompt, sentence, add_stop_token=add_stop_token)
        prompt_text = tokenizer.apply_chat_template(msg_ls, tokenize=False, add_generation_prompt=True)
        cur_sentence_token_counts = int(min(limit, len(tokenizer.encode(sentence)) * OUTPUT_MULTIPLIER))
        cur_token_counts = len(tokenizer.encode(prompt_text)) + cur_sentence_token_counts
        if token_counts + cur_token_counts >= limit:
            chunk_output_dict = get_batch_entity_output(prompt_text_ls, sent_text_ls, max_new_tokens, add_stop_token=add_stop_token, isDebug=isDebug)
            result_output_dict.update(chunk_output_dict)
            max_new_tokens = cur_sentence_token_counts
            prompt_text_ls = [prompt_text]
            sent_text_ls = [(sent_idx, sentence)]
            token_counts = cur_token_counts
        else:
            if cur_sentence_token_counts > max_new_tokens: max_new_tokens = cur_sentence_token_counts
            prompt_text_ls.append(prompt_text)
            sent_text_ls.append((sent_idx, sentence))
            token_counts += cur_token_counts

    if isDebug: print(len(prompt_text_ls), len(sent_text_ls))
    if len(prompt_text_ls) > 0:
        chunk_output_dict = get_batch_entity_output(prompt_text_ls, sent_text_ls, max_new_tokens, add_stop_token=add_stop_token, isDebug=isDebug)
        result_output_dict.update(chunk_output_dict)

    return result_output_dict

def get_msg_ls_validateResults(instruction_prompt:str, key_entity: str, sentence:str, entity_val_ls: list, postive_testing:bool = True) -> list[dict]:
    if postive_testing:
        validate_prompt = ("Yes", "No")
    else:
        validate_prompt = ("No", "Yes")

    msg_ls = [
        {
            "role": "system",
            "content": "You are an expert in dentistry and clinical natural language processing.\n\n"
                    "Task:\n"
                    "Validate whether the provided extracted entity or entities correctly fit and comply with the rules defined in the entity instruction, given the target sentence.\n\n"
                    "Instruction:\n"
                    f"{instruction_prompt}\n\n"
                    "The decision must be based solely on whether the extracted entity or entities are valid according to the definition, inclusion criteria, and exclusion criteria specified in the entity instruction, when evaluated against the target sentence.\n"
                    "Do not perform new entity extraction. Only assess the correctness and appropriateness of the provided extracted entities.\n\n"
                    "Output Rules:\n"
                    f"- Respond {validate_prompt[0]} only if all provided extracted entities are valid and correctly match the instruction and the sentence context.\n"
                    f"- Respond {validate_prompt[1]} if any extracted entity is invalid, unsupported by the sentence, ambiguous, or violates the instruction rules.\n"
                    "- Respond with exactly one token: Yes or No.\n"
                    "- Do not include explanations, punctuation, whitespace, or any additional text.\n"
                    "\nYou must strictly follow the output rules."
        },
        {
            "role": "user",
            "content": f"Extracted {key_entity} entities:\n{entity_val_ls}\n\nTarget sentence:\n{sentence}\n\nAnswer:"
        }
    ]
    if no_sys_prompt: msg_ls = replace_msg_ls_to_user(msg_ls)
    return msg_ls

def get_batch_msg_ls_validateResults(instruction_prompt:str, sentence_dict: dict, key_entity: str, postive_testing:bool = True) -> list[list[dict]]:
    batch_msg_ls = [
        get_msg_ls_validateResults(instruction_prompt, key_entity, e_data.sentence, e_data.entity_ls, postive_testing=postive_testing)
        for _, e_data in sentence_dict.items() if e_data
    ]
    return batch_msg_ls

def remove_STOP(text: str, isDebug=False) -> str:
    """Remove STOP token and trailing characters from text."""
    if STOP in text:
        if isDebug: print('output found STOP')
        text = text.split(STOP)[0].strip()
        if text.endswith("'"):
            text = text[:-1].strip()
    return text.strip()

def get_batch_entity_output(prompt_text_ls, sent_text_ls: list[tuple[int, str]], 
                            max_new_tokens: int, add_stop_token: bool = True, isDebug=False) -> dict[int, EntityData]:

    global key_entity_prompts_tokens_count, key_entity_output_tokens_count
    assert tokenizer is not None and model is not None, "Tokenizer and model must be initialized."
    # --- Generation parameters ---
    generate_kwargs = dict(
        do_sample=False,
        # temperature=None,
        # top_p=None,
        # top_k=None,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    model_inputs = tokenizer(
        prompt_text_ls, 
        return_tensors="pt",
        padding=True,  # Pad to longest sequence in batch
        truncation=False
    ).to(device)

    # batch_size = model_inputs.input_ids.size(0)
    key_entity_prompts_tokens_count += model_inputs.attention_mask.sum().item() 

    with torch.inference_mode():
        generate_ids = model.generate(**model_inputs, **generate_kwargs)

    # Slice and decode only new tokens for each sample
    input_len = model_inputs.input_ids.size(1)
    new_token_ids = generate_ids[:, input_len:]

    output_mask = (new_token_ids != tokenizer.pad_token_id)
    key_entity_output_tokens_count += output_mask.sum().item()
    
    outputs = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    if isDebug:
        print(f"[DEBUG] Number of Raw outputs: {len(outputs)}")
    
    # Parse each output
    output_result_dict = {}
    for output, (t_sent_idx, target_sentence) in zip(outputs, sent_text_ls):
        output_result_dict[t_sent_idx] = EntityData(sentence=target_sentence, entity_ls=[])
        if add_stop_token: output = remove_STOP(output, isDebug=isDebug)
        if isDebug: print('[isDebug] - ', repr(output))
        
        try:
            entity_ls_response = ast.literal_eval(output)
            if isDebug: print('[isDebug] - entity_ls_response', repr(entity_ls_response))
            if isinstance(entity_ls_response, list): 
                if isDebug: print('[isDebug] - entity_ls_response is a list -->', repr(entity_ls_response))
                entity_ls_response = entity_ls_response[0]

            if ENTITY_LS in str(entity_ls_response):
                pred_entity_ls = entity_ls_response.get(ENTITY_LS, [])
                if isDebug: print('[isDebug] - pred_entity_ls:', pred_entity_ls)
                if pred_entity_ls is not None and any([pred_entity not in target_sentence for pred_entity in pred_entity_ls]):
                    break # try again hallucination found
                
                output_result_dict[t_sent_idx] = EntityData.from_dict(entity_ls_response)

            elif isDebug:
                print('[isDebug] - entity_ls_response is not a list of dict')
                break
                
        except Exception as e:
            if isDebug: log_msg( "".join(traceback.format_exception(type(e), e, e.__traceback__)), isError=True)
            break
        
    return output_result_dict

def get_msg_ls_revise_resultsInstruction(instruction_prompt:str, sentence:str, entity_ls: list, add_stop_token: bool = True) -> list[dict]:
    msg_ls = [
        {
            "role": "system",
            "content": "You are an expert in dentistry and clinical natural language processing.\n\n"
                    "Task:\n"
                    "Review and revise the provided target entity list so that it fully complies with the rules defined in the entity instruction, based on the target sentence.\n\n"
                    "Instruction:\n"
                    f"{instruction_prompt}\n\n"
                    "Revision Rules:\n"
                    "- Evaluate each provided target entity against the entity definition, inclusion criteria, and exclusion criteria specified in the instruction.\n"
                    "- Remove any target entity that is invalid, unsupported by the sentence, ambiguous, or violates the instruction rules.\n"
                    "- Identify and add any missing target entities that are clearly present in the sentence and should have been extracted according to the instruction.\n"
                    "- Do not add entities that are ambiguous, inferred, or weakly implied.\n\n"
                    "Output Requirements:\n"
                    # f"{OUTPUT_FORMAT}\n"
                    "- Return the revised list of target entities after applying all removals and additions.\n"
                    "- Preserve the original output format and structure exactly as provided in the input entity list.\n"
                    "- Do not change field names, value types, or data structure; only modify the list contents.\n"
                    "- If all provided entities are invalid and no new valid entities should be added, return an empty list in the same format.\n"
                    "- Ensure the output contains only valid target entities, follows the original format, and is complete and not truncated."
                    + (f"\n- Append the word {STOP!r} on a new line at the very end of the output." if add_stop_token else "")
        },
        {
            "role": "user",
            "content": f"Initial target entity list:\n{str(entity_ls)}\n\nTarget sentence:\n{sentence}\n\nOutput:"
        }
    ]
    if no_sys_prompt: msg_ls = replace_msg_ls_to_user(msg_ls)
    return msg_ls

def get_reviseResult(instruction_prompt: str, entity_data: EntityData, 
                                   model_input_limit_ratio=0.8, add_stop_token: bool = True, isDebug=False) -> EntityData:
    """
    Revise the entity list for a single sentence using the provided instruction prompt.
    Args:
        instruction_prompt: The instruction prompt for the model
        entity_data: EntityData object containing the data to be processed
        model_input_limit_ratio: Ratio of model input limit to use
        add_stop_token: Whether to append a stop token at the end of the output
        isDebug: Enable debug output

    Returns:
        List of EntityData objects with predicted entities
    """
    ###########################################################################################
    assert tokenizer is not None and model is not None, "Tokenizer and model must be initialized."
    # Build message list for each sentence
    limit = model_max_window * model_input_limit_ratio
    idx = 0 # only one data
    msg_ls = get_msg_ls_revise_resultsInstruction(instruction_prompt, entity_data.sentence, entity_data.entity_ls, add_stop_token=add_stop_token)
    cur_sentence_token_counts = int(min(limit, len(tokenizer.encode(entity_data.sentence)) * OUTPUT_MULTIPLIER))
    prompt_text = tokenizer.apply_chat_template(msg_ls, tokenize=False, add_generation_prompt=True)
    revised_dict = get_batch_entity_output([prompt_text], [(idx, entity_data.sentence)], cur_sentence_token_counts, add_stop_token=add_stop_token, isDebug=isDebug)

    return revised_dict[idx]

def get_batch_instruction_ls_result_ls(instruction_prompt_ls: list, sent_dict: dict[int, str], 
                                   model_input_limit_ratio=0.8, add_stop_token: bool = True, isDebug=False) -> dict[int, EntityData]:
    """
    Batch version of checkInstruction - processes multiple sentences in one forward pass.

    Args:
        instruction_prompt_ls: List of instruction prompts for the model
        sent_dict: Dictionary of sent_idx to sentences to check
        isDebug: Enable debug output
        isPromptOnly: If True, return the message list without running inference

    Returns:
        Dictionary of sent_idx to EntityData objects with predicted entities
    """
    # instruction_prompt = instruction_prompt
    # sentence_ls = target_sentence_ls
    # model_input_limit_ratio = 0.8
    # isDebug = True
    def update_sentence_entity_results():
        """
        Update src_entity_dict with results from result_dict.

        Args:
            src_entity_dict: Dictionary of sent_idx to EntityData to be updated
            result_dict: Dictionary of sent_idx to EntityData with new results

        Returns:
            Updated src_entity_dict
        """
        for sent_idx, entity_data in chunk_output_dict.items():
            if sent_idx in result_output_dict:
                result_output_dict[sent_idx].update(entity_data.entity_ls)
            else:
                result_output_dict[sent_idx] = entity_data

    ###########################################################################################
    assert tokenizer is not None and model is not None, "Tokenizer and model must be initialized."
    if isDebug:
        print(f"Processing batch of {len(sent_dict)} sentences")
        # Build message list for each sentence
    limit = int(model_max_window*model_input_limit_ratio)

    prompt_text_ls = []
    sent_text_ls = []
    token_counts = 0 # calculate for model limitation
    result_output_dict = {}
    max_new_tokens = -2 # calculate the max new tokens for generation
    try:
        for sent_idx, sentence in sent_dict.items():
            for instruction_prompt in instruction_prompt_ls:
                msg_ls = get_msg_ls_resultsInstruction(instruction_prompt, sentence, add_stop_token=add_stop_token)
                prompt_text = tokenizer.apply_chat_template(msg_ls, tokenize=False, add_generation_prompt=True)
                cur_sentence_token_counts = int(min(limit, len(tokenizer.encode(sentence)) * OUTPUT_MULTIPLIER))
                cur_token_counts = len(tokenizer.encode(prompt_text)) + cur_sentence_token_counts
                if token_counts + cur_token_counts >= limit:
                    # print(len(prompt_text_ls), len(sent_text_ls))
                    chunk_output_dict = get_batch_entity_output(prompt_text_ls, sent_text_ls, max_new_tokens, add_stop_token=add_stop_token, isDebug=isDebug)
                    # print(chunk_output_dict)
                    update_sentence_entity_results()
                    # result_output_dict.update(chunk_output_dict)
                    max_new_tokens = cur_sentence_token_counts
                    prompt_text_ls = [prompt_text]
                    sent_text_ls = [(sent_idx, sentence)]
                    token_counts = cur_token_counts
                else:
                    if cur_sentence_token_counts > max_new_tokens: max_new_tokens = cur_sentence_token_counts
                    prompt_text_ls.append(prompt_text)
                    sent_text_ls.append((sent_idx, sentence))
                    token_counts += cur_token_counts

        if len(prompt_text_ls) > 0:
            # print(len(prompt_text_ls), len(sent_text_ls))
            chunk_output_dict = get_batch_entity_output(prompt_text_ls, sent_text_ls, max_new_tokens, add_stop_token=add_stop_token, isDebug=isDebug)
            # print(chunk_output_dict)
            update_sentence_entity_results()
    
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        new_model_limit_ratio = model_input_limit_ratio * 0.8
        if "out of memory" in str(e).lower() and new_model_limit_ratio > 0.1:
            return get_batch_instruction_ls_result_ls(instruction_prompt_ls, sent_dict, 
                                   model_input_limit_ratio=new_model_limit_ratio, add_stop_token=add_stop_token, isDebug=isDebug) 

    return result_output_dict


def get_batch_reviseResult(instruct_ls: list, entity_data: EntityData, 
                                   model_input_limit_ratio=0.8, add_stop_token: bool = True, isDebug=False):#-> list(EntityData):
    """
    Revise the entity list for a single sentence using the provided instruction prompt list.
    Note: This expected to be limited instruction prompt (e.g. 2-3) for the same sentence to further refine the entity extraction results.
    For larger number of instruction prompts, OOM may happen and get_batch_instruction_ls_result_ls should be used instead.
    Args:
        instruction_prompt: The instruction prompt for the model
        entity_data: EntityData object containing the data to be processed
        model_input_limit_ratio: Ratio of model input limit to use
        add_stop_token: Whether to append a stop token at the end of the output
        isDebug: Enable debug output

    Returns:
        List of EntityData objects with predicted entities
    """
    ###########################################################################################
    assert tokenizer is not None and model is not None, "Tokenizer and model must be initialized."
    # Build message list for each sentence
    limit = model_max_window * model_input_limit_ratio
    cur_sentence_token_counts = 0
    prompt_text_ls = []
    data_ls = []
    for instruct_idx, instruction_prompt in enumerate(instruct_ls):
        msg_ls = get_msg_ls_revise_resultsInstruction(instruction_prompt, entity_data.sentence, entity_data.entity_ls, add_stop_token=add_stop_token)
        cur_sentence_token_counts = int(min(limit, len(tokenizer.encode(entity_data.sentence)) * OUTPUT_MULTIPLIER))
        prompt_text = tokenizer.apply_chat_template(msg_ls, tokenize=False, add_generation_prompt=True)
        prompt_text_ls.append(prompt_text)
        data_ls.append((instruct_idx, entity_data.sentence))
    
    print(f"Data list {len(data_ls)}: {data_ls}")
    revised_dict = get_batch_entity_output(prompt_text_ls, data_ls, cur_sentence_token_counts, add_stop_token=add_stop_token, isDebug=isDebug)

    return revised_dict

def get_reviseResult_from_instruct_ls(batch_msg_ls: list[list[dict]],
                                      target_sent: str,
                                      model_input_limit_ratio=0.8,
                                      add_stop_token=True, isDebug=False) -> dict[int, EntityData]:
    """
    Revise the entity list for a single sentence using the provided instruction prompt.
    Args:
        batch_msg_ls: List of message lists for the model, each corresponding to an instruction prompt
        target_sent: The target sentence to be revised
        model_input_limit_ratio: Ratio of model input limit to use
        add_stop_token: Whether to add stop token to the prompt
        isDebug: Enable debug output

    Returns:
        Dictionary of sent_idx to EntityData with revised entities
    """
    ###########################################################################################
    assert tokenizer is not None and model is not None, "Tokenizer and model must be initialized."
    # Build message list for each sentence
    limit = model_max_window * model_input_limit_ratio
    
    cur_sentence_token_counts = int(min(limit, len(tokenizer.encode(target_sent)) * OUTPUT_MULTIPLIER))
    if isDebug: print( len(tokenizer.encode(target_sent)),  len(tokenizer.encode(target_sent)) * OUTPUT_MULTIPLIER, cur_sentence_token_counts)
    prompt_text_ls = tokenizer.apply_chat_template(batch_msg_ls, tokenize=False, add_generation_prompt=True)
    revised_dict = get_batch_entity_output(prompt_text_ls, 
                                           [(i, target_sent) for i in range(len(prompt_text_ls))], cur_sentence_token_counts, add_stop_token=add_stop_token, isDebug=isDebug)

    return revised_dict


# Function to parse command-line arguments
def parse_args(args_list=None):
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Model initialization script")

    parser.add_argument("--version",  type=str, required=True, help="Version string or project name, e.g., 'v2', 'xxx_project'")

    parser.add_argument("--gpu_num",    type=str, required=True, help="GPU number (e.g., '0')")
    parser.add_argument("--device",     type=str, help="Device to use (e.g., 'cuda')")
    parser.add_argument("--base_model", type=str, default=None, help="Name of the base model")
    parser.add_argument("--load_model_fct", type=str, default=None, help="Function to load the model")
    parser.add_argument("--model_name", type=str, default=None, help="Name of the model with specific settings")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model directory")
    parser.add_argument("--model_input_limit_ratio", type=float, default=0.8, help="Ratio of model input limit to use for batch generation")

    parser.add_argument("--train_data_path", type=str, default=None, help="Path to the training json files")
    parser.add_argument("--dst_root", type=str, default=r"/home/FullMouth/data", help="Directory to save processed data")
    parser.add_argument("--result_type_dir", type=str, default=None, help="Directory to save results - 'gold_dpo', 'gold_sft', 'gold_prompt'")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory for model checkpoints")
    parser.add_argument("--start_idx", type=int, default=0, help="idx for processing")
    parser.add_argument("--end_idx", type=int, default=-1, help="idx for processing")
    parser.add_argument("--test_dir", type=str, default=None, help="Directory for test data")

    parser.add_argument("--num_of_instructions", type=int, default=1, required=True, help="Number of instruction prompts to generate/use")
    parser.add_argument("--makeup", action="store_true", default=False, help="Flag to enable makeup the number of instruction used")
    parser.add_argument("--instruction_length", type=int, default=500, help="Length of the instruction prompt")
    parser.add_argument("--validation_threshold", type=float, default=0.8, help="Threshold for validation")

    parser.add_argument("--train_neg_ratio", type=int, default=3, help="Ratio of negative samples in training set")
    parser.add_argument("--val_neg_ratio", type=int, default=10, help="Ratio of negative samples in validation set")
    parser.add_argument("--test_neg_ratio", type=int, default=100, help="Ratio of negative samples in test set")

    parser.add_argument("--reset", action="store_true", default=False, help="Flag to trigger start with a blank folder")
    parser.add_argument("--add_stop_token", action="store_true", default=True, help="Flag to add STOP token at the end of generated instruction")
    
    parser.add_argument("--revised_training_set",   action="store_true", default=False, help="Flag to use revised training set")
    parser.add_argument("--include_examples",       action="store_true", default=False, help="Flag to include examples in the prompt")
    parser.add_argument("--include_description",    action="store_true", default=False, help="Flag to include description in the prompt")
    parser.add_argument("--error_feedback_in_loop", action="store_true", default=False, help="Enable error feedback in the processing loop")
    parser.add_argument("--do_revise", action="store_true", default=False, help="[Deprecated] Flag to perform result revision step")

    parser.add_argument("--run_train", action="store_true", default=False, help="Flag to run on training set")
    parser.add_argument("--run_test",  action="store_true", default=False, help="Flag to run on test set")
    parser.add_argument("--run_extra",  action="store_true", default=False, help="Flag to run on extra test set")
    parser.add_argument("--debug_mode", action="store_true", default=False, help="Enable debug mode with verbose output")
    parser.add_argument("--go_reverse", action="store_true", default=False, help="Flag to reverse the order of processing files")
    parser.add_argument("--no_system_prompt", action="store_true", default=False, help="Flag to exclude system prompt from the messages")
    
    args = parser.parse_args(args_list)
    return args