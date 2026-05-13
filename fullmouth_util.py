from rapidfuzz import fuzz
from typing import Iterable, List, Callable, Optional, Any
from collections import defaultdict
import re
import pickle
import random
import importlib
import os
import sys

import json
from pathlib import Path

def write_json(data: Any, file_path: str | Path) -> None:
    """Write Python data to a JSON file."""
    path = Path(file_path)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def load_json(file_path: str | Path) -> Any:
    """Load JSON data from a file."""
    path = Path(file_path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)

def savePickle(file_name:str, obj):
    with open(file_name, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def loadPickle(file_name:str):
    # Backward-compatibility for older pickles that reference top-level
    # module names (e.g., function_util_v3) instead of py_files.* names.
    module_dir = os.path.dirname(os.path.abspath(__file__))
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    for mod_name in ('function_util_v2', 'function_util_v2_3', 'function_util_v3', 'function_util_v4'):
        if mod_name not in sys.modules:
            try:
                sys.modules[mod_name] = importlib.import_module(mod_name)
            except Exception:
                # Keep loading even if some legacy modules are unavailable.
                pass

    with open(file_name, 'rb') as handle:
        return pickle.load(handle)

# Instruct_prompt_dict [entity]'s key
ROUND_THRES = 5

INSTRUCT_PROMPT = 'instruct_prompt'
CHECK_PROMPT = 'check_prompt'
GET_RESULT_PROMPT = 'getResult_prompt'

# v3
NUM_CORRECT_VAL = 'num_correct_val'
NUM_VAL = 'num_val'
NUM_CORRECT_TEST = 'num_correct_test'
NUM_TEST = 'num_test'

# v4
EVAL_MTX  = 'eval_metrix'
TEST_MTX = 'test_metrix'

# global STOP
PRED = 'pred'
STOP = '<|STOP|>'
CHECK = 'check_val'
GET_RESULT = 'getResult_val'
ENTITY = 'entity'
ENTITY_LS = 'entity_ls'
ENTITY_DICT = 'entity_dict'
SENT = 'sentence'
OUTPUT_FORMAT = "Output format: {{ '{}': SENTENCE, '{}': [TEXT_1, TEXT_2]}}".format(SENT, ENTITY_LS)
NA = 'Na'
HALU  = 'Hallu'
############################################################

# LOCAL_MODEL_DICT = {
                    
#                     'Ministral-8B-Instruct-2410' : 'minstral-8B',
#                     'medgemma-4b-it' : 'medgemma-4b-it',
#                     'Qwen2-7B-Instruct': 'Qwen2-7B',
#                     'Qwen2.5-7B-Instruct': 'Qwen2.5-7B',
#                     'Qwen2.5-14B-Instruct': 'Qwen2.5-14B',
#                     'Qwen2.5-32B-Instruct': 'Qwen2-32B',
#                     "gpt-4o":"gpt-4o",
#                     }

label_fp = r'./label_v4.json'
COMMENTS = 'comments'
DESC = 'desc'
IS_INCLUDE = 'is_include'

class FM_label:
    def __init__(self):
        label_fp = r'./label_v4.json'
        assert os.path.exists(label_fp), f"Label file not found at {label_fp}. Please ensure the label file exists."
        self.gold_entity_dict = load_json(label_fp)
        self.gold_entity_desc_dict = {}
        self.gold_entity_ls = self.get_gold_entity_ls(self.gold_entity_dict)

        self.exclude_old_entity_ls = [
            "Furcation involvement", "TMJ", "Periodontal Charting Indices", 
            "Pocket/probing depth (PD/PPD)", "Clinical attachment level (CAL)", 
            "Bleeding on probing (BoP)", "Gingival recession", 
            "Gingival description", "Mobility", "Plaque Score", 
            '◆', '▲', '▮',
            'crown/root ratio',
            'Periodontal/Ginval Health',]
        
    def get_gold_entity_ls(self, entity_dict):
        ls = []
        for key in entity_dict.keys():
            if isinstance(entity_dict[key], str):
                ls.append(key)
                self.gold_entity_desc_dict[key] = entity_dict[key]
            else:
                for sub_key in entity_dict[key].keys():
                    if sub_key != 'desc':
                        ls.append(sub_key)
                        self.gold_entity_desc_dict[sub_key] = entity_dict[key][sub_key]
        return ls

    def get_entity_description(self, entity_name: str) -> str:
        return self.gold_entity_desc_dict.get(entity_name, '')


EXTRA_SPACE = ' -' + '　'*100 + ' - '
def preprocess_notes(note:str) -> str:
    return note.replace('\n', EXTRA_SPACE)

def reverse_notes(note:str) -> str:
    return note.replace(EXTRA_SPACE, '\n')


def get_txt_section_from_reshaped(gold_content_ls:list, level_idx_name='text_segment_index') -> list:
    def merge_section(sec_ls):
        """Merge sentences and entity dicts from a section list."""
        sent_ls = [line[SENT] for line in sec_ls]
        sec_entity_dict = defaultdict(list)

        for sec_line in sec_ls:
            for key, value_list in sec_line[ENTITY_DICT].items():
                sec_entity_dict[key].extend(value_list)

        return {
            SENT: '\n'.join(sent_ls),
            ENTITY_DICT: dict(sec_entity_dict),
            ENTITY: list(sec_entity_dict.keys())
        }

    collect_ls = []
    sec_ls = []
    cur_num = 0

    for line_dict in gold_content_ls:
        section_num =  line_dict[level_idx_name]
        # section_num =  line_dict['sent_index']

        # same section → keep collecting
        if cur_num == section_num:
            sec_ls.append(line_dict)
        else:
            # section changed → finalize previous section
            collect_ls.append(merge_section(sec_ls))
            sec_ls = [line_dict]
            cur_num = section_num

    # handle the last section
    if sec_ls:
        collect_ls.append(merge_section(sec_ls))

    return collect_ls

def find_entity_locations(fn: str, text: str, entity_dict: dict, nlp, combined_sentences: bool = False) -> list[dict]:
    '''
    Returns a list like:
    [
      {
        "entity_dict": {},
        "file_name": "file.txt",
        "text_segment_index": 0,     # sentence section
        "sent_index": 0,     #  sent index in the text segment
        "sentence": "D: 16 y/o male presents with father for Recall.",
      },
      ...
    ]
    '''

    assert nlp is not None, "nlp model is not loaded"
    # Split the text by newlines
    text_ls = text.split('\n')

    text_entity_ls = []

    for text_segment_index, text_segment in enumerate(text_ls):
        doc = nlp(text_segment)
        sentences = [sent.text for sent in doc.sents]  # Split into sentences
        if combined_sentences:
            merged_sentences = []

            for sentence_text in sentences:
                if len(sentence_text) < 100 and merged_sentences:
                    merged_sentences[-1] = f"{merged_sentences[-1]} {sentence_text}"
                else:
                    merged_sentences.append(sentence_text)
            sentences = merged_sentences[:]

        for i, sentence_text in enumerate(sentences):
            # sentence_text = sentence
            one_sent_dict = {
                            ENTITY_DICT: {},
                            "file_name": fn,
                            "text_segment_index": text_segment_index,
                            "sent_index": i,
                            SENT: sentence_text, 
                        }
            for entity_name, entity_val_ls in entity_dict.items():
                if entity_name == COMMENTS: continue
                tmp_entity_val_ls = []
                for entity_val in entity_val_ls:
                    start_char = sentence_text.find(entity_val)
                    if start_char != -1:
                        if entity_name not in one_sent_dict[ENTITY_DICT]:
                            one_sent_dict[ENTITY_DICT][entity_name] = [entity_val]
                        else:
                            one_sent_dict[ENTITY_DICT][entity_name].append(entity_val)  
                    else:
                        tmp_entity_val_ls.append(entity_val)
                entity_dict[entity_name] = tmp_entity_val_ls

            text_entity_ls.append(one_sent_dict)

    return text_entity_ls


def add_position2entity_dict(entity_dict, note):
    for entity in entity_dict:
        if entity == COMMENTS: continue
        if entity_dict[entity]:
            cur_idx = 0
            src_entity_ls = entity_dict[entity][:]
            new_ls = []
            for val in src_entity_ls:
                st_idx = note[cur_idx:].find(val) + cur_idx
                end_idx = st_idx + len(val)
                assert st_idx >= 0, f'Entity Val "{val}" not found in the note'
                new_ls.append({'start': st_idx, 'end': end_idx, 'label': val})
                cur_idx = end_idx
            entity_dict[entity] = new_ls
    return entity_dict

def is_similarity(compared_entity: str, entity_ls: list, threshold: int = 80):
    """
    Returns (is_match, best_index) where:
      - is_match: True/False depending on whether any entity >= threshold
      - best_index: index of the best-matching entity in entity_ls, or None
    """
    best_score = -1
    best_idx = None
    for i, entity in enumerate(entity_ls):
        score = fuzz.partial_ratio(compared_entity, entity)
        if score > best_score:
            best_score = score
            best_idx = i

    return (best_score >= threshold, best_idx)

# Functions
def get_instruction_data(entity_dict_list):
    tmp_dict = {}
    for e_dict in entity_dict_list:
        text_key = e_dict['text_segment_index']
        if text_key not in tmp_dict:
            tmp_dict[text_key] = []
        tmp_dict[text_key].append(e_dict)

    collect_ls = []
    for key in tmp_dict.keys():
        if len(tmp_dict[key]) > 1:
            # print(key, len(tmp_dict[key]))
            new_sent_dict = {'sentence': '', ENTITY: []}
            for e_dict in tmp_dict[key]:
                new_sent_dict[ENTITY].append(e_dict['text'])
            new_sent_dict['sentence'] = e_dict['sentence']
            collect_ls.append(new_sent_dict)
        else:
            e_dict = tmp_dict[key][0]
            collect_ls.append({'sentence': e_dict['sentence'], ENTITY: [e_dict['text']]})
    
    return collect_ls


def merge_list_dicts(src_dict, changed_dict):
    """
    Merge two dictionaries whose values are lists.
    For keys in both dictionaries, merge lists while preserving order
    and removing duplicates.
    """
    result = src_dict.copy()
    for key, values in changed_dict.items():
        result.setdefault(key, [])
        result[key] = list(dict.fromkeys(result[key] + values))
    return result


def update_collect_ls(collect_ls, gold_collect_ls, is_debug=False):
    collect_ls_idx = 0
    gold_count = 0
    for idx, sent_dict_gold in enumerate(gold_collect_ls):
        if sent_dict_gold[ENTITY_DICT]:
            if is_debug: 
                print(idx, sent_dict_gold[ENTITY_DICT].keys())
            gold_count += 1
            while collect_ls_idx < len(collect_ls):
                sent_dict = collect_ls[collect_ls_idx]
                if sent_dict_gold[SENT] in sent_dict[SENT]:
                    # print(collect_ls_idx, sent_dict)
                    collect_ls[collect_ls_idx][ENTITY_DICT] = merge_list_dicts(collect_ls[collect_ls_idx][ENTITY_DICT],
                                                                                           sent_dict_gold[ENTITY_DICT])
                    break
                collect_ls_idx += 1
    
    if is_debug:
        print('Total gold with entities:', gold_count)
    return collect_ls



def _sym_partial_ratio(a: str, b: str) -> int:
    """More robust than one-way partial_ratio for uneven lengths."""
    return max(fuzz.partial_ratio(a, b), fuzz.partial_ratio(b, a))

def keep_longest_fuzzy(
    entities: Iterable[str],
    threshold: int = 80,
    normalize: Optional[Callable[[str], str]] = lambda s: s.strip().lower(),
    scorer: Callable[[str, str], int] = _sym_partial_ratio,
) -> List[str]:
    """
    Deduplicate entities by fuzzy overlap:
      - if two entities are similar (score >= threshold), keep the longer one
      - uses greedy length-desc strategy (works well for substring-ish overlaps)

    Returns a list (deterministic order: longest-first).
    """
    # drop empties / None, keep as list
    ents = [e for e in entities if e and str(e).strip()]
    # deterministic: length desc, then lex
    ents.sort(key=lambda x: (-len(x), x))

    kept: List[str] = []
    kept_norm: List[str] = []

    for e in ents:
        e_norm = normalize(e) if normalize else e

        is_dup = False
        for kn in kept_norm:
            if scorer(e_norm, kn) >= threshold:
                is_dup = True
                break

        if not is_dup:
            kept.append(e)
            kept_norm.append(e_norm)

    return kept

