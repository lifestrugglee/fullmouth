# %cd /home/FullMouth/code
from pathlib import Path
import os
import re
import time
import shutil
from glob import glob
import argparse
import numpy as np
import pandas as pd

from fullmouth_util import *
STR_NA = 'Na'
STR_HALU  = 'Hallu'

fm_util = FMUtil()
from tqdm import tqdm
import spacy

def main(args):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    spacy.require_gpu()
    nlp = spacy.load("en_core_web_trf")

    src_txt_root = r'/home/FullMouth/data/notes/UTH_notes_type_filter'
    src_txt_fp_ls = glob(os.path.join(src_txt_root, '*.txt'))
    src_txt_fn_ls = [os.path.basename(fp) for fp in src_txt_fp_ls]
    # len(src_txt_fp_ls)

    for src_txt_fp in tqdm(src_txt_fp_ls[args.st_idx:args.end_idx]):
        src_txt_fn = os.path.basename(src_txt_fp)
        save_fp = os.path.join(src_txt_root, src_txt_fn.replace('.txt', '.pkl'))
        if os.path.exists(save_fp):
            continue
        # print(f'Processing {src_txt_fn}...')
        with open(src_txt_fp, 'r') as f:
            note_txt = f.read()

        content_ls = find_entity_locations(src_txt_fn, note_txt, {}, nlp)
        savePickle(save_fp, content_ls)


def parse_args(args_list=None):
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Model initialization script")
    parser.add_argument("--gpu_num", type=str, required=True, help="GPU number (e.g., '0')")
    parser.add_argument("--st_idx",  type=int, default=0,  help="Start index (e.g., '0')")
    parser.add_argument("--end_idx", type=int, default=-1, help="End index (e.g., '1')")

    args = parser.parse_args(args_list)
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)

'''
python 1-5_convert_note2sent.py --gpu_num 0 
python 1-5_convert_note2sent.py --gpu_num 0 --st_idx 21000 --end_idx 38000
'''