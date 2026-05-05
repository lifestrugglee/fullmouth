# /usr/bin/env python
from pathlib import Path
import os
from glob import glob
import argparse
from tqdm import tqdm
import spacy
from spacy.util import is_package

from fullmouth_util import find_entity_locations, write_json

def main(args):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    spacy.prefer_gpu()
    model_name = "en_core_web_trf"

    if is_package(model_name):
        nlp = spacy.load(model_name)
        print(f"Success: '{model_name}' is loaded and ready to use.")
    else:
        raise ValueError(f"Warning: '{model_name}' is not available in your environment.\n"
                         f"Please install it by running: python -m spacy download {model_name}")


    src_txt_root = args.text_data_dir
    src_txt_fp_ls = list(Path(src_txt_root).glob('*.txt'))
    if not args.end_idx: args.end_idx = len(src_txt_fp_ls)

    output_dir = args.output_dir if args.output_dir else src_txt_root
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    for src_txt_fp in tqdm(src_txt_fp_ls[args.st_idx:args.end_idx]):
        src_txt_fn = os.path.basename(src_txt_fp)
        dst_json_fp = os.path.join(output_dir, src_txt_fn.replace('.txt', '.json'))
        if os.path.exists(dst_json_fp): continue

        with open(src_txt_fp, 'r') as f:
            note_txt = f.read()

        content_ls = find_entity_locations(src_txt_fn, note_txt, {}, nlp, args.combined_sentences)
        write_json(content_ls, dst_json_fp)
    
    print(f'Finished processing {args.end_idx - args.st_idx} notes in {src_txt_root}.')
    print(f'Output JSON files are saved in {output_dir}.')


def parse_args(args_list=None):
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Text to sentence conversion script via spaCy")
    parser.add_argument("--text_data_dir", type=str, help="Target directory containing text notes")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for JSON files (default: same as input directory)")
    parser.add_argument("--gpu_num", type=str, required=True, help="GPU number (e.g., '0')")
    parser.add_argument("--st_idx",  type=int, default=0,  help="Start index (e.g., '0')")
    parser.add_argument("--end_idx", type=int, default=None, help="End index (e.g., '100')")
    parser.add_argument("--combined_sentences", type=bool, default=False, help="Whether to combine short sentences (<100 characters) into longer ones (default: False)")

    args = parser.parse_args(args_list)
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)

'''
python convert_note2sent.py --text_data_dir "/home/FullMouth/data/dataset/training_notes/" --gpu_num 0
python convert_note2sent.py --text_data_dir "/home/FullMouth/data/dataset/test_notes/" --gpu_num 0 --combined_sentences True
'''