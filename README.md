# FullMouth LLM NER pipeline (prompt generation + inference)

This folder contains a two-stage pipeline for extracting dental NER entities from clinical notes using local LLMs via HuggingFace `transformers`.

- Stage A: prompt generation (LLMs_prompt_generation.py) generates and evaluates instruction prompts per entity.
- Stage B: inference (LLMs_inferences.py) runs entity screening + extraction using the saved prompts.

Shell wrappers (prompt_generation.sh, llm_inference.sh) are provided as runnable examples, but their default paths are FullMouth-environment specific (e.g., /home/FullMouth/...).

## What’s in this folder

Entry points

- LLMs_prompt_generation.py: builds dst_root/version/<target_dir>/instruct_prompt_dict.json
- LLMs_inferences.py: reads that instruct_prompt_dict.json and writes per-note JSON outputs with a pred field
- convert_note2sent.py: converts raw .txt notes to sentence-level .json suitable for inference input

Core utilities / schema

- function_util.py: CLI args (parse_args) + model init + batching + evaluation helpers
- fullmouth_util.py: constants and schema loader (FM_label)
- label_v4_1.json: the entity schema + descriptions used by FM_label (loaded via relative path)

Config / deps

- config.yml: required at runtime; at minimum contains model_root for local model paths
- requirements.txt: baseline runtime dependencies for prompt generation / inference

Fine-tuning (optional)

- SFT_DPO/: notebooks + extra requirements for QLoRA SFT and DPO
    - SFT_DPO/SFT-int4.ipynb
    - SFT_DPO/DPO_reward_init.ipynb
    - SFT_DPO/DPO_training.ipynb
    - SFT_DPO/requirements-sft_dpo.txt

Misc

- __req_scan_py/: copies of the main scripts (used for dependency scanning in some environments)
- LICENSE

## Setup

1) Create / edit config.yml in this directory (the scripts require it in the current working directory)

Minimal example:

```yaml
model_root: /data_sys/lang_model
```

The checked-in config.yml also includes keys like root_dir and src_txt_root that may be used by other tools, but the two main entry points only require model_root.

2) Install Python dependencies

```bash
pip install -r requirements.txt
```

Notes:

- requirements.txt pins torch/transformers versions for the FullMouth runtime. If you’re not on the same CUDA / driver stack, you may need to adjust torch accordingly.
- Models are loaded with local_files_only=True. Make sure the model exists on disk under model_root.

## Data formats

Stage A input (training json)

LLMs_prompt_generation.py expects a directory of .json files, each containing a list of sentence records like:

```json
{
    "sentence": "D: 16 y/o male presents with father for Recall.",
    "sent_key": ".txt_0_0",
    "entity_ls": ["Age", "Gender"],
    "entity_dict": {"Age": ["16 y/o"], "Gender": ["male"]}
}
```

Stage B input (inference json)

LLMs_inferences.py expects a directory of .json files, each containing a list of records with at least:

```json
{ "sentence": "..." }
```

The script writes a pred field per sentence for entities that were predicted:

```json
{ "pred": {"Age": ["16 y/o"], "Gender": ["male"]} }
```

convert_note2sent.py can produce this inference-ready JSON from raw .txt notes.

## Quickstart

### Stage A — generate instruction prompts

Option 1: wrapper script (edit paths inside the .sh file as needed)

```bash
bash prompt_generation.sh
```

Option 2: run Python directly (example)

```bash
python LLMs_prompt_generation.py \
    --version test_project --reset \
    --train_data_path /home/FullMouth/data/dataset/training_notes/ \
    --dst_root /home/FullMouth/data/ \
    --gpu_num 0 --base_model Qwen2.5-7B-Instruct \
    --num_of_instructions 3 --instruction_length 500 --validation_threshold 0.8 \
    --model_input_limit_ratio 0.8 \
    --revised_training_set --include_description --include_examples --error_feedback_in_loop
```

Outputs

- Prompt artifacts (including instruct_prompt_dict.json) are written under:
    - <dst_root>/<version>/<target_dir>/
- target_dir is derived from base_model plus flags (see get_model_name() in function_util.py)

### Stage B — run inference

Option 1: wrapper script (edit paths inside the .sh file as needed)

```bash
bash llm_inference.sh
```

Option 2: run Python directly (example)

```bash
python LLMs_inferences.py \
    --version test_project --gpu_num 6 --base_model Qwen2.5-7B-Instruct \
    --test_dir /home/FullMouth/data/dataset/test_notes \
    --dst_root /home/FullMouth/data/ \
    --num_of_instructions 3 --validation_threshold 0.9 \
    --result_type_dir gold_prompt \
    --model_input_limit_ratio 0.8 \
    --revised_training_set --include_description --include_examples --error_feedback_in_loop
```

Outputs

- The script reads instruct_prompt_dict.json from:
    - <dst_root>/<version>/<target_dir>/instruct_prompt_dict.json
- It creates selected prompts at:
    - <dst_root>/<version>/<target_dir>/selected_prompt_<postfix>.json
- It writes predictions to:
    - <dst_root>/<version>/<target_dir>/<result_type_dir>_<postfix>/*.json

Where postfix is computed as:

- <postfix> = <num_of_instructions>instructs<validation_threshold-without-dot>
    - example: 3instructs09 for validation_threshold=0.9

## Notes / gotchas

- gpu_num is required by the CLI and is used to set CUDA_VISIBLE_DEVICES inside the scripts.
- If you pass --model_name for inference, the model path becomes <model_root>/<version>/<model_name> (intended for SFT/DPO outputs). Otherwise it loads <model_root>/<base_model>.
- The pred field is only added for sentences/entities that pass the “checkInstruction” vote; absence of pred means “no prediction written”, not necessarily a negative label.

## Optional: SFT + DPO notebooks

The fine-tuning workflow is under SFT_DPO/ and is designed for the FullMouth environment (it contains hard-coded paths like /home/FullMouth/... that you’ll likely need to edit).

Install notebook-only deps with:

```bash
pip install -r SFT_DPO/requirements-sft_dpo.txt
```

## Utility: convert raw notes to JSON

convert_note2sent.py converts .txt notes to sentence-level .json suitable as inference input:

```bash
python convert_note2sent.py \
    --text_data_dir /home/FullMouth/data/dataset/test_notes/ \
    --output_dir /home/FullMouth/data/dataset/test_notes_json/ \
    --gpu_num 0 \
    --combined_sentences True
```

It requires spaCy and the en_core_web_trf model:

```bash
python -m spacy download en_core_web_trf
```

