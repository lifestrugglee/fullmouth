# FullMouth LLM NER pipeline (prompt generation + inference)

This folder contains a two-stage pipeline for extracting dental NER entities from clinical notes using *local* LLMs (HuggingFace `transformers`).

- **Stage A: prompt generation** (`LLMs_prompt_generation.py`) generates and evaluates instruction prompts per entity.
- **Stage B: inference** (`LLMs_inferences.py`) runs ensemble screening + extraction using the generated prompts.

The `.sh` files (`prompt_generation.sh`, `llm_inference.sh`) are thin wrappers with example arguments.

## Folder contents

- `LLMs_prompt_generation.py`: generate `instruct_prompt_dict.json`.
- `LLMs_inferences.py`: run inference and write per-note JSON outputs with a `pred` field.
- `SFT-int4.ipynb`: supervised fine-tuning (SFT) with QLoRA + 4-bit quantization (TRL `SFTTrainer`).
- `DPO_reward_init.ipynb`: builds DPO preference pairs (`chosen`/`rejected`) from predictions vs gold.
- `DPO_training.ipynb`: DPO training with QLoRA + 4-bit quantization (TRL `DPOTrainer`).
- `function_util.py`: shared CLI (`parse_args`) and LLM / batching / evaluation utilities.
- `fullmouth_util.py`: constants and label schema loader (`FM_label`).
- `convert_note2sent.py`: utility to convert raw `.txt` notes to sentence-level `.json` (useful as inference input).
- `config.yml`: **required**; contains `model_root` for local model paths.

## Requirements

- Linux + NVIDIA GPU recommended.
- Python environment with (at minimum): `torch`, `transformers`, `pyyaml`, `scikit-learn`, `rapidfuzz`, `tqdm`.
- A local model directory under `model_root` (see config section).

Important: entity schema is loaded from absolute paths in `fullmouth_util.py` (e.g., `/home/FullMouth/GUI/label_v4.pkl`). If your environment differs, update those paths.

## Configuration (`config.yml`)

Both Python entrypoints require a `config.yml` in the current working directory.

Minimal example:

```yaml
model_root: /data_sys/lang_model
```

This repo includes extra keys (`root_dir`, `src_txt_root`) that may be used by other scripts; the two main entrypoints only require `model_root`.

## Data formats

### Training JSON format (Stage A)

`LLMs_prompt_generation.py` expects a directory of `.json` files, each containing a list of sentence records like:

```json
{
    "sentence": "D: 16 y/o male presents with father for Recall.",
    "sent_key": ".txt_0_0",
    "entity_ls": ["Age", "Gender"],
    "entity_dict": {"Age": ["16 y/o"], "Gender": ["male"]}
}
```

### Inference JSON format (Stage B)

`LLMs_inferences.py` expects a directory of `.json` files, each containing a list of records with at least:

```json
{ "sentence": "..." }
```

The script writes a `pred` field per sentence:

```json
{ "pred": {"Age": ["16 y/o"], "Gender": ["male"]} }
```

`convert_note2sent.py` can produce this inference-ready JSON from raw `.txt` notes.

## Quickstart

Run everything from this folder so the scripts can find `config.yml`:

```bash
cd FullMouth/code/github
```

### Stage A — generate instruction prompts

Option 1 (recommended): use the wrapper:

```bash
bash prompt_generation.sh
```

Option 2: run Python directly (example):

```bash
python LLMs_prompt_generation.py \
    --version test_project --reset \
    --train_data_path /home/FullMouth/data/dataset/training_notes/ \
    --dst_root /home/FullMouth/data/ \
    --gpu_num 0 --base_model Qwen2.5-7B-Instruct \
    --num_of_instructions 3 --instruction_length 500 --validation_threshold 0.8 \
    --add_stop_token --model_input_limit_ratio 0.8 \
    --revised_training_set --include_description --include_examples --error_feedback_in_loop
```

Outputs are written under:

`<dst_root>/<version>/<model-flags-dir>/instruct_prompt_dict.json`

### Stage B — run inference

```bash
bash llm_inference.sh
```

Or directly:

```bash
python LLMs_inferences.py \
    --version test_project --gpu_num 6 --base_model Qwen2.5-7B-Instruct \
    --test_dir /home/FullMouth/data/dataset/test_notes \
    --dst_root /home/FullMouth/data/ \
    --num_of_instructions 3 --validation_threshold 0.9 \
    --result_type_dir gold_prompt \
    --add_stop_token --model_input_limit_ratio 0.8 \
    --revised_training_set --include_description --include_examples --error_feedback_in_loop
```

This reads `instruct_prompt_dict.json` from Stage A and writes predictions into:

`<dst_root>/<version>/<model-flags-dir>/<result_type_dir>_<postfix>/*.json`

Where `<postfix>` looks like `3instructs09` (derived from `num_of_instructions` and `validation_threshold`).

## Notes / gotchas

- `LLMs_inferences.py` requires `--result_type_dir` (it is used to name the output folder).
- Both entrypoints set `CUDA_VISIBLE_DEVICES` internally based on `--gpu_num`.
- Models are loaded with `local_files_only=True`. If the model isn’t present under `model_root`, loading will fail.
- `--model_name` / `--checkpoint_dir` change *which model weights are loaded*, but output folders are still named from `--base_model` plus the prompt-related flags (see `get_model_name()` in `function_util.py`). If you use these options, rely on the script’s printed `model_dir` / `Source model dir path` to confirm paths.

## Fine-tuning pipelines (SFT + DPO)

This folder also includes an optional fine-tuning workflow implemented as notebooks:

1) **SFT (supervised fine-tuning)** to teach the model the “checkInstruction” and “resultsInstruction” behaviors.
2) **DPO (Direct Preference Optimization)** to further optimize the model using preference pairs created from model mistakes.

These notebooks are designed for the FullMouth environment and include hard-coded paths like `%cd /home/FullMouth/code` and dataset roots under `/home/FullMouth/data/...`. Expect to edit paths before running elsewhere.

### Dependencies (notebooks)

In addition to the inference/prompt-generation deps, the SFT/DPO notebooks use:

- `datasets` (HF)
- `trl` (for `SFTTrainer`, `DPOTrainer`)
- `peft` (LoRA / QLoRA)
- `bitsandbytes` (4-bit quantization via `BitsAndBytesConfig`)

Some notebook configs set `attn_implementation="flash_attention_2"`; if your environment doesn’t have FlashAttention, disable it in the notebook.

### SFT: `SFT-int4.ipynb`

What it does:

- Loads `selected_prompt_<postfix>.json` (the top-N prompts per entity) from a prompt-generation run.
- Builds supervised examples for both:
    - **checkInstruction**: message list + gold answer (`Yes`/`No`)
    - **resultsInstruction**: message list + gold JSON output (`{ "sentence": ..., "entity_ls": [...] }`)
- Writes the SFT datasets under the model’s `data_dir`:
    - `SFT_training_data.json`
    - `SFT_val_data.json`
- Fine-tunes a base model using **QLoRA (LoRA + 4-bit NF4)** via TRL `SFTTrainer`.

Key paths/outputs (as written in the notebook):

- `data_dir`: `/home/FullMouth/data/instruct_<version>/<model_name>`
- `selected_prompt_fp`: `${data_dir}/selected_prompt_<num>instructs<threshold>.json`
- `model_output_dir`: `/data_sys/lang_model/<version>/<model_name>`

### DPO data creation: `DPO_reward_init.ipynb`

What it does:

- Loads a folder of model predictions (`*.json`) that contain `pred` per sentence.
- Loads the corresponding gold training JSON from `/home/FullMouth/data/<json_fn>`.
- For mismatches, creates DPO preference pairs for both:
    - **resultsInstruction**: `chosen` = gold entities, `rejected` = model prediction (or `{}`)
    - **checkInstruction**: `chosen` = correct `Yes/No`, `rejected` = incorrect `Yes/No`
- Writes per-file DPO cases to `output_dir`, then compiles them into:
    - `dpo_data_train.jsonl`
    - `dpo_data_val.jsonl`

The generated DPO units are JSON-lines of the form:

```json
{"prompt": [...chat messages...], "chosen": [...assistant msg...], "rejected": [...assistant msg...]}
```

### DPO training: `DPO_training.ipynb`

What it does:

- Loads `dpo_data_train.jsonl` and `dpo_data_val.jsonl` created above.
- Loads a base model from a local directory (example pattern in the notebook):
    - `load_model_dir = /data_sys/lang_model/<version>/<model_name>_<model_settings>`
- Trains with TRL `DPOTrainer` using QLoRA + 4-bit NF4.
- Saves the trained output to:
    - `model_output_dir = load_model_dir + "_DPO"`

### Using SFT/DPO models with inference

`LLMs_inferences.py` supports loading from `<model_root>/<version>/<model_name>` when you pass `--model_name <model_name>` (and the same `--version`). That matches the SFT notebook’s output layout (`/data_sys/lang_model/<version>/<model_name>`).

## Utility: convert raw notes to JSON

`convert_note2sent.py` converts `.txt` notes to sentence-level `.json` suitable as inference input:

```bash
python convert_note2sent.py \
    --text_data_dir /home/FullMouth/data/dataset/test_notes/ \
    --gpu_num 0 --combined_sentences True
```

It requires spaCy and the `en_core_web_trf` model.

