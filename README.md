# FullMouth LLM NER Pipeline

[![arXiv](https://img.shields.io/badge/arXiv-2401.12345-b31b1b.svg)](https://arxiv.org/abs/2605.04221)
[![GitHub](https://img.shields.io/github/license/FullMouth/LLM-NER)](https://github.com/lifestrugglee/fullmouth/blob/main/LICENSE)


FullMouth is a two-stage pipeline for extracting dental named entities from clinical notes with locally hosted large language models (LLMs). It uses Hugging Face `transformers` to generate entity-specific extraction prompts, validate those prompts, and then apply the selected prompts to sentence-level clinical-note data.

The pipeline is designed for local model execution. Models are loaded from disk with `local_files_only=True`, so you must have the required model checkpoints available under the configured `model_root`.

## Pipeline overview

```text
Raw or annotated notes
        │
        ├── Stage A: Prompt generation
        │       LLMs_prompt_generation.py
        │       - generates candidate instructions per entity
        │       - validates and evaluates instructions
        │       - saves instruct_prompt_dict.json
        │
        └── Stage B: Inference
                LLMs_inferences.py
                - loads saved prompts
                - screens sentences for candidate entities
                - extracts entity mentions
                - writes per-note JSON predictions
```

Stage A learns reusable extraction instructions for each entity type. Stage B uses those instructions to run a two-step extraction process: first a boolean screening pass, then entity extraction for sentences that pass screening.

## Repository contents

### Main entry points

| File | Purpose |
| --- | --- |
| `LLMs_prompt_generation.py` | Generates, validates, and saves entity-specific instruction prompts to `<dst_root>/<version>/<target_dir>/instruct_prompt_dict.json`. |
| `LLMs_inferences.py` | Loads `instruct_prompt_dict.json`, selects prompts, runs screening and extraction, and writes JSON outputs with a `pred` field. |
| `convert_note2sent.py` | Converts raw `.txt` notes into sentence-level `.json` files suitable for inference. |

### Utilities and schema

| File | Purpose |
| --- | --- |
| `function_util.py` | Command-line parsing, model initialization, batching, generation helpers, and evaluation utilities. |
| `fullmouth_util.py` | FullMouth constants and schema-loading helpers. |
| `label_v4_1.json` | Dental NER entity schema and entity descriptions used by `FM_label`. |

### Configuration and dependencies

| File | Purpose |
| --- | --- |
| `config.yml` | Runtime configuration. At minimum, the main scripts require `model_root`. |
| `requirements.txt` | Baseline Python dependencies for prompt generation and inference. |

### Optional fine-tuning assets

The `SFT_DPO/` directory contains notebooks and dependencies for QLoRA supervised fine-tuning and DPO experiments:

- `SFT_DPO/SFT-int4.ipynb`
- `SFT_DPO/DPO_reward_init.ipynb`
- `SFT_DPO/DPO_training.ipynb`
- `SFT_DPO/requirements-sft_dpo.txt`

These notebooks are environment-specific and may include hard-coded FullMouth paths that need to be edited before use.

## Setup

### 1. Create or update `config.yml`

The scripts expect `config.yml` to be available in the current working directory. A minimal configuration is:

```yaml
model_root: /data_sys/lang_model
```

The checked-in configuration may include additional keys such as `root_dir` and `src_txt_root`. The two main entry points require `model_root`; other utilities may use the additional values.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The pinned `torch` and `transformers` versions reflect the FullMouth runtime environment. If your CUDA, driver, or Python environment differs, adjust the PyTorch installation accordingly.

### 3. Verify local models

Models are loaded from local paths only. For example, if you run with:

```bash
--base_model Qwen2.5-7B-Instruct
```

then the model is expected at:

```text
<model_root>/Qwen2.5-7B-Instruct
```

For fine-tuned models passed through `--model_name`, inference expects:

```text
<model_root>/<version>/<model_name>
```

## Data formats

preprocess raw clinical `.txt` notes into sentence-level `.json` files with `convert_note2sent.py`. This step uses spaCy sentence segmentation, so install the spaCy transformer English model first:

```bash
python -m spacy download en_core_web_trf
```

Then convert the notes:

```bash
python convert_note2sent.py \
  --text_data_dir /home/FullMouth/data/dataset/test_notes/ \
  --output_dir /home/FullMouth/data/dataset/test_notes_json/ \
  --gpu_num 0 \
  --combined_sentences True
```


### Stage A training input

`LLMs_prompt_generation.py` expects a directory of `.json` files. Each file should contain a list of sentence records like this:

```json
{
  "sentence": "D: 16 y/o male presents with father for Recall.",
  "sent_key": ".txt_0_0",
  "entity_ls": ["Age", "Gender"],
  "entity_dict": {
    "Age": ["16 y/o"],
    "Gender": ["male"]
  }
}
```

Required fields:

- `sentence`: source sentence text
- `entity_ls`: list of entity labels present in the sentence
- `entity_dict`: mapping from entity label to one or more annotated spans

### Stage B inference input

`LLMs_inferences.py` expects a directory of `.json` files. Each file should contain a list of records with at least a `sentence` field:

```json
{
  "sentence": "D: 16 y/o male presents with father for Recall."
}
```

The inference script writes a `pred` field for predicted entities:

```json
{
  "sentence": "D: 16 y/o male presents with father for Recall.",
  "pred": {
    "Age": ["16 y/o"],
    "Gender": ["male"]
  }
}
```

A missing `pred` field means no prediction was written for that sentence. It does not necessarily mean the sentence was explicitly classified as negative.

## Quickstart

The repository includes shell wrappers with example commands. Their default paths are specific to the FullMouth environment, so review and update them before running.

### Stage A: generate instruction prompts

Run the wrapper:

```bash
bash prompt_generation.sh
```

Or run Python directly:

```bash
python LLMs_prompt_generation.py \
  --version test_project \
  --reset \
  --train_data_path /home/FullMouth/data/dataset/training_notes/ \
  --dst_root /home/FullMouth/data/ \
  --gpu_num 0 \
  --base_model Qwen2.5-7B-Instruct \
  --num_of_instructions 3 \
  --instruction_length 500 \
  --validation_threshold 0.8 \
  --model_input_limit_ratio 0.8 \
  --revised_training_set \
  --include_description \
  --include_examples \
  --error_feedback_in_loop
```

Stage A writes prompt artifacts to:

```text
<dst_root>/<version>/<target_dir>/
```

The key output is:

```text
<dst_root>/<version>/<target_dir>/instruct_prompt_dict.json
```

`target_dir` is derived from the base model and selected flags. See `get_model_name()` in `function_util.py` for the exact naming logic.

### Stage B: run inference

Run the wrapper:

```bash
bash llm_inference.sh
```

Or run Python directly:

```bash
python LLMs_inferences.py \
  --version test_project \
  --gpu_num 6 \
  --base_model Qwen2.5-7B-Instruct \
  --test_dir /home/FullMouth/data/dataset/test_notes \
  --dst_root /home/FullMouth/data/ \
  --num_of_instructions 3 \
  --validation_threshold 0.9 \
  --result_type_dir gold_prompt \
  --model_input_limit_ratio 0.8 \
  --revised_training_set \
  --include_description \
  --include_examples \
  --error_feedback_in_loop
```

Stage B reads prompts from:

```text
<dst_root>/<version>/<target_dir>/instruct_prompt_dict.json
```

It saves selected prompts to:

```text
<dst_root>/<version>/<target_dir>/selected_prompt_<postfix>.json
```

It writes prediction files to:

```text
<dst_root>/<version>/<target_dir>/<result_type_dir>_<postfix>/*.json
```

The postfix is computed from the number of instructions and the validation threshold:

```text
<num_of_instructions>instructs<validation_threshold-without-dot>
```

For example, `--num_of_instructions 3 --validation_threshold 0.9` produces:

```text
3instructs09
```

## Convert raw notes to inference JSON

Use `convert_note2sent.py` to convert raw `.txt` notes into sentence-level `.json` files:

```bash
python convert_note2sent.py \
  --text_data_dir /home/FullMouth/data/dataset/test_notes/ \
  --output_dir /home/FullMouth/data/dataset/test_notes_json/ \
  --gpu_num 0 \
  --combined_sentences True
```

This utility requires spaCy and the transformer English model:

```bash
python -m spacy download en_core_web_trf
```

## Common command-line options

| Option | Used by | Description |
| --- | --- | --- |
| `--version` | Both | Dataset, experiment, or model version namespace. |
| `--gpu_num` | Both | CUDA device index. The scripts set `CUDA_VISIBLE_DEVICES` internally. |
| `--base_model` | Both | Base model directory name under `model_root`. |
| `--model_name` | Inference | Fine-tuned model name. When provided, the model path becomes `<model_root>/<version>/<model_name>`. |
| `--num_of_instructions` | Both | Number of prompts to generate or use per entity. |
| `--validation_threshold` | Both | Minimum quality threshold used for validation or prompt selection. |
| `--instruction_length` | Prompt generation | Target instruction length. |
| `--model_input_limit_ratio` | Both | Fraction of the model context window to use for batching. |
| `--revised_training_set` | Prompt generation | Enables LLM-assisted training-set revision. |
| `--include_description` | Prompt generation | Includes schema descriptions when generating instructions. |
| `--include_examples` | Prompt generation | Includes examples in prompt-generation context. |
| `--error_feedback_in_loop` | Prompt generation | Feeds validation errors back into later instruction-generation rounds. |
| `--result_type_dir` | Inference | Prefix for the output prediction directory. |

## Operational notes

- `--gpu_num` is required by the main scripts.
- Wrapper scripts are examples, not portable defaults. Edit FullMouth-specific paths such as `/home/FullMouth/...` before running in another environment.
- Keep `config.yml` in the directory from which you launch the scripts.
- Make sure local model checkpoints exist before running; the scripts do not download models automatically.
- The inference pipeline only writes `pred` for entities that pass the screening vote.
- If you change prompt-generation flags, use the same relevant flags during inference so `target_dir` resolves to the same prompt directory.

## Optional: SFT and DPO

Install notebook-specific dependencies with:

```bash
pip install -r SFT_DPO/requirements-sft_dpo.txt
```

Then review the notebooks in `SFT_DPO/`. Update hard-coded paths before running them outside the original FullMouth environment.

## License

See `LICENSE` for licensing information.
