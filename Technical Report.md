# Technical Report: LLM-Based Entity Extraction Pipeline

**Date:** May 5, 2026  
**Scope:** `LLMs_prompt_generation.py` (Instruction Generation) and `LLMs_inferences.py` (Gold-Standard Inference)

---

## 1. Executive Summary

This report documents a two-stage pipeline for **automated Named Entity Recognition (NER) in dental clinical notes** using locally deployed Large Language Models (LLMs). The system follows a meta-learning paradigm: Stage 1 generates, validates, and selects entity-specific instruction prompts via an iterative optimization loop; Stage 2 applies the optimized prompts to extract entities from unseen clinical text.

Both scripts share a common utility layer (`function_util.py`, `fullmouth_util.py`) and operate on the FullMouth dental annotation dataset.

---

## 2. Architecture Overview

```
┌───────────────────────────────────────┐
│   Stage 1: Instruction Generation     │
│   (LLMs_prompt_generation.py)       │
│                                       │
│ Training Data ──► LLM Generates       │
│                   Instruction Prompt   │
│                       │                │
│                   Verify & Revise      │
│                       │                │
│               Validate on Val Set      │
│                       │                │
│              Evaluate on Test Set      │
│                       │                │
│          Save instruct_prompt_dict.json │
└───────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│   Stage 2: Gold-Standard Inference    │
│   (LLMs_inferences.py)  │
│                                       │
│ instruct_prompt_dict.json ──►          │
│   Prompt Selection (top-N by F1)      │
│                       │                │
│  Clinical Notes ──► Phase 1:          │
│                   Boolean Screening    │
│                       │                │
│                   Phase 2:             │
│                   Entity Extraction    │
│                       │                │
│          Save per-file predictions     │
└───────────────────────────────────────┘
```

---

## 3. Stage 1 — Instruction Generation (`LLMs_prompt_generation.py`)

### 3.1 Purpose

Automatically generates high-quality NER instruction prompts for each target entity type defined in the FullMouth annotation schema (loaded from `label.json`). Each prompt acts as a reusable extraction specification that tells the LLM precisely how to identify and extract a particular entity from dental clinical text.

### 3.2 Data Preparation

| Step | Description |
|------|-------------|
| **Data Loading** | Annotated training sentences are loaded from json files under `training_data_swap_revised/`. Each record contains a sentence, entity list, and entity dictionary. |
| **Entity Splitting** | For each target entity, data is partitioned into positive samples (sentence contains the entity) and negative samples (sentence does not). |
| **Simplification** | Raw annotation dicts are converted to `EntityData(sentence, entity_ls)` dataclass objects. Negative samples receive empty entity lists. |
| **Train/Val/Test Split** | When ≥ 10 positive samples exist: 80/10/10 split (with stratified neg sampling). Otherwise: all positives go to training; no validation set is created. |
| **Negative Mixing** | Each split is augmented with negative samples at configurable ratios (`train_neg_ratio=3`, `val_neg_ratio=10`, `test_neg_ratio=100`) via `build_mixed_split()`. |

### 3.3 Optional Training Set Revision

When `--revised_training_set` is enabled, positive training samples are deduplicated and curated by the LLM itself (`getRevisedTrainingSet()`). This removes near-duplicate or low-quality examples before instruction generation.

### 3.4 Instruction Generation Loop

For each entity type and each instruction number (1..`num_of_instructions`), the system executes an iterative refinement loop (max `ROUND_THRES = 5` rounds):

1. **Generation:** `getInstruction()` prompts the LLM with training examples to produce a structured NER instruction. Configurable options include entity description, example inclusion, instruction length, and error feedback from prior rounds.

2. **Verification:** `verifyInstruction()` evaluates the generated prompt against 9 quality criteria (clarity, completeness, non-truncation, no embedded examples, no duplication, NER-appropriate, etc.) via batched True/False LLM queries. All 9 must pass.

3. **Revision:** If verification fails, `reviseInstruction()` asks the LLM to improve the prompt for clarity, structure, and completeness.

4. **Validation Evaluation:**
   - **Phase 1 (Boolean Screening):** Each validation sentence is checked for entity relevance via `get_bool_batch_result_ls()` — a batched Yes/No classification that respects token window limits.
   - **Phase 2 (Entity Extraction):** Relevant sentences are processed through `get_batch_InstructionResult_ls()` to extract entities.
   - **Metrics:** `evaluate_mixed_entity_extraction()` computes precision, recall, F1, negative-sentence accuracy, and positive-sentence exact accuracy using greedy 1-to-1 matching with a similarity function.

5. **Convergence:** The loop converges when validation F1 ≥ `validation_threshold` (default: 0.8) or after exhausting `ROUND_THRES` rounds. The best-scoring prompt (by F1) is retained regardless of convergence.

6. **Error Feedback (Optional):** When `--error_feedback_in_loop` is enabled, false positives and false negatives from the validation evaluation are fed back into the next instruction generation round as revision guidance.

### 3.5 Test Set Evaluation

After loop completion, the best instruction prompt is evaluated on the held-out test set using the same two-phase pipeline (Boolean Screening → Entity Extraction). Test metrics are stored alongside the prompt.

### 3.6 Low-Data Handling

For entities with fewer than 10 positive samples, the system skips validation-based optimization:
- All positives are used for training.
- One instruction per `num_of_instructions` is generated without iterative refinement.
- Verification and revision are still applied.
- Test evaluation is performed against negatives only.

### 3.7 Outputs

- `instruct_prompt_dict.json` — Nested dictionary: `{entity → {instruction_num → {INSTRUCT_PROMPT, EVAL_MTX, TEST_MTX}}}` plus per-entity execution times.
- Log files with timestamped entries (rotating, max 10 MB per file, 5 backups).
- Telegram notification on completion.

---

## 4. Stage 2 — Gold-Standard Inference (`LLMs_prompt_generation_gold.py`)

### 4.1 Purpose

Applies the optimized instruction prompts from Stage 1 to extract entities from full clinical note corpora (training or test partitions), producing per-file prediction outputs for downstream evaluation.

### 4.2 Prompt Selection

`instruct_prompt_preparation()` filters and ranks instructions per entity:
1. Filters by test F1 ≥ `validation_threshold`.
2. Selects top-N prompts (by score) per entity.
3. If insufficient valid prompts exist, fills from the highest-scoring failures (makeup strategy).
4. Persists selection to `selected_prompt_{postfix}.json` for reproducibility.

### 4.3 Two-Phase Inference Pipeline

For each clinical note file and each entity type:

**Phase 1 — Multi-Prompt Boolean Screening:**
- All N instruction prompts independently classify each sentence as containing/not-containing the target entity.
- A **majority voting** mechanism aggregates: a sentence passes if `sum(votes) > N/2`.
- This reduces false negatives while controlling false positives through ensemble consensus.

**Phase 2 — Multi-Prompt Entity Extraction:**
- Sentences that pass screening are processed through **all** instruction prompts simultaneously via `get_batch_instruction_ls_result_ls()`.
- Results across prompts are merged per sentence using `EntityData.update()` (union with deduplication, preserving first-seen order).
- This ensemble approach improves recall by capturing entities that individual prompts might miss.

### 4.4 Result Aggregation

Extracted entities are written into the gold-standard data structure under a `PRED` key:
```python
gold_content_ls[sent_idx]['PRED'][entity] = entity_data.entity_ls
```

Each file's augmented content is saved as a json file in the result directory.

### 4.5 Operational Features

| Feature | Description |
|---------|-------------|
| **Resumability** | Existing output files are skipped (`if os.path.exists(dst_json_fp): continue`). |
| **Reverse Processing** | `--go_reverse` enables processing files in reverse order (useful for parallel execution from both ends). |
| **Index Slicing** | `--start_idx` and `--end_idx` enable batch parallelism across multiple runs. |
| **Progress Reporting** | Telegram notifications every 200 files, plus completion notification. |
| **Timing** | Per-file and cumulative execution time tracking. |

---

## 5. Shared Infrastructure

### 5.1 Model Management

- Models are loaded via `llms_setup()` which initializes `AutoModelForCausalLM` + `AutoTokenizer` from a local directory.
- A `MODEL_DISPATCH` table supports specialized loaders (e.g., `mistral3` → `Mistral3ForConditionalGeneration` with FP8 quantization).
- EOS/PAD token handling ensures compatibility across model families.
- Global state: `model`, `tokenizer`, `device`, `model_max_window`, `no_sys_prompt`.

### 5.2 Token-Aware Batching

Both scripts use a chunking strategy that respects model context windows:
- Input messages are partitioned into chunks where total prompt tokens ≤ `model_max_window × model_input_limit_ratio`.
- OOM errors trigger automatic retry with `ratio × 0.8`.
- Token usage is tracked globally (`printTokensUpdate()`).

### 5.3 Evaluation Framework

`evaluate_mixed_entity_extraction()` implements:
- **Entity-level metrics:** Precision, recall, F1 via greedy 1-to-1 matching with a configurable similarity function (`is_similarity` from `fullmouth_util`).
- **Sentence-level metrics:** Negative sentence accuracy (FP rate), positive sentence exact match accuracy.
- **Error case tracking:** Lists of missed entities, spurious entities, and false-positive negative sentences for feedback.

### 5.4 `EntityData` Dataclass

```python
@dataclass(slots=True)
class EntityData:
    sentence: str           # Source clinical text
    entity_ls: List[str]    # Extracted entity mentions
```

Supports serialization (`to_dict`, `from_dict`, `get_json_str`), incremental update with deduplication, and string-normalized sentence equality.

### 5.5 Entity Schema (`FM_label`)

- Loaded from `label.json` at module initialization.
- `gold_entity_ls`: Flat list of all annotable dental NER entity types.
- `get_entity_description(name)`: Returns human-readable description for inclusion in instruction prompts.

---

## 6. Configuration & Command-Line Interface

### Key Arguments

| Argument | Default | Stage | Description |
|----------|---------|-------|-------------|
| `--version` | (required) | Both | Dataset/model version identifier |
| `--gpu_num` | (required) | Both | CUDA device index |
| `--base_model` | None | Both | Base model name for path construction |
| `--model_name` | None | Both | Fine-tuned model name (overrides base) |
| `--num_of_instructions` | 1 | Both | Number of instruction prompts per entity |
| `--validation_threshold` | 0.8 | Both | F1 threshold for prompt quality |
| `--instruction_length` | 500 | Stage 1 | Target token length for generated instructions |
| `--train_neg_ratio` | 3 | Stage 1 | Negative sample multiplier for training |
| `--val_neg_ratio` | 10 | Stage 1 | Negative sample multiplier for validation |
| `--test_neg_ratio` | 100 | Stage 1 | Negative sample multiplier for testing |
| `--reset` | False | Stage 1 | Clear output directory before run |
| `--revised_training_set` | False | Stage 1 | Enable LLM-based training set curation |
| `--include_description` | False | Stage 1 | Include entity descriptions in prompts |
| `--include_examples` | False | Stage 1 | Include examples in prompts |
| `--error_feedback_in_loop` | False | Stage 1 | Feed errors back to instruction generation |
| `--model_input_limit_ratio` | 0.8 | Both | Fraction of context window to use |
| `--run_train` / `--run_test` | False | Stage 2 | Select training or test partition |
| `--go_reverse` | False | Stage 2 | Process files in reverse order |
| `--start_idx` / `--end_idx` | 0 / -1 | Stage 2 | File index range for batch processing |

---

## 7. Data Flow Diagram

```
label.json ──► FM_label (entity schema)
                      │
training_data_swap_revised/*.json ──► Stage 1 ──► instruct_prompt_dict.json
                                                         │
                                          instruct_prompt_preparation()
                                                         │
                                                selected_prompt.json
                                                         │
gold_dir/*.json (annotated notes) ──► Stage 2 ──► result_dir/*.json (with PRED)
```

---

## 8. Design Observations & Considerations

### Strengths

1. **Self-improving prompts:** The iterative generation–verification–validation loop with error feedback enables the system to progressively refine instruction quality without human intervention.

2. **Ensemble inference:** Multi-prompt majority voting (screening) combined with union-based extraction (prediction) balances precision and recall effectively.

3. **Robustness:** Token-aware batching with OOM recovery, rotating log files, resumable processing, and graceful degradation for low-data entities demonstrate production-grade engineering.

4. **Flexibility:** Extensive CLI arguments allow systematic ablation studies (e.g., with/without descriptions, examples, error feedback, training set revision).

### Areas for Consideration

1. **Global mutable state:** The LLM model, tokenizer, and configuration are maintained as module-level globals in `function_util.py`. This works for single-process execution but limits testability and potential multi-model workflows.

2. **Duplicate line:** In Stage 1 (line ~137), `instruction_prompt = best_validation_result_dict[INSTRUCT_PROMPT]` appears twice consecutively — a harmless but unnecessary duplication.

3. **Low-data entity path (Stage 1, `else` branch):** References `pos_val` and `pos_test` in a log statement, but these variables are not defined in the `else` branch (they belong to the `if len(entity_data) >= 10` branch). This would cause a `NameError` at runtime if an entity has < 10 samples.

4. **No system prompt mode:** The `--no_system_prompt` flag merges system messages into user messages, enabling compatibility with models that don't support system roles (e.g., certain Mistral variants).

5. **Deterministic reproducibility:** Random seeds are fixed for splits (42, 43, 44) and negative sampling, supporting reproducible experiments. However, LLM generation itself (temperature, sampling) is controlled in the utility layer's generation settings, which are not exposed as CLI arguments in these scripts.

---

## 9. Runtime Dependencies

| Component | Source |
|-----------|--------|
| `function_util.py` | Core utility layer (LLM calls, evaluation, data handling) |
| `fullmouth_util.py` | Constants, `FM_label`, `is_similarity`, json I/O |
| `transformers` | HuggingFace model/tokenizer loading |
| `torch` | GPU inference |
| `scikit-learn` | `train_test_split` |
| `tg_bot_send` | Telegram progress notifications |
| Local model files | Under `/data_sys/lang_model/` |
| Annotation data | Under `/home/FullMouth/data/annotation/` |

---

## 10. Conclusion

The pipeline implements an end-to-end, LLM-driven NER system for dental clinical notes. Stage 1 automates the traditionally manual process of crafting extraction instructions through a generate–verify–validate loop, while Stage 2 applies these instructions at scale with ensemble-based multi-prompt inference. The architecture balances automation with configurability, enabling systematic experimentation across model variants, prompt strategies, and evaluation thresholds.
