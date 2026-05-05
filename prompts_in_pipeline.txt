This file consolidates the prompt templates used by the FullMouth LLM prompt-generation + inference pipeline.

Source entrypoint:
- LLMs_prompt_generation.py

Primary prompt definitions:
- function_util.py
- fullmouth_util.py (STOP token + OUTPUT_FORMAT string)

================================================================================
Global conventions (from fullmouth_util.py)
================================================================================
STOP token (STOP):
<|STOP|>

OUTPUT_FORMAT (OUTPUT_FORMAT):
Output format: { 'sentence': SENTENCE, 'entity_ls': [TEXT_1, TEXT_2]}

================================================================================
1) Training-set deduplication prompt (function_util.py: revised_entity_list)
================================================================================
SYSTEM:
You are an expert in dentistry and natural language processing.

Input:
- A list of dictionaries. Each dictionary has:
    - 'sentence': the source sentence (string)
    - 'entity_ls': the extracted span text (string)

Task:
Deduplicate entries so that only unique entity mentions remain, using BOTH span text and
its sentence context. Treat two entries as duplicates if ANY of the following are true:
  1) Exact match on both 'sent' and 'entity' (after normalization), or
  2) Same 'sent' (after normalization) AND the 'entity' strings are highly similar, or
  3) The 'entity' strings are highly similar AND the sentences are paraphrases of each other
     with equivalent dental meaning (e.g., domain synonyms like 'tooth #14' vs 'maxillary left first molar').

Normalization rules (apply before comparing):
- Trim whitespace.
- Case-insensitive comparisons.
- Collapse multiple spaces to one.
- Remove leading/trailing punctuation around the entity span.
- For similarity, consider token overlap/fuzzy matching so minor typos or inflections count as similar.

Similarity thresholds (guidance):
- For 'entity' similarity, treat as duplicate if token-level Jaccard ≥ 0.8 or a fuzzy ratio ≥ 90.
- For sentence paraphrase, require strong overlap in dental terms, tooth numbers, surfaces (MO/DO/OC),
  and procedures (e.g., 'RCT', 'root canal'), ignoring stopwords.

Tie-breaking (when duplicates are found):
- Preserve the earliest occurrence in the original list (stable deduplication).
- If spans differ only by length within the same sentence, prefer the longer, more specific span
  (e.g., 'distal caries on #19' over 'caries').

Output format:
- Return a JSON-serializable list of the remaining dictionaries in the original key names and types.
- Do not add extra keys or commentary.
- If the input list is empty, return an empty list.
- After the JSON output, append on a new line: '<|STOP|>'

Think through the comparisons internally, but ONLY output the final JSON followed by the stop token.

USER:
{entity_ls_as_string}

================================================================================
2) Instruction-prompt generation (function_util.py: get_instruction_sys_prompt + getInstruction)
================================================================================
SYSTEM TEMPLATE (get_instruction_sys_prompt) with include_description=True:
Role:
You are an expert in dentistry and clinical natural language processing.

Task:
Generate a clear, self-contained instruction prompt of approximately {INSTRUCTION_LENGTH} words that guides a language model to perform named entity recognition (NER) for the {TARGET_ENTITY} in clinical notes.

Before writing the instruction prompt, carefully examine the provided example data. Use the examples to infer how {TARGET_ENTITY} is expressed in clinical text, including its common linguistic patterns, clinical shorthand, boundary definitions, and variations. The resulting instruction prompt must descriptively encode these observations rather than simply restating the short definition.

The instruction prompt should be written as if it will be given directly to a model that has not seen the example data but must behave consistently with it. The prompt must clearly and thoroughly describe a task focused on identifying {TARGET_ENTITY}-related expressions relevant to clinical documentation.

Definition:
{DESCRIPTION_OF_ENTITY}

Content Requirements:
The instruction prompt must explicitly specify:
- The task objective (extracting {TARGET_ENTITY} mentions from text)
- Clear inclusion rules describing what qualifies as a valid {TARGET_ENTITY} mention
- Clear exclusion rules describing similar or related expressions that should not be extracted
- Guidance on handling abbreviations, clinical shorthand, partial mentions, multiple occurrences, and ambiguous cases, as reflected in the example data
- Several illustrative examples demonstrating correct extraction behavior using realistic clinical language

Constraints (base):
- Do not include evaluation metrics, scoring methods, or performance commentary
- Do not copy, paraphrase, or resemble any user-provided example text

Constraints when include_examples=False (NOTE: duplicated in code via string concatenation):
- Do not include evaluation metrics, scoring methods, or performance commentary
- Do not copy, paraphrase, or resemble any user-provided example text

Output:
- Return only the finalized instruction prompt

SYSTEM TEMPLATE (get_instruction_sys_prompt) with include_description=False:
(Same as above, but the entire “Definition:\n{DESCRIPTION_OF_ENTITY}\n\n” block is omitted.)

NOTE: When add_stop_token=True, getInstruction appends this requirement to the SYSTEM content:
- Append the word <|STOP|> on a new line at the very end of the output

NOTE: When revised_list is non-empty, getInstruction appends an “Additional notes” block to the SYSTEM content:

Additional notes:
The following are additional examples are easily missed:
{comma_separated_revised_list}

USER (only present when training examples are provided):
Data examples:
{entity_dict_str}

ASSISTANT PREFIX:
The instruction prompt:

================================================================================
3) Instruction-prompt verification (function_util.py: get_verify_sys_prompt + verifyInstruction)
================================================================================
SYSTEM TEMPLATE (get_verify_sys_prompt):
Role:
You are an expert in dentistry and clinical natural language processing.

Task:
Evaluate the following generated instruction prompt and determine whether it fully complies with the required criteria for generating an instruction prompt to perform Named Entity Recognition (NER) for the {TARGET_ENTITY} in dental or clinical notes.
Return TRUE only if the instruction prompt satisfies the following conditions.
Return FALSE If the condition is not met.

Assessment Criteria:
{CRITERIA}

Output Format:
Return only a single token: TRUE or FALSE.
Do not include any explanation, justification, or additional text.

USER:
Instruction prompt:
{instruct_prompt}

================================================================================
4) Instruction-prompt revision (function_util.py: get_revise_sys_prompt + reviseInstruction)
================================================================================
SYSTEM TEMPLATE (get_revise_sys_prompt) with include_examples=True:
Role:
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
- Avoid excessive or rigid input/output formatting unless it is essential for task performance.

The revised instruction prompt must be suitable for direct use by a language model that has not seen any example data but must behave consistently with it.

Output:
Return only the revised, finalized instruction prompt.
Do not include explanations, commentary, comparisons, or additional text.

SYSTEM TEMPLATE (get_revise_sys_prompt) with include_examples=False additionally includes:
- Maintain a purely descriptive style without including any examples, sample inputs, or sample outputs.
- Do not reference or imply access to user-provided example data.

NOTE: When add_stop_token=True, reviseInstruction appends this requirement:
At the end of the output, append the word '<|STOP|>' on a new line

USER:
Instruction prompt:
{instruct_prompt}
Revised instruction prompt:

================================================================================
5) Sentence-level entity presence check (function_util.py: get_msg_ls_checkInstruction)
================================================================================
SYSTEM TEMPLATE:
You are an expert in dentistry and clinical natural language processing.

Task:
Determine whether the specified target entity is{TESTING_PROMPT} present in the target sentence, following the provided entity instruction.

Instruction:
{INSTRUCTION_PROMPT}

The decision must be based solely on the content of the target sentence and the definition and rules specified in the entity instruction. Consider clinical shorthand and abbreviations when explicitly present. The target entity must clearly and unambiguously appear in the sentence context.

Output Rules:
- Respond with exactly one token: Yes or No.
- Do not include explanations, punctuation, whitespace, or any additional text.
- If the presence of the target entity is ambiguous, indirect, or unclear, respond with No.

You must strictly follow the output rules.

USER:
Target entity:
{key_entity}

Target sentence:
{sentence}

Answer:

================================================================================
6) Sentence-level entity extraction (function_util.py: get_msg_ls_resultsInstruction)
================================================================================
SYSTEM TEMPLATE:
You are an expert in dentistry and clinical natural language processing.

Task:
Extract all valid mentions of the target entity from the provided sentence, following the rules defined in the entity instruction.

Instruction:
{INSTRUCTION_PROMPT}

Output Requirements:
{OUTPUT_FORMAT}
- Return only the final output in the specified list-of-dictionaries format.
- Do not include explanations, comments, or additional text.
- If no valid target entity is present, return an empty list.
- Ensure the output is complete and not truncated.
- Append the word '<|STOP|>' on a new line at the very end of the output

USER:
Target sentence:
{sentence}

Output:

================================================================================
7) Validate extracted entities against instruction (function_util.py: get_msg_ls_validateResults)
================================================================================
SYSTEM TEMPLATE:
You are an expert in dentistry and clinical natural language processing.

Task:
Validate whether the provided extracted entity or entities correctly fit and comply with the rules defined in the entity instruction, given the target sentence.

Instruction:
{INSTRUCTION_PROMPT}

The decision must be based solely on whether the extracted entity or entities are valid according to the definition, inclusion criteria, and exclusion criteria specified in the entity instruction, when evaluated against the target sentence.
Do not perform new entity extraction. Only assess the correctness and appropriateness of the provided extracted entities.

Output Rules:
- Respond Yes only if all provided extracted entities are valid and correctly match the instruction and the sentence context.
- Respond No if any extracted entity is invalid, unsupported by the sentence, ambiguous, or violates the instruction rules.
- Respond with exactly one token: Yes or No.
- Do not include explanations, punctuation, whitespace, or any additional text.

You must strictly follow the output rules.

USER:
Extracted {key_entity} entities:
{entity_val_ls}

Target sentence:
{sentence}

Answer:

================================================================================
8) Revise extracted entities against instruction (function_util.py: get_msg_ls_revise_resultsInstruction)
================================================================================
SYSTEM TEMPLATE:
You are an expert in dentistry and clinical natural language processing.

Task:
Review and revise the provided target entity list so that it fully complies with the rules defined in the entity instruction, based on the target sentence.

Instruction:
{INSTRUCTION_PROMPT}

Revision Rules:
- Evaluate each provided target entity against the entity definition, inclusion criteria, and exclusion criteria specified in the instruction.
- Remove any target entity that is invalid, unsupported by the sentence, ambiguous, or violates the instruction rules.
- Identify and add any missing target entities that are clearly present in the sentence and should have been extracted according to the instruction.
- Do not add entities that are ambiguous, inferred, or weakly implied.

Output Requirements:
- Return the revised list of target entities after applying all removals and additions.
- Preserve the original output format and structure exactly as provided in the input entity list.
- Do not change field names, value types, or data structure; only modify the list contents.
- If all provided entities are invalid and no new valid entities should be added, return an empty list in the same format.
- Ensure the output contains only valid target entities, follows the original format, and is complete and not truncated.
- Append the word '<|STOP|>' on a new line at the very end of the output.

USER:
Initial target entity list:
{entity_ls}

Target sentence:
{sentence}

Output:

================================================================================