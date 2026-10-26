# Persona Vectors: source data

These JSON files are byte-for-byte copies from
https://github.com/safety-research/persona_vectors at commit
`b8e0f044fe2410a6fad579f38324f03f13b4e917`.

| Local file | Upstream path |
| --- | --- |
| `sycophantic_extract.json` | `data_generation/trait_data_extract/sycophantic.json` |
| `sycophantic_eval.json` | `data_generation/trait_data_eval/sycophantic.json` |

Each contains 20 questions, five positive/negative instruction pairs, and a trait
judge prompt. We preserve all fields and all text, including punctuation.
The extraction and evaluation questions differ. Evaluation jobs use only the
evaluation questions; their instruction pairs are not applied.

Copyright 2025 The Persona Vectors authors. The upstream Apache-2.0 license is
included as `LICENSE`. No upstream JSON content has been modified.

SHA-256 checksums, source revision, repeat counts and the fixed evaluation split
are recorded in `configs/persona_sycophantic.yaml` at the repository root.
The `.gitattributes` rule preserves JSON bytes on Windows as well as Linux.

Our dev/test split is a project decision inherited from the pilot, not an
upstream split. Indices start at zero in the evaluation JSON. Do not use the test
questions to tune layers, coefficients or other experimental choices.

Prompt construction follows `load_persona_questions` in upstream
`eval/eval_persona.py`: a trait assistant prefix for positive instructions,
a helpful assistant prefix for negative instructions, and no explicit system
message for evaluation. The model's chat template may still insert its own
default system message.
