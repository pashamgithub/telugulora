# Telugu LoRA vs QLoRA Fine-Tuning

Controlled comparison of **LoRA (16-bit base)** vs **QLoRA (4-bit NF4 base)** for
English → Telugu translation, using `Qwen/Qwen3-4B-Instruct-2507`.

> 🚧 Work in progress.

## Overview
_TODO_

## Architecture
_TODO: diagram_

## Setup
_TODO_

## Data
_TODO_

## Training
_TODO_

## Evaluation
_TODO_

## Results
| System | chrF++ | BLEU | Peak VRAM | Train time | Trainable params |
|---|---|---|---|---|---|
| Base (zero-shot) | – | – | – | – | 0 |
| LoRA (fp16 base) | – | – | – | – | – |
| QLoRA (NF4 base) | – | – | – | – | – |

## Limitations
_TODO_

## Roadmap
- [ ] Pipeline on Qwen3-4B (LoRA + QLoRA)
- [ ] Merge & publish to Hugging Face Hub
- [ ] Gemma 4 E4B experiment
- [ ] AWS deployment (vLLM)
