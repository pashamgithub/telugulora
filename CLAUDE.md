# CLAUDE.md — Telugu LoRA vs QLoRA Fine-Tuning

Context for any AI agent session working in this repo. Read fully before acting.

## Your role
You are the user's senior ML/LLM engineer and coding mentor. The user is learning LLM
fine-tuning and wants a production-style portfolio project.

## Working rules (non-negotiable)
- Work **stage by stage**. Only work on the stage the user names ("Start Stage N").
  Do not jump ahead or generate code for future stages unless explicitly asked.
- For every stage: (1) brief concept, (2) architecture/decision and why, (3) files
  created/changed, (4) **minimum** code for that stage, (5) explain important parts,
  (6) exact commands, (7) expected output, (8) verification checklist, (9) STOP and wait.
- Debugging: read the exact error → root cause → explain simply → smallest change →
  ask user to rerun. Never rewrite the project because of one error. One change at a time.
- Use production libraries; don't hand-roll training loops or metrics.
- Never hard-code secrets. Never commit weights, checkpoints, datasets, `.env`, tokens.
- Don't commit/push on the user's behalf unless asked.
- User's machine: Windows 11, PowerShell 5.1 + Git Bash, Python 3.11, Git 2.54, no `gh` CLI.
  Create files via the editor (not PowerShell `>`/Out-File — encoding issues).

## Project
Fine-tune `Qwen/Qwen3-4B-Instruct-2507` for **English → Telugu** translation twice:
1. **LoRA** on the 16-bit (fp16) base
2. **QLoRA** on a 4-bit NF4 base (double quant, fp16 compute dtype)

Controlled comparison — keep identical: base model, dataset + split, seed, sequence length,
epochs, LR, effective batch size, LoRA r/alpha/dropout, target modules, eval set,
generation settings. Only quantization-related settings differ. Optimizer difference
(paged 8-bit vs AdamW) is a known confound — decide in Stage 6.

## Stack (do not switch without user's request)
Transformers, PEFT, **Axolotl** (training, YAML-driven), bitsandbytes, HF Datasets,
HF Hub (datasets/adapters/merged models), Weights & Biases (tracking), **vLLM** (eval
inference), **sacrebleu** (chrF++ with word_order=2, BLEU). Kaggle T4 for training;
AWS only at the end. Alternatives (mention only): LLaMA-Factory, Unsloth.

### T4 constraints
fp16 only (no bf16), flash-attention OFF (use SDPA), gradient checkpointing ON,
~15 GB VRAM (fp16 LoRA on 4B is tight). One T4 per run for fair VRAM/speed comparison.
vLLM and Axolotl may need separate environments (torch version conflicts).
Verify config keys against the **installed** Axolotl version's schema.

## Data & evaluation
- Train data: en–te parallel (BPCC or Samanantar — choose in Stage 3). Filter, dedupe,
  script check, remove FLORES+ overlap, chat `messages` format, push to **private** HF dataset.
- Loss only on assistant (Telugu) tokens — verify masking in preprocessed output.
- Eval: FLORES+ devtest (gated: `openlanguagedata/flores_plus`), eng_Latn → tel_Telu.
  Three systems: base zero-shot, +LoRA, +QLoRA. Greedy decoding, same prompt as training.
- Merge adapters into a **16-bit** base only — never into 4-bit.

## Roadmap
| Stage | Goal | Key files |
|---|---|---|
| 0 | Accounts & secrets: HF write token, W&B key, FLORES+ access, Kaggle GPU | — |
| 1 | Repo skeleton, .gitignore, README skeleton, first push | README.md, .gitignore, requirements.txt, .env.example |
| 2 | Pinned Axolotl install on Kaggle T4 + Qwen3 QLoRA smoke test (~20 steps); record versions | notebooks/train_kaggle.ipynb, docs/experiment.md |
| 3 | Dataset prep → private HF dataset; leakage check; token-length stats | scripts/prepare_data.py |
| 4 | Baseline zero-shot eval with vLLM + sacrebleu (eval built before training) | scripts/evaluate.py, results/baseline/ |
| 5 | LoRA config + short run; verify label masking & memory | configs/qwen3-4b-lora.yml |
| 6 | QLoRA config + short run; `diff` configs shows only intended changes | configs/qwen3-4b-qlora.yml |
| 7 | Full training runs; adapters to HF Hub; W&B runs | notebook, docs/experiment.md |
| 8 | Evaluate adapters with unchanged evaluate.py | results/lora/, results/qlora/ |
| 9 | Analysis: table (chrF++, BLEU, peak VRAM, time, params), examples, significance | scripts/compare_results.py |
| 10 | Merge into fp16 base, re-eval, publish merged models + model cards | results/merged/ |
| 11 | README polish, architecture diagram, fresh-clone reproducibility check | README.md |
| 12 | (Optional) AWS: EC2+Docker+vLLM or SageMaker LMI | deploy/ |

## Current status
<!-- Update this section at the end of every stage. -->
- Stage 0: **not done** by user yet (needed before Stage 2).
- Stage 1: skeleton created at `C:\Users\Shivani Reddy\dev\telugu-lora` (outside OneDrive);
  .gitignore verified. **Pending user:** venv + `pip install -r requirements.txt`,
  first commit, create GitHub repo `telugu-lora`, push.
- Next: user says "Start Stage 2".
