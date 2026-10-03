# Experiment Log

## Environment & versions
Training runs on Kaggle (GPU T4, one GPU per run via `CUDA_VISIBLE_DEVICES=0`).
Axolotl lives in an isolated uv venv (`/tmp/axo-venv`, Python 3.12) so Kaggle's
preinstalled torch is untouched. Full freeze: `versions-axolotl.txt` (Kaggle output).

| Component | Version |
|---|---|
| Kaggle GPU / driver | 2× Tesla T4 (15360 MiB each, 1 used) / driver 580.178.04 (CUDA ≤ 13.0) |
| Python (venv) | 3.12 |
| torch / CUDA build | 2.13.0+cu132 / 13.2 (runs on driver 13.0 via CUDA minor-version compatibility) |
| transformers | 5.17.0 |
| peft | 0.21.0 |
| bitsandbytes | 0.50.2 |
| accelerate | 1.15.0 |
| trl | 1.13.0 |
| datasets | 4.8.4 |
| axolotl | 0.20.0 |

Kaggle's own Python is 3.13.15 (not used for training). Checked 2026-10-03.

### Stage 2 smoke test (QLoRA, 20 steps) — 2026-10-03
- Status: **pass** (8 toy pairs, seq_len 512, micro-batch 1 × grad-accum 4)
- Loss: 0.467 → 0.0018 (memorising 8 examples — expected, not meaningful)
- `train_runtime`: 53.45 s (~2.7 s/step); peak `memory/max_allocated`: ~3.0 GiB
- Trainable params: 33,030,144 / 4,055,498,240 (0.81%), r=16, all 7 linear projections
- Loss masking sanity: step 1 had 115 trainable of 256 total tokens → prompt tokens
  are masked (full verification in Stage 5)

### Gotchas found in Stage 2 (carry forward)
- `axolotl train` shells out to `accelerate` **from PATH** → must
  `export PATH=/tmp/axo-venv/bin:$PATH`, else Kaggle's system accelerate (Py 3.13) runs.
- Kaggle has 2 T4s → accelerate auto-starts 2 processes. Force one:
  `axolotl train cfg.yml -- --num_processes 1`.
- Axolotl **auto-enables LoRA kernels** (`lora_mlp_kernel`, `lora_qkv_kernel`,
  `lora_o_kernel`, `lora_embedding_kernel`). Set these **explicitly and identically** in
  both LoRA and QLoRA configs (Stages 5–6) so they can't differ silently.
- Implicit defaults seen in resolved config: `lr_scheduler: cosine`, `weight_decay: 0.0`,
  gradient checkpointing `use_reentrant: true`, `train_on_inputs: false`. Pin explicitly later.
- Axolotl reports `bf16: true` capability on T4 (emulated) — ignore; we stay on fp16.
- Warning to act on: pre-tokenise with `axolotl preprocess cfg.yml` before real runs.

## Data (Stage 3)
Built by `scripts/prepare_data.py` → private HF dataset `thirumalreddy0172/telugu-en-te-bpcc`.

- Source: BPCC-Human **Wiki + Daily**, en→te (human-translated, CC-BY-4.0). Chosen over
  Samanantar (web-mined, noisy, CC-BY-NC) — quality matters more than size on a T4 budget.
- Prompt (shared with eval via `scripts/prompt.py`), no system message:
  `Translate the following English text to Telugu.\n\n{en}`

| Step | Pairs |
|---|---|
| Input (wiki 29,726 + daily 8,512) | 38,238 |
| − duplicate English source (kept first) | −747 |
| − target < 50% Telugu script (half-translated rows) | −25 |
| − length ratio te/en outside 0.3–3.0 | −7 |
| − FLORES+ dev/devtest overlap (exact or shared 8-gram) | **−2** |
| **Kept** | **37,457** |
| train / validation (seed 42) | 36,957 / 500 |

Leakage: 0 exact matches, but 2 near-copies of FLORES+ sentences (Sikhism; Sundarbans
tigers) — caught only by the 8-gram check.

Token length (Qwen3 tokenizer, full chat incl. template): p50 173 · p95 306 · p99 368 ·
max 1031. >512 tokens: 34 (0.09%). Telugu is token-expensive (~17-word sentence ≈ 170 tokens).

## Controlled variables
_Filled in Stages 5–6._

## Runs
_Filled in Stage 7._
