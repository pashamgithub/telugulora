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

## Evaluation setup & baseline (Stage 4)
`scripts/evaluate.py` — vLLM 0.30.0 (fp16, `TRITON_ATTN` on T4), greedy, max 512 new
tokens, same prompt as training. FLORES+ devtest eng_Latn → tel_Telu, 1012 sentences.

| System | chrF++ | spBLEU | Truncated (hit 512) | Gen time (1 T4) |
|---|---|---|---|---|
| Base Qwen3-4B-Instruct-2507, zero-shot | **27.88** | **11.0** | 138 / 1012 (13.6%) | 363 s |

Signatures: chrF++ `nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0` ·
spBLEU `nrefs:1|case:mixed|eff:no|tok:flores200|smooth:exp|version:2.6.0`

Observations (base model):
- Gist usually right, but transliterates English words ("mice" → మీస్), mistranslates
  facts (Monday → మంగళవారం/Tuesday), invents words.
- Greedy **repetition loops** (e.g. devtest id 2 repeats "సంబంధించిన వారికి" to the limit) —
  the main reason for the 138 truncations: **124 / 1012 outputs are heavy repetition
  loops** (unique-4-gram ratio < 0.5). 0 empty outputs, 0 outputs not in Telugu script. Not "fixed" with a repetition penalty: decoding
  must stay identical across systems, and loop rate is itself a finding.
- **Reference noise in FLORES+ tel_Telu:** CSV-escaping artifacts (`"""""""…`), a duplicated
  phrase (id 1), an incomplete translation (id 3). Same refs for every system, so the
  comparison stays fair; note as a limitation in Stage 9. References are not edited.

## Controlled variables
Source of truth: `configs/qwen3-4b-lora.yml` (Stage 5). The QLoRA config (Stage 6) may
differ **only** in the quantization block.

| Variable | Value |
|---|---|
| Base model | Qwen/Qwen3-4B-Instruct-2507 |
| Data | `thirumalreddy0172/telugu-en-te-bpcc` train (36,957) / validation (500) |
| Loss | assistant turn only (`roles_to_train: [assistant]`, `train_on_eos: turn`) |
| Sequence length / packing | 512 / off |
| LoRA | r 16 · alpha 32 · dropout 0.05 · all linear projections |
| LoRA kernels | mlp/qkv/o on, embedding off (pinned explicitly) |
| Precision / attention | fp16 · SDPA · gradient checkpointing (reentrant) |
| Batch | micro 4 × grad-accum 4 = **16** effective |
| Optimizer | `adamw_torch_fused` — **same for both runs** (see below) |
| LR schedule | 2e-4 · cosine · warmup 5% · weight decay 0 · grad-clip 1.0 |
| Seed | 42 |
| Epochs | 1 (provisional; fixed in Stage 7 from measured throughput) |

**Optimizer confound — resolved by using one optimizer for both.** Paged 8-bit AdamW only
saves optimizer-state memory, and with LoRA that state is tiny (33M trainable params × 8 bytes
≈ 0.26 GB in fp32 AdamW). So there's no memory reason for QLoRA to use a different optimizer,
and using the same one removes the confound. To confirm in Stage 6.

### Stage 5 short run (LoRA fp16, 30 steps) — 2026-10-12
- Status: **pass** — fits on one T4, no NaN.
- Peak memory: `max_allocated` **11.73 GiB**, `device_reserved` **12.57 GiB** of 15 GiB
  → ~2.4 GiB headroom. No need to shrink micro-batch.
- Eval loss (500 val): **0.4816** at step 0 (base model) → **0.3661** at step 30.
- Train loss (every 10 steps): 0.430 → 0.377 → 0.396 (noisy at this scale; expected).
- Speed: 30 steps in ~278 s of pure training ≈ **9.3 s/step** (~230 trainable tok/s);
  one validation pass ≈ 80–95 s. `train_runtime` 453.5 s includes both evals.
- Padding: `tokens/total` ≈ 2× `tokens/trainable` — includes prompt tokens *and* batch
  padding (no packing). Prompt masking itself checked separately via `preprocess --debug`.
- **Full-run estimate (1 epoch):** 36,957 / 16 = 2,310 steps × 9.3 s ≈ 6.0 h, + 9 evals
  (~14 min) ≈ **6.3 h** → fits Kaggle's 12 h session limit.
- **Masking check: pass** (`axolotl preprocess --debug`, 1 example, 208 tokens, 156 trained):
  every prompt token incl. `<|im_start|>user`, the instruction, the English text,
  `<|im_end|>` and `<|im_start|>assistant\n` → label **-100**; every Telugu token and the
  closing `<|im_end|>` (151645) → own id (trained); trailing `\n` → -100.
  Telugu prints as `�` pieces in the debug log because Qwen's byte-level BPE splits Telugu
  characters into several byte tokens — display artefact, not corruption (also why Telugu is
  token-expensive).
- Axolotl drops the **34** train examples longer than 512 tokens (matches Stage 3 stats) →
  effective train size **36,923**. Validation: max 433 tokens, none dropped.

### Stage 6 short run (QLoRA 4-bit NF4, 30 steps) — 2026-10-12
`git diff --no-index` of the two configs: only the title comment, the quantization block
(`adapter`, `load_in_4bit`, `nf4_backend`, `bnb_config_kwargs`) and `output_dir` differ.
Resolved config confirms nf4 + double quant + float16 compute, `adamw_torch_fused`.

| 30-step short run (same data, seed, overrides) | LoRA fp16 | QLoRA NF4 | Δ |
|---|---|---|---|
| Peak `max_allocated` | 11.73 GiB | **6.71 GiB** | −5.0 GiB (−43%) |
| Peak `device_reserved` | 12.57 GiB | **7.57 GiB** | −5.0 GiB |
| Eval loss at step 0 (base model, before training) | 0.4816 | **0.5240** | +0.042 — cost of 4-bit base |
| Eval loss at step 30 | 0.3661 | **0.3870** | +0.021 |
| Pure training time, 30 steps | ~278 s | ~313 s | |
| Seconds / step | ~9.3 | **~10.4** | +13% slower |
| Trainable tok/s | ~230 | ~207 | −10% |
| Trainable params | 33,030,144 | 33,030,144 | identical |

(Step time = from end of step-0 eval to start of step-30 eval, from log timestamps.)

Early read (30 steps only — not a result): QLoRA saves ~43% memory and costs ~13% speed;
quantizing the base raises its loss, and the gap narrows after training starts (0.042 → 0.021).
Whether it closes is what Stages 7–8 measure.

**Full-run estimates (1 epoch, 2,308 steps + ~9 evals):** LoRA ≈ 6.3 h · QLoRA ≈ 7.0 h →
both fit Kaggle's 12 h session; ≈ 13.3 h of the ~30 h weekly GPU quota.

## Runs
_Filled in Stage 7._
