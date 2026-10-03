"""Stages 4/8/10: translate FLORES+ eng_Latn -> tel_Telu with vLLM and score with sacrebleu.

The same script evaluates every system, so only --adapter / --model and --name change:
    python scripts/evaluate.py --name baseline
    python scripts/evaluate.py --name lora  --adapter <hf-repo-or-path>     (Stage 8)

Writes results/<name>/predictions.jsonl and results/<name>/metrics.json.
HF_TOKEN must be set in the environment (FLORES+ is gated).
"""

import argparse
import json
import time
from pathlib import Path

import sacrebleu
import vllm
from huggingface_hub import hf_hub_download, snapshot_download
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from prompt import PROMPT_TEMPLATE, build_messages

FLORES_REPO = "openlanguagedata/flores_plus"
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
SEED = 42


def load_flores(split: str, lang: str) -> dict[int, str]:
    path = hf_hub_download(FLORES_REPO, f"{split}/{lang}.jsonl", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        return {row["id"]: row["text"] for row in map(json.loads, f)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="results/<name>/ output folder")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--adapter", default=None, help="LoRA adapter: HF repo id or local path")
    parser.add_argument("--split", default="devtest")
    parser.add_argument("--limit", type=int, default=None, help="first N sentences (quick test)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-lora-rank", type=int, default=64)
    args = parser.parse_args()

    src, ref = load_flores(args.split, "eng_Latn"), load_flores(args.split, "tel_Telu")
    ids = sorted(src)[: args.limit]
    assert all(i in ref for i in ids), "FLORES+ eng/tel ids do not align"

    llm = LLM(
        model=args.model,
        dtype="float16",                 # T4: no bf16
        attention_backend="TRITON_ATTN",  # T4 (sm_75): FlashInfer/FlashAttention unsupported
        max_model_len=2048,
        seed=SEED,
        enable_lora=args.adapter is not None,
        max_lora_rank=args.max_lora_rank,
    )
    lora = LoRARequest("adapter", 1, snapshot_download(args.adapter)) if args.adapter else None
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)  # greedy

    start = time.perf_counter()
    outputs = llm.chat([build_messages(src[i]) for i in ids], sampling, lora_request=lora)
    gen_seconds = time.perf_counter() - start

    hyps = [o.outputs[0].text.strip() for o in outputs]
    refs = [ref[i] for i in ids]
    truncated = sum(o.outputs[0].finish_reason == "length" for o in outputs)

    chrf = sacrebleu.metrics.CHRF(word_order=2)            # chrF++
    bleu = sacrebleu.metrics.BLEU(tokenize="flores200")    # spBLEU, standard for FLORES
    chrf_score, bleu_score = chrf.corpus_score(hyps, [refs]), bleu.corpus_score(hyps, [refs])

    out_dir = Path("results") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for i, hyp in zip(ids, hyps):
            f.write(json.dumps({"id": i, "en": src[i], "ref": ref[i], "hyp": hyp}, ensure_ascii=False) + "\n")

    metrics = {
        "name": args.name,
        "model": args.model,
        "adapter": args.adapter,
        "split": args.split,
        "n_sentences": len(ids),
        "chrf++": round(chrf_score.score, 2),
        "spbleu": round(bleu_score.score, 2),
        "chrf_signature": str(chrf.get_signature()),
        "bleu_signature": str(bleu.get_signature()),
        "truncated_outputs": truncated,
        "generation_seconds": round(gen_seconds, 1),
        "prompt_template": PROMPT_TEMPLATE,
        "decoding": {"temperature": 0.0, "max_new_tokens": args.max_new_tokens},
        "versions": {"vllm": vllm.__version__, "sacrebleu": sacrebleu.__version__},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
