"""Stage 3: build the en->te training dataset and push it to a private HF dataset.

Source: BPCC-Human Wiki + Daily (ai4bharat/BPCC, gated, CC-BY-4.0).
Leakage guard: FLORES+ dev + devtest English (openlanguagedata/flores_plus, gated).

Usage:
    python scripts/prepare_data.py           # build + report + token stats (no upload)
    python scripts/prepare_data.py --push    # same, then push private dataset to HF Hub
"""

import argparse
import csv
import json
import os
import re
import unicodedata

from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from prompt import build_messages

BPCC_REPO = "ai4bharat/BPCC"
BPCC_SUBSETS = ["wiki", "daily"]
FLORES_REPO = "openlanguagedata/flores_plus"
FLORES_SPLITS = ["dev", "devtest"]
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

SEED = 42
VAL_SIZE = 500
NGRAM = 8                  # any shared 8-word sequence with FLORES+ => drop the pair
MIN_TELUGU_FRACTION = 0.5  # share of non-space target chars that must be Telugu script
RATIO_RANGE = (0.3, 3.0)   # allowed len(te) / len(en) in characters

TELUGU_CHAR = re.compile(r"[ఀ-౿]")
WORD = re.compile(r"[a-z0-9]+")


def clean(text: str) -> str:
    """Unicode NFC + collapse whitespace. Keeps ZWNJ (U+200C), which Telugu uses."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def ngrams(text: str, n: int = NGRAM) -> set[tuple[str, ...]]:
    words = WORD.findall(text.lower())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def telugu_fraction(text: str) -> float:
    chars = re.sub(r"\s", "", text)
    return len(TELUGU_CHAR.findall(chars)) / max(1, len(chars))


def load_bpcc(token: str) -> list[dict]:
    rows = []
    for subset in BPCC_SUBSETS:
        path = hf_hub_download(BPCC_REPO, f"{subset}/tel_Telu.tsv", repo_type="dataset", token=token)
        with open(path, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                if (r["src_lang"], r["tgt_lang"]) == ("eng_Latn", "tel_Telu"):
                    rows.append({"en": clean(r["src"] or ""), "te": clean(r["tgt"] or ""), "source": subset})
    return rows


def load_flores_english(token: str) -> list[str]:
    texts = []
    for split in FLORES_SPLITS:
        path = hf_hub_download(FLORES_REPO, f"{split}/eng_Latn.jsonl", repo_type="dataset", token=token)
        with open(path, encoding="utf-8") as f:
            texts += [json.loads(line)["text"] for line in f]
    return texts


def filter_pairs(rows: list[dict], flores: list[str]) -> list[dict]:
    flores_exact = {clean(t).lower() for t in flores}
    flores_grams = set().union(*(ngrams(t) for t in flores))
    report, seen, kept = {}, set(), []

    def drop(reason: str) -> None:
        report[reason] = report.get(reason, 0) + 1

    for r in rows:
        en, te = r["en"], r["te"]
        key = en.lower()
        if not en or not te:
            drop("empty")
        elif telugu_fraction(te) < MIN_TELUGU_FRACTION:
            drop("target not mostly Telugu script")
        elif TELUGU_CHAR.search(en):
            drop("Telugu script in English source")
        elif not RATIO_RANGE[0] <= len(te) / len(en) <= RATIO_RANGE[1]:
            drop("length ratio outlier")
        elif key in flores_exact or ngrams(en) & flores_grams:
            drop(f"FLORES+ overlap (exact or shared {NGRAM}-gram)")
        elif key in seen:
            drop("duplicate English source (kept first)")
        else:
            seen.add(key)
            kept.append(r)

    print(f"\nInput pairs: {len(rows)}")
    for reason, n in sorted(report.items(), key=lambda kv: -kv[1]):
        print(f"  dropped {n:6d}  {reason}")
    print(f"Kept pairs:  {len(kept)}")
    return kept


def token_stats(ds: Dataset, tokenizer) -> None:
    texts = [tokenizer.apply_chat_template(m, tokenize=False) for m in ds["messages"]]
    lengths = sorted(len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"])
    pct = lambda p: lengths[min(len(lengths) - 1, int(p / 100 * len(lengths)))]
    print(f"\nToken length (full chat, Qwen3 tokenizer) over {len(lengths)} train examples:")
    print(f"  p50 {pct(50)} | p90 {pct(90)} | p95 {pct(95)} | p99 {pct(99)} | max {lengths[-1]}")
    for limit in (256, 384, 512):
        over = sum(n > limit for n in lengths)
        print(f"  > {limit} tokens: {over} ({100 * over / len(lengths):.2f}%)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true", help="push the private dataset to HF Hub")
    args = parser.parse_args()

    load_dotenv()
    token = os.environ["HF_TOKEN"]
    repo_id = f"{os.environ['HF_USERNAME']}/telugu-en-te-bpcc"

    rows = filter_pairs(load_bpcc(token), load_flores_english(token))
    ds = Dataset.from_list(rows).map(lambda r: {"messages": build_messages(r["en"], r["te"])})
    splits = ds.train_test_split(test_size=VAL_SIZE, seed=SEED)
    dsd = DatasetDict(train=splits["train"], validation=splits["test"])

    print(f"\nSplits: train {len(dsd['train'])} | validation {len(dsd['validation'])}")
    print("By source (train):", {s: dsd["train"]["source"].count(s) for s in BPCC_SUBSETS})
    print("\nExample:", json.dumps(dsd["train"][0]["messages"], ensure_ascii=False, indent=1))

    token_stats(dsd["train"], AutoTokenizer.from_pretrained(BASE_MODEL, token=token))

    if args.push:
        dsd.push_to_hub(repo_id, private=True, token=token)
        print(f"\nPushed (private): https://huggingface.co/datasets/{repo_id}")
    else:
        print(f"\nDry run. Re-run with --push to upload to {repo_id} (private).")


if __name__ == "__main__":
    main()
