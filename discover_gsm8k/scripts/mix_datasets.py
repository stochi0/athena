#!/usr/bin/env python3
"""
Create a mixed rubric-discovery dataset by sampling rows from multiple
existing JSONL datasets and writing a shuffled union.

Example:
    uv run scripts/mix_datasets.py \\
        --gsm8k data/data_gsm8k.jsonl \\
        --ifeval data/data_ifeval.jsonl \\
        --openmed data/data_openmed_pubmedqa.jsonl \\
        --out data/data_mixed.jsonl \\
        --max-gsm8k 50 --max-ifeval 50 --max-openmed 50
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _sample_rows(rows: list[dict[str, Any]], k: int | None, rng: random.Random) -> list[dict[str, Any]]:
    if k is None or k >= len(rows):
        return list(rows)
    return rng.sample(rows, k)


def mix_datasets(
    gsm8k_path: Path,
    ifeval_path: Path,
    openmed_path: Path,
    out_path: Path,
    *,
    max_gsm8k: int | None = None,
    max_ifeval: int | None = None,
    max_openmed: int | None = None,
    seed: int = 42,
) -> None:
    rng = random.Random(seed)

    gsm8k_rows = _load_jsonl(gsm8k_path)
    ifeval_rows = _load_jsonl(ifeval_path)
    openmed_rows = _load_jsonl(openmed_path)

    mixed: list[dict[str, Any]] = []
    mixed.extend(_sample_rows(gsm8k_rows, max_gsm8k, rng))
    mixed.extend(_sample_rows(ifeval_rows, max_ifeval, rng))
    mixed.extend(_sample_rows(openmed_rows, max_openmed, rng))

    rng.shuffle(mixed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in mixed:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(mixed)} mixed rows → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mix existing rubric-discovery JSONL datasets into one file.")
    parser.add_argument("--gsm8k", type=Path, required=True, help="Path to GSM8K JSONL (e.g. data/data_gsm8k.jsonl)")
    parser.add_argument("--ifeval", type=Path, required=True, help="Path to IFEval JSONL (e.g. data/data_ifeval.jsonl)")
    parser.add_argument(
        "--openmed",
        type=Path,
        required=True,
        help="Path to OpenMed PubMedQA JSONL (e.g. data/data_openmed_pubmedqa.jsonl)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/data_mixed.jsonl"),
        help="Output JSONL path (default: data/data_mixed.jsonl)",
    )
    parser.add_argument("--max-gsm8k", type=int, default=None, help="Max GSM8K rows to sample (default: all)")
    parser.add_argument("--max-ifeval", type=int, default=None, help="Max IFEval rows to sample (default: all)")
    parser.add_argument("--max-openmed", type=int, default=None, help="Max OpenMed rows to sample (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for sampling/shuffling")
    args = parser.parse_args()

    mix_datasets(
        gsm8k_path=args.gsm8k,
        ifeval_path=args.ifeval,
        openmed_path=args.openmed,
        out_path=args.out,
        max_gsm8k=args.max_gsm8k,
        max_ifeval=args.max_ifeval,
        max_openmed=args.max_openmed,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

