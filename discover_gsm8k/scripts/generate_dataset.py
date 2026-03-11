#!/usr/bin/env python3
"""Generate rubric-discovery dataset (data.json / data.jsonl) from a source env.

  uv run python scripts/generate_dataset.py --out data.json
  uv run python scripts/generate_dataset.py --out data.jsonl --n 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI
import verifiers as vf

load_dotenv()

QUESTION_KEYS = ("prompt", "question", "input", "problem", "instruction")


def _question(row: dict[str, Any]) -> str | None:
    p = row.get("prompt")
    if isinstance(p, str) and p.strip():
        return p.strip()
    if isinstance(p, list):
        for m in p:
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str) and c.strip():
                    return c.strip()
    for k in QUESTION_KEYS:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


async def _score_one(
    env: vf.Environment,
    prompt: list[dict],
    completion: str,
    answer: Any,
    info: dict,
    task: str,
) -> float:
    state: vf.State = {
        "prompt": prompt,
        "completion": [{"role": "assistant", "content": completion}],
        "answer": answer,
        "info": info,
        "task": task,
        "trajectory": [],
        "timing": {"total_ms": 0},
    }
    await env.rubric.score_group([state])
    r = state.get("reward")
    return float(r) if r is not None else 0.0


async def _completion(client: Any, question: str, model: str, temperature: float, max_tokens: int = 512) -> str:
    r = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return str(r.choices[0].message.content or "")


def get_provider_config(args) -> dict:
    """Return config dict for api_key_var and api_base_url based on the args."""
    if args.openrouter:
        return dict(
            api_key_var="OPENROUTER_API_KEY",
            api_base_url="https://openrouter.ai/api/v1",
        )
    if args.openai:
        return dict(
            api_key_var="OPENAI_API_KEY",
            api_base_url="https://api.openai.com/v1",
        )
    # Prime (default)
    return dict(
        api_key_var="PRIME_API_KEY",
        api_base_url="https://api.pinference.ai/api/v1",
    )


async def run(
    source_env: str,
    out_path: Path,
    *,
    n: int = 50,
    responses_per_example: int = 4,
    train_ratio: float = 0.6,
    task_hint: str = "Score whether the final numeric answer is correct.",
    model: str = "openai/gpt-4.1-mini",
    temperatures: list[float] | None = None,
    seed: int = 42,
    api_key_var: str = "PRIME_API_KEY",
    api_base_url: str = "https://api.pinference.ai/api/v1",
) -> list[dict[str, Any]]:
    vf.ensure_keys([api_key_var])
    api_key = os.getenv(api_key_var, "")
    if not api_key:
        raise RuntimeError(f"Missing API key in environment variable {api_key_var}")

    env = vf.load_environment(source_env)
    try:
        ds = env.get_dataset(n=n, seed=seed)
    except ValueError:
        ds = env.get_eval_dataset(n=n, seed=seed)
    rows = ds.to_list() if hasattr(ds, "to_list") else list(ds)

    temps = temperatures or [0.0, 0.5, 1.0, 1.5][: max(1, responses_per_example)]
    rng = random.Random(seed)
    client = AsyncOpenAI(api_key=api_key, base_url=api_base_url)
    out: list[dict[str, Any]] = []

    for raw in rows:
        row = dict(raw)
        question = _question(row)
        if not question:
            continue
        answer = row.get("answer", "")
        info = row.get("info") if isinstance(row.get("info"), dict) else {}
        task = str(row.get("task", "math"))
        prompt = row.get("prompt")
        prompt_list = prompt if isinstance(prompt, list) and prompt else [{"role": "user", "content": question}]

        scored = []
        for t in temps:
            text = await _completion(client, question, model, t)
            if not text.strip():
                continue
            score = await _score_one(env, prompt_list, text, answer, info, task)
            scored.append({"input": question[:2000], "response": text[:4000], "score": round(float(score), 4)})

        if len(scored) < 2:
            continue
        rng.shuffle(scored)
        split = max(1, int(len(scored) * train_ratio))
        train, test = scored[:split], scored[split:]
        if not test:
            continue
        out.append({"task_hint": task_hint.strip(), "train_examples": train, "test_examples": test})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".json":
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        with out_path.open("w", encoding="utf-8") as f:
            for row in out:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(out)} rows → {out_path}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate rubric-discovery data.json / data.jsonl")
    parser.add_argument("--out", default="data.json", help="Output path (.json = array, .jsonl = lines)")
    parser.add_argument("--source-env", default="primeintellect/gsm8k", help="Source env (get_dataset + rubric)")
    parser.add_argument("--n", type=int, default=50, help="Number of source examples")
    parser.add_argument("--responses-per-example", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--task-hint", default="Score whether the final numeric answer is correct.")
    parser.add_argument("--model", default="openai/gpt-4.1-mini")
    parser.add_argument("--temperatures", default="0,0.5,1.0,1.5", help="Comma-separated")
    parser.add_argument("--seed", type=int, default=42)
    # Provider flags
    parser.add_argument("--prime", action="store_true", help="Use Prime Inference (default)")
    parser.add_argument("--openrouter", action="store_true", help="Use OpenRouter")
    parser.add_argument("--openai", action="store_true", help="Use OpenAI")

    args = parser.parse_args()

    provider_cfg = get_provider_config(args)
    api_key_var = provider_cfg["api_key_var"]
    api_base_url = provider_cfg["api_base_url"]

    # Handle temperature parsing centrally
    temps = [float(s.strip()) for s in args.temperatures.split(",") if s.strip()]
    if not temps:
        temps = [0.0, 0.5, 1.0, 1.5]

    asyncio.run(
        run(
            args.source_env,
            Path(args.out),
            n=args.n,
            responses_per_example=args.responses_per_example,
            train_ratio=args.train_ratio,
            task_hint=args.task_hint,
            model=args.model,
            temperatures=temps[:args.responses_per_example],
            seed=args.seed,
            api_key_var=api_key_var,
            api_base_url=api_base_url,
        )
    )


if __name__ == "__main__":
    main()
