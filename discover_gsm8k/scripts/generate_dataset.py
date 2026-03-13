#!/usr/bin/env python3
"""Generate rubric-discovery dataset (data/data.jsonl) from one or more source envs.

Single env (CLI):
  uv run scripts/generate_dataset.py --out data/data.jsonl --n 50
  uv run scripts/generate_dataset.py --source-env arcee-ai/ifeval --out data3.jsonl --train-per-task 3 --test-per-task 2

Multiple envs (YAML):
  uv run scripts/generate_dataset.py --config config/envs.yaml
  uv run scripts/generate_dataset.py --config config/envs.yaml --out data/mixed.jsonl

YAML schema: out, model, seed, responses_per_example, train_ratio, temperatures, provider; envs: list of
  { source_env, n?, train_per_task?, test_per_task?, task_hint?, seed?, responses_per_example?, train_ratio?, temperatures? }
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from pathlib import Path
from typing import Any

import yaml
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


def get_provider_config(args: argparse.Namespace) -> dict:
    """Return config dict for api_key_var and api_base_url based on the args."""
    if getattr(args, "openrouter", False):
        return dict(
            api_key_var="OPENROUTER_API_KEY",
            api_base_url="https://openrouter.ai/api/v1",
        )
    if getattr(args, "openai", False):
        return dict(
            api_key_var="OPENAI_API_KEY",
            api_base_url="https://api.openai.com/v1",
        )
    return dict(
        api_key_var="PRIME_API_KEY",
        api_base_url="https://api.pinference.ai/api/v1",
    )


def _parse_temperatures(s: str) -> list[float]:
    if not isinstance(s, str) or not s.strip():
        return [0.0, 0.5, 1.0, 1.5]
    return [float(x.strip()) for x in s.split(",") if x.strip()]


_DEFAULT_TEMPS = [0.0, 0.5, 1.0, 1.5]


def _ensure_temperatures(temperatures: list[float], responses_per_example: int) -> list[float]:
    """Ensure we have at least responses_per_example temperatures (pad from default if needed)."""
    if len(temperatures) >= responses_per_example:
        return temperatures[:responses_per_example]
    out = list(temperatures)
    while len(out) < responses_per_example:
        out.append(_DEFAULT_TEMPS[len(out) % len(_DEFAULT_TEMPS)])
    return out


def _validate_run_params(
    *,
    responses_per_example: int,
    train_ratio: float,
    train_per_task: int | None,
    test_per_task: int | None,
) -> None:
    """Validate relationship between responses_per_example, train_ratio, train_per_task, test_per_task."""
    if responses_per_example < 2:
        raise ValueError(
            "responses_per_example must be >= 2 (need at least 2 scored responses for train+test)"
        )
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be in (0, 1)")
    if train_per_task is not None and train_per_task < 1:
        raise ValueError("train_per_task must be >= 1 when set")
    if test_per_task is not None and test_per_task < 1:
        raise ValueError("test_per_task must be >= 1 when set")
    # After split we have at most floor(n*train_ratio) train and rest test; caps must be achievable
    max_train = max(1, int(responses_per_example * train_ratio))
    max_test = max(1, responses_per_example - max_train)
    if train_per_task is not None and train_per_task > max_train:
        raise ValueError(
            f"train_per_task ({train_per_task}) cannot exceed train split size ({max_train}) "
            f"for responses_per_example={responses_per_example}, train_ratio={train_ratio}"
        )
    if test_per_task is not None and test_per_task > max_test:
        raise ValueError(
            f"test_per_task ({test_per_task}) cannot exceed test split size ({max_test}) "
            f"for responses_per_example={responses_per_example}, train_ratio={train_ratio}"
        )


async def run(
    source_env: str,
    out_path: Path,
    *,
    n: int = 50,
    responses_per_example: int = 4,
    train_ratio: float = 0.6,
    train_per_task: int | None = None,
    test_per_task: int | None = None,
    task_hint: str | None = None,
    model: str = "openai/gpt-4.1-mini",
    temperatures: list[float] | None = None,
    seed: int = 42,
    api_key_var: str = "PRIME_API_KEY",
    api_base_url: str = "https://api.pinference.ai/api/v1",
    write_output: bool = True,
) -> list[dict[str, Any]]:
    _validate_run_params(
        responses_per_example=responses_per_example,
        train_ratio=train_ratio,
        train_per_task=train_per_task,
        test_per_task=test_per_task,
    )
    temperatures = _ensure_temperatures(
        temperatures or _DEFAULT_TEMPS, responses_per_example
    )

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

    temps = temperatures[:responses_per_example]
    rng = random.Random(seed)
    client = AsyncOpenAI(api_key=api_key, base_url=api_base_url)
    out: list[dict[str, Any]] = []

    resolved_task_hint = (task_hint or source_env).strip()

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
        if train_per_task is not None:
            train = train[:train_per_task]
        if test_per_task is not None:
            test = test[:test_per_task]
        if not train or not test:
            continue
        out.append({"task_hint": resolved_task_hint, "train_examples": train, "test_examples": test})

    if write_output:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == ".json":
            out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            with out_path.open("w", encoding="utf-8") as f:
                for row in out:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {len(out)} rows → {out_path}")
    return out


async def _run_from_yaml(config_path: Path, out_override: Path | None) -> None:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    envs = cfg.get("envs") or []
    if not envs:
        raise SystemExit("Config must have at least one entry under 'envs'")

    final_out = out_override or Path(cfg.get("out") or "data/mixed.jsonl")
    model = cfg.get("model") or "openai/gpt-4.1-mini"
    seed = int(cfg.get("seed", 42))
    responses_per_example = int(cfg.get("responses_per_example", 4))
    train_ratio = float(cfg.get("train_ratio", 0.6))
    temps_str = cfg.get("temperatures", "0,0.5,1.0,1.5")
    provider = (cfg.get("provider") or "prime").strip().lower()
    args = argparse.Namespace(
        openrouter=(provider == "openrouter"),
        openai=(provider == "openai"),
        prime=(provider != "openrouter" and provider != "openai"),
    )
    provider_cfg = get_provider_config(args)
    api_key_var = provider_cfg["api_key_var"]
    api_base_url = provider_cfg["api_base_url"]

    all_rows: list[dict[str, Any]] = []
    for i, env_cfg in enumerate(envs):
        if not isinstance(env_cfg, dict):
            continue
        source_env = env_cfg.get("source_env")
        if not source_env:
            print(f"Skip env entry {i}: missing source_env")
            continue
        n = int(env_cfg.get("n", 50))
        train_per_task = env_cfg.get("train_per_task")
        train_per_task = int(train_per_task) if train_per_task is not None else None
        test_per_task = env_cfg.get("test_per_task")
        test_per_task = int(test_per_task) if test_per_task is not None else None
        task_hint = env_cfg.get("task_hint")
        env_seed = int(env_cfg.get("seed", seed))
        env_responses = int(env_cfg.get("responses_per_example", responses_per_example))
        env_train_ratio = float(env_cfg.get("train_ratio", train_ratio))
        env_temps = _ensure_temperatures(
            _parse_temperatures(env_cfg.get("temperatures", temps_str)),
            env_responses,
        )
        _validate_run_params(
            responses_per_example=env_responses,
            train_ratio=env_train_ratio,
            train_per_task=train_per_task,
            test_per_task=test_per_task,
        )

        print(f"Generating from {source_env} (n={n}, train_per_task={train_per_task}, test_per_task={test_per_task}) …")
        rows = await run(
            source_env,
            final_out,
            n=n,
            responses_per_example=env_responses,
            train_ratio=env_train_ratio,
            train_per_task=train_per_task,
            test_per_task=test_per_task,
            task_hint=task_hint,
            model=model,
            temperatures=env_temps,
            seed=env_seed,
            api_key_var=api_key_var,
            api_base_url=api_base_url,
            write_output=False,
        )
        all_rows.extend(rows)
        print(f"  → {len(rows)} rows (total so far: {len(all_rows)})")

    final_out.parent.mkdir(parents=True, exist_ok=True)
    if final_out.suffix == ".json":
        final_out.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        with final_out.open("w", encoding="utf-8") as f:
            for row in all_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_rows)} rows → {final_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate rubric-discovery data.json / data.jsonl")
    parser.add_argument("--config", "-c", help="YAML config: multiple source envs (envs list + defaults); ignores other CLI env flags")
    parser.add_argument("--out", "-o", default="data/data.jsonl", help="Output path (.json = array, .jsonl = lines)")
    parser.add_argument("--source-env", default="primeintellect/gsm8k", help="Source env (get_dataset + rubric)")
    parser.add_argument("--n", type=int, default=50, help="Number of source examples")
    parser.add_argument("--responses-per-example", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument(
        "--train-per-task",
        type=int,
        default=2,
        metavar="N",
        help="Cap train examples per row; must be <= floor(responses_per_example*train_ratio); default 2",
    )
    parser.add_argument(
        "--test-per-task",
        type=int,
        default=2,
        metavar="N",
        help="Cap test examples per row; must be <= remainder after train split; default 2",
    )
    parser.add_argument("--task-hint", default="Score whether the final numeric answer is correct.")
    parser.add_argument("--model", default="openai/gpt-4.1-mini")
    parser.add_argument("--temperatures", default="0,0.5,1.0,1.5", help="Comma-separated")
    parser.add_argument("--seed", type=int, default=42)
    # Provider flags
    parser.add_argument("--prime", action="store_true", help="Use Prime Inference (default)")
    parser.add_argument("--openrouter", action="store_true", help="Use OpenRouter")
    parser.add_argument("--openai", action="store_true", help="Use OpenAI")

    args = parser.parse_args()

    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise SystemExit(f"Config not found: {config_path}")
        out_override = Path(args.out) if args.out != "data/data.jsonl" else None
        asyncio.run(_run_from_yaml(config_path, out_override))
        return

    provider_cfg = get_provider_config(args)
    api_key_var = provider_cfg["api_key_var"]
    api_base_url = provider_cfg["api_base_url"]

    temps = _parse_temperatures(args.temperatures)
    if not temps:
        temps = [0.0, 0.5, 1.0, 1.5]

    asyncio.run(
        run(
            args.source_env,
            Path(args.out),
            n=args.n,
            responses_per_example=args.responses_per_example,
            train_ratio=args.train_ratio,
            train_per_task=args.train_per_task,
            test_per_task=args.test_per_task,
            task_hint=args.task_hint,
            model=args.model,
            temperatures=temps[: args.responses_per_example],
            seed=args.seed,
            api_key_var=api_key_var,
            api_base_url=api_base_url,
        )
    )


if __name__ == "__main__":
    main()
