#!/usr/bin/env python3
"""Generate rubric-discovery dataset from source envs. Config is YAML only.

  uv run scripts/generate_dataset.py --config config/envs.yaml
  uv run scripts/generate_dataset.py --config config/envs.yaml --out data/mixed.jsonl

YAML: out, model, seed, responses_per_example, train_ratio, temperatures, provider; envs: list of
  { source_env, n?, train_per_task?, test_per_task?, ... }
  Rows use verifiers RolloutInput shape: prompt (Messages), answer, task, etc.
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


def extract_question(row: dict[str, Any]) -> str | None:
    """Get question text from row. Verifiers RolloutInput uses 'prompt' (Messages). Handles string or list-of-messages."""
    val = row.get("prompt")
    if isinstance(val, str) and val.strip():
        return val.strip()
    if isinstance(val, list):
        for m in val:
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str) and c.strip():
                    return c.strip()
    return None


async def score_one(
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


async def call_completion(client: Any, question: str, model: str, temperature: float, max_tokens: int = 512) -> str:
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


def parse_temperatures(s: str) -> list[float]:
    if not isinstance(s, str) or not s.strip():
        return [0.0, 0.5, 1.0, 1.5]
    return [float(x.strip()) for x in s.split(",") if x.strip()]


DEFAULT_TEMPERATURES = [0.0, 0.5, 1.0, 1.5]


def ensure_temperatures(temperatures: list[float], responses_per_example: int) -> list[float]:
    """Ensure we have at least responses_per_example temperatures (pad from default if needed)."""
    if len(temperatures) >= responses_per_example:
        return temperatures[:responses_per_example]
    out = list(temperatures)
    while len(out) < responses_per_example:
        out.append(DEFAULT_TEMPERATURES[len(out) % len(DEFAULT_TEMPERATURES)])
    return out


def validate_run_params(
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
    validate_run_params(
        responses_per_example=responses_per_example,
        train_ratio=train_ratio,
        train_per_task=train_per_task,
        test_per_task=test_per_task,
    )
    temperatures = ensure_temperatures(
        temperatures or DEFAULT_TEMPERATURES, responses_per_example
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
        question = extract_question(row)
        if not question:
            continue
        answer = row.get("answer", "")
        info = row.get("info") if isinstance(row.get("info"), dict) else {}
        task = str(row.get("task", "math"))
        prompt = row.get("prompt")
        prompt_list = prompt if isinstance(prompt, list) and prompt else [{"role": "user", "content": question}]

        scored = []
        for t in temps:
            text = await call_completion(client, question, model, t)
            if not text.strip():
                continue
            score = await score_one(env, prompt_list, text, answer, info, task)
            scored.append({"prompt": question[:2000], "completion": text[:4000], "score": round(float(score), 4)})

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


async def run_from_yaml(config_path: Path, out_override: Path | None) -> None:
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
        env_temps = ensure_temperatures(
            parse_temperatures(env_cfg.get("temperatures", temps_str)),
            env_responses,
        )
        validate_run_params(
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
    parser = argparse.ArgumentParser(description="Generate rubric-discovery dataset from YAML config")
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config (envs + defaults)")
    parser.add_argument("--out", "-o", default=None, help="Override output path from config")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")
    out_override = Path(args.out) if args.out else None
    asyncio.run(run_from_yaml(config_path, out_override))


if __name__ == "__main__":
    main()
