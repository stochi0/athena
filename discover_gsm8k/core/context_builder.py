from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset
from verifiers.types import Info


TASK_FILE = "task.json"

PROMPT = (
    "Write rubric_fn(prompt: str, completion: str) -> float (0–1).\n\n"
    f"REPL: {TASK_FILE} has `hint` and `train` [{{prompt, completion, score}}].\n"
    "1) Test in the REPL: get_rubric_run_result_tool(fn_code_string, train).\n"
    "2) **Required** — Submit in the REPL so your run is scored: set answer and mark ready in the same session:\n"
    "   answer['content'] = fn_code_string  # or the string of your rubric code\n"
    "   answer['ready'] = True\n"
    "If you never execute the line answer['ready'] = True in the REPL, the run gets reward 0 (no rubric is recorded)."
)

SYSTEM_PROMPT = (
    "Rubric from examples. REPL: task.json → hint, train. "
    "Test with get_rubric_run_result_tool(fn_code_string, train). "
    "You must finish by executing in the REPL: answer['content'] = <rubric_code>; answer['ready'] = True — otherwise no reward is computed. "
    "Stdlib only; return 0.0 on error."
)

def task_from_row(row: Info, cfg) -> dict:
    train = [
        {
            "prompt": str(ex["prompt"]),
            "completion": str(ex["completion"]),
            "score": float(ex.get("score", 0.0)),
        }
        for ex in (row.get("train_examples") or [])
        if isinstance(ex, dict) and "prompt" in ex and "completion" in ex
    ]
    if cfg.max_train_per_task is not None:
        train = train[: cfg.max_train_per_task]
    return {
        "hint": row.get("task_hint", "Infer the scoring rule from examples."),
        "train": train,
    }


def build_dataset(rows: list[Info], cfg, stage_dir: Path) -> Dataset:
    """One row → one context dir → one task.json (flat {hint, train})."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i, row in enumerate(rows):
        if cfg.max_examples is not None and i >= cfg.max_examples:
            break
        ctx = stage_dir / str(i)
        ctx.mkdir(exist_ok=True)
        (ctx / TASK_FILE).write_text(
            json.dumps(task_from_row(row, cfg)),
            encoding="utf-8",
        )
        test = list(row.get("test_examples") or [])
        if cfg.max_test_per_task is not None:
            test = test[: cfg.max_test_per_task]
        records.append(
            {
                "prompt": [{"role": "user", "content": PROMPT}],
                "answer": json.dumps({"test_examples": test}, separators=(",", ":")),
                "task": "discover_gsm8k",
                "example_id": i,
                "info": {"context_dir": str(ctx.resolve())},
            }
        )
    return Dataset.from_list(records)

