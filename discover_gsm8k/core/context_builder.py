from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset
from verifiers.types import Info


TASK_FILE = "task.json"

PROMPT = (
    f"Write rubric_fn(input_text, response) -> float (0–1). REPL: {TASK_FILE} contains hint and train. "
    "All tool usage, testing, and submission MUST be performed by executing Python statements inside this REPL; describing actions is insufficient. "
    "Test by running: result = get_rubric_run_result_tool(fn_code_string, train). "
    "Submit by running: answer['content'] = fn_code_string; answer['ready'] = True. "
    "Any deviation or failure should lead to returning 0.0."
)

SYSTEM_PROMPT = (
    "Single Python REPL: your code, tools, and rubric_fn run here; task.json = hint + train. "
    "get_rubric_run_result_tool is preloaded — call it directly and do NOT import, redefine, or shadow it. "
    "Tool calls and submissions take effect only when you execute the corresponding Python code in the REPL "
    "(e.g. answer['content'] = fn_code_string; answer['ready'] = True). "
    "Use only the Python standard library; no external network, non-stdlib modules, or side-effecting I/O. "
    "On any error or exception, return 0.0."
)

def task_from_row(row: Info, cfg) -> dict:
    train = [
        {
            "input": str(ex["input"]),
            "response": str(ex["response"]),
            "score": float(ex.get("score", 0.0)),
        }
        for ex in (row.get("train_examples") or [])
        if isinstance(ex, dict)
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

