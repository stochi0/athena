"""
GSM8K rubric-discovery environment.

Each JSONL row: {train_examples, test_examples, task_hint?}
Each example:   {input, response, score}

Model writes rubric_fn(input_text, response) -> float in the REPL,
tested against train via get_rubric_run_result.
Reward = test-set agreement rate. Metric = Spearman(predicted, GT).

Usage:
    env = load_environment({"dataset_path": "data.jsonl"})
    prime eval run discover-gsm8k -m gpt-4.1-mini
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import verifiers as vf
from typing_extensions import TypedDict
from verifiers.envs.experimental.rlm_env import RLMEnv
from verifiers.types import Info, State

from core.rubric_execution import RubricExecutionService
from core.types import EvalSample
from core.context_builder import SYSTEM_PROMPT, build_dataset


# Tool schema TypedDict (includes num_examples for REPL)
class RubricRunResult(TypedDict, total=False):
    rubric_agreement_rate: float
    scores: list[float]
    correct: list[bool]
    num_examples: int
    error: str


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    dataset_path: str = "data.jsonl"
    rlm_model: str = "gpt-4.1-mini"
    max_turns: int = 100
    repl_language: str = "python"
    sub_llm_max_turns: int = 5
    sub_max_completion_tokens: int | None = None
    root_max_completion_tokens: int | None = None
    max_examples: int | None = None
    timeout_s: int = 30
    margin: float = 0.3
    parallelism: int = 5
    # Dataset / context layout
    max_train_per_task: int | None = None
    max_test_per_task: int | None = None
    context_dir_name: str = "contexts"

    @classmethod
    def from_input(cls, cfg: Config | dict | None) -> Config:
        if cfg is None:
            return cls()
        if isinstance(cfg, cls):
            return cfg
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Rubric code from state (RLMEnv only)
# ---------------------------------------------------------------------------


def rubric_code_from_state(state: State) -> str:
    """Return rubric code from state. RLMEnv sets state['final_answer'] = answer['content'] when the model sets answer['ready'] = True in the REPL. No parsing — use as-is."""
    if not isinstance(state, dict):
        return ""
    final = state.get("final_answer", "")
    if isinstance(final, str) and final.strip():
        return final.strip()
    return ""


# ---------------------------------------------------------------------------
# Rubric execution (sandbox)
# ---------------------------------------------------------------------------

_rubric_service: RubricExecutionService | None = None


def _get_rubric_service(cfg: Config) -> RubricExecutionService:
    global _rubric_service
    if _rubric_service is None:
        _rubric_service = RubricExecutionService(margin=cfg.margin)
    return _rubric_service


# ---------------------------------------------------------------------------
# Test-example extraction from answer JSON
# ---------------------------------------------------------------------------


def get_test_examples(state: State) -> list[Info]:
    if not isinstance(state, dict):
        return []
    raw = state.get("answer", "")
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return []
    return [
        {
            "input": str(ex["input"]),
            "response": str(ex["response"]),
            "score": float(ex.get("score", 0.0)),
        }
        for ex in (data.get("test_examples") or [])
        if isinstance(ex, dict) and "input" in ex and "response" in ex
    ]


# ---------------------------------------------------------------------------
# Reward + metric (shared eval, cached on state)
# ---------------------------------------------------------------------------

_CACHE = "_eval"


async def _eval(state: State, cfg: Config) -> Info | None:
    if not isinstance(state, dict):
        return None
    if _CACHE in state:
        return state[_CACHE]
    code = rubric_code_from_state(state)
    examples = get_test_examples(state)
    if not code or not examples:
        state[_CACHE] = None
        return None
    service = _get_rubric_service(cfg)
    result = await service.get_rubric_run_result(code, examples, timeout=cfg.timeout_s)
    state[_CACHE] = result
    return result


async def agreement_reward(state: State, cfg: Config) -> float:
    r = await _eval(state, cfg)
    return float((r or {}).get("reward", 0.0))


def _spearman(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n != len(b) or n < 2:
        return 0.0

    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        for i, idx in enumerate(order):
            r[idx] = float(i + 1)
        return r

    rp, rg = rank(a), rank(b)
    d2 = sum((rp[i] - rg[i]) ** 2 for i in range(n))
    denom = n * (n * n - 1)
    return 0.0 if denom == 0 else 1.0 - 6.0 * d2 / denom


async def spearman_metric(state: State, cfg: Config) -> float:
    r = await _eval(state, cfg)
    if not r:
        return 0.0
    scores = r.get("scores", [])
    examples = get_test_examples(state)
    if len(scores) != len(examples):
        return 0.0
    gt = [float(ex["score"]) for ex in examples]
    return _spearman(list(scores), gt)


# ---------------------------------------------------------------------------
# REPL tool exposed to the model (module-level so worker can import by name)
# ---------------------------------------------------------------------------

_current_cfg: Config | None = None


async def get_rubric_run_result(
    fn_code: str, examples: list[EvalSample], cfg: Config
) -> RubricRunResult:
    """Run rubric_fn against examples. Returns {rubric_agreement_rate, scores, correct, num_examples, error?}."""
    service = _get_rubric_service(cfg)
    result = await service.get_rubric_run_result(
        fn_code, examples, timeout=cfg.timeout_s
    )
    out: RubricRunResult = {
        "rubric_agreement_rate": float(result.get("reward", 0.0)),
        "scores": list(result.get("scores", [])),
        "correct": list(result.get("correct", [])),
        "num_examples": len(examples),
    }
    if result.get("error"):
        out["error"] = str(result["error"])
    return out


async def get_rubric_run_result_tool(
    fn_code: str, examples: list[EvalSample]
) -> RubricRunResult:
    """Run rubric_fn against examples. Returns rubric_agreement_rate, scores, correct, num_examples, error?."""
    cfg = _current_cfg
    if cfg is None:
        return {
            "rubric_agreement_rate": 0.0,
            "scores": [],
            "correct": [],
            "num_examples": len(examples),
            "error": "config not set (load_environment not called)",
        }
    return await get_rubric_run_result(fn_code, examples, cfg)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
#
# Context building is separated into `core/context_builder.py` (prompt + dataset staging).


# ---------------------------------------------------------------------------
# load_environment
# ---------------------------------------------------------------------------


def load_environment(config: Config | dict | None = None) -> vf.Environment:
    cfg = Config.from_input(config)
    if not cfg.dataset_path:
        raise ValueError("dataset_path required")

    path = Path(cfg.dataset_path)
    if not path.exists():
        raise FileNotFoundError(path)

    # Stream JSONL rows to keep memory usage bounded and allow early stopping.
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if cfg.max_examples is not None and len(rows) >= cfg.max_examples:
                break
    stage_dir = path.parent / cfg.context_dir_name
    dataset = build_dataset(rows, cfg, stage_dir)

    rubric = vf.Rubric(funcs=[agreement_reward], weights=[1.0])
    rubric.add_metric(spearman_metric)
    rubric.add_class_object("cfg", cfg)

    global _current_cfg
    _current_cfg = cfg
    _get_rubric_service(cfg)

    return RLMEnv(
        dataset=dataset,
        rubric=rubric,
        system_prompt=SYSTEM_PROMPT,
        max_iterations=cfg.max_turns,
        repl_language=cfg.repl_language,
        sub_model=cfg.rlm_model,
        sub_llm_max_turns=cfg.sub_llm_max_turns,
        sub_max_completion_tokens=cfg.sub_max_completion_tokens,
        root_max_completion_tokens=cfg.root_max_completion_tokens,
        root_tools=[get_rubric_run_result_tool],
        env_id="discover_gsm8k",
    )
