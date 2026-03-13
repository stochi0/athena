"""Isolated subprocess entry-point for evaluating a candidate `rubric_fn`.

Run as:  python -m core.rubric_execution.subprocess_runner
Stdin:   JSON {"code": <str>, "examples": [{"prompt", "completion", "score"}, ...], "margin": <float?>}
Stdout:  JSON {"rubric_agreement_rate", "scores", "correct"[, "error"]}

Standalone: no imports from core so the same source can run in a sandbox.
Naming aligned with verifiers: prompt = input text, completion = model output text.
"""

from __future__ import annotations

import json
import sys

DEFAULT_MARGIN = 0.3
MARGIN_EPSILON = 1e-9


def run(data: dict) -> dict:
    """Execute rubric_fn on examples; return result dict or error."""
    code = str(data.get("code", ""))
    examples = list(data.get("examples", []))
    try:
        margin = max(0.0, float(data.get("margin", DEFAULT_MARGIN)))
    except (TypeError, ValueError):
        margin = DEFAULT_MARGIN

    try:
        compiled = compile(code, "<rubric_fn>", "exec")
    except SyntaxError as e:
        return {"error": str(e), "scores": [], "correct": [], "rubric_agreement_rate": 0.0}

    namespace: dict = {}
    try:
        exec(compiled, namespace)  # noqa: S102
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "scores": [], "correct": [], "rubric_agreement_rate": 0.0}

    fn = namespace.get("rubric_fn")
    if not callable(fn):
        return {"error": "No rubric_fn defined", "scores": [], "correct": [], "rubric_agreement_rate": 0.0}

    scores: list[float] = []
    correct: list[bool] = []
    for ex in examples:
        prompt_str = str(ex.get("prompt", ""))
        completion_str = str(ex.get("completion", ""))
        try:
            gt = float(ex.get("score", 0.0))
        except (TypeError, ValueError):
            gt = 0.0
        try:
            s = float(fn(prompt_str, completion_str))  # type: ignore[misc]
            s = max(0.0, min(1.0, s))  # clamp to [0, 1]
            scores.append(s)
            correct.append(abs(s - gt) <= margin + MARGIN_EPSILON)
        except Exception:  # noqa: BLE001
            scores.append(0.0)
            correct.append(False)

    rate = (sum(correct) / len(correct)) if correct else 0.0
    return {"rubric_agreement_rate": rate, "scores": scores, "correct": correct}


def main() -> None:
    payload = json.loads(sys.stdin.read() or "{}")
    if not isinstance(payload, dict):
        payload = {}
    result = run(payload)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
