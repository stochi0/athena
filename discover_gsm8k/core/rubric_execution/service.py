"""Execute rubric_fn in sandbox; no parsing — code used as-is."""

from __future__ import annotations

from core.rubric_execution.backends import SandboxBackend
from core.types import EvalSample
from verifiers.types import Info


class RubricExecutionService:
    def __init__(self, *, margin: float) -> None:
        self._margin = float(margin)
        self._backend = SandboxBackend()

    async def get_rubric_run_result(
        self,
        fn_code: str,
        examples: list[EvalSample],
        *,
        timeout: int,
        margin: float | None = None,
    ) -> Info:
        m = self._margin if margin is None else float(margin)
        result = await self._backend.run(
            fn_code,
            examples,
            timeout=timeout,
            margin=m,
        )
        return {
            "reward": float(result.get("reward", 0.0) or 0.0),
            "scores": list(result.get("scores", [])),
            "correct": list(result.get("correct", [])),
            "num_examples": len(examples),
            "error": result.get("error"),
        }
