"""Sandbox backend: run rubric_fn in sandbox via subprocess_runner."""

from __future__ import annotations

import base64
import json
from importlib import resources

from verifiers.envs.sandbox_env import CommandTimeoutError

from core.rubric_execution.sandbox_client import get_sandbox_client
from core.types import EvalResult, EvalSample


def _load_runner_source() -> str:
    return (
        resources.files("core.rubric_execution")
        .joinpath("subprocess_runner.py")
        .read_text(encoding="utf-8")
    )


def _decode(stdout: str, invalid: str, empty: str) -> EvalResult:
    s = (stdout or "").strip()
    if not s:
        return {"reward": 0.0, "scores": [], "correct": [], "error": empty}
    try:
        raw = json.loads(s)
    except json.JSONDecodeError:
        return {"reward": 0.0, "scores": [], "correct": [], "error": invalid}
    if not isinstance(raw, dict):
        return {"reward": 0.0, "scores": [], "correct": [], "error": invalid}
    err = raw.get("error")
    return {
        "reward": float(raw.get("rubric_agreement_rate") or 0.0),
        "scores": list(raw.get("scores") or []),
        "correct": list(raw.get("correct") or []),
        "error": err if isinstance(err, str) else None,
    }


class SandboxBackend:
    async def run(
        self,
        rubric_fn_code: str,
        examples: list[EvalSample],
        *,
        timeout: int,
        margin: float,
    ) -> EvalResult:
        client = get_sandbox_client()
        sandbox_id = await client.get_or_create_sandbox()
        runner_b64 = base64.b64encode(_load_runner_source().encode()).decode()
        payload = json.dumps(
            {"code": rubric_fn_code, "examples": examples, "margin": margin}
        )
        payload_b64 = base64.b64encode(payload.encode()).decode()
        cmd = (
            'bash -lc \'\nset -euo pipefail\npython - <<"PY"\nimport base64\nfrom pathlib import Path\n\n'
            f'Path("/tmp/rubric_runner.py").write_bytes(base64.b64decode("{runner_b64}"))\n'
            f'Path("/tmp/payload.json").write_bytes(base64.b64decode("{payload_b64}"))\n'
            "PY\npython /tmp/rubric_runner.py < /tmp/payload.json\n'\n"
        )
        try:
            result = await client.execute_command(sandbox_id, cmd, timeout=timeout)
        except CommandTimeoutError:
            return {"reward": 0.0, "scores": [], "correct": [], "error": "Timeout"}
        stdout = (getattr(result, "stdout", "") or "").strip()
        stderr = (getattr(result, "stderr", "") or "").strip()
        return _decode(
            stdout,
            "Invalid sandbox runner output",
            stderr or "No output from sandbox runner",
        )
