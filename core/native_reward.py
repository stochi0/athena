from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import verifiers as vf
from verifiers.utils.data_utils import extract_boxed_answer

from .transcript import extract_ask_user_interactions, normalize_info


@dataclass(frozen=True)
class NativeTrialResult:
    success: bool
    score: float
    total: float
    questions_asked: int = 0
    used_ask_user: bool = False


def pass_at_k(n: int, c: int, k: int) -> float:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if n <= 0:
        return 0.0
    if n < k:
        p = c / n
        return 1.0 - (1.0 - p) ** k
    if n - c < k:
        return 1.0
    if c == 0:
        return 0.0
    numerator = 1.0
    for i in range(k):
        numerator *= (n - c - i) / (n - i)
    return 1.0 - numerator


def _coerce_float(value: object) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _coerce_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def _read_json_path(path_value: object) -> object:
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_trial_result(raw: object) -> NativeTrialResult | None:
    if not isinstance(raw, dict):
        return None

    score = _coerce_float(raw.get("score"))
    total = _coerce_float(raw.get("total"))
    reward = _coerce_float(raw.get("reward"))
    success = _coerce_bool(raw.get("success"))

    if score is None and total is None and reward is not None:
        score = reward
        total = 1.0

    if score is None and success is not None:
        score = 1.0 if success else 0.0
        total = 1.0

    if score is None or total is None:
        return None

    if total <= 0:
        total = 1.0

    if success is None:
        success = score >= total

    questions_asked = int(_coerce_float(raw.get("questions_asked")) or 0.0)
    used_ask_user = _coerce_bool(raw.get("used_ask_user"))
    if used_ask_user is None:
        used_ask_user = questions_asked > 0

    return NativeTrialResult(
        success=success,
        score=score,
        total=total,
        questions_asked=questions_asked,
        used_ask_user=used_ask_user,
    )


def _extract_trial_results(raw: object) -> list[NativeTrialResult]:
    if not isinstance(raw, list):
        return []
    trials: list[NativeTrialResult] = []
    for item in raw:
        normalized = _normalize_trial_result(item)
        if normalized is not None:
            trials.append(normalized)
    return trials


def _load_result_payload(info: vf.Info, state: vf.State) -> object:
    for key in ("native_result",):
        payload = state.get(key)
        if payload is not None:
            return payload
        payload = info.get(key)
        if payload is not None:
            return payload

    for key in ("native_result_path",):
        payload = state.get(key)
        if payload:
            return _read_json_path(payload)
        payload = info.get(key)
        if payload:
            return _read_json_path(payload)
    return None


def _normalized_reward_from_payload(payload: object) -> float | None:
    trial = _normalize_trial_result(payload)
    if trial is None:
        return None
    return max(0.0, min(1.0, trial.score / trial.total))


def _metric_from_summary(summary: object, key: str) -> float | None:
    if not isinstance(summary, dict):
        return None
    value = _coerce_float(summary.get(key))
    if value is None:
        return None
    if key.endswith("_percent"):
        return value / 100.0
    return value


def _summary_from_trials(
    trials: list[NativeTrialResult], baseline_trials: list[NativeTrialResult]
) -> dict[str, float]:
    if not trials:
        return {}

    n = len(trials)
    c = sum(1 for trial in trials if trial.success)
    pass_at_3 = pass_at_k(n, c, min(3, n))
    checkpoint_percent = sum((trial.score / trial.total) * 100.0 for trial in trials) / n
    ask_count = sum(1 for trial in trials if trial.used_ask_user)
    ask_percent = 100.0 * ask_count / n
    total_questions = sum(trial.questions_asked for trial in trials)
    avg_questions = total_questions / ask_count if ask_count > 0 else 0.0

    baseline_pass_at_3 = None
    if baseline_trials:
        baseline_n = len(baseline_trials)
        baseline_c = sum(1 for trial in baseline_trials if trial.success)
        baseline_pass_at_3 = pass_at_k(baseline_n, baseline_c, min(3, baseline_n))

    gain_per_question = 0.0
    if baseline_pass_at_3 is not None and total_questions > 0:
        gain_per_question = (pass_at_3 - baseline_pass_at_3) / total_questions

    return {
        "pass_at_3": pass_at_3,
        "checkpoint_percent": checkpoint_percent,
        "ask_percent": ask_percent,
        "avg_questions_per_trajectory": avg_questions,
        "gain_per_question": gain_per_question,
    }


class NativeRewardRubric(vf.Rubric):
    """Reward using benchmark-native results rather than an LLM reconstruction judge."""

    def __init__(self) -> None:
        super().__init__()
        self.add_reward_func(self.native_reward, weight=1.0)
        self.add_metric(self.native_score_metric)
        self.add_metric(self.native_success_metric)
        self.add_metric(self.questions_asked_metric)
        self.add_metric(self.asked_user_metric)
        self.add_metric(self.pass_at_3_metric)
        self.add_metric(self.checkpoint_percent_metric)
        self.add_metric(self.ask_percent_metric)
        self.add_metric(self.avg_questions_per_trajectory_metric)
        self.add_metric(self.gain_per_question_metric)

    def _cached_summary(self, state: vf.State) -> dict[str, float]:
        cached = state.get("_native_summary_cache")
        if isinstance(cached, dict):
            return {
                key: float(value)
                for key, value in cached.items()
                if isinstance(value, (int, float))
            }

        info = normalize_info(state.get("info", {}))
        summary = info.get("native_summary")
        summary_pairs = [
            ("pass_at_3", _metric_from_summary(summary, "pass_at_3")),
            ("checkpoint_percent", _metric_from_summary(summary, "checkpoint_percent")),
            ("ask_percent", _metric_from_summary(summary, "ask_percent")),
            (
                "avg_questions_per_trajectory",
                _metric_from_summary(summary, "avg_questions_per_trajectory"),
            ),
            ("gain_per_question", _metric_from_summary(summary, "gain_per_question")),
        ]
        computed = {key: value for key, value in summary_pairs if value is not None}

        if not computed:
            trials = _extract_trial_results(info.get("native_trials", []))
            baseline_trials = _extract_trial_results(info.get("native_baseline_trials", []))
            computed = _summary_from_trials(trials, baseline_trials)

        state["_native_summary_cache"] = computed
        return computed

    def _require_reward(self, state: vf.State) -> float:
        payload = _load_result_payload(normalize_info(state.get("info", {})), state)
        reward = _normalized_reward_from_payload(payload)
        if reward is None:
            raise ValueError(
                "native_reward mode requires native benchmark outputs. "
                "Provide one of info['native_result'] or info['native_result_path']."
            )
        return reward

    async def native_reward(self, state: vf.State, **_kwargs: object) -> float:
        _ = extract_boxed_answer(str(state.get("final_answer", ""))).strip()
        return self._require_reward(state)

    async def native_score_metric(self, state: vf.State, **_kwargs: object) -> float:
        return self._require_reward(state)

    async def native_success_metric(self, state: vf.State, **_kwargs: object) -> float:
        return 1.0 if self._require_reward(state) >= 1.0 else 0.0

    async def questions_asked_metric(self, state: vf.State, **_kwargs: object) -> float:
        return float(len(extract_ask_user_interactions(state.get("completion", []))))

    async def asked_user_metric(self, state: vf.State, **_kwargs: object) -> float:
        interactions = extract_ask_user_interactions(state.get("completion", []))
        return 1.0 if interactions else 0.0

    async def pass_at_3_metric(self, state: vf.State, **_kwargs: object) -> float:
        return self._cached_summary(state).get("pass_at_3", 0.0)

    async def checkpoint_percent_metric(self, state: vf.State, **_kwargs: object) -> float:
        return self._cached_summary(state).get("checkpoint_percent", 0.0) / 100.0

    async def ask_percent_metric(self, state: vf.State, **_kwargs: object) -> float:
        return self._cached_summary(state).get("ask_percent", 0.0) / 100.0

    async def avg_questions_per_trajectory_metric(
        self, state: vf.State, **_kwargs: object
    ) -> float:
        return self._cached_summary(state).get("avg_questions_per_trajectory", 0.0)

    async def gain_per_question_metric(self, state: vf.State, **_kwargs: object) -> float:
        return self._cached_summary(state).get("gain_per_question", 0.0)
