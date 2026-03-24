from __future__ import annotations

from typing import cast

import verifiers as vf


PRIVATE_METADATA_KEY = "private_metadata"
PRIVATE_INFO_FALLBACK_KEYS = frozenset(
    {
        "original_task",
        "removed_segments",
        "expected_questions",
        "original_prompt",
        "underspecified_prompt",
        "terminal_states",
        "native_result",
        "native_result_path",
        "native_trials",
        "native_baseline_trials",
        "native_summary",
    }
)


def normalize_private_metadata(value: object) -> vf.Info:
    return cast(vf.Info, value) if isinstance(value, dict) else {}


def get_private_metadata(state: vf.State) -> vf.Info:
    private_metadata = normalize_private_metadata(state.get(PRIVATE_METADATA_KEY, {}))
    if private_metadata:
        return private_metadata

    info = state.get("info", {})
    if not isinstance(info, dict):
        return {}

    return cast(
        vf.Info,
        {key: info[key] for key in PRIVATE_INFO_FALLBACK_KEYS if key in info},
    )
