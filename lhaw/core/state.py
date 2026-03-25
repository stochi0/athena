from __future__ import annotations

from typing import cast

import verifiers as vf

PRIVATE_METADATA_KEY = "private_metadata"


def normalize_private_metadata(value: object) -> vf.Info:
    return cast(vf.Info, value) if isinstance(value, dict) else {}


def get_private_metadata(state: vf.State) -> vf.Info:
    return normalize_private_metadata(state.get(PRIVATE_METADATA_KEY, {}))
