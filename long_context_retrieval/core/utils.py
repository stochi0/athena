from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def ensure_json(value: str | None, default: Any, label: str) -> Any:
    if value in (None, ""):
        return default
    return json.loads(value)


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return normalized or "item"


def short_hash_for_path(path: Path) -> str:
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()
    return digest[:10]


def stable_document_id(path: Path) -> str:
    stem = slugify(path.stem)
    return f"{stem}-{short_hash_for_path(path.resolve())}"


def normalize_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        # Accept both raw strings and JSON-encoded payloads/lists.
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        return normalize_items(decoded)
    if isinstance(value, dict):
        answer = value.get("answer")
        if isinstance(answer, str):
            return [answer.strip()] if answer.strip() else []
        return [json.dumps(value, sort_keys=True)]
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for entry in value:
            out.extend(normalize_items(entry))
        return [item for item in out if item]
    return [str(value).strip()]


def parse_final_answer_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        payload = dict(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {"answer": "", "citations": []}
        decoded = json.loads(stripped)
        if isinstance(decoded, dict):
            payload = decoded
        elif isinstance(decoded, list):
            payload = {"answer": ", ".join(normalize_items(decoded)), "citations": []}
        else:
            payload = {"answer": str(decoded), "citations": []}
    else:
        payload = {"answer": str(value), "citations": []}

    answer = payload.get("answer", "")
    citations = payload.get("citations", [])
    if not isinstance(answer, str):
        answer = str(answer)
    if not isinstance(citations, list):
        citations = []
    return {"answer": answer, "citations": citations}
