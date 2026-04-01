from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _safe_segment(s: str, max_len: int = 48) -> str:
    t = re.sub(r"[^a-zA-Z0-9_.-]+", "_", s.strip())[:max_len]
    return t or "unknown"


def materialize_trajectory_dir(
    workspace_root: Path,
    messages: list[dict[str, Any]],
) -> None:
    """Write ``trajectory/NNNN_role.txt`` plus ``manifest.json`` under ``workspace_root``."""
    traj = workspace_root / "trajectory"
    traj.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    for i, m in enumerate(messages):
        role = str(m.get("role", "unknown"))
        fname = f"{i:04d}_{_safe_segment(role, 32)}.txt"
        content = m.get("content", "")
        if isinstance(content, list):
            content = json.dumps(content, ensure_ascii=False)
        else:
            content = str(content)
        (traj / fname).write_text(content, encoding="utf-8")
        files.append(fname)
    manifest = {"files": files, "format": "one UTF-8 text file per message, in order"}
    (traj / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    readme = (
        "Conversation turns live in this directory.\n"
        "Read manifest.json for the ordered list of files.\n"
        "Each file is one message body; the filename prefix is the turn index and role.\n"
    )
    (traj / "README.txt").write_text(readme, encoding="utf-8")


def read_trajectory_from_context_dir(context_dir: str | Path) -> str:
    """Concatenate trajectory files in manifest order (for judges / tools)."""
    root = Path(context_dir)
    traj = root / "trajectory"
    man_path = traj / "manifest.json"
    if not man_path.is_file():
        return ""
    try:
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    files = manifest.get("files")
    if not isinstance(files, list):
        return ""
    blocks: list[str] = []
    for name in files:
        if not isinstance(name, str) or not name:
            continue
        path = traj / name
        if not path.is_file():
            continue
        stem = path.stem
        role = stem.split("_", 1)[-1] if "_" in stem else "unknown"
        try:
            body = path.read_text(encoding="utf-8")
        except OSError:
            continue
        blocks.append(f"[{role}]\n{body}")
    return "\n\n".join(blocks)
