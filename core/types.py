from __future__ import annotations

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


class AskUserInteraction(TypedDict):
    question: str
    context: str
    response: str
