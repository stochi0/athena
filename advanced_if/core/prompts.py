from __future__ import annotations

# Task text only. RLM mechanics (``answer``, ``call_python_repl``, ``llm_batch``) come from
# verifiers' built-in RLM scaffold — do not replace ``system_prompt`` on ``RLMEnv`` unless you
# re-embed that contract (see ``lhaw/core/constants.py``, ``long_context_retrieval/core/config.py``).

RLM_TASK_PROMPT = """You are operating in an RLM environment with a persistent Python REPL and a sandbox working directory (``call_python_repl``). The framework injects the global ``answer`` dict and ``llm_batch()``; follow that scaffolding: work in small steps, inspect output, then continue.

## Task

Infer a **testable rubric** for judging the **assistant** in the multi-turn conversation stored on disk. Output is a JSON object with a ``rubrics`` key: a list of short, atomic criterion strings (what a grader would check).

## Workspace (evidence only from here)

- ``./trajectory/manifest.json`` — ordered list of filenames (read this first).
- ``./trajectory/*.txt`` — one message body per file; the filename prefix is turn index and role (e.g. ``0003_user.txt``, ``0004_assistant.txt``).
- There is **no** gold rubric in the filesystem; do not assume labels you have not derived from the messages.

## Workflow (iterative)

1. **Orient** — ``import os; print(os.getcwd()); print(os.listdir("."))`` then list ``trajectory/``.
2. **Read the manifest** — load JSON and decide which files to open (you do not need every file if you sample strategically; re-open if unsure).
3. **Optional** — use ``llm_batch([...])`` in parallel to summarize or compare chunks of the conversation (strings only; no message dicts).
4. **Draft** — keep a working JSON string in ``answer["content"]`` and update it as you read more.
5. **Partial feedback** — call ``submit_candidate_rubrics(draft_string)`` where ``draft_string`` includes JSON shaped like ``{{"rubrics": ["...", ...]}}``. The host judge sees full gold internally; **your** feedback channel is: **{feedback_channel}**
6. **Finalize** — after you have **seen** REPL output for your last edits, set ``answer["content"]`` to the final JSON string (same ``rubrics`` schema), then set ``answer["ready"] = True`` in the **same** REPL session.

## Rubric quality

- Each item is one checkable requirement; no duplicates; preserve task-specific constraints and formatting the assistant was held to.

## Starter snippet (manifest)

```python
import json
from pathlib import Path
m = json.loads(Path("trajectory/manifest.json").read_text())
print(m.get("files", [])[:5], "...")
```
"""


PER_CRITERION_JUDGE_PROMPT = """You grade a candidate rubric list against each GOLD criterion for the same conversation.

For each gold rubric in order, answer: does the candidate list (as a whole) adequately cover that requirement?
Be strict: partial or vague coverage counts as not satisfied.

Trajectory:
{trajectory}

Gold rubrics (in order, indices 0..n-1):
{numbered_gold}

Candidate rubrics (model output, may be JSON or prose):
{candidate}

Return one JSON object (no markdown) with key "satisfied": an array of exactly {n} booleans,
one per gold rubric in order (true = adequately covered by the candidate list).
"""


JUDGE_PROMPT = """Compare a candidate rubric list to the gold rubric list for the same conversation.

Conversation trajectory:
{trajectory}

Gold rubrics:
{answer}

Candidate rubrics:
{response}

Return a single JSON object with exactly these three boolean keys (no markdown, no prose):
- "coverage": true if the candidate covers the essential constraints from the gold list.
- "faithful": true if the candidate does not add major incorrect or hallucinated constraints.
- "non_redundant": true if the candidate list is not overly repetitive vs gold.

Example shape (values are yours to judge): {{"coverage": true, "faithful": true, "non_redundant": false}}
"""
