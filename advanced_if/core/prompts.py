from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a rubric induction assistant. Read the full trajectory and infer a concise, "
    "testable checklist of criteria for evaluating the assistant's response quality."
)

USER_TEMPLATE = """Infer evaluation rubrics from this full conversation trajectory.

Return JSON only with:
{{"rubrics": ["criterion 1", "criterion 2", ...]}}

Guidelines:
- Each criterion must be atomic and testable.
- Avoid duplicates and vague wording.
- Preserve task-specific constraints and formatting requirements.

Trajectory:
{trajectory}
"""

JUDGE_PROMPT = """Compare a candidate rubric list to the gold rubric list for the same conversation.

Conversation trajectory:
{question}

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
