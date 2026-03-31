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

Return JSON only with keys:
- "coverage" (boolean): candidate covers the essential gold constraints.
- "faithful" (boolean): candidate does not introduce major incorrect constraints.
- "non_redundant" (boolean): candidate list is not overly repetitive.
"""
