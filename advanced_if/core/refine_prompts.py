from __future__ import annotations

REFINE_SYSTEM_PROMPT = (
    "You are a rubric induction assistant. Read the conversation trajectory and infer a concise, "
    "testable checklist for evaluating the assistant's response.\n\n"
    "You have a tool `submit_candidate_rubrics`: pass your current rubric list as a JSON string "
    'with shape {"rubrics": ["...", ...]}. The judge sees the full gold rubric but only returns '
    "limited feedback (configured per run).\n\n"
    "When you are ready to finish, respond with a normal assistant message (no tool calls) "
    'containing only JSON: {"rubrics": ["criterion 1", ...]} — same schema as the task.'
)

REFINE_USER_TEMPLATE = """Infer evaluation rubrics from this trajectory. Use the tool to refine; then submit your final JSON (no tools).

Trajectory:
{trajectory}
"""

# Internal: judge returns full structure; only a subset is shown to the model via the tool.
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
