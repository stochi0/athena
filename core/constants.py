HF_DATASET_NAME = "ScaleAI/lhaw"

SOURCE_DATASETS = frozenset({"MCP-Atlas", "TheAgentCompany", "SWE-Bench Pro"})
AMBIGUITY_CLASSES = frozenset({"outcome-critical", "divergent", "benign"})
INFORMATION_DIMENSIONS = frozenset({"goal", "constraint", "input", "context"})
REWARD_MODES = frozenset({"reconstruction_judge", "native_reward"})

RECONSTRUCTION_ENV_TIPS = """
<env_tips>
You are operating in an RLM environment with a persistent Python REPL.
Work iteratively:
1. Inspect the underspecified task and identify what information is missing
2. Use `ask_user(question, context="")` only when the missing information is necessary
3. Keep a working draft in `answer["content"]` and refine it as you learn more
4. Do not execute the underlying task; your job is to produce the clarified task specification only
5. Only set `answer["ready"] = True` once `answer["content"]` is a complete, executable task specification
</env_tips>"""

NATIVE_REWARD_ENV_TIPS = """
<env_tips>
You are operating in an RLM environment with a persistent Python REPL.
Work iteratively:
1. Inspect the underspecified task and identify blockers before acting
2. Use `ask_user(question, context="")` only for genuinely missing information tied to the task
3. Solve the task itself rather than rewriting it
4. Prefer autonomous progress when clarification is unnecessary
5. Only set `answer["ready"] = True` once `answer["content"]` contains the completed task output or execution summary
</env_tips>"""

RECONSTRUCTION_TASK_PROMPT_PREFIX = (
    "You are operating in an RLM environment with a persistent Python REPL.\n"
    "Below is an underspecified task.\n"
    "Your job is to reconstruct a fully specified, executable version of that task.\n"
    "If information is genuinely missing, call `ask_user(question, context='')`.\n"
    'Use the REPL iteratively and maintain your current best draft in `answer["content"]`.\n'
    "Do not execute or solve the task itself; produce the clarified task specification only.\n"
    'When your draft is complete, set `answer["ready"] = True`.\n\n'
)

NATIVE_REWARD_TASK_PROMPT_PREFIX = (
    "You are operating in an RLM environment with a persistent Python REPL.\n"
    "Below is an underspecified task.\n"
    "Your job is to solve the task itself while strategically deciding when clarification is necessary.\n"
    "If critical information is missing, call `ask_user(question, context='')`.\n"
    'Use the REPL iteratively and maintain your current best solution or execution summary in `answer["content"]`.\n'
    "Avoid unnecessary user interruptions for information you can infer or proceed without.\n"
    'When you have completed the task as far as possible, set `answer["ready"] = True`.\n\n'
)


def get_env_tips(reward_mode: str) -> str:
    if reward_mode == "native_reward":
        return NATIVE_REWARD_ENV_TIPS
    return RECONSTRUCTION_ENV_TIPS


def get_task_prompt_prefix(reward_mode: str) -> str:
    if reward_mode == "native_reward":
        return NATIVE_REWARD_TASK_PROMPT_PREFIX
    return RECONSTRUCTION_TASK_PROMPT_PREFIX
