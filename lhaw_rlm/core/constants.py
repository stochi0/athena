HF_DATASET_NAME = "ScaleAI/lhaw"

SOURCE_DATASETS = frozenset({"MCP-Atlas", "TheAgentCompany", "SWE-Bench Pro"})
AMBIGUITY_CLASSES = frozenset({"outcome-critical", "divergent", "benign"})
INFORMATION_DIMENSIONS = frozenset({"goal", "constraint", "input", "context"})

ENV_TIPS = """
<env_tips>
Use the Python REPL to reason about the task and call `ask_user(...)` when you
need missing information. Once you have enough information, write a fully
specified version of the task into `answer["content"]` and then set
`answer["ready"] = True`.
</env_tips>"""

TASK_PROMPT_PREFIX = (
    "Below is an underspecified task.\n"
    "Your job is to recover a fully specified, executable version of the task.\n"
    "If information is missing, use the `ask_user(question, context='')` tool.\n"
    "Once you have enough information, write the fully specified task into `answer[\"content\"]`.\n"
    "Do not execute the task itself; produce the clarified task specification only.\n\n"
)
