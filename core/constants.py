HF_DATASET_NAME = "ScaleAI/lhaw"

SOURCE_DATASETS = frozenset({"MCP-Atlas", "TheAgentCompany", "SWE-Bench Pro"})
AMBIGUITY_CLASSES = frozenset({"outcome-critical", "divergent", "benign"})
INFORMATION_DIMENSIONS = frozenset({"goal", "constraint", "input", "context"})

ENV_TIPS = """
<env_tips>
You are operating in an RLM environment with a persistent Python REPL.
Work iteratively:
1. Inspect the underspecified task and identify what information is missing
2. Use `ask_user(question, context="")` only when the missing information is necessary
3. Keep a working draft in `answer["content"]` and refine it as you learn more
4. Do not execute the underlying task; your job is to produce the clarified task specification only
5. Only set `answer["ready"] = True` once `answer["content"]` is a complete, executable task specification
</env_tips>"""

TASK_PROMPT_PREFIX = (
    "You are operating in an RLM environment with a persistent Python REPL.\n"
    "Below is an underspecified task.\n"
    "Your job is to reconstruct a fully specified, executable version of that task.\n"
    "If information is genuinely missing, call `ask_user(question, context='')`.\n"
    "Use the REPL iteratively and maintain your current best draft in `answer[\"content\"]`.\n"
    "Do not execute or solve the task itself; produce the clarified task specification only.\n"
    "When your draft is complete, set `answer[\"ready\"] = True`.\n\n"
)
