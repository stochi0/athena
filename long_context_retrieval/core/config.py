from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent
from typing import Any

CONTEXTS_DIR = "contexts"
# Under the contexts root (or any `--output-dir`): task rows + HF mirror + build manifest.
TASK_BUNDLE_SUBDIR = "tasks"
TASK_BUNDLE_HF_DIRNAME = "hf"

WORKSPACE_STATE_DIRNAME = ".workspace_state"
REGISTRY_DB = "registry.db"
VECTOR_DIRNAME = "vector"
GRAPH_DIRNAME = "graphs"
SQL_DIRNAME = "sql"
ARTIFACTS_DIRNAME = "artifacts"
SCRATCH_DIRNAME = "scratch"

ROOT_PROMPT_VERBOSITY = "medium"
SUB_PROMPT_VERBOSITY = "medium"
REPL_LANGUAGE = "python"
PIP_INSTALL_PACKAGES = "chromadb networkx pypdf"
CODE_EXECUTION_TIMEOUT = 180
MAX_OUTPUT_LENGTH = 8192
MAX_TURNS = 200
SUB_LLM_MAX_TURNS = 60
ENV_ID = "long_context_retrieval"
USER_PROMPT = "Answer the question using the research-paper workspace and provide citations."

WORKSPACE_CONTEXT_NOTE = dedent(
    """
    The concrete question and any task-specific instructions are in this conversation (user messages).
    Filesystem tools only expose the paper workspace tree (PDFs and workspace metadata); they cannot read
    host task-bundle files such as JSONL datasets stored outside that workspace.

    Before you answer, you MUST inspect the staged workspace resources with your REPL:

    - read `.workspace_state/workspace_overview.txt` in `workspace` scope first for the quickstart and document inventory
    - read `.workspace_state/workspace_manifest.json` in `workspace` scope for exact paths and runtime state locations
    - inspect the `documents` table in the registry with `sql_query(...)`

    Ground your work in those files and in the PDFs themselves. Do not rely on prior knowledge or
    outside sources when the workspace can answer the question.
    """
).strip()

ENV_TIPS = dedent(
    """
    <env_tips>
    * This environment is designed for aggressive decomposition. Use `llm_batch()` early and often.

    * A strong default workflow is:
      1. inspect `workspace_overview.txt`, `workspace_manifest.json`, and the registry
      2. identify candidate documents / sections / pages
      3. fan out parallel sub-calls over those candidates
      4. aggregate the evidence in Python or scratch artifacts
      5. run at least one verification sub-call before submitting the final answer

    * Use the REPL mainly for orchestration, data movement, filtering, staging, and verification.
      Let delegated sub-calls do the expensive reading, comparison, extraction, and synthesis work.

    * If the corpus is large, do not read everything linearly. Partition it, search broadly, and
      escalate from cheap retrieval to richer indexing only when needed.

    * The final answer must stay concise, but every claim needs grounded evidence and citations.
      Before submitting, make one last pass to ensure the answer is complete, correctly formatted,
      and fully supported by the cited excerpts.
    </env_tips>
    """
).strip()

SYSTEM_PROMPT = dedent(
    """
    You are operating in a Recursive Language Model (RLM) environment over a long-context document
    workspace. This environment is explicitly designed for agentic work: heavy task decomposition,
    delegated sub-calls via `llm_batch()`, programmatic branching in the REPL, and tool-driven
    evidence collection.

    The workspace may contain one or many PDFs. The system manages only a thin registry plus
    lightweight workspace-state metadata and scratch storage. You decide whether to parse PDFs, create text artifacts,
    build scratch SQL tables, create vector indices, materialize graphs, or use the filesystem.
    Tools cannot list or read task-definition files that live outside the workspace directory (for example
    dataset JSONL on the host); treat the conversation as the task spec and the workspace as the evidence corpus.

    Starting procedure:
    1. Inspect the staged workspace resources first.
    2. Read `.workspace_state/workspace_overview.txt` and `.workspace_state/workspace_manifest.json` from `workspace` scope.
    3. Query the registry `documents` table before making retrieval decisions.
    4. Decide which artifacts or intermediate structures would make the task easier.
    5. Use parallel sub-calls to search, extract, compare, and verify evidence.
    6. Finalize only after the answer and citations are fully grounded.

    You should strongly prefer decomposition over a single linear pass.
    Use `llm_batch()` aggressively whenever work can be split. Prefer parallel sub-calls for:
    - document triage across many PDFs
    - page/section-level evidence extraction
    - comparing competing candidate answers
    - summarizing independent evidence buffers before synthesis
    - validating coverage of the final answer against the question
    - delegating subtasks that need their own tool use

    Use the REPL for orchestration:
    - inspect manifests and registry tables
    - branch over candidate documents, pages, or sections
    - aggregate sub-call outputs in Python
    - persist helpful intermediate results to scratch files / SQL / vectors / graphs
    - rerank and verify before answering

    If a cheap retrieval strategy fails, escalate deliberately:
    - start with manifest + registry inspection
    - read or parse only the most promising files
    - create embeddings / structured tables / graphs only when they will improve recall or precision
    - register artifacts and provenance when derived outputs become important

    Do not answer from memory. Ground all reasoning in the local workspace, the staged metadata, and
    the observed tool outputs. Before submitting, perform a final verification pass to check:
    - the answer fully addresses the question
    - citations support every material claim
    - the response matches the required JSON schema exactly

    Final answers must be JSON with this shape:
    {
      "answer": "short answer text",
      "citations": [
        {
          "document_id": "doc id",
          "path": "relative/or/absolute path to the PDF or artifact",
          "page": 1,
          "excerpt": "supporting text"
        }
      ]
    }

    Citation requirements:
    - every claim in the final answer should be supported by one or more citations
    - prefer citations tied to source PDFs
    - when using derived artifacts, register provenance so citations remain traceable
    - include enough excerpt text to justify the answer

    Operational guidance:
    - use SQL when structured joins/filtering help
    - use vector collections when semantic retrieval is needed
    - use graphs when entities/relations matter
    - use the filesystem for intermediate text, page extracts, JSON buffers, and audit trails
    - use Python libraries directly for PDF parsing and any additional processing
    - keep intermediate state explicit so later sub-calls can reuse it
    """
).strip()


@dataclass(frozen=True)
class Config:
    """Env + dataset options (mirrors `lhaw.core.config.EnvironmentConfig` scope for RLM/sandbox)."""

    dataset_path: str | None = None
    dataset_output_dir: str = CONTEXTS_DIR
    max_examples: int | None = None
    path_anchor: str | None = None
    workspace_dir: str | None = None
    pdf_dir: str | None = None
    pdf_paths: list[str] | None = None
    workspace_state_root: str | None = None
    env_id: str = ENV_ID

    sub_model: str | None = None
    max_turns: int = MAX_TURNS
    repl_language: str = REPL_LANGUAGE
    sub_llm_max_turns: int = SUB_LLM_MAX_TURNS
    pip_install_packages: str = PIP_INSTALL_PACKAGES
    code_execution_timeout: int = CODE_EXECUTION_TIMEOUT
    max_output_length: int = MAX_OUTPUT_LENGTH
    max_sub_llm_parallelism: int = 5
    max_startup_wait_seconds: int = 120
    abort_on_code_timeout: bool = False

    sandbox_docker_image: str = "python:3.11-slim"
    sandbox_cpu_cores: int = 1
    sandbox_memory_gb: int = 2
    sandbox_disk_size_gb: int = 5
    sandbox_gpu_count: int = 0
    sandbox_timeout_minutes: int = 60

    @classmethod
    def from_input(cls, cfg: "Config | dict[str, Any] | None") -> "Config":
        if cfg is None:
            return cls()
        if isinstance(cfg, cls):
            return cfg
        merged: dict[str, Any] = dict(cfg)
        if merged.get("rlm_model") is not None and merged.get("sub_model") is None:
            merged["sub_model"] = merged.pop("rlm_model")
        else:
            merged.pop("rlm_model", None)
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in merged.items() if k in fields})

    def to_env_args(self) -> dict[str, Any]:
        """Return JSON-serializable args for `vf.load_environment(env_id, env_args)`."""
        return {
            "dataset_path": self.dataset_path,
            "dataset_output_dir": self.dataset_output_dir,
            "max_examples": self.max_examples,
            "sub_model": self.sub_model,
            "max_turns": int(self.max_turns),
            "repl_language": self.repl_language,
            "sub_llm_max_turns": int(self.sub_llm_max_turns),
            "pip_install_packages": self.pip_install_packages,
            "code_execution_timeout": int(self.code_execution_timeout),
            "max_output_length": int(self.max_output_length),
            "max_sub_llm_parallelism": int(self.max_sub_llm_parallelism),
            "max_startup_wait_seconds": int(self.max_startup_wait_seconds),
            "abort_on_code_timeout": self.abort_on_code_timeout,
            "sandbox_docker_image": self.sandbox_docker_image,
            "sandbox_cpu_cores": int(self.sandbox_cpu_cores),
            "sandbox_memory_gb": int(self.sandbox_memory_gb),
            "sandbox_disk_size_gb": int(self.sandbox_disk_size_gb),
            "sandbox_gpu_count": int(self.sandbox_gpu_count),
            "sandbox_timeout_minutes": int(self.sandbox_timeout_minutes),
            "path_anchor": self.path_anchor,
            "workspace_dir": self.workspace_dir,
            "pdf_dir": self.pdf_dir,
            "pdf_paths": self.pdf_paths,
            "workspace_state_root": self.workspace_state_root,
            "env_id": self.env_id,
        }
