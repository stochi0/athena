from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent
from typing import Any

CONTEXTS_DIR = "contexts"

CACHE_DIRNAME = ".contexts_cache"
REGISTRY_DB = "registry.db"
VECTOR_DIRNAME = "vector"
GRAPH_DIRNAME = "graphs"
SQL_DIRNAME = "sql"
ARTIFACTS_DIRNAME = "artifacts"
SCRATCH_DIRNAME = "scratch"

ROOT_PROMPT_VERBOSITY = "heavy"
SUB_PROMPT_VERBOSITY = "heavy"
REPL_LANGUAGE = "python"
PIP_INSTALL_PACKAGES = "chromadb networkx pypdf"
CODE_EXECUTION_TIMEOUT = 300
MAX_OUTPUT_LENGTH = 8192
MAX_TURNS = 20
SUB_LLM_MAX_TURNS = 4
ENV_ID = "long_context_retrieval"
USER_PROMPT = (
    "Answer the question using the research-paper workspace and provide citations."
)

SYSTEM_PROMPT = dedent(
    """
    You are operating in a Recursive Language Model (RLM) environment over documents.
    This environment is meant to be used agentically. You should heavily decompose tasks,
    delegated sub-calls via `llm_batch`, programmatic branching, and tool-driven iteration.

    The workspace may contain one or many PDF files. The system manages only a thin registry and
    persistent cache. You decide whether to parse PDFs, create chunks, embed text, define scratch
    SQL schemas, build vector collections, create graph structures, or use the filesystem.

    This is an iterative environment:
    1. Inspect the workspace and registry first.
    2. Decide what evidence or intermediate structure is needed.
    3. Create only the artifacts you need.
    4. Reuse persistent cache artifacts when helpful.
    5. Materialize intermediate results into scratch namespaces when they simplify reasoning.
    6. Only finalize once the answer and citations are grounded.

    You are strongly encouraged to use `llm_batch()` as much as possible when tasks can be split.
    Prefer parallel sub-calls over sequential semantic work. Use it for:
    - searching across multiple PDFs or document groups in parallel
    - extracting evidence from multiple pages, chunks, or candidate documents
    - comparing competing hypotheses or answers
    - summarizing independent evidence buffers before synthesis
    - delegating sub-problems that need their own tool use

    Build a programmatic strategy in the REPL:
    - branch over candidate documents, sections, or artifacts
    - aggregate sub-call results in Python
    - decide whether more retrieval, indexing, graph construction, or SQL materialization is needed
    - rerank and verify before answering

    Do not default to a single linear pass if the question can benefit from decomposition.
    If the corpus is large, partition it and query sub-LLMs in parallel. If a subtask needs its own
    tool use, delegate it. If a cheap approach fails, escalate to richer artifact creation.

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
    dataset_path: str | None = None
    dataset_output_dir: str = CONTEXTS_DIR
    max_examples: int | None = None
    rlm_model: str | None = None
    max_turns: int = MAX_TURNS
    repl_language: str = REPL_LANGUAGE
    sub_llm_max_turns: int = SUB_LLM_MAX_TURNS
    sub_prompt_verbosity: str = SUB_PROMPT_VERBOSITY
    root_prompt_verbosity: str = ROOT_PROMPT_VERBOSITY
    pip_install_packages: str = PIP_INSTALL_PACKAGES
    code_execution_timeout: int = CODE_EXECUTION_TIMEOUT
    max_output_length: int = MAX_OUTPUT_LENGTH
    sub_max_completion_tokens: int | None = None
    root_max_completion_tokens: int | None = None
    path_anchor: str | None = None
    context_dir: str | None = None
    workspace_dir: str | None = None
    pdf_dir: str | None = None
    pdf_paths: list[str] | None = None
    workspace_cache_root: str | None = None
    env_id: str = ENV_ID

    @classmethod
    def from_input(cls, cfg: Config | dict[str, Any] | None) -> Config:
        if cfg is None:
            return cls()
        if isinstance(cfg, cls):
            return cfg
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})

    def to_env_args(self) -> dict[str, Any]:
        """Return JSON-serializable args for `vf.load_environment(env_id, env_args)`."""
        return {
            "dataset_path": self.dataset_path,
            "dataset_output_dir": self.dataset_output_dir,
            "max_examples": self.max_examples,
            "rlm_model": self.rlm_model,
            "max_turns": int(self.max_turns),
            "repl_language": self.repl_language,
            "sub_llm_max_turns": int(self.sub_llm_max_turns),
            "sub_prompt_verbosity": self.sub_prompt_verbosity,
            "root_prompt_verbosity": self.root_prompt_verbosity,
            "pip_install_packages": self.pip_install_packages,
            "code_execution_timeout": int(self.code_execution_timeout),
            "max_output_length": int(self.max_output_length),
            "sub_max_completion_tokens": self.sub_max_completion_tokens,
            "root_max_completion_tokens": self.root_max_completion_tokens,
            "path_anchor": self.path_anchor,
            "context_dir": self.context_dir,
            "workspace_dir": self.workspace_dir,
            "pdf_dir": self.pdf_dir,
            "pdf_paths": self.pdf_paths,
            "workspace_cache_root": self.workspace_cache_root,
            "env_id": self.env_id,
        }
