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

WORKSPACE_CONTEXT_NOTE = dedent(
    """
    The full task materials live inside the local workspace and cache.
    Before you answer, you MUST inspect the staged workspace resources with your REPL:

    - read `workspace_overview.txt` in `cache` scope first for the quickstart and document inventory
    - read `workspace_manifest.json` in `cache` scope for exact paths and cache locations
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

    * Use the REPL mainly for orchestration, data movement, filtering, caching, and verification.
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
    persistent cache and scratch storage. You decide whether to parse PDFs, create text artifacts,
    build scratch SQL tables, create vector indices, materialize graphs, or use the filesystem.

    Starting procedure:
    1. Inspect the staged workspace resources first.
    2. Read `workspace_overview.txt` and `workspace_manifest.json` from `cache` scope.
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

    Do not answer from memory. Ground all reasoning in the local workspace, the cache artifacts, and
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
