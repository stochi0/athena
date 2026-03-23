# long-context-retrieval

`long-context-retrieval` is a Prime / `verifiers` `RLMEnv` for autonomous question answering over one or more research-paper PDFs in a local workspace.

The environment gives the model:

- a copied PDF workspace inside the Python REPL
- raw CRUD tools for SQL, vector, graph, and filesystem storage
- a thin system-owned registry for documents, artifacts, and provenance
- persistent cache storage across queries and rollout-scoped scratch namespaces
- `llm_batch()` support from `verifiers` for task decomposition

The model decides whether to parse PDFs, chunk text, create embeddings, define scratch schemas, build graphs, or reuse cached artifacts.
The root RLM is explicitly prompted to decompose work aggressively and use `llm_batch()` in parallel whenever the task can be split. Delegated sub-LLMs inherit the same shared SQL/vector/graph/filesystem/provenance tool surface.

## Install

```bash
uv sync
```

## Workspace Model

Each example points at a workspace, not at prebuilt `papers.db` / `chroma` / `graph.pkl` artifacts.

Accepted input shapes in `info`:

- `workspace_dir`: existing directory containing PDFs
- `pdf_dir`: directory to scan for PDFs
- `pdf_paths`: explicit list of PDF files
- `workspace_cache_root`: optional persistent cache root override

The environment initializes a registry under the cache root with:

- `documents`
- `artifacts`
- `artifact_provenance`
- `namespaces`

## Root Tools

The root model accesses these tools from inside the Python REPL:

- `sql_query(query, scope="registry", db_name="registry")`
- `sql_write(stmt, scope="scratch", db_name="main")`
- `vector_list_collections(scope="cache")`
- `vector_search(query, collection, n=5, scope="cache", where_json="{}")`
- `vector_upsert(ids, docs, meta_json, collection, scope="scratch")`
- `vector_delete(collection, ids_json="[]", where_json="{}", scope="scratch")`
- `graph_query(op, params_json="{}", graph_name="main", scope="scratch")`
- `graph_write(nodes_json="[]", edges_json="[]", graph_name="main", scope="scratch")`
- `fs_list(path=".", scope="workspace")`
- `fs_read(path, scope="workspace", encoding="utf-8")`
- `fs_write(path, content, scope="scratch", overwrite=False, encoding="utf-8")`
- `fs_mkdir(path, scope="scratch")`
- `fs_delete(path, scope="scratch", recursive=False)`
- `register_artifact(...)`
- `register_provenance(...)`

PDF parsing and other low-level processing happen in Python with installed libraries such as `pypdf`.

## Answer Contract

Final answers should be JSON:

```json
{
  "answer": "short answer text",
  "citations": [
    {
      "document_id": "doc-id",
      "path": "pdfs/paper.pdf",
      "page": 1,
      "excerpt": "supporting text"
    }
  ]
}
```

## Usage

### Load from a workspace directly

```python
from long_context_retrieval import load_environment

env = load_environment(
    workspace_dir="/abs/path/to/workspace",
)
```

You can also pass `pdf_dir` or `pdf_paths` instead of `workspace_dir`.

### Load from a dataset path

```python
from long_context_retrieval import load_environment

env = load_environment(
    dataset_path="contexts/dataset.jsonl",  # or a dataset_hf directory
)
```

### Build from in-memory rows

```python
from datasets import Dataset

from core.environment import create_environment
from core.rewards import build_default_rubric

dataset = Dataset.from_list(
    [
        {
            "prompt": [
                {
                    "role": "user",
                    "content": "Answer the question using the workspace and provide citations.",
                }
            ],
            "answer": '["Alice Chen"]',
            "info": {"workspace_dir": "/abs/path/to/workspace"},
        }
    ]
)

env = create_environment(dataset=dataset, rubric=build_default_rubric())
```

## Dataset Generation

Generate a shared arXiv-backed PDF workspace plus dataset rows:

```bash
uv run python scripts/build_dataset.py --query "cat:cs.IR" --max-papers 10
```

This creates:

- `paper_workspace_dataset_out/workspace/`
- `paper_workspace_dataset_out/.paper_workspace_cache/`
- `paper_workspace_dataset_out/dataset_hf/`
- `paper_workspace_dataset_out/dataset.jsonl`

## Verification

Run lint checks:

```bash
uv run ruff check .
```

Run tests (if/when present):

```bash
uv run pytest -q
```

## Evaluation

Push the environment:

```bash
prime env push --path .
```

Run eval:

```bash
prime eval run eval.toml --skip-upload
```

The eval config in this repo uses `env_id = "long_context_retrieval"`.
