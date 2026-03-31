#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

# Prefer this repo's `core` package over the unrelated PyPI `core` (site-packages is often
# ordered before the project root when using `uv run`).
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from typing import Any
from xml.etree import ElementTree as ET

import requests
from core import config
from core.types import WorkspaceConfig
from core.workspace import get_paths, init_workspace
from datasets import Dataset

ARXIV_API = "https://export.arxiv.org/api/query"
ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}
WORD_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")
STOPWORDS = {
    "about",
    "across",
    "after",
    "against",
    "allow",
    "allows",
    "also",
    "among",
    "approach",
    "approaches",
    "based",
    "because",
    "become",
    "becomes",
    "been",
    "being",
    "between",
    "both",
    "challenge",
    "challenges",
    "data",
    "different",
    "effectively",
    "from",
    "have",
    "however",
    "into",
    "many",
    "more",
    "most",
    "need",
    "paper",
    "papers",
    "present",
    "propose",
    "proposed",
    "proposes",
    "provide",
    "provides",
    "results",
    "show",
    "shows",
    "such",
    "task",
    "than",
    "that",
    "their",
    "there",
    "these",
    "this",
    "using",
    "where",
    "which",
    "with",
    "within",
    "work",
}


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def safe_filename(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[^\w\-.]+", "_", text)
    return text[:180] or "item"


def parse_published_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def tokenize_keywords(*parts: str) -> set[str]:
    tokens: set[str] = set()
    for part in parts:
        for token in WORD_RE.findall((part or "").lower()):
            if len(token) < 4 or token.isdigit() or token in STOPWORDS:
                continue
            tokens.add(token)
    return tokens


def count_words(text: str) -> int:
    return len(WORD_RE.findall((text or "").lower()))


def parse_arxiv_entries(xml_text: str) -> list[dict[str, Any]]:
    root = ET.fromstring(xml_text)
    entries: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        arxiv_id_url = entry.findtext("atom:id", default="", namespaces=ATOM_NS)
        arxiv_id = arxiv_id_url.rsplit("/abs/", 1)[-1] if "/abs/" in arxiv_id_url else arxiv_id_url
        title = normalize_ws(entry.findtext("atom:title", default="", namespaces=ATOM_NS))
        summary = normalize_ws(entry.findtext("atom:summary", default="", namespaces=ATOM_NS))
        authors = [
            normalize_ws(author.findtext("atom:name", default="", namespaces=ATOM_NS))
            for author in entry.findall("atom:author", ATOM_NS)
        ]
        authors = [author for author in authors if author]
        pdf_url = ""
        for link in entry.findall("atom:link", ATOM_NS):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href", "")
                break
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        entries.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "summary": summary,
                "authors": authors,
                "published": entry.findtext("atom:published", default="", namespaces=ATOM_NS),
                "updated": entry.findtext("atom:updated", default="", namespaces=ATOM_NS),
                "pdf_url": pdf_url,
            }
        )
    return entries


def fetch_arxiv_papers(
    query: str,
    max_papers: int,
    batch_size: int = 50,
    sleep_s: float = 1.0,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    start = 0
    headers = {"User-Agent": "paper-workspace-dataset/1.0"}
    while len(out) < max_papers:
        remaining = max_papers - len(out)
        take = min(batch_size, remaining)
        response = requests.get(
            ARXIV_API,
            params={
                "search_query": query,
                "start": start,
                "max_results": take,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()
        batch = parse_arxiv_entries(response.text)
        if not batch:
            break
        out.extend(batch)
        start += len(batch)
        if len(batch) < take:
            break
        time.sleep(sleep_s)
    return out[:max_papers]


def download_pdf(pdf_url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(
        pdf_url,
        stream=True,
        timeout=120,
        headers={"User-Agent": "paper-workspace-dataset/1.0"},
    ) as response:
        response.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 128):
                if chunk:
                    fh.write(chunk)


def build_workspace(output_dir: Path, papers: list[dict[str, Any]]) -> Path:
    workspace_root = output_dir / "workspace"
    pdf_dir = workspace_root / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    for paper in papers:
        pdf_name = safe_filename(f"{paper['arxiv_id']}.pdf")
        download_pdf(paper["pdf_url"], pdf_dir / pdf_name)

    metadata_path = workspace_root / "papers.json"
    metadata_path.write_text(json.dumps(papers, ensure_ascii=False, indent=2), encoding="utf-8")

    init_workspace(
        get_paths(
            WorkspaceConfig(
                workspace_root=workspace_root,
                state_root=workspace_root / config.WORKSPACE_STATE_DIRNAME,
            )
        )
    )
    return workspace_root


def prepare_papers(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for paper in papers:
        title = normalize_ws(str(paper.get("title", "")))
        summary = normalize_ws(str(paper.get("summary", "")))
        authors = [
            normalize_ws(str(author))
            for author in paper.get("authors", [])
            if normalize_ws(str(author))
        ]
        published = str(paper.get("published", ""))
        prepared.append(
            {
                **paper,
                "title": title,
                "summary": summary,
                "authors": authors,
                "first_author": authors[0] if authors else "Unknown Author",
                "published": published,
                "published_date": published[:10] if published else "",
                "published_ts": parse_published_timestamp(published),
                "title_tokens": tokenize_keywords(title),
                "summary_tokens": tokenize_keywords(summary),
                "all_tokens": tokenize_keywords(title, summary),
                "author_count": len(authors),
                "title_word_count": count_words(title),
                "summary_word_count": count_words(summary),
            }
        )
    return prepared


def informative_keyword_counts(prepared: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for paper in prepared:
        counts.update(paper["all_tokens"])
    return Counter(
        {
            keyword: count
            for keyword, count in counts.items()
            if 2 <= count <= max(2, len(prepared) - 1)
        }
    )


def make_row(
    *,
    prompt: str,
    answer_items: list[str],
    workspace_root: Path,
    task_type: str,
) -> dict[str, Any]:
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "answer": json.dumps(answer_items, ensure_ascii=False),
        "info": {
            "workspace_dir": str(workspace_root),
            "task_type": task_type,
            "difficulty": "edge",
        },
    }


def maybe_add_row(
    rows: list[dict[str, Any]],
    seen_prompts: set[str],
    *,
    prompt: str,
    answer_items: list[str],
    workspace_root: Path,
    task_type: str,
) -> None:
    if not prompt or prompt in seen_prompts:
        return
    seen_prompts.add(prompt)
    rows.append(
        make_row(
            prompt=prompt,
            answer_items=answer_items,
            workspace_root=workspace_root,
            task_type=task_type,
        )
    )


def add_collaboration_hub_task(
    rows: list[dict[str, Any]],
    seen_prompts: set[str],
    prepared: list[dict[str, Any]],
    workspace_root: Path,
) -> None:
    coauthors: dict[str, set[str]] = defaultdict(set)
    for paper in prepared:
        authors = sorted(set(paper["authors"]))
        for left, right in combinations(authors, 2):
            coauthors[left].add(right)
            coauthors[right].add(left)
    if not coauthors:
        return
    best_author = min(
        coauthors,
        key=lambda author: (-len(coauthors[author]), author.lower(), author),
    )
    maybe_add_row(
        rows,
        seen_prompts,
        prompt=(
            "Build the undirected coauthorship graph over every paper in this workspace. "
            "Which author has the largest number of distinct coauthors? "
            "Break ties lexicographically by author name and return only that author."
        ),
        answer_items=[best_author],
        workspace_root=workspace_root,
        task_type="coauthor_graph_hub",
    )


def add_repeated_author_window_task(
    rows: list[dict[str, Any]],
    seen_prompts: set[str],
    prepared: list[dict[str, Any]],
    workspace_root: Path,
) -> None:
    author_to_timestamps: dict[str, list[datetime]] = defaultdict(list)
    for paper in prepared:
        published_ts = paper["published_ts"]
        if published_ts is None:
            continue
        for author in sorted(set(paper["authors"])):
            author_to_timestamps[author].append(published_ts)
    repeated = {
        author: timestamps
        for author, timestamps in author_to_timestamps.items()
        if len(timestamps) >= 2
    }
    if not repeated:
        return
    best_author = min(
        repeated,
        key=lambda author: (
            -(max(repeated[author]) - min(repeated[author])).total_seconds(),
            author.lower(),
            author,
        ),
    )
    maybe_add_row(
        rows,
        seen_prompts,
        prompt=(
            "Consider only authors who appear on at least two papers in this workspace. "
            "Which author spans the largest publication window, defined as the time gap between their earliest "
            "and latest paper timestamp in the workspace? Break ties lexicographically by author name "
            "and return only that author."
        ),
        answer_items=[best_author],
        workspace_root=workspace_root,
        task_type="author_publication_window",
    )


def add_similarity_pair_task(
    rows: list[dict[str, Any]],
    seen_prompts: set[str],
    prepared: list[dict[str, Any]],
    workspace_root: Path,
) -> None:
    best_pair: tuple[str, str] | None = None
    best_score = 0
    for left, right in combinations(prepared, 2):
        overlap = left["all_tokens"] & right["all_tokens"]
        score = len(overlap)
        candidate_pair = tuple(sorted([left["title"], right["title"]]))
        if score > best_score:
            best_score = score
            best_pair = candidate_pair
        elif score == best_score and best_pair is not None and candidate_pair < best_pair:
            best_pair = candidate_pair
    if not best_pair or best_score < 2:
        return
    maybe_add_row(
        rows,
        seen_prompts,
        prompt=(
            "Normalize every paper by lowercasing the title and summary, tokenizing into words, "
            "dropping stopwords and tokens shorter than 4 characters, and deduplicating tokens per paper. "
            "Under that normalization, which two papers share the largest number of informative tokens? "
            "Break ties lexicographically by the pair of titles and return the two titles sorted lexicographically."
        ),
        answer_items=list(best_pair),
        workspace_root=workspace_root,
        task_type="pairwise_keyword_overlap",
    )


def add_keyword_timeline_tasks(
    rows: list[dict[str, Any]],
    seen_prompts: set[str],
    prepared: list[dict[str, Any]],
    workspace_root: Path,
    *,
    max_tasks: int,
) -> None:
    counts = informative_keyword_counts(prepared)
    added = 0
    for keyword, _count in sorted(counts.items(), key=lambda item: (item[1], item[0])):
        matches = [paper for paper in prepared if keyword in paper["all_tokens"]]
        if len(matches) < 2:
            continue
        ordered = sorted(
            matches,
            key=lambda paper: (
                paper["published"],
                paper["title"].lower(),
                paper["title"],
            ),
        )
        maybe_add_row(
            rows,
            seen_prompts,
            prompt=(
                f"Consider only papers whose title or summary contains the keyword '{keyword}' after lowercasing "
                "and tokenization. List the matching paper titles from earliest published to latest published. "
                "Break ties lexicographically by title."
            ),
            answer_items=[paper["title"] for paper in ordered],
            workspace_root=workspace_root,
            task_type="keyword_timeline",
        )
        added += 1
        if added >= max_tasks:
            break


def add_keyword_author_count_tasks(
    rows: list[dict[str, Any]],
    seen_prompts: set[str],
    prepared: list[dict[str, Any]],
    workspace_root: Path,
    *,
    max_tasks: int,
) -> None:
    counts = informative_keyword_counts(prepared)
    added = 0
    for keyword, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        matches = [paper for paper in prepared if keyword in paper["all_tokens"]]
        unique_dates = sorted(
            {paper["published_date"] for paper in matches if paper["published_date"]}
        )
        if len(matches) < 2 or len(unique_dates) < 2:
            continue
        cutoff = unique_dates[len(unique_dates) // 2 - 1]
        filtered = [paper for paper in matches if paper["published_date"] > cutoff]
        if len(filtered) < 2:
            continue
        distinct_authors = sorted({author for paper in filtered for author in paper["authors"]})
        if len(distinct_authors) < 2:
            continue
        maybe_add_row(
            rows,
            seen_prompts,
            prompt=(
                f"How many distinct authors appear on papers published after {cutoff} whose title or summary "
                f"contains the keyword '{keyword}' after lowercasing and tokenization? Count every listed author, "
                "not just first authors, and return only the integer."
            ),
            answer_items=[str(len(distinct_authors))],
            workspace_root=workspace_root,
            task_type="keyword_author_count",
        )
        added += 1
        if added >= max_tasks:
            break


def add_author_threshold_tasks(
    rows: list[dict[str, Any]],
    seen_prompts: set[str],
    prepared: list[dict[str, Any]],
    workspace_root: Path,
    *,
    max_tasks: int,
) -> None:
    counts = informative_keyword_counts(prepared)
    added = 0
    for keyword, _count in sorted(counts.items(), key=lambda item: (item[1], item[0])):
        matches = [paper for paper in prepared if keyword in paper["all_tokens"]]
        if len(matches) < 2:
            continue
        thresholds = sorted({paper["author_count"] for paper in matches})
        if len(thresholds) < 2:
            continue
        threshold = thresholds[len(thresholds) // 2]
        filtered = [paper for paper in matches if paper["author_count"] >= threshold]
        if len(filtered) < 2:
            continue
        best_paper = min(
            filtered,
            key=lambda paper: (
                paper["title_word_count"],
                paper["published"],
                paper["title"].lower(),
                paper["title"],
            ),
        )
        maybe_add_row(
            rows,
            seen_prompts,
            prompt=(
                f"Among papers whose title or summary contains the keyword '{keyword}' after lowercasing and "
                f"tokenization, keep only the papers with at least {threshold} listed authors. "
                "Within that filtered set, find the paper with the shortest title in word count. "
                "Break ties by earliest publication timestamp and then lexicographically by title. "
                "Return only the first author of that paper."
            ),
            answer_items=[best_paper["first_author"]],
            workspace_root=workspace_root,
            task_type="keyword_author_threshold",
        )
        added += 1
        if added >= max_tasks:
            break


def add_global_ranking_tasks(
    rows: list[dict[str, Any]],
    seen_prompts: set[str],
    prepared: list[dict[str, Any]],
    workspace_root: Path,
) -> None:
    if len(prepared) >= 3:
        ranked = sorted(
            prepared,
            key=lambda paper: (
                -paper["author_count"],
                paper["published"],
                paper["title"].lower(),
                paper["title"],
            ),
        )
        maybe_add_row(
            rows,
            seen_prompts,
            prompt=(
                "Sort all papers by number of listed authors descending, then by publication timestamp ascending, "
                "then lexicographically by title. Which paper is ranked third in that ordering? Return only the title."
            ),
            answer_items=[ranked[2]["title"]],
            workspace_root=workspace_root,
            task_type="global_author_ranking",
        )
    ranked_by_summary = sorted(
        prepared,
        key=lambda paper: (
            -paper["summary_word_count"],
            paper["published"],
            paper["title"].lower(),
            paper["title"],
        ),
    )
    maybe_add_row(
        rows,
        seen_prompts,
        prompt=(
            "Which paper has the longest summary by word count? Break ties by earliest publication timestamp "
            "and then lexicographically by title. Return only the title."
        ),
        answer_items=[ranked_by_summary[0]["title"]],
        workspace_root=workspace_root,
        task_type="summary_length_extreme",
    )


def make_rows_for_workspace(
    papers: list[dict[str, Any]], workspace_root: Path
) -> list[dict[str, Any]]:
    prepared = prepare_papers(papers)
    rows: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()

    add_collaboration_hub_task(rows, seen_prompts, prepared, workspace_root)
    add_repeated_author_window_task(rows, seen_prompts, prepared, workspace_root)
    add_similarity_pair_task(rows, seen_prompts, prepared, workspace_root)
    add_keyword_timeline_tasks(rows, seen_prompts, prepared, workspace_root, max_tasks=3)
    add_keyword_author_count_tasks(rows, seen_prompts, prepared, workspace_root, max_tasks=2)
    add_author_threshold_tasks(rows, seen_prompts, prepared, workspace_root, max_tasks=2)
    if len(rows) < 6:
        add_global_ranking_tasks(rows, seen_prompts, prepared, workspace_root)
    if not rows:
        raise RuntimeError("Failed to generate any dataset rows from the paper metadata.")
    return rows


def generate_dataset(
    query: str,
    max_papers: int,
    output_dir: Path,
    batch_size: int = 50,
    sleep_s: float = 1.0,
) -> Dataset:
    output_dir.mkdir(parents=True, exist_ok=True)
    papers = fetch_arxiv_papers(
        query=query,
        max_papers=max_papers,
        batch_size=batch_size,
        sleep_s=sleep_s,
    )
    if not papers:
        raise RuntimeError("No arXiv papers were returned for the query.")

    workspace_root = build_workspace(output_dir, papers)
    rows = make_rows_for_workspace(papers, workspace_root)
    dataset = Dataset.from_list(rows)

    tasks_dir = output_dir / config.TASK_BUNDLE_SUBDIR
    tasks_dir.mkdir(parents=True, exist_ok=True)
    hf_dir = tasks_dir / config.TASK_BUNDLE_HF_DIRNAME
    dataset.save_to_disk(str(hf_dir))

    jsonl_path = tasks_dir / "dataset.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "query": query,
        "max_papers": max_papers,
        "num_rows": len(rows),
        "workspace_root": str(workspace_root),
        "tasks_dir": str(tasks_dir),
        "hf_dataset_dir": str(hf_dir),
        "jsonl_path": str(jsonl_path),
    }
    (tasks_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an arXiv paper workspace (output-dir/workspace/) and task bundle "
            "(output-dir/tasks/: dataset.jsonl, hf/, manifest.json)."
        ),
    )
    parser.add_argument("--query", type=str, default="cat:cs.IR", help="arXiv query string.")
    parser.add_argument("--max-papers", type=int, default=10, help="Number of papers to fetch.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=f"./{config.CONTEXTS_DIR}",
        help="Output directory.",
    )
    parser.add_argument("--batch-size", type=int, default=50, help="arXiv API batch size.")
    parser.add_argument(
        "--sleep-s", type=float, default=1.0, help="Sleep between arXiv API requests."
    )
    args = parser.parse_args()

    generate_dataset(
        query=args.query,
        max_papers=args.max_papers,
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        sleep_s=args.sleep_s,
    )


if __name__ == "__main__":
    main()
