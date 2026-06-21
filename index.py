#!/usr/bin/env python3
"""Build a summary index of transcriptions and query it to find relevant files."""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
TRANSCRIPTIONS_DIR = SCRIPT_DIR / "transcriptions"
INDEXES_DIR = SCRIPT_DIR / "indexes"

# ── Constants ────────────────────────────────────────────────────────────────
SUMMARY_MODEL = "gpt-4o-mini"
QUERY_MODEL = "gpt-4o-mini"
TEXT_TRUNCATE_CHARS = 12000  # ~3k tokens, keeps API cost low per file
API_DELAY_SECONDS = 1

SUMMARY_SYSTEM_PROMPT = """\
Du er en indholdsanalytiker. Du modtager en transskription af en dansk video/podcast.

Returner et JSON-objekt med præcis disse felter:
{
  "summary": "2-3 sætninger der beskriver hvad episoden handler om",
  "topics": ["emne1", "emne2", ...],
  "people": ["person1", "person2", ...],
  "key_points": ["pointe1", "pointe2", ...]
}

Regler:
- summary: Kort og præcist, 2-3 sætninger på dansk
- topics: 3-8 emneord/tags på dansk (f.eks. "Meta Ads", "timepriser", "SEO")
- people: Navne nævnt i episoden (gæster, eksperter, virksomheder)
- key_points: 3-6 vigtige pointer eller konklusioner fra episoden, på dansk
- Returner KUN valid JSON, ingen markdown eller ekstra tekst"""

QUERY_SYSTEM_PROMPT = """\
Du er en søgeassistent. Du modtager en liste af video-/podcastresumeer med indeks-numre, \
og et spørgsmål fra brugeren.

Returner et JSON-objekt:
{
  "results": [
    {"index": 0, "relevance": "high|medium|low", "reason": "kort forklaring på dansk"}
  ]
}

Regler:
- Inkluder KUN filer der er relevante for spørgsmålet
- Sortér efter relevans (mest relevant først)
- "reason" skal kort forklare HVORFOR filen er relevant
- Returner KUN valid JSON, ingen markdown eller ekstra tekst
- Hvis ingen filer er relevante, returner {"results": []}"""


# ── Indexing ─────────────────────────────────────────────────────────────────

def summarize_transcription(text: str, title: str, client: OpenAI) -> dict:
    """Generate a structured summary of a transcription using OpenAI."""
    truncated = text[:TEXT_TRUNCATE_CHARS]
    if len(text) > TEXT_TRUNCATE_CHARS:
        truncated += "\n\n[... resten af transskriptionen er afkortet ...]"

    response = client.chat.completions.create(
        model=SUMMARY_MODEL,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": f"Titel: {title}\n\nTransskription:\n{truncated}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    return json.loads(response.choices[0].message.content)


def build_index(group: str, client: OpenAI, force: bool = False) -> dict:
    """Build or update the summary index for a transcription group."""
    group_dir = TRANSCRIPTIONS_DIR / group
    if not group_dir.exists():
        raise FileNotFoundError(f"Transcription group not found: {group_dir}")

    json_files = sorted(group_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No transcription JSON files in {group_dir}")

    # Load existing index for incremental updates
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)
    index_path = INDEXES_DIR / f"{group}.json"
    existing = {}
    if not force and index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            old_index = json.load(f)
        existing = {entry["file"]: entry for entry in old_index.get("files", [])}

    entries = []
    for i, json_file in enumerate(json_files):
        stem = json_file.stem

        # Skip if already indexed (unless --force)
        if stem in existing:
            print(f"  [{i+1}/{len(json_files)}] SKIP (indexed): {stem}")
            entries.append(existing[stem])
            continue

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        title = data.get("title", stem)
        text = data.get("full_text", "")

        print(f"  [{i+1}/{len(json_files)}] Summarizing: {title}")
        summary_data = summarize_transcription(text, title, client)

        entries.append({
            "file": stem,
            "title": title,
            "url": data.get("url", ""),
            "duration_seconds": data.get("duration_seconds", 0),
            "language": data.get("language", "da"),
            "summary": summary_data.get("summary", ""),
            "topics": summary_data.get("topics", []),
            "people": summary_data.get("people", []),
            "key_points": summary_data.get("key_points", []),
        })

        if i < len(json_files) - 1:
            time.sleep(API_DELAY_SECONDS)

    index = {
        "group": group,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(entries),
        "files": entries,
    }

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"\nIndex saved: {index_path.relative_to(SCRIPT_DIR)} ({len(entries)} files)")
    return index


# ── Querying ─────────────────────────────────────────────────────────────────

def query_index(group: str, question: str, client: OpenAI) -> list[dict]:
    """Query the index to find transcriptions relevant to a question."""
    index_path = INDEXES_DIR / f"{group}.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"No index found for '{group}'. Run: python index.py build --group {group}"
        )

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    # Build context block from summaries
    file_list = []
    for i, entry in enumerate(index["files"]):
        topics = ", ".join(entry.get("topics", []))
        people = ", ".join(entry.get("people", []))
        points = "; ".join(entry.get("key_points", []))
        file_list.append(
            f"[{i}] \"{entry['title']}\"\n"
            f"    Resumé: {entry['summary']}\n"
            f"    Emner: {topics}\n"
            f"    Personer: {people}\n"
            f"    Pointer: {points}"
        )
    context = "\n\n".join(file_list)

    response = client.chat.completions.create(
        model=QUERY_MODEL,
        messages=[
            {"role": "system", "content": QUERY_SYSTEM_PROMPT},
            {"role": "user", "content": f"Filer:\n{context}\n\nSpørgsmål: {question}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    result = json.loads(response.choices[0].message.content)
    matches = result.get("results", [])

    # Enrich results with file metadata
    for match in matches:
        idx = match["index"]
        if 0 <= idx < len(index["files"]):
            entry = index["files"][idx]
            match["title"] = entry["title"]
            match["file"] = entry["file"]
            match["url"] = entry["url"]
            match["summary"] = entry["summary"]
            match["topics"] = entry["topics"]

    return matches


def print_query_results(matches: list[dict], group: str):
    """Pretty-print query results."""
    if not matches:
        print("No relevant files found.")
        return

    print(f"\nFound {len(matches)} relevant file(s):\n")
    for i, m in enumerate(matches, 1):
        relevance_marker = {"high": "***", "medium": "**", "low": "*"}.get(m.get("relevance", ""), "")
        print(f"  {i}. [{m.get('relevance', '?').upper()}] {m.get('title', '?')}")
        print(f"     {m.get('reason', '')}")
        print(f"     File: transcriptions/{group}/{m.get('file', '?')}.json")
        print(f"     URL:  {m.get('url', '?')}")
        print(f"     Topics: {', '.join(m.get('topics', []))}")
        print()


# ── List ─────────────────────────────────────────────────────────────────────

def list_index(group: str):
    """Print the index contents as a readable overview."""
    index_path = INDEXES_DIR / f"{group}.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"No index found for '{group}'. Run: python index.py build --group {group}"
        )

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    print(f"\nIndex: {group} ({index['file_count']} files, indexed {index['indexed_at']})\n")
    for i, entry in enumerate(index["files"], 1):
        duration_min = int(entry.get("duration_seconds", 0) // 60)
        topics = ", ".join(entry.get("topics", []))
        print(f"  {i:>3}. {entry['title']}  ({duration_min} min)")
        print(f"       {entry['summary']}")
        print(f"       Emner: {topics}")
        people = entry.get("people", [])
        if people:
            print(f"       Personer: {', '.join(people)}")
        print()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build and query transcription summary indexes"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # build
    build_parser = sub.add_parser("build", help="Build/update the summary index for a group")
    build_parser.add_argument("--group", required=True, help="Transcription group name")
    build_parser.add_argument("--force", action="store_true", help="Re-index all files")

    # query
    query_parser = sub.add_parser("query", help="Find relevant files for a question")
    query_parser.add_argument("--group", required=True, help="Transcription group name")
    query_parser.add_argument("question", help="The question to search for")

    # list
    list_parser = sub.add_parser("list", help="Show the index contents")
    list_parser.add_argument("--group", required=True, help="Transcription group name")

    args = parser.parse_args()

    load_dotenv(SCRIPT_DIR / ".env")

    if args.command == "list":
        list_index(args.group)
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Check your .env file.")
        return
    client = OpenAI(api_key=api_key)

    if args.command == "build":
        build_index(args.group, client, force=args.force)
    elif args.command == "query":
        matches = query_index(args.group, args.question, client)
        print_query_results(matches, args.group)


if __name__ == "__main__":
    main()
