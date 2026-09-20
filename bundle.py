#!/usr/bin/env python3
"""Build bundle/: retrieval-ready Markdown files for a ChatGPT / Claude Project.

Why: 366 small transcripts with slug filenames retrieve poorly. A Project knowledge base
(file search) works far better with a few large, well-labelled files where every episode
carries its publish date, source, speakers, summary and topics right next to the text.

Output (all under bundle/):
  00_catalog.md                 every episode newest-first with date, summary, topics, URL,
                                and which bundle file holds the full transcript
  <label>_partNN_<from>_<to>.md chronological chunks per source (~1 MB each); each starts
                                with a table of contents, then one section per episode
  README.md                     how to use it + suggested project instructions
  manifest.json                 sha256 per file so you can see which files changed and
                                only re-upload those after a new batch

Inputs: transcriptions/ (.speakers.txt preferred, plain .txt fallback), indexes/<group>.json
(summary/topics/key points), channel/video_meta.json (publish dates, from video_meta.py),
channel/channels.json (source names).

    python bundle.py             # rebuild everything, report changed files
"""
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
TRANSCRIPTIONS_DIR = SCRIPT_DIR / "transcriptions"
INDEXES_DIR = SCRIPT_DIR / "indexes"
META_PATH = SCRIPT_DIR / "channel" / "video_meta.json"
CHANNELS_PATH = SCRIPT_DIR / "channel" / "channels.json"
BUNDLE_DIR = SCRIPT_DIR / "bundle"

CHUNK_CHARS = 1_000_000          # target size per chunk file (chars, ~300k tokens)
# Short file-name labels per group; groups sharing a label are bundled together.
LABELS = {"group1": "fuldt-booket", "group2": "workshops", "group3": "marketingpod", "group4": "workshops"}
SOURCE_KIND = {"group1": "YouTube", "group2": "Vimeo", "group3": "YouTube", "group4": "Webinar"}

_SPEAKERS_HDR = re.compile(r"^\[Speakers:\s*([^|\]]+)")


def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_indexes() -> dict[str, dict]:
    """(group, stem) -> index entry."""
    out = {}
    for p in INDEXES_DIR.glob("*.json"):
        for e in load_json(p).get("files", []):
            out[(p.stem, e["file"])] = e
    return out


def transcript_body(group: str, stem: str) -> tuple[str, list[str] | None]:
    """Return (text, speakers). Prefer the speaker-labelled version."""
    spk = TRANSCRIPTIONS_DIR / group / f"{stem}.speakers.txt"
    if spk.exists():
        raw = spk.read_text(encoding="utf-8")
        head, _, body = raw.partition("\n\n")
        speakers = None
        for line in head.splitlines():
            m = _SPEAKERS_HDR.match(line.strip())
            if m:
                speakers = [s.strip() for s in m.group(1).split(",")]
        return body.strip(), speakers
    plain = TRANSCRIPTIONS_DIR / group / f"{stem}.txt"
    return plain.read_text(encoding="utf-8").strip(), None


def fmt_minutes(sec) -> str:
    try:
        return f"{round(float(sec) / 60)} min"
    except (TypeError, ValueError):
        return "ukendt varighed"


def episode_section(ep: dict) -> str:
    lines = [f"# {ep['title']}", ""]
    if ep["date"]:
        lines.append(f"- Dato (udgivet): {ep['date']}")
    else:
        lines.append(f"- Dato (udgivet): ukendt — transskriberet {ep['transcribed'] or 'ukendt dato'}")
    lines.append(f"- Kilde: {ep['source']} ({ep['kind']}) — {ep['url']}")
    lines.append(f"- Varighed: {fmt_minutes(ep['duration'])}")
    if ep["speakers"]:
        lines.append(f"- Talere: {', '.join(ep['speakers'])}")
    idx = ep["index"]
    if idx:
        if idx.get("summary"):
            lines.append(f"- Resumé: {idx['summary']}")
        if idx.get("topics"):
            lines.append(f"- Emner: {', '.join(idx['topics'])}")
        if idx.get("people"):
            lines.append(f"- Personer nævnt: {', '.join(idx['people'])}")
        if idx.get("key_points"):
            lines.append("- Nøglepointer:")
            lines += [f"  - {k}" for k in idx["key_points"]]
    lines += ["", "## Transskription", "", ep["text"], ""]
    return "\n".join(lines)


def build_episodes() -> list[dict]:
    meta = load_json(META_PATH)["videos"] if META_PATH.exists() else {}
    by_transcript = {(e["group"], e["transcript"]): e for e in meta.values()}
    sources = {c["group"]: c["name"] for c in load_json(CHANNELS_PATH)["channels"]}
    indexes = load_indexes()
    eps = []
    for group_dir in sorted(TRANSCRIPTIONS_DIR.iterdir()):
        if not group_dir.is_dir():
            continue
        g = group_dir.name
        for jf in sorted(group_dir.glob("*.json")):
            if jf.name.endswith(".speakers.json"):
                continue
            stem = jf.name[:-len(".json")]
            d = load_json(jf)
            m = by_transcript.get((g, stem), {})
            text, speakers = transcript_body(g, stem)
            eps.append({
                "group": g, "stem": stem,
                "title": m.get("live_title") or d.get("title", stem),
                "url": d.get("url", ""),
                "date": m.get("upload_date"),
                "transcribed": (d.get("transcribed_at") or "")[:10] or None,
                "duration": m.get("duration_seconds") or d.get("duration_seconds"),
                "source": sources.get(g, g), "kind": SOURCE_KIND.get(g, ""),
                "speakers": speakers,
                "index": indexes.get((g, stem)),
                "text": text,
            })
    return eps


def chunk(eps: list[dict]) -> list[list[dict]]:
    """Greedy chronological chunks; prefix-stable so old chunks don't churn."""
    eps = sorted(eps, key=lambda e: (e["date"] or e["transcribed"] or "9999", e["title"]))
    chunks, cur, size = [], [], 0
    for e in eps:
        n = len(e["text"])
        if cur and size + n > CHUNK_CHARS:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(e)
        size += n
    if cur:
        chunks.append(cur)
    return chunks


def write_file(path: Path, content: str):
    path.write_text(content, encoding="utf-8", newline="\n")


def main() -> int:
    BUNDLE_DIR.mkdir(exist_ok=True)
    old_manifest = load_json(BUNDLE_DIR / "manifest.json") if (BUNDLE_DIR / "manifest.json").exists() else {}
    for p in BUNDLE_DIR.glob("*.md"):
        p.unlink()

    eps = build_episodes()
    by_label = defaultdict(list)
    for e in eps:
        by_label[LABELS.get(e["group"], e["group"])].append(e)

    files: dict[str, dict] = {}
    for label, group_eps in sorted(by_label.items()):
        parts = chunk(group_eps)
        for i, part in enumerate(parts, 1):
            dates = [e["date"] for e in part if e["date"]]
            span = f"{min(dates)}_{max(dates)}" if dates else "udateret"
            name = f"{label}_part{i:02d}_of{len(parts):02d}_{span}.md"
            source = part[0]["source"]
            toc = "\n".join(f"- {e['date'] or '????-??-??'} · {e['title']} ({fmt_minutes(e['duration'])})" for e in part)
            head = (f"# {source} — del {i} af {len(parts)} ({span.replace('_', ' → ')}), {len(part)} episoder\n\n"
                    f"Hver episode nedenfor starter med en overskrift (`# titel`), metadata (dato, kilde, talere, "
                    f"resumé, emner, nøglepointer) og derefter den fulde transskription. Kronologisk rækkefølge, "
                    f"ældste først.\n\n## Indhold\n\n{toc}\n\n---\n\n")
            body = "\n---\n\n".join(episode_section(e) for e in part)
            write_file(BUNDLE_DIR / name, head + body)
            for e in part:
                e["bundle_file"] = name
            files[name] = {"episodes": len(part), "chars": len(head) + len(body), "source": source}

    # Catalog: newest first across all sources.
    cat = ["# Katalog over alle episoder (nyeste først)", "",
           "Én linje pr. episode: dato · kilde · titel · varighed · talere · resumé · emner · URL · "
           "bundle-fil med den fulde transskription. Brug denne fil til at afgøre hvilke episoder der er "
           "\"seneste\", og slå derefter op i den nævnte bundle-fil.", ""]
    for e in sorted(eps, key=lambda e: (e["date"] or e["transcribed"] or "", e["title"]), reverse=True):
        idx = e["index"] or {}
        bits = [e["date"] or f"ukendt dato (transskriberet {e['transcribed']})", e["source"], f"**{e['title']}**", fmt_minutes(e["duration"])]
        if e["speakers"]:
            bits.append("talere: " + ", ".join(e["speakers"]))
        if idx.get("summary"):
            bits.append(idx["summary"])
        if idx.get("topics"):
            bits.append("emner: " + ", ".join(idx["topics"]))
        bits.append(e["url"])
        bits.append(f"fil: {e['bundle_file']}")
        cat.append("- " + " · ".join(bits))
    cat_text = "\n".join(cat) + "\n"
    write_file(BUNDLE_DIR / "00_catalog.md", cat_text)
    files["00_catalog.md"] = {"episodes": len(eps), "chars": len(cat_text), "source": "alle"}

    readme = f"""# Bundle — transcripts packaged for a ChatGPT / Claude Project

Generated by `bundle.py` from `transcriptions/`, `indexes/` and `channel/video_meta.json`.
{len(eps)} episodes in {len(files)} files. Regenerate after every new batch; `manifest.json`
tells you which files changed so you only re-upload those.

## How to use on a phone (no computer needed)

1. Put this folder in iCloud Drive / Google Drive once (e.g. sync the repo, or copy `bundle/`).
2. In the ChatGPT or Claude app, create a Project, add every `.md` file here as project
   knowledge (Files app → pick the files), and paste the instructions below.
3. Ask by voice. Recency questions work because `00_catalog.md` lists episodes newest-first
   with publish dates, and every transcript section carries its own date.
4. After a new batch: re-run `python bundle.py`, then replace only the files that changed.

## Suggested project instructions

```
Du er assistent for VideoTranscribe: et arkiv af danske marketing-podcasts og videoer, uploadet som
Markdown-filer i dette projekt. Svar altid ud fra arkivet, aldrig fra din egen viden om episoderne.

ARKIVETS OPBYGNING
- 00_catalog.md: én linje pr. episode, sorteret nyeste først, med udgivelsesdato, kilde, titel,
  varighed, talere, resumé, emner, URL og navnet på den fil der har den fulde transskription.
- marketingpod_partNN_*.md: "Marketingpod med Halfdan Timm og Kristian Tinho", kronologiske dele;
  filnavnet viser datointervallet. fuldt-booket_*.md: Alexander (Fuldt Booket), klinikker/behandlere.
  workshops_*.md: to workshops og en Obsidian-webinar.
- Hver episode starter med "# titel" og metadata (dato, kilde, talere, resumé, emner, nøglepointer)
  og derefter "## Transskription" med replikker som "Halfdan: ..." / "Kristian: ..." / gæstens fornavn.

SÅDAN SVARER DU
1. Slå først op i 00_catalog.md for at finde de relevante episoder. "Seneste"/"nyeste" betyder den
   højeste udgivelsesdato i kataloget, ikke den episode du tilfældigvis finder først.
2. Læs derefter den fulde transskription i den fil kataloget peger på, og svar ud fra hvad der
   faktisk bliver sagt der.
3. Nævn altid hvilken episode svaret bygger på: titel og dato. Er flere episoder relevante, sig det
   kort og vælg den nyeste medmindre brugeren spørger om noget andet.
4. Spørgsmål om hvad en bestemt person siger: brug talernavnene i transskriptionen. Værterne hedder
   Halfdan (Timm) og Kristian (Tinho); Morten er fast gæstevært; gæster står med fornavn.
   Selve teksten staver navne forkert (Kristian bliver til "Christian", "Tinju", "Tino", "Tinjo";
   Halfdan til "Halvdan", "Halvstand", "Halvsten"). Det er de samme personer.
   Episoder før juni 2026 kan have talere som bogstaver (A, B, C); sig det, og udled hvem der taler
   fra sammenhængen kun hvis det er tydeligt.
5. Svar på dansk, kort og konkret, som til en kollega, medmindre brugeren beder om andet. Svarene
   bliver ofte læst højt: start med selve svaret i 2–4 sætninger, brug ingen overskrifter eller
   tabeller, og læg detaljer og citater efter, hvis der er brug for dem.
6. Gæt ikke. Dækker arkivet ikke spørgsmålet, så sig det og foreslå den nærmeste episode.
   Citér gerne ordret fra transskriptionen når det styrker svaret, og skriv hvem der sagde det.
```

## Files

| File | Episodes | Size (chars) |
|---|---:|---:|
""" + "\n".join(f"| `{n}` | {f['episodes']} | {f['chars']:,} |" for n, f in sorted(files.items())) + "\n"
    write_file(BUNDLE_DIR / "README.md", readme)

    manifest = {}
    for p in sorted(BUNDLE_DIR.glob("*.md")):
        manifest[p.name] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), **files.get(p.name, {})}
    with open(BUNDLE_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")

    changed = [n for n, m in manifest.items() if old_manifest.get(n, {}).get("sha256") != m["sha256"]]
    removed = [n for n in old_manifest if n not in manifest]
    undated = sum(1 for e in eps if not e["date"])
    print(f"Wrote {len(manifest)} files for {len(eps)} episodes ({undated} without publish date) to bundle/")
    print(f"Changed since last build: {len(changed)}" + (f" -> {', '.join(changed)}" if changed else ""))
    if removed:
        print(f"Removed: {', '.join(removed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
