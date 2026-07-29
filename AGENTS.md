# AGENTS.md — Transcribe & Diarize runbook

Instructions for an AI agent (with access to this repo) to transcribe videos and add
speaker labels. Follow these steps exactly; the invariants at the bottom prevent the
mistakes that have actually happened here.

Pipeline: **download audio → Whisper transcript → AssemblyAI diarization → (optional)
name speakers → index**. Whisper produces the text; AssemblyAI is used *only* for "who
spoke when" (its own transcript is discarded).

---

## 0. Prerequisites (check before doing anything)

```bash
pip install openai python-dotenv yt-dlp      # runtime deps
ffmpeg -version && ffprobe -version          # must be on PATH
```

API keys — **never commit these**; supply at run time via `.env` or environment:
- `OPENAI_API_KEY` — required for transcription (Whisper).
- `ASSEMBLYAI_API_KEY` — required for diarization. It is deliberately absent from the
  repo. If it is not set, ask the user for it; do not invent one.

Validate the AssemblyAI key before a big batch (cheap GET, avoids wasting time):
```bash
curl -s -o /dev/null -w "%{http_code}" \
  -H "authorization: $ASSEMBLYAI_API_KEY" \
  "https://api.assemblyai.com/v2/transcript?limit=1"     # 200 = OK, 401 = bad key
```

---

## 1. Transcribe

Groups are defined in `VIDEO_GROUPS` in `transcribe.py`, or scraped from a channel.
Source channels per group are recorded in [`channel/channels.json`](channel/channels.json).

```bash
# A predefined group (group1..group4)
python transcribe.py --group group1

# Scrape a whole channel into a group (cache saved to channel/channel_cache/<name>.json)
python transcribe.py --channel "https://www.youtube.com/@handle" --channel-name group3 --limit 0

# Split phases for large channels
python transcribe.py --channel "<url>" --channel-name group3 --limit 0 --download-only
python transcribe.py --channel "<url>" --channel-name group3 --limit 0 --transcribe-only
```

Outputs per video: `audio/<group>/<slug>.mp3`, `transcriptions/<group>/<slug>.{txt,json}`.
Already-done videos (`.txt` **and** `.json` present) are skipped unless `--force` — so
re-running is safe and cheap. **Never pass `--force` to pick up new channel videos** (it
re-transcribes everything and costs money); instead refresh the cache (see §4).

Language defaults to Danish (`--language da`). Downloads sometimes fail with a transient
**HTTP 403** — just retry those videos; it is not a hard block.

## 2. Diarize — ALWAYS do this for newly transcribed videos

Diarization is **not optional**: every newly transcribed video must get speaker labels in
the same job. Expected end state per video: a `<slug>.speakers.json` + `<slug>.speakers.txt`
next to the transcript.

```bash
# one file (audio auto-resolved from audio/<group>/<slug>.mp3)
python diarize.py transcriptions/group3/<slug>.json

# override audio if the auto-match is ambiguous
python diarize.py transcriptions/group3/<slug>.json --audio audio/group3/<file>.mp3
```

To diarize a batch, loop over every transcript that lacks a correctly-named
`*.speakers.json`. Parallelize (AssemblyAI work is I/O-bound; ~6 workers is fine) and
retry each file up to ~3× on transient errors. Skip anything already diarized.

Solo-speaker videos (e.g. group1 monologues) are fine — they simply come back as one
speaker. Diarize them anyway rather than special-casing.

## 3. (Optional) Name speakers — only when asked

Diarization labels speakers as letters (A/B/C). Mapping to real names is a **separate,
manual** step, not part of the automatic flow.

```bash
# offline relabel of an existing .speakers.json (no AssemblyAI call, no key, no cost)
python diarize.py --relabel --names "A=Olivia,B=Kasper,C=Morten" \
  transcriptions/group4/<slug>.speakers.json

# or name during a fresh diarization run
python diarize.py transcriptions/group4/<slug>.json --names "A=Olivia,B=Kasper"
```

## 4. Check a tracked channel for new videos

Channel URLs live in `channel/channels.json` (the scrape cache does **not** store them).

1. Scrape the channel listing with yt-dlp `extract_flat` on `<url>/videos`.
2. Diff by **video id** (`?v=…`) against the cached/known set (curated groups: the
   hardcoded `VIDEO_GROUPS` list; scraped groups: `channel/channel_cache/<group>.json`).
3. For a **curated** group (e.g. group1) add the new entries to `VIDEO_GROUPS` first;
   for a **scraped** group refresh the cache. Then run §1 → §2 on the new videos.

---

## Invariants — do not violate

- **Filenames may contain dots** (`...5_mio._kr.json`, `...3.489...`). Never build sibling
  names with `Path.with_suffix(".speakers.json")` — it truncates at the last dot. Strip
  the trailing `.json` by string slice: `name[:-len(".json")] + ".speakers.json"`.
  `diarize.py` already does this; keep it that way.
- **Identity is the YouTube video id, not the filename.** A video's upstream title can
  change, which changes its slug; the name-based skip check then misses it and it gets
  re-transcribed under a new name. Detect duplicates by id, and match a diarization to a
  transcript by `(url, segment-count)` — never by filename alone.
- **Never `--force`** to fetch new videos (see §1).
- **Keys are never committed.** Supply via env/`.env` only.
- **Atomic writes**: `diarize.py` writes `.speakers.*` atomically (temp + `os.replace`) so
  an interrupted run never leaves a truncated file that looks "done". Preserve this.

## Verify before declaring done

```bash
# every transcript has a correctly-named speaker pair; no orphans
python - <<'PY'
from pathlib import Path
for g in ['group1','group2','group3','group4']:
    T=Path('transcriptions')/g
    if not T.exists(): continue
    plain={f.name[:-5] for f in T.glob('*.json') if not f.name.endswith('.speakers.json')}
    spk={f.name[:-len('.speakers.json')] for f in T.glob('*.speakers.json')}
    print(g, '| transcripts', len(plain), '| missing speakers', sorted(plain-spk)[:5],
          '| orphan speakers', sorted(spk-plain)[:5])
PY
```

(group1's 13 legacy monologues are intentionally not diarized; everything transcribed
after 2026-07-29 should be.)
