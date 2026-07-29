# CLAUDE.md — working notes for this repo

Pipeline: `transcribe.py` (YouTube/Vimeo → Whisper text) → `diarize.py` (AssemblyAI
speaker labels) → `index.py` (searchable summary index). Source channels and per-group
status live in [`channel/`](channel/README.md); derived answers in [`queries/`](queries/README.md).

**Step-by-step how-to for transcribing + diarizing is in [`AGENTS.md`](AGENTS.md)** — read
it before running the pipeline. The rule and gotchas below are the essentials.

## STANDING RULE: always diarize newly-transcribed videos

Whenever new videos are transcribed, **always run `diarize.py` (AssemblyAI) on them as
part of the same job** — transcription alone is not "done". The expected end state for a
new video is a `*.speakers.json` + `*.speakers.txt` alongside the plain transcript.

- Applies to every group. Solo-speaker videos (e.g. group1 monologues) simply come back
  as a single speaker — that's fine; diarize them anyway rather than special-casing.
- Speaker **naming** (letters → real names) is a separate, heavier step; do it when asked,
  it is not part of the automatic "new video" flow.
- The **AssemblyAI key is deliberately never committed**. Supply it at run time via
  `ASSEMBLYAI_API_KEY` (env or `.env`); do not write it into any tracked file.

Standard flow for new videos:
```bash
python transcribe.py --group <g>            # or --channel <url> --channel-name <g>
python diarize.py transcriptions/<g>/<name>.json   # for each newly-transcribed file
```

## Gotchas

- **Dotted filenames**: several transcript names contain dots (e.g. `...5_mio._kr.json`,
  `...3.489...`). Never derive sibling names with `Path.with_suffix(".speakers.json")` —
  it truncates at the last dot. Strip the trailing `.json` by string slice instead
  (`name[:-len(".json")] + ".speakers.json"`). `diarize.py` was fixed to do this; keep it.
- **Title drift → duplicates**: the skip-check keys off the slugified title, so if a
  video's title changes upstream, a re-run re-transcribes it under a new filename. Dedup
  by YouTube video id (`?v=…`), not by filename. Match a diarization to its transcript by
  `(url, segment-count)`, not by name.
- Channel URLs are **not** stored in the scrape cache — they live in `channel/channels.json`.
