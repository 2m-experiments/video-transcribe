# Video Transcribe & Index

Download videos from YouTube/Vimeo, transcribe them with OpenAI Whisper, add speaker
labels with AssemblyAI, and build a searchable summary index.

## Where to look

| File / folder | What it is |
|---|---|
| [`HANDOVER.md`](HANDOVER.md) | **Start here.** Current state, next actions, per-machine setup. Updated at the end of every session. |
| [`AGENTS.md`](AGENTS.md) | Step-by-step runbook for transcribing + diarizing, including the invariants that prevent past mistakes. |
| [`CLAUDE.md`](CLAUDE.md) | Working notes for AI agents: the standing rules and gotchas. |
| [`sessionlog/`](sessionlog/README.md) | One short log per working session (history behind `HANDOVER.md`). |
| [`channel/`](channel/README.md) | Which channel each video group comes from, plus pipeline status per group. |
| [`queries/`](queries/README.md) | Derived answers and summaries produced from the transcripts. |

## Setup

```bash
pip install -r requirements.txt        # openai, python-dotenv, yt-dlp (>= 2026.3)
ffmpeg -version && ffprobe -version    # must be on PATH
node --version                         # JS runtime for yt-dlp (deno also works)
```

- **ffmpeg** on Windows: `winget install --id Gyan.FFmpeg.Essentials -e`, then open a
  new shell.
- **node** (or deno) is needed because YouTube requires a JS runtime for yt-dlp to
  derive signatures; `transcribe.py` enables node explicitly.

Create a `.env` file in the repo root (git-ignored, never commit it):

```
OPENAI_API_KEY=sk-...          # Whisper transcription + index summaries
ASSEMBLYAI_API_KEY=...         # speaker diarization
```

## Pipeline Overview

```
1. Download audio       transcribe.py --download-only
2. Transcribe           transcribe.py --transcribe-only
3. Add speakers         diarize.py <transcript>.json        (required for every new video)
4. Build index          index.py build --group <name>
5. Query index          index.py query --group <name> "your question"
```

Steps 1 to 3 are the routine flow for new videos; step 3 is **not optional** (see
`CLAUDE.md`). Naming speakers (letters to real names) is a separate manual step.

## 1. Download & Transcribe

### Process predefined video groups

Video groups are defined in `VIDEO_GROUPS` inside `transcribe.py`. The channel each
group comes from is recorded in `channel/channels.json`.

```bash
# Process all groups (download + transcribe)
python transcribe.py

# Process a specific group
python transcribe.py --group group1

# Download only (no transcription)
python transcribe.py --group group1 --download-only

# Transcribe only (audio must already exist)
python transcribe.py --group group1 --transcribe-only
```

### Scrape and process a YouTube channel

```bash
# Scrape channel, download and transcribe first 50 videos
python transcribe.py --channel "https://youtube.com/@channelname"

# Custom group name + all videos
python transcribe.py --channel "https://youtube.com/@channelname" --channel-name mygroup --limit 0

# Download only (useful for large channels - download first, transcribe later)
python transcribe.py --channel "https://youtube.com/@channelname" --channel-name mygroup --limit 0 --download-only

# Then transcribe separately
python transcribe.py --channel "https://youtube.com/@channelname" --channel-name mygroup --limit 0 --transcribe-only
```

The scraped video list is cached in `channel/channel_cache/<group>.json` (git-ignored).
`--channel` reads that cache and only re-scrapes with `--force`. To pick up new uploads
without re-transcribing everything, refresh the cache alone as described in
`AGENTS.md` §4.

### Re-runs are safe

A video is skipped when its `.txt` and `.json` transcript already exist, **or** when a
transcript with the same YouTube video id exists under another filename (titles change
upstream, which changes the slug). Re-running is therefore cheap. `--force` disables both
checks and re-transcribes everything at cost, so never use it just to fetch new videos.

Downloads occasionally fail with a transient HTTP 403; simply re-run.

### Options

| Flag | Description |
|------|-------------|
| `--channel URL` | YouTube channel URL to scrape |
| `--channel-name NAME` | Group name for channel videos (default: derived from @handle) |
| `--limit N` | Max videos to process, 0 for all (default: 50) |
| `--language CODE` | Whisper language code (default: `da`) |
| `--cookies FILE` | Netscape-format cookies file for YouTube auth |
| `--proxy URL` | HTTP/SOCKS proxy for yt-dlp |
| `--download-only` | Only download audio, skip transcription |
| `--transcribe-only` | Only transcribe existing audio |
| `--force` | Re-scrape and re-process even if output already exists (costs money) |
| `--group NAME` | Process only a specific predefined group |

### Output structure

```
audio/<group>/*.mp3                    # Downloaded audio (committed; diarization input)
transcriptions/<group>/*.txt           # Plain text transcriptions
transcriptions/<group>/*.json          # JSON with metadata + timestamped segments
transcriptions/<group>/*.speakers.txt  # Transcript grouped by speaker
transcriptions/<group>/*.speakers.json # Segments annotated with speaker
channel/channel_cache/<group>.json     # Cached channel video list (git-ignored, per machine)
indexes/<group>.json                   # Summary index (see below)
```

Filenames may contain dots (`...5_mio._kr.json`). Never derive sibling names with
`Path.with_suffix()`; strip the literal `.json` instead. Both scripts do this.

## 1b. Add Speaker Labels

Whisper produces the best Danish *text* but cannot tell speakers apart.
`diarize.py` adds "who said what" by using **AssemblyAI for diarization only** —
it sends the audio to AssemblyAI, keeps only its speaker timeline ("who spoke
when"), discards AssemblyAI's own transcript, and assigns each existing Whisper
segment to the speaker whose turn overlaps it most. The merge is purely
timestamp-based, so it runs on an **already-transcribed** file with no
re-transcription cost, and writes **new** files — the originals are untouched.

```bash
# Requires ASSEMBLYAI_API_KEY in .env (or the environment)
python diarize.py transcriptions/group4/<name>.json

# Audio path is inferred as audio/<group>/<name>.mp3; override if needed
python diarize.py transcriptions/group4/<name>.json --audio path/to/audio.mp3
```

Outputs `<name>.speakers.txt` (conversation grouped as `Speaker A: ...`) and
`<name>.speakers.json` (every segment annotated with a `speaker` field). Files are
written atomically, so an interrupted run never leaves a truncated result that looks
done. Solo-speaker videos simply come back as one speaker; diarize them anyway.

Diarization only knows speakers as letters (A/B/C/D). Once you know who is who,
map the letters to real names — either in the same run, or **offline** on an
already-diarized file (no AssemblyAI call, no key, no cost):

```bash
# Offline relabel of an existing *.speakers.json (rewritten in place)
python diarize.py --relabel \
  --names "A=Olivia,B=Kasper,C=Morten,D=Andreas" \
  transcriptions/group4/<name>.speakers.json

# Or name speakers during a fresh diarization run
python diarize.py transcriptions/group4/<name>.json \
  --names "A=Olivia,B=Kasper,C=Morten,D=Andreas"
```

Named outputs read as real turns (`Olivia: ...`, `Morten: ...`), and the json
records the mapping under `diarization.speaker_names` so it stays auditable.

> Note: AssemblyAI's `speaker_labels` must support the audio language. Danish
> works via the multilingual model; if rejected, try `--language-code en`
> (diarization is acoustic, so speakers still group correctly).

## 2. Build Summary Index

Generate AI summaries, topics, people, and key points for each transcription.

```bash
# Build index for a group
python index.py build --group group1

# Re-index all files (ignore cache)
python index.py build --group group1 --force
```

Indexing is incremental: new transcriptions are added without re-processing existing
entries, and `*.speakers.json` siblings are ignored so each video is indexed once.
Rebuild the index after every batch of new videos so queries see them.

### Output

```
indexes/<group>.json   # Summary index with topics, people, key points per file
```

## 3. Browse the Index

```bash
python index.py list --group group1
```

Shows a readable overview of all indexed files with summaries, topics, and people.

## 4. Query (Minimal RAG)

Ask a question and get ranked relevant files with explanations.

```bash
python index.py query --group group1 "Hvordan sætter jeg den rigtige pris?"
```

Returns files ranked by relevance (high/medium/low) with a reason for each match. Use the returned file paths to feed full transcriptions into an AI conversation.

### Example output

```
Found 3 relevant file(s):

  1. [HIGH] 3 Tegn Paa At Dine Behandlinger Er For Billige
     Diskussion om prissætning og strategier til at hæve priserne.
     File: transcriptions/group1/3_tegn_paa_at_dine_behandlinger_er_for_billige.json
     Topics: prissætning, behandlinger, økonomisk frihed

  2. [HIGH] Mikkels klinik gik fra 50K til 120k
     Fokus på priselasticitet og ændringer i prissætning.
     File: transcriptions/group1/mikkels_klinik_gik_fra_50k_til_120k.json
     Topics: priselasticitet, klinikdrift, omsætning
```

## Full workflow example

```bash
# 1. Scrape and download a channel
python transcribe.py --channel "https://youtube.com/@marketingpod" \
  --channel-name marketing --limit 0 --download-only

# 2. Transcribe all downloaded audio
python transcribe.py --channel "https://youtube.com/@marketingpod" \
  --channel-name marketing --limit 0 --transcribe-only

# 3. Add speaker labels to every new transcript
for f in transcriptions/marketing/*.json; do
  case "$f" in *.speakers.json) ;; *) python diarize.py "$f" ;; esac
done

# 4. Build the summary index
python index.py build --group marketing

# 5. Search across all episodes
python index.py query --group marketing "Hvad er best practice for Meta Ads?"
```
