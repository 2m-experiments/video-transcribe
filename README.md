# Video Transcribe & Index

Download videos from YouTube/Vimeo, transcribe them with OpenAI Whisper, and build a searchable summary index.

## Setup

```bash
pip install openai python-dotenv yt-dlp
```

Create a `.env` file:

```
OPENAI_API_KEY=sk-...
```

Requires `ffmpeg` and `ffprobe` on PATH.

## Pipeline Overview

```
1. Download audio    transcribe.py --download-only
2. Transcribe        transcribe.py --transcribe-only
3. Build index       index.py build --group <name>
4. Query index       index.py query --group <name> "your question"
```

## 1. Download & Transcribe

### Process predefined video groups

Video groups are defined in `VIDEO_GROUPS` inside `transcribe.py`.

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

### Options

| Flag | Description |
|------|-------------|
| `--channel URL` | YouTube channel URL to scrape |
| `--channel-name NAME` | Group name for channel videos (default: derived from @handle) |
| `--limit N` | Max videos to process, 0 for all (default: 50) |
| `--language CODE` | Whisper language code (default: `da`) |
| `--cookies FILE` | Netscape-format cookies file for YouTube auth |
| `--download-only` | Only download audio, skip transcription |
| `--transcribe-only` | Only transcribe existing audio |
| `--force` | Re-process even if output already exists |
| `--group NAME` | Process only a specific predefined group |

### Output structure

```
audio/<group>/*.mp3           # Downloaded audio files
transcriptions/<group>/*.txt  # Plain text transcriptions
transcriptions/<group>/*.json # JSON with metadata + timestamped segments
channel_cache/<group>.json    # Cached channel video list (for resume)
```

## 2. Build Summary Index

Generate AI summaries, topics, people, and key points for each transcription.

```bash
# Build index for a group
python index.py build --group group1

# Re-index all files (ignore cache)
python index.py build --group group1 --force
```

Indexing is incremental - new transcriptions are added without re-processing existing entries.

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

# 3. Build the summary index
python index.py build --group marketing

# 4. Search across all episodes
python index.py query --group marketing "Hvad er best practice for Meta Ads?"
```
