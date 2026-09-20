# HANDOVER.md — current state for whoever picks this up next

Living document: **overwrite** the "Current state" and "Next actions" sections at the end
of every session (on any machine), then commit and push so the other machine sees it.
Per-session narrative goes in [`sessionlog/`](sessionlog/README.md), not here.

Last updated: **2026-09-20** on **blackbox-silent** (stationary).

---

## Current state

- Repo is in sync with GitHub `main` at `f949423` plus this session's docs commit.
- All four groups are fully transcribed **and** diarized as of 2026-09-07 (see
  [`channel/README.md`](channel/README.md)). Nothing is half-done.
- **New videos are waiting** (found 2026-09-20, not yet downloaded or transcribed):

  | Group | Video id | Uploaded | Length | Title |
  |---|---|---|---|---|
  | group3 | `AqzbeUR3Ajk` | 2026-09-10 | 19 min | Performance Max er døende - men hvad gør Google nu? |
  | group3 | `xTsPejw87uo` | 2026-09-14 | 65 min | Performance-branding Masterclass med Jacob Holst Mouritzen |
  | group3 | `-LjhtfdBj-k` | 2026-09-17 | 10 min | Only Halfdan: Metas læringsfase forklaret på 12 minutter |
  | group1 | `3KGvqLOQ8nw` | 2026-09-15 | 16 min | De 3 ting, der tog en behandler fra 0 til 3 klinikker |

  The 17 other untracked group1 videos are the old testimonial clips that are
  deliberately excluded from the curated set (one of them, `flMsd1NCtyA`, is no longer
  playable on YouTube). Ignore them.
- The group3 scrape cache (`channel/channel_cache/group3.json`, git-ignored) was
  refreshed on 2026-09-20 on blackbox-silent and lists all 337 videos. **On another
  machine that cache does not exist**; `transcribe.py --channel …` will simply re-scrape,
  which is fine (see `AGENTS.md` §4 for the cache-only refresh).

## Next actions

1. Add the group1 video `3KGvqLOQ8nw` to `VIDEO_GROUPS["group1"]` in `transcribe.py`.
2. `python transcribe.py --group group1` and
   `python transcribe.py --channel "https://www.youtube.com/@marketingpod" --channel-name group3 --limit 0`
   (no `--force`).
3. Diarize the 4 new transcripts with `diarize.py` (mandatory, see `CLAUDE.md`).
4. Run the verify snippet at the bottom of `AGENTS.md`.
5. Update the status tables in `channel/channels.json` and `channel/README.md`
   (`last_checked`, tracked/transcribed/diarized counts).
6. Commit the outputs, update this file, write a session log, push.

---

## Machine setup checklist

Run this on any machine before starting; every line should print a version, not an error.

```bash
python -c "import openai, dotenv, yt_dlp; print('deps ok', yt_dlp.version.__version__)"
ffmpeg -version | head -1 && ffprobe -version | head -1
node --version          # JS runtime for yt-dlp signature solving (deno also works)
git status -sb          # must be in sync with origin/main before you start
ls .env                 # OPENAI_API_KEY + ASSEMBLYAI_API_KEY; never committed
```

Fixes if something is missing:

- Python deps: `python -m pip install -r requirements.txt`. yt-dlp must be
  **>= 2026.3.0** (older versions break on YouTube).
- ffmpeg: `winget install --id Gyan.FFmpeg.Essentials -e` (Windows). Restart the shell
  afterwards. Essentials is enough (has ffprobe + libmp3lame); do not also install the
  `Gyan.FFmpeg` Full build, both land on PATH and the first one wins silently.
- node: `winget install OpenJS.NodeJS.LTS` (Windows). The repo enables the node runtime
  for yt-dlp in `transcribe.py`; without any JS runtime yt-dlp falls back to deprecated
  clients and downloads become flaky.
- `.env`: copy it from the other machine or a password manager. It is git-ignored and
  contains the two API keys. Validate the AssemblyAI key with the curl in `AGENTS.md` §0.

### Per-machine notes

| Machine | Role | State on 2026-09-20 |
|---|---|---|
| blackbox-silent | stationary | Python 3.12 ×2 (Store `python` + python.org `py -3.12`, both have deps), yt-dlp 2026.08.19, ffmpeg 9.0.1 Essentials via winget (only copy on PATH; an older 2025-11-24 git build sits unused under the TV-Syd JV Player Download project dir), node 24.0.1, no deno. `.env` present, both keys validated 2026-09-20. **Ready to run.** |
| laptop | mobile | Unknown. Run the checklist and fill this row in. |

Note on Windows Python: `python` on PATH may be the Microsoft Store build while
`py -3.12` is the python.org build. They have separate site-packages. Install requirements
into whichever one you use, or both.

---

## Working across machines

1. **Start of session**: `git pull`, read this file, run the setup checklist.
2. **End of session**: update "Current state" + "Next actions" above, add a file under
   `sessionlog/`, commit, push. Never leave a session with unpushed transcript output.
3. `audio/` mp3s are committed (they are the diarization input), so a pull can be large.
   `channel/channel_cache/` and `.env` are git-ignored and per-machine.
4. If a run is interrupted mid-batch, the skip-checks in `transcribe.py`/`diarize.py`
   make re-running safe; just say in the session log which videos were in flight.
