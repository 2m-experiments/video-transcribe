# HANDOVER.md — current state for whoever picks this up next

Living document: **overwrite** the "Current state" and "Next actions" sections at the end
of every session (on any machine), then commit and push so the other machine sees it.
Per-session narrative goes in [`sessionlog/`](sessionlog/README.md), not here.

Last updated: **2026-09-24** on **blackbox-silent**.

---

## Current state

- **All groups complete as of 2026-09-24**: group1 27/27 transcribed (14 diarized, 13 legacy
  monologues intentionally not), group3 339/339 transcribed and diarized (299 named), group2,
  group4 and group5 unchanged. The verify snippet in `AGENTS.md` passes and there are no orphans.
- 2026-09-24 channel diff found 1 new upload (2026-09-24), now transcribed, diarized, named,
  indexed and bundled:
  - group3 `Pr_IHodzhWw` (7 min) "Only Halfdan: Metas algoritme forklaret", 1 speaker, named
    Halfdan from the series title (same basis as the earlier "Only Halfdan" episode).
- Earlier, 2026-09-22 brought group1 `3OQc7u3pri8` (named Alexander) and group3 `Mnl3r5vXLFQ`
  (named Halfdan/Kristian).
- group1 channel listing showed 44 videos on 2026-09-24 (27 tracked + 17 old untracked
  testimonial clips; it showed 45/18 on 2026-09-22). All old, deliberately outside the curated set.
- **Phone/voice retrieval** is the Claude Project "VideoTranscribe" (shared with the 2ai org):
  `bundle/` is 5.57M chars (~91% of the ~6.1M cap). `bundle.py` keeps full transcripts
  within a 4.9M-char budget, newest first (now 2025-10-30 →), older episodes are
  summary-only in `zz_digest_*`. **Do not raise the budget; never upload
  `README.md`/`manifest.json`.** After a batch, delete the files `bundle.py` lists as
  changed/removed from the project and upload their replacements.
- **The Claude Project has not been re-uploaded since before 2026-09-22**, so it is now two
  batches behind (see Next actions). After `bundle.py`, `git status` lists every bundle part
  as modified, but that is LF/CRLF noise; the manifest hashes show which files really changed.
- **Speaker naming is part of the standard flow** (user decision 2026-09-22). Known speakers,
  nicknames and Whisper misspellings are in `channel/speakers.json`. 299 of 339 Marketingpod
  episodes are named; the other 40 stay letter-labelled for lack of evidence. The 13 older
  diarized group1 videos are not named yet.
- The scrape caches under `channel/channel_cache/` are per-machine and git-ignored; a
  fresh machine simply re-scrapes (refresh per `AGENTS.md` §4). blackbox-silent's group3
  cache was refreshed 2026-09-24.

## Next actions

1. Re-upload the bundle to the Claude Project. Remove from the project:
   `fuldt-booket_part01_of01_2026-01-15_2026-09-15.md`, every `marketingpod_part0?_of04_…`
   file, `zz_digest_part01_of01_2024-05-09_2025-10-20.md` and the old `00_catalog.md`.
   Upload from `bundle/`: `00_catalog.md`, `fuldt-booket_part01_of01_2026-01-15_2026-09-21.md`,
   `group5_part01_of01_2026-04-30_2026-08-18.md`, `marketingpod_part01_of04_2025-10-30_2026-01-19.md`,
   `marketingpod_part02_of04_2026-01-22_2026-04-16.md`, `marketingpod_part03_of04_2026-04-20_2026-07-09.md`,
   `marketingpod_part04_of04_2026-07-13_2026-09-24.md` and
   `zz_digest_part01_of01_2024-05-09_2025-10-27.md`. `workshops_…` is unchanged.
2. Optional: name the other 13 diarized group1 solo videos (Alexander). Only 2 of them
   self-introduce, so check each one first.
3. Test the project by voice on the phone. If retrieval is weak, the fallback is a small
   remote MCP server over the index.
4. Next channel diff: look for uploads after 2026-09-24.
5. The remaining 40 unnamed group3 episodes would need voice matching (not just text) to go further.

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
| blackbox-silent | stationary | 2026-09-24: ran the full flow OK (Python 3.12.10, yt-dlp 2026.08.19, ffmpeg 9.0.1, node 24.0.1, keys valid). Python 3.12 ×2 (Store `python` + python.org `py -3.12`, both have deps), yt-dlp 2026.08.19, ffmpeg 9.0.1 Essentials via winget (only copy on PATH; an older 2025-11-24 git build sits unused under the TV-Syd JV Player Download project dir), node 24.0.1, no deno. `.env` present, both keys validated 2026-09-20. **Ready to run.** |
| laptop | mobile | Unknown. Run the checklist and fill this row in. |
| asus-copilot | — | 2026-09-22: Python 3.14.6, yt-dlp 2026.08.19, ffmpeg 8.1 full build (gyan.dev), node 26.4.0. `.env` present with both keys; ran the full pipeline OK. Console needs `PYTHONIOENCODING=utf-8` (cp1252 chokes on emoji titles). |

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
