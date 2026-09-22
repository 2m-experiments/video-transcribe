# HANDOVER.md — current state for whoever picks this up next

Living document: **overwrite** the "Current state" and "Next actions" sections at the end
of every session (on any machine), then commit and push so the other machine sees it.
Per-session narrative goes in [`sessionlog/`](sessionlog/README.md), not here.

Last updated: **2026-09-22** on **asus-copilot**.

---

## Current state

- **All groups complete as of 2026-09-22**: group1 27/27 transcribed (14 diarized, 13 legacy
  monologues intentionally not), group3 338/338 transcribed and diarized, group2, group4 and
  group5 unchanged. The verify snippet in `AGENTS.md` passes and there are no orphans.
- 2026-09-22 channel diff found 2 new uploads (both 2026-09-21), now transcribed, diarized,
  indexed and bundled:
  - group1 `3OQc7u3pri8` (15 min) "Lær hvorfor din klinik har ramt et loft - og hvordan du bryder det", 1 speaker
  - group3 `Mnl3r5vXLFQ` (54 min) "Tester AI agenter til marketing: Claude vs Manus vs Grok Bot vs OpenClaw", speakers A/B **not yet named**
- group1 channel listing now shows 45 videos: 27 tracked + 18 old untracked testimonial
  clips (17 were counted on 2026-09-20; the extra one wasn't identified and may be
  `flMsd1NCtyA` playable again). All old, deliberately outside the curated set.
- **Phone/voice retrieval** is the Claude Project "VideoTranscribe" (shared with the 2ai org):
  `bundle/` is 5.56M chars (~91% of the ~6.1M cap). `bundle.py` keeps full transcripts
  within a 4.9M-char budget, newest first (now 2025-10-30 →), older episodes are
  summary-only in `zz_digest_*`. **Do not raise the budget; never upload
  `README.md`/`manifest.json`.** After a batch, delete the files `bundle.py` lists as
  changed/removed from the project and upload their replacements.
- The 2026-09-22 rebuild changed 8 upload files (every part shifts its date window) and
  bundled group5 for the first time (`group5_part01_of01_…`). **The Claude Project has
  not been re-uploaded yet** (see Next actions).
- Speaker names: 262 of 338 Marketingpod episodes are named. Hosts are **Halfdan** and
  **Kristian**; guests use the first name from the title. Naming is offline
  (`diarize.py --relabel`) and the bundle picks it up on rebuild.
- The scrape caches under `channel/channel_cache/` are per-machine and git-ignored; a
  fresh machine simply re-scrapes (refresh per `AGENTS.md` §4).

## Next actions

1. Re-upload the bundle to the Claude Project: remove `fuldt-booket_part01_of01_2026-01-15_2026-09-15.md`,
   the four `marketingpod_part0?_of04_…` files, `zz_digest_part01_of01_2024-05-09_2025-10-20.md`
   and the old `00_catalog.md`; upload `00_catalog.md`, `fuldt-booket_part01_of01_2026-01-15_2026-09-21.md`,
   `group5_part01_of01_2026-04-30_2026-08-18.md`, the four new `marketingpod_part0?_of04_…` files and
   `zz_digest_part01_of01_2024-05-09_2025-10-27.md`. `workshops_…` is unchanged.
2. Optional: name speakers A/B in `Mnl3r5vXLFQ` (likely Halfdan/Kristian; check the text
   first), then `bundle.py` again.
3. Test the project by voice on the phone. If retrieval is weak, the fallback is a small
   remote MCP server over the index.
4. Next channel diff: look for uploads after 2026-09-21.
5. Optional: name the remaining 76 letter-labelled group3 episodes.

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
