# 2026-09-20 — blackbox-silent (stationary)

**Goal:** sync with GitHub, find out which tracked-channel videos are new, verify the
local toolchain, and set up handover/session-log files for working across machines.

**Done:**
- `git pull` fast-forwarded `459e5c7 → f949423` (16 commits, 961 files, mostly the
  group1/group3 diarization output + `AGENTS.md` runbook + filename-truncation fix).
- Scraped both YouTube channels with yt-dlp `extract_flat` (no downloads) and diffed
  video ids against the transcripts on disk. Result: **3 new group3 episodes + 1 new
  group1 video**, nothing removed upstream, hardcoded group1 list matches disk. Details
  and ids in `HANDOVER.md`. Refreshed `channel/channel_cache/group{1,3}.json` locally.
- Toolchain check: yt-dlp 2026.08.19 (latest on PyPI), openai 3.16.2, node 24.0.1
  present. **ffmpeg/ffprobe were not on PATH**; I installed the `Gyan.FFmpeg` Full build,
  the user separately installed `Gyan.FFmpeg.Essentials` 9.0.1. Both ended up on the user
  PATH, so I uninstalled the Full build again; Essentials 9.0.1 is now the only ffmpeg on
  PATH (libmp3lame confirmed). An older unmanaged copy lives under the TV-Syd project dir.
  Installed `requirements.txt` into both Python 3.12 installs (Store `python` and
  python.org `py -3.12`) because the runbook uses bare `python`.
- Created `HANDOVER.md`, `sessionlog/README.md`, this file; linked them from `CLAUDE.md`.

**Decisions / gotchas:**
- Did **not** download or transcribe anything; the diff was wanted first.
- The 17 extra untracked group1 videos in the scrape are the known excluded testimonial
  clips, not new content. One (`flMsd1NCtyA`) now returns "This video is not available".
- `.env` was missing at first; the user added it later in the session. Both keys validated
  with cheap GETs (AssemblyAI 200, OpenAI 200). `.env` confirmed git-ignored.
- `channel/channel_cache/` did not exist after the pull (git-ignored); expected on a
  fresh machine, the scrape recreates it.

**Left for next time:** the 6-step "Next actions" list in `HANDOVER.md` (add group1 video
to `VIDEO_GROUPS`, transcribe 4 videos, diarize, verify, update channel status, commit).

**Commits:** docs-only commit adding HANDOVER.md + sessionlog/ + CLAUDE.md pointer.
