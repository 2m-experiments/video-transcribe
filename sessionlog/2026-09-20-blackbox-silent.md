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
- **Ran the update** after the user added `.env`: added `3KGvqLOQ8nw` to `VIDEO_GROUPS["group1"]`,
  `transcribe.py --group group1` (1 new, 16 min), `transcribe.py --channel @marketingpod` (3 new,
  ~94 min, one needed 2 Whisper chunks), then `diarize.py` on all 4. Speakers found: group1 video
  1 (A); group3: Only Halfdan 1 (A), Masterclass 3 (A,B,C), Performance Max 3 (A,B,C).
  Verify snippet passes: group1 26/13, group3 337/337, no orphans.
- Fixed `transcribe.py` twice, both tested with no-op re-runs (337 + 26 skipped, 0 failed, ~1 s):
  delay-after-skip idle bug, and an id-aware skip check (see gotchas).
- Updated `channel/channels.json` + `channel/README.md` status to 2026-09-20.
- **Follow-up (same day):** rebuilt `indexes/group1.json` (11 → 26) and `indexes/group3.json`
  (288 → 337) with `index.py build`. First patched `index.py` to skip `*.speakers.json`, which
  the `*.json` glob would otherwise have summarized as 337 extra "videos". Deleted two
  byte-identical duplicate mp3s in `audio/group3/` (checked size, duration and md5):
  the retitled `TVexnIlT-ps` copy and `…_og_vandt..mp3` (the code expects the single-dot
  name). Rewrote `README.md`: requirements.txt, both keys, node runtime, ffmpeg install,
  cache moved to `channel/channel_cache/`, diarization required, id-based skip, `--proxy`,
  and a "Where to look" table pointing at HANDOVER/AGENTS/CLAUDE/sessionlog/channel/queries.

**Decisions / gotchas:**
- Did **not** download or transcribe anything; the diff was wanted first.
- The 17 extra untracked group1 videos in the scrape are the known excluded testimonial
  clips, not new content. One (`flMsd1NCtyA`) now returns "This video is not available".
- `.env` was missing at first; the user added it later in the session. Both keys validated
  with cheap GETs (AssemblyAI 200, OpenAI 200). `.env` confirmed git-ignored.
- `channel/channel_cache/` did not exist after the pull (git-ignored); expected on a
  fresh machine, the scrape recreates it.

- The group3 channel run kept running ~15 min after the last transcript was saved: the
  anti-blocking sleep (5–15 s) ran after every *skipped* video too, i.e. ~334 × 10 s of idling.
  Killed it (all outputs were already written and verified) and fixed the loop.
- **Title drift hit for real**: video 19 in the channel list, `TVexnIlT-ps`, was transcribed
  2026-07-29 as "HVEJSEL ER TILBAGE…" and has been retitled upstream. The name-based skip
  check missed it and the run tried to re-transcribe it (it failed only because that shell had
  no ffmpeg on PATH; the earlier killed run may have burned one partial Whisper call on it).
  `transcribe.py` now scans the group's transcripts once for `url` → video id and skips by id.
- The Bash tool's shell predates the ffmpeg install; runs there need the winget bin dir
  prepended to `PATH`. New terminals are fine.
- **YouTube bot check**: fetching per-video metadata with 4 threads tripped "Sign in to
  confirm you're not a bot" after ~230 videos (IP-level, persisted for the rest of the
  session). Fix in `video_meta.py`: 1 worker, 2 s pause, and a fallback to the android
  player client with `player_skip=webpage` + `ignore_no_formats_error`, which still returns
  `upload_date` while blocked (cross-checked against a known date). Group2 (Vimeo player
  URLs) and group4 (webinar page) give no upload_date; bundle falls back to transcription date.
- `python -` heredocs that `print()` non-cp1252 characters crash on this Windows console
  (`UnicodeEncodeError`); the file writes before the print still land. Use `git diff` to check.

- **Retrieval bundle (same day, after the user asked how to query from a phone by voice):**
  - `video_meta.py` (new): keeps `channel/video_meta.json`, publish date + live title per
    transcript, keyed by YouTube id. Needed because transcripts only carry `transcribed_at`
    and "seneste episoder" questions need publish dates. Incremental.
  - `bundle.py` (new): writes `bundle/` = `00_catalog.md` (all episodes newest-first with
    date/speakers/summary/topics/URL + which file holds the transcript) and ~1 MB
    chronological chunk files per source, each episode section = metadata + full
    speaker-labelled transcript. `manifest.json` (sha256) shows which files changed so only
    those need re-uploading to the ChatGPT/Claude Project. `bundle/README.md` has phone
    steps + suggested project instructions.
  - Created the missing `indexes/group2.json` + `group4.json` (3 files) so every episode has a
    summary in the bundle.
  - Docs: AGENTS.md §2b (dates → index → bundle after every batch), CLAUDE.md flow, README.

- **Speaker naming (same day):** renamed "Christian" → "Kristian" in all 217 named group3
  files (offline relabel; `diarize.py --relabel` now composes with an earlier name map so the
  letter → name audit trail survives). Named the newest 30 unnamed episodes: built a
  per-episode evidence packet (opening turns + every turn with a name cue), had gpt-4o
  propose letter → name maps, then **verified every map** against who-addresses-whom cue
  counts and by reading the cue turns. gpt-4o labelled all 30 "high" but had the two hosts
  swapped in 6 (self-introductions like "jeg er Christian Tinho" and guests talking *about*
  Christian fooled it). 29 applied, 1 left unnamed (2026-06-29 solo, no clue). group3 now
  262/337 named. Bundle rebuilt.

- **Claude Project upload:** the full 12 MB bundle was rejected; 11 files (6.65M chars) showed
  108% capacity, so the cap is ~6.1M chars. `bundle.py` got a `--budget` (default 4.9M chars of
  full text, strict newest-first date cutoff) + summary-only digest for the rest. Final upload:
  8 files, 5.56M chars, 91%. Project instructions in `bundle/README.md` (no forced title+date
  citing, per user).

**Left for next time:** nothing pending; see `HANDOVER.md` "Next actions" (diff for new
uploads after 2026-09-17, optional speaker naming, optional index rebuild).

**Commits:** `c6fd6e1` handover docs, `1b41650`/`c01d5da`/`5e4c7e2` handover updates, then one
commit with the 4 new videos + transcribe.py fixes + status/handover updates (see git log).
