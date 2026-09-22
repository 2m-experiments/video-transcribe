# 2026-09-22 — asus-copilot

**Goal:** Check the tracked channels for new uploads and bring them in.

**Done:**
- Scraped @AlexanderFuldtBooket and @marketingpod (`extract_flat`) and diffed by video id
  against all 363 transcripts. There were 2 new videos, both uploaded 2026-09-21:
  group1 `3OQc7u3pri8` (15 min) and group3 `Mnl3r5vXLFQ` (54 min).
- Added `3OQc7u3pri8` to `VIDEO_GROUPS["group1"]`, refreshed the group3 cache
  (`scrape_channel_videos` + `save_channel_cache`) and transcribed both without `--force`.
- Diarized both: group1 has 1 speaker, group3 has 2 (A/B, not yet named).
- Ran `video_meta.py`, `index.py build` (group1 → 27, group3 → 338) and `bundle.py`:
  10 files, 372 episodes, 5.56M chars of upload files. group5 is bundled for the first time.
- Updated `channel/channels.json`, `channel/README.md` and `HANDOVER.md`. Also removed a
  stale leftover block (the 2026-09-20 pre-run plan) from HANDOVER.

**Decisions / gotchas:**
- Windows console is cp1252 on this machine, so emoji titles crash `print`. Run
  scripts with `PYTHONIOENCODING=utf-8`.
- The group1 listing shows 18 old untracked clips instead of the 17 counted before. They're
  all old testimonial clips, so they were left out.
- `YKgkPMoTbB0` (33 s) is the known group3 Short, which the scraper filters out.

**Left for next time:** Re-upload the changed bundle files to the Claude Project, and
optionally name the speakers in `Mnl3r5vXLFQ`.

**Commits:** see git log for 2026-09-22.

## Later: speaker naming made automatic

- The user asked why the new episode still said Speaker A/B. Naming had been treated as an
  on-request step. The user's decision: the project exists to automate, so naming is part of
  "done". Updated `CLAUDE.md` and `AGENTS.md` §3.
- New `channel/speakers.json`: known speakers per group with aliases and Whisper misspellings
  ("Tinho" → Tini/Tinju/Tino/Signe/Sinju…, "TikTok-boy"; "Halfdan" → Halvdan/Halvsten/Alfden…).
- Named `Mnl3r5vXLFQ` (A=Halfdan, B=Kristian) and the new group1 video (A=Alexander, from
  "Mit navn er Alexander").
- Went through all 75 unnamed group3 episodes. Built an evidence file per episode (opening
  turns + name-cue turns), and 5 parallel reviewers proposed maps (22 high, 14 medium, 39
  none). A script then flagged any letter mapped to a host that says that host's name
  (02, 30, 53); the lines turned out to be diarization bleed, so the majority maps stand.
  Rejected 03 ("halvdelen" is ordinary Danish, not Halfdan). Corrected the guest in 60 to
  Niklas (same interview as 37). Applied 35 → group3 is 298/338 named.
- Traps found: some episodes have Kristian as A and Halfdan as B (14, 31, and solo 20
  is Kristian), so "the opener is Halfdan" isn't a safe rule. Several short clips have both
  hosts under one letter (41, 49, 55); those stay unnamed.
