# Channels

Source channels behind each video group, plus pipeline status. This folder is the
single place that records **where each group's videos come from** — that mapping is
otherwise not stored in the code (`transcribe.py` only holds the video lists).

- **`channels.json`** — machine-readable registry (details + status per group).
- **`channel_cache/`** — cached scraped video lists (`<group>.json`), git-ignored and
  regenerable. Written/read by `transcribe.py` (`CHANNEL_CACHE_DIR`).

## Groups at a glance

| Group | Source | Channel | Tracked | Transcribed | Diarized | New available |
|-------|--------|---------|--------:|------------:|---------:|--------------:|
| group1 | curated | [@AlexanderFuldtBooket](https://www.youtube.com/@AlexanderFuldtBooket) | 19 | 19 | 0¹ | 0 |
| group2 | fixed | — (2 Vimeo workshops) | 2 | 2 | 2 | 0 |
| group3 | channel scrape | [@marketingpod](https://www.youtube.com/@marketingpod) | 321 | 322 | 288³ | 0 |
| group4 | fixed | 1 Obsidian webinar | 1 | 1 | 1 | 0 |

*Status last updated: 2026-07-29 (after transcribing 6 new group1 + 34 new group3 videos).*

¹ group1 is solo monologues — diarization is intentionally skipped (single speaker).
² ~18 older testimonial/case clips on the channel are deliberately left out of the
curated set (not counted as "new").
³ The 34 newest group3 transcripts are not yet diarized/speaker-named (transcript-only).

## Source types

- **`channel_scrape`** — the whole channel is scraped via yt-dlp and cached in
  `channel_cache/<group>.json`. Re-run `transcribe.py --channel <url> --channel-name
  <group>` to refresh.
- **`curated`** — a hand-picked subset of a channel, hardcoded in `VIDEO_GROUPS`
  inside `transcribe.py`. New channel uploads are **not** auto-included; add them to
  the list manually.
- **`fixed`** — standalone videos/webinars with no parent channel to track.

## Checking for new videos

For a scraped or curated group, scrape the channel's `/videos` page with yt-dlp
(`extract_flat`) and diff the video IDs against the cached / hardcoded set:

```bash
# group3 example — refresh cache + download + transcribe only the new episodes
python ../transcribe.py --channel "https://www.youtube.com/@marketingpod" \
  --channel-name group3 --limit 0
```

The scrape cache does **not** store the channel URL itself — it lives here in
`channels.json`. (Historically it could be recovered from any cached video's
`uploader_url` via yt-dlp.)
