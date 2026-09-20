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
| group1 | curated | [@AlexanderFuldtBooket](https://www.youtube.com/@AlexanderFuldtBooket) | 26 | 26 | 13¹ | 0 |
| group2 | fixed | — (2 Vimeo workshops) | 2 | 2 | 2 | 0 |
| group3 | channel scrape | [@marketingpod](https://www.youtube.com/@marketingpod) | 337 | 337 | 337² | 0 |
| group4 | fixed | 1 Obsidian webinar | 1 | 1 | 1 | 0 |

*Status last updated: 2026-09-20 (1 new group1 + 3 new group3 videos, all transcribed and diarized).*

¹ group1's 13 pre-2026-07-29 monologues are intentionally not diarized; the 13 newer ones are.
² group3 is complete — all 337 transcribed and diarized; 262 have real speaker names, 75 are
letter-labeled (hosts appear as Halfdan / Kristian; the newest 30 were named 2026-09-20). Three "Mr. Beast" clips were missing until 2026-09-07: every dotted title used
to collapse to a truncated filename (`Mr. Beasts …` → `mr.json`), so four videos overwrote each
other and only one survived. Fixed in `transcribe.py` and re-transcribed.
³ 17 older testimonial/case clips on the group1 channel are deliberately left out of the
curated set (not counted as "new"). The channel listing also carries 1 group3 Short, which the
scraper filters out (338 listed → 337 tracked).

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
