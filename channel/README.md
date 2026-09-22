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
| group1 | curated | [@AlexanderFuldtBooket](https://www.youtube.com/@AlexanderFuldtBooket) | 27 | 27 | 14¹ | 0 |
| group2 | fixed | — (2 Vimeo workshops) | 2 | 2 | 2 | 0 |
| group3 | channel scrape | [@marketingpod](https://www.youtube.com/@marketingpod) | 338 | 338 | 338² | 0 |
| group4 | fixed | 1 Obsidian webinar | 1 | 1 | 1 | 0 |
| group5 | fixed | — (4 MP4s on 2marketing.nu) | 4 | 4 | 4 | — |

*Status last updated: 2026-09-22 (1 new group1 + 1 new group3 video, both transcribed and diarized; group5 added 2026-09-10).*

¹ group1's 13 pre-2026-07-29 monologues are intentionally not diarized; the 14 newer ones are.
² group3 is complete — all 338 transcribed and diarized; 298 have real speaker names, 40 are
letter-labelled because the transcript doesn't prove who is speaking (hosts appear as Halfdan / Kristian; 36 more were named on 2026-09-22 using [`speakers.json`](speakers.json)). Three "Mr. Beast" clips were missing until 2026-09-07: every dotted title used
to collapse to a truncated filename (`Mr. Beasts …` → `mr.json`), so four videos overwrote each
other and only one survived. Fixed in `transcribe.py` and re-transcribed.
³ The older testimonial/case clips (17 counted 2026-09-20; the listing showed 18 on 2026-09-22) on the group1 channel are deliberately left out of the
curated set (not counted as "new"). The channel listing also carries 1 group3 Short, which the
scraper filters out (339 listed → 338 tracked).

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
