#!/usr/bin/env python3
"""Maintain channel/video_meta.json: publish date + live title per transcribed video.

Transcripts only record when they were *transcribed*; retrieval ("one of the latest
episodes...") needs when they were *published*. This script walks every plain
transcript, keys it by source URL, and fetches upload_date / duration / live title via
yt-dlp for entries that are not already in the registry. Incremental and cheap: only
new videos hit the network. Re-run after every transcription batch.

    python video_meta.py            # fetch missing entries
    python video_meta.py --refresh  # re-fetch everything (titles drift upstream)
"""
import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import yt_dlp

SCRIPT_DIR = Path(__file__).parent.resolve()
TRANSCRIPTIONS_DIR = SCRIPT_DIR / "transcriptions"
META_PATH = SCRIPT_DIR / "channel" / "video_meta.json"
WORKERS = 1            # YouTube bot-checks bursts; 4 workers tripped it after ~230 videos
PAUSE_SECONDS = 2.0    # between metadata fetches

_YT_ID = re.compile(r"[?&]v=([\w-]{11})")


def video_key(url: str) -> str:
    """Stable key for a source URL: YouTube id when present, else the URL itself."""
    m = _YT_ID.search(url or "")
    return m.group(1) if m else (url or "")


def load_registry() -> dict:
    if META_PATH.exists():
        with open(META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"_comment": "Publish dates + live metadata per transcribed video, keyed by YouTube id "
                        "(or source URL for non-YouTube). Maintained by video_meta.py.",
            "updated_at": None, "videos": {}}


def save_registry(reg: dict):
    reg["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    reg["videos"] = dict(sorted(reg["videos"].items(), key=lambda kv: (kv[1].get("upload_date") or "", kv[0])))
    tmp = META_PATH.with_name(META_PATH.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(META_PATH)


def scan_transcripts() -> dict[str, dict]:
    """key -> {group, transcript (stem), url, title (as transcribed), duration_seconds}."""
    found = {}
    for group_dir in sorted(TRANSCRIPTIONS_DIR.iterdir()):
        if not group_dir.is_dir():
            continue
        for jf in sorted(group_dir.glob("*.json")):
            if jf.name.endswith(".speakers.json"):
                continue
            with open(jf, "r", encoding="utf-8") as f:
                d = json.load(f)
            url = d.get("url", "")
            if not url:
                continue
            found[video_key(url)] = {
                "group": group_dir.name,
                "transcript": jf.name[:-len(".json")],
                "url": url,
                "title": d.get("title", jf.stem),
                "duration_seconds": d.get("duration_seconds"),
            }
    return found


_BASE_OPTS = {"quiet": True, "no_warnings": True, "skip_download": True,
              "nocheckcertificate": True, "js_runtimes": {"deno": {}, "node": {}}}
# Fallback when YouTube answers "Sign in to confirm you're not a bot" (IP-level, after a
# burst of requests): the android player client, skipping the watch page, still returns
# upload_date/title/duration. Formats are irrelevant here, so ignore format errors.
_FALLBACK_OPTS = {**_BASE_OPTS, "ignore_no_formats_error": True,
                  "extractor_args": {"youtube": {"player_client": ["android"], "player_skip": ["webpage"]}}}


def fetch_one(url: str) -> dict:
    try:
        with yt_dlp.YoutubeDL(_BASE_OPTS) as y:
            info = y.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as e:
        if "not a bot" not in str(e):
            raise
        with yt_dlp.YoutubeDL(_FALLBACK_OPTS) as y:
            info = y.extract_info(url, download=False)
    time.sleep(PAUSE_SECONDS)
    ud = info.get("upload_date")  # YYYYMMDD
    return {
        "upload_date": f"{ud[:4]}-{ud[4:6]}-{ud[6:]}" if ud else None,
        "live_title": info.get("title"),
        "duration_seconds": info.get("duration"),
        "uploader": info.get("uploader") or info.get("channel"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="re-fetch every entry, not only missing ones")
    args = ap.parse_args()

    reg = load_registry()
    videos = reg["videos"]
    found = scan_transcripts()

    # Register / update the local facts for every transcript.
    for key, local in found.items():
        entry = videos.setdefault(key, {})
        entry.update({k: local[k] for k in ("group", "transcript", "url", "title")})
        entry.setdefault("duration_seconds", local["duration_seconds"])
        entry.setdefault("upload_date", None)
        entry.setdefault("live_title", None)
        entry.setdefault("fetch_error", None)

    todo = [k for k, e in videos.items() if args.refresh or not e.get("upload_date")]
    print(f"{len(found)} transcripts, {len(videos)} registered, {len(todo)} to fetch")
    save_registry(reg)

    ok = err = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(fetch_one, videos[k]["url"]): k for k in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            k = futs[fut]
            try:
                videos[k].update(fut.result())
                videos[k]["fetch_error"] = None
                ok += 1
                print(f"  [{i}/{len(todo)}] {videos[k]['upload_date']}  {videos[k]['title'][:70]}")
            except Exception as e:  # noqa: BLE001 - record and continue
                videos[k]["fetch_error"] = str(e).splitlines()[0][:200]
                err += 1
                print(f"  [{i}/{len(todo)}] ERROR {videos[k]['title'][:60]}: {videos[k]['fetch_error'][:80]}")
            if i % 25 == 0:
                save_registry(reg)
    save_registry(reg)

    missing = [e["title"] for e in videos.values() if not e.get("upload_date")]
    print(f"\nDONE: fetched {ok}, errors {err}, still without upload_date: {len(missing)}")
    for t in missing[:10]:
        print(f"  - {t}")
    print(f"Saved: {META_PATH.relative_to(SCRIPT_DIR)}")
    return 0 if err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
