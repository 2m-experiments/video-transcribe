#!/usr/bin/env python3
"""Download YouTube/Vimeo videos and transcribe them using OpenAI Whisper API."""

import argparse
import json
import math
import os
import random
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import yt_dlp
from dotenv import load_dotenv
from openai import OpenAI

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
AUDIO_DIR = SCRIPT_DIR / "audio"
CHUNKS_DIR = SCRIPT_DIR / "chunks"
TRANSCRIPTIONS_DIR = SCRIPT_DIR / "transcriptions"
CHANNEL_CACHE_DIR = SCRIPT_DIR / "channel_cache"

# ── Constants ────────────────────────────────────────────────────────────────
MAX_CHUNK_SIZE_MB = 20  # Whisper API limit is 25MB, generous margin for re-encoding variance
CHUNK_OVERLAP_SECONDS = 30
API_DELAY_SECONDS = 2
DOWNLOAD_DELAY_MIN = 5   # seconds between video downloads (anti-blocking)
DOWNLOAD_DELAY_MAX = 15
DEFAULT_CHANNEL_LIMIT = 50

DANISH_PROMPT = (
    "Dette er en dansk podcast om klinikdrift, behandlere, fysioterapi, "
    "bookinger, timepriser og markedsføring af behandlerklinikker."
)

# ── Video Groups ─────────────────────────────────────────────────────────────
VIDEO_GROUPS = {
    "group1": [
        {"url": "https://www.youtube.com/watch?v=qQjC1SDvDzg", "title": "Fuldt Booket Men Ingen Penge"},
        {"url": "https://www.youtube.com/watch?v=TAB7YcIUzUk", "title": "6 Trin Til En Klinik Du Kan Leve Af"},
        {"url": "https://www.youtube.com/watch?v=zeSmnxIZ4K0", "title": "2 Fysioterapeuter Gik Fra Deltid til 6 Millioner"},
        {"url": "https://www.youtube.com/watch?v=HeKzYxfCm6Q", "title": "Gratis Bookinger De Fleste Behandlere Overser"},
        {"url": "https://www.youtube.com/watch?v=Llcy3B2J1zg", "title": "3 Tegn Paa At Dine Behandlinger Er For Billige"},
        {"url": "https://www.youtube.com/watch?v=8eiZrKSqHa0", "title": "Det Perfekte Foerste Besoeg"},
        {"url": "https://www.youtube.com/watch?v=xZIubFBq98o", "title": "Mikkels klinik gik fra 50K til 120k"},
        {"url": "https://www.youtube.com/watch?v=81qN_HdoOxo", "title": "Oege Timeprisen Uden Mere Arbejde"},
        {"url": "https://www.youtube.com/watch?v=qjJhDvjQdfQ", "title": "Fundamentet i en velfungerende klinik"},
        {"url": "https://www.youtube.com/watch?v=spitLSYlMjc", "title": "Laer at saelge behandlingsforloeb baseret paa forskning"},
        {"url": "https://www.youtube.com/watch?v=6OW5meZW4VY", "title": "Hvorfor din markedsfoering ikke virker og saadan fikser du det"},
    ],
    "group2": [
        {"url": "https://player.vimeo.com/video/1097517089", "title": "100 bookinger workshop"},
        {"url": "https://player.vimeo.com/video/1114869545", "title": "WORKSHOP Sep 1 2025 TRIM"},
    ],
}


# ── yt-dlp Helpers ──────────────────────────────────────────────────────────

def _yt_dlp_base_opts(cookies: str | None = None) -> dict:
    """Return shared yt-dlp options (SSL, anti-blocking, cookies)."""
    opts: dict = {
        "nocheckcertificate": True,
        "no_warnings": True,
        "quiet": True,
    }
    if cookies:
        opts["cookiefile"] = cookies
    return opts


# ── Channel Scraping ─────────────────────────────────────────────────────────

def scrape_channel_videos(channel_url: str, cookies: str | None = None) -> list[dict]:
    """Extract all video URLs and titles from a YouTube channel.

    Uses yt-dlp's extract_flat mode to fetch only the listing metadata
    (minimal HTTP requests). Filters out Shorts (< 60s).
    Returns list of {"url": str, "title": str} matching VIDEO_GROUPS format.
    """
    # Ensure we hit the /videos tab for a complete listing
    if "/videos" not in channel_url:
        channel_url = channel_url.rstrip("/") + "/videos"

    ydl_opts = {
        **_yt_dlp_base_opts(cookies),
        "extract_flat": "in_playlist",
        "ignoreerrors": True,
        "sleep_interval_requests": 1,
    }
    print(f"Scraping channel: {channel_url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)

    if not info:
        raise RuntimeError(f"Failed to extract info from {channel_url}")

    entries = info.get("entries", [])
    videos = []
    skipped_shorts = 0
    for entry in entries:
        if entry is None:
            continue
        # Skip Shorts (under 60 seconds)
        duration = entry.get("duration") or 0
        if duration and duration < 60:
            skipped_shorts += 1
            continue
        video_id = entry.get("id") or entry.get("url")
        title = entry.get("title") or f"video_{video_id}"
        url = entry.get("url") or entry.get("webpage_url")
        if url and not url.startswith("http"):
            url = f"https://www.youtube.com/watch?v={url}"
        videos.append({"url": url, "title": title})

    print(f"Found {len(videos)} videos on channel"
          + (f" (skipped {skipped_shorts} Shorts)" if skipped_shorts else ""))
    return videos


def save_channel_cache(group_name: str, videos: list[dict]):
    """Cache the scraped video list for resume capability."""
    ensure_dirs(CHANNEL_CACHE_DIR)
    cache_path = CHANNEL_CACHE_DIR / f"{group_name}.json"
    data = {
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "video_count": len(videos),
        "videos": videos,
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Cached {len(videos)} videos to {cache_path.relative_to(SCRIPT_DIR)}")


def load_channel_cache(group_name: str) -> list[dict] | None:
    """Load cached video list if it exists."""
    cache_path = CHANNEL_CACHE_DIR / f"{group_name}.json"
    if not cache_path.exists():
        return None
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {data['video_count']} cached videos (scraped {data['scraped_at']})")
    return data["videos"]


# ── Utilities ────────────────────────────────────────────────────────────────

def sanitize_filename(title: str) -> str:
    """Convert title to a safe filename."""
    name = title.strip().lower()
    name = re.sub(r'[\\/:*?"<>|]', '', name)
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'[^\w\-.]', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')
    return name


def get_audio_path(group: str, video: dict) -> Path:
    return AUDIO_DIR / group / f"{sanitize_filename(video['title'])}.mp3"


def get_transcript_base(group: str, video: dict) -> Path:
    return TRANSCRIPTIONS_DIR / group / sanitize_filename(video['title'])


def ensure_dirs(*dirs: Path):
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# ── Download ─────────────────────────────────────────────────────────────────

def download_audio(url: str, output_path: Path, cookies: str | None = None):
    """Download audio from a video URL using yt-dlp Python API."""
    ensure_dirs(output_path.parent)
    # yt-dlp manages the intermediate format and final extension via postprocessor
    outtmpl = str(output_path.with_suffix("")) + ".%(ext)s"
    ydl_opts = {
        **_yt_dlp_base_opts(cookies),
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "5",
        }],
        "outtmpl": outtmpl,
        "noplaylist": True,
        "retries": 3,
        # Anti-blocking: random sleep between yt-dlp internal requests
        "sleep_interval": 3,
        "max_sleep_interval": 8,
        "sleep_interval_requests": 1,
    }
    print(f"    Downloading audio via yt-dlp ...")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as e:
        raise RuntimeError(f"yt-dlp failed for {url}: {e}")


# ── Audio Chunking ───────────────────────────────────────────────────────────

def get_audio_info(audio_path: Path) -> tuple[float, int]:
    """Get audio duration (seconds) and bitrate (bps) using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration,bit_rate",
        "-of", "csv=p=0",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    parts = result.stdout.strip().split(",")
    duration = float(parts[0])
    bitrate = int(parts[1]) if len(parts) > 1 and parts[1].strip() else 128000
    return duration, bitrate


def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration in seconds."""
    duration, _ = get_audio_info(audio_path)
    return duration


def split_audio(audio_path: Path, chunks_dir: Path) -> list[Path]:
    """Split audio into chunks under the Whisper API size limit (25MB)."""
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)

    if file_size_mb <= MAX_CHUNK_SIZE_MB:
        return [audio_path]

    duration, bitrate = get_audio_info(audio_path)

    # Calculate target chunk duration based on actual bitrate to stay under 20MB
    target_bytes = MAX_CHUNK_SIZE_MB * 1024 * 1024
    target_duration = target_bytes * 8 / max(bitrate, 32000)  # seconds per chunk
    num_chunks = max(2, math.ceil(duration / target_duration))
    chunk_duration = duration / num_chunks

    # Use source bitrate for re-encoding (avoid inflating with 128k)
    encode_bitrate = f"{max(bitrate // 1000, 32)}k"

    print(f"    Splitting {file_size_mb:.1f}MB file into {num_chunks} chunks "
          f"({chunk_duration:.0f}s each, {encode_bitrate}bps)")

    video_chunks_dir = chunks_dir / audio_path.stem
    ensure_dirs(video_chunks_dir)

    # Clean up any old chunks from previous failed runs
    if video_chunks_dir.exists():
        for f in video_chunks_dir.iterdir():
            f.unlink()

    chunks = []
    for i in range(num_chunks):
        start_time = max(0, i * chunk_duration - CHUNK_OVERLAP_SECONDS) if i > 0 else 0
        chunk_path = video_chunks_dir / f"chunk_{i:03d}.mp3"

        cmd = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-ss", str(start_time),
            "-t", str(chunk_duration + (CHUNK_OVERLAP_SECONDS if i > 0 else 0)),
            "-acodec", "libmp3lame",
            "-ab", encode_bitrate,
            "-v", "quiet",
            str(chunk_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        # Safety check: verify chunk is under 25MB
        chunk_size_mb = chunk_path.stat().st_size / (1024 * 1024)
        if chunk_size_mb > 24.5:
            print(f"    WARNING: chunk {i} is {chunk_size_mb:.1f}MB, may exceed API limit")

        chunks.append(chunk_path)

    return chunks


def cleanup_chunks(audio_path: Path, chunks_dir: Path):
    """Remove temporary chunk files."""
    video_chunks_dir = chunks_dir / audio_path.stem
    if video_chunks_dir.exists():
        for f in video_chunks_dir.iterdir():
            f.unlink()
        video_chunks_dir.rmdir()


# ── Transcription ────────────────────────────────────────────────────────────

def transcribe_file(file_path: Path, client: OpenAI, prompt: str, language: str = "da") -> dict:
    """Transcribe a single audio file using Whisper API."""
    with open(file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=language,
            response_format="verbose_json",
            prompt=prompt,
        )
    return response


def transcribe_audio(audio_path: Path, client: OpenAI, chunks_dir: Path, language: str = "da") -> tuple[str, list]:
    """Transcribe audio, handling chunking for large files."""
    chunks = split_audio(audio_path, chunks_dir)
    is_chunked = len(chunks) > 1

    all_text = []
    all_segments = []
    base_prompt = DANISH_PROMPT if language == "da" else ""
    prompt = base_prompt
    time_offset = 0.0

    for i, chunk_path in enumerate(chunks):
        if is_chunked:
            print(f"    Transcribing chunk {i + 1}/{len(chunks)}...")
        else:
            print(f"    Transcribing...")

        response = transcribe_file(chunk_path, client, prompt, language)

        chunk_text = response.text
        all_text.append(chunk_text)

        # Add segments with adjusted timestamps for chunked files
        if hasattr(response, 'segments') and response.segments:
            for seg in response.segments:
                all_segments.append({
                    "start": round(seg.start + time_offset, 2),
                    "end": round(seg.end + time_offset, 2),
                    "text": seg.text,
                })
            if is_chunked and response.segments:
                # Estimate the time offset for the next chunk
                time_offset += response.segments[-1].end

        # Use tail of this chunk's text as context for next chunk
        if is_chunked:
            prompt = base_prompt + " " + chunk_text[-200:]

        if i < len(chunks) - 1:
            time.sleep(API_DELAY_SECONDS)

    if is_chunked:
        cleanup_chunks(audio_path, chunks_dir)

    full_text = " ".join(all_text)
    return full_text, all_segments


# ── Output ───────────────────────────────────────────────────────────────────

def save_transcript(base_path: Path, video: dict, group: str, text: str, segments: list, duration: float, language: str = "da"):
    """Save transcription as .txt and .json files."""
    ensure_dirs(base_path.parent)

    # Plain text
    txt_path = base_path.with_suffix(".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"{video['title']}\n")
        f.write(f"[Source: {video['url']}]\n")
        f.write(f"[Duration: {int(duration // 60)}:{int(duration % 60):02d}]\n")
        f.write(f"[Language: {language}]\n\n")
        f.write(text)
    print(f"    Saved: {txt_path.relative_to(SCRIPT_DIR)}")

    # JSON with metadata and segments
    json_path = base_path.with_suffix(".json")
    data = {
        "title": video["title"],
        "url": video["url"],
        "group": group,
        "language": language,
        "duration_seconds": round(duration, 1),
        "transcribed_at": datetime.now(timezone.utc).isoformat(),
        "full_text": text,
        "segments": segments,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"    Saved: {json_path.relative_to(SCRIPT_DIR)}")


# ── Orchestration ────────────────────────────────────────────────────────────

def process_video(video: dict, group: str, client: OpenAI, args, language: str = "da", cookies: str | None = None) -> bool:
    """Process a single video: download + transcribe. Returns True on success."""
    audio_path = get_audio_path(group, video)
    transcript_base = get_transcript_base(group, video)

    # Skip if already transcribed (unless --force)
    if not args.force and transcript_base.with_suffix(".txt").exists() and transcript_base.with_suffix(".json").exists():
        print(f"  SKIP (already done): {video['title']}")
        return True

    # Download
    if not args.transcribe_only:
        if audio_path.exists():
            print(f"  SKIP download (exists): {video['title']}")
        else:
            print(f"  Downloading: {video['title']}")
            download_audio(video["url"], audio_path, cookies)

    if args.download_only:
        return True

    # Verify audio exists before transcribing
    if not audio_path.exists():
        print(f"  ERROR: Audio not found at {audio_path}")
        return False

    # Transcribe
    print(f"  Transcribing: {video['title']}")
    duration = get_audio_duration(audio_path)
    text, segments = transcribe_audio(audio_path, client, CHUNKS_DIR, language)
    save_transcript(transcript_base, video, group, text, segments, duration, language)
    return True


def main():
    parser = argparse.ArgumentParser(description="Download and transcribe videos using OpenAI Whisper")
    parser.add_argument("--download-only", action="store_true", help="Only download audio, skip transcription")
    parser.add_argument("--transcribe-only", action="store_true", help="Only transcribe (audio must exist)")
    parser.add_argument("--group", type=str, help="Process only this group (e.g., group1)")
    parser.add_argument("--force", action="store_true", help="Re-process even if output exists")
    parser.add_argument("--channel", type=str,
                        help="YouTube channel URL to scrape and process (e.g., https://youtube.com/@marketingpod)")
    parser.add_argument("--channel-name", type=str, default=None,
                        help="Group name for channel videos (default: derived from channel URL)")
    parser.add_argument("--limit", type=int, default=DEFAULT_CHANNEL_LIMIT,
                        help=f"Max number of channel videos to process, 0 for all (default: {DEFAULT_CHANNEL_LIMIT})")
    parser.add_argument("--language", type=str, default="da",
                        help="Whisper language code (default: da)")
    parser.add_argument("--cookies", type=str, default=None,
                        help="Path to Netscape-format cookies file for YouTube authentication")
    args = parser.parse_args()

    load_dotenv(SCRIPT_DIR / ".env")

    client = None
    if not args.download_only:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY not set. Check your .env file.")
            return
        client = OpenAI(api_key=api_key)

    # ── Resolve which videos to process ──────────────────────────────────
    is_channel_group = False

    if args.channel:
        # Derive group name from @handle in URL or use provided name
        if args.channel_name:
            channel_group_name = args.channel_name
        else:
            match = re.search(r"@([\w.-]+)", args.channel)
            channel_group_name = match.group(1) if match else "channel"

        # Try cache first; re-scrape with --force
        channel_videos = None
        if not args.force:
            channel_videos = load_channel_cache(channel_group_name)

        if channel_videos is None:
            channel_videos = scrape_channel_videos(args.channel, args.cookies)
            save_channel_cache(channel_group_name, channel_videos)

        # Apply limit (0 = no limit)
        if args.limit > 0:
            channel_videos = channel_videos[:args.limit]

        groups_to_process = {channel_group_name: channel_videos}
        is_channel_group = True

    elif args.group:
        if args.group not in VIDEO_GROUPS:
            print(f"ERROR: Unknown group '{args.group}'. Available: {', '.join(VIDEO_GROUPS.keys())}")
            return
        groups_to_process = {args.group: VIDEO_GROUPS[args.group]}
    else:
        groups_to_process = VIDEO_GROUPS

    # ── Process videos ───────────────────────────────────────────────────
    results = {"success": [], "failed": []}

    for group_name, videos in groups_to_process.items():
        print(f"\n{'=' * 60}")
        print(f"Processing {group_name} ({len(videos)} videos)")
        print(f"{'=' * 60}")

        batch_start = time.time()

        for i, video in enumerate(videos, 1):
            # ETA estimate for channel batches
            elapsed = time.time() - batch_start
            if is_channel_group and i > 1:
                avg_per_video = elapsed / (i - 1)
                remaining = avg_per_video * (len(videos) - i + 1)
                eta_min = remaining / 60
                print(f"\n[{i}/{len(videos)}] {video['title']}  (ETA: ~{eta_min:.0f}min)")
            else:
                print(f"\n[{i}/{len(videos)}] {video['title']}")

            try:
                ok = process_video(video, group_name, client, args, args.language, args.cookies)
                if ok:
                    results["success"].append(video["title"])
                else:
                    results["failed"].append(video["title"])
            except Exception as e:
                print(f"  ERROR: {e}")
                results["failed"].append(video["title"])

            # Anti-blocking delay between downloads for channel scrapes
            if is_channel_group and i < len(videos):
                delay = random.uniform(DOWNLOAD_DELAY_MIN, DOWNLOAD_DELAY_MAX)
                print(f"  (waiting {delay:.0f}s before next video)")
                time.sleep(delay)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"DONE: {len(results['success'])} succeeded, {len(results['failed'])} failed")
    if results["failed"]:
        print("Failed videos:")
        for title in results["failed"]:
            print(f"  - {title}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
