#!/usr/bin/env python3
"""Download YouTube/Vimeo videos and transcribe them using OpenAI Whisper API."""

import argparse
import json
import math
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
YT_DLP_PATH = SCRIPT_DIR / "yt-dlp.exe"
AUDIO_DIR = SCRIPT_DIR / "audio"
CHUNKS_DIR = SCRIPT_DIR / "chunks"
TRANSCRIPTIONS_DIR = SCRIPT_DIR / "transcriptions"

# ── Constants ────────────────────────────────────────────────────────────────
MAX_CHUNK_SIZE_MB = 20  # Whisper API limit is 25MB, generous margin for re-encoding variance
CHUNK_OVERLAP_SECONDS = 30
API_DELAY_SECONDS = 2

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
    ],
    "group2": [
        {"url": "https://player.vimeo.com/video/1097517089", "title": "100 bookinger workshop"},
        {"url": "https://player.vimeo.com/video/1114869545", "title": "WORKSHOP Sep 1 2025 TRIM"},
    ],
}


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

def download_audio(url: str, output_path: Path):
    """Download audio from a video URL using yt-dlp."""
    ensure_dirs(output_path.parent)
    cmd = [
        str(YT_DLP_PATH),
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "5",
        "-o", str(output_path),
        "--no-playlist",
        "--retries", "3",
        "--no-warnings",
        url,
    ]
    print(f"    Running: yt-dlp -x --audio-format mp3 ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    yt-dlp stderr: {result.stderr[:500]}")
        raise RuntimeError(f"yt-dlp failed for {url}")


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

def transcribe_file(file_path: Path, client: OpenAI, prompt: str) -> dict:
    """Transcribe a single audio file using Whisper API."""
    with open(file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="da",
            response_format="verbose_json",
            prompt=prompt,
        )
    return response


def transcribe_audio(audio_path: Path, client: OpenAI, chunks_dir: Path) -> tuple[str, list]:
    """Transcribe audio, handling chunking for large files."""
    chunks = split_audio(audio_path, chunks_dir)
    is_chunked = len(chunks) > 1

    all_text = []
    all_segments = []
    prompt = DANISH_PROMPT
    time_offset = 0.0

    for i, chunk_path in enumerate(chunks):
        if is_chunked:
            print(f"    Transcribing chunk {i + 1}/{len(chunks)}...")
        else:
            print(f"    Transcribing...")

        response = transcribe_file(chunk_path, client, prompt)

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
            prompt = DANISH_PROMPT + " " + chunk_text[-200:]

        if i < len(chunks) - 1:
            time.sleep(API_DELAY_SECONDS)

    if is_chunked:
        cleanup_chunks(audio_path, chunks_dir)

    full_text = " ".join(all_text)
    return full_text, all_segments


# ── Output ───────────────────────────────────────────────────────────────────

def save_transcript(base_path: Path, video: dict, group: str, text: str, segments: list, duration: float):
    """Save transcription as .txt and .json files."""
    ensure_dirs(base_path.parent)

    # Plain text
    txt_path = base_path.with_suffix(".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"{video['title']}\n")
        f.write(f"[Source: {video['url']}]\n")
        f.write(f"[Duration: {int(duration // 60)}:{int(duration % 60):02d}]\n")
        f.write(f"[Language: Danish]\n\n")
        f.write(text)
    print(f"    Saved: {txt_path.relative_to(SCRIPT_DIR)}")

    # JSON with metadata and segments
    json_path = base_path.with_suffix(".json")
    data = {
        "title": video["title"],
        "url": video["url"],
        "group": group,
        "language": "da",
        "duration_seconds": round(duration, 1),
        "transcribed_at": datetime.now(timezone.utc).isoformat(),
        "full_text": text,
        "segments": segments,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"    Saved: {json_path.relative_to(SCRIPT_DIR)}")


# ── Orchestration ────────────────────────────────────────────────────────────

def process_video(video: dict, group: str, client: OpenAI, args) -> bool:
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
            download_audio(video["url"], audio_path)

    if args.download_only:
        return True

    # Verify audio exists before transcribing
    if not audio_path.exists():
        print(f"  ERROR: Audio not found at {audio_path}")
        return False

    # Transcribe
    print(f"  Transcribing: {video['title']}")
    duration = get_audio_duration(audio_path)
    text, segments = transcribe_audio(audio_path, client, CHUNKS_DIR)
    save_transcript(transcript_base, video, group, text, segments, duration)
    return True


def main():
    parser = argparse.ArgumentParser(description="Download and transcribe videos using OpenAI Whisper")
    parser.add_argument("--download-only", action="store_true", help="Only download audio, skip transcription")
    parser.add_argument("--transcribe-only", action="store_true", help="Only transcribe (audio must exist)")
    parser.add_argument("--group", type=str, help="Process only this group (e.g., group1)")
    parser.add_argument("--force", action="store_true", help="Re-process even if output exists")
    args = parser.parse_args()

    load_dotenv(SCRIPT_DIR / ".env")

    client = None
    if not args.download_only:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY not set. Check your .env file.")
            return
        client = OpenAI(api_key=api_key)

    groups_to_process = {args.group: VIDEO_GROUPS[args.group]} if args.group else VIDEO_GROUPS

    results = {"success": [], "failed": []}

    for group_name, videos in groups_to_process.items():
        print(f"\n{'=' * 60}")
        print(f"Processing {group_name} ({len(videos)} videos)")
        print(f"{'=' * 60}")

        for i, video in enumerate(videos, 1):
            print(f"\n[{i}/{len(videos)}] {video['title']}")
            try:
                ok = process_video(video, group_name, client, args)
                if ok:
                    results["success"].append(video["title"])
                else:
                    results["failed"].append(video["title"])
            except Exception as e:
                print(f"  ERROR: {e}")
                results["failed"].append(video["title"])

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
