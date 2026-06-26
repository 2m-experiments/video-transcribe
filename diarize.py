#!/usr/bin/env python3
"""Add speaker labels to an existing Whisper transcript using AssemblyAI diarization.

Why this is a separate, optional step:

  Whisper (OpenAI) gives us the best Danish *text*, but it cannot tell speakers
  apart. AssemblyAI can diarize ("who spoke when"), so we use it ONLY for its
  speaker timeline and throw away its transcript text. Each existing Whisper
  segment is assigned to the speaker whose diarized turn overlaps it most.

  The original *.txt / *.json are never modified. We write NEW files:
      <name>.speakers.txt    – transcript grouped into "Speaker A: ..." turns
      <name>.speakers.json   – original segments, each annotated with "speaker"

  Because the merge is purely timestamp-based, this runs on an ALREADY
  transcribed file — no need (and no cost) to re-transcribe.

Usage:
    export ASSEMBLYAI_API_KEY=...
    python diarize.py transcriptions/group4/<name>.json
    # audio path is inferred (audio/<group>/<name>.mp3); override with --audio

Caveat: AssemblyAI's `speaker_labels` must support the audio's language. Danish
is provided via the multilingual model; if a given account/model rejects it,
pass --language-code en (diarization is acoustic and still groups speakers,
though the discarded AssemblyAI text will be wrong — which we don't use anyway).
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
AUDIO_DIR = SCRIPT_DIR / "audio"

AAI_BASE = "https://api.assemblyai.com/v2"
POLL_INTERVAL_SECONDS = 5
UPLOAD_CHUNK = 5 * 1024 * 1024  # 5MB streaming chunks


# ── Pure merge logic (no network — unit-testable) ────────────────────────────

def assign_speakers(segments: list[dict], turns: list[dict]) -> list[dict]:
    """Annotate each Whisper segment with the best-overlapping diarized speaker.

    `segments` are Whisper segments with float `start`/`end` (seconds).
    `turns` are diarized spans: {"speaker": "A", "start": s, "end": s} (seconds).

    Each segment is matched to the speaker whose turn shares the most time with
    it. Segments with no overlap inherit the previous segment's speaker (a short
    aside between two of A's turns is almost always still A), falling back to the
    first known speaker, then "?".
    """
    out = []
    last_speaker = None
    for seg in segments:
        s, e = seg["start"], seg["end"]
        best_speaker, best_overlap = None, 0.0
        for t in turns:
            overlap = min(e, t["end"]) - max(s, t["start"])
            if overlap > best_overlap:
                best_overlap, best_speaker = overlap, t["speaker"]
        if best_speaker is None:
            best_speaker = last_speaker
        else:
            last_speaker = best_speaker
        out.append({**seg, "speaker": best_speaker or "?"})
    return out


def format_speaker_transcript(segments: list[dict]) -> str:
    """Render speaker-annotated segments as grouped 'Speaker X: ...' turns.

    Consecutive segments from the same speaker are merged into one paragraph so
    the output reads as a conversation rather than one line per Whisper segment.
    """
    blocks: list[str] = []
    cur_speaker = object()  # sentinel distinct from any real label
    buf: list[str] = []
    for seg in segments:
        spk = seg.get("speaker") or "?"
        if spk != cur_speaker:
            if buf:
                blocks.append(f"Speaker {cur_speaker}: " + " ".join(buf).strip())
            cur_speaker, buf = spk, []
        text = seg["text"].strip()
        if text:
            buf.append(text)
    if buf:
        blocks.append(f"Speaker {cur_speaker}: " + " ".join(buf).strip())
    return "\n\n".join(blocks)


# ── AssemblyAI diarization (network) ─────────────────────────────────────────

def _request(url: str, *, method: str, api_key: str, data: bytes | None = None,
             content_type: str | None = None) -> dict:
    headers = {"authorization": api_key}
    if content_type:
        headers["content-type"] = content_type
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        raise RuntimeError(f"AssemblyAI {method} {url} -> HTTP {e.code}: {body}") from e


def _upload_audio(audio_path: Path, api_key: str) -> str:
    """Stream-upload a local audio file, return the AssemblyAI upload URL."""
    def chunks():
        with open(audio_path, "rb") as f:
            while data := f.read(UPLOAD_CHUNK):
                yield data

    req = urllib.request.Request(
        f"{AAI_BASE}/upload",
        data=chunks(),
        method="POST",
        headers={
            "authorization": api_key,
            "content-type": "application/octet-stream",
            # Content-Length unknown when streaming a generator → chunked encoding
            "transfer-encoding": "chunked",
        },
    )
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)["upload_url"]


def assemblyai_diarize(audio_path: Path, api_key: str,
                       language_code: str = "da") -> list[dict]:
    """Diarize an audio file via AssemblyAI; return speaker turns in seconds.

    Returns a list of {"speaker": "A", "start": float, "end": float}. We use the
    `utterances` (one entry per continuous speaker turn) and DISCARD the text.
    """
    print(f"  Uploading audio to AssemblyAI: {audio_path.name}")
    upload_url = _upload_audio(audio_path, api_key)

    print("  Requesting diarized transcript ...")
    payload = json.dumps({
        "audio_url": upload_url,
        "speaker_labels": True,
        "language_code": language_code,
    }).encode("utf-8")
    created = _request(f"{AAI_BASE}/transcript", method="POST", api_key=api_key,
                       data=payload, content_type="application/json")
    transcript_id = created["id"]

    poll_url = f"{AAI_BASE}/transcript/{transcript_id}"
    while True:
        result = _request(poll_url, method="GET", api_key=api_key)
        status = result.get("status")
        if status == "completed":
            break
        if status == "error":
            raise RuntimeError(f"AssemblyAI transcription failed: {result.get('error')}")
        print(f"    status={status} ... waiting {POLL_INTERVAL_SECONDS}s")
        time.sleep(POLL_INTERVAL_SECONDS)

    utterances = result.get("utterances") or []
    if not utterances:
        raise RuntimeError(
            "AssemblyAI returned no utterances — speaker_labels may be "
            f"unsupported for language_code={language_code!r}."
        )
    return [
        {"speaker": u["speaker"], "start": u["start"] / 1000.0, "end": u["end"] / 1000.0}
        for u in utterances
    ]


# ── Orchestration ────────────────────────────────────────────────────────────

def infer_audio_path(transcript_json: dict, transcript_path: Path) -> Path:
    """Locate the source mp3 for a transcript (audio/<group>/<stem>.mp3)."""
    group = transcript_json.get("group", transcript_path.parent.name)
    return AUDIO_DIR / group / f"{transcript_path.stem}.mp3"


def diarize_transcript(transcript_path: Path, api_key: str,
                       audio_path: Path | None = None,
                       language_code: str = "da") -> tuple[Path, Path]:
    """Add speaker labels to one transcript; write *.speakers.{json,txt}."""
    with open(transcript_path, encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments") or []
    if not segments:
        raise RuntimeError(f"No segments in {transcript_path}")

    if audio_path is None:
        audio_path = infer_audio_path(data, transcript_path)
    if not audio_path.exists():
        raise RuntimeError(f"Audio not found: {audio_path} (pass --audio)")

    turns = assemblyai_diarize(audio_path, api_key, language_code)
    speakers = sorted({t["speaker"] for t in turns})
    print(f"  Diarization found {len(speakers)} speaker(s): {', '.join(speakers)}")

    labeled = assign_speakers(segments, turns)

    transcript_path = transcript_path.resolve()
    stem = transcript_path.with_suffix("")  # drop .json
    json_out = stem.with_suffix(".speakers.json")
    txt_out = stem.with_suffix(".speakers.txt")

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump({
            **{k: v for k, v in data.items() if k != "segments"},
            "diarization": {"provider": "assemblyai", "speakers": speakers,
                            "language_code": language_code},
            "segments": labeled,
        }, f, ensure_ascii=False, indent=2)

    with open(txt_out, "w", encoding="utf-8") as f:
        f.write(f"{data.get('title', transcript_path.stem)} — with speakers\n")
        f.write(f"[Source: {data.get('url', '')}]\n")
        f.write(f"[Speakers: {', '.join(speakers)} | text: Whisper, "
                f"diarization: AssemblyAI]\n\n")
        f.write(format_speaker_transcript(labeled))

    print(f"  Saved: {json_out.relative_to(SCRIPT_DIR)}")
    print(f"  Saved: {txt_out.relative_to(SCRIPT_DIR)}")
    return json_out, txt_out


def main():
    parser = argparse.ArgumentParser(
        description="Add AssemblyAI speaker labels to an existing Whisper transcript.")
    parser.add_argument("transcript", type=Path, help="Path to a transcript .json file")
    parser.add_argument("--audio", type=Path, default=None,
                        help="Audio file (default: audio/<group>/<stem>.mp3)")
    parser.add_argument("--language-code", default="da",
                        help="AssemblyAI language_code for diarization (default: da)")
    args = parser.parse_args()

    try:  # .env support is optional; env var works on its own
        from dotenv import load_dotenv
        load_dotenv(SCRIPT_DIR / ".env")
    except ModuleNotFoundError:
        pass
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        print("ERROR: ASSEMBLYAI_API_KEY not set. Add it to your .env file.")
        raise SystemExit(1)

    diarize_transcript(args.transcript, api_key, args.audio, args.language_code)


if __name__ == "__main__":
    main()
