#!/usr/bin/env python3
import os
import re
import uuid
import time
import math
import json
import shutil
import textwrap
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, send_from_directory, redirect, url_for, flash
from werkzeug.utils import secure_filename

# Local processing
import subprocess
import yt_dlp
from faster_whisper import WhisperModel

# OpenAI (use raw HTTP to avoid library version mismatches)
import requests

# ---------------- Configuration ----------------
BASE_DIR = Path(__file__).resolve().parent
WORK_DIR = BASE_DIR / "work"
WORK_DIR.mkdir(exist_ok=True)

# Load environment (optional)
from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env", override=True)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")  # adjust if you prefer another model
DEVICE = os.environ.get("WHISPER_DEVICE", "auto")  # "auto" | "cpu"
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")  # "int8" is fast & light for faster-whisper

# Allowed Whisper sizes: tiny, base, small, medium, large-v2, large-v3 (if available)
DEFAULT_WHISPER_SIZE = os.environ.get("WHISPER_SIZE", "small")

# Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-me")

# ---------------- Utilities ----------------
def _timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def sanitize_filename(name: str) -> str:
    name = secure_filename(name) or "audio"
    return name[:120]

def download_youtube_audio(youtube_url: str, outdir: Path) -> Path:
    """
    Downloads bestaudio from YouTube using yt-dlp and returns the audio file path (wav/m4a/mp3 depending on availability).
    Ensures FFmpeg is installed for conversions if needed.
    """
    print(f"Downloading audio...{youtube_url}")
    outdir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(outdir / "%(title).200B.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        #ydl.download(['youtube_url'])
        info = ydl.extract_info(youtube_url, download=True)
        title = info.get('title', 'audio')
        # After postprocessor, expect .mp3
        filename = sanitize_filename(title) + ".mp3"
        audio_path = outdir / filename
        if not audio_path.exists():
            # Fallback: search for any audio in outdir
            for p in outdir.glob("*.mp3"):
                audio_path = p
                break
        return audio_path

def transcribe_with_faster_whisper(audio_path: Path, model_size: str = DEFAULT_WHISPER_SIZE, language: str = None):
    """
    Transcribe using faster-whisper locally.
    Returns (segments, full_text, srt_text).
    """
    model = WhisperModel(model_size, device=DEVICE, compute_type=COMPUTE_TYPE)
    segments, info = model.transcribe(str(audio_path), language=language, vad_filter=True)
    segments = list(segments)

    # Build plain text + srt
    lines = []
    srt_lines = []
    for i, seg in enumerate(segments, start=1):
        text = seg.text.strip()
        lines.append(text)

        def to_srt_ts(t):
            ms = int((t - int(t)) * 1000)
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = int(t % 60)
            return f"{h:02}:{m:02}:{s:02},{ms:03}"

        start = to_srt_ts(seg.start)
        end = to_srt_ts(seg.end)
        srt_lines.append(f"{i}\n{start} --> {end}\n{text}\n")

    full_text = "\n".join(lines)
    srt_text = "\n".join(srt_lines)
    return segments, full_text, srt_text

def chunk_text(text, max_chars=12000):
    # conservative chunking by characters; avoids token counting complexities
    parts = []
    cur = []
    cur_len = 0
    for para in text.split("\n"):
        if cur_len + len(para) + 1 > max_chars:
            parts.append("\n".join(cur))
            cur = [para]
            cur_len = len(para) + 1
        else:
            cur.append(para)
            cur_len += len(para) + 1
    if cur:
        parts.append("\n".join(cur))
    return [p for p in parts if p.strip()]

def call_openai_chat(messages, model=OPENAI_MODEL, temperature=0.2):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing. Put it in .env or environment variables.")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI API error: {resp.status_code} {resp.text}")
    data = resp.json()
    return data["choices"][0]["message"]["content"]

def summarize_transcript(transcript_text: str):
    # Hierarchical summarize to handle long transcripts
    chunks = chunk_text(transcript_text, max_chars=12000)
    partial_summaries = []
    for i, ch in enumerate(chunks, 1):
        messages = [
            {"role": "system", "content": "You are a concise expert summarizer. Produce a tight bullet summary in the source language."},
            {"role": "user", "content": f"Summarize part {i}/{len(chunks)} of a talk:\n\n{ch}"}
        ]
        partial = call_openai_chat(messages)
        partial_summaries.append(partial)

    # Final merge
    messages = [
        {"role": "system", "content": "You are a senior editor. Merge the partial summaries into one cohesive summary with clear sections and key takeaways."},
        {"role": "user", "content": "Combine and refine these partial summaries into one:\n\n" + "\n\n---\n\n".join(partial_summaries)}
    ]
    final_summary = call_openai_chat(messages)
    return final_summary

def rewrite_full_version(transcript_text: str):
    # Produce a cleaned, readable 'full version' with light editing.
    chunks = chunk_text(transcript_text, max_chars=12000)
    edited_chunks = []
    for i, ch in enumerate(chunks, 1):
        messages = [
            {"role": "system", "content": "You are a careful editor. Output a polished, readable 'full version' preserving the speaker's meaning. Fix obvious transcription errors, punctuate, paragraph, remove filler, but do NOT invent new content."},
            {"role": "user", "content": f"Edit part {i}/{len(chunks)} of the transcript for clarity and flow. Keep original language and meaning:\n\n{ch}"}
        ]
        edited = call_openai_chat(messages, temperature=0.1)
        edited_chunks.append(edited)

    # Optional final pass to smooth transitions
    messages = [
        {"role": "system", "content": "You are a professional copyeditor. Smooth transitions and ensure consistent style across sections without adding new content."},
        {"role": "user", "content": "\n\n".join(edited_chunks)}
    ]
    final = call_openai_chat(messages, temperature=0.1)
    return final

# ---------------- Routes ----------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html",
                           default_model=DEFAULT_WHISPER_SIZE,
                           openai_model=OPENAI_MODEL)

@app.route("/process", methods=["POST"])
def process():
    yt_url = request.form.get("youtube_url", "").strip()
    whisper_size = request.form.get("whisper_size", DEFAULT_WHISPER_SIZE)
    language = request.form.get("language", "").strip() or None
    want_summary = bool(request.form.get("want_summary"))
    want_full = bool(request.form.get("want_full"))

    if not yt_url:
        flash("Please provide a YouTube URL.", "danger")
        return redirect(url_for("index"))

    # Each job goes in its own folder
    job_id = f"{_timestamp()}_{uuid.uuid4().hex[:8]}"
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1) Download audio
        audio_path = download_youtube_audio(yt_url, job_dir)

        # 2) Transcribe locally
        segments, full_text, srt_text = transcribe_with_faster_whisper(audio_path, model_size=whisper_size, language=language)

        # Save artifacts
        (job_dir / "transcript.txt").write_text(full_text, encoding="utf-8")
        (job_dir / "subtitles.srt").write_text(srt_text, encoding="utf-8")

        summary_text = ""
        full_version_text = ""

        if want_summary:
            summary_text = summarize_transcript(full_text)
            (job_dir / "summary.md").write_text(summary_text, encoding="utf-8")

        if want_full:
            full_version_text = rewrite_full_version(full_text)
            (job_dir / "full_version.md").write_text(full_version_text, encoding="utf-8")

        return render_template(
            "result.html",
            job_id=job_id,
            audio_file=audio_path.name,
            full_text=full_text,
            summary_text=summary_text,
            full_version_text=full_version_text
        )

    except Exception as e:
        # cleanup on failure
        shutil.rmtree(job_dir, ignore_errors=True)
        flash(f"Error: {e}", "danger")
        return redirect(url_for("index"))

@app.route("/download/<job_id>/<path:filename>")
def download(job_id, filename):
    d = WORK_DIR / job_id
    return send_from_directory(d, filename, as_attachment=True)

if __name__ == "__main__":
    # For local dev
    app.run(host="127.0.0.1", port=5050, debug=True)
