# YouTube Transcriber

A tiny local web GUI that:
1) Downloads a YouTube video's audio with **yt-dlp**
2) Transcribes locally with **faster-whisper** (no cloud)
3) Sends the transcript to **ChatGPT** to create:
   - A concise **summary** (Markdown)
   - A cleaned **full version** of the talk (Markdown)

> Runs fully on your Mac for steps (1) and (2). Only step (3) calls the OpenAI API.

---

## Prerequisites (macOS)

- **Homebrew** (optional but recommended)
- **FFmpeg** (required by yt-dlp for audio)
  ```bash
  brew install ffmpeg
  ```
- **Python 3.10+** (recommend `python3 -m venv .venv`)

## Install

```bash
cd youtube_whisper_local_app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` from the example and set your key:
```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY
```

Optional performance knobs in `.env`:
- `WHISPER_SIZE` = tiny|base|small|medium|large-v2
- `WHISPER_DEVICE` = auto|cpu (on Apple Silicon, `auto` will try to use Metal via CTranslate2 if available)
- `WHISPER_COMPUTE_TYPE` = int8|int8_float16|float16|float32 (tradeoff quality/speed)

## Run

```bash
source .venv/bin/activate
python app.py
# open http://127.0.0.1:5050
```

Paste a YouTube URL, choose the Whisper model size, and check the boxes for "Summary" and/or "Full version".

Outputs are saved under `work/<job_id>/`:
- `transcript.txt`
- `subtitles.srt`
- `summary.md` (optional)
- `full_version.md` (optional)

You can download each file from the results page.

## Notes

- If you prefer **openai-whisper** instead of **faster-whisper**, you can swap the transcription function easily.
- For very long talks, the app uses a hierarchical summarization pass (chunk → partial summaries → merged summary).
- The "full version" is a **light edit**: corrects errors, punctuation, paragraphs, removes filler—no new content is invented.

## Troubleshooting

- **FFmpeg not found**: `brew install ffmpeg`
- **yt-dlp errors**: update: `pip install -U yt-dlp`
- **OpenAI auth error**: ensure `OPENAI_API_KEY` is set in `.env`
- **Slow transcription**: try a smaller `WHISPER_SIZE` or set `WHISPER_COMPUTE_TYPE=int8`
- **Metal acceleration**: `faster-whisper` via CTranslate2 may use Metal with `device=auto` on Apple Silicon.
