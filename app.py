"""
Simple transcription pipeline:
    audio file -> validate -> normalize (ffmpeg) -> split if long -> Whisper -> timestamped JSON

Run the API:  uvicorn app:app --reload
Run the CLI:  python app.py path/to/audio.mp3
"""
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from faster_whisper import WhisperModel

# ---------- Config (overridable via environment variables) ----------
ALLOWED_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4"}
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))
MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")       # tiny/base/small/medium/large-v3
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")           # cpu or cuda
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE", "int8")   # int8 is fast on CPU
CHUNK_SECONDS = int(os.getenv("CHUNK_SECONDS", "600"))  # split files longer than 10 min

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("transcriber")


# ---------- Model (loaded once, reused for every request) ----------
@lru_cache(maxsize=1)
def get_model() -> WhisperModel:
    log.info("Loading Whisper model '%s' on %s (%s)", MODEL_SIZE, DEVICE, COMPUTE_TYPE)
    return WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)


# ---------- Audio helpers ----------
def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg/ffprobe not found. Install ffmpeg and add it to PATH.")


def normalize_audio(src: Path, dst: Path) -> None:
    """Convert any input format to 16 kHz mono 16-bit PCM WAV (the format Whisper expects)."""
    _require_ffmpeg()
    cmd = ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000",
           "-c:a", "pcm_s16le", str(dst)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(f"Could not decode audio file: {result.stderr[-300:]}")


def get_duration(path: Path) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        raise ValueError("Could not read audio duration")


def split_audio(wav: Path, out_dir: Path, chunk_seconds: int) -> list[tuple[Path, float]]:
    """Split long audio into fixed-length chunks. Returns (chunk_path, start_offset_seconds)."""
    if get_duration(wav) <= chunk_seconds:
        return [(wav, 0.0)]

    cmd = ["ffmpeg", "-y", "-i", str(wav), "-f", "segment",
           "-segment_time", str(chunk_seconds), "-c", "copy",
           str(out_dir / "chunk_%04d.wav")]
    subprocess.run(cmd, capture_output=True, check=True)

    chunks, offset = [], 0.0
    for chunk in sorted(out_dir.glob("chunk_*.wav")):
        chunks.append((chunk, offset))
        offset += get_duration(chunk)
    log.info("Split audio into %d chunks", len(chunks))
    return chunks


# ---------- Core pipeline ----------
def transcribe_file(path: Path) -> dict:
    model = get_model()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        wav = tmp / "normalized.wav"
        normalize_audio(path, wav)
        duration = get_duration(wav)

        segments_out, language = [], None
        for chunk_path, offset in split_audio(wav, tmp, CHUNK_SECONDS):
            segments, info = model.transcribe(str(chunk_path), vad_filter=True, beam_size=5)
            language = language or info.language
            for seg in segments:  # generator: segments are produced as decoding happens
                text = seg.text.strip()
                if not text:
                    continue
                segments_out.append({
                    "id": len(segments_out),
                    "start": round(seg.start + offset, 2),
                    "end": round(seg.end + offset, 2),
                    "text": text,
                })

    return {
        "language": language,
        "duration": round(duration, 2),
        "text": " ".join(s["text"] for s in segments_out),
        "segments": segments_out,
    }


# ---------- HTTP API ----------
app = FastAPI(title="Transcription Service")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(415, f"Unsupported format '{ext}'. Allowed: {sorted(ALLOWED_EXT)}")

    fd, tmp_name = tempfile.mkstemp(suffix=ext)
    tmp_path = Path(tmp_name)
    try:
        # Stream upload to disk in 1 MB pieces instead of loading it all into memory
        size, limit = 0, MAX_UPLOAD_MB * 1024 * 1024
        with os.fdopen(fd, "wb") as out:
            while piece := await file.read(1024 * 1024):
                size += len(piece)
                if size > limit:
                    raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB")
                out.write(piece)

        # Transcription is CPU-heavy, so run it off the event loop
        return await run_in_threadpool(transcribe_file, tmp_path)
    except ValueError as e:
        raise HTTPException(422, str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------- CLI ----------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python app.py <audio_file>")
        sys.exit(1)
    result = transcribe_file(Path(sys.argv[1]))
    print(json.dumps(result, indent=2, ensure_ascii=False))
