# Transcription Pipeline

Converts an audio file (WAV, MP3, M4A, FLAC, OGG, WEBM, MP4) into text with per-segment timestamps.

## Pipeline
1. **Ingest** – accept file via HTTP upload or CLI, validate extension and size, stream to disk.
2. **Normalize** – ffmpeg converts any format to 16 kHz mono PCM WAV.
3. **Split** – files longer than `CHUNK_SECONDS` (default 600s) are split into chunks.
4. **Transcribe** – faster-whisper (open-source Whisper) with VAD to skip silence.
5. **Output** – JSON with language, duration, full text, and timestamped segments (offsets corrected across chunks).

## Setup
Requires Python 3.10+ and ffmpeg on PATH.
```bash
pip install -r requirements.txt
```

## Usage
CLI:
```bash
python app.py sample.mp3
```
API:
```bash
uvicorn app:app --reload
curl -F "file=@sample.mp3" http://localhost:8000/transcribe
```

## Example output
```json
{
  "language": "en",
  "duration": 12.4,
  "text": "Hello and welcome. Today we discuss the pipeline.",
  "segments": [
    {"id": 0, "start": 0.0, "end": 3.2, "text": "Hello and welcome."},
    {"id": 1, "start": 3.4, "end": 7.9, "text": "Today we discuss the pipeline."}
  ]
}
```

## Config (env vars)
`WHISPER_MODEL` (base), `WHISPER_DEVICE` (cpu/cuda), `WHISPER_COMPUTE` (int8), `MAX_UPLOAD_MB` (200), `CHUNK_SECONDS` (600)

## Design decisions & trade-offs
- **faster-whisper** over OpenAI Whisper: ~4x faster on CPU with int8, same accuracy, no API cost, data stays local.
- **ffmpeg normalization** means one code path for every format instead of format-specific handling.
- **Chunking** keeps memory bounded on long files. Trade-off: a word can be cut at a chunk boundary; next step would be overlapping chunks or splitting on silence.
- **Model cached** once per process to avoid reloading on every request.
- **Production next steps:** async job queue (Celery/Redis) returning a job ID for long files, GPU workers, SRT/VTT export, speaker diarization.

## System design (how this would scale in production)

The current code is a working single-process prototype. For production I would split it into an API layer and a worker layer connected by a queue.

```
Client -> API (FastAPI) -> Object storage (S3/MinIO)
              |                 ^
              v                 |
          PostgreSQL <---- Workers (Celery + faster-whisper, GPU)
              ^                 ^
              |                 |
              +---- Queue (Redis/RabbitMQ) 
```

### Concurrent uploads
The API never transcribes inside the request. It validates the file, streams it to object storage (or gives the client a pre-signed URL to upload directly), creates a job row, pushes the job ID onto a queue and returns `202 Accepted` with the job ID. A pool of workers, each with the model loaded once, pulls jobs from the queue and scales horizontally with queue length. The queue absorbs traffic spikes, and per-client rate limits and file size limits protect the system.

### Storage
- **Audio:** object storage keyed by job ID (`audio/{job_id}.mp3`), with lifecycle rules to delete or archive after N days for cost and privacy.
- **Metadata and transcripts:** PostgreSQL. A `jobs` table (id, user, status, file key, SHA-256 hash, duration, language, model version, attempts, error, timestamps) and the result stored as JSONB or a `segments` table (job_id, start, end, text).
- **Search:** Postgres full-text index, or Elasticsearch if search grows.
- The file hash lets identical uploads reuse an existing transcript instead of re-running the model.

### Retries and recovery
- Job status flows `queued -> processing -> completed | failed`, with an attempt counter.
- Transient failures (worker crash, out of memory, timeout) are retried with exponential backoff, up to 3 attempts. Jobs are idempotent because the audio stays in storage, so re-running is safe.
- The queue uses late acknowledgement, so if a worker dies mid-job the message goes back on the queue.
- Permanent failures (corrupt or unsupported file) fail immediately with a clear error and are not retried.
- After max retries the job goes to a dead-letter queue and triggers an alert; an admin endpoint can re-run it.
- For long files, progress is saved per chunk so a retry resumes from the last finished chunk.
- A periodic sweeper re-queues jobs stuck in `processing` past a timeout.

### API
Versioned REST API:

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/v1/transcriptions` | Upload a file (or URL), returns `202` + `job_id` |
| GET | `/v1/transcriptions/{id}` | Status, and result when done |
| GET | `/v1/transcriptions/{id}?format=json\|srt\|vtt` | Export in different formats |
| DELETE | `/v1/transcriptions/{id}` | Delete audio and transcript |

Clients can pass a `callback_url` to receive a webhook on completion instead of polling. Authentication via API keys, rate limiting, auto-generated OpenAPI docs at `/docs`, and consistent error codes (400, 413, 415, 422, 429).
