# Podly Core (Standalone)

This folder is a **new standalone repo scaffold** for extracting Podly's core episode-processing flow into a CLI-first service.

## Goals

- decouple from Flask, SQLAlchemy, and writer IPC
- run one episode end-to-end from command line
- produce deterministic artifacts (`transcript.json`, `classification.json`, `output.mp3`)

## Commands

```bash
python -m podly_core_cli.cli process \
  --audio ./input.mp3 \
  --title "Episode Title" \
  --description "Episode description" \
  --out ./output.mp3
```

Optional steps:

```bash
python -m podly_core_cli.cli transcribe ...
python -m podly_core_cli.cli classify ...
python -m podly_core_cli.cli cut ...
```

## Config

A JSON config can be passed with `--config path.json`.

Example:

```json
{
  "whisper_provider": "groq",
  "whisper_model": "whisper-large-v3-turbo",
  "whisper_api_key": "...",
  "llm_model": "groq/openai/gpt-oss-120b",
  "llm_api_key": "...",
  "openai_base_url": "https://api.openai.com/v1",
  "segment_batch_size": 60,
  "min_confidence": 0.8
}
```

## Notes

- This package intentionally avoids DB writes and web app concepts.
- The JSON artifact format is meant to be stable for future service wrappers.
