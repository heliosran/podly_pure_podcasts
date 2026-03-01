from __future__ import annotations

import json
import shutil
from pathlib import Path

import requests

from podly_core_cli.audio import build_cut_windows, render_without_windows
from podly_core_cli.config import CoreConfig
from podly_core_cli.models import EpisodeInput, ProcessResult, TranscriptArtifact
from podly_core_cli.providers.classify import AdClassifier
from podly_core_cli.providers.transcribe import make_transcriber


def process_episode(
    episode: EpisodeInput,
    config: CoreConfig,
    output_audio_path: str,
    work_dir: str,
) -> ProcessResult:
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    transcriber = make_transcriber(config)
    segments = transcriber.transcribe(episode.audio_path)
    transcript = TranscriptArtifact(episode_title=episode.title, segments=segments)
    transcript_path = work / "transcript.json"
    transcript_path.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")

    classifier = AdClassifier(config)
    classification = classifier.classify(segments)
    classification_path = work / "classification.json"
    classification_path.write_text(
        classification.model_dump_json(indent=2), encoding="utf-8"
    )

    windows = build_cut_windows(
        segments,
        ad_indices={p.segment_index for p in classification.predictions},
        min_len_sec=config.min_ad_segment_length_seconds,
        min_sep_sec=config.min_ad_segment_separation_seconds,
    )

    output = Path(output_audio_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    render_without_windows(episode.audio_path, str(output), windows)

    return ProcessResult(
        output_audio_path=str(output),
        transcript_path=str(transcript_path),
        classification_path=str(classification_path),
        cut_windows=windows,
    )


def ensure_local_audio(audio: str, work_dir: str) -> str:
    if audio.startswith("http://") or audio.startswith("https://"):
        out = Path(work_dir) / "input.mp3"
        with requests.get(audio, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with out.open("wb") as f:
                shutil.copyfileobj(resp.raw, f)
        return str(out)
    return audio


def write_json(path: str, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
