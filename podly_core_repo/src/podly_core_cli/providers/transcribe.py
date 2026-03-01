from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List

from groq import Groq
from openai import OpenAI
from podly_core_cli.config import CoreConfig
from podly_core_cli.models import Segment

logger = logging.getLogger(__name__)


class Transcriber(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def transcribe(self, audio_path: str) -> List[Segment]:
        raise NotImplementedError


class TestTranscriber(Transcriber):
    @property
    def model_name(self) -> str:
        return "test_whisper"

    def transcribe(self, audio_path: str) -> List[Segment]:
        del audio_path
        return [
            Segment(index=0, start=0.0, end=2.0, text="Intro and content"),
            Segment(index=1, start=2.0, end=4.0, text="Sponsor message"),
        ]


class LocalWhisperTranscriber(Transcriber):
    def __init__(self, model: str) -> None:
        self._model = model

    @property
    def model_name(self) -> str:
        return f"local_{self._model}"

    def transcribe(self, audio_path: str) -> List[Segment]:
        import whisper  # type: ignore[import-untyped]

        model = whisper.load_model(self._model)
        result = model.transcribe(audio_path, fp16=False, language="English")
        out: List[Segment] = []
        for idx, seg in enumerate(result["segments"]):
            out.append(
                Segment(
                    index=idx,
                    start=float(seg["start"]),
                    end=float(seg["end"]),
                    text=str(seg["text"]),
                )
            )
        return out


class OpenAIWhisperTranscriber(Transcriber):
    def __init__(self, config: CoreConfig) -> None:
        if not config.whisper_api_key:
            raise ValueError("whisper_api_key required for remote whisper")
        self._model = config.whisper_model
        self._language = config.whisper_language
        self._client = OpenAI(
            base_url=config.whisper_base_url,
            api_key=config.whisper_api_key,
        )

    @property
    def model_name(self) -> str:
        return self._model

    def transcribe(self, audio_path: str) -> List[Segment]:
        with open(audio_path, "rb") as f:
            tx = self._client.audio.transcriptions.create(
                model=self._model,
                file=f,
                timestamp_granularities=["segment"],
                language=self._language,
                response_format="verbose_json",
            )

        segments: List[Segment] = []
        for idx, seg in enumerate(tx.segments or []):
            segments.append(
                Segment(
                    index=idx,
                    start=float(seg.start),
                    end=float(seg.end),
                    text=str(seg.text),
                )
            )
        return segments


class GroqWhisperTranscriber(Transcriber):
    def __init__(self, config: CoreConfig) -> None:
        if not config.whisper_api_key:
            raise ValueError("whisper_api_key required for groq whisper")
        self._model = config.whisper_model
        self._language = config.whisper_language
        self._client = Groq(api_key=config.whisper_api_key)

    @property
    def model_name(self) -> str:
        return f"groq_{self._model}"

    def transcribe(self, audio_path: str) -> List[Segment]:
        tx = self._client.audio.transcriptions.create(
            file=Path(audio_path),
            model=self._model,
            response_format="verbose_json",
            language=self._language,
        )
        segments: List[Segment] = []
        for idx, seg in enumerate(getattr(tx, "segments", []) or []):
            item: Any = seg
            segments.append(
                Segment(
                    index=idx,
                    start=float(item["start"]),
                    end=float(item["end"]),
                    text=str(item["text"]),
                )
            )
        return segments


def make_transcriber(config: CoreConfig) -> Transcriber:
    if config.whisper_provider == "test":
        return TestTranscriber()
    if config.whisper_provider == "local":
        return LocalWhisperTranscriber(config.whisper_model)
    if config.whisper_provider == "remote":
        return OpenAIWhisperTranscriber(config)
    if config.whisper_provider == "groq":
        return GroqWhisperTranscriber(config)
    raise ValueError(f"Unsupported whisper provider: {config.whisper_provider}")
