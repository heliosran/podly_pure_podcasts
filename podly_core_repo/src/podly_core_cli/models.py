from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class EpisodeInput(BaseModel):
    title: str
    description: str = ""
    audio_path: str


class Segment(BaseModel):
    index: int
    start: float
    end: float
    text: str


class TranscriptArtifact(BaseModel):
    episode_title: str
    segments: List[Segment]


class AdPrediction(BaseModel):
    segment_index: int
    confidence: float = Field(ge=0.0, le=1.0)


class ClassificationArtifact(BaseModel):
    model: str
    predictions: List[AdPrediction]


class CutWindow(BaseModel):
    start_sec: float
    end_sec: float


class ProcessResult(BaseModel):
    output_audio_path: str
    transcript_path: str
    classification_path: str
    cut_windows: List[CutWindow]
