from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

WhisperProvider = Literal["local", "remote", "groq", "test"]


class CoreConfig(BaseModel):
    whisper_provider: WhisperProvider = "groq"
    whisper_model: str = "whisper-large-v3-turbo"
    whisper_api_key: Optional[str] = None
    whisper_base_url: str = "https://api.openai.com/v1"
    whisper_language: str = "en"

    llm_model: str = "groq/openai/gpt-oss-120b"
    llm_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None

    segment_batch_size: int = 60
    min_confidence: float = 0.8
    min_ad_segment_length_seconds: float = 14.0
    min_ad_segment_separation_seconds: float = 60.0


def load_config(path: str | None) -> CoreConfig:
    if not path:
        return CoreConfig()
    content = Path(path).read_text(encoding="utf-8")
    return CoreConfig.model_validate(json.loads(content))
