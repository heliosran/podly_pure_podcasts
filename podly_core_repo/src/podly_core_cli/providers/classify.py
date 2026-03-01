from __future__ import annotations

import json
import logging
from typing import Any, Iterable, List, cast

import litellm
from podly_core_cli.config import CoreConfig
from podly_core_cli.models import AdPrediction, ClassificationArtifact, Segment

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You label podcast transcript segments as ad or content.
Return ONLY JSON with shape: {\"ad_segment_indices\": [int], \"confidence\": number}
Where indices refer to the provided segment indices.
"""


class AdClassifier:
    def __init__(self, config: CoreConfig) -> None:
        self.config = config
        litellm.api_key = config.llm_api_key
        litellm.api_base = config.openai_base_url

    def classify(self, segments: List[Segment]) -> ClassificationArtifact:
        predictions: List[AdPrediction] = []
        for batch in _batched(segments, self.config.segment_batch_size):
            predictions.extend(self._classify_batch(batch))

        # de-dup best confidence by segment index
        best: dict[int, float] = {}
        for p in predictions:
            best[p.segment_index] = max(best.get(p.segment_index, 0.0), p.confidence)

        final = [
            AdPrediction(segment_index=idx, confidence=conf)
            for idx, conf in sorted(best.items())
            if conf >= self.config.min_confidence
        ]
        return ClassificationArtifact(model=self.config.llm_model, predictions=final)

    def _classify_batch(self, batch: List[Segment]) -> List[AdPrediction]:
        user = {
            "segments": [
                {
                    "index": s.index,
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                }
                for s in batch
            ]
        }
        resp = litellm.completion(
            model=self.config.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user)},
            ],
            timeout=120,
            max_tokens=512,
        )

        content = resp.choices[0].message.content
        if content is None:
            return []

        payload = _parse_json(content)
        raw_indices = payload.get("ad_segment_indices", [])
        indices = raw_indices if isinstance(raw_indices, list) else []
        raw_confidence = payload.get("confidence", 1.0)
        confidence = (
            float(raw_confidence) if isinstance(raw_confidence, (int, float)) else 1.0
        )
        out: List[AdPrediction] = []
        valid = {s.index for s in batch}
        for idx in indices:
            if isinstance(idx, int) and idx in valid:
                out.append(AdPrediction(segment_index=idx, confidence=confidence))
        return out


def _batched(items: List[Segment], n: int) -> Iterable[List[Segment]]:
    n = max(1, n)
    for i in range(0, len(items), n):
        yield items[i : i + n]


def _parse_json(content: str) -> dict[str, object]:
    text = content.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    parsed = json.loads(text)
    return cast(dict[str, object], parsed)
