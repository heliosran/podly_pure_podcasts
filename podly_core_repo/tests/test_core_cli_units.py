from __future__ import annotations

import json
from pathlib import Path

from podly_core_cli.audio import build_cut_windows
from podly_core_cli.config import CoreConfig, load_config
from podly_core_cli.models import Segment
from podly_core_cli.providers.classify import _parse_json


def test_build_cut_windows_merges_and_filters() -> None:
    segments = [
        Segment(index=0, start=0.0, end=10.0, text="a"),
        Segment(index=1, start=10.0, end=20.0, text="b"),
        Segment(index=2, start=100.0, end=110.0, text="c"),
    ]

    windows = build_cut_windows(
        segments,
        ad_indices={0, 1, 2},
        min_len_sec=8.0,
        min_sep_sec=5.0,
    )

    assert len(windows) == 2
    assert windows[0].start_sec == 0.0
    assert windows[0].end_sec == 20.0
    assert windows[1].start_sec == 100.0
    assert windows[1].end_sec == 110.0


def test_parse_json_handles_fenced_block() -> None:
    payload = """```json
    {"ad_segment_indices": [1,2], "confidence": 0.91}
    ```"""
    parsed = _parse_json(payload)
    assert parsed["ad_segment_indices"] == [1, 2]
    assert parsed["confidence"] == 0.91


def test_load_config_from_file(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "whisper_provider": "test",
                "llm_model": "openai/gpt-4o-mini",
                "segment_batch_size": 10,
            }
        ),
        encoding="utf-8",
    )

    cfg = load_config(str(path))
    assert isinstance(cfg, CoreConfig)
    assert cfg.whisper_provider == "test"
    assert cfg.llm_model == "openai/gpt-4o-mini"
    assert cfg.segment_batch_size == 10
