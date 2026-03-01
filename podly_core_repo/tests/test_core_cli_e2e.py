from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_cli_process_e2e_with_known_fixture(tmp_path: Path) -> None:
    """Runs the CLI end-to-end on a known fixture and checks output artifacts.

    The fixture is the repository's stable sample podcast audio and the classifier
    runs in `test` mode so this is deterministic and offline.
    """
    repo_root = Path(__file__).resolve().parents[2]
    fixture_audio = repo_root / "src" / "tests" / "data" / "count_0_99.mp3"
    assert fixture_audio.exists(), "fixture audio missing"

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "whisper_provider": "test",
                "classifier_provider": "test",
                "segment_batch_size": 10,
                "min_ad_segment_length_seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )

    output_audio = tmp_path / "output.mp3"
    work_dir = tmp_path / "work"

    cmd = [
        sys.executable,
        "-m",
        "podly_core_cli.cli",
        "process",
        "--config",
        str(config_path),
        "--work-dir",
        str(work_dir),
        "--audio",
        str(fixture_audio),
        "--title",
        "Known fixture podcast",
        "--description",
        "Podcast with known sponsored segment in test transcriber",
        "--out",
        str(output_audio),
    ]

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    extra = str((repo_root / "podly_core_repo" / "src").resolve())
    env["PYTHONPATH"] = f"{extra}:{existing}" if existing else extra

    result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    payload = json.loads(result.stdout)

    transcript_path = Path(payload["transcript_path"])
    classification_path = Path(payload["classification_path"])

    assert output_audio.exists()
    assert transcript_path.exists()
    assert classification_path.exists()

    assert output_audio.stat().st_size > 0

    cut_windows = payload.get("cut_windows", [])
    assert isinstance(cut_windows, list)
    assert len(cut_windows) >= 1
    assert float(cut_windows[0]["end_sec"]) > float(cut_windows[0]["start_sec"])

    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    classification = json.loads(classification_path.read_text(encoding="utf-8"))

    assert len(transcript["segments"]) == 2
    assert len(classification["predictions"]) >= 1
