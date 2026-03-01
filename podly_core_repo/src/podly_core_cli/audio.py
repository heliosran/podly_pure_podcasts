from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Tuple

import ffmpeg  # type: ignore[import-untyped]

from podly_core_cli.models import CutWindow, Segment


def build_cut_windows(
    segments: List[Segment],
    ad_indices: set[int],
    min_len_sec: float,
    min_sep_sec: float,
) -> List[CutWindow]:
    windows: List[Tuple[float, float]] = []
    for s in segments:
        if s.index in ad_indices:
            windows.append((s.start, s.end))

    if not windows:
        return []

    windows.sort(key=lambda x: x[0])
    merged: List[Tuple[float, float]] = [windows[0]]
    for start, end in windows[1:]:
        prev_s, prev_e = merged[-1]
        if prev_e + min_sep_sec >= start:
            merged[-1] = (prev_s, max(prev_e, end))
        else:
            merged.append((start, end))

    merged = [w for w in merged if (w[1] - w[0]) >= min_len_sec]
    return [CutWindow(start_sec=s, end_sec=e) for s, e in merged]


def render_without_windows(input_path: str, output_path: str, windows: List[CutWindow]) -> None:
    duration_ms = _get_audio_duration_ms(input_path)
    if duration_ms is None:
        raise ValueError(f"Unable to get duration for: {input_path}")

    ad_segments_ms = [(int(w.start_sec * 1000), int(w.end_sec * 1000)) for w in windows]
    _clip_segments_simple(ad_segments_ms, input_path, output_path, duration_ms)


def _get_audio_duration_ms(file_path: str) -> int | None:
    try:
        probe = ffmpeg.probe(file_path)
        return int(float(probe["format"]["duration"]) * 1000)
    except Exception:
        return None


def _clip_segments_simple(
    ad_segments_ms: List[Tuple[int, int]], in_path: str, out_path: str, audio_duration_ms: int
) -> None:
    keep_segments: List[Tuple[int, int]] = []
    last_end = 0
    for start_ms, end_ms in sorted(ad_segments_ms):
        if start_ms > last_end:
            keep_segments.append((last_end, start_ms))
        last_end = end_ms
    if last_end < audio_duration_ms:
        keep_segments.append((last_end, audio_duration_ms))

    if not keep_segments:
        raise ValueError("No audio left after cuts")

    with tempfile.TemporaryDirectory() as temp_dir:
        segment_files = []
        for i, (start_ms, end_ms) in enumerate(keep_segments):
            segment_path = Path(temp_dir) / f"segment_{i}.mp3"
            start_sec = start_ms / 1000.0
            duration_sec = (end_ms - start_ms) / 1000.0
            (
                ffmpeg.input(in_path)
                .output(
                    str(segment_path), ss=start_sec, t=duration_sec, acodec="libmp3lame", q=2
                )
                .overwrite_output()
                .run(quiet=True)
            )
            segment_files.append(segment_path)

        concat_file = Path(temp_dir) / "concat_list.txt"
        concat_file.write_text(
            "".join([f"file '{p}'\n" for p in segment_files]),
            encoding="utf-8",
        )

        (
            ffmpeg.input(str(concat_file), format="concat", safe=0)
            .output(out_path, acodec="libmp3lame", q=2)
            .overwrite_output()
            .run(quiet=True)
        )
