from __future__ import annotations

import argparse
import json
from pathlib import Path

from podly_core_cli.audio import build_cut_windows, render_without_windows
from podly_core_cli.config import CoreConfig, load_config
from podly_core_cli.models import EpisodeInput, Segment, TranscriptArtifact
from podly_core_cli.pipeline import ensure_local_audio, process_episode
from podly_core_cli.providers.classify import AdClassifier
from podly_core_cli.providers.transcribe import make_transcriber


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="podly-core")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=str, default=None)
    common.add_argument("--work-dir", type=str, default="./.podly-core")

    p_process = sub.add_parser("process", parents=[common])
    p_process.add_argument("--audio", required=True)
    p_process.add_argument("--title", required=True)
    p_process.add_argument("--description", default="")
    p_process.add_argument("--out", required=True)

    p_transcribe = sub.add_parser("transcribe", parents=[common])
    p_transcribe.add_argument("--audio", required=True)
    p_transcribe.add_argument("--title", required=True)
    p_transcribe.add_argument("--out", required=True)

    p_classify = sub.add_parser("classify", parents=[common])
    p_classify.add_argument("--transcript", required=True)
    p_classify.add_argument("--out", required=True)

    p_cut = sub.add_parser("cut", parents=[common])
    p_cut.add_argument("--audio", required=True)
    p_cut.add_argument("--transcript", required=True)
    p_cut.add_argument("--classification", required=True)
    p_cut.add_argument("--out", required=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    if args.cmd == "process":
        audio_path = ensure_local_audio(args.audio, args.work_dir)
        episode = EpisodeInput(
            title=args.title,
            description=args.description,
            audio_path=audio_path,
        )
        result = process_episode(
            episode=episode,
            config=config,
            output_audio_path=args.out,
            work_dir=args.work_dir,
        )
        print(result.model_dump_json(indent=2))
        return

    if args.cmd == "transcribe":
        audio_path = ensure_local_audio(args.audio, args.work_dir)
        transcriber = make_transcriber(config)
        segments = transcriber.transcribe(audio_path)
        artifact = TranscriptArtifact(episode_title=args.title, segments=segments)
        Path(args.out).write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
        print(json.dumps({"transcriber": transcriber.model_name, "segments": len(segments)}))
        return

    if args.cmd == "classify":
        payload = json.loads(Path(args.transcript).read_text(encoding="utf-8"))
        transcript = TranscriptArtifact.model_validate(payload)
        classifier = AdClassifier(config)
        result = classifier.classify(transcript.segments)
        Path(args.out).write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(json.dumps({"model": result.model, "predictions": len(result.predictions)}))
        return

    if args.cmd == "cut":
        transcript_payload = json.loads(Path(args.transcript).read_text(encoding="utf-8"))
        classification_payload = json.loads(
            Path(args.classification).read_text(encoding="utf-8")
        )
        transcript = TranscriptArtifact.model_validate(transcript_payload)
        ad_indices = {
            int(p["segment_index"]) for p in classification_payload.get("predictions", [])
        }
        windows = build_cut_windows(
            transcript.segments,
            ad_indices=ad_indices,
            min_len_sec=config.min_ad_segment_length_seconds,
            min_sep_sec=config.min_ad_segment_separation_seconds,
        )
        render_without_windows(args.audio, args.out, windows)
        print(json.dumps({"windows": [w.model_dump() for w in windows]}))
        return


if __name__ == "__main__":
    main()
