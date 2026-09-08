"""
NOP AI Developer Challenge 2026 - Main CLI Entry Point
Executes the video perception, tracking, event detection, and evidence export pipeline.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline import VisionPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="NOP Pro+ Vision Intelligence Challenge - Candidate Solution"
    )
    parser.add_argument("--input", required=True, help="Path to input video file (MP4, MKV, AVI)")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to JSON configuration. If not specified, auto-matches scenario.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to save evidence, CSVs, JSONL, and annotated video",
    )
    parser.add_argument(
        "--scenario-id",
        default=None,
        help="Scenario ID (e.g., S01_BASIC_GOODS). Auto-inferred if omitted.",
    )
    parser.add_argument("--display", action="store_true", help="Display live annotated video window")
    parser.add_argument(
        "--no-video", action="store_true", help="Skip rendering output MP4 video for max speed"
    )
    return parser.parse_args()


def resolve_config(input_path: str, config_path: str = None) -> tuple:
    """Finds or builds the best matching configuration based on input file."""
    p = Path(input_path).stem.upper()

    scenario_map = {
        "S01": ("S01_BASIC_GOODS", "config/s01_basic_goods.json"),
        "S02": ("S02_OCCLUSION_REVERSAL", "config/s02_occlusion_reversal.json"),
        "S03": ("S03_DENSE_CROSSING", "config/s03_dense_crossing.json"),
        "S04": ("S04_DWELL_QUEUE", "config/s04_dwell_queue.json"),
    }

    inferred_scenario = "GENERIC_VIDEO"
    candidate_cfg = "config/default_config.json"

    for prefix, (scen_id, cfg_file) in scenario_map.items():
        if prefix in p:
            inferred_scenario = scen_id
            candidate_cfg = cfg_file
            break

    target_cfg = config_path or candidate_cfg

    # Resolve relative path if needed
    base_dir = Path(__file__).parent
    cfg_file_path = Path(target_cfg)
    if not cfg_file_path.is_file():
        cfg_file_path = base_dir / target_cfg
    if not cfg_file_path.is_file():
        cfg_file_path = base_dir / "config/default_config.json"

    with open(cfg_file_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    return config_data, inferred_scenario


def main():
    args = parse_args()
    config_data, inferred_scenario = resolve_config(args.input, args.config)
    scenario_id = args.scenario_id or config_data.get("scenario_id", inferred_scenario)
    config_data["scenario_id"] = scenario_id

    print(f"\n========================================================")
    print(f" NOP Pro+ Vision Intelligence - Pipeline Starting")
    print(f" Scenario:   {scenario_id}")
    print(f" Input:      {args.input}")
    print(f" Output Dir: {args.output_dir}")
    print(f"========================================================\n")

    pipeline = VisionPipeline(
        config=config_data, output_dir=args.output_dir, scenario_id=scenario_id
    )

    result = pipeline.process_video(
        video_path=args.input,
        display=args.display,
        save_video=not args.no_video,
    )

    print("\n---------------- Processing Complete ----------------")
    print(f"Processed Frames: {result['processed_frames']}")
    print(f"Total Events:     {result['events_count']}")
    print("Event Breakdown:")
    for etype, count in result["counts"].items():
        print(f"  - {etype}: {count}")
    print(f"Outputs saved to: {args.output_dir}")
    print("========================================================\n")


if __name__ == "__main__":
    main()
