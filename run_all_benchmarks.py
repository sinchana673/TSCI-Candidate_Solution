"""
NOP Vision Intelligence Pipeline - Comprehensive Benchmark Runner
Processes all 4 public challenge scenarios, consolidates output CSVs,
evaluates precision/recall/F1 with tools/evaluate_events.py,
and produces an audit summary.
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

# Add candidate-solution/src to path
sys.path.insert(0, str(Path(__file__).parent / "candidate-solution" / "src"))
from pipeline import VisionPipeline

SCENARIOS = [
    ("S01_BASIC_GOODS", "data/generated/S01_BASIC_GOODS.mp4", "candidate-solution/config/s01_basic_goods.json"),
    ("S02_OCCLUSION_REVERSAL", "data/generated/S02_OCCLUSION_REVERSAL.mp4", "candidate-solution/config/s02_occlusion_reversal.json"),
    ("S03_DENSE_CROSSING", "data/generated/S03_DENSE_CROSSING.mp4", "candidate-solution/config/s03_dense_crossing.json"),
    ("S04_DWELL_QUEUE", "data/generated/S04_DWELL_QUEUE.mp4", "candidate-solution/config/s04_dwell_queue.json"),
]


def main():
    root_dir = Path(__file__).parent
    output_base = root_dir / "candidate-solution" / "output"
    output_base.mkdir(parents=True, exist_ok=True)

    consolidated_csv_path = output_base / "consolidated_events.csv"
    benchmark_summary_path = output_base / "benchmark_summary.json"

    all_event_rows = []
    benchmark_metrics = {}

    print("\n" + "=" * 70)
    print("  NOP PRO+ VISION INTELLIGENCE - FULL BENCHMARK SUITE EXECUTION")
    print("=" * 70 + "\n")

    total_start_time = time.time()

    for scen_id, video_rel, config_rel in SCENARIOS:
        vid_path = root_dir / video_rel
        cfg_path = root_dir / config_rel
        scen_out_dir = output_base / scen_id
        scen_out_dir.mkdir(parents=True, exist_ok=True)

        if not vid_path.is_file():
            print(f"[ERROR] Video file not found: {vid_path}")
            continue

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)

        print(f"\n>>> Running Scenario: {scen_id}")
        print(f"    Video:  {video_rel}")
        print(f"    Config: {config_rel}")

        pipeline = VisionPipeline(config=cfg_data, output_dir=str(scen_out_dir), scenario_id=scen_id)

        t0 = time.time()
        res = pipeline.process_video(str(vid_path), display=False, save_video=True)
        elapsed = time.time() - t0

        fps_effective = res["processed_frames"] / max(0.001, elapsed)
        print(f"    Done: {res['processed_frames']} frames in {elapsed:.2f}s ({fps_effective:.1f} FPS)")
        print(f"    Events detected: {res['events_count']}")
        for etype, cnt in res["counts"].items():
            print(f"      - {etype}: {cnt}")

        benchmark_metrics[scen_id] = {
            "processed_frames": res["processed_frames"],
            "elapsed_seconds": round(elapsed, 2),
            "effective_fps": round(fps_effective, 1),
            "counts": res["counts"],
            "analytics": res.get("analytics", {}),
        }

        # Collect rows from scenario events.csv
        scen_csv = scen_out_dir / "events.csv"
        if scen_csv.is_file():
            with open(scen_csv, "r", newline="", encoding="utf-8") as cf:
                reader = csv.DictReader(cf)
                for r in reader:
                    all_event_rows.append(r)

    # Write consolidated CSV
    if all_event_rows:
        fieldnames = [
            "scenario_id",
            "event_id",
            "track_id",
            "event_type",
            "object_type",
            "observed_at_seconds",
            "confidence",
            "evidence_image",
        ]
        with open(consolidated_csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_event_rows:
                writer.writerow(r)
        print(f"\n[OK] Wrote consolidated events to: {consolidated_csv_path}")

    # Write benchmark summary JSON
    with open(benchmark_summary_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_metrics, f, indent=2)
    print(f"[OK] Wrote benchmark summary JSON to: {benchmark_summary_path}")

    total_elapsed = time.time() - total_start_time
    print(f"\nAll scenarios completed in {total_elapsed:.2f} seconds.")

    # Run official evaluate_events.py
    print("\n" + "=" * 70)
    print("  OFFICIAL PUBLIC EVENT EVALUATOR RESULTS")
    print("=" * 70)
    evaluator_script = root_dir / "tools" / "evaluate_events.py"
    if evaluator_script.is_file() and consolidated_csv_path.is_file():
        import subprocess
        cmd = [sys.executable, str(evaluator_script), "--candidate", str(consolidated_csv_path)]
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
