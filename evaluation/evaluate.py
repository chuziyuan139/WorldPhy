#!/usr/bin/env python3
"""Evaluate generated task videos with three WorldPhyReward scoring passes."""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
RUBRICS = [f"R{i}" for i in range(1, 8)]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(rows):
    """Match the released results: invalid passes fail the strict 7/7 test."""
    valid = [row for row in rows if row["status"] == "valid"
             and len(row["passes"]) == 3 and all(p["valid"] for p in row["passes"])]
    result = {"videos": len(rows), "valid_videos": len(valid),
              "evaluations": sum(len(row["passes"]) for row in rows)}
    for rubric in RUBRICS:
        result[rubric] = mean(mean(p["scores"][rubric] for p in row["passes"])
                              for row in valid) if valid else None
    result["total_mean"] = sum(result[r] for r in RUBRICS) if valid else None
    for number in (1, 2, 3):
        passed = 0
        for row in rows:
            p = next((p for p in row["passes"] if p["pass_id"] == number), None)
            passed += bool(p and p["valid"] and all(p["scores"][r] == 1 for r in RUBRICS))
        result[f"passed_run{number}"] = passed
        result[f"pass_rate_run{number}"] = 100 * passed / len(rows) if rows else None
    rates = [result[f"pass_rate_run{number}"] for number in (1, 2, 3)]
    result["pass_rate_mean"] = mean(rates) if rows else None
    result["pass_rate_std"] = stdev(rates) if rows else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True,
                        help="Directory containing <task_id>.mp4 for each generated task")
    parser.add_argument("--model", required=True, help="WorldPhyReward checkpoint path or Hugging Face model ID")
    parser.add_argument("--output", type=Path, required=True, help="Directory for new evaluation outputs")
    parser.add_argument("--tasks", type=Path, default=ROOT / "task/instruction/tasks.json")
    parser.add_argument("--name", help="Generation-model name in the result tables (default: videos directory name)")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261011,
                        help="Use this same seed for each of the three actual scoring passes")
    args = parser.parse_args()
    if args.tp < 1 or args.batch_size < 1:
        parser.error("--tp and --batch-size must be positive")
    tasks_path, videos, output = args.tasks.resolve(), args.videos.resolve(), args.output.resolve()
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list) or not tasks:
        parser.error("--tasks must contain a nonempty JSON list")
    ids = [task["task_id"] for task in tasks]
    if len(ids) != len(set(ids)):
        parser.error("Task IDs must be unique")
    name = args.name or videos.name
    rows, missing = [], []
    for task in tasks:
        task_id = task["task_id"]
        video = videos / f"{task_id}.mp4"
        if not video.is_file():
            missing.append(task_id)
            continue
        image = (tasks_path.parent / task["image"]).resolve()
        if not image.is_file():
            parser.error(f"Missing task image: {image}")
        rows.append({"task_id": task_id, "video": os.path.relpath(video, output),
                     "image": os.path.relpath(image, output), "caption": task["caption"],
                     "classification": task["classification"]})
    if not rows:
        parser.error("No matching <task_id>.mp4 videos found")
    config = {"model": args.model, "name": name, "seed": args.seed, "tp": args.tp,
              "batch_size": args.batch_size, "tasks": str(tasks_path), "videos": str(videos)}
    output.mkdir(parents=True, exist_ok=True)
    records_path, config_path = output / "records.json", output / "evaluation_config.json"
    if any(output.iterdir()):
        if not records_path.is_file() or not config_path.is_file():
            parser.error("Use an empty output directory or resume this evaluator's existing output")
        if (json.loads(records_path.read_text()) != rows
                or json.loads(config_path.read_text()) != config):
            parser.error("Existing output has different inputs or settings; use a new --output directory")
    save_json(records_path, rows)
    save_json(config_path, config)
    print(f"Evaluating {len(rows)}/{len(tasks)} task videos; {len(missing)} missing.", flush=True)
    runs = []
    for number in (1, 2, 3):
        run_dir = output / f"pass_{number}"
        subprocess.run([sys.executable, str(ROOT / "worldphyreward/inference/score.py"),
                        "--records", str(records_path), "--output", str(run_dir),
                        "--model", args.model, "--tp", str(args.tp),
                        "--batch-size", str(args.batch_size), "--seed", str(args.seed)], check=True)
        predictions = json.loads((run_dir / "predictions.json").read_text())
        run = {p["video"]: p for p in predictions}
        if len(run) != len(predictions):
            raise ValueError(f"Duplicate video predictions in {run_dir}")
        runs.append(run)
    scored = []
    for row in rows:
        passes = []
        for number, run in enumerate(runs, 1):
            raw = run.get(row["video"], {"format_error": "Missing prediction"})
            prediction = raw.get("prediction")
            valid = prediction is not None and not raw.get("format_error")
            scores = {r: prediction[r]["score"] for r in RUBRICS} if valid else None
            reasons = {r: prediction[r]["reason"] for r in RUBRICS} if valid else None
            passes.append({"pass_id": number, "status": "valid" if valid else "invalid",
                           "valid": valid, "scores": scores, "reasons": reasons,
                           "total_score": sum(scores.values()) if valid else None,
                           "text": raw.get("text", ""), "format_error": raw.get("format_error"),
                           "raw_output": f"pass_{number}/predictions.json"})
        valid = all(p["valid"] for p in passes)
        scores = {r: mean(p["scores"][r] for p in passes) for r in RUBRICS} if valid else None
        scored.append({**row, "status": "valid" if valid else "invalid", "passes": passes,
                       "scores": scores, "total_score": sum(scores.values()) if valid else None})
    save_json(output / "scores.json", {"model": name, "scoring_model": "WorldPhyReward",
              "score_aggregation": f"mean of three actual passes; same seed={args.seed}, temperature=0",
              "expected_tasks": len(tasks), "video_count": len(scored),
              "missing_task_ids": missing, "records": scored})
    overall = {"model": name, "expected": len(tasks), **metrics(scored)}
    write_csv(output / "overall.csv", [overall])
    categories = []
    for field in ("gripper_hand", "arm_amount", "camera_view"):
        for value in sorted({row["classification"][field] for row in scored}):
            subset = [row for row in scored if row["classification"][field] == value]
            categories.append({"model": name, "category": field, "value": value, **metrics(subset)})
    write_csv(output / "categories.csv", categories)
    print(f"7/7 pass rate: {overall['pass_rate_mean']:.2f} +/- {overall['pass_rate_std']:.2f}% "
          f"(sample SD across three runs; denominator {len(scored)} videos)")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
