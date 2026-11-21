"""Week13: Part A Thunderbird chunk13 + Part D position-based selection.

From repository root (after installing requirements):
    python part_C/week13/run_position_experiment.py

This script selects whole log lines and their token indices; it does NOT run
Qwen forward or compress model-produced KV cache. Alert labels are read only
AFTER both FRONT/BACK selections, for a separate retention evaluation.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "part_C" / "src"
sys.path.insert(0, str(SRC))

from prepare_week6_thunderbird import make_inputs
from line_kv_compressor_position import load_positions, select_position_lines


def load_alert_ids(csv_path: Path) -> set[int]:
    """Load Alert ground truth ONLY for evaluation, never selection."""
    result: set[int] = set()
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["is_alert"] == "1":
                result.add(int(row["line_id"]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Week13 Thunderbird position experiment")
    parser.add_argument("--window-id", type=int, default=314927)
    parser.add_argument("--chunk-index", type=int, default=13)
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--chunks-dir", type=Path,
                        default=ROOT / "Part_A" / "week6" / "thunderbird" / "chunks")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "part_C" / "week13" / "output")
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Produces position scores for the entire original 1-hour window first.
    files = make_inputs(args.chunks_dir, output, window_id=args.window_id,
                        chunk_index=args.chunk_index)
    with Path(files["token_map_jsonl"]).open(encoding="utf-8") as f:
        chunk = json.loads(next(line for line in f if line.strip()))
    token_map = chunk["token_map"]
    if chunk["window_idx"] != args.chunk_index:
        raise ValueError("토큰 맵 Chunk ID 불일치")

    positions = load_positions(files["positions_csv"])
    selections = {}
    summary = {
        "window_id": args.window_id,
        "chunk_index": args.chunk_index,
        "window_lines": files["window_lines"],
        "window_alerts": files["window_alerts"],
        "chunk_lines": files["chunk_lines"],
        "chunk_tokens": files["chunk_tokens"],
        "keep_ratio": args.keep_ratio,
        "results": {},
    }
    for direction in ("front", "back"):
        selection = select_position_lines(token_map, positions,
                                          keep_ratio=args.keep_ratio,
                                          direction=direction)
        selections[direction] = selection
        # Same Selection schema as the existing CLI; can inspect indices later.
        selection_file = output / f"chunk{args.chunk_index}_selection_{direction}.json"
        data = asdict(selection)
        data.update({
            "window_id": args.chunk_index, "parent_window_id": args.window_id,
            "direction": direction, "used_tokens": selection.used_tokens,
            "original_positions": selection.kept_token_indices,
            "next_position": selection.total_tokens,
            "note": "로그/토큰 선택 결과입니다. 실제 LLM의 KV 압축 실행은 별도 검증이 필요합니다.",
        })
        selection_file.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                  encoding="utf-8")

    # Gold labels are loaded ONLY after the two selection policies have finished.
    alert_ids = load_alert_ids(Path(files["alert_labels_csv"]))
    chunk_line_ids = {int(token["line_id"]) for token in token_map}
    chunk_alerts = alert_ids & chunk_line_ids
    summary["chunk_alerts"] = len(chunk_alerts)

    print(f"Window {args.window_id} / Chunk {args.chunk_index} ")
    print(f"원본 {len(token_map):,} tokens | {len(chunk_line_ids)} lines | "
          f"Alert {len(chunk_alerts)}")
    print("정책        보존토큰     선택라인   보존Alert/전체Alert    Alert보존율")
    for direction, selection in selections.items():
        saved = set(selection.selected_line_ids)
        retained = saved & chunk_alerts
        retention = (len(retained) / len(chunk_alerts)) if chunk_alerts else None
        summary["results"][direction] = {
            "score": "1-P" if direction == "front" else "P",
            "selected_lines": len(saved),
            "selected_tokens": selection.used_tokens,
            "budget_tokens": selection.budget_tokens,
            "token_reduction_ratio": 1 - selection.used_tokens / selection.total_tokens,
            "retained_alerts": len(retained),
            "lost_alerts": len(chunk_alerts - saved),
            "alert_retention": retention,
            "lost_alert_ids": sorted(chunk_alerts - saved),
        }
        retention_text = f"{retention:.2%}" if retention is not None else "N/A"
        print(f"{direction:6}     {selection.used_tokens:>7,}      {len(saved):>5}         "
              f"{len(retained):>2}/{len(chunk_alerts):<2}             {retention_text}")

    summary_file = output / f"chunk{args.chunk_index}_results.json"
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"\n결과 저장: {summary_file}")
    print("주의: 여기까지는 위치 기반 로그/토큰 선택과 Alert 보존율 측정입니다. "
          "실제 모델 KV 압축·RCA 원인 보존율은 아직 아닙니다.")


if __name__ == "__main__":
    main()
