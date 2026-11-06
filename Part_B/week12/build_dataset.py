import json
from pathlib import Path
from collections import Counter

base_dir = Path(__file__).resolve().parent
output_dir = base_dir / "output"

files = [
    "frontend_exception_candidates.json",
    "checkout_exception_candidates.json",
    "adservice_exception_candidates.json"
]

dataset = []
total_candidates = 0

for filename in files:

    path = output_dir / filename

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    candidates = data["evidence_candidates"]

    # 원본 CSV 위치 중복 검사
    locations = [
        (r["csv_file"], r["csv_row_index"])
        for r in candidates
    ]

    if len(locations) != len(set(locations)):
        raise ValueError(
            f"{filename}: 중복된 CSV 위치 발견"
        )

    # TraceID 개수 및 중복 확인
    trace_counts = Counter(
        r["trace_id"] for r in candidates
    )

    duplicated_traces = {
        trace_id: count
        for trace_id, count in trace_counts.items()
        if count > 1
    }

    print(f"\n===== {data['incident_id']} =====")
    print("증거 후보:", len(candidates))
    print("TraceID:", len(trace_counts))
    print("중복 등장한 TraceID:", len(duplicated_traces))

    for trace_id, count in duplicated_traces.items():
        print(f"  {trace_id}: {count}줄")

        for record in candidates:
            if record["trace_id"] == trace_id:
                print(
                    f"    {record['csv_file']} "
                    f"Row {record['csv_row_index']}"
                )

    # Incident마다 후보 형식 통일
    normalized_candidates = []

    for record in candidates:

        normalized_candidates.append({
            "timestamp_ns": record["timestamp_ns"],
            "trace_id": record["trace_id"],
            "csv_file": record["csv_file"],
            "csv_row_index": record["csv_row_index"],
            "message_pattern": record["message_pattern"],
            "evidence_role": record.get(
                "evidence_role",
                "injected_service_error_candidate"
            ),
            "evidence_status": "candidate_unverified"
        })

    dataset.append({
        "incident_id": data["incident_id"],
        "incident": data["incident"],
        "candidate_count": len(normalized_candidates),
        "trace_count": len(trace_counts),
        "evidence_candidates": normalized_candidates
    })

    total_candidates += len(normalized_candidates)

# 통합 JSON
result = {
    "dataset_name": "Nezha Exception Evidence Candidates",
    "label_status": "candidate_unverified",
    "incident_count": len(dataset),
    "candidate_count": total_candidates,
    "incidents": dataset
}

output_path = output_dir / "nezha_exception_dataset.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("\n===== 통합 결과 =====")
print("Incident 수:", len(dataset))
print("증거 후보 총합:", total_candidates)
print("저장 위치:", output_path)

import csv
import hashlib

log_dir = Path(
    r"C:\Users\qkral\projects\Nezha\rca_data\2022-08-22\log"
)

targets = {
    "07_54_log.csv": 24098,
    "07_55_log.csv": 23078
}

print("\n===== adservice 중복 TraceID 상세 비교 =====")

for filename, target_idx in targets.items():
    path = log_dir / filename

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row_idx, row in enumerate(reader):
            if row_idx != target_idx:
                continue

            outer = json.loads(row["Log"])
            inner = outer.get("log", "")
            data = json.loads(inner) if isinstance(inner, str) else inner

            print(f"\n파일: {filename}")
            print("CSV Row:", row_idx)
            print("Timestamp:", row["Timestamp"])
            print("TraceID:", row["TraceID"])
            print("SpanID:", row["SpanID"])
            print("Message:", data.get("message", ""))
            print(
                "원본 Log SHA256:",
                hashlib.sha256(
                    row["Log"].encode("utf-8")
                ).hexdigest()
            )

            break