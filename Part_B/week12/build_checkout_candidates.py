import csv
import json
from pathlib import Path

data_dir = Path(
    r"C:\Users\qkral\projects\Nezha\rca_data\2022-08-22"
)

# 1. 공식 장애 정보 읽기
fault_path = data_dir / "2022-08-22-fault_list.json"

with open(fault_path, "r", encoding="utf-8") as f:
    fault_data = json.load(f)

faults = [
    item
    for incidents in fault_data.values()
    for item in incidents
]

incident = next(
    item for item in faults
    if item["inject_time"] == "2022-08-22 05:01:13"
    and item["inject_type"] == "exception"
)

inject_ts = int(incident["inject_timestamp"]) * 1_000_000_000
end_ts = inject_ts + 60 * 1_000_000_000

# 2. 직접 오류 증거 후보 추출
candidates = []

for filename in ["05_01_log.csv", "05_02_log.csv"]:
    path = data_dir / "log" / filename

    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row_idx, row in enumerate(reader):
            ts = int(row["TimeUnixNano"])

            if not (inject_ts <= ts < end_ts):
                continue

            if row["PodName"] != incident["inject_pod"]:
                continue

            try:
                outer = json.loads(row["Log"])
                inner = outer.get("log", "")
                data = (
                    json.loads(inner)
                    if isinstance(inner, str)
                    else inner
                )
            except (ValueError, TypeError, AttributeError):
                continue

            if not isinstance(data, dict):
                continue

            if str(data.get("severity", "")).lower() != "error":
                continue

            message = str(data.get("message", ""))

            pattern = "Failed to connect payment service: connect error."

            if pattern not in message:
                continue

            candidates.append({
                "timestamp_ns": str(ts),
                "trace_id": row["TraceID"],
                "csv_file": filename,
                "csv_row_index": row_idx,
                "message_pattern": pattern,
                "evidence_role": "injected_service_error_candidate",
                "evidence_status": "candidate_unverified"
            })

# 3. 정렬 및 TraceID 검증
candidates.sort(
    key=lambda r: (
        int(r["timestamp_ns"]),
        r["csv_file"],
        r["csv_row_index"]
    )
)

trace_ids = [r["trace_id"] for r in candidates]

if not all(trace_ids):
    raise ValueError("TraceID가 없는 후보가 존재합니다.")

if len(set(trace_ids)) != len(candidates):
    raise ValueError("동일 TraceID에서 중복 후보가 발견됐습니다.")

# 4. JSON 저장
result = {
    "incident_id": "2022-08-22_050113_checkout_exception",
    "incident": incident,
    "investigation_window_seconds": 60,
    "evidence_status": "candidate_unverified",
    "label_basis": "injected pod, time window, severity, message",
    "candidate_count": len(candidates),
    "trace_count": len(set(trace_ids)),
    "evidence_candidates": candidates
}

output_dir = Path(__file__).resolve().parent / "output"
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "checkout_exception_candidates.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("JSON 저장 완료:", output_path)
print("증거 로그 후보:", len(candidates))
print("관련 TraceID:", len(set(trace_ids)))