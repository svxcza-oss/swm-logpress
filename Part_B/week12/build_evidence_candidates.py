import csv
import json
from pathlib import Path
from collections import defaultdict

# 1. 데이터 경로
data_dir = Path(
    r"C:\Users\qkral\projects\Nezha\rca_data\2022-08-22"
)

fault_path = data_dir / "2022-08-22-fault_list.json"
log_dir = data_dir / "log"

# 2. 공식 장애 정보에서 분석 대상 찾기
with open(fault_path, "r", encoding="utf-8") as f:
    fault_data = json.load(f)

faults = [
    incident
    for hour_incidents in fault_data.values()
    for incident in hour_incidents
]

incident = next(
    item for item in faults
    if item["inject_time"] == "2022-08-22 04:18:35"
    and item["inject_type"] == "exception"
)

inject_ts = int(incident["inject_timestamp"]) * 1_000_000_000
end_ts = inject_ts + 60 * 1_000_000_000

# 3. 장애 이후 ERROR 로그 후보 추출
candidates = []
searched_files = []

for filename in [
    "04_18_log.csv",
    "04_19_log.csv",
    "04_20_log.csv"
]:
    path = log_dir / filename

    if not path.exists():
        print(f"주의: 파일 없음 - {filename}")
        continue

    searched_files.append(filename)

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
                data = json.loads(inner) if isinstance(inner, str) else inner
            except (ValueError, TypeError, AttributeError):
                continue

            if not isinstance(data, dict):
                continue

            if str(data.get("severity", "")).lower() != "error":
                continue

            if data.get("http.req.path") != "/cart/checkout":
                continue

            message = str(data.get("message", ""))

            if "Failed to complete the order" in message:
                pattern = "Failed to complete the order"

            elif (
                "Request error" in message
                and data.get("error")
                == "failed to complete the order: Place order error."
            ):
                pattern = "Request error"

            else:
                continue

            candidates.append({
                "timestamp_ns": str(ts),
                "trace_id": row["TraceID"],
                "csv_file": filename,
                "csv_row_index": row_idx,
                "message_pattern": pattern,
                "evidence_status": "candidate_unverified"
            })

# 4. 각 TraceID에 두 오류가 모두 존재하는지 검증
trace_groups = defaultdict(list)

for record in candidates:
    trace_groups[record["trace_id"]].append(record)

for trace_id, logs in trace_groups.items():
    patterns = {log["message_pattern"] for log in logs}

    if len(logs) != 2 or patterns != {
        "Failed to complete the order",
        "Request error"
    }:
        raise ValueError(f"예상과 다른 TraceID: {trace_id}")

candidates.sort(
    key=lambda item: (
        int(item["timestamp_ns"]),
        item["csv_file"],
        item["csv_row_index"]
    )
)

# 5. JSON 저장
result = {
    "incident_id": "2022-08-22_041835_frontend_exception",
    "incident": incident,
    "investigation_window_seconds": 60,
    "searched_csv_files": searched_files,
    "evidence_status": "candidate_unverified",
    "evidence_rule": "frontend exception checkout ERROR pattern",
    "candidate_count": len(candidates),
    "trace_count": len(trace_groups),
    "evidence_candidates": candidates
}

output_dir = Path(__file__).resolve().parent / "output"
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "frontend_exception_candidates.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("JSON 저장 완료:", output_path)
print("증거 로그 후보:", len(candidates))
print("관련 TraceID:", len(trace_groups))