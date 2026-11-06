import csv
import json
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

log_dir = Path(
    r"C:\Users\qkral\projects\Nezha\rca_data\2022-08-22\log"
)

# 첫 번째 오류가 발생했던 TraceID
target_trace = "d9dda648d9d21b19f2140c3768f9da23"

records = []

for filename in [
    "05_01_log.csv",
    "05_02_log.csv",
    "05_03_log.csv"
]:
    path = log_dir / filename

    if not path.exists():
        continue

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row_idx, row in enumerate(reader):

            if row["TraceID"] != target_trace:
                continue

            # 실제 로그 메시지 추출
            try:
                outer = json.loads(row["Log"])
                inner = outer.get("log", "")
                data = json.loads(inner) if isinstance(inner, str) else inner

                severity = data.get("severity", "-")
                message = data.get("message", str(data))

            except (ValueError, TypeError, AttributeError):
                severity = "-"
                message = row["Log"]

            records.append({
                "ts": int(row["TimeUnixNano"]),
                "pod": row["PodName"],
                "severity": severity,
                "message": message,
                "file": filename,
                "row": row_idx
            })

# 발생 시간순 정렬
records.sort(key=lambda x: x["ts"])

print("===== TraceID 전체 로그 분석 =====")
print("TraceID:", target_trace)
print("전체 로그 수:", len(records))

# 서비스별 로그 수
pod_counts = Counter(record["pod"] for record in records)

print("\n===== Pod별 로그 수 =====")
for pod, count in pod_counts.items():
    print(f"{pod}: {count}개")

print("\n===== 시간순 로그 =====")

for record in records:
    ts = record["ts"]
    seconds = ts // 1_000_000_000
    milliseconds = (ts % 1_000_000_000) // 1_000_000

    time_str = datetime.fromtimestamp(
        seconds, tz=timezone.utc
    ).strftime("%H:%M:%S") + f".{milliseconds:03d}"

    print(
        f"{time_str} | "
        f"{record['pod'].split('-')[0]} | "
        f"{record['severity']} | "
        f"{record['message']}"
    )
