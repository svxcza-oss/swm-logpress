from collections import Counter
import re
import csv
import json
from pathlib import Path
from datetime import datetime, timezone

log_dir = Path(
    r"C:\Users\qkral\projects\Nezha\rca_data\2022-08-22\log"
)

# 첫 번째 장애 정보
inject_ts = 1661140434 * 1_000_000_000
target_pod = "frontend-579b9bff58-t2dbm"

# 장애 발생 후 60초만 조사
end_ts = inject_ts + 60 * 1_000_000_000

def extract_message(raw_log):
    try:
        outer = json.loads(raw_log)
        inner = outer.get("log", "")

        data = json.loads(inner) if isinstance(inner, str) else inner

        if isinstance(data, dict):
            severity = data.get("severity", "-")
            message = data.get("message", str(data))
            return severity, message

        return "-", str(data)

    except (ValueError, TypeError, AttributeError):
        return "-", str(raw_log)


records = []

for filename in ["03_54_log.csv", "03_55_log.csv"]:
    path = log_dir / filename

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row_idx, row in enumerate(reader):
            ts = int(row["TimeUnixNano"])

            # 장애 발생 이후 + 동일 Pod만
            if not (inject_ts <= ts < end_ts):
                continue

            if row["PodName"] != target_pod:
                continue

            severity, message = extract_message(row["Log"])

            records.append((
                ts, filename, row_idx, severity, message
            ))

# 실제 발생 시각순으로 출력
records.sort(key=lambda x: x[0])

print("장애 이후 frontend 로그 수:", len(records))

for ts, filename, row_idx, severity, message in records[:15]:
    time_str = datetime.fromtimestamp(
        ts / 1_000_000_000,
        tz=timezone.utc
    ).strftime("%H:%M:%S.%f")[:-3]

    print(
        f"{time_str} | {severity} | "
        f"{filename} Row {row_idx} | "
        f"{str(message)[:160]}"
    )
    
# ==========================================
# 장애 이후 로그 분석
# ==========================================

# 1. Severity별 로그 개수
severity_counts = Counter(
    str(record[3]).lower() for record in records
)

print("\n===== Severity별 로그 수 =====")

for severity, count in severity_counts.items():
    print(f"{severity}: {count}개")


# 2. TraceID, SpanID를 제거한 메시지별 빈도
def normalize_message(message):
    message = re.sub(
        r"TraceID:\s*\S+\s*SpanID:\s*\S+",
        "",
        str(message)
    )
    return message.strip()


message_counts = Counter(
    normalize_message(record[4])
    for record in records
)

print("\n===== 가장 자주 발생한 메시지 TOP 10 =====")

for message, count in message_counts.most_common(10):
    print(f"{count}회 | {message[:150]}")


# 3. Error / Warning 로그만 따로 확인
print("\n===== Error / Warning 로그 =====")

error_logs = [
    record for record in records
    if str(record[3]).lower() in (
        "error", "warn", "warning", "fatal", "critical"
    )
]

print("오류 및 경고 로그 수:", len(error_logs))

for ts, filename, row_idx, severity, message in error_logs[:15]:
    print(
        f"{severity} | {filename} Row {row_idx} | "
        f"{normalize_message(message)[:150]}"
    )    