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
inject_ts = 1661141915 * 1_000_000_000
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

for filename in [
    "04_18_log.csv",
    "04_19_log.csv",
    "04_20_log.csv"
]:
    path = log_dir / filename

    if not path.exists():
        print(f"파일 없음: {filename}")
        continue

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

print("Exception 장애 이후 frontend 로그 수:", len(records))

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

# ==========================================
# 장애 발생 전후 로그 비교
# ==========================================

before_start = inject_ts - 60 * 1_000_000_000
before_records = []

for filename in [
    "04_18_log.csv",
    "04_19_log.csv",
    "04_20_log.csv"
]:
    path = log_dir / filename

    if not path.exists():
        continue

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row_idx, row in enumerate(reader):
            ts = int(row["TimeUnixNano"])

            # 장애 발생 이전 60초 + 동일 Pod
            if not (before_start <= ts < inject_ts):
                continue

            if row["PodName"] != target_pod:
                continue

            severity, message = extract_message(row["Log"])

            before_records.append(
                (ts, filename, row_idx, severity, message)
            )

before_errors = [
    record for record in before_records
    if str(record[3]).lower() in (
        "error", "warn", "warning", "fatal", "critical"
    )
]

print("\n===== 장애 발생 전후 비교 =====")

print(f"장애 전 60초: {len(before_records)}줄")
print(f"장애 전 ERROR/WARN: {len(before_errors)}줄")

print(f"장애 후 60초: {len(records)}줄")
print(f"장애 후 ERROR/WARN: {len(error_logs)}줄")


# 장애 직후 첫 번째 오류 발생 시각 확인
print("\n===== 장애 이후 첫 번째 오류 10개 =====")

for ts, filename, row_idx, severity, message in error_logs[:10]:
    elapsed = (ts - inject_ts) / 1_000_000_000

    print(
        f"+{elapsed:.3f}초 | "
        f"{filename} Row {row_idx} | "
        f"{normalize_message(message)[:100]}"
    )    

from collections import defaultdict

# 1. ERROR 로그의 원본 CSV 위치 기록
error_lookup = {}

for ts, filename, row_idx, severity, message in error_logs:
    error_lookup[(filename, row_idx)] = {
        "ts": ts,
        "message": normalize_message(message),
    }

# 2. CSV에서 해당 ERROR의 TraceID 확인
trace_groups = defaultdict(list)

for filename in sorted({key[0] for key in error_lookup}):
    path = log_dir / filename

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row_idx, row in enumerate(reader):
            key = (filename, row_idx)

            if key not in error_lookup:
                continue

            trace_id = row["TraceID"]

            # TraceID가 없다면 별개 로그로 취급
            if not trace_id:
                trace_id = f"UNKNOWN_{filename}_{row_idx}"

            trace_groups[trace_id].append({
                "ts": error_lookup[key]["ts"],
                "filename": filename,
                "row_idx": row_idx,
                "message": error_lookup[key]["message"],
            })

# 3. 결과 출력
print("\n===== ERROR TraceID 분석 =====")
print("ERROR 로그 수:", len(error_logs))
print("서로 다른 TraceID 수:", len(trace_groups))

# 가장 이른 ERROR를 가진 Trace부터 출력
sorted_groups = sorted(
    trace_groups.items(),
    key=lambda item: min(log["ts"] for log in item[1])
)

for trace_id, logs in sorted_groups[:5]:
    print(f"\nTraceID: {trace_id}")
    print(f"오류 로그 {len(logs)}개")

    for log in sorted(logs, key=lambda x: x["ts"]):
        elapsed = (log["ts"] - inject_ts) / 1_000_000_000

        print(
            f"  +{elapsed:.3f}초 | "
            f"{log['filename']} Row {log['row_idx']} | "
            f"{log['message']}"
        )    

# ==========================================
# 전체 ERROR 로그의 상세 패턴 분석
# ==========================================

pattern_counts = Counter()
trace_error_counts = Counter()

for filename in sorted({key[0] for key in error_lookup}):
    path = log_dir / filename

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row_idx, row in enumerate(reader):
            if (filename, row_idx) not in error_lookup:
                continue

            outer = json.loads(row["Log"])
            inner = outer.get("log", "")

            data = json.loads(inner) if isinstance(inner, str) else inner

            if not isinstance(data, dict):
                continue

            message = normalize_message(data.get("message", ""))
            error = data.get("error", "<없음>")
            request_path = data.get("http.req.path", "<없음>")

            pattern_counts[(message, error, request_path)] += 1
            trace_error_counts[row["TraceID"]] += 1

print("\n===== 전체 ERROR 상세 패턴 =====")

for (message, error, request_path), count in pattern_counts.most_common():
    print(f"\n발생 횟수: {count}")
    print(f"메시지: {message}")
    print(f"오류 내용: {error}")
    print(f"요청 경로: {request_path}")

print("\n===== Trace별 ERROR 개수 분포 =====")

distribution = Counter(trace_error_counts.values())

for error_count, trace_count in sorted(distribution.items()):
    print(f"ERROR {error_count}개인 Trace: {trace_count}개")        