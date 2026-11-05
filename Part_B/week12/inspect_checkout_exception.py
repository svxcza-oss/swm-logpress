import csv
import json
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

log_dir = Path(
    r"C:\Users\qkral\projects\Nezha\rca_data\2022-08-22\log"
)

# 공식 장애 주입 정보
inject_ts = 1661144473 * 1_000_000_000
target_pod = "checkoutservice-578fcf4766-9csqn"

before_start = inject_ts - 60 * 1_000_000_000
after_end = inject_ts + 60 * 1_000_000_000


def parse_log(raw_log):
    try:
        outer = json.loads(raw_log)
        inner = outer.get("log", "")

        try:
            data = json.loads(inner) if isinstance(inner, str) else inner
        except json.JSONDecodeError:
            data = inner

        if isinstance(data, dict):
            return {
                "severity": str(data.get("severity", "-")).lower(),
                "message": str(data.get("message", "")),
                "error": str(data.get("error", "")),
            }

        return {
            "severity": "-",
            "message": str(data),
            "error": "",
        }

    except (ValueError, TypeError, AttributeError):
        return {
            "severity": "-",
            "message": str(raw_log),
            "error": "",
        }


def format_time(ts):
    return datetime.fromtimestamp(
        ts // 1_000_000_000,
        tz=timezone.utc
    ).strftime("%H:%M:%S")


before = []
after = []
file_ranges = []

# CSV 탐색
for filename in [
    "05_01_log.csv",
    "05_02_log.csv",
    "05_03_log.csv"
]:
    path = log_dir / filename

    if not path.exists():
        print("파일 없음:", filename)
        continue

    min_ts = None
    max_ts = None

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row_idx, row in enumerate(reader):
            ts = int(row["TimeUnixNano"])

            min_ts = ts if min_ts is None else min(min_ts, ts)
            max_ts = ts if max_ts is None else max(max_ts, ts)

            if row["PodName"] != target_pod:
                continue

            if not (before_start <= ts < after_end):
                continue

            log = parse_log(row["Log"])

            record = {
                "ts": ts,
                "file": filename,
                "row": row_idx,
                "trace_id": row["TraceID"],
                **log
            }

            if ts < inject_ts:
                before.append(record)
            else:
                after.append(record)

    if min_ts is not None:
        file_ranges.append((min_ts, max_ts))
        print(
            f"{filename}: "
            f"{format_time(min_ts)} ~ {format_time(max_ts)}"
        )

# 시간 범위 확인
if file_ranges:
    earliest = min(start for start, _ in file_ranges)
    latest = max(end for _, end in file_ranges)

    if earliest > before_start:
        print("주의: 장애 전 60초 구간의 시작 부분이 누락됐을 수 있음")

    if latest < after_end:
        print("주의: 장애 후 60초 구간의 끝 부분이 누락됐을 수 있음")

# 장애 전후 분석
before_errors = [
    r for r in before
    if r["severity"] in ("error", "warn", "warning", "fatal")
]

after_errors = [
    r for r in after
    if r["severity"] in ("error", "warn", "warning", "fatal")
]

print("\n===== 장애 전후 비교 =====")
print("장애 전 전체 로그:", len(before))
print("장애 전 ERROR/WARN:", len(before_errors))
print("장애 후 전체 로그:", len(after))
print("장애 후 ERROR/WARN:", len(after_errors))

print("\n===== 장애 이후 Severity 분포 =====")
for severity, count in Counter(r["severity"] for r in after).items():
    print(f"{severity}: {count}개")

# 오류 메시지 출력
after_errors.sort(key=lambda r: r["ts"])

print("\n===== 첫 번째 오류 로그 15개 =====")

for record in after_errors[:15]:
    elapsed = (record["ts"] - inject_ts) / 1_000_000_000

    print(
        f"\n+{elapsed:.3f}초 | "
        f"{record['file']} Row {record['row']}"
    )
    print("TraceID:", record["trace_id"])
    print("Message:", record["message"][:200])

    if record["error"]:
        print("Error:", record["error"][:200])

import re

# TraceID / SpanID 제거
def normalize_message(message):
    return re.sub(
        r"^TraceID:\s*\S+\s+SpanID:\s*\S+\s*",
        "",
        message
    ).strip()


# ERROR 메시지별 개수
message_counts = Counter(
    normalize_message(r["message"])
    for r in after_errors
)

print("\n===== ERROR 메시지별 개수 =====")

for message, count in message_counts.most_common():
    print(f"{count}회 | {message}")


# TraceID별 ERROR 개수
trace_counts = Counter(
    r["trace_id"]
    for r in after_errors
    if r["trace_id"]
)

print("\n===== TraceID 분석 =====")
print("ERROR 총 개수:", len(after_errors))
print("서로 다른 TraceID 수:", len(trace_counts))

distribution = Counter(trace_counts.values())

for error_count, trace_count in sorted(distribution.items()):
    print(f"ERROR {error_count}개인 Trace: {trace_count}개")        