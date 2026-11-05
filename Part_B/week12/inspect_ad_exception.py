import csv
import json
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

log_dir = Path(
    r"C:\Users\qkral\projects\Nezha\rca_data\2022-08-22\log"
)

# 공식 장애 주입 정보
inject_ts = 1661154813 * 1_000_000_000
target_pod = "adservice-5f6585d649-fnmft"

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
    "07_53_log.csv",
    "07_54_log.csv",
    "07_55_log.csv"
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

import re

print("\n===== Adservice 실제 로그 형식 확인 =====")

print("\n[장애 발생 전 로그 샘플]")
for record in before[:3]:
    print(repr(record["message"][:250]))

print("\n[장애 발생 후 로그 샘플]")
for record in after[:5]:
    print(repr(record["message"][:250]))


# Severity와 관계없이 메시지 본문에서 오류 키워드 탐색
error_pattern = re.compile(
    r"error|exception|fail|panic|timeout|refused|unavailable",
    re.IGNORECASE
)

keyword_matches = [
    record for record in after
    if error_pattern.search(record["message"])
]

print("\n===== 오류 키워드 탐색 =====")
print("키워드 포함 로그 수:", len(keyword_matches))

for record in keyword_matches[:10]:
    elapsed = (record["ts"] - inject_ts) / 1_000_000_000

    print(
        f"+{elapsed:.3f}초 | "
        f"{record['file']} Row {record['row']} | "
        f"{record['message'][:200]}"
    )    

# ==========================================
# adservice 텍스트 로그의 Severity 분석
# ==========================================

def extract_text_severity(message):
    match = re.search(
        r"^\s*\d{2}:\d{2}:\d{2}(?:\.\d+)?\s+"
        r"(TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\b",
        str(message),
        re.IGNORECASE
    )

    if match:
        return match.group(1).lower()

    return "unknown"


before_levels = Counter(
    extract_text_severity(r["message"]) for r in before
)

after_levels = Counter(
    extract_text_severity(r["message"]) for r in after
)

print("\n===== 실제 텍스트 Severity 비교 =====")
print("장애 전:", dict(before_levels))
print("장애 후:", dict(after_levels))


# ERROR/WARN/FATAL 등이 있는지 확인
bad_levels = {"error", "warn", "warning", "fatal", "critical"}

suspicious = [
    r for r in after
    if extract_text_severity(r["message"]) in bad_levels
]

print("\n실제 ERROR/WARN 로그 수:", len(suspicious))

for r in sorted(suspicious, key=lambda x: x["ts"])[:10]:
    elapsed = (r["ts"] - inject_ts) / 1_000_000_000
    print(f"+{elapsed:.3f}초 | {r['message'][:200]}")


# 장애 발생 후 첫 로그 시각도 확인
if after:
    first = min(after, key=lambda x: x["ts"])
    elapsed = (first["ts"] - inject_ts) / 1_000_000_000

    print(f"\n장애 후 첫 adservice 로그: +{elapsed:.3f}초")
    print(first["message"][:200])    

# ==========================================
# adservice 장애가 다른 서비스에 영향을 주었는지 확인
# ==========================================

# 장애 이후 adservice 로그에 등장한 TraceID
ad_trace_ids = {
    record["trace_id"]
    for record in after
    if record["trace_id"]
}

other_errors = []

for filename in [
    "07_53_log.csv",
    "07_54_log.csv",
    "07_55_log.csv"
]:
    path = log_dir / filename

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row_idx, row in enumerate(reader):
            ts = int(row["TimeUnixNano"])

            # 장애 발생 이후 60초
            if not (inject_ts <= ts < after_end):
                continue

            # adservice 자체는 이미 확인했으므로 제외
            if row["PodName"] == target_pod:
                continue

            log = parse_log(row["Log"])

            if log["severity"] not in (
                "error", "warn", "warning", "fatal", "critical"
            ):
                continue

            other_errors.append({
                "ts": ts,
                "pod": row["PodName"],
                "trace_id": row["TraceID"],
                "message": log["message"],
                "file": filename,
                "row": row_idx,
            })

# adservice와 동일 TraceID인 다른 서비스 오류
related_errors = [
    record for record in other_errors
    if record["trace_id"] in ad_trace_ids
]

print("\n===== 다른 서비스로의 오류 전파 조사 =====")

print("adservice TraceID 수:", len(ad_trace_ids))
print("다른 서비스 ERROR/WARN 수:", len(other_errors))
print("adservice TraceID와 일치하는 ERROR/WARN:", len(related_errors))

print("\n===== 서비스별 ERROR/WARN 수 =====")

for pod, count in Counter(
    record["pod"] for record in other_errors
).items():
    print(f"{pod}: {count}개")

print("\n===== adservice와 TraceID가 연결된 오류 =====")

related_errors.sort(key=lambda r: r["ts"])

for record in related_errors[:15]:
    elapsed = (record["ts"] - inject_ts) / 1_000_000_000

    print(
        f"\n+{elapsed:.3f}초 | "
        f"{record['pod']} | "
        f"{record['file']} Row {record['row']}"
    )
    print("TraceID:", record["trace_id"])
    print("Message:", record["message"][:200])    

# ==========================================
# frontend 오류의 장애 전후 비교 및 시간 검증
# ==========================================

before_frontend_errors = []

for filename in [
    "07_53_log.csv",
    "07_54_log.csv",
    "07_55_log.csv"
]:
    path = log_dir / filename

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            ts = int(row["TimeUnixNano"])

            if not (before_start <= ts < inject_ts):
                continue

            if not row["PodName"].startswith("frontend-"):
                continue

            log = parse_log(row["Log"])

            if log["severity"] in (
                "error", "warn", "warning", "fatal", "critical"
            ):
                before_frontend_errors.append(log["message"])

print("\n===== frontend 장애 전후 ERROR 비교 =====")
print("장애 전 60초 frontend ERROR/WARN:", len(before_frontend_errors))
print("장애 후 60초 frontend ERROR/WARN:", len(other_errors))


# 장애 이후 adservice TraceID별 첫 로그 시간
ad_first_times = {}

for record in after:
    trace_id = record["trace_id"]

    if not trace_id:
        continue

    if trace_id not in ad_first_times:
        ad_first_times[trace_id] = record["ts"]
    else:
        ad_first_times[trace_id] = min(
            ad_first_times[trace_id],
            record["ts"]
        )

# frontend ERROR 발생 전에 같은 TraceID의 adservice 로그가 있었는지
ordered_matches = [
    record for record in related_errors
    if (
        record["trace_id"] in ad_first_times
        and ad_first_times[record["trace_id"]] <= record["ts"]
    )
]

print("\n===== TraceID 시간순 검증 =====")
print("연결된 frontend ERROR:", len(related_errors))
print("그 이전에 adservice 로그가 확인된 ERROR:", len(ordered_matches))


print("\n===== 연결된 frontend ERROR 메시지 패턴 =====")

message_counts = Counter(
    re.sub(
        r"^TraceID:\s*\S+\s+SpanID:\s*\S+\s*",
        "",
        record["message"]
    ).strip()
    for record in related_errors
)

for message, count in message_counts.most_common():
    print(f"{count}회 | {message}")    
    
# ==========================================
# adservice Exception 증거 후보 JSON 저장
# ==========================================

# 기존 분석 결과가 일관적인지 확인
if len(related_errors) != len(ordered_matches):
    raise ValueError("adservice 로그보다 먼저 발생한 frontend 오류가 존재합니다.")

# 공식 장애 주입 정보 읽기
fault_path = log_dir.parent / "2022-08-22-fault_list.json"

with open(fault_path, "r", encoding="utf-8") as f:
    fault_data = json.load(f)

faults = [
    item
    for incidents in fault_data.values()
    for item in incidents
]

incident = next(
    item for item in faults
    if item["inject_type"] == "exception"
    and item["inject_timestamp"] == "1661154813"
)

# 증거 후보 생성
candidates = []

for record in sorted(related_errors, key=lambda r: r["ts"]):
    if "Failed to retrieve ads" not in record["message"]:
        raise ValueError("예상하지 못한 ERROR 메시지가 발견됐습니다.")

    candidates.append({
        "timestamp_ns": str(record["ts"]),
        "trace_id": record["trace_id"],
        "csv_file": record["file"],
        "csv_row_index": record["row"],
        "source_pod": record["pod"],
        "message_pattern": "Failed to retrieve ads",
        "evidence_role": "downstream_error_candidate",
        "evidence_status": "candidate_unverified",
        "injected_pod_prior_log_ns": str(
            ad_first_times[record["trace_id"]]
        )
    })

# 원본 CSV 위치 중복 확인
locations = [
    (r["csv_file"], r["csv_row_index"])
    for r in candidates
]

if len(locations) != len(set(locations)):
    raise ValueError("중복된 CSV 행이 발견됐습니다.")

# 결과 저장
result = {
    "incident_id": "2022-08-22_075333_adservice_exception",
    "incident": incident,
    "investigation_window_seconds": 60,
    "evidence_status": "candidate_unverified",
    "evidence_role": "downstream_error_candidate",
    "label_basis": "frontend error with matching adservice TraceID and temporal order",
    "candidate_count": len(candidates),
    "trace_count": len({r["trace_id"] for r in candidates}),
    "evidence_candidates": candidates
}

output_dir = Path(__file__).resolve().parent / "output"
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "adservice_exception_candidates.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("\n===== adservice 후보 JSON 저장 =====")
print("저장 위치:", output_path)
print("증거 로그 후보:", len(candidates))
print("관련 TraceID:", result["trace_count"])