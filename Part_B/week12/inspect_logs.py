import csv
from pathlib import Path

log_path = Path(
    r"C:\Users\qkral\projects\Nezha\rca_data\2022-08-22\log\03_53_log.csv"
)

with open(log_path, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    # CSV에 어떤 컬럼이 존재하는지 확인
    print("CSV 컬럼:", reader.fieldnames)

    # 앞의 로그 3개만 확인
    for idx, row in enumerate(reader):
        if idx >= 3:
            break

        print(f"\n========== Row {idx} ==========")

        for column, value in row.items():
            print(f"{column}: {str(value)[:200]}")

from datetime import datetime, timezone

# 첫 번째 장애 발생 시각
inject_ts = 1661140434 * 1_000_000_000
target_pod = "frontend-579b9bff58-t2dbm"

# 장애 발생 1분 전 ~ 2분 후
start_ts = inject_ts - 60 * 1_000_000_000
end_ts = inject_ts + 120 * 1_000_000_000

log_dir = log_path.parent

for filename in [
    "03_53_log.csv",
    "03_54_log.csv",
    "03_55_log.csv"
]:
    path = log_dir / filename

    min_ts = None
    max_ts = None
    nearby_count = 0
    target_count = 0

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            ts = int(row["TimeUnixNano"])

            min_ts = ts if min_ts is None else min(min_ts, ts)
            max_ts = ts if max_ts is None else max(max_ts, ts)

            if start_ts <= ts <= end_ts:
                nearby_count += 1

                if row["PodName"] == target_pod:
                    target_count += 1

    def format_time(ns):
        return datetime.fromtimestamp(
            ns // 1_000_000_000,
            tz=timezone.utc
        ).strftime("%H:%M:%S")

    print(f"\n파일: {filename}")
    print(f"시간 범위: {format_time(min_ts)} ~ {format_time(max_ts)}")
    print(f"장애 주변 로그: {nearby_count}개")
    print(f"대상 Pod 로그: {target_count}개")            