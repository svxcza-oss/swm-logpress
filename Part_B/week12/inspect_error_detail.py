import csv
import json
from pathlib import Path
from pprint import pprint

path = Path(
    r"C:\Users\qkral\projects\Nezha\rca_data\2022-08-22\log\04_19_log.csv"
)

# 첫 번째 TraceID에서 발견한 ERROR 2개
target_rows = {9391, 9390}

with open(path, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    for row_idx, row in enumerate(reader):
        if row_idx not in target_rows:
            continue

        outer = json.loads(row["Log"])
        inner = outer.get("log", "")

        try:
            data = json.loads(inner) if isinstance(inner, str) else inner
        except json.JSONDecodeError:
            data = inner

        print(f"\n===== CSV Row {row_idx} =====")

        if isinstance(data, dict):
            print("로그 필드:", list(data.keys()))

            # 메시지 및 오류 관련 필드만 확인
            for key, value in data.items():
                if any(word in key.lower() for word in (
                    "message", "severity", "error",
                    "exception", "stack"
                )):
                    print(f"{key}: {str(value)[:500]}")
        else:
            print(str(data)[:500])

            