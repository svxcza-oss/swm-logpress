import json
from pathlib import Path

# 1. Nezha 장애 정보 파일 경로
fault_path = Path(
    r"C:\Users\qkral\projects\Nezha\rca_data\2022-08-22\2022-08-22-fault_list.json"
)

# 2. JSON 읽기
with open(fault_path, "r", encoding="utf-8") as f:
    fault_data = json.load(f)

# 3. 시간별 장애 목록을 하나의 리스트로 합치기
faults = []

for hour, incidents in fault_data.items():
    for incident in incidents:
        faults.append(incident)

# 4. 전체 장애 수 출력
print("전체 장애 기록 수:", len(faults))

# 5. 처음 5개 출력
for incident in faults[:5]:
    print(
        incident["inject_time"],
        "|", incident["inject_pod"],
        "|", incident["inject_type"]
    )

from collections import Counter

# 장애 유형별 개수 계산
fault_types = Counter()

for incident in faults:
    fault_type = incident["inject_type"]
    fault_types[fault_type] += 1

print("\n장애 유형별 기록 수:")

for fault_type, count in fault_types.items():
    print(f"{fault_type}: {count}건")

print("\n===== Exception 장애 목록 =====")

for incident in faults:
    if incident["inject_type"] == "exception":
        print(
            f"발생 시각: {incident['inject_time']}"
        )
        print(
            f"대상 Pod: {incident['inject_pod']}"
        )
        print(
            f"Timestamp: {incident['inject_timestamp']}"
        )
        print("-" * 40)    