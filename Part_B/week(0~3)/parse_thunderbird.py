"""Thunderbird 로그 스트리밍 파서 (2k 개발용, 2억 줄 확장 대비).

포맷: label timestamp date user month day time location component(pid): content
component 필드는 공백으로 구분된 단일 토큰이지만 내부에 콜론이 중첩될 수 있음
(예: "audit(1131538222.234:0):", "ioctl32(fdisk:515):"). 콜론 기준 정규식 대신
공백 기준으로 9번째 토큰을 통째로 집어 trailing ':'만 벗겨내는 방식으로 처리.
"""

import re

ERROR_KEYWORDS = re.compile(r"error|fail|exception", re.IGNORECASE)

FIELD_NAMES = ("label", "timestamp", "date", "user", "month", "day", "time", "location")


def parse_line(line):
    """한 줄을 dict로 파싱. 형식이 맞지 않으면 None."""
    parts = line.split(maxsplit=8)
    if len(parts) < 9:
        return None

    record = dict(zip(FIELD_NAMES, parts[:8]))
    rest = parts[8]

    comp_content = rest.split(maxsplit=1)
    component_token = comp_content[0]
    if component_token.endswith(":"):
        record["component"] = component_token[:-1]
        record["content"] = comp_content[1] if len(comp_content) > 1 else ""
    else:
        # component 뒤 콜론이 없는 비정형 라인 -> component 없음, 나머지 전부 content
        record["component"] = ""
        record["content"] = rest

    # Thunderbird에는 근본원인(root cause) 라벨이 없음. alert_tag(label)가 '-'가
    # 아니면 "이상(anomaly)으로 표시된 줄"이라는 뜻일 뿐, 원인이라는 의미가 아님.
    record["is_anomaly"] = record["label"] != "-"
    record["level"] = "ERROR" if ERROR_KEYWORDS.search(record["content"]) else "INFO"

    return record


def parse_thunderbird(path):
    """로그 파일을 한 줄씩 읽어 파싱 결과를 순차적으로 내보내는 제너레이터.
    전체를 메모리에 올리지 않으므로 대용량 파일에도 동일하게 적용 가능."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            record = parse_line(line)
            if record is None:
                yield {"lineno": lineno, "raw": line, "parse_error": True}
                continue
            record["lineno"] = lineno
            record["parse_error"] = False
            yield record


if __name__ == "__main__":
    import sys
    from collections import Counter

    path = sys.argv[1] if len(sys.argv) > 1 else "Thunderbird_2k.log"

    total = 0
    anomaly = 0
    error_level = 0
    parse_errors = []

    for record in parse_thunderbird(path):
        total += 1
        if record["parse_error"]:
            parse_errors.append(record)
            continue
        if record["is_anomaly"]:
            anomaly += 1
        if record["level"] == "ERROR":
            error_level += 1

    print(f"total lines: {total}")
    print(f"parse errors: {len(parse_errors)}")
    print(f"is_anomaly (label != '-'): {anomaly}")
    print(f"level == ERROR (keyword match): {error_level}")

    if parse_errors:
        print("\n=== parse error samples ===")
        for r in parse_errors[:5]:
            print(r["lineno"], r["raw"])
