from pathlib import Path
import csv
import json
import re

def load_thunderbird_gt(log_path, metadata_path, num_lines):

    log_path = Path(log_path)
    metadata_path = Path(metadata_path)

    # 1. 파일명에서 chunk, subchunk 번호 추출
    match = re.search(
        r"_chunk(\d+)_sub(\d+)\.txt$",
        log_path.name
    )

    if match is None:
        raise ValueError(
            f"subchunk 파일명 형식을 인식할 수 없습니다: {log_path.name}"
        )

    chunk_idx = int(match.group(1))
    sub_idx = int(match.group(2))

    # 2. Metadata CSV에서 해당 chunk 찾기
    with metadata_path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        reader = csv.DictReader(f)

        chunk_row = next(
            (
                row
                for row in reader
                if row["chunk"] == f"chunk{chunk_idx}"
            ),
            None
        )

    if chunk_row is None:
        raise ValueError(
            f"metadata에서 chunk{chunk_idx}를 찾지 못했습니다."
        )

    chunk_start_line = int(chunk_row["start_line"])
    alerts = json.loads(chunk_row["alerts"])

    # 3. 이전 subchunk들의 라인 수 누적
    subchunk_offset = 0

    for prev_sub_idx in range(sub_idx):

        prev_name = re.sub(
            r"_sub\d+\.txt$",
            f"_sub{prev_sub_idx}.txt",
            log_path.name
        )

        prev_sub_path = log_path.with_name(prev_name)

        prev_lines = prev_sub_path.read_text(
            encoding="utf-8"
        ).splitlines()

        subchunk_offset += len(prev_lines)

    subchunk_start_line = (
        chunk_start_line + subchunk_offset
    )

    # 4. 현재 subchunk에 해당하는 GT만 추출
    gt_lines = set()

    for alert_line_str in alerts:
        alert_line = int(alert_line_str)

        local_line_idx = (
            alert_line - subchunk_start_line
        )

        if 0 <= local_line_idx < num_lines:
            gt_lines.add(local_line_idx)

    return gt_lines