import csv
import json
from pathlib import Path
from transformers import AutoTokenizer

log_path = Path(r"C:\datasets\thunderbird\Thunderbird.log")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

output_dir = Path("Part_A/week6/thunderbird/chunks")
output_dir.mkdir(parents=True, exist_ok=True)

current_window_lines = []
alert_count = 0
alert_line = {}
line_idx = 0
current_window_id = None

MAX_TOKENS = 32768
chunk = {}
chunk_idx = 0
token_size = 0

chunk_alerts = {}


with log_path.open(encoding="utf-8", errors="replace") as file:
    for line_no, line in enumerate(file):
        log_lst = line.split()

        timestamp = int(log_lst[1])
        new_window_id = timestamp // 3600

        # 첫 번째 로그
        if current_window_id is None:
            current_window_id = new_window_id

        # 새로운 1시간 window에 진입
        if new_window_id != current_window_id:

            # 방금 끝난 window가 우리가 찾는 window라면 확인
            if current_window_id == 314927:
                for target_idx, target_line in enumerate(current_window_lines):

                    # label과 실제 모델 입력 분리
                    label, _, clean_line = target_line.partition(" ")
                    

                    # label을 제거한 로그로 토큰 계산
                    token_ids = tokenizer(
                        clean_line,
                        add_special_tokens=False
                    )["input_ids"]

                    line_token_size = len(token_ids)

                    if token_size + line_token_size > MAX_TOKENS:
                        chunk_idx += 1
                        token_size = 0

                    chunk_name = f"chunk{chunk_idx}"

                    if chunk_name not in chunk:
                        chunk[chunk_name] = []

                    # 실제 chunk에는 label이 제거된 로그 저장
                    chunk[chunk_name].append(clean_line)

                    token_size += line_token_size

                    if chunk_name not in chunk_alerts:
                        chunk_alerts[chunk_name] = {}

                    if target_idx in alert_line:
                        chunk_alerts[chunk_name][target_idx] = alert_line[target_idx]

                for chunk_name, chunk_lines in chunk.items():
                    print(
                        chunk_name,
                        "logs:",
                        len(chunk_lines),
                        "tokens:",
                        len(
                            tokenizer(
                                "".join(chunk_lines),
                                add_special_tokens=False
                            )["input_ids"]
                        )
                    )
                for chunk_name in chunk:
                    print(
                        chunk_name,
                        "alerts:",
                        len(chunk_alerts[chunk_name]),
                        chunk_alerts[chunk_name]
                    )
                    
                for chunk_name, chunk_lines in chunk.items():
                    output_path = output_dir / f"window_314927_{chunk_name}.txt"

                    output_path.write_text(
                        "".join(chunk_lines),
                        encoding="utf-8"
                    )
                    
                break

            # 다음 window 준비
            current_window_id = new_window_id
            current_window_lines = []
            alert_count = 0
            alert_line = {}
            line_idx = 0

        # 현재 window의 alert 기록
        if log_lst[0] != "-":
            alert_count += 1
            alert_line[line_idx] = log_lst[0]

        current_window_lines.append(line)
        line_idx += 1
        
metadata_path = output_dir / "window_314927_metadata.csv"

with metadata_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
    writer = csv.writer(csv_file)

    writer.writerow([
        "chunk",
        "start_line",
        "end_line",
        "log_count",
        "token_count",
        "alert_count",
        "alerts"
    ])

    start_line = 0

    for chunk_name, chunk_lines in chunk.items():
        end_line = start_line + len(chunk_lines) - 1

        token_count = len(
            tokenizer(
                "".join(chunk_lines),
                add_special_tokens=False
            )["input_ids"]
        )

        writer.writerow([
            chunk_name,
            start_line,
            end_line,
            len(chunk_lines),
            token_count,
            len(chunk_alerts[chunk_name]),
            json.dumps(chunk_alerts[chunk_name], ensure_ascii=False)
        ])

        start_line = end_line + 1