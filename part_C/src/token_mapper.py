from transformers import AutoTokenizer
import json
from pathlib import Path

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

INPUT_FIELDS = [
    "timestamp",
    "user",
    "location",
    "component",
    "content"
]


def tokenizer_log(text, spans, tokenizer):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False
    )

    input_ids = encoded["input_ids"][0]
    attention_mask = encoded["attention_mask"][0]
    offsets = encoded["offset_mapping"][0].tolist()

    token_map = []

    for token_idx, (token_start, token_end) in enumerate(offsets):
        token_info = {
            "token_idx": token_idx,
            "token_id": int(input_ids[token_idx]),
            "token": tokenizer.convert_ids_to_tokens(
                int(input_ids[token_idx])
            ),
            "start": token_start,
            "end": token_end,
            "line_id": None,
            "row_idx": None,
            "field": None
        }

        # 문자 범위가 없는 토큰은 매핑하지 않음
        if token_start == token_end:
            token_map.append(token_info)
            continue

        # 토큰 범위와 필드값 범위를 비교
        for span in spans:
            overlap_start = max(
                token_start,
                span["start"]
            )

            overlap_end = min(
                token_end,
                span["end"]
            )

            if overlap_start < overlap_end:
                token_info["line_id"] = span["line_id"]
                token_info["row_idx"] = span["row_idx"]
                token_info["field"] = span["field"]
                break

        token_map.append(token_info)

    for token_idx, token_info in enumerate(token_map):
        start = token_info["start"]
        end = token_info["end"]

        if token_info["line_id"] is None and text[start:end] == " \n":
            if token_idx > 0:
                previous = token_map[token_idx - 1]
                token_info["line_id"] = previous["line_id"]
                token_info["row_idx"] = previous["row_idx"]

    return {
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "token_map": token_map
    }


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    input_path = input(
        "window JSONL 파일 경로를 입력하세요: "
    )
    input_path = Path(input_path)

    if not input_path.exists():
        print("입력 파일을 찾을 수 없습니다.")
        return

    output_path = input_path.with_name(
        input_path.stem + "_token_map.jsonl"
    )

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as fin, open(
        output_path,
        "w",
        encoding="utf-8"
    ) as fout:

        for window_idx, line in enumerate(fin):
            window = json.loads(line)

            text = ""
            spans = []

            # 모델 입력 문자열과 외부 span 정보 생성
            for row_idx, row in enumerate(window):
                line_id = int(row["line_id"])

                for field in INPUT_FIELDS:
                    value = str(row.get(field, ""))

                    start = len(text)
                    text += value
                    end = len(text)

                    spans.append({
                        "line_id": line_id,
                        "row_idx": row_idx,
                        "field": field,
                        "start": start,
                        "end": end
                    })

                    text += " "

                text += "\n"

            # 토큰화 및 토큰 매핑
            mapped = tokenizer_log(
                text=text,
                spans=spans,
                tokenizer=tokenizer
            )

            result = {
                "window_idx": window_idx,
                "text": text,
                "input_ids": mapped["input_ids"],
                "attention_mask": mapped["attention_mask"],
                "token_map": mapped["token_map"]
            }

            fout.write(
                json.dumps(
                    result,
                    ensure_ascii=False
                ) + "\n"
            )

            # 확인용 출력
            print(f"\n===== Window {window_idx} =====")
            print("\n모델 입력:")
            print(text)

            print("\n토큰 매핑:")
            for token_info in mapped["token_map"]:
                print(token_info)

    print(f"\n저장 완료: {output_path}")


if __name__ == "__main__":
    main()