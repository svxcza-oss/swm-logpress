from transformers import AutoTokenizer
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Span:
    line_id: int
    row_idx: int
    field: str
    start: int
    end: int


@dataclass
class TokenInfo:
    token_idx: int
    token_id: int
    token: str
    start: int
    end: int
    line_id: Optional[int] = None
    row_idx: Optional[int] = None
    field: Optional[str] = None


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

INPUT_FIELDS = [
    "timestamp",
    "user",
    "location",
    "component",
    "content"
]


def map_token_to_span(token_start, token_end, spans):
    for span in spans:
        overlap_start = max(token_start, span.start)
        overlap_end = min(token_end, span.end)

        if overlap_start < overlap_end:
            return span

    return None


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    input_path = input("window JSONL 파일 경로를 입력하세요: ")
    input_path = Path(input_path)

    if not input_path.exists():
        print("입력 파일을 찾을 수 없습니다.")
        return

    output_path = input_path.with_name(
        input_path.stem + "_token_map.jsonl"
    )

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for window_idx, line in enumerate(fin):
            window = json.loads(line)

            text = ""
            spans = []

            # 모델 입력 문자열과 span 정보 생성
            for row_idx, row in enumerate(window):
                line_id = int(row["line_id"])

                # [수정 1]
                # 현재 로그 line 전체의 시작 위치
                line_start = len(text)

                for field in INPUT_FIELDS:
                    value = str(row.get(field, ""))

                    start = len(text)
                    text += value
                    end = len(text)

                    spans.append(
                        Span(
                            line_id=line_id,
                            row_idx=row_idx,
                            field=field,
                            start=start,
                            end=end
                        )
                    )

                    text += " "

                text += "\n"

                # [수정 2]
                # 공백 / newline token도 해당 line으로 매핑하기 위해
                # 전체 line span 추가
                spans.append(
                    Span(
                        line_id=line_id,
                        row_idx=row_idx,
                        field="line",
                        start=line_start,
                        end=len(text)
                    )
                )

            # 토큰화 및 token -> line/field 매핑
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
                "token_map": mapped["token_map"],

                # 기존 label 정보는 유지
                "line_labels": [
                    {
                        "line_id": int(row["line_id"]),
                        "label": str(row.get("label", "-"))
                    }
                    for row in window
                ],

                # [수정 3]
                # Anchor Hypothesis 평가에 사용할 line metadata
                "line_meta": [
                    {
                        "line_id": int(row["line_id"]),

                        # ERROR / WARN / INFO
                        "level": str(
                            row.get("level", row.get("label", "-"))
                        ),

                        # root / propagation / symptom / none
                        "evidence_role": str(
                            row.get("evidence_role", "none")
                        )
                    }
                    for row in window
                ]
            }

            fout.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )

            # 확인용 출력
            print(f"\n===== Window {window_idx} =====")

            print("\n모델 입력:")
            print(text)

            print("\n토큰 매핑:")
            for token_info in mapped["token_map"]:
                print(token_info)

    print(f"\n저장 완료: {output_path}")


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
        token_info = TokenInfo(
            token_idx=token_idx,
            token_id=int(input_ids[token_idx]),
            token=tokenizer.convert_ids_to_tokens(
                int(input_ids[token_idx])
            ),
            start=token_start,
            end=token_end
        )

        # 문자 범위가 없는 토큰은 매핑하지 않음
        if token_start == token_end:
            token_map.append(token_info)
            continue

        mapped_span = map_token_to_span(
            token_start,
            token_end,
            spans
        )

        if mapped_span is not None:
            token_info.line_id = mapped_span.line_id
            token_info.row_idx = mapped_span.row_idx
            token_info.field = mapped_span.field

        token_map.append(token_info)

    # [수정 4]
    # 기존 " \n" 전용 예외처리는 제거.
    # 전체 line span이 있으므로 공백/newline도 자동으로 line에 매핑됨.

    return {
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "token_map": [
            asdict(token_info)
            for token_info in token_map
        ]
    }


if __name__ == "__main__":
    main()