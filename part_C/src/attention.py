from transformers import AutoModelForCausalLM
from cross_line import convert_to_line_attention
from pathlib import Path
import torch
import json

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


def main():
    input_path = input(
        "token_map JSONL 파일 경로를 입력하세요: "
    )
    input_path = Path(input_path)

    if not input_path.exists():
        print("입력 파일을 찾을 수 없습니다.")
        return

   
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    print(f"사용 장치: {device}")

    model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    attn_implementation="eager",
    dtype=torch.float16 if device.type == "xpu" else torch.float32,
     ).to(device)
    model.eval()

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as fin:
        line_score, line_ids = token_attention(
            fin,
            model,
            device
        )

    if line_score is None:
        print("분석할 데이터가 없습니다.")
        return

    print("\nLine ID 순서:")
    print(line_ids)

    print("\nLine attention score:")
    print(line_score)


def token_attention(fin, model, device):
    for window_idx, line in enumerate(fin):
        data = json.loads(line)
        for token_info in data["token_map"]:
         if token_info["line_id"] is None:
          start = token_info["start"]
          end = token_info["end"]

          print(
            "미매핑 토큰:",
            token_info["token_idx"],
            repr(data["text"][start:end])
          )

        input_ids = torch.tensor(
            [data["input_ids"]],
            dtype=torch.long,
            device=device
        )

        attention_mask = torch.tensor(
            [data["attention_mask"]],
            dtype=torch.long,
            device=device
        )

        print(f"\n윈도우 번호: {window_idx}")
        print(f"input_ids shape: {input_ids.shape}")

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                use_cache=False,
                return_dict=True
            )

        line_attentions, line_ids = convert_to_line_attention(
            layer_attentions=outputs.attentions,
            token_map=data["token_map"]
        )
        print("라인에 매핑되지 않은 토큰 수:",
        sum(t["line_id"] is None for t in data["token_map"]))

        print("라인 attention의 행별 합:",
      line_attentions.mean(dim=(0, 1)).sum(dim=-1))

        print(
            "전체 line attention shape:",
            line_attentions.shape
        )

        # 마지막 레이어 선택 후 모든 헤드 평균
        line_score = line_attentions[-1].mean(dim=0)

        # 테스트 단계에서는 첫 번째 윈도우만 반환
        return line_score, line_ids

    # JSONL 파일이 비어 있는 경우
    return None, []


if __name__ == "__main__":
    main()