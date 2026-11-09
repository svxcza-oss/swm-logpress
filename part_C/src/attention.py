from transformers import AutoModelForCausalLM
from cross_line import convert_to_line_attention
from token_mapper import TokenInfo
from token_attention__probe import save_token_probe
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

    output_path = input_path.with_name(
        input_path.stem + "_line_attention.pt"
    )

    # 사용 가능한 장치 선택
    if torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"사용 장치: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        attn_implementation="eager",
        dtype=(
            torch.float16
            if device.type in ["xpu", "cuda"]
            else torch.float32
        )
    ).to(device)

    model.eval()

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as fin:

        results = token_attention(
            fin,
            model,
            device
        )

    torch.save(
        results,
        output_path
    )

    print(f"\n저장 완료: {output_path}")


def token_attention(fin, model, device):

    # [수정 1]
    # 기존에는 첫 번째 window 결과를 바로 return했지만,
    # 이제는 모든 window 결과를 저장한다.
    results = []

    for window_idx, line in enumerate(fin):

        data = json.loads(line)

        print(
            f"\n===== Window {window_idx} ====="
        )

        # JSON dict -> TokenInfo 구조체 변환
        token_map = [
            TokenInfo(**token)
            for token in data["token_map"]
        ]

        # ---------------------------------
        # 미매핑 token 확인
        # ---------------------------------

        unmapped_count = 0

        for token_info in token_map:

            if token_info.line_id is None:

                unmapped_count += 1

                start = token_info.start
                end = token_info.end

                print(
                    "미매핑 토큰:",
                    token_info.token_idx,
                    repr(
                        data["text"][start:end]
                    )
                )

        print(
            "미매핑 토큰 수:",
            unmapped_count
        )

        # ---------------------------------
        # 모델 입력 tensor 생성
        # ---------------------------------

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

        # ---------------------------------
        # Qwen attention 추출
        #
        # outputs.attentions:
        #
        # layer별
        # [batch, head, query_token, key_token]
        # ---------------------------------

        with torch.no_grad():

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                use_cache=False,
                return_dict=True
            )
            save_token_probe(attentions=outputs.attentions,
                             token_map=token_map,
                             window_idx=window_idx,
                             line_meta=data.get("line_meta", []),
                             output_dir="token_probe_results"
                             )

        # ---------------------------------
        # token attention
        #       ↓
        # line attention
        #
        # 결과:
        # [layer, head, query_line, key_line]
        # ---------------------------------

        line_attentions, line_ids = (
            convert_to_line_attention(
                layer_attentions=outputs.attentions,
                token_map=token_map
            )
        )

        print(
            "line attention shape:",
            line_attentions.shape
        )

        print(
            "line_ids:",
            line_ids
        )

        # ---------------------------------
        # [수정 2]
        # 각 line이 몇 개의 token으로 이루어졌는지 계산
        #
        # 이후 line length confound 분석에 사용
        # ---------------------------------

        line_token_counts = []

        for line_id in line_ids:

            count = sum(
                1
                for token_info in token_map
                if token_info.line_id == line_id
            )

            line_token_counts.append(count)

        print(
            "line token counts:",
            line_token_counts
        )

        # ---------------------------------
        # sanity check
        #
        # 각 query line이 전체 key line에
        # 분배한 attention mass 확인
        #
        # shape:
        # [layer, head, query_line]
        # ---------------------------------

        row_sums = line_attentions.sum(
            dim=-1
        )

        print(
            "row sums shape:",
            row_sums.shape
        )

        print(
            "첫 layer / 첫 head row sums:",
            row_sums[0, 0]
        )

        # ---------------------------------
        # [수정 3]
        # 해당 window의 결과를 리스트에 저장
        #
        # line_meta는 코드 1에서 만든
        # level / evidence_role 정보를 그대로 전달
        # ---------------------------------

        result = {
            "window_idx": data.get(
                "window_idx",
                window_idx
            ),

            "line_ids": line_ids,

            # 원본 head별 attention
            # [layer, head, query_line, key_line]
            "line_attentions": line_attentions,

            # 참고용 head 평균
            # [layer, query_line, key_line]
            "head_mean_attentions":
                line_attentions.mean(dim=1),

            # line별 token 수
            "line_token_counts":
                line_token_counts,

            # ERROR/WARN/INFO와
            # root/propagation/symptom GT
            "line_meta":
                data.get("line_meta", []),

            # 기존 label도 필요할 경우 유지
            "line_labels":
                data.get("line_labels", [])
        }

        results.append(result)

        # 모델 output에서 큰 tensor reference 제거
        del outputs

        if device.type == "cuda":
            torch.cuda.empty_cache()

        elif device.type == "xpu":
            torch.xpu.empty_cache()

    # [수정 4]
    # 기존:
    #
    # return line_attentions, line_ids
    #
    # 였기 때문에 첫 window에서 끝났음.
    #
    # 이제 모든 window 결과를 반환.
    return results


if __name__ == "__main__":
    main()