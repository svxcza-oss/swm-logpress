import torch
import json
from pathlib import Path


# 원본 로그에서 Query Anchor로 사용할 severity
ANCHOR_LEVELS = ["ERROR", "WARN"]


def main():

    input_path = input(
        "line_attention PT 파일 경로를 입력하세요: "
    )

    input_path = Path(input_path)

    if not input_path.exists():
        print("입력 파일을 찾을 수 없습니다.")
        return

    output_path = input_path.with_name(
        input_path.stem + "_anchor.jsonl"
    )

    # attention.py에서 저장한 결과 불러오기
    results = torch.load(
        input_path,
        map_location="cpu",
        weights_only=False
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as fout:

        for result in results:

            anchor_results = evaluate_anchor(
                result
            )

            for anchor_result in anchor_results:

                fout.write(
                    json.dumps(
                        anchor_result,
                        ensure_ascii=False
                    ) + "\n"
                )

    print(
        f"\n저장 완료: {output_path}"
    )


def evaluate_anchor(result):

    window_idx = result["window_idx"]

    # 실제 로그 line 순서
    line_ids = result["line_ids"]

    # shape:
    # [layer, head, query_line, key_line]
    line_attentions = result[
        "line_attentions"
    ]

    line_token_counts = result.get(
        "line_token_counts",
        []
    )

    # 코드 1에서 저장한 metadata
    line_meta = result.get(
        "line_meta",
        []
    )

    line_labels = result.get(
        "line_labels",
        []
    )

    # ---------------------------------
    # line_id -> level 정보 생성
    # ---------------------------------

    level_map = {}

    # line_meta가 있으면 우선 사용
    for meta in line_meta:

        line_id = meta["line_id"]

        level = str(
            meta.get("level", "-")
        ).upper()

        level_map[line_id] = level

    # line_meta에 정보가 없는 경우
    # 기존 line_labels에서 level을 가져옴
    for label_info in line_labels:

        line_id = label_info["line_id"]

        if line_id not in level_map:

            level_map[line_id] = str(
                label_info.get(
                    "label",
                    "-"
                )
            ).upper()

    # ---------------------------------
    # Ground Truth 정보
    #
    # evidence_role이 존재하는 경우에만
    # 결과에 같이 표시한다.
    #
    # Anchor 선택에는 사용하지 않는다.
    # ---------------------------------

    evidence_map = {}

    for meta in line_meta:

        evidence_map[
            meta["line_id"]
        ] = meta.get(
            "evidence_role",
            "none"
        )

    # ---------------------------------
    # V1:
    # 모든 layer / head attention 평균
    #
    # [layer, head, Q, K]
    #           ↓
    #          [Q, K]
    #
    # 이후에는 layer/head별 분석 가능
    # ---------------------------------

    mean_attention = line_attentions.mean(
        dim=(0, 1)
    )

    anchor_results = []

    # ---------------------------------
    # 각 log line 확인
    # ---------------------------------

    for query_idx, query_line_id in enumerate(
        line_ids
    ):

        level = level_map.get(
            query_line_id,
            "-"
        )

        # ERROR / WARN이 아니면
        # Anchor로 사용하지 않음
        if level not in ANCHOR_LEVELS:
            continue

        # ---------------------------------
        # 현재 ERROR/WARN line을 Query로 선택
        # ---------------------------------

        scores = mean_attention[
            query_idx
        ]

        candidates = []

        # ---------------------------------
        # Query보다 이전에 등장한 line만
        # Key 후보로 사용
        #
        # query_idx 자체는 제외되므로
        # Anchor line 자기 자신도 제외됨
        # ---------------------------------

        for key_idx in range(query_idx):

            key_line_id = line_ids[
                key_idx
            ]

            score = float(
                scores[key_idx]
            )

            # Anchor로부터 몇 줄 앞에 있는지
            distance = (
                query_idx - key_idx
            )

            # 해당 line의 token 개수
            if key_idx < len(
                line_token_counts
            ):
                token_count = (
                    line_token_counts[
                        key_idx
                    ]
                )
            else:
                token_count = None

            candidate = {
                "line_id":
                    key_line_id,

                "attention_score":
                    score,

                "distance":
                    distance,

                "token_count":
                    token_count,

                "level":
                    level_map.get(
                        key_line_id,
                        "-"
                    ),

                "evidence_role":
                    evidence_map.get(
                        key_line_id,
                        "none"
                    )
            }

            candidates.append(
                candidate
            )

        # ---------------------------------
        # Attention이 높은 line부터 정렬
        # ---------------------------------

        candidates.sort(
            key=lambda x: x[
                "attention_score"
            ],
            reverse=True
        )

        # ranking 번호 추가
        for rank, candidate in enumerate(
            candidates,
            start=1
        ):
            candidate["rank"] = rank

        # ---------------------------------
        # Anchor 하나에 대한 최종 결과
        # ---------------------------------

        anchor_result = {
            "window_idx":
                window_idx,

            "anchor_line_id":
                query_line_id,

            "anchor_level":
                level,

            "anchor_evidence_role":
                evidence_map.get(
                    query_line_id,
                    "none"
                ),

            "candidates":
                candidates
        }

        anchor_results.append(
            anchor_result
        )

        # ---------------------------------
        # 확인용 출력
        # ---------------------------------

        print(
            f"\n===== Window {window_idx} ====="
        )

        print(
            f"Anchor: line={query_line_id}, "
            f"level={level}"
        )

        print("\nTop 10 Lines")

        for candidate in candidates[:10]:

            print(
                f"rank={candidate['rank']} "
                f"line={candidate['line_id']} "
                f"score={candidate['attention_score']:.6f} "
                f"distance={candidate['distance']} "
                f"tokens={candidate['token_count']} "
                f"level={candidate['level']} "
                f"GT={candidate['evidence_role']}"
            )

    return anchor_results


if __name__ == "__main__":
    main()