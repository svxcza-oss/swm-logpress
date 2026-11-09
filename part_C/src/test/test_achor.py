
import pytest
import torch
import anchor


def run_anchor(line_attentions, line_ids, line_meta):
    """
    테스트용 보조 함수

    목적:
    - 실제 evaluate_anchor()에 필요한 입력 데이터를 구성한다.
    - evaluate_anchor()의 결과를 테스트하기 편한 ranking 형태로 변환한다.
    """

    # evaluate_anchor()가 받는 result 딕셔너리 구성
    result = {
        "window_idx": 0,
        "line_ids": line_ids,
        "line_attentions": line_attentions,
        "line_meta": line_meta,
        "line_token_counts": [1] * len(line_ids),
        "line_labels": []
    }

    # 실제 anchor.py의 평가 함수 실행
    records = anchor.evaluate_anchor(result)

    # candidates에서 line_id만 추출하여 ranking 리스트 생성
    # 예: candidates = [{"line_id": 10, ...}, {"line_id": 20, ...}]
    #     ranking = [10, 20]
    return [
        {
            "anchor_line_id": record["anchor_line_id"],
            "ranking": [
                candidate["line_id"]
                for candidate in record["candidates"]
            ],
            "candidates": record["candidates"]
        }
        for record in records
    ]


@pytest.fixture
def sample():
    """
    모든 테스트에서 공통으로 사용하는 가상 로그 데이터

    목적:
    - 4개의 로그 라인 생성
    - INFO 2개, WARN 1개, ERROR 1개 구성
    - Attention Score를 직접 설정하여 예상 순위를 계산 가능하게 함
    """

    # 로그 라인 ID
    ids = [10, 20, 30, 40]

    # 각 로그의 메타데이터
    meta = [
        {"line_id": 10, "level": "INFO", "evidence_role": "root"},
        {"line_id": 20, "level": "INFO", "evidence_role": "none"},
        {"line_id": 30, "level": "WARN", "evidence_role": "symptom"},
        {"line_id": 40, "level": "ERROR", "evidence_role": "symptom"},
    ]

    # Attention Tensor 생성
    # Shape: [layer, head, query_line, key_line]
    # layer=1, head=1, query=4, key=4
    attention = torch.zeros(1, 1, 4, 4)

    # WARN(line_id=30)이 이전 로그를 바라보는 Attention
    # line 10 = 0.8
    # line 20 = 0.2
    # 예상 Ranking: [10, 20]
    attention[0, 0, 2, :2] = torch.tensor([0.8, 0.2])

    # ERROR(line_id=40)가 이전 로그를 바라보는 Attention
    # line 10 = 0.1
    # line 20 = 0.7
    # line 30 = 0.2
    # 예상 Ranking: [20, 30, 10]
    attention[0, 0, 3, :3] = torch.tensor([0.1, 0.7, 0.2])

    return attention, ids, meta


def _get(records, anchor_id):
    """
    테스트용 보조 함수

    목적:
    - 여러 Anchor 결과 중 특정 Anchor의 결과만 추출한다.

    예:
    _get(records, 30) -> WARN Anchor 결과
    _get(records, 40) -> ERROR Anchor 결과
    """
    return next(
        r for r in records
        if r["anchor_line_id"] == anchor_id
    )


def test_only_warn_error_are_anchors(sample):
    """
    테스트 1: WARN/ERROR 로그만 Anchor로 선택되는가?

    검증 목적:
    - INFO 로그는 Anchor에서 제외되는지 확인
    - WARN/ERROR 로그만 Anchor로 선택되는지 확인

    예상 결과:
    - line 10 (INFO)  -> 제외
    - line 20 (INFO)  -> 제외
    - line 30 (WARN)  -> 선택
    - line 40 (ERROR) -> 선택
    """

    # Anchor 평가 실행
    output = run_anchor(*sample)

    # 선택된 모든 Anchor ID를 집합으로 추출
    # 실제 Anchor 집합이 {30, 40}과 정확히 일치해야 PASS
    assert {
        r["anchor_line_id"]
        for r in output
    } == {30, 40}


def test_ranking_uses_descending_attention_score(sample):
    """
    테스트 2: Attention Score가 높은 순서대로 정렬되는가?

    검증 목적:
    - Candidate Ranking이 Attention Score 내림차순인지 확인

    WARN(30):
    - line 10 = 0.8
    - line 20 = 0.2
    - 예상 Ranking = [10, 20]

    ERROR(40):
    - line 10 = 0.1
    - line 20 = 0.7
    - line 30 = 0.2
    - 예상 Ranking = [20, 30, 10]
    """

    # Anchor별 Ranking 생성
    output = run_anchor(*sample)

    # WARN Anchor의 정렬 결과 검증
    assert _get(output, 30)["ranking"] == [10, 20]

    # ERROR Anchor의 정렬 결과 검증
    assert _get(output, 40)["ranking"] == [20, 30, 10]


def test_excludes_current_and_future_lines_from_candidates(sample):
    """
    테스트 3: 현재 및 미래 로그가 Candidate에서 제외되는가?

    검증 목적:
    - Anchor 자기 자신이 Candidate에 포함되지 않는지 확인
    - Anchor 이후 로그가 Candidate에 포함되지 않는지 확인
    - Anchor 이전 로그만 Candidate로 사용되는지 확인

    예상 결과:
    - Anchor 30 -> Candidate [10, 20]
    - Anchor 40 -> Candidate [10, 20, 30]

    주의:
    - Candidate의 순서는 Attention Score에 따라 달라질 수 있음
    - 여기서는 순위가 아닌 시간적 위치 조건을 검증
    """

    output = run_anchor(*sample)

    # 원본 로그의 시간적 순서를 나타내는 line_ids
    line_ids = sample[1]

    # 각 Anchor에 대해 Candidate 검증
    for record in output:
        anchor_id = record["anchor_line_id"]
        ranking = record["ranking"]

        # 조건 1: Anchor 자기 자신이 Ranking에 없어야 함
        assert anchor_id not in ranking

        # 원본 로그 배열에서 Anchor 위치 확인
        anchor_index = line_ids.index(anchor_id)

        # 조건 2: 모든 Candidate가 Anchor보다 앞에 있어야 함
        # Candidate의 위치 < Anchor의 위치
        assert all(
            line_ids.index(candidate) < anchor_index
            for candidate in ranking
        )


def test_ranking_does_not_use_evidence_ground_truth(sample):
    """
    테스트 4: Ground Truth(evidence_role)가 Ranking에 영향을 주는가?

    검증 목적:
    - Ranking이 Attention Score를 기준으로 생성되는지 확인
    - 정답 라벨(evidence_role)을 변경해도 Ranking이 유지되는지 확인
    - 평가 정답이 Ranking 생성에 사용되는 것을 방지

    테스트 방법:
    1. 원본 데이터로 Ranking 생성
    2. evidence_role의 root와 none을 서로 변경
    3. 동일한 Attention으로 다시 Ranking 생성
    4. 두 Ranking이 같은지 비교

    예상 결과:
    - Ground Truth를 변경해도 Ranking은 동일해야 함
    """

    # 원본 Ground Truth를 사용한 Ranking
    original = run_anchor(*sample)

    attention, ids, meta = sample

    # Ground Truth 변경
    # root -> none
    # root가 아닌 모든 값 -> root
    # Attention Score 자체는 변경하지 않음
    swapped_meta = [
        dict(
            row,
            evidence_role=(
                "none"
                if row["evidence_role"] == "root"
                else "root"
            )
        )
        for row in meta
    ]

    # 변경된 Ground Truth를 사용한 Ranking
    swapped = run_anchor(
        attention,
        ids,
        swapped_meta
    )

    # 원본 Anchor ID와 Ranking 추출
    original_ranks = [
        (r["anchor_line_id"], r["ranking"])
        for r in original
    ]

    # Ground Truth 변경 후 Anchor ID와 Ranking 추출
    swapped_ranks = [
        (r["anchor_line_id"], r["ranking"])
        for r in swapped
    ]

    # Ground Truth 변경 전후 Ranking이 같아야 PASS
    assert original_ranks == swapped_ranks
