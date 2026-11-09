
from types import SimpleNamespace

import pytest
import torch

from cross_line import convert_to_line_attention


def _tokens(ids):
    """
    테스트용 Token Mapping 데이터를 생성하는 보조 함수.

    목적:
    - 각 Token의 token_idx와 line_id를 생성
    - 실제 TokenInfo 객체 대신 SimpleNamespace 사용

    예:
    ids = [10, 10, 20, 20]

    결과:
    - Token 0 -> Line 10
    - Token 1 -> Line 10
    - Token 2 -> Line 20
    - Token 3 -> Line 20
    """

    return [
        SimpleNamespace(
            token_idx=i,
            line_id=line_id
        )
        for i, line_id in enumerate(ids)
    ]


def test_key_sum_and_query_mean_exact_values():
    """
    테스트 1: Key SUM과 Query MEAN 계산이 정확한가?

    검증 목적:
    - 같은 Line에 속한 Key Token의 Attention을 합산하는지 확인
    - 같은 Line에 속한 Query Token의 Attention을 평균하는지 확인
    - 최종 Line Attention 값이 예상 결과와 일치하는지 확인

    입력:
    - Token 0, 1 -> Line 10
    - Token 2, 3 -> Line 20

    예상 결과:
    Line Attention =
    [[1.0, 0.0],
     [0.3, 0.7]]
    """

    # Token-level Attention Matrix
    # 행: Query Token
    # 열: Key Token
    a = torch.tensor([
        [0.6, 0.4, 0., 0.],
        [0.2, 0.8, 0., 0.],
        [0.1, 0.2, 0.7, 0.],
        [0.1, 0.2, 0.3, 0.4],
    ], dtype=torch.float32)

    # Token Attention을 Line Attention으로 변환
    # a[None, None] -> [batch=1, head=1, query=4, key=4]
    # (a[None, None],) -> Layer 1개를 담은 튜플
    actual, line_ids = convert_to_line_attention(
        (a[None, None],),
        _tokens([10, 10, 20, 20])
    )

    # 두 개의 Line이 올바르게 생성되었는지 확인
    assert line_ids == [10, 20]

    # 출력 Shape 확인
    # [layer=1, head=1, query_line=2, key_line=2]
    assert actual.shape == (1, 1, 2, 2)

    # 직접 계산한 정답 Line Attention
    expected = torch.tensor([
        [1., 0.],
        [0.3, 0.7]
    ])

    # 실제 결과와 예상 결과의 값이 일치하는지 확인
    torch.testing.assert_close(
        actual[0, 0],
        expected
    )


def test_preserves_layer_and_head_dimensions():
    """
    테스트 2: Layer와 Head 차원이 유지되는가?

    검증 목적:
    - Token -> Line 변환 과정에서 Layer 차원 유지
    - 여러 Attention Head가 평균되거나 합쳐지지 않는지 확인
    - Line Attention Shape이 올바른지 확인

    입력:
    - Layer: 2개
    - Head: 2개
    - Token: 3개
    - Line: 3개 (각 Line에 Token 1개)

    예상 결과:
    - Shape = [2, 2, 3, 3]
    - 각 Line에 Token이 하나이므로 Identity Matrix 유지
    """

    # 3x3 Identity Matrix 생성 후 Head 2개로 복제
    # Shape: [batch=1, head=2, query=3, key=3]
    a = torch.eye(3).repeat(1, 2, 1, 1)

    # Layer 2개를 전달하여 Line Attention 변환
    actual, ids = convert_to_line_attention(
        (a, a),
        _tokens([1, 2, 3])
    )

    # 모든 Line ID 유지 여부 확인
    assert ids == [1, 2, 3]

    # Layer와 Head 차원이 유지되는지 확인
    assert actual.shape == (2, 2, 3, 3)

    # 첫 번째 Layer/Head의 값이 Identity Matrix인지 확인
    torch.testing.assert_close(
        actual[0, 0],
        torch.eye(3)
    )


def test_skips_unmapped_tokens_but_does_not_renormalize():
    """
    테스트 3: 미매핑 Token을 제외하면서 재정규화하지 않는가?

    검증 목적:
    - line_id=None인 Token을 Line 집계에서 제외하는지 확인
    - 정상 매핑된 Token의 Attention 값이 유지되는지 확인
    - 미매핑 Token 제거 후 Row Sum을 강제로 1로 만들지 않는지 확인

    입력:
    - Token 0 -> None (미매핑)
    - Token 1 -> Line 5
    - Token 2 -> Line 6

    예상 결과:
    Line Attention =
    [[0.8, 0.0],
     [0.2, 0.4]]

    주의:
    - 미매핑 Token에 배분된 Attention은 최종 결과에서 제외
    - 따라서 Row Sum이 1보다 작을 수 있음
    """

    # 3개 Token의 Attention Matrix
    a = torch.tensor([
        [1., 0., 0.],
        [.2, .8, 0.],
        [.4, .2, .4]
    ])

    # 첫 Token은 미매핑, 나머지는 Line 5와 6
    actual, ids = convert_to_line_attention(
        (a[None, None],),
        _tokens([None, 5, 6])
    )

    # 실제로 매핑된 Line만 반환되었는지 확인
    assert ids == [5, 6]

    # 예상 Line Attention과 값 비교
    torch.testing.assert_close(
        actual[0, 0],
        torch.tensor([
            [.8, 0.],
            [.2, .4]
        ])
    )

    # 두 번째 Query Line의 Row Sum이 1보다 작은지 확인
    # 0.2 + 0.4 = 0.6
    # 미매핑 Token으로 향한 Attention 0.4를 재분배하지 않음
    assert actual[0, 0, 1].sum() < 1


def test_line_ids_keep_first_appearance_order():
    """
    테스트 4: Line ID가 처음 등장한 순서대로 유지되는가?

    검증 목적:
    - Line ID를 숫자 크기순으로 정렬하지 않는지 확인
    - Token Map에서 처음 등장한 순서가 유지되는지 확인

    입력:
    Token 0 -> Line 30
    Token 1 -> Line 10
    Token 2 -> Line 20

    예상 결과:
    line_ids = [30, 10, 20]
    """

    # 3개 Token의 Identity Attention Matrix
    a = torch.eye(3)[None, None]

    # Line ID가 정렬되지 않은 순서로 입력됨
    _, ids = convert_to_line_attention(
        (a,),
        _tokens([30, 10, 20])
    )

    # 오름차순이 아닌 최초 등장 순서 유지 확인
    assert ids == [30, 10, 20]


def test_raises_when_attention_length_does_not_match_token_map():
    """
    테스트 5: Attention Token 수와 Token Map 길이가 다르면 오류가 발생하는가?

    검증 목적:
    - Attention Tensor와 Token Map의 길이 불일치 감지
    - 잘못된 입력으로 Line Attention을 계산하는 것을 방지

    입력:
    - Attention Token 수: 3개
    - Token Map 길이: 2개

    예상 결과:
    - ValueError 또는 AssertionError 발생
    """

    # Attention은 3개 Token을 기준으로 생성
    a = torch.eye(3)[None, None]

    # Token Map에는 2개 Token만 전달
    # 두 길이가 다르므로 예외가 발생해야 PASS
    with pytest.raises((ValueError, AssertionError)):
        convert_to_line_attention(
            (a,),
            _tokens([1, 2])
        )


def test_raises_when_no_token_is_mapped():
    """
    테스트 6: 모든 Token이 미매핑 상태이면 오류가 발생하는가?

    검증 목적:
    - 유효한 Line ID가 하나도 없는 상황 감지
    - Line Attention 계산이 불가능한 입력 처리

    입력:
    - Token 0 -> None
    - Token 1 -> None

    예상 결과:
    - ValueError 또는 AssertionError 발생
    """

    # 2개 Token의 Attention Matrix
    a = torch.eye(2)[None, None]

    # 모든 Token이 미매핑 상태
    # 생성 가능한 Line이 없으므로 오류가 발생해야 PASS
    with pytest.raises((ValueError, AssertionError)):
        convert_to_line_attention(
            (a,),
            _tokens([None, None])
        )
