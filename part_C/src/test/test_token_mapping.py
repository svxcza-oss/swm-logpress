
import json
from types import SimpleNamespace

import pytest

import token_mapper as tm


class FakeTokenizer:
    """
    실제 Hugging Face Tokenizer 대신 사용하는 가짜 Tokenizer.

    목적:
    - 모델을 불러오지 않고 Token Mapping 기능만 테스트
    - 토큰의 시작/끝 위치(offset)를 직접 통제
    - 공백과 개행을 포함해 입력 문자열의 각 문자를 하나의 토큰으로 처리

    예:
    text = "abc"
    offset_mapping = [(0, 1), (1, 2), (2, 3)]
    """

    def __call__(self, text, **kwargs):

        # tokenizer_log()가 올바른 옵션을 전달하는지 확인
        assert kwargs['return_offsets_mapping'] is True
        assert kwargs['add_special_tokens'] is False

        # 각 문자를 하나의 토큰으로 분리
        # 공백 및 개행도 토큰으로 포함
        offsets = [(i, i + 1) for i in range(len(text))]

        class FakeTensor:
            """
            실제 Tensor를 사용하지 않고 필요한 기능만 흉내 내는 객체.
            """

            def __init__(self, value):
                self.value = value

            def __getitem__(self, idx):
                if isinstance(idx, int):
                    value = self.value[idx]
                    return FakeTensor(value) if isinstance(value, list) else value
                raise IndexError(idx)

            def __iter__(self):
                return iter(self.value)

            def __len__(self):
                return len(self.value)

            def tolist(self):
                return self.value

        # 실제 Tokenizer와 유사한 출력 형식 반환
        return {
            'input_ids': FakeTensor([[100 + i for i in range(len(text))]]),
            'attention_mask': FakeTensor([[1] * len(text)]),
            'offset_mapping': FakeTensor([offsets]),
        }

    def convert_ids_to_tokens(self, token_id):
        # Token ID를 테스트용 문자열로 변환
        return f't{token_id}'


def test_map_token_to_span_returns_object_not_dictionary():
    """
    테스트 1: Token Mapping 결과가 Span 객체로 반환되는가?

    검증 목적:
    - Token과 Span이 겹치면 해당 Span 객체를 반환하는지 확인
    - 결과를 딕셔너리가 아닌 객체의 속성으로 접근할 수 있는지 확인

    입력:
    - Span 범위: [0, 5)
    - Token 범위: [1, 3)

    예상 결과:
    - Token이 Span 안에 있으므로 line_id=7에 매핑
    """

    # 테스트용 Span 객체 생성
    spans = [
        tm.Span(
            line_id=7,
            row_idx=0,
            field='content',
            start=0,
            end=5
        )
    ]

    # Token 범위 [1, 3)을 Span과 매핑
    result = tm.map_token_to_span(1, 3, spans)

    # 반환 결과가 원래 Span 객체와 동일한지 확인
    assert result is spans[0]

    # 딕셔너리 접근이 아닌 객체 속성 접근이 가능한지 확인
    assert result.line_id == 7


def test_map_token_to_span_no_overlap():
    """
    테스트 2: Token과 Span이 겹치지 않으면 None을 반환하는가?

    검증 목적:
    - 겹치지 않는 Token을 잘못된 로그에 매핑하지 않는지 확인

    입력:
    - Span 범위: [0, 5)
    - Token 범위: [5, 6)

    예상 결과:
    - 두 범위는 경계에서 만나지만 겹치지 않음
    - 따라서 None 반환
    """

    spans = [
        tm.Span(7, 0, 'content', 0, 5)
    ]

    # Token이 Span 범위 밖에 있으므로 None이어야 PASS
    assert tm.map_token_to_span(5, 6, spans) is None


def test_tokenizer_log_maps_fields_and_line_fallback():
    """
    테스트 3: Token이 올바른 Field와 Line에 매핑되는가?

    검증 목적:
    - 일반 문자 토큰이 content 필드에 매핑되는지 확인
    - 공백 및 개행이 line 필드로 매핑되는지 확인
    - 모든 토큰이 동일한 line_id를 유지하는지 확인
    - 토큰 개수와 인덱스가 일치하는지 확인

    입력:
    text = 'abc \\n'

    Span:
    - content: [0, 3) -> 'abc'
    - line: [0, 5) -> 전체 문자열

    예상 결과:
    - 'a', 'b', 'c' -> content
    - ' ', '\\n' -> line
    - 모든 토큰의 line_id = 10
    """

    text = 'abc \n'

    # content와 전체 line의 문자 위치 정의
    spans = [
        tm.Span(10, 0, 'content', 0, 3),
        tm.Span(10, 0, 'line', 0, len(text)),
    ]

    # Tokenizer 실행 및 Token Mapping 생성
    result = tm.tokenizer_log(text, spans, FakeTokenizer())

    # 생성된 Token 개수가 문자열 길이와 같은지 확인
    assert len(result['input_ids']) == len(text)

    # Attention Mask 개수가 문자열 길이와 같은지 확인
    assert len(result['attention_mask']) == len(text)

    mapped = result['token_map']

    # 모든 Token Index가 0부터 순서대로 부여되었는지 확인
    assert [m['token_idx'] for m in mapped] == list(range(len(text)))

    # 모든 Token이 line_id=10에 매핑되었는지 확인
    assert all(m['line_id'] == 10 for m in mapped)

    # 각 토큰이 올바른 Field에 매핑되었는지 확인
    assert [m['field'] for m in mapped] == [
        'content',
        'content',
        'content',
        'line',
        'line'
    ]


def test_tokenizer_log_empty_offset_is_unmapped():
    """
    테스트 4: Offset이 비어 있는 Token을 미매핑 처리하는가?

    검증 목적:
    - 시작 위치와 끝 위치가 같은 (0, 0) Token을 확인
    - 실제 문자열 범위가 없는 Token에 line_id를 부여하지 않는지 확인

    입력:
    - text = 'a'
    - Token Offset = (0, 0)

    예상 결과:
    - line_id = None
    """

    class WithEmpty(FakeTokenizer):
        """
        첫 번째 Token의 Offset을 강제로 (0, 0)으로 변경.
        """

        def __call__(self, text, **kwargs):

            encoded = super().__call__(text, **kwargs)

            # 원래 (0, 1)인 Offset을 (0, 0)으로 변경
            encoded['offset_mapping'].value[0][0] = (0, 0)

            return encoded

    # 빈 Offset을 가진 토큰에 대해 매핑 실행
    result = tm.tokenizer_log(
        'a',
        [tm.Span(1, 0, 'content', 0, 1)],
        WithEmpty()
    )

    # 매핑할 문자열 범위가 없으므로 None이어야 PASS
    assert result['token_map'][0]['line_id'] is None


def test_tokenizer_log_first_overlap_wins_for_cross_boundary_token():
    """
    테스트 5: 하나의 Token이 두 Span에 걸치면 어떻게 처리되는가?

    검증 목적:
    - Token이 서로 다른 두 로그 라인에 걸치는 상황 확인
    - 현재 구현이 첫 번째로 겹치는 Span을 선택하는지 검증

    입력:
    - Line 1: [0, 2) -> 'ab'
    - Line 2: [2, 4) -> 'cd'
    - Token:  [1, 3) -> 'bc'

    예상 결과:
    - Token 'bc'는 Line 1과 Line 2에 모두 겹침
    - 현재 구현은 먼저 발견한 Line 1에 매핑
    - 따라서 line_id = 1
    """

    class OneToken(FakeTokenizer):
        """
        입력 문자열을 하나의 Token처럼 처리하도록
        Tokenizer 출력을 강제로 변경.
        """

        def __call__(self, text, **kwargs):

            enc = super().__call__(text, **kwargs)

            # Token 하나만 생성
            enc['input_ids'].value = [[101]]
            enc['attention_mask'].value = [[1]]

            # Token이 두 로그 라인에 걸치도록 Offset 지정
            enc['offset_mapping'].value = [[(1, 3)]]

            return enc

    # 서로 다른 두 로그 라인 정의
    spans = [
        tm.Span(1, 0, 'content', 0, 2),
        tm.Span(2, 1, 'content', 2, 4)
    ]

    # 경계에 걸친 Token Mapping 실행
    result = tm.tokenizer_log(
        'abcd',
        spans,
        OneToken()
    )

    # 현재 구현은 먼저 겹친 Span(Line 1)을 선택
    assert result['token_map'][0]['line_id'] == 1

def test_qwen_token_does_not_cross_line_boundary():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct"
    )

    text = "ERROR disk failure \nINFO recovery started \n"

    boundary = len("ERROR disk failure \n")

    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False
    )

    for start, end in encoded["offset_mapping"]:
        assert not (start < boundary < end)
