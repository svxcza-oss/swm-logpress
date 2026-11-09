
import io
import json
from types import SimpleNamespace

import torch
import attention as at


def _window(window_idx, line_ids=(10, 11)):
    """
    테스트용 Window 데이터를 생성하는 보조 함수.

    목적:
    - 실제 token_map JSONL과 동일한 구조의 가상 데이터를 생성
    - 2개의 Log Line과 2개의 Token으로 간단하게 테스트

    구성:
    - Line 10: INFO / root
    - Line 11: ERROR / symptom
    - Token 'A' -> Line 10
    - Token 'B' -> Line 11
    """

    return {
        "window_idx": window_idx,
        "text": "AB",

        # 모델에 전달할 Token ID와 Attention Mask
        "input_ids": [101, 102],
        "attention_mask": [1, 1],

        # 각 Token이 어느 로그 라인에 속하는지 나타내는 정보
        "token_map": [
            {
                "token_idx": i,
                "token_id": 101 + i,
                "token": c,
                "start": i,
                "end": i + 1,
                "line_id": line_ids[i],
                "row_idx": i,
                "field": "content"
            }
            for i, c in enumerate("AB")
        ],

        # 로그 레벨 및 Ground Truth 메타데이터
        "line_meta": [
            {
                "line_id": line_ids[0],
                "level": "INFO",
                "evidence_role": "root"
            },
            {
                "line_id": line_ids[1],
                "level": "ERROR",
                "evidence_role": "symptom"
            }
        ],

        # 기존 로그의 Label 정보
        "line_labels": [
            {"line_id": x, "label": "-"}
            for x in line_ids
        ]
    }


class DummyModel:
    """
    실제 Qwen 모델 대신 사용하는 가짜 모델.

    목적:
    - 실제 LLM 추론 없이 attention.py의 기능을 테스트
    - 모델이 어떤 인자를 전달받았는지 기록
    - Attention Tensor를 미리 정해진 형태로 반환

    반환 Attention Shape:
    - Layer 수: 2
    - Head 수: 2
    - Query Token 수: 2
    - Key Token 수: 2

    주의:
    - 실제 Attention 계산을 검증하는 모델은 아님
    """

    def __init__(self):

        # 모델 호출 시 전달받은 인자를 저장
        self.calls = []

    def __call__(self, **kwargs):

        # 모델 호출 정보 기록
        self.calls.append(kwargs)

        # 2개 Layer의 Attention Tensor를 반환
        # 각 Tensor Shape: [batch, head, query_token, key_token]
        return SimpleNamespace(
            attentions=(
                torch.ones(1, 2, 2, 2),
                torch.ones(1, 2, 2, 2)
            )
        )


def test_token_attention_processes_all_windows_and_preserves_metadata(
    monkeypatch
):
    """
    테스트 1: 모든 Window를 처리하고 결과 구조를 유지하는가?

    검증 목적:
    1. 여러 Window를 빠짐없이 처리하는지 확인
    2. 각 Window마다 모델을 호출하는지 확인
    3. Token Map이 TokenInfo 객체로 변환되는지 확인
    4. Line Attention의 Shape이 올바른지 확인
    5. Head 평균 계산이 올바른지 확인
    6. 기존 Window의 메타데이터가 유지되는지 확인
    7. 모델 추론 옵션이 올바르게 설정되는지 확인

    예상 결과:
    - 입력 Window: 2개
    - 출력 Result: 2개
    - 모델 호출: 2회
    - Line Attention Shape: [2, 2, 2, 2]
    - Head Mean Shape: [2, 2, 2]
    """

    # 테스트용 가짜 모델 생성
    model = DummyModel()

    # Line Attention 변환 함수 호출을 기록할 리스트
    calls = []

    def fake_convert(layer_attentions, token_map):
        """
        실제 convert_to_line_attention() 대신 실행할 가짜 함수.

        목적:
        - 실제 집계 연산 대신 입력 데이터의 구조 검증
        - TokenInfo 변환이 정상적으로 이루어졌는지 확인
        """

        # 함수 호출 시 전달된 인자 저장
        calls.append((layer_attentions, token_map))

        # 모든 Token이 TokenInfo 객체인지 확인
        assert all(
            isinstance(t, at.TokenInfo)
            for t in token_map
        )

        # Token Index가 [0, 1]인지 확인
        assert [
            t.token_idx for t in token_map
        ] == [0, 1]

        # 가상의 Line Attention 반환
        # Shape: [layer, head, query_line, key_line]
        # 2 Layer, 2 Head, 2 Query Line, 2 Key Line
        return (
            torch.ones(2, 2, 2, 2) / 2,
            [t.line_id for t in token_map]
        )

    # 실제 변환 함수를 테스트용 함수로 임시 교체
    monkeypatch.setattr(
        at,
        "convert_to_line_attention",
        fake_convert
    )

    # 2개의 Window를 JSONL 형식으로 생성
    # StringIO는 파일을 열지 않고 문자열을 파일처럼 다루는 객체
    stream = io.StringIO(
        "\n".join(
            json.dumps(_window(i))
            for i in range(2)
        )
    )

    # 실제 token_attention() 실행
    results = at.token_attention(
        stream,
        model,
        torch.device("cpu")
    )

    # 반환 자료형이 리스트인지 확인
    assert isinstance(results, list)

    # 2개 Window 모두 처리되었는지 확인
    assert len(results) == 2

    # Window마다 모델이 한 번씩 호출되었는지 확인
    assert len(model.calls) == 2

    # Line Attention 변환 함수도 2번 호출되었는지 확인
    assert len(calls) == 2

    # 각 Window의 출력 결과 검증
    for i, row in enumerate(results):

        # Window Index가 유지되는지 확인
        assert row["window_idx"] == i

        # 원래 Line ID가 유지되는지 확인
        assert row["line_ids"] == [10, 11]

        # Line Attention Shape 검증
        # [layer, head, query_line, key_line]
        assert row["line_attentions"].shape == (
            2, 2, 2, 2
        )

        # Head Mean Shape 검증
        # Head 차원이 평균되므로 [layer, query_line, key_line]
        assert row["head_mean_attentions"].shape == (
            2, 2, 2
        )

        # Head 평균 결과가 직접 계산한 평균과 같은지 확인
        # dim=1은 Head 차원을 의미
        assert torch.allclose(
            row["head_mean_attentions"],
            row["line_attentions"].mean(dim=1)
        )

        # 각 Line에 속한 Token 개수 확인
        # Line 10 = 1개, Line 11 = 1개
        assert row["line_token_counts"] == [1, 1]

        # ERROR 로그의 메타데이터가 유지되는지 확인
        assert row["line_meta"][1]["level"] == "ERROR"

        # Label 정보가 누락되지 않았는지 확인
        assert len(row["line_labels"]) == 2

        # 모델이 Attention을 반환하도록 설정되었는지 확인
        assert model.calls[i]["output_attentions"] is True

        # KV Cache를 사용하지 않도록 설정되었는지 확인
        assert model.calls[i]["use_cache"] is False

        # 모델 출력을 딕셔너리 형태로 접근 가능한
        # ModelOutput 객체로 받도록 설정되었는지 확인
        assert model.calls[i]["return_dict"] is True

        # 모델 입력 Tensor Shape 확인
        # [batch_size, token_count] = [1, 2]
        assert model.calls[i]["input_ids"].shape == (1, 2)


def test_token_attention_returns_empty_list_for_empty_jsonl():
    """
    테스트 2: 입력 JSONL이 비어 있으면 정상 처리하는가?

    검증 목적:
    - 입력 Window가 없는 경우 확인
    - 프로그램 오류 없이 빈 리스트를 반환하는지 확인

    입력:
    - 빈 문자열 ""

    예상 결과:
    - []
    """

    # 빈 JSONL을 입력으로 전달
    result = at.token_attention(
        io.StringIO(""),
        DummyModel(),
        torch.device("cpu")
    )

    # 처리할 Window가 없으므로 빈 리스트 반환
    assert result == []


def test_token_attention_reports_unmapped_tokens_without_dropping_them(
    monkeypatch,
    capsys
):
    """
    테스트 3: 미매핑 Token을 감지하고 보고하는가?

    검증 목적:
    1. line_id가 None인 미매핑 Token을 확인
    2. 미매핑 Token 정보를 변환 함수에 전달하는지 확인
    3. 정상적으로 매핑된 Line의 결과가 유지되는지 확인
    4. 미매핑 Token 개수를 출력하는지 확인

    입력:
    - Token A -> line_id=None
    - Token B -> line_id=11

    예상 결과:
    - 미매핑 Token 수: 1
    - Line 11의 Token 수: 1
    - Line Attention Shape: [2, 2, 1, 1]
    """

    # 기본 Window 데이터 생성
    data = _window(0)

    # 첫 번째 Token을 미매핑 상태로 변경
    data["token_map"][0]["line_id"] = None

    def fake_convert(layer_attentions, token_map):
        """
        실제 변환 함수 대신 사용하는 테스트용 함수.

        목적:
        - 미매핑 Token이 변환 함수까지 전달되는지 확인
        """

        # 첫 번째 Token이 미매핑 상태인지 확인
        assert token_map[0].line_id is None

        # 미매핑 Token을 포함해 총 2개가 전달되었는지 확인
        assert len(token_map) == 2

        # 정상 매핑된 Line 11만 결과에 포함된다고 가정
        # Shape: [layer, head, query_line, key_line]
        return torch.zeros(2, 2, 1, 1), [11]

    # 실제 Line Attention 변환 함수를 가짜 함수로 교체
    monkeypatch.setattr(
        at,
        "convert_to_line_attention",
        fake_convert
    )

    # 미매핑 Token이 포함된 Window 처리
    result = at.token_attention(
        io.StringIO(json.dumps(data)),
        DummyModel(),
        torch.device("cpu")
    )

    # 정상 Line 11에 속한 Token 개수가 1개인지 확인
    assert result[0]["line_token_counts"] == [1]

    # 정상 Line의 Attention Shape 확인
    assert result[0]["line_attentions"].shape == (
        2, 2, 1, 1
    )

    # 프로그램이 출력한 메시지를 가져옴
    captured = capsys.readouterr()

    # 미매핑 Token이 1개라고 출력되었는지 확인
    assert "미매핑 토큰 수: 1" in captured.out
