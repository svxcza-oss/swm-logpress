# test 디렉터리 쉽게 이해하기

## 1. 테스트가 확인하는 것

작은 입력에서 예상되는 결과를 미리 정하고, 코드가 같은 값을 내는지 pytest로 확인한다.
연구 가설의 성능을 검증하는 실험과는 다르다.
테스트 통과는 정해 둔 입력과 조건에서 코드가 기대대로 동작했다는 뜻이다.

## 2. 파일별 역할

| 파일 | 테스트 수 | 핵심 검증 |
|---|---:|---|
| test_token_mapping.py | 6 | Span 겹침, 공백·개행, 빈 Offset, 경계 토큰 |
| test_cross_line.py | 6 | Key SUM / Query MEAN 수치, 차원, 순서, 오류 |
| test_attention.py | 3 | 모든 Window 처리, 메타데이터, 빈 입력, 미매핑 보고 |
| test_achor.py | 4 | WARN/ERROR 선택, 정렬, 이전 후보, GT 독립성 |
| test_window_extractor.py | 54개 실행 사례 | 크기·stride·끝부분·정답·CSV·CLI·Mapper 연결 |

Window 테스트는 매개변수화로 함수 하나를 여러 입력에 실행하므로 함수 수와 사례 수가 다르다.
현재 전체 실행 사례는 73개다. token_attention__probe.py 전용 테스트 파일은 없다.

## 3. 모델 없이 어떻게 검사하는가?

### FakeTokenizer

문자 하나를 토큰 하나처럼 취급해 Offset을 쉽게 예상할 수 있게 한다.
실제 Qwen 토큰 분할 품질을 검증하는 도구는 아니다.
test_token_mapping.py의 마지막 테스트 하나는 실제 Qwen Tokenizer로 특정 문장의
라인 경계 침범 여부를 확인한다. 캐시 또는 네트워크가 필요할 수 있지만 모델 가중치는 로드하지 않는다.

### DummyModel

test_attention.py에서 정해진 Tensor를 반환하는 가짜 모델이다.
모델 호출 인자와 후속 데이터 전달을 확인한다. 실제 모델의 Attention 정확성을 검증하지 않는다.

### monkeypatch

테스트 동안 실제 함수나 입력을 임시로 바꾼다.
예를 들어 convert_to_line_attention 대신 가짜 함수를 사용해 TokenInfo 전달을 확인한다.
테스트가 끝나면 원래 상태로 복원된다.

### tmp_path

소형 CSV와 결과 파일을 임시 폴더에 만든다.
단, 기존 Attention 테스트는 진단 함수를 대체하지 않으므로
현재 작업 폴더에 token_probe_results CSV를 생성하거나 추가할 수 있다.

## 4. 각 테스트 파일을 읽는 방법

test_token_mapping.py의 FakeTokenizer부터 보면 입력을 통제하는 방법이 보인다.
test_cross_line.py는 직접 만든 작은 Attention 행렬과 예상값을 비교하므로 집계 규칙 이해에 좋다.
test_attention.py의 _window()는 파이프라인 중간 JSON 구조의 예시이며,
DummyModel과 fake_convert가 실제 추론 없이 연결을 검사한다.
test_achor.py의 sample()은 정해진 점수로 정렬 결과를 예상하고,
run_anchor()는 evaluate_anchor() 호출을 테스트하기 쉬운 형태로 감싼다.
파일명 achor는 현재 저장된 이름 그대로다.

test_window_extractor.py의 make_csv(), run(), read_jsonl()은 입력 생성과 결과 읽기를 돕는다.
실제 token_mapper.main()을 가짜 Tokenizer와 연결하는 통합 테스트도 있다.
경계 테스트는 입력 길이 15가지 × 크기 6가지 × stride 7가지 × tail 정책 2가지,
총 1,260가지 조합을 단순 배열 슬라이싱과 대조한다.

## 5. 실행 방법

프로젝트에서 쓰는 Python 환경을 활성화하고 src에서 실행한다.
pytest가 필요하고, 기존 모듈 import에는 torch와 transformers도 필요하다.

```powershell
python -m pytest test -q
python -m pytest test/test_cross_line.py -q
python -m pytest test/test_window_extractor.py -q
```

실제 Tokenizer를 사용하는 기존 테스트를 제외하려면:

```powershell
python -m pytest test -q -k "not qwen_token_does_not_cross_line_boundary"
```

Window 추출 로직만 확인하고 Mapper 통합 테스트를 제외하려면:

```powershell
python -m pytest test/test_window_extractor.py -q -k "not actual_token_mapper"
```

통과는 passed, 실패는 failed, 실행 준비 중 문제는 error, 선택에서 제외한 것은 deselected다.
임시 폴더 권한, PyTorch DLL, 의존성 문제를 코드의 계산 오류와 구분해야 한다.

## 6. 확인된 실행 기록과 한계

앞선 구현 검증에서 전체 73개가 통과했다.
2026-11-09 Window 재점검에서는 Mapper 통합 1개를 제외한 53개가 통과했다.
별도로 실제 Tokenizer의 167토큰 매핑과 Qwen 모델의 Line Attention·Anchor 순위를 실행했다.
이는 이전 검증 기록이며 이번 문서 작성에서는 테스트나 모델 추론을 재실행하지 않았다.

대용량 CSV 전체 처리 시간, 모든 로그의 토큰 경계, keyword Anchor의 실제 Severity 정확도,
Alert 검색 성능이나 Root Cause 성능을 이 테스트만으로 보장하지 않는다.

처음으로 돌아가기: [팀원용 전체 안내](TEAM_CODE_GUIDE.md).
