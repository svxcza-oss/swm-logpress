# token_attention__probe.py 쉽게 이해하기

## 1. 왜 필요한가?

첫 로그가 높은 Attention을 받았을 때, 그 로그 전체가 고르게 관심을 받는지
첫 토큰 하나에 대부분의 점수가 몰리는지 구분하는 **진단 도구**다.
Attention이나 순위 점수를 수정하는 보정 코드는 아니다.
파일명에는 attention과 probe 사이에 밑줄이 두 개 있다.

## 2. 언제 실행되는가?

attention.py가 모델 forward 직후 `save_token_probe()`를 호출한다.
독립 실행용 main()이나 CLI는 없다. attention.py를 실행하면 함께 실행된다.
파일을 직접 `python token_attention__probe.py`로 실행해도 추론이나 진단을 시작하지 않는다.

## 3. 주요 함수

| 함수 | 역할 |
|---|---|
| `_attr()` | 딕셔너리 또는 TokenInfo 객체에서 같은 방식으로 속성 읽기 |
| `save_token_probe()` | Anchor별·Layer별·Head별 Attention 분해 후 CSV 저장 |

입력 attentions는 Layer별 `[1,H,T,T]` Tensor 튜플이다.
token_map의 순서는 토큰 순서여야 한다. line_meta의 level로 WARN/ERROR를 찾는다.
기본 Anchor 레벨은 ERROR와 WARN이다.

## 4. 어떻게 분해하는가?

먼저 **처음 매핑된 라인**과 그 라인의 **첫 토큰**을 찾는다.
첫 매핑 토큰은 미매핑 토큰이 앞에 있을 경우 전체 token_idx=0과 다를 수 있다.

각 Anchor에 속한 Query 토큰들을 골라 평균하면 Key 토큰별 점수가 나온다.
Layer와 Head는 평균하지 않는다. 이 점수를 네 묶음으로 나눈다.

```text
Anchor의 Query 토큰 평균
  ├─ 첫 라인의 첫 토큰
  ├─ 첫 라인의 나머지 토큰
  ├─ 다른 매핑 라인들의 토큰
  └─ 미매핑 토큰
```

이 진단은 anchor.py처럼 이전 라인만 후보로 제한하지 않는다.
다른 라인 묶음에는 Anchor 자기 라인도 들어갈 수 있다.
미래 토큰에 대한 값은 모델이 제공한 Attention 그대로 사용한다.

## 5. 출력 CSV 읽는 법

기본 경로는 현재 작업 폴더의 `token_probe_results/token_bias_by_head.csv`다.
한 행은 `(window_idx, anchor_line_id, layer, head)` 하나에 해당한다.

| 컬럼 | 의미 |
|---|---|
| window_idx / anchor_line_id | 어떤 Window의 어느 Query인가 |
| first_line_id | 첫 매핑 라인 ID |
| layer / head | 0-based Layer·Head 번호 |
| query_token_count | 해당 Anchor의 토큰 수 |
| key_token_count | Window 전체 토큰 수 |
| first_token_score | 첫 라인 첫 토큰에 준 점수 |
| first_line_rest_score | 첫 라인의 나머지 토큰에 준 합계 |
| first_line_total_score | 위 두 점수의 합 |
| other_lines_score | 다른 매핑 라인들에 준 합계 |
| unmapped_score | 미매핑 Key 토큰에 준 합계 |
| first_token_share_of_first_line | 첫 라인 점수 중 첫 토큰이 차지하는 비율 |
| first_line_token_count | 첫 라인 토큰 수 |

예를 들어 첫 토큰 점수가 0.40이고 나머지 첫 라인 점수가 0.10이면,
first_line_total_score=0.50, first_token_share_of_first_line=0.80이다.
즉 첫 라인에 준 Attention의 80%가 첫 토큰 하나에 집중됐다. 설명용 수치다.
첫 라인 전체 점수가 0이면 비율은 코드에서 0으로 저장한다.

## 6. 결과를 해석할 때

집중 현상이 있다는 관찰과 그것이 실제 중요성을 뜻한다는 해석은 구분해야 한다.
이 CSV만으로 Attention Sink의 원인이나 Root Cause를 확정하지 않는다.
특정 Layer·Head에서만 집중되는지도 확인할 수 있도록 차원을 유지했다.

CSV는 append 모드다. 재실행하면 같은 Window 번호라도 새로운 행이 추가된다.
자동 중복 제거 또는 실행 ID 부여는 없다. 실험별 폴더를 구분해서 관리한다.
입력 PT/JSONL 옆에 저장되는 것이 아니라 실행 위치 기준이라는 점도 확인한다.

WARN/ERROR 메타데이터가 없으면 메시지를 출력하고 빈 리스트를 반환한다.
Attention이 없거나 모든 토큰이 미매핑이거나 Tensor 형식이 잘못되면 예외가 발생한다.
이 경우 호출자인 attention.py도 중단될 수 있다.

관련 문서: [Attention 추출](ATTENTION_EASY_GUIDE.md), [테스트 안내](TESTS_EASY_GUIDE.md).
