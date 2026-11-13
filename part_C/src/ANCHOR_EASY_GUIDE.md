# anchor.py 쉽게 이해하기

## 1. 역할

**WARN/ERROR 라인을 Query로 골라 이전 로그를 Attention 점수순으로 정렬**한다.
모델을 다시 실행하지 않고 attention.py가 저장한 PT 파일을 읽는다.
함수 이름은 evaluate_anchor지만 MAP/MRR 등의 평가 지표를 계산하는 함수는 아니다.

## 2. 입력과 출력

입력: Window별 Line Attention과 메타데이터를 담은 `*_line_attention.pt`.
출력: 입력 옆의 `*_line_attention_anchor.jsonl`.

이 JSONL은 **한 줄이 Window 하나가 아니라 Anchor 하나**다.
한 Window에 Anchor가 4개면 결과도 4줄이다.

| 출력 필드 | 뜻 |
|---|---|
| window_idx | Anchor가 속한 Window |
| anchor_line_id | Query로 선택한 원본 로그 ID |
| anchor_level | WARN 또는 ERROR |
| anchor_evidence_role | 입력 메타데이터의 역할 표시, 기본 none |
| candidates | 이전 로그들의 점수와 순위 |

후보에는 line_id, attention_score, distance, token_count, level, evidence_role, rank가 있다.
rank는 1부터 시작한다. distance는 Window 내 위치 차이이며 원본 ID의 뺄셈이 아니다.

## 3. 주요 함수와 실행 흐름

### `main()`

입력 경로를 받고 PT를 CPU로 읽는다. 각 Window에 evaluate_anchor()를 호출하고
반환된 Anchor 결과를 JSONL에 한 줄씩 저장한다.

### `evaluate_anchor(result)`

```text
line_meta에서 로그 레벨 읽기
  ↓
Layer와 Head 전체 평균: [L,H,Q,K] → [Q,K]
  ↓
WARN/ERROR 라인마다 반복
  ↓
해당 Query 행에서 앞선 Key 라인 점수만 선택
  ↓
점수 내림차순 정렬 → rank 부여 → 결과에 추가
```

line_meta에 해당 ID가 없을 때만 line_labels를 대체 정보로 사용한다.
새 Window 추출기는 level을 별도로 전달하므로 실제 Anchor는 이 level로 선택된다.
Anchor 파일은 content를 직접 검사하지 않는다.

정렬 점수는 모든 Layer·Head를 동일하게 평균한 값이다.
동점은 Python의 안정 정렬에 의해 기존 후보 순서가 유지된다.
출력은 전체 후보를 저장하고 화면에는 최대 10개만 보여준다.

## 4. 작은 예시

line_ids가 `[100,101,102,103]`이고 103이 ERROR라면 후보는 `[100,101,102]`다.
103 자체와 이후 라인은 후보에 없다. 이전 WARN/ERROR는 후보가 될 수 있다.

앞선 실제 실행의 Anchor 103 결과는 다음과 같았다.

| 순위 | Line ID | Attention | 사후 연결한 Alert 여부 |
|---:|---:|---:|---|
| 1 | 100 | 0.476459 | 아니오 |
| 2 | 102 | 0.184215 | 아니오 |
| 3 | 101 | 0.108218 | 예 |

점수 합이 1일 필요는 없다. 전체 Line Attention에서 자기 라인을 제외한 뒤
후보 점수를 재정규화하지 않기 때문이다.
Window 첫 줄이 Anchor라면 candidates=[]인 결과를 그대로 저장한다.

## 5. Alert Ground Truth와 연결하는 방법

현재 anchor.py는 alert_ground_truth.jsonl을 읽지 않는다.
출력의 evidence_role이나 화면의 GT=none은 Alert 여부가 아니다.
현재 추출 파이프라인에서는 evidence_role이 기본 none으로 전달된다.

나중에 평가할 때 다음 두 키로 별도 정답을 연결한다.

```text
순위 결과: (window_idx, anchor_line_id)
정답 결과: (window_id,  anchor_line_id)
```

이는 Window 순서를 바꾸거나 일부만 떼어내 재번호를 매기지 않았다는 전제다.
같은 원본 라인이 겹치는 Window에 반복될 수 있어 line_id만으로 연결하면 안 된다.

추출기의 ranking_eligible, evaluation_eligible, min_prior_lines를 이 코드는 읽지 않는다.
WARN/ERROR면 모두 순위를 생성하며, 실제 평가 대상 선정은 향후 평가 단계에서 해야 한다.
Alert 정답은 점수 계산에 사용되지 않는다.

## 6. 실행과 결과 확인

src에서 `python anchor.py` 실행 후 PT 경로를 입력한다. 같은 이름의 출력은 덮어쓴다.
Anchor 개수와 레벨, 후보의 자기 자신·미래 라인 제외 여부, 점수 내림차순을 확인한다.
test/test_achor.py에서 이 동작과 evidence_role 변경에 대한 순위 독립성을 검증한다.
테스트 파일명은 실제로 achor라는 철자로 저장되어 있다.

첫 라인이 계속 1위여도 그것이 장애 원인임을 의미하지 않는다.
위 예시는 인공 로그 1개 Window이며 Baseline 대비 성능 결론이 아니다.

관련 문서: [Token 진단](TOKEN_ATTENTION_PROBE_EASY_GUIDE.md).
