# Window 추출 코드 쉽게 이해하기

이 문서는 `window_extractor.py`의 동작을 예시 중심으로 설명한다.
기존 `WINDOW_EXTRACTOR.md`와 별도로 만든 입문용 문서다.

## 1. 이 코드가 하는 일

긴 CSV 로그를 일정한 크기의 묶음으로 자른다. 그 묶음을 **Window**라고 부른다.
각 Window에서 WARN/ERROR 로그를 찾고, 각 로그보다 앞에 있는 후보와 Alert 정답을 연결한다.

예를 들어 로그가 1,000줄이고 Window 크기가 100, stride가 100이면
겹치지 않는 Window 10개가 만들어진다.

이 단계에서는 모델을 실행하거나 중요도 순위를 계산하지 않는다.
나중에 Attention으로 순위를 계산하고 평가할 수 있도록 입력과 정답을 준비한다.

```text
이미 파싱된 CSV
    ↓
Window 단위로 나누기
    ↓
Window 안의 WARN/ERROR Anchor 찾기
    ↓
각 Anchor의 이전 후보·이전 Alert 정답 연결
    ↓
모델용 로그 / 평가 정답 / Window 정보 / 전체 통계 저장
```

## 2. 먼저 알아둘 용어

| 용어 | 쉬운 뜻 |
|---|---|
| Window | 한 번에 분석할 로그 묶음 |
| window_size | 한 묶음에 담을 최대 로그 수 |
| stride | 다음 묶음의 시작점을 몇 줄 이동할지 |
| Anchor | Query로 사용할 WARN/ERROR 로그 |
| Candidate | 같은 Window 안에서 Anchor보다 앞에 있는 로그 |
| Alert | 원본 label이 나타내는 평가 정답 |
| line_id | CSV에 들어 있는 원본 로그 ID |
| anchor_position | Window 안의 위치. 1부터 시작 |

**Anchor와 Alert는 다르다.**
Anchor는 content의 키워드로 고르고, Alert는 원본 label로 판단한다.
ERROR라는 단어가 있어도 label이 `-`이면 Alert 정답은 아니다.
Alert는 실제 장애의 근본 원인이라는 뜻도 아니다.

## 3. Window는 어떻게 잘리는가?

실제로 검증한 입력은 Line ID 100부터 107까지의 8줄이다.
아래 숫자는 Window 안의 위치가 아니라 원본 Line ID다.

### 겹치지 않게 자르기: 크기 4, stride 4

```text
Window 0: [100, 101, 102, 103]
Window 1: [104, 105, 106, 107]
```

모든 로그가 한 번씩 들어간다.

### 겹치게 자르기: 크기 4, stride 2

```text
Window 0: [100, 101, 102, 103]
Window 1: [102, 103, 104, 105]
Window 2: [104, 105, 106, 107]
Window 3: [106, 107]
```

시작점이 두 줄씩 이동하므로 같은 로그가 여러 Window에 들어간다.
기본 설정은 불완전 Window도 보존하므로 마지막 `[106, 107]`도 출력된다.
이는 처음 정한 시작점 규칙을 EOF까지 적용한 결과다.

### 끝에 남은 로그 처리: 크기 3, stride 3

```text
include: [100,101,102] / [103,104,105] / [106,107]
drop:    [100,101,102] / [103,104,105]
```

`--tail-policy include`가 기본값이다. `drop`이면 마지막 2줄이 출력에서 빠지고
요약의 `unwindowed_log_count`에 2가 기록된다.

stride가 Window 크기보다 크면 중간에 빠지는 행이 생긴다.
모든 로그를 겹침 없이 확인하려면 stride를 Window 크기와 같게 사용한다.

## 4. Anchor별 후보와 정답은 어떻게 만들어지는가?

8줄을 하나의 Window로 넣었던 예시다.

| 위치 | Line ID | content 요약 | 원본 Alert 여부 | Anchor |
|---:|---:|---|---|---|
| 1 | 100 | WARN startup is slow | 아니오 | WARN |
| 2 | 101 | disk latency exceeded threshold | 예 | 없음 |
| 3 | 102 | worker heartbeat received | 아니오 | 없음 |
| 4 | 103 | ERROR failed to read disk | 아니오 | ERROR |
| 5 | 104 | connection timeout detected | 예 | 없음 |
| 6 | 105 | WARNING retry queue is growing | 아니오 | WARN |
| 7 | 106 | ERROR exception during recovery | 아니오 | ERROR |
| 8 | 107 | service became unavailable | 예 | 없음 |

이때 Anchor별 결과는 다음과 같다.

| Anchor | 후보 | 이전 Alert 정답 |
|---|---|---|
| 100 | 없음 | 없음 |
| 103 | 100, 101, 102 | 101 |
| 105 | 100, 101, 102, 103, 104 | 101, 104 |
| 106 | 100, 101, 102, 103, 104, 105 | 101, 104 |

자기 자신과 미래 로그는 후보가 아니다. 이전 Anchor는 이후 Anchor의 후보가 될 수 있다.
107은 Alert이지만 모든 Anchor 뒤에 있으므로 어떤 Anchor의 정답에도 들어가지 않는다.

**Window 밖의 이전 로그는 후보에 포함되지 않는다.**
크기 4로 나누면 두 번째 Window는 `[104,105,106,107]`이다.
이때 Anchor 105의 후보는 `[104]`이며, 앞 Window의 101은 정답에 들어가지 않는다.
그래서 같은 원본 Anchor라도 Window 구성이 바뀌면 후보와 정답이 달라질 수 있다.

## 5. 주요 함수: 읽기 → 묶기 → 연결하기 → 저장하기

### `main()` — 실행 옵션을 받는 입구

CSV 경로, 저장 폴더, 크기, stride 등의 옵션을 읽고 `extract_windows()`를 호출한다.

### `extract_windows()` — 전체 작업 관리자

옵션을 검사하고 출력 파일을 연다. 아래 함수들을 연결해 Window를 처리하며,
Window 하나가 완성될 때마다 즉시 파일에 기록한다. 마지막에는 통계를 저장한다.
기존 출력 파일이 있으면 덮어쓰지 않고 중단한다.

### `_read_rows()` — CSV를 한 행씩 읽기

필수 11개 필드와 line_id를 검사한다. 원본 순서를 유지하며 ID는 바꾸지 않는다.
line_id는 비음수 정수이고 엄격히 증가해야 한다. ID 사이의 간격은 허용하고 기록한다.
CSV 구조나 ID가 잘못되면 행을 건너뛰지 않고 오류로 중단한다.

여기서 각 행에 다음 두 함수를 적용한다.

| 함수 | 입력 | 하는 일 |
|---|---|---|
| `classify_label()` | 원본 label | Alert / Non-alert / 불명확 구분 |
| `detect_level()` | content | ERROR / WARN / INFO 구분 |

label은 양끝 공백을 제거한 후 `-`이면 정상이다. 누락값, 빈 값, 알려진 누락 표시,
내부 공백이 있는 라벨은 불명확으로 처리한다. 나머지 단일 태그는 Alert다.
유효 태그 사전이 없으므로 모든 라벨 오타를 검출하는 것은 아니다.

키워드는 error/errors, fail/failed/failing/failure/failures, exception/exceptions,
warn/warning/warnings를 단어 경계와 대소문자 무시 규칙으로 검사한다.
ERROR와 WARN이 함께 있으면 ERROR가 우선한다. 없으면 INFO다.
`no errors`도 ERROR로 매칭하므로 실제 Severity와 일치한다고 보장할 수 없다.
혼합 키워드·부정 표현·빈 content는 통계에 따로 기록한다.

### `_iter_windows()` — 로그를 묶기

`deque`라는 버퍼에 최대 Window 크기만큼 행을 담는다.
가득 차면 Window를 내보내고, 다음 시작점보다 앞선 행을 버린다.
겹치는 행은 버퍼에 남겨 다음 Window에 재사용한다.

`yield`는 결과를 하나씩 전달하는 방식이다. 전체 CSV를 한꺼번에 메모리에 넣지 않는다.

### `_build_records()` — 후보와 정답 연결하기

Window를 처음부터 끝까지 읽으며 이전 로그 ID와 이전 Alert ID를 누적한다.
Anchor를 만나면 **지금까지 누적한 목록을 복사**해 해당 Anchor에 연결한다.
그다음 현재 행을 누적 목록에 추가한다. 따라서 자기 자신은 후보에서 빠진다.

이 함수는 모델용 로그 배열, 평가 정답, Window 메타데이터를 반환한다.
Anchor별 후보 목록은 한 Window 안에서 최대 크기의 제곱에 비례할 수 있지만,
전체 데이터셋의 후보 목록을 계속 메모리에 쌓아 두지는 않는다.

### `_write_jsonl()` — 한 Window씩 저장하기

JSON 객체를 한 줄로 기록한다. JSONL은 이런 JSON 줄들이 모인 파일이다.
화면에서 길어 보여도 windows.jsonl의 한 줄은 Window 하나다.

## 6. 생성되는 파일 4개

| 파일 | 무엇이 들어 있나? | 언제 사용하나? |
|---|---|---|
| `windows.jsonl` | 로그 행 배열, 파생 level | token_mapper.py에 직접 입력 |
| `alert_ground_truth.jsonl` | Anchor별 후보·이전 Alert·원본 라벨 | 나중에 순위와 정답 비교 |
| `window_metadata.jsonl` | 원본 범위, 크기, Anchor 요약, 중복 정보 | Window가 잘 잘렸는지 확인 |
| `extraction_summary.json` | 전체 개수, 설정, 오류, 실행 상태 | 전체 추출이 성공했는지 확인 |

세 JSONL 파일의 같은 번째 줄은 같은 Window다. window_id는 0부터 시작한다.
파일을 그대로 Token Mapper에 전달하면 window_id와 window_idx가 대응한다.

정답 label은 windows.jsonl에서 제외된다. 기존 Token Mapper는
timestamp/user/location/component/content만 모델 입력 텍스트로 만든다.
level은 메타데이터로 보존된다. Token Mapper 출력의 기본 line_labels=`-`를
실제 평가 정답으로 사용하면 안 된다. 정답은 별도 Ground Truth 파일에서 읽는다.

## 7. '순위를 만들 수 있음'과 '평가할 수 있음'의 차이

| 항목 | true가 되는 조건 |
|---|---|
| `ranking_eligible` | 이전 후보가 1개 이상이고 min_prior_lines 이상 |
| `evaluation_eligible` | 위 조건 + 모든 후보의 라벨을 알고 있음 + 이전 Alert가 존재 |

후보는 있지만 Alert가 없으면 순위 생성은 가능해도 정답 기반 평가는 불가능하다.
두 경우 모두 삭제하지 않고 파일에 남긴다.
`--min-anchors`도 Window를 필터링하지 않고 조건 충족 여부만 표시한다.
이 설정은 Alert나 Anchor가 많은 Window만 골라 생기는 표본 편향을 피하기 위한 것이다.

## 8. 실제로 잘 추출됐는지 확인한 결과

검증 날짜: 2026-11-09.
추출 코드와 기존 설명 문서를 변경하지 않고 다시 실행했다.

### 자동 테스트

추출 관련 **53개 테스트 통과**.
Window 크기 10/50/100, 순서 보존, 다중 Anchor, 이전 후보·정답,
겹침, 불완전 Window, 라벨 정책, 잘못된 CSV, CLI 등을 확인했다.
Window 경계 테스트에서는 길이·크기·stride·끝부분 정책의 1,260개 조합을
단순 배열 슬라이싱 결과와 대조했다.

이번 실행에서는 `actual_token_mapper` 통합 테스트 1개를 제외했다.
Token Mapper 통합과 실제 Qwen Tokenizer 연결은 앞선 작업에서 별도로 확인했다.

### 예시 CSV를 실제 출력 파일로 추출

| 설정 | 실제 Window 구성 | 출력에 포함되지 않은 원본 행 | 결과 |
|---|---|---:|---|
| 크기 4 / stride 4 | 100–103, 104–107 | 0 | 통과 |
| 크기 4 / stride 2 / include | 100–103, 102–105, 104–107, 106–107 | 0 | 통과 |
| 크기 3 / stride 3 / include | 100–102, 103–105, 106–107 | 0 | 통과 |
| 크기 3 / stride 3 / drop | 100–102, 103–105 | 2 | 통과 |

각 결과의 Anchor별 후보가 같은 Window의 이전 행과 정확히 일치하는지,
정답에 미래 Alert가 들어가지 않는지, 모델용 행에 label/is_anomaly가 없는지도 확인했다.

결과는 `examples/window_check_20261109/` 아래에 보존했다.

- `nonoverlap/`: 겹침 없는 출력 파일 4개
- `overlap/`: 겹치는 출력 파일 4개
- `partial_include/`: 마지막 불완전 Window를 보존한 출력
- `partial_drop/`: 마지막 불완전 Window를 제외한 출력
- `verification_results.json`: 네 경우의 Window ID 목록과 검증 결과

소형 예시와 테스트 범위에서는 정상이다. 실제 대용량 Thunderbird CSV 전체를
이번에 실행한 것은 아니므로 전체 데이터의 품질이나 처리 시간까지 검증한 것은 아니다.

## 9. 직접 실행하고 확인하기

src 폴더에서 다음처럼 실행한다. 출력 폴더는 기존 결과가 없는 새 이름을 사용한다.

```powershell
python window_extractor.py --input "examples/multi_anchor_demo/sample.csv" --output-dir "examples/my_window_check" --window-size 4 --stride 4
```

확인 순서는 다음과 같다.

1. extraction_summary.json에서 `status`가 `complete`인지 확인한다.
2. 처리 로그 수와 Window 수가 예상과 맞는지 확인한다.
3. window_metadata.jsonl에서 각 시작·끝 ID와 실제 크기를 확인한다.
4. windows.jsonl의 행 순서를 원본 CSV와 비교한다.
5. alert_ground_truth.jsonl에서 Anchor 이전 후보·정답만 들어 있는지 확인한다.

`--max-windows`는 개발용으로 앞부분만 실행하는 옵션이다.
이 옵션으로 중단한 결과를 전체 CSV 통계로 해석하면 안 된다.
겹치는 Window와 같은 Window 안의 여러 Anchor는 독립적인 장애 사건이 아니다.
