# token_mapper.py 쉽게 이해하기

## 1. 한 문장으로 설명

로그를 토큰으로 나누면서, **각 토큰에 원본 로그의 주소를 붙이는 코드**다.
모델 추론은 하지 않고 Qwen Tokenizer를 사용한다.

이 주소가 있어야 이후 Token Attention을 Line Attention으로 합칠 수 있다.

## 2. 입력과 출력

입력은 Window 추출기가 만든 JSONL이다. 한 줄마다 로그 행 배열이 들어 있다.

```json
[{"line_id":100,"timestamp":"123","user":"server1","location":"node01","component":"demo","content":"WARN slow","level":"WARN"}]
```

이 한 줄은 로그가 하나뿐인 Window 예시다. 실제로는 여러 로그 객체가 배열에 들어간다.
출력은 입력 파일 옆에 `<입력 이름>_token_map.jsonl`로 생성된다.

| 출력 필드 | 의미 |
|---|---|
| window_idx | JSONL에서 Window의 순서, 0부터 시작 |
| text | 실제로 토큰화한 전체 문자열 |
| input_ids | 모델에 전달할 토큰 ID 배열 |
| attention_mask | 토큰 사용 여부를 나타내는 배열 |
| token_map | 토큰별 원본 라인·필드·문자 위치 |
| line_labels | 입력에 label이 있으면 보존하고, 없으면 `-` |
| line_meta | level과 evidence_role 메타데이터 |

## 3. 실행 흐름

```text
main(): Tokenizer 로드 → Window 한 개 읽기
  ↓
필드 값을 공백으로 연결하고 각 로그 끝에 개행 추가
  ↓
필드별 문자 범위와 전체 라인 문자 범위 생성
  ↓
tokenizer_log(): 토큰 ID와 Offset Mapping 받기
  ↓
map_token_to_span(): 토큰과 겹치는 범위 찾기
  ↓
결과를 JSONL 한 줄에 저장 → 다음 Window
```

## 4. 주요 구조체와 함수

### `Span` — 원본 텍스트의 구역 표시

line_id, row_idx, field, start, end를 가진다.
예를 들어 content가 전체 문자열의 20번째부터 29번째 문자라면 `[20,30)`이다.
start는 포함하고 end는 포함하지 않는다. 바이트 위치가 아니라 문자 Offset이다.

### `TokenInfo` — 토큰 하나의 주소록

| 필드 | 의미 |
|---|---|
| token_idx | Window 전체 토큰 중 위치, 0-based |
| token_id | Tokenizer 사전의 ID |
| token | Tokenizer가 표시하는 토큰 문자열 |
| start / end | 전체 입력 문자열에서 토큰의 범위 |
| line_id | 원본 로그 ID |
| row_idx | Window 안의 로그 위치, 0-based |
| field | content, timestamp 등의 필드 또는 line |

### `map_token_to_span()` — 처음 겹친 구역 선택

Span들을 앞에서부터 검사해 토큰 범위와 처음 겹치는 Span을 반환한다.
가장 많이 겹치는 구역을 찾는 방식은 아니다. 하나의 토큰이 라인 경계를 넘으면
먼저 겹친 라인에 배정된다. 어떤 문자열에서도 경계를 넘지 않는다고 보장할 수는 없다.

### `tokenizer_log()` — 토큰화와 매핑

`return_offsets_mapping=True`로 토큰의 문자 위치를 얻는다.
`add_special_tokens=False`이므로 별도 특수 토큰을 추가하지 않는다.
문자 범위가 없는 토큰(start=end)은 미매핑 상태로 보존한다.
최종적으로 Tensor와 구조체를 JSON 저장 가능한 리스트·딕셔너리로 바꾼다.

### `main()` — 파일 처리

Qwen/Qwen2.5-3B-Instruct Tokenizer를 불러오고 입력 경로를 받는다.
Window별 문자열·Span을 만들고 결과를 저장하며 확인용 내용을 화면에 출력한다.

## 5. 공백과 개행도 어느 라인에 속하는가?

각 필드의 범위 외에 공백·개행까지 포함한 전체 라인 Span을 추가한다.
필드에 걸친 토큰은 해당 필드를 사용하고, 나머지 공백·개행은 field=`line`으로 매핑한다.

실제 예시에서 Line 103의 끝은 다음과 같이 매핑됐다.

| token_idx | 원문 조각 | line_id | field |
|---:|---|---:|---|
| 78 | ` ERROR` | 103 | content |
| 79 | ` failed` | 103 | content |
| 80 | ` to` | 103 | content |
| 81 | ` read` | 103 | content |
| 82 | ` disk` | 103 | content |
| 83 | 공백 + 개행 | 103 | line |

Tokenizer 표시의 `Ġ`, `Ċ` 같은 기호는 그대로의 원문 글자가 아닐 수 있다.
원문을 확인하려면 `text[start:end]`를 사용한다.

## 6. 정답 라벨이 모델에 들어가는가?

입력 문자열은 INPUT_FIELDS의 timestamp/user/location/component/content 값만 연결한다.
필드 이름, line_id, label, level, evidence_role은 추가하지 않는다.
chat template이나 별도 지시문도 붙이지 않는다.

현재 Window 추출기는 label 자체를 모델용 JSONL에서 제외한다.
따라서 mapper 결과의 line_labels가 모두 `-`여도 모든 로그가 정상이라는 뜻은 아니다.
평가 정답은 alert_ground_truth.jsonl에서 읽어야 한다.

## 7. 실행 및 확인

src에서 `python token_mapper.py`를 실행하고 windows.jsonl 경로를 입력한다.
같은 이름의 출력은 덮어쓴다. 생성된 파일은 attention.py에 전달한다.

확인할 것은 토큰 배열 길이 일치, line_id/row_idx, 문자 범위, 미매핑 토큰,
라인 경계 침범, text 안의 정답 태그 유무다.
이러한 모든 검사가 main()에 자동 구현되어 있는 것은 아니다.
앞선 실제 8줄 예시에서는 별도 점검으로 167개 토큰의 매핑과 라벨 비유출을 확인했다.

다음 문서: [Attention 추출](ATTENTION_EASY_GUIDE.md).
