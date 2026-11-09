# LogPress 팀원용 코드 안내

2026-11-09 현재 src의 실제 코드를 기준으로 작성했다. 처음 합류한 팀원은 이 문서부터 읽으면 된다.

## 프로젝트에서 지금 하는 일

로그를 Window로 나누고, 모델이 로그의 어느 부분에 Attention을 주는지 살펴본다.
WARN/ERROR 로그를 Query로 골라 이전 로그에 순위를 매기는 단계까지 연결되어 있다.
Alert 정답은 평가용으로 분리한다. Alert를 근본 원인 정답이라고 해석하지 않는다.

## 추천 읽기 순서

| 순서 | 문서 | 핵심 질문 |
|---|---|---|
| 1 | [Window 추출](WINDOW_EXTRACTOR_EASY_GUIDE.md) | 로그를 어떻게 나누고 후보·정답을 준비하는가? |
| 2 | [Token Mapper](TOKEN_MAPPER_EASY_GUIDE.md) | 토큰이 원래 어느 로그에 속했는가? |
| 3 | [Attention 추출](ATTENTION_EASY_GUIDE.md) | 실제 모델에서 어떤 Tensor를 받는가? |
| 4 | [Line Attention 집계](CROSS_LINE_EASY_GUIDE.md) | 토큰 점수를 어떻게 로그 점수로 합치는가? |
| 5 | [Anchor 순위](ANCHOR_EASY_GUIDE.md) | WARN/ERROR에서 이전 로그를 어떻게 정렬하는가? |
| 6 | [Token 진단](TOKEN_ATTENTION_PROBE_EASY_GUIDE.md) | 첫 토큰·첫 라인에 점수가 몰리는가? |
| 7 | [테스트 안내](TESTS_EASY_GUIDE.md) | 무엇을 검증했고 어디까지 믿을 수 있는가? |

## 파일 간 연결

```text
파싱된 CSV
  ↓ window_extractor.py
windows.jsonl ─────────────── 별도 alert_ground_truth.jsonl
  ↓ token_mapper.py                      ↓ 향후 평가에서 사용
windows_token_map.jsonl
  ↓ attention.py → token_attention__probe.py → 진단 CSV
  │       └──────→ cross_line.py
  ↓
windows_token_map_line_attention.pt
  ↓ anchor.py
windows_token_map_line_attention_anchor.jsonl
```

현재 src에는 `evaluation.py`가 없다. 과거 버전의 설명을 현재 구현으로 간주하지 않는다.
Anchor 순위와 새 Alert Ground Truth를 자동 평가하는 단계는 이 문서의 실행 흐름에 포함하지 않는다.

## 헷갈리기 쉬운 표현

| 표현 | 의미 |
|---|---|
| Query / Q | Attention을 주는 쪽 |
| Key / K | Attention을 받는 쪽 |
| Layer | 모델 내부의 처리 층 |
| Head | 각 층에서 별도로 계산되는 Attention 단위 |
| Anchor | Query로 선택한 WARN/ERROR 로그 |
| Alert | CSV의 원본 label로 얻은 정답 |
| window_id / window_idx | 원본 JSONL 순서를 유지할 때 대응하는 Window 번호, 0-based |
| line_id | 원본 로그 식별자. Tensor의 배열 인덱스가 아님 |

## 직접 실행하는 순서

아래 명령은 src 폴더에서 실행한다. Python 환경에는 torch와 transformers가 필요하다.
모델이나 Tokenizer가 캐시에 없으면 Hugging Face에서 내려받으려 할 수 있다.
Window 추출기는 표준 라이브러리만으로 실행할 수 있다.

```powershell
python window_extractor.py --input "examples/multi_anchor_demo/sample.csv" --output-dir "examples/team_trial" --window-size 8
python token_mapper.py
```

Token Mapper가 경로를 물으면 `examples/team_trial/windows.jsonl`을 입력한다.

```powershell
python attention.py
```

경로 질문에 `examples/team_trial/windows_token_map.jsonl`을 입력한다.

```powershell
python anchor.py
```

경로 질문에 `examples/team_trial/windows_token_map_line_attention.pt`를 입력한다.
경로를 묻는 프롬프트에는 경로 자체만 입력한다. 따옴표까지 붙이면 파일명 일부로 해석된다.

Window 추출기는 기존 출력 파일이 있으면 중단한다. 반면 mapper, attention, anchor는
같은 이름의 결과를 덮어쓴다. 진단 CSV는 현재 작업 폴더의 token_probe_results에 누적한다.
실험을 구분하려면 입력·출력 이름과 실행 위치를 함께 관리한다.

## 실제 예시를 볼 위치

`examples/multi_anchor_demo/output/`에 이전 실제 실행 결과가 있다.
8줄·167토큰 입력에서 Line Attention은 `[36,16,8,8]`이었다.
이 숫자는 해당 모델과 예시에 대한 결과이며 모든 입력의 고정 크기가 아니다.

`anchor_ranking_with_alerts.json`은 순위와 별도 Alert 정답을 사후 연결해 만든 보조 결과다.
anchor.py가 기본 생성하는 파일은 아니다. 같은 폴더의 model_input.txt,
token_mapping_validation.json, attention_validation.json도 실행 후 만든 점검 자료다.

## 실험 해석 시 공통 기준

- 모델 입력은 timestamp/user/location/component/content다. 별도 정답 라벨은 넣지 않는다.
- Attention 점수가 높다는 사실만으로 해당 로그가 원인 또는 Alert라고 확정하지 않는다.
- 겹치는 Window와 같은 Window의 여러 Anchor를 독립적인 장애 사건으로 세지 않는다.
- 현재 구현은 Sink 보정, KV Cache 압축, Baseline 성능 비교를 수행하지 않는다.
- 모델 추론은 Window 단위지만 Line Attention 결과는 전체 목록으로 메모리에 모은다.
  추출기가 스트리밍이라는 이유만으로 전체 파이프라인의 메모리가 일정하다고 보면 안 된다.

이번 문서 작업에서는 코드를 변경하거나 모델을 다시 실행하지 않았다.
