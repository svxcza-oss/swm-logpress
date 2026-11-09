# attention.py 쉽게 이해하기

## 1. 이 파일의 역할

**실제 Qwen 모델을 실행하고, 토큰 사이 Attention을 로그 라인 사이 Attention으로 연결하는 관리자**다.
Tokenizer 실행 단계와 달리 모델 가중치를 로드하므로 메모리와 가속 장치 사용이 발생한다.
텍스트를 생성하는 generate() 대신 입력 전체에 대해 한 번의 forward를 수행한다.

## 2. 입력과 출력

입력: token_mapper.py의 `*_token_map.jsonl`.
모델에는 input_ids와 attention_mask만 전달한다. 메타데이터는 분석용으로 보존한다.

출력: 입력 옆의 `*_token_map_line_attention.pt`.
내용은 Window별 딕셔너리를 담은 Python 리스트다.

| 필드 | 내용 |
|---|---|
| window_idx | 원본 Window 번호 |
| line_ids | Tensor의 라인 축과 대응하는 원본 ID 목록 |
| line_attentions | `[layer, head, query_line, key_line]` |
| head_mean_attentions | Head만 평균한 `[layer, query_line, key_line]` |
| line_token_counts | line_ids 순서에 맞춘 토큰 수 |
| line_meta / line_labels | 입력의 메타데이터 |

원본 Token Attention 전체는 PT에 저장하지 않는다.
별도로 token_probe_results/token_bias_by_head.csv에 진단 결과를 추가한다.

## 3. 실행 흐름

```text
main(): 입력 경로 확인 → 장치 선택 → 모델 로드
  ↓
token_attention(): 각 Window 순회
  ├─ token_map 딕셔너리를 TokenInfo로 복원
  ├─ input_ids, attention_mask를 [1,T] Tensor로 변환
  ├─ 모델 forward → Layer별 Token Attention
  ├─ save_token_probe() → 첫 토큰·첫 라인 진단 CSV
  ├─ convert_to_line_attention() → Line Attention
  └─ 토큰 수·메타데이터와 함께 결과 리스트에 추가
  ↓
main(): 모든 Window 결과를 PT 파일로 저장
```

## 4. 주요 함수

### `main()` — 환경 설정과 저장

장치는 XPU → CUDA → CPU 순서로 선택한다.
XPU/CUDA는 float16, CPU는 float32를 사용한다.
Qwen/Qwen2.5-3B-Instruct를 `attn_implementation="eager"`로 불러온다.
`model.eval()`로 평가 모드를 설정한다.

### `token_attention(fin, model, device)` — Window별 실제 작업

파일을 순회하며 토큰 매핑을 복원한다. 미매핑 토큰은 출력으로 알리지만 삭제하지 않는다.
모델 호출 옵션은 다음과 같다.

| 옵션 | 의미 |
|---|---|
| output_attentions=True | Attention을 결과에 포함 |
| use_cache=False | KV Cache를 사용하지 않음 |
| return_dict=True | outputs.attentions처럼 이름으로 접근 |
| torch.no_grad() | 학습용 Gradient를 기록하지 않음 |

여기서 KV Cache를 압축하는 것은 아니다.

## 5. Tensor 모양 읽는 법

T는 토큰 수, N은 로그 라인 수다.

```text
모델 입력: [1, T]
outputs.attentions: Layer별 Tensor를 담은 튜플
각 Layer Tensor: [1, head, T, T]
Line 집계 후: [layer, head, N, N]
```

Q축은 Attention을 주는 토큰/라인, K축은 받는 토큰/라인이다.
앞선 8줄·167토큰 예시에서 출력은 `[36,16,8,8]`이었다.
축 크기를 임의로 고정하지 말고 실제 결과의 shape를 확인한다.

## 6. 결과가 정상인지 확인하기

화면에 line_ids, 라인별 토큰 수, 첫 Layer/Head의 행 합이 출력된다.
행 합은 한 Query 라인이 전체 Key 라인에 준 Attention의 총합이다.
미매핑 토큰이 없으면 보통 1 부근이고, 낮은 정밀도의 반올림 차이가 있을 수 있다.
미매핑 Key에 향한 Attention은 집계에서 빠지므로 행 합이 작아질 수 있다.
현재 코드는 행 합을 출력하며 1인지 강제로 검사하지 않는다.

앞선 실제 실행은 NaN/무한대 없이 행 합 약 0.99933~1.00067이었다.
이는 저장된 예시 실행 결과이며 이번 문서 작성에서 재실행한 결과는 아니다.

## 7. 실행 시 알아둘 점

src에서 `python attention.py`를 실행하고 Token Mapping JSONL 경로를 입력한다.
출력 PT는 기존 파일을 덮어쓴다. 진단 CSV는 **실행 위치 기준** 경로에 누적된다.
진단 함수도 매번 호출되므로 그 함수의 오류가 전체 처리를 중단시킬 수 있다.

Window마다 큰 모델 출력의 참조를 지우지만, CPU의 Line Attention 결과는 리스트에 계속 쌓는다.
Token Attention은 토큰 수의 제곱에 비례하므로 Window가 길면 메모리 부담이 커진다.
추출기에서 큰 Window를 만들 수 있다는 사실과 모델에서 처리 가능하다는 사실은 다르다.
현재 코드는 자동 토큰 길이 제한이나 Window별 중간 PT 저장을 제공하지 않는다.

다음 문서: [Line 집계](CROSS_LINE_EASY_GUIDE.md), [Anchor 순위](ANCHOR_EASY_GUIDE.md).
