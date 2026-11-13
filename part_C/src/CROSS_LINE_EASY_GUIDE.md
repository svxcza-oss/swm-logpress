# cross_line.py 쉽게 이해하기

## 1. 역할

모델은 토큰 단위로 Attention을 계산하지만 연구에서는 로그 한 줄 단위의 관계가 필요하다.
이 파일은 **같은 로그에 속한 토큰들을 묶어 Line Attention을 계산**한다.
주요 함수는 `convert_to_line_attention(layer_attentions, token_map)` 하나다.
파일 입출력이나 독립 실행 CLI는 없으며 attention.py에서 호출한다.

## 2. 입력과 반환값

| 항목 | 내용 |
|---|---|
| layer_attentions | Layer별 `[1,head,Q_token,K_token]` Tensor |
| token_map | token_idx와 line_id에 접근 가능한 객체 목록 |
| 반환 line_attentions | `[layer,head,Q_line,K_line]` CPU float32 Tensor |
| 반환 line_ids | 라인 축의 원본 ID 목록 |

예를 들어 line_ids가 `[100,103]`이면 배열 인덱스 0은 Line 100이고 1은 Line 103이다.
ID 숫자 자체를 Tensor 인덱스로 사용하면 안 된다.
순서는 토큰 매핑에 처음 등장한 순서이며 ID 오름차순 정렬은 하지 않는다.

## 3. 핵심 규칙: Key는 더하고 Query는 평균

- Key SUM: 같은 로그에 속한 Key 토큰들이 받은 점수를 더한다.
- Query MEAN: 같은 로그에 속한 Query 토큰들의 결과를 평균한다.
- Layer와 Head는 합치지 않고 그대로 유지한다.

### 숫자로 이해하기

Line A에 Key 토큰 k0, k1이 있고, Line B에 Query 토큰 q2, q3이 있다고 하자.

| Query | k0에 준 점수 | k1에 준 점수 | Line A에 준 합계 |
|---|---:|---:|---:|
| q2 | 0.1 | 0.2 | 0.3 |
| q3 | 0.2 | 0.4 | 0.6 |

Line B가 Line A에 준 점수는 `(0.3 + 0.6) / 2 = 0.45`다.
Key를 평균해서 0.225로 만들거나, Query를 합산만 해서 0.9로 만들지 않는다.

수식은 다음과 같다.

```text
A(Line Q, Line K)
  = Q 라인의 각 토큰이 K 라인 전체에 준 Attention 합의 평균
  = Σ(q ∈ Q) Σ(k ∈ K) A(q,k) / Q 라인의 토큰 수
```

## 4. 함수 안에서 하는 일

1. 매핑된 line_id 목록을 등장 순서대로 만든다.
2. 토큰이 어떤 라인에 속하는지 나타내는 member 행렬을 만든다.
3. Layer 하나씩 가져와 batch=1 축을 제거한다.
4. Query·Key 토큰 수가 token_map 길이와 일치하는지 검사한다.
5. `attention @ member`로 Key를 라인별 합산한다.
6. `member.T @ 결과`로 Query를 라인별 합산한다.
7. Query 라인별 토큰 수로 나누어 평균한다.
8. Layer별 결과를 CPU float32로 옮겨 쌓는다.

member는 예를 들어 다음과 같다.

```text
          Line A  Line B
token 0      1      0
token 1      1      0
token 2      0      1
token 3      0      1
```

행렬곱은 이 소속 정보를 이용해 같은 라인의 점수를 한 번에 모으는 계산이다.

## 5. 미매핑 토큰과 예외

line_id=None인 토큰은 Query와 Key 집계에서 제외한다.
제외된 점수를 다른 라인에 다시 나누어 주지 않는다.
예를 들어 0.4가 미매핑 Key로 향했다면 나머지 라인의 행 합이 0.6일 수 있다.

유효한 라인이 하나도 없거나 Attention의 토큰 수가 맞지 않으면 ValueError를 발생시킨다.
batch=1과 정상적인 token_idx를 전제로 하며 모든 잘못된 입력을 별도로 검사하는 것은 아니다.
계산은 입력 Tensor의 dtype으로 수행한 뒤 결과를 float32로 변환한다.
따라서 float16 계산에서 생긴 오차가 마지막 변환으로 사라지지는 않는다.

## 6. 해석 시 기억할 점

Key는 SUM이므로 긴 Key 라인은 토큰 수의 영향을 받을 수 있다.
이 코드는 그 영향을 보정하지 않는다. attention.py가 line_token_counts를 저장하는 이유다.
집계 행렬에는 자기 라인도 포함된다. 자기 자신을 순위 후보에서 제외하는 일은 anchor.py에서 한다.

관련 테스트는 test/test_cross_line.py의 6개 테스트다.
정확한 집계 수치, Layer/Head 유지, 미매핑 제외, 순서와 입력 길이 오류를 검사한다.

다음 문서: [Anchor 순위](ANCHOR_EASY_GUIDE.md).
