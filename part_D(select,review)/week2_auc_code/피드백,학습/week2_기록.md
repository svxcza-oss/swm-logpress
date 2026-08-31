# AUC 판정 — 공부 노트 (사고 흐름 정리)

> 원인 줄에 신호가 있는지 판정하는 AUC를, 개념부터 코드까지 흐름대로 정리한 노트.

---

## 1. 출발: TP·FP로 AUC를 구하는 표준 방식

먼저 분류의 기본 표(혼동 행렬)부터 봤다.

| | 실제 양성 | 실제 음성 |
|---|---|---|
| **예측 양성** | TP (정답) | FP (오답) |
| **예측 음성** | FN (오답) | TN (정답) |

- T = 정답(True), F = 오답(False), P = 양성 예측, N = 음성 예측
- **TPR (재현율)** = TP / (TP + FN) — 실제 양성 중 맞힌 비율
- **FPR** = FP / (FP + TN) — 실제 음성 중 틀린 비율

**ROC 곡선** = 임계값을 쭉 훑으며 (FPR, TPR) 점을 찍어 그린 곡선.
**AUC** = 그 ROC 곡선 아래 넓이. (0~1, 높을수록 구별 잘함)

> 첫 의문: "이걸 어떻게 파일로 그리지? 일단 AUC·ROC는 내장 함수가 해준다."

---

## 2. 막힘: 우리 파일엔 TP·FP를 셀 수가 없다

`positives_grid_1h_2h.csv`에서 줄마다 라벨을 세어 TP·FP·FN·TN을 구하려 했는데 —
표준 방식(임계값→예측→TP/FP)은 **모델의 예측 점수**를 전제한다. 그런데 킬스위치엔 모델이 없다.

그래서 방향을 틀었다:

```
양성/음성  구별  →  정답 (라벨)
    ↓        ↓
   위치    희귀도    → 신호(score)
    ↓        ↓
   ③ hard-case 위치
    ↓        ↓
  True     False
```

→ **TP/FP를 세는 게 아니라, "위치라는 score로 양성/음성이 갈리나"를 보는 문제**로 재정의.

---

## 3. 전환: 위치 점수 → Mann-Whitney U

TP/FP 없이 AUC를 구하는 길이 있다 — **순위(rank) 공식**.

$$\text{AUC} = \frac{R_{pos} - \dfrac{n_{pos}(n_{pos}+1)}{2}}{n_{pos} \times n_{neg}}$$

- $R_{pos}$ : 양성들의 순위 합 (양성+음성 전체를 정렬해 순위 매긴 뒤 양성 것만 합산)
- $n_{pos}, n_{neg}$ : 양성·음성 개수 (전체 기준)

> 의문: "왜 AUC가 이렇게 계산되지?" → 순위만 보므로 값의 크기가 아니라 **순서**만 반영하기 때문.

---

## 4. 핵심 이해: 왜 `y_score`에 위치만 넣으면 되나

```
roc_auc_score(y_true, y_score)
              ↑        ↑
          양성/음성   score
```

- 원래 `y_score`는 "모델이 예측한 양성일 확률"로 설명된다.
- **하지만 확률일 필요가 없다.** 함수는 값의 의미를 안 보고 **순서(순위)만** 본다.
- 따라서:
  - 모델 확률 넣으면 → 모델이 양성/음성 잘 나누나
  - **위치를 넣으면 → 위치가 양성/음성 잘 나누나**
  - 둘 다 함수는 똑같이 순위로 처리한다.

> **결론: `y_score`는 "양성일수록 크거나 작은, 순서 있는 점수"면 된다. 위치가 바로 그것.**
> `roc_auc_score`가 내부에서 하는 계산이 곧 Mann-Whitney U다.

---

## 5. 코드로 옮기기

**뼈대 (③ 기준)**

```python
import pandas as pd
from sklearn.metrics import roc_auc_score

# 1. 읽기
df = pd.read_csv('positives_grid_1h_2h.csv')

# 2. 시간순 정렬 (창 안에서 순서를 매기려면 필수)
df = df.sort_values(['node', 'grid_id_1h', 'timestamp'])

# 3. 창마다 rank(0부터), 창 크기 n
df['rank'] = df.groupby(['node', 'grid_id_1h']).cumcount()
df['n']    = df.groupby(['node', 'grid_id_1h'])['line_id'].transform('size')

# 4. 1줄 창 제외 (n=1이면 위치 정의 불가)
df = df[df['n'] >= 2]

# 5. 위치, 정답
y_score = df['rank'] / (df['n'] - 1)
y_test  = df['is_hardcase']

# 6. AUC
print(roc_auc_score(y_test, y_score))
```

**막혔던 문법 3가지 (C 사고 → pandas 사고)**

| 헷갈린 것 | 바른 것 | 이유 |
|---|---|---|
| `len(rank)` = 전체 길이 | `transform('size')` | 창마다 다른 n이 필요 |
| `df[df.n>=2]`를 for/if로 | `df[df['n']>=2]` 한 줄 | pandas는 전체를 한 번에 거름 |
| `y_test = 'node'` | `y_test = df['is_hardcase']` | 정답은 양성/음성 라벨 |

> 배운 것: pandas는 하나씩 도는(for) 게 아니라 **전체를 한 번에** 처리한다.
> 계산 결과는 `df['이름']`으로 저장해야 걸러도 짝이 안 깨진다.

---

## 6. 순열검정 & p값

AUC 값만으론 반쪽. "이 신호가 우연이 아님"을 순열검정으로 증명한다.

**원리**
```
1. 라벨(y_test)을 무작위로 섞는다 → 위치-정답 관계를 끊음 = 신호 없는 가짜 상황(H0)
2. 섞은 라벨로 AUC를 잰다        → 방금 짠 roc_auc_score 재활용
3. 5,000번 반복 → "우연일 때의 AUC 분포"
4. 실제 AUC가 그 분포보다 극단적인 비율 = p값
```

**코드**
```python
import numpy as np
from sklearn.metrics import roc_auc_score

real_auc = roc_auc_score(y_test, y_score)
n_perm = 5000
count = 0
rng = np.random.default_rng(42)   # 재현용 시드

for i in range(n_perm):
    shuffled = rng.permutation(y_test.values)
    perm_auc = roc_auc_score(shuffled, y_score)
    if abs(perm_auc - 0.5) >= abs(real_auc - 0.5):   # 양측
        count += 1

p_value = (count + 1) / (n_perm + 1)
print(real_auc, p_value)
```

- `permutation` : 라벨만 섞기 (위치는 그대로)
- `abs(auc - 0.5)` : 양측검정 — 0.5에서 벗어난 정도로 비교
- `(count+1)/(n_perm+1)` : p값. +1은 관례(0 방지)
- **핵심: 여기선 for가 맞다.** "5,000번 반복"은 진짜 반복 작업.

> 판정: **AUC ≥ 0.65 (크기) AND p < 0.05 (우연 아님) → GO**

---

## 7. 마무리 감상

- 저번주 멘토에게 피드백을 듣고, 난 이해했다고 생각했는데 막상 질문을 받고 말하는 태도에서 문제점이 있었던것 같다. 한번 문서를 작성했다고 완전히 이해했다고 말할수는 없을것 같다. 반복해서 말해보고, 사고흐름을 연습해야겠다는 생각이 많이 들었다. 
- 모르거나 당황할때는 잠시 이해하는 시간을 양해하고 말을 하는것이 좋은 대처법이라 배웠다.
- 파이썬의 사고방식 적응이 안됨. 반복하면 나아질것 같다.
- 그래프(ROC)만으론 ①②③ 차이가 잘 안 보인다 — 셋 다 0.5 근처라 곡선이 비슷하기 때문. 차이는 **"0.5에서 얼마나·어느 방향으로 벗어났나"**로 봐야 한다.