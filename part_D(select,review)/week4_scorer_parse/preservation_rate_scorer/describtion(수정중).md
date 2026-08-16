# 보존율 채점기 (Preservation-Rate Scorer) — 설계·진행 정리

> **정의.** KV 캐시 압축(KV cache compression) 이후 cause line의 생존 여부를
> keep mask 기준으로 판정하여 preservation rate를 산출하는 evaluation harness.
> 담당: D(리드/평가) · 마일스톤: M5(Tier-1 평가, 안전판)
>
> *본문은 프로젝트 표준 용어로 기술한다. 각 용어의 평이한 설명은 **부록 A. 용어 해설**에 둔다.*

---

## 1. 목적 — Tier-1 평가로서의 안전판 논거

프로젝트의 핵심 가설은 다음과 같다. *기존 KV 압축법은 long-context 로그를 압축할 때
rare(저빈도)·non-ERROR·lost-in-the-middle 특성을 갖는 cause line을 우선적으로 제거한다. 
LogPress는 이를 보존한다.* 보존율 채점기는 이 가설을 검정 가능한
정량 지표로 환산하는 장치다.

평가 체계는 2-tier로 구성된다.

- **Tier-2 (RCA delta).** 압축 후 backbone LLM의 root-cause analysis 정확도 향상분을 측정한다.
  결정타지만, task 난도가 높을 경우 모든 method가 하한에 수렴하는 **floor effect**(바닥효과) 로 인해
  method 간 우열이 소실될 리스크가 있다.
- **Tier-1 (preservation rate).** cause line의 생존 여부를 측정한다. **model accuracy와
  독립적으로** 산출되므로 floor effect에 강건하다.

따라서 Tier-2가 floor로 붕괴해도 **Tier-1 결과만으로 논문이 성립한다.** 이것이 M5를
"발표 가능 최소선(minimum viable result)" 이자 안전판으로 규정하고, 채점기를 D의 저부하
구간에 **pre-implementation** 하도록 지정한 근거다.

---

## 2. 아키텍처 — 3개 컴포넌트

| # | 컴포넌트 | 역할 | 상태 |
|---|---|---|---|
| A | **Data loading** | cause / keep mask / is_error / pos 를 입력 파일에서 로드 | ☐ 미착수 |
| B | **Scoring** | cause line 보존 판정 → stratified aggregation → preservation rate | ☑ **완료·검증** |
| C | **Comparison & verdict** | LogPress vs baseline, hypothesis test로 유의성 판정 | ☐ 별도 진행 |

본 작업의 산출은 **B(Scoring)** 이며, 합성 데이터로 검증까지 완료했다. A·C는 후속 단계다.

---

## 3. Interface — 입출력 명세

| 식별자 | 의미 | 타입 | Source |
|---|---|---|---|
| `cause` | cause line index 집합 | int list | dataset label (B 트랙) |
| `keep` | 압축 후 토큰 생존 여부 (keep mask) | bool array | LogPress output (C 트랙) |
| `is_error` | ERROR-level 여부 | bool array | dataset label |
| `pos` | 상대 위치 (0.0=head, 1.0=tail) | float array | dataset label |

**Output:** preservation rate 3종 — `overall` / `non_error` / `middle`.

> **의존성.** keep mask와 token-to-line mapping은 **C의 LogPress**에서 산출된다.
> 두 산출물의 포맷·좌표계(coordinate frame)를 C와 사전 합의(interface contract)해야 하며,
> 합의 이전에는 §5의 synthetic data로 채점기를 pre-implementation·검증한다.

---

## 4. Scoring 로직 (완료분)

### 4.1 보존 판정 기준 (`rule='any'`)

cause line 하나는 다수 토큰으로 subword tokenization된다. 프로젝트는 **구성 토큰 중
하나라도 keep mask에서 생존하면 해당 line을 보존**으로 판정한다(`rule='any'`). 아래 코드는
line 단위 생존/사망으로 축약한 형태이며, 실 파이프라인에서는 token-to-line mapping을 경유해
이 규칙이 적용된다.

### 4.2 Preservation rate 산출

```python
preserved = 0
for c in cause:                          # cause line iteration
    if keep[c]:                          # keep mask 조회
        preserved += 1
overall_rate = preserved / len(cause)    # preserved / total
```

### 4.3 Stratification — hard-case 부분집합 분리

overall rate 단독으로는 *level filter로도 달성 가능한 영역*이라는 반박을 차단하지 못한다.
프로젝트의 기여는 **hard case**(non-ERROR cause · 중간 위치 cause)에서 발현되므로 두
stratum을 분리한다.

```python
non_error = []                           # non-ERROR stratum
mid_position = []                        # middle-position stratum
for i in cause:
    if not is_error[i]:
        non_error.append(i)
    if 0.25 <= pos[i] <= 0.75:
        mid_position.append(i)
```

### 4.4 Stratified aggregation

scoring 로직은 불변, target stratum만 교체한다.

```python
def rate(target):
    preserved = 0
    for c in target:
        if keep[c]:
            preserved += 1
    return preserved / len(target)

overall_rate   = rate(cause)
non_error_rate = rate(non_error)
mid_rate       = rate(mid_position)
```

3개 stratum — **overall / non_error / middle** — 이 최종 output이다.

---

## 5. 검증 (sanity check)

10-line synthetic case로 수기 계산과 코드 output을 대조했다.

| cause line | keep | ERROR | pos |
|---|---|---|---|
| 2 | 생존 | O | 0.30 |
| 5 | 사망 | X | 0.55 |
| 7 | 생존 | X | 0.80 |
| 8 | 생존 | O | 0.90 |

- **overall:** 4개 중 생존 3(2·7·8) → **0.75**
- **non_error:** target 5·7 중 생존 7 → **0.50**
- **middle (0.25–0.75):** target 2·5 중 생존 2 → **0.50**

코드 실행 결과 **0.75 / 0.50 / 0.50**, 수기 계산과 일치. scoring 로직 검증 완료.

---

## 6. Progress

- ☑ Preservation rate 산출
- ☑ non_error / middle stratification
- ☑ Stratified aggregation
- ☑ Synthetic data 실행·검증 (0.75 / 0.50 / 0.50)
- ☐ Data loading (A) — 실제 CSV·array 입력
- ☐ Comparison & verdict (C) — McNemar + BH-FDR, Tier-1 판정

**현 상태:** Scoring 컴포넌트 완료. synthetic data로 pre-implementation·검증까지 마쳤으며,
C의 실제 keep mask·mapping 인계 시 §5의 mock 입력을 교체해 실측에 진입한다.

---

## 7. Next steps

1. **Data loading (A)** — 하드코딩된 `cause`·`keep`을 실제 파일 입력으로 전환. 저난도 작업.
2. **Comparison & verdict (C)** — paired **McNemar test**로 LogPress vs baseline 우열을
   검정하고, multiple comparison이므로 **BH-FDR**로 보정. hard-case stratum에서 전 baseline
   대비 유의(p_bh < 0.05) & 효과 방향 일치 시 **Tier-1 PASS**. 개념 부담이 크므로 선행 학습 필요.
3. **Interface contract with C** — keep mask 포맷·coordinate frame, 공유 token-to-line mapping을
   문서로 확정. 8주차 산출물 인계 전 완료.

---

## 부록 A. 용어 해설 (평이한 설명)

| 용어 | 쉬운 뜻 |
|---|---|
| evaluation harness | 실험 결과를 자동으로 채점하는 코드 묶음 |
| KV 캐시 압축 (KV cache compression) | LLM이 긴 글 읽을 때 쌓는 임시 메모리에서 덜 중요한 부분을 버려 용량을 줄이는 것 |
| cause line | 장애의 실제 원인이 적힌 로그 한 줄 |
| preservation rate (보존율) | 압축 후에도 원인 줄이 안 버려지고 남은 비율. 높을수록 좋음 |
| keep mask | 어느 토큰이 살아남았는지 참/거짓으로 표시한 배열 |
| eviction | 압축 과정에서 토큰(=그 자리 정보)을 버리는 것 |
| token / subword tokenization | 모델이 글을 잘게 쪼갠 최소 단위. 한 줄이 여러 조각으로 쪼개짐 |
| token-to-line mapping | 각 토큰이 로그 몇 번째 줄에서 왔는지 연결한 표 |
| rule = 'any' | 한 줄의 토큰이 하나라도 살면 그 줄은 보존된 것으로 침 |
| stratification (층화) | 전체 평균만 보지 않고, 어려운 케이스만 따로 떼어 성적을 확인하는 것 |
| hard case | 찾기 어려운 원인. ERROR가 아닌 원인 / 로그 중간에 묻힌 원인 |
| lost-in-the-middle | 긴 글의 중간에 있는 정보를 모델이 잘 놓치는 현상 |
| floor effect (바닥효과) | 문제가 너무 어려워 모든 방법이 다 0점 근처라 우열이 안 갈리는 상태 |
| Tier-1 / Tier-2 | 1층=원인이 얼마나 남았나(보존율). 2층=그래서 RCA 정답률이 오르나 |
| RCA delta | 압축 방식만 바꿨을 때 근본원인 분석 정답률이 달라지는 차이값 |
| backbone LLM | 실험에 쓰는 기반 모델 (Llama-3.1-8B 등) |
| baseline | 우리 방법과 비교할 기존 압축법 (Random·StreamingLLM·H2O·PyramidKV) |
| McNemar test | 같은 대상에 두 방법을 짝지어 대볼 때 쓰는 통계 검정 |
| BH-FDR | 여러 번 비교하면 우연히 이길 수 있어, 그걸 보정하는 방법 |
| pre-implementation | 진짜 재료가 오기 전에 미리 짜두는 것 |
| synthetic / mock data | 검증용으로 지어낸 가짜 데이터 |
| interface contract | C한테 "이 형식으로, 이 좌표계로 넘겨줘"를 못 박는 규격 합의 |
| coordinate frame (좌표계) | keep mask의 토큰 번호가 전체 시퀀스 기준인지 로그만 기준인지 하는 기준틀 |