# ObservedAttention Alert Retention Results

## 1. 실험 목적

Thunderbird 로그 데이터에서 alert 로그가 KV cache 압축 이후 얼마나 보존되는지 확인하기 위해
`ObservedAttentionPress`를 사용하여 line-level retention을 측정했다.

본 결과는 `window_314927_chunk13`을 약 8K token 단위의 line-preserving subchunk로 분할한 뒤,
각 subchunk에 대해 ObservedAttention을 독립적으로 적용한 **preliminary 결과**이다.

---

## 2. 실험 설정

- Model: `Qwen/Qwen2.5-3B-Instruct`
- Attention implementation: `eager`
- dtype: `torch.float16`
- Press: `ObservedAttentionPress`
- Tracking layer: `layer 0`
- Compression ratios: `0.5`, `0.7`, `0.9`
- Token survival definition:
  - 2개 KV head 중 **하나라도 해당 token을 선택하면 생존**으로 간주
- Alert ground truth:
  - `window_314927_metadata.csv` 기준
- 대상 원본 chunk:
  - `window_314927_chunk13`
  - 289 logs
  - 32,727 tokens
  - 54 alerts
- sub4에는 alert가 없으므로 alert retention 통합 계산에서는 sub0~sub3만 사용

---

## 3. Subchunk 구성

| Subchunk | Logs | Tokens | Alerts |
|---|---:|---:|---:|
| sub0 | 65 | 8,123 | 17 |
| sub1 | 74 | 8,135 | 11 |
| sub2 | 73 | 8,153 | 14 |
| sub3 | 75 | 8,174 | 12 |
| sub4 | 2 | 142 | 0 |
| **Total** | **289** | **32,727** | **54** |

---

## 4. Line retention 정의

각 로그 line에 포함된 token 중 layer 0에서 생존한 token 수를 기준으로 다음과 같이 분류했다.

- **KEEP**
  - 해당 line의 모든 token이 생존
- **PARTIAL**
  - 일부 token만 생존
- **DROP**
  - 해당 line의 token이 하나도 생존하지 않음

두 가지 retention 지표를 사용한다.

### Strict retention

```text
Strict retention = KEEP / 전체 alert 수
```

alert line의 모든 token이 살아남은 경우만 성공으로 본다.

### Any-token retention

```text
Any-token retention = (KEEP + PARTIAL) / 전체 alert 수
```

alert line에서 token이 하나라도 살아남으면 성공으로 본다.

> Any-token retention은 매우 permissive한 지표이므로, 이후에는 alert별 token coverage
> (`kept_tokens / total_tokens`)도 함께 확인할 필요가 있다.

---

## 5. Subchunk별 Alert Retention

### sub0

- Logs: 65
- Tokens: 8,123
- Alerts: 17

| Compression ratio | KEEP | PARTIAL | DROP | Strict retention | Any-token retention |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 4 | 13 | 0 | 23.5% | 100.0% |
| 0.7 | 2 | 15 | 0 | 11.8% | 100.0% |
| 0.9 | 0 | 6 | 11 | 0.0% | 35.3% |

---

### sub1

- Logs: 74
- Tokens: 8,135
- Alerts: 11

| Compression ratio | KEEP | PARTIAL | DROP | Strict retention | Any-token retention |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 3 | 8 | 0 | 27.3% | 100.0% |
| 0.7 | 1 | 10 | 0 | 9.1% | 100.0% |
| 0.9 | 1 | 3 | 7 | 9.1% | 36.4% |

---

### sub2

- Logs: 73
- Tokens: 8,153
- Alerts: 14

| Compression ratio | KEEP | PARTIAL | DROP | Strict retention | Any-token retention |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 4 | 10 | 0 | 28.6% | 100.0% |
| 0.7 | 2 | 12 | 0 | 14.3% | 100.0% |
| 0.9 | 2 | 5 | 7 | 14.3% | 50.0% |

---

### sub3

- Logs: 75
- Tokens: 8,174
- Alerts: 12

| Compression ratio | KEEP | PARTIAL | DROP | Strict retention | Any-token retention |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 3 | 9 | 0 | 25.0% | 100.0% |
| 0.7 | 3 | 9 | 0 | 25.0% | 100.0% |
| 0.9 | 1 | 3 | 8 | 8.3% | 33.3% |

---

## 6. Integrated Result — chunk13 Alert 54개

sub0~sub3의 alert retention 결과를 합산하면 다음과 같다.

| Compression ratio | KEEP | PARTIAL | DROP | Strict retention | Any-token retention |
|---:|---:|---:|---:|---:|---:|
| **0.5** | **14** | **40** | **0** | **25.9%** | **100.0%** |
| **0.7** | **8** | **46** | **0** | **14.8%** | **100.0%** |
| **0.9** | **4** | **17** | **33** | **7.4%** | **38.9%** |

### 주요 관찰

- compression ratio `0.5`
  - 54개 alert 모두 최소 1개 이상의 token 생존
  - 완전 DROP된 alert 없음
- compression ratio `0.7`
  - 역시 54개 alert 모두 최소 1개 이상의 token 생존
  - 완전 DROP된 alert 없음
- compression ratio `0.9`
  - 54개 중 33개 alert가 완전히 DROP
  - Any-token retention이 38.9%로 급격히 감소

즉, 본 preliminary 결과에서는 ObservedAttention이 중간 압축률(`0.5`, `0.7`)에서는
alert line을 완전히 제거하지 않는 경향을 보였지만,
극단적 압축(`0.9`)에서는 alert 완전 소실이 크게 증가했다.

---

## 7. SnapKV와의 Preliminary 비교

동일한 subchunk 구성과 line-level retention 정의를 사용한 SnapKV 결과와 비교하면 다음과 같다.

| Compression ratio | Method | KEEP | PARTIAL | DROP | Strict retention | Any-token retention |
|---:|---|---:|---:|---:|---:|---:|
| 0.5 | SnapKV | 19 | 34 | 1 | 35.2% | 98.1% |
| 0.5 | ObservedAttention | 14 | 40 | 0 | 25.9% | 100.0% |
| 0.7 | SnapKV | 8 | 41 | 5 | 14.8% | 90.7% |
| 0.7 | ObservedAttention | 8 | 46 | 0 | 14.8% | 100.0% |
| 0.9 | SnapKV | 2 | 31 | 21 | 3.7% | 61.1% |
| 0.9 | ObservedAttention | 4 | 17 | 33 | 7.4% | 38.9% |

### 해석

- `0.5`
  - SnapKV가 alert 전체 token을 완전히 유지한 비율(Strict retention)은 더 높음
  - ObservedAttention은 완전 DROP된 alert가 없음
- `0.7`
  - Strict retention은 동일
  - ObservedAttention은 54개 alert 모두 최소 1개 token을 보존
- `0.9`
  - ObservedAttention은 Strict retention은 약간 높지만
  - 완전 DROP이 33개로 증가하여 Any-token retention은 SnapKV보다 낮음

따라서 현재 결과만 보면 ObservedAttention은 `0.5~0.7` 구간에서
**alert의 최소 생존**에는 강한 모습을 보이지만,
`0.9`의 극단적 압축에서는 alert 소실이 급격하게 증가한다.

---

## 8. 매우 중요한 해석 제한

이 결과를 **32K context에서 ObservedAttention을 한 번 실행한 결과로 해석하면 안 된다.**

실제 절차는 다음과 같다.

```text
32K chunk13
    ↓
약 8K token 단위로 line-preserving split
    ↓
sub0, sub1, sub2, sub3, sub4
    ↓
각 subchunk에서 ObservedAttention을 독립적으로 실행
    ↓
line-level alert retention 결과를 사후 통합
```

따라서 본 결과는 다음과 같이 표현해야 한다.

> **32K chunk13을 약 8K subchunk로 분할한 뒤, 각 subchunk에서 독립적으로
> ObservedAttention을 수행한 preliminary alert preservation 결과**

32K 전체 context에서 한 번에 압축한 결과와 달라질 수 있는 이유는 다음과 같다.

1. 각 실행에서 후보 token pool이 전체 32K가 아니라 해당 subchunk로 제한된다.
2. subchunk 밖 context는 attention score 계산에 사용되지 않는다.
3. compression budget이 각 subchunk에 독립적으로 배정된다.
4. 각 subchunk 경계에서 attention context가 새로 시작된다.

따라서 SnapKV와 ObservedAttention 간 비교 역시
**동일한 8K subchunk 기반 preliminary 조건 안에서만 비교**해야 한다.

---

## 9. 다음 평가 방향

현재 KEEP / PARTIAL / DROP만으로는 PARTIAL line이 실제로 얼마나 보존됐는지 알기 어렵다.

후속 평가에서는 다음 지표를 추가하는 것이 적절하다.

```text
alert token coverage = kept alert tokens / total alert tokens
```

그리고 alert별 coverage를 기반으로:

- macro-average alert token coverage
- coverage >= 50% 비율
- coverage >= 90% 비율

등을 추가하면 SnapKV와 ObservedAttention의 차이를 더 정밀하게 비교할 수 있다.
