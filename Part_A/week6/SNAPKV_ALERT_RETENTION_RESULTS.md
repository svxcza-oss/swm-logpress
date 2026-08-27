# SnapKV Alert Preservation Results — Thunderbird window 314927 / chunk13

## 1. 실험 목적

Thunderbird `window 314927`의 `chunk13`에 포함된 alert 로그가 SnapKV 압축 이후 얼마나 보존되는지 확인한다.

- Model: `Qwen/Qwen2.5-3B-Instruct`
- Baseline: `SnapKVPress`
- 평가 레이어: `layer 0`
- SnapKV 설정: `window_size=64`, `kernel_size=5`
- compression ratio: `0.5`, `0.7`, `0.9`
- 원본 `chunk13`
  - logs: `289`
  - tokens: `32,727`
  - alerts: `54`
  - alert label: `VAPI`

32K 입력을 한 번에 실행하면 GPU OOM이 발생했기 때문에, `chunk13`을 로그 line 순서를 유지한 채 최대 약 8K token 단위의 subchunk로 다시 나누어 실험했다.

---

## 2. Subchunk 구성

| subchunk | logs | tokens | alerts | chunk13 내부 line 범위 |
|---|---:|---:|---:|---:|
| sub0 | 65 | 8,123 | 17 | 0–64 |
| sub1 | 74 | 8,135 | 11 | 65–138 |
| sub2 | 73 | 8,153 | 14 | 139–211 |
| sub3 | 75 | 8,174 | 12 | 212–286 |
| sub4 | 2 | 142 | 0 | 287–288 |
| **합계** | **289** | **32,727** | **54** | **0–288** |

`sub4`에는 alert가 없으므로 alert preservation 계산에서는 제외했다.

---

## 3. 보존 판정 기준

각 alert log의 token 중 SnapKV 이후 살아남은 token 수에 따라 다음과 같이 분류했다.

- `KEEP`: 해당 로그의 모든 token이 생존
- `PARTIAL`: 일부 token만 생존
- `DROP`: 해당 로그의 token이 하나도 생존하지 않음

보존율은 두 가지 기준으로 함께 기록한다.

### Strict retention

로그 전체가 온전히 보존된 경우만 성공으로 계산한다.

```text
Strict retention = KEEP / 전체 alert 수
```

### Any-token retention

로그에서 token 하나라도 살아남으면 보존된 것으로 계산한다.

```text
Any-token retention = (KEEP + PARTIAL) / 전체 alert 수
```

현재 token 생존 판정은 `layer 0`의 KV head 선택 결과를 합쳐, 두 KV head 중 하나라도 해당 token을 선택하면 생존으로 간주한다.

---

## 4. Subchunk별 SnapKV alert 보존 결과

### sub0

alert 수: `17`

| compression ratio | KEEP | PARTIAL | DROP | Strict retention | Any-token retention |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 4 | 12 | 1 | 4/17 = **23.5%** | 16/17 = **94.1%** |
| 0.7 | 1 | 13 | 3 | 1/17 = **5.9%** | 14/17 = **82.4%** |
| 0.9 | 0 | 8 | 9 | 0/17 = **0.0%** | 8/17 = **47.1%** |

### sub1

alert 수: `11`

| compression ratio | KEEP | PARTIAL | DROP | Strict retention | Any-token retention |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 5 | 6 | 0 | 5/11 = **45.5%** | 11/11 = **100.0%** |
| 0.7 | 2 | 8 | 1 | 2/11 = **18.2%** | 10/11 = **90.9%** |
| 0.9 | 1 | 7 | 3 | 1/11 = **9.1%** | 8/11 = **72.7%** |

### sub2

alert 수: `14`

| compression ratio | KEEP | PARTIAL | DROP | Strict retention | Any-token retention |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 6 | 8 | 0 | 6/14 = **42.9%** | 14/14 = **100.0%** |
| 0.7 | 2 | 12 | 0 | 2/14 = **14.3%** | 14/14 = **100.0%** |
| 0.9 | 1 | 8 | 5 | 1/14 = **7.1%** | 9/14 = **64.3%** |

### sub3

alert 수: `12`

| compression ratio | KEEP | PARTIAL | DROP | Strict retention | Any-token retention |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 4 | 8 | 0 | 4/12 = **33.3%** | 12/12 = **100.0%** |
| 0.7 | 3 | 8 | 1 | 3/12 = **25.0%** | 11/12 = **91.7%** |
| 0.9 | 0 | 8 | 4 | 0/12 = **0.0%** | 8/12 = **66.7%** |

---

## 5. Subchunk 통합 결과

`sub0`~`sub3`의 alert 결과를 합산하면 `chunk13`의 전체 alert `54개`에 대한 결과는 다음과 같다.

| compression ratio | KEEP | PARTIAL | DROP | Strict retention | Any-token retention |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 19 | 34 | 1 | 19/54 = **35.2%** | 53/54 = **98.1%** |
| 0.7 | 8 | 41 | 5 | 8/54 = **14.8%** | 49/54 = **90.7%** |
| 0.9 | 2 | 31 | 21 | 2/54 = **3.7%** | 33/54 = **61.1%** |

### 요약

- compression ratio `0.5`
  - alert 54개 중 53개가 최소 1 token 이상 생존
  - Any-token retention: **98.1%**
- compression ratio `0.7`
  - alert 54개 중 49개가 최소 1 token 이상 생존
  - Any-token retention: **90.7%**
- compression ratio `0.9`
  - alert 54개 중 33개가 최소 1 token 이상 생존
  - Any-token retention: **61.1%**

압축률이 높아질수록 `KEEP`은 감소하고 `DROP`은 증가했다.

---

## 6. 중요 주의사항 — 32K 전체 실행 결과와 동일하지 않음

이 결과를 **32,727 token 전체 chunk에 SnapKV를 한 번 적용한 결과로 해석하면 안 된다.**

실제 수행 방식은 다음과 같다.

```text
chunk13: 32,727 tokens
        ↓ GPU OOM
sub0 ~ sub3: 각각 약 8K tokens
        ↓
각 subchunk에 SnapKV를 독립적으로 실행
        ↓
alert preservation 결과를 마지막에 합산
```

따라서 현재 결과는 정확히 말하면:

> **32K chunk13을 약 8K subchunk로 분할한 뒤, 각 subchunk에서 독립적으로 SnapKV를 수행한 preliminary alert preservation 결과**

이다.

### 왜 32K를 한 번에 실행한 것과 결과가 달라질 수 있는가?

SnapKV는 입력 안에서 token 중요도를 계산하고 남길 token을 선택한다. Subchunk로 나누면 각 구간에서 selection이 독립적으로 다시 수행되므로 다음 요소가 달라진다.

1. **비교 대상 token 집합이 달라짐**
   - 32K 전체 실행에서는 전체 32K token 사이에서 선택이 이뤄진다.
   - 8K 분할 실행에서는 각 8K 내부 token 사이에서만 선택이 이뤄진다.

2. **문맥 범위가 달라짐**
   - subchunk 밖의 이전/이후 로그는 현재 subchunk의 SnapKV score 계산에 참여하지 않는다.

3. **압축 budget이 subchunk마다 별도로 할당됨**
   - 예를 들어 compression ratio `0.5`라면 각 subchunk에서 각각 약 절반을 남긴다.
   - 이는 32K 전체에서 한 번에 절반을 선택하는 것과 선택 결과가 같다고 보장할 수 없다.

4. **SnapKV의 local window 기반 score도 subchunk 경계에서 다시 시작함**
   - 현재 설정의 `window_size=64`, `kernel_size=5` 계산 역시 subchunk 단위로 독립 수행된다.

따라서 현재 수치는 baseline 경향과 alert preservation을 빠르게 확인하기 위한 **preliminary result**로 사용하고, 추후 더 큰 GPU 또는 32K 실행이 가능한 환경이 확보되면 full-chunk 결과와 별도로 비교해야 한다.

---

## 7. 현재 결과 해석 범위

현재 결과에서 확정적으로 말할 수 있는 것은 다음이다.

- 약 8K subchunk 단위 SnapKV 실행은 정상적으로 완료됨
- compression ratio가 증가할수록 alert의 완전 보존(`KEEP`)은 감소함
- compression ratio가 증가할수록 alert의 완전 제거(`DROP`)는 증가함
- `chunk13`의 alert 54개에 대해 subchunk 결과를 합산한 Any-token retention은:
  - `0.5` → **98.1%**
  - `0.7` → **90.7%**
  - `0.9` → **61.1%**

단, 이 값은 **32K full-chunk SnapKV 결과가 아니라 8K subchunk 독립 실행 결과**라는 제한을 항상 함께 명시한다.
