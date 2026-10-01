# Week 11 - ObservedAttention 평가 방식 수정 진행상황

## 이번 주에 바꾼 이유

기존 방식은 한 layer만 보고, 여러 KV head에서 살아남은 token을 합쳐서 line 생존 여부를 판단했습니다.

이 방식은
- 어떤 KV head에서만 token이 살아 있어도 line이 살아 있는 것처럼 보일 수 있고
- compression ratio를 바꿔도 line 보존 결과가 거의 비슷하게 나오는 문제가 있었습니다.

그래서 이제는 **모든 layer × KV head를 따로 보고**, 각 log line의 token이 얼마나 살아남았는지를 점수로 계산하도록 바꿨습니다.

---

## 현재 방식

각 log line마다 다음 순서로 계산합니다.

1. line에 속한 token 목록을 구함
2. 각 layer × KV head에서 그 token이 몇 개 살아남았는지 계산
3. `생존 token 수 / line 전체 token 수` 계산
4. 모든 layer × KV head 결과를 평균내서 `LineScore` 생성
5. LineScore가 높은 순서대로 Top-K line 선택

현재 Qwen2.5-3B에서는

- 36 layers
- 2 KV heads
- 총 72개의 layer × KV-head 결과를 평균

해서 LineScore를 계산합니다.

KV head 수는 `2`로 하드코딩하지 않고 실제 tensor에서 자동으로 읽도록 수정했습니다.  
그래서 이후 Qwen2.5-7B로 모델을 바꿔도 같은 코드 구조를 사용할 수 있습니다.

---

## 현재까지 완료한 것

- [x] ObservedAttention에서 layer별 surviving token index 추적
- [x] KV head 수 동적 처리
- [x] token → line 매핑
- [x] line → token 매핑
- [x] 특정 layer 하나만 보는 방식 제거
- [x] KV head union 방식 제거
- [x] 모든 layer × KV head별 line token 생존율 계산
- [x] 전체 평균으로 LineScore 계산
- [x] LineScore 기준으로 line 정렬
- [x] Top-K line 선택

현재 테스트에서는 30개 log line, 3307 tokens를 사용했습니다.

| KV compression ratio | Token-weighted average LineScore |
|---|---:|
| 0.5 | 0.4998 |
| 0.7 | 0.3000 |
| 0.9 | 0.0998 |

예상한 token keep 비율과 거의 동일하게 나와서 현재 LineScore 계산은 정상적으로 동작하는 것을 확인했습니다.

---

## 아직 남은 것

### 1. 최종 보존율 자동 계산

현재 Top-K line까지는 자동으로 뽑힙니다.

다음으로 GT 원인/증거 line을 넣어서

`Top-K에 남은 GT line 수 / 전체 GT line 수`

를 코드에서 바로 계산하고 출력하도록 만들 예정입니다.

즉 예전처럼 결과를 따로 복사해서 보존율을 계산할 필요가 없도록 만드는 것이 목표입니다.

### 2. Line Top-K 비율 결정

현재 `line_keep_ratio = 0.5`는 테스트용입니다.

`KV를 50% 남겼다고 line도 50% 남기는 것이 맞는가?`

이 부분은 아직 최종 결정되지 않았습니다.

### 3. 다른 baseline에도 동일 방식 적용

ObservedAttention이 끝나면 같은 평가 방식을

- SnapKV
- StreamingLLM
- H2O

순서로 적용할 예정입니다.

---

## 현재 진행도

ObservedAttention 기준으로는 **최종 보존율 자동 출력까지 약 90% 완료**된 상태입니다.

남은 핵심은

**GT line 연결 → 최종 retention 숫자 자동 출력**

입니다.
