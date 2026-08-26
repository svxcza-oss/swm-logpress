# M3 Plan — Part A / Part B

## 전제

M2 Baseline 재현이 끝난 뒤 M3는 **라인→토큰 매핑 및 kvpress PoC 관문**으로 진입한다.

로드맵 기준 M3의 핵심 주담당은 Part C이며, Part A는 기술 백업 경로를 지원한다.  
Part B는 M3 자체의 직접 담당은 아니지만, 병렬 관문인 **Nezha 전처리(M-N)** 를 계속 진행해야 한다.

---

# 0. M2 종료 조건 확인

M3 진입 전 Part A baseline 작업을 먼저 마무리한다.

## Part A — M2

- [ ] baseline 구현 완료
- [ ] 동일 Thunderbird chunk 입력 형식 확정
- [ ] 압축률 sweep 조건 통일
- [ ] baseline별 alert 보존율 계산
- [ ] baseline 보존율 곡선 생성
- [ ] 두 번째 backbone 동작 확인

### 현재 프로젝트 baseline 진행 상태

- [x] StreamingLLM
- [x] SnapKV
- [x] ObservedAttention 기본 재현
- [ ] H2O
- [ ] 동일 chunk 기반 retention curve 통합 실험

> 기존 확정 로드맵의 baseline 목록은 Random / StreamingLLM / H2O / PyramidKV이다.  
> 현재 구현에서는 팀 논의에 따라 StreamingLLM / SnapKV / ObservedAttention 중심으로 재구성 중이며, 최종 baseline 목록은 M2 종료 전에 팀에서 고정한다.

---

# 1. M3 목표

## M3 관문

목표:

```text
로그 line
→ token 범위 매핑
→ "이 line을 버린다"는 mask 생성
→ 해당 token의 KV가 실제 cache에서 삭제되는지 확인
```

M3 Exit:

- token-line mapping 완료
- BPE 분절 예외 처리 확인
- kvpress 압축 경로 이해
- line 단위 mask를 KV 삭제까지 연결
- PoC 수준에서 실제 KV 길이 감소 확인

M3 GO 조건:

```text
로그 신호 기반 mask
→ 목표 token KV 삭제
→ 이후 layer에서도 압축된 KV 사용
```

---

# 2. Part A Plan — M3 기술 지원 / 백업 경로

Part A는 M2 이후 baseline 주담당에서 **M3 기술 지원 역할**로 전환한다.

## A-1. kvpress 내부 경로 조사

- [ ] `BasePress` 구조 확인
- [ ] `compress()` 호출 흐름 확인
- [ ] press가 어느 시점에 KV cache를 수정하는지 확인
- [ ] layer / KV head / token dimension 다시 정리
- [ ] 기존 baseline press들의 token 선택 방식 비교

목표:

```text
외부에서 만든 line mask를
kvpress 압축 로직에 어디서 전달할 수 있는지 파악
```

---

## A-2. M3 백업 구현 경로 조사

C의 PoC가 기존 kvpress API만으로 막힐 가능성에 대비한다.

우선순위:

```text
1. 기존 kvpress 확장
2. custom Press / kvpress fork
3. transformers Cache 직접 제어
```

체크리스트:

- [ ] custom Press 작성 가능 지점 확인
- [ ] 외부 token indices / mask 주입 가능 여부 확인
- [ ] `DynamicCache` 수정 가능 지점 확인
- [ ] transformers Cache subclassing 구조 조사
- [ ] 직접 Hook 방식 필요 여부 판단

### NO-GO 대비

기존 kvpress에서 원하는 line mask를 넣을 수 없다면:

```text
custom press
    ↓ 실패
Cache subclass / 직접 Hook
```

경로로 전환할 수 있도록 최소 PoC를 준비한다.

---

## A-3. C와 인터페이스 맞추기

C가 전달할 결과 형식을 먼저 합의한다.

예:

```text
line_idx
→ token_indices
→ keep/drop mask
```

Part A가 확인할 것:

- [ ] mask shape
- [ ] token index 기준
- [ ] batch dimension
- [ ] KV head dimension
- [ ] layer별 동일 mask 적용 가능 여부
- [ ] 초기 layer 이후부터 압축 적용하는 방법

M3에서는 **판단(mask 계산)은 한 번**, 실제 KV 압축은 설계된 초기 layer 이후 적용하는 구조를 목표로 한다.

---

# 3. Part B Plan — Nezha 병렬 전처리

Part B는 M3 기간에도 M-N 관문을 향해 Nezha 전처리를 계속한다.

목표 마감:

```text
11주차 말
```

최종 목표:

```text
장애 원인 label
→ 실제 로그 line
```

으로 정확하게 매핑 가능한 평가셋 확보.

---

## B-1. Nezha 데이터 구조 파악

- [ ] 원본 데이터 구조 확인
- [ ] 로그 파일 / 라벨 파일 관계 확인
- [ ] 장애 case 구분 방식 확인
- [ ] timestamp 구조 확인
- [ ] 원인 annotation 형식 확인

---

## B-2. 원인 label → line mapping

- [ ] 원인 label이 어떤 단위로 제공되는지 확인
- [ ] 각 원인을 실제 로그 line으로 연결
- [ ] 하나의 장애에 원인 line이 여러 개인 경우 처리 규칙 정의
- [ ] 매핑 실패 / 애매한 case 별도 기록
- [ ] 표본을 직접 확인해 mapping 검증

핵심 산출물 예시:

```text
case_id
log_line
cause_label
cause_line_idx
```

---

## B-3. 데이터 누수 / 정합성 관리

Thunderbird에서 적용한 원칙을 Nezha에도 동일하게 유지한다.

- [ ] 정답 label을 모델 입력에서 제거
- [ ] 정답은 metadata로 별도 관리
- [ ] 원본 line 순서 유지
- [ ] preprocessing 규칙 문서화
- [ ] train/dev/held-out 분리 여부 확인
- [ ] line index가 preprocessing 이후에도 유지되는지 검증

---

## B-4. M-N 관문 준비

M-N GO 조건:

```text
Nezha의 원인 label을
실제 로그 line 단위로 안정적으로 매핑 가능
```

완료 산출물:

- [ ] Nezha 정제 데이터셋
- [ ] 원인 line metadata
- [ ] loader
- [ ] mapping 검증 결과
- [ ] D에게 평가셋 handoff

---

# 4. M3 기간의 Part A / B 병렬 진행 구조

```text
                    M2 Baseline 종료
                           │
             ┌─────────────┴─────────────┐
             │                           │
          Part A                      Part B
             │                           │
    M3 기술 지원 / 백업              Nezha 전처리
             │                           │
    kvpress 내부 조사                 구조 파악
    custom Press 조사                 원인 label 파싱
    Cache fallback 조사               line mapping
    C PoC 지원                        정합성 검증
             │                           │
             ▼                           ▼
         M3 PoC 관문                 M-N 관문
          9주차 말                    11주차 말
```

---

# 5. 최종 체크리스트

## Part A

- [ ] M2 baseline 보존율 곡선 완료
- [ ] kvpress 내부 압축 흐름 이해
- [ ] C의 token-line mask 입력 형식 확정
- [ ] custom Press 가능성 확인
- [ ] Cache / Hook fallback 조사
- [ ] M3 PoC 문제 발생 시 바로 지원 가능한 상태 확보

## Part B

- [ ] Nezha 구조 파악
- [ ] 원인 label 파싱
- [ ] cause label → log line mapping
- [ ] mapping 검증
- [ ] label leakage 방지
- [ ] metadata / loader 정리
- [ ] 11주차 말 M-N handoff 준비

---

## 한 줄 요약

```text
Part A = M2를 닫고 C의 M3 PoC가 막히지 않도록 kvpress/Cache 백업 경로를 준비
Part B = M3와 병렬로 Nezha 원인 label → line 평가셋을 11주차까지 완성
```
