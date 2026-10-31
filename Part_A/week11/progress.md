# Week 11 — Baseline 통합 및 로그 라인 보존율 평가

- **프로젝트:** LogPress
- **담당:** Part A
- **정리일:** 2026-10-31
- **진행 상태:** Baseline 4종을 같은 방식으로 실행하고 평가하는 코드 연결 완료

## 1. 이번 주 작업

- Baseline마다 따로 있던 평가 코드를 `common/` 폴더로 통합했다.
- **ObservedAttention, SnapKV, StreamingLLM, H2O**를 공통 평가 코드에 연결했다.
- Thunderbird 로그 한 파일에 압축률 **0.5 / 0.7 / 0.9**를 적용해 결과를 비교했다.
- 모든 모델 Layer와 KV Head에서 살아남은 토큰 번호를 기록했다.
- H2O는 Qwen의 GQA 구조에 맞게 수정하고, Heavy Hitter + Recent 선택을 확인했다.

## 2. 코드 구성

```text
Part_A/week11/
├── common/
│   ├── line_mapping.py     # 토큰과 로그 라인 연결
│   ├── gt_loader.py        # Thunderbird 이상 로그 정답 불러오기
│   ├── line_score.py       # 라인별 토큰 생존 점수 계산
│   ├── line_selection.py   # 점수가 높은 로그 라인 선택
│   ├── retention.py        # 정답 라인 보존율 계산
│   └── evaluator.py        # 전체 평가 과정 실행
├── ObservedAttention/ObservedAttention.py
├── SnapKV/SnapKV.py
├── StreamingLLM/StreamingLLM.py
├── H2O/
│   ├── H2O.py              # H2O 압축 및 토큰 추적
│   └── H2O_eval.py         # 모델 실행 및 평가
└── progress.md
```

## 3. 공통 평가 방식

1. 각 Layer와 KV Head에서 살아남은 토큰 번호를 저장한다.
2. 로그 라인마다 토큰이 얼마나 남았는지 계산하고 평균을 내 **LineScore**를 구한다.
3. LineScore가 높은 라인을 정해진 개수만큼 선택한다(Top-K).
4. 선택된 라인 중 정답 라인이 몇 개 있는지 계산한다.

**보존율 = 선택된 정답 라인 수 / 전체 정답 라인 수**

이번 테스트에서는 Thunderbird의 **이상(anomaly) 로그 17줄**을 정답으로 사용했다. 최종 RCA 평가에서 사용할 **원인·증거 로그 정답**과는 다르다.

## 4. 테스트 환경

| 항목 | 내용 |
|---|---|
| 모델 | `Qwen/Qwen2.5-3B-Instruct` |
| 모델 구조 | 36 Layers / Attention Heads 16개 / KV Heads 2개 |
| 로그 파일 | `window_314927_chunk13_sub0.txt` |
| 로그 수 | 65줄 |
| 정답 이상 로그 | 17줄 |
| KV 압축률 | 0.5 / 0.7 / 0.9 |
| 선택한 로그 라인 | 32 / 19 / 6줄 |

현재는 테스트 편의상 `line_keep_ratio = 1 - compression_ratio`를 사용했다.

## 5. Baseline 평가 결과

**Anomaly Line Retention** — 동일한 Thunderbird 로그와 정답 17줄 기준

| Baseline | 압축률 0.5 | 압축률 0.7 | 압축률 0.9 |
|---|---:|---:|---:|
| StreamingLLM | 8/17 (47.06%) | 5/17 (29.41%) | 0/17 (0.00%) |
| H2O (Prefill 변형) | 7/17 (41.18%) | 4/17 (23.53%) | 1/17 (5.88%) |
| ObservedAttention | 6/17 (35.29%) | 4/17 (23.53%) | 0/17 (0.00%) |
| SnapKV | 6/17 (35.29%) | 2/17 (11.76%) | 1/17 (5.88%) |

이번 파일에서는 StreamingLLM이 압축률 0.5와 0.7에서 가장 높은 결과를 보였다. **로그 한 파일의 테스트 결과이므로 전체 성능 순위로 볼 수는 없다.**

## 6. 구현 확인 및 주의사항

| Baseline | 확인한 내용 |
|---|---|
| ObservedAttention | 36개 Layer의 생존 토큰 추적 및 공통 평가 성공 |
| SnapKV | 36개 Layer의 생존 토큰 추적 및 공통 평가 성공 |
| StreamingLLM | 36개 Layer 추적, 처음 4개 Sink 토큰과 최근 토큰 보존 확인 |
| H2O | GQA 대응, Heavy Hitter + Recent 선택, KV 압축 및 36개 Layer 인덱스 검증 성공 |

- **H2O:** 현재는 입력 전체를 한 번에 처리하는 *Prefill 변형*이며, 원본 H2O의 생성 과정 중 반복 압축까지 구현한 것은 아니다.
- **평가 방식:** 출력되는 생존 라인은 KV에서 로그 문장 전체가 그대로 남았다는 뜻이 아니라, LineScore로 선택한 라인이다.
- **비교 기준:** 로그마다 토큰 수가 다르므로 KV 압축률과 라인 선택 비율을 똑같이 두는 방식은 최종 평가 전에 검토가 필요하다.
- **결과 범위:** 이번 작업은 한 로그 파일에서의 **코드 실행 및 평가 연결 검증**이다. 최종 RCA 성능 평가는 아니다.

## 7. 실행 명령어

프로젝트 루트(`swm-logpress`)에서 가상환경 활성화 후:

```powershell
python -m Part_A.week11.ObservedAttention.ObservedAttention
python -m Part_A.week11.SnapKV.SnapKV
python -m Part_A.week11.StreamingLLM.StreamingLLM
python -m Part_A.week11.H2O.H2O_eval
```
