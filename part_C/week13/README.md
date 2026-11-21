# Week 13 — Part C × Part D 위치 기반 압축 실험

**한 줄 요약:** Part D에서 검증한 로그 **위치 정보**를 Part C의 **라인·토큰 선택기**에 연결해, Thunderbird 32K Chunk에서 어떤 Alert가 남는지 확인했다.

## 1. 왜 이 실험을 했나?

기존 Part C의 `line_kv_compressor.py`는 **Anchor(기준 로그)**를 필요로 하고, 최근성(60%)·구조(20%)·식별자(5%)·Attention(15%)을 합쳐 로그를 선택했다. 다만 Anchor를 입력의 **마지막 로그**로 가정하고, 가중치도 초기 실험용 값이라 32K Chunk를 바로 평가하기엔 제약이 있었다. 또 기존 코드는 선택한 K/V를 추출하는 함수는 있지만 실제 LLM 생성까지 연결된 완성형 압축기는 아니었다.

그래서 이번에는 **Anchor 없이 작동하는 위치 단독 방식부터 연결**했다. 기존 압축기 파일은 삭제하거나 변경하지 않았다.

## 2. Part D 수식을 어떻게 합쳤나?

Part D는 같은 **노드 + 정각 기준 1시간 윈도우** 안에서 로그의 위치를 아래처럼 정의했다.

```text
P = rank / (N - 1)
rank: 해당 노드·시간창에서의 순번 (0부터)
N: 해당 노드·시간창의 로그 수
```

Part D의 AUC 실험에서 **앞쪽에 이상 로그가 상대적으로 많이 나타나는 경향**이 관측됐다. 이를 바탕으로 이번 실험에서는 **FRONT 점수 = 1 - P**를 시도했고, 반대 방향의 **BACK 점수 = P**도 비교했다. **`1-P`는 우리가 제안한 압축 선택 정책이며 Part D가 최종 압축 공식으로 확정한 값은 아니다.**

흐름은 간단하다.

```text
Part A의 1시간 Thunderbird 로그 (전체 6,022줄)
  → Part D 공식으로 노드별 위치 P 계산
  → Chunk13의 각 로그에 위치값 연결
  → FRONT(1-P) 또는 BACK(P) 기준으로 로그 줄 선택
  → 선택한 줄의 토큰 인덱스 추출 (기존 Part C Selection 재사용)
  → 저장해 둔 Alert 라벨과 비교해 보존율 계산
```

> **AUC와 Alert 보존율은 다르다.** AUC는 위치 신호의 이상/정상 구별력이고, 여기서 측정한 보존율은 실제 선택 후 남은 Alert의 비율이다. **Alert 정답 라벨은 로그 선택에 사용하지 않고 평가할 때만 사용했다.**

## 3. 실제 실험 결과 (Thunderbird Window 314927, Chunk13)

- 원본 1시간 윈도우: **6,022줄 / Alert 121개**
- 실험 Chunk13: **289줄 / 32,727토큰 / Alert 54개**
- 설정: **토큰 보존 예산 50%**, 전체 로그 줄 단위 선택

| 항목 | FRONT (1-P, 앞쪽 우선) | BACK (P, 뒤쪽 우선) |
|---|---:|---:|
| 보존 Alert | **38 / 54** | **16 / 54** |
| Alert 보존율 | **70.37%** | **29.63%** |
| 손실 Alert | 16개 | 38개 |
| 선택된 로그 줄 | 120줄 | 169줄 |
| 보존 토큰 | 16,359개 | 50% 예산 내 선택 (정확한 수 미기록) |

FRONT는 원본 32,727토큰에서 16,359토큰을 남겨 **약 50.01% 토큰 감소**를 기록했다. 같은 토큰 예산을 사용한 이 Chunk에서는 FRONT가 Alert를 **22개 더 보존(+40.74%p)**했다.

이 결과는 **Chunk13 한 개**에서의 실험이며, 아직 전체 데이터에서 더 좋다는 결론이나 **RCA 원인·증거 보존율**을 뜻하지 않는다.

## 4. 팀원이 다시 실행하는 방법

저장소 루트(`swm-logpress`)에서 실행. 기존 `Part_A/week6/thunderbird/chunks/` 데이터가 필요하다.

```bash
# 맥북 등에서 가상환경 활성화 (이미 있으면 기존 환경 사용)
source .venv/bin/activate

# 필요한 패키지가 없다면 설치
python -m pip install "transformers==4.57.6" torch scikit-learn pytest

# 기존 Part C 테스트 11개 + 위치 통합 테스트 7개
python -m pytest part_C/src/test/test_line_kv_compressor.py \
  part_C/src/test/test_line_kv_compressor_position.py -q

# Week13 실험 재실행 (FRONT와 BACK 결과를 함께 계산)
python part_C/week13/run_position_experiment.py
```

실행 결과는 `part_C/week13/output/`에 저장된다. 생성된 CSV/JSON/토큰 매핑과 원본 대용량 데이터는 Git에 올리지 않는다. 이번 실험은 사용자의 맥북에서 **18개 테스트 통과** 및 위 보존율 결과를 확인했다.

## 5. 관련 코드와 남은 과제

| 파일 | 내용 |
|---|---|
| `part_C/week13/run_position_experiment.py` | 실험 재현: 데이터 준비 → 위치 기반 선택 → Alert 보존율 비교 |
| `part_C/src/prepare_week6_thunderbird.py` | Part A Chunk TXT/metadata를 위치 CSV·토큰 맵으로 변환 |
| `part_C/src/line_kv_compressor_position.py` | Part D의 `P` 계산, FRONT/BACK 로그 선택, AUC 보조 평가 함수 |
| `part_C/src/line_kv_compressor.py` | 기존 Part C의 `Selection`, `compress_kv()` 재사용 |
| `part_C/src/test/test_line_kv_compressor_position.py` | 신규 단위 테스트 7개 |
| `part_D(select,review)/week3_m1_go/~26.8.5(판정결과)/matching.ipynb` | Part D의 AUC 위치 실험 원본 |

**다음 작업:** (1) 실제 Qwen 모델에서 KV를 생성하고 선택 인덱스를 적용해 압축 검증, (2) 더 많은 Chunk·윈도우와 Baseline 비교, (3) Attention/희귀도/시간 신호 결합. 현재는 **로그·토큰 선택 및 기존 legacy K/V 추출 함수의 단위 테스트**까지 확인한 상태이며, 모델의 실제 캐시 압축과 연속 생성은 완료되지 않았다.
