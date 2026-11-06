# Part B | Nezha 로그 정제 진행 공유 (Week 12)

## 1. 어떤 작업인가요?

LogPress는 **장애 원인을 찾는 데 중요한 로그가 KV 캐시 압축 후에도 남는지** 평가하는 프로젝트입니다.

이를 위해 Part B에서는 Nezha의 **장애 기록**과 **실제 로그**를 연결하고, 원인 분석에 도움이 될 **증거 로그 후보**를 찾았습니다.

- `fault_list.json`: 언제, 어느 서비스에 어떤 장애를 주입했는지 알려주는 기록
- `log/*.csv`: 당시 서비스들이 남긴 실제 로그
- `TraceID`: 여러 서비스의 로그 중 **같은 요청**에 속한 로그를 연결하는 번호

## 2. 어떻게 진행했나요?

1. **데이터 확인:** 2022-08-22의 장애 24건과 로그 CSV 구조를 살펴봤습니다.
2. **장애 전후 비교:** 먼저 `exception` 유형 3건을 골라 장애 발생 전후의 로그를 분석했습니다.
3. **관련 로그 추적:** 오류가 기록된 서비스와 `TraceID`를 확인해 다른 서비스에 나타난 오류도 찾았습니다.
4. **결과 저장:** 후보 로그의 CSV 파일명·행 번호·TraceID를 JSON에 기록하고 3건을 통합했습니다.

### 분석 결과

| 장애 주입 서비스 | 발견한 증거 후보 | 특징 |
|---|---:|---|
| `frontend` | 42줄 | 주문 실패 오류가 직접 기록됨 |
| `checkoutservice` | 27줄 | 결제 연결 실패 오류가 직접 기록됨 |
| `adservice` | 47줄 | 자체 오류는 없지만 `frontend`에서 광고 조회 실패가 나타남 |
| **합계** | **116줄** | 중복 제거 전 |

**특이사항:** `adservice` 후보 중 **동일한 로그 1줄이 서로 다른 CSV에 중복 저장된 것**을 확인했습니다. 현재 파일에는 중복이 남아 있어 116줄이며, 확인된 중복만 빼면 115줄입니다(전체 중복 검사 전).

> **중요:** 여기서 말하는 116줄은 **공식 정답(GT)이 아니라 미검증 증거 후보**입니다. 로그에 오류가 있다고 반드시 장애의 원인이라는 뜻은 아닙니다.

## 3. 각 파일은 무슨 역할인가요?

위치: `Part_B/week12/`

| 파일 | 하는 일 |
|---|---|
| `inspect_faults.py` | 장애 목록과 유형 확인 |
| `inspect_logs.py` | 로그 CSV의 구조와 시간 범위 확인 |
| `inspect_incident.py` | CPU 경합 장애 예시 조사 |
| `inspect_exception.py` | `frontend` 장애 분석 |
| `inspect_error_detail.py` | 오류 메시지 세부 내용 확인 |
| `inspect_trace.py` | 같은 TraceID의 서비스별 로그 추적 |
| `inspect_checkout_exception.py` | `checkoutservice` 장애 분석 |
| `inspect_ad_exception.py` | `adservice` 및 연결된 `frontend` 오류 분석·후보 저장 |
| `build_evidence_candidates.py` | `frontend` 후보 JSON 생성 |
| `build_checkout_candidates.py` | `checkoutservice` 후보 JSON 생성 |
| `build_dataset.py` | 세 후보 JSON 통합 및 중복 점검 |

**결과 파일:** `Part_B/week12/output/`

- `frontend_exception_candidates.json` / `checkout_exception_candidates.json` / `adservice_exception_candidates.json`: 장애별 후보 목록
- **`nezha_exception_dataset.json`**: 위 세 목록을 합친 통합 파일

## 4. 앞으로 할 일

- CSV 간 **중복 로그를 제거**하고, 서비스별로 다른 로그 형식을 **공통 파서**로 처리하기
- 아직 다루지 않은 장애 유형까지 **전체 24건으로 분석 확장**하기
- 후보 로그의 근거를 검증해 **평가용 정답(GT) 기준** 정하기
- CSV의 원본 행 번호를 **LLM 입력 로그의 행 번호**와 연결하기

**현재 단계:** Exception 3건의 후보 추출·통합 완료 → **중복 제거와 공통 전처리 자동화 예정**
