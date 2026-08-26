# Week 6 Plan — Thunderbird Parsing + ObservedAttentionPress

## 이번 주 목표

- [x] Thunderbird 원본 로그를 직접 파싱해 모델 입력용 데이터 준비
- [x] 1시간 window 기준으로 로그 묶기
- [x] 모델 최대 입력 길이를 고려해 32K token chunk로 분할
- [x] alert label을 모델 입력에서 제거하고 metadata로 분리
- [x] ObservedAttentionPress 재현 및 압축률별 KV shape 확인

---

## 1. Thunderbird 원본 로그 파싱

### 원본 데이터

- 파일: `Thunderbird.log`
- 크기: 약 31.8GB
- 첫 번째 필드
  - `-` : non-alert
  - `ECC`, `VAPI`, `CPU` 등 : alert label
- 두 번째 필드: Unix timestamp

### Parsing 체크리스트

- [x] 31GB 전체를 메모리에 올리지 않고 line-by-line streaming 처리
- [x] UTF-8 decoding 오류 대응 (`errors="replace"`)
- [x] timestamp 파싱
- [x] `timestamp // 3600` 기준의 고정 1시간 window 생성
- [x] window별 로그 수 / alert 수 확인
- [x] 실험용 window 선정
  - `window_id = 314927`
  - logs: `6022`
  - alerts: `121`
- [x] alert label 제거
  - 모델 입력: label을 제외한 실제 로그 내용
  - 평가 정보: label을 metadata로 별도 보관

---

## 2. 32K Token Chunking

### 이유

Thunderbird 1시간 window를 그대로 모델에 넣기에는 입력이 너무 큼.

`window 314927` 기준:

```text
label 포함 전체 token 수: 487,232
```

따라서 1시간 window 자체는 유지하되, 내부 로그 순서를 보존하면서 최대 32K token 단위로 분할.

### Chunking 규칙

- 최대 token 수: `32768`
- 로그 line 순서 유지
- 한 로그 line을 중간에서 자르지 않음
- 다음 line 추가 시 32K를 넘으면 다음 chunk로 이동
- alert 위치를 기준으로 chunk를 자르지 않음
- label 제거 후의 `clean_line` 기준으로 token 수 계산

### 결과

총 `15 chunks`

```text
chunk0  ~ chunk11 : alert 0
chunk12           : alert 37
chunk13           : alert 54
chunk14           : alert 30
```

alert 검증:

```text
37 + 54 + 30 = 121
```

원본 alert 121개가 누락 없이 각 chunk에 매핑됨.

### 생성 파일

```text
Part_A/week6/thunderbird/chunks/
├── window_314927_chunk0.txt
├── ...
├── window_314927_chunk14.txt
└── window_314927_metadata.csv
```

- `.txt` : baseline / 모델 입력용 로그
- `.csv` : chunk별 line 범위, token 수, alert 수, alert 위치/종류

---

## 3. ObservedAttentionPress 재현

### 환경

- Model: `Qwen/Qwen2.5-3B-Instruct`
- kvpress: `ObservedAttentionPress`
- attention implementation: `eager`
- GPU 실행
- KV cache: `DynamicCache`

ObservedAttention은 attention weight가 필요하므로 모델 로드 시:

```python
attn_implementation="eager"
```

사용.

### 기본 동작 확인

테스트 입력: `11 tokens`

압축 전 KV:

```text
torch.Size([1, 2, 11, 128])
```

압축률별 결과:

| compression_ratio | KV tokens | shape |
|---:|---:|---|
| original | 11 | `[1, 2, 11, 128]` |
| 0.5 | 5 | `[1, 2, 5, 128]` |
| 0.7 | 3 | `[1, 2, 3, 128]` |
| 0.9 | 1 | `[1, 2, 1, 128]` |

### 체크리스트

- [x] `ObservedAttentionPress` import
- [x] press 객체 생성
- [x] Qwen2.5-3B에서 forward 성공
- [x] DynamicCache 생성 확인
- [x] 압축 전 / 후 KV shape 비교
- [x] 0.5 / 0.7 / 0.9 압축률 sweep 확인

---

## Week 6 완료 상태

- [x] Thunderbird raw log parsing
- [x] 1시간 window 구성
- [x] label leakage 방지를 위한 label 분리
- [x] 32K chunk 생성
- [x] chunk metadata CSV 생성
- [x] ObservedAttentionPress baseline 기본 재현

## 다음 단계

지금은 retention 실험을 바로 수행하지 않는다.

M2 baseline 구현이 모두 준비된 뒤 동일한 Thunderbird chunk를 사용해:

```text
StreamingLLM
SnapKV
ObservedAttention
```

에 같은 입력 / 같은 압축률을 적용하고 alert line 보존율을 한 번에 비교한다.
