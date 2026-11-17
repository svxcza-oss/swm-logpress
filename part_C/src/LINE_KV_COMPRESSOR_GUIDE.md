# 라인 매핑 기반 KV 압축기

## 구현 범위

`line_kv_compressor.py`는 최근성, 호스트·컴포넌트, 오류 코드·식별자,
Anchor Attention을 합쳐 로그 줄을 선택하고 그 줄의 K/V 텐서를 보존한다.
선택은 라인 단위, 예산은 원본 token_map의 토큰 수 기준이다.
Alert 라벨은 선택에 사용하지 않는다.

현재는 **선택기 + legacy K/V 텐서 압축 함수**다. Transformers `generate()`에
연결하는 자동 어댑터, DynamicCache 직접 수정, 연속 디코딩 및 생성 품질 검증은
구현 범위에 포함되지 않는다. 기존 attention.py는 use_cache=False이므로 그 출력에
실제 KV는 없다. 실제 캐시 압축에는 별도로 확보한 use_cache=True의 K/V가 필요하다.

## 기본 점수

`0.60 × 최근성 + 0.20 × 구조 일치 + 0.05 × 식별자 유사도 + 0.15 × Attention`

- 최근성: 후보의 시간순 위치 / 후보 개수. 실제 초 단위 거리는 아니다.
- 구조 일치: (같은 호스트 여부 + 같은 컴포넌트 여부) / 2.
  Thunderbird의 user 필드를 호스트로 사용하고 컴포넌트 끝 PID를 제거한다.
- 식별자: 명시적 EventID/error code/return code/node 등의 값, 밑줄 포함 이름,
  InfiniHost 장치명을 추출한 집합의 Jaccard 유사도. 소스 파일 위치는 제외한다.
- Attention: [층, 헤드, Query라인, Key라인]에서 층·헤드를 평균한 Anchor 행.
  보호 라인을 제외한 후보의 최솟값/최댓값으로 0~1 정규화한다.
  모든 값이 같으면 Attention 점수는 0이다.

첫 줄 전체와 마지막 Anchor 전체를 보호하고 그 비용을 예산에 포함한다.
보호 비용이 예산보다 크면 오류를 내며 예산을 몰래 늘리지 않는다.
나머지를 점수순으로 담고 안 들어가는 줄은 건너뛴다. 동점은 최신 줄 우선이다.
처음 소수 토큰 보호 정책이나 배낭 최적화는 이번 코드에 넣지 않았다.
기본 가중치는 탐색 설정이며 최적값이 아니다. 모든 층/헤드에서 같은 라인을 남긴다.

## 선택 결과 실행 예시

저장된 파일럿 데이터가 있는 로컬 src에서 실행한다. 표본/Attention 텐서는 이번
커밋에 업로드하지 않았으므로 새 체크아웃에서는 자신의 입력 경로로 바꿔야 한다.

```powershell
python line_kv_compressor.py `
  --token-map examples/thunderbird_pilot_20261114_135108/windows_token_map.jsonl `
  --source-csv examples/thunderbird_pilot_20261114_135108/source.csv `
  --line-attention examples/thunderbird_pilot_20261114_135108/line_attention_793.pt `
  --window-id 793 --keep-ratio 0.5 `
  --output output/selection_793.json
```

가중치를 바꾸려면 `--weights 0.60 0.20 0.05 0.15`를 지정한다.
순서는 최근성, 구조 일치, 식별자, Attention이다. 합은 1이어야 한다.
결과 JSON은 선택 라인, 원본 토큰 인덱스, 보호 라인, 각 특징/점수, 예산을 포함한다.
원본 토큰 비용을 사용하므로 선택한 텍스트를 재토큰화한 길이를 의미하지 않는다.

## 실제 K/V 압축

```python
from line_kv_compressor import compress_kv

# selection: select_lines(...)의 결과
# legacy_cache: ((K_layer0, V_layer0), (K_layer1, V_layer1), ...)
# K와 V shape: [1, KV_heads, original_tokens, head_dim]
compressed = compress_kv(legacy_cache, selection)
original_positions = selection.kept_token_indices
next_position = selection.total_tokens
```

CLI에서도 `--kv-cache original_kv.pt --kv-output compressed_kv.pt`를 추가하면
실제 압축 텐서와 원본 위치를 저장한다. 신뢰하는 .pt 파일만 사용한다.
배치 1, 원본 프리필 길이와 매핑 길이가 같은 legacy 캐시만 지원한다.
압축은 입력을 수정하지 않고 새 텐서를 반환하므로 입력 캐시를 계속 참조하면
메모리가 즉시 줄지 않는다. 복사 중에는 일시적으로 양쪽 캐시 메모리가 필요하다.

RoPE가 적용된 K를 다시 회전하거나 선택 인덱스로 위치를 새로 매기지 않는다.
생성 연결 시 모델별 position_ids, cache_position, causal mask, 캐시 길이 관리가
필요하다. next_position은 원래 프리필 길이의 메타데이터이며 Transformers에
그대로 전달해도 된다는 의미가 아니다. 반환 텐서를 generate()에 바로 넣지 않는다.

## 검증 결과 (2026-11-17)

PyTorch/pytest 환경에서 `python -m pytest test/test_line_kv_compressor.py -q`:
**11 passed**. 토큰/라인 선택, 보호 비용 초과, 누락/비정상 Attention,
라벨 불변성, 실제 K/V 값과 순서, 원본 캐시 미변경을 확인했다.
모델의 압축 후 생성 품질 테스트는 아니다.

저장된 16개 윈도우에 기본 가중치로 3개 예산을 적용했다. 이전 Alert가 있는
7개 윈도우의 macro recall이며, 9개는 정답이 없어 평균에서 제외했다.

| 방법 | 25% 보존 예산 | 50% 보존 예산 | 75% 보존 예산 |
|---|---:|---:|---:|
| 최근성 | 11.9% | 66.7% | 92.9% |
| 최근성 + 호스트·컴포넌트 | 11.9% | 81.0% | 92.9% |
| 최근성 + 구조 + 식별자 | 11.9% | 66.7% | 92.9% |
| 이번 압축기: 위 정보 + Attention | 11.9% | 66.7% | 100.0% |

이번 압축기의 보존 Alert 출현 수는 각각 2/12, 8/12, 12/12이다.
겹치는 윈도우의 반복 Alert 출현을 포함하며 독립 사건 12개가 아니다.
75%에서는 좋아졌지만 50%에서는 호스트·컴포넌트 조건보다 낮다.
가중치 구성이 서로 다르므로 추가 정보만의 인과적 효과라고 해석하지 않는다.
이번 실행은 저장된 Attention을 사용하는 선택 검증이며 모델 재실행/실제 표본의
KV 압축/압축 후 RAM 절감/생성 품질을 측정한 결과가 아니다.

## 함께 업로드한 자료

- `run_structural_retention.py`: 직전 Attention 없는 비교 실험 재현 코드.
- `output/pdf/LogPress_구조정보_라인보존_실험결과_20261117.pdf`: 직전 4가지 비교 실험.
- `output/pdf/LogPress_진행현황_종합정리_20261117.pdf`: 작성 당시의 종합 현황.
- `output/pdf/LogPress_파일럿_결과분석_개선계획.pdf`: 초기 파일럿 분석.

이전 두 PDF는 당시 기록을 그대로 보존했다. 거기에 '미실행'으로 적힌 일부
평가는 이후 진행되었으므로 현재 구현 상태는 이 가이드와 코드를 기준으로 한다.
구조정보 PDF에는 이번 Attention 추가 압축기 결과가 포함되어 있지 않다.
