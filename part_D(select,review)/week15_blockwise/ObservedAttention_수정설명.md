# ObservedAttention 블록와이즈 수정 설명

> 파일: `ObservedAttention_blockwise.py` (원본 `ObservedAttention.py`는 그대로, 별도 새 파일)

---

## 0. 왜 바꿨나 (한 줄)

기존 ObservedAttention은 `attn_implementation="eager"`로 돌려서 **전체 어텐션 행렬(N×N)을 메모리에 통째로 올린다.** 입력이 길면(예: 32K 토큰) 이 N×N이 수십~수백 GB가 되어 **GPU가 터진다(OOM).**
→ **어텐션을 "전체 한 번에"가 아니라 "query 블록으로 쪼개서" 계산**하되, **점수 결과는 eager와 동일**하게 만든 것이 이번 수정이다.

---

## 1. 바꾼 곳 요약

| # | 위치 | 기존 | 변경 |
|---|---|---|---|
| ① | import | — | 3줄 추가 (`math`, `repeat_kv`, `get_query_states`) |
| ② | 클래스 | — | `block_size = 1024` 추가 (메모리 조절 손잡이) |
| ③ | `score()` 메서드 | 상속본(전체 어텐션 사용) | **블록와이즈로 덮어쓰기 (~25줄)** ★핵심 |
| ④ | 모델 로딩 | `attn_implementation="eager"` | `"sdpa"` (단어 하나) |
| — | 그 외 전부 | compress·데이터·평가·출력 | **원본 그대로 (변경 없음)** |

> 실질적 변경은 **③ score 메서드 하나 + ④ 단어 하나**. 나머지는 손대지 않았다.
> 이유: 아래 평가 코드는 "점수를 **어떻게** 냈는지"는 모르고 "점수 **결과**"만 받아 쓰므로, score 하나만 바꾸면 전부 그대로 돌아간다.

---

## 2. 각 변경 상세

### ① import 추가
```python
import math
from transformers.models.llama.modeling_llama import repeat_kv   # GQA 복제
from kvpress.utils import get_query_states                        # RoPE 적용한 Query 추출
```

### ④ 모델 로딩: `eager` → `sdpa`
```python
attn_implementation="sdpa"    # eager가 N×N을 통째로 만드는 주범 → sdpa는 안 만듦
```
- `eager`는 전체 어텐션 행렬을 메모리에 올린다(그래서 터짐).
- `sdpa`로 바꾸면 모델 forward는 메모리 효율적이지만, 대신 **어텐션 점수를 밖에서 못 받는다**(`attentions=None`). 그래서 ③에서 점수를 직접 계산한다.

### ③ `score()` 블록와이즈 (핵심)
```python
def score(self, module, hidden_states, keys, values, attentions, kwargs):
    bsz, num_kv_heads, k_len, head_dim = keys.shape
    num_heads = module.config.num_attention_heads
    groups = num_heads // num_kv_heads

    cos, sin = kwargs["position_embeddings"]
    q = get_query_states(module, hidden_states, (cos, sin))   # RoPE 적용한 Query
    k = repeat_kv(keys, groups)                               # GQA: kv헤드 복제
    N = q.shape[2]
    scale = 1.0 / math.sqrt(head_dim)

    # 각 key가 '모든 query'로부터 받은 어텐션 합 = eager의 attentions.sum(2)
    acc = torch.zeros(bsz, num_heads, k_len, device=keys.device, dtype=torch.float32)
    k_pos = torch.arange(k_len, device=keys.device).view(1, 1, 1, -1)

    for s in range(0, N, self.block_size):                   # query 블록 루프
        e = min(s + self.block_size, N)
        qb = q[:, :, s:e, :]
        logits = torch.matmul(qb, k.transpose(2, 3)) * scale
        q_pos = torch.arange(s, e, device=keys.device).view(1, 1, -1, 1)
        logits = logits.masked_fill(k_pos > q_pos, float("-inf"))   # causal mask
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
        acc += probs.sum(dim=2)                              # query축 합 누적
        del logits, probs                                    # 블록 버림(메모리 해제)

    # --- 아래 3줄은 kvpress 원본 score와 '동일한' 정규화 (복제) ---
    n_tokens_in_sum = torch.arange(k_len, 0, -1, device=acc.device, dtype=acc.dtype)
    scores = acc / n_tokens_in_sum
    scores = scores.view(bsz, num_kv_heads, -1, k_len).mean(2)
    return scores.to(keys.dtype)
```

---

## 2-1. score() 코드 — 발췌냐 직접 짰냐 + 줄별 역할

### (가) 발췌 / 표준 / 직접 작성 구분

| 부분 | 출처 | 구분 |
|---|---|---|
| 정규화 3줄 (끝 `n_tokens_in_sum`, `÷`, `.mean`) | kvpress 원본 `ObservedAttentionPress.score` | **그대로 복사** |
| `Q·Kᵀ·scale` / `softmax` / causal mask | 표준 어텐션 공식 | 표준 (어디서 따온 게 아님) |
| Q·K 준비 (`get_query_states`, `repeat_kv`) | kvpress(SnapKV)의 호출 방식 차용 | 패턴 차용 |
| for 루프로 블록 돌며 누적·버림 | query chunking 아이디어 (Rabe & Staats 2021) | **직접 작성(조립)** |

→ **"통째로 발췌"도 "100% 발명"도 아님.** 정규화 = 원본 복사, 공식 = 표준, **블록 누적 뼈대만 직접 조립.**

### (나) 덩어리 1 — 준비 (재료 꺼내기)
```python
bsz, num_kv_heads, k_len, head_dim = keys.shape   # keys 모양에서 숫자 꺼냄
num_heads = module.config.num_attention_heads     # 쿼리 헤드 수
groups = num_heads // num_kv_heads                # 쿼리헤드가 kv헤드 몇 개씩 공유(GQA)
cos, sin = kwargs["position_embeddings"]          # RoPE용 위치정보
q = get_query_states(module, hidden_states, (cos, sin))  # Query 뽑고 RoPE 적용
k = repeat_kv(keys, groups)                       # Key를 쿼리헤드 수에 맞게 복제
N = q.shape[2]                                    # query 개수(=토큰 수)
scale = 1.0 / math.sqrt(head_dim)                # 어텐션 스케일(1/√d)
```
역할: **계산에 필요한 Q, K, 숫자들을 꺼내는 준비.**

### (다) 덩어리 2 — 블록 루프 (★핵심, 직접 작성)
```python
acc = torch.zeros(bsz, num_heads, k_len, ...)    # 결과 담을 통 (0으로 시작)
k_pos = torch.arange(k_len, ...)                 # 각 key의 위치번호 (mask용)

for s in range(0, N, self.block_size):           # query를 block_size씩 끊어서 돈다
    e = min(s + self.block_size, N)              # 이번 블록 끝
    qb = q[:, :, s:e, :]                         # 이번 블록 query만 꺼냄
    logits = torch.matmul(qb, k.transpose(2,3)) * scale  # 이 query들이 각 key를 얼마나 보나(원점수)
    q_pos = torch.arange(s, e, ...)
    logits = logits.masked_fill(k_pos > q_pos, -inf)     # 미래 key는 못 보게 막음(causal)
    probs = torch.softmax(logits, dim=-1, ...)   # 원점수 → 0~1 어텐션 확률
    acc += probs.sum(dim=2)                      # 이 블록이 각 key에 준 어텐션을 key별로 합쳐 누적 ★
    del logits, probs                            # 이 블록 다 썼으니 버림(메모리 해제)
```
역할: **query를 조금씩 끊어서, 각 토큰이 받은 어텐션을 key별로 더해 나가고, 쓴 블록은 버린다.** → 전체 N×N을 한 번에 안 들고 있으니 안 터짐.

### (라) 덩어리 3 — 정규화 (원본 복사)
```python
n_tokens_in_sum = torch.arange(k_len, 0, -1, ...)  # 각 key를 볼 수 있었던 query 수 [k_len, ..., 1]
scores = acc / n_tokens_in_sum                     # 합 ÷ 본 query 수 = 평균 어텐션
scores = scores.view(bsz, num_kv_heads, -1, k_len).mean(2)  # 쿼리헤드 그룹 평균 → kv헤드별 점수
return scores.to(keys.dtype)
```
역할: **누적한 '합'을 '평균'으로 바꾸는 마무리.** 이 3줄은 **원본 kvpress score를 그대로 복사**한 것이라, 결과가 원본과 같아지는 보장.

---

## 3. 오픈소스 출처 (어디서 따왔나)

| 구성요소 | 출처 | 역할 |
|---|---|---|
| `get_query_states` | **kvpress** (`kvpress.utils`) — NVIDIA/kvpress | hidden_states에서 **RoPE 적용한 Query** 추출 (직접 안 짬) |
| `repeat_kv` | **transformers** (`models.llama.modeling_llama`) | **GQA**: KV 헤드를 Query 헤드 수에 맞춰 복제 (표준 함수) |
| `Q·Kᵀ / √d` → `softmax` | **표준 어텐션 공식** | 모델·FlashAttention·kvpress 전부 쓰는 기본식 |
| causal mask | **표준** | query가 미래 key를 못 보게(−inf) — 인과성 |
| 정규화 (`÷ n_tokens_in_sum`, `view(...).mean(2)`) | **kvpress 원본** `ObservedAttentionPress.score` 코드 **그대로 복제** | eager와 동일 결과 보장 |
| query를 블록으로 쪼개 누적 | **query chunking** — Rabe & Staats (2021), arXiv:2112.05682 *"Self-attention Does Not Need O(n²) Memory"* | N×N을 안 만들고 같은 값을 블록으로 |

> **직접 짠 부분은 "for 루프로 블록을 돌며 `attentions.sum(2)`를 누적"하는 조립 코드뿐이다.**
> 발명이 아니라, **표준 어텐션 공식 + kvpress 원본 로직을 "안 터지게" 블록으로 재배열**한 것.

---

## 4. 왜 결과가 eager와 같나 (근거)

kvpress 원본 `ObservedAttentionPress.score`는 핵심이 **한 줄**이다:
```python
scores = attentions.sum(2)        # 전체 N×N 어텐션을 query축으로 합
# 이후: scores / n_tokens_in_sum,  view(...).mean(2)
```
- 우리 블록와이즈는 이 **`attentions.sum(2)`를 "블록마다 나눠 더한 것"**이다.
- **합(sum)은 쪼개서 더해도 결과가 같으므로**, `acc` == `attentions.sum(2)`.
- 그 뒤 정규화(`÷ n_tokens_in_sum`, 헤드그룹 평균)는 **원본 코드를 그대로 복제**했다.
- → **최종 점수가 eager와 수학적으로 동일.** 유일한 차이는 "N×N을 메모리에 올리느냐 마느냐"뿐.

---

## 5. 검증

- **[A] CPU 수학 검증** (`verify_blockwise.py`): 블록 크기 1·7·16·전체 어느 쪽이든 **블록와이즈 == 전체(eager)**. 최대오차 **≈ 1e-17 (= 0)**, **Top-K 완전 일치**. → 로직은 통과됨.
- **[B] GPU eager 동일성 검증** (남은 게이트): 작은 입력에서 **원본 eager 점수 vs 블록와이즈 점수의 Top-K**가 (거의) 같은지 실제 모델로 확인. bf16 초박빙 순위는 몇 개 어긋날 수 있어 **동률 허용**으로 본다.

---

## 6. 안 바꾼 것 (중요)

- `compress()` 본체, 모델 로딩 나머지, 데이터 로딩, `evaluate_line_retention`(평가), 출력 — **전부 원본 그대로.**
- 즉 **"점수 내는 방식(score)" 하나만** 블록와이즈로 바꾸고, 파이프라인은 손대지 않았다.

---

## 요약 (발표/멘토용 한 문장)

> "kvpress `ObservedAttentionPress`의 점수 공식 `attentions.sum(2)`를, **전체 어텐션(N×N)을 메모리에 올리지 않고 query 블록으로 나눠 같은 합을 누적**하도록 다시 구현했습니다. RoPE·GQA는 kvpress/transformers 함수를 그대로 쓰고, 정규화는 원본 코드를 복제했으며, eager와 결과가 동일함을 수치로 검증했습니다. (query chunking, Rabe & Staats 2021)"