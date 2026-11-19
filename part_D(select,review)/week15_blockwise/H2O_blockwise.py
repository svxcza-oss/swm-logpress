# ============================================================
#  H2O  (블록와이즈 버전)  — 기존 H2O_eval.py 대체
#  - 기존: attentions(N×N) 전체를 H2OKVCache_LayerWise 에 넘겨 heavy-hitter 계산 → 터짐
#  - 변경: sdpa 로 돌리고, hh_score(= 각 key가 받은 어텐션 합)를 '블록와이즈'로 계산
#  - H2O 선택 규칙(heavy-hitter + 최근 윈도우)은 원본 H2O.py 와 동일하게 재현
#  - 수학적으로 eager와 동일 (verify_blockwise.py 로 검증됨)
#
#  ※ 이 버전은 H2O.py(H2OKVCache_LayerWise)를 import 하지 않음 — 자체 완결.
# ============================================================

# ============================================================
# 1. Imports
# ============================================================
from Part_A.week11.common.line_mapping import build_token_line_mapping
from Part_A.week11.common.gt_loader import load_thunderbird_gt
from transformers import AutoTokenizer, AutoModelForCausalLM
from Part_A.week11.common.evaluator import evaluate_line_retention
from pathlib import Path
from kvpress.presses.scorer_press import ScorerPress
import torch

# 블록와이즈용 추가 import
import math
from transformers.models.llama.modeling_llama import repeat_kv
from kvpress.utils import get_query_states


# ============================================================
# 2. H2O Token Survival Tracking (블록와이즈)
# ============================================================

class TrackingH2OPress(ScorerPress):

    block_size = 1024   # 메모리 조절 손잡이

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        batch_size, num_kv_heads, k_len, head_dim = keys.shape
        assert batch_size == 1

        if self.compression_ratio == 0:
            indices = torch.arange(k_len, device=keys.device)
            indices = indices.view(1, 1, k_len).expand(batch_size, num_kv_heads, k_len)
            self.layers[module.layer_idx] = indices.detach().cpu()
            return keys, values

        n_kept = int(k_len * (1 - self.compression_ratio))
        if n_kept < 2:
            raise ValueError("H2O 보존 토큰 예산이 너무 작습니다.")

        hh_size = n_kept // 2
        recent_size = n_kept - hh_size

        # --- hh_score 를 블록와이즈로 계산 (= 원본 H2O 의 _update_hh_score 결과와 동일) ---
        hh = self._hh_score_blockwise(module, hidden_states, keys, kwargs)   # [num_kv_heads, k_len]

        # --- H2O 선택 규칙 (원본 H2O.py __call__ 과 동일) ---
        select_hh = hh[:, : k_len - recent_size]
        _, keep_topk = torch.topk(select_hh, hh_size, dim=-1)
        keep_topk = keep_topk.sort().values                                 # [KVH, hh_size]
        keep_recent = torch.arange(
            k_len - recent_size, k_len, device=hh.device
        ).repeat(num_kv_heads, 1)                                           # [KVH, recent_size]
        keep_idx = torch.cat([keep_topk, keep_recent], dim=-1)             # [KVH, n_kept]

        # 생존 인덱스 기록: [1, KVH, n_kept]
        self.layers[module.layer_idx] = keep_idx.unsqueeze(0).detach().cpu()

        # 실제 KV 압축
        gather_indices = keep_idx.unsqueeze(0).unsqueeze(-1).expand(
            batch_size, num_kv_heads, n_kept, head_dim
        )
        keys = keys.gather(2, gather_indices).contiguous()
        values = values.gather(2, gather_indices).contiguous()
        return keys, values

    # ----- 블록와이즈 hh_score -----
    #  hh_score[kv_head, key] = 모든 query-head, 모든 query 로부터 받은 어텐션 합
    #  (원본 H2O: grouped_attn.sum(dim=(2,3)) 와 동일)
    def _hh_score_blockwise(self, module, hidden_states, keys, kwargs):
        bsz, num_kv_heads, k_len, head_dim = keys.shape
        num_heads = module.config.num_attention_heads
        groups = num_heads // num_kv_heads

        cos, sin = kwargs["position_embeddings"]
        q = get_query_states(module, hidden_states, (cos, sin))   # [B, H, N, d] (RoPE)
        k = repeat_kv(keys, groups)                               # [B, H, k_len, d] (GQA)
        N = q.shape[2]
        scale = 1.0 / math.sqrt(head_dim)

        acc = torch.zeros(bsz, num_heads, k_len, device=keys.device, dtype=torch.float32)
        k_pos = torch.arange(k_len, device=keys.device).view(1, 1, 1, -1)

        for s in range(0, N, self.block_size):
            e = min(s + self.block_size, N)
            qb = q[:, :, s:e, :]
            logits = torch.matmul(qb, k.transpose(2, 3)) * scale
            q_pos = torch.arange(s, e, device=keys.device).view(1, 1, -1, 1)
            logits = logits.masked_fill(k_pos > q_pos, float("-inf"))
            probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
            acc += probs.sum(dim=2)
            del logits, probs

        # query-head 그룹을 합(sum) → [num_kv_heads, k_len]  (H2O는 평균이 아니라 '합')
        hh = acc.view(bsz, num_kv_heads, -1, k_len).sum(2).squeeze(0)
        return hh


# ============================================================
# 3. Model Loading
# ============================================================

model_name = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="sdpa",   # eager → sdpa
    dtype=torch.float16
    )

# ============================================================
# 4. Log / Metadata Loading
# ============================================================

log_path = (
    Path(__file__).resolve().parents[2]
    / "week6"
    / "thunderbird"
    / "subchunks"
    / "window_314927_chunk13_sub0.txt"
)

metadata_path = (
    Path(__file__).resolve().parents[2]
    / "week6"
    / "thunderbird"
    / "chunks"
    / "window_314927_metadata.csv"
)

logs = log_path.read_text(encoding="utf-8")
lines = logs.splitlines(keepends=True)

# ============================================================
# 5. Metadata CSV → Subchunk GT Mapping
# ============================================================

gt_lines = load_thunderbird_gt(log_path, metadata_path, len(lines))

# ============================================================
# 6. Token → Log Line Mapping
# ============================================================

inputs = tokenizer(logs, return_tensors="pt", return_offsets_mapping=True)
offset_mapping = inputs.pop("offset_mapping")

model.to("cuda")
inputs.to("cuda")
model.eval()

token_to_line, line_to_tokens = build_token_line_mapping(lines, offset_mapping)

# ============================================================
# 7. Compression Experiment
# ============================================================

compression = [0.5, 0.7, 0.9]

for compression_ratio in compression:
    press = TrackingH2OPress(compression_ratio)
    press.layers = {}

    with torch.no_grad():
        with press(model):
            outputs = model(**inputs, use_cache=True)

    # H2O Tracking Validation (원본과 동일)
    k_len = inputs["input_ids"].shape[1]
    n_kept = int(k_len * (1 - compression_ratio))
    assert len(press.layers) == model.config.num_hidden_layers
    for layer_idx, indices in press.layers.items():
        assert indices.shape == (1, model.config.num_key_value_heads, n_kept)
        assert torch.all(indices >= 0) and torch.all(indices < k_len)
        for kv_head_idx in range(indices.shape[1]):
            assert torch.unique(indices[0, kv_head_idx]).numel() == n_kept
    print(f"H2O Tracking 검증 성공: {len(press.layers)} / {model.config.num_hidden_layers} layers")

    line_keep_ratio = 1 - compression_ratio
    (line_scores, top_k_lines, retained_gt_lines, retention) = evaluate_line_retention(
        press.layers, line_to_tokens, gt_lines, line_keep_ratio
    )

    # ============================================================
    # 8. Result Output
    # ============================================================
    print("\n" + "=" * 80)
    print(f"압축률: {compression_ratio}")
    print("=" * 80)
    print("\n생존 라인:")
    for line_idx in sorted(top_k_lines):
        print(f"Line {line_idx:02d} | {lines[line_idx].strip()}")
    print()

    if retention is None:
        print("Cause/Evidence Line Retention: N/A (GT line 없음)")
    else:
        print(
            f"Anomaly Line Retention: "
            f"{len(retained_gt_lines)}/{len(gt_lines)} = {retention:.4f}"
        )
