# ============================================================
#  ObservedAttention  (블록와이즈 버전)
#  - 기존: attn_implementation="eager" 로 N×N 어텐션을 통째로 만들어 터짐
#  - 변경: sdpa 로 돌리고, score()를 '블록와이즈'로 직접 계산
#  - 수학적으로 eager와 동일 (verify_blockwise.py 로 검증됨)
#
#  바뀐 곳은 딱 3군데:
#   ① import 3줄 추가 (math, repeat_kv, get_query_states)
#   ② 클래스 안에 score() 블록와이즈로 '덮어쓰기' (compress는 그대로)
#   ③ 모델 로딩 attn_implementation="eager" → "sdpa"
#  나머지(평가·출력)는 원본 그대로.
# ============================================================

# ============================================================
# 1. Imports
# ============================================================
from Part_A.week11.common.line_mapping import build_token_line_mapping
from Part_A.week11.common.gt_loader import load_thunderbird_gt
from transformers import AutoTokenizer, AutoModelForCausalLM
from Part_A.week11.common.evaluator import evaluate_line_retention
from pathlib import Path
from kvpress import ObservedAttentionPress
import torch

# ① 블록와이즈용 추가 import ----------------------------------
import math
from transformers.models.llama.modeling_llama import repeat_kv
from kvpress.utils import get_query_states


# ============================================================
# 2. ObservedAttention Token Survival Tracking (블록와이즈)
# ============================================================

class TrackingObservedAttentionPress(ObservedAttentionPress):

    # 블록 크기 = 메모리 조절 손잡이. 작을수록 메모리 적게, 느리게.
    block_size = 1024

    # ----- compress()는 원본 그대로 (self.score 를 부름) -----
    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        # keys shape: [batch_size, num_kv_heads, seq_len, head_dim]
        batch_size, num_kv_heads, k_len, head_dim = keys.shape

        if self.compression_ratio == 0:
            indices = torch.arange(k_len, device=keys.device)
            indices = indices.view(1, 1, k_len).expand(batch_size, num_kv_heads, k_len)
            self.layers[module.layer_idx] = indices.detach().cpu()
            return keys, values

        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)

        n_kept = int(k_len * (1 - self.compression_ratio))
        indices = scores.topk(n_kept, dim=-1).indices
        self.layers[module.layer_idx] = indices.detach().cpu()

        gather_indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        keys = keys.gather(2, gather_indices).contiguous()
        values = values.gather(2, gather_indices).contiguous()
        return keys, values

    # ② score()를 '블록와이즈'로 덮어쓰기 ----------------------
    #    원본 ObservedAttentionPress.score 는 attentions(N×N) 전체를 씀.
    #    여기선 attentions 없이, query를 블록으로 나눠 '각 key가 받은 어텐션 합'을 누적.
    #    최종 정규화(÷ n_tokens_in_sum, 헤드그룹 평균)는 원본과 '똑같이' 맞춤.
    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        bsz, num_kv_heads, k_len, head_dim = keys.shape
        num_heads = module.config.num_attention_heads
        groups = num_heads // num_kv_heads

        cos, sin = kwargs["position_embeddings"]
        q = get_query_states(module, hidden_states, (cos, sin))   # [B, H, N, d] (RoPE 적용됨)
        k = repeat_kv(keys, groups)                               # [B, H, k_len, d] (GQA 복제)
        N = q.shape[2]
        scale = 1.0 / math.sqrt(head_dim)

        # 각 key가 '모든 query'로부터 받은 어텐션 합 = eager의 attentions.sum(2)
        acc = torch.zeros(bsz, num_heads, k_len, device=keys.device, dtype=torch.float32)
        k_pos = torch.arange(k_len, device=keys.device).view(1, 1, 1, -1)

        for s in range(0, N, self.block_size):                   # ← query 블록 루프
            e = min(s + self.block_size, N)
            qb = q[:, :, s:e, :]
            logits = torch.matmul(qb, k.transpose(2, 3)) * scale # [B, H, bq, k_len]
            q_pos = torch.arange(s, e, device=keys.device).view(1, 1, -1, 1)
            logits = logits.masked_fill(k_pos > q_pos, float("-inf"))   # causal mask
            probs = torch.softmax(logits, dim=-1, dtype=torch.float32)  # fp32
            acc += probs.sum(dim=2)                              # query축 합 누적
            del logits, probs

        # --- 원본 ObservedAttentionPress.score 와 '동일한' 정규화 ---
        n_tokens_in_sum = torch.arange(k_len, 0, -1, device=acc.device, dtype=acc.dtype)
        scores = acc / n_tokens_in_sum
        scores = scores.view(bsz, num_kv_heads, -1, k_len).mean(2)   # 헤드그룹 평균
        return scores.to(keys.dtype)


# ============================================================
# 3. Model Loading
# ============================================================

model_name = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="sdpa",   # ③ eager → sdpa (N×N 안 만들어서 안 터짐)
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
    press = TrackingObservedAttentionPress(compression_ratio)
    press.layers = {}

    with torch.no_grad():
        with press(model):
            outputs = model(**inputs, use_cache=True)

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
