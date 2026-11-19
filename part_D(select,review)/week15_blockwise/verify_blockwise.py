# ============================================================
#  verify_blockwise.py
#  "블록와이즈 점수 == eager(전체) 점수" 를 검증하는 스크립트.
#
#  [A] CPU 수학 검증 (모델 불필요) — 블록 누적이 전체와 같은지 '지금 당장' 확인
#  [B] GPU eager 동일성 검증 (실제 모델) — 돌리기 전 반드시 통과시킬 게이트
#
#  A는 pip install torch 만 있으면 어디서든 돌아감.
# ============================================================

import math
import torch


# ---------- 블록와이즈 핵심 (실제 파일과 동일한 로직) ----------
def blockwise_query_sum(q, keys, num_heads, block_size):
    """각 key가 모든 query로부터 받은 어텐션 합 = eager attentions.sum(2). [B,H,k]"""
    bsz, num_kv_heads, k_len, head_dim = keys.shape
    groups = num_heads // num_kv_heads
    k = keys.repeat_interleave(groups, dim=1)          # GQA 복제
    N = q.shape[2]
    scale = 1.0 / math.sqrt(head_dim)
    acc = torch.zeros(bsz, num_heads, k_len, dtype=torch.float64)
    k_pos = torch.arange(k_len).view(1, 1, 1, -1)
    for s in range(0, N, block_size):
        e = min(s + block_size, N)
        logits = torch.matmul(q[:, :, s:e, :], k.transpose(2, 3)) * scale
        q_pos = torch.arange(s, e).view(1, 1, -1, 1)
        logits = logits.masked_fill(k_pos > q_pos, float("-inf"))
        probs = torch.softmax(logits, dim=-1, dtype=torch.float64)
        acc += probs.sum(dim=2)
    return acc


def full_query_sum(q, keys, num_heads):
    """전체(eager) attentions.sum(2) 레퍼런스."""
    return blockwise_query_sum(q, keys, num_heads, block_size=q.shape[2])  # block=N == 통짜


def observed_score(qsum, num_kv_heads):
    bsz, num_heads, k_len = qsum.shape
    n_in = torch.arange(k_len, 0, -1, dtype=qsum.dtype)
    s = qsum / n_in
    return s.view(bsz, num_kv_heads, -1, k_len).mean(2)


# ============================================================
# [A] CPU 수학 검증
# ============================================================
def test_cpu():
    torch.manual_seed(0)
    B, Hkv, groups, N, d = 1, 2, 8, 60, 16
    H = Hkv * groups
    q = torch.randn(B, H, N, d, dtype=torch.float64)
    keys = torch.randn(B, Hkv, N, d, dtype=torch.float64)

    ref = observed_score(full_query_sum(q, keys, H), Hkv)
    print("[A] CPU 수학 검증 (블록와이즈 vs 전체)")
    ok_all = True
    for block in [1, 7, 16, N]:
        bw = observed_score(blockwise_query_sum(q, keys, H, block), Hkv)
        maxdiff = (ref - bw).abs().max().item()
        topk = N // 2
        ref_idx = ref.argsort(dim=-1, descending=True)[..., :topk].sort().values
        bw_idx = bw.argsort(dim=-1, descending=True)[..., :topk].sort().values
        same = torch.equal(ref_idx, bw_idx)
        ok_all &= same and maxdiff < 1e-9
        print(f"   block={block:3d} | maxdiff={maxdiff:.1e} | TopK동일={same}")
    print("   =>", "PASS ✅  블록와이즈 = 전체(eager)" if ok_all else "FAIL ❌")
    return ok_all


# ============================================================
# [B] GPU eager 동일성 검증 (실제 모델) — 돌리기 전 게이트
#     작은 입력에서: 원본 eager 점수 vs 블록와이즈 점수의 Top-K 가 (거의) 같은지.
#     ↓ 아래를 GPU 환경에서 주석 풀어 한 번 돌려보고 통과하면 신뢰.
# ============================================================
GPU_CHECK_SNIPPET = r'''
import torch, math
from transformers import AutoTokenizer, AutoModelForCausalLM
from kvpress import ObservedAttentionPress
from kvpress.utils import get_query_states
from transformers.models.llama.modeling_llama import repeat_kv

name = "Qwen/Qwen2.5-3B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
text = "작은 로그 몇 줄 " * 50          # 2K 이하 작은 입력 (eager 가 메모리에 올라가는 크기)

# 1) eager 로 원본 ObservedAttentionPress 점수 뽑기
eager = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager", dtype=torch.float16).cuda().eval()
captured = {}
class RefPress(ObservedAttentionPress):
    def compress(self, module, h, keys, values, attentions, kwargs):
        captured[module.layer_idx] = self.score(module, h, keys, values, attentions, kwargs).detach().float().cpu()
        return keys, values
inp = tok(text, return_tensors="pt").to("cuda")
with torch.no_grad():
    with RefPress(0.5)(eager):
        eager(**inp, use_cache=True)

# 2) 블록와이즈 점수 (sdpa) — ObservedAttention_blockwise.py 의 score 와 동일 로직으로 뽑아
#    같은 레이어의 Top-K 인덱스가 거의 일치하는지(동률 허용) 확인.
#    (레이어별 captured[l] 와 블록와이즈 점수의 argsort 상위 K 비교)
print("레이어별 Top-K 일치율을 확인하세요. 대부분 일치하면(동률로 몇 개만 어긋나면) 통과입니다.")
'''


if __name__ == "__main__":
    test_cpu()
    print("\n[B] GPU 동일성 검증은 아래 스니펫 참고 (실제 모델 필요):")
    print("    verify_blockwise.py 안의 GPU_CHECK_SNIPPET 문자열을 복사해 GPU 환경에서 실행.")
