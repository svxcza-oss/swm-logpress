# ============================================================
# 1. Imports
# ============================================================
from Part_A.week11.common.line_mapping import build_token_line_mapping
from Part_A.week11.common.gt_loader import load_thunderbird_gt
from transformers import AutoTokenizer, AutoModelForCausalLM
from Part_A.week11.common.evaluator import evaluate_line_retention
from pathlib import Path 
from kvpress import StreamingLLMPress
import torch

# ============================================================
# 2. SnapKV Token Survival Tracking
# ============================================================

class TrackingStreamingLLMPress(StreamingLLMPress):
    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        # keys shape:
        # [batch_size, num_kv_heads, seq_len, head_dim]
        
        batch_size, num_kv_heads, k_len, head_dim = keys.shape
        if self.compression_ratio == 0:
            indices = torch.arange(
                k_len,
                device=keys.device
            )
            
            indices = indices.view(
                1, 1, k_len
            ).expand(
                batch_size,
                num_kv_heads,
                k_len
            )
            
            self.layers[module.layer_idx] = (
                indices.detach().cpu()
            )
            
            return keys, values
        
        scores = self.score(
            module,
            hidden_states,
            keys,
            values,
            attentions,
            kwargs
        )
        
        n_kept = int(k_len * (1 - self.compression_ratio))
        
        indices = scores.topk(n_kept, dim=-1).indices
        
        # shape:
        # [batch_size, num_kv_heads, n_kept]
        self.layers[module.layer_idx] = indices.detach().cpu()
        
        # gather용으로 마지막 head_dim 축 확장
        gather_indices = indices.unsqueeze(-1).expand(
            -1, -1, -1, head_dim
        )
        
        keys = keys.gather(2, gather_indices).contiguous()
        values = values.gather(2, gather_indices).contiguous()
        
        return keys, values
        
# ============================================================
# 3. Model Loading
# ============================================================

model_name = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="eager",
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

gt_lines = load_thunderbird_gt(
    log_path,
    metadata_path,
    len(lines)
)

# ============================================================
# 6. Token → Log Line Mapping
# ============================================================

inputs = tokenizer(
    logs,
    return_tensors="pt",
    return_offsets_mapping=True
    )

offset_mapping = inputs.pop("offset_mapping")

model.to("cuda")
inputs.to("cuda")
model.eval()

token_to_line, line_to_tokens = build_token_line_mapping(
    lines, offset_mapping
)

    
# ============================================================
# 7. Compression Experiment
# ============================================================

compression = [0.5, 0.7, 0.9]
        
for compression_ratio in compression:
    press = TrackingStreamingLLMPress(
        compression_ratio,
        n_sink  = 4
        )
    press.layers = {}
    
    with torch.no_grad():
        with press(model):
            outputs = model(**inputs, use_cache=True)

    print(
        f"Tracked layers: {len(press.layers)} "
        f"/ {model.config.num_hidden_layers}"
    )
    assert len(press.layers) == model.config.num_hidden_layers, (
        "일부 layer에서 생존 토큰 추적이 누락되었습니다."
    )
    # ===========================================
    # StreamingLLM Sink / Recent Validation
    # ===========================================

    k_len = inputs["input_ids"].shape[1]
    n_kept = int(k_len * (1 - compression_ratio))

    assert n_kept >= press.n_sink

    expected_kept = (
        set(range(press.n_sink))
        | set(range(k_len - (n_kept - press.n_sink), k_len))
    )

    for layer_indices in press.layers.values():
        for kv_head_idx in range(layer_indices.shape[1]):
            actual_kept = set(
                layer_indices[0, kv_head_idx].tolist()
            )

            assert actual_kept == expected_kept, (
                "StreamingLLM의 Sink/Recent 보존 결과가 다릅니다."
            )

    print("StreamingLLM Sink/Recent 검증 성공")

    # 공통 평가
    line_keep_ratio = 1 - compression_ratio

    (
        line_scores,
        top_k_lines,
        retained_gt_lines,
        retention
    ) = evaluate_line_retention(
        press.layers,
        line_to_tokens,
        gt_lines,
        line_keep_ratio
    )
# ============================================================
# 8. Result Output
# ============================================================

    print("\n" + "=" * 80)
    print(f"압축률: {compression_ratio}")
    print("=" * 80)

    print("\n생존 라인:")

    for line_idx in sorted(top_k_lines):
        print(
            f"Line {line_idx:02d} | "
            f"{lines[line_idx].strip()}"
        )

    print()

    if retention is None:
        print(
            "Cause/Evidence Line Retention: "
            "N/A (GT line 없음)"
        )
    else:
        print(
            f"Anomaly Line Retention: "
            f"{len(retained_gt_lines)}/{len(gt_lines)} "
            f"= {retention:.4f}"
        )