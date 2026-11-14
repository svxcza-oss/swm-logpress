# ============================================================
# 1. Imports
# ============================================================
from Part_A.week11.common.line_mapping import build_token_line_mapping
from Part_A.week11.common.gt_loader import load_thunderbird_gt
from transformers import AutoTokenizer, AutoModelForCausalLM
from Part_A.week11.common.evaluator import evaluate_line_retention
from pathlib import Path 
from kvpress.presses.scorer_press import ScorerPress
from .H2O import H2OKVCache_LayerWise
import torch

# ============================================================
# 2. ObservedAttention Token Survival Tracking
# ============================================================

class TrackingH2OPress(ScorerPress):

    def compress(
        self, module, hidden_states,
        keys, values, attentions, kwargs
    ):
        batch_size, num_kv_heads, k_len, head_dim = keys.shape

        if self.compression_ratio == 0:
            indices = torch.arange(
                k_len, device=keys.device
            ).view(1, 1, k_len).expand(
                batch_size, num_kv_heads, k_len
            )

            self.layers[module.layer_idx] = (
                indices.detach().cpu()
            )

            return keys, values

        assert attentions is not None
        assert batch_size == 1

        n_kept = int(k_len * (1 - self.compression_ratio))

        if n_kept < 2:
            raise ValueError("H2O 보존 토큰 예산이 너무 작습니다.")

        hh_size = n_kept // 2
        recent_size = n_kept - hh_size

        h2o = H2OKVCache_LayerWise(
            hh_size=hh_size,
            recent_size=recent_size
        )

        compressed_keys, compressed_values = h2o(
            (keys, values),
            attentions
        )

        # [KVH, n_kept] → [1, KVH, n_kept]
        self.layers[module.layer_idx] = (
            h2o.kept_indices.unsqueeze(0).detach().cpu()
        )

        return compressed_keys, compressed_values
        
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
    press = TrackingH2OPress(compression_ratio)
    press.layers = {}
    
    with torch.no_grad():
        with press(model):
            outputs = model(**inputs, use_cache=True)

    # H2O Layer / KV Head Tracking Validation

    k_len = inputs["input_ids"].shape[1]
    n_kept = int(k_len * (1 - compression_ratio))

    assert len(press.layers) == model.config.num_hidden_layers

    for layer_idx, indices in press.layers.items():

        assert indices.shape == (
            1,
            model.config.num_key_value_heads,
            n_kept
        )

        assert torch.all(indices >= 0)
        assert torch.all(indices < k_len)

        for kv_head_idx in range(indices.shape[1]):
            selected = indices[0, kv_head_idx]

            assert torch.unique(selected).numel() == n_kept

    print(
        f"H2O Tracking 검증 성공: "
        f"{len(press.layers)} / "
        f"{model.config.num_hidden_layers} layers"
    )

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