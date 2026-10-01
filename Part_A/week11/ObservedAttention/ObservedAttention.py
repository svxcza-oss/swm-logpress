from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path 
from kvpress import ObservedAttentionPress
import torch

class TrackingObservedAttentionPress(ObservedAttentionPress):
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
        

model_name = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="eager",
    dtype=torch.float16
    )

log_path = (
    Path(__file__).resolve().parents[2]
    / "week6"
    / "thunderbird"
    / "subchunks"
    / "window_314927_chunk13_sub3.txt"
)
logs = log_path.read_text(encoding="utf-8")
all_lines = logs.splitlines(keepends=True)
cursor = 0

# 디버깅용: 앞 30줄만 사용
lines = all_lines[:30]

logs = "".join(lines)

inputs = tokenizer(
    logs,
    return_tensors="pt",
    return_offsets_mapping=True
    )
offset_mapping = inputs.pop("offset_mapping")

print("입력 토큰 수:", inputs["input_ids"].shape[1])

model.to("cuda")
inputs.to("cuda")
model.eval()

line_spans = []

for line_no, line in enumerate(lines):
    line_start = cursor
    cursor += len(line)
    line_end = cursor
    
    line_spans.append([line_start, line_end])

token_to_line = []
line_chk = 0


for token_no, token in enumerate(offset_mapping[0]):
    token_start = token[0].item()
    token_end = token[1].item()
    
    while(token_start >= line_spans[line_chk][1]):
        line_chk += 1
        
    token_to_line.append(line_chk)
    

line_to_tokens = [[] for _ in range(len(lines))]

for token_idx, line_idx in enumerate(token_to_line):
    line_to_tokens[line_idx].append(token_idx)
    

compression = [0.5, 0.7, 0.9]

for compression_ratio in compression:
    print("\n" + "=" * 80)
    print(f"Compression Ratio: {compression_ratio}")
    print("=" * 80)
    
    press = TrackingObservedAttentionPress(compression_ratio)
    press.layers = {}
    
    with torch.no_grad():
        with press(model):
            outputs = model(**inputs, use_cache=True)
            
    line_scores = [0.0] * len(lines)
    slot_count = 0

    # 모든 layer 순회
    for layer_idx, layer_indices in press.layers.items():

        # layer_indices shape:
        # [batch_size, num_kv_heads, n_kept]
        num_kv_heads = layer_indices.shape[1]

        # 현재 layer의 모든 KV head 순회
        for kv_head_idx in range(num_kv_heads):

            kept_tokens = set(
                layer_indices[0, kv_head_idx].tolist()
            )

            # 모든 line에 대해 survival ratio 계산
            for line_idx, token_indices in enumerate(line_to_tokens):

                if len(token_indices) == 0:
                    continue

                kept_count = sum(
                    token_idx in kept_tokens
                    for token_idx in token_indices
                )

                survival_ratio = (
                    kept_count / len(token_indices)
                )

                line_scores[line_idx] += survival_ratio

            # layer × kv_head 한 slot 처리 완료
            slot_count += 1


    # 모든 layer × KV head를 다 돈 뒤 검사
    if slot_count == 0:
        raise RuntimeError(
            "압축된 layer x KV head slot이 없습니다."
        )


    # 지금까지 누적한 survival ratio를
    # 전체 slot 개수로 딱 한 번 나눔
    for line_idx, token_indices in enumerate(line_to_tokens):

        if len(token_indices) == 0:
            continue

        line_scores[line_idx] /= slot_count

    total_mapped_tokens = sum(
        len(token_indices)
        for token_indices in line_to_tokens
    )

    weighted_line_score = sum(
        line_scores[line_idx] * len(token_indices)
        for line_idx, token_indices in enumerate(line_to_tokens)
        if len(token_indices) > 0
    ) / total_mapped_tokens

    print(
        f"Token-weighted average LineScore: "
        f"{weighted_line_score:.4f}"
    )

    line_keep_ratio = 0.5

    # token이 실제로 존재하는 line만 ranking 대상으로 사용
    valid_line_indices = [
        line_idx
        for line_idx, token_indices in enumerate(line_to_tokens)
        if len(token_indices) > 0
    ]

    num_lines_to_keep = int(
        len(valid_line_indices) * line_keep_ratio
    )

    # LineScore 내림차순
    # 점수가 같으면 원래 line 번호가 작은 것을 먼저 선택
    ranked_lines = sorted(
        valid_line_indices,
        key=lambda line_idx: (
            -line_scores[line_idx],
            line_idx
        )
    )

    top_k_lines = ranked_lines[:num_lines_to_keep]
    
    print("\n--- Top-K Line Debug ---")

    print(
        f"valid lines: {len(valid_line_indices)}"
    )

    print(
        f"line keep ratio: {line_keep_ratio}"
    )

    print(
        f"selected lines: {num_lines_to_keep}"
    )

    for rank, line_idx in enumerate(top_k_lines, start=1):
        print(
            f"Rank {rank:02d} | "
            f"Line {line_idx:02d} | "
            f"Score={line_scores[line_idx]:.4f} | "
            f"{lines[line_idx].strip()}"
        )
    # --------------------------------------------------
    # LineScore 출력
    # --------------------------------------------------

    print("\n--- LineScore Debug ---")

    print(f"slot count: {slot_count}")

    for line_idx in range(min(5, len(lines))):

        if len(line_to_tokens[line_idx]) == 0:
            print(
                f"Line {line_idx}: NO_TOKEN"
            )
            continue

        print(
            f"Line {line_idx}: "
            f"tokens={len(line_to_tokens[line_idx])}, "
            f"LineScore={line_scores[line_idx]:.4f}"
        )


    # --------------------------------------------------
    # Line 0 수동 검증
    # --------------------------------------------------

    debug_line_idx = 0
    debug_tokens = line_to_tokens[debug_line_idx]

    print("\n--- Manual Line Debug ---")

    print(
        f"Line {debug_line_idx} "
        f"total tokens: {len(debug_tokens)}"
    )

    first_layer_idx = sorted(press.layers.keys())[0]
    first_layer_indices = press.layers[first_layer_idx]

    num_kv_heads = first_layer_indices.shape[1]

    for kv_head_idx in range(num_kv_heads):

        kept_tokens = set(
            first_layer_indices[0, kv_head_idx].tolist()
        )

        kept_count = sum(
            token_idx in kept_tokens
            for token_idx in debug_tokens
        )

        ratio = (
            kept_count / len(debug_tokens)
        )

        print(
            f"Layer {first_layer_idx}, "
            f"Head {kv_head_idx}: "
            f"{kept_count}/{len(debug_tokens)} "
            f"= {ratio:.4f}"
        )

    print(
        f"Final LineScore: "
        f"{line_scores[debug_line_idx]:.4f}"
    )
            
    
    print("\n--- Tracking Debug ---")

    print(
        f"tracked layers: {len(press.layers)}"
    )

    first_layer_idx = sorted(press.layers.keys())[0]

    first_layer_indices = press.layers[first_layer_idx]

    print(
        f"first layer: {first_layer_idx}"
    )

    print(
        f"indices shape: {first_layer_indices.shape}"
    )

    print(
        f"num kv heads: {first_layer_indices.shape[1]}"
    )
    
    print("\n--- Line Mapping Debug ---")

    for line_idx in range(min(5, len(lines))):
        print(
            f"Line {line_idx}: "
            f"{line_to_tokens[line_idx]}"
        )
    
    num_layers = len(outputs.past_key_values.layers)
    original_tokens = inputs["input_ids"].shape[1]
    compressed_tokens = outputs.past_key_values.layers[0].keys.shape[2]
    
