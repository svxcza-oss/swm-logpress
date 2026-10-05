from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path 
from kvpress import ObservedAttentionPress
import torch
import csv
import json
import re

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
    / "window_314927_chunk13_sub0.txt"
)

metadata_path = (
    Path(__file__).resolve().parents[2]
    / "week6"
    / "thunderbird"
    / "chunks"
    / "window_314927_metadata.csv"
)

match = re.search(
    r"_chunk(\d+)_sub(\d+)\.txt$",
    log_path.name
)

if match is None:
    raise ValueError(
        f"subchunk 파일명 형식을 인식할 수 없습니다: {log_path.name}"
    )

chunk_idx = int(match.group(1))
sub_idx = int(match.group(2))

with metadata_path.open(
    "r",
    encoding="utf-8",
    newline=""
) as f:
    reader = csv.DictReader(f)

    chunk_row = next(
        (
            row
            for row in reader
            if row["chunk"] == f"chunk{chunk_idx}"
        ),
        None
    )
    
if chunk_row is None:
    raise ValueError(
        f"metadata에서 chunk{chunk_idx}를 찾지 못했습니다."
    )
chunk_start_line = int(chunk_row["start_line"])
alerts = json.loads(chunk_row["alerts"])

subchunk_offset = 0

for prev_sub_idx in range(sub_idx):
    prev_sub_path = log_path.with_name(
        f"window_314927_chunk{chunk_idx}_sub{prev_sub_idx}.txt"
    )

    prev_lines = prev_sub_path.read_text(
        encoding="utf-8"
    ).splitlines()

    subchunk_offset += len(prev_lines)

# metadata의 alert line을 현재 subchunk local line index로 변환
logs = log_path.read_text(encoding="utf-8")
lines = logs.splitlines(keepends=True)

inputs = tokenizer(
    logs,
    return_tensors="pt",
    return_offsets_mapping=True
    )
offset_mapping = inputs.pop("offset_mapping")

model.to("cuda")
inputs.to("cuda")
model.eval()

cursor = 0
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

# 임시 디버깅용 GT
# 현재 lines는 앞 30줄만 사용하므로 0~29 범위의 line index 사용

subchunk_start_line = (
    chunk_start_line + subchunk_offset
)

full_gt_lines = set()

for alert_line_str in alerts:
    alert_line = int(alert_line_str)

    local_line_idx = (
        alert_line - subchunk_start_line
    )

    if 0 <= local_line_idx < len(lines):
        full_gt_lines.add(local_line_idx)

gt_lines = full_gt_lines
        
for compression_ratio in compression:
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
    
    selected_lines = set(top_k_lines)
    
    invalid_gt_lines = [
        gt_line
        for gt_line in gt_lines
        if gt_line < 0 or gt_line >= len(lines)
    ]

    if invalid_gt_lines:
        raise ValueError(
            f"현재 입력 범위를 벗어난 GT line: {invalid_gt_lines}"
        )
        
    retained_gt_lines = gt_lines & selected_lines
    
    if len(gt_lines) == 0:
        retention = None
    else:
        retention = len(retained_gt_lines) / len(gt_lines)
    
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