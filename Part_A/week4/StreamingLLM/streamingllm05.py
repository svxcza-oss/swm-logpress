from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from kvpress import StreamingLLMPress
from pathlib import Path

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

log_path = Path(__file__).with_name("thunderbird_sample.txt")
logs = log_path.read_text(encoding="utf-8")
lines = logs.splitlines(keepends=True)
cursor = 0

inputs = tokenizer(
    logs,
    return_tensors = "pt",
    return_offsets_mapping=True
    )

offset_mapping = inputs.pop("offset_mapping")


model.to("cuda")
inputs.to("cuda")
model.eval()

with torch.no_grad():
    outputs = model(**inputs)
compression_ratio = 0.5
press = StreamingLLMPress(compression_ratio)

with torch.no_grad(), press(model):
    press_outputs = model(**inputs)

line_ranges = []
line_lst = []
kept_token_indices = []

k_len = inputs["input_ids"].shape[1]
n_sink = 4
n_pruned = k_len - int(k_len * (1 - compression_ratio))

for token_idx in range(k_len):
    if token_idx < n_sink or token_idx >= n_sink + n_pruned:
        kept_token_indices.append(token_idx)

keep = []
partial = []
drop = []
line_token_counts = []

for line_no, line in enumerate(lines, start=1):
    start = cursor
    end = start + len(line)

    line_ranges.append([line_no, start, end])

    cursor = end

    token_indices = []

    for token_idx, offset in enumerate(offset_mapping[0]):
        token_start = offset[0].item()
        token_end = offset[1].item()
        
        if token_start >= start and token_end <= end:
            token_indices.append(token_idx)

    line_lst.append(token_indices)

    token_count = 0

    for i in token_indices:
        if i in kept_token_indices:
            token_count += 1

    line_token_counts.append([token_count, len(token_indices)])

    if token_count == len(token_indices):
        keep.append(line_no)
    elif token_count > 0:
        partial.append(line_no)
    else:
        drop.append(line_no)
    

print("Input file:", log_path.name)
print("Inputs tokens: ", k_len)
print("Full KV: ", outputs.past_key_values.layers[0].keys.shape)
print("StreamingLLM: ", press_outputs.past_key_values.layers[0].keys.shape)

for line_no, line in enumerate(lines, start=1):
    survived = line_token_counts[line_no - 1][0]
    total = line_token_counts[line_no - 1][1]

    if line_no in keep:
        print("[KEEP]    line", line_no, ":", line.rstrip(), end=" ")
    elif line_no in partial:
        print("[PARTIAL] line", line_no, ":", line.rstrip(), end=" ")
    elif line_no in drop:
        print("[DROP]    line", line_no, ":", line.rstrip(), end=" ")
    print("(", survived, "/", total, ")")

