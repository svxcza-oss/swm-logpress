from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path 
from kvpress import ObservedAttentionPress
import torch

class TrackingObservedAttentionPress(ObservedAttentionPress):
    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        
        scores = self.score(
            module,
            hidden_states,
            keys,
            values,
            attentions,
            kwargs
        )
        k_len = keys.shape[2]
        n_kept = int(k_len * (1 - self.compression_ratio))
        
        indices = scores.topk(n_kept, dim=-1).indices
        
        self.layers[module.layer_idx] = indices.detach().cpu()
        
        indices = indices.unsqueeze(-1).expand(
            -1, -1, -1, module.head_dim
        )
        
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()
        
        return keys, values
        

model_name = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="eager"
    )

log_path = Path(
    r"C:\Users**\qkral\OneDrive\바탕 화면\박민규\swm-project\Part_A\week6\thunderbird\c**hunks\window_314927_chunk13.txt"
)
logs = log_path.read_text(encoding="utf-8")
lines = logs.splitlines(keepends=True)
cursor = 0

inputs = tokenizer(
    logs,
    return_tensors="pt"
    return_offsets_mapping=True
    )
offset_mapping = inputs.pop("offset_mapping")

model.to("cuda")
inputs.to("cuda")
model.eval()

line_spans = []

for line_no, line in enumerate(lines):
    line_start = cursor
    cursor += len(line)
    line_end = cursor
    
    print(f"line: {line_no} ({line_start}, {line_end})")
    
    line_spans.append([line_start, line_end])

token_to_line = []
line_chk = 0

for token_no, token in enumerate(offset_mapping[0]):
    token_start = token[0]
    token_end = token[1]
    
    while(token_start >= line_spans[line_chk][1]):
        line_chk += 1
        
    token_to_line.append(line_chk)
    
print(len(token_to_line))
print(token_to_line[:20])


compression = [0.5, 0.7, 0.9]

for compression_ratio in compression:
    press = TrackingObservedAttentionPress(compression_ratio)
    press.layers = {}
    
    with torch.no_grad():
        with press(model):
            outputs = model(**inputs, use_cache=True)
    num_layers = len(outputs.past_key_values.layers)
    original_tokens = inputs["input_ids"].shape[1]
    compressed_tokens = outputs.past_key_values.layers[0].keys.shape[2]
    
    print()
    print("=" * 80)
    print(f"압축률: {compression_ratio}")
    print(f"원본 토큰 수: {original_tokens}")
    print(f"압축 후 KV 토큰 수: {compressed_tokens}")
    
    target_layer = 0
    
    
    
    print(outputs.past_key_values.layers[0].keys.shape)
    
    
    print(press.layers[0].shape)
    print(press.layers[0])