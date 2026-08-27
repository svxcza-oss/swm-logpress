from transformers import AutoTokenizer, AutoModelForCausalLM 
import torch 
from kvpress import SnapKVPress 
from pathlib import Path 
 
class TrackingSnapKVPress (SnapKVPress): 
     
    def compress(self, module, hidden_states, keys, values, attentions, kwargs): 
        if self.compression_ratio == 0: 
            return keys, values 
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

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    torch_dtype=torch.float16
    ) 

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct") 

log_path = (
    Path(__file__).resolve().parents[2]
    / "week6"
    / "thunderbird"
    / "subchunks"
    / "window_314927_chunk13_sub3.txt"
)
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


line_spans = [] 
for line_no, line in enumerate(lines): 
    line_start = cursor 
    cursor += len(line) 
    line_end = cursor 
     
    # print("line ", line_no, "(", line_start, ",", line_end, ")") 
    line_spans.append([line_start, line_end]) 
     
print(line_spans[:3]) 
print(cursor, len(logs)) 
     
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

for i in compression: 
    compression_ratio = i 
    press = TrackingSnapKVPress( 
        compression_ratio=compression_ratio, 
        window_size=64, 
        kernel_size=5) 
    press.layers = {} 

    with torch.no_grad(), press(model): 
        press_outputs = model(**inputs) 
         
    num_layers = len(press_outputs.past_key_values.layers) 
     
    original_tokens = inputs["input_ids"].shape[1] 
    compressed_tokens = press_outputs.past_key_values.layers[0].keys.shape[2] 

    print() 
    print("=" * 80) 
    print(f"압축률: {compression_ratio}") 
    print(f"원본 토큰 수: {original_tokens}") 
    print(f"압축 후 KV 토큰 수: {compressed_tokens}") 
     

    target_layer = 0 
    
    layer_indices = press.layers[target_layer][0] 

    
    kept_tokens = set(layer_indices.reshape(-1).tolist()) 
    

    line_total = [0] * len(lines) 
    line_kept = [0] * len(lines) 
    
    for token_idx, line_idx in enumerate(token_to_line): 
    
        line_total[line_idx] += 1 
    
        if token_idx in kept_tokens: 
            line_kept[line_idx] += 1 
    
    
    print(f"압축률: {compression_ratio}") 
    print(f"기준 레이어: {target_layer}") 
    print(f"생존 unique token 수: {len(kept_tokens)}") 
    print("=" * 80) 
    
    
    keep_count = 0 
    partial_count = 0 
    drop_count = 0 
    
    for line_no, line in enumerate(lines): 
    
        total = line_total[line_no] 
        kept = line_kept[line_no] 
    
        if total == 0: 
            status = "NO_TOKEN" 
    
        elif kept == 0: 
            status = "DROP" 
            drop_count += 1 
    
        elif kept == total: 
            status = "KEEP" 
            keep_count += 1 
    
        else: 
            status = "PARTIAL" 
            partial_count += 1 
    
        print( 
            f"Line {line_no:02d} | " 
            f"{status:7} | " 
            f"{kept}/{total} | " 
            f"{line.rstrip(chr(10)).rstrip(chr(13))}" 
        ) 
    
        
    print("-" * 80) 
    print( 
        f"KEEP={keep_count}, " 
        f"PARTIAL={partial_count}, " 
        f"DROP={drop_count}" 
    ) 

