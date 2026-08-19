from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from kvpress import SnapKVPress
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

print(offset_mapping.shape)
print(offset_mapping[0][:10])

compression = [0.5, 0.6, 0.7, 0.8, 0.9]

for i in compression:
    compression_ratio = i
    press = SnapKVPress(compression_ratio=compression_ratio, 
                        window_size=64, 
                        kernel_size=5
                        )

    with torch.no_grad(), press(model): 
        press_outputs = model(**inputs)

    num_layers = len(press_outputs.past_key_values.layers)
    print("[압축률: ", compression_ratio, "]")
    print("전체 레이어 수: ", num_layers)
    print("첫 레이어: ", press_outputs.past_key_values.layers[0].keys.shape)
    print("중간 레이어: ", press_outputs.past_key_values.layers[num_layers//2].keys.shape)
    print("마지막 레이어: ", press_outputs.past_key_values.layers[-1].keys.shape)