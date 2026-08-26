from transformers import AutoTokenizer, AutoModelForCausalLM
from kvpress import ObservedAttentionPress
import torch

model_name = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="eager"
    )

text = "This is a simple test log.\nAnother log line."

inputs = tokenizer(
    text,
    return_tensors="pt"
    )
model.to("cuda")
inputs.to("cuda")
model.eval()



with torch.no_grad():
    original_outputs = model(**inputs, use_cache=True)

print(f"원본: {original_outputs.past_key_values.layers[0].keys.shape}")


compression = [0.5, 0.7, 0.9]

for compression_ratio in compression:
    press = ObservedAttentionPress(compression_ratio)
    
    with torch.no_grad():
        with press(model):
            outputs = model(**inputs, use_cache=True)
    print(f"압축률: {compression_ratio}")
    print(outputs.past_key_values.layers[0].keys.shape)