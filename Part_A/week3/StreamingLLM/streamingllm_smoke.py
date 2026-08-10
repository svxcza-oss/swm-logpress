from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from kvpress import StreamingLLMPress

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

logs = """ERROR database connection timeout
99892767,1143750684,en248,104,-,removing telnet,99572410
99892768,1143750684,en248,104,-,removing time,99572410
99892769,1143750684,en248,104,-,removing time,99572410"""

inputs = tokenizer(
    logs,
    return_tensors = "pt"
    )

model.to("cuda")
inputs.to("cuda")

with torch.no_grad():
    outputs = model(**inputs)


print(outputs.past_key_values.layers[0].keys.shape)
print(outputs.past_key_values.layers[0].values.shape)