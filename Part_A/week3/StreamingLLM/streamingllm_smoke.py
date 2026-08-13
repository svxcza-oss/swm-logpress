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
model.eval()

with torch.no_grad():
    outputs = model(**inputs)

press = StreamingLLMPress(compression_ratio=0.5)

with torch.no_grad(), press(model):
    press_outputs = model(**inputs)


print("Input tokens: ", inputs["input_ids"].shape[1])

print("Full KV: ")
print(outputs.past_key_values.layers[0].keys.shape)

print("StreamingLLM KV: ")
print(press_outputs.past_key_values.layers[0].keys.shape)