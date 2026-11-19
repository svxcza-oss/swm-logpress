from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

inputs = tokenizer(
    "ERROR database connection timeout",
    return_tensors = "pt"
    )

model.to("cuda:0")
inputs.to("cuda:0")

outputs = model(**inputs)

print(inputs)

# print(next(model.parameters()).device)
# print(inputs["input_ids"].device)
# print(type(outputs))
# print(outputs.keys())
# print(type(outputs.past_key_values))
# print(len(outputs.past_key_values))
print(outputs.past_key_values.layers[0].keys.shape)
print(outputs.past_key_values.layers[0].values.shape)