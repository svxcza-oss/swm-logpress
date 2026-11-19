import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-3B-Instruct"

text = """[12:00:01] INFO health check ok
[12:00:02] WARN db connection pool 95%
[12:00:03] ERROR payment-service timeout
"""

def get_first_layer_kv(past):
    # Case 1: old-style tuple/list cache
    if isinstance(past, (tuple, list)):
        k, v = past[0]
        return k, v, "tuple/list past[0]"

    # Case 2: DynamicCache with to_legacy_cache()
    if hasattr(past, "to_legacy_cache"):
        try:
            legacy = past.to_legacy_cache()
            k, v = legacy[0]
            return k, v, "DynamicCache.to_legacy_cache()[0]"
        except Exception as e:
            print("to_legacy_cache failed:", repr(e))

    # Case 3: DynamicCache with key_cache/value_cache
    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        k = past.key_cache[0]
        v = past.value_cache[0]
        return k, v, "DynamicCache.key_cache/value_cache"

    # Case 4: newer DynamicCache with layers
    if hasattr(past, "layers"):
        layer0 = past.layers[0]
        print("layer0 type:", type(layer0))
        print("layer0 attrs:", [a for a in dir(layer0) if not a.startswith("_")])

        # Common newer names
        for k_name, v_name in [
            ("keys", "values"),
            ("key_states", "value_states"),
            ("key_cache", "value_cache"),
        ]:
            if hasattr(layer0, k_name) and hasattr(layer0, v_name):
                k = getattr(layer0, k_name)
                v = getattr(layer0, v_name)
                return k, v, f"DynamicCache.layers[0].{k_name}/{v_name}"

    raise RuntimeError("Could not find key/value tensors in past_key_values cache structure.")

print("model:", model_name)
print("cuda available:", torch.cuda.is_available())

print("loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

model.eval()

inputs = tokenizer(text, return_tensors="pt").to(model.device)

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

print("running forward...")
with torch.no_grad():
    outputs = model(**inputs, use_cache=True)

past = outputs.past_key_values

print("forward success")
print("past_key_values type:", type(past))

try:
    print("num layers:", len(past))
except Exception as e:
    print("len(past) failed:", repr(e))

k, v, access_method = get_first_layer_kv(past)

print("cache access method:", access_method)
print("layer0 key shape:", k.shape)
print("layer0 value shape:", v.shape)

input_ids = inputs["input_ids"][0]
print("input token count:", input_ids.shape[0])

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("max memory GB:", torch.cuda.max_memory_allocated() / 1024**3)
