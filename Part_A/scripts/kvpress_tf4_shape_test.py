import torch
import importlib.metadata as metadata
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from kvpress import KnormPress


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

TEXT = """[12:00:01] INFO health check ok
[12:00:02] WARN db connection pool 95%
[12:00:03] ERROR payment-service timeout
"""


def package_version(package_name):
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def get_first_layer_kv(past):
    if isinstance(past, (tuple, list)):
        k, v = past[0]
        return k, v, "tuple/list past[0]"

    if hasattr(past, "to_legacy_cache"):
        try:
            legacy = past.to_legacy_cache()
            k, v = legacy[0]
            return k, v, "DynamicCache.to_legacy_cache()[0]"
        except Exception:
            pass

    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        k = past.key_cache[0]
        v = past.value_cache[0]
        return k, v, "DynamicCache.key_cache/value_cache"

    if hasattr(past, "layers"):
        layer0 = past.layers[0]
        for k_name, v_name in [
            ("keys", "values"),
            ("key_states", "value_states"),
            ("key_cache", "value_cache"),
        ]:
            if hasattr(layer0, k_name) and hasattr(layer0, v_name):
                k = getattr(layer0, k_name)
                v = getattr(layer0, v_name)
                return k, v, f"DynamicCache.layers[0].{k_name}/{v_name}"

    raise RuntimeError("Could not find key/value tensors in past_key_values.")


def cache_info(outputs):
    past = outputs.past_key_values
    k, v, method = get_first_layer_kv(past)

    return {
        "past_type": str(type(past)),
        "access_method": method,
        "num_layers": len(past),
        "key_shape": tuple(k.shape),
        "value_shape": tuple(v.shape),
        "seq_len": k.shape[-2],
    }


def main():
    result_path = Path("Part_A/results/kvpress_tf4_shape_result.md")
    result_path.parent.mkdir(parents=True, exist_ok=True)

    transformers_version = package_version("transformers")
    kvpress_version = package_version("kvpress")
    torch_version = package_version("torch")
    numpy_version = package_version("numpy")

    print("model:", MODEL_NAME)
    print("torch:", torch_version)
    print("transformers:", transformers_version)
    print("kvpress:", kvpress_version)
    print("numpy:", numpy_version)
    print("cuda available:", torch.cuda.is_available())

    print("loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    inputs = tokenizer(TEXT, return_tensors="pt").to(model.device)
    input_token_count = inputs["input_ids"].shape[1]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print("running forward without press...")
    with torch.no_grad():
        outputs_before = model(**inputs, use_cache=True)

    before = cache_info(outputs_before)

    press = KnormPress(compression_ratio=0.5)

    print("running forward with KnormPress(compression_ratio=0.5)...")
    with torch.no_grad(), press(model):
        outputs_after = model(**inputs, use_cache=True)

    after = cache_info(outputs_after)

    compressed = after["seq_len"] < before["seq_len"]

    gpu_name = "N/A"
    max_memory_gb = "N/A"

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        max_memory_gb = torch.cuda.max_memory_allocated() / 1024**3

    report = []
    report.append("# transformers 4.x + kvpress 0.5.4 Shape Compression Test")
    report.append("")
    report.append("## Purpose")
    report.append("")
    report.append("Verify whether kvpress actually reduces the `past_key_values` sequence length in a transformers 4.x environment.")
    report.append("")
    report.append("## Required Records")
    report.append("")
    report.append(f"- transformers version: `{transformers_version}`")
    report.append(f"- kvpress version: `{kvpress_version}`")
    report.append(f"- before layer0 key shape: `{before['key_shape']}`")
    report.append(f"- after layer0 key shape: `{after['key_shape']}`")
    report.append("")
    report.append("## Environment")
    report.append("")
    report.append(f"- Model: `{MODEL_NAME}`")
    report.append("- Press: `KnormPress(compression_ratio=0.5)`")
    report.append(f"- torch version: `{torch_version}`")
    report.append(f"- numpy version: `{numpy_version}`")
    report.append(f"- GPU: `{gpu_name}`")
    report.append(f"- CUDA available: `{torch.cuda.is_available()}`")
    report.append(f"- Input token count: `{input_token_count}`")
    report.append("")
    report.append("## Input Text")
    report.append("")
    report.append("```text")
    report.append(TEXT)
    report.append("```")
    report.append("")
    report.append("## Before Press")
    report.append("")
    report.append(f"- past_key_values type: `{before['past_type']}`")
    report.append(f"- cache access method: `{before['access_method']}`")
    report.append(f"- number of layers: `{before['num_layers']}`")
    report.append(f"- layer0 key shape: `{before['key_shape']}`")
    report.append(f"- layer0 value shape: `{before['value_shape']}`")
    report.append(f"- KV seq_len: `{before['seq_len']}`")
    report.append("")
    report.append("## After Press")
    report.append("")
    report.append(f"- past_key_values type: `{after['past_type']}`")
    report.append(f"- cache access method: `{after['access_method']}`")
    report.append(f"- number of layers: `{after['num_layers']}`")
    report.append(f"- layer0 key shape: `{after['key_shape']}`")
    report.append(f"- layer0 value shape: `{after['value_shape']}`")
    report.append(f"- KV seq_len: `{after['seq_len']}`")
    report.append("")
    report.append("## Summary")
    report.append("")
    report.append(f"- Before seq_len: `{before['seq_len']}`")
    report.append(f"- After seq_len: `{after['seq_len']}`")
    report.append(f"- Compressed: `{compressed}`")
    report.append(f"- Max memory allocated GB: `{max_memory_gb}`")
    report.append("")
    report.append("## Interpretation")
    report.append("")
    if compressed:
        report.append("kvpress successfully reduced the KV cache sequence length.")
    else:
        report.append("kvpress did not reduce the KV cache sequence length in this run.")
    report.append("")

    result_path.write_text("\n".join(report), encoding="utf-8")

    print()
    print("== required records ==")
    print("transformers version:", transformers_version)
    print("kvpress version:", kvpress_version)
    print("before layer0 key shape:", before["key_shape"])
    print("after layer0 key shape:", after["key_shape"])

    print()
    print("== summary ==")
    print("before seq_len:", before["seq_len"])
    print("after seq_len:", after["seq_len"])
    print("compressed:", compressed)
    print("saved result to:", result_path)


if __name__ == "__main__":
    main()
