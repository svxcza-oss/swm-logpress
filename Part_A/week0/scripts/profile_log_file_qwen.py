import argparse
import csv
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_first_layer_kv(past):
    if isinstance(past, (tuple, list)):
        k, v = past[0]
        return k, v, "tuple/list past[0]"

    if hasattr(past, "to_legacy_cache"):
        try:
            legacy = past.to_legacy_cache()
            k, v = legacy[0]
            return k, v, "DynamicCache.to_legacy_cache()[0]"
        except Exception as e:
            print("to_legacy_cache failed:", repr(e))

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

    raise RuntimeError("Could not find key/value tensors in past_key_values cache structure.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to log text file")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--max-lines", type=int, default=None)
    parser.add_argument("--output", default="results/qwen3b_log_memory_profile.csv")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = input_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    if args.max_lines is not None:
        lines = lines[:args.max_lines]

    text = "\n".join(lines)

    print("model:", args.model)
    print("input file:", input_path)
    print("line count:", len(lines))
    print("cuda available:", torch.cuda.is_available())

    print("loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print("tokenizing...")
    inputs = tokenizer(text, return_tensors="pt")
    input_token_count = inputs["input_ids"].shape[1]
    print("input token count:", input_token_count)

    print("loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    inputs = inputs.to(model.device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print("running forward...")
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    past = outputs.past_key_values
    k, v, access_method = get_first_layer_kv(past)

    kv_seq_len = k.shape[-2]

    print("forward success")
    print("past_key_values type:", type(past))
    print("cache access method:", access_method)
    print("num layers:", len(past))
    print("layer0 key shape:", k.shape)
    print("layer0 value shape:", v.shape)
    print("kv seq len:", kv_seq_len)

    max_memory_gb = None
    gpu_name = None

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        max_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
        print("gpu:", gpu_name)
        print("max memory GB:", max_memory_gb)

    seq_match = input_token_count == kv_seq_len
    print("token count == kv seq len:", seq_match)

    file_exists = output_path.exists()

    with output_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "input_file",
                "line_count",
                "input_token_count",
                "kv_seq_len",
                "seq_match",
                "num_layers",
                "layer0_key_shape",
                "layer0_value_shape",
                "gpu",
                "max_memory_gb",
                "result",
            ],
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "model": args.model,
                "input_file": str(input_path),
                "line_count": len(lines),
                "input_token_count": input_token_count,
                "kv_seq_len": kv_seq_len,
                "seq_match": seq_match,
                "num_layers": len(past),
                "layer0_key_shape": str(tuple(k.shape)),
                "layer0_value_shape": str(tuple(v.shape)),
                "gpu": gpu_name,
                "max_memory_gb": max_memory_gb,
                "result": "success",
            }
        )

    print("saved result to:", output_path)


if __name__ == "__main__":
    main()
