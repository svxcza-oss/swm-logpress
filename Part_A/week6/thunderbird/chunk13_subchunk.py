from pathlib import Path
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

log_path = (
    Path(__file__).resolve().parent
    / "chunks"
    / "window_314927_chunk13.txt"
)

output_dir = Path(__file__).resolve().parent / "subchunks"
output_dir.mkdir(parents=True, exist_ok=True)

MAX_TOKENS = 8192

lines = log_path.read_text(
    encoding="utf-8"
).splitlines(keepends=True)

chunks = {}
chunk_idx = 0
token_size = 0

for line in lines:

    line_token_size = len(
        tokenizer(
            line,
            add_special_tokens=False
        )["input_ids"]
    )

    if token_size + line_token_size > MAX_TOKENS:
        chunk_idx += 1
        token_size = 0

    chunk_name = f"sub{chunk_idx}"

    if chunk_name not in chunks:
        chunks[chunk_name] = []

    chunks[chunk_name].append(line)
    token_size += line_token_size


for chunk_name, chunk_lines in chunks.items():

    actual_tokens = len(
        tokenizer(
            "".join(chunk_lines),
            add_special_tokens=False
        )["input_ids"]
    )

    print(
        chunk_name,
        "logs:", len(chunk_lines),
        "tokens:", actual_tokens
    )

    output_path = (
        output_dir
        / f"window_314927_chunk13_{chunk_name}.txt"
    )

    output_path.write_text(
        "".join(chunk_lines),
        encoding="utf-8"
    )