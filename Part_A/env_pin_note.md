# A Track Local Environment Pin Note

## Local Environment

- OS: Windows
- Python: 3.11.9
- GPU: NVIDIA GeForce RTX 5060 Ti
- VRAM: 16GB
- Project path: C:\Users\qkral\projects\log-rca-kv

## Verified Packages

- torch: installed
- transformers: installed
- kvpress: installed
- huggingface_hub: installed
- accelerate: installed
- bitsandbytes: installed, but Windows local 8bit forward was not practical

## Qwen/Qwen2.5-3B-Instruct Test

Result: success

- Model load: success
- Forward pass: success
- past_key_values access: success
- KV cache shape access: success
- CUDA available: True

Initial short log test:

- input token count: 48
- num layers: 36
- layer0 key shape: torch.Size([1, 2, 48, 128])
- layer0 value shape: torch.Size([1, 2, 48, 128])
- max memory GB: 5.77GB

Open-source log profiling:

- 50 lines: success
- 100 lines: success
- 200 lines: success
- 250 lines: CUDA out of memory
- 300 lines: CUDA out of memory

Conclusion:

Local RTX 5060 Ti 16GB is enough for small-model KV cache access tests and short-to-medium log window profiling.
Longer log windows and 7B~8B baseline experiments should be run on a larger GPU server such as RunPod or lab GPU.

## Quantization Test

### Qwen/Qwen2.5-3B-Instruct 8bit

Result: not practical on local Windows environment

- 8bit bitsandbytes test with 200 log lines stayed at running forward for more than 2 hours.
- This suggests that Windows local bitsandbytes 8bit inference is not suitable for this experiment.
- Quantization should be re-tested on Linux GPU server such as RunPod.

## Hugging Face / Llama Status

- Hugging Face login: success
- Active token saved locally by Hugging Face CLI
- Do not commit token files to GitHub
- Llama-3.1-8B-Instruct: access request submitted / pending approval
- Llama-3.2-1B-Instruct: access request submitted / awaiting review

Current Llama status:

- Tokenizer/model loading is blocked by gated repo approval.
- Error type: 403 Forbidden / awaiting review
- Next step: retry Llama tokenizer loading after approval.

## Next Steps

1. Retry Llama tokenizer access after Hugging Face approval.
2. Run Llama-3.2-1B-Instruct smoke test first.
3. If 1B succeeds, try Llama-3.1-8B-Instruct with 1-line input.
4. Keep Qwen 3B fp16 as current local baseline.
5. Move official 7B~8B baseline experiments to RunPod or lab GPU.
