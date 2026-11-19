# Part A - Infrastructure / Environment Verification

## Summary

This folder contains the initial local environment verification for Track A.

The goal of this test is to verify:

- CUDA availability
- PyTorch / Transformers / kvpress import
- Qwen model loading
- Forward pass on short log input
- `past_key_values` / KV cache access
- Basic GPU memory usage measurement

## Local Test Environment

- OS: Windows
- Python: 3.11.9
- GPU: RTX 5060 Ti
- VRAM: 16GB
- Model: Qwen/Qwen2.5-3B-Instruct
- Result: Success

## KV Cache Test Result

- input token count: 48
- num layers: 36
- layer0 key shape: torch.Size([1, 2, 48, 128])
- layer0 value shape: torch.Size([1, 2, 48, 128])
- max memory GB: 5.772336959838867

## Files

- `setup.md`: detailed local environment verification record
- `requirements-local-freeze.txt`: local Python package freeze
- `scripts/kv_access_qwen_test.py`: Qwen KV cache access test script
- `scripts/check_gpu.py`: CUDA/GPU check script, if available

## Note

This is a local development environment record.
Official 7B~8B model experiments and baseline reproduction should be pinned again on RunPod or lab GPU servers.
