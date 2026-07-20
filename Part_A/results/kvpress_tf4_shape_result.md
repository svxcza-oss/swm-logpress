# transformers 4.x + kvpress 0.5.4 Shape Compression Test

## Purpose

Verify whether kvpress actually reduces the `past_key_values` sequence length in a transformers 4.x environment.

## Required Records

- transformers version: `4.57.6`
- kvpress version: `0.5.4`
- before layer0 key shape: `(1, 2, 48, 128)`
- after layer0 key shape: `(1, 2, 24, 128)`

## Environment

- Model: `Qwen/Qwen2.5-3B-Instruct`
- Press: `KnormPress(compression_ratio=0.5)`
- torch version: `2.8.0`
- numpy version: `2.4.6`
- GPU: `N/A`
- CUDA available: `False`
- Input token count: `48`

## Input Text

```text
[12:00:01] INFO health check ok
[12:00:02] WARN db connection pool 95%
[12:00:03] ERROR payment-service timeout

```

## Before Press

- past_key_values type: `<class 'transformers.cache_utils.DynamicCache'>`
- cache access method: `DynamicCache.to_legacy_cache()[0]`
- number of layers: `36`
- layer0 key shape: `(1, 2, 48, 128)`
- layer0 value shape: `(1, 2, 48, 128)`
- KV seq_len: `48`

## After Press

- past_key_values type: `<class 'transformers.cache_utils.DynamicCache'>`
- cache access method: `DynamicCache.to_legacy_cache()[0]`
- number of layers: `36`
- layer0 key shape: `(1, 2, 24, 128)`
- layer0 value shape: `(1, 2, 24, 128)`
- KV seq_len: `24`

## Summary

- Before seq_len: `48`
- After seq_len: `24`
- Compressed: `True`
- Max memory allocated GB: `N/A`

## Interpretation

kvpress successfully reduced the KV cache sequence length.
