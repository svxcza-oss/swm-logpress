# Qwen 3B kvpress Shape Compression Test

## 목적

현재 로컬 Qwen 3B 환경에서 kvpress가 정상적으로 적용되는지 확인한다.
핵심 확인 내용은 press 적용 후 `past_key_values`의 sequence length 차원이 줄어드는지 여부이다.

## 환경

- Model: `Qwen/Qwen2.5-3B-Instruct`
- Press: `KnormPress(compression_ratio=0.5)`
- GPU: `NVIDIA GeForce RTX 5060 Ti`
- CUDA available: `True`
- Input token count: `48`

## 입력 로그

```text
[12:00:01] INFO health check ok
[12:00:02] WARN db connection pool 95%
[12:00:03] ERROR payment-service timeout

```

## 압축 전

- past_key_values type: `<class 'transformers.cache_utils.DynamicCache'>`
- cache access method: `DynamicCache.layers[0].keys/values`
- number of layers: `36`
- layer0 key shape: `(1, 2, 48, 128)`
- layer0 value shape: `(1, 2, 48, 128)`
- KV seq_len: `48`

## 압축 후

- past_key_values type: `<class 'transformers.cache_utils.DynamicCache'>`
- cache access method: `DynamicCache.layers[0].keys/values`
- number of layers: `36`
- layer0 key shape: `(1, 2, 24, 128)`
- layer0 value shape: `(1, 2, 24, 128)`
- KV seq_len: `24`

## 요약

- Before seq_len: `48`
- After seq_len: `24`
- Compressed: `True`

## 해석

`Compressed`가 `True`이면 kvpress가 KV cache sequence length를 실제로 줄인 것이다.
예를 들어 압축 전 shape가 `(1, 2, 48, 128)`이고 압축 후 shape가 `(1, 2, 24, 128)`이면 sequence length가 48에서 24로 줄어든 것이다.

## GPU Memory

- Max memory allocated GB: `5.786745071411133`
