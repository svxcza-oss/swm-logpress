# LogPress Week 3 진행 요약

## 이번 주 한 일

- `Practice01.py`에서 tokenizer, model forward, KV Cache 접근 흐름을 실습했다.
- Llama와 Qwen 모델에서 `past_key_values` 구조를 확인했다.
- Qwen2.5-3B 기준으로 첫 번째 layer의 Key/Value shape를 출력했다.
- StreamingLLM 실험용 가상환경을 별도로 구성했다.
- `StreamingLLMPress` import가 정상 동작하는 것을 확인했다.
- `streamingllm_smoke.py`에서 Qwen2.5-3B로 Full KV shape를 확인했다.

## 확인한 내용

- tokenizer 출력은 `input_ids`, `attention_mask`로 구성된다.
- `return_tensors="pt"`를 사용하면 PyTorch Tensor로 변환된다.
- 모델과 입력 Tensor는 같은 device에 있어야 한다.
- `past_key_values`는 layer별 Key/Value cache를 담는다.
- KV shape의 세 번째 차원은 입력 token 길이와 연결된다.

## 현재 결과

```text
Practice01.py
- Llama tokenizer/model 실습
- DynamicCache 확인
- layer 0 K/V shape 확인

streamingllm_smoke.py
- Qwen/Qwen2.5-3B-Instruct 사용
- Full KV shape 확인
- 짧은 입력: [1, 2, 4, 128]
- 여러 줄 로그 입력: [1, 2, 131, 128]
```

## 환경

```text
Python: 3.11.9
PyTorch: 2.13.0+cu130
Transformers: 4.57.6
kvpress: 0.5.4
GPU: NVIDIA GeForce RTX 5060 Ti
```

## 아직 하지 않은 것

- StreamingLLMPress 실제 압축 적용
- 압축 전후 KV shape 비교
- Thunderbird 최종 1시간 윈도우 평가
- 이상 로그 보존율 계산

## 다음 작업

- `StreamingLLMPress(compression_ratio=0.5)` 적용
- Full KV와 압축 KV shape 비교
- 실제 Thunderbird message 기반 입력으로 확장
- 실행 결과를 CSV 또는 Markdown으로 정리
