# A Track 로컬 환경 기록

## 중요 안내

이 문서는 Windows 로컬 개발 환경 기록입니다.

이 파일은 RunPod/Linux 공식 lock 파일이 아닙니다.

RunPod/Linux 환경에서는 다음 파일을 기준으로 사용합니다.

- requirements.in: 패키지 의도 목록
- requirements-runpod-freeze.txt: RunPod에서 실제 설치와 검증 후 생성할 진짜 lock 파일

## 로컬 환경

- OS: Windows
- Python: 3.11.9
- GPU: NVIDIA GeForce RTX 5060 Ti
- VRAM: 16GB
- 프로젝트 경로: C:\Users\qkral\projects\log-rca-kv

## 로컬에서 확인한 패키지

- torch: 설치 확인
- transformers: 설치 확인
- kvpress: 설치 확인
- huggingface_hub: 설치 확인
- accelerate: 설치 확인
- bitsandbytes: 설치 확인

단, bitsandbytes 8bit 실험은 Windows 로컬 환경에서 실용적이지 않았습니다.

## Qwen/Qwen2.5-3B-Instruct 테스트

결과: 성공

확인한 내용:

- 모델 로드 성공
- forward pass 성공
- past_key_values 접근 성공
- KV cache shape 확인 성공
- CUDA 사용 가능 확인

짧은 로그 입력 테스트 결과:

- input token count: 48
- num layers: 36
- layer0 key shape: torch.Size([1, 2, 48, 128])
- layer0 value shape: torch.Size([1, 2, 48, 128])
- max memory GB: 약 5.77GB

## 실제 로그 입력 메모리 프로파일링

Qwen/Qwen2.5-3B-Instruct 기준:

- 50줄: 성공
- 100줄: 성공
- 200줄: 성공
- 250줄: CUDA out of memory
- 300줄: CUDA out of memory

해석:

로컬 RTX 5060 Ti 16GB 환경은 작은 모델의 KV cache 접근 테스트와 짧은/중간 길이 로그 window 검증에는 사용할 수 있습니다.

하지만 긴 로그 window나 7B~8B 모델 기반 공식 baseline 실험은 RunPod 또는 연구실 GPU 서버에서 진행하는 것이 적절합니다.

## 양자화 실험

### Qwen/Qwen2.5-3B-Instruct 8bit

결과: Windows 로컬 환경에서는 실용적이지 않음

- 200줄 로그 입력 기준으로 running forward 상태가 2시간 이상 지속되었습니다.
- 따라서 Windows 로컬에서 bitsandbytes 8bit 추론은 이번 실험에 적합하지 않다고 판단했습니다.
- 양자화 실험은 RunPod/Linux 환경에서 다시 검증해야 합니다.

## Hugging Face / Llama 상태

- Hugging Face 로그인: 성공
- Hugging Face CLI가 로컬에 active token 저장 완료
- Hugging Face token 파일은 GitHub에 올리면 안 됩니다.
- Llama-3.1-8B-Instruct: 접근 요청 완료, 승인 대기 중
- Llama-3.2-1B-Instruct: 접근 요청 완료, 승인 대기 중

현재 Llama 상태:

- gated repository 승인 대기 때문에 tokenizer/model 로딩이 아직 차단되어 있습니다.
- 에러 유형: 403 Forbidden / awaiting review
- 다음 단계: 승인 완료 후 Llama tokenizer 로드 재시도

## 결론

현재 Windows 로컬 환경은 Qwen 3B 기반 소규모 KV cache 검증용으로 사용 가능합니다.

공식 7B~8B 실험과 최종 패키지 lock 생성은 RunPod/Linux 환경에서 진행해야 합니다.
