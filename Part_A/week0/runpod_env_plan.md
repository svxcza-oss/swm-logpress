# A Track RunPod 환경 계획

## 목적

이 문서는 A Track의 공식 RunPod/Linux 실험 환경을 어떻게 만들고 고정할지 정리한 문서입니다.

현재 로컬 Windows 환경은 초기 개발과 작은 모델 검증용입니다.
공식 실험 환경의 lock 파일로 사용하면 안 됩니다.

## 현재 로컬 환경 상태

- OS: Windows
- GPU: RTX 5060 Ti 16GB
- Python: 3.11.9
- Qwen/Qwen2.5-3B-Instruct KV cache 접근 테스트 성공
- 실제 로그 입력 메모리 프로파일링 결과
  - 50줄: 성공
  - 100줄: 성공
  - 200줄: 성공
  - 250줄: CUDA out of memory
  - 300줄: CUDA out of memory

## Windows 8bit 양자화 실험 결과

- Qwen/Qwen2.5-3B-Instruct 8bit 실험을 Windows 로컬에서 진행했습니다.
- 200줄 입력 기준 running forward 상태에서 2시간 이상 완료되지 않았습니다.
- 따라서 Windows 로컬 환경에서는 bitsandbytes 8bit 추론이 실용적이지 않다고 판단했습니다.
- 양자화 실험은 RunPod/Linux 환경에서 다시 검증해야 합니다.

## 의존성 파일 구분

### requirements.in

RunPod/Linux에서 사용할 패키지 의도 목록입니다.
실제 고정 버전 파일이 아니라, 대략적으로 어떤 패키지를 사용할지 정리한 계획서입니다.

### requirements-windows-local-freeze.txt

Windows 로컬 개발 환경에서 생성한 패키지 기록입니다.
RunPod/Linux에 그대로 사용하면 안 됩니다.

### requirements-runpod-freeze.txt

나중에 RunPod에서 실제 설치와 검증이 끝난 뒤 생성할 진짜 lock 파일입니다.

생성 명령어:

pip freeze > requirements-runpod-freeze.txt

## RunPod 환경 구축 순서

1. RunPod에서 Linux GPU pod를 생성합니다.
2. PyTorch/CUDA 기반 이미지를 선택합니다.
3. 팀 GitHub 저장소를 clone합니다.
4. Python 가상환경을 생성합니다.
5. Part_A/requirements.in을 참고하여 필요한 패키지를 설치합니다.
6. flash-attn은 Linux/CUDA 환경에서 별도 설치가 필요할 수 있습니다.
7. Hugging Face에 로그인합니다.
8. 모델 로드를 확인합니다.
   - Qwen/Qwen2.5-7B-Instruct
   - meta-llama/Llama-3.1-8B-Instruct
9. KV cache 접근 테스트를 진행합니다.
10. 메모리 프로파일링 스크립트를 실행합니다.
11. 검증이 끝난 뒤 RunPod에서 진짜 lock 파일을 생성합니다.

## Llama 현재 상태

- Hugging Face 로그인: 성공
- Llama-3.2-1B-Instruct: 접근 요청 완료, 승인 대기 중
- Llama-3.1-8B-Instruct: 접근 요청 완료, 승인 대기 중
- 현재 차단 원인: Meta Llama gated repository 승인 대기

## 다음 할 일

1. Llama 접근 승인이 완료되면 tokenizer 로드부터 다시 확인합니다.
2. 먼저 Llama-3.2-1B-Instruct로 smoke test를 진행합니다.
3. 이후 Llama-3.1-8B-Instruct는 1줄 입력으로 최소 테스트를 진행합니다.
4. 공식 7B~8B 실험은 RunPod/Linux에서 진행합니다.
5. RunPod에서 검증이 끝난 뒤 requirements-runpod-freeze.txt를 생성합니다.
