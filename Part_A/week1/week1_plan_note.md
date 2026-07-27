# A Track Week 1 작업 계획

## 이번 주 목표

Week 0에서는 로컬 GPU에서 Qwen 모델 실행, KV Cache 확인, kvpress 압축이 가능한지 검증했다.

Week 1에서는 이 환경을 Linux GPU 서버로 옮기고, 이후 팀 실험에 사용할 공식 환경을 만드는 것이 목표다.

## 해야 할 작업

### 1. RunPod/Linux 환경 구축

- GPU 서버 생성 및 저장소 clone
- Python 가상환경 생성
- PyTorch, Transformers, kvpress 설치
- CUDA가 정상적으로 인식되는지 확인
- 모델과 실험 결과가 사라지지 않도록 저장 경로 설정

### 2. 로컬 실험 재현

Week 0에서 성공한 실험을 Linux 환경에서도 다시 실행한다.

- Qwen 모델 로드
- KV Cache 접근
- 압축 전 길이 48 확인
- kvpress 적용 후 길이 24 확인
- GPU 사용량과 실행 시간 기록

같은 결과가 나오면 Linux 환경에서도 kvpress가 정상 동작한다고 판단한다.

### 3. 공식 버전 고정

Linux에서 검증이 끝난 뒤 실제 설치된 패키지 버전을 파일로 저장한다.

- PyTorch
- Transformers
- kvpress
- CUDA 관련 정보
- 사용한 GPU와 모델

결과 파일: `requirements-runpod-freeze.txt`

### 4. 7B~8B 모델 확인

- Qwen2.5-7B 모델 로드 및 KV Cache 접근 확인
- Llama 접근 승인이 완료되면 Llama-3.1-8B도 최소 입력으로 테스트
- 모델별 GPU 메모리 사용량 기록

### 5. 양자화 실험

양자화는 모델 자체의 크기를 줄여 GPU 메모리 사용량을 낮추는 방법이다.

Week 0의 Windows 8bit 실험은 지나치게 느렸으므로, Linux 환경에서 다시 확인한다.

비교 대상:

- 기본 FP16 모델
- 8bit 모델
- 가능하면 4bit 모델

기록할 항목:

- 모델 로드 성공 여부
- GPU 메모리 사용량
- 실행 시간
- KV Cache 접근 가능 여부
- kvpress와 함께 사용할 수 있는지

양자화는 KV Cache 압축과 다른 기술이므로, 공식 kvpress 성능 결과와 섞지 않고 별도 실험으로 기록한다.

## 이번 주 완료 기준

- RunPod/Linux에서 CUDA와 모델 실행 성공
- kvpress 압축 결과 48 → 24 재현
- 공식 패키지 버전 파일 생성
- Qwen 7B 또는 Llama 8B에서 KV Cache 접근 확인
- Linux에서 8bit 또는 4bit 양자화 가능성 확인
- 실행 방법과 결과를 GitHub에 정리

## 주의사항

양자화는 모델 크기를 줄이는 것이고, kvpress는 모델이 로그를 읽으며 만드는 임시 저장 공간을 줄이는 것이다.

두 방법의 효과가 섞이지 않도록 기본 모델 실험과 양자화 모델 실험을 분리해서 진행한다.