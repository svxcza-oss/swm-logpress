# A Track Week 0 작업 메모

## 왜 이 작업을 했는가

우리 프로젝트는 긴 장애 로그를 AI 모델에 넣을 때 생기는 GPU 메모리 문제를 줄이는 것이 목표다.

본격적인 실험 전에 먼저 내 로컬 GPU에서 모델이 정상적으로 실행되는지, 모델의 임시 저장 공간인 KV Cache를 확인할 수 있는지, 그리고 kvpress로 실제 압축할 수 있는지 검증할 필요가 있었다.

또한 작은 로컬 GPU로 가능한 실험 범위를 확인하여, 이후 어떤 작업부터 큰 GPU 서버로 옮겨야 하는지 판단하고자 했다.

## 진행한 작업

- Windows 로컬 GPU 환경에 PyTorch, Transformers, kvpress 설치
- Qwen2.5-3B 모델 로드 및 실행
- 모델이 만든 KV Cache의 크기와 구조 확인
- kvpress 적용 전후 KV Cache 크기 비교
- 실제 로그 줄 수를 늘리며 GPU 메모리 사용량 측정
- 이후 RunPod/Linux 환경에서 사용할 패키지와 설치 계획 정리

## 주요 결과

- Qwen 모델 로드와 실행 성공
- 입력 토큰 48개에 대해 KV Cache 길이 48 확인
- kvpress 적용 후 KV Cache 길이가 48에서 24로 감소
- Transformers 4.57.6과 kvpress 0.5.4 조합에서 정상 동작 확인
- 로컬 RTX 5060 Ti 16GB에서는 약 200줄까지 실행 가능했고, 250줄 이상에서는 메모리 부족 발생

따라서 로컬 환경은 짧은 로그와 기능 테스트에는 사용할 수 있지만, 긴 로그와 7B~8B 모델을 이용한 공식 실험은 더 큰 GPU 서버에서 진행해야 한다.

## 파일 구성

- `README.md`, `setup.md`, `env_pin_note.md`: 환경과 실험 내용 설명
- `requirements-*.txt`: 설치된 패키지 버전 기록
- `kv_access_qwen_test.py`: KV Cache 확인 코드
- `kvpress_*_test.py`: 압축 전후 크기 비교 코드
- `profile_log_file_qwen.py`: 로그 길이별 메모리 측정 코드
- `results` 파일: 실제 실행 결과 기록