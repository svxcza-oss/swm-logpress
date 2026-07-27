# A Track Week 0 작업 메모

## 목적

Week 0에서는 본격적인 실험 전에, 내 로컬 GPU에서 모델과 KV 압축 도구가 실제로 동작하는지 확인했다.

KV Cache는 모델이 앞에서 읽은 내용을 임시로 저장하는 공간이다. 이번 프로젝트는 이 저장 공간에서 덜 중요한 부분을 줄이면서, 장애 원인이 담긴 로그는 남기는 것이 목표다.

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