# LogPress

시스템 로그의 중요 라인을 분석하고 KV Cache 보존 전략을 연구하는 프로젝트입니다.

## 바로 찾기

- **[Part C 시작하기](part_C/README.md)**: 코드, 가이드, 실험 보고서 안내
- **[현재 라인 단위 압축기](part_C/src/line_kv_compressor.py)** · [사용 방법과 검증 범위](part_C/src/LINE_KV_COMPRESSOR_GUIDE.md)
- **[압축 방법과 선택 근거 PDF](part_C/src/output/pdf/LogPress_압축방법과_선택근거_20261117.pdf)**
- **[Part C 보고서 목록](part_C/src/output/pdf/README.md)**: 주제와 작성 시점별 안내

## 저장소 구성

| 폴더 | 내용 |
|---|---|
| [Part_A](Part_A/) | 압축 baseline 구현과 주차별 실험 |
| [Part_B](Part_B/) | Part B 코드 및 자료 |
| [part_C](part_C/) | 로그 라인 매핑, Attention 분석, 라인 단위 압축기 |
| [part_D(select,review)](part_D%28select%2Creview%29/) | Part D 선택·검토 자료 |
| [문서](문서/) | 공통 프로젝트 문서 |
| [주간 보고서](주간%20보고서/) | 주차별 진행 보고 |
| [회의록·피드백](피드백%2C%20%20회의록%20기록/) | 회의 기록과 피드백 |

## 자료를 읽을 때

Part C의 현재 압축기는 라인 선택과 legacy K/V 텐서 압축을 구현한 연구용 코드입니다.
모델의 연속 생성 연결과 압축 후 생성 품질은 아직 검증되지 않았습니다.
이전 PDF는 작성 당시 기록이며, 현재 지원 범위는 최신 코드와 해당 가이드를 확인하세요.
