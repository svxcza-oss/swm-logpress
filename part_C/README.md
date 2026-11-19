# Part C · 라인 매핑과 압축 연구

로그 전처리 → Token/Line 매핑 → Attention 분석 → 보존 라인 선택 → K/V 추출을 다룹니다.

## 처음 보는 팀원에게

1. **[압축 방법과 선택 근거 PDF](src/output/pdf/LogPress_압축방법과_선택근거_20261117.pdf)**: 무엇을 만들었고 왜 이 방식을 택했는지
2. **[코드 목록](src/README.md)**: 파일별 역할과 실행 흐름
3. **[압축기 사용 가이드](src/LINE_KV_COMPRESSOR_GUIDE.md)**: 입력 형식, 실행 예시, 검증 결과
4. **[보고서 목록](src/output/pdf/README.md)**: 현재 설명과 과거 실험 기록 구분

## 자료 위치

| 위치 | 내용 |
|---|---|
| [src/](src/) | 실행 코드와 모듈별 가이드 |
| [src/test/](src/test/) | 기능 검증 테스트 |
| [src/output/pdf/](src/output/pdf/) | 실험 보고서와 압축 방식 설명 |
| [전체 실험 개요](LogPress_전체_실험_개요_WARN_ERROR_Anchor.pdf) | 기존 WARN/ERROR Anchor 실험 개요 |
| [평가 설계](WARN_ERROR_Query_Anchor_평가설계.pdf) | 기존 Query Anchor 평가 설계 |

현재 압축기는 `최근성 + 호스트·컴포넌트 + 코드·식별자 + Anchor Attention`으로
완전한 줄을 선택합니다. 예산은 토큰 수 기준이며 Alert 라벨은 평가에만 사용합니다.
실제 모델의 압축 후 생성 성능까지 검증된 완성형 파이프라인은 아닙니다.

로컬 실험의 CSV·Attention 텐서·일부 후속 코드는 저장소에 모두 포함되어 있지 않습니다.
보고서의 실행 이력과 현재 저장소에 존재하는 파일을 구분해서 확인하세요.
