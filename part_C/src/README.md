# Part C 코드 안내

**[압축기 사용 가이드](LINE_KV_COMPRESSOR_GUIDE.md)** · **[PDF 보고서 목록](output/pdf/README.md)** · **[Part C 전체 안내](../README.md)**

## 현재 압축기와 실험

| 파일 | 역할 |
|---|---|
| [line_kv_compressor.py](line_kv_compressor.py) | 네 가지 점수로 라인을 선택하고 매핑된 토큰의 legacy K/V를 추출 |
| [LINE_KV_COMPRESSOR_GUIDE.md](LINE_KV_COMPRESSOR_GUIDE.md) | 실행 방법, 기본 가중치, 입력·출력, 테스트와 지원 범위 |
| [run_structural_retention.py](run_structural_retention.py) | Attention 없이 최근성·구조·식별자 조건을 비교하는 직전 실험 |
| [test/test_line_kv_compressor.py](test/test_line_kv_compressor.py) | 라인 선택, 예산, 라벨 불변성, K/V 내용과 순서 검증 |

압축기 실행 예시에서 사용하는 로컬 파일럿 데이터와 저장된 Attention은 이번 업로드에
포함되어 있지 않습니다. 자신의 입력 파일을 준비하고 가이드의 경로를 바꾸세요.

## 기존 분석 파이프라인

| 순서 | 코드 | 설명 |
|---|---|---|
| 1 | [window_extractor.py](window_extractor.py) | [윈도우 추출 안내](WINDOW_EXTRACTOR_EASY_GUIDE.md) |
| 2 | [token_mapper.py](token_mapper.py) | [토큰과 원본 라인 연결](TOKEN_MAPPER_EASY_GUIDE.md) |
| 3 | [attention.py](attention.py) | [모델 Attention 추출](ATTENTION_EASY_GUIDE.md) |
| 4 | [cross_line.py](cross_line.py) | [토큰 점수를 라인으로 집계](CROSS_LINE_EASY_GUIDE.md) |
| 5 | [anchor.py](anchor.py) | [Anchor 기준 이전 로그 순위](ANCHOR_EASY_GUIDE.md) |
| 진단 | [token_attention__probe.py](token_attention__probe.py) | [토큰·위치 효과 진단](TOKEN_ATTENTION_PROBE_EASY_GUIDE.md) |

[기존 팀원용 설명](TEAM_CODE_GUIDE.md)과 [기존 테스트 안내](TESTS_EASY_GUIDE.md)는
작성 시점의 파이프라인 설명입니다. 새 압축기는 별도의 가이드를 참고하세요.

## 디렉터리

```text
src/
  *.py                 실행 코드와 모듈
  *_GUIDE.md           기능별 설명
  test/                테스트
  output/pdf/          PDF 보고서와 보고서 목록
```

## 검증 범위

```powershell
python -m pytest test/test_line_kv_compressor.py -q
```

PyTorch와 pytest가 필요합니다. 현재 압축기는 배치 1의 legacy K/V 텐서를 지원합니다.
DynamicCache/연속 생성 어댑터와 압축 후 모델 품질 검증은 아직 포함하지 않습니다.
Alert 보존율과 모델 진단 성능은 같은 지표가 아닙니다.
