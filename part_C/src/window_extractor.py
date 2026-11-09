"""파싱된 Thunderbird CSV를 순차적으로 읽어 Window와 별도의 Alert 정답을 생성한다.

고정 키워드 규칙, 불명확한 라벨 처리 정책, 실행 방법은 WINDOW_EXTRACTOR.md를 참고한다.
Python 표준 라이브러리만 사용하며 모델은 로드하지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, deque
from contextlib import ExitStack
from pathlib import Path


# 기존 Notebook 파서가 생성하는 필수 CSV 필드.
CSV_FIELDS = (
    "line_id", "label", "timestamp", "date", "user", "month", "day",
    "time", "location", "component", "content",
)
# 정답 라벨은 모델용 Window에서 제외하고 Ground Truth 파일에만 보존한다.
MODEL_FIELDS = tuple(field for field in CSV_FIELDS if field != "label")
# 단어 경계를 사용하며 대소문자는 구분하지 않는다. ERROR가 WARN보다 우선한다.
ERROR_PATTERN = r"\b(?:errors?|fail(?:ed|ing|ure|ures)?|exceptions?)\b"
WARN_PATTERN = r"\bwarn(?:ing|ings)?\b"
ERROR_RE = re.compile(ERROR_PATTERN, re.IGNORECASE)
WARN_RE = re.compile(WARN_PATTERN, re.IGNORECASE)
NEGATED_RE = re.compile(
    r"\b(?:no|not|without)\s+(?:errors?|fail(?:ed|ing|ure|ures)?|exceptions?|warn(?:ing|ings)?)\b",
    re.IGNORECASE,
)
UNKNOWN_LABELS = {"", "nan", "null", "none", "n/a", "na", "unknown", "?"}
OUTPUT_NAMES = ("windows.jsonl", "alert_ground_truth.jsonl", "window_metadata.jsonl",
                "extraction_summary.json")


def classify_label(value: str | None) -> bool | None:
    """원본 라벨을 Alert(True), 정상(False), 불명확(None)으로 구분한다.

    양끝 공백을 제거한 뒤 누락값이나 잘못된 형식은 Alert로 분류하지 않는다.
    파서가 유효한 태그 목록을 제공하지 않으므로, 나머지 단일 토큰은
    원본 기준인 label != '-'에 따라 판별한다.
    """
    if value is None:
        return None
    label = value.strip()
    if label.casefold() in UNKNOWN_LABELS or any(c.isspace() for c in label):
        return None
    return label != "-"


def detect_level(content: str) -> dict:
    """content만으로 keyword_anchor_v1 규칙을 적용하고 판별 진단 정보를 반환한다.

    ERROR > WARN > INFO 순서이며 정답 라벨은 사용하지 않는다.
    실제 Severity를 보장하지 않는다. 부정 표현도 매칭을 유지하되 별도 표시한다.
    """
    error = bool(ERROR_RE.search(content))
    warn = bool(WARN_RE.search(content))
    return {
        "level": "ERROR" if error else "WARN" if warn else "INFO",
        "mixed_keywords": error and warn,
        "negated_keyword": bool(NEGATED_RE.search(content)),
        "empty_content": not content.strip(),
    }


def _write_jsonl(stream, value):
    """한 객체를 한글을 유지한 JSON 한 줄로 기록한다."""
    stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def _read_rows(stream, summary):
    """CSV를 한 행씩 검증하고 (로그 행, Alert 여부, 1-based 원본 행 위치)를 내보낸다.

    라벨과 키워드를 판별하면서 원본 행 단위 통계를 갱신한다.
    CSV 구조나 ID가 잘못되면 건너뛰지 않고 중단하여 로그 순서의 왜곡을 막는다.
    """
    reader = csv.DictReader(stream, strict=True)
    try:
        header = reader.fieldnames
        missing = set(CSV_FIELDS) - set(header or [])
        if missing:
            raise ValueError(f"Missing CSV fields: {', '.join(sorted(missing))}")
        if len(header) != len(set(header)):
            raise ValueError("Duplicate CSV column names")
        summary["csv_fields"] = header
        previous_id = None
        for row in reader:
            summary["total_processed_logs"] += 1
            if None in row or any(value is None for value in row.values()):
                raise ValueError("CSV row width does not match header")
            raw_id = row["line_id"].strip()
            if not re.fullmatch(r"[0-9]+", raw_id):
                raise ValueError(f"Invalid line_id: {raw_id!r}")
            line_id = int(raw_id)
            if previous_id is not None:
                if line_id <= previous_id:
                    raise ValueError("line_id must be unique and strictly increasing in source order")
                if line_id != previous_id + 1:
                    summary["line_id_gap_count"] += 1
            previous_id = line_id
            row["line_id"] = line_id
            alert = classify_label(row["label"])
            summary["unknown_label_count"] += int(alert is None)
            summary["source_alert_count"] += int(alert is True)
            diagnostic = detect_level(row["content"])
            stats = summary["anchor_detection"]
            stats["mixed_keyword_rows"] += int(diagnostic["mixed_keywords"])
            stats["negated_keyword_rows"] += int(diagnostic["negated_keyword"])
            stats["empty_content_rows"] += int(diagnostic["empty_content"])
            stats["no_keyword_rows"] += int(diagnostic["level"] == "INFO")
            if diagnostic["empty_content"]:
                stats["unclassifiable_rows"] += 1
            row["level"] = diagnostic["level"]
            yield row, alert, summary["total_processed_logs"]
    except (csv.Error, ValueError, UnicodeError) as exc:
        summary["csv_parse_error_count"] += 1
        raise ValueError(f"CSV physical line {reader.line_num}: {exc}") from exc


def _iter_windows(rows, size, stride, tail_policy):
    """0, stride, 2*stride, ... 위치에서 시작하는 Window를 하나씩 내보낸다.

    deque에는 최대 size개 행만 보관한다. 겹치는 행은 다음 Window에 재사용하며,
    파일 끝의 불완전 Window는 tail_policy가 include일 때만 내보낸다.
    """
    buffer = deque()
    start = 0
    for row in rows:
        position = row[2] - 1
        if position < start:  # stride가 size보다 크면 Window 사이의 행은 포함하지 않는다.
            continue
        buffer.append(row)
        if len(buffer) == size:
            yield list(buffer)
            start += stride
            while buffer and buffer[0][2] - 1 < start:
                buffer.popleft()
    if tail_policy == "include":
        while buffer:
            yield list(buffer)
            start += stride
            while buffer and buffer[0][2] - 1 < start:
                buffer.popleft()


def _build_records(window, window_id, source, size, stride, min_prior_lines,
                   min_anchors, previous_end):
    """한 Window를 모델용 로그 배열, Anchor별 정답, Window 메타데이터로 나눈다.

    previous_end는 이전 출력 Window까지 포함된 원본 행 위치의 최댓값이다.
    이를 이용해 겹치는 Window에서 로그와 Anchor의 중복 출현을 표시한다.
    """
    model_rows = []
    anchors = []
    candidates, alerts, unknown = [], [], []
    labels = []
    repeated_anchors = 0
    for position, (row, is_alert, source_position) in enumerate(window, 1):
        line_id = row["line_id"]
        # 허용한 필드만 복사하여 is_anomaly, evidence_role 등의 정답 유출을 막는다.
        # 파생 level은 token_mapper에서 메타데이터로 전달되며 입력 텍스트에는 들어가지 않는다.
        model_rows.append({**{key: row[key] for key in MODEL_FIELDS}, "level": row["level"]})
        labels.append({"line_id": line_id, "label": row["label"], "is_alert": is_alert})
        if row["level"] in ("WARN", "ERROR"):
            # 순위 생성 가능 여부와 정답 기반 평가 가능 여부를 별도로 판단한다.
            reason = ("no_prior_lines" if not candidates else
                      "insufficient_prior_lines" if len(candidates) < min_prior_lines else None)
            evaluation_reason = reason or ("unknown_candidate_labels" if unknown else
                                           "no_prior_alerts" if not alerts else None)
            repeated = source_position <= previous_end
            repeated_anchors += int(repeated)
            anchors.append({
                "anchor_line_id": line_id, "anchor_position": position,
                "anchor_level": row["level"], "anchor_type": "keyword_anchor",
                "candidate_line_ids": candidates.copy(), "candidate_count": len(candidates),
                "alert_line_ids": alerts.copy(), "alert_count": len(alerts),
                "unknown_label_line_ids": unknown.copy(),
                "ranking_eligible": reason is None, "exclusion_reason": reason,
                "evaluation_eligible": evaluation_reason is None,
                "evaluation_exclusion_reason": evaluation_reason,
                "repeated_from_previous_windows": repeated,
            })
        # 현재 Anchor를 기록한 다음 현재 행을 추가해야 자기 자신이 후보에 포함되지 않는다.
        # 이 행은 이후 Anchor에서는 정상적인 이전 후보가 된다.
        candidates.append(line_id)
        if is_alert is True:
            alerts.append(line_id)
        elif is_alert is None:
            unknown.append(line_id)
    gt = {"window_id": window_id, "anchors": anchors, "line_labels": labels}
    # Anchor별 전체 후보 및 정답 배열은 Ground Truth 파일에만 기록한다.
    metadata = {
        "window_id": window_id, "source_file": str(source),
        "start_line_id": window[0][0]["line_id"], "end_line_id": window[-1][0]["line_id"],
        "source_start_position": window[0][2], "source_end_position": window[-1][2],
        "window_size": len(window), "configured_window_size": size, "stride": stride,
        "is_partial": len(window) < size, "anchor_count": len(anchors),
        "anchors": [{key: a[key] for key in (
            "anchor_line_id", "anchor_position", "anchor_level", "anchor_type",
            "ranking_eligible", "evaluation_eligible", "repeated_from_previous_windows",
        )} for a in anchors],
        "has_alert": bool(alerts), "has_unknown_labels": bool(unknown),
        "has_eligible_anchor": any(a["ranking_eligible"] for a in anchors),
        "has_evaluation_eligible_anchor": any(a["evaluation_eligible"] for a in anchors),
        "meets_min_anchors": len(anchors) >= min_anchors,
        "repeated_line_count": sum(item[2] <= previous_end for item in window),
        "repeated_anchor_count": repeated_anchors,
    }
    return model_rows, gt, metadata


def extract_windows(input_path, output_dir, *, window_size=100, stride=None,
                    min_prior_lines=1, min_anchors=0, max_windows=None,
                    seed=2026, tail_policy="include") -> dict:
    """추출 전체 과정을 실행하여 파일 4개를 생성하고 최종 통계 딕셔너리를 반환한다.

    기존 파일은 덮어쓰지 않는다. 처리 중 실패하면 요약 파일에 실패 상태를 기록한다.
    min_anchors는 조건 충족 여부만 표시하며 Window를 제거하지 않는다.
    무작위 표본 추출은 하지 않으므로 seed는 설정 기록에만 사용한다.
    """
    stride = window_size if stride is None else stride
    if window_size < 1 or stride < 1:
        raise ValueError("window_size and stride must be positive")
    if min_prior_lines < 0 or min_anchors < 0:
        raise ValueError("minimum counts must be nonnegative")
    if max_windows is not None and max_windows < 1:
        raise ValueError("max_windows must be positive")
    if tail_policy not in ("include", "drop"):
        raise ValueError("tail_policy must be include or drop")
    source, dest = Path(input_path).resolve(), Path(output_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    paths = [dest / name for name in OUTPUT_NAMES]
    for path in paths:
        if path == source or path.exists():
            raise FileExistsError(f"Output already exists or is the source file: {path}")
    dest.mkdir(parents=True, exist_ok=True)
    summary = dict.fromkeys((
        "total_processed_logs", "total_windows", "windows_without_anchors",
        "windows_with_one_anchor", "windows_with_multiple_anchors", "error_anchor_count",
        "warn_anchor_count", "ranking_eligible_anchor_count", "evaluation_eligible_anchor_count",
        "anchors_with_prior_alerts", "anchors_without_prior_alerts", "insufficient_candidate_anchor_count",
        "unknown_label_count", "source_alert_count", "csv_parse_error_count", "line_id_gap_count",
        "repeated_line_occurrences", "repeated_anchor_occurrences", "unique_anchor_count",
        "unique_windowed_log_count", "partial_window_count", "windows_meeting_min_anchors",
    ), 0)
    summary.update({
        "status": "running", "source_file": str(source), "stopped_at_max_windows": False,
        "config": {"window_size": window_size, "stride": stride, "min_prior_lines": min_prior_lines,
                   "min_anchors": min_anchors, "max_windows": max_windows, "seed": seed,
                   "tail_policy": tail_policy, "min_anchors_action": "annotate_only",
                   "sampling": "none"},
        "anchor_rule": {"version": "keyword_anchor_v1", "type": "keyword_anchor",
                        "error_pattern": ERROR_PATTERN, "warn_pattern": WARN_PATTERN,
                        "case_sensitive": False, "priority": ["ERROR", "WARN", "INFO"],
                        "negation_policy": "retain_match_and_report"},
        "anchor_detection": dict.fromkeys(("mixed_keyword_rows", "negated_keyword_rows",
                                           "empty_content_rows", "no_keyword_rows", "unclassifiable_rows"), 0),
        "alert_position_distribution": {},
        "counting_unit": "window/anchor occurrences unless named source or unique",
        "independence_note": "Anchors within a window and overlapping windows are not independent incidents.",
        "label_policy": "strip; '-'=non-alert; missing/sentinel/whitespace-containing=unknown; other tokens=alert",
    })
    positions = Counter()
    previous_end = 0
    # 긴 로그 content를 읽을 수 있도록 CSV 필드 길이 제한을 임시로 늘린다.
    old_limit = csv.field_size_limit()
    csv.field_size_limit(max(old_limit, 2**31 - 1))
    try:
        with ExitStack() as stack:
            # 여러 파일을 함께 관리하여 정상 종료와 예외 발생 모두에서 파일을 닫는다.
            fin = stack.enter_context(source.open(encoding="utf-8-sig", newline=""))
            windows_out, gt_out, meta_out, summary_out = [
                stack.enter_context(path.open("x", encoding="utf-8", newline="")) for path in paths
            ]
            try:
                rows = _read_rows(fin, summary)
                for window in _iter_windows(rows, window_size, stride, tail_policy):
                    # 한 Window를 즉시 기록한다. 전체 Window 목록을 메모리에 쌓지 않는다.
                    model_rows, gt, meta = _build_records(
                        window, summary["total_windows"], source, window_size, stride,
                        min_prior_lines, min_anchors, previous_end,
                    )
                    _write_jsonl(windows_out, model_rows)
                    _write_jsonl(gt_out, gt)
                    _write_jsonl(meta_out, meta)
                    summary["total_windows"] += 1
                    # 아래는 Window 출현 단위 통계다. 겹침을 제외한 수는 unique_*에 기록한다.
                    count = meta["anchor_count"]
                    category = ("windows_without_anchors" if count == 0 else
                                "windows_with_one_anchor" if count == 1 else "windows_with_multiple_anchors")
                    summary[category] += 1
                    summary["partial_window_count"] += int(meta["is_partial"])
                    summary["windows_meeting_min_anchors"] += int(meta["meets_min_anchors"])
                    summary["repeated_line_occurrences"] += meta["repeated_line_count"]
                    summary["repeated_anchor_occurrences"] += meta["repeated_anchor_count"]
                    summary["unique_anchor_count"] += count - meta["repeated_anchor_count"]
                    summary["unique_windowed_log_count"] += len(window) - meta["repeated_line_count"]
                    for a in gt["anchors"]:
                        summary[f"{a['anchor_level'].lower()}_anchor_count"] += 1
                        summary["ranking_eligible_anchor_count"] += int(a["ranking_eligible"])
                        summary["evaluation_eligible_anchor_count"] += int(a["evaluation_eligible"])
                        summary["anchors_with_prior_alerts"] += int(a["alert_count"] > 0)
                        summary["anchors_without_prior_alerts"] += int(a["alert_count"] == 0)
                        summary["insufficient_candidate_anchor_count"] += int(not a["ranking_eligible"])
                    for position, (_, alert, _) in enumerate(window, 1):
                        if alert is True:
                            positions[str(position)] += 1
                    previous_end = window[-1][2]
                    if max_windows is not None and summary["total_windows"] >= max_windows:
                        summary["stopped_at_max_windows"] = True
                        break
                summary["status"] = "complete"
            except Exception as exc:
                summary["status"] = "failed"
                summary["error"] = str(exc)
                raise
            finally:
                # 처리 중 오류가 나도 현재까지의 통계와 실행 상태를 남긴다.
                summary["alert_position_distribution"] = dict(positions)
                summary["unwindowed_log_count"] = summary["total_processed_logs"] - summary["unique_windowed_log_count"]
                summary_out.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    finally:
        csv.field_size_limit(old_limit)
    return summary


def main(argv=None):
    """명령행 옵션을 읽고 추출기를 실행한 뒤 요약 통계를 화면에 출력한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="파싱된 Thunderbird CSV 경로 (UTF-8/BOM 허용)")
    parser.add_argument("--output-dir", type=Path, required=True, help="동일한 출력 파일이 아직 없는 저장 폴더")
    parser.add_argument("--window-size", type=int, default=100, help="Window 최대 행 수 (기본 100)")
    parser.add_argument("--stride", type=int, default=None, help="Window 시작점 간격 (기본 window-size)")
    parser.add_argument("--min-prior-lines", type=int, default=1, help="Anchor 순위 생성에 필요한 최소 이전 후보 수 (기본 1)")
    parser.add_argument("--min-anchors", type=int, default=0, help="최소 Anchor 수 충족 여부만 표시하며 Window는 제거하지 않음")
    parser.add_argument("--max-windows", type=int, default=None, help="지정한 Window 수를 저장하면 중단 (개발용)")
    parser.add_argument("--seed", type=int, default=2026, help="설정에만 기록하며 무작위 표본 추출은 하지 않음")
    parser.add_argument("--tail-policy", choices=("include", "drop"), default="include", help="마지막 불완전 Window를 보존(include)하거나 제외(drop)")
    args = parser.parse_args(argv)
    try:
        summary = extract_windows(args.input, args.output_dir, window_size=args.window_size,
                                  stride=args.stride, min_prior_lines=args.min_prior_lines,
                                  min_anchors=args.min_anchors, max_windows=args.max_windows,
                                  seed=args.seed, tail_policy=args.tail_policy)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Extraction failed: {exc}\n")
    # Windows의 기존 콘솔 인코딩에서도 출력 리다이렉션이 가능하도록 ASCII로 이스케이프한다.
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
