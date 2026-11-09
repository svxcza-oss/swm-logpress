import csv
import json
from pathlib import Path

import pytest

import window_extractor as we


def make_csv(tmp_path, rows):
    path = tmp_path / "한글 입력.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=we.CSV_FIELDS)
        writer.writeheader()
        for i, changes in enumerate(rows):
            row = dict.fromkeys(we.CSV_FIELDS, "")
            row.update(line_id=i, label="-", timestamp="123", content="normal")
            row.update(changes)
            writer.writerow(row)
    return path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def run(tmp_path, rows, **options):
    source = make_csv(tmp_path, rows)
    output = tmp_path / "결과 폴더"
    summary = we.extract_windows(source, output, **options)
    return (read_jsonl(output / "windows.jsonl"),
            read_jsonl(output / "alert_ground_truth.jsonl"),
            read_jsonl(output / "window_metadata.jsonl"), summary)


@pytest.mark.parametrize("size", [10, 50, 100])
def test_sizes_order_and_default_stride(tmp_path, size):
    windows, gt, meta, summary = run(tmp_path, [{} for _ in range(size * 2)], window_size=size)
    assert [len(w) for w in windows] == [size, size]
    assert [r["line_id"] for w in windows for r in w] == list(range(size * 2))
    assert [g["window_id"] for g in gt] == [0, 1]
    assert meta[0]["stride"] == size
    assert summary["windows_without_anchors"] == 2
    assert summary["total_processed_logs"] == size * 2


@pytest.mark.parametrize("content,level", [
    ("WARN queue", "WARN"), ("warning: slow", "WARN"),
    ("ERROR disk", "ERROR"), ("failed to open", "ERROR"),
    ("failure", "ERROR"), ("Exception raised", "ERROR"),
    ("warn then error", "ERROR"), ("normal", "INFO"),
    ("errorless failover prewarning", "INFO"), ("", "INFO"),
])
def test_fixed_keyword_rules(content, level):
    assert we.detect_level(content)["level"] == level


@pytest.mark.parametrize("label,expected", [
    ("-", False), (" - ", False), ("ALERT_X", True), (" ALERT_X ", True),
    ("", None), ("  ", None), (None, None), ("NaN", None),
    ("null", None), ("bad label", None), ("?", None),
])
def test_label_policy(label, expected):
    assert we.classify_label(label) is expected


def test_multiple_anchors_positions_candidates_and_alerts(tmp_path):
    windows, gt, meta, summary = run(tmp_path, [
        {"line_id": 10, "content": "WARN first", "label": "FIRST_ALERT"},
        {"line_id": 20, "label": "OTHER_ALERT"},
        {"line_id": 30, "content": "ERROR middle"},
        {"line_id": 40, "label": "FUTURE_ALERT"},
        {"line_id": 50, "content": "WARNING last"},
    ], window_size=5)
    first, middle, last = gt[0]["anchors"]
    assert [a["anchor_position"] for a in (first, middle, last)] == [1, 3, 5]
    assert first["candidate_line_ids"] == []
    assert not first["ranking_eligible"]
    assert first["exclusion_reason"] == "no_prior_lines"
    assert middle["candidate_line_ids"] == [10, 20]
    assert middle["alert_line_ids"] == [10, 20]
    assert middle["candidate_count"] == middle["alert_count"] == 2
    assert last["candidate_line_ids"] == [10, 20, 30, 40]
    assert last["alert_line_ids"] == [10, 20, 40]
    assert meta[0]["anchor_count"] == 3
    assert meta[0]["start_line_id"] == 10
    assert summary["windows_with_multiple_anchors"] == 1
    assert summary["warn_anchor_count"] == 2
    assert summary["error_anchor_count"] == 1
    assert summary["alert_position_distribution"] == {"1": 1, "2": 1, "4": 1}
    assert all("label" not in row and "is_anomaly" not in row for row in windows[0])


def test_no_truth_and_minimum_candidates_are_separate(tmp_path):
    _, gt, _, summary = run(tmp_path, [{}, {"content": "ERROR"}, {}, {"content": "WARN"}],
                            window_size=4, min_prior_lines=2)
    early, late = gt[0]["anchors"]
    assert not early["ranking_eligible"]
    assert early["exclusion_reason"] == "insufficient_prior_lines"
    assert late["ranking_eligible"]
    assert not late["evaluation_eligible"]
    assert late["evaluation_exclusion_reason"] == "no_prior_alerts"
    assert late["alert_line_ids"] == []
    assert summary["anchors_without_prior_alerts"] == 2


def test_unknown_labels_never_become_alerts(tmp_path):
    _, gt, _, summary = run(tmp_path, [{"label": ""}, {"label": "REAL"}, {"content": "ERROR"}], window_size=3)
    anchor = gt[0]["anchors"][0]
    assert anchor["alert_line_ids"] == [1]
    assert anchor["unknown_label_line_ids"] == [0]
    assert anchor["ranking_eligible"] and not anchor["evaluation_eligible"]
    assert anchor["evaluation_exclusion_reason"] == "unknown_candidate_labels"
    assert summary["unknown_label_count"] == 1


def test_overlap_tracks_duplicate_lines_and_anchors(tmp_path):
    windows, gt, meta, summary = run(tmp_path, [{}, {}, {"content": "WARN"}, {}, {}], window_size=3, stride=2)
    assert [[r["line_id"] for r in w] for w in windows] == [[0, 1, 2], [2, 3, 4], [4]]
    assert gt[0]["anchors"][0]["anchor_line_id"] == gt[1]["anchors"][0]["anchor_line_id"] == 2
    assert meta[1]["repeated_line_count"] == 1
    assert meta[1]["repeated_anchor_count"] == 1
    assert summary["repeated_line_occurrences"] == 2
    assert summary["repeated_anchor_occurrences"] == 1
    assert summary["unique_anchor_count"] == 1


@pytest.mark.parametrize("tail,expected", [("include", [3, 2]), ("drop", [3])])
def test_partial_tail(tmp_path, tail, expected):
    windows, _, _, summary = run(tmp_path, [{}] * 5, window_size=3, tail_policy=tail)
    assert [len(w) for w in windows] == expected
    assert summary["total_processed_logs"] == 5
    assert summary["unwindowed_log_count"] == (2 if tail == "drop" else 0)


def test_empty_input(tmp_path):
    windows, gt, meta, summary = run(tmp_path, [])
    assert windows == gt == meta == []
    assert summary["total_windows"] == 0


def test_min_anchors_annotates_without_selection_bias(tmp_path):
    windows, _, meta, summary = run(tmp_path, [{}, {}, {"content": "WARN"}, {}], window_size=2, min_anchors=2)
    assert len(windows) == 2
    assert all(not m["meets_min_anchors"] for m in meta)
    assert summary["windows_with_one_anchor"] == 1


def test_max_windows_stops_reading_and_seed_does_not_sample(tmp_path):
    windows, _, _, summary = run(tmp_path, [{}] * 20, window_size=3, max_windows=2, seed=42)
    assert len(windows) == 2
    assert summary["total_processed_logs"] == 6
    assert summary["stopped_at_max_windows"]


def test_stride_gaps_are_reported(tmp_path):
    windows, _, _, summary = run(tmp_path, [{}] * 8, window_size=2, stride=3)
    assert [[r["line_id"] for r in w] for w in windows] == [[0, 1], [3, 4], [6, 7]]
    assert summary["unwindowed_log_count"] == 2


def test_diagnostics_for_ambiguous_and_empty_content(tmp_path):
    _, _, _, summary = run(tmp_path, [{"content": "WARN error"}, {"content": "no errors"}, {"content": ""}], window_size=3)
    assert summary["anchor_detection"]["mixed_keyword_rows"] == 1
    assert summary["anchor_detection"]["negated_keyword_rows"] == 1
    assert summary["anchor_detection"]["empty_content_rows"] == 1


@pytest.mark.parametrize("options", [{"window_size": 0}, {"stride": 0}, {"min_prior_lines": -1},
                                      {"min_anchors": -1}, {"max_windows": 0}, {"tail_policy": "bad"}])
def test_invalid_options(tmp_path, options):
    with pytest.raises(ValueError):
        run(tmp_path, [{}], **options)


@pytest.mark.parametrize("bad_id", ["bad", "0", "-1"])
def test_bad_ids_fail_without_silently_rejoining_logs(tmp_path, bad_id):
    source = make_csv(tmp_path, [{}, {"line_id": bad_id}])
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        we.extract_windows(source, out)
    summary = json.loads((out / "extraction_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["csv_parse_error_count"] == 1


def test_malformed_csv_and_missing_header(tmp_path):
    source = tmp_path / "bad.csv"
    source.write_text("line_id,content\n0,normal\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing CSV fields"):
        we.extract_windows(source, tmp_path / "out")


def test_source_is_never_overwritten(tmp_path):
    source = make_csv(tmp_path, [{}])
    original = source.read_bytes()
    with pytest.raises(FileExistsError):
        we.extract_windows(source, tmp_path)
        we.extract_windows(source, tmp_path)
    assert source.read_bytes() == original


def test_actual_token_mapper_main_compatibility_and_label_nonleakage(tmp_path, monkeypatch):
    import token_mapper as tm
    from test_token_mapping import FakeTokenizer

    source = make_csv(tmp_path, [{"label": "SECRET_ALERT_TAG"}, {"content": "WARN disk"}])
    out = tmp_path / "out"
    we.extract_windows(source, out, window_size=2)
    monkeypatch.setattr(tm.AutoTokenizer, "from_pretrained", lambda *a, **k: FakeTokenizer())
    monkeypatch.setattr("builtins.input", lambda _: str(out / "windows.jsonl"))
    tm.main()
    mapped = read_jsonl(out / "windows_token_map.jsonl")[0]
    assert "SECRET_ALERT_TAG" not in json.dumps(mapped)
    assert mapped["line_meta"][1]["level"] == "WARN"
    assert mapped["window_idx"] == 0
    assert {t["line_id"] for t in mapped["token_map"]} == {0, 1}


def test_cli(tmp_path):
    import subprocess
    import sys
    source = make_csv(tmp_path, [{}, {"content": "WARN"}])
    result = subprocess.run([sys.executable, str(Path(we.__file__)), "--input", str(source),
                             "--output-dir", str(tmp_path / "cli 결과"), "--window-size", "2"],
                            capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["total_windows"] == 1


@pytest.mark.parametrize("tail", ["include", "drop"])
def test_streaming_windows_match_reference_slices(tail):
    # Includes size=1, overlap tails, stride gaps, and exact EOF boundaries.
    for length in range(15):
        rows = [({}, False, i + 1) for i in range(length)]
        for size in range(1, 7):
            for stride in range(1, 8):
                expected = [rows[i:i + size] for i in range(0, length, stride)
                            if tail == "include" or i + size <= length]
                assert list(we._iter_windows(iter(rows), size, stride, tail)) == expected


@pytest.mark.parametrize("bad_record", ["0,-\n", "0,-,1,d,u,m,d,t,l,c,x,extra\n", '0,-,1,d,u,m,d,t,l,c,"unclosed'])
def test_bad_csv_records_report_error(tmp_path, bad_record):
    source = tmp_path / "bad.csv"
    source.write_text(",".join(we.CSV_FIELDS) + "\n" + bad_record, encoding="utf-8")
    with pytest.raises(ValueError, match="CSV physical line"):
        we.extract_windows(source, tmp_path / "out")
    summary = json.loads((tmp_path / "out" / "extraction_summary.json").read_text(encoding="utf-8"))
    assert summary["csv_parse_error_count"] == 1


def test_quoted_multiline_content_and_extra_gt_columns(tmp_path):
    source = tmp_path / "extra.csv"
    fields = (*we.CSV_FIELDS, "is_anomaly", "evidence_role", "level")
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        row = dict.fromkeys(fields, "")
        row.update(line_id=0, label="-", content='normal, "quoted"\nsecond line',
                   is_anomaly="true", evidence_role="root", level="ERROR")
        writer.writerow(row)
    out = tmp_path / "out"
    summary = we.extract_windows(source, out)
    model_row = read_jsonl(out / "windows.jsonl")[0][0]
    assert model_row["content"] == row["content"]
    assert model_row["level"] == "INFO"
    assert "is_anomaly" not in model_row and "evidence_role" not in model_row
    assert summary["total_processed_logs"] == 1
