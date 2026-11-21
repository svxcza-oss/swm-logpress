"""Part D position formula + Part C line/KV integration regression tests."""
import csv
import json
from pathlib import Path

import pytest
import torch
from line_kv_compressor_position import (
    build_position_rows, select_position_lines, load_positions,
    matched_auc, prepare_positions,
)
from line_kv_compressor import compress_kv


def _rows():
    # The original full-hour group has five logs, but our chunk contains three.
    return [
        {'line_id': str(lid), 'node': 'n1', 'timestamp': str(3600 + minute), 'label': 'ALERT' if lid==20 else '-'}
        for minute, lid in enumerate((10, 20, 30, 40, 50))
    ]


def _tokens():
    lids = [30, 30, 40, 40, 50, 50]
    return [{'token_idx': i, 'line_id': lid} for i, lid in enumerate(lids)]


def test_original_hour_positions_unchanged_by_chunking():
    rows = build_position_rows(_rows())
    pos = {r['line_id']: r for r in rows}
    assert [pos[l]['position'] for l in (10,20,30,40,50)] == [0,.25,.5,.75,1]
    assert all(pos[l]['n'] == 5 for l in pos)
    choice = select_position_lines(_tokens(), pos, keep_ratio=2/3)
    assert choice.selected_line_ids == [30,40]
    assert choice.kept_token_indices == [0,1,2,3]
    assert choice.used_tokens == 4
    assert choice.protected_line_ids == []
    assert choice.scores[30]['position_score'] == .5
    assert choice.scores[40]['position_score'] == .25


def test_back_direction_and_label_independence():
    rows = build_position_rows(_rows())
    positions = {r['line_id']: r for r in rows}
    front = select_position_lines(_tokens(), positions, keep_ratio=2/3)
    back = select_position_lines(_tokens(), positions, keep_ratio=2/3, direction='back')
    assert back.selected_line_ids == [40,50]
    for r in _rows():
        r['label'] = 'poison'
    assert front == select_position_lines(_tokens(), positions, keep_ratio=2/3)


def test_exact_tensor_indexing_and_no_mutation():
    selection = select_position_lines(_tokens(), {r['line_id']:r for r in build_position_rows(_rows())}, keep_ratio=2/3)
    kv = tuple((torch.arange(h*6*3).view(1,h,6,3).float(),
                torch.arange(h*6*3).view(1,h,6,3).float()+100) for h in (1,2))
    out = compress_kv(kv, selection)
    for (k, v), (ck, cv) in zip(kv, out):
        assert torch.equal(ck, k[:,:,selection.kept_token_indices])
        assert torch.equal(cv, v[:,:,selection.kept_token_indices])
        assert ck.shape[2] == 4 and k.shape[2] == 6


def test_n1_neutral_but_auc_excludes():
    rows = build_position_rows([{'line_id': '1','node':'solo','timestamp':'5'}])
    assert rows[0]['position'] is None
    selected = select_position_lines([{'token_idx':0,'line_id':1}], {1:rows[0]}, keep_ratio=1.0)
    assert selected.scores[1]['position_score'] == .5


def test_auc_is_evaluation_only_and_reverses():
    rows = build_position_rows(_rows())
    positions = {r['line_id']: r for r in rows}
    labels = {10:1,20:1,30:0,40:0,50:0}
    result = matched_auc(positions, labels, ratio=1, seed=42)
    assert result['n_positive']==2 and result['n_negative']==2
    assert result['auc_P'] < .5
    assert result['auc_1_minus_P'] == pytest.approx(1-result['auc_P'])
    # AUC inputs never enter line selection.
    assert select_position_lines(_tokens(), positions,keep_ratio=2/3) == select_position_lines(_tokens(), positions,keep_ratio=2/3)


def test_positions_csv_round_trip(tmp_path):
    origin = tmp_path/'whole.csv'; output = tmp_path/'positions.csv'
    with origin.open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['line_id','node','timestamp','label'])
        writer.writeheader();writer.writerows(_rows())
    prepare_positions(origin,output)
    pos = load_positions(output)
    assert pos[40]['position'] == .75


def test_reject_missing_parent_position():
    rows = build_position_rows(_rows())
    positions = {r['line_id']:r for r in rows}
    del positions[40]
    with pytest.raises(ValueError, match='누락'):
        select_position_lines(_tokens(), positions)
