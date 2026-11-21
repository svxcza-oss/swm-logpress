"""LogPress Part C × Part D position integration (research prototype).

This module reuses Part C's Selection/compress_kv.  It brings in the Part D
rank/(n-1) position feature and optional *offline* matched ROC-AUC diagnostics.

Pipeline:
  prepare: full parent-window CSV -> node × fixed-clock-hour positions CSV
  select:  32K chunk token_map + positions CSV -> selected complete lines / tokens
  evaluate: positions CSV + separate gold labels CSV -> AUC (never a feature)

Only the legacy ((K,V),...) format is supported for cache extraction.  No
transformers DynamicCache or model generation adapter is implemented here.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from line_kv_compressor import Selection, compress_kv


def _epoch_seconds(raw: str) -> float:
    """Thunderbird parsed CSV's numeric UNIX timestamp, seconds (or milliseconds)."""
    try:
        t = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"숫자형 UNIX timestamp가 아닙니다: {raw!r}") from None
    if not math.isfinite(t):
        raise ValueError("timestamp는 유한한 값이어야 합니다.")
    if abs(t) >= 1e11:  # allow common millisecond UNIX timestamps
        t /= 1000.0
    return t


def build_position_rows(rows: list[dict]) -> list[dict]:
    """Part D: node + fixed-clock 1h, rank / (N-1).

    `rows` MUST cover the original parent 1h window, not only a 32K chunk.
    Node values come from `node` or Thunderbird `user`.  label/is_hardcase is
    not read.  The original line_id breaks equal-timestamp ties stably.

    N==1: mark position as None; Part D excluded these from its AUC.
    """
    groups: dict[tuple[str, str], list[tuple[float, int]]] = defaultdict(list)
    seen: set[int] = set()
    for row in rows:
        line_id = int(row['line_id'])
        if line_id in seen:
            raise ValueError(f"중복 line_id: {line_id}")
        seen.add(line_id)
        node = str(row.get('node') or row.get('user') or '').strip()
        if not node:
            raise ValueError(f"node/user가 없는 line_id: {line_id}")
        timestamp = _epoch_seconds(row['timestamp'])
        # Explicit grid can be supplied by Part D preprocessing.  Otherwise
        # use an absolute clock-aligned UNIX 1h grid (not a rolling 1h window).
        grid = str(row.get('grid_id_1h') or int(math.floor(timestamp / 3600)))
        groups[(node, grid)].append((timestamp, line_id))
    result: list[dict] = []
    for (node, grid), entries in sorted(groups.items()):
        entries.sort(key=lambda x: (x[0], x[1]))
        n = len(entries)
        for rank, (_, line_id) in enumerate(entries):
            result.append({
                'line_id': line_id, 'node': node, 'grid_id_1h': grid,
                'rank': rank, 'n': n,
                'position': rank / (n - 1) if n > 1 else None,
            })
    return sorted(result, key=lambda row: row['line_id'])


def prepare_positions(source_csv: str | Path, output_csv: str | Path) -> list[dict]:
    with Path(source_csv).open(newline='', encoding='utf-8-sig') as f:
        rows = build_position_rows(list(csv.DictReader(f)))
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['line_id', 'node', 'grid_id_1h', 'rank', 'n', 'position'])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def load_positions(positions_csv: str | Path) -> dict[int, dict]:
    lookup: dict[int, dict] = {}
    with Path(positions_csv).open(newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            line_id = int(row['line_id'])
            if line_id in lookup:
                raise ValueError(f'중복 위치 line_id: {line_id}')
            n = int(row['n'])
            value = None if row['position'] in ('', None) else float(row['position'])
            if n < 1 or (value is not None and (not math.isfinite(value) or not 0 <= value <= 1)):
                raise ValueError(f'잘못된 position: {line_id}')
            lookup[line_id] = {
                'line_id': line_id, 'node': row['node'], 'grid_id_1h': row['grid_id_1h'],
                'rank': int(row['rank']), 'n': n, 'position': value,
            }
    return lookup


def select_position_lines(token_map: list[dict], positions: dict[int, dict], *,
                          keep_ratio: float = 0.5, direction: str = 'front') -> Selection:
    """Line-granularity budgeted selection, anchor-free.

    front: score = 1-P (new policy proposed from Part D's AUC < 0.5)
    back:  score = P (control experiment)
    Neither score is a formula Part D finalized for compression.
    """
    if direction not in ('front', 'back'):
        raise ValueError('direction은 front 또는 back이어야 합니다.')
    if not math.isfinite(keep_ratio) or not 0 < keep_ratio <= 1:
        raise ValueError('keep_ratio는 (0, 1] 범위여야 합니다.')
    groups: dict[int, list[int]] = {}
    for index, token in enumerate(token_map):
        if token.get('token_idx') != index or not isinstance(token.get('line_id'), int):
            raise ValueError('token_idx는 연속적이고 모든 토큰은 정수 line_id에 매핑되어야 합니다.')
        groups.setdefault(token['line_id'], []).append(index)
    if not groups:
        raise ValueError('비어 있는 token_map입니다.')
    for lid, indices in groups.items():
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError(f'하나의 로그 라인에 속한 토큰은 연속되어야 합니다: {lid}')
        if lid not in positions:
            raise ValueError(f'원본 윈도우의 위치가 누락된 라인: {lid}')
    budget = math.floor(len(token_map) * keep_ratio)
    if budget < 1:
        raise ValueError('토큰 보존 예산이 1보다 작습니다.')
    scores = {}
    for lid, indices in groups.items():
        p = positions[lid]['position']
        if p is None:
            # D excluded N=1 groups from AUC.  Use neutral score for compression;
            # record it rather than falsely treating such lines as first/last.
            score = 0.5
        else:
            score = 1.0 - p if direction == 'front' else p
        scores[lid] = {
            'position': p, 'position_score': score,
            'rank': positions[lid]['rank'], 'n': positions[lid]['n'],
            'cost': len(indices),
        }
    chosen: set[int] = set()
    used = 0
    # deterministic tie-break: input order, NOT source labels nor Anchor.
    order = {lid: idx for idx, lid in enumerate(groups)}
    for lid in sorted(groups, key=lambda lid: (-scores[lid]['position_score'], order[lid])):
        cost = len(groups[lid])
        if used + cost <= budget:
            chosen.add(lid)
            used += cost
    if not chosen:
        raise ValueError('토큰 예산에 들어가는 완전한 로그 라인이 없습니다.')
    kept = sorted(index for lid in chosen for index in groups[lid])
    return Selection(
        total_tokens=len(token_map), budget_tokens=budget,
        selected_line_ids=[lid for lid in groups if lid in chosen],
        kept_token_indices=kept, protected_line_ids=[], scores=scores,
    )


def matched_auc(positions: dict[int, dict], labels: dict[int, int], *,
                ratio: int = 4, seed: int = 42) -> dict:
    """Part D's node × 1h matched AUC; gold labels used ONLY in evaluation.

    Based on Part D matching.ipynb: per-group all positives + up to `ratio`
    negatives per positive, then one pooled ROC AUC.  For N=1 no P exists.
    """
    if ratio not in (1, 4):
        raise ValueError('Part D 비교 조건은 ratio=1 또는 ratio=4입니다.')
    from sklearn.metrics import roc_auc_score  # optional, evaluation only
    by_group: dict[tuple[str, str], dict[int, list[int]]] = defaultdict(lambda: {0: [], 1: []})
    for lid, data in positions.items():
        if data['position'] is None or lid not in labels:
            continue
        label = labels[lid]
        if label not in (0, 1):
            raise ValueError(f'잘못된 이진 라벨: {lid}')
        by_group[(data['node'], data['grid_id_1h'])][label].append(lid)
    rng = random.Random(seed)
    chosen: list[int] = []
    for group, pair in sorted(by_group.items()):
        pos, neg = pair[1], pair[0]
        if not pos or not neg:
            continue
        chosen.extend(pos)
        chosen.extend(rng.sample(neg, k=min(len(neg), len(pos) * ratio)))
    if not chosen:
        raise ValueError('같은 노드·시간창에 양성과 음성이 함께 있는 평가 데이터가 없습니다.')
    y = [labels[lid] for lid in chosen]
    s = [positions[lid]['position'] for lid in chosen]
    auc = float(roc_auc_score(y, s))
    return {
        'ratio': ratio, 'auc_P': auc, 'auc_1_minus_P': 1 - auc,
        'n_positive': sum(y), 'n_negative': len(y) - sum(y),
        'note': 'AUC는 압축 선택에 사용하지 않습니다. 기존 D의 위치 분류력 진단입니다.',
    }


def _load_binary_labels(path: str | Path) -> dict[int, int]:
    labels = {}
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            lid = int(row['line_id'])
            if lid in labels:
                raise ValueError(f'라벨 중복 line_id: {lid}')
            value = str(row['is_hardcase']).strip()
            if value not in ('0', '1'):
                raise ValueError('평가 라벨은 is_hardcase 0/1이어야 합니다.')
            labels[lid] = int(value)
    return labels


def _load_chunk(token_map_jsonl: str | Path, chunk_id: int) -> dict:
    found = []
    with Path(token_map_jsonl).open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get('window_idx') == chunk_id:
                found.append(item)
    if len(found) != 1:
        raise ValueError(f'window_idx={chunk_id} 레코드가 정확히 하나 필요합니다. 실제 {len(found)}개')
    chunk = found[0]
    if len(chunk['input_ids']) != len(chunk['token_map']):
        raise ValueError('input_ids와 token_map 토큰 수가 다릅니다.')
    return chunk


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    build = sub.add_parser('prepare', help='원본 노드별 1시간 전체 CSV에서 position 생성')
    build.add_argument('--full-window-csv', required=True)
    build.add_argument('--positions-out', required=True)

    run = sub.add_parser('compress', help='position 점수로 라인 선택, 선택적 KV 압축')
    run.add_argument('--token-map', required=True, help='Part C의 windows_token_map.jsonl')
    run.add_argument('--window-id', required=True, type=int)
    run.add_argument('--positions', required=True)
    run.add_argument('--keep-ratio', type=float, default=0.5)
    run.add_argument('--direction', choices=('front', 'back'), default='front')
    run.add_argument('--output', required=True)
    run.add_argument('--kv-cache', help='신뢰할 수 있는 legacy KV 캐시 .pt')
    run.add_argument('--kv-output', help='압축된 legacy KV 캐시 .pt')

    evaluate = sub.add_parser('auc', help='별도 라벨로 Part D 매칭 AUC만 평가')
    evaluate.add_argument('--positions', required=True)
    evaluate.add_argument('--labels', required=True, help='line_id,is_hardcase(0/1) CSV')
    evaluate.add_argument('--output', help='평가 결과 JSON')
    args = p.parse_args()

    if args.cmd == 'prepare':
        rows = prepare_positions(args.full_window_csv, args.positions_out)
        print(f'position {len(rows)}줄 저장: {args.positions_out}')
    elif args.cmd == 'auc':
        positions = load_positions(args.positions)
        labels = _load_binary_labels(args.labels)
        results = [matched_auc(positions, labels, ratio=r) for r in (1, 4)]
        output = json.dumps(results, ensure_ascii=False, indent=2)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(output, encoding='utf-8')
        print(output)
    elif args.cmd == 'compress':
        if bool(args.kv_cache) != bool(args.kv_output):
            p.error('--kv-cache와 --kv-output은 함께 지정해야 합니다.')
        chunk = _load_chunk(args.token_map, args.window_id)
        positions = load_positions(args.positions)
        selection = select_position_lines(
            chunk['token_map'], positions,
            keep_ratio=args.keep_ratio, direction=args.direction,
        )
        result = asdict(selection)
        result.update({
            'window_id': args.window_id, 'direction': args.direction,
            'used_tokens': selection.used_tokens,
            'original_positions': selection.kept_token_indices,
            'next_position': selection.total_tokens,
            'note': '실제 generate() 연결은 미구현. gold label/AUC는 선택에 사용하지 않음.',
        })
        if args.kv_cache:
            import torch
            original = torch.load(args.kv_cache, map_location='cpu', weights_only=True)
            compressed = compress_kv(original, selection)
            target = Path(args.kv_output)
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'past_key_values': compressed,
                'original_positions': selection.kept_token_indices,
                'next_position': selection.total_tokens,
            }, target)
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'선택 완료: {selection.total_tokens} -> {selection.used_tokens} tokens '
              f'(budget={selection.budget_tokens}, direction={args.direction})')


if __name__ == '__main__':
    main()
