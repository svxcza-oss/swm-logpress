"""Bridge Part A week6 Thunderbird chunk TXT/metadata into Part C x Part D inputs.

Usage from swm-logpress/part_C/src:
  python prepare_week6_thunderbird.py \
    --chunks-dir ../../Part_A/week6/thunderbird/chunks \
    --output-dir output/week6_314927 --chunk-index 0

This adapter intentionally does NOT use Alert labels in position calculation or
selection. It saves separate Alert labels only for later proxy evaluation.
Part D position is node x clock-hour rank/(N-1), calculated from all parent
window lines before processing a chunk.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import re
from pathlib import Path

from line_kv_compressor_position import prepare_positions


def make_inputs(chunks_dir: Path, output_dir: Path, *, window_id: int = 314927,
                chunk_index: int = 0, model: str = 'Qwen/Qwen2.5-3B-Instruct') -> dict:
    metadata = chunks_dir / f'window_{window_id}_metadata.csv'
    if not metadata.is_file():
        raise FileNotFoundError(f'Chunk 메타데이터가 없습니다: {metadata}')
    with metadata.open(newline='', encoding='utf-8-sig') as f:
        records = list(csv.DictReader(f))
    if not records:
        raise ValueError('Chunk 메타데이터가 비어 있습니다.')
    required = {'chunk', 'start_line', 'end_line', 'log_count', 'token_count', 'alerts'}
    if not required.issubset(records[0]):
        raise ValueError(f'필수 metadata 컬럼 누락: {sorted(required-set(records[0]))}')
    records.sort(key=lambda r: int(r['start_line']))

    all_rows = []
    all_alerts = {}
    selected_text = None
    selected_ids = None
    selected_expected_tokens = None
    next_start = 0
    seen_chunks = set()

    for meta in records:
        chunk_name = meta['chunk']
        match = re.fullmatch(r'chunk(\d+)', chunk_name)
        if not match:
            raise ValueError(f'예상과 다른 chunk 이름: {chunk_name!r}')
        idx = int(match.group(1))
        if idx in seen_chunks:
            raise ValueError(f'중복 chunk 번호: {idx}')
        seen_chunks.add(idx)
        start = int(meta['start_line'])
        end = int(meta['end_line'])
        count = int(meta['log_count'])
        if start != next_start or end-start+1 != count:
            raise ValueError(f'{chunk_name} 원본 line_id가 불연속이거나 개수가 틀립니다.')
        path = chunks_dir / f'window_{window_id}_{chunk_name}.txt'
        if not path.is_file():
            raise FileNotFoundError(f'Chunk 파일 없음: {path}')
        text = path.read_text(encoding='utf-8')
        lines = text.splitlines(keepends=True)
        if len(lines) != count:
            raise ValueError(f'{chunk_name} 로그 라인 수가 metadata와 다름: {len(lines)} != {count}')
        for offset, line in enumerate(lines):
            fields = line.split(maxsplit=3)
            if len(fields) < 3:
                raise ValueError(f'{chunk_name} line {offset}: Thunderbird 형식 파싱 실패')
            try:
                ts = int(fields[0])
            except ValueError as e:
                raise ValueError(f'{chunk_name} line {offset}: UNIX 초 단위 timestamp 파싱 실패') from e
            if ts // 3600 != window_id:
                raise ValueError(f'{chunk_name} line {offset}: 원본 윈도우 ID 불일치 ({ts // 3600})')
            all_rows.append({'line_id': start+offset, 'timestamp': ts, 'user': fields[2]})

        alerts = json.loads(meta['alerts'] or '{}')
        if not isinstance(alerts, dict):
            raise ValueError(f'{chunk_name}: alerts는 JSON 객체여야 합니다.')
        for raw_lid, alert_value in alerts.items():
            lid = int(raw_lid)
            if not start <= lid <= end or lid in all_alerts:
                raise ValueError(f'잘못되었거나 중복된 Alert line_id: {lid}')
            all_alerts[lid] = str(alert_value)
        if int(meta['alert_count']) != len(alerts):
            raise ValueError(f'{chunk_name}: Alert 수가 metadata와 다릅니다.')
        if idx == chunk_index:
            selected_text = text
            selected_ids = list(range(start, end+1))
            selected_expected_tokens = int(meta['token_count'])
        next_start = end+1

    if selected_text is None or selected_ids is None:
        raise ValueError(f'chunk{chunk_index}이 metadata에 없습니다.')
    if sorted(seen_chunks) != list(range(len(records))):
        raise ValueError('chunk0부터 연속된 chunk들이 모두 필요합니다.')

    output_dir.mkdir(parents=True, exist_ok=True)
    full_csv = output_dir / f'window_{window_id}_full_window.csv'
    with full_csv.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['line_id', 'timestamp', 'user'])
        writer.writeheader()
        writer.writerows(all_rows)
    positions_csv = output_dir / f'window_{window_id}_positions.csv'
    positions = prepare_positions(full_csv, positions_csv)

    # Do not pass this file to compression: it is a proxy evaluation label.
    alerts_csv = output_dir / f'window_{window_id}_alert_labels.csv'
    with alerts_csv.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['line_id', 'is_alert', 'alert_label'])
        writer.writerows((r['line_id'], int(r['line_id'] in all_alerts),
                          all_alerts.get(r['line_id'], '')) for r in all_rows)

    # Reproduce exactly the Part A chunk TXT stream. Retokenizing each line
    # separately is NOT equivalent to tokenizing the concatenated chunk.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError('오프셋 매핑을 위해 fast tokenizer가 필요합니다.')
    encoded = tokenizer(selected_text, add_special_tokens=False,
                        return_offsets_mapping=True, truncation=False)
    input_ids = encoded['input_ids']
    offsets = encoded['offset_mapping']
    if len(input_ids) != selected_expected_tokens:
        raise ValueError(
            f'chunk{chunk_index} 토큰 수가 Part A metadata와 다릅니다: '
            f'{len(input_ids)} != {selected_expected_tokens}. '
            '모델/토크나이저 버전 또는 원본 텍스트가 다를 수 있습니다.'
        )
    lines = selected_text.splitlines(keepends=True)
    starts, ends = [], []
    current = 0
    for line in lines:
        starts.append(current)
        current += len(line)
        ends.append(current)

    token_map = []
    for i, (token_id, (start, end)) in enumerate(zip(input_ids, offsets)):
        if start >= end:
            raise ValueError(f'빈 문자 오프셋 token_idx={i}: ({start}, {end})')
        local_line = bisect.bisect_right(starts, start)-1
        if local_line < 0 or local_line >= len(starts) or end > ends[local_line]:
            raise ValueError(
                f'토큰이 두 로그 라인을 걸침: token_idx={i}, offset=({start},{end}). '
                '정확한 라인 단위 선택을 위해 자동 매핑을 중단했습니다.'
            )
        token_map.append({
            'token_idx': i, 'token_id': token_id,
            'start': start, 'end': end, 'line_id': selected_ids[local_line]
        })
    if not token_map or set(t['line_id'] for t in token_map) != set(selected_ids):
        raise ValueError('토큰이 없는 로그 라인 또는 매핑 누락이 있습니다.')

    token_map_file = output_dir / f'window_{window_id}_chunk{chunk_index}_token_map.jsonl'
    payload = {
        'window_idx': chunk_index, 'parent_window_id': window_id,
        'text': selected_text, 'input_ids': input_ids, 'token_map': token_map,
    }
    token_map_file.write_text(json.dumps(payload, ensure_ascii=False)+'\n', encoding='utf-8')
    return {
        'full_window_csv': str(full_csv), 'positions_csv': str(positions_csv),
        'alert_labels_csv': str(alerts_csv), 'token_map_jsonl': str(token_map_file),
        'window_lines': len(all_rows), 'window_alerts': len(all_alerts),
        'chunk_lines': len(selected_ids), 'chunk_tokens': len(input_ids),
        'positions': len(positions), 'chunk_index': chunk_index,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chunks-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--window-id', type=int, default=314927)
    parser.add_argument('--chunk-index', type=int, default=0)
    parser.add_argument('--model', default='Qwen/Qwen2.5-3B-Instruct')
    args = parser.parse_args()
    result = make_inputs(args.chunks_dir, args.output_dir, window_id=args.window_id,
                         chunk_index=args.chunk_index, model=args.model)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
