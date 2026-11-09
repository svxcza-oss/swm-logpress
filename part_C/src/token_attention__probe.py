"""LogPress token-level position-bias diagnostic.

Use immediately after a Qwen forward pass with output_attentions=True.
Does not modify attention tensors or the production ranking.
"""
import csv
from pathlib import Path
import torch


def _attr(t, name, default=None):
    return t.get(name, default) if isinstance(t, dict) else getattr(t, name, default)


def save_token_probe(attentions, token_map, window_idx, output_dir='token_probe_results', anchor_levels=None, line_meta=None):
    """Write per-layer/head breakdown of attention received by first token/line.

    attentions: tuple of [1, H, T, T] tensors, from outputs.attentions
    token_map: list of TokenInfo objects or dictionaries in token order
    line_meta: optional list containing line_id, level to select WARN/ERROR query lines
    """
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    if not attentions:
        raise ValueError('No attention tensors. Call model with output_attentions=True and eager attention.')
    token_count = len(token_map)
    token_line_ids = [_attr(t, 'line_id') for t in token_map]
    if token_count == 0 or all(x is None for x in token_line_ids):
        raise ValueError('No mapped tokens')
    first_line_id = next(x for x in token_line_ids if x is not None)
    first_line_tokens = [i for i, x in enumerate(token_line_ids) if x == first_line_id]
    first_token_idx = first_line_tokens[0]
    first_rest = [i for i in first_line_tokens if i != first_token_idx]
    other_tokens = [i for i, x in enumerate(token_line_ids) if x is not None and x != first_line_id]
    unmapped_tokens = [i for i, x in enumerate(token_line_ids) if x is None]
    levels = {str(_attr(m, 'line_id')): str(_attr(m, 'level', '-')).upper() for m in (line_meta or [])}
    anchor_set = set(anchor_levels or ('ERROR', 'WARN'))
    anchor_ids = [lid for lid in dict.fromkeys(x for x in token_line_ids if x is not None)
                  if levels.get(str(lid), '-') in anchor_set]
    if not anchor_ids:
        print(f'[token-probe] Window {window_idx}: WARN/ERROR metadata not found, skipping.')
        return []
    fname = dest / 'token_bias_by_head.csv'
    new_file = not fname.exists() or fname.stat().st_size == 0
    fields = ['window_idx','anchor_line_id','first_line_id','layer','head','query_token_count',
              'key_token_count','first_token_score','first_line_rest_score',
              'first_line_total_score','other_lines_score','unmapped_score',
              'first_token_share_of_first_line','first_line_token_count']
    rows = []
    for layer_idx, attn in enumerate(attentions):
        if attn is None:
            raise ValueError(f'Layer {layer_idx}: attention is None')
        if attn.ndim != 4 or attn.shape[0] != 1 or attn.shape[-1] != token_count or attn.shape[-2] != token_count:
            raise ValueError(f'Layer {layer_idx}: expected [1,H,{token_count},{token_count}], got {tuple(attn.shape)}')
        # Only select query tokens in a single Anchor line; avoids copying full T x T attention to CPU.
        for anchor_id in anchor_ids:
            qidx = [i for i, x in enumerate(token_line_ids) if x == anchor_id]
            if not qidx:
                continue
            # [H, Q, K] -> [H, K] average query tokens; keep on device for reduction.
            qmean = attn[0, :, qidx, :].float().mean(dim=1)
            # Move only 4-5 small vectors per layer to CPU.
            first = qmean[:, first_token_idx]
            rest = qmean[:, first_rest].sum(dim=-1) if first_rest else torch.zeros_like(first)
            others = qmean[:, other_tokens].sum(dim=-1) if other_tokens else torch.zeros_like(first)
            unmapped = qmean[:, unmapped_tokens].sum(dim=-1) if unmapped_tokens else torch.zeros_like(first)
            measures = torch.stack((first, rest, others, unmapped), dim=1).detach().cpu().tolist()
            for head_idx, (a, b, c, d) in enumerate(measures):
                total = a+b
                rows.append(dict(window_idx=window_idx, anchor_line_id=anchor_id, first_line_id=first_line_id,
                                 layer=layer_idx, head=head_idx, query_token_count=len(qidx),
                                 key_token_count=token_count, first_token_score=a,
                                 first_line_rest_score=b, first_line_total_score=total,
                                 other_lines_score=c, unmapped_score=d,
                                 first_token_share_of_first_line=(a/total if total else 0.0),
                                 first_line_token_count=len(first_line_tokens)))
            del qmean
    with fname.open('a', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
    print(f'[token-probe] Window {window_idx}: {len(rows)} head records saved -> {fname}')
    return rows
