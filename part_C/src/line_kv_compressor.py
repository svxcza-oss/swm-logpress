"""라인 매핑 기반 실험용 KV 압축기.

최근성/구조 정보/Anchor Attention으로 완전한 로그 줄을 선택한다.
압축된 텐서는 원래 위치와 함께 반환한다. generate() 자동 연결기는 아니다.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class Weights:
    # 성능 최적값이 아니라 네 정보를 함께 검증하기 위한 기본값이다.
    recency: float = .60
    structure: float = .20
    identifier: float = .05
    attention: float = .15

    def validate(self):
        values = list(asdict(self).values())
        if any(not math.isfinite(v) or v < 0 for v in values) or not math.isclose(sum(values), 1):
            raise ValueError('가중치는 유한한 음이 아닌 값이며 합이 1이어야 합니다.')


@dataclass
class Selection:
    total_tokens: int
    budget_tokens: int
    selected_line_ids: list[int]
    kept_token_indices: list[int]
    protected_line_ids: list[int]
    scores: dict

    @property
    def used_tokens(self):
        return len(self.kept_token_indices)


def identifiers(text):
    """소스 위치를 빼고 본문의 코드·노드·함수·장치 이름을 추출한다."""
    text = re.sub(r'\[[^\]]*\.(?:c|h):\d+\]', ' ', text.lower())
    result = {k+':'+v for k,v in re.findall(
        r'\b(eventid|error code|return code|qpn|aid|class|node)\s*[:=]?\s*(-?0x[0-9a-f]+|-?[0-9a-f]+)\b', text)}
    result.update(w for w in re.findall(r'\b[a-z][a-z0-9_]*\b', text)
                  if '_' in w or re.fullmatch(r'infinihost\d+', w))
    return result


def structure_features(row, anchor):
    # 이 Thunderbird 전처리에서는 user가 호스트명이다. label은 읽지 않는다.
    host = row.get('user', '').strip().lower()
    h = float(host not in ('', '-', '#1#') and host == anchor.get('user', '').strip().lower())
    normalize = lambda x: re.sub(r'\[\d+\]$', '', x.strip().lower())
    c = normalize(row.get('component', ''))
    c = float(c not in ('', '-', '#1#') and c == normalize(anchor.get('component', '')))
    a,b = identifiers(row.get('content', '')), identifiers(anchor.get('content', ''))
    return (h+c)/2, len(a & b)/len(a | b) if a | b else 0.0


def select_lines(token_map, source: Mapping, anchor_id: int, attention_scores: Mapping,
                 *, keep_ratio=.5, weights=Weights(), protect_first_line=True):
    """토큰 비용을 넘지 않게 점수 순서로 완전한 라인을 선택한다.

    attention_scores는 후보별 비음수 Anchor Attention 값이다.
    전체 줄 보호를 유지해 이전 파일럿과 비교한다. 예산 부족은 명시적으로 실패한다.
    """
    weights.validate()
    if not math.isfinite(keep_ratio) or not 0 < keep_ratio <= 1:
        raise ValueError('보존 비율은 0보다 크고 1 이하여야 합니다.')
    groups = {}
    for index, token in enumerate(token_map):
        if token.get('token_idx') != index or token.get('line_id') is None:
            raise ValueError('token_idx가 연속적이어야 하며 미매핑 토큰은 지원하지 않습니다.')
        lid = token['line_id']
        if not isinstance(lid, int):
            raise ValueError('line_id는 정수여야 합니다.')
        groups.setdefault(lid, []).append(index)
    if not groups or list(groups)[-1] != anchor_id:
        raise ValueError('마지막 라인을 Anchor로 지정해야 합니다. 미래 로그는 사용할 수 없습니다.')
    if any(v != list(range(v[0],v[-1]+1)) for v in groups.values()):
        raise ValueError('각 로그 라인의 토큰은 연속되어야 합니다.')
    if any(i not in source for i in groups):
        raise ValueError('원본 라인 정보가 누락되었습니다.')
    candidates = list(groups)[:-1]
    raw = {}
    for i in candidates:
        if weights.attention and i not in attention_scores:
            raise ValueError(f'라인 {i}의 Attention 값이 없습니다.')
        value = float(attention_scores.get(i, 0))
        if not math.isfinite(value) or value < 0:
            raise ValueError('Attention 값은 유한한 비음수여야 합니다.')
        raw[i] = value
    protected = {anchor_id}
    if protect_first_line:
        protected.add(next(iter(groups)))
    # 첫 줄은 이미 보호되므로 큰 sink 점수가 나머지 점수를 축소하지 않도록 제외한다.
    scoring = [raw[i] for i in candidates if i not in protected]
    lo,hi = (min(scoring),max(scoring)) if scoring else (0,0)
    details = {}
    for pos,i in enumerate(candidates):
        r = (pos+1)/len(candidates)
        m,j = structure_features(source[i],source[anchor_id])
        a = max(0,min(1,(raw[i]-lo)/(hi-lo))) if hi>lo else 0.0
        score = weights.recency*r + weights.structure*m + weights.identifier*j + weights.attention*a
        details[i] = dict(recency=r,structure=m,identifier=j,attention_raw=raw[i],attention_normalized=a,score=score,cost=len(groups[i]))
    budget = math.floor(len(token_map)*keep_ratio)
    chosen = set(protected)
    used = sum(len(groups[i]) for i in chosen)
    if used > budget:
        raise ValueError(f'보호 라인 비용 {used}토큰이 예산 {budget}토큰보다 큽니다.')
    order = {i:pos for pos,i in enumerate(groups)}
    for i in sorted(candidates,key=lambda i:(-details[i]['score'],-order[i])):
        if i not in chosen and used+len(groups[i])<=budget:
            chosen.add(i)
            used += len(groups[i])
    kept = sorted(index for i in chosen for index in groups[i])
    return Selection(len(token_map),budget,[i for i in groups if i in chosen],kept,
                     [i for i in groups if i in protected],details)


def anchor_scores(line_attentions, line_ids, anchor_id):
    """[층, 헤드, Query라인, Key라인]을 층·헤드 평균하여 Anchor 행을 읽는다."""
    import torch
    if (line_attentions.ndim != 4 or min(line_attentions.shape[:2])<1 or
            tuple(line_attentions.shape[-2:]) != (len(line_ids),len(line_ids)) or
            len(set(line_ids)) != len(line_ids) or anchor_id not in line_ids):
        raise ValueError('라인 Attention shape 또는 line_ids가 올바르지 않습니다.')
    if not torch.isfinite(line_attentions).all() or (line_attentions<0).any():
        raise ValueError('라인 Attention에 잘못된 값이 있습니다.')
    row = line_attentions.float().mean(dim=(0,1))[line_ids.index(anchor_id)].cpu().tolist()
    return dict(zip(line_ids,row))


def compress_kv(past_key_values, selection: Selection):
    """legacy ((K,V), ...) 형식의 실제 텐서를 복사해 압축한다.

    각 텐서 형식은 [1, KV헤드, 원본토큰, head_dim]. 모든 층/헤드가 같은
    라인을 남긴다. 입력 캐시는 수정하지 않는다. DynamicCache는 직접 지원하지
    않으며 생성 모델의 위치/마스크 처리는 호출자가 별도로 연결해야 한다.
    """
    import torch
    if not isinstance(past_key_values,(tuple,list)) or not past_key_values:
        raise ValueError('비어 있지 않은 legacy (K,V) 목록을 전달하세요.')
    indices = selection.kept_token_indices
    if (not indices or indices!=sorted(set(indices)) or min(indices)<0 or
            max(indices)>=selection.total_tokens or len(indices)>selection.budget_tokens):
        raise ValueError('선택한 토큰 인덱스/예산이 잘못되었습니다.')
    # 전체 검증을 마친 뒤 복사하므로 중간 층의 실패가 입력 캐시를 변경하지 않는다.
    for pair in past_key_values:
        if not isinstance(pair,(tuple,list)) or len(pair)!=2:
            raise ValueError('각 층은 (K,V) 쌍이어야 합니다.')
        k,v = pair
        if (not isinstance(k,torch.Tensor) or not isinstance(v,torch.Tensor) or
                k.ndim!=4 or k.shape!=v.shape or k.shape[0]!=1 or
                k.shape[2]!=selection.total_tokens or k.device!=v.device or k.dtype!=v.dtype):
            raise ValueError('K/V shape, 장치 또는 dtype이 일치하지 않습니다. 배치 크기는 1입니다.')
    compressed=[]
    for k,v in past_key_values:
        idx=torch.tensor(indices,device=k.device,dtype=torch.long)
        compressed.append((k.index_select(2,idx),v.index_select(2,idx)))
    return tuple(compressed)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--token-map',required=True)
    parser.add_argument('--source-csv',required=True)
    parser.add_argument('--line-attention',required=True)
    parser.add_argument('--window-id',required=True,type=int)
    parser.add_argument('--keep-ratio',type=float,default=.5)
    parser.add_argument('--weights',nargs=4,type=float,default=[.6,.2,.05,.15],metavar=('RECENT','STRUCTURE','IDENTIFIER','ATTENTION'))
    parser.add_argument('--output',required=True,help='선택 결과 JSON 경로')
    parser.add_argument('--kv-cache',help='선택 사항: legacy (K,V) 목록을 저장한 신뢰하는 .pt 파일')
    parser.add_argument('--kv-output',help='실제 압축 K/V와 원본 위치를 저장할 .pt 경로')
    args=parser.parse_args()
    if bool(args.kv_cache)!=bool(args.kv_output):
        parser.error('--kv-cache와 --kv-output은 함께 지정해야 합니다.')
    import torch
    windows=[json.loads(l) for l in Path(args.token_map).read_text(encoding='utf-8').splitlines() if l.strip()]
    matching=[w for w in windows if w.get('window_idx')==args.window_id]
    if len(matching)!=1:
        raise ValueError('window-id에 해당하는 입력이 정확히 하나여야 합니다.')
    window=matching[0]
    if len(window['input_ids'])!=len(window['token_map']):
        raise ValueError('입력과 토큰 매핑 길이가 다릅니다.')
    with open(args.source_csv,encoding='utf-8-sig',newline='') as f:
        source={int(r['line_id']):{k:v for k,v in r.items() if k!='label'} for r in csv.DictReader(f)}
    attn=torch.load(args.line_attention,map_location='cpu',weights_only=True)
    ids=list(dict.fromkeys(t['line_id'] for t in window['token_map']))
    if attn.get('window_idx')!=args.window_id or attn['line_ids']!=ids:
        raise ValueError('다른 윈도우의 Attention 파일입니다.')
    scores=anchor_scores(attn['line_attentions'],ids,ids[-1])
    weights=Weights(*args.weights)
    selection=select_lines(window['token_map'],source,ids[-1],scores,keep_ratio=args.keep_ratio,weights=weights)
    result=asdict(selection)
    result.update(window_id=args.window_id,weights=asdict(weights),used_tokens=selection.used_tokens,
                  original_positions=selection.kept_token_indices,next_position=len(window['input_ids']),
                  note='원본 토큰 위치 기준. 생성용 cache_position/마스크 어댑터는 별도로 필요합니다.')
    if args.kv_cache:
        original=torch.load(args.kv_cache,map_location='cpu',weights_only=True)
        compressed=compress_kv(original,selection)
        Path(args.kv_output).parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(past_key_values=compressed,original_positions=selection.kept_token_indices,next_position=selection.total_tokens),args.kv_output)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'{selection.total_tokens} → {selection.used_tokens}토큰 / 예산 {selection.budget_tokens}')


if __name__=='__main__':
    main()
