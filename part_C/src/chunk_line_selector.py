"""완료된 청크 전체의 Alert 보존을 비교하는 다중 Anchor 선택기.

마지막 줄을 Anchor로 가정하지 않는다. 모든 줄이 보존 후보이며 Anchor도
자동 보호하지 않는다. 원인 정답을 사용하지 않는 오프라인 청크 선택 정책이다.
"""
import hashlib
import math

from line_kv_compressor import Selection, Weights, structure_features
from window_extractor import detect_level


def find_anchors(line_ids, source):
    """원본 label/level을 읽지 않고 기존 본문 키워드 규칙을 적용한다."""
    return [i for i in line_ids if detect_level(source[i].get('content', ''))['level'] in ('WARN', 'ERROR')]


def prepare_chunk_features(token_map, source, attention_rows=None, *, distance_scale=1024):
    """각 Anchor보다 앞선 라인에만 점수를 준다. 라벨은 참조하지 않는다.

    attention_rows 형식은 {anchor_id: {key_line_id: attention}}.
    모든 Anchor 행이 있는 경우에만 Attention 조건을 지원한다.
    라인마다 관측 가능한 이후 Anchor 수로 나눠 단순 합산의 노출 편향을 줄인다.
    """
    if not math.isfinite(distance_scale) or distance_scale <= 0:
        raise ValueError('거리 척도는 양의 유한한 토큰 수여야 합니다.')
    groups={}
    for pos,t in enumerate(token_map):
        if t.get('token_idx')!=pos or not isinstance(t.get('line_id'),int):
            raise ValueError('연속 token_idx와 정수 line_id 매핑이 필요합니다.')
        groups.setdefault(t['line_id'],[]).append(pos)
    if not groups or any(i not in source for i in groups):
        raise ValueError('비어 있거나 원본 라인 정보가 없습니다.')
    if any(v!=list(range(v[0],v[-1]+1)) for v in groups.values()):
        raise ValueError('라인별 토큰은 연속되어야 합니다.')
    ids=list(groups)
    anchors=find_anchors(ids,source)
    positions={i:p for p,i in enumerate(ids)}
    features={i:dict(recency=(p+1)/len(ids),anchor_distance=0.,structure=0.,identifier=0.,attention=0.,observations=0,keyword=float(i in anchors)) for p,i in enumerate(ids)}
    for anchor in anchors:
        previous=ids[:positions[anchor]]
        normalized={}
        if attention_rows is not None:
            if anchor not in attention_rows:
                raise ValueError('일부 Anchor의 Attention 행이 누락되었습니다.')
            if any(i not in attention_rows[anchor] for i in previous):
                raise ValueError('일부 이전 라인의 Attention 값이 누락되었습니다.')
            values={i:float(attention_rows[anchor][i]) for i in previous}
            if any(not math.isfinite(v) or v<0 for v in values.values()):
                raise ValueError('Attention은 유한한 비음수여야 합니다.')
            lo,hi=(min(values.values()),max(values.values())) if values else (0,0)
            normalized={i:(v-lo)/(hi-lo) if hi>lo else 0. for i,v in values.items()}
        for i in previous:
            m,j=structure_features(source[i],source[anchor])
            f=features[i]
            f['structure']+=m
            f['identifier']+=j
            f['attention']+=normalized.get(i,0.)
            # 라인 사이에 끼어 있는 원본 토큰 수. 직전 줄이면 거리는 0이다.
            distance=groups[anchor][0]-groups[i][-1]-1
            f['anchor_distance']+=math.exp(-distance/distance_scale)
            f['observations']+=1
    for f in features.values():
        if f['observations']:
            for name in ('structure','identifier','attention','anchor_distance'):
                f[name]/=f['observations']
    return dict(groups=groups,features=features,anchors=anchors,total_tokens=len(token_map),
                has_attention=attention_rows is not None,distance_scale=distance_scale)


def select_prepared_chunk(prepared, *, keep_ratio=.5, weights=Weights(0,.5,.5,0),
                          recency_mode='end', seed=0, protect_first_line=False,
                          baseline=None):
    """같은 후보/보호/예산에서 비교한다. 기본값은 최근성 없는 구조 정보 조건.

    동점은 고정 seed 해시로 처리해 최근성 0 조건에 최근성 tie-break가 섞이지
    않게 한다. 이는 정답과 무관한 임의 순서이며 사건마다 성능이 달라질 수 있다.
    baseline은 None, random, keyword 중 하나. 마지막 두 항목은 비교용이다.
    """
    weights.validate()
    if not math.isfinite(keep_ratio) or not 0<keep_ratio<=1:
        raise ValueError('보존 비율은 (0,1]이어야 합니다.')
    if recency_mode not in ('end','anchor') or baseline not in (None,'random','keyword'):
        raise ValueError('지원하지 않는 점수 조건입니다.')
    if weights.attention and not prepared['has_attention']:
        raise ValueError('Attention 가중치가 있는데 실제 Attention이 없습니다.')
    groups=prepared['groups']
    ids=list(groups)
    scores={}
    for i,f in prepared['features'].items():
        r=f['recency'] if recency_mode=='end' else f['anchor_distance']
        value=weights.recency*r+weights.structure*f['structure']+weights.identifier*f['identifier']+weights.attention*f['attention']
        if baseline=='random': value=0.
        if baseline=='keyword': value=f['keyword']
        scores[i]=dict(f,score=value,cost=len(groups[i]))
    order=sorted(ids,key=lambda i:(-scores[i]['score'],hashlib.sha256(f'{seed}:{i}'.encode()).hexdigest()))
    budget=math.floor(prepared['total_tokens']*keep_ratio)
    protected=[ids[0]] if protect_first_line else []
    chosen=set(protected)
    used=sum(len(groups[i]) for i in chosen)
    if used>budget: raise ValueError('보호 비용이 예산보다 큽니다.')
    for i in order:
        if i not in chosen and used+len(groups[i])<=budget:
            chosen.add(i)
            used+=len(groups[i])
    return Selection(prepared['total_tokens'],budget,[i for i in ids if i in chosen],
                     sorted(t for i in chosen for t in groups[i]),protected,scores)
