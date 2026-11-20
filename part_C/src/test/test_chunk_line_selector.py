import pytest
from chunk_line_selector import prepare_chunk_features,select_prepared_chunk
from line_kv_compressor import Weights


def data(contents):
    tokens=[dict(token_idx=j,line_id=j//2) for j in range(2*len(contents))]
    source={i:dict(user='host1',component='kernel',content=c) for i,c in enumerate(contents)}
    return tokens,source


def test_internal_anchors_and_no_auto_protection():
    tokens,source=data(['normal','error code 1','normal','warning','normal'])
    p=prepare_chunk_features(tokens,source)
    assert p['anchors']==[1,3]
    assert p['features'][0]['observations']==2
    assert p['features'][1]['observations']==1
    assert p['features'][3]['observations']==0
    assert p['features'][4]['observations']==0
    s=select_prepared_chunk(p,keep_ratio=.2)
    assert s.protected_line_ids==[] and s.used_tokens<=2


def test_future_keys_and_self_attention_do_not_influence_scores():
    tokens,source=data(['normal','error','normal'])
    p=prepare_chunk_features(tokens,source,{1:{0:.3,1:999,2:999}})
    q=prepare_chunk_features(tokens,source,{1:{0:.3,1:0,2:0}})
    assert p['features']==q['features']


def test_no_anchor_policy_deterministic_and_budget():
    tokens,source=data(['ok']*8)
    p=prepare_chunk_features(tokens,source)
    assert not p['anchors']
    a=select_prepared_chunk(p,keep_ratio=.5)
    b=select_prepared_chunk(p,keep_ratio=.5)
    assert a==b and a.used_tokens==8
    assert a.selected_line_ids!=[4,5,6,7]


def test_no_label_leak_and_full_budget():
    tokens,source=data(['ok','error','ok'])
    a=prepare_chunk_features(tokens,source)
    for row in source.values(): row['label']='ALERT';row['level']='WARN'
    b=prepare_chunk_features(tokens,source)
    assert a==b
    assert select_prepared_chunk(a,keep_ratio=1).kept_token_indices==list(range(6))


def test_missing_attention_rejected_not_fabricated():
    tokens,source=data(['ok','error','normal'])
    p=prepare_chunk_features(tokens,source)
    with pytest.raises(ValueError):select_prepared_chunk(p,weights=Weights(0,0,0,1))
    with pytest.raises(ValueError):prepare_chunk_features(tokens,source,{})


def test_average_corrects_number_of_observations():
    tokens,source=data(['ok','warning','error','normal'])
    p=prepare_chunk_features(tokens,source)
    assert p['features'][0]['structure']==1
    assert p['features'][1]['structure']==1
    assert p['features'][0]['observations']==2
    assert p['features'][1]['observations']==1


def test_anchor_distance_uses_tokens_and_keeps_chronological_positions():
    tokens,source=data(['ok','ok','error','ok'])
    p=prepare_chunk_features(tokens,source,distance_scale=2)
    assert p['features'][1]['anchor_distance']==1
    assert p['features'][0]['anchor_distance']==pytest.approx(0.367879441)
    s=select_prepared_chunk(p,keep_ratio=.5,weights=Weights(1,0,0,0),recency_mode='anchor')
    assert s.selected_line_ids==[0,1]
    assert s.kept_token_indices==[0,1,2,3]
