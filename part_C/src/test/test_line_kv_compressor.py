"""라인 선택 예산, 라벨 누수, 실제 KV 내용 보존을 검증한다."""
import pytest
import torch
from line_kv_compressor import Weights, Selection, select_lines, anchor_scores, compress_kv


def sample():
    ids=[10,10,20,20,30,30,40,40]
    tokens=[dict(token_idx=i,line_id=x) for i,x in enumerate(ids)]
    source={i:dict(user='h1',component='kernel',content='error code 2') for i in set(ids)}
    return tokens,source


def test_exact_kv_contents_and_original_not_mutated():
    tokens,source=sample()
    s=select_lines(tokens,source,40,{10:99,20:8,30:1},keep_ratio=.75,
                   weights=Weights(0,0,0,1))
    assert s.selected_line_ids==[10,20,40]
    assert s.kept_token_indices==[0,1,2,3,6,7]
    # 두 층, 서로 다른 KV 헤드 수에서도 동일한 토큰을 남기는지 확인한다.
    cache=tuple((torch.arange(h*8*3).reshape(1,h,8,3).float(),
                 torch.arange(h*8*3).reshape(1,h,8,3).float()+1000) for h in (1,2))
    result=compress_kv(cache,s)
    for (k,v),(ck,cv) in zip(cache,result):
        assert ck.shape[2]==6
        assert torch.equal(ck,k[:,:,[0,1,2,3,6,7]])
        assert torch.equal(cv,v[:,:,[0,1,2,3,6,7]])
        assert k.shape[2]==8 and ck.data_ptr()!=k.data_ptr()


def test_labels_do_not_change_selection():
    tokens,source=sample()
    first=select_lines(tokens,source,40,{10:0,20:.2,30:.4},keep_ratio=.75)
    for i,row in source.items():
        row['label']='ALERT' if i==20 else '-'
    assert first==select_lines(tokens,source,40,{10:0,20:.2,30:.4},keep_ratio=.75)


def test_protected_budget_and_full_budget():
    tokens,source=sample()
    with pytest.raises(ValueError,match='보호'):
        select_lines(tokens,source,40,{10:0,20:0,30:0},keep_ratio=.25)
    s=select_lines(tokens,source,40,{10:0,20:0,30:0},keep_ratio=1)
    assert s.kept_token_indices==list(range(8))


@pytest.mark.parametrize('change', ['unmapped','index','future','missing_attention','nan'])
def test_bad_inputs_fail(change):
    tokens,source=sample()
    scores={10:0,20:.2,30:.4}
    anchor=40
    if change=='unmapped': tokens[0]['line_id']=None
    if change=='index': tokens[0]['token_idx']=3
    if change=='future': anchor=20
    if change=='missing_attention': del scores[20]
    if change=='nan': scores[20]=float('nan')
    with pytest.raises(ValueError):
        select_lines(tokens,source,anchor,scores,keep_ratio=.75)


def test_anchor_axis_and_mean():
    attn=torch.zeros(2,2,3,3)
    attn[:,:,2,:]=torch.tensor([.1,.2,.7])
    scores=anchor_scores(attn,[10,20,40],40)
    assert scores[20]==pytest.approx(.2)
    with pytest.raises(ValueError):
        anchor_scores(attn,[10,20],20)


def test_invalid_cache_and_weights():
    tokens,source=sample()
    s=select_lines(tokens,source,40,{10:0,20:0,30:0},keep_ratio=1)
    with pytest.raises(ValueError):
        compress_kv(((torch.zeros(1,2,7,3),torch.zeros(1,2,7,3)),),s)
    with pytest.raises(ValueError):
        select_lines(tokens,source,40,{},weights=Weights(1,1,0,0))


def test_attention_changes_choice_and_equal_scores_are_finite():
    tokens,source=sample()
    s=select_lines(tokens,source,40,{10:999,20:0,30:0},keep_ratio=.75,weights=Weights(0,0,0,1))
    assert s.selected_line_ids==[10,30,40]
    assert all(v['attention_normalized']==0 for v in s.scores.values())
