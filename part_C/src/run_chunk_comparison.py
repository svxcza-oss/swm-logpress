"""실제 32K 청크의 비Attention 비교와 기존 소형 Attention 비교를 분리 실행한다."""
import argparse
import bisect
import csv
import io
import json
from pathlib import Path

from chunk_line_selector import prepare_chunk_features,select_prepared_chunk,find_anchors
from line_kv_compressor import Weights,anchor_scores

ROOT=Path(__file__).resolve().parent
PILOT=ROOT/'examples/thunderbird_pilot_20261114_135108'
OUT=ROOT/'examples/chunk_comparison_20261120'


def evaluate(prepared,truth,chunk,cohort,with_attention=False):
    policies={
        'recent':(Weights(1,0,0,0),'end',None),
        'multi_structure':(Weights(0,.5,.5,0),'end',None),
        'multi_end_recent':(Weights(.6,.2,.2,0),'end',None),
        'multi_anchor_distance':(Weights(.6,.2,.2,0),'anchor',None),
        'keyword':(Weights(0,.5,.5,0),'end','keyword'),
    }
    if with_attention:
        policies.update(multi_attention=(Weights(0,0,0,1),'end',None),
                        multi_hybrid_no_recent=(Weights(0,.25,.25,.5),'end',None),
                        multi_hybrid_end=(Weights(.6,.2,.05,.15),'end',None),
                        multi_hybrid_anchor=(Weights(.6,.2,.05,.15),'anchor',None))
    results=[]
    for ratio in (.25,.5,.75):
        trials=[(name,*settings,0) for name,settings in policies.items()]
        trials += [('random',Weights(0,.5,.5,0),'end','random',seed) for seed in range(20)]
        for name,weights,recency,baseline,seed in trials:
            s=select_prepared_chunk(prepared,keep_ratio=ratio,weights=weights,recency_mode=recency,baseline=baseline,seed=seed)
            selected=set(s.selected_line_ids)
            assert s.used_tokens<=s.budget_tokens
            results.append(dict(chunk=chunk,cohort=cohort,method=name,seed=seed,budget_ratio=ratio,
                                alert_count=len(truth),kept_alert_ids=sorted(truth&selected),
                                missed_alert_ids=sorted(truth-selected),recall=len(truth&selected)/len(truth) if truth else None,
                                used_tokens=s.used_tokens,total_tokens=s.total_tokens,selected_line_ids=s.selected_line_ids))
    return results


def summarize(rows):
    result=[]
    for cohort in sorted({r['cohort'] for r in rows}):
        for method in sorted({r['method'] for r in rows if r['cohort']==cohort}):
            for ratio in (.25,.5,.75):
                subset=[r for r in rows if r['cohort']==cohort and r['method']==method and r['budget_ratio']==ratio]
                valid=[r for r in subset if r['recall'] is not None]
                result.append(dict(cohort=cohort,method=method,budget_ratio=ratio,
                    macro_recall=sum(r['recall'] for r in valid)/len(valid) if valid else None,
                    micro_recall=sum(len(r['kept_alert_ids']) for r in valid)/sum(r['alert_count'] for r in valid) if valid else None,
                    positive_chunks=len({r['chunk'] for r in valid}),total_chunks=len({r['chunk'] for r in subset}),
                    mean_used_ratio=sum(r['used_tokens']/r['total_tokens'] for r in subset)/len(subset)))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--small-attention',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    with (PILOT/'source.csv').open(encoding='utf-8-sig',newline='') as f:
        original={int(r['line_id']):r for r in csv.DictReader(f)}
    source={i:{k:v for k,v in row.items() if k!='label'} for i,row in original.items()}
    rows=[]
    manifest=[]
    if not args.small_attention:
        from tokenizers import Tokenizer
        snapshot_path=ROOT/'examples/thunderbird_audit/source_snapshot.json'
        snapshot=json.loads(snapshot_path.read_text(encoding='utf-8'))
        metadata=list(csv.DictReader(io.StringIO(snapshot['window_314927_metadata.csv'].lstrip('\ufeff'))))
        tokenizer_path=next(Path.home().glob('.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/*/tokenizer.json'))
        tokenizer=Tokenizer.from_file(str(tokenizer_path))
        for row in metadata:
            number=int(row['chunk'].replace('chunk',''))
            text=snapshot[f'window_314927_{row["chunk"]}.txt']
            lines=text.splitlines(keepends=True)
            start=int(row['start_line'])
            assert len(lines)==int(row['log_count'])
            starts=[]
            offset=0
            for n,line in enumerate(lines):
                assert line.rstrip().endswith(source[start+n]['content'].rstrip())
                starts.append(offset)
                offset+=len(line)
            encoded=tokenizer.encode(text,add_special_tokens=False)
            assert len(encoded.ids)==int(row['token_count'])
            tokens=[]
            for n,(a,b) in enumerate(encoded.offsets):
                idx=bisect.bisect_right(starts,a)-1
                assert idx>=0 and b> a
                boundary=starts[idx+1] if idx+1<len(starts) else len(text)
                assert not text[boundary:b].strip(), '토큰이 서로 다른 줄의 본문을 가로지릅니다.'
                tokens.append(dict(token_idx=n,line_id=start+idx))
            prepared=prepare_chunk_features(tokens,source)
            ids=list(prepared['groups'])
            # 선택용 특징을 만든 다음 평가 전용 정답 라벨을 확인한다.
            truth={i for i in ids if original[i]['label'] not in ('','-')}
            assert len(truth)==int(row['alert_count'])
            cohort='full_32k' if len(encoded.ids)>=32000 else 'tail_16k'
            manifest.append(dict(chunk=number,cohort=cohort,tokens=len(tokens),lines=len(ids),anchors=len(prepared['anchors']),
                                 end_is_anchor=ids[-1] in prepared['anchors'],alerts=len(truth),
                                 alerts_also_anchors=len(truth&set(prepared['anchors']))))
            rows.extend(evaluate(prepared,truth,number,cohort))
            if number==12:
                (OUT/'chunk12_token_map.jsonl').write_text(json.dumps(dict(window_idx=12,input_ids=encoded.ids,token_map=tokens))+'\n',encoding='utf-8')
        name='actual_chunks'
    else:
        import torch
        for w in map(json.loads,(PILOT/'windows_token_map.jsonl').read_text(encoding='utf-8').splitlines()):
            wid=w['window_idx']
            data=torch.load(PILOT/f'line_attention_{wid}.pt',map_location='cpu',weights_only=True)
            ids=list(dict.fromkeys(t['line_id'] for t in w['token_map']))
            assert data['window_idx']==wid and data['line_ids']==ids
            anchors=find_anchors(ids,source)
            attn={a:anchor_scores(data['line_attentions'],ids,a) for a in anchors}
            prepared=prepare_chunk_features(w['token_map'],source,attn)
            truth={i for i in ids if original[i]['label'] not in ('','-')}
            manifest.append(dict(chunk=wid,tokens=len(w['input_ids']),lines=len(ids),anchors=len(anchors),alerts=len(truth)))
            rows.extend(evaluate(prepared,truth,wid,'small_attention',True))
        name='small_attention'
    result=dict(manifest=manifest,summary=summarize(rows),rows=rows,
                note='완료된 청크 전체 평가. 모든 라인을 후보/정답 대상으로 사용하고 보호 라인은 없음. 32K Attention 실행은 하지 않음.')
    (OUT/f'{name}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(manifest=manifest,summary=result['summary']),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
