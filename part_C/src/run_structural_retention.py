"""저장된 라인 매핑으로 구조 정보 기반 보존을 비교한다. 모델/KV는 변경하지 않는다."""
import csv
import json
import re
import time
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent / 'examples/thunderbird_pilot_20261114_135108'
OUT = BASE / 'structural_retention'


def identifiers(text):
    """본문의 명시적 코드, 장치/노드 ID, 함수 이름만 사용한다. 경로·행 번호는 제외한다."""
    text = re.sub(r'\[[^\]]*\.(?:c|h):\d+\]', ' ', text.lower())
    found = set()
    for key, value in re.findall(r'\b(eventid|error code|return code|qpn|aid|class|node)\s*[:=]?\s*(-?0x[0-9a-f]+|-?[0-9a-f]+)\b', text):
        found.add(key + ':' + value)
    for word in re.findall(r'\b[a-z][a-z0-9_]*\b', text):
        if '_' in word or re.fullmatch(r'infinihost\d+', word):
            found.add(word)
    return found


def features(row, anchor):
    # 이 자료의 user 필드에는 호스트명이 들어 있다. 알 수 없는 #1#는 일치 처리하지 않는다.
    host = row['user'].lower()
    h = float(host not in ('', '-', '#1#') and host == anchor['user'].lower())
    component = lambda x: re.sub(r'\[\d+\]$', '', x.lower().strip())
    c = float(bool(component(row['component'])) and component(row['component']) == component(anchor['component']))
    a, b = identifiers(row['content']), identifiers(anchor['content'])
    j = len(a & b) / len(a | b) if a | b else 0.0
    return h, c, j, sorted(a & b)


def select(costs, protected, order, budget):
    chosen = set(protected)
    used = sum(costs[i] for i in chosen)
    assert used <= budget
    for i in order:
        if i not in chosen and used + costs[i] <= budget:
            chosen.add(i)
            used += costs[i]
    return sorted(chosen), used


def main():
    started = time.perf_counter()
    # 선택 함수에 라벨이 전달되지 않도록 읽은 직후 제거한다.
    with (BASE / 'source.csv').open(encoding='utf-8-sig', newline='') as f:
        source = {int(r['line_id']): {k: v for k, v in r.items() if k != 'label'} for r in csv.DictReader(f)}
    maps = [json.loads(x) for x in (BASE / 'windows_token_map.jsonl').read_text(encoding='utf-8').splitlines()]
    gt = {x['window_id']: x['anchors'][0] for x in map(json.loads, (BASE / 'selected_ground_truth.jsonl').read_text(encoding='utf-8').splitlines())}
    rows, audits = [], []
    for window in maps:
        wid = window['window_idx']
        costs = Counter(t['line_id'] for t in window['token_map'])
        assert None not in costs and sum(costs.values()) == len(window['input_ids'])
        ids = sorted(costs)
        anchor = ids[-1]
        assert anchor == gt[wid]['anchor_line_id']
        candidates = ids[:-1]
        scores = {m: {} for m in ('recent', 'host_component', 'identifiers', 'combined')}
        for pos, i in enumerate(candidates):
            r = (pos + 1) / len(candidates)
            h, c, j, overlap = features(source[i], source[anchor])
            # 정답을 보기 전에 고정한 가중치. 점수는 모두 0~1 범위다.
            scores['recent'][i] = r
            scores['host_component'][i] = .75*r + .25*(h+c)/2
            scores['identifiers'][i] = .75*r + .25*j
            scores['combined'][i] = .75*r + .125*(h+c)/2 + .125*j
            audits.append(dict(window_id=wid,line_id=i,anchor_id=anchor,cost=costs[i],recency=r,host=h,component=c,identifier_similarity=j,shared_identifiers=overlap))
        for ratio in (.25, .5, .75):
            budget = int(len(window['input_ids'])*ratio)
            for method, values in scores.items():
                order = sorted(candidates, key=lambda i: (-values[i], -i))
                chosen, used = select(costs, {ids[0], anchor}, order, budget)
                # 선택이 끝난 뒤 정답을 참조한다. Anchor 자신은 평가 대상에서 제외한다.
                truth = set(gt[wid]['alert_line_ids'])
                kept = sorted(truth & set(chosen))
                rows.append(dict(window_id=wid,method=method,budget_ratio=ratio,budget_tokens=budget,used_tokens=used,total_tokens=len(window['input_ids']),selected_line_ids=chosen,kept_alert_ids=kept,missed_alert_ids=sorted(truth-set(chosen)),alert_count=len(truth),recall=len(kept)/len(truth) if truth else None))
    # 이전 시뮬레이션의 최근순 선택과 48개 조건 모두 일치하는지 확인한다.
    old = json.loads((BASE/'token_budget_mixture/simulation.json').read_text(encoding='utf-8'))['rows']
    old = {(r['window_id'],r['budget_ratio']):r for r in old if r['recent_ratio']==1}
    for r in rows:
        if r['method']=='recent':
            assert r['selected_line_ids']==old[r['window_id'],r['budget_ratio']]['selected_line_ids']
    summary = []
    for method in scores:
        for ratio in (.25,.5,.75):
            subset=[r for r in rows if r['method']==method and r['budget_ratio']==ratio]
            valid=[r for r in subset if r['recall'] is not None]
            summary.append(dict(method=method,budget_ratio=ratio,macro_recall=sum(r['recall'] for r in valid)/len(valid),kept_alerts=sum(len(r['kept_alert_ids']) for r in valid),alert_occurrences=sum(r['alert_count'] for r in valid),mean_used_ratio=sum(r['used_tokens']/r['total_tokens'] for r in subset)/len(subset)))
    OUT.mkdir(exist_ok=True)
    result=dict(summary=summary,rows=rows,features=audits,elapsed_seconds=time.perf_counter()-started,baseline_verified_conditions=48)
    (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
