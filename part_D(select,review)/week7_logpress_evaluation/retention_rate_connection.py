import pandas as pd, json, re, glob, os
import numpy as np
import os
from math import comb
from collections import defaultdict

CHUNK_NAME  = "chunk13"
COMPRESSION = 0.7     # 압축률 0.5 / 0.7 / 0.9
PARTIAL_AS  = 1       # 1=Any-token(KEEP·PARTIAL 생존), 0=Strict(KEEP만 생존)
ERROR_CATS  = {"VAPI"} 

# 원인 줄의 일부 토큰만 남아도 llm이 원인을 유추할수 있다고 보기때문에, partial도 1로 치환.(any-token)
# 다만 어떤 토큰이 남았는지의 기준으로 구분을 다 하지는 못한다.
# 그래서 strict(전부 남아야 생존)기준도 같이 돌려서 두 기준 다 확인한다.
# 0 | 1 로 나누는 이유는 멕네마 검정이 이진 비교법이기 때문에.

try:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass

def _find(kw, ext):
    h = [f for f in glob.glob(f"*{ext}") if kw in f.lower()]
    return h[0] if h else None
 
def _parse(path):
    data = open(path, encoding="utf-8", errors="replace").read()
    blocks, cur = {}, None
    for l in data.splitlines():
        m = re.match(r"압축률:\s*([\d.]+)", l)
        if m and "Line" not in l:
            cur = float(m.group(1)); blocks.setdefault(cur, {})
        lm = re.match(r"Line (\d+)\s*\|\s*(KEEP|PARTIAL|DROP|NO_TOKEN)", l)
        if lm and cur is not None:
            blocks[cur][int(lm.group(1))] = lm.group(2)
    return blocks
 
def _offset(result_file, chunk_txt):
    data = open(result_file, encoding="utf-8", errors="replace").read()
    first = re.search(r"Line 00\s*\|\s*\w+\s*\|\s*\d+/\d+\s*\|\s*(.+)", data).group(1)[:60]
    for i, line in enumerate(open(chunk_txt, encoding="utf-8", errors="replace").read().splitlines()):
        if line[:60] == first:
            return i
    return None

# ── 실행: 폴더에서 자동으로 다 찾아서 변수 생성 ──
_meta  = _find("metadata", ".csv")
_chunk = _find(CHUNK_NAME, ".txt")
_r = pd.read_csv(_meta)
_r = _r[_r["chunk"] == CHUNK_NAME].iloc[0]
_start = _r["start_line"]
_alerts = json.loads(_r["alerts"])                       # {절대줄: 카테고리}
_cat = {int(k) - _start: v for k, v in _alerts.items()}  # chunk 내부줄: 카테고리
cause = sorted(_cat.keys())    

_files = [f for f in glob.glob("*subchunk*.txt")
          if CHUNK_NAME not in f and "metadata" not in f.lower()]
_method_status = defaultdict(dict)   # {방식: {chunk내부줄: status}}
for f in _files:
    name = re.split(r"_subchunk", os.path.basename(f))[0]
    off = _offset(f, _chunk)
    if off is None:
        print(f"[!] {f}: 위치 매칭 실패, 건너뜀"); continue
    blk = _parse(f).get(COMPRESSION, {})
    for local, st in blk.items():
        _method_status[name][off + local] = st   # 로컬줄 → chunk13 전체줄
 
def _surv(status_map):
    return [1 if status_map.get(c) == "KEEP"
            else PARTIAL_AS if status_map.get(c) == "PARTIAL"
            else 0 for c in cause]
 
_methods = {n: _surv(s) for n, s in _method_status.items()}
 
# ── logpress / baseline 분리 (네 코드가 기대하는 형식) ──
_lp = next((n for n in _methods if "logpress" in n.lower()), None)
logpress = _methods.get(_lp, [0] * len(cause))
baseline = {n: v for n, v in _methods.items() if n != _lp}
 
# ── is_error / pos / strata ──
is_error = {c: (_cat[c] in ERROR_CATS) for c in cause}
_n = len(cause)
pos = {c: (cause.index(c) / (_n - 1) if _n > 1 else 0.0) for c in cause}
 
overall   = list(range(_n))
non_error = [i for i in range(_n) if not is_error[cause[i]]]
middle    = [i for i in range(_n) if 0.25 <= pos[cause[i]] <= 0.75]
strata = {"overall": overall, "non_error": non_error, "middle": middle}
 
print(f"[준비 완료] 압축률 {COMPRESSION} · PARTIAL={'Any' if PARTIAL_AS else 'Strict'}")
print(f"  chunk13 alert(원인) {len(cause)}개 · 방식: {list(_methods.keys())}")
if _lp is None:
    print("  ※ LogPress 결과 없음 → logpress는 전부 0 (baseline끼리만 의미 있음)")
for n, v in {"logpress": logpress, **baseline}.items():
    print(f"  {n:20s}: {sum(v)}/{len(v)} 보존")
print(f"  strata: overall={len(overall)} non_error={len(non_error)} middle={len(middle)}")
# ↓↓↓ 이 아래에 네 metric_KPI 셀들 그대로 ↓↓↓



def preservation(result, target): # 보존율 함수(결과, 타겟(압축기))
    if len(target) == 0:
        return 0.0
    return sum(1 for c in target if result[c]) / len(target) 
# target 안에 원소 c를 하나씩 꺼내서, 만약 c번째 인덱스 결과배열에서 값이 true(1)이라면 + 1 합계 구하기. / 타겟의 전체길이. 
# --> true가 전체 타겟에서 얼마나 남아있는가?의 비율.

print("-- 층별 보존율 -- ")
print(f"{'method':10}", *[f"{s:>10}" for s in strata]) #왼쪽기준으로 10칸 확보, *는 내용물 출력(리스트 언팩킹), :>10은 오른쪽 버전
for name, res in {"logpress" : logpress, **baseline}.items(): # **(딕셔너리 언팩킹)
    rates = [preservation(res, t) for t in strata.values()]
    print(f"{name:10}", *[f"{r:>10.2f}" for r in rates])
    
    
from math import comb

# n값은 동전 던지기 총 횟수(=의견이 엇갈린 횟수)
# k값은 동전 앞면이 나온 횟수 (=원인을 더 적게 맞힌 쪽 횟수)
# 조합으로 확률을 누적하는 과정 자체가 멕네마 검정 자체.
# 실력이 똑같다면? (0.5) 엇갈린 것이 우연인가?를 알아야함.

def binom_cdf_half(k,n): # 원인을 잘 못살린 값을 받아서 동전던지기 확률을 구함. 
    # 실력이 같으면 엇갈림은 운이라는 귀무가설에서 0.5 ** n. 즉 엇갈림이 동전던지기 확률인가를 구함.
    pmf = 0.5 ** n  # 동전을 n번 던지는데, 특정 결과 하나가 나올 확률. 그래서 0.5 n제곱 형태. 0번째 돌렸을때 확률.
    cdf = pmf # 0번(앞면 0개)일 때 확률부터 담고 시작. cdf에 첫 항 저장된 상태
    for i in range(1, k+1): # 0번~k번(앞면 0개부터 min개까지) 확률을 다 누적 = "이만큼 또는 더 극단적일 확률"
        pmf *= (n - i + 1) / i # 다음항 계산.(combiantion 점화식 자체) pmf(개별 확률질량함수)
        cdf += pmf # pmf의 누적(누적 확률질량함수)
    return cdf

def mcnemar(x,y, idx): #치우친 결과가 우연히 나올 확률? -> 누적 확률
    p = q = 0
    for i in idx:
        if x[i] != y[i]: #두쌍을 비교, 같은건 제외하고 값이 다를때만 카운트
            if x[i] == 1: # x가 1이면 p에 +1
                p += 1
            else: #아니면 q에 +1
                q += 1
    n = p+q # 얼마나 다른게 있었냐를 저장. ex) x = [1 , 0, 0, 1] y = [0, 1, 0, 1] --> n == 2
    if n == 0: return p,q,1.0 # 다른게 없었다면? (n == 0) 0,0,1.0 반환하게 됨.
    
    cdf = binom_cdf_half(min(p,q),n) # 원인을 더 잘 살리지 못한 쪽을 binom함수로 넘김. (큰값이 기준이던 작은값이던 결과는 수학적으로 동일. 계산이 편한 작은값 활용)
    p_value = min(2*cdf, 1.0) # 분포의 좌우대칭성 이용. 반쪽만 측정후, 2배. 왜?
    return p,q, p_value
# 왜 좌우대칭성이 되는가? 두 모델의 성능이 같다, 즉 비율이 같다는 가정에서 출발. 
# 확률이 정확히 0.5일때, 이항분포 그래프를 그리면 좌우대칭이 됨.(수학적 성질)
# 조합의 성질또한 대칭을 이룸. 
# 통계학에서는 이를 양측 검정이라 부르며, 검증하고자 하는 목표가 방향과 상관없이 극단적인 격차가 벌어지냐?이기 때문에 
# 구하기 쉬운 작은 값 쪽의 cdf를 구한뒤 단순 *2를 해주면 양쪽 극단의 확률을 포괄하는 p_value값이 완성됨.


#bh보정 전의 raw값들
results = []
print("BH 보정전 값들\n")

for b_name, base in baseline.items(): # 베이스라인 딕셔너리 분리해서 꺼내서
    for s_name, idx in strata.items(): # 모델 이름과, 번호로 나눠서
        p,q,p_value = mcnemar(logpress, base, idx) # 각 층별로 멕네마 검정 돌리기
        results.append([b_name, s_name, p, q, p_value]) # 결과를 results에 append
        print(f"{b_name:10} {s_name:10} p = {p} q = {q} p_value = {round(p_value,3)}") # 출력형식


def bh_fdr(pvals, alpha=0.05):
    m = len(pvals) # 테스트 1번 = p-value 1개 탄생 = 리스트에 1개 추가 = 테스트 횟수가 나옴
    order = sorted(range(m), key=lambda i: pvals[i])   # p값 오름차순 정렬
    p_adj = [0.0] * m # 테스트 횟수만큼 빈리스트 만들기
    prev = 1.0
    for rank in range(m, 0, -1):        # 뒤(큰 p)에서 앞으로
        i = order[rank - 1] # 원래 값 저장.
        prev = min(prev, pvals[i] * m / rank) #BH보정 공식 
        p_adj[i] = prev # 결과 저장
    return p_adj                        # ← 보정된 p값 반환

results = []
for b_name, base in baseline.items():
    for s_name, idx in strata.items():
        b,c,p = mcnemar(logpress, base, idx)
        results.append((b_name, s_name, b,c,p)) # 멕네마 거친것을 결과에 저장
        
p_adj = bh_fdr([r[4] for r in results], alpha = 0.05) # 보정 4번째 인덱스에 있는 값을 가져옴(p_value)

for r,pa in zip(results, p_adj):
    b_name, s_name, l_win, b_win , _ = r
    if l_win > b_win: verdict = f"logpress 우세 ({l_win} vs {b_win})"
    elif b_win > l_win: verdict = f"{b_name} 우세  ({b_win} vs {l_win})"
    else:
        verdict = "동률"
    print(f"[{s_name:10}] logpress vs {b_name:10} -> {verdict}, p_bh = {pa:.3f}")
        
hard = [(r,pa) for r, pa in zip(results, p_adj) if r[1] != "overall"]
passed = all(pa < 0.05 and r[2] > r[3] for r, pa in hard)
print("\nTier-1:", "Pass" if passed else "Fail(표본이 작으면 정상)")

