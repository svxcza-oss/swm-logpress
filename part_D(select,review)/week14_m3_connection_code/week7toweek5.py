#  week7 (측정) → week5 (판정) 연결 코드

import pandas as pd, json, re, glob, os
from collections import defaultdict

# ---------------- 설정 ----------------
BASE_DIR    = "/Users/mac/무제 폴더/swm-logpress/part_D(select,review)/week7_logpress_evaluation/chunk_library"        # ★빈칸A: chunk_library 폴더 경로 (네 환경/대여 서버에 맞게)
CHUNK_NAME  = "chunk13"
COMPRESSION = 0.6           # 압축률(= 버리는 비율) 0.5 / 0.7 / 0.9 중 하나
KEEP_RATIO  = 1 - COMPRESSION   # 보존비(= 남길 라인 비율) = top-k의 k   
ERROR_CATS  = {"VAPI"}


#  PART 1 — 측정 (week7): 압축기 출력 txt → 생존 배열
def _parse(path):
    """결과 txt를 {압축률: {줄번호: (생존토큰수, 전체토큰수)}} 로 파싱
       (예전엔 KEEP/DROP 상태만 뽑았지만, 이제 n/m 토큰 수를 뽑아 '비율'을 만든다)"""
    data = open(path, encoding="utf-8", errors="replace").read()
    blocks = {}
    cur = None
    for l in data.splitlines():
        m = re.match(r"압축률:\s*([\d.]+)", l)
        if m and "Line" not in l:
            cur = float(m.group(1))
            blocks.setdefault(cur, {})
        # 줄 형식:  Line NN | STATUS | n/m | 본문   →  n(생존), m(전체) 을 뽑는다
        lm = re.match(r"Line (\d+)\s*\|\s*\w+\s*\|\s*(\d+)/(\d+)", l)
        if lm and cur is not None:
            line_no = int(lm.group(1))
            kept    = int(lm.group(2))   # 생존 토큰 수 n
            total   = int(lm.group(3))   # 전체 토큰 수 m
            blocks[cur][line_no] = (kept, total)
    return blocks

def _offset(result_file, chunk_txt):
    """결과 파일의 Line 0 이 원본 chunk의 몇 번째 줄에서 시작하는지 찾기"""
    data = open(result_file, encoding="utf-8", errors="replace").read()
    first = re.search(r"Line 00\s*\|\s*\w+\s*\|\s*\d+/\d+\s*\|\s*(.+)", data).group(1)[:60]
    for i, line in enumerate(open(chunk_txt, encoding="utf-8", errors="replace").read().splitlines()):
        if line[:60] == first:
            return i
    return None

# --- 정답(원인 줄) 메타 로드 : 이게 ground truth(측정 아님, 데이터셋 라벨) ---
_meta  = f"{BASE_DIR}/window_314927_metadata.csv"
_chunk = f"{BASE_DIR}/window_314927_chunk13.txt"
_r = pd.read_csv(_meta)
_r = _r[_r["chunk"] == CHUNK_NAME].iloc[0]
_start  = _r["start_line"]
_alerts = json.loads(_r["alerts"])                      # {절대줄: 카테고리}
_cat    = {int(k) - _start: v for k, v in _alerts.items()}   # chunk 내부줄: 카테고리
cause   = sorted(_cat.keys())                           # 원인 줄 번호들(정답지)

# --- 각 방법의 '라인별 생존토큰 비율' 모으기 ---
_files = [f for f in glob.glob(f"{BASE_DIR}/*subchunk*.txt")
          if CHUNK_NAME not in f and "metadata" not in f.lower()]
_method_ratio = defaultdict(dict)       # {방식: {chunk내부줄: 생존비율}}
for f in _files:
    name = re.split(r"_subchunk", os.path.basename(f))[0]
    off = _offset(f, _chunk)
    if off is None:
        print(f"[!] {f}: 위치 매칭 실패, 건너뜀"); continue
    blk = _parse(f).get(COMPRESSION, {})
    for local, km in blk.items():
        kept, total = km
        ratio = kept / total if total > 0 else 0.0   # ← 핵심: 비율 = 생존/전체
        _method_ratio[name][off + local] = ratio

def _surv(ratio_map):
    """[새 방식] 라인별 '생존토큰 비율'로 줄 세워, 상위 KEEP_RATIO 만큼 라인을 보존으로 친다.
       원인 줄이 그 상위 k 안에 들면 1, 아니면 0.  (McNemar가 이진이라 0/1로 떨어뜨림)"""
    # 1) (비율, 줄번호) 쌍을 만들어 비율 큰 순으로 정렬  (람다 안 씀: 튜플은 앞값으로 정렬됨)
    pairs = []
    for line_no in ratio_map:
        pairs.append((ratio_map[line_no], line_no))
    pairs.sort(reverse=True)

    # 2) 전체 라인 중 상위 몇 줄 남길지 = 라인 수 × 보존비(KEEP_RATIO)
    n_keep = int(round(len(pairs) * KEEP_RATIO))

    # 3) 상위 n_keep 줄의 '줄번호'만 모은다 = 보존된 라인 집합
    kept_lines = set()
    for i in range(n_keep):
        kept_lines.add(pairs[i][1])

    # 4) 원인 줄마다: 보존 집합에 들어 있으면 1, 아니면 0
    out = []
    for c in cause:
        if c in kept_lines:
            out.append(1)
        else:
            out.append(0)
    return out

_methods = {n: _surv(s) for n, s in _method_ratio.items()}

# --- logpress / baseline 분리 ---
_lp      = next((n for n in _methods if "logpress" in n.lower()), None)
logpress = _methods.get(_lp, [0] * len(cause))          # LogPress 없으면 전부 0
baseline = {n: v for n, v in _methods.items() if n != _lp}

# --- 층(strata) 나누기 ---
is_error  = {c: (_cat[c] in ERROR_CATS) for c in cause}
_n        = len(cause)
pos       = {c: (cause.index(c) / (_n - 1) if _n > 1 else 0.0) for c in cause}
overall   = list(range(_n))
non_error = [i for i in range(_n) if not is_error[cause[i]]]
middle    = [i for i in range(_n) if 0.25 <= pos[cause[i]] <= 0.75]
strata    = {"overall": overall, "non_error": non_error, "middle": middle}

print(f"[측정 완료] 압축률 {COMPRESSION} · 원인 줄 {len(cause)}개 · 방식 {list(_methods.keys())}")



#  PART 2 — 판정 엔진 (week5): McNemar + BH-FDR  (이미 완성, 그대로)
def binom_cdf_half(k, n):
    pmf = 0.5 ** n
    cdf = pmf
    for i in range(1, k + 1):
        pmf *= (n - i + 1) / i
        cdf += pmf
    return cdf

def mcnemar(x, y, idx):
    p = q = 0
    for i in idx:
        if x[i] != y[i]:                 # 엇갈린 것만 (같으면 무시)
            if x[i] == 1:                
                p += 1
            else:
                q += 1
    n = p + q
    if n == 0:
        return p, q, 1.0
    cdf     = binom_cdf_half(min(p, q), n)
    p_value = min(2 * cdf, 1.0)         
    return p, q, p_value

def bh_fdr(pvals, alpha=0.05):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    p_adj = [0.0] * m
    prev  = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        prev = min(prev, pvals[i] * m / rank)
        p_adj[i] = prev
    return p_adj


#  PART 3 — 연결 (핵심 고리): 측정 배열 → 판정
results = []
for b_name, base in baseline.items():
    for s_name, idx in strata.items():
        b, c, p = mcnemar(logpress, base, idx)  
        results.append((b_name, s_name, b, c, p))

pvals = []
for r in results:
    pvals.append(r[4])          # r = (b_name, s_name, b, c, p) → p값은 인덱스 4
p_adj = bh_fdr(pvals, alpha=0.05)               # 모든 검정 p값 모아 BH 보정

for r, pa in zip(results, p_adj):
    b_name, s_name, l_win, b_win, _ = r
    verdict = ("logpress 우세" if l_win > b_win else
               f"{b_name} 우세"  if b_win > l_win else "동률")
    print(f"[{s_name:10}] logpress vs {b_name:12} -> {verdict:16} p_bh = {pa:.3f}")

# Tier-1 판정 (hard-case = overall 제외)
hard   = [(r, pa) for r, pa in zip(results, p_adj) if r[1] != "overall"]
passed = all(pa < 0.05 and r[2] > r[3] for r, pa in hard)
print("\nTier-1:", "Pass" if passed else "Fail (표본 작으면 정상)")