# %%
import random #랜덤뽑기


# %%
def average(num):
    if len(num) == 0: # 존재하지 않을때
        return 0.0
    total = 0.0 #초기화
    for n in num: 
        total = total + n #누적합
    return total / len(num) #누적합을 전체 원소개수로 나눠서 평균 구하기

# %%
def make_candidate(line, attention, is_groundT): #후보줄을 딕셔너리로 반환해주기
    candidate = {}
    candidate["line"] = line #라인번호
    candidate["attention"] = attention #어텐션 점수
    candidate["is_groundT"] = is_groundT #GroundTrue/False (실제 답안지)
    return candidate

# %%
def score_anchor(candidate, anchor_line):
    return candidate["attention"] # 후보가 가진 어텐션 점수를 그대로 점수로 돌려줌.
# anchor가 이 후보줄에 얼마나 주목했나?

def score_recency(candidate, anchor_line):
    distance = anchor_line - candidate["line"]
    return -distance # 상대적 거리 재기. distance[1,3,5]라면, -를 붙이면 -1,-3,-5가 되는데 -1 > -3 > -5이므로 거리가 가까운값을 바로 알 수 있다.
# 로그에서 후보줄 <-> anchor 사이의 줄 간격. 몇줄 떨어졌나?
#앵커보다 앞에 있음. 그래서 음수로 비교.
# 즉 에러 줄에서 몇줄 위로 올라가야 후보냐를 나타냄.
# 가까운것을 의심. 보통 에러 직전에 원인이 있을 확률이 높다라는 가정.

def score_random(candidate, anchor_line):
    return random.random()
# 0-1 사이 아무 숫자를 뱉음. 후보들이 랜덤한 순서로 정렬. 우연 수준의 바닥을 보여줌

# %% [markdown]
# ## Anchor Evaluation — 방법별 비교 (우리 + baseline 4개)
# 
# | 방법 | 뭐냐 | 뭘 통제(검증)하냐 |
# |---|---|---|
# | **Anchor (우리)** | ERROR/WARN 줄을 query로 써서 그 어텐션으로 이전 줄 랭킹 | — (이게 증명하려는 주인공) |
# | **Position-matched INFO** | anchor를 ERROR/WARN이 아니라 비슷한 위치의 평범한 INFO 줄로 바꿔서 query | "ERROR/WARN이라서 잘 찾나? 아니면 그 위치 줄이면 아무나 잘 찾나?" → ERROR/WARN의 특별함 검증 |
# | **Recency** | 어텐션 무시, 그냥 가까운 줄 우선 | "어텐션이 찾나? 아니면 그냥 가까운 게 원인이라 가까운 거 찍으면 맞나?" |
# | **All-Query** | 한 줄만 안 쓰고 여러 줄의 query 어텐션을 평균 | "ERROR/WARN 한 줄에 집중하는 게 이득이냐? 그냥 다 평균내도 되냐?" |
# | **Random** | 무작위 | 우연 수준 바닥선 |
# 
# > baseline 4개는 각각 "어텐션이 아니라 사실 ○○ 때문 아니야?"라는 딴지를 하나씩 막는 장치.
# > INFO(위치) · Recency(가까움) · All-Query(한 줄 집중 무의미) · Random(우연) → 4개 다 이겨야 "ERROR/WARN 어텐션이 진짜 원인을 찾는다"가 증명됨.

# %%
def ranked_labels(candidates, anchor_lines, score_fn):  # (후보 딕셔너리, anchor의 줄 번호[기준점 줄번호], 점수를 체점하는 방식함수)
    scored_pairs = []
    for candidate in candidates: #각 후보를 각 점수산출 방식마다 계산.
        score = score_fn(candidate, anchor_lines)  #점수 매김
        is_groundT = candidate["is_groundT"] #정답 여부 확인
        scored_pairs.append((score, is_groundT)) #(점수, 정답)으로 묶어 담기
        
    scored_pairs.sort(reverse=True) # 점수 높은 순으로 정렬
    
    labels = []
    for pair in scored_pairs:
        is_groundT = pair[1] # 짝의 정답여부를 꺼냄
        if is_groundT == True:
            labels.append(1) #정답이면 1로 저장
        else:
            labels.append(0) # 오답이면 0으로 저장
    return labels

#후보들을 점수로 줄 세운 다음, 정답여부만 뽑아 0/1 리스트로 만듦

# %%
def count_ones(labels): #위에서 구한 정답라벨 개수 구하기(전체 정답 수)
    count = 0
    for v in labels:
        if v ==1:
            count = count +1
    return count

def recall_at_k(labels, k): #전체 정답중 top-k안에 든 비율
    total_gt = count_ones(labels)
    if total_gt == 0:
        return 0.0

    found = 0
    for i in range(k):
        if i < len(labels):
            if labels[i] == 1:
                found = found + 1
    return found / total_gt

def precision_at_k(labels,k): #top-k중 정답비율
    found = 0
    for i in range(k):
        if i<len(labels):
            if labels[i] == 1:
                found = found +1
    return found / k


def mrr(labels): # 첫 정답이 몇등인가? 바로 한번찾고 끝
    rank = 0
    for v in labels:
        rank = rank+1
        if v ==1:
            return 1.0/ rank #등수의 역수 돌려줌
    return 0.0

def average_precision(labels): # 정답 나올때마다 precision(정밀도) 재서 평균
    total_gt = count_ones(labels)
    if total_gt == 0:
        return 0.0
    
    hit = 0 #지금까지 만난 정답 수 
    precision_sum = 0.0 
    rank = 0 # 현재 몇등?
    for v in labels:
        rank = rank +1
        if v ==1: # 정답이면
            hit = hit + 1
            precision_here = hit / rank #현재 위치까지의 정답비율
            precision_sum = precision_sum + precision_here
    return precision_sum / total_gt


def evidence_auc(labels): #정답이 오답보다 위에인 비율
    gt_positions = []
    non_positions = []
    for i in range(len(labels)):
        if labels[i] == 1:
            gt_positions.append(i) #내림차순으로 어펜드
        else:
            non_positions.append(i)

    if len(gt_positions) == 0 or len(non_positions) == 0:
        return 0.5 #한쪽만 있으면 비교 불가
    
    wins = 0.0
    for g in gt_positions:
        for n in non_positions: # 모든 정답과 오답 짝비교
            if g < n:           # 정답이 더 위면
               wins = wins + 1.0 
            elif g == n:        # 같은자리(예시 데이터에는 없음)는 없지만 예외처리
                wins = wins + 0.5
                
    total_pairs = len(gt_positions) * len(non_positions)
    return wins/total_pairs
        

# %%
# 한 장애엔 앵커가 여러개 있을 수 있음 -> 앵커마다 0/1 리스트 하나씩

def incident_ap(anchor_label_lists): #장애 1개 안에서 앵커들 AP 평균 (AP = 앵커 리스트들)
    ap_list = []
    for labels in anchor_label_lists: #앵커 하나씩 꺼내기
        ap = average_precision(labels) # 그 앵커 AP계산
        ap_list.append(ap)
    return average(ap_list) # 앵커들 AP 평균 = 이 장애 점수

def MAP(incidents): # 전체 장애들
    incident_scores = []
    for inc in incidents: # 장애 하나씩 꺼내기
        incident_scores.append(incident_ap(inc)) # 그 장애 점수 계산
    return average(incident_scores) # 장애들 점수 평균

# %%
def delta_map(our_incidents, base_incidents):
    return MAP(our_incidents) - MAP(base_incidents)

def bootstrap_ci(our_incidents, base_incidents, B = 5000, seed = 0):
    rng = random.Random(seed)
    n = len(our_incidents)
    
    diffs = []
    for _ in range(B):
        chosen_our = []
        chosen_base = []
        for _ in range(n):
            idx = rng.randrange(n)
            chosen_our.append(our_incidents[idx])
            chosen_base.append(base_incidents[idx])
        diffs.append(delta_map(chosen_our, chosen_base))
    
    diffs.sort()
    low_index = int(0.025 * B)
    high_index = int(0.975 * B)
    ci_low = diffs[low_index]
    ci_high = diffs[high_index]
    return ci_low, ci_high

# %%
def decide(dmap_allq, p_allq, dmap_recency, p_recency):
    both_positive = (dmap_allq > 0) and (dmap_recency > 0)
    both_significant = (p_allq < 0.05) and (p_recency < 0.05)
    
    if both_positive and both_significant:
        return "GO"
    
    allq_ok = (dmap_allq > 0) and (p_allq < 0.05)
    recency_ok = (dmap_recency > 0) and (p_recency < 0.05)
    if allq_ok or recency_ok:
        return "CONDITIONAL"
    
    return "NO-GO"

# %%

if __name__ == "__main__":
    anchor_line = 6
    candidates = [
        make_candidate(1, 0.14, False),
        make_candidate(2, 0.34, True),     # 정답
        make_candidate(3, 0.09, False),
        make_candidate(4, 0.25, True),     # 정답
        make_candidate(5, 0.18, False),
    ]

    labels = ranked_labels(candidates, anchor_line, score_anchor)
    print("정렬된 정답여부(우리):", labels)
    print("  Recall@3 :", recall_at_k(labels, 3))
    print("  Prec@3   :", precision_at_k(labels, 3))
    print("  MRR      :", round(mrr(labels), 3))
    print("  AP       :", round(average_precision(labels), 3))
    print("  AUC      :", round(evidence_auc(labels), 3))

    labels_recency = ranked_labels(candidates, anchor_line, score_recency)
    print("정렬된 정답여부(recency):", labels_recency,
          " AP:", round(average_precision(labels_recency), 3))

