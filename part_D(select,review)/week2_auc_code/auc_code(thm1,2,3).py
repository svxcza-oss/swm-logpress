
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# 가설 1
df = pd.read_csv("killswitch_dataset.csv")

y_test = ['is_cause']
y_score = ['position_ratio']

fpr,tpr,threshold = roc_curve(y_test, y_score)

plt.plot(fpr,tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

print(roc_auc_score(y_test, y_score))

# %%
# 가설 2

df = pd.read_csv("killswitch_dataset.csv")

df['5min_grid'] = df['timestamp'] // 300 #5분은 300초

df = df.sort_values(['node', '5min_grid', 'template_id', 'timestamp'])

df['rank'] = df.groupby(['node', '5min_grid', 'template_id']).cumcount()
df['n'] = df.groupby(['node', '5min_grid', 'template_id'])[line_id].transform('size')

df = df[df['n'] >= 2]

y_score = df['rank'] / (df['n'] - 1)
y_test = df['is_cause']

fpr, tpr, threshold = roc_curve(y_test, y_score)

plt.plot(fpr,tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

print(roc_auc_score(y_test, y_score))

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# 가설 3 파일

# group = df.groupby(['node','grid_id_1h'])
# DataFrame.rank(self,  // 랭크함수 기본api
# 		axis = 0, # 기본값 0(index)으로, index 축을 기준으로 랭크가 계산됩니다. 
#     		method = 'average', # 동점을 가진 데이터들의 순위를 매기는 방법입니다.
#     		numeric_only = None, # True로 설정된 경우 숫자 열만 순위를 매깁니다. 
#     		na_option = 'keep', # NaN 값 순위를 매기는 방법입니다.
#     		ascending = True, # 오름차순, 내림차순 정렬인지 정합니다.
#     		pct = False) # 반환 된 순위를 백분위 수 형식으로 표시할지 여부입니다.
# df.groupby('그룹기준열')['순위매길열'].rank() 이건 그룹바이한 열을 기준으로 순서매김 공식

# 1. node + grid_id_1h로 묶어서 rank, 창크기(n) 계산
# 2. n >= 2인 줄만 남김 (1줄 창 제외)   ← 여기서 걸러짐
# 3. 남은 줄들로:
#    - 위치 = rank/(n-1)
#    - is_hardcase = 그 남은 줄들의 is_hardcase
# 4. roc_auc_score(그 is_hardcase, 그 위치)


df = pd.read_csv('positives_grid_1h_2h.csv')

df = df.sort_values(['node', 'grid_id_1h', 'timestamp'])

df['rank'] = df.groupby(['node', 'grid_id_1h']).cumcount()
df['n'] = df.groupby(['node', 'grid_id_1h'])['line_id'].transform('size')

df = df[df['n'] >= 2]

y_score = df['rank'] / (df['n'] - 1)
y_test = df['is_hardcase']

fpr,tpr,threshold = roc_curve(y_test, y_score)

plt.plot(fpr,tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('위치 AUC')
plt.show()

print(roc_auc_score(y_test, y_score)) # csv파일에서 불러와서 넣기, 내장함수

# %%
# 순열검정, p값
from tqdm import tqdm #진행율 보여주는 라이브러리

result_auc = roc_auc_score(y_test, y_score)

n_perm = 5000
count = 0

rng = np.random.default_rng(42)

for i in tqdm(range(n_perm)):
    shuffle = rng.permutation(y_test.values)
    perm_auc = roc_auc_score(shuffle, y_score)
    
    if abs(perm_auc - 0.5) >= abs(result_auc - 0.5):
        count += 1
        
p_value = (count + 1) / (n_perm + 1)

print(f"실제 AUC: {result_auc: .4f}")
print(f"p값: {p_value: .4f}")
print("신호 존재" if p_value < 0.05 else "신호 무의미")


# 서명
# | 트랙 | 역할 | 서명 | 날짜 |
# |---|---|---|---|
# | A | 인프라 | ||
# | B | 데이터 | | |
# | C | 매핑/압축 | | |
# | D | 리드/평가 | ||