import pandas as pd 
import numpy as np
from sklearn.metrics import roc_auc_score

#창 안에서 음성 뽑음.
df = pd.read_csv("positives_grid_1h_2h.csv")

df = df.sort_values(['node', 'grid_id_1h', 'timestamp'])

df['rank'] = df.groupby(['node', 'grid_id_1h']).cumcount()
df['n'] = df.groupby(['node', 'grid_id_1h'])['line_id'].transform('size')

df = df[df['n'] >= 2]

df['position'] = df['rank'] / (df['n'] - 1)

def matching_auc(df, ratio, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    
    for (node, grid), g in df.groupby(['node', 'grid_id_1h']):
        pos = g[g['is_hardcase'] == 1]
        neg = g[g['is_hardcase'] == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        need = len(pos) * ratio
        if len(neg) >= need:
            take = neg.sample(need, random_state = rng.integers(1e9))
        else:
            take = neg
        rows.append(pd.concat([pos, take]))
    matched = pd.concat(rows)
    auc = roc_auc_score(matched['is_hardcase'], matched['position'])
    return auc, len(matched[matched['is_hardcase'] == 1]), len(matched[matched['is_hardcase'] == 0])


for ratio in [1,4]:
    auc, npos, nneg = matching_auc(df, ratio)
    print(f"1:{ratio} AUC = {auc:.4f} |0.5(auc 기준) 차이| = {abs(auc-0.5):.4f} 양성개수 = {npos} 음성개수 = {nneg}")