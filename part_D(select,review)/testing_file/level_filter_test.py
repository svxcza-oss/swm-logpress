import re
LEVEL_RE = re.compile(r"(ERROR|FAIL|FATAL|EXCEPTION|CRITICAL)", re.I)

def level_filter_preservation(df, cause_idx):
    kept = df[df["template"].str.contains(LEVEL_RE) |
              df["level"].str.upper().isin(["ERROR","FATAL","CRITICAL"])]
    kept_idx = set(kept["idx"])
    preserved = sum(1 for c in cause_idx if c in kept_idx)
    return preserved / len(cause_idx)
# 보존율 측정.. 원본 데이터와 얼마나 남아있는지 테스트(neazh)