#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LogPress — 킬스위치 실험 (Role D, 1~3주차 / 관문 ①)
====================================================

GPU 불필요. 인텔 맥 CPU에서 그대로 돌아간다.

이 스크립트가 답하는 질문
-------------------------
  Q1. level 필터(ERROR/FAIL/...)만으로 원인 라인을 몇 % 보존하는가?
  Q2. 필터가 놓친 원인을 rarity(희귀도) + position(위치) 신호로 건질 수 있는가?
  Q3. 하드케이스 표본이 통계적으로 충분한가?

  → 세 답을 합쳐 GO / NO-GO 를 판정한다.

판정식 (사전 확정 — 결과 보고 바꾸면 p-hacking)
------------------------------------------------
  GO ①  n_hard >= 300
  GO ②  결합신호 AUC >= 0.65  AND  순열검정 p < 0.05
  보조   level-filter 보존율 < 0.99

사용법
------
  # 1) 데모 (합성 로그로 파이프라인 검증 — 데이터 없어도 지금 바로 실행)
  python killswitch.py --demo

  # 2) 실제 데이터
  python killswitch.py --input data/BGL.log --format bgl

설치
----
  pip install drain3 pandas numpy scipy scikit-learn statsmodels matplotlib
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────
# 판정식 상수 — 실험 시작 전에 팀 합의로 확정할 것
# ─────────────────────────────────────────────────────────────
N_HARD_MIN = 300      # GO ①  하드케이스 최소 표본
AUC_MIN = 0.65        # GO ②  결합신호 최소 판별력
ALPHA = 0.05          # GO ②  유의수준
LEVEL_PRES_MAX = 0.99 # 보조   이 이상이면 LogPress의 여지가 없음

N_PERM = 5000         # 순열검정 반복
SEED = 42

# ERROR 계열 판정용 정규식 (0단계에서 팀이 확정한 키워드)
LEVEL_RE = re.compile(r"\b(?:ERROR|ERR|FAIL|FAILED|FAILURE|FATAL|EXCEPTION|CRITICAL|SEVERE|PANIC)\b", re.I)
ERROR_LEVELS = {"ERROR", "ERR", "FATAL", "CRITICAL", "SEVERE", "PANIC"}


# ═════════════════════════════════════════════════════════════
# 1. 로그 파싱  (Drain3)
# ═════════════════════════════════════════════════════════════
def build_miner():
    """Drain3 TemplateMiner 생성. 스트리밍 파서라 메모리를 거의 안 쓴다."""
    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig

    cfg = TemplateMinerConfig()
    cfg.load("drain3.ini")          # 파일 없으면 기본값 사용 (sim_th=0.4, depth=4)
    cfg.profiling_enabled = False
    return TemplateMiner(cfg)


def parse_logs(records: list[dict]) -> pd.DataFrame:
    """
    records: [{"incident_id", "line_idx", "level", "message", "is_cause"}, ...]
    return : 위 컬럼 + cluster_id, template
    """
    miner = build_miner()
    rows = []
    for r in records:
        res = miner.add_log_message(r["message"])
        rows.append({
            **r,
            "cluster_id": res["cluster_id"],
            "template": res["template_mined"],
        })
    df = pd.DataFrame(rows)
    print(f"[parse] {len(df):,} 라인 → 템플릿 {df['cluster_id'].nunique():,}종")
    return df


# ═════════════════════════════════════════════════════════════
# 2. 신호 계산  (rarity / position)
# ═════════════════════════════════════════════════════════════
def add_rarity(df: pd.DataFrame) -> pd.DataFrame:
    """
    rarity(t) = log( N / df(t) )      = 자기정보량 log(1/p)

    희귀한 템플릿일수록 값이 크다.
    가정: "장애의 원인 라인은 대개 희귀하다"
    ⚠️ 전체 코퍼스 기준으로 계산 (인시던트별 아님)
    """
    N = len(df)
    doc_freq = df.groupby("cluster_id")["line_idx"].transform("count")
    df["rarity"] = np.log(N / doc_freq)
    return df


def add_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    pos      = 인시던트 내 정규화 위치 [0, 1]
    mid_dist = |pos - 0.5|   → 0에 가까울수록 '중간'(= lost-in-the-middle 영역)

    LogPress는 중간을 살리려는 것이므로, 신호로는 mid_dist를 쓴다(작을수록 원인 가능성↑).
    """
    g = df.groupby("incident_id")["line_idx"]
    lo, hi = g.transform("min"), g.transform("max")
    span = (hi - lo).replace(0, 1)
    df["pos"] = (df["line_idx"] - lo) / span
    df["mid_dist"] = (df["pos"] - 0.5).abs()
    return df


def add_is_error(df: pd.DataFrame) -> pd.DataFrame:
    """level 필터가 잡아내는 라인인가."""
    by_level = df["level"].astype(str).str.upper().isin(ERROR_LEVELS)
    by_regex = df["message"].astype(str).str.contains(LEVEL_RE, na=False)
    df["is_error"] = by_level | by_regex
    return df


def add_hard_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    하드케이스 = level 필터가 놓치기 쉬운 원인 라인
      (a) 비-ERROR 원인            → 필터에 아예 안 걸림
      (b) 중간 위치(25~75%) 원인   → lost-in-the-middle
    """
    df["hard_nonerror"] = df["is_cause"] & ~df["is_error"]
    df["hard_middle"] = df["is_cause"] & df["pos"].between(0.25, 0.75)
    df["is_hard"] = df["hard_nonerror"] | df["hard_middle"]
    return df


# ═════════════════════════════════════════════════════════════
# 3. GO 조건 — 보조: level 필터 보존율
# ═════════════════════════════════════════════════════════════
def level_filter_preservation(df: pd.DataFrame) -> float:
    """level 필터만 썼을 때 원인 라인의 보존율."""
    causes = df[df["is_cause"]]
    if len(causes) == 0:
        raise ValueError("원인 라인이 하나도 없습니다. is_cause 라벨을 확인하세요.")
    return float(causes["is_error"].mean())


# ═════════════════════════════════════════════════════════════
# 4. GO 조건 ② — 신호 판별력 (AUC + 순열검정)
# ═════════════════════════════════════════════════════════════
def auc_permutation_test(y, score, n_perm=N_PERM, seed=SEED):
    """
    라벨을 무작위로 섞어 만든 귀무분포와 관측 AUC를 비교.
    분포 가정이 없고 어떤 지표에도 쓸 수 있는 게 장점.
    """
    rng = np.random.default_rng(seed)
    obs = roc_auc_score(y, score)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = roc_auc_score(rng.permutation(y), score)
    p = (np.sum(null >= obs) + 1) / (n_perm + 1)   # +1 보정 (p=0 방지)
    return float(obs), float(p), null


def single_signal_auc(df: pd.DataFrame, col: str, higher_is_cause=True):
    """
    단일 신호의 판별력.
    Mann-Whitney U 는 AUC 와 수학적으로 동치(AUC = U / (n1·n2)).
    정규성 가정이 없어 로그 데이터에 적합.
    효과크기는 rank-biserial 상관으로 보고.
    """
    from scipy.stats import mannwhitneyu

    y = df["is_cause"].astype(int).values
    x = df[col].values
    if not higher_is_cause:
        x = -x

    n1, n0 = int(y.sum()), int((1 - y).sum())
    U, p = mannwhitneyu(x[y == 1], x[y == 0], alternative="greater")
    auc = U / (n1 * n0)
    r_rb = 2 * auc - 1                      # rank-biserial (0=무판별, 1=완전)
    return float(auc), float(p), float(r_rb)


def combined_auc(df: pd.DataFrame, seed=SEED):
    """
    rarity + mid_dist 를 로지스틱 회귀로 결합한 판별력.

    ⚠️ 라벨 누수 방지 — 반드시 인시던트 단위로 dev/test 분리.
       (같은 인시던트가 dev와 test에 모두 들어가면 누수)
    """
    feats = ["rarity", "mid_dist"]
    X = df[feats].values
    y = df["is_cause"].astype(int).values
    groups = df["incident_id"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    dev_idx, test_idx = next(gss.split(X, y, groups))

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000),
    )
    clf.fit(X[dev_idx], y[dev_idx])                       # dev 에서만 학습
    score_test = clf.predict_proba(X[test_idx])[:, 1]     # test 에서만 평가

    auc, p, _ = auc_permutation_test(y[test_idx], score_test, seed=seed)

    lr = clf.named_steps["logisticregression"]
    coefs = dict(zip(feats, lr.coef_[0]))
    return auc, p, coefs, len(test_idx)


# ═════════════════════════════════════════════════════════════
# 5. 검정력 분석 — "왜 300개인가"
# ═════════════════════════════════════════════════════════════
def power_analysis(p_logpress=0.80, p_baseline=0.65, power=0.80, alpha=ALPHA):
    """
    보존율 p_baseline → p_logpress 개선을 검출하려면 표본이 몇 개 필요한가.
    statsmodels 가 없으면 정규근사로 직접 계산 (동일 결과).
    """
    try:
        from statsmodels.stats.power import NormalIndPower
        from statsmodels.stats.proportion import proportion_effectsize
        es = proportion_effectsize(p_logpress, p_baseline)
        n = NormalIndPower().solve_power(es, power=power, alpha=alpha, ratio=1.0)
        return float(n), float(es)
    except ImportError:
        from scipy.stats import norm
        h = 2 * np.arcsin(np.sqrt(p_logpress)) - 2 * np.arcsin(np.sqrt(p_baseline))
        z_a, z_b = norm.ppf(1 - alpha / 2), norm.ppf(power)
        n = ((z_a + z_b) / h) ** 2
        return float(n), float(h)


# ═════════════════════════════════════════════════════════════
# 6. 판정
# ═════════════════════════════════════════════════════════════
@dataclass
class Verdict:
    n_hard: int
    level_pres: float
    auc: float
    p_value: float
    coefs: dict = field(default_factory=dict)

    @property
    def go1(self) -> bool:
        return self.n_hard >= N_HARD_MIN

    @property
    def go2(self) -> bool:
        return (self.auc >= AUC_MIN) and (self.p_value < ALPHA)

    @property
    def aux(self) -> bool:
        return self.level_pres < LEVEL_PRES_MAX

    @property
    def go(self) -> bool:
        return self.go1 and self.go2 and self.aux


def report(v: Verdict, df: pd.DataFrame):
    ok = lambda b: "\033[92m✅ PASS\033[0m" if b else "\033[91m❌ FAIL\033[0m"
    bar = "═" * 62

    n_causes = int(df["is_cause"].sum())
    n_nonerr = int(df["hard_nonerror"].sum())
    n_mid = int(df["hard_middle"].sum())

    print(f"\n{bar}\n  킬스위치 판정 (관문 ①)\n{bar}")
    print(f"  전체 라인          {len(df):>8,}")
    print(f"  원인 라인          {n_causes:>8,}")
    print(f"    ├ 비-ERROR 원인  {n_nonerr:>8,}")
    print(f"    └ 중간위치 원인  {n_mid:>8,}")
    print(f"    └ 하드케이스(합) {v.n_hard:>8,}   ← 중복 제거")
    print(f"\n{'─'*62}")

    print(f"\n  [보조] level 필터 보존율")
    print(f"         {v.level_pres:.1%}   (기준: < {LEVEL_PRES_MAX:.0%})   {ok(v.aux)}")
    if not v.aux:
        print("         → 필터만으로 거의 다 잡힘. LogPress의 여지 없음 = 주제 재설정 신호")

    print(f"\n  [GO ①] 하드케이스 표본 수")
    print(f"         n_hard = {v.n_hard:,}   (기준: >= {N_HARD_MIN})   {ok(v.go1)}")
    if not v.go1:
        print("         → 대응: BGL 병합 / 층화 완화 / 라인 단위 검정")

    print(f"\n  [GO ②] 신호 판별력 (rarity + position)")
    print(f"         AUC = {v.auc:.4f}   (기준: >= {AUC_MIN})")
    print(f"         순열검정 p = {v.p_value:.4f}   (기준: < {ALPHA})   {ok(v.go2)}")
    print(f"         계수: " + ", ".join(f"{k}={val:+.3f}" for k, val in v.coefs.items()))
    if not v.go2:
        print("         → 대응: 신호 재설계(시간간격·버스트성 추가) 또는 주제 리셋")

    n_req, es = power_analysis()
    print(f"\n  [참고] 검정력 분석")
    print(f"         보존율 0.65 → 0.80 검출 (power=0.8, α=0.05)")
    print(f"         필요 표본 ≈ {n_req:.0f} / 그룹   (효과크기 h={es:.3f})")

    print(f"\n{bar}")
    if v.go:
        print("  \033[92m▶ GO — 3개 조건 모두 통과. 다음 단계 진행.\033[0m")
    else:
        print("  \033[91m▶ NO-GO — 대응 회의 소집 (D 주관).\033[0m")
    print(f"{bar}\n")


# ═════════════════════════════════════════════════════════════
# 7. 시각화
# ═════════════════════════════════════════════════════════════
def _setup_font():
    """한글 폰트 자동 탐색 (macOS / Windows / Linux). 없으면 영문 라벨로 폴백."""
    import matplotlib
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("AppleGothic", "Malgun Gothic", "NanumGothic",
                 "Noto Sans CJK KR", "Noto Sans KR"):
        if cand in have:
            matplotlib.rcParams["font.family"] = cand
            matplotlib.rcParams["axes.unicode_minus"] = False   # 한글폰트 마이너스 깨짐 방지
            return True
    return False


def plot(df: pd.DataFrame, out="killswitch.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    ko = _setup_font()
    T = (lambda k, e: k if ko else e)   # 한글 폰트 있으면 한글, 없으면 영문

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    cause = df[df["is_cause"]]
    noise = df[~df["is_cause"]]

    # (1) rarity 분포
    ax = axes[0]
    bins = np.linspace(0, df["rarity"].max(), 40)
    ax.hist(noise["rarity"], bins=bins, alpha=.55, density=True, label="non-cause", color="#94a3b8")
    ax.hist(cause["rarity"], bins=bins, alpha=.75, density=True, label="cause", color="#dc2626")
    ax.set_xlabel("rarity  = log(N / df)"); ax.set_ylabel("density")
    ax.set_title(T("① 희귀도 분포", "(1) Rarity distribution")); ax.legend()

    # (2) 위치 분포
    ax = axes[1]
    bins = np.linspace(0, 1, 21)
    ax.hist(noise["pos"], bins=bins, alpha=.55, density=True, label="non-cause", color="#94a3b8")
    ax.hist(cause["pos"], bins=bins, alpha=.75, density=True, label="cause", color="#dc2626")
    ax.axvspan(.25, .75, alpha=.10, color="#2563eb")
    ax.text(.5, ax.get_ylim()[1]*.93, T("중간 25~75%", "middle 25-75%"), ha="center", fontsize=9, color="#2563eb")
    ax.set_xlabel("normalized position"); ax.set_title(T("② 위치 분포", "(2) Position distribution")); ax.legend()

    # (3) ROC
    ax = axes[2]
    y = df["is_cause"].astype(int).values
    for col, sign, name, c in [("rarity", 1, "rarity", "#dc2626"),
                               ("mid_dist", -1, "position", "#2563eb")]:
        s = sign * df[col].values
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, label=f"{name}  (AUC={roc_auc_score(y, s):.3f})", color=c)
    ax.plot([0, 1], [0, 1], "k--", lw=.8, alpha=.5, label="random")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(T("③ 신호별 ROC", "(3) ROC by signal")); ax.legend()

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"[plot] 저장 → {out}")


# ═════════════════════════════════════════════════════════════
# 8. 데이터 로더
# ═════════════════════════════════════════════════════════════
def load_bgl(path, window=200):
    """
    BGL/Thunderbird 형식:  <label> <ts> <date> <node> ... <message>
    label 이 '-' 면 정상, 아니면 alert.

    ⚠️ 중요: BGL/Thunderbird는 '이상' 라벨이지 '원인' 라벨이 아니다.
       킬스위치용 프록시로만 쓰고, 진짜 원인 검증은 Nezha에서 한다.
    """
    records, inc, idx = [], 0, 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(maxsplit=9)
            if len(parts) < 10:
                continue
            label, msg = parts[0], parts[9]
            lvl = "ERROR" if label != "-" else "INFO"
            m = re.search(r"\b(INFO|WARN(?:ING)?|ERROR|FATAL|DEBUG)\b", msg, re.I)
            if m:
                lvl = m.group(1).upper()
            records.append({
                "incident_id": inc, "line_idx": idx,
                "level": lvl, "message": msg,
                "is_cause": label != "-",       # 프록시 라벨
            })
            idx += 1
            if idx % window == 0:              # 고정 윈도우 = 가짜 인시던트
                inc += 1
    print(f"[load] {len(records):,} 라인 / {inc+1:,} 윈도우")
    return records


def make_demo(n_inc=400, lines=120, seed=SEED):
    """
    합성 로그 — 데이터 없이 파이프라인을 지금 바로 검증하기 위한 것.
    실제 데이터의 성질을 흉내낸다:
      · 원인의 40%만 ERROR (나머지는 INFO/WARN)   → 하드케이스 생성
      · 원인은 희귀 템플릿에서 더 자주 나옴        → rarity 신호
      · 원인은 중간 위치에 몰림                     → position 신호
    """
    rng = np.random.default_rng(seed)
    common = [f"heartbeat node-{{}} ok seq {{}}", "cache hit ratio {} pct",
              "request served {} ms", "gc pause {} ms", "connection pool size {}"]
    rare = ["nvme timeout on device {}", "ecc uncorrectable error addr {}",
            "config reload applied version {}", "quota exceeded for tenant {}",
            "clock skew detected {} ms", "route flap detected peer {}"]

    recs = []
    for inc in range(n_inc):
        # 원인 위치: 중간에 몰리게 (Beta(3,3) → 0.5 근처)
        c_pos = float(rng.beta(3, 3))
        c_idx = int(np.clip(c_pos * (lines - 1), 0, lines - 1))
        for i in range(lines):
            is_cause = (i == c_idx)
            if is_cause:
                # 원인의 75%는 희귀 템플릿에서
                tpl = rng.choice(rare) if rng.random() < 0.75 else rng.choice(common)
                # 원인의 40%만 ERROR → 60%가 하드케이스(비-ERROR)
                lvl = "ERROR" if rng.random() < 0.40 else rng.choice(["INFO", "WARN"])
            else:
                # 잡음의 8%도 희귀 템플릿 → '드문 무해' (진짜 난제)
                tpl = rng.choice(rare) if rng.random() < 0.08 else rng.choice(common)
                lvl = rng.choice(["INFO", "WARN", "ERROR"], p=[.80, .15, .05])
            msg = f"{lvl} " + tpl.format(*[rng.integers(1, 9999) for _ in range(tpl.count('{}'))])
            recs.append({"incident_id": inc, "line_idx": i,
                         "level": lvl, "message": msg, "is_cause": is_cause})
    print(f"[demo] 합성 로그 {len(recs):,} 라인 / {n_inc} 인시던트")
    return recs


# ═════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="LogPress 킬스위치 실험 (Role D)")
    ap.add_argument("--demo", action="store_true", help="합성 로그로 파이프라인 검증")
    ap.add_argument("--input", help="로그 파일 경로")
    ap.add_argument("--format", default="bgl", choices=["bgl"])
    ap.add_argument("--window", type=int, default=200, help="인시던트 윈도우 크기")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--save", default="killswitch_parsed.parquet")
    args = ap.parse_args()

    if args.demo:
        recs = make_demo()
        print("\n\033[93m[!] 데모 모드 — 합성 로그입니다. 신호를 일부러 심어놨으므로")
        print("    AUC가 실제 데이터보다 훨씬 높게 나옵니다. 파이프라인 검증용일 뿐,")
        print("    이 GO 판정을 실제 킬스위치 결과로 착각하지 마세요.\033[0m")
    elif args.input:
        recs = load_bgl(args.input, args.window)
    else:
        ap.error("--demo 또는 --input 중 하나를 지정하세요")

    # ── 파이프라인 ────────────────────────────────────────────
    df = parse_logs(recs)
    df = add_rarity(df)
    df = add_position(df)
    df = add_is_error(df)
    df = add_hard_flag(df)

    # ── 신호별 판별력 (단일) ─────────────────────────────────
    print(f"\n[신호] 단일 판별력 (Mann-Whitney U = AUC)")
    for col, hi in [("rarity", True), ("mid_dist", False)]:
        auc, p, r = single_signal_auc(df, col, hi)
        name = "rarity  (희귀할수록 원인)" if col == "rarity" else "position(중앙일수록 원인)"
        print(f"       {name:<26} AUC={auc:.4f}  p={p:.2e}  효과크기 r={r:+.3f}")

    # ── 결합 판별력 (GO ②) ──────────────────────────────────
    auc, pval, coefs, n_test = combined_auc(df)
    print(f"\n[신호] 결합 판별력 (로지스틱 회귀, dev→test 인시던트 분리)")
    print(f"       test set {n_test:,} 라인  |  AUC={auc:.4f}  순열검정 p={pval:.4f}")

    # ── 판정 ─────────────────────────────────────────────────
    v = Verdict(
        n_hard=int(df["is_hard"].sum()),
        level_pres=level_filter_preservation(df),
        auc=auc, p_value=pval, coefs=coefs,
    )
    report(v, df)

    try:
        df.to_parquet(args.save)                       # parquet 권장 (작고 빠름)
        print(f"[save] 파싱 결과 → {args.save}")
    except ImportError:
        alt = args.save.rsplit(".", 1)[0] + ".csv"     # pyarrow 없으면 CSV
        df.to_csv(alt, index=False)
        print(f"[save] 파싱 결과 → {alt}  (parquet 쓰려면: pip install pyarrow)")

    if not args.no_plot:
        plot(df)

    return 0 if v.go else 1


if __name__ == "__main__":
    sys.exit(main())
