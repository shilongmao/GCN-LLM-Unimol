# -*- coding: utf-8 -*-
# Copyright (c) 2026 GCN-LLM-UniMol 项目作者（个人）. All Rights Reserved.
# 本项目基于 DrugCLIP 与 GCN-LLM 构建，仅限非商业用途；
# 未经作者书面许可，不得用于论文、学位论文、专利等学术发表。完整条款见根目录 LICENSE。
"""
vs_metrics.py — 把 GCN-LLM-UniMol 的评估从 acc/F1 升级到虚拟筛选指标
=====================================================================

为什么需要
----------
原代码（``Final/train.py:339-347``）用的是::

    acc  = accuracy_score(labels, sims > threshold)
    f1   = f1_score(labels, sims > threshold)
    auc  = roc_auc_score(labels, sims)

问题有三：

1. **阈值依赖**：acc/F1 需要一个阈值，而阈值是在验证集上"搜"出来的
   （``np.arange(-0.5, 1.0, 0.05)`` 扫 30 个点），本质上是在优化一个
   与实际筛选无关的指标。
2. **类别不平衡下几乎无区分力**：630 个分子里 60% 是阳性，
   「全判 active」就能拿 acc ≈ 0.60 / F1 ≈ 0.75。模型做到 0.88/0.94
   看起来很好，其实只比平凡解强一点。
3. **与虚拟筛选的实用目标不一致**：真实场景是"从库里捞头部"，
   关心的是 **早期富集**（BEDROC / EF），而不是全局准确率。

本模块提供 **纯 numpy 实现**，不依赖 sklearn / rdkit / torch：

    auc(y_true, y_score)                        ROC-AUC（含 ties 平均秩）
    enrichment_factor(y_true, y_score, frac)    EF@frac
    bedroc(y_true, y_score, alpha=80.5)         BEDROC
    evaluate(y_true, y_score, ef_fractions)     一次性返回 dict

BEDROC 口径
-----------
Truchon & Bayly (2007) 定义::

    RIE = Σ_{i ∈ actives} exp(-α · r_i / N)          # r_i 为降序名次(1-based)

本模块提供两种归一化，用 ``mode`` 选择：

    mode="max"     BEDROC = RIE / RIE_max            # 完美排序 = 1
    mode="minmax"  BEDROC = (RIE-RIE_min)/(RIE_max-RIE_min)  # 完美=1, 最差=0

其中 RIE_max = 所有 active 排在最前时的 RIE，
      RIE_min = 所有 active 排在最后时的 RIE。

**随机排序的基线不是 0，也不是 0.5，而是取决于阳性率 p = Ra/N**

推导（N 较大、α 较大时）::

    RIE_random / RIE_max  ≈  p / (1 - exp(-α·p))

举例（α=80.5）：

    p = 0.05  →  随机 BEDROC ≈ 0.051     （1000 个分子 / 50 阳性）
    p = 0.40  →  随机 BEDROC ≈ 0.402     （630 个分子 / 253 阳性，即 XOD）

> 这一点非常关键：**在阳性率高的数据集上，随机基线的 BEDROC 本身就很高**。
> 直接报 BEDROC=0.4 看似不错，但在 p=0.40 时它等于随机。
> 必须同时报告 `random_bedroc(p, alpha)` 作为对照。

``mode="minmax"`` 在 N 大、α 大时与 ``mode="max"`` 几乎相同，
因为 ``RIE_min ≈ 0``（actives 排到最后时 exp(-α·r/N) 已下溢）。

（ConGLUDe 用的是 RDKit 的 ``CalcBEDROC(..., alpha=80.5)``；
 若要与它的数字严格对齐，请用 rdkit 复算一遍做标定。）

自测::

    python vs_metrics.py --selftest
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 基础
# ---------------------------------------------------------------------------
def _check(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true).ravel()
    ys = np.asarray(y_score, dtype=float).ravel()
    if yt.shape != ys.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {ys.shape}")
    if yt.size == 0:
        raise ValueError("empty input")
    if not np.all(np.isfinite(ys)):
        raise ValueError("y_score contains NaN/inf")
    yt = (yt > 0).astype(np.int64)
    return yt, ys


def _desc_order(y_score: np.ndarray) -> np.ndarray:
    """降序名次；分数相同时用稳定排序（原顺序），与 ConGLUDe 的
    ``torch.randperm`` 打破平局的思路一致。"""
    return np.argsort(-y_score, kind="mergesort")


# ---------------------------------------------------------------------------
# ROC-AUC（Mann-Whitney U，含 ties 平均秩）
# ---------------------------------------------------------------------------
def auc(y_true: Sequence, y_score: Sequence) -> float:
    yt, ys = _check(np.asarray(y_true), np.asarray(y_score))
    n = yt.size
    ra = int(yt.sum())
    rn = n - ra
    if ra == 0 or rn == 0:
        return float("nan")

    order = np.argsort(ys, kind="mergesort")
    sorted_scores = ys[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0          # 1-based
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1

    return float((ranks[yt == 1].sum() - ra * (ra + 1) / 2.0) / (ra * rn))


# ---------------------------------------------------------------------------
# 富集因子
# ---------------------------------------------------------------------------
def enrichment_factor(
    y_true: Sequence,
    y_score: Sequence,
    fraction: float,
) -> float:
    """EF@fraction = (top-k 中的阳性比例) / (全局阳性比例)。"""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    yt, ys = _check(np.asarray(y_true), np.asarray(y_score))
    n = yt.size
    ra = int(yt.sum())
    if ra == 0:
        return float("nan")
    k = max(1, int(math.ceil(fraction * n)))
    top = _desc_order(ys)[:k]
    hits = int(yt[top].sum())
    return float((hits / k) / (ra / n))


# ---------------------------------------------------------------------------
# BEDROC
# ---------------------------------------------------------------------------
def _rie(active_ranks: np.ndarray, n: int, alpha: float) -> float:
    return float(np.exp(-alpha * active_ranks / n).sum())


def bedroc(
    y_true: Sequence,
    y_score: Sequence,
    alpha: float = 80.5,
    mode: str = "max",
) -> float:
    """BEDROC。``alpha=80.5`` 对应 ≈20% 早期识别。"""
    if mode not in ("max", "minmax"):
        raise ValueError("mode must be 'max' or 'minmax'")
    yt, ys = _check(np.asarray(y_true), np.asarray(y_score))
    n = yt.size
    ra = int(yt.sum())
    if ra == 0 or ra == n:
        return float("nan")

    order = _desc_order(ys)
    ranks = np.arange(1, n + 1, dtype=float)
    is_active = yt[order] == 1
    rie = _rie(ranks[is_active], n, alpha)

    rie_max = _rie(np.arange(1, ra + 1, dtype=float), n, alpha)
    if mode == "max":
        return float(rie / rie_max)

    rie_min = _rie(np.arange(n - ra + 1, n + 1, dtype=float), n, alpha)
    if rie_max == rie_min:
        return float("nan")
    return float((rie - rie_min) / (rie_max - rie_min))


# ---------------------------------------------------------------------------
# 一站式
# ---------------------------------------------------------------------------
def evaluate(
    y_true: Sequence,
    y_score: Sequence,
    ef_fractions: Sequence[float] = (0.005, 0.01, 0.05),
    alpha: float = 80.5,
    bedroc_mode: str = "max",
) -> Dict[str, float]:
    """返回 auc / bedroc / ef_x 以及阳性率等诊断量。"""
    yt, ys = _check(np.asarray(y_true), np.asarray(y_score))
    out: Dict[str, float] = {
        "n": int(yt.size),
        "n_pos": int(yt.sum()),
        "pos_rate": float(yt.mean()),
        "auc": auc(yt, ys),
        "bedroc": bedroc(yt, ys, alpha=alpha, mode=bedroc_mode),
    }
    for f in ef_fractions:
        out[f"ef_{f}"] = enrichment_factor(yt, ys, f)
    return out


def top_k_hits(
    y_true: Sequence,
    y_score: Sequence,
    k: int,
) -> Dict[str, float]:
    """虚拟筛选最关心的量：top-k 里捞到多少真阳性。"""
    yt, ys = _check(np.asarray(y_true), np.asarray(y_score))
    k = max(1, min(int(k), yt.size))
    top = _desc_order(ys)[:k]
    hits = int(yt[top].sum())
    return {
        "k": k,
        "hits": hits,
        "precision": hits / k,
        "recall": hits / max(1, int(yt.sum())),
    }


def random_bedroc(pos_rate: float, alpha: float = 80.5) -> float:
    """随机排序下 BEDROC(mode='max') 的解析期望值。

    RIE_random / RIE_max ≈ p / (1 - exp(-alpha * p))
    —— 注意它**依赖阳性率 p**，不是常数。
    """
    p = float(pos_rate)
    if not 0.0 < p < 1.0:
        return float("nan")
    return p / (1.0 - math.exp(-alpha * p))


def random_baselines(
    n: int,
    n_pos: int,
    alpha: float = 80.5,
    n_trials: int = 200,
    seed: int = 0,
) -> Dict[str, float]:
    """用随机排序估计各指标的期望基线 —— 判断模型是否真的有信号。"""
    rng = np.random.default_rng(seed)
    aucs, br_max, br_mm, ef5 = [], [], [], []
    for _ in range(n_trials):
        y = np.zeros(n, dtype=np.int64)
        y[rng.choice(n, size=n_pos, replace=False)] = 1
        s = rng.random(n)
        aucs.append(auc(y, s))
        br_max.append(bedroc(y, s, alpha, "max"))
        br_mm.append(bedroc(y, s, alpha, "minmax"))
        ef5.append(enrichment_factor(y, s, 0.05))
    return {
        "auc": float(np.mean(aucs)),
        "bedroc(max)": float(np.mean(br_max)),
        "bedroc(minmax)": float(np.mean(br_mm)),
        "ef_0.05": float(np.mean(ef5)),
    }


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
def _selftest() -> int:
    fails = 0

    def check(cond: bool, msg: str) -> None:
        nonlocal fails
        if cond:
            print(f"  [ok]   {msg}")
        else:
            fails += 1
            print(f"  [FAIL] {msg}")

    n, ra = 1000, 50                      # 5% 阳性率
    rng = np.random.default_rng(0)

    # (1) 完美排序
    y = np.zeros(n, dtype=int)
    y[:ra] = 1
    s_perfect = np.arange(n, 0, -1, dtype=float)   # 前 50 个分数最高
    r = evaluate(y, s_perfect)
    print("== 完美排序 ==")
    print(f"   {r}")
    check(abs(r["auc"] - 1.0) < 1e-9, "AUC == 1")
    check(abs(r["bedroc"] - 1.0) < 1e-9, "BEDROC(max) == 1")
    check(r["ef_0.05"] > 19.0, f"EF@0.05 ≈ 20（实际 {r['ef_0.05']:.2f}）")

    # (2) 最差排序
    s_worst = -s_perfect
    r = evaluate(y, s_worst)
    print("== 最差排序 ==")
    print(f"   {r}")
    check(abs(r["auc"] - 0.0) < 1e-9, "AUC == 0")
    check(abs(r["bedroc"]) < 1e-9, "BEDROC(max) == 0")

    # (3) 随机排序 & 基线
    print("== 随机排序基线 ==")
    base = random_baselines(n, ra, n_trials=100)
    analytic = random_bedroc(ra / n, alpha=80.5)
    print(f"   {base}")
    print(f"   解析期望 random_bedroc(p={ra/n:.3f}) = {analytic:.4f}")
    check(abs(base["auc"] - 0.5) < 0.03, "随机 AUC ≈ 0.5")
    check(abs(base["bedroc(max)"] - analytic) < 0.02,
          f"随机 BEDROC 与解析式一致（{base['bedroc(max)']:.4f} vs {analytic:.4f}）")
    check(analytic > 0.04,
          "随机基线 BEDROC > 0 —— 不能当作「接近 0」来解读")
    check(abs(base["bedroc(minmax)"] - base["bedroc(max)"]) < 0.05,
          "N 大、alpha 大时 minmax ≈ max（RIE_min≈0）")

    # (4) ties 处理
    print("== ties ==")
    y2 = np.array([1, 0, 1, 0])
    s2 = np.array([0.5, 0.5, 0.5, 0.5])
    check(abs(auc(y2, s2) - 0.5) < 1e-9, "全并列时 AUC == 0.5")

    # (5) top-k
    print("== top-k ==")
    y3 = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    s3 = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    h = top_k_hits(y3, s3, 3)
    check(h["hits"] == 3 and abs(h["precision"] - 1.0) < 1e-9, f"top-3 全中 {h}")

    print()
    print("全部通过" if fails == 0 else f"{fails} 项失败")
    return fails


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="虚拟筛选指标（BEDROC/EF/AUC）")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(_selftest())

    # 演示：用 GCN-LLM-UniMol 的真实规模做基线对照
    print("XOD 数据集规模下的随机基线（630 分子 / 253 阳性）：")
    print(random_baselines(n=630, n_pos=253, n_trials=300))
    print()
    print("若模型报的 AUC 是 0.827，与上面的 auc 基线比较即可知道增益；")
    print("注意这仍是**同靶点同系物随机划分**下的数字，不代表泛化能力。")
