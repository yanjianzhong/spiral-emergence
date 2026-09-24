# -*- coding: utf-8 -*-
#
# Copyright (c) 2024-2026 Jianzhong Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0
#
"""spiral_metric_v16._metric.tier3_desc —— 由 spiral_metric_v16.py 拆分。

第三层·结构描述子与谱工具

**纯移动产物**: 每个顶层单元按源码行号逐字切片, 函数体未改一字;
改动只在靠 __file__ 推导的路径与 W11 源码扫描范围两处 (见门面说明)。
"""

import os
import sys
import json
import platform
import time
import warnings
import contextlib
import io

import numpy as np
import scipy.sparse as sp
from scipy import ndimage   # 只有 tier3 用到, 但宽 import 才挡得住"漏 import"这类事故
from scipy.sparse.linalg import eigsh
from scipy.optimize import linprog
import networkx as nx

import matplotlib
matplotlib.use('Agg')      # 原文件里 matplotlib 一进来就是 Agg (无窗口); 见门面
import matplotlib.pyplot as plt

import torch
import quimb.tensor as qtn

# ---- 跨模块依赖 (自动推导, 仅列本模块实际用到的名字) ----
from _metric.registry import (
    PROM_MIN, PROM_WIN, _band_power, _delta2, _first_zero, _mode_census, _xi_from_fft,
)
from _metric.solvers import _char_scale_checked




# ===========================================================================
# 第三层 · 结构对结构
# ===========================================================================
def _late_frame(mod, t=None):
    """(T, N, N) 序列取最后一帧 (最接近稳态斑图), 也支持已经是 (N, N) 的输入。"""
    a = np.asarray(mod)
    if a.ndim == 3:
        a = a[-1 if t is None else t]
    return a




def _binarize_median(a):
    """与 spiral_v13_prepare.py 完全相同的二值化规则 (中位数阈值)。

    两侧用同一个规则是必须的: 若模型侧用中位数而参考侧用固定阈值,
    测到的差异里就混进了"阈值选择"这个与物理无关的自由度。
    """
    a = np.asarray(a, dtype=np.float64)
    return a > np.median(a)




def _ac_len(b):
    """场的自相关长度 (格): 径向平均自相关首次过零的滞后。

    这是**与谱峰无关**的尺度估计 —— 正因为 _char_scale 靠 argmax, 它会被
    二值场的高频噪声壳骗到 (归档 bool 场就是例子, 见 A 段诊断项
    A-归档参考侧管线不可复现)。自相关是零阶量, 不挑壳, 所以可以拿来交叉核对
    lam_pk 是不是伪影。
    """
    z = np.asarray(b, dtype=np.float64)
    z = z - z.mean()
    f = np.fft.fft2(z)
    ac = np.fft.fftshift(np.fft.ifft2(f * np.conj(f)).real)
    c = ac.shape[0] // 2
    prof = ac[c, c:c + max(ac.shape[0] // 2, 1)]
    if prof.size == 0 or prof[0] <= 0:
        return float('nan')
    prof = prof / prof[0]
    neg = np.flatnonzero(prof < 0)
    return float(neg[0]) if neg.size else float(prof.size)




def _ds(b, k=4):
    """块平均降采样, 只用于作图 (把场画成缩略图), 不参与任何量化。"""
    a = np.asarray(b, dtype=np.float64)
    n = a.shape[0] // k
    if n < 1:
        return a
    return a[:n * k, :n * k].reshape(n, k, n, k).mean(axis=(1, 3))




def structural_descriptors(b, lam=None):
    """
    一个二维周期二值场的 6 个**无量纲**结构描述子。

    为什么全部无量纲: 模型侧盒子 36, 参考侧 100 (CLIP) 或 450x4517
    (Southampton 时空图), 尺度和分辨率都不同。任何带量纲的量 (边界总长、
    关联长度绝对值) 直接比就是在比盒子大小。所以一律除以该场自己的特征波长
    lambda_pk, 或者除以总模数 —— 剩下的才是"形状"。

    六个量:
      phi     占空比              (面积率)
      v1_lam  边界长度 x lam       (界面密度)
      v2_lam2 Euler 示性数 x lam^2 (连通域/空洞的净数)
      kpk_k1  谱峰所在模 / 基频    (斑图的锁模位置)
      w50     承载 50% 功率所需模数 / 总非零模数 (自由度集中度)
      xi_lam  关联长度 / lam
    """
    b = np.asarray(b, dtype=bool)
    box = float(max(b.shape))
    z = b.astype(np.float64)
    phi = float(z.mean())

    # 特征波长: 用连续化后的场 (z - <z>) 的谱峰, 与 _char_scale 同一约定。
    # 走校验包装: 平坦场在那里被标成 invalid (原路径会静默返回 lam = box)
    delta = z - z.mean()
    lam_pk, pk, pk_valid = _char_scale_checked(delta, box, deconv=False)
    if not pk_valid:
        lam_pk = float(box)
    if lam is None:
        lam = lam_pk

    # Minkowski V1/V2 在唯一的二值阈值上 (二值场没有别的阈值可选)
    dh = np.abs(z - np.roll(z, -1, 1)).sum()
    dv = np.abs(z - np.roll(z, -1, 0)).sum()
    v1 = float((dh + dv) / float(b.size))                   # 单位: 1/格
    p00 = b.astype(np.int8)
    p01 = np.roll(b, -1, 1).astype(np.int8)
    p10 = np.roll(b, -1, 0).astype(np.int8)
    p11 = np.roll(np.roll(b, -1, 0), -1, 1).astype(np.int8)
    s = p00 + p01 + p10 + p11
    v2 = 0.25 * float(np.sum((s == 1).astype(np.float64)
                             - (s == 3).astype(np.float64))) / float(b.size)

    w50, _w90, n_modes = _mode_census(delta)
    xi = _xi_from_fft(delta, box, deconv=False)
    xi0 = _first_zero(xi['r'], xi['xi'])
    if not np.isfinite(xi0) or xi0 <= 0:
        # 无过零点时退到 xi(r) 的积分尺度, 仍是尺度无关的形式
        r, y = xi['r'], xi['xi']
        xi0 = (float(np.trapezoid(np.clip(y, 0.0, None), r) / max(y[0], 1e-12))
               if len(r) > 2 and y[0] > 0 else float(box))

    return {'phi': phi, 'v1_lam': float(v1 * lam), 'v2_lam2': float(v2 * lam ** 2),
            'kpk_k1': float(box / lam) if lam > 0 else float('nan'),
            'w50': float(w50) / max(n_modes, 1), 'xi_lam': float(xi0 / lam),
            'lam_pk': float(lam), 'w50_n': int(w50), 'n_modes': int(n_modes)}




_KEYS = ('phi', 'v1_lam', 'v2_lam2', 'kpk_k1', 'w50', 'xi_lam')




def descriptor_distance(da, db):
    """
    两个描述子向量之间的距离: 逐项 d_i = |a-b| / sqrt(a^2+b^2), 取 RMS。

    为什么用这个归一化 (与 spiral_metric._mf_distance 同一约定): 它对称、
    有界于 sqrt(2), 且分母恒非负 —— V2 (Euler 示性数密度) 会在零点附近穿过,
    用 "|a-b|/|a|" 会爆出几百上千的假距离。它对"两个量都接近 0"的情形
    自动给出接近 0 的差异, 这正是我们想要的 (都接近 0 就是都接近 0)。
    """
    d = []
    for k in _KEYS:
        a, b = float(da[k]), float(db[k])
        den = float(np.hypot(a, b))
        d.append(abs(a - b) / den if den > 0 else 0.0)
    d = np.array(d, dtype=float)
    return float(np.sqrt(np.mean(d ** 2))), dict(zip(_KEYS, d.tolist()))




def _norm_pk_curve(b, n_bins=24):
    """
    归一化功率谱形状: 横轴 k/k_pk, 纵轴 Delta^2 归一到 ln k 上积分为 1。
    这样比较的只剩**形状**, 与盒子大小、分辨率、振幅都无关。
    """
    box = float(max(b.shape))
    delta = np.asarray(b, dtype=np.float64)
    delta = delta - delta.mean()
    pk = _band_power(delta, box, 2, deconv=False)
    if len(pk['k']) < 3:
        return np.zeros(0), np.zeros(0)
    k, P = pk['k'], pk['P']
    k_pk = k[int(np.argmax(_delta2(k, P, 2)))]
    if not np.isfinite(k_pk) or k_pk <= 0:
        return np.zeros(0), np.zeros(0)
    x = k / k_pk
    y = np.clip(_delta2(k, P, 2), 1e-30, None)
    lx = np.log(x)
    w = lx.max() - lx.min()
    if w <= 0:
        return np.zeros(0), np.zeros(0)
    return x, y / (np.trapezoid(y, lx) / w)




def _peak_period_stats(peaks, sec_per_px):
    """
    Southampton 波峰标记 (时刻, 幅值) -> 振荡周期的统计量。

    这是 B 段里**唯一不依赖轴语义类比**的量: 波峰时刻是一维时序, 与
    "把时空图当二维斑图"无关, 也不经过 .tif 图版渲染 (它是原始作者从
    数据里提出的数值列)。返回相邻峰间隔的中位数、变异系数 (CV) 与秒。

    sec_per_px 必须由源文档标定给出: 0_Process_steps.txt 写
    "15 pixels = 2.5 seconds", 即 1/6 s per px。这里的 X 列单位是**像素**
    (取值直到 25297, 与时空图宽度 27100 同量级), 不是帧 —— 第一版把
    这个单位弄错过两次, 所以参数名里直接把单位写死。
    """
    t = np.sort(np.asarray(peaks, dtype=float)[:, 0])
    if len(t) < 4:
        return None
    dt = np.diff(t)
    dt = dt[dt > 0]
    if len(dt) < 3:
        return None
    med = float(np.median(dt))
    return {'n_peaks': int(len(t)), 'dt_median_px': med,
            'dt_cv': float(np.std(dt) / np.mean(dt)),
            'period_s': med * sec_per_px,
            'span_px': float(t.max() - t.min())}




def _series_lag_agreement(series, period_px, skip_periods=3.0,
                          min_lag_cycles=12.0):
    """
    独立交叉检验: 波峰间隔给出的周期 T, 在**原始强度序列**上对得上吗?

    为什么会需要这个: 波峰标记和强度序列是两套**独立**的提取 —— 前者是原始
    作者在时空图上标的峰位, 后者是在固定空间位置上量的强度随时间的变化。
    如果两者都对, 强度序列在滞后 ≈ T 处的自相关应当出现极大 (隔一个周期,
    波形回到自己), 且明显高于滞后 T/2 处 (反相, 应当低)。

    它同时解释了"为什么谱方法反而检不出周期": 谱峰要求波形**严格周期**,
    而这里波峰间隔的变异系数 CV 从 0.27 到 0.90 不等 —— CV 越大, 谱线被
    自身的不规则性抹得越平。所以三个量 (CV / 自相关滞后峰 / 谱突出度)
    是同一件事的三个侧面, 它们的**一致性**才是可以交付的结论。

    返回: ac (各滞后的自相关系数) / ratio (ac_T / ac_{T/2}) /
          是否为严格滞后峰 -> 三档判定。

    **两个样本量门槛按调用域给 (v15·W8)**: skip_periods / min_lag_cycles 的默认
    值 (跳 3 个周期 / 至少 12 个周期) 是按 **BZ 空间序列**定的 —— 那批序列
    n≈3万~10万 px、lag≈10³ px, 跨 20~40 个周期, 两条都宽松。但同一个函数也被
    用在 **Reading 时间序列**上 (2161 帧 / lag=270 帧, 全长只有 8.0 个周期),
    那里 "至少 12 个周期" 是**结构性不可达**的 (2161 < 12*270 = 3240, 连不跳都
    够不着), 于是该检验在 Reading 侧**从来没运行过**; 而 "跳 3 个周期" 会跳掉
    整条记录的 37%, 把自相关洗白。默认值保持原样以**不改动 BZ 侧任何读数**,
    Reading 调用点显式传自己的值 —— 见 tier3_structural_ladder 的 Reading 分支。
    """
    s = np.asarray(series, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] < 2 or s.shape[0] < 16:
        return None
    x, y = s[:, 0], s[:, 1]
    lag = int(round(float(period_px)))
    if lag < 2:
        return None
    m = x > skip_periods * lag             # 跳过混合之前的那一段
    if int(m.sum()) < min_lag_cycles * lag:
        return None
    yy = y[m]
    idx = np.arange(len(yy))
    yy = yy - np.polyval(np.polyfit(idx, yy, 1), idx)

    def ac(L):
        if L <= 0 or L >= len(yy) - 2:
            return float('nan')
        a, b = yy[:-L], yy[L:]
        if a.std() <= 0 or b.std() <= 0:
            return float('nan')
        return float(np.corrcoef(a, b)[0, 1])

    lags = {'T/2': max(1, lag // 2), 'T': lag, '1.5T': int(round(1.5 * lag)),
            '2T': 2 * lag}
    acs = {k: ac(v) for k, v in lags.items()}
    ac_T = acs['T']
    ac_half = acs['T/2']
    if not np.isfinite(ac_T) or not np.isfinite(ac_half):
        return None
    ratio = float(ac_T / ac_half) if ac_half > 0 else float('inf')
    finite = {k: v for k, v in acs.items() if np.isfinite(v)}
    best = max(finite, key=finite.get)
    if best == 'T':
        verdict = '独立确认 (滞后 T 处是严格自相关极大)'
    elif ac_T >= 0.95 * finite[best]:
        verdict = f'弱确认 (极大在 {best}, 但 ac(T) 与它相差 <5%)'
    else:
        verdict = f'未确认 (极大在 {best}, 不在 T)'
    return {'ac': acs, 'ac_T': ac_T, 'ac_half': float(ac_half),
            'ratio_T_over_half': ratio, 'lag_best': best,
            'lag_T_px': lag, 'verdict': verdict,
            'confirmed': bool(best == 'T'),
            'n_used': int(m.sum())}




def _log_bin(k, p, n_bin=12):
    """
    把 (k, P) 按对数间隔分箱取均值 —— **只用于画图, 判定绝不用它**。

    单次实现的周期图自由度只有 2, 方差极大: 1080 根谱线叠在 log-log 上是一团
    噪声, 完全看不出"功率随 k 单调下降"这个唯一重要的形状。分箱把方差压下去,
    代价是**抹掉窄峰** —— 所以它只能展示趋势。判定 (带内局部极大 + 突出度)
    一律在未分箱的原始谱上做, 见 _series_period。
    """
    k = np.asarray(k, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(k) & np.isfinite(p) & (k > 0) & (p > 0)
    if int(m.sum()) < n_bin:
        return k[m], p[m]
    kk, pp = k[m], p[m]
    edges = np.logspace(np.log10(kk.min()), np.log10(kk.max()), n_bin + 1)
    idx = np.clip(np.digitize(kk, edges) - 1, 0, n_bin - 1)
    kb, pb = [], []
    for bidx in range(n_bin):
        sel = idx == bidx
        if sel.any():
            kb.append(float(kk[sel].mean()))
            pb.append(float(pp[sel].mean()))
    return np.asarray(kb), np.asarray(pb)




def _leading_dark(y, frac=0.5):
    """
    头部连续"暗帧"的个数 (相机预热 / 快门未开)。

    为什么需要这个数: Reading 三条序列的**前 16 帧**亮度约 25.6, 而稳态是
    84~115 —— 这是一段硬阶跃, 不是物理。阶跃的谱是宽频的, 会给整条谱灌进
    高频功率, 从而把"漂移占比"**压小**。实测: 参考序列不去预热段漂移占比
    9.9%, 去掉后 4.1%; 而再往后去掉上升沿则升到 ~88%。也就是说照原样报出的
    漂移占比是**被低估**的, 不是高估。

    阈值取 0.5 x 中位数 (自标定, 不写死绝对值): 三条序列的稳态亮度不同
    (115 / 90 / 84), 固定的绝对阈值会漏掉最暗的那条。
    """
    yy = np.asarray(y, dtype=np.float64)
    if yy.size == 0:
        return 0
    med = float(np.median(yy))
    if med <= 0:
        return 0
    thr = frac * med
    i = 0
    while i < yy.size and yy[i] < thr:
        i += 1
    return int(i)




def _window_drift_ratio(y, frac=0.1):
    """
    前 frac 段与后 frac 段各自的最小二乘斜率之比的绝对值。

    为什么需要这个数: 头部暗帧是一段硬阶跃, 它给整条序列灌进一个**单边**
    趋势。前后窗口的斜率比就是"这个单边趋势占了多大比重"的一个直接读数 ——
    比值远大于 1 说明前段的漂移是阶跃造成的, 而不是物理漂移。

    返回 (比值, 前段斜率, 后段斜率); 段长不足或后段斜率退化到 0 时返回 nan。

    诚实边界: 这个数**只作诊断**。它不去自动裁剪序列 —— 自动裁掉前段等于
    改变已经报出去的口径。序列是否裁剪由 trim_scan 的敏感性扫描裁决, 本函数
    只负责把"漂移比有多大"这件事量化出来。
    """
    yy = np.asarray(y, dtype=np.float64)
    n_w = int(round(frac * yy.size))
    if yy.size < 20 or n_w < 8:
        return float('nan'), float('nan'), float('nan')
    sl = lambda v: float(np.polyfit(np.arange(v.size), v, 1)[0])  # noqa: E731
    s_head, s_tail = sl(yy[:n_w]), sl(yy[-n_w:])
    if not (np.isfinite(s_head) and np.isfinite(s_tail)) or abs(s_tail) < 1e-30:
        return float('nan'), s_head, s_tail
    return float(abs(s_head) / abs(s_tail)), s_head, s_tail




def _series_period(lum, n_keep=6, min_cycles=8):
    """
    一维时序 (逐帧平均亮度) 的主周期 —— 带**居中的检出判定**, 检不出就报检不出。

    第一版直接把谱峰当主周期, 结果三条序列全报 2161.0 帧 = 100.0% 全长, 即谱峰
    恒在 k=1。那不是发现, 是残余趋势: 线性去趋势后 std 只从 8.66 降到 8.38,
    k=1 仍占 51% 功率, 说明主导成分是**非线性慢漂移** (凝胶干燥/照明漂移),
    线性拟合吃不掉它, Hann 窗又把它压成一根高谱线。把趋势读成周期是这类分析
    最典型的自欺, 所以这里做三件事:

      1. 高通: 除线性拟合外, 再减去窗口为 n/min_cycles 的滑动平均, 压掉慢漂移。
      2. 限带: 只在 k >= min_cycles 的谱线里找峰。周期必须短于全长的 1/8,
         否则"周期"和"整段记录"无法区分。
      3. 判定: 同时要求 (a) 带外漂移功率占比 < drift_max, (b) 峰功率占比
         >= peak_min。任一不满足就 detected=False, 周期置 None。

    谱峰仍做**抛物线插值**。这不是装饰: 2161 帧的记录里周期 ~137 帧对应第 15.8
    根谱线, 取整会锁到第 16 根 = 135.1 帧, 引入 -1.3% 量化误差; B3 报的是三个
    等长序列的**周期比值**, 量化误差在三者上不同, 会污染比值。

    返回字段: detected / period_frames / drift_frac / peak_power_frac / verdict。
    """
    y = np.asarray(lum, dtype=np.float64)
    n = len(y)
    if n < 4 * min_cycles:
        return None
    idx = np.arange(n)
    y = y - np.polyval(np.polyfit(idx, y, 1), idx)

    # 漂移占比: 在**高通之前**的谱上量, 否则高通会把它一起减掉、量不到
    F0 = np.fft.rfft(y)
    p0 = np.abs(F0) ** 2
    p0[0] = 0.0
    ks = np.arange(len(F0))
    tot0 = float(p0[1:].sum())
    if tot0 <= 0:
        return None
    drift = float(p0[(ks >= 1) & (ks < min_cycles)].sum() / tot0)

    # 高通: 直接把 k < min_cycles 的谱线置零。用谱域而不是滑动平均 ——
    # 减滑动平均的频响是**梳状**的 (在 k = n/w 的整数倍处有零点), 会在零点之间
    # 造出假的峰; 谱域置零的搜索带就是通带, 没有这个问题。
    F2 = F0.copy()
    F2[:min_cycles] = 0.0
    yh = np.fft.irfft(F2, n)
    f = np.abs(np.fft.rfft(yh * np.hanning(n)))
    p = f ** 2
    p[0] = 0.0
    tot = float(p[1:].sum())
    if tot <= 0:
        return None

    p_band = p.copy()
    p_band[ks < min_cycles] = 0.0
    k0 = int(np.argmax(p_band))
    peak_frac = float(p[k0] / tot)

    # 局部背景: 峰附近 (不含峰本身) 的中位数功率。用中位数而不是均值, 免得
    # 背景被峰自己抬高。
    lo = max(min_cycles, k0 - PROM_WIN)
    hi = min(len(p), k0 + PROM_WIN + 1)
    bg_vals = np.concatenate([p[lo:k0], p[k0 + 1:hi]])
    bg = float(np.median(bg_vals)) if len(bg_vals) else 0.0
    prominence = float(p[k0] / bg) if bg > 0 else float('inf')

    # 【检出判定的核心, 两步】
    # (a) 峰必须是**带内内部的局部极大**。只要求"带内最大"不够: 亮度谱是红的
    #     (功率随 k 单调下降), 于是带内最大值总落在带的最低端, 即频带边缘。
    #     把边缘当峰, 换个 min_cycles 就换一个"周期" —— 那是把滤波器边界读成
    #     了物理量。真振荡必须在带内部拱起一个局部极大。
    # (b) 峰必须**高出局部背景** PROM_MIN 倍。这一条不可省: 红噪声的谱很散
    #     (单次实现的周期图方差极大), 天然会在带内冒出一堆局部极大; 只看
    #     "有没有局部极大"必然假阳性 —— 实测纯红噪声 300 次里就有 226 次冒峰。
    #
    # PROM_MIN 与 PROM_WIN 不是拍出来的, 是按零假设标定的:
    #   纯红噪声 (随机游走+漂移) 300 次抽样, prominence 中位 23, p99 85, 最大 113;
    #   正弦+30% 噪声+漂移, T=25~200 帧, 20 次抽样里最小 2309。
    #   两者之间有约 20 倍的空隙, 取 300 落在空隙中间 -> 假阳性 < 1/300,
    #   而 T>=25 帧的信号 100% 检出。标定脚本见 _v13_diag_tier3.py。
    interior = bool(min_cycles < k0 < len(p) - 1)
    local_max = bool(p[k0] > p[k0 - 1] and p[k0] > p[k0 + 1])
    detected = bool(interior and local_max and prominence >= PROM_MIN)
    if detected:
        verdict = '检出周期'
    elif not interior or not local_max:
        verdict = '谱为红噪声, 无带内峰 (边缘极大不是峰)'
    else:
        verdict = (f'峰不显著 (突出度 {prominence:.0f} < {PROM_MIN}), 未检出')

    def refine(k0_):
        """在谱线 k0 处对 |F| 做三点抛物线定峰, 返回小数谱线位置。"""
        if k0_ <= 1 or k0_ + 1 >= len(f):
            return float(k0_)
        a, b, c = f[k0_ - 1], f[k0_], f[k0_ + 1]
        den = a - 2.0 * b + c
        return float(k0_) + (0.5 * (a - c) / den if den != 0 else 0.0)

    k_ref = refine(k0)
    order = ks[p > 0][np.argsort(p[p > 0])[::-1]][:n_keep]
    return {'detected': detected,
            'period_frames': float(n / k_ref) if detected and k_ref > 0 else None,
            'k_refined': k_ref if detected else None,
            'k_band_argmax': k0,
            'period_frames_band_argmax': float(n / k0),
            'top_periods_frames': [float(n / refine(int(t))) for t in order],
            'drift_frac': drift, 'peak_power_frac': peak_frac,
            'prominence': prominence, 'prom_min': PROM_MIN,
            'min_cycles': int(min_cycles), 'n_frames': n,
            # 判定所依据的那条谱本身。作图时要把它画出来 —— "未检出"是个结论,
            # 只有把谱亮出来 (带内最大值落在频带边缘、且低于突出度阈值),
            # 读者才能自己复核这个结论, 而不是被告知一个 verdict 字符串。
            # power[i] 对应谱线 k = i+1 (已丢掉 k=0 的直流项)。
            'power': p[1:].tolist(),
            'power_bg': bg,
            'power_thresh': float(bg * PROM_MIN),
            'verdict': verdict}
