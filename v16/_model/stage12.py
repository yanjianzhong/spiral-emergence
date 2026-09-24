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
"""spiral_model_v16._model.stage12 —— 由 spiral_model_v16.py 拆分。

阶段一·空无 / 阶段二·对称性破缺 + v14·F2a h/J 负对照扫描

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
from _model.core import (
    KNOBS, derive_L2_entanglement_gap, derive_L3_schmidt_rank, derive_L4_bond_dimension, exact_ground_state, fit_central_charge, record_link,
)




# ============================================================================
# 阶段一 · 空无基底等权叠加蕴潜能
# ============================================================================
def stage1_void(dimension=2, seed=0):
    """
    空无 = 最大对称态 (等权叠加), 也是幺半群的单位元。

    因果链 L1: 默认维度取 2 —— 空无就是**单个量子比特**, 即阶段二自旋链的
    局部自由度。整个 Hilbert 空间是它的 L 次张量幂, 由
    derive_L1_local_to_full() 派生 (dim_full = d_local ** L)。

    维度 4 曾是一个与后续任何阶段都无关的随手值; 取 2 之后,
    "空无"与"链"的关系从修辞变成了集合论事实。
    """
    rng = np.random.default_rng(seed)
    H = (rng.standard_normal((dimension, dimension)) +
         1j * rng.standard_normal((dimension, dimension)))
    Q, _ = np.linalg.qr(H)
    equal = np.ones(dimension, dtype=complex) / np.sqrt(dimension)
    print(f"[阶段一] 空无基底: 局部维度 d={dimension}, "
          f"酉条件数={np.linalg.cond(Q):.2f}")
    return {'unitary': Q, 'equal_state': equal, 'dimension': dimension}




# ============================================================================
# 阶段二 · 量子涨落扰动对称性破缺 (临界 Ising 基态的 Schmidt 极化)
# ============================================================================
def stage2_critical_break(L=16, J=1.0, h=1.0, void_state=None):
    """
    h/J = 1 是横场 Ising 的热力学临界点。基态的 Schmidt 谱呈现极化 ——
    这是"涨落打破对称性"在纠缠谱上的可计算投影。
    诚实边界: 有限尺寸下纠缠熵峰值偏向 h/J < 1, 严格临界点只在热力学极限。

    v16·B1: 新增**可选**入参 void_state = L1 张成的域态 (维数 2**L)。
    **纯增量接线**: 只做维数断言 + 因果链登记, 下面所有物理读数 (E0 / gs /
    spectrum / probabilities / entropy / polarization / gap) 一字不动;
    不传 void_state 时行为与 v15 完全一致。
    """
    print(f"[阶段二] 临界点对称性破缺: L={L}, h/J={h/J}")
    if void_state is not None:
        void_state = np.asarray(void_state, dtype=complex)
        assert void_state.size == 2 ** L, (
            f"L1 域态维数 {void_state.size} != 2**{L} = {2 ** L}")
        record_link('L1', f'空无域态 (维数 {void_state.size}) -> 阶段二链 L={L}',
                    'assert void_state.size == 2 ** L', '已接线', '未接线',
                    note='v16·B1: L1 到 L2 的第一条真实连线。该断言只校验维数'
                         '匹配, **不改变阶段二的任何物理读数**。上游是否真的'
                         '"被用上", 由 derive_L1_void_to_chain 的重叠曲线回答 —— '
                         '维数相符本身不构成物理因果。')
    E0, gs = exact_ground_state(L, J, h)
    half = L // 2
    s = np.linalg.svd(gs.reshape(2 ** half, 2 ** (L - half)), compute_uv=False)
    p = s ** 2
    p = p[p > 1e-15]
    p = p / p.sum()
    pol = float(s[0] / (np.mean(s[1:]) + 1e-30)) if len(s) > 1 else 1.0
    S = float(-np.sum(p * np.log(p)))
    # v14·F2: gap = p1/p0 显式返回。此前只由 derive_L2_entanglement_gap() 从
    # probabilities 里算, 走 h/J 扫描时需要一个自包含的读数。
    # 注意它是**单调量** (h/J 从 0 增大: GHZ 态 gap->1, 乘积态 gap->0),
    # 在 h/J=1.0 处**不是**极值 —— 不可当作临界判据, 见 hj_scan() 的说明。
    gap = float(p[1] / p[0]) if len(p) > 1 else float('nan')
    print(f"  E0/L={E0/L:.6f}, S(L/2)={S:.4f}, Schmidt 极化={pol:.3f}, "
          f"gap=p1/p0={gap:.6f}")
    return {'E0': E0, 'gs': gs, 'spectrum': s, 'probabilities': p,
            'entropy': S, 'polarization': pol, 'gap': gap}




# ============================================================================
# v14·F2a · 负对照: h/J 扫描 (临界性是不是事实)
# ============================================================================
#
# 这是 v14 的生死检验: 若"临界模拟"成立, 把 h/J 推离 1.0 之后所有临界指标
# 都应当**塌缩**。
#
# ⚠ 判据的选取 —— v14 实测**修正了任务书的原判据**。
#   任务书 Task 3 要求: "h/J=1.0 处 gap=p1/p0 是全局/局部极小值, 否则临界模拟
#   存疑", 并给出验收 "gap 比邻近点低 (比值<0.8)"。这个判据在物理上是错的,
#   实测会**伪失败**:
#       gap 对 h/J **单调递减** (0.5 -> 0.9967, 1.0 -> 0.3662, 1.5 -> 0.0398),
#       因为 h/J->0 时基态趋于 GHZ 猫态 (两个等权 Schmidt 值, gap->1),
#       h/J->inf 时基态趋于乘积态 (p0->1, gap->0)。整条曲线上没有内部极值。
#   照原判据: gap(1.0)/gap(1.05) = 1.41 > 0.8 -> 判"未通过" -> 触发备选方案
#   (twisted BC / 上 L=24)。那是判据错误导致的伪失败, 不是物理结论。
#
#   改用三条 (本模块实测值见下):
#     主判据: c_fit(h/J) 在 h/J=1.0 取全局**最大**。实测 0.5072, 而 0.5 处
#             0.0646、1.5 处 0.1317。含义是 **Calabrese-Cardy ansatz 只在临界
#             点成立** —— 比"某个数小"强, 它是"拟合公式本身失效"。
#     次判据: S(L/2) 峰位落在 h/J=1.0 附近 (实测 0.95)。有限尺寸下熵峰偏向
#             h/J<1, 这与 stage2_critical_break 的 docstring 声明一致, 故窗口
#             取 [0.9, 1.05] 而非单点, 并把漂移量显式报出。
#     区分力: 框架必须在非临界点**正确失败** (审计 §五.2: 一个能在非临界点
#             正确失败的框架, 比一个永远输出 3% 的框架可靠)。要求 h/J 在
#             {0.5, 0.8, 1.2, 1.5} 上 |c-0.5|/0.5 > 20% (实测 87/45/39/74%)。
#   gap 降级为**单调性诊断**: 只报它是单调的, 不测极值。

HJ_SCAN_DEFAULT = (0.5, 0.8, 0.9, 0.95, 0.98, 1.0, 1.02, 1.05, 1.1, 1.2, 1.5)


HJ_CRIT = 1.0                  # 热力学临界点


HJ_S_WINDOW = (0.9, 1.05)      # 次判据: S(L/2) 峰位允许窗口 (有限尺寸漂移)


HJ_DISCRIM = (0.5, 0.8, 1.2, 1.5)   # 区分力判据: 这些点上框架必须"正确失败"


HJ_DISCRIM_MIN = 0.20          # 区分力门槛: |c-0.5|/0.5 至少这么大




def hj_scan(hj_values=HJ_SCAN_DEFAULT, L=16, J=1.0, verbose=True):
    """
    F2a 负对照: 在 h/J != 1.0 上重跑阶段二, 看临界指标是否塌缩。

    每点记录 E0/L, S(L/2), gap=p1/p0, Schmidt 极化, 非零 Schmidt 数,
    以及**由同一组 Schmidt 谱派生的下游参数** (L2 gap, L3 N_selfref, L4 chi),
    外加该点的 c_fit。这样"参数偏离临界 -> 下游派生量是否跟着动"一目了然。

    返回 dict, 含 'points' (逐点记录) 与 'verdict' (三条判据的裁决)。
    """
    print("\n" + "=" * 76)
    print(f"  v14·F2a 负对照 · h/J 扫描 (L={L}, J={J})")
    print("=" * 76)
    print("  这是 v14 的生死检验: 若临界模拟是事实, h/J 偏离 1.0 后指标必须塌缩。")
    print("  ⚠ 判据已修正: 任务书原判据 (gap 在 h/J=1.0 取极小值) 物理上不成立 ——")
    print("     gap 对 h/J 单调递减, 曲线无内部极值。改用 c_fit 峰值 + S 峰位 + 区分力。")

    points = []
    for hj in hj_values:
        E0, gs = exact_ground_state(L, J, hj)
        half = L // 2
        s = np.linalg.svd(gs.reshape(2 ** half, 2 ** (L - half)),
                          compute_uv=False)
        p = s ** 2
        n_nonzero = int(np.sum(p > 1e-15))
        p = p[p > 1e-15]
        p = p / p.sum()
        S = float(-np.sum(p * np.log(p)))
        gap = float(p[1] / p[0]) if len(p) > 1 else float('nan')
        pol = float(s[0] / (np.mean(s[1:]) + 1e-30)) if len(s) > 1 else 1.0

        # 下游派生量: 走**与 main() 完全相同**的派生函数, 保证口径一致
        l2 = derive_L2_entanglement_gap(p)
        l3 = derive_L3_schmidt_rank(p, KNOBS['schmidt_tol'])
        l4 = derive_L4_bond_dimension(S)

        fit = fit_central_charge(gs, L)
        c_fit = float(fit['c'])
        rms = float(fit['rms'])

        points.append({'hj': float(hj), 'E0': float(E0), 'E0_per_L': float(E0 / L),
                       'S': S, 'gap': gap, 'polarization': pol,
                       'n_schmidt_nonzero': n_nonzero,
                       'N_selfref': int(l3['N_selfref']), 'chi': int(l4['chi']),
                       'chi_lower_bound': float(l4['chi_lower_bound']),
                       'c_fit': c_fit, 'c_rms': rms,
                       'eps_vs_half': abs(c_fit - 0.5) / 0.5})
        if verbose:
            print(f"    h/J={hj:5.2f}: E0/L={E0/L:10.6f} S={S:8.5f} "
                  f"gap={gap:8.5f} pol={pol:9.3f} N_sr={l3['N_selfref']:3d} "
                  f"chi={l4['chi']:2d} c_fit={c_fit:.4f} "
                  f"|c-0.5|/0.5={abs(c_fit-0.5)/0.5:.1%}")

    pts = np.array([(q['hj'], q['S'], q['gap'], q['c_fit']) for q in points])
    hj_arr, S_arr, gap_arr, c_arr = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]

    i_c = int(np.argmax(c_arr))
    i_s = int(np.argmax(S_arr))
    hj_c, hj_s = float(hj_arr[i_c]), float(hj_arr[i_s])

    # 主判据: c_fit 峰值恰在临界点
    ok_c = (abs(hj_c - HJ_CRIT) < 1e-12)
    # 次判据: S 峰位落在窗口内 (有限尺寸漂移是**预期**的, 故给窗口)
    ok_s = (HJ_S_WINDOW[0] <= hj_s <= HJ_S_WINDOW[1])
    # 区分力: 非临界点上必须"正确失败"
    disc = {}
    for hj in HJ_DISCRIM:
        m = np.isclose(hj_arr, hj)
        if m.any():
            disc[float(hj)] = float(abs(c_arr[m][0] - 0.5) / 0.5)
    ok_d = bool(disc) and all(v > HJ_DISCRIM_MIN for v in disc.values())

    # gap 单调性诊断 (不作判据, 只报告"它确实是单调的")
    dgap = np.diff(gap_arr)
    gap_monotone = bool(np.all(dgap < 0))

    verdict = {'ok_c_peak_at_critical': ok_c,
               'hj_at_c_peak': hj_c,
               'ok_s_peak_in_window': ok_s,
               'hj_at_s_peak': hj_s,
               's_window': list(HJ_S_WINDOW),
               'ok_discriminating': ok_d,
               'discrim_min': HJ_DISCRIM_MIN,
               'discrim': disc,
               'gap_monotone_decreasing': gap_monotone,
               'passed': bool(ok_c and ok_s and ok_d)}

    print("\n  --- go/no-go 裁决 ---")
    print(f"    [主] argmax c_fit 在 h/J = {hj_c:.2f}  -> "
          f"{'PASS' if ok_c else 'FAIL'} (要求 = {HJ_CRIT})")
    print(f"         实测 c_fit = {c_arr[i_c]:.4f}; 偏离处: "
          f"h/J={hj_arr[0]:.2f} -> {c_arr[0]:.4f}, "
          f"h/J={hj_arr[-1]:.2f} -> {c_arr[-1]:.4f}")
    print(f"    [次] argmax S(L/2) 在 h/J = {hj_s:.2f}  -> "
          f"{'PASS' if ok_s else 'FAIL'} (窗口 {HJ_S_WINDOW})")
    print(f"         漂移 {hj_s - HJ_CRIT:+.2f}: 有限尺寸熵峰偏向 h/J<1, 已声明")
    print(f"    [区分力] 非临界点 |c-0.5|/0.5 > {HJ_DISCRIM_MIN:.0%} -> "
          f"{'PASS' if ok_d else 'FAIL'}")
    for k, v in sorted(disc.items()):
        print(f"         h/J={k:4.2f}: {v:.1%}")
    print(f"    [诊断] gap 单调递减: {'是' if gap_monotone else '否'} "
          f"({gap_arr[0]:.4f} -> {gap_arr[-1]:.4f}) —— 不作判据, 见函数头说明")
    print(f"\n    >>> 总裁决: {'临界性成立 (继续后续任务)' if verdict['passed'] else '未通过 (触发备查)'}")

    return {'points': points, 'verdict': verdict,
            'hj_values': [float(x) for x in hj_values], 'L': L, 'J': J}




def plot_hj_scan(scan, out_png):
    """三联图: c_fit(h/J) 主判据 | S(L/2) 次判据 | gap 单调性诊断。"""
    pts = scan['points']
    hj = np.array([q['hj'] for q in pts])
    S = np.array([q['S'] for q in pts])
    gap = np.array([q['gap'] for q in pts])
    c = np.array([q['c_fit'] for q in pts])
    v = scan['verdict']

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    ax.plot(hj, c, 'o-', color='#c0392b', lw=2, ms=6)
    ax.axhline(0.5, color='k', ls='--', lw=1.2, label='真实 c = 0.5')
    ax.axvline(HJ_CRIT, color='gray', ls=':', lw=1.2)
    ax.plot([v['hj_at_c_peak']], [c[np.argmax(c)]], '*', color='gold',
            ms=18, mec='k', zorder=5, label=f"峰位 h/J={v['hj_at_c_peak']:.2f}")
    ax.set_xlabel('h/J'); ax.set_ylabel('$c_{fit}$ (Calabrese-Cardy)')
    ax.set_title(f"[主判据] c_fit 峰值在临界点\n{'PASS' if v['ok_c_peak_at_critical'] else 'FAIL'}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(hj, S, 's-', color='#2980b9', lw=2, ms=6)
    ax.axvline(HJ_CRIT, color='gray', ls=':', lw=1.2, label='h/J=1.0')
    ax.axvspan(*HJ_S_WINDOW, color='green', alpha=0.12,
               label=f"窗口 {HJ_S_WINDOW}")
    ax.axvline(v['hj_at_s_peak'], color='orange', ls='--', lw=1.5,
               label=f"峰位 h/J={v['hj_at_s_peak']:.2f}")
    ax.set_xlabel('h/J'); ax.set_ylabel('S(L/2)')
    ax.set_title(f"[次判据] 半链纠缠熵峰位\n{'PASS' if v['ok_s_peak_in_window'] else 'FAIL'}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(hj, gap, '^-', color='#27ae60', lw=2, ms=6)
    ax.axvline(HJ_CRIT, color='gray', ls=':', lw=1.2, label='h/J=1.0')
    ax.set_xlabel('h/J'); ax.set_ylabel('gap = $p_1/p_0$')
    ax.set_title(f"[诊断·不作判据] gap 单调递减 = "
                 f"{'是' if v['gap_monotone_decreasing'] else '否'}\n"
                 f"(无内部极值 -> 原判据不可用)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f"v14·F2a 负对照: h/J 扫描 (L={scan['L']})  "
                 f"总裁决 {'PASS' if v['passed'] else 'FAIL'}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_png
