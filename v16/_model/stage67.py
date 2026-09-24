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
"""spiral_model_v16._model.stage67 —— 由 spiral_model_v16.py 拆分。

阶段六·时空孕育显生命 (Gray-Scott) + 闭环螺旋 + 阶段七·意识自照见

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
    boundary_correlation_graph, boundary_correlation_length, exact_ground_state, fit_central_charge,
)
from _model.mera import (
    mera_fit, mera_init,
)




# ============================================================================
# 闭环螺旋 · spiral_loop()
# ============================================================================
def spiral_loop(turns=6, L0=8, scale_knob=1.0, max_L=16, loop_tol=0.02,
                seed=0):
    """
    多圈自举: 把整条七阶段链当作一个映射, 反复应用到自身。

    ---------------------------------------------------------------------------
    尺度更新律 (闭环的"自指"部分)
    ---------------------------------------------------------------------------
        第 t 圈的系统尺寸由第 t-1 圈**实测**的关联长度定向:
            L_t = clip(round(scale_knob * xi_{t-1}), L0, max_L)
        第 0 圈没有上一圈, 用旋钮 L0 起步。
    也就是说: 模型测出来的关联长度, 决定模型下一次在多大的尺度上运行。
    尺度不再是外部写死的常数: 它是上一圈自己测出来的。

    ---------------------------------------------------------------------------
    跟踪哪三个量, 以及为什么必须跟踪三个而不是一个
    ---------------------------------------------------------------------------
    闭环到底收不收敛, **取决于问哪个量**。这三个的答案是不一样的:

      (1) c_t          中心荷。应趋于 1/2 —— 这是共形场论不动点,
                       是"模型自我应用时复现自己"的那一面。

      (2) xi_t / L_t   无量纲比值。应趋于常数 (实测约 1.2) ——
                       这才叫尺度不变。**关键**: 绝不能用 xi 的绝对值
                       判断收敛。临界系统的关联长度被系统尺寸截断,
                       实测 xi = 19.314 > L = 16, xi 就是跟着 L 涨的。
                       用 xi 判收敛会得到"永远发散"的错误结论。

      (3) 相对谱隙 1 - p1/p0。应趋于 0 —— 这是**临界**的标志。
          (p1/p0 本身是收敛比, 它趋于 1; 初稿说"p1/p0 -> 0"方向反了。)
                       (能隙闭合)。这个量就是因果链 L2 里那个派生量,
                       所以闭环与因果链在这里咬合上了。

    收敛判据只作用于无量纲量 (c 与 xi/L), 因为只有它们才是尺度不变量。

    ---------------------------------------------------------------------------
    诚实边界 (必须在代码里说清楚, 否则这个闭环会自我欺骗)
    ---------------------------------------------------------------------------
    (a) **xi 随 L 发散不是失败, 是临界的定义。** 若把"xi 不收敛"报成闭环
        失败, 那就把最漂亮的物理结论说反了。脚本对这一点有专门打印。
    (b) **max_L 是旋钮, 且实测会触顶。** 去掉上限 L 会一直涨 —— 又是临界性。
        脚本把触顶如实打印, 不假装它是自然收敛。**触顶原因分两条, 都不是
        "算力上限"** (v15·W5 实测更正): (i) 每圈主结果走精确对角化, 它**没有**
        16 这个上限 —— L=18 实测建 H 1.53s + eigsh 3.86s + 峰值 433MB, 而且
        基态仍与 Jordan-Wigner 闭合式吻合到 2.5e-14; (ii) 真正卡住 16 的是
        **末圈 MERA 交叉校验**: quimb 的 qtn.MERA 要求 L 为 2 的幂, 而 16 = 2^4。
        所以 16 是"既落在本机算力内、又是 2 的幂"的那个值, 不是 2^L 维稀疏
        对角化的边界。故"闭环能否真自组织到临界"在本工作条件下仍无法回答 ——
        但那是因为**存在上限**这件事本身, 不是因为 16 这个数。
    (c) **本函数用精确对角化做 oracle**, 每圈直接从基态算 c / xi / gap,
        不走 MERA。理由: 若每圈都跑 MERA, 键维截断误差会与尺度流动混淆,
        分不清"c 在流动"还是"截断在漂移"。末圈单独做一次 MERA 交叉校验。
    (d) 每圈 L 取偶数 (自旋链半链切分需要), 且 <= max_L。**注意"偶数"并不够**:
        末圈 MERA 交叉校验还额外要求 L 是 2 的幂, 非 2 的幂时该步降到不大于
        L_last 的最大 2 的幂并如实打印降级 (W5 之前, max_L ∈ {10,12,14} 会在
        这里抛 ValueError, 把整轮已经算完的结果一起带走)。稀疏 Lanczos 在
        L=18 上实测 5.4s/433MB, 所以"本机舒服的范围"远不止 2^16。

    返回每圈的记录; 同时返回一个 verdict 字符串, 说明"收敛"是对哪些量说的。
    """
    print("\n" + "=" * 76)
    print("  闭环螺旋 spiral_loop() · 多圈自举")
    print("=" * 76)
    print(f"  尺度律: L_t = clip(round({scale_knob} * xi_(t-1)), {L0}, {max_L})")
    print(f"  收敛判据: 无量纲量 (c, xi/L) 的相对变化 < {loop_tol:.1%}")
    print(f"  注意: xi 的绝对值**不**参与收敛判据 —— 临界时它随 L 发散, ")
    print(f"        那是临界的定义, 不是失败。")

    rows = []
    L_prev = None
    prev_c, prev_ratio = None, None
    clamped = False
    converged_at = None

    for t in range(turns):
        if t == 0:
            L = int(L0)
            L_source = f'旋钮 L0={L0} (第 0 圈无上一圈)'
        else:
            raw = int(round(scale_knob * rows[-1]['xi']))
            L = int(np.clip(raw, L0, max_L))
            clamped = clamped or (raw != L)
            L_source = (f'round({scale_knob}*xi_{t-1}={rows[-1]["xi"]:.3f}) '
                        f'= {raw}' + (f' -> clip 到 {L}' if raw != L else ''))
        # 半链切分要求偶数
        L = L - (L % 2)

        # ---- 本圈的物理量: 全部来自精确对角化 (oracle) ----
        E0, gs = exact_ground_state(L, 1.0, 1.0)
        half = L // 2
        s = np.linalg.svd(gs.reshape(2 ** half, 2 ** (L - half)),
                          compute_uv=False)
        p = s ** 2
        p = p / p.sum()
        p_nz = p[p > 1e-15]
        c_fit = fit_central_charge(gs, L)
        _, C, _ = boundary_correlation_graph(gs, L)   # C_ij = <sigma^z_i sigma^z_j>
        xi = boundary_correlation_length(C, L)
        gap = float(p_nz[1] / p_nz[0]) if len(p_nz) > 1 else float('nan')
        ratio = xi / L

        # 无量纲量的相对变化 (只有它们参与收敛判据)
        dc = abs(c_fit['c'] - prev_c) / abs(prev_c) if prev_c else float('nan')
        dr = abs(ratio - prev_ratio) / abs(prev_ratio) if prev_ratio else float('nan')

        rows.append({'turn': t, 'L': L, 'L_source': L_source,
                     'E0_per_L': E0 / L, 'c': c_fit['c'],
                     'xi': float(xi), 'xi_over_L': float(ratio),
                     'gap': gap, 'entropy': float(-np.sum(p_nz * np.log(p_nz))),
                     'dc_rel': dc, 'dratio_rel': dr})

        print(f"\n  第 {t} 圈: L={L}   [{L_source}]")
        print(f"     E0/L={E0/L:.6f}, c={c_fit['c']:.4f}, "
              f"xi={xi:.3f}, xi/L={ratio:.4f}, gap={gap:.6f}")
        if t > 0:
            print(f"     相对上圈: |dc|/c={dc:.4f}, |d(xi/L)|/(xi/L)={dr:.4f}")

        if (t > 0 and np.isfinite(dc) and np.isfinite(dr)
                and dc < loop_tol and dr < loop_tol and converged_at is None):
            converged_at = t
            print(f"     -> 无量纲量已收敛 (第 {t} 圈)")

        prev_c, prev_ratio = c_fit['c'], ratio

    # ---- 末圈 MERA 交叉校验 (诚实边界 (c)) ----
    # v15·W5 修正: MERA 库 (quimb 的 qtn.MERA) **要求 L 是 2 的幂**, 而
    # L_last = clip(round(xi), L0, max_L) 可以是任意偶数。实测 max_L ∈ {10,12,14}
    # 时这里直接抛 ValueError("L should be a power of 2"), 而且是在每圈物理都算完、
    # "无量纲量已收敛"都打印出来之后才崩 —— 整条已经做出来的结果被一起带走。
    # 修法不是"把 L 凑成 2 的幂"(那会篡改闭环自己的尺度律), 而是让**这一步降级**:
    # 交叉校验只是"键维截断量级"的旁证, 不是主结论 (主结论走精确对角化, 见边界 (c)),
    # 所以取不大于 L_last 的最大 2 的幂做这一步, 并如实打印降级这件事。
    L_last = rows[-1]['L']
    L_mera = 2 ** int(np.floor(np.log2(max(2, L_last))))
    chi_cross = max(2, int(2 ** int(np.ceil(np.log(np.exp(rows[-1]['entropy']))
                                             / np.log(2)))))
    print(f"\n  [交叉校验] 末圈 L={L_last} 用 MERA (chi={chi_cross}) 复核 c:")
    if L_mera != L_last:
        print(f"    注意: quimb 的 MERA 要求 L 为 2 的幂, {L_last} 不是 -> "
              f"本步降到 L={L_mera}")
        print(f"    降级只影响这条旁证, 不动上面的主结果。")
    m_cross, raws_cross = mera_init(L_mera, chi_cross, seed=0)
    _, gs_mera = exact_ground_state(L_mera, 1.0, 1.0)
    # 步数远少于主流程 (V13_STEPS=1500): 这是**有意的欠优化** —— 这一步问的是
    # "截断到 chi 时 c 会偏多少", 不是把 MERA 调到最好。它必须连同 c 一起进 json,
    # 否则单独看 c 会把它读成一个中心荷读数。
    cross_steps = 400
    psi_cross, _ = mera_fit(m_cross, raws_cross, gs_mera, L_mera,
                            steps=cross_steps, verbose=False)
    c_cross = fit_central_charge(psi_cross, L_mera)
    c_exact_mera = float(fit_central_charge(gs_mera, L_mera)['c'])
    ov_cross = float(abs(np.vdot(gs_mera, psi_cross)))
    # |dc| 必须**同 L 比** (MERA vs 精确都在 L_mera 上), 否则混进尺寸差。
    # 降级时尤其要守这条: 拿 L=8 的 MERA 去比 L=12 的精确值, 那个差里既有
    # 截断又有尺寸, 而这句话声称的只是截断。
    print(f"     MERA(L={L_mera}): c={c_cross['c']:.4f}, 重叠={ov_cross:.4f} "
          f"| 精确(同 L={L_mera}): c={c_exact_mera:.4f}")
    print(f"     差 |dc| = {abs(c_cross['c'] - c_exact_mera):.4f} "
          f"—— 同 L 比, 所以这是键维截断的量级, 不是流动")
    if L_mera != L_last:
        print(f"     主结果不受影响: 末圈精确 c(L={L_last}) = "
              f"{rows[-1]['c']:.4f}")

    # ---- 收尾判定: 三个量分开下结论 ----
    #
    # 两处必须说清楚的坑 (初稿都踩了, 这里已修正):
    #
    # (坑一) 不能拿第 0 圈当"流动的起点"来判收敛。
    #   第 0 圈的 L0 是**旋钮**, 不是尺度律的产物; 它必然离不动点远。
    #   拿 seed 和末圈比, 得到的一定是"不稳定", 而这与逐圈的
    #   |d(xi/L)|/(xi/L) -> 0 直接矛盾。所以收敛判据只对**非 seed 圈**
    #   (t >= 1) 生效, seed 单列报告。
    #
    # (坑二) p1/p0 **不是**"谱隙", 它是幂迭代的**收敛比**。
    #   记 r = p1/p0, 则相对谱隙 = 1 - r。临界时 Schmidt 谱变密变慢,
    #   r -> 1, 于是**相对谱隙 1-r -> 0**, 这才是"谱隙闭合"。
    #   初稿写成 "gap_t = p1/p0 应趋于 0", 方向搞反了 —— 实测
    #   r: 0.309 -> 0.343 -> 0.366 一路上升, 对应的 1-r: 0.691 -> 0.657
    #   -> 0.634 一路下降, **谱隙确实在闭合**。判定按 1-r 下。
    c_all = [r['c'] for r in rows]
    ratio_all = [r['xi_over_L'] for r in rows]
    gap_all = [r['gap'] for r in rows]           # 收敛比 r = p1/p0
    relgap_all = [1.0 - g for g in gap_all]      # 相对谱隙 1 - r
    c_flow, ratio_flow, relgap_flow = c_all[1:], ratio_all[1:], relgap_all[1:]
    verdicts = {
        'c_toward_half': bool(abs(c_all[-1] - 0.5) < 0.05),
        # 判据用非 seed 圈: 比值在流动中是否稳定下来
        'ratio_stable': bool(np.ptp(ratio_flow) < 0.15 * np.mean(ratio_flow)),
        # 收敛比上升 <=> 相对谱隙闭合 (趋向临界)
        'relgap_closing': bool(relgap_flow[-1] < relgap_flow[0]),
        'scale_clamped': bool(clamped),
        'converged_at_turn': converged_at,
    }
    print("\n  闭环裁决 (三个量分开说, 不合并成一个'收敛'):")
    print(f"    [seed] 第 0 圈由旋钮 L0={L0} 给定, 不是尺度律的产物,")
    print(f"           故不参与收敛判定 (与末圈比一定'不稳定', 那是假信号)。")
    print(f"    c_t: {c_all[0]:.4f} -> {c_all[-1]:.4f}  "
          f"(理论 0.5)  {'趋于 1/2' if verdicts['c_toward_half'] else '未收敛到 1/2'}")
    print(f"    xi_t/L_t (t>=1): {ratio_flow[0]:.4f} -> {ratio_flow[-1]:.4f}  "
          f"{'尺度不变 (比值稳定)' if verdicts['ratio_stable'] else '不稳定'}")
    print(f"    收敛比 r=p1/p0: {gap_all[0]:.6f} -> {gap_all[-1]:.6f}  "
          f"(上升 = Schmidt 谱变密)")
    print(f"    相对谱隙 1-r:   {relgap_all[0]:.6f} -> {relgap_all[-1]:.6f}  "
          f"{'谱隙闭合 (趋向临界)' if verdicts['relgap_closing'] else '谱隙未闭合'}")
    print(f"    xi 绝对值: {rows[0]['xi']:.3f} -> {rows[-1]['xi']:.3f}  "
          f"(随 L 增长 —— 这是临界的定义, 不是失败)")
    if clamped:
        print(f"    旋钮 max_L={max_L} 已触顶: 尺度 L 的流动被上限截断。")
        print(f"    触顶原因分两条, 都不是『算力上限』(v15·W5 实测更正):")
        print(f"      * 每圈主结果走精确对角化, 它**没有** 16 这个上限 —— L=18 "
              f"实测建 H 1.53s")
        print(f"        + eigsh 3.86s + 433MB, 基态仍与 Jordan-Wigner 吻合到 2.5e-14;")
        print(f"      * 卡住 16 的是末圈 MERA 交叉校验: quimb 的 MERA 要求 L 是 "
              f"2 的幂。")
        print(f"    所以 {max_L} 是『既落在本机算力内、又是 2 的幂』的那个值。")
        print(f"    所以『闭环能否真自组织到临界』在本工作条件下无法回答 ——")
        print(f"    但这是因为**存在上限**这件事本身, 不是因为 {max_L} 这个数。")
        print(f"    去掉上限 L 会继续涨 —— 同样是临界性, 不是闭环缺陷。")
        print(f"    注意 (诚实边界): 谱隙闭合依赖 L 增长; L 被夹住之后")
        print(f"    1-r 会停在 L={max_L} 的值上不再下降, 这不是模型停止趋向")
        print(f"    临界, 而是**闭环再也拿不到更大的 L**。")
    # v15 修正 (既存缺陷): 原文把两个**不同源**的判据塞进同一句 —— 前半句
    # "收敛"来自 verdicts['ratio_stable'] (ptp(xi/L 流) < 15% * mean, 松,
    # 且只覆盖 t>=1), 括号里的"第 N 圈起逐圈变化 < 2%" 来自 converged_at
    # (逐圈 |dc|/c 与 |d(xi/L)|/(xi/L) **都** < 2%, 严)。闭环从未满足后者时
    # converged_at 是 None, 于是印出「收敛 (第 None 圈起 …)」—— 自相矛盾,
    # 而且把 15% 判据的结论挂到 2% 的措辞上。两个判据分开报, 各自说自己的阈值。
    _conv_txt = (f"第 {converged_at} 圈起满足"
                 if converged_at is not None
                 else "从未有某一圈同时满足")
    print(f"\n  结论: 无量纲不变量"
          f"{'收敛' if verdicts['ratio_stable'] else '未收敛'}"
          f" (尺度不变判据: xi/L 流的极差 < 15% 均值), ")
    print(f"        逐圈判据 (阈 {loop_tol:.0%}): {_conv_txt};")
    print(f"        而尺度本身发散。模型自我应用时复现 CFT 不动点;")
    print(f"        唯一'不收敛'的那一项恰好是临界性。")

    # 这条 mera_cross 必须**自描述**。原先把 c 单独存出去, 配对的 c_exact_mera
    # 只活在打印里 —— 只拿到 json 的人看到 c=0.6004 会把它读成这个系统的中心荷,
    # 而它其实是 cross_steps 下的欠优化旁证, 只有和**同 L** 的精确值并排才读得出
    # "键维截断的量级"这个意思。另外原先存的是 L_last, 降级发生时 (L_last 不是
    # 2 的幂) 存的就不是真正用来算的那个 L —— 那是记错账。
    return {'rows': rows, 'verdicts': verdicts, 'mera_cross': {
        'L_mera': L_mera, 'L_last': L_last,
        'degraded': bool(L_mera != L_last),
        'chi': chi_cross, 'steps': cross_steps,
        'c': c_cross['c'],
        'c_exact_same_L': c_exact_mera,
        'dc_abs': abs(c_cross['c'] - c_exact_mera),
        'overlap': ov_cross,
        'note': ('旁证而非独立读数: 同一 L 上 MERA(欠优化) 与精确值的差, '
                 '读作键维截断的量级; 不可单独引用为 c')}}




def stage6_life(F=0.035, k=0.060, N=40, L=0.5, steps=40000, dt=1.0,
                Du=2e-5, Dv=1e-5):
    """
    标准 Gray-Scott 反应-扩散 (Pearson 的斑点相区参数):
        du/dt = Du Lap u - u v^2 + F(1-u)
        dv/dt = Dv Lap v + u v^2 - (F+k) v
    在 N x N 的周期域 L x L 上做显式欧拉。

    【数值稳定性的硬约束 —— 这是本函数最容易被糊弄过去的地方】
    二维 5 点拉普拉斯 + 显式欧拉的稳定条件是  dt * Du / h^2 <= 1/4 (h = L/N)。
    超限不会得到"更剧烈的斑图", 只会得到棋盘状的数值失稳 (饱和到 [0,1] 的
    假斑图)。本函数把 dt*Du/h^2 算出来并报告, 且不用 np.clip 去掩盖失稳。
    分辨率也必须够: 模式波长 ~0.05 域长单位, h 大到 0.06 时每个波长只有
    1 个格点, 拍不出来。经典 Pearson 用 h ~ 0.01; 这里 L=0.5, N=40 给
    h=0.0125, 与之一致。

    诚实边界: 这是反应-扩散的数学斑图, 不是生物生命。而且它跑在平直网格上
    —— 本函数不假装扩散张量来自阶段五的关联几何; 两者的特征尺度由
    boundary_correlation_length() 与下面的 pattern_spacing 分别报出。
    """
    h = L / N
    stab = dt * Du / h ** 2
    if stab > 0.25:
        return {'ok': False, 'h': h, 'stab': stab, 'v': None,
                'contrast': float('nan'), 'emerged': False,
                'reason': f'不满足稳定性条件 dt*Du/h^2={stab:.4f} > 0.25, '
                          f'拒绝给出可能失稳的结果'}

    u = np.ones((N, N))
    v = np.zeros((N, N))
    c = N // 2
    u[c - 2:c + 2, c - 2:c + 2] = 0.5
    v[c - 2:c + 2, c - 2:c + 2] = 0.25
    for _ in range(steps):
        lu = (np.roll(u, 1, 0) + np.roll(u, -1, 0) +
              np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4 * u) / h ** 2
        lv = (np.roll(v, 1, 0) + np.roll(v, -1, 0) +
              np.roll(v, 1, 1) + np.roll(v, -1, 1) - 4 * v) / h ** 2
        uv2 = u * v ** 2
        u = u + dt * (Du * lu - uv2 + F * (1 - u))
        v = v + dt * (Dv * lv + uv2 - (F + k) * v)

    finite = bool(np.isfinite(u).all() and np.isfinite(v).all())
    contrast = float(v.max() - v.min()) if finite else float('nan')
    active = float((v > 0.1).mean()) if finite else 0.0
    # 斑图波长: v 沿 x 的平均零交叉间隔 * 2 (一个完整周期有两次穿越)
    zc = np.mean([len(np.where(np.diff(np.sign(v[i] - v[i].mean())))[0])
                  for i in range(N)]) if finite else 0.0
    spacing = float(2.0 * L / zc) if zc > 0 else float('inf')
    emerged = bool(finite and contrast > 0.2 and active > 0.05)

    print(f"[阶段六] Gray-Scott 生命斑图 (平直周期网格):")
    print(f"  网格 h={h:.4f}, dt*Du/h^2={stab:.4f} (稳定上限 0.25) -> 数值稳定")
    print(f"  对比度={contrast:.3f}, 活化面积占比={active:.3f}, "
          f"斑图波长~{spacing:.4f} (域长 {L})")
    print(f"  -> {'涌现' if emerged else '未涌现'}"
          f"  (判据: 对比度>0.2 且 活化面积>5%)")
    return {'ok': True, 'u': u, 'v': v, 'contrast': contrast,
            'active_frac': active, 'spacing': spacing, 'h': h, 'stab': stab,
            'steps': steps, 'emerged': emerged, 'finite': finite}




def gray_scott_invariants(N=40, L=0.5, steps=20, F=0.035, k=0.060,
                          dt=1.0, Du=2e-5, Dv=1e-5):
    """
    Gray-Scott 显式欧拉格式的两条**精确**离散不变量。

    【为什么不是"u+v 质量守恒"】
    更新式是
        du/dt = Du Lap u - u v^2 + F(1-u)
        dv/dt = Dv Lap v + u v^2 - (F+k) v
    反应项 -u v^2 与 +u v^2 逐点相消, 但**源汇项不相消**。对全空间取平均:
        d/dt <u+v> = Du <Lap u> + Dv <Lap v> + F(1-<u>) - (F+k)<v>
    周期域上两个 Laplacian 项精确为 0, 剩下的 F(1-<u>) - (F+k)<v> 一般不为零
    (只有恰好已在稳态时才为零)。所以"u+v 守恒"是个**伪不变量**: 实测第一步
    就偏离 6.25e-5, 比"守恒到 1e-6"的容差大 62 倍, 累计 2000 步漂移 0.278。
    照着它写守卫, 只会把正确的物理判成失败。

    换成的两条都是真命题, 且各管一段:

      (a) **离散拉普拉斯零和** <Sum Lap u> == 0。
          周期包裹下四个 roll 是同一组数的置换, 所以四个和严格相等、分子
          精确为 0 (浮点上只留抵消误差)。它测的是 stencil 系数 (-4 中心)
          与 roll 的包裹方向 —— 与守恒律无关, 纯粹是"算子写对了没有"。

      (b) **精确离散平衡恒等式**: 用**更新前**的场算
            <u+v>_{n+1} - <u+v>_n == dt [ F(1-<u>_n) - (F+k)<v>_n ]
          这不是物理守恒, 而是同一个更新式的算术恒等式 (Laplacian 项在求
          平均时精确消掉)。它测的是整个 RHS 的装配: 少一项、符号反、dt 位置
          错, 都会立刻在这里暴露。实测吻合到 3.6e-16。

    诚实边界: 两条都是**格式性质**, 不是"Gray-Scott 守恒量"这样的物理命题。
    它们保证的是"这段代码按写下的方程在算", 不保证那个方程描述的是生命。
    """
    h = L / N
    u = np.ones((N, N))
    v = np.zeros((N, N))
    c = N // 2
    u[c - 2:c + 2, c - 2:c + 2] = 0.5
    v[c - 2:c + 2, c - 2:c + 2] = 0.25

    lap_rel_max = 0.0
    bal_abs_max = 0.0
    drift_first = float('nan')
    for n in range(steps):
        lu = (np.roll(u, 1, 0) + np.roll(u, -1, 0) +
              np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4 * u) / h ** 2
        lv = (np.roll(v, 1, 0) + np.roll(v, -1, 0) +
              np.roll(v, 1, 1) + np.roll(v, -1, 1) - 4 * v) / h ** 2

        # (a) 零和: 归一化成无量纲的抵消残差 (除以 4<u>/h^2 这个量级)
        lu_rel = abs(float(lu.sum())) * h ** 2 / (4.0 * abs(float(u.sum())))
        lv_rel = abs(float(lv.sum())) * h ** 2 / (4.0 * abs(float(v.sum())) + 1e-30)
        lap_rel_max = max(lap_rel_max, lu_rel, lv_rel)

        # (b) 平衡恒等式: 预言的增量用更新前的场算
        m_before = float((u + v).mean())
        pred = dt * (F * (1.0 - float(u.mean())) - (F + k) * float(v.mean()))
        uv2 = u * v ** 2
        u = u + dt * (Du * lu - uv2 + F * (1 - u))
        v = v + dt * (Dv * lv + uv2 - (F + k) * v)
        m_after = float((u + v).mean())
        if n == 0:
            drift_first = float(m_after - m_before)
        bal_abs_max = max(bal_abs_max, abs((m_after - m_before) - pred))

    return {'lap_zero_rel_max': lap_rel_max,
            'balance_abs_err_max': bal_abs_max,
            'drift_first_step': drift_first,
            'N': N, 'L': L, 'h': h, 'steps': steps, 'dt': dt,
            'F': F, 'k': k}




# ============================================================================
# 阶段七 · 生命涌现意识自照见空无 (二阶自指闭环)
# ============================================================================
def stage7_consciousness(coupled):
    """
    二阶自指 = 以阶段三-c 的联合不动点 W* 作为"照见"闭环:
      D* = W* 的主特征方向; 熵比 = H(D*)/ln N。
      '照见空无' = D* 接近等权 (熵比 -> 1); '照见结构' = D* 偏离等权。
    诚实: 照见的内容取决于缘起的条件。变体 A (状态自由) 在 g=1 下**结构上
    没有非零不动点** —— 不动点要求 W* = x* x*^T 且 ||W*||_2 = 1 (每步谱归一),
    故 ||x*|| = 1; 代入状态式得 W* x* = x*, 于是 x* = tanh(g x*), g=1 时只有
    x* = 0 解。所以 A 只能塌到 x* = 0, 此时如实报"空无", 不伪造结构。
    变体 B (状态归一) 每步归一使 ||x*|| = 1 自洽, 才有真不动点 W* = x* x*^T,
    其主方向就是状态自己 —— 这才是"照见自己"的数学形式。

    v15 修正 (W4): v14 用**变体 A** 的 collapsed 做判据, 却取**变体 B** 的 W 做
    读数, 非塌缩支的 note 还直接引 B 的 sigma 比 —— 两个变体混在一条结论里,
    而 docstring 的语义(「A 塌缩 => 报空无」)与代码的 else 支(A 没塌缩时反而
    报结构)互相矛盾。v14 之所以没暴露: 生产路径上 N=17 时 A 一律塌缩, 走的是
    不带 W 的那一支, 这处不一致是**死代码**。
    现改为: 各变体**读自己的 W**; A/B 两套读数**分列**; 顶层读数据 A(阶段七
    叙事的主体), 与原生产路径的数值一致。
    """
    g = float(coupled.get('_g', 1.0))     # 由 selfref_coupled 带出, 不另写常量

    def _readout(tag, data, W_t):
        """从**该变体自己的** W* 读出 D*/熵比/等权度。"""
        n = W_t.shape[0]
        if data['collapsed']:
            # x* = 0 时 W* 没有可读的主方向, D* 不存在。这一支取值**一律是约定**:
            # probs = 1/N 与 entropy = ln N 是同一件事的换写, 所以熵比恒等于 1;
            # 等权度也恒等于 1。它们是"空无"这个名字的定义, 不是关于系统的观测。
            # 如实标 defined=False, 免得被当成"照见空无"的独立证据。
            return {'N': n, 'probs': np.ones(n) / n, 'entropy': float(np.log(n)),
                    'eqw': 1.0, 'status': '空无', 'defined': False, 'D': None,
                    'note': f'{tag} 塌缩到 x*=0, 无可读的 D*; '
                            f'均匀分布是本处约定, 非测量'}
        W = W_t.detach().numpy()
        ev, evec = np.linalg.eig(W)
        i = int(np.argmax(np.abs(ev)))
        D = evec[:, i]
        D = D / np.linalg.norm(D)
        probs = np.abs(D) ** 2
        probs = probs / probs.sum()
        entropy = float(-np.sum(probs * np.log(probs + 1e-15)))
        eqw = float(abs(np.vdot(np.ones(n) / np.sqrt(n), D)))
        return {'N': n, 'probs': probs, 'entropy': entropy, 'eqw': eqw,
                'status': '结构', 'defined': True, 'D': D,
                'note': f'D* = {tag} 的 W* 主方向 '
                        f'(sigma1/sigma2={data["sigma_ratio"]:.1e})'}

    def _rho_jac(data, W_t, x_t):
        """
        不动点 Jacobian 的谱半径 —— 自指深度。
          x_{n+1,i} = tanh(g (W x)_i) ⇒ J = g · diag(1 - x*^2) · W*
        (不动点处 tanh(g W x*)_i = x*_i, 故 1 - tanh^2 = 1 - x*_i^2)
        返回 (rho, is_identity)。is_identity=True 表示这个数**不是测量**:
        塌缩支 x* = 0 ⇒ J = g·W*, 而 W 每步做谱归一故 sigma_max(W*) = 1,
        于是 rho = g 恒成立 —— 它只复述了"W 被归一"这个定义。
        """
        W = W_t.detach().numpy()
        if data['collapsed']:
            return float(g), True
        xs = x_t.detach().numpy()
        J = g * (1.0 - xs ** 2)[:, None] * W
        return float(np.max(np.abs(np.linalg.eigvals(J)))), False

    print("[阶段七] 二阶自指闭环:")
    A = _readout('A(状态自由)', coupled['free'], coupled['_W_free'])
    B = _readout('B(状态归一)', coupled['norm'], coupled['_W'])
    rho_A, idA = _rho_jac(coupled['free'], coupled['_W_free'], coupled['_x_free'])
    rho_B, idB = _rho_jac(coupled['norm'], coupled['_W'], coupled['_x'])

    # 顶层读数 = 变体 A。阶段七的叙事主体是"真自指"(状态不被外部归一),
    # 即 A; 取它与 v14 生产路径的数值一致(N=17 时 A 塌缩, 走的都是约定支)。
    N = A['N']
    probs, entropy, eqw = A['probs'], A['entropy'], A['eqw']
    status, defined, note = A['status'], A['defined'], A['note']
    ratio = entropy / np.log(len(probs))
    # 上界: 熵 <= ln N (均匀分布取最大熵) 与 |<等权向量, D*>| <= 1 (Cauchy-Schwarz)。
    # 塌缩支两条都恒取等号, 所以只有**非塌缩支**的这两条才带信息量。
    bound_ok = bool(ratio <= 1.0 + 1e-12 and eqw <= 1.0 + 1e-12)
    print(f"  变体 A(状态自由): 熵比={A['entropy'] / np.log(A['N']):.3f}, "
          f"等权度={A['eqw']:.3f} -> 照见{A['status']}"
          f"{'' if A['defined'] else ' (约定值, 非测量)'}")
    print(f"     ({A['note']})")
    print(f"  变体 B(状态归一): 熵比={B['entropy'] / np.log(B['N']):.3f}, "
          f"等权度={B['eqw']:.3f} -> 照见{B['status']}"
          f"{'' if B['defined'] else ' (约定值, 非测量)'}")
    print(f"     ({B['note']})")
    print(f"  自指深度 rho(J*): A={rho_A:.4f}{' [恒等式, 非测量]' if idA else ''}, "
          f"B={rho_B:.4f}{' [恒等式, 非测量]' if idB else ''}")
    if idA:
        print(f"     A 支恒等的原因: x*=0 ⇒ J*=g·W*, 而 sigma_max(W*)≡1 是谱归一"
              f"的定义, 故 rho≡g={g:g}。")
    return {'entropy_ratio': ratio, 'equal_weight': eqw, 'status': status,
            'note': note, 'defined': defined, 'bound_ok': bound_ok,
            'variant': 'A', 'rho_jac': rho_A, 'rho_is_identity': idA,
            'rho_jac_B': rho_B, 'rho_is_identity_B': idB,
            # probs 与 D 是**作图素材**(长度 N 的向量), 按本文件"中间量不落盘"
            # 的约定滤掉: 它们是"熵比/等权度是怎么算出来的"的原料, 不是指标本身。
            # 漏掉 D 会让 s7 带一个 ndarray 进 data, json.dump 直接崩在收尾处。
            'A': {k: v for k, v in A.items() if k not in ('probs', 'D')},
            'B': {k: v for k, v in B.items() if k not in ('probs', 'D')}}
