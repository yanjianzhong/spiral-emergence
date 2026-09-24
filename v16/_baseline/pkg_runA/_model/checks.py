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
"""spiral_model_v16._model.checks —— 由 spiral_model_v16.py 拆分。

F1 跨边界条件中心荷 + v15·W3a/W5/W3b + 任务A + F5 一致性检查束

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
    boundary_correlation_graph, boundary_correlation_length, entanglement_curve, exact_ground_state, fit_central_charge, jw_ground_energy, tfi_periodic_sparse,
)
from _model.mera import (
    V13_STEPS, mera_energy_only, mera_fit, mera_fit_v13, mera_init, mera_isometry_check,
)
from _model.stage67 import gray_scott_invariants




# ============================================================================
# F1 · 跨边界条件中心荷交叉检验 (v14)
# ============================================================================
#
# 审计 (§三之补 F1①) 要求的是**独立族估计量**: E0/L 与 S(n) 共用同一个
# Calabrese-Cardy ansatz, 只能算同族交叉检验, 必须降级。据此把 F1 拆成两块,
# 每一块都显式标注是哪一族:
#
#   (同族)   跨边界条件: PBC vs θ=π Z2 twist (反周期键)。
#            换的是**态**与**边界条件**, 不换估计量。
#   (独立族) **未建立**。
#            已试的 Rényi 指数扫描实测验证失败 (见 renyi_alpha_scan 的函数头),
#            如实记录为负面结果, 不伪造一个"看起来在检验"的守卫。
F1_L = 16


F1_CHI = 4


F1_SEEDS = (0, 1, 2)


F1_C_TRUE = 0.5                  # 临界 Ising 的中心荷真值


F1_REL_MAX = 0.05                # 跨边界条件相对差判据 (审计要求不放宽)



RENYI_ALPHAS = (0.5, 0.75, 1.5, 2.0, 3.0, 5.0, 8.0)


RENYI_NS = (2, 3, 4, 5, 6, 7, 8)


RENYI_STD_MAX = 0.02             # c(alpha) 在 alpha 上一致性的门槛


RENYI_DEV_TRUE_MAX = 0.05        # c(alpha) 均值对 0.5 的偏差门槛




def tfi_z2_twisted_sparse(L, J=1.0, h=1.0):
    """
    周期 TF-Ising, 但边界键 (L-1, 0) 取 +J 而非 -J。

    这就是 θ=π 的 Z2 twist, 物理上等于 Jordan-Wigner 的反周期 (NS) 扇区。
    与连续 U(1) twist 的区别是实测出来的 (见 _v14_probe_twist.py): 连续 twist
    给出复矩阵, 而 Z2 版本保持 H 为**实**矩阵 —— 所以能直接喂 dtype=float 的
    quimb MERA, 不需要动任何现有张量网络代码。

    位序与 tfi_periodic_sparse 完全相同 (站点 i <-> 位 L-1-i), 唯一差别是边界键
    符号, 这样 S(n) 的块划分与 PBC 情形**逐位可比** —— 这是"只换边界条件"这句
    话能成立的前提。
    """
    sx = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=float))
    sz = sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=float))

    Hx = None
    for i in range(L):
        b = L - 1 - i
        m = sp.kron(sp.eye(2 ** b, format='csr'),
                    sp.kron(sx, sp.eye(2 ** (L - 1 - b), format='csr')),
                    format='csr')
        Hx = m if Hx is None else Hx + m

    H = (-h) * Hx
    for (a, b) in [(i, (i + 1) % L) for i in range(L)]:
        mat = 1
        for j in range(L):
            mat = sp.kron(mat, sz if j in (a, b) else sp.eye(2, format='csr'),
                          format='csr')
        sgn = +1.0 if (a, b) == (L - 1, 0) else -1.0
        H = H + (sgn * J) * mat
    return ((H + H.T) / 2).tocsr()




def _schmidt_probs(psi, L, n):
    """块 A = 前 n 个连续站点的 Schmidt 概率谱 (已归一, 已截掉 < 1e-15)。"""
    psi = np.asarray(psi).reshape(-1)
    psi = psi / np.linalg.norm(psi)
    s = np.linalg.svd(psi.reshape(2 ** n, 2 ** (L - n)), compute_uv=False)
    p = s ** 2
    p = p[p > 1e-15]
    return p / p.sum()




def renyi_alpha_scan(gs, L=F1_L, alphas=RENYI_ALPHAS, ns=RENYI_NS):
    """
    F1 · 独立族估计量的**失败尝试**, 保留为可复现的负面证据。

    想法: Rényi 熵满足 S_alpha = (c/6)(1 + 1/alpha) ln[(L/pi) sin(pi n/L)]
    + const_alpha。对**固定 alpha 扫 n** 拟合得 c(alpha) —— 与 alpha=1 的 S(n)
    拟合相比, 它多检验了一个**独立的系数预言** (1 + 1/alpha), 失效方式也不同
    (若态不共形, alpha 依赖会先于 n 依赖垮掉)。看上去像个独立族。

    实测 (L=16, h/J=1.0, 解析 c=0.5):
        alpha  0.50     0.75     1.50     2.00     3.00     5.00     8.00
        c      0.53819  0.50838  0.51114  0.53184  0.56138  0.56820  0.55991
      均值 0.53986, 标准差 0.02252, 对 0.5 的偏差 7.97%。

    alpha=1 附近是自洽的 (0.508/0.511 正确夹住 0.5072, 说明实现本身没写错),
    但 alpha >= 2 后系统性飘高: Rényi 熵的次领头修正比 alpha=1 大得多, L=16
    的有限尺寸压不住。

    => **验证不通过, 不作独立族估计量**。在指标层注册为诊断项
       (expect_pass=False), 不计入通过率 —— 它的作用是留下"为什么没有独立族"
       的可复现证据, 而不是充当一个假的第二把尺子。
    """
    ns = list(ns)
    x = np.array([np.log((L / np.pi) * np.sin(np.pi * n / L)) for n in ns])
    probs = {n: _schmidt_probs(gs, L, n) for n in ns}
    rows = []
    for a in alphas:
        y = np.array([float(np.log(np.sum(probs[n] ** a)) / (1.0 - a))
                      for n in ns])
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        rows.append({'alpha': float(a),
                     'c': float(6.0 * coef[0] / (1.0 + 1.0 / a)),
                     'rms': float(np.sqrt(np.mean((y - A @ coef) ** 2)))})
    cs = np.array([r['c'] for r in rows])
    dev = float(abs(cs.mean() - F1_C_TRUE) / F1_C_TRUE)
    return {'alphas': rows, 'mean': float(cs.mean()), 'std': float(cs.std()),
            'range': float(cs.max() - cs.min()), 'dev_vs_true': dev,
            'L': L, 'ns': ns,
            'validated': bool(cs.std() < RENYI_STD_MAX
                              and dev < RENYI_DEV_TRUE_MAX)}




def cross_boundary_twist(L=F1_L, J=1.0, h=1.0, chi=F1_CHI, seeds=F1_SEEDS,
                         steps=None, lr=0.03, verbose=True):
    """
    F1 (同族) · 跨边界条件的中心荷交叉检验: PBC vs θ=π Z2 twist。

    **这一项换的是边界条件与态, 不是估计量** —— 两边都用同一个
    fit_central_charge (同一个 Calabrese-Cardy log 式)。所以它回答的是
    "c 的读数对边界条件稳不稳", 而不是"换一把尺子量"。审计要求的独立族
    估计量本工作**未建立** (原因见 renyi_alpha_scan 的函数头)。

    为何用检查点选择协议 (mera_fit_v13) 而不是朴素拟合:
    实测 (_v14_probe_twistmera.py) 朴素 600 步/单种子在 PBC 侧给出 c=0.56530、
    重叠 0.98829 —— 那正是"11.45% 未收敛假象": 它会把跨边界条件的差夸大成
    5.98%, **是优化没跑到位, 不是物理失配**。改用检查点选择 + 1500 步后
    PBC 侧 c=0.52257, 差降到 3.13%。

    每种边界 3 个种子, 取 eps_c 处于**中位数**的那条轨迹作代表 —— 与
    v13_tier1_runs 同一条可审计规则 (中位数 = 典型表现, 不用最好的一条)。
    """
    if steps is None:                    # 延迟解析: V13_STEPS 在本文件后面才定义
        steps = V13_STEPS

    E0_p, gs_p = exact_ground_state(L, J, h)
    H_t = tfi_z2_twisted_sparse(L, J, h)
    E_t, V_t = eigsh(H_t, k=1, which='SA')
    E0_t = float(E_t[0])
    gs_t = V_t[:, 0] / np.linalg.norm(V_t[:, 0])

    fit_ex = {'pbc': fit_central_charge(gs_p, L),
              'twist': fit_central_charge(gs_t, L)}
    twist_is_real = not np.iscomplexobj(H_t)

    if verbose:
        print("=" * 76)
        print("  任务 F1 · 跨边界条件中心荷交叉检验 (同族)")
        print("=" * 76)
        print(f"  边界条件: PBC vs θ=π Z2 twist (边界键 -J -> +J), "
              f"H 保持实矩阵 = {twist_is_real}")
        print(f"  L={L}, chi={chi}, 每种边界 {len(seeds)} 个种子 x {steps} 步 "
              f"(检查点按 overlap 选择, 不偷看 eps_c)")
        print(f"  精确态 E0/L: PBC={E0_p/L:+.6f}  twist={E0_t/L:+.6f}  "
              f"(边界条件确实换了态)")
        print(f"  精确态 S(L/2): PBC={fit_ex['pbc']['S'][-1]:.5f}  "
              f"twist={fit_ex['twist']['S'][-1]:.5f}")
        print(f"  精确态拟合 c: PBC={fit_ex['pbc']['c']:.5f}  "
              f"twist={fit_ex['twist']['c']:.5f}   "
              f"(rms {fit_ex['pbc']['rms']:.4f} / {fit_ex['twist']['rms']:.4f})")

    sides = {}
    for name, target in (('pbc', gs_p), ('twist', gs_t)):
        runs = []
        for seed in seeds:
            mera, raws = mera_init(L, chi, seed=seed)
            psi, ov, cstep, _c = mera_fit_v13(mera, raws, target, L,
                                              steps=steps, lr=lr, c_exact=None)
            fm = fit_central_charge(psi, L)
            runs.append({'seed': int(seed), 'c': float(fm['c']),
                         'rms': float(fm['rms']), 'overlap': float(ov),
                         'chosen_step': int(cstep)})
            if verbose:
                print(f"      [{name:>5} seed={seed}] ov={ov:.6f} "
                      f"@step{cstep:4d}  c={fm['c']:.6f}  (rms={fm['rms']:.4f})")
        cs = np.array([r['c'] for r in runs])
        i_med = int(np.argmin(np.abs(cs - np.median(cs))))
        sides[name] = {
            'runs': runs, 'i_representative': i_med,
            'c_median': float(np.median(cs)),
            'c_representative': float(cs[i_med]),
            'c_std': float(cs.std()),
            'c_exact': float(fit_ex[name]['c']),
            'rms_exact': float(fit_ex[name]['rms']),
            'S_half_exact': float(fit_ex[name]['S'][-1]),
            'E0_per_L': float((E0_p if name == 'pbc' else E0_t) / L),
        }

    c_p = sides['pbc']['c_representative']
    c_t = sides['twist']['c_representative']
    rel = abs(c_t - c_p) / abs(c_p)
    dev_true = {k: abs(sides[k]['c_representative'] - F1_C_TRUE) / F1_C_TRUE
                for k in sides}
    dev_ex = {k: abs(sides[k]['c_representative'] - sides[k]['c_exact'])
              / abs(sides[k]['c_exact']) for k in sides}
    min_ov = min(r['overlap'] for s in sides.values() for r in s['runs'])

    renyi = renyi_alpha_scan(gs_p, L)

    if verbose:
        print(f"\n  跨边界条件 (代表态 = eps_c 中位数轨迹):")
        print(f"     PBC   c = {c_p:.5f}  (种子间 std={sides['pbc']['c_std']:.5f})")
        print(f"     twist c = {c_t:.5f}  (种子间 std={sides['twist']['c_std']:.5f})")
        print(f"     相对差 = {rel:.3%}   (判据 < {F1_REL_MAX:.0%})")
        print(f"  两个偏差并列报告 (审计 F1② 要求, 5% 标准不放宽):")
        print(f"     对 c_exact(L) = {sides['pbc']['c_exact']:.5f} / "
              f"{sides['twist']['c_exact']:.5f}: "
              f"PBC {dev_ex['pbc']:.3%}, twist {dev_ex['twist']:.3%}")
        print(f"     对真值 0.5                    : "
              f"PBC {dev_true['pbc']:.3%}, twist {dev_true['twist']:.3%}")
        print(f"  [独立族尝试] Rényi 指数扫描: 均值 c={renyi['mean']:.5f}, "
              f"std={renyi['std']:.5f}, 对 0.5 偏差={renyi['dev_vs_true']:.2%}")
        print(f"     -> {'通过验证' if renyi['validated'] else '未通过验证'}: "
              f"{'可作独立族估计量' if renyi['validated'] else '不作独立族估计量 (如实记录为负面结果)'}")
        print("  诚实边界: 本项换的是**边界条件**与**态**, 不是估计量; ansatz 仍是")
        print("            同一个 Calabrese-Cardy log 式, 因此属**同族**交叉检验。")
        print("            本工作**未建立**真正的独立族中心荷估计量 (见上条负面结果)。")

    return {'L': L, 'chi': chi, 'seeds': [int(s) for s in seeds],
            'steps': int(steps), 'J': J, 'h': h,
            'c_true': F1_C_TRUE, 'rel_max': F1_REL_MAX,
            'boundary': {'E0_pbc': float(E0_p), 'E0_twist': E0_t,
                         'twist_h_is_real': bool(twist_is_real),
                         'S_half_pbc': float(fit_ex['pbc']['S'][-1]),
                         'S_half_twist': float(fit_ex['twist']['S'][-1])},
            'sides': sides, 'rel_diff': float(rel),
            'min_overlap': float(min_ov),
            'dev_vs_true': dev_true, 'dev_vs_c_exact': dev_ex,
            'renyi': renyi,
            'passed': bool(rel < F1_REL_MAX)}




# ============================================================================
# v15 · W3a 面积律 vs 对数律 —— 判据是 S(L/2) 随 L 的标度, 不是 S(l) 的形状
# ============================================================================
def area_vs_log_law(hj_list=(0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0),
                    L_list=(8, 10, 12, 14, 16), J=1.0):
    """
    答审计表 L3 行 ("纠缠熵 -> 面积律, 需扫描切割位置", 🔶) 与接口建议 P0
    ("L3 -> 纠缠熵面积律: 扫描切割位置, 验证 S(l) 形状")。

    **实测定下的结论: P0 那条按字面做不出来, 换一个判据才成立。**
    分两部分, 因为两部分的结果正好相反:

      部分 A (S(l) 的**形状**, L=16, h/J in {0.5, 1.0, 1.5}):
        对每个 h/J 把 对数律 S=(c/3)ln[(L/pi)sin(pi*l/L)]+const (2 参) 与
        面积律 S=const (1 参) 同时拟合, 比 RMSE 与 AIC。
        实测**对数律在每一个 h/J 上都赢** —— 包括 h/J=0.5 与 2.0 这种深在
        能隙相深处的点。所以"扫切割位置看形状"在 L<=16 上**没有区分力**:
        l 最大只到 L/2=8, 而 h/J=0.5 的关联长度约 2~3 格, S(l) 从 l≈3 就平了;
        但带自由 c 的对数形式能把"快速饱和"吸收掉 (h/J=0.5 时 c_log 缩到 0.065),
        于是它的 RMSE 反而比常数小。**这条照实登记为负面结果, 不粉饰成通过。**

      部分 B (S(L/2) 随 **L** 的标度) —— 这个才分得开:
        面积律: S(L/2) -> const;  对数律: S(L/2) = (c/3)ln L + b 发散。
        以 S=(c/3)ln L+b 拟合, c_scaling 即判别量:
          h/J=0.5  c=0.0007 (五点变化 <3e-4, 教科书式面积律)
          h/J=1.0  c=0.4976, R^2=1.0000 (对数律, c≈1/2)
          h/J=2.0  c=-0.0050 (平)
        非临界点上 c_scaling 反而轻微为负 (S 随 L 略降), 如实报出。

    判据 (按物理**先定**, 不是事后拟合):
      crit  : |c_scaling(1.0) - 0.5| < 0.05 且 R^2 > 0.99
      gapped: |c_scaling| < 0.05 于 h/J in {0.5, 1.5}
      peak  : argmax|c_scaling| 恰在 h/J = 1.0

    诚实边界:
      1. 部分 B 的杠杆臂只有 L: 8->16, 即 ln 上仅 2 倍。所以它是**定性区分**
         (平 vs 对数), 不是定量外推。c_scaling=0.4976 只当**弱杠杆臂下的
         交叉检验**读, 不当第三把精确尺子 —— 它有别于 W3b 那种闭式读数。
      2. 部分 B 用的是同一条 S 数据、只是换自变量, 故它与 Calabrese-Cardy 的
         l-拟合**不独立**; 真正独立的第二族是 W3b (能谱)。
      3. 五个点、2 倍范围的 R^2=1.0000 不能读成"对数律被证实"; 有区分力的是
         h/J=0.5 那个 c=0.0007 —— 同一套拟合在那边给出"完全不涨"。
    """
    partA, partB = [], []
    for hj in hj_list:
        s_half, cache = [], {}
        for L in L_list:
            _, gs = exact_ground_state(L, J, hj)
            cache[L] = gs
            s_half.append(float(entanglement_curve(gs, L)[-1]))

        # ---- 部分 A: S(l) 的形状 (只在计划点名的三个 h/J 上做) ----
        if hj in (0.5, 1.0, 1.5):
            L = L_list[-1]
            ns = np.arange(1, L // 2 + 1)
            S = entanglement_curve(cache[L], L)
            x = np.log((L / np.pi) * np.sin(np.pi * ns / L))
            A = np.vstack([x, np.ones_like(x)]).T
            coef, *_ = np.linalg.lstsq(A, S, rcond=None)
            rl, rc = S - A @ coef, S - S.mean()
            n = len(S)
            rss_l, rss_c = float(rl @ rl), float(rc @ rc)
            # AIC = n*ln(RSS/n) + 2k, k = 参数个数 (对数律 2, 面积律 1)
            aic_l = n * np.log(rss_l / n) + 2 * 2
            aic_c = n * np.log(rss_c / n) + 2 * 1
            partA.append({'hj': float(hj), 'L': int(L),
                          'c_logshape': 3.0 * float(coef[0]),
                          'rmse_log': float(np.sqrt(rss_l / n)),
                          'rmse_const': float(np.sqrt(rss_c / n)),
                          'aic_log': float(aic_l), 'aic_const': float(aic_c),
                          'dAIC_const_minus_log': float(aic_c - aic_l),
                          'winner': 'log' if aic_l < aic_c else 'const'})

        # ---- 部分 B: S(L/2) 随 L ----
        xL = np.log(np.array(L_list, dtype=float))
        Ab = np.vstack([xL, np.ones_like(xL)]).T
        yb = np.array(s_half)
        cb, *_ = np.linalg.lstsq(Ab, yb, rcond=None)
        rb = yb - Ab @ cb
        r2 = 1.0 - float(rb @ rb) / float(((yb - yb.mean()) ** 2).sum())
        partB.append({'hj': float(hj), 'S_half': s_half,
                      'c_scaling': 3.0 * float(cb[0]), 'b': float(cb[1]),
                      'r2': float(r2)})

    crit = next(p for p in partB if abs(p['hj'] - 1.0) < 1e-12)
    gapped = [p for p in partB if p['hj'] in (0.5, 1.5)]
    ok_crit = bool(abs(crit['c_scaling'] - 0.5) < 0.05 and crit['r2'] > 0.99)
    ok_gap = bool(all(abs(p['c_scaling']) < 0.05 for p in gapped))
    i_pk = int(np.argmax([abs(p['c_scaling']) for p in partB]))
    ok_peak = bool(abs(partB[i_pk]['hj'] - 1.0) < 1e-12)

    winners = {p['hj']: p['winner'] for p in partA}
    a_has_power = bool(len(set(winners.values())) > 1)
    verdict = {'ok_crit_log_law': ok_crit,
               'ok_gapped_area_law': ok_gap,
               'ok_peak_at_critical': ok_peak,
               'hj_at_c_peak': partB[i_pk]['hj'],
               'c_scaling_at_critical': crit['c_scaling'],
               'r2_at_critical': crit['r2'],
               'gapped_c_scaling': {p['hj']: p['c_scaling'] for p in gapped},
               'partA_winners': winners,
               'partA_has_discriminating_power': a_has_power,
               'passed': bool(ok_crit and ok_gap and ok_peak)}

    print(f"\n[v15·W3a] 面积律 vs 对数律 (L={L_list[0]}..{L_list[-1]})")
    print(f"  部分 A · S(l) 的形状 (L={L_list[-1]}), 对数律 vs 常数:")
    for p in partA:
        print(f"    h/J={p['hj']:4.2f}: c_logshape={p['c_logshape']:6.4f} "
              f"RMSE 对数={p['rmse_log']:.5f} 常数={p['rmse_const']:.5f} "
              f"dAIC(常数-对数)={p['dAIC_const_minus_log']:+8.3f} "
              f"-> {p['winner']} 胜")
    print(f"    **结论: 对数律在每一个 h/J 上都赢 -> S(l) 形状在 L<=16 下"
          f"无区分力**"
          if not a_has_power else "    (有区分力)")
    print(f"  部分 B · S(L/2) 随 L 的标度 S=(c/3)lnL+b:")
    print(f"    {'h/J':>5} {'c_scaling':>10} {'R2':>8}   S(L/2) 逐 L")
    for p in partB:
        print(f"    {p['hj']:5.2f} {p['c_scaling']:10.4f} {p['r2']:8.4f}   "
              + " ".join(f"{v:.4f}" for v in p['S_half']))
    print(f"  裁决: 临界点对数律 (|c-0.5|<0.05 且 R2>0.99) -> "
          f"{'PASS' if ok_crit else 'FAIL'} (c={crit['c_scaling']:.4f}, "
          f"R2={crit['r2']:.4f})")
    print(f"        非临界点面积律 (|c_scaling|<0.05) -> "
          f"{'PASS' if ok_gap else 'FAIL'} "
          f"({verdict['gapped_c_scaling']})")
    print(f"        |c_scaling| 峰位在 h/J={partB[i_pk]['hj']:.2f} -> "
          f"{'PASS' if ok_peak else 'FAIL'}")
    print(f"  诚实边界: 杠杆臂仅 L: 8->16 (ln 上 2 倍), 故这是**定性区分**"
          f"(平 vs 对数), 不是定量外推;")
    print(f"            c_scaling={crit['c_scaling']:.4f} 只当弱杠杆臂下的交叉"
          f"检验, 不当第三把精确尺子 (独立第二族见 W3b)。")
    return {'hj_list': [float(h) for h in hj_list],
            'L_list': [int(L) for L in L_list],
            'partA': partA, 'partB': partB, 'verdict': verdict}




# ============================================================================
# v15 · W5 有限尺寸标度 —— 把 L 当自变量, 逐 L 单独测同一套读数
# ============================================================================
def finite_size_scaling(L_list=(8, 10, 12, 14, 16, 18), J=1.0, h=1.0):
    """
    答审计表 P2 行 ("去掉 max_L 上限使 xi/L 收敛", 方向记反了那一条) 与
    W5 工作包 ("有限尺寸标度 / 弱 2x 杠杆臂")。

    与 spiral_loop 的分工 (这是本函数存在的理由)
    ---------------------------------------------------------------------------
      * spiral_loop 回答"闭环能否自组织到临界"。它的 L 是**上一圈自己测出来的**
        (L_t = clip(round(xi), L0, max_L)), 所以 L 序列不是我们能选的, 而且一定
        会被 max_L 夹住 —— 想扫 L 就得改 max_L, 那就改了闭环本身。
      * 本函数回答另一个问题: 那几个读数**作为 L 的函数**长什么样。L 是自变量,
        直接给出来, 不经过尺度律。于是它能问一句 spiral_loop 问不出来的话:
        **"L<=16 那条天花板到底限制了什么?"**

    与 spiral_loop 的读数一致性 (必须守住, 否则两者测的不是同一个量)
    ---------------------------------------------------------------------------
      估计量逐字同源: exact_ground_state / fit_central_charge /
      boundary_correlation_graph + boundary_correlation_length / 半链 SVD 的
      p1/p0 比。偶数处理也一样 (L - L%2, 半链切分要求)。所以同一个 L 上, 本函数
      的 c / xi / gap 与 spiral_loop 那一圈**必须是同一个数**; 若不同, 那是 bug,
      不是"两种口径"。

    边界声明 (v15·W5 实测, 必读)
    ---------------------------------------------------------------------------
      (1) **MERA 那条腿要求 L 是 2 的幂** (quimb 的 qtn.MERA 直接抛
          ValueError)。所以 {10,12,14,18} 上只有本函数的读数, 没有 MERA 读数。
          输出里逐行标注 MERA 可用性, 不假装能跑。MERA 可用集 = {8,16}。
      (2) **精确对角化那条腿没有 16 这个上限**: L=18 实测建 H 1.53s + eigsh
          3.86s + 峰值 433MB。所以本函数给的 L 范围比 max_L=16 大 —— 这正是
          它能回答 (1) 那个问题的原因。注释里"L<=16 是本机算力上限"的说法已按
          本实测更正。
      (3) 每个 L 都用 **Jordan-Wigner 闭合式独立核对 E0**。eigsh 只回答"给定这
          个 H, 最小本征值是多少", 对 H 本身对不对 (位序约定 / 周期键 / 场符号)
          一句话都没有; JW 是第二条独立路径, 所以它能测出 eigsh 测不出的东西。
      (4) 这里报的是**有限尺寸**读数, 不是热力学极限外推。c(L) 趋向 1/2 是收敛
          趋势, 不做 v->infinity 的拟合外推 (那需要 L 大得多)。所以本函数**不**
          声称"测到了 c=1/2"。
      (5) 杠杆臂 (两把尺子必须分开说, 混了就是夸大): L 的**比值** 8->18 是
          2.25x (旧的 8->16 是 2.00x), 看着宽了 12%; 但标度拟合的自变量是
          **ln L**, 在 log2 上只是 1.00x -> 1.17x。所以真正的杠杆臂只宽了
          **17%**, **不足以改善任何标度拟合** —— 扩 L 的价值不在拟合, 而在
          (2): 证明 16 不是这几个读数的墙。这条必须说准, 否则是夸大。
    """
    print("\n" + "=" * 76)
    print("  v15·W5 有限尺寸标度 · L 作自变量, 逐 L 单独测 spiral_loop 那套读数")
    print("=" * 76)
    print("  用途: 回答『L<=16 那条天花板限制了谁』—— 不是提高拟合质量。")
    print("  读数与 spiral_loop 逐字同源 (同估计量, 同偶数处理), 故同 L 必须同数。")

    rows = []
    for L0_ in L_list:
        L = int(L0_) - (int(L0_) % 2)      # 半链切分要求偶数, 与 spiral_loop 同一处理
        t0 = time.perf_counter()
        E0, gs = exact_ground_state(L, J, h)
        half = L // 2
        s = np.linalg.svd(gs.reshape(2 ** half, 2 ** (L - half)),
                          compute_uv=False)
        p = s ** 2
        p = p / p.sum()
        p_nz = p[p > 1e-15]
        c_fit = fit_central_charge(gs, L)
        _, C, _ = boundary_correlation_graph(gs, L)
        xi = float(boundary_correlation_length(C, L))
        gap = float(p_nz[1] / p_nz[0]) if len(p_nz) > 1 else float('nan')
        ejw = jw_ground_energy(L, J, h)
        dev = abs(E0 - ejw)
        # L 是 2 的幂 <=> L & (L-1) == 0; 这决定 MERA 那条腿能不能碰
        mera_ok = L > 0 and (L & (L - 1)) == 0
        rows.append({'L': L, 'E0_per_L': float(E0 / L), 'c': float(c_fit['c']),
                     'c_rms': float(c_fit['rms']), 'xi': xi,
                     'xi_over_L': float(xi / L), 'gap': gap,
                     'entropy_half': float(-np.sum(p_nz * np.log(p_nz))),
                     'jw_dev': float(dev), 'mera_ok': bool(mera_ok),
                     'cost': float(time.perf_counter() - t0)})
        print(f"    L={L:>3d}: E0/L={E0 / L:>10.6f}  c={c_fit['c']:.6f}  "
              f"xi={xi:>7.3f}  xi/L={xi / L:.4f}  gap={gap:.6f}  "
              f"S(L/2)={rows[-1]['entropy_half']:.4f}  "
              f"|E0-E_jw|={dev:.1e}  "
              f"MERA={'可用' if mera_ok else '不可用(非2的幂)'}  "
              f"({rows[-1]['cost']:.2f}s)")

    # ---- 三条独立的读数, 分开报, 不合并成一个"收敛" ----
    c_arr = [r['c'] for r in rows]
    e_arr = [r['E0_per_L'] for r in rows]
    E0_INF = -4.0 / np.pi                  # 临界 TFIM 的 E0/L 热力学极限 (解析)
    c_mono = all(c_arr[i] > c_arr[i + 1] for i in range(len(c_arr) - 1))
    e_mono = all(e_arr[i] < e_arr[i + 1] for i in range(len(e_arr) - 1))
    jw_worst = max(r['jw_dev'] for r in rows)
    mera_L = [r['L'] for r in rows if r['mera_ok']]
    # 下面所有"末两点/首两点/维度"都从 rows 现取, 不写死 16/18 ——
    # 本函数允许传任意 L_list (冒烟就只跑 3 个点), 写死会印出对不上的数。
    L_lo, L_hi = rows[0]['L'], rows[-1]['L']
    d_tail = abs(c_arr[-1] - c_arr[-2])            # 末两点 c 的变化
    d_head = abs(c_arr[0] - c_arr[1])              # 首两点 c 的变化
    d_half = abs(c_arr[-1] - 0.5)                  # 末点离 1/2 的距离
    arm_now = float(np.log(L_hi / L_lo) / np.log(2.0))

    print(f"\n  三条读数分开说:")
    print(f"    c(L):      " + " -> ".join(f"{v:.6f}" for v in c_arr)
          + f"\n               单调下降趋向 1/2 = {c_mono}; "
            f"末两点(L={rows[-2]['L']}->{L_hi})差 {d_tail:.6f}, "
            f"而末点离 1/2 还差 {d_half:.6f} —— "
            f"末两点差是离极限距离的 1/{d_half / d_tail:.1f}")
    print(f"    E0/L:      " + " -> ".join(f"{v:.6f}" for v in e_arr)
          + f"\n               单调上升趋向 -4/pi={E0_INF:.6f} = {e_mono}; "
            f"末点偏离 {abs(e_arr[-1] - E0_INF):.2e}")
    print(f"    JW 核对:   逐 L 最大偏差 {jw_worst:.1e} "
          f"({'全部通过' if jw_worst < 1e-10 else '**有超差**'}) —— "
          f"这是对 H 构造本身的独立检验")
    print(f"    MERA 可用集: {mera_L if mera_L else '无'} —— 只在这里能碰 MERA")

    print(f"\n  【W5 的结论, 说准】")
    print(f"    (i)  对 c / E0/L / xi / gap 这几个读数, **扩到 {L_hi} 也没撞墙**: "
          f"L={rows[-2]['L']}->{L_hi} 时 c 只动")
    print(f"         {d_tail:.6f}, 小于 L={L_lo}->{rows[1]['L']} 的 {d_head:.6f} "
          f"(步长比 {d_head / d_tail:.1f}x), 也小于离 1/2 尚余的 {d_half:.6f} "
          f"({d_half / d_tail:.1f}x)。")
    print(f"         漂移在收, 不在散。但两个数都从 rows 现取, 且本函数**不做外推** ——")
    print(f"         只跑了少数几个 L 时, 这个趋势不构成『已到极限』的证据。")
    print(f"    (ii) max_L=16 那类约束来自 **MERA 那条腿要求 L 为 2 的幂** (quimb),")
    print(f"         不来自稀疏对角化的算力 —— 后者在 2^{L_hi}={2 ** L_hi} 维上, "
          f"本函数整行读数 (建 H + eigsh + SVD + 关联图 + JW 核对)")
    print(f"         合计 {rows[-1]['cost']:.2f}s。MERA 可用集实测 {mera_L}, "
          f"不可用 {[r['L'] for r in rows if not r['mera_ok']]}。")
    print(f"    (iii) 所以 max_L 限制的是**闭环能走多远**(要 MERA 交叉校验),")
    print(f"          不是**这几个读数能测多准**。审计表 P2 那条要把方向反过来读:")
    print(f"          xi/L 早就在收敛, 发散的是 L 本身 (尺度律 L_t=round(xi))。")
    print(f"    (iv) 杠杆臂要分两把尺子说: L 的比值 {L_lo}->{L_hi} = "
          f"{L_hi / L_lo:.2f}x, 但拟合的自变量是 ln L ——")
    print(f"          在全 log2 轴上只跨了 {arm_now:.2f} 个 octave "
          f"(8->16 是 1.00 个, 8->18 是 1.17 个)。")
    print(f"          **不足以改善标度拟合**, 价值全在 (i)(ii)。")
    return {'L_list': [r['L'] for r in rows], 'rows': rows,
            'c_mono': bool(c_mono), 'E0_over_L_mono': bool(e_mono),
            'E0_over_L_limit': float(E0_INF), 'jw_worst_dev': float(jw_worst),
            'mera_admissible_L': mera_L,
            'mera_inadmissible_L': [r['L'] for r in rows if not r['mera_ok']]}




# ============================================================================
# v15 · W3b 谱学中心荷 —— 独立族 (低能能谱, 不是纠缠谱)
# ============================================================================
def spectral_central_charge(L_list=(8, 10, 12, 14, 16), J=1.0, h=1.0):
    """
    用**低能能谱**给 c=1/2 建第二把尺子, 答审计表 §7 第 4 条
    ("独立族中心荷估计量未建立", Rényi 族实测均值 0.53986, 对 0.5 偏差 7.97%)。

    为什么这在原理上独立于纠缠谱那条路 (2.7 / 任务 A):
      数据不同 —— 用的是 H 的头三个本征值, 不是 |psi> 的约化密度矩阵;
      函数不同 —— 闭式比值, **无最小二乘拟合、无有限尺寸外推**;
      v 消去   —— 见下, 不需要知道费米速度。

    推导 (临界周期 TF-Ising, 低能端由 c=1/2 的 Ising CFT 描述):
        E0      = e_inf*L - pi*c*v/(6L)        Casimir 项
        E1 - E0 = (2*pi*v/L) * Delta1          R 扇区真空 (即 sigma, Delta1 = 1/8)
        E2 - E0 = (2*pi*v/L) * Delta2          NS 双粒子   (即 epsilon, Delta2 = 1)
    后两式相除消去 v, 得到**与 v 无关**的算符内容指纹
        (E2-E0)/(E1-E0) = Delta2/Delta1 = 8
    Casimir 式与 E1 式联立消去 v, 得到闭式读数
        c = 12 * Delta1 * (e_inf*L - E0) / (E1 - E0)
    代入 Delta1 = 1/8 即 c = 1.5*(e_inf*L - E0)/(E1 - E0)。没有拟合参数。

    诚实边界 (四处, 不要读过头):
      1. **它以算符内容 (Delta1=1/8, Delta2=1) 为前提**, 不是无假设的 c 测量。
         纠缠谱那条路以 Calabrese-Cardy 形式为前提, 本条以算符内容为前提 ——
         两者同等带假设, 只是换了假设。真正独立的是**数据**与**函数形式**,
         不是"不需要任何先验"。
      2. 上述前提本身**由比值 8 独立佐证**: 比值只含 Delta2/Delta1, 不含 c
         也不含 v。实测 L=8..16 为 7.9231..7.9807 且单调趋 8, 说明低能端确实
         是 Ising 算符内容。这是"两条路互证"里唯一站得住的那一半。
      3. e_inf 不写死, 由**同一条自由费米子色散**数值积分得到; 函数内再与解析
         值 -4/pi 比对并报出偏差。它属于本模型已在用的那个解, 不算第三方输入,
         但也不是从 ED 数据反推的 —— 该假设照实记在这里。
      4. E1 究竟是不是 R 扇区真空, **不靠假设**: 把 R 真空与 NS 双粒子的解析
         值都算出来同台比对, 谁对上就记谁。对不上会在下面直接报出来。
      5. 读数**不随 L 单调收敛** (L=8..16 落在 0.49894..0.50034, L=16 反而最差),
         残差是 O(1/L) 有限尺寸修正, 不是数值误差 (用解析值代入结果逐位相同)。
         所以只能说"读数与 0.5 相容到 0.2%", 不能说"外推到 0.5"。
    """
    def _eps(q):
        """单粒子色散; H = sum_q eps(q) (eta^dag eta - 1/2)。"""
        return 2.0 * np.sqrt(J ** 2 + h ** 2 - 2.0 * J * h * np.cos(q))

    # e_inf: 基态能量密度 = -(1/2) * (1/pi) * Int_0^pi eps(q) dq。
    # 数值积分 (矩形法, 被积函数解析), 解析值是 -4/pi, 这里只把它当**校验**。
    qg = np.linspace(0.0, np.pi, 200001)
    e_inf = float(-0.5 * np.sum(_eps(qg)) * (np.pi / (len(qg) - 1.0)) / np.pi)
    e_inf_analytic = -4.0 / np.pi
    e_inf_dev = abs(e_inf - e_inf_analytic)

    DELTA1 = 1.0 / 8.0          # sigma;  c = 12*DELTA1*(...)/... 里的那一个
    rows = []
    for L in L_list:
        H = tfi_periodic_sparse(L, J, h)
        ev = np.sort(eigsh(H, k=3, which='SA', return_eigenvectors=False))
        E0, d1, d2 = float(ev[0]), float(ev[1] - ev[0]), float(ev[2] - ev[0])

        # 解析对照: NS 真空 / R 真空 / NS 双粒子
        qNS = np.pi * (2.0 * np.arange(L) + 1.0) / L
        qR = 2.0 * np.pi * np.arange(L) / L
        E_NS = float(-0.5 * np.sum(_eps(qNS)))
        E_R = float(-0.5 * np.sum(_eps(qR)))
        d1_pred_R = E_R - E_NS            # Delta1 = 1/8 的候选
        d1_pred_NS2 = 2.0 * _eps(np.pi / L)   # Delta2 = 1 的候选
        d2_pred = d1_pred_NS2

        v_meas = L * d1 / (2.0 * np.pi * DELTA1)     # 由 E1 反解出的 v
        c_spec = 12.0 * DELTA1 * (e_inf * L - E0) / d1
        rows.append({
            'L': int(L), 'E0': E0, 'd1': d1, 'd2': d2,
            'ratio': d2 / d1,
            'ratio_pred': d2_pred / d1_pred_R,
            'd1_pred_R': d1_pred_R, 'd1_pred_NS2': d1_pred_NS2,
            'd2_pred': d2_pred,
            # _eps 返回 np.float64, 比较结果会是 np.bool_ (json 不认), 显式转 bool。
            'd1_is_R': bool(abs(d1 - d1_pred_R) < 1e-8),
            'd2_is_NS2': bool(abs(d2 - d2_pred) < 1e-6),
            'v_meas': v_meas, 'c_spec': c_spec,
            'c_err_pct': 100.0 * abs(c_spec - 0.5) / 0.5})

    last = rows[-1]
    out = {'L_list': [int(L) for L in L_list], 'e_inf': e_inf,
           'e_inf_analytic': e_inf_analytic, 'e_inf_dev': e_inf_dev,
           'delta1': DELTA1, 'rows': rows,
           'ratio_L16': last['ratio'], 'c_L16': last['c_spec'],
           'v_L16': last['v_meas'],
           'ratio_monotone': all(rows[i]['ratio'] < rows[i + 1]['ratio']
                                 for i in range(len(rows) - 1)),
           'all_d1_is_R': all(r['d1_is_R'] for r in rows),
           'all_d2_is_NS2': all(r['d2_is_NS2'] for r in rows)}

    c_lo = min(r['c_spec'] for r in rows)
    c_hi = max(r['c_spec'] for r in rows)
    c_devmax = max(r['c_err_pct'] for r in rows)
    out['c_lo'] = c_lo
    out['c_hi'] = c_hi
    out['c_devmax_pct'] = c_devmax

    print(f"[v15·W3b] 谱学中心荷 (能谱族, 独立于纠缠谱)")
    print(f"  e_inf = {e_inf:.10f} (解析 -4/pi = {e_inf_analytic:.10f}, "
          f"偏差 {e_inf_dev:.1e})")
    print(f"  {'L':>3} {'E0':>12} {'E1-E0':>11} {'E2-E0':>11} {'ratio':>8} "
          f"{'解析比':>8} {'v_meas':>8} {'c_spec':>8} {'偏差%':>7}")
    for r in rows:
        print(f"  {r['L']:3d} {r['E0']:12.6f} {r['d1']:11.6f} {r['d2']:11.6f} "
              f"{r['ratio']:8.4f} {r['ratio_pred']:8.4f} {r['v_meas']:8.4f} "
              f"{r['c_spec']:8.5f} {r['c_err_pct']:7.3f}")
    print(f"  判读: E1 逐点等于 R 扇区真空 (Delta1=1/8)? "
          f"{'是' if out['all_d1_is_R'] else '**否**'}; "
          f"E2 逐点等于 NS 双粒子 (Delta2=1)? "
          f"{'是' if out['all_d2_is_NS2'] else '**否**'}")
    print(f"  比值 (E2-E0)/(E1-E0): {rows[0]['ratio']:.4f} (L={rows[0]['L']}) -> "
          f"{last['ratio']:.4f} (L={last['L']}), "
          f"{'单调趋 8' if out['ratio_monotone'] else '**非单调**'}; "
          f"v 无关, 故这是算符内容的独立指纹。")
    print(f"  c = 12*Delta1*(e_inf*L-E0)/(E1-E0): L={rows[0]['L']}..{last['L']} "
          f"读数落在 [{c_lo:.5f}, {c_hi:.5f}], 对 0.5 最大偏差 {c_devmax:.3f}%")
    print(f"    **不随 L 单调收敛**: 两端分别是 {rows[0]['c_spec']:.5f} (L={rows[0]['L']})"
          f" 与 {last['c_spec']:.5f} (L={last['L']}), L=16 反而是最差点。残差是"
          f" O(1/L) 有限尺寸修正,")
    print(f"    所以这条只能说'读数与 0.5 相容到 {c_devmax:.2f}%', "
          f"**不能**说'外推到 0.5'。对照纠缠谱族的 Rényi 扫描: 偏差 7.97%。")
    print(f"  诚实边界: 本条以算符内容 Delta1=1/8 为前提, 不是无假设的 c 测量; "
          f"换的是数据与函数形式, 不是'不需要先验'。")
    return out




# ============================================================================
# 任务 A · 中心荷 c 验证 (MERA 键维扫描 + 精确对照)
# ============================================================================
def central_charge_verification(configs=((8, 2, 600), (8, 4, 600),
                                         (16, 2, 600), (16, 4, 600))):
    """
    双重证据:
      (1) 精确对角化 (有限尺寸基线): 拟合 c, 应约等于 0.5;
      (2) MERA (不同键维 chi): 拟合 c, 应随 chi 增大单调趋近 0.5。
    诚实边界: 有限尺寸 + 有限键维下报告的是"收敛趋势", 不是 c=1/2 的证明。
    """
    print("=" * 76)
    print("  任务 A · 中心荷 c 验证 (quimb MERA 变分 + 精确对照)")
    print("=" * 76)
    out = {'exact': [], 'mera': [], 'energy_control': None}

    exact_cache = {}
    for (L, _, _) in configs:
        if L in exact_cache:
            continue
        E0, gs = exact_ground_state(L, 1.0, 1.0)
        fit = fit_central_charge(gs, L)
        exact_cache[L] = (E0, gs, fit)
        out['exact'].append({'L': L, 'E0_per_L': E0 / L, 'c': float(fit['c']),
                             'rms': fit['rms'], 'S': fit['S'].tolist()})
        print(f"  [精确对照] L={L}: E0/L={E0/L:.6f}, 拟合 c={fit['c']:.4f} "
              f"(rms={fit['rms']:.4f})")

    for (L, chi, steps) in configs:
        E0, gs, _ = exact_cache[L]
        mera, raws = mera_init(L, chi, seed=0)
        iso = mera_isometry_check(mera, L)
        print(f"\n  [MERA] L={L}, chi={chi}: 张量数={mera.num_tensors}, "
              f"酉性误差={iso['unitary_err']:.2e}, "
              f"等距误差={iso['isometry_err']:.2e}")
        psi, hist = mera_fit(mera, raws, gs, L, steps=steps)
        E = float(psi @ (tfi_periodic_sparse(L) @ psi))
        fit = fit_central_charge(psi, L)
        ov = abs(float(gs @ psi))
        out['mera'].append({'L': L, 'chi': chi, 'c': float(fit['c']),
                            'rms': fit['rms'], 'overlap': ov,
                            'E_per_L': E / L, 'E0_per_L': E0 / L,
                            'n_unitary': iso['n_unitary'],
                            'n_isometry': iso['n_isometry'],
                            'unitary_err': iso['unitary_err'],
                            'isometry_err': iso['isometry_err'],
                            'S': fit['S'].tolist(), 'fit': hist})
        print(f"     -> 重叠={ov:.5f}, E/L={E/L:+.5f} (精确{E0/L:+.5f}), "
              f"拟合 c={fit['c']:.4f} (rms={fit['rms']:.4f})")

    # 对照实验: 能量目标 -> 平均场塌缩
    print("\n  [对照实验] 若改用'最小化能量'作变分目标 (L=8, chi=4):")
    mera_e, raws_e = mera_init(8, 4, seed=0)
    psi_e, E_e = mera_energy_only(mera_e, raws_e, 8,
                                  tfi_periodic_sparse(8), steps=400)
    fit_e = fit_central_charge(psi_e, 8)
    E0_8, gs8, _ = exact_cache[8]
    ov_e = abs(float(gs8 @ psi_e))
    out['energy_control'] = {'E_per_L': E_e / 8, 'E0_per_L': E0_8 / 8,
                             'c': float(fit_e['c']), 'overlap': ov_e,
                             'S': fit_e['S'].tolist()}
    print(f"     能量目标: E/L={E_e/8:+.5f} (精确{E0_8/8:+.5f}), "
          f"重叠={ov_e:.5f}, 拟合 c={fit_e['c']:.4f}")
    print("     -> 能量看似接近, 但重叠低、c 完全错误: 塌缩到平均场局部极小")
    print("        (最优乘积态 E/L = -1.25, 与精确值只差 2.5%, 却有 S(n)~0)")
    return out





# ============================================================================
# 主程序
# ============================================================================
# ============================================================================
# v14 · F5 一致性检查束 (数值路径 <-> 解析路径)
# ============================================================================
def consistency_checks(s3a, curv_val, s7, gray=None):
    """
    五条检查, 每条都把同一个量用**两条独立路径**算出来对表。

    为什么必须是这个形式: "某个数很小"本身没有内容 —— 任何实现都能让它很小。
    有内容的是"数值路径与解析路径给出同一个数", 因为解析路径不共享任何被检验
    的代码, 它既错不了也蒙不对。

      L2  基态能量:  稀疏 eigsh        vs  Jordan-Wigner 闭合式
      L3  幂迭代:    迭代残差 ||A x - <x,A x> x|| 是否到阈值 (不是"迭代步数")
      L5  Forman:    环 C_n (n>=4) 的 F 是否恒为 0 (4 - deg(u) - deg(v))
      L6  Gray-Scott: 离散拉普拉斯零和 + 精确离散平衡恒等式
      L7  自指闭环:   非塌缩支的熵比与等权度是否满足上界

    三条被删掉的候选, 记在这里免得以后又被加回来:
      * "L2 的迭代步数与预设值无关" —— eigsh 没有"步数"这个参数。用 maxiter
        翻译之后, 30 以上逐位相同, 零分辨力; 换成 JW 闭合式才真的在做对照。
      * "L3 的 Jordan 峰 k*=(N-1)/|ln lam|" —— 已经是第二层的指标 2.4,
        在这里重算只是把同一个数算第二遍, 不构成第二条路径。
      * "L4 每步优化后的酉性/等距性残差" —— 双重恒真: 进优化器的是 raws
        参数, MERA 张量对象从未被优化触碰 (而且两个检查点都排在优化之前);
        就算搬进循环, QR 参数化也按构造给出等距性。metric_mera_consistency
        已经断言过这件事。

    诚实边界: 这五条测的都是"实现与它声称的解析式一致", 不测"物理结论成立"。
    L6 的两条是**格式性质** —— 保证代码按写下的方程在算, 不保证那个方程描述
    的是生命。L7 的上界在塌缩支恒取等号, 那一支不带信息量。
    """
    # ---- L2: 基态能量 vs Jordan-Wigner 闭合式 ----
    jw_pts = []
    for L, h in ((8, 1.0), (10, 0.5), (12, 1.0), (14, 1.5), (16, 1.0)):
        e_num, _ = exact_ground_state(L, 1.0, h)
        e_jw = jw_ground_energy(L, 1.0, h)
        jw_pts.append({'L': L, 'h': h, 'E_eigsh': e_num, 'E_jw': e_jw,
                       'abs_dev': float(abs(e_num - e_jw))})
    jw_max_dev = float(max(p['abs_dev'] for p in jw_pts))

    # ---- L3: 幂迭代残差 (物理算例 + 四个受控算例, 取最差的一个) ----
    res_all = [float(s['residual']) for s in s3a['gap_sweep']]
    if s3a.get('physical'):
        res_all.append(float(s3a['physical']['residual']))
    l3_max_res = float(max(res_all)) if res_all else float('nan')
    l3_n_case = len(res_all)

    # ---- L5: Forman 环锚点 ----
    cd = curv_val['forman']

    # ---- L6: Gray-Scott 不变量 ----
    gs = gray if gray is not None else gray_scott_invariants()

    # ---- L7: 自指闭环上界 ----
    ratio, eqw = float(s7['entropy_ratio']), float(s7['equal_weight'])
    l7_defined = bool(s7.get('defined'))
    l7_bound_ok = bool(s7.get('bound_ok'))

    checks = {
        'L2': {'points': jw_pts, 'max_abs_dev': jw_max_dev,
               'n_points': len(jw_pts)},
        'L3': {'max_residual': l3_max_res, 'n_case': l3_n_case,
               'residuals': res_all},
        'L5': {'cycle_mean': cd['cycle_mean'],
               'cycle_max_abs': float(cd['cycle_max_abs']),
               'cycle_min_n': int(cd['cycle_min_n'])},
        'L6': gs,
        'L7': {'entropy_ratio': ratio, 'equal_weight': eqw,
               'defined': l7_defined, 'bound_ok': l7_bound_ok,
               'status': s7['status'],
               # v15·W4: 自指深度 rho(J*) + A/B 分列读数。
               # rho 在塌缩支是**恒等式** (x*=0 ⇒ J*=g·W*, 而 sigma_max(W*)≡1
               # 是谱归一的定义), 靠 rho_is_identity 标出来, 免得被当成测量。
               # entropy_ratio/equal_weight 仍是变体 A 的读数 —— 阶段七叙事
               # 的主体是 A, 且这样与 v14 生产路径的数值一致。
               'rho_jac': float(s7['rho_jac']),
               'rho_is_identity': bool(s7['rho_is_identity']),
               'rho_jac_B': float(s7['rho_jac_B']),
               'rho_is_identity_B': bool(s7['rho_is_identity_B']),
               'variant': s7['variant'],
               'ratio_B': float(s7['B']['entropy'] / np.log(s7['B']['N'])),
               'defined_B': bool(s7['B']['defined']),
               'status_B': s7['B']['status']},
    }

    print("\n  ── v14·F5 一致性检查 (数值路径 <-> 解析路径) ──")
    print(f"      L2 基态能量: {len(jw_pts)} 个 (L,h) 点, "
          f"max|eigsh - JW| = {jw_max_dev:.2e}")
    print(f"      L3 幂迭代残差: {l3_n_case} 个算例, max = {l3_max_res:.2e}")
    print(f"      L5 Forman 环 C_n (n>=4): max|F| = {cd['cycle_max_abs']:.2e} "
          f"(解析 0)")
    print(f"      L6 Gray-Scott: 拉普拉斯零和相对残差 max = "
          f"{gs['lap_zero_rel_max']:.2e}; 平衡恒等式 abs 误差 max = "
          f"{gs['balance_abs_err_max']:.2e} (首步 <u+v> 漂移 = "
          f"{gs['drift_first_step']:+.3e})")
    print(f"      L7 自指闭环: 熵比 = {ratio:.3f} <= 1, 等权度 = {eqw:.3f} <= 1"
          f"{'' if l7_defined else ' (塌缩支: 两条均为约定值, 无信息量)'}")
    # v15·W4: 自指深度。rho 与熵比/等权度**不同类** —— 前者是不动点的动力学
    # 稳定性, 后者是权重的几何形状。两条一起才分得清"照见了什么"与"稳不稳"。
    if s7['rho_is_identity']:
        print(f"      L7 自指深度 rho(J*): A = {s7['rho_jac']:.4f} "
              f"[恒等式, 非测量: 塌缩支 x*=0 ⇒ J*=g·W*, sigma_max(W*)≡1]")
    else:
        print(f"      L7 自指深度 rho(J*): A = {s7['rho_jac']:.4f}")
    print(f"                           B = {s7['rho_jac_B']:.4f} "
          f"(B 每步归一 ||x||≡1, 有真不动点 W* = x* x*^T)")
    return checks
