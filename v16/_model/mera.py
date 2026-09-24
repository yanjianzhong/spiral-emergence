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
"""spiral_model_v16._model.mera —— 由 spiral_model_v16.py 拆分。

阶段四·自指画定边界投影出全息 + 第一层的物理迭代 (v13 检查点选择)

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
from _model.core import fit_central_charge




# ============================================================================
# 阶段四 · 自指画定边界投影出全息
# ============================================================================
def mera_init(L, chi, seed=0):
    """
    构造 quimb 二进制 MERA (L 为 2 的幂, 周期边界), 并为每个可训练张量
    分配一个无约束的原始参数矩阵 (后面用 QR 投到 Stiefel 流形上)。
    形状必须从张量自身的 shape 推导 —— 不同层的键维不同 (底层 4x4, 更高层更大)。
    """
    torch.manual_seed(seed)
    mera = qtn.MERA.rand(L, phys_dim=2, max_bond=chi, dtype=float)
    raws = []
    for t in mera.tensors:
        if t.ndim == 4:
            d = t.shape[0]
            raws.append(torch.randn(d * d, d * d, requires_grad=True))
        elif t.ndim == 3:
            a, b, c = t.shape
            raws.append(torch.randn(a * b, c, requires_grad=True))
    return mera, raws




def _qr_isometry(raw, cols=None):
    """
    把无约束矩阵参数化为 (半) 酉张量: M = Q R -> Q, 再做列符号规范。
    这是 Stiefel 流形上的标准参数化, QR 在 torch 中可微, 约束恒满足 ——
    所以 MERA 的酉性/等距性是"张量性质保证", 不是事后凑出来的。
    """
    Q, R = torch.linalg.qr(raw)
    Q = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)
    return Q if cols is None else Q[:, :cols]




def mera_dense(mera, raws, L):
    """
    把 MERA 收缩成 2^L 维稠密态矢量 (全程可微)。
    output_inds 强制 k0 为第 0 轴 -> reshape(-1) 后 k0 是最高有效位,
    与 tfi_periodic_sparse 的位序约定一致。
    注意: 必须按张量属性 (ndim) 遍历并逐个替换, 不能按下标索引 ——
    下标在 mera 与 mera.copy() 之间不保证对应。
    """
    tn = mera.copy()
    k = 0
    for t in tn.tensors:
        if t.ndim == 4:
            t.modify(data=_qr_isometry(raws[k]).reshape(t.shape))
            k += 1
        elif t.ndim == 3:
            t.modify(data=_qr_isometry(raws[k], t.shape[2]).reshape(t.shape))
            k += 1
        else:                                   # 顶层固定的边界张量
            t.modify(data=torch.as_tensor(np.asarray(t.data)))
    out = tn.contract(all, optimize='greedy',
                      output_inds=[f'k{i}' for i in range(L)])
    if hasattr(out, 'data'):                    # quimb 收缩可能返回 Tensor
        out = out.data
    return out.reshape(-1)




def mera_isometry_check(mera, L):
    """
    验证 MERA 的定义性质 —— 这才是"真正的张量网络", 而不是随机 QR 矩阵:
        酉性   U^dag U = I   (解纠缠器)
        等距性 W^dag W = I   (等距)
    """
    eu, ew = [], []
    for t in mera.tensors:
        d = np.asarray(t.data)
        if t.ndim == 4:
            n = t.shape[0]
            U = d.reshape(n * n, n * n)
            eu.append(float(np.abs(U.conj().T @ U - np.eye(n * n)).max()))
        elif t.ndim == 3:
            a, b, c = t.shape
            W = d.reshape(a * b, c)
            ew.append(float(np.abs(W.conj().T @ W - np.eye(c)).max()))
    return {'unitary_err': max(eu) if eu else 0.0,
            'isometry_err': max(ew) if ew else 0.0,
            'n_unitary': len(eu), 'n_isometry': len(ew)}




def mera_fit(mera, raws, target, L, steps=600, lr=0.03, verbose=True):
    """
    变分拟合: 最大化 |<target|psi(mera)>|^2。

    为何不用能量作目标 (实测结论, 见下方对照实验):
      以 <psi|H|psi> 为目标时梯度下降会塌缩到平均场乘积态 —— 临界 Ising
      的最优乘积态就有 E/L = -1.25 (精确 -1.2815), 只差 2.5%, 却只有
      S(n) ~ 0.02 的纠缠。普通梯度法看不出这点差别, 重叠目标才直接奖励
      "像基态", 因此能落到正确的纠缠分支。
    """
    gst = torch.as_tensor(np.asarray(target).reshape(-1))
    opt = torch.optim.Adam(raws, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr / 50)
    hist = []
    t0 = time.time()
    for step in range(steps):
        opt.zero_grad()
        psi = mera_dense(mera, raws, L)
        loss = -((gst @ psi) ** 2) / (psi @ psi)
        loss.backward()
        opt.step()
        sched.step()
        if verbose and (step % 200 == 0 or step == steps - 1):
            with torch.no_grad():
                ov = abs(float(gst @ (psi / torch.norm(psi))))
            hist.append((step, ov))
            print(f"     step {step:4d} |<gs|psi>|={ov:.5f}  ({time.time()-t0:.0f}s)")
    with torch.no_grad():
        psi = mera_dense(mera, raws, L).numpy()
    return psi / np.linalg.norm(psi), hist




# ===========================================================================
# 第一层的物理迭代: 带检查点选择的多种子变分
# ===========================================================================
#
# 为什么 11.45% 不是 chi=4 的表示极限 (诊断证据, 见 _v13_diag_*.py):
#   证据一: 同 chi=4, 换种子/换学习率, eps_c 在 1%~60% 之间跳 —— 谱宽远超 chi 的影响。
#   证据二: chi=4 在充分优化下达标 (见 _v13_diag_chi.py 的 chi 扫描)。
#   证据三: 重叠从 0.98816 涨到 0.99961 (1.2%), eps_c 从 11.45% 掉到 1.09% (40 倍)。
#           eps_c 在重叠 0.98~0.999 区间**极度敏感**, 所以"跑没跑到位"是主导项。
#   证据四: QR 参数化的等距矩阵带 sign(diag(R)) 定号, Stiefel 流形上这个不连续点
#           会让 ADAM 的动量偶尔跨过符号翻转 -> 重叠出现**悬崖**。实测 lr=0.06 同一种子:
#           step 400 给 eps_c=3.13%, step 800 给 29.42%, step 1800 给 59.52%。
#
# 由此两条对策, 都必须显式声明, 不能悄悄用:
#   (1) **检查点选择**: 每 100 步记一次, 取 overlap 最高的那个。
#       选的是**训练目标本身** (overlap), 不是 eps_c —— 偷看 eps_c 选点就是
#       测试集泄漏, 那才会让"达标"变得不值钱。
#   (2) **多轨迹**: 报分布而不是单点。判据卡**最坏轨迹** (见 metric_design_robustness)。
#
# 目标阈值一个字都没改: eps_c < 5%, R_conf < 1e-2。改的只有预算与统计。
V13_STEPS = 1500                    # 迭代步数预算


V13_LR_SEEDS = ((0.03, (0, 1, 2)),  # 默认学习率, 3 个种子 (headline)
                (0.12, (0, 1)))     # 稳健性对照, 2 个种子


V13_EVERY = 100                     # 收敛曲线与检查点的采样间隔


V13_CHI_CTRL = (8, 0)               # 键维对照: (chi, seed) —— 同种子同预算, 只改 chi




def mera_fit_v13(mera, raws, target, L, steps=V13_STEPS, lr=0.03,
                 every=V13_EVERY, c_exact=None):
    """
    朴素拟合**同一个损失、同一个优化器、同一个调度、同一个 lr 默认值**,
    只加两件不改变优化目标的事: 检查点选择 + 沿途记录。

    检查点选择 (按 overlap, 即训练目标): 因为悬崖的存在, 末态不一定是轨迹上的
    最好点。取最好点是把"优化没跑好"这件事从测量里剔掉 —— 注意这只影响**报什么数**,
    不影响任何物理量; 状态本身仍是纯粹的变分 MERA, 没有任何人工构造。

    返回 (best_psi, best_overlap, best_step, curve);
    curve = [(step, eps_c, R_conf, overlap), ...] 每 every 步一个点。
    """
    gst = torch.as_tensor(np.asarray(target).reshape(-1))
    opt = torch.optim.Adam(raws, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr / 50)

    curve, best = [], None
    for step in range(steps + 1):
        if step % every == 0 or step == steps - 1:
            with torch.no_grad():
                psi = mera_dense(mera, raws, L).numpy()
                psi = psi / np.linalg.norm(psi)
            ov = abs(float(gst.numpy() @ psi))
            if best is None or ov > best[1]:
                best = (psi, ov, step)
            if c_exact is not None:
                fm = fit_central_charge(psi, L)
                curve.append((step, abs(fm['c'] - c_exact) / c_exact,
                              fm['rms'] / float(np.mean(fm['S'])), ov))
        if step == steps:
            break
        opt.zero_grad()
        psi = mera_dense(mera, raws, L)
        loss = -((gst @ psi) ** 2) / (psi @ psi)
        loss.backward()
        opt.step()
        sched.step()
    return best[0], float(best[1]), int(best[2]), curve




def v13_tier1_runs(gs, L, chi, c_exact, verbose=True):
    """
    跑完第一层的全部轨迹 (含键维对照), 返回 (记录 dict, 代表态)。

    代表态 = eps_c 处于**中位数**的那条轨迹的检查点态。选它而不是"最好的那条",
    是因为第二层的纠缠谱 KL 是拿 MERA 与精确态比, 用最好的一条会夸大第二层的成绩。
    中位数 = 典型表现, 这是可审计的选择。
    """
    t0 = time.time()
    runs, curves = [], {}
    for lr, seeds in V13_LR_SEEDS:
        for seed in seeds:
            mera, raws = mera_init(L, chi, seed=seed)
            psi, ov, cstep, curve = mera_fit_v13(
                mera, raws, gs, L, steps=V13_STEPS, lr=lr, c_exact=c_exact)
            fm = fit_central_charge(psi, L)
            eps = abs(fm['c'] - c_exact) / c_exact
            rc = fm['rms'] / float(np.mean(fm['S']))
            runs.append({'lr': lr, 'seed': seed, 'eps_c': float(eps),
                         'R_conf': float(rc), 'overlap': ov,
                         'c': float(fm['c']), 'chosen_step': cstep,
                         'psi': psi})
            if lr == V13_LR_SEEDS[0][0]:
                curves[seed] = curve
            if verbose:
                print(f"      [chi={chi} lr={lr} seed={seed}] "
                      f"ov={ov:.6f} @step{cstep:4d}  c={fm['c']:.6f}  "
                      f"eps_c={eps:>7.3%}  R_conf={rc:.3e}  "
                      f"({time.time() - t0:.0f}s)")

    # 收敛曲线取 headline 学习率下各 seed 的**逐点中位数**, 供预算诊断
    steps_axis = sorted({s for c in curves.values() for (s, _e, _r, _o) in c})
    curve_med = []
    for s in steps_axis:
        vals = [t for c in curves.values() for t in c if t[0] == s]
        curve_med.append((s,
                          float(np.median([v[1] for v in vals])),
                          float(np.median([v[2] for v in vals])),
                          float(np.median([v[3] for v in vals]))))

    # 键维对照: 同种子同预算, 只改 chi -> 直接测"提 chi 有没有用"
    cchi, cseed = V13_CHI_CTRL
    mera_c, raws_c = mera_init(L, cchi, seed=cseed)
    psi_c, ov_c, cstep_c, _ = mera_fit_v13(
        mera_c, raws_c, gs, L, steps=V13_STEPS, lr=V13_LR_SEEDS[0][0],
        c_exact=c_exact)
    fm_c = fit_central_charge(psi_c, L)
    eps_c_ctrl = float(abs(fm_c['c'] - c_exact) / c_exact)
    if verbose:
        print(f"      [chi={cchi} 对照 seed={cseed}] ov={ov_c:.6f} @step{cstep_c}  "
              f"eps_c={eps_c_ctrl:>7.3%}  (同种子同预算, 只改 chi)")

    # v15·W12: 键维**向下**对照 —— 面积律下界的「推离」。
    # 既有的 chi_ctrl 是**向上**的, 只回答"提 chi 有没有用"; **向下**才回答
    # "面积律那条下界到底约不约束下游"。chi_down = chi // 2 是 2 的幂里严格小于
    # exp(S) 的那个 (由 chi = 2**ceil(log2(exp(S))) 得 chi/2 < exp(S) <= chi),
    # 即**在界下** —— 框架应当在这里**正确失败**。
    chi_down = chi // 2
    down = None
    if chi_down >= 2:
        mera_d, raws_d = mera_init(L, chi_down, seed=cseed)
        psi_d, ov_d, cstep_d, _ = mera_fit_v13(
            mera_d, raws_d, gs, L, steps=V13_STEPS, lr=V13_LR_SEEDS[0][0],
            c_exact=c_exact)
        fm_d = fit_central_charge(psi_d, L)
        down = {'chi': chi_down, 'seed': cseed,
                'eps_c': float(abs(fm_d['c'] - c_exact) / c_exact),
                'overlap': float(ov_d), 'step': int(cstep_d)}
        if verbose:
            print(f"      [chi={chi_down} 向下对照 seed={cseed}] ov={ov_d:.6f} "
                  f"@step{cstep_d}  eps_c={down['eps_c']:>7.3%}  "
                  f"(面积律下界 exp(S) 之上是 chi={chi}, 本点在**界下**)")
    elif verbose:
        print(f"      [chi={chi_down} 向下对照] **未构造**: chi={chi} 已是最小"
              f"合法键维 (MERA 要求 2 的幂), 界下没有可用格点")

    # Schmidt 逐切割截断参照 (chi=4): 不是严格下界, 见 spiral_metric_v13 的说明
    floor4 = _schmidt_floor(gs, L, chi)
    floor8 = _schmidt_floor(gs, L, cchi)
    if verbose:
        print(f"      [参照] Schmidt 逐切割截断 (非严格界): "
              f"chi={chi} -> eps_c={floor4:.3%}, chi={cchi} -> eps_c={floor8:.3%}")

    eps_all = np.array([r['eps_c'] for r in runs])
    i_med = int(np.argmin(np.abs(eps_all - np.median(eps_all))))
    runs_stripped = [{k: v for k, v in r.items() if k != 'psi'} for r in runs]
    return {'runs': runs_stripped, 'curve': curve_med,
            'curves_by_seed': {int(k): v for k, v in curves.items()},
            'steps': V13_STEPS,
            'lrs': [lr for lr, _s in V13_LR_SEEDS],
            'chi_sweep': {chi: {'eps_c': float(eps_all[i_med]),
                                'R_conf': runs[i_med]['R_conf'],
                                'overlap': runs[i_med]['overlap']},
                          cchi: {'eps_c': eps_c_ctrl,
                                 'R_conf': float(fm_c['rms'] / float(np.mean(fm_c['S']))),
                                 'overlap': ov_c}},
            'floor_chi': floor4, 'floor_chi_ctrl': floor8,
            'chi_ctrl': cchi, 'chi_down': down,
            'chosen_run_index': i_med,
            'elapsed': time.time() - t0}, runs[i_med]['psi']




def _schmidt_floor(psi, L, chi):
    """
    精确基态在每个切割点把 Schmidt 谱截到 chi 项并归一化后的 eps_c。

    **这不是严格下界** (同一个态不可能在所有切割点同时取到该 rank, 且 rank-chi
    态的熵可在 [0, ln chi] 内任意)。它只用来做**定性归因**: 实测 MERA 的 eps_c
    明显高于这条线 -> 瓶颈在优化; 贴着这条线 -> 瓶颈在表示能力。
    """
    psi = np.asarray(psi, dtype=float).reshape(-1)
    psi = psi / np.linalg.norm(psi)
    y = []
    for n in range(1, L // 2 + 1):
        m = psi.reshape(2 ** n, 2 ** (L - n))
        s = np.linalg.svd(m, compute_uv=False)[:chi]
        p = s ** 2
        p = p[p > 1e-15]
        p = p / p.sum()
        y.append(float(-np.sum(p * np.log(p))))
    y = np.array(y)
    ns = np.arange(1, L // 2 + 1)
    x = np.log((L / np.pi) * np.sin(np.pi * ns / L))
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    c_ex = fit_central_charge(psi, L)['c']
    return float(abs(3.0 * coef[0] - c_ex) / abs(c_ex))




def mera_energy_only(mera, raws, L, H_sparse, steps=400, lr=0.03):
    """对照实验: 用"最小化能量"作目标, 展示会塌缩到平均场极小。"""
    Hd = torch.as_tensor(H_sparse.toarray())
    opt = torch.optim.Adam(raws, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr / 50)
    for _ in range(steps):
        opt.zero_grad()
        psi = mera_dense(mera, raws, L)
        loss = (psi @ (Hd @ psi)) / (psi @ psi)
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        psi = mera_dense(mera, raws, L).numpy()
    psi = psi / np.linalg.norm(psi)
    return psi, float(psi @ (H_sparse @ psi))




def mera_causal_cone(mera, L):
    """
    MERA 的全息结构证据: 区域 A = {0..n-1} 的因果锥张量数 |cone(n)|。
    quimb 已在每个张量上打好 I{j} 标签 (j 属于其因果锥), 所以这里是直接
    数张量, 不是模拟出来的。|cone(n)| 应与 n 成体积律 (线性), 而 S(n)
    只随 ln n 增长 —— 体是体积律, 纠缠是面积律, 这正是全息的定义性质。
    返回 (n 列表, cone 列表, 线性拟合 R^2, 对数拟合 R^2), 让数据自己说话。
    诚实边界: 这是对全息结构的直接张量计数, 不是 Ryu-Takayanagi 面积计算。
    """
    ns = np.arange(1, L // 2 + 1)
    cone = np.array([int(mera.select_any([f'I{j}' for j in range(n)]).num_tensors)
                     for n in ns], dtype=float)
    r2 = {}
    for name, x in (('linear', ns.astype(float)), ('log', np.log(ns))):
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, cone, rcond=None)
        r2[name] = float(1.0 - (cone - A @ coef).var() / cone.var())
    return ns, cone, r2




def build_mera_graph(mera):
    """
    MERA 的张量网络图: 节点 = 张量 (quimb 的整数 tid), 边 = 共享指标。
    这就是离散的体几何 —— 阶段五的曲率就算在它上面。
    (quimb 的 Tensor 没有 .tid 属性, 但 ind_map 的值就是整数 tid。)
    """
    G = nx.Graph()
    G.add_nodes_from(range(mera.num_tensors))
    for _, tids in mera.ind_map.items():
        if len(tids) == 2:
            a, b = sorted(int(x) for x in tids)
            G.add_edge(a, b)
    return G




def mera_bulk_invariance(L=32, seeds=(0, 1, 7), chis=(2, 4, 8)):
    """
    L5 的体几何图 G 依赖**什么**? —— 实测, 不靠读代码推断。

    问三件事, 每一件都可能是**未声明的耦合**:
      * **态**: 换 seed 重建, 图变不变? (v15·W10 问的)
      * **键维**: 换 chi 重建, 图变不变? (v15·B1 问的)
        chi 只定指标维度, 理论上是接线之外的自由度 —— 但 L4 派生的 chi_dev 与
        L5 实际用的 chi 若不同, 这一点**必须实测**, 否则就是一个没被声明的耦合。
      * **布局**: |V| 是否 = 2L-2 (二进制 MERA 的解析值)?

    返回 dict 供指标侧与 json 使用。键 'same' 是 'seed_invariant' 的历史名,
    保留是为了不让既有消费者 (spiral_metric_v16 F6d) 改签名。

    **必须存还全局 torch RNG 状态** —— mera_init 内部调 torch.manual_seed,
    这些额外构造会把状态留在最后一个 seed 上, 污染下游。
    """
    _rng_bak = torch.get_rng_state()
    try:
        gs = [build_mera_graph(mera_init(L, chi, seed=s)[0])
              for chi, s in [(2, s) for s in seeds] + [(c, 0) for c in chis]]
    finally:
        torch.set_rng_state(_rng_bak)
    es = [frozenset(tuple(sorted(e)) for e in g.edges()) for g in gs]
    n_seed = len(seeds)
    seed_eq = len(set(es[:n_seed])) == 1        # 固定 chi=2, 只变 seed
    chi_eq = len(set(es[n_seed:])) == 1         # 固定 seed=0, 只变 chi
    V = gs[0].number_of_nodes()
    return {'L': int(L),
            'seeds': [int(s) for s in seeds], 'same': bool(seed_eq),
            'seed_invariant': bool(seed_eq),
            'chis': [int(c) for c in chis], 'chi_same': bool(chi_eq),
            'chi_invariant': bool(chi_eq),
            'n_edges': int(gs[0].number_of_edges()), 'n_nodes': int(V),
            'nodes_match_binary_mera': bool(V == 2 * L - 2),
            'expected_nodes_2L_minus_2': int(2 * L - 2)}
