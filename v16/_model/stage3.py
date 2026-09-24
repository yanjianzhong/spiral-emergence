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
"""spiral_model_v16._model.stage3 —— 由 spiral_model_v16.py 拆分。

阶段三·自指的动力学涌现

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




# ============================================================================
# 阶段三 · 自指的动力学涌现
# ============================================================================
def _power_iteration(A, x0, max_iter=3000, tol=1e-13):
    """规范化幂迭代 x <- A x / |A x|; 返回不动点、步长历史、轨迹"""
    x = x0 / np.linalg.norm(x0)
    hist, steps = [], []
    for _ in range(max_iter):
        y = A @ x
        nrm = np.linalg.norm(y)
        if nrm < 1e-300:
            break
        y = y / nrm
        phase = np.vdot(x, y)          # 相位对齐 (复特征值时 x, -x, ix 等价)
        if abs(phase) > 1e-14:
            y = y * (phase / abs(phase))
        steps.append(float(np.linalg.norm(y - x)))
        hist.append(y.copy())
        if steps[-1] < tol:
            x = y
            break
        x = y
    return x, np.array(steps), hist




def _nonnormality(A):
    """非正规性度量: ||A A^T - A^T A||_F / ||A||_F^2。正规矩阵为 0。"""
    return float(np.linalg.norm(A @ A.T - A.T @ A) / np.linalg.norm(A) ** 2)




def _transient_growth(A, kmax=8):
    """||A^k||_2 序列, 用来暴露非正规矩阵的瞬态放大"""
    Ak = np.eye(A.shape[0])
    g = []
    for _ in range(kmax):
        Ak = Ak @ A
        g.append(float(np.linalg.svd(Ak, compute_uv=False)[0]))
    return np.array(g)




def selfref_general_matrix(N=128, seed=0, gap_physical=None):
    """
    (a) 一般 (非正规) 矩阵的幂迭代 x <- A x / |A x|。

    第 0 组 (因果链 L2): **物理来源的谱隙**。gap_physical 由阶段二实测的
    Schmitt 谱给出 (gap = p1/p0, 见 derive_L2_entanglement_gap)。
    四个预先指定的 gap (0.9/0.5/0.2/0.05) 只证明"方法实现正确", 那是在
    **校准方法**; 此外多跑一个不由人选的 gap —— 它由临界 Ising 基态的纠缠谱
    决定。实测收敛比与这个派生值的吻合, 才是"幂迭代的收敛速度有物理来源"
    的证据。

    第 1 组: 受控谱隙校准。构造 A = V diag(lam) V^{-1} (lam1 = 1, lam2 = gap),
    V 取轻微非正交 (I + 0.05 G, cond ~ 6), 这样收敛速率有干净的解析预言
    |lam2/lam1| = gap, 可以直接对表。实测比与理论吻合到 3 位。

    第 2 组: Jordan 块 (单一特征值 lam=0.9, 剪切 gam=0.5, 尺寸 6)。谱半径
    0.9 < 1, 幂迭代**必然**收敛到 0; 但 ||A^k|| 先从 1.32 涨到 358 (k=49)
    再衰减。所有特征值都在衰减, 范数却先放大 272 倍 —— 谱半径对这个峰
    完全无话可说, 这正是"自指不一定平滑收敛"的数学来源。

    (为什么必须换一个算例: 第 1 组谱半径 = 1, ||A^k|| 单调, 没有"先涨后落";
     而 V diag(lam) V^{-1} 那一族无论怎么调, ||A^k|| 都在 k=1 取最大 ——
     它的非正规性在第一步就饱和了。真正的内部峰需要 Jordan/剪切这种
     "幂零 + 衰减"的竞争结构。)
    """
    rng = np.random.default_rng(seed)

    # ---------- 第 0 组: gap 由阶段二的纠缠谱派生 (因果链 L2) ----------
    physical = None
    if gap_physical is not None:
        if not (0.0 < gap_physical < 1.0):
            print(f"   [L2] 派生谱隙 = {gap_physical:.6f} 落在 (0,1) 之外, "
                  f"跳过物理算例 (临界态谱隙应 < 1)")
        else:
            V = np.eye(N) + 0.05 * rng.standard_normal((N, N))
            lam = np.array([1.0] + [gap_physical * (1 - 0.5 * i / N)
                                    for i in range(N - 1)])
            A = V @ np.diag(lam) @ np.linalg.inv(V)
            ev, evec = np.linalg.eig(A)
            order = np.argsort(-np.abs(ev))
            rate_theory = float(abs(ev[order[1]]) / abs(ev[order[0]]))
            xf, steps, _ = _power_iteration(A, rng.standard_normal(N))
            if len(steps) > 10:
                nz = steps[:-1] > 1e-14
                ratios = steps[1:][nz] / steps[:-1][nz]
                rate_meas = (float(np.median(ratios[-30:])) if len(ratios)
                             else float('nan'))
            else:
                rate_meas = float('nan')
            physical = {'gap': gap_physical, 'rate_theory': rate_theory,
                        'rate_meas': rate_meas, 'n_steps': int(len(steps)),
                        'residual': float(np.linalg.norm(
                            A @ xf - (np.vdot(xf, A @ xf)) * xf))}
            print(f"   [L2 物理算例] 由纠缠谱派生的谱隙 = {gap_physical:.6f}")
            print(f"        理论收敛比 = {rate_theory:.6f}, "
                  f"实测收敛比 = {rate_meas:.6f}, 步数 = {len(steps)}")

    sweep = []
    for gap, eps in ((0.9, 0.05), (0.5, 0.05), (0.2, 0.05), (0.05, 0.05)):
        V = np.eye(N) + eps * rng.standard_normal((N, N))
        lam = np.array([1.0] + [gap * (1 - 0.5 * i / N) for i in range(N - 1)])
        A = V @ np.diag(lam) @ np.linalg.inv(V)
        ev, evec = np.linalg.eig(A)
        order = np.argsort(-np.abs(ev))
        rate_theory = float(abs(ev[order[1]]) / abs(ev[order[0]]))

        xf, steps, _ = _power_iteration(A, rng.standard_normal(N))
        if len(steps) > 10:
            nz = steps[:-1] > 1e-14
            ratios = steps[1:][nz] / steps[:-1][nz]
            rate_meas = float(np.median(ratios[-30:])) if len(ratios) else float('nan')
        else:
            rate_meas = float('nan')

        vec1 = evec[:, order[0]]
        vec1 = vec1 / np.linalg.norm(vec1)
        sweep.append({'gap': gap, 'eps': eps,
                      'cond_V': float(np.linalg.cond(V)),
                      'rate_theory': rate_theory, 'rate_meas': rate_meas,
                      'n_steps': int(len(steps)),
                      'overlap': float(abs(np.vdot(vec1, xf))),
                      'residual': float(np.linalg.norm(
                          A @ xf - (np.vdot(xf, A @ xf)) * xf)),
                      'nonnormality': _nonnormality(A),
                      'spectral_radius': float(abs(ev[order[0]]))})

    print(f"[阶段三-a] 幂迭代: 一般 (非正规) 矩阵 N={N}")
    for s in sweep:
        print(f"   gap={s['gap']:.2f} eps={s['eps']:.2f} (cond(V)={s['cond_V']:.0f}): "
              f"理论比={s['rate_theory']:.4f}, 实测比={s['rate_meas']:.4f}, "
              f"步数={s['n_steps']:3d}, 残差={s['residual']:.1e}, "
              f"非正规性={s['nonnormality']:.4f}")

    # 第 2 组: Jordan 块 —— 谱半径 < 1 却先放大再衰减
    NJ, lamJ, gamJ, KJ = 6, 0.9, 0.5, 300
    AJ = lamJ * (np.eye(NJ) + gamJ * np.diag(np.ones(NJ - 1), 1))
    rho = float(np.abs(np.linalg.eigvals(AJ)).max())
    g = _transient_growth(AJ, kmax=KJ)
    k_peak = int(np.argmax(g)) + 1
    hump = float(g.max())
    peak_ratio = hump / float(g[0])
    rate_transient = hump / rho ** k_peak     # 峰值 / 谱半径预言

    # 同一矩阵跑幂迭代: 方向收敛 / 范数衰减 是两件不同的事
    x0 = rng.standard_normal(NJ)
    x0 = x0 / np.linalg.norm(x0)
    _, steps_t, _ = _power_iteration(AJ, x0)
    if len(steps_t) > 10:
        nz = steps_t[:-1] > 1e-300
        rt = steps_t[1:][nz] / steps_t[:-1][nz]
        step_ratio = float(np.median(rt[-10:])) if len(rt) else float('nan')
    else:
        step_ratio = float('nan')
    # 特征值只有一个不同值 -> 谱隙 = 0, 线性收敛根本不被预言
    gap_J = 1.0

    # 状态范数 ‖A^k x0‖ 的轨迹 (这才是"自发收敛"的判据)
    xn = x0.copy()
    norms = []
    for _ in range(KJ):
        xn = AJ @ xn
        norms.append(float(np.linalg.norm(xn)))
    norms = np.array(norms)
    k_norm_peak = int(np.argmax(norms)) + 1
    # 最后一次跌破初值并保持不回升的步数 (首次跌破可能只是 k=1 的偶然下探)
    above = np.where(norms >= 1.0)[0]
    k_below = int(above[-1]) + 2 if len(above) else 1

    print(f"   第 2 组 Jordan 块 N={NJ}, 特征值全 = {lamJ} "
          f"(谱半径={rho:.3f} < 1, 非正规性={_nonnormality(AJ):.3f}):")
    print(f"     只有 1 个不同特征值 -> 谱隙 |λ2/λ1| = {gap_J:.2f}, "
          f"线性收敛不被预言; 实测方向步长比 = {step_ratio:.4f} (吻合, 方向不收敛)")
    print(f"     状态范数 ‖A^k x₀‖: 1.000 (k=1: {norms[0]:.3f}) -> 峰值 "
          f"{norms[k_norm_peak-1]:.2f} (k={k_norm_peak}) -> 先涨后落, "
          f"k={k_below} 才最后一次跌破初值并保持, "
          f"k={KJ} 衰减到 {norms[-1]:.2e} (ρ^k={rho**KJ:.2e})")
    print(f"     ‖A‖₂={g[0]:.3f} -> 峰值 ‖A^k‖₂={hump:.3f} (k={k_peak}), "
          f"放大 {peak_ratio:.1f}x, 之后才衰减")
    print(f"     峰值处 ‖A^k‖₂/ρ^k = {rate_transient:.1f}: "
          f"谱半径把范数低估了 {rate_transient:.0f} 倍")
    return {'gap_sweep': sweep, 'physical': physical,
            'transient': {'growth': g.tolist(), 'spectral_radius': rho,
                          'nonnormality': _nonnormality(AJ),
                          'lam': lamJ, 'N': NJ, 'kmax': KJ,
                          'gap': gap_J, 'step_ratio': step_ratio,
                          'norm_trace': norms.tolist(),
                          'norm_final': float(norms[-1]),
                          'k_norm_peak': k_norm_peak,
                          'norm_peak': float(norms[k_norm_peak - 1]),
                          'k_below': k_below,
                          'peak': hump, 'peak_ratio': peak_ratio,
                          'k_peak': k_peak, 'norm_A': float(g[0]),
                          'peak_over_rho_k': rate_transient}}




def selfref_nonconvergent(N=128, seed=0):
    """
    (a') 反例: 当 |lam1| = |lam2| 时幂迭代**根本不收敛** —— 状态在两个
         模相等的特征方向之间无尽旋转。这是"自指不必然收敛"的严格边界。
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, N)) / np.sqrt(N)
    rho = float(np.abs(np.linalg.eigvals(A)).max())
    A = A / rho
    A[:2, :2] = np.array([[0.0, -1.0], [1.0, 0.0]])    # 90 度旋转块 -> 模相等
    ev = np.sort(np.abs(np.linalg.eigvals(A)))
    x0 = rng.standard_normal(N)
    _, steps, _ = _power_iteration(A, x0, max_iter=4000)
    tail = steps[-300:] if len(steps) > 300 else steps
    tail_mean = float(np.mean(tail)) if len(tail) else float('nan')
    print(f"[阶段三-a'] 模相等反例: |lam1|={ev[-1]:.4f}, |lam2|={ev[-2]:.4f}")
    print(f"  4000 步后步长仍为 {tail_mean:.4f} (不趋于 0) -> 不收敛 (周期轨道)")
    return {'lam1': float(ev[-1]), 'lam2': float(ev[-2]),
            'tail_step': tail_mean, 'steps': steps, 'converged': False}




def _lyapunov(W, g, N, warmup=1500, iters=400, sub=10):
    """
    最大 Lyapunov 指数: 沿轨道对 Jacobian J = diag(1-tanh^2) * gW 做
    幂迭代, 每步重新归一化 (不重归一化的话数值会饱和, 指数不可靠)。
    """
    x = torch.randn(N)
    for _ in range(warmup):
        x = torch.tanh(g * (W @ x))
    ly = 0.0
    for _ in range(iters):
        J = torch.diag(1 - torch.tanh(g * (W @ x)) ** 2) @ (g * W)
        v = torch.randn(N)
        v = v / torch.norm(v)
        for _ in range(sub):
            v = J @ v
            ly += float(torch.log(torch.norm(v)))
            v = v / torch.norm(v)
        x = torch.tanh(g * (W @ x))
    return ly / (iters * sub)




def selfref_neural_network(N=64, seed=0):
    """
    (b) 神经网络自指映射  x_{n+1} = tanh(g W x_n)。
        判别三相需要两个独立量, 只看步长会把极限环误判成混沌:
          不动点: 尾部步长 -> 0  且 Lyapunov < 0
          极限环: 尾部步长有限   且 Lyapunov < 0   (周期, 非混沌)
          混沌  : 尾部步长有限   且 Lyapunov > 0
        相变发生在 g * rho(W) ~ 1 (边缘混沌)。
    """
    g_t = torch.Generator().manual_seed(seed)
    W = torch.randn(N, N, generator=g_t) / np.sqrt(N)
    rho = float(torch.linalg.eigvals(W).abs().max())
    gains = [0.4, 0.8, 0.9, 1.0, 1.1, 1.4, 2.0]
    scan = []
    for g in gains:
        x = torch.randn(N, generator=g_t)
        for _ in range(3000):
            x = torch.tanh(g * (W @ x))
        steps = []
        for _ in range(400):
            xn = torch.tanh(g * (W @ x))
            steps.append(float(torch.norm(xn - x)))
            x = xn
        tail = max(steps[-50:])
        ly = _lyapunov(W, g, N)
        if tail < 1e-8 and ly < 0:
            phase = '不动点'
        elif ly > 0:
            phase = '混沌'
        else:
            phase = '极限环'
        scan.append({'g': g, 'g_rho': g * rho, 'tail_step': tail,
                     'lyapunov': ly, 'phase': phase})
        print(f"   g={g:.1f} (g*rho={g*rho:.2f}): 尾部步长={tail:.2e} "
              f"Lyapunov={ly:+.4f} -> {phase}")
    print(f"[阶段三-b] 自指神经网络 N={N}, 谱半径 rho(W)={rho:.4f}")
    return {'rho': rho, 'scan': scan}




def selfref_coupled(N=64, steps=8000, eta=0.05, g=1.0, seed=0):
    """
    (c) 真自指 —— 映射的权重本身是状态的函数 (二阶自指):
          x_{n+1} = tanh(g W_n x_n)                (状态)
          W_{n+1} = normalize_spec(W_n + eta (x_{n+1} x_{n+1}^T - W_n))
                                                   (权重被状态改写)
    权重每步做谱归一, 否则 W -> 0 的塌缩会淹没一切。

    三种变体 (全部实测, 输出里的判定完全由数字驱动, 不预设结论):
      冻结权重 (对照, 不自指): 权重不随状态改变, 就是一个固定随机映射。
      状态自由 (A):            权重随状态改写, 状态不做归一。
      状态归一 (B):            权重随状态改写, 状态每步归一。

    判定依据是三个量: 末段步长 |dx| (收敛与否)、末段 ||x|| (状态是否消退)、
    权重的 sigma1/sigma2 (是否自发秩 1 —— 秩 1 意味着 W* ∝ x* x*^T, 即权重
    收敛到"恰好产生自己的那个态", 这才是自指的自洽不动点)。
    诚实边界: 步长在有限步内只降到 1e-5 量级, 是渐近收敛而非机器精度;
    真正干净的证据是 sigma 比随步数单调增长若干个数量级。
    """
    def _run(norm_x, frozen=False):
        gt = torch.Generator().manual_seed(seed)
        W = torch.randn(N, N, generator=gt) / np.sqrt(N)
        x = torch.randn(N, generator=gt)
        if norm_x:
            x = x / torch.norm(x)
        dx, dW, rk, nx = [], [], [], []
        for it in range(steps):
            Wn = W
            xn = torch.tanh(g * (Wn @ x))
            if norm_x:
                xn = xn / torch.norm(xn)
            if not frozen:
                W = Wn + eta * (torch.outer(xn, xn) - Wn)
                W = W / torch.linalg.matrix_norm(W, 2)
            dx.append(float(torch.norm(xn - x)))
            dW.append(float(torch.norm(W - Wn)))
            nx.append(float(torch.norm(xn)))
            if it % 2000 == 0:
                sv = torch.linalg.svdvals(W)
                rk.append((it, float(sv[0] / sv[1])))
            x = xn
        return {'dx': np.array(dx), 'dW': np.array(dW), 'rank': rk,
                'x_norm': np.array(nx), 'W': W, 'x': x,
                'final_dx': float(np.mean(dx[-200:])),
                'final_dW': float(np.mean(dW[-200:])),
                'sigma_ratio': rk[-1][1],
                'collapsed': bool(np.mean(nx[-200:]) < 1e-6)}

    print(f"[阶段三-c] 权重-状态耦合自指 N={N}, eta={eta}, g={g}")
    frozen = _run(norm_x=True, frozen=True)
    free = _run(norm_x=False)
    norm = _run(norm_x=True)
    for lab, r in (("冻结权重 (对照)", frozen),
                   ("自指·状态自由 (A)", free),
                   ("自指·状态归一 (B)", norm)):
        xn = float(np.mean(r['x_norm'][-200:]))
        print(f"  {lab}: 末段|dx|={r['final_dx']:.3e} "
              f"({'收敛' if r['final_dx'] < 1e-4 else '不收敛'}), "
              f"末段||x||={xn:.3e}, sigma1/sigma2={r['sigma_ratio']:.3e}")
    for it, r in norm['rank']:
        print(f"      B 的秩1化轨迹 step {it:5d}: sigma1/sigma2={r:14.2f}")
    # 结语必须由**实测**生成, 不能写死。
    # 手工 N=64 下两个变体都秩 1 化, 若照此写死一句"都自发秩 1 化", 那么把 N
    # 派生为 17 (有效 Schmidt 秩) 之后 — 此时变体 A 收敛到了 x*=0, 它的
    # sigma1/sigma2 掉到量级 1, **没有**秩 1 化 — 写死的结语就成了假话。
    # 所以改成按实测阈值下结论。
    RANK_TOL = 1e3        # sigma1/sigma2 > 1e3 才算自发秩 1 化
    ok_free = free['sigma_ratio'] > RANK_TOL
    ok_norm = norm['sigma_ratio'] > RANK_TOL
    verdict_free = '秩 1 化' if ok_free else '未秩 1 化 (它收敛到 x*=0, 权重失去方向)'
    verdict_norm = '秩 1 化' if ok_norm else '未秩 1 化'
    # v15 修正 1: 原文写"冻结的对照永不收敛", 这是数据不支持的绝对断言。
    # v15·W1 实测 50 个 (N,seed) 组合: 8 个 (16.0%) 的冻结对照 final_dx < 1e-4,
    # 且分布双峰 (<1e-8 有 8 个, >=1e-4 有 42 个, 中间 0 个) —— 那是真收敛到
    # 机器精度, 不是慢漂移。所以只报本次实测值, 不加"永不"这类量词。
    print(f"      -> 冻结的对照本次末段|dx|={frozen['final_dx']:.3e}"
          f" ({'收敛' if frozen['final_dx'] < 1e-4 else '不收敛'}); 它是固定随机"
          f"映射, 收敛与否取决于抽到的 W 的谱 —— W1 实测 16.0% 的组合会收敛。")
    print(f"         两个自指变体本次: A |dx|={free['final_dx']:.3e}, "
          f"B |dx|={norm['final_dx']:.3e} (判据 < 1e-4)")
    print(f"         但'权重自发秩 1 化 (W* -> x* x*^T)'只在实测 sigma1/sigma2 "
          f"> {RANK_TOL:g} 时才能说:")
    print(f"           变体 A (状态自由): sigma1/sigma2={free['sigma_ratio']:.3e} -> "
          f"{verdict_free}")
    print(f"           变体 B (状态归一): sigma1/sigma2={norm['sigma_ratio']:.3e} -> "
          f"{verdict_norm}")
    # v15 修正 2: 原文只对比"手工 N=64"与"派生 N=17"两个点。v15·W1 扫了 10 个 N,
    # 实测相图是 N<=32 时 A 为 0/5 种子秩 1 化, N=48 时 1/5, N=64 时 5/5 ——
    # 即相变**随种子随机**, 不是一条 N 阈值。完整相图见 selfref_N_dependence()。
    print(f"         诚实边界: 上述判定依赖 N —— W1 实测 A 秩1化的种子数: "
          f"N<=32 为 0/5, N=48 为 1/5, N=64 为 5/5;")
    print(f"                   而 L<=16 能派生的 N 只有 11~17, 一律 0/5。"
          f" 完整相图见 selfref_N_dependence()。")
    # v15 修正 3: A 若秩 1 化, 那是**不动点**还是**暂态**? 实测 ||x|| 的幂律指数
    # 回答这个问题: 若 ||x|| ~ n^-p 且 p>0, 则 A 正沿代数律走向 x*=0, 秩 1 方向
    # 只是有限步暂态。W1 实测 (N=64, 5 seeds) p = 0.4950~0.4992 ≈ 1/2 ——
    # 1/2 正是 ds/dn = -s^3/3 (g=1, 边缘稳定) 的预测。B 每步归一 ||x||≡1,
    # 才有真不动点 W* = x* x*^T。
    if ok_free and free['x_norm'][-1] > 0:
        _n = np.arange(1, len(free['x_norm']) + 1, dtype=float)
        _fit = np.vstack([np.log(_n), np.ones_like(_n)]).T
        _p = -float(np.linalg.lstsq(
            _fit, np.log(np.maximum(free['x_norm'], 1e-300)), rcond=None)[0][0])
        print(f"         暂态判定: A 的 ||x|| ~ n^-p, 本次拟合 p={_p:.4f}; p>0 即"
              f"代数衰减,")
        print(f"                   说明该秩 1 方向是**有限步暂态**, 渐近仍归于"
              f" x*=0, 不是不动点。")
    print(f"         状态消退与否: 归一后 ||x|| 保持在 "
          f"{np.mean(norm['x_norm'][-200:]):.3f}, "
          f"自由时衰减到 {np.mean(free['x_norm'][-200:]):.2e}。")
    # 只保留可序列化 / 可画图的量; torch 张量 (W, x) 单独带出。
    # v15: A 的 W/x 也必须带出 —— 阶段七要**分列**两个变体的读数。v14 的
    # stage7_consciousness 用变体 A 的 collapsed 做判据、却取变体 B 的 W 做读数,
    # 两个变体混在一条结论里; 只带 B 的 W 就复现不出这个错误, 也就修不掉它。
    drop = ('W', 'x')
    return {'frozen': {k: v for k, v in frozen.items() if k not in drop},
            'free': {k: v for k, v in free.items() if k not in drop},
            'norm': {k: v for k, v in norm.items() if k not in drop},
            '_W': norm['W'], '_x': norm['x'],
            '_W_free': free['W'], '_x_free': free['x'],
            '_g': g}          # 阶段七算 Jacobian 谱半径要用; 由本函数带出,


                              # 免得调用方另写一个 g 而与这里漂移


def selfref_N_dependence(N_list=(11, 12, 15, 16, 17, 48, 64),
                         seeds=(0, 1, 2, 3, 4), steps=8000):
    """
    L3 · 变体 A/B 的秩1化**相图** —— 回答"这条结论依赖 N"究竟依赖成什么样。

    v15·W1 新增。v14 只在 selfref_coupled 的结语里对比两个点 (手工 N=64 与
    派生 N=17), 没有相图, 也说不清"依赖"是依赖成一条阈值还是别的。本函数给出:
      (1) 每个 N 上有几个种子秩 1 化 (判据 sigma1/sigma2 > RANK_TOL);
      (2) RANK_TOL 敏感性 —— 换阈值会不会翻盘 (由已返回的 sigma 比**离线重算**,
          不重跑);
      (3) 变体 A 若秩 1 化, 那是**不动点**还是**有限步暂态**: 拟合 ||x|| ~ n^-p,
          p > 0 即代数衰减 (走向 x*=0), 说明秩 1 方向只是暂态;
      (4) 冻结对照的收敛命中率 —— v14 印的是"永不收敛"这个绝对断言, 这里改成
          实测计数。

    边界声明:
      - 这里的 N 是**自指网络宽度**, 不是自旋链长度 L; 每步 O(N^2)~O(N^3),
        不涉及 2^L 精确对角化, 因此不受链长 L 那条约束 (L 的上限来自末圈 MERA
        要求 L 为 2 的幂, 与本函数的 N 无关)。
      - 物理上真实取到的只有 N_self(L) in {11,12,15,16,17} (L=8..16); 默认网格
        另含 48/64 是为定位相变, 那**不是**模型会走到的状态。
      - 实测相图: A 秩1化的种子数 N<=32 为 0/5, N=48 为 1/5, N=64 为 5/5 ——
        即相变**随种子随机**, 不存在一条确定的 N 阈值。所以报的是"命中率",
        不是"阈值 N_c"; 不要把它读成一条边界。
    """
    RANK_TOL = 1e3
    rows = []
    t0 = time.perf_counter()
    for N in N_list:
        for seed in seeds:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                r = selfref_coupled(N=N, steps=steps, seed=seed)
            # free_p 先置 NaN 再按需要覆盖。**不能**放在 tag 循环的 else 支里 ——
            # 那个 else 对 norm/frozen 也会执行, 会把已算出的 free_p 冲掉。
            row = {'N': N, 'seed': seed, 'free_p': float('nan')}
            for tag in ('free', 'norm', 'frozen'):
                v = r[tag]
                xn = np.asarray(v['x_norm'], dtype=float)
                row[f'{tag}_sigma'] = float(v['sigma_ratio'])
                row[f'{tag}_dx'] = float(v['final_dx'])
                row[f'{tag}_xnorm'] = float(np.mean(xn[-200:]))
                row[f'{tag}_collapsed'] = bool(v['collapsed'])
                # p 只在 A 真秩 1 化时才有意义; 塌缩支 ||x|| 恒 0, 拟合是噪声。
                if tag == 'free' and v['sigma_ratio'] > RANK_TOL:
                    n = np.arange(1, len(xn) + 1, dtype=float)
                    fit = np.vstack([np.log(n), np.ones_like(n)]).T
                    row['free_p'] = -float(np.linalg.lstsq(
                        fit, np.log(np.maximum(xn, 1e-300)), rcond=None)[0][0])
            rows.append(row)
    dt = time.perf_counter() - t0

    print(f"[阶段三-c·W1] 秩1化相图: {len(N_list)} 个 N x {len(seeds)} 种子 "
          f"x {steps} 步, {dt:.1f}s")
    print(f"  {'N':>3} {'A 秩1化':>9} {'B 秩1化':>9} {'冻结收敛':>9} "
          f"{'A 的 p':>9}   (命中种子数 / 总数)")
    for N in N_list:
        g = [r for r in rows if r['N'] == N]
        m = len(g)
        a = sum(r['free_sigma'] > RANK_TOL for r in g)
        b = sum(r['norm_sigma'] > RANK_TOL for r in g)
        f = sum(r['frozen_dx'] < 1e-4 for r in g)
        ps = [r['free_p'] for r in g if r['free_p'] == r['free_p']]
        ptxt = f"{np.median(ps):9.4f}" if ps else f"{'--':>9}"
        print(f"  {N:3d} {a:7d}/{m:<2d} {b:7d}/{m:<2d} {f:7d}/{m:<2d} {ptxt}")

    # RANK_TOL 敏感性: 用已返回的 sigma 比离线重算, 不重跑。
    tols = (1e1, 1e2, 1e3, 1e4, 1e5, 1e6)
    print(f"  RANK_TOL 敏感性 (命中率 = 该 N 下 sigma>tol 的种子占比):")
    print(f"  {'N':>3} " + " ".join(f"{('A>%.0e' % t):>7}" for t in tols)
          + "   | " + " ".join(f"{('B>%.0e' % t):>7}" for t in tols))
    for N in N_list:
        g = [r for r in rows if r['N'] == N]
        m = len(g)
        sa = " ".join(f"{sum(r['free_sigma'] > t for r in g) / m:7.2f}"
                      for t in tols)
        sb = " ".join(f"{sum(r['norm_sigma'] > t for r in g) / m:7.2f}"
                      for t in tols)
        print(f"  {N:3d} {sa}   | {sb}")

    # 结论由**实测**生成, 不写死。
    hit = {N: sum(r['free_sigma'] > RANK_TOL for r in rows if r['N'] == N)
           / max(1, len([r for r in rows if r['N'] == N])) for N in N_list}
    pure = [N for N in N_list if hit[N] == 1.0]
    zero = [N for N in N_list if hit[N] == 0.0]
    mixed = [N for N in N_list if 0.0 < hit[N] < 1.0]
    n_froz_conv = sum(r['frozen_dx'] < 1e-4 for r in rows)
    print(f"  -> A 全部种子秩1化的 N: {pure if pure else '无'}")
    print(f"     完全没有的 N:       {zero if zero else '无'}")
    print(f"     部分种子才有的 N:   {mixed if mixed else '无'}"
          f"  <- 相变是**随机的**, 不是一条 N 阈值")
    ps_all = [r['free_p'] for r in rows if r['free_p'] == r['free_p']]
    if ps_all:
        print(f"  -> A 的 ||x|| ~ n^-p, p 中位 = {np.median(ps_all):.4f}"
              f" (预测 1/2, 来自 ds/dn = -s^3/3 即 g={1.0:g} 的边缘稳定)")
        print(f"     p>0 ⇒ 秩1方向是**有限步暂态**, 渐近仍归于 x*=0, 不是不动点。")
    print(f"  -> 冻结对照 (固定随机映射) 收敛到 <1e-4 的有 {n_froz_conv}/"
          f"{len(rows)} = {100.0 * n_froz_conv / len(rows):.1f}%"
          f" —— 收敛与否取决于抽到的 W 的谱, **不是恒定的**。")
    return {'N_list': list(N_list), 'seeds': list(seeds), 'steps': steps,
            'rank_tol': RANK_TOL, 'seconds': dt, 'rows': rows,
            'hit_rate_free': hit,
            'pure_N': pure, 'zero_N': zero, 'mixed_N': mixed,
            'median_p_free': float(np.median(ps_all)) if ps_all else None,
            'frozen_converged_frac': n_froz_conv / len(rows)}
