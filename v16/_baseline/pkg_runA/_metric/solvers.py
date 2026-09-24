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
"""spiral_metric_v16._metric.solvers —— 由 spiral_metric_v16.py 拆分。

共用求解器 + 既有指标函数 (第一层 1.3~1.5 / 第二层 2.1~2.4)

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
    _char_scale, record_metric,
)



def entanglement_curve(psi, L):
    """S(n) = -Tr rho_A ln rho_A, A = 前 n 个连续站点 (n = 1..L//2)"""
    psi = np.asarray(psi).reshape(-1)
    psi = psi / np.linalg.norm(psi)
    out = []
    for n in range(1, L // 2 + 1):
        m = psi.reshape(2 ** n, 2 ** (L - n))
        s = np.linalg.svd(m, compute_uv=False)
        p = s ** 2
        p = p[p > 1e-15]
        out.append(float(-np.sum(p * np.log(p))))
    return np.array(out)



def fit_central_charge(psi, L):
    """
    周期 CFT 的 Calabrese-Cardy 公式:
        S(n) = (c/3) ln[ (L/pi) sin(pi n / L) ] + const
    对 [ln(...), 1] 做线性最小二乘, 斜率 -> c = 3 * slope。
    返回的 rms 是"这个公式是否成立"的残差, 不是 c 的误差棒。
    """
    ns = np.arange(1, L // 2 + 1)
    x = np.log((L / np.pi) * np.sin(np.pi * ns / L))
    y = entanglement_curve(psi, L)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    rms = float(np.sqrt(np.mean((y - A @ coef) ** 2)))
    return {'c': 3.0 * coef[0], 'const': coef[1], 'rms': rms,
            'x': x, 'S': y, 'ns': ns}




# ============================================================================
# v14 · 指标层 I/II: 内部自洽性 + 微观物理对标
# ============================================================================
#
# 每个函数算完就 record_metric(...) 落一条账, 并返回自己的明细 dict 供打印/JSON。
# 三条约定:
#   1. 目标都是本项目自设的**工程阈值**, 不是文献标准 —— 报告里如实标注。
#   2. **只报告不追目标**: 没达标时给的是差距诊断, 不是调参重算。
#   3. 参照物分两类: 第 1 层的参照是 v11 的实测基线 (单次运行, 无误差棒);
#      第 2 层的参照是**解析解** (c=1/2, eta=1/4, k*=(N-1)/|ln lam|, 精确基态谱)。


def _schmidt_spectrum(psi, L, tol=1e-12):
    """
    半链二分的**归一化 Schmidt 谱** p_k = s_k^2 / sum(s_k^2)。

    与 entanglement_curve 用同一个二分 (前 n 个连续站点), 区别是这里要整个谱,
    不是它的熵。tol 截断是熵/KL 计算的常规做法 (否则 ln 0 发散); 截断本身是一条
    诚实边界, 所以把被丢掉的分量数一并返回, 让调用方可以如实报告。
    """
    n = L // 2
    M = psi.reshape(2 ** n, 2 ** (L - n))
    s = np.linalg.svd(M, compute_uv=False)
    p_all = s ** 2
    keep = p_all > tol
    p = p_all[keep]
    return p / p.sum(), int((~keep).sum())




def metric_gap_closure(rows):
    """
    1.3 谱隙闭合残差 —— 逼近临界时相对谱隙 1-r 是否按 1/L 收拢?

    定义: r = p1/p0 是约化密度矩阵前两个本征值之比 (= 幂迭代收敛率)。
      临界时 r -> 1 (谱隙闭合)。记 g(L) = 1 - r(L), 拟合 g(L) = a/L + b。
      指标取该拟合的残差 RMS; 并单独检查 a 的符号 (a > 0 才是"在闭合")。
    诚实边界: 只有 3 个 L (8/12/16), 拟一条直线只剩 1 个自由度 ——
      残差 RMS 很小是**弱证据**, 不是强证据。这条写进 note。
    """
    Ls = np.array([r['L'] for r in rows], dtype=float)
    g = np.array([1.0 - r['gap'] for r in rows], dtype=float)
    a, b = np.polyfit(1.0 / Ls, g, 1)
    resid = float(np.sqrt(np.mean((g - (a / Ls + b)) ** 2)))
    closing = bool(a > 0)
    target = 5e-3
    ok = bool(resid < target and closing)
    record_metric(
        1, '谱隙闭合残差', 'RMS of  g(L) = a/L + b,  g = 1 - p1/p0',
        '相对谱隙 g = 1 - p1/p0 随系统尺寸 L 的闭合行为对 1/L 律的偏离。'
        '临界性的标志是 g -> 0; 这里量的是它收拢得规不规律。',
        'spiral_loop() 每圈实测的 gap = p1/p0, 对 L in {8,12,16} 拟合 a/L + b',
        '解析对照: 临界点处谱隙随 1/L 闭合',
        resid, None, f'< {target:.0e} 且 a > 0 (方向正确)', ok,
        note=(f'拟合得 a = {a:.4f} (a > 0 表示确实在闭合), b = {b:.4f}; '
              f'g 的六个实测值 = {np.round(g, 4).tolist()} '
              f'(spiral_loop 跑了 6 圈, 但 L 被 max_L=16 夹住, 末 3 圈与第 3 圈同值)。'
              f'**只有 3 个点拟一条直线, 仅 1 个自由度** —— 残差小是弱证据。'))
    return {'resid': resid, 'a': float(a), 'b': float(b),
            'g': g.tolist(), 'closing': closing, 'passed': ok}




def metric_mera_consistency(iso, r2_cone):
    """
    1.4 MERA 一致性 —— 张量网络自身的约束满足得怎么样?

    定义: 两个子量同时成立才算达标
      (a) 张量一致性 err = max(等距误差 ||W^dag W - I||, 酉性误差 ||U^dag U - I||)
      (b) 因果锥线性度 R^2_linear > R^2_log (体是体积律, 而 S(n) 只是面积律)
    为什么 (a) 要卡到 1e-12: 等距性是 MERA 的**定义性质**, 由 QR 构造保证,
      所以它应当到机器精度。v11 只把它 print 出来 ("6.66e-16") 而没有判定 ——
      这正是一个"有数字无判定"的洞。
    """
    err = float(max(iso['unitary_err'], iso['isometry_err']))
    lin, lg = float(r2_cone['linear']), float(r2_cone['log'])
    ok = bool(err < 1e-12 and lin > 0.95 and lin > lg)
    record_metric(
        1, 'MERA 一致性', 'max(||W^dag W - I||, ||U^dag U - I||)  与  R^2_linear > R^2_log',
        '张量网络自身必须满足的两条约束: (a) 解纠缠器酉、等距张量等距, 且到机器精度;'
        '(b) 因果锥张量数随 n 线性增长 (体是体积律), 优于对数增长。',
        f'mera_isometry_check(): {iso["n_unitary"]} 个酉 + {iso["n_isometry"]} 个等距; '
        f'mera_causal_cone(): 线性拟合 R^2={lin:.4f} vs 对数拟合 R^2={lg:.4f}',
        '解析对照: 等距性是 MERA 的定义性质, 不是拟合结果',
        err, 6.66e-16, '< 1e-12 且 R^2_linear > 0.95 且 线性 > 对数', ok,
        note=('v11 只把这两个误差 print 出来, 没有判定 —— v12 补成可证伪的 bool。'
              '这里能说的是"张量性质成立", **不是**"MERA 拟合得好"; '
              '拟合质量由 1.1 的中心荷误差负责。'))
    return {'err': err, 'r2_linear': lin, 'r2_log': lg, 'passed': ok}




def metric_chain_rmse(s3a, l2_phys=None):
    """
    2.1 因果链拟合 RMSE —— 理论收敛比 与 实测幂迭代率 的均方根偏差。

    定义: 对每组受控谱隙 (gap, eps), 理论预言收敛比 = |lam2/lam1| (由矩阵特征值
      直接算出), 实测收敛比 = 幂迭代末段步长比的中位数。RMSE = sqrt(mean((t-m)^2))。
    参照物是**解析量** |lam2/lam1|, 不是拟合值 —— 这是这条指标之所以叫"对标"的原因。
    """
    pairs = [(float(s['rate_theory']), float(s['rate_meas']))
             for s in s3a['gap_sweep']]
    good = [(t, m) for t, m in pairs if np.isfinite(t) and np.isfinite(m)]
    rmse = float(np.sqrt(np.mean([(t - m) ** 2 for t, m in good])))
    target = 1e-2
    ok = rmse < target
    phys_note = ''
    if l2_phys:
        phys_note = (f'物理算例 (谱隙由纠缠谱派生 gap={l2_phys["gap"]:.6f}): '
                     f'理论 {l2_phys["rate_theory"]:.6f} vs 实测 '
                     f'{l2_phys["rate_meas"]:.6f}, 偏差 '
                     f'{abs(l2_phys["rate_theory"] - l2_phys["rate_meas"]):.2e}。')
    record_metric(
        2, '因果链拟合 RMSE', 'RMSE(rate_theory, rate_meas)',
        '受控谱隙扫描下, 幂迭代的理论收敛比 |lam2/lam1| 与实测收敛比之间的均方根偏差。'
        '量的是 L2 这条因果链"理论预言能不能被数值复现"。',
        f'selfref_general_matrix() 的 gap_sweep: {len(good)} 组 (gap, eps) 各取一对 (理论, 实测)',
        '解析对照: |lam2/lam1| (矩阵特征值直接算出, 非拟合)',
        rmse, 7.612e-3, f'< {target:.0e}', ok,
        note=(phys_note + '基线 7.612e-3 由 v11 的 gap_sweep 四组 '
              '(0.9000/0.8955, 0.5000/0.4878, 0.2000/0.1926, 0.0500/0.0473) '
              '直接算出, 与本次实测同源 —— 这是复现, 不是改进。'
              '逐组偏差: 4.49e-3 / 1.22e-2 / 7.41e-3 / 2.74e-3, '
              '与各组步数 (260/45/22/12) 不对应, 故这里不给出偏差来源的解释。'))
    return {'rmse': rmse, 'pairs': good, 'passed': ok}




def metric_entropy_kl(psi_mera, gs, L):
    """
    2.2 纠缠熵 KL 散度 —— MERA 的 Schmidt 谱与精确谱差多远?

    定义: D_KL(p || q) = sum_k p_k ln(p_k / q_k), p = MERA 的归一化 Schmidt 谱,
      q = 精确基态的。另外报对称化的 Jensen-Shannon 散度 (它天然有限)。
    为什么用谱而不是只用熵: 熵是一个数, 谱是一个分布 —— 两个态可以有相同的熵
      却完全不同的谱。KL 对分布形状敏感, 这是它比"熵差多少"更强的地方。
    诚实边界: 两条谱都在 tol=1e-12 处截断过, 所以这里报的 KL 是**截断后的**,
      是真实 KL 的一个下界; 被丢掉的分量数一并报告。
    """
    p, n_drop_p = _schmidt_spectrum(psi_mera, L)
    q, n_drop_q = _schmidt_spectrum(gs, L)
    n = max(len(p), len(q))
    pp = np.zeros(n); pp[:len(p)] = p
    qq = np.zeros(n); qq[:len(q)] = q
    mask = (pp > 0) & (qq > 0)
    kl = float(np.sum(pp[mask] * np.log(pp[mask] / qq[mask])))
    m = 0.5 * (pp + qq)
    # 每一项只在**自己**非零处求和: 若按 mm=(m>0) 统一取支撑, pp 为 0 而 qq 非零的
    # 位置上会出现 0*log(0) = nan —— 而 JS 之所以要和 KL 并排报, 正是因为它在
    # 支撑不匹配时**仍然有限**。用 mm 会让这个性质凭空消失。
    mp, mq = pp > 0, qq > 0
    js = 0.5 * float(np.sum(pp[mp] * np.log(pp[mp] / m[mp]))) \
        + 0.5 * float(np.sum(qq[mq] * np.log(qq[mq] / m[mq])))
    target, target_js = 1e-2, 1e-3
    ok = bool(kl < target and js < target_js)
    record_metric(
        2, '纠缠熵 KL 散度', 'D_KL(p || q),  p,q = 归一化 Schmidt 谱',
        'MERA 态的 Schmidt 谱 p 相对精确基态谱 q 的 KL 散度; 另报对称化的 JS。'
        '比"熵差多少"更强: 熵相同而谱不同的两个态, 熵判据看不出来, KL 能。',
        f'半链二分 SVD, 两条谱均在 1e-12 处截断后重归一化 (L={L})',
        '解析对照: 精确对角化基态的 Schmidt 谱',
        kl, None, f'< {target:.0e} nats 且 JS < {target_js:.0e}', ok,
        note=(f'JS = {js:.3e}。谱在 1e-12 处截断: MERA 丢 {n_drop_p} 个分量, '
              f'精确态丢 {n_drop_q} 个 —— 所以这个 KL 是真实值的**下界**。'
              f'支撑长度: MERA {len(p)}, 精确 {len(q)}, 共同 {int(mask.sum())} '
              f'(KL 只在这个交集上求和)。**反直觉但属实**: 精确基态在 L=16 '
              f'时已是有效低秩 (最小的保留分量 s={np.sqrt(q[-1]):.2e}), 而 chi=4 的 '
              f'MERA 反而拖出一条更长的弱尾 —— 熵匹配不代表尾部匹配。'))
    return {'kl': kl, 'js': js, 'n_p': len(p), 'n_q': len(q),
            'n_common': int(mask.sum()), 'n_drop_p': n_drop_p,
            'n_drop_q': n_drop_q, 'passed': ok}





def metric_jordan_peak(transient):
    """
    2.4 Jordan 块收敛率 —— 退化矩阵的瞬态峰值位置能不能被解析预言?

    定义: A = lam*I + N_N (单个 N 阶 Jordan 块, |lam| < 1)。谱隙 |lam2/lam1| = 1,
      所以**谱分析预言"不收敛"**; 而 ||A^k||_2 会先暴涨再衰减。
      峰值位置的解析估计: k* = (N-1) / |ln lam|
        (推导: ||A^k|| 由 Toeplitz 矩阵最大元 C(k,N-1)|lam|^(k-N+1) 主导,
         对 k 求极值得 (N-1)/k + ln|lam| = 0)
      指标 = |k_peak_measured - k*| / k*。

    为什么这个指标有分量: 它是一条**真正有预言力的**解析结果 —— 不是"数值和理论一致"
      这种事后说法, 而是先算出 47.5 再去看数值落在哪里。实测 (N=6, lam=0.9):
      ||A^k||_2 峰值在 k=49, 状态范数峰值在 k=48。误差 1%~3%。
    """
    N = int(transient['N'])
    lam = float(transient['lam'])
    k_meas = int(transient['k_peak'])
    k_norm_meas = int(transient['k_norm_peak'])
    k_theory = float((N - 1) / abs(np.log(lam))) if 0 < lam < 1 else float('nan')
    rel = float(abs(k_meas - k_theory) / k_theory) if np.isfinite(k_theory) else float('nan')
    target = 0.05
    ok = bool(np.isfinite(rel) and rel < target)
    record_metric(
        2, 'Jordan 块收敛率', 'k* = (N-1)/|ln lam|  vs  实测 ||A^k||_2 峰值位置',
        f'{N} 阶单个 Jordan 块 (lam={lam}) 的幂次范数 ||A^k||_2 的峰值位置, '
        '与解析估计 k* = (N-1)/|ln lam| 的相对误差。'
        '它 одновременно证明谱隙判据在此失效 (|lam2/lam1| = 1 却仍有结构) '
        '且瞬态峰值位置是可预言的。',
        'selfref_general_matrix() 的 transient: 扫 k=1..300 取 ||A^k||_2 的 argmax',
        '解析对照: k* = (N-1)/|ln lam| (对 Toeplitz 主元求极值导出)',
        rel, None, f'相对误差 < {target:.0%}', ok,
        note=(f'实测: ||A^k||_2 峰值 k={k_meas} (解析 {k_theory:.2f}), '
              f'状态范数 ||A^k x0|| 峰值 k={k_norm_meas}。'
              f'谱半径 rho={transient["spectral_radius"]:.4f} < 1, '
              f'峰值放大 {transient["peak_ratio"]:.1f} 倍, '
              f'峰值处 ||A^k||_2/rho^k = {transient["peak_over_rho_k"]:.3g} —— '
              f'谱半径把范数低估了这么多倍, 这就是非正规矩阵的"假收敛"陷阱。'))
    return {'k_meas': k_meas, 'k_theory': k_theory, 'k_norm_meas': k_norm_meas,
            'rel': rel, 'passed': ok}



def metric_correlation_exponent(gs, L):
    """
    2.3 关联衰减指数 —— 临界 Ising 的关联函数指数 eta 对不对?

    定义: 临界 1+1 维 Ising 普适类里 spin 算符的标度维 Delta_sigma = 1/8,
      等时关联函数 |C(r)| = A * [(pi/L)/sin(pi r/L)]^(2*Delta_sigma), 即 eta = 2*Delta = 1/4。
      指标 = 拟合出的 2*Delta 相对 1/4 的相对误差。

    为什么**必须**用共形形式而不是朴素幂律 log|C| ~ -eta log r:
      本脚本实测过 —— 朴素幂律给 eta = 0.205 (18% 误差), 共形形式给 2*Delta = 0.2426
      (3% 误差)。有限环上的 sin 修正不可忽略, 朴素幂律是在用错误的函数形式做拟合。
      朴素幂律的斜率仍然并排打印出来, 作为"选错形式的代价"的对照, 不掩盖。
    """
    _, C, _ = boundary_correlation_graph(gs, L)
    r = np.arange(1, L // 2)
    y = np.abs(C[0, r])
    f = (np.pi / L) / np.sin(np.pi * r / L)   # CFT 的周期像, 随 r 单调下降
    two_delta = float(np.polyfit(np.log(f), np.log(y), 1)[0])   # |C| = A * f^(2*Delta)
    eta_naive = float(-np.polyfit(np.log(r), np.log(y), 1)[0])
    exact = 0.25
    rel = float(abs(two_delta - exact) / exact)
    target = 0.10
    ok = rel < target
    record_metric(
        2, '关联衰减指数', 'eta = 2*Delta_sigma,  |C(r)| = A*[(pi/L)sin(pi r/L)]^(2*Delta)',
        '临界 TF-Ising 基态的等时自旋关联函数 C(r) = <sigma^z_0 sigma^z_r> 的衰减指数,'
        '用共形不变性给出的周期形式拟合 (不是朴素幂律)。',
        f'boundary_correlation_graph(gs,{L}) 取 C[0,r], r=1..{L//2 - 1}, '
        f'对 ln|C| 与 ln[(pi/L)/sin(pi r/L)] 做线性拟合',
        '解析对照: Delta_sigma = 1/8 -> eta = 1/4 (2D Ising / c=1/2 普适类)',
        two_delta, exact, f'相对误差 < {target:.0%}', ok,
        note=(f'朴素幂律 log|C| ~ -eta log r 给 eta = {eta_naive:.4f}, '
              f'相对误差 {abs(eta_naive - exact) / exact:.1%} —— '
              f'远差于共形形式。这正说明有限环上不能用朴素幂律。'
              f'MERA 态 (非精确态) 的同量在下方"差距诊断"里给出。'))
    return {'two_delta': two_delta, 'eta_naive': eta_naive, 'exact': exact,
            'rel': rel, 'passed': ok}



def exact_ground_state(L, J=1.0, h=1.0):
    """稀疏 eigsh 求基态。**本函数没有 L<=16 这个上限**: 同构造的
    spiral_model_v16.exact_ground_state 在 L=18 上实测建 H 1.53s + eigsh 3.86s
    + 433MB, 基态与 Jordan-Wigner 闭合式仍吻合 2.5e-14。16 这个数来自末圈 MERA
    交叉校验要求 L 为 2 的幂, 与本函数无关。返回 (E0, gs)"""
    H = tfi_periodic_sparse(L, J, h)
    E, V = eigsh(H, k=1, which='SA')
    gs = V[:, 0]
    return float(E[0]), gs / np.linalg.norm(gs)



def boundary_correlation_graph(psi, L, keep_per_node=3):
    """
    从 (MERA 或精确) 态的边界关联构造涌现几何:
        C_ij = <sigma^z_i sigma^z_j>,   d_ij = 1 - |C_ij|
    每个节点保留距离最近的 keep_per_node 条边 —— 稀疏化, 使图连通且规模可控。
    站点 0 = 最高有效位 (与 tfi_periodic_sparse 的约定一致)。
    """
    psi = np.asarray(psi).reshape(-1)
    psi = psi / np.linalg.norm(psi)
    idx = np.arange(2 ** L)
    bits = (idx[:, None] >> np.arange(L - 1, -1, -1)) & 1
    z = 1.0 - 2.0 * bits
    prob = psi ** 2
    C = np.einsum('s,si,sj->ij', prob, z, z)

    Dm = 1.0 - np.abs(C)
    np.fill_diagonal(Dm, 0.0)
    G = nx.Graph()
    G.add_nodes_from(range(L))
    for i in range(L):
        added = 0
        for j in np.argsort(Dm[i]):
            if j == i:
                continue
            G.add_edge(i, int(j), weight=float(Dm[i, j]))
            added += 1
            if added >= keep_per_node:
                break
    return G, C, Dm




def tfi_periodic_sparse(L, J=1.0, h=1.0):
    """
    周期边界 (PBC) 横场 Ising 稀疏哈密顿量:
        H = -J Sum_i sigma^z_i sigma^z_{i+1} - h Sum_i sigma^x_i

    位序约定 (关键, 与 quimb MERA 的输出指标 k0..k_{L-1} 对齐):
        站点 i 对应二进制位 (L-1-i), 即 k0 是最高有效位。这样 quimb 收缩
        出的稠密矢量 reshape(-1) 后, 前 n 个分量恰好对应前 n 个连续站点,
        纠缠熵 S(n) 的块划分才与物理一致。
        (v9 的 tfi_hamiltonian 用的是相反位序; 二者由环的反射对称性联系,
         本征值完全相同, 已实测一致到 7e-15。)
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
        H = H + (-J) * mat
    return ((H + H.T) / 2).tocsr()




def _char_scale_checked(delta, box, deconv=False):
    """
    _char_scale 的输入校验包装 —— 返回 (lam, pk, valid)。

    为什么需要: _char_scale 内部是 `lam = 2*pi / k[argmax(Delta^2(k))]`, 那个
    argmax **不要求峰有内部性**, 也不检查输入场是否退化。两种退化输入会给出
    看起来正常的数字:
      * **空数组** —— _band_power 里的 np.fft.rfftn 直接抛 ValueError, 崩在
        它自己的 len==0 守卫之前, 报错信息与"场为空"这件事没有可读的关联;
      * **平坦场** —— 频谱恒为零, argmax 退化为第 0 个 bin, 于是返回
        lam = box, 一个**有限且看起来合理**的伪值。调用方原本的
        `isfinite(lam) and lam > 0` 守卫接不住它 —— 那不是守卫失效, 是那个
        守卫本来就不是为这件事写的。

    本包装只加两件事: 空场给出可读的报错, 退化场把 valid 标成 False。
    **取峰规则一个字节都没改**。换掉 argmax 需要一个有物理依据的替代规则;
    已试过的替换方案 (在 lambda >= k 的壳里取 P(k) 最大者) 在合成条纹上验证
    通过、在真实场上退化到盒子尺度的低频团块, 稳定性不达标。拿一个未验证的
    估计量换掉一个已知有缺陷的估计量, 风险大于收益, 所以缺口如实保留。

    诚实边界: valid=False 只说"这一次的输入不满足定波长的前提", 不说 lam 一定
    是错的; valid=True 也不说 lam 有独立佐证 (这一条由 A 段的诊断项回答)。
    """
    d = np.asarray(delta, dtype=np.float64)
    if d.size == 0:
        raise ValueError('_char_scale_checked: 输入场为空, 无法定特征波长; '
                         '(原路径会在 _band_power 的 rfftn 处抛 ValueError)')
    if not np.isfinite(d).all():
        return float('nan'), None, False
    lam, pk = _char_scale(delta, box, deconv=deconv)
    # 平坦性判据必须**相对**, 不能用 `std() > 0`。常数场取浮点非整数时 (如 3.7),
    # 1296 个元素的成对求和均值并不位等于 3.7, 偏差在 1 ulp 量级 —— std 于是算
    # 出 ~1e-16 而不是 0, 恰好从 `> 0` 里漏过去。这与上面批评原守卫的是同一类
    # 毛病: 判据的量级没跟被判对象对齐。改用"中心化后是否还有相对波动"。
    dc = d - d.mean()
    scale = max(float(np.abs(d).max()), 1e-300)
    valid = bool(float(np.abs(dc).max()) > 1e-12 * scale
                 and np.isfinite(lam) and lam > 0)
    return lam, pk, valid




def metrics_for_json(obj):
    """
    递归转成 json 可序列化的原生类型; 同时**丢掉所有下划线开头的键**。

    为什么要丢: 第三层会把作图用的紧凑数组挂在 `_plot` 下 (时空图的抽样像、
    功率谱曲线、逐帧亮度序列)。那些是几百 KB 的数组, 进 json 既撑大文件又
    没有审计价值 (它们是已经算出来的指标的原始素材, 不是指标本身)。
    命名约定 `_` 前缀 = 内存内传递, 不落盘。
    """
    if isinstance(obj, dict):
        return {str(k): metrics_for_json(v) for k, v in obj.items()
                if not str(k).startswith('_')}
    if isinstance(obj, (list, tuple)):
        return [metrics_for_json(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
