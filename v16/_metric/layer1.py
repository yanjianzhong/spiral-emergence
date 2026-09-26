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
"""spiral_metric_v16._metric.layer1 —— 由 spiral_metric_v16.py 拆分。

v14 第一层·分布层 (多种子 / 稳健性 / 预算 / chi 瓶颈)

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
    EPS_T, RC_T, record_metric,
)




# ===========================================================================
# 第一层 (分布口径)
# ===========================================================================
def metric_central_charge_error_seeds(runs, chi, L, baseline=0.1145):
    """
    1.1 中心荷误差 —— 从"一条轨迹的单点值"改成"多条轨迹的分布"

    11.45% 不是 chi=4 的表示极限, 而是一条**未收敛轨迹的瞬时值**:
    同预算下换种子/换学习率, 这个数在 1%~60% 之间跳 (见 _v13_diag_conv)。
    所以单点报法缺的不是精度, 是**方差** —— 一个没有误差棒的单点值
    无法区分"表示能力不够"和"这次没跑好"。

    判据仍然卡在最坏情况 (max < 5%), 不是中位数 —— 中位数达标但某条轨迹
    60% 的话, "chi=4 能做到 5%" 这句话就不成立。
    """
    e = np.array([r['eps_c'] for r in runs], dtype=float)
    med, lo, hi = float(np.median(e)), float(e.min()), float(e.max())
    ok = bool(hi < EPS_T)
    lrs = sorted({r['lr'] for r in runs})
    record_metric(
        1, '中心荷误差', 'eps_c = |c_MERA - c_exact| / c_exact',
        'MERA 态按 CFT 形式拟合出的中心荷 c, 相对同 L 精确值的偏差。'
        '报多条轨迹的分布 (中位数与全距), 判据卡最大值。',
        f'fit_central_charge(MERA 态, L={L}) vs 精确基态同 L 拟合; '
        f'chi={chi}, {len(runs)} 条轨迹 (lr={" / ".join(str(x) for x in lrs)}), '
        f'每条取 overlap 最高检查点',
        '解析对照: c=1/2 的 CFT (Ising 普适类)',
        {'median': med, 'min': lo, 'max': hi, 'all': e.tolist()},
        baseline, f'所有轨迹 < {EPS_T:.0%}', ok,
        note=(f'基线 {baseline:.4f} 是单条 600 步轨迹的瞬时值, 无误差棒。'
              f'本层实测 {len(runs)} 条轨迹全距 [{lo:.2%}, {hi:.2%}] —— '
              f'那个基线落在本分布的右尾, 说明它是**未收敛**而不是**表示极限**。'
              f'判据用 max 而非 median: 只要有一条轨迹做不到, "chi={chi} 能做到"'
              f'就不成立。目标 5% 未放宽。'),
        # 供 `G6 最坏轨迹也达标` 指认 (v16.1·C-1)。两者读**同一份 runs、同一个
        # EPS_T** ⇒ 登记为**同源**, 不构成独立佐证。逐条登记见
        # `v16/spiral_v16_audit.md` §6.4。
        claim_id='1.1')
    return {'median': med, 'min': lo, 'max': hi, 'passed': ok}




def metric_conformal_invariance_seeds(runs, rc_exact):
    """
    1.2 共形不变性 —— 同样改成分布, 判据卡最大。

    精确基态同量 (rc_exact) 是有限尺寸造成的残差地板; 它不随优化预算变,
    所以"MERA 相对地板高多少倍"这个说法依然成立, 依然写进 note。
    """
    r = np.array([x['R_conf'] for x in runs], dtype=float)
    med, lo, hi = float(np.median(r)), float(r.min()), float(r.max())
    ok = bool(hi < RC_T)
    record_metric(
        1, '共形不变性', 'R_conf = rms(CFT拟合) / mean(S(n))',
        'S(n) 对 CFT 形式 S(n) = (c/3)ln[(L/pi)sin(pi n/L)] + const 的归一化'
        '拟合残差。量的是"形式对不对", 与"c 的数值准不准"是两件事。',
        f'fit_central_charge(psi, L)["rms"] / mean(["S"]); {len(runs)} 条轨迹, '
        f'同 L 下同时算 MERA 与精确基态',
        '解析对照: 共形场论的 Cardy-Calabrese 公式本身',
        {'median': med, 'min': lo, 'max': hi, 'all': r.tolist()},
        None, f'所有轨迹 < {RC_T:.0%}', ok,
        note=(f'精确基态同量 = {rc_exact:.3e}, 是有限尺寸造成的残差地板; '
              f'实测全距 [{lo:.3e}, {hi:.3e}]。'
              f'基线未达标值 0.0102 与本分布的右尾同量级, 同样是预算问题而非形式问题。'
              f'目标 1e-2 未放宽。'))
    return {'median': med, 'min': lo, 'max': hi, 'R_conf_exact': rc_exact,
            'passed': ok}




def metric_design_robustness(runs, chi, L):
    """
    1.5 全程稳健性 —— 结论必须在**每一条**轨迹上同时成立

    这是本层唯一新增的达标项, 它防的是一个具体的作弊路径:
    多跑几个种子, 然后把中位数报出来当结论。若 3 条轨迹里 2 条达标、1 条
    是 60%, 中位数可能仍然"达标", 但"chi=4 能复现 c=1/2"这句话是假的。

    所以判据写成: 每一条轨迹 (种子 x 学习率) 都要达标。这不是新目标,
    就是把 1.1 与 1.2 的阈值同时施加到每一条轨迹上。
    """
    eps = np.array([r['eps_c'] for r in runs], dtype=float)
    rc = np.array([r['R_conf'] for r in runs], dtype=float)
    i_worst = int(np.argmax(eps / EPS_T + rc / RC_T))
    ok = bool(eps.max() < EPS_T and rc.max() < RC_T)
    record_metric(
        1, '全程稳健性', f'max over {len(runs)} 条轨迹 (eps_c, R_conf) < ({EPS_T:.0%}, {RC_T:.0%})',
        '1.1 与 1.2 的阈值同时施加到每一条独立轨迹 (不同随机种子 x 不同学习率), '
        '取最坏情况。防止"多跑几个种子然后报中位数"把失败轨迹平均掉。',
        f'chi={chi}, L={L}, 各轨迹独立初始化 (种子) 与独立优化器 (学习率)',
        '无外部参照 —— 这是自设的稳健性要求, 不是文献标准',
        {'max_eps_c': float(eps.max()), 'max_R_conf': float(rc.max()),
         'n_runs': len(runs),
         'worst_run': {'lr': runs[i_worst]['lr'], 'seed': runs[i_worst]['seed'],
                       'eps_c': runs[i_worst]['eps_c'],
                       'R_conf': runs[i_worst]['R_conf']}},
        None, f'max eps_c < {EPS_T:.0%} 且 max R_conf < {RC_T:.0%}', ok,
        note=(f'最坏轨迹: lr={runs[i_worst]["lr"]}, seed={runs[i_worst]["seed"]}, '
              f'eps_c={eps[i_worst]:.4%}, R_conf={rc[i_worst]:.3e}。'
              f'这一项比 1.1/1.2 严格 —— 它不可能在 1.1/1.2 失败时通过。'))
    return {'max_eps_c': float(eps.max()), 'max_R_conf': float(rc.max()),
            'n_runs': len(runs), 'passed': ok}




def metric_budget_diagnostic(curve, steps, chi):
    """
    1.D 预算漂移 (诊断项, passed=None) —— 报出"还没收敛多少", 不假装已收敛

    诚实边界必须是可读的数字, 不是一句"大致收敛了"。这里给出
    eps_c(steps) 相对 eps_c(steps/2) 的相对变化。若它明显是负的,
    说明报出的 eps_c 是**上界**而不是最优值。

    为什么这反而是好事: 一个达标的上界比一个达标的最优值更强。
    "chi=4 最终能做到 <5%" 需要最优值达标; 而"chi=4 在 1500 步时
    已经 <5%, 且还在下降"直接堵死"你只是没跑够"这条反驳。
    """
    cs = {int(s): (e, r) for (s, e, r, _o) in curve}
    keys = sorted(cs)
    near = lambda target: min(keys, key=lambda s: abs(s - target))
    k_h, k_f = near(int(steps // 2)), near(int(steps))
    e_h, r_h = cs[k_h]
    e_f, r_f = cs[k_f]
    drift = float((e_f - e_h) / e_h) if e_h > 0 else float('nan')
    record_metric(
        1, '预算漂移', 'drift = (eps_c(N) - eps_c(N/2)) / eps_c(N/2)',
        '训练预算翻倍时中心荷误差的相对变化。负值 = 还在下降, 说明报出的值'
        '是上界; 接近 0 = 已进平台, 说明报出的值就是该 chi 的最优。'
        '这是一个诊断项, 不给通过判定 —— 它的作用是把"收敛了多少"变成可读数字。',
        f'训练过程中每 100 步记录一次; 取 step={k_h} 与 step={k_f} 两点',
        '无外部参照 —— 这是对本次运行自身的收敛性描述',
        {'drift': drift, 'eps_c_half': float(e_h), 'eps_c_full': float(e_f),
         'step_half': k_h, 'step_full': k_f,
         'R_conf_half': float(r_h), 'R_conf_full': float(r_f)},
        None, '仅诊断, 不设阈值', None,
        note=(f'chi={chi}, N={steps}: eps_c 从 {e_h:.3%} (step {k_h}) 到 '
              f'{e_f:.3%} (step {k_f}), 相对变化 {drift:+.1%}。'
              f'{"仍在下降, 故报出值是上界 (对达标有利, 不是不利)" if drift < -0.05 else "已接近平台"}; '
              f'但**不能**声称已收敛到该 chi 的最优值。'))
    return {'drift': drift, 'eps_c_half': float(e_h), 'eps_c_full': float(e_f)}




def metric_chi_bottleneck(chi_sweep, chi_used, floor_chi, baseline=0.1145):
    """
    1.D2 chi 瓶颈诊断 (诊断项, passed=None) —— 基线 note 那句断言到底对不对

    基线 note 写"残差的主因是有限键维截断 (chi=4)"。这是一个**经验断言**,
    但它从没被检验过 —— 因为没有第二个 chi 的结果可比。

    【判定规则为什么是最小充分性, 而不是"实测 vs 参照线的比值"】
    第一版用的是 ratio = eps_c(chi) / Schmidt 截断参照, ratio < 2 就判"表示能力"。
    这个规则在真实数据上直接给反了: 实测 eps_c(chi=4) = 3.03%, 而参照线 = 5.93%,
    ratio = 0.51 < 2 -> 判成"表示能力受限"。但实测值**低于**参照线, 恰恰说明
    参照线不是有效约束 (逐切割 Schmidt 截断本就不是严格下界, 见下), 因此它
    什么也证明不了; 拿它做判据是把一个无效参照当成了标尺。

    改用**受控比较**, 只看同预算下换 chi 的效果:
      chi_gain   = eps_c(chi=4) / eps_c(chi=8)   同种子同步数同 lr, 只有 chi 变
      optim_gain = baseline(chi=4, 600步) / eps_c(chi=4, 1500步)  同 chi, 只有预算变
    再叠加一条不需要任何比较的事实: **chi=4 在同预算下已经达标**。
    一个已经达标的设置不可能是当前瓶颈 —— 这条比任何比值都硬。

    诚实边界 (两个增益并不对称, 必须写明):
      - chi_gain 是严格受控的 (单变量)。
      - optim_gain 把"更多步数 + 最佳检查点选择 + 种子"捆在一起, 是合并数字,
        不能单独归因给步数; 它和 chi_gain 的比较只用于判断**量级**谁更大。
      - 参照线 (逐切割保留前 chi 个精确 Schmidt 分量) **不是严格下界**: 同一个态
        不可能在所有切割点同时取到该 rank, 且 rank-chi 态的熵可在 [0, ln chi] 内
        任意取值。它只作为"表示能力的大致量级"附列, 不参与判定。
    """
    chis = sorted(chi_sweep)
    measured = {c: chi_sweep[c]['eps_c'] for c in chis}
    ctrl = next((c for c in chis if c > chi_used), None)
    chi_gain = (float(measured[chi_used] / measured[ctrl])
                if ctrl is not None and measured[ctrl] > 0 else float('nan'))
    optim_gain = float(baseline / measured[chi_used]) if measured[chi_used] > 0 \
        else float('nan')
    passes = bool(measured[chi_used] < EPS_T)

    if passes:
        verdict = (f'chi 不是瓶颈: chi={chi_used} 同预算下已达标 '
                   f'({measured[chi_used]:.2%} < {EPS_T:.0%})')
    elif chi_gain > optim_gain:
        verdict = '表示能力是主因 (换 chi 的增益大于换预算)'
    else:
        verdict = '预算/优化是主因 (换预算的增益大于换 chi)'

    monotone = all(measured[chis[i]] >= measured[chis[i + 1]]
                   for i in range(len(chis) - 1))
    ratio = float(measured[chi_used] / floor_chi) if floor_chi > 0 else float('nan')
    record_metric(
        1, 'chi 瓶颈归因', '受控比较: 换 chi 的增益 vs 换预算的增益',
        '判定基线 note 的"主因是有限键维截断"这句断言是否成立。'
        '做法: (a) 同种子同步数只改 chi, 量出 chi_gain; '
        '(b) 同 chi 只改预算, 量出 optim_gain; 两者比量级。'
        '另加一条无需比较的硬事实: chi=4 同预算下是否已达标。',
        f'chi in {chis}; 控制组 chi={ctrl}; 每条轨迹取最佳检查点 (按 overlap 选)',
        f'单条 600 步轨迹的 eps_c = {baseline:.4f} (无误差棒) 作为预算对照',
        {'measured': measured, 'chi_gain': chi_gain, 'optim_gain': optim_gain,
         'chi_used': chi_used, 'chi_ctrl': ctrl, 'passes_at_chi_used': passes,
         'monotone_in_chi': monotone,
         'floor_chi_used': floor_chi, 'ratio_over_floor_supporting_only': ratio,
         'verdict': verdict},
        None, '仅诊断, 不设阈值', None,
        note=(f'同种子同步数, chi {chi_used}->{ctrl}: eps_c '
              f'{measured[chi_used]:.3%} -> {measured[ctrl]:.3%}, '
              f'chi_gain = {chi_gain:.2f}x。同 chi={chi_used}, 预算 '
              f'600->1500 步: {baseline:.2%} -> {measured[chi_used]:.3%}, '
              f'optim_gain = {optim_gain:.2f}x。'
              f'chi 单调性: {"成立" if monotone else "不成立"}。'
              f'**结论**: {verdict}。提 chi 只能再降 {chi_gain:.2f}x, '
              f'而基线的 {baseline:.2%} 与本层的 {measured[chi_used]:.3%} 差 '
              f'{optim_gain:.2f}x。注: optim_gain 把步数/检查点选择/种子捆在一起, '
              f'是合并数字; chi_gain 才是单变量受控量。'
              f'Schmidt 截断参照 {floor_chi:.3%} 仅作量级附列, 非严格下界, '
              f'不参与判定。'))
    return {'chi_gain': chi_gain, 'optim_gain': optim_gain, 'verdict': verdict,
            'monotone': monotone, 'measured': measured}
