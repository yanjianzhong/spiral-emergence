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
"""spiral_metric_v16._metric.negctrl —— 由 spiral_metric_v16.py 拆分。

负对照 T3 段 (F2a / F1 / F5)

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
    record_guard, record_metric,
)




# ===========================================================================
# v14 · F2a 负对照 (h/J 扫描) 的指标与守卫
# ===========================================================================
def metric_hj_negative_control(hj):
    """
    2.5 临界性负对照 —— 把 h/J 推离 1.0, 看指标是否塌缩。

    这是 v14 的生死检验。判据**已修正** (任务书 Task 3 的原判据物理上不成立,
    详见 spiral_model_v16.hj_scan 的函数头):
      主判据: argmax c_fit == h/J=1.0。含义是 Calabrese-Cardy ansatz 只在临界
              点成立 —— 实测 0.5072, 而 0.5 处 0.0646、1.5 处 0.1317。
      次判据: argmax S(L/2) 落在 [0.9, 1.05]。有限尺寸熵峰偏向 h/J<1 是
              **已声明**的效应 (stage2_critical_break docstring), 故给窗口。
      区分力: 非临界点上框架必须**正确失败** —— 审计 §五.2「一个能在非临界点
              正确失败的框架, 比一个永远输出 3% 的框架可靠」。要求
              |c-0.5|/0.5 > 20%。
      gap=p1/p0 是**单调量**, 无内部极值, 降级为诊断, **不作判据**。
    """
    if not hj:
        record_metric(2, '5 临界性负对照 (h/J 扫描)', 'argmax c_fit @ h/J=1',
                      '把 h/J 推离 1.0, 检验临界指标是否塌缩',
                      'spiral_model_v16.hj_scan()',
                      '横场 Ising 热力学临界点 h/J=1', None, None,
                      '未运行 (ctx 无 hj)', None,
                      note='负对照未执行, 无法判定。')
        record_guard('F2 负对照已运行', 'collect_metrics_v16',
                     'ctx["hj"] 存在', False)
        return None

    v = hj['verdict']
    pts = hj['points']
    measured = {
        'hj_at_c_peak': v['hj_at_c_peak'],
        'c_peak': float(max(q['c_fit'] for q in pts)),
        'c_at_hj_first': float(pts[0]['c_fit']),
        'c_at_hj_last': float(pts[-1]['c_fit']),
        'hj_at_s_peak': v['hj_at_s_peak'],
        'gap_monotone': v['gap_monotone_decreasing'],
        'discrim': v['discrim'],
    }
    record_metric(
        2, '5 临界性负对照 (h/J 扫描)', 'argmax c_fit @ h/J=1',
        '把 h/J 推离 1.0, 检验临界指标是否塌缩 —— 临界模拟是不是事实',
        'spiral_model_v16.hj_scan(): 逐点精确对角化 + Calabrese-Cardy 拟合',
        '横场 Ising 热力学临界点 h/J=1 (解析已知)',
        measured,
        '基线只在 h/J=1.0 单点检验, 无负对照',
        'c_fit 峰位 = 1.0 且 S 峰位 in [0.9,1.05] 且非临界点 |c-0.5|/0.5 > 20%',
        v['passed'],
        note=f"**判据已修正**: 任务书原判据 (gap 在 h/J=1.0 取极小值) 物理上"
             f"不成立 —— gap 对 h/J 单调递减 ({pts[0]['gap']:.4f} -> "
             f"{pts[-1]['gap']:.4f}), 曲线无内部极值, 照原判据会**伪失败**。"
             f"改用 c_fit 峰 (0.5072 @ h/J=1.0, 首点 {pts[0]['c_fit']:.4f}, "
             f"末点 {pts[-1]['c_fit']:.4f}) + S 峰位 ({v['hj_at_s_peak']:.2f}) "
             f"+ 区分力。诚实边界: S 峰位漂移 {v['hj_at_s_peak']-1.0:+.2f} 是"
             f"有限尺寸效应 (L=16 熵峰偏向 h/J<1), 已声明;")

    # ---- 三条可证伪守卫 ----
    record_guard('F2a 主判据: c_fit 峰在临界点', 'hj_scan',
                 f"argmax c_fit = h/J {v['hj_at_c_peak']:.2f} (要求 = 1.0)",
                 v['ok_c_peak_at_critical'],
                 note='Calabrese-Cardy ansatz 只在临界点成立 —— 比"某个数小"强。')
    record_guard('F2a 次判据: S(L/2) 峰位在窗口内', 'hj_scan',
                 f"argmax S = h/J {v['hj_at_s_peak']:.2f} "
                 f"(窗口 {v['s_window']})",
                 v['ok_s_peak_in_window'],
                 note='窗口而非单点: 有限尺寸熵峰偏向 h/J<1 是预期效应。')
    record_guard('F2a 区分力: 非临界点正确失败', 'hj_scan',
                 f"h/J in {sorted(v['discrim'])} 上 |c-0.5|/0.5 > "
                 f"{v['discrim_min']:.0%}",
                 v['ok_discriminating'],
                 note='审计 §五.2: 能在非临界点正确失败的框架才可靠。')
    record_guard('F2a 诊断: gap 单调 (不作判据)', 'hj_scan',
                 f"gap 对 h/J 单调递减 = {v['gap_monotone_decreasing']}",
                 v['gap_monotone_decreasing'], expect_pass=False,
                 note='这条**故意不计入通过率**: 单调是物理预期 (GHZ->乘积态), '
                      '不是"通过"。它的作用是防止有人再把 gap 当临界判据用。')

    print(f"\n  ── v14·F2a 负对照 (h/J 扫描) ──")
    print(f"      c_fit 峰位 h/J={v['hj_at_c_peak']:.2f} "
          f"(c={measured['c_peak']:.4f}) | S 峰位 h/J={v['hj_at_s_peak']:.2f} "
          f"| 区分力 {'PASS' if v['ok_discriminating'] else 'FAIL'}")
    print(f"      裁决: {'临界性成立' if v['passed'] else '未通过'}")
    return measured




# ===========================================================================
# v14 · F1 跨边界条件中心荷交叉检验的指标与守卫
# ===========================================================================
def metric_f1_cross_boundary(f1):
    """
    2.6 跨边界条件中心荷交叉检验。

    **分层标注** (审计 §三之补 F1① 的降级要求, 必须写在指标名里):
      同族   —— PBC vs θ=π Z2 twist。换的是**边界条件与态**, 不换估计量;
                两边都用同一个 Calabrese-Cardy log 拟合。
      独立族 —— **未建立**。已试的 Rényi 指数扫描实测验证失败 (L=16 上
                c(alpha) 标准差 0.0225、均值对 0.5 偏 7.97%), 如实登记为诊断项,
                不冒充第二把尺子。

    两个偏差并列报告 (审计 F1② 要求, 5% 标准不放宽):
      对 c_exact(L) —— 消掉有限尺寸偏差后的纯方法误差;
      对真值 0.5    —— 把有限尺寸偏差也算进去的端到端误差。
    只报后者会掩盖有限尺寸, 只报前者会让读者误以为到达了精确值。
    """
    if not f1:
        record_metric(2, '6 跨边界条件中心荷交叉检验 (同族)',
                      '|c_twist - c_PBC| / c_PBC',
                      '换边界条件后中心荷读数是否稳定', 'spiral_model_v16'
                      '.cross_boundary_twist()',
                      '临界 Ising c=0.5', None, None,
                      '未运行 (ctx 无 f1)', None,
                      note='未执行, 无法判定。')
        record_guard('F1 跨边界条件已运行', 'collect_metrics_v16',
                     'ctx["f1"] 存在', False)
        return None

    sd = f1['sides']
    c_p = sd['pbc']['c_representative']
    c_t = sd['twist']['c_representative']
    bd, ry = f1['boundary'], f1['renyi']
    dt, de = f1['dev_vs_true'], f1['dev_vs_c_exact']

    # 边界条件真的换了态: E0 与 S(L/2) **都**必须变 (只变一个说明实现有问题)
    bc_changed = bool(bd['E0_twist'] != bd['E0_pbc']
                      and bd['S_half_twist'] != bd['S_half_pbc'])
    devs_reported = bool(
        all(k in dt for k in ('pbc', 'twist'))
        and all(k in de for k in ('pbc', 'twist'))
        and all(np.isfinite([dt['pbc'], dt['twist'], de['pbc'], de['twist']])))

    measured = {
        'c_pbc': c_p, 'c_twist': c_t,
        'c_exact_pbc': sd['pbc']['c_exact'], 'c_exact_twist': sd['twist']['c_exact'],
        'rel_diff': f1['rel_diff'],
        'dev_vs_exact_pbc': de['pbc'], 'dev_vs_exact_twist': de['twist'],
        'dev_vs_true_pbc': dt['pbc'], 'dev_vs_true_twist': dt['twist'],
        'c_std_pbc': sd['pbc']['c_std'], 'c_std_twist': sd['twist']['c_std'],
        'min_overlap': f1['min_overlap'],
        'renyi_mean': ry['mean'], 'renyi_std': ry['std'],
        'renyi_dev_vs_true': ry['dev_vs_true'], 'renyi_validated': ry['validated'],
    }
    record_metric(
        2, '6 跨边界条件中心荷交叉检验 (同族)', '|c_twist - c_PBC| / c_PBC',
        '换边界条件 (PBC -> θ=π Z2 twist) 后中心荷读数是否稳定 —— '
        '**同族**交叉检验 (换态与边界条件, 不换估计量)',
        'spiral_model_v16.cross_boundary_twist(): 精确对角化 + 检查点选择协议 MERA '
        f'(每边 {len(f1["seeds"])} 种子 x {f1["steps"]} 步, 取 eps_c 中位数轨迹)',
        '临界 Ising c=0.5 (体中心荷不应随边界条件改变)',
        measured,
        '基线只有 PBC 一个边界条件, 无跨边界条件检验',
        f'相对差 <= {f1["rel_max"]:.0%}, 且两个偏差 (vs c_exact(L) 与 vs 0.5) 并列报告',
        f1['passed'],
        note=f"**同族**标注: ansatz 仍是同一个 Calabrese-Cardy log 式, 换的只是"
             f"边界条件与态 —— 它检验稳健性, **不是**独立族估计量。"
             f"独立族**: **未建立**。已试 Rényi 指数扫描 (alpha in "
             f"{[r['alpha'] for r in ry['alphas']]}), 实测均值 c="
             f"{ry['mean']:.5f}, std={ry['std']:.5f}, 对 0.5 偏差 "
             f"{ry['dev_vs_true']:.2%} -> 验证不通过, 降为诊断项。"
             f"诚实边界: 这是受 L<=16 天花板限制的结果, 已如实声明; "
             f"未伪造第二把尺子。")

    # ---- 可证伪守卫 ----
    record_guard('F1 同族: 跨边界条件已运行 (PBC vs θ=π twist)',
                 'cross_boundary_twist',
                 f"两侧均有 {len(f1['seeds'])} 条轨迹, L={f1['L']}, chi={f1['chi']}",
                 bool(sd['pbc']['runs'] and sd['twist']['runs']))
    record_guard('F1 twist 保持实矩阵 (可喂 float MERA)',
                 'cross_boundary_twist',
                 f"θ=π twist 的 H 为实矩阵 = {bd['twist_h_is_real']}",
                 bd['twist_h_is_real'],
                 note='连续 U(1) twist 给复矩阵, 那样就得改张量网络 dtype —— 本项'
                      '刻意选 Z2 (θ=π), 只为不引入复张量网络。')
    record_guard('F1 边界条件确实换了态 (E0 与 S(L/2) 都变)',
                 'cross_boundary_twist',
                 f"E0/L: {bd['E0_pbc']/f1['L']:+.6f} vs "
                 f"{bd['E0_twist']/f1['L']:+.6f}; S(L/2): "
                 f"{bd['S_half_pbc']:.5f} vs {bd['S_half_twist']:.5f}",
                 bc_changed,
                 note='防"伪对照": 若两者相同, 说明 twist 没生效, 这条会抓住。')
    record_guard('F1 MERA 全部轨迹重叠 >= 0.99', 'cross_boundary_twist',
                 f"min overlap = {f1['min_overlap']:.5f} (两侧共 "
                 f"{len(f1['seeds']) * 2} 条轨迹)",
                 f1['min_overlap'] >= 0.99,
                 note='重叠低说明是**优化没跑到位**, 不是物理 —— 实测朴素协议 '
                      '(600 步/单种子) 在 PBC 侧只到 0.98829, 会把跨边界条件差'
                      '夸大成 5.98%。')
    record_guard('F1 跨边界条件相对差 <= 5%', 'cross_boundary_twist',
                 f"|c_twist - c_PBC|/c_PBC = {f1['rel_diff']:.3%} "
                 f"({c_t:.5f} vs {c_p:.5f})",
                 f1['passed'],
                 note='审计要求 5% 标准**不放宽**。')
    record_guard('F1 两个偏差并列报告 (vs c_exact(L) 与 vs 0.5)',
                 'cross_boundary_twist',
                 f"vs c_exact(L): PBC {de['pbc']:.3%} / twist {de['twist']:.3%}; "
                 f"vs 0.5: PBC {dt['pbc']:.3%} / twist {dt['twist']:.3%}",
                 devs_reported,
                 note='审计 F1②: 只报一个偏差会分别掩盖有限尺寸或方法误差。')
    record_guard('F1 诊断: Rényi 扫描不作独立族估计量 (L<=16 实测失效)',
                 'renyi_alpha_scan',
                 f"c(alpha) std = {ry['std']:.5f} (门槛 "
                 f"{0.02}), 均值对 0.5 偏差 = {ry['dev_vs_true']:.2%}",
                 ry['validated'], expect_pass=False,
                 note='这条**故意不计入通过率**: 它是负面结果的可复现证据。'
                      '结论是"本工作未建立独立族估计量", 而不是"通过了"。')

    print(f"\n  ── v14·F1 跨边界条件中心荷 (同族) ──")
    print(f"      PBC c={c_p:.5f} | θ=π twist c={c_t:.5f} | "
          f"相对差 {f1['rel_diff']:.3%} "
          f"{'PASS' if f1['passed'] else 'FAIL'}")
    print(f"      [独立族] Rényi 扫描 std={ry['std']:.5f} "
          f"-> {'可作估计量' if ry['validated'] else '验证失败, 未建立独立族估计量'}")
    return measured




# ===========================================================================
# v14 · F5 一致性检查束的指标与守卫
# ===========================================================================
def metric_consistency_checks(f5):
    """
    2.7 一致性检查束 —— 五条, 每条都用两条独立路径对同一个量。

    与第二层其它指标的分工: 其它指标对标的是**物理真值** (c=0.5, 2Δ=0.25,
    临界点 h/J=1), 检验物理结论; 这一项对标的是**解析式**, 检验实现本身。
    它通过与否不改变任何物理结论 —— 改的是"这些数字是怎么算出来的"这件事
    的可信度, 以及"解析路径没被绕过"这件事有没有被验过。

    七条判据, 每条都刻意不做成"某个数很小"(那种判据没有内容, 任何实现都能
    让它很小), 而是"两条路径给出同一个数", 容差取各自噪声的量级:
      L2   tol 1e-10  eigsh 与 JW 闭合式都只差舍入
      L3   tol 1e-12  幂迭代自身阈 1e-13, 残差应在同量级
      L5   tol 1e-12  环上 4-2-2=0 是恒等式
      L6a  tol 1e-13  零和是严格换写, 只剩抵消误差
      L6b  tol 1e-12  平衡恒等式两侧是同一更新式的算术
      L7   上界       熵 <= ln N (均匀分布最大熵) 且 |<等权, D*>| <= 1
    删掉的三条 (L2 步数 / L3 Jordan 峰 / L4 酉性) 的理由见 consistency_checks。
    """
    if not f5:
        record_metric(2, '7 一致性检查束 (数值路径 <-> 解析路径)', '5 条各自的双路径偏差',
                      '实现与它声称的解析式是否一致', 'spiral_model_v16'
                      '.consistency_checks()',
                      'JW 闭合式 / 幂迭代残差 / Forman 恒等式 / '
                      '离散 Laplacian 零和 / 平衡恒等式 / 熵上界',
                      None, None, '未运行 (ctx 无 f5)', None,
                      note='未执行, 无法判定。')
        record_guard('F5 一致性检查已运行', 'collect_metrics_v16',
                     'ctx["f5"] 存在', False)
        return None

    c2, c3, c5, c6, c7 = f5['L2'], f5['L3'], f5['L5'], f5['L6'], f5['L7']
    measured = {
        'l2_max_abs_dev': c2['max_abs_dev'], 'l2_n_points': c2['n_points'],
        'l3_max_residual': c3['max_residual'], 'l3_n_case': c3['n_case'],
        'l5_cycle_max_abs': c5['cycle_max_abs'],
        'l5_cycle_min_n': c5['cycle_min_n'],
        'l6_lap_zero_rel_max': c6['lap_zero_rel_max'],
        'l6_balance_abs_err_max': c6['balance_abs_err_max'],
        'l6_drift_first_step': c6['drift_first_step'],
        'l7_entropy_ratio': c7['entropy_ratio'],
        'l7_equal_weight': c7['equal_weight'],
        'l7_defined': c7['defined'], 'l7_bound_ok': c7['bound_ok'],
        # v15·W4 新增: 自指深度 (不动点 Jacobian 谱半径) 与 A/B 分列读数
        'l7_rho_jac': c7['rho_jac'],
        'l7_rho_is_identity': c7['rho_is_identity'],
        'l7_rho_jac_B': c7['rho_jac_B'],
        'l7_ratio_B': c7['ratio_B'], 'l7_defined_B': c7['defined_B'],
        'l7_status_B': c7['status_B'],
    }
    record_metric(
        2, '7 一致性检查束 (数值路径 <-> 解析路径)', '5 条各自的双路径偏差',
        '实现与它声称的解析式是否一致 —— 测的是**实现**, 不是物理结论',
        'spiral_model_v16.consistency_checks(): eigsh vs JW 闭合式 / 幂迭代残差 '
        '/ Forman 环恒等式 / 离散 Laplacian 零和 / 精确平衡恒等式 / 熵上界',
        'Jordan-Wigner 闭合式, 4-deg(u)-deg(v), 周期域上 Laplacian 零和, '
        '同一更新式的离散平衡',
        measured,
        '既有实现完全没有这类对照 —— 三份检查里两份恒真、一份是范畴错误',
        'L2/L3/L5/L6a/L6b < 1e-12 (L2 < 1e-10), L7 两条上界成立',
        bool(c2['max_abs_dev'] < 1e-10 and c3['max_residual'] < 1e-12
             and c5['cycle_max_abs'] < 1e-12
             and c6['lap_zero_rel_max'] < 1e-13
             and c6['balance_abs_err_max'] < 1e-12
             and c7['bound_ok']),
        note=f"**五条各自的两条路径**: L2 用 Jordan-Wigner 闭合式 "
             f"E0=-Sum sqrt(J^2+h^2-2Jh cos(pi(2k+1)/L)) 对 eigsh, "
             f"{c2['n_points']} 个 (L,h) 点最大偏差 {c2['max_abs_dev']:.2e} —— "
             f"这条是**真的独立路径**, 它能测出 H 的位序、周期键、横场符号错, "
             f"而 eigsh 对 H 本身对不对一句话都没有。L3 断言幂迭代残差 "
             f"{c3['max_residual']:.2e} ({c3['n_case']} 个算例) —— "
             f"原判据写的是「与 Lanczos 100/200/500 步无关」, 但 eigsh 没有"
             f"步数参数, 换成 maxiter 后 30 以上逐位相同, 零分辨力。"
             f"L5 环 C_n (n>={c5['cycle_min_n']}) 的 Forman max|F|="
             f"{c5['cycle_max_abs']:.2e} (解析 0); **n=3 已排除** —— C_3 就是 "
             f"K_3, augmented 的 +3 项把它顶到 +3。"
             f"L6 两条: 离散 Laplacian 零和相对残差 "
             f"{c6['lap_zero_rel_max']:.2e}, 精确平衡恒等式 abs 误差 "
             f"{c6['balance_abs_err_max']:.2e}。"
             f"L7 熵比 {c7['entropy_ratio']:.4f} <= 1 且等权度 "
             f"{c7['equal_weight']:.4f} <= 1。"
             f"**诚实边界**: L6 的两条是**格式性质** —— 保证代码按写下的方程在"
             f"算, 不保证那个方程描述的是生命; 首步 <u+v> 漂移 "
             f"{c6['drift_first_step']:+.3e} 就是「u+v 守恒」这个伪不变量的反例 "
             f"(源汇项 F(1-u) 与 (F+k)v 不相消)。"
             f"L7 上界在塌缩支恒取等号, "
             f"本项 defined={c7['defined']}, 那一支不带信息量。")

    record_guard('F5-L2 基态能量与 JW 闭合式一致', 'consistency_checks',
                 f"{c2['n_points']} 个 (L,h) 点 max|eigsh - JW| = "
                 f"{c2['max_abs_dev']:.2e} < 1e-10",
                 bool(c2['max_abs_dev'] < 1e-10),
                 note='这是唯一一条真的独立路径: 它同时验 H 的构造、位序约定与'
                      '周期边界。')
    record_guard('F5-L3 幂迭代残差达阈', 'consistency_checks',
                 f"{c3['n_case']} 个算例 max ||A x - <x,A x> x|| = "
                 f"{c3['max_residual']:.2e} < 1e-12",
                 bool(c3['max_residual'] < 1e-12),
                 note='判据是残差, 不是步数 —— 步数没有解析预期, 残差有。')
    record_guard('F5-L5 Forman 在环 C_n (n>=4) 上为 0', 'validate_curvature',
                 f"n = {sorted(int(k) for k in c5['cycle_mean'])}, "
                 f"max|F| = {c5['cycle_max_abs']:.2e} < 1e-12",
                 bool(c5['cycle_max_abs'] < 1e-12),
                 note='判据是恒等式 4-deg(u)-deg(v); **n=3 必须排除** (C_3=K_3, '
                      'augmented 项给 +3)。')
    record_guard('F5-L6a 离散拉普拉斯零和', 'gray_scott_invariants',
                 f"周期域上 max |Sum Lap u| / (4<u>/h^2) = "
                 f"{c6['lap_zero_rel_max']:.2e} < 1e-13",
                 bool(c6['lap_zero_rel_max'] < 1e-13),
                 note='测 stencil 系数与 roll 的包裹方向, 与守恒律无关。')
    record_guard('F5-L6b Gray-Scott 精确离散平衡恒等式', 'gray_scott_invariants',
                 f"max |d<u+v> - dt[F(1-<u>) - (F+k)<v>]| = "
                 f"{c6['balance_abs_err_max']:.2e} < 1e-12",
                 bool(c6['balance_abs_err_max'] < 1e-12),
                 note='**替代"u+v 质量守恒"这个错误判据**: 源汇项不相消, 照原'
                      '判据会伪失败 (实测首步漂移已超其容差 62 倍)。这条用'
                      '更新前的场算, 两侧是同一更新式的算术, 检验整个 RHS 装配。')
    record_guard('F5-L7 熵比与等权度满足上界', 'stage7_consciousness',
                 f"熵比 {c7['entropy_ratio']:.4f} <= 1 且等权度 "
                 f"{c7['equal_weight']:.4f} <= 1",
                 bool(c7['bound_ok']),
                 note='上界: 熵 <= ln N (均匀分布最大熵) 与 Cauchy-Schwarz。'
                      '**只在非塌缩支有内容** —— 塌缩支 probs=1/N 是约定, 两条'
                      '恒取等号; 该项 defined='
                      f"{c7['defined']}, 如实标注。")
    record_guard('F5 诊断: Gray-Scott 的 u+v 不是守恒量', 'stage6_life',
                 f"首步 <u+v> 漂移 = {c6['drift_first_step']:+.3e} != 0 "
                 f"(源汇项 F(1-<u>) - (F+k)<v> 不相消)",
                 bool(abs(c6['drift_first_step']) < 1e-6), expect_pass=False,
                 note='**故意不计入通过率**: 这是负面事实的可复现证据 —— '
                      '"u+v 守恒到 1e-6"物理上不成立, 只有恰好已在稳态时才成立。'
                      '留着它是为了防止这个错误判据以后又被加回来。')

    # v15·W11: W6 台账登记 L1「空无基底等权叠加蕴潜能」**没有负对照**。本轮按
    # 计划试了"偏置初态 -> 下游 c/xi 退化"这个形式, 结论是**这个形式不成立**,
    # 不是"没做": stage1_void 返回的 equal_state 在整个仓库里**没有被取用过**。
    # 判据用**源码级属性访问**扫描, 所以它可证伪 —— 谁哪天把 equal_state 接上
    # 下游, 这条就会翻成 True。
    # 注意: 本函数**没有** _HERE 参数 (它只收 f5), 所以这里自取模块目录。
    # 第一版写成了 _HERE, 全流程在 NameError 上崩掉 —— 冒烟因为是复制的逻辑
    # 而没照出作用域错误, 这类错误只有真调函数才能发现。
    # 拆分后 model 的实体在 _model/ 子包, 门面只剩 re-export —— 只扫门面
    # 会扫不到函数体, 所以改扫子包全部源码 (扫描面反而更全)。
    _mod_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _mdl_dir = os.path.join(_mod_dir, '_model')
    try:
        _eq_lines = []
        for _fn in sorted(os.listdir(_mdl_dir)):
            if _fn.endswith('.py'):
                with open(os.path.join(_mdl_dir, _fn),
                          encoding='utf-8') as _f:
                    _eq_lines += [l for l in _f.read().splitlines()
                                  if 'equal_state' in l]
        _eq_consumed = any(("['equal_state']" in l) or ('.equal_state' in l)
                           for l in _eq_lines)
    except OSError:
        _eq_lines, _eq_consumed = [], True
    record_guard('W11 诊断: L1 有到下游的因果通路 (负对照的前提)',
                 'stage1_void',
                 f"spiral_model_v16.py 中 equal_state 出现 {len(_eq_lines)} 处, "
                 f"其中被下游取用的 = {_eq_consumed} -> 计划里的"
                 f"「偏置初态 -> 下游 c/xi 退化」可构造 = {_eq_consumed}",
                 bool(_eq_consumed), expect_pass=False,
                 note='**诊断项, 不计入通过率 (本项设计上就该不通过)**: 本条把'
                      'L1 负对照缺口的**原因**定住, 而不是再报一次"缺口存在"。'
                      '(1) **为什么该问**: 负对照的判据是"结论不成立时它必须失败"。'
                      '「空无基底等权叠加蕴潜能」要能被推离, 前提是 L1 的初态'
                      '**真的流到下游**; 否则推离它不改变任何读数, 控制就是摆设。'
                      '(2) **v15 实测**: stage1_void 返回 {unitary, equal_state, '
                      'dimension}。整仓扫描 (v14/v15 两版) —— equal_state '
                      '**从未被取用**; unitary 只取其条件数上报 '
                      '(spiral_model_v16.main 的 void_cond); dimension 只喂 '
                      'derive_L1_local_to_full, 而后者算出的 dim_full '
                      '**只被 print** (main 的 [L1] 打印行), 从不与链的实际'
                      '维数 2**L_chain 比对。'
                      '(3) **所以 v15 里**: L1 到下游**没有因果通路**。计划里那个形式'
                      '造不出来 —— 这是本条与"回头再做"的区别。'
                      '**边界**: 本条只否证那一个形式, 不给替代。'
                      '**v16·B1 的更新**: B1 给 L1 建了一条真实的下游连线 '
                      '(derive_L1_void_to_chain 把 equal_state 张成域态并交给'
                      '阶段二), 故上面的源码扫描**已翻成 True**, 而 L1 的负对照'
                      '也由 B1 的推离守卫补上 (见 _NEG_CTRL_TABLE 的 L1 行)。'
                      '该翻转的含义只到接线存在为止: 本守卫仍登记为 '
                      'expect_pass=False 的**诊断项, 不计分**, 通过率分母不因它'
                      '变化。')

    # v15·W11: W4 新增的 l7_rho_jac 在台账里登记为"只报告、无守卫"。本轮按计划
    # 把它登记为**诊断**, 并把**为什么不能计分**一并写死。
    _rhoA, _rhoB = float(c7['rho_jac']), float(c7['rho_jac_B'])
    _b_res = abs(_rhoB - 1.0)
    record_guard('W11 诊断: l7_rho_jac_B 非恒等 (弱守卫, 登记但不计分)',
                 'selfref_variant_fixpoint',
                 f"A 支 rho_jac = {_rhoA:.10f} (rho_is_identity="
                 f"{c7['rho_is_identity']} -> 恒等, 不可守); "
                 f"B 支 rho_jac_B = {_rhoB:.10f}, |rho_B - 1| = {_b_res:.3e}",
                 bool(_b_res > 1e-9), expect_pass=False,
                 note='**登记但不计分 (弱守卫)**: 计分会往通过率掺水 —— 这一点'
                      '比"有没有守"重要, 所以写死在这里。'
                      '(1) **A 支不可守**: rho_jac(A) 恒等于 1.0, 守它就是守一个'
                      '恒假的命题, 任何参数下都不会通过。'
                      '(2) **B 支可守但弱**: rho_jac_B 非恒等, 说明这一支**有'
                      '分辨力**, 这是 L7 从 D 升 C 的依据; 但它是同一套自指约定'
                      '在特定参数下的产物, 参数一变就可能塌回 1.0, 通过与否**不'
                      '反映物理结论对不对**。把它算进分母, 等于用一个自己挑的'
                      '参数换一个百分点。'
                      '(3) **升级条件 (留给以后)**: 若 B 支出现**参数无关**的'
                      '解析预期 (例如从自指映射的结构本身推出谱半径上界), '
                      '再升级成计分守卫; 在那之前它只是"报告"。')

    print(f"\n  ── v14·F5 一致性检查 (数值路径 <-> 解析路径) ──")
    print(f"      L2 JW 闭合式  max|eigsh-JW| = {c2['max_abs_dev']:.2e} | "
          f"L3 幂迭代残差 max = {c3['max_residual']:.2e} | "
          f"L5 环 Forman max|F| = {c5['cycle_max_abs']:.2e}")
    print(f"      L6 Laplacian 零和 = {c6['lap_zero_rel_max']:.2e}, "
          f"平衡恒等式 = {c6['balance_abs_err_max']:.2e} | "
          f"L7 熵比 {c7['entropy_ratio']:.4f}, 等权度 {c7['equal_weight']:.4f}"
          f"{'' if c7['defined'] else ' (塌缩支, 约定值)'}")
    return measured
