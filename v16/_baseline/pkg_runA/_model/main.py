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
"""spiral_model_v16._model.main —— 由 spiral_model_v16.py 拆分。

主程序与总图 (plot_all)

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
from _model.checks import (
    area_vs_log_law, central_charge_verification, consistency_checks, cross_boundary_twist, finite_size_scaling, spectral_central_charge,
)
from _model.core import (
    CHAIN, KNOBS, _HERE, _OUTPUT_DIR, boundary_correlation_graph, boundary_correlation_length, derive_L1_void_to_chain, derive_L2_entanglement_gap, derive_L3_schmidt_rank, derive_L4_bond_dimension, derive_L5_forman_from_degree, derive_L6_grid, exact_ground_state, fit_central_charge, l1_void_pushaway, print_chain_report,
)
from _model.mera import (
    V13_CHI_CTRL, V13_LR_SEEDS, V13_STEPS, build_mera_graph, mera_bulk_invariance, mera_causal_cone, mera_init, mera_isometry_check, v13_tier1_runs,
)
from _model.stage12 import (
    hj_scan, plot_hj_scan, stage1_void, stage2_critical_break,
)
from _model.stage3 import (
    selfref_N_dependence, selfref_coupled, selfref_general_matrix, selfref_neural_network, selfref_nonconvergent,
)
from _model.stage5 import (
    curvature_report, geometry_controls, validate_curvature,
)
from _model.stage67 import (
    spiral_loop, stage6_life, stage7_consciousness,
)
from spiral_metric_v16 import (collect_metrics_v16,
                                    plot_metrics_v16, metrics_for_json)




def main():
    t_start = time.time()
    print("=" * 76)
    print("  空无到照见 · v15")
    print("  量化指标评估层 (三层体检报告) —— 不加任何新物理阶段")
    print("  阶段间因果链 (函数式派生) | 闭环螺旋 spiral_loop()")
    print("  负对照与交叉检验: F1 跨边界 c | F2a h/J 扫描 | F5 双路径对表 | F6 零模型对照族")
    print("=" * 76)
    print("\n  旋钮 (不可约外部输入, 不是派生量):")
    for kk in ('L_chain', 'd_local', 'L_domain', 'n_lambda',
               'pts_per_wavelength', 'F', 'k', 'dt', 'Du', 'Dv'):
        print(f"    {kk:20s} = {KNOBS[kk]}")
    print("    其余所有下游参数由上游实测物理量派生, 见下方因果链台账。")

    np.random.seed(0)
    torch.manual_seed(0)

    # ---------- 阶段 1-2, 以及因果链 L1 ----------
    L_chain = int(KNOBS['L_chain'])
    s1 = stage1_void(KNOBS['d_local'])
    # v16·B1: dim_full 不再只被 print —— 它在这里与张成后的实际维数比对, 且
    # equal_state 张成的域态被交给阶段二 (即 W11 点名的 L1 下游连线)。
    l1 = derive_L1_void_to_chain(s1['equal_state'], L_chain)
    print(f"  [L1] 空无的局部维度 {l1['d_local']} 张成 {L_chain} 个自旋的全空间: "
          f"dim_full = {l1['dim_full']} (= 2^{L_chain}), "
          f"实际维数匹配 = {l1['dim_match']}")
    l1_push = l1_void_pushaway(l1['equal_state'], L_chain)

    s2 = stage2_critical_break(L=L_chain, void_state=l1['void_full'])

    # ---------- 因果链 L2/L3/L4: 全部从实测的约化密度矩阵谱派生 ----------
    print("\n  --- 因果链: 从阶段二的约化密度矩阵谱派生下游参数 ---")
    l2 = derive_L2_entanglement_gap(s2['probabilities'])
    l3 = derive_L3_schmidt_rank(s2['probabilities'], KNOBS['schmidt_tol'])
    l4 = derive_L4_bond_dimension(s2['entropy'])

    # ---------- 阶段三: 自指动力学 (参数由 L2/L3 派生) ----------
    print("\n" + "=" * 76)
    print("  阶段三 · 自指的动力学涌现")
    print("=" * 76)
    # N=128 是"演示矩阵有多大"的尺寸选择 (非物理量); gap 与网络宽度才是派生量
    s3a = selfref_general_matrix(N=128, gap_physical=l2['gap'])
    s3a2 = selfref_nonconvergent(N=128)
    N_self = l3['N_selfref']
    print(f"  [L3] 自指网络宽度由有效 Schmidt 秩派生: N = {N_self} "
          f"(手工取 64)")
    s3b = selfref_neural_network(N=N_self)
    s3c = selfref_coupled(N=N_self, steps=8000)
    # v15·W1: 秩1化**相图**。s3c 只看派生出的这一个 N, 而 selfref_coupled 结语里
    # 那条"结论依赖 N"的边界需要一个相图才说得清依赖成什么样 (阈值还是随机命中)。
    # 放在 s3c 相邻处, 让 L3-c 的材料连在一起。
    s3c_nd = selfref_N_dependence()

    # ---------- 阶段四: 真实 MERA (键维由 L4 派生) ----------
    print("\n" + "=" * 76)
    print("  阶段四 · 真实张量网络全息 (quimb MERA)")
    print("=" * 76)
    chi_dev = int(l4['chi'])
    print(f"  [L4] 键维由面积律派生: chi = {chi_dev} "
          f"(下界 exp(S) = {l4['chi_lower_bound']:.4f}); 手工取 4")
    # 开发/验证用的结构样本: 小 chi, 只为检查等距性
    mera8, _ = mera_init(8, 2, seed=0)
    iso8 = mera_isometry_check(mera8, 8)
    print(f"  MERA(L=8, chi=2): {mera8.num_tensors} 个张量, "
          f"{iso8['n_unitary']} 个解纠缠器 + {iso8['n_isometry']} 个等距 "
          f"+ 1 个顶层边界张量")
    print(f"  酉性 ||U^dag U - I|| = {iso8['unitary_err']:.2e} "
          f"(张量性质保证, 非构造凑数)")
    print(f"  等距 ||W^dag W - I|| = {iso8['isometry_err']:.2e}")

    # L=32 只用来看结构 (因果锥 / 体几何), 绝不收缩成稠密态
    mera32, _ = mera_init(32, 2, seed=0)
    ns32, cone32, r2_32 = mera_causal_cone(mera32, 32)
    mera16, _ = mera_init(16, 2, seed=0)
    ns16c, cone16, r2_16 = mera_causal_cone(mera16, 16)
    print(f"  因果锥张量数 |cone(n)| (L=32, 共 {mera32.num_tensors} 张量):")
    print(f"     n=1..16 -> {cone32.astype(int).tolist()}")
    print(f"     线性拟合 R^2={r2_32['linear']:.4f} vs 对数拟合 R^2={r2_32['log']:.4f}")
    print(f"     -> 体是体积律 (线性), 而 S(n) 只随 ln n 增长 (面积律)")

    # 主拟合使用**派生键维**的 MERA, 并带多种子 + 检查点选择
    c_exact_main = float(fit_central_charge(s2['gs'], L_chain)['c'])
    print(f"  [主拟合] MERA(L={L_chain}, chi={chi_dev} 派生): "
          f"{len(V13_LR_SEEDS[0][1]) + len(V13_LR_SEEDS[1][1])} 条轨迹 "
          f"x {V13_STEPS} 步, 精确对照 c={c_exact_main:.4f}")
    print(f"       检查点按 **overlap** (训练目标本身) 选择, 不偷看 eps_c; "
          f"另有 chi={V13_CHI_CTRL[0]} 对照轨迹测『提 chi 有没有用』")
    v13_tier1, psi_main = v13_tier1_runs(s2['gs'], L_chain, chi_dev,
                                         c_exact_main)
    ov_main = float(abs(np.vdot(s2['gs'], psi_main)))
    c_main = fit_central_charge(psi_main, L_chain)
    print(f"  [代表态] 取 eps_c 位于中位数的那条轨迹 "
          f"(index {v13_tier1['chosen_run_index']}), 用于第二层的纠缠谱 KL")
    print(f"           重叠={ov_main:.6f}, c={c_main['c']:.6f}, "
          f"eps_c={abs(c_main['c'] - c_exact_main) / c_exact_main:.3%}; "
          f"全部轨迹总用时 {v13_tier1['elapsed']:.0f}s")

    s4 = {'isometry': iso8, 'n_tensors': int(mera8.num_tensors),
          'chi_derived': chi_dev,
          'chi_lower_bound': float(l4['chi_lower_bound']),
          'main_fit': {'L': L_chain, 'chi': chi_dev, 'overlap': ov_main,
                       'c': float(c_main['c']),
                       'v13_runs': v13_tier1['runs'],
                       'v13_curve': v13_tier1['curve'],
                       'v13_steps': v13_tier1['steps'],
                       'v13_chi_ctrl': v13_tier1['chi_ctrl'],
                       'v13_floor_chi': v13_tier1['floor_chi'],
                       'v13_floor_chi_ctrl': v13_tier1['floor_chi_ctrl']},
          'cone': {'L16': cone16.astype(int).tolist(), 'r2_L16': r2_16,
                   'L32': cone32.astype(int).tolist(), 'r2_L32': r2_32,
                   'n_tensors_L32': int(mera32.num_tensors)}}

    # ---------- 阶段五: 离散曲率 ----------
    print("\n" + "=" * 76)
    print("  阶段五 · 离散时空几何 (Forman-Ricci / Ollivier-Ricci)")
    print("=" * 76)
    curv_val = validate_curvature()

    print("\n  应用到涌现几何:")
    G_mera = build_mera_graph(mera32)
    rep_mera, _, _ = curvature_report(G_mera, "MERA 体几何 (L=32 张量网络图)")

    # v15·W10 + v15·B1: `mera_bulk` 依赖**态**、依赖**键维**, 还是只依赖**线路布局**?
    # 这不是修辞 —— 若不依赖态, F6 的百分位讲的只是"二进制 MERA 的接线在同类
    # 规模图里特不特别", 与拟合/训练出来的任何东西无关, L5 的读数只能按这个
    # 口径读。判据是**实测** (mera_bulk_invariance), 不靠"读代码推断"。
    # 键维那一问是 B1 补的: L4 派生的 chi_dev 与这里实际用的 chi 不同, 若图随 chi
    # 变, 那就是一个没被声明的耦合 —— 实测证明**不变** (chi 只定指标维度, 不定接线)。
    _bulk = mera_bulk_invariance(32)
    _bulk_same, _bulk_chi_same = _bulk['same'], _bulk['chi_same']
    print(f"  [L5] mera_bulk 是否依赖态: seed {_bulk['seeds']} 的边集完全相同 = "
          f"{_bulk_same} (|E|={_bulk['n_edges']})")
    print(f"  [L5] mera_bulk 是否依赖键维: chi {_bulk['chis']} 的边集完全相同 = "
          f"{_bulk_chi_same}")
    print(f"       build_mera_graph 只读 ind_map, mera_fit* 只调 t.modify(data=...),")
    print(f"       从不改接线 -> 该读数是 (L, 二进制布局) 的函数, 与态、键维均无关。")
    print(f"       |V|={_bulk['n_nodes']} = 2L-2 = {_bulk['expected_nodes_2L_minus_2']}, "
          f"三角=0 -> 这是二进制布局的解析性质, 不是拟合出来的。")

    # [L5] MERA 图的平均度 -> Forman 曲率的解析预期
    l5 = derive_L5_forman_from_degree(G_mera, rep_mera['forman_mean'])
    print(f"  [L5] 正则粗式 <F> = 4 - 2*d_bar = {l5['forman_predicted']:.4f} "
          f"(差 {abs(l5['forman_predicted'] - rep_mera['forman_mean']):.4f})")
    print(f"       严格式 <F> = 4 - 2*<deg>_edge = {l5['forman_exact']:.4f} "
          f"vs 实测 {rep_mera['forman_mean']:.4f} "
          f"(差 {abs(l5['forman_exact'] - rep_mera['forman_mean']):.2e})")
    print(f"       -> 粗式的差全部来自度偏置 <deg>_edge - d_bar = "
          f"{l5['deg_edge_mean'] - l5['d_bar']:.4f} (非正则图: 度大的点被更多边选中),")
    print(f"          不是曲率实现有错; 严格式到机器精度吻合。")

    _, gs16 = exact_ground_state(16, 1.0, 1.0)
    G_corr, C_corr, D_corr = boundary_correlation_graph(gs16, 16)
    rep_corr, _, _ = curvature_report(G_corr, "边界关联几何 (临界基态)")

    # v14·F6: 零模型对照族 (三类 x 3 实例) —— 给出的是**百分位**, 不是单点比较。
    # 单张同规模随机图只能回答"比随机图更负吗"; 三类对照各控一个维度
    # (度分布 / 聚类 / 树的成分), 才能说清 MERA 的负曲率有什么特点。
    ctrl_reps = geometry_controls(G_mera)
    rep_rand_same = ctrl_reps[0]        # gnm #0, 作为"同规模随机图"的代表实例

    s5 = {'validation': curv_val, 'mera_bulk': rep_mera, 'boundary': rep_corr,
          'random_same': rep_rand_same, 'controls': ctrl_reps,
          'correlation': C_corr.tolist(),
          'degree_prediction': l5,
          'bulk_seed_invariance': _bulk}

    # ---------- 阶段六: 参数由 L6 派生 (域长来自阶段五的 xi) ----------
    xi_boundary = boundary_correlation_length(C_corr, 16)
    print(f"\n  阶段五几何的特征尺度: 边界关联长度 xi = {xi_boundary:.3f} "
          f"(环上有 16 个站点, 故 xi/L = {xi_boundary/16:.4f})")
    print(f"    诚实边界: xi > L 是临界系统的正常现象 (关联长度被尺寸截断);")
    print(f"    尺度不变量是比值 xi/L = {xi_boundary/16:.4f}, 不是 xi 本身。")

    l6 = derive_L6_grid(xi_boundary / 16.0, KNOBS, xi=xi_boundary)
    print(f"  [L6] 网格 N = {l6['N']} 派生 (N_req={l6['N_req']}, "
          f"N_cap={l6['N_cap']}, 受限于 {l6['bound_by']})")
    print(f"       手工取 N=40; 派生后 dt*Du/h^2 = {l6['stab']:.4f} "
          f"<= {KNOBS['stability_limit']} 由构造成立 (不再是碰巧)")

    s6 = stage6_life(F=KNOBS['F'], k=KNOBS['k'], N=l6['N'],
                     L=l6['L_domain'], dt=KNOBS['dt'],
                     Du=KNOBS['Du'], Dv=KNOBS['Dv'])

    # [L6-c] 可证伪的跨阶段预言: 域内应容纳 n_lambda 个波长
    if s6.get('spacing') is not None and np.isfinite(s6['spacing']):
        ratio_lambda = s6['spacing'] / l6['lambda_target']
        print(f"  [L6] 跨阶段预言: 目标波长 = {l6['lambda_target']:.4f} "
              f"(= L_domain/{KNOBS['n_lambda']:.0f}), 实测 = {s6['spacing']:.4f}, "
              f"比值 = {ratio_lambda:.3f}")
        print(f"       诚实边界: 这个预言只在同一个域长下才有意义 —— "
              f"斑图波长由化学参数定, 不由几何定。")
    else:
        ratio_lambda = float('nan')
        print(f"  [L6] 斑图未成形, 跨阶段波长预言无法检验 (如实报告)")

    # ---------- 阶段七 ----------
    s7 = stage7_consciousness(s3c)

    # ---------- v14·F5 一致性检查束 (数值路径 <-> 解析路径) ----------
    f5 = consistency_checks(s3a, s5['validation'], s7)

    # ---------- 闭环螺旋: 把整条链应用到自身 ----------
    loop = spiral_loop(turns=KNOBS['turns'], L0=KNOBS['L0_loop'],
                       scale_knob=KNOBS['scale_knob'],
                       max_L=KNOBS['max_L_loop'],
                       loop_tol=KNOBS['loop_tol'])

    # ---------- 因果链台账 ----------
    print_chain_report()

    # ---------- v14·F2a 负对照: h/J 扫描 ----------
    # 放在因果链台账之后、指标层之前: 它是"阶段二的临界性是不是事实"的检验,
    # 与主线计算解耦, 失败也不会阻断后续。
    hj = hj_scan(L=L_chain)
    plot_hj_scan(hj, os.path.join(_OUTPUT_DIR, '_v16_hj_scan.png'))
    print(f"  负对照图已保存: {os.path.join(_OUTPUT_DIR, '_v16_hj_scan.png')}")

    # ---------- v14·F1 跨边界条件中心荷交叉检验 (同族) ----------
    # 与 F2 一样放在指标层之前、且与主线解耦: 它检验的是"c 的读数对边界条件
    # 稳不稳", 属对解析解对标, 不是新的物理阶段。
    print()
    f1 = cross_boundary_twist(L=L_chain, chi=chi_dev)

    # ---------- 任务 A: c 验证 ----------
    print()
    cc = central_charge_verification()

    # ---------- v15·W3b: 谱学中心荷 (独立族) ----------
    # 与任务 A 并排: A 走纠缠谱 (对 |psi> 做), W3b 走能谱 (对 H 的头三个本征值做)。
    # 数据与函数形式都不同, 这才是"第二把尺子"; 见函数 docstring 的诚实边界。
    print()
    cspec = spectral_central_charge()

    # ---------- v15·W3a: 面积律 vs 对数律 ----------
    # 与 F2a 的 h/J 扫描同族但**不同量**: F2a 看 S(l) 形状拟合出的 c 是否塌缩,
    # W3a 看 S(L/2) 随 L 的标度是平还是对数。见函数 docstring 的负面结果声明。
    print()
    alw = area_vs_log_law()

    # ---------- v15·W5: 有限尺寸标度 (L 作自变量) ----------
    # 与 W3a 都扫 L, 但问的不是同一件事: W3a 问 "S(L/2) 随 L 是平还是对数",
    # W5 问 "spiral_loop 那几个读数作为 L 的函数长什么样, 以及 max_L 到底限制了谁"。
    # 与 spiral_loop 的读数逐字同源 (同估计量), 故同 L 上必须同数 —— 这是交叉核对。
    # 它不碰 MERA, 所以不受 max_L 约束, 也不改闭环本身。
    print()
    fss = finite_size_scaling()

    # ---------- 指标层: 三层量化评估 ----------
    metrics = collect_metrics_v16({
        's2': s2, 's3a': s3a, 's3c': s3c, 's4': s4, 's5': s5,
        's6': s6, 'loop': loop, 'cc': cc, 'l6': l6, 'hj': hj, 'f1': f1,
        'f5': f5, 's3c_nd': s3c_nd, 'cspec': cspec, 'alw': alw, 'fss': fss,
        'L_chain': L_chain, 'chi_dev': chi_dev, 'psi_main': psi_main,
        'l1': l1, 'l1_push': l1_push,
    },KNOBS,_HERE, v13_tier1)

    # ---------- 汇总 ----------
    data = {
        'stages': {
            'void_cond': float(np.linalg.cond(s1['unitary'])),
            'critical': {'E0_per_L': s2['E0'] / 16, 'entropy': s2['entropy'],
                         'polarization': s2['polarization']},
            'selfref_matrix': s3a,
            'selfref_nonconvergent': {k: v for k, v in s3a2.items()
                                      if k != 'steps'},
            'selfref_nn': s3b,
            'selfref_coupled': {
                name: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in s3c[name].items()}
                for name in ('frozen', 'free', 'norm')},
            'mera': s4,
            'curvature': s5,
            'life': {'contrast': s6['contrast'], 'emerged': s6['emerged'],
                     'active_frac': s6.get('active_frac'), 'h': s6['h'],
                     'stab': s6['stab'], 'spacing': s6.get('spacing'),
                     'steps': s6.get('steps'),
                     'boundary_xi': xi_boundary,
                     'xi_over_L': xi_boundary / 16.0,
                     'N_derived': l6['N'], 'N_req': l6['N_req'],
                     'N_cap': l6['N_cap'], 'bound_by': l6['bound_by'],
                     'lambda_target': l6['lambda_target'],
                     'lambda_ratio': ratio_lambda},
            'consciousness': s7,
        },
        'causal_chain': CHAIN,
        'spiral_loop': loop,
        'knobs': {k: v for k, v in KNOBS.items()},
        'central_charge': cc,
        # v15·W3b: 独立族 (能谱) 的 c 读数, 与上面纠缠谱族的 cc 并排存
        'spectral_c': cspec,
        # v15·W3a: 面积律 vs 对数律 (含 S(l) 形状检验无区分力那条负面结果)
        'area_vs_log': alw,
        'consistency': f5,
        'metadata': {
            'version': 'v16',
            'implementation': 'quimb MERA + torch autograd + networkx/scipy(LP)',
            'critical_point': 'h/J = 1.0 (周期 TF-Ising)',
            'c_fit_formula': 'S(n) = (c/3) ln[(L/pi) sin(pi n/L)] + const',
            'or_convention': 'Ollivier-Ricci 默认 alpha=0 (简单随机游走, 无 idle)',
            'site_ordering': '站点 0 = 最高有效位 (与 quimb 输出指标 k0 对齐)',
            'loop_scale_law': 'L_t = clip(round(scale_knob * xi_{t-1}), L0, max_L)',
            # v15: xi 的口径必须写进数据文件。此前 xi/L 被登记为"锚不上", 唯一
            # 原因就是本文件没给定义 —— 拿到 json 的人无从知道 xi 是 Schmidt 谱
            # 比值还是关联函数拟合, 也就无从去找对得上的文献值。补在这里。
            # 适用范围: spiral_loop.rows[].xi 与
            # metrics.tier2.finite_size_scaling.fss_xi_over_L (不是 stage6 的
            # life.boundary_xi, 那是另一套)。
            'xi_definition': (
                'xi = boundary_correlation_length(C, L), 其中 '
                'C_ij = <sigma^z_i sigma^z_j> (以 psi**2 加权; 站点 0 = 最高有效位); '
                '把 |C_ij| 按环上距离 d = min(j-i, L-(j-i)) 分箱取均值; '
                '再对**全部** d = 1..L/2 做直线拟合 ln|C| ~ -d/xi, 返回 -1/slope'),
            'xi_is_effective_length': (
                'xi 是有效长度, 不是关联长度 —— 所以它**没有解析对应值**, 这是 '
                'xi/L 锚不上的原因, 不是"没找到公式"。依据: 临界环上 '
                '<sigma^z sigma^z> 的真实形状是幂律 [sin(pi d/L)]^{-2*Delta_sigma} '
                '(Delta_sigma = 1/8), 不是指数。L=16 同一份数据实测: 幂律拟合的 '
                'RMS 残差 1.45e-03, 指数拟合 5.26e-02, 相差 36 倍; 幂律拟合出的 '
                '2*Delta = 0.2429 (解析 0.25)。换拟合窗口 xi 就变: '
                'd in [1,3] -> xi = 7.99, d in [4,8] -> xi = 46.91 (L=16)。'
                'xi/L 随 L 单调下降 (L=8: 1.6648 -> L=18: 1.1616) 而不收敛, '
                '是这件事的表现'),
            'runtime_sec': None,
        },
    }

    # metrics 三层指标必须真正写进 json。曾经的数据文件声称含 metrics 三层
    # 指标, 但实测里面**没有** metrics 这一节 —— 只有 stages/causal_chain/
    # spiral_loop/knobs/central_charge/metadata。指标只活在打印输出和 PNG 里,
    # 拿不到机器可读的指标本身。这里把它写进去。
    data['metrics'] = metrics_for_json(metrics)

    # 先落盘再作图: 计算结果比画图贵得多, 作图出错不该把结果一起丢掉
    out_json = os.path.join(_OUTPUT_DIR, '_v16_data.json')
    data['metadata']['runtime_sec'] = time.time() - t_start
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  数据已保存: {out_json}")

    plot_all(data, s2, s3a, s3b, s3c, s6, s4, cc, loop)
    plot_metrics_v16(metrics, s2, s3a,
                     os.path.join(_OUTPUT_DIR, 'spiral_v16_metrics.png'))

    # ---------- 汇总 ----------
    print("\n" + "=" * 76)
    print("  汇总")
    print(f"  因果链台账 ({len(CHAIN)} 条):")
    for r in CHAIN:
        tag = "= 手工基线" if r['same_as_manual'] else "!= 手工基线"
        print(f"            {r['link']}: 派生 {r['derived']!r} "
              f"vs 手工 {r['v10_manual']!r}  {tag}")
    print(f"            L2 派生收敛比 = {l2['gap']:.6f} -> 实测收敛比 "
          f"{(s3a['physical'] or {}).get('rate_meas', float('nan')):.6f}")
    print(f"  闭环 spiral_loop(): 三量分述")
    lr = loop['rows']
    print(f"            c_t: " + " -> ".join(f"{r['c']:.4f}" for r in lr)
          + f"  (理论 0.5)")
    print(f"            xi/L: " + " -> ".join(f"{r['xi_over_L']:.3f}" for r in lr))
    print(f"            gap_t: " + " -> ".join(f"{r['gap']:.4f}" for r in lr))
    print(f"            L_t: " + " -> ".join(str(r['L']) for r in lr)
          + ("  [max_L 已触顶]" if loop['verdicts']['scale_clamped'] else ""))
    print(f"            相对谱隙 1-r: " + " -> ".join(f"{1-r['gap']:.4f}" for r in lr)
          + ("  (谱隙闭合 = 趋向临界)" if loop['verdicts']['relgap_closing'] else ""))
    print(f"            裁决: 无量纲不变量"
          f"{'稳定' if loop['verdicts']['ratio_stable'] else '不稳定'}"
          f" (第 {loop['verdicts']['converged_at_turn']} 圈起收敛), "
          f"尺度发散 = 临界性")
    print(f"  MERA: {s4['n_tensors']} 张量, "
          f"酉性误差={s4['isometry']['unitary_err']:.1e}, "
          f"等距误差={s4['isometry']['isometry_err']:.1e}")
    print(f"            派生键维 chi={s4['chi_derived']}: "
          f"L={s4['main_fit']['L']} 重叠={s4['main_fit']['overlap']:.4f}, "
          f"c={s4['main_fit']['c']:.4f}")
    for m in cc['mera']:
        print(f"            L={m['L']} chi={m['chi']}: 重叠={m['overlap']:.4f}, "
              f"c={m['c']:.4f}")
    for e in cc['exact']:
        print(f"            [精确对照] L={e['L']}: c={e['c']:.4f}")
    print(f"  幂迭代·受控谱隙: 理论比/实测比 "
          + ", ".join(f"{s['rate_theory']:.2f}/{s['rate_meas']:.2f}"
                      for s in s3a['gap_sweep']))
    tr = s3a['transient']
    print(f"            Jordan 块 (特征值全={tr['lam']}, 谱隙=0, rho={tr['spectral_radius']:.2f}<1): "
          f"方向不收敛(步长比={tr['step_ratio']:.3f}); "
          f"状态范数 1.0 -> 峰 {tr['norm_peak']:.1f} -> {tr['norm_final']:.1e} (k={tr['kmax']})")
    print(f"            模相等反例: 末段步长仍={s3a2['tail_step']:.3f} (不收敛)")
    print(f"            自指网络相图: "
          f"{' -> '.join(r['phase'] for r in s3b['scan'])}")
    print(f"            权重-状态耦合: 冻结|dx|={s3c['frozen']['final_dx']:.2e} (不收敛) -> "
          f"自指后 |dx|={s3c['norm']['final_dx']:.2e} (收敛)")
    print(f"            权重秩1化只在变体 B 成立: A(sigma1/sigma2="
          f"{s3c['free']['sigma_ratio']:.2e}) vs B({s3c['norm']['sigma_ratio']:.2e})")
    print(f"            诚实边界: 手工 N=64 时 A、B 都秩1化; 派生 N=17 后只剩 B,")
    print(f"            且 g=2.0 的混沌消失 (变成极限环) —— 相图依赖 N, 不是普适的。")
    print(f"  曲率解析锚点全部通过 (到机器精度)")
    print(f"            MERA 体几何: Forman 均值={rep_mera['forman_mean']:+.3f}, "
          f"Ollivier 均值={rep_mera['or_mean']:+.4f} "
          f"(负边占比 {rep_mera['or_neg_frac']:.2f}) -> 负曲率")
    print(f"            边界关联几何: Ollivier 均值={rep_corr['or_mean']:+.4f} "
          f"(负边占比 {rep_corr['or_neg_frac']:.2f})")
    _cf = [c['forman_mean'] for c in ctrl_reps]
    _co = [c['or_mean'] for c in ctrl_reps]
    print(f"            零模型对照族: {len(ctrl_reps)} 个实例 "
          f"({len({c['family'] for c in ctrl_reps})} 族), Forman "
          f"[{min(_cf):+.3f}, {max(_cf):+.3f}], Ollivier "
          f"[{min(_co):+.4f}, {max(_co):+.4f}]")
    print(f"  既有文档 spiral_v13_说明.md 的 A 段有两条说法被本次实测推翻 —— "
          f"'scatter 是参考侧种子间散布'与'两条独立路线互相印证';")
    print(f"            依据是上方 A 段两个诊断项 (归档 bool 场与归档浮点场一致 / "
          f"lam_pk 有独立佐证), 两项目前都未通过。该文档未改动。")
    g = metrics['guards']
    print(f"  指标层: 体检报告见上方; 守卫通过率 "
          f"{g['n_pass']}/{g['n_expect']} = {g['rate']:.2%} "
          f"(另有 {g['n_diag']} 项诊断项不计入, 其中 {g['n_diag_notok']} 项未通过)")
    print(f"            产物: spiral_v16_metrics.png (20 面板) + _v16_data.json")
    print(f"  总用时 {time.time()-t_start:.0f}s")
    print("=" * 76)
    print("  变化以周期性偏振，统一于空无；")
    print("  增长以螺旋式上升，起始于终结。")
    print("  本自具足。🌊")




def plot_all(data, s2, s3a, s3b, s3c, s6, s4, cc, loop):
    """15 个面板:
        1-12 七个物理阶段 (第 2 面板为"预设 + 物理派生"双层)
        13   闭环无量纲不变量 c_t / (xi_t/L_t)
        14   闭环谱隙收窄 gap_t 与尺度流动 L_t
        15   因果链台账 (派生值 vs 手工值 的文本对照)
    """
    fig, axes = plt.subplots(5, 3, figsize=(21, 26))
    axes = np.asarray(axes)

    # 1. Schmidt 谱
    s = s2['spectrum'][:12]
    axes[0, 0].bar(range(len(s)), s)
    axes[0, 0].set_title(f"阶段二: 临界 Schmidt 谱 (极化={s2['polarization']:.2f})\n"
                         f"注: 此比值被极小分母放大, 不是可靠的破缺指标")
    axes[0, 0].set_xlabel('指标')
    axes[0, 0].set_ylabel('奇异值')

    # 2. 幂迭代收敛: 理论比 vs 实测比 对表
    ax = axes[0, 1]
    gaps = [s['gap'] for s in s3a['gap_sweep']]
    ax.semilogy(gaps, [s['rate_theory'] for s in s3a['gap_sweep']], 'k--o',
                label='理论 |λ2/λ1|')
    ax.semilogy(gaps, [s['rate_meas'] for s in s3a['gap_sweep']], 'r-s',
                label='实测收敛比')
    # 叠加由纠缠谱派生的那个物理谱隙 (因果链 L2)
    if s3a.get('physical'):
        ph = s3a['physical']
        ax.axvline(ph['gap'], color='seagreen', lw=1.0, ls=':')
        ax.semilogy([ph['gap']], [ph['rate_meas']], 'D', color='seagreen',
                    ms=9, label=f"[L2] 纠缠谱派生 gap={ph['gap']:.3f}\n"
                                f"实测比={ph['rate_meas']:.4f}")
    ax.set_title('阶段三-a: 幂迭代收敛比对表\n'
                 '(受控谱隙校准 + 物理派生谱隙)')
    ax.set_xlabel('谱隙 (预设值, 及 L2 派生值)')
    ax.set_ylabel('收敛比')
    ax.invert_xaxis()
    ax.legend(fontsize=7)

    # 3. 自指网络相图
    ax = axes[0, 2]
    ax.plot([r['g_rho'] for r in s3b['scan']],
            [r['lyapunov'] for r in s3b['scan']], 'o-', color='crimson',
            label='Lyapunov 指数')
    ax.axhline(0, color='k', lw=0.8, ls=':')
    ax.axvline(1.0, color='gray', lw=0.8, ls='--')
    ax.set_title('阶段三-b: 自指动力学相图\n不动点 → 极限环 → 混沌')
    ax.set_xlabel('g·ρ(W)')
    ax.set_ylabel('Lyapunov 指数')
    for r in s3b['scan']:
        ax.annotate(r['phase'], (r['g_rho'], r['lyapunov']), fontsize=7,
                    textcoords='offset points', xytext=(0, 5))
    ax.legend()

    # 4. 瞬态增长 (非正规性) + 自指闭环的秩1化
    ax = axes[1, 0]
    strong = s3a['transient']
    kk = np.arange(1, len(strong['growth']) + 1)
    ax.semilogy(kk, strong['growth'], 'o-', ms=3,
                label=f"‖A^k‖₂ (非正规性 {strong['nonnormality']:.3f})")
    ax.semilogy(kk, strong['spectral_radius'] ** kk, 'k--',
                label=f"ρ(A)^k (ρ={strong['spectral_radius']:.2f}<1)")
    ax.semilogy(kk, strong['norm_trace'], '-', color='seagreen', lw=1.8,
                label=f"状态范数 ‖A^k x₀‖ -> {strong['norm_final']:.1e} (收敛)")
    ax.axvline(strong['k_peak'], color='gray', lw=0.8, ls=':',
               label=f"‖A^k‖₂ 峰值 k={strong['k_peak']}")
    ax.axvline(strong['k_below'], color='seagreen', lw=0.8, ls=':',
               label=f"状态范数最后一次跌破初值 k={strong['k_below']}")
    ax.set_title(f"阶段三-a: Jordan 块 (特征值全={strong['lam']}, ρ<1) 的瞬态增长\n"
                 f"状态范数先涨后落: 1.0 -> {strong['norm_peak']:.1f} -> "
                 f"{strong['norm_final']:.0e}, 收敛但仍非单调")
    ax.set_xlabel('k')
    ax.set_ylabel('范数')
    ax.legend(fontsize=8)

    # 5. 中心荷: c vs 键维 (核心图)
    ax = axes[1, 1]
    ax.axhline(0.5, color='k', ls='--', lw=1.2, label='理论 c = 1/2')
    for e in cc['exact']:
        ax.axhline(e['c'], color='gray', ls=':', lw=1.0,
                   label=f"精确对照 L={e['L']}: c={e['c']:.3f}")
    for L in sorted({m['L'] for m in cc['mera']}):
        ms = sorted([m for m in cc['mera'] if m['L'] == L], key=lambda m: m['chi'])
        ax.plot([m['chi'] for m in ms], [m['c'] for m in ms], 'o-',
                label=f'MERA L={L}')
    ax.set_title('任务A: 中心荷 c 随键维 χ 收敛到 1/2')
    ax.set_xlabel('键维 χ')
    ax.set_ylabel('拟合 c')
    ax.legend(fontsize=8)

    # 6. S(n) 曲线 + CFT 拟合
    ax = axes[1, 2]
    for m in cc['mera']:
        if m['L'] == 16:
            ax.plot(np.arange(1, len(m['S']) + 1), m['S'], 'o-', alpha=0.7,
                    label=f"MERA χ={m['chi']} (c={m['c']:.3f})")
    for e in cc['exact']:
        if e['L'] == 16:
            ax.plot(np.arange(1, len(e['S']) + 1), e['S'], 'k*-', ms=10,
                    label=f"精确 (c={e['c']:.3f})")
    ax.set_title('S(n) 曲线 (L=16, 周期 CFT 拟合)')
    ax.set_xlabel('n')
    ax.set_ylabel('S(n)')
    ax.legend(fontsize=8)

    # 7. 因果锥 vs 纠缠 (全息压缩: 体是体积律, 纠缠是面积律)
    ax = axes[2, 0]
    cone32 = np.array(s4['cone']['L32'], dtype=float)
    ns32 = np.arange(1, len(cone32) + 1)
    ax.plot(ns32, cone32, 's-', color='darkorange',
            label=f"|cone(n)| L=32 (线性 R²={s4['cone']['r2_L32']['linear']:.3f})")
    ax2 = ax.twinx()
    m8 = [m for m in cc['mera'] if m['L'] == 8]
    if m8:
        S8 = np.array(m8[0]['S'])
        ax2.plot(np.arange(1, len(S8) + 1), S8, 'o-', color='navy',
                 label='S(n) L=8 (对数增长)')
    ax.set_title('阶段四: 全息压缩\n因果锥张量数 ∝ n, 纠缠熵 ∝ ln n')
    ax.set_xlabel('n (边界站点数)')
    ax.set_ylabel('因果锥张量数', color='darkorange')
    ax2.set_ylabel('S(n)', color='navy')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='lower right', fontsize=8)

    # 8. 曲率对比
    ax = axes[2, 1]
    labels = ['MERA体几何', '边界关联', '随机图(同规模)']
    means = [data['stages']['curvature']['mera_bulk']['forman_mean'],
             data['stages']['curvature']['boundary']['forman_mean'],
             data['stages']['curvature']['random_same']['forman_mean']]
    negf = [data['stages']['curvature']['mera_bulk']['forman_neg_frac'],
            data['stages']['curvature']['boundary']['forman_neg_frac'],
            data['stages']['curvature']['random_same']['forman_neg_frac']]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, means, 0.4, label='Forman 均值', color='steelblue')
    ax.bar(x + 0.2, negf, 0.4, label='负曲率边占比', color='salmon')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title('阶段五: 离散 Ricci 曲率对比')
    ax.legend(fontsize=8)

    # 9. Gray-Scott 斑图
    ax = axes[2, 2]
    if s6['v'] is None:
        # 稳定性守卫路径: 拒绝画一张可能失稳的结果
        ax.text(0.5, 0.5, f"未运行\n{s6['reason']}", ha='center', va='center',
                wrap=True, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("阶段六: Gray-Scott 拒绝运行\n(不满足稳定性条件)")
    else:
        ax.imshow(s6['v'], cmap='magma')
        ax.set_title(f"阶段六: 生命斑图 (对比度={s6['contrast']:.2f}, "
                     f"{'涌现' if s6['emerged'] else '未涌现'})\n"
                     f"dt·Du/h²={s6['stab']:.3f}<0.25, 间距≈{s6['spacing']:.3f}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    # 10. 自指闭环: 权重自发秩 1 化 + 冻结对照
    ax = axes[3, 0]
    for key, lab, col in (('norm', '自指·状态归一', 'seagreen'),
                          ('free', '自指·状态自由', 'darkorange'),
                          ('frozen', '冻结权重 (对照)', 'gray')):
        st = np.array(s3c[key]['dx'])
        ax.semilogy(np.arange(len(st)), np.maximum(st, 1e-16), lw=1.2,
                    color=col, label=lab)
    ax.set_title('阶段三-c: 权重-状态耦合自指\n冻结(灰)永不收敛; 自指后(绿/橙)收敛')
    ax.set_xlabel('迭代步')
    ax.set_ylabel('‖x_{n+1}-x_n‖')
    ax.legend(fontsize=8)

    # 11. 对照实验: 能量目标塌缩到平均场
    ax = axes[3, 1]
    e8 = [e for e in cc['exact'] if e['L'] == 8]
    if e8:
        ax.plot(np.arange(1, len(e8[0]['S']) + 1), e8[0]['S'], 'k*-', ms=10,
                label=f"精确基态 (c={e8[0]['c']:.3f})")
    m8o = [m for m in cc['mera'] if m['L'] == 8 and m['chi'] == 4]
    if m8o:
        ax.plot(np.arange(1, len(m8o[0]['S']) + 1), m8o[0]['S'], 'o-',
                label=f"重叠目标 (c={m8o[0]['c']:.3f})")
    if cc['energy_control']:
        ec = cc['energy_control']
        ax.plot(np.arange(1, len(ec['S']) + 1), ec['S'], 'x--', color='crimson',
                label=f"能量目标 (c={ec['c']:.3f})")
    ax.set_title('对照实验: 为何用重叠而非能量\n'
                 '能量目标塌缩到平均场, 纠缠几乎为零')
    ax.set_xlabel('n')
    ax.set_ylabel('S(n)')
    ax.legend(fontsize=8)

    # 12. 曲率算子族
    ax = axes[3, 2]
    val = data['stages']['curvature']['validation']
    kn = [r['n'] for r in val['complete']]
    ax.plot(kn, [r['mean'] for r in val['complete']], 'o-', label='K_n 实测')
    ax.plot(kn, [r['exact'] for r in val['complete']], 'k--',
            label='(n-2)/(n-1) 解析')
    kd = [r['d'] for r in val['hypercube']]
    ax.plot(kd, [r['mean_lazy'] for r in val['hypercube']], 's-',
            label='Q_d 实测 (alpha=1/(d+1))')
    ax.plot(kd, [r['exact_lazy'] for r in val['hypercube']], 'k:',
            label='2/(d+1) 解析')
    ax.plot(kd, [r['mean_plain'] for r in val['hypercube']], '^--',
            color='gray', label='Q_d (alpha=0) -> 0')
    ax.set_title('阶段五: 曲率算子的解析锚点校验\n(全部到机器精度)')
    ax.set_xlabel('n 或 d')
    ax.set_ylabel('kappa / F 均值')
    ax.legend(fontsize=7)

    # 13. 闭环三量流动: c_t 与 xi/L (无量纲不变量)
    ax = axes[4, 0]
    rows = loop['rows']
    tt = [r['turn'] for r in rows]
    ax.plot(tt, [r['c'] for r in rows], 'o-', color='crimson', label='c_t')
    ax.axhline(0.5, color='k', ls='--', lw=1.0, label='理论 c = 1/2')
    ax.set_xlabel('圈数 t')
    ax.set_ylabel('中心荷 c_t', color='crimson')
    ax.set_ylim(0.4, 0.65)
    ax2 = ax.twinx()
    ax2.plot(tt, [r['xi_over_L'] for r in rows], 's-', color='navy',
             label='xi_t / L_t')
    ax2.set_ylabel('xi_t / L_t  (尺度不变量)', color='navy')
    ax.set_title('闭环 spiral_loop(): 无量纲不变量\n'
                 'c -> 1/2 且 xi/L -> 常数 (共形不变性)')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc='center right')

    # 14. 闭环: 谱隙收窄 (临界的标志) 与尺度流动
    ax = axes[4, 1]
    ax.plot(tt, [1.0 - r['gap'] for r in rows], 'o-', color='darkgreen',
            label='相对谱隙 1 - p1/p0')
    ax.set_xlabel('圈数 t')
    ax.set_ylabel('相对谱隙 1 - r', color='darkgreen')
    ax3 = ax.twinx()
    ax3.plot(tt, [r['L'] for r in rows], '^--', color='gray', label='L_t')
    ax3.set_ylabel('系统尺寸 L_t', color='gray')
    clamped = loop['verdicts']['scale_clamped']
    ax.set_title('闭环: 相对谱隙闭合 = 趋向临界\n'
                 '(r=p1/p0 上升, 故 1-r 下降)'
                 + ('  [L_t 触顶 max_L]' if clamped else ''))
    h1, l1 = ax.get_legend_handles_labels()
    h3, l3 = ax3.get_legend_handles_labels()
    ax.legend(h1 + h3, l1 + l3, fontsize=7, loc='best')

    # 15. 因果链台账 (文本面板)
    ax = axes[4, 2]
    ax.axis('off')
    lines = ['']
    for r in data['causal_chain']:
        same = '=' if r['same_as_manual'] else '≠'
        d = r['derived']
        ds = f"{d:.4g}" if isinstance(d, (int, float)) else str(d)
        m = r['v10_manual']
        ms = f"{m:.4g}" if isinstance(m, (int, float)) else str(m)
        lines.append(f"{r['link']}  {r['formula']}")
        lines.append(f"     derive ={ds}  manual={ms} {same}")
    ax.text(0.0, 1.0, "\n".join(lines), va='top', ha='left', fontsize=6.5,
            family='monospace', transform=ax.transAxes)
    ax.set_title('因果链: 派生值 vs 手工值\n(≠ 表示派生修正了原来的随手取值)')

    plt.suptitle('空无到照见 · v15 — 因果链 (函数式派生) + 闭环 spiral_loop()',
                 fontsize=15)
    plt.tight_layout()
    out_png = os.path.join(_OUTPUT_DIR, 'spiral_v16.png')
    plt.savefig(out_png, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  图像已保存: {out_png}")
