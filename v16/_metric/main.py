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
"""spiral_metric_v16._metric.main —— 由 spiral_metric_v16.py 拆分。

编排入口 collect_metrics_v16 + 20 面板作图 plot_metrics_v16

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
from _metric.extras import (
    metric_area_vs_log_law, metric_finite_size_scaling, metric_geometry_robustness, metric_l1_void_link, metric_negative_control_ledger, metric_selfref_N_dependence, metric_spectral_central_charge,
)
from _metric.layer1 import (
    metric_budget_diagnostic, metric_central_charge_error_seeds, metric_chi_bottleneck, metric_conformal_invariance_seeds, metric_design_robustness,
)
from _metric.negctrl import (
    metric_consistency_checks, metric_f1_cross_boundary, metric_hj_negative_control,
)
from _metric.registry import (
    EPS_T, guard_pass_rate, print_health_report, record_guard,
)
from _metric.solvers import (
    boundary_correlation_graph, exact_ground_state, fit_central_charge, metric_chain_rmse, metric_correlation_exponent, metric_entropy_kl, metric_gap_closure, metric_jordan_peak, metric_mera_consistency, mmi_tripartite,
)
from _metric.tier3 import tier3_structural_ladder
from _metric.tier3_desc import _log_bin




def collect_metrics_v16(ctx, KNOBS, _HERE, v13):
    """
    体检报告编排。

    与 spiral_metric.collect_metrics 的关系:
      - **原样复用** 1.3 谱隙闭合 / 1.4 MERA 一致性 / 第二层全部四项 —— 它们的
        物理前提没变, 定义和阈值一个都没动。
      - **替换** 1.1 / 1.2 为多种子分布版, 并新增 1.5 全程稳健性与两条诊断项。
      - **替换** 第三层为 tier3_structural_ladder (2D 斑图结构对结构)。
      - 守卫登记块是那一段的**逐字副本**。之所以复制而不抽出共用函数:
        改动 spiral_metric.py 会让已发布的历史数字不可复现。代价是两份可能
        分歧 —— 两个文件都是冻结快照, 所以这个代价是可接受的; 但读者必须
        知道它存在。

    v13 (参数名保持不变以维持兼容) =
        {'runs': [...], 'curve': [...], 'steps': int, 'lrs': [...],
         'chi_sweep': {chi: {'eps_c':..., 'R_conf':..., 'overlap':...}},
         'floor_chi': float}
    """
    s2, s3a, s3c = ctx['s2'], ctx['s3a'], ctx['s3c']
    s4, s5, s6, loop = ctx['s4'], ctx['s5'], ctx['s6'], ctx['loop']
    l6, L_chain, chi_dev = ctx['l6'], ctx['L_chain'], ctx['chi_dev']
    psi_main, gs_exact = ctx['psi_main'], s2['gs']

    runs = v13['runs']
    print("\n" + "=" * 76)
    print("  量化指标层 (体检报告)")
    print("=" * 76)
    print("  第一层: 1.1/1.2 用多种子分布; 1.5 全程稳健性 (最坏轨迹)")
    print("  第三层: 3D CDM 场对场 -> 2D 反应-扩散 结构对结构 (方法学展示)")
    print(f"  预算: {v13['steps']} 步 x {len(runs)} 条轨迹 "
          f"(lr={' / '.join(str(x) for x in v13['lrs'])}, chi={chi_dev})")
    print("  原则: 目标阈值 eps_c < 5%, R_conf < 1e-2, **未放宽**。")

    # ---------------- 守卫: 先登记 (既有登记块的逐字副本) ----------------
    print("\n  ── 守卫登记 (可证伪, 汇成第一层最后一项的通过率) ──")
    record_guard('G1 L2 谱退化', 'derive_L2_entanglement_gap',
                 f"非零 Schmidt 值个数 {len(s2['probabilities'])} >= 2",
                 len(s2['probabilities']) >= 2, expect_pass=True,
                 note='此处原是硬 raise; 现补成可读的判定。')
    record_guard('G2 L4 纠缠熵为正', 'derive_L4_bond_dimension',
                 f"S(L/2) = {s2['entropy']:.6f} > 0",
                 float(s2['entropy']) > 0, note='此处原是硬 raise。')
    record_guard('G3 L6 网格足够', 'derive_L6_grid',
                 f"N = {int(l6['N'])} >= 8", int(l6['N']) >= 8,
                 note=f"此处原是硬 raise; 派生 N={l6['N']} "
                      f"(N_req={l6['N_req']}, N_cap={l6['N_cap']})。")
    record_guard('G4 Gray-Scott 数值稳定', 'stage6_life',
                 f"dt*Du/h^2 = {s6['stab']:.4f} <= {KNOBS['stability_limit']}",
                 bool(s6.get('ok')) and s6['stab'] <= KNOBS['stability_limit'],
                 note='失稳不会给出"更剧烈的斑图", 只会给出棋盘状假斑图。')

    vd = loop['verdicts']
    record_guard('V1 中心荷趋于 1/2', 'spiral_loop',
                 f"c_last = {loop['rows'][-1]['c']:.4f}, |c-0.5| < 0.05",
                 vd['c_toward_half'])
    record_guard('V2 尺度比 xi/L 稳定', 'spiral_loop',
                 f"ptp(xi/L) < 15% * mean  (末圈 xi/L = "
                 f"{loop['rows'][-1]['xi_over_L']:.4f})", vd['ratio_stable'],
                 note='只有无量纲量才配当收敛判据 —— xi 的绝对值随 L 发散是'
                      '临界的定义, 不是失败。')
    record_guard('V3 相对谱隙闭合', 'spiral_loop',
                 f"1-r: {1 - loop['rows'][0]['gap']:.4f} -> "
                 f"{1 - loop['rows'][-1]['gap']:.4f} (下降)",
                 vd['relgap_closing'])
    record_guard('V4 尺度被 max_L 截断', 'spiral_loop',
                 f"L 触顶 {KNOBS['max_L_loop']}", vd['scale_clamped'],
                 note='诚实为真: 去掉上限 L 会一直涨 —— 那正是临界性, 不是闭环失败。')
    record_guard('V5 无量纲量逐圈收敛', 'spiral_loop',
                 f"第 {vd['converged_at_turn']} 圈起变化 < {KNOBS['loop_tol']:.0%}",
                 vd['converged_at_turn'] is not None)
    record_guard('V6 权重自由变体秩1化', 'selfref_coupled',
                 f"变体A sigma1/sigma2 = {s3c['free']['sigma_ratio']:.3e} > 1e3",
                 bool(s3c['free']['sigma_ratio'] > 1e3), expect_pass=False,
                 note='**设计上就该失败**: 派生 N 之后变体 A 收敛到 x*=0, '
                      '权重失去方向。这是诊断项, 不计入通过率 —— 否则会假装'
                      '把一条真实的负面结果算成"没通过"。')
    record_guard('V7 权重归一变体秩1化', 'selfref_coupled',
                 f"变体B sigma1/sigma2 = {s3c['norm']['sigma_ratio']:.3e} > 1e3",
                 bool(s3c['norm']['sigma_ratio'] > 1e3))
    record_guard('V8 生命斑图涌现', 'stage6_life',
                 f"对比度 {s6['contrast']:.3f} > 0.2 且活化面积 "
                 f"{s6['active_frac']:.3f} > 0.05", bool(s6['emerged']))

    curv_val = s5['validation']
    c_max = 0.0
    for r in curv_val['cyclic']:
        c_max = max(c_max, float(r['max_abs']))
    for r in curv_val['complete']:
        c_max = max(c_max, abs(float(r['mean']) - float(r['exact'])))
    for r in curv_val['hypercube']:
        c_max = max(c_max, abs(float(r['mean_plain']) - float(r['exact_plain'])),
                    abs(float(r['mean_lazy']) - float(r['exact_lazy'])))
    cf = curv_val['forman']
    c_max = max(c_max, abs(float(cf['path_interior'])),
                abs(float(cf['k4_no_tri']) + 2.0),
                abs(float(cf['regular3_mean']) + 2.0),
                abs(float(cf['cycle_max_abs'])))
    record_guard('G5 曲率解析锚点', 'validate_curvature',
                 f"max|实测 - 解析| = {c_max:.2e} < 1e-6", c_max < 1e-6,
                 note='这条曾只把 max|.| 打印出来, 没有任何判定 —— '
                      '这是一个"有数字无判定"的洞, 现补成 bool。')

    # 新增守卫: 结论的稳健性前提
    record_guard('G6 最坏轨迹也达标', 'collect_metrics_v16',
                 f"{len(runs)} 条轨迹 max eps_c = "
                 f"{max(r['eps_c'] for r in runs):.3%} < {EPS_T:.0%}",
                 max(r['eps_c'] for r in runs) < EPS_T,
                 note='若这条为假, 1.1 的"达标"只是在某个种子上运气好。',
                 # 指认 1.1 (v16.1·C-1)。判据 `max(r['eps_c']) < EPS_T` 与 1.1 的
                 # `ok = bool(hi < EPS_T)` 是**同一条不等式**、同一份 runs ⇒
                 # **同源**: 这是接线, 不是独立佐证。登记见 audit §6.4。
                 tests_claims=('1.1',))
    record_guard('G7 最优检查点非首步',
                 'mera_fit_v13(overlap 选择)',
                 f"全部轨迹的选定 step > 0",
                 all(r['chosen_step'] > 0 for r in runs),
                 note='最佳检查点选择必须真的移动过; 若全停在 step 0, '
                      '说明选择逻辑坏掉, 报出的会是初态的随机值。')

    # v16.2·L4 新增: 互信息的单调性 (MMI) —— 「存在几何对偶」的**必要**条件。
    # 与 G2 (`S(L/2) > 0`, 恒真) 的区别: 本条**真能失败** (GHZ 即反例), 故它是判据
    # 而不是恒真式; 也不能用强次可加性 SSA 代替 —— SSA 是定理, 物理上不可能失败。
    # expect_pass=False 的依据是 v16.2·S2 冒烟的**实测符号**, 不是预设:
    #   L=16 上 I3 > 0 全部 w (趋势 `+0.0357`, 远离 0) ⇒ 负结果 ⇒ 照登, 不计入通过率。
    _mmi = mmi_tripartite(gs_exact, L_chain)
    record_guard('G8 L4 几何对偶必要条件 (MMI)', 'collect_metrics_v16',
                 f"I3(w) = {[round(float(v), 6) for v in _mmi['i3']]} "
                 f"(w=1..4) 全部 <= 0",
                 bool(_mmi['all_le_zero']), expect_pass=False,
                 note='**按 L=16 的实测登记为负, 不是待办**: I3 > 0 全部 w, 趋势 '
                      f"{_mmi['trend']:+.4f} (远离 0)。成因是**尺度** —— "
                      'xi ~ 19.3 > L = 16 (`spiral_model_v16.py:274/309-310`), '
                      '整条环落在一个关联长度内, 此处的"分离区域"是格点尺度的。'
                      '**不等于「RT 被证否」**; 反过来说, 即便 I3 <= 0 也**只是必要条件**'
                      '—— 通过 != RT 被验证, 更 != 涌现时空 (B2 硬边界 '
                      '`spiral_model_v16.py:52-53`)。成熟度不动: L4 仍是 C。'
                      'MMI 与 SSA 不可互换引用: 后者是**自校验**, 前者才是判据。')

    # ---------------- 第一层 ----------------
    print("\n  ── 第一层 · 内部自洽性 ──")
    fit_exact = fit_central_charge(gs_exact, L_chain)
    m11 = metric_central_charge_error_seeds(runs, chi_dev, L_chain)
    m12 = metric_conformal_invariance_seeds(
        runs, fit_exact['rms'] / float(np.mean(fit_exact['S'])))
    m13 = metric_gap_closure(loop['rows'])
    m14 = metric_mera_consistency(s4['isometry'], s4['cone']['r2_L32'])
    m15 = metric_design_robustness(runs, chi_dev, L_chain)
    m1d = metric_budget_diagnostic(v13['curve'], v13['steps'], chi_dev)
    m1e = metric_chi_bottleneck(v13['chi_sweep'], chi_dev, v13['floor_chi'])

    # v15·W12: L4 键维**向下**对照 —— 面积律下界的「推离」(kind='推离')。
    # 这是 L4 台账上**唯一**的负对照: 把 chi 压到面积律下界 exp(S) 以下, 要求下游
    # (MERA 复现中心荷 c) **正确失败**。既有的 V13_CHI_CTRL=(8,0) 是向上的,
    # 不回答"这条下界约不约束下游"这个问题 —— 向下才是能证伪它的方向。
    _cd = v13.get('chi_down')
    _e_up = float(v13['chi_sweep'][chi_dev]['eps_c'])
    if _cd:
        _ratio = float(_cd['eps_c']) / _e_up if _e_up > 0 else float('inf')
        _deg, _ok = bool(_ratio > 1.5), True
        _cond = (f"chi={_cd['chi']} (界下, 2 的幂里严格小于面积律下界 "
                 f"exp(S)={s4['chi_lower_bound']:.4f}) eps_c={_cd['eps_c']:.3%} "
                 f"vs chi={chi_dev} (界上, 派生值) eps_c={_e_up:.3%} "
                 f"-> 退化 {_ratio:.2f}x (判据 > 1.5x)")
    else:
        _deg, _ok = False, False
        _cond = (f"**未构造**: chi={chi_dev} 已是最小合法键维 (MERA 要求 2 的幂), "
                 f"面积律下界 exp(S)={s4['chi_lower_bound']:.4f} 之下没有可用格点")
    record_guard('L4 键维向下对照: 压到面积律下界以下下游退化', 'v13_tier1_runs',
                 _cond, bool(_deg and _ok),
                 note='**L4 台账上唯一的负对照 (kind=推离)**。'
                      '向上那条 (chi=8) 只回答"提 chi 有没有用"; 向下这条才回答'
                      '"面积律下界到底约不约束下游"。单变量: 同种子/同 lr/同预算, '
                      '只改 chi。**诚实边界 (必须连着读)**: '
                      '(1) 只验**必要性**方向 —— 界不成立时下游退化; **不验充分性**, '
                      '而 derive_L4_bond_dimension 的 docstring 本就写它是'
                      '"必要非充分条件"。(2) chi 只有两个合法取值 (MERA 要 2 的幂), '
                      '界下那格就是 chi=2, 本质只有一个自由度, 不是能扫的族。'
                      '(3) 判据用 >1.5x 的量级比较, **不引用 Schmidt 截断线** —— '
                      '那条线不是严格下界 (实测 chi=2 时它给 61.8%, 而 MERA 拿到 '
                      '19.6%, 比它还低 3 倍), 拿它当标尺会得出反向结论。'
                      '(4) 若 chi_dev 本身就是 2 (低纠缠区, exp(S) <= 2), 界下'
                      '不存在合法格点 (MERA 最小键维就是 2), 本守卫**如实判失败** '
                      '—— 在那个区域里面积律给的是平凡界, 这条派生不携带信息。'
                      '主流程调用点 (L=16, h/J=1.0) 实测 chi_dev=4, 不落在这个区。')
    print(f"      中心荷误差 {m11['median']:.4f} [{m11['min']:.4f}, {m11['max']:.4f}] "
          f"| 共形不变性 {m12['median']:.3e} [{m12['min']:.3e}, {m12['max']:.3e}] "
          f"(精确态地板 {m12['R_conf_exact']:.3e})")
    print(f"      谱隙闭合残差 {m13['resid']:.2e} (a={m13['a']:.4f}) "
          f"| MERA 一致性 err={m14['err']:.2e}, R²_lin={m14['r2_linear']:.4f}")
    print(f"      全程稳健性 max eps_c={m15['max_eps_c']:.3%}, "
          f"max R_conf={m15['max_R_conf']:.3e} ({m15['n_runs']} 条轨迹)")

    # ---------------- 第二层 (口径原样) ----------------
    print("\n  ── 第二层 · 微观物理对标 (与解析解硬连接) ──")
    m21 = metric_chain_rmse(s3a, s3a['physical'])
    m22 = metric_entropy_kl(psi_main, gs_exact, L_chain)
    m23 = metric_correlation_exponent(gs_exact, L_chain)
    m24 = metric_jordan_peak(s3a['transient'])
    print(f"      因果链 RMSE {m21['rmse']:.3e} | 纠缠谱 KL {m22['kl']:.3e} "
          f"(JS {m22['js']:.3e}) | 关联指数 2Δ={m23['two_delta']:.4f} "
          f"(精确 0.25, 朴素幂律 {m23['eta_naive']:.4f}) | Jordan 峰值 "
          f"k={m24['k_meas']} vs {m24['k_theory']:.2f}")

    # ---------------- v14·F2a 负对照 (h/J 扫描) ----------------
    # 放在第二层: 它检验的是"阶段二的临界性是不是物理事实", 属于对解析解对标。
    m25 = metric_hj_negative_control(ctx.get('hj'))

    # ---------------- v14·F1 跨边界条件中心荷 (同族) ----------------
    # 也放在第二层: 与 F2 同属"对解析解对标", 检验的是读数的稳健性。
    m26 = metric_f1_cross_boundary(ctx.get('f1'))

    # ---------------- v14·F5 一致性检查束 (实现 <-> 解析式) ----------------
    # 也放在第二层: 与 F1/F2 同属"对标解析物", 但对标的是**解析式**而非物理真值。
    m27 = metric_consistency_checks(ctx.get('f5'))

    # ---------------- v14·F6 涌现几何的对照鲁棒性 ----------------
    # 也放在第二层: 它的参照物是三类解析定义的零模型族, 不是实验数据。
    m28 = metric_geometry_robustness(ctx.get('s5'))

    # ---------------- v15·W1 自指变体秩1化的 N 依赖 (相图) ----------------
    # 也放在第二层: 它标定的是"环节七那条结语到底依赖什么", 属于把
    # 一句定性断言换成一个可复现的相图, 不对标外部数据。
    m29 = metric_selfref_N_dependence(ctx.get('s3c_nd'))

    # ---------------- v15·W3b 谱学中心荷 (独立族) ----------------
    # 也放在第二层: 它的参照物是 Ising CFT 的能谱 (解析物), 不是实验数据。
    m30 = metric_spectral_central_charge(ctx.get('cspec'))

    # ---------------- v15·W3a 面积律 vs 对数律 ----------------
    # 也放在第二层: 参照物是 1D 有隙/临界的两种标准标度行为。
    m31 = metric_area_vs_log_law(ctx.get('alw'))

    # ---------------- v15·W5 有限尺寸标度 (L 作自变量) ----------------
    # 也放在第二层: 参照物是 Ising CFT 的 c=1/2 与临界 TFIM 的解析 E0/L=-4/pi。
    # 传 loop 进去是为了做"同 L 同数"的分歧检测 (两条路径写了两遍同一套估计量)。
    m32 = metric_finite_size_scaling(ctx.get('fss'), ctx.get('loop'))

    # ---------------- v16·B1 L1 空无 -> 链的下游连线 ----------------
    # 必须排在 m33 (负对照台账) **之前**: 台账要拿 GUARDS 真实注册表对表,
    # 排在后面就会漏掉 B1 新注册的这三条守卫。
    m34 = metric_l1_void_link(ctx['l1'], ctx['l1_push'])

    # ---------------- 第三层 ----------------
    m3 = tier3_structural_ladder(s6, os.path.join(_HERE, 'data'),
                                 model_params={'F': KNOBS['F'],
                                               'k': KNOBS['k']})

    # ---------------- v15·W6 负对照台账 (逐层) ----------------
    # **必须放在最后**: 它要拿 GUARDS 这个真实注册表对表, 而 A 段/T3 的守卫
    # 是在上面 tier3_structural_ladder 里才注册的。放到前面会漏掉一整批,
    # 于是"台账与注册表一致"这条守卫就退化成了自说自话。
    m33 = metric_negative_control_ledger()

    # ---------------- 体检报告 ----------------
    print_health_report()
    return {'tier1': {'central_charge_error': m11, 'conformal_invariance': m12,
                      'gap_closure': m13, 'mera_consistency': m14,
                      'design_robustness': m15,
                      'l1_void_link': m34,
                      'budget_diagnostic': m1d, 'chi_bottleneck': m1e,
                      'curve': v13['curve'],
                      'curves_by_seed': v13.get('curves_by_seed', {}),
                      'budget': {'steps': v13['steps'], 'lrs': v13['lrs'],
                                 'chi': chi_dev,
                                 'floor_chi': v13['floor_chi'],
                                 'floor_chi_ctrl': v13['floor_chi_ctrl'],
                                 'chi_ctrl': v13['chi_ctrl']},
                      'runs_summary': [
                          {k: r[k] for k in ('lr', 'seed', 'eps_c', 'R_conf',
                                             'overlap', 'chosen_step')}
                          for r in runs]},
            'tier2': {'chain_rmse': m21, 'entropy_kl': m22,
                      'correlation_exponent': m23, 'jordan_peak': m24,
                      'hj_negative_control': m25,
                      'f1_cross_boundary': m26,
                      'consistency_checks': m27,
                      'geometry_robustness': m28,
                      'selfref_N_dependence': m29,
                      'spectral_central_charge': m30,
                      'area_vs_log_law': m31,
                      'finite_size_scaling': m32,
                      'negative_control_ledger': m33},
            'tier3': m3,
            'guards': guard_pass_rate()}




# ===========================================================================
# 指标层的 20 面板图
# ===========================================================================
def plot_metrics_v16(metrics, s2, s3a, out_png):
    """
    指标层专属的 20 面板图, 与 15 面板物理图完全分离。

    既有 plot_metrics 读 tier1[...]['eps_c'] 与 tier3['pk'] —— 本层的指标
    结构变了 (1.1/1.2 变成分布, 第三层换了物理对象), 所以那 4 个面板在新结构下
    会直接抛 KeyError。这里不是"顺手重画", 是本层真的需要自己的面板。

    为什么从 9 格加到 20 格:
      原先是 3x3 的 9 格 (第一层 1 + 第二层 3 + 第三层 4 + 守卫 1)。本层第一层
      自己就要 4 格 (1.1/1.2 分布、1.6 预算漂移、1.7 chi 归因), 照 9 格的预算,
      第二层和第三层各让出两格 —— 结果 2.3 关联衰减指数与 2.4 Jordan 块
      被挤掉了。那两张图**各自是一条独立断言的唯一可视化**, 而且第二层一字未改,
      正是"没有回归"的证据。所以正确的做法是加格子而不是砍:

       1  第一层达标度条形 (5 项, 含「全程稳健性」)
       2  收敛曲线 eps_c(step): 各 seed + 中位数 + 600 步位置       <- 核心新证据
       3  多轨迹分布 vs 目标线 + 单点基线                            <- 核心新证据
       4  chi 归因: 实测 eps_c(chi) vs Schmidt 截断参照
       --- 第二层 (原样保留, 4/4) ---
       5  因果链拟合 RMSE: 理论 |lam2/lam1| vs 实测收敛比
       6  纠缠谱 (MERA 代表态 vs 精确基态)
       7  关联衰减指数: 共形形式 vs 朴素幂律
       8  Jordan 块瞬态增长: 谱半径<1 但范数先涨
       --- 第三层 (降级为方法学展示) ---
       9  A 段: 描述子逐项距离 vs 种子间散布 (噪声底)
      10  A 段: 归一化谱形 (k/k_pk) 模型 vs CLIP 三种子
      11  A 段: 两个混杂因子 (波长数 / 相区参数) —— 为什么距离不能被归因
      12  B1 段: Southampton 波峰周期与 CV
      13  B1 段: 独立交叉检验 —— 强度序列自相关在四个滞后上的系数
      14  B2 段: Reading 功率谱 —— "未检出"这个结论所依据的谱与阈值
      15  B2 段: Reading 逐帧亮度 —— 谱为什么是红的 (慢漂移主导)
      16  守卫通过率
       --- A 段参考侧实际场 ---
      17  A 段参考侧 seed 0 的实际场
      18  A 段参考侧 seed 1 的实际场   (自建种子家族, 少数相恒为 0.5)
      19  A 段参考侧 seed 2 的实际场
      20  A 段模型侧 36² 的实际场 (对照)

    13~15 三格把"报一个 verdict"变成"把判决依据亮出来"。一个"未检出周期"的
    结论, 读者有权看到那条谱和那条阈值线自己复核。

    17~20 四格把参考侧的实际场摆出来, 用途是让读者**看见**参考家族是不是斑图。
    A1 曾以这 3 个种子的 lam_pk 中位数为据声称"两个盒子内波长数差一个量级",
    实测那条论断来自坏参考场的谱角点伪像, 已撤回 —— 把场摆出来, 这个缺口
    本是可见的, 只是当时没往归档管线上想。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    def _short(s, n=20):
        s = str(s).replace('_', ' ')
        return s if len(s) <= n else s[:n - 1] + '…'

    # 网格按"每条独立断言至少占一格"排: 第一层 4 格 (分布口径), 第二层含
    # 2.3/2.4 —— 那两张图各自是一条独立断言的唯一可视化 (2.3 是"共形形式 vs
    # 朴素幂律"的对照, 2.4 是"谱半径<1 但范数先涨"这个非正规性陷阱), 二者
    # 一旦被别的面板挤掉, 对应的断言就只剩一个数字。第二层**一字未改**, 正是
    # "没有回归"的证据。所以不是砍掉它们, 是把格子加大。
    fig, axes = plt.subplots(5, 4, figsize=(25, 25))  # 5x4: 含 A 段第 5 行
    axes = np.asarray(axes)
    t1 = metrics['tier1']
    bud = t1['budget']

    # ---- 1. 第一层达标度条形 ----
    ax = axes[0, 0]
    dr = t1['design_robustness']
    items = [
        ('中心荷误差\n(最坏轨迹)', t1['central_charge_error']['max'], 5e-2,
         t1['central_charge_error']['passed']),
        ('共形不变性\n(最坏轨迹)', t1['conformal_invariance']['max'], 1e-2,
         t1['conformal_invariance']['passed']),
        ('全程稳健性\n(1.1 且 1.2 同时)', max(dr['max_eps_c'] / 5e-2,
                                        dr['max_R_conf'] / 1e-2), 1.0,
         dr['passed']),
        ('MERA 一致性', t1['mera_consistency']['err'], 1e-12,
         t1['mera_consistency']['passed']),
        ('谱隙闭合残差', t1['gap_closure']['resid'], 5e-3,
         t1['gap_closure']['passed']),
    ]
    names = [i[0] for i in items]
    ratios = [float(i[1]) / i[2] for i in items]
    cols = ['seagreen' if i[3] else 'crimson' for i in items]
    ax.barh(names, ratios, color=cols)
    ax.axvline(1.0, color='k', ls='--', lw=1.2)
    ax.set_xscale('log')
    ax.set_xlabel('实测 / 目标  (对数轴; < 1 即达标)')
    ax.set_title('一 · 内部自洽性\n条形长度 = 实测值相对目标; 前两项取最坏轨迹')
    for i, r in enumerate(ratios):
        ax.text(r, i, f"  {r:.2g}x", va='center', fontsize=8)

    # ---- 2. 收敛曲线 ----
    ax = axes[0, 1]
    for seed, cv in sorted(t1.get('curves_by_seed', {}).items(),
                           key=lambda kv: int(kv[0])):
        if not cv:
            continue
        cv = np.asarray(cv, dtype=float)
        ax.plot(cv[:, 0], cv[:, 1] * 100, lw=0.9, alpha=0.55,
                label=f'seed {seed}')
    med = np.asarray(t1['curve'], dtype=float)
    if med.size:
        ax.plot(med[:, 0], med[:, 1] * 100, 'k-', lw=2.3,
                label=f'中位数 (以上 {len(t1.get("curves_by_seed", {}))} 条)')
    ax.axhline(5.0, color='crimson', ls='--', lw=1.3, label='目标 5%')
    ax.axhline(11.45, color='gray', ls=':', lw=1.3, label='已发布基线 11.45%')
    ax.axvline(600, color='gray', lw=0.9, alpha=0.7)
    ax.annotate('已发布基线止步于此\n(600 步)', xy=(600, 30), fontsize=8,
                color='gray')
    ax.set_yscale('log')
    ax.set_xlabel('变分步数')
    ax.set_ylabel('eps_c  [%]')
    n_cv = len(t1.get('curves_by_seed', {}))
    other_lr = (f'lr={bud["lrs"][1]} 的 {len(t1["runs_summary"]) - n_cv} 条'
                f'未逐步记录, 只出现在面板三\n' if len(bud['lrs']) > 1 else '')
    ax.set_title(f'二 · 收敛曲线 (lr={bud["lrs"][0]}, chi={bud["chi"]}, '
                 f'{n_cv} 个种子)  ★核心证据\n'
                 + other_lr
                 + f'预算翻倍时 eps_c 相对变化 = '
                   f'{t1["budget_diagnostic"]["drift"]:+.1%}')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    # ---- 3. 多轨迹分布 ----
    ax = axes[0, 2]
    for lr, mk in zip(bud['lrs'], ('o', 's')):
        rs = [r for r in t1['runs_summary'] if r['lr'] == lr]
        ax.semilogy([r['eps_c'] for r in rs], [r['R_conf'] for r in rs], mk,
                    ms=9, alpha=0.85, label=f'lr={lr} ({len(rs)} 条)')
        for r in rs:
            ax.annotate(f"s{r['seed']}", (r['eps_c'], r['R_conf']),
                        fontsize=7, xytext=(3, 3), textcoords='offset points')
    ax.axvline(5e-2, color='crimson', ls='--', lw=1.3)
    ax.axhline(1e-2, color='crimson', ls='--', lw=1.3)
    ax.axvline(0.1145, color='gray', ls=':', lw=1.3)
    ax.axhline(t1['conformal_invariance']['R_conf_exact'], color='steelblue',
               ls='-.', lw=1.2, label='有限尺寸地板')
    ax.set_xscale('log')
    ax.set_xlabel('eps_c  (目标 < 5%)')
    ax.set_ylabel('R_conf  (目标 < 1e-2)')
    ax.set_title('三 · 多轨迹分布 (判据卡最坏轨迹)\n'
                 '虚线 = 目标; 点线 = 已发布基线 eps_c')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    # ---- 4. chi 归因 ----
    ax = axes[0, 3]
    chis = [bud['chi'], bud['chi_ctrl']]
    meas = [t1['chi_bottleneck']['measured'][str(c)] if str(c)
            in t1['chi_bottleneck']['measured']
            else t1['chi_bottleneck']['measured'][c] for c in chis]
    floors = [bud['floor_chi'], bud['floor_chi_ctrl']]
    xpos = np.arange(len(chis))
    ax.bar(xpos - 0.18, [m * 100 for m in meas], 0.36, color='steelblue',
           label='MERA 实测')
    ax.bar(xpos + 0.18, [f * 100 for f in floors], 0.36, color='lightgray',
           label='Schmidt 截断参照 (非严格界)')
    ax.axhline(5.0, color='crimson', ls='--', lw=1.3, label='目标 5%')
    ax.set_xticks(xpos)
    ax.set_xticklabels([f'chi={c}' for c in chis])
    ax.set_ylabel('eps_c  [%]')
    ax.set_title('四 · chi 瓶颈归因 (同种子同预算, 只改 chi)\n'
                 f'{t1["chi_bottleneck"]["verdict"]}')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, axis='y')

    # ---- 5. 因果链拟合 RMSE (2.1) ----
    ax = axes[1, 0]
    cr = metrics['tier2']['chain_rmse']
    pr = np.atleast_2d(np.asarray(cr.get('pairs', []), dtype=float))
    if pr.size and pr.shape[1] == 2:
        th, me = pr[:, 0], pr[:, 1]
        lo = float(min(th.min(), me.min()))
        hi = float(max(th.max(), me.max()))
        pad = 10 ** (0.08 * (np.log10(hi) - np.log10(lo) + 1e-9))
        ax.loglog([lo / pad, hi * pad], [lo / pad, hi * pad], 'k--', lw=1.3,
                  label='理想: 实测 = 理论 (y=x)')
        ax.loglog(th, me, 'o', ms=10, mfc='none', mec='navy', mew=2,
                  label=f'{len(th)} 组受控谱隙')
        for t, m in zip(th, me):
            ax.annotate(f'{abs(t - m):.1e}', (t, m), fontsize=7, color='navy',
                        xytext=(6, -3), textcoords='offset points')
        ax.set_xlim(lo / pad, hi * pad)
        ax.set_ylim(lo / pad, hi * pad)
    ax.set_xlabel('理论收敛比  |lam2/lam1|  (矩阵特征值直接算出)')
    ax.set_ylabel('实测收敛比 (幂迭代末段步长比中位数)')
    ax.set_title(f'五 · 微观物理 · 因果链拟合 RMSE (2.1)\n'
                 f'RMSE = {cr["rmse"]:.3e}  (目标 < 1e-2, 与已发布基线同阈值)')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, which='both')

    # ---- 6. 纠缠谱 KL (2.2) ----
    ax = axes[1, 1]
    kl = metrics['tier2']['entropy_kl']
    spar = np.asarray(s2['spectrum'])
    ax.semilogy(np.arange(1, len(spar) + 1), np.maximum(spar, 1e-18), 'o-',
                ms=3, label=f'精确基态 Schmidt 谱 ({len(spar)} 个奇异值)')
    ax.set_xlabel('Schmidt 指标 n')
    # 注: s2['spectrum'] 是**奇异值** s_n 本身 (spiral_model_v13.py:638 的 SVD 输出,
    # 没有平方)。第一版这里写成 s_i^2 是笔误, 竖轴与横轴对不上 —— 已改回 s_n。
    ax.set_ylabel('奇异值  s_n')
    ax.set_title(f'六 · 微观物理 · 纠缠熵 KL (2.2)\n'
                 f'KL(MERA‖精确) = {kl["kl"]:.3e} nats, JS = {kl["js"]:.3e}\n'
                 f'支撑 MERA {kl["n_p"]} / 精确 {kl["n_q"]}, '
                 f'共同 {kl["n_common"]} (KL 只在交集上求和)')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    # ---- 7. 关联衰减指数 (2.3) ----
    # 为什么必须画出: 这是"共形形式 (π/L)/sin(πr/L) vs 朴素幂律 r"这个对照的
    # 唯一可视化, 而已发布基线的"朴素幂律给 η=0.205、共形形式给 0.2426"正是
    # 那条诚实边界的依据。只报一个数字看不出形式选错有多大代价。
    ax = axes[1, 2]
    ce = metrics['tier2']['correlation_exponent']
    L16 = 16
    _, gs16 = exact_ground_state(L16, 1.0, 1.0)
    _, C16, _ = boundary_correlation_graph(gs16, L16)
    rs = np.arange(1, L16 // 2 + 1)
    y16 = np.abs(np.array([C16[0, r] for r in rs]))
    f16 = (np.pi / L16) / np.sin(np.pi * rs / L16)
    good = y16 > 1e-12
    ax.loglog(f16[good], y16[good], 'ko', ms=5, label='|C(r)| 精确 (L=16)')
    if good.sum() >= 2:
        sl, ic = np.polyfit(np.log(f16[good]), np.log(y16[good]), 1)
        ff = np.linspace(np.log(f16[good].min()), np.log(f16[good].max()), 50)
        ax.loglog(np.exp(ff), np.exp(ic + sl * ff), 'r-', lw=1.5,
                  label=f'共形形式拟合 斜率={sl:.4f}\n(→ 2Δ={ce["two_delta"]:.4f}, '
                        f'精确 0.25, 相对误差 {ce["rel"]:.1%})')
    ax.loglog(rs, y16, 'b--', lw=1.0,
              label=f'朴素幂律对照 η={ce["eta_naive"]:.4f}')
    ax.set_title('七 · 微观物理 · 关联衰减指数 (2.3)\n'
                 '共形形式 vs 朴素幂律 (后者在有 PBC 的环上必然偏,\n'
                 '这是形状效应不是数值误差)')
    ax.set_xlabel('f = (π/L)/sin(πr/L)  [实心点],   r  [虚线]')
    ax.set_ylabel('|⟨σᶻ₀σᶻ_r⟩|')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, which='both')
    # loglog 上 matplotlib 会给次刻度也标数字 (范围 2e-3~7e0), 在 625 px 宽的
    # 格子里这些标签互相压在一起完全读不出来。只保留主刻度标签。
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # ---- 8. Jordan 块瞬态增长 (2.4) ----
    # 为什么必须恢复: 这是"谱半径 0.90 < 1 却先涨 272 倍"这个非正规性陷阱的
    # 唯一可视化。只看谱半径会得出"收敛"的结论, 而范数曲线直接反驳它。
    ax = axes[1, 3]
    tr = s3a['transient']
    jp = metrics['tier2']['jordan_peak']
    kk = np.arange(1, len(tr['growth']) + 1)
    ax.semilogy(kk, tr['growth'], 'o-', ms=3, color='navy',
                label=f"‖A^k‖₂ 实测峰在 k={jp['k_meas']}")
    ax.axvline(jp['k_theory'], color='crimson', ls='--', lw=1.5,
               label=f"理论 k* = (N−1)/|ln λ| = {jp['k_theory']:.2f}\n"
                     f"相对误差 {jp['rel']:.2%} (目标 < 5%)")
    ax.set_title(f"八 · 微观物理 · Jordan 块瞬态增长 (2.4)\n"
                 f"N={tr['N']}, λ={tr['lam']}, 谱半径 "
                 f"{tr['spectral_radius']:.2f} < 1 但范数先涨后落\n"
                 f"— 非正规性的直接证据, 不是数值错误")
    ax.set_xlim(0, 1.06 * max(len(kk), float(jp['k_theory'])))
    ax.set_xlabel('迭代步 k')
    ax.set_ylabel('‖A^k‖₂')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    # ---- 9. A 段描述子距离 vs 噪声底 ----
    ax = axes[2, 0]
    a = metrics['tier3'].get('_plot', {}).get('A')
    if a:
        per = metrics['tier3']['A1']['per_key']
        ks = list(a['keys'])
        ax.bar(ks, [per[k] for k in ks], color='steelblue')
        ax.axhline(a['scatter'], color='darkorange', ls='--', lw=1.5,
                   label=f'种子间散布 (噪声底) = {a["scatter"]:.4f}')
        ax.axhline(3 * a['scatter'], color='crimson', ls='--', lw=1.3,
                   label=f'判据上限 3x = {3 * a["scatter"]:.4f}')
        ax.set_ylabel('逐项距离 d_i')
        ax.set_title(f'九 · 第三层 A 段 · 描述子距离 (36^2 vs 100^2)\n'
                     f'D_struct = {a["D"]:.4f}  (不设达标线, 见十一)')
        ax.legend(fontsize=7)
        ax.tick_params(axis='x', rotation=30)
    else:
        ax.set_title('九 · A 段\n(第三层跳过)')
    ax.grid(alpha=0.25, axis='y')

    # ---- 10. A 段归一化谱形 ----
    ax = axes[2, 1]
    if a:
        pk = a['pk']
        ax.loglog(pk['xr'], pk['yr'], 'o-', ms=3, color='darkorange',
                  label='CLIP GS (3 种子均值)')
        ax.loglog(pk['xm'], pk['ym'], 's--', ms=3, color='steelblue',
                  label='阶段六 GS (36^2)')
        ax.set_xlabel('k / k_peak')
        ax.set_ylabel('Delta^2 (归一到积分 1)')
        ax.set_title(f'十 · 第三层 A 段 · 归一化谱形\n'
                     f'R_P = {a["R_P"]:.4f}   (已发布基线同量是 1.277,\n'
                     f'当年是拿 2D GS 比 3D CDM 晕谱)')
        ax.legend(fontsize=7)
    else:
        ax.set_title('十 · A 段谱形\n(第三层跳过)')
    ax.grid(alpha=0.25, which='both')

    # ---- 11. A 段: 两个尺度估计器对不上 -> 混杂因子 (2) 已撤回 ----
    # 【本格已改写】原版画的是"模型 4.1 vs 参考 68.9 个波长",
    # 用来支撑混杂因子 (2)。后查明参考侧那个 68.9 是 _char_scale 的
    # argmax 抓到二维离散谱**角点** (λ=√2=1.414 格, 网格能表示的最短波长)
    # 造成的伪像 —— 1 格的二值台阶, 不是斑图波长。
    # 所以这一格改成把两个**互相独立**的尺度估计器并排画出来:
    #   lam_pk = 谱峰 argmax    (会被高频壳骗, 参考侧失效)
    #   xi_ac  = 自相关首过零  (零阶量, 不挑壳, 可作交叉核对)
    # 模型侧两者差 2.2 倍 (正常), 参考侧差 29 倍 (估计器失效)。这比原来的
    # 单估计器柱图诚实: 它把"为什么这个数是伪像"直接画出来了。
    ax = axes[2, 2]
    a1 = metrics['tier3'].get('A1')
    if a1 and a.get('lam_mod') is not None:
        _lm, _lr = float(a['lam_mod']), float(a['lam_ref'])
        _lc = float(a.get('lam_corner_cells', float('nan')))
        _xm = float(a1.get('xi_ac_model', float('nan')))
        _xr_list = [float(v) for v in (a1.get('xi_ac_ref') or []) if v == v]
        _xr = float(np.median(_xr_list)) if _xr_list else float('nan')
        _w = 0.36
        ax.bar([-_w / 2, 1 - _w / 2], [_lm, _lr], _w, color='steelblue',
               label='lam_pk (谱峰 argmax)')
        ax.bar([_w / 2, 1 + _w / 2], [_xm, _xr], _w, color='seagreen',
               label='xi_ac (自相关首次过零)')
        ax.axhline(_lc, ls='--', lw=1.2, color='crimson')
        # 角点线**不写图内文字**: 这张图几乎每个位置都有柱子, 文字压上去
        # 不是被柱子盖住就是被右边缘切掉 (成图上两种都实测遇到过)。
        # 改成一个 Line2D 图例项, 信息不丢, 也不和任何东西重叠。
        _cnr = plt.Line2D([], [], ls='--', lw=1.2, color='crimson',
                          label=f'网格角点 {_lc:.3f} 格 (最短可表示波长)')
        _h, _l = ax.get_legend_handles_labels()
        ax.legend(_h + [_cnr], _l + [_cnr.get_label()], fontsize=6.5,
                  loc='upper left', framealpha=0.92)
        for _xi, _v in zip([-_w / 2, _w / 2, 1 - _w / 2, 1 + _w / 2],
                           [_lm, _xm, _lr, _xr]):
            ax.text(_xi, _v * 1.06, f'{_v:.2f}', ha='center', va='bottom',
                    fontsize=8)
        ax.set_yscale('log')
        # 留出顶部余量, 否则最高那根柱顶到坐标框、数值标签飘到框外
        ax.set_ylim(bottom=max(0.3, _lc * 0.22),
                    top=max(_lm, _xr) * 1.9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f'模型 36^2\n{a1["params_model"]}',
                            f'参考 100^2\n{a1["params_ref"]}'], fontsize=7.5)
        ax.set_ylabel('特征尺度 (格)')
        ax.set_title(f'十一 · 第三层 A 段 · 两个尺度估计器对不上\n'
                     f'模型 {_lm / max(_xm, 1e-9):.1f}x (正常), 参考 '
                     f'{_xr / max(_lr, 1e-9):.0f}x ⇒ 参考侧 lam_pk 是伪像\n'
                     f'混杂因子 (2) 已撤回, 只剩 (1) 相区参数不同')
        ax.grid(alpha=0.25, axis='y', which='both')
    else:
        ax.set_title('十一 · A 段混杂因子\n(第三层跳过)')

    # ---- 12. B1 段: Southampton 波峰周期 (数值列, 非图版) ----
    ax = axes[2, 3]
    b = metrics['tier3'].get('_plot', {}).get('B')
    if b and b.get('peaks'):
        names = sorted(b['peaks'])
        pers = [b['peaks'][n]['period_s'] for n in names]
        cvs = [b['peaks'][n]['dt_cv'] for n in names]
        cols = ['#2b7bba' if c < 0.5 else '#d1653a' for c in cvs]
        ax.bar(range(len(names)), pers, color=cols)
        for i, (p, c) in enumerate(zip(pers, cvs)):
            ax.text(i, p, f'{p:.1f}s\nCV={c:.2f}', ha='center', va='bottom',
                    fontsize=7)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel('波峰间隔中位数 (s)')
        ax.set_ylim(0, max(pers) * 1.35)
        ax.set_title('十二 · 第三层 B1 段 · BZ 波峰周期\n'
                     '来源 = 作者的波峰标记数值列 (不经 .tif 图版)\n'
                     '橙 = CV>0.5, 是该记录的真实性质而非流程失败')
        ax.grid(alpha=0.25, axis='y')
    else:
        ax.set_title('十二 · B 段\n(第三层跳过)')

    # ---- 13. B1 段的独立交叉检验 (两层提取是否指向同一个 T) ----
    # 这是第三层作为"方法学展示"真正要交付的东西: 波峰间隔统计与强度序列
    # 自相关是两套**互相独立**的提取 (前者来自作者标注的峰位, 后者来自原始
    # 强度列, 不经过任何波峰检测)。把四个滞后的系数全画出来, 读者才能自己
    # 核对"极大落在哪个滞后", 而不是接受一句 verdict。
    ax = axes[3, 0]
    lagk = ['T/2', 'T', '1.5T', '2T']
    if b and b.get('peaks'):
        names = sorted(b['peaks'])
        pal = ['#2b7bba', '#d1653a', '#3f8f5f', '#8a5fb0']
        xs = np.arange(len(lagk))
        w = 0.8 / max(len(names), 1)
        any_ac = False
        for i, nm in enumerate(names):
            pk = b['peaks'][nm]
            acs = pk.get('ac')
            if not acs:
                continue
            any_ac = True
            vals = [float(acs.get(k, np.nan)) for k in lagk]
            ax.bar(xs + (i - (len(names) - 1) / 2) * w, vals, w,
                   color=pal[i % len(pal)],
                   label=f"{nm}  (CV={pk['dt_cv']:.2f}, 极大在 {pk.get('lag_best')})")
        if any_ac:
            ax.set_xticks(xs)
            ax.set_xticklabels(lagk)
        ax.axhline(0.0, color='k', lw=0.8)
    ax.set_xlabel('滞后 (以波峰间隔中位数 T 为单位)')
    ax.set_ylabel('强度序列自相关系数 ac(τ)')
    ax.set_title('十三 · 第三层 B1 段 · 独立交叉检验\n'
                 '波峰标记 vs 强度序列自相关 —— 两套独立提取\n'
                 '三个记录的"极大位置"排序与 CV 排序完全一致')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=0.25, axis='y')

    # ---- 14. B2 段: Reading 功率谱 —— "未检出"是个结论, 把证据画出来 ----
    ax = axes[3, 1]
    rd = (b or {}).get('reading', {})
    if rd:
        pal = ['#2b7bba', '#d1653a', '#3f8f5f', '#8a5fb0']
        kmin = None
        for i, (nm, v) in enumerate(sorted(rd.items())):
            pw = np.asarray(v['power'], dtype=float)
            kk = np.arange(1, len(pw) + 1)
            kmin = v['min_cycles']
            c = pal[i % len(pal)]
            kb, pb = _log_bin(kk, pw, 14)
            ax.loglog(kb, pb, lw=1.3, color=c,
                      label=f"{_short(nm)}: 带内最大在 k={v['k_band_argmax']}, "
                            f"突出度 {v['prominence']:.0f}")
            ax.axhline(v['power_thresh'], color=c, ls=':', lw=1.0, alpha=0.85)
            ax.plot([v['k_band_argmax']], [pw[v['k_band_argmax'] - 1]], 'v',
                    color=c, ms=10)
        if kmin:
            ax.axvline(kmin, color='k', ls='--', lw=1.2,
                       label=f'频带下界 k={kmin} (以下被高通置零)')
            ax.set_xlim(kmin * 0.75, 600)
        ax.set_xlabel('谱线 k  (周期 = 2161 / k 帧)')
        ax.set_ylabel('Hann 窗后 |F|²  (对数分箱显示, 判定用未分箱谱)')
        ax.set_title('十四 · 第三层 B2 段 · Reading 功率谱 (高通后)\n'
                     '▼ = 带内最大值, 三条都恰好落在频带**下界** k=8 上;\n'
                     '点线 = 各自的突出度阈值 (局部背景中位数 x 300)\n'
                     '三条都**越过**了点线 —— 检不出不是因为峰不够突出,\n'
                     '而是因为 k=8 不是"带内内部的局部极大"。带下界随\n'
                     'min_cycles 变, 把边缘当峰等于把滤波器边界读成物理量')
        ax.legend(fontsize=6.5, loc='lower left')
    else:
        ax.set_title('十四 · B2 段 Reading 谱\n(第三层跳过)')
    ax.grid(alpha=0.25, which='both')

    # ---- 15. B2 段: Reading 逐帧亮度 + 预热段裁剪敏感性 ----
    ax = axes[3, 2]
    if rd:
        pal = ['#2b7bba', '#d1653a', '#3f8f5f', '#8a5fb0']
        nd_max = 0
        lo, hi = np.inf, -np.inf
        for i, (nm, v) in enumerate(sorted(rd.items())):
            y = np.asarray(v['lum'], dtype=float)
            x = np.arange(len(y)) * 4.0            # 已按 ::4 降采样
            nd_max = max(nd_max, int(v.get('n_dark', 0)))
            lo, hi = min(lo, float(y.min())), max(hi, float(y.max()))
            ax.plot(x, y, lw=0.9, color=pal[i % len(pal)], label=_short(nm))
        if nd_max:
            ax.axvspan(0, float(nd_max), color='k', alpha=0.10)
        span = max(hi - lo, 1e-9)
        ax.set_ylim(lo - 0.05 * span, hi + 0.30 * span)   # 顶部留白给图例
        ax.set_xlabel('帧  (三条长度均为 2161; 显示按 ::4 降采样)')
        ax.set_ylabel('逐帧平均亮度 (绝对灰度)')
        ax.set_title('十五 · 第三层 B2 段 · Reading 逐帧亮度\n'
                     '左端灰带 = 相机预热暗帧 (25.6 vs 稳态 84~115),\n'
                     '一段硬阶跃: 它给谱灌宽频功率, 把漂移占比**压小**')
        lines = ['裁剪敏感性  (漂移占比; trim = 掐掉头部帧数)']
        tot = tot_det = 0
        for nm, v in sorted(rd.items()):
            tr = v.get('trim_trials', [])
            tot += len(tr)
            tot_det += sum(1 for t in tr if t['detected'])
            fr = '  '.join(f"{t['trim']}:{t['drift_frac']:.0%}" for t in tr)
            lines.append(f"{_short(nm, 13):<13} {fr}")
        lines.append(f'合计检出 {tot_det}/{tot} => 结论不依赖预处理')
        ax.text(0.985, 0.02, '\n'.join(lines), transform=ax.transAxes,
                ha='right', va='bottom', fontsize=6.5,
                # family 必须是**列表**: 单调 'monospace' 会解析到 DejaVu Sans
                # Mono, 它没有 CJK 字形, 中文标签会全变成方块 (□□) —— 而这张
                # 表的标题正是中文。matplotlib >= 3.6 按字形逐个回退, 所以
                # 数字仍用等宽体保持列对齐, 中文回退到全局 CJK 字体。
                family=['monospace'] + list(plt.rcParams['font.sans-serif']),
                bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.92))
        ax.legend(fontsize=6, loc='upper right', framealpha=0.92)
    else:
        ax.set_title('十五 · B2 段 Reading 序列\n(第三层跳过)')
    ax.grid(alpha=0.25)

    # ---- 16. 守卫通过率 ----
    ax = axes[3, 3]
    g = metrics['guards']
    ax.bar(['应当通过', '诊断项\n(设计上就该失败)'],
           [g['rate'] * 100, (g['n_diag'] - g['n_diag_notok'])
            / max(g['n_diag'], 1) * 100],
           color=['seagreen', 'gray'])
    ax.set_ylim(0, 105)
    ax.axhline(100, color='k', ls='--', lw=1.0)
    ax.set_ylabel('通过率 [%]')
    ax.set_title(f'十六 · 守卫通过率\n{g["n_pass"]}/{g["n_expect"]} = '
                 f'{g["rate"]:.1%} (诊断项 {g["n_diag"]} 条单列, 不计入分母;\n'
                 f'分母 31→22 的拆解见 spiral_v13_说明.md §5)')
    for i, v in enumerate([g['rate'] * 100,
                           (g['n_diag'] - g['n_diag_notok'])
                           / max(g['n_diag'], 1) * 100]):
        ax.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=9)

    # ---- 17~20. A 段参考侧/模型侧实际场 (v14·F3 改写) ----
    # 这四张图让读者**看见**参考家族 (自建种子) 与模型侧的实际形态, 并同时
    # 看到两者的自相关长度 —— 参考侧是不是斑图, 自己看即可, 不必只听结论。
    # 参考侧只放前 PANEL_SEEDS 个 (网格只有 4 列 x 1 行可用), 全部 12 个种子的
    # 描述子在 result/_v16_data.json 的 A1 里。
    _fld = a.get('fields') if a else None
    if _fld:
        _order = [f'参考 seed {s}' for s in (a.get('panel_seeds') or [])] + ['模型']
        _order = [n for n in _order if n in _fld] + \
                 [n for n in _fld if n not in _order]
        _rom = ['十七', '十八', '十九', '二十']
        for _i, _nm in enumerate(_order[:len(_rom)]):
            _r, _c = 4 + _i // 4, _i % 4
            _ax = axes[_r, _c]
            _fi = _fld[_nm]
            _img = np.asarray(_fi['img'], dtype=float)
            _ax.imshow(_img, cmap='gray_r', vmin=0, vmax=1,
                       interpolation='nearest', origin='lower',
                       extent=(0, _fi['box'], 0, _fi['box']))
            _cn = (a.get('conn_model') if _nm.startswith('模型') else
                   (a.get('conn_ref', {}) or {}).get(
                       _nm.replace('参考 seed ', ''), None))
            _tag = (f', 少数相 {_cn["n_comp"]} 块, 平均 {_cn["mean_size"]:.0f} 格'
                    if _cn else '')
            _ax.set_title(f'{_rom[_i]} · A 段实际场\n{_nm}   {_fi["box"]}²  '
                          f'占空比 {_fi["phi"]:.4f} (中位数阈恒等式)',
                          fontsize=8.5)
            _ax.text(0.5, -0.10, f'自相关长度 {_fi["xi_ac"]:.1f} 格 = '
                                 f'{_fi["xi_ac"] / _fi["box"]:.2f} 盒长{_tag}',
                     transform=_ax.transAxes, ha='center', va='top', fontsize=7.5)
            # **不设 xlabel**: 它与下面那行说明文字都以 axes 的 x=0.5 居中,
            # 必然重叠 (成图上实测到的排版缺陷: 黑色的 '格' 压在
            # 红色说明文字上)。单位由 ylabel 和说明文字里的"格"给出。
            _ax.set_ylabel('格', fontsize=8)
            _ax.tick_params(labelsize=7)
        _lm = a.get('lam_mod', float('nan'))
        _lr = a.get('lam_ref', float('nan'))
        _lc = a.get('lam_corner_cells', float('nan'))
        _lleg = a.get('lam_legacy_cells', float('nan'))
        fig.text(0.5, 0.012,
                 f'A 段诊断 (v14·F3 改写, 诊断项不计入分母): '
                 f'参考侧已换为自建种子 (共 {a.get("n_seeds_total", "?")} 个, '
                 f'图中前 3 个), 两侧同用 (v > 中位数) 二值化。'
                 f'lam_pk 模型 {_lm:.2f} 格 vs 参考中位数 {_lr:.2f} 格 -> '
                 f'"盒子内波长数" {a.get("n_lambda_model", float("nan")):.1f} vs '
                 f'{a.get("n_lambda_ref", float("nan")):.1f} (但该量是盒子边长/lam_pk '
                 f'的换写, 不是独立测量)。\n'
                 f'**两条照着事实报**: (i) 旧口径用的归档 bool 场与同一归档的浮点场'
                 f'任何简单二值化都对不上 (逐格一致率≈0.5), 其生成管线不在归档里 '
                 f'-> 那个 lam_pk {_lleg:.2f} 格落在二维离散谱角点 '
                 f'{_lc:.3f} 格 (网格能表示的**最短**波长), 是那种场的 1 格台阶伪像, '
                 f'也是"波长数差一个量级"那条混杂因子的来源; '
                 f'(ii) _char_scale 取 Δ²(k) 在**离散壳**上的 argmax, 所以 lam_pk 只'
                 f'能取离散值, 且与**独立**的自相关尺度 4*xi_ac 系统性不一致 '
                 f'(中位偏差 {a.get("lam_pk_ac_dev_median", float("nan")):.0%}) —— '
                 f'它在本段给出的读数既不保证是斑图波长, 也没有独立佐证, '
                 f'不能当连续波长读。**未改 _char_scale**, 缺口如实记录。',
                 ha='center', va='bottom', fontsize=8.5,
                 bbox=dict(boxstyle='round', fc='#f5f8ff', ec='steelblue',
                           alpha=0.95))

    fig.suptitle('v15 量化指标层 · 20 面板 · 第一层分布口径 (多种子 + 预算) · '
                 '第二层原样 (4/4, 含 2.3/2.4) · '
                 '第三层降级 (2D 斑图 结构对结构, 不设达标线) · '
                 'F3: 参考侧换自建种子',
                 fontsize=14, y=0.996)
    fig.tight_layout(rect=(0, 0.035, 1, 0.982))
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    print(f"\n  指标图已保存: {out_png}")
