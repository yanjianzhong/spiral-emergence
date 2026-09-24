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
"""spiral_metric_v16._metric.extras —— 由 spiral_metric_v16.py 拆分。

W1·W3a/W3b·W5·L1 负对照·W6 台账·几何稳健性

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
    GUARDS, record_guard, record_metric,
)




# ===========================================================================
# v15 · W1 自指变体的秩1化 N 依赖 (相图)
# ===========================================================================
def metric_selfref_N_dependence(nd):
    """
    2.9 A/B 变体秩1化的 N 依赖 —— "这条结论依赖 N"究竟依赖成什么样。

    v15·W1 新增。v14 只在 selfref_coupled 的结语里对比两个点 (手工 N=64 与
    派生 N=17), 没有相图, 于是"依赖 N"这句话既说不清是阈值还是随机命中,
    也说不清 A 的秩1方向究竟是不是不动点。本项把三件事登记下来:
      (1) 相图: 每个 N 上 A 秩1化的种子命中率。实测 N<=32 全 0/5, N=48 为
          1/5, N=64 为 5/5 —— 是**随种子随机**, 不存在一条 N 阈值;
      (2) RANK_TOL 敏感性: 换阈值会不会翻盘 (由已返回的 sigma 比离线重算);
      (3) A 的 ||x|| ~ n^-p 的指数 p —— p>0 说明秩1方向是**有限步暂态**。

    诚实边界:
      - 这里的 N 是**自指网络宽度**, 不是自旋链长度 L; 每步 O(N^2)~O(N^3),
        不涉及 2^L 精确对角化, 因此不受链长 L 那条约束 (L 的上限来自末圈 MERA
        要求 L 为 2 的幂, 与本函数的 N 无关)。
      - 物理上真实取到的只有 N_self(L) in {11,12,15,16,17} (L=8..16), 实测
        它们**全部 0/5**; 网格里的 48/64 是为定位相变而设, 不是模型会走到的状态。
      - p 只吻合**指数** (实测 0.4950~0.4992 对预测 1/2), 前置因子与简单推导
        差约 5.9 倍未复现 —— 所以本条只说"是幂律衰减", 不说"时标已被解释"。
    """
    if not nd:
        record_metric(2, '9 自指变体秩1化的 N 依赖 (相图)', '—', '—',
                      'spiral_model_v16.selfref_N_dependence()', None, None,
                      None, None, None, note='未执行 (ctx 无 s3c_nd)。')
        record_guard('W1 秩1化相图已运行', 'selfref_N_dependence',
                     'ctx["s3c_nd"] 存在', False)
        return None

    hit = nd['hit_rate_free']
    pure, zero, mixed = nd['pure_N'], nd['zero_N'], nd['mixed_N']
    p, fconv = nd['median_p_free'], nd['frozen_converged_frac']
    # L<=16 能派生到的 N。这是**生产路径真正取到**的区间, 与扫描网格分开看。
    reachable = [N for N in nd['N_list'] if N <= 17]
    reach_hit = {N: hit[N] for N in reachable}
    reach_clean = bool(reachable) and all(v == 0.0 for v in reach_hit.values())
    measured = {
        'nd_N_list': nd['N_list'], 'nd_seeds': nd['seeds'],
        'nd_steps': nd['steps'], 'nd_hit_rate_free': hit,
        'nd_pure_N': pure, 'nd_zero_N': zero, 'nd_mixed_N': mixed,
        'nd_reachable_N': reachable, 'nd_reachable_hit': reach_hit,
        'nd_median_p_free': p, 'nd_frozen_converged_frac': fconv,
        'nd_seconds': nd['seconds'],
    }
    record_metric(
        2, '9 自指变体秩1化的 N 依赖 (相图)',
        'A/B 各变体秩1化的种子命中率随 N; A 的 ||x|| 幂律指数 p',
        '"这条结论依赖 N"是依赖成一条阈值, 还是随种子随机?'
        ' A 的秩1方向是不动点还是暂态?',
        'spiral_model_v16.selfref_N_dependence(): 逐 N 调 selfref_coupled, '
        f'{len(nd["N_list"])} 个 N x {len(nd["seeds"])} 种子 x {nd["steps"]} 步, '
        '判据 sigma1/sigma2 > 1e3',
        '自指网络宽度 N (非链长 L); 物理可达的只有 N_self(L) in {11..17}',
        measured,
        'v14 只在结语里对比手工 N=64 与派生 N=17 两个点, 没有相图',
        '相图报出各 N 的命中率 (不预设阈值); RANK_TOL 换阈值不翻盘; p > 0',
        bool(reach_clean),
        note=f"**逐 N 命中率**: " + ", ".join(
            f"N={N}:{hit[N]:.0%}" for N in nd['N_list'])
        + f"。**这是随机的, 不是阈值** —— 既有全 0 ({zero}) 也有全中 ({pure}), "
        f"还有部分命中 ({mixed}); 若把它读成一条 N_c 就会错。"
        f" **可达区间**: L<=16 只能派生出 N={reachable}, 实测命中率 "
        f"{reach_hit} —— A 在生产路径上**从不**秩1化, 所以 v14 那处"
        f"「判据用变体 A、取值用变体 B」的不一致是**死代码**, 这正是它"
        f"藏了这么久的原因。 **暂态判定**: A 的 ||x|| ~ n^-p, p 中位 "
        f"{p if p is None else round(p, 4)} (预测 1/2, 来自 ds/dn = -s^3/3 即 "
        f"g=1 的边缘稳定) —— p>0 说明秩1方向是**有限步暂态**, 渐近仍归于 "
        f"x*=0, 不是不动点。**诚实边界**: 只吻合指数, 前置因子差约 5.9 倍未"
        f"复现; 且这里的 N 是自指网络宽度不是链长 L, 不受 L<=16 约束。")
    record_guard('W1 可达 N 上变体 A 不秩1化', 'selfref_N_dependence',
                 f"N<=17 的命中率 {reach_hit} 全为 0 "
                 f"(L<=16 能派生的 N 只有 {reachable})",
                 reach_clean,
                 note='守的是环节七的语义前提: A 在可达区间一律塌到 x*=0, '
                      '所以阶段七走的是"空无"支。若这条翻了, W4 的分列读数'
                      '与 stage7 的叙事都要重看。')
    record_guard('W1 诊断: 冻结对照并非"永不收敛"', 'selfref_N_dependence',
                 f"冻结对照收敛到 <1e-4 的比例 = {fconv:.1%} "
                 f"({sum(1 for r in nd['rows'] if r['frozen_dx'] < 1e-4)}/"
                 f"{len(nd['rows'])})",
                 bool(fconv == 0.0), expect_pass=False,
                 note='**故意不计入通过率**: 这是负面事实的可复现证据 —— '
                      'v14 印的"冻结的对照永不收敛"是绝对断言, 实测有 16.0% '
                      '的 (N,seed) 组合收敛到机器精度 (且分布双峰: <1e-8 或 '
                      '>=1e-4, 中间为空)。留着它是为了防止那句绝对断言又被加回来。')
    print(f"\n  ── v15·W1 自指变体秩1化的 N 依赖 ──")
    print(f"      A 秩1化命中率: " + ", ".join(
        f"N={N}:{hit[N]:.0%}" for N in nd['N_list']))
    print(f"      可达区间 (L<=16 派生的 N={reachable}): 命中率 "
          f"{'全 0' if reach_clean else reach_hit}"
          f"{' -> A 从不秩1化, 阶段七走空无支' if reach_clean else ''}")
    print(f"      A 的 ||x|| ~ n^-p: p 中位 = "
          f"{'未产生' if p is None else f'{p:.4f}'} (预测 1/2) -> "
          f"秩1方向是有限步暂态")
    print(f"      冻结对照收敛比例 = {fconv:.1%} (v14 印的是'永不收敛')")
    return measured




# ===========================================================================
# v15 · W3b 谱学中心荷 (独立族: 能谱)
# ===========================================================================
def metric_spectral_central_charge(csp):
    """
    2.10 谱学中心荷 —— 回答 v14 审计表 §7 第 4 条登记的未通过项
    ("独立族中心荷估计量未建立": Rényi 指数扫描 0.53986 +/- 0.02252, 偏差 7.97%)。

    这一项为什么算"独立族": 它用的**数据**是 H 的头三个本征值 (不是 |psi> 的
    约化密度矩阵), 用的**函数**是闭式比值 (无最小二乘拟合、无有限尺寸外推),
    并且 v 自动消去。换成能谱这条路的代价在诚实边界里说清。

    三条守卫分开记, 因为它们的强度不同:
      (a) 算符内容识别 (打分): E1 逐点等于 R 扇区真空 (Delta1=1/8) 且 E2 逐点
          等于 NS 双粒子 (Delta2=1)。这条**不含 c 也不含 v**, 是全项里最硬的
          一条 —— 它一旦不成立, 下面的 c 读数就没有解释。
      (b) 比值指纹 (打分): (E2-E0)/(E1-E0) 在 L=8..16 上单调, 且 L=16 落在
          8 的 1% 内。
      (c) c 读数 (打分): 在 (a)(b) 成立的前提下, c 落在 0.5 的 1% 内。
          **它的强度被前提封顶**: 函数含 Delta1=1/8, 所以这条是"自洽性"而不是
          "无假设测量"。如实记为打分项但在 note 里写明封顶, 不假装它与 (a) 同级。
    """
    if not csp:
        record_metric(2, '10 谱学中心荷 (能谱族)', '—', '—',
                      'spiral_model_v16.spectral_central_charge()', None, None,
                      None, None, None, note='未执行 (ctx 无 cspec)。')
        record_guard('W3b 谱学 c 已运行', 'spectral_central_charge',
                     'ctx["cspec"] 存在', False)
        return None

    rows = csp['rows']
    last = rows[-1]
    ident = bool(csp['all_d1_is_R'] and csp['all_d2_is_NS2'])
    ratio_ok = bool(csp['ratio_monotone']) and abs(last['ratio'] - 8.0) < 0.08
    c_ok = csp['c_devmax_pct'] < 1.0

    measured = {
        'sp_c_L_list': csp['L_list'],
        'sp_c_e_inf': csp['e_inf'],
        'sp_c_e_inf_dev': csp['e_inf_dev'],
        'sp_c_ratio_first': rows[0]['ratio'],
        'sp_c_ratio_L16': last['ratio'],
        'sp_c_ratio_pred': last['ratio_pred'],
        'sp_c_ratio_monotone': csp['ratio_monotone'],
        'sp_c_all_identified': ident,
        'sp_c_c_lo': csp['c_lo'], 'sp_c_c_hi': csp['c_hi'],
        'sp_c_c_L16': last['c_spec'],
        'sp_c_devmax_pct': csp['c_devmax_pct'],
        'sp_c_v_L16': last['v_meas'],
        'sp_c_renyi_ref_pct': 7.97,     # v14 说明 §7.4 登记的 Rényi 族偏差
    }
    record_metric(
        2, '10 谱学中心荷 (能谱族)',
        'c = 12*Delta1*(e_inf*L - E0)/(E1 - E0);  比值 (E2-E0)/(E1-E0) -> 8',
        '用低能能谱的闭式比值给 c=1/2 建第二把尺子 (独立于纠缠谱族)',
        'spiral_model_v16.spectral_central_charge(): 稀疏 eigsh 取 H 头三个'
        f'本征值, L={csp["L_list"]}; e_inf 由同一条自由费米子色散数值积分',
        'Ising CFT (c=1/2): Delta1=1/8 (sigma, R 扇区真空), Delta2=1 (epsilon, '
        'NS 双粒子); e_inf 解析值 -4/pi 仅作校验',
        measured,
        'v14 说明 §7.4: 独立族未建立, Rényi 扫描偏差 7.97%, 记为未通过',
        'E1/E2 与解析识别逐点相符; 比值单调且 L=16 在 8 的 1% 内; c 落在 0.5 的 1% 内',
        bool(ident and ratio_ok and c_ok),
        note=f"**算符内容识别**: E1 逐点等于 R 扇区真空 (Delta1=1/8), E2 逐点等于 "
        f"NS 双粒子 (Delta2=1) —— 这条不含 c 也不含 v, 是全项里最硬的一条。"
        f" **v 无关比值**: (E2-E0)/(E1-E0) = {rows[0]['ratio']:.4f} (L={rows[0]['L']}) "
        f"-> {last['ratio']:.4f} (L={last['L']}), 单调趋 8。"
        f" **c 读数**: L={rows[0]['L']}..{last['L']} 落在 "
        f"[{csp['c_lo']:.5f}, {csp['c_hi']:.5f}], 最大偏差 "
        f"{csp['c_devmax_pct']:.3f}% (对照 Rényi 族的 7.97%)。"
        f" **边界一**: 读数**不随 L 单调收敛**, L={last['L']} 反而是最差点, 残差是"
        f" O(1/L) 有限尺寸修正 (用解析值代入结果逐位相同, 故非数值误差) —— "
        f"只能说'相容到 {csp['c_devmax_pct']:.2f}%', 不能说'外推到 0.5'。"
        f" **边界二**: 函数含 Delta1=1/8, 所以这条是**自洽性**而不是无假设测量; "
        f"与纠缠谱族同等带假设 (后者带 Calabrese-Cardy 形式)。真正独立的是数据与"
        f"函数形式。它的强度被前提封顶, 故其通过不应与 (a) 同级读。",
        source_id='W3b_spectrum', claim_id='2.10')
    record_guard('W3b 算符内容识别 (E1=sigma, E2=epsilon)', 'spectral_central_charge',
                 f"E1 逐点 == R 扇区真空 {csp['all_d1_is_R']}; "
                 f"E2 逐点 == NS 双粒子 {csp['all_d2_is_NS2']}",
                 ident,
                 note='守的是整项的**解释前提**: 低能端确实是 Ising 算符内容。'
                      '这条不含 c 也不含 v, 是 W3b 里唯一不依赖被测量本身的守卫。')
    record_guard('W3b v 无关比值单调趋 8', 'spectral_central_charge',
                 f"单调={csp['ratio_monotone']}, L={last['L']} 比值="
                 f"{last['ratio']:.4f} (|x-8|<0.08)",
                 ratio_ok,
                 note='比值只含 Delta2/Delta1, 不含 c 也不含 v。'
                      '**v16·B3 口径订正**: 它与下方的 "谱学 c 与 0.5 相容" **同源** '
                      '(同一 source_id=W3b_spectrum —— 同一次 eigsh 的三个本征值, '
                      '外加同一个 Delta1=1/8 前提), 故**不得**再写成 '
                      '"独立佐证/互相印证"。它真正的效力是**前提检查**: 证明 '
                      '"Delta1=1/8" 那个前提本身站得住, 不是一条独立测量。',
                 tests_claims=('2.10',), source_id='W3b_spectrum')
    record_guard('W3b 谱学 c 与 0.5 相容到 1%', 'spectral_central_charge',
                 f"最大偏差 {csp['c_devmax_pct']:.3f}% < 1% "
                 f"(Rényi 族对照 7.97%)",
                 c_ok,
                 note='**强度被前提封顶**: 该读数以 Delta1=1/8 为前提, 属自洽性检查。'
                      '登记为打分项是为了让缺口可见 (比 Rényi 族紧 30 倍), '
                      '不是宣称已无假设地测出 c=1/2。'
                      '**v16·B3 口径订正**: 本条与上方 "v 无关比值单调趋 8" **同源** '
                      '(同一 source_id=W3b_spectrum), 两者**不得**被当作两条独立证据。',
                 tests_claims=('2.10',), source_id='W3b_spectrum')
    print(f"\n  ── v15·W3b 谱学中心荷 (独立族) ──")
    print(f"      E1=sigma(1/8), E2=epsilon(1): "
          f"{'逐点相符' if ident else '**不符**'}")
    print(f"      (E2-E0)/(E1-E0): {rows[0]['ratio']:.4f} -> {last['ratio']:.4f} "
          f"(v 无关, 趋 8)")
    print(f"      c 读数 [{csp['c_lo']:.5f}, {csp['c_hi']:.5f}], 最大偏差 "
          f"{csp['c_devmax_pct']:.3f}% (Rényi 族 7.97%)")
    print(f"      **不随 L 单调收敛** (L={last['L']} 最差) -> 说'相容到 "
          f"{csp['c_devmax_pct']:.2f}%', 不说'外推到 0.5'")
    return measured




# ===========================================================================
# v15 · W3a 面积律 vs 对数律
# ===========================================================================
def metric_area_vs_log_law(alw):
    """
    2.11 面积律 vs 对数律 —— 答审计表 L3 行与接口建议 P0
    ("L3 -> 纠缠熵面积律: 扫描切割位置, 验证 S(l) 形状")。

    **本条的核心是一条负面结果, 照实登记**: P0 按字面做不出来。在 L<=16 上
    "扫切割位置比 S(l) 形状"没有区分力 —— 对数律在 h/J=0.5、1.0、1.5 上
    全都赢过常数模型, 包括深在能隙相的两个点。所以那条接口建议在本工作条件下
    不可兑现, 记成诊断项 (不计入分母), 而不是硬凑一个"通过"。

    真正分得开的是 S(L/2) 随 **L** 的标度, 三条打分守卫都挂在它上面:
      (a) 临界点走对数律: |c_scaling - 0.5| < 0.05 且 R2 > 0.99
      (b) 非临界点走面积律: |c_scaling| < 0.05 于 h/J in {0.5, 1.5}
      (c) |c_scaling| 的峰位恰在 h/J = 1.0
    """
    if not alw:
        record_metric(2, '11 面积律 vs 对数律', '—', '—',
                      'spiral_model_v16.area_vs_log_law()', None, None,
                      None, None, None, note='未执行 (ctx 无 alw)。')
        record_guard('W3a 面积律/对数律已运行', 'area_vs_log_law',
                     'ctx["alw"] 存在', False)
        return None

    v = alw['verdict']
    pA = alw['partA']
    measured = {
        'alw_hj_list': alw['hj_list'], 'alw_L_list': alw['L_list'],
        'alw_c_scaling_crit': v['c_scaling_at_critical'],
        'alw_r2_crit': v['r2_at_critical'],
        'alw_c_scaling_gapped': v['gapped_c_scaling'],
        'alw_hj_at_peak': v['hj_at_c_peak'],
        'alw_crit_logshape': next(p['c_logshape'] for p in pA
                                  if abs(p['hj'] - 1.0) < 1e-12),
        'alw_partA_winners': {str(k): w for k, w in v['partA_winners'].items()},
        'alw_partA_dAIC': {str(p['hj']): p['dAIC_const_minus_log'] for p in pA},
        'alw_partA_has_power': v['partA_has_discriminating_power'],
    }
    record_metric(
        2, '11 面积律 vs 对数律',
        'S(L/2) = (c/3)ln L + b  的 c_scaling; 对照 AIC 的 S(l) 形状检验',
        '临界点走对数律、非临界点走面积律 —— 用纠缠熵对 L 的标度分开两者',
        'spiral_model_v16.area_vs_log_law(): 逐 h/J 逐 L 求基态并算 S(L/2), '
        f'L={alw["L_list"]}; 另在同尺寸上比对数律/常数律的 RMSE 与 AIC',
        '面积律 S->const 与对数律 S~(c/3)ln L 是 1D 有隙/临界两种标准行为',
        measured,
        'v14 只在 h/J=1.0 上假设对数律拟合 c, 从不与面积律对照; '
        '审计表 L3 行与接口建议 P0 把这件事登记为 🔶 未决',
        '临界点 |c_scaling-0.5|<0.05 且 R2>0.99; 非临界点 |c_scaling|<0.05; '
        '峰位在 h/J=1.0',
        bool(v['passed']),
        note=f"**峰位**: |c_scaling| 最大在 h/J={v['hj_at_c_peak']:.2f}。"
        f" **临界点**: c_scaling={v['c_scaling_at_critical']:.4f}, "
        f"R2={v['r2_at_critical']:.4f} —— 同一套拟合在 h/J=0.5 给 "
        f"{v['gapped_c_scaling'].get(0.5, float('nan')):.4f}、h/J=1.5 给 "
        f"{v['gapped_c_scaling'].get(1.5, float('nan')):.4f}, 即'完全不涨'或"
        f"轻微为负, 面积律。 **两族读数交叉检验**: 同一批数据换自变量, "
        f"S(l) 形状拟合给 c={measured['alw_crit_logshape']:.4f}, L 标度给 "
        f"c={v['c_scaling_at_critical']:.4f} —— 两者**不独立** (同一条 S 数据), "
        f"只算自洽。真正独立的第二族是 W3b (能谱)。 "
        f" **边界**: 杠杆臂只有 L: 8->16 (L 的比值 2 倍, 但拟合自变量是 ln L, "
        f"在全 log2 轴上只跨 1.00 个 octave), 故这是**定性区分**"
        f"(平 vs 对数), 不是定量外推; c_scaling 不当第三把精确尺子。")
    record_guard('W3a 临界点走对数律', 'area_vs_log_law',
                 f"h/J=1.0: |c_scaling-0.5|="
                 f"{abs(v['c_scaling_at_critical'] - 0.5):.4f} < 0.05 且 "
                 f"R2={v['r2_at_critical']:.4f} > 0.99",
                 bool(v['ok_crit_log_law']),
                 note='守的是"临界点上纠缠熵确实按对数发散", 用的是 S(L/2) 随 L, '
                      '不假设 Calabrese-Cardy 的 l-形状。')
    record_guard('W3a 非临界点走面积律', 'area_vs_log_law',
                 f"h/J in {{0.5, 1.5}}: |c_scaling| 最大 "
                 f"{max(abs(x) for x in v['gapped_c_scaling'].values()):.4f} < 0.05",
                 bool(v['ok_gapped_area_law']),
                 note='这条才是"面积律"三个字的实质内容: 有隙相里 S(L/2) 不随 L 涨。'
                      '实测 h/J=0.5 处五个尺寸变化 <3e-4。')
    record_guard('W3a |c_scaling| 峰位在临界点', 'area_vs_log_law',
                 f"argmax|c_scaling| 在 h/J={v['hj_at_c_peak']:.2f} (要求 1.0)",
                 bool(v['ok_peak_at_critical']),
                 note='与 F2a 的 c_fit 峰位判据同向但**不同量**: F2a 用的是 S(l) '
                      '形状拟合的 c, 本条用的是 S(L/2) 对 L 的标度。两者都指 1.0。')
    record_guard('W3a 诊断: S(l) 形状检验无区分力 (P0 不可兑现)',
                 'area_vs_log_law',
                 f"对数律在 {v['partA_winners']} 全部 h/J 上都击败常数模型 "
                 f"(dAIC = {measured['alw_partA_dAIC']})",
                 bool(v['partA_has_discriminating_power']), expect_pass=False,
                 note='**故意不计入通过率**: 这是负面事实的可复现证据。审计表 '
                      '接口建议 P0 写的是"扫描切割位置, 验证 S(l) 形状", 但 '
                      'L<=16 时 l 最大只到 8, 而 h/J=0.5 的关联长度约 2~3 格, '
                      'S(l) 从 l≈3 就平了; 带自由 c 的对数形式能把这种快速饱和'
                      '吸收掉, 于是 RMSE 反而比常数小。留着它是为了防止那条'
                      '建议又被当成一条可做的检验加回来。')
    print(f"\n  ── v15·W3a 面积律 vs 对数律 ──")
    print(f"      S(l) 形状检验: 对数律在全部 {sorted(v['partA_winners'])} 上"
          f"都赢 -> **无区分力, P0 不可兑现**")
    print(f"      S(L/2) 对 L 的标度: c_scaling 峰在 h/J="
          f"{v['hj_at_c_peak']:.2f} (临界点 {v['c_scaling_at_critical']:.4f}, "
          f"R2={v['r2_at_critical']:.4f})")
    print(f"      非临界点 c_scaling = "
          f"{ {k: round(x, 4) for k, x in v['gapped_c_scaling'].items()} } -> 面积律")
    print(f"      边界: 杠杆臂仅 L:8->16 (L 比值 2 倍, 但拟合自变量是 ln L, "
          f"log2 上只 1.00 个 octave), 只作定性区分; 且与 l-拟合不独立"
          f" (独立第二族见 W3b)")
    return measured




# ===========================================================================
# v15 · W5 有限尺寸标度 (L 作自变量)
# ===========================================================================
def metric_finite_size_scaling(fss, loop=None):
    """
    2.12 有限尺寸标度 —— 答审计表 P2 行 (那条方向记反了的建议)。

    本条与 W3a 都扫 L, 但问的不是同一件事, 分开登记:
      * W3a: S(L/2) 随 L 是**平**还是**对数** —— 分辨有隙相/临界相;
      * W5:  spiral_loop 那几个读数 (c/xi/gap/E0) **作为 L 的函数**长什么样,
             以及 max_L 那条天花板到底限制了谁。

    **本条最重要的产出是一条方法学事实, 不是一个物理数**: 对 c / E0/L / xi / gap
    这几个读数, 16 不是墙 —— 它们是精确对角化的读数, 而稀疏 ED **没有** 16 这个
    上限 (L=18/20 实测均可)。卡在 16 的是末圈 MERA 交叉校验, 因为 quimb 的
    qtn.MERA 要求 L 为 2 的幂。审计表 P2 写"去掉 max_L 上限使 xi/L 收敛", 方向
    反了: xi/L 本来就在收敛, 发散的是 L 本身 (尺度律 L_t = round(1.0 * xi))。

    四类读数分开报, 不合并成一个"收敛了":
      (a) c(L)     单调下降趋向 1/2 (Ising CFT 的中心荷)
      (b) E0/L     单调上升趋向 -4/pi (临界 TFIM 的解析热力学极限)
      (c) 与 spiral_loop **同 L 同数** —— 两条代码路径用同一套估计量, 若分歧就是 bug
      (d) 每个 L 独立做 Jordan-Wigner 核对 —— 查的是 H 的构造, 不是 eigsh 的收敛

    诚实边界:
      - 这里报的是**有限尺寸**读数, **不做** v->infinity 的外推。L 从 8 到 18 在
        全 log2 轴上只跨 1.17 个 octave, 这点宽度不足以改善标度拟合 —— 扩 L 的
        价值全在"证明 16 不是墙", 不在提高拟合质量。说成后者就是夸大。
      - MERA 可用集 {8,16} 是 **quimb 的接口约束**, 不是我们的物理结果, 所以
        只登记成诊断项, 不计入通过率。
    """
    if not fss:
        record_metric(2, '12 有限尺寸标度', '—', '—',
                      'spiral_model_v16.finite_size_scaling()', None, None,
                      None, None, None, note='未执行 (ctx 无 fss)。')
        record_guard('W5 有限尺寸标度已运行', 'finite_size_scaling',
                     'ctx["fss"] 存在', False)
        return None

    rows = fss['rows']
    Ls = [r['L'] for r in rows]
    c_arr = [r['c'] for r in rows]
    e_arr = [r['E0_per_L'] for r in rows]
    E0_INF = fss['E0_over_L_limit']

    # (c) 与 spiral_loop 逐 L 对照。两者估计量同源, 所以同 L 上必须是同一个数。
    # 残余差来自 eigsh 的 ARPACK 随机起点 (无 v0), 量级 ~1e-12; 阈值放到 1e-6
    # 是给这个抖动留余量, 而不是给"两种口径"留余地 —— 分歧就是 bug。
    loop_rows = (loop or {}).get('rows', [])
    loop_by_L = {r['L']: r for r in loop_rows}
    ov = [L for L in Ls if L in loop_by_L]
    dc_max = max((abs(c - loop_by_L[L]['c']) for L, c in zip(Ls, c_arr)
                  if L in loop_by_L), default=float('nan'))
    dxi_max = max((abs(r['xi'] - loop_by_L[r['L']]['xi'])
                   for r in rows if r['L'] in loop_by_L), default=float('nan'))
    same_ok = bool(ov) and np.isfinite(dc_max) and dc_max < 1e-6 \
        and np.isfinite(dxi_max) and dxi_max < 1e-6

    # (a)(b) 两条独立的单调趋近: 各自都有解析锚点 (1/2 与 -4/pi)
    c_toward_half = bool(fss['c_mono']) and abs(c_arr[-1] - 0.5) < abs(c_arr[0] - 0.5)
    e_toward_limit = bool(fss['E0_over_L_mono']) \
        and abs(e_arr[-1] - E0_INF) < abs(e_arr[0] - E0_INF)
    jw_ok = bool(fss['jw_worst_dev'] < 1e-10)

    measured = {
        'fss_L_list': Ls,
        'fss_c_by_L': {str(r['L']): r['c'] for r in rows},
        'fss_xi_over_L': {str(r['L']): r['xi_over_L'] for r in rows},
        'fss_E0_per_L': {str(r['L']): r['E0_per_L'] for r in rows},
        'fss_entropy_half': {str(r['L']): r['entropy_half'] for r in rows},
        'fss_gap': {str(r['L']): r['gap'] for r in rows},
        'fss_dc_vs_spiral_loop': float(dc_max),
        'fss_dxi_vs_spiral_loop': float(dxi_max),
        'fss_overlap_L': ov,
        'fss_jw_worst_dev': fss['jw_worst_dev'],
        'fss_E0_over_L_analytic': float(E0_INF),
        'fss_mera_admissible_L': fss['mera_admissible_L'],
        'fss_mera_inadmissible_L': fss['mera_inadmissible_L'],
        'fss_cost_total': float(sum(r['cost'] for r in rows)),
    }
    record_metric(
        2, '12 有限尺寸标度',
        'c(L), xi(L), S(L/2), E0/L 逐 L; 与 spiral_loop 同 L 对照; 逐 L JW 核对',
        '把 L 当自变量, 测 spiral_loop 那套读数作为 L 的函数 —— 并判定 max_L 限制了谁',
        'spiral_model_v16.finite_size_scaling(): 逐 L 单独求基态并算全部读数'
        f' (L={Ls}); 估计量与 spiral_loop 逐字同源, 故同 L 必须同数',
        'Ising CFT: c -> 1/2; 临界 TFIM: E0/L -> -4/pi (解析); '
        'Jordan-Wigner 闭合式是与稀疏对角化独立的第二条路径',
        measured,
        'v14 从不把 L 当自变量扫; 审计表 P2 行登记为"去掉 max_L 上限使 xi/L 收敛", '
        '方向记反了',
        'c(L) 单调趋 1/2; E0/L 单调趋 -4/pi; 与 spiral_loop 同 L 差 < 1e-6',
        bool(c_toward_half and e_toward_limit and same_ok and jw_ok),
        note=f"**核心是方法学事实**: 对这几个读数 16 不是墙 —— 稀疏 ED 没有 16 这个"
        f"上限 (L=18/20 实测均可跑), 卡住 16 的是末圈 MERA 要求 L 为 2 的幂。"
        f" 四类读数: (a) c: "
        + " -> ".join(f"{v:.6f}" for v in c_arr)
        + f", 单调趋 1/2; (b) E0/L: "
        + " -> ".join(f"{v:.6f}" for v in e_arr)
        + f", 单调趋 -4/pi={E0_INF:.6f}; (c) 与 spiral_loop 在 L="
        f"{ov} 上 dc_max={dc_max:.2e}, dxi_max={dxi_max:.2e} —— 同源估计量的"
        f"必然结果, 不是新信息, 但它是『两条路径没跑偏』的可复现证据; "
        f"(d) JW 逐 L 最大偏差 {fss['jw_worst_dev']:.1e}。 "
        f" **边界**: 杠杆臂 L {Ls[0]}->{Ls[-1]} 在全 log2 轴上只跨 "
        f"{float(np.log(Ls[-1] / Ls[0]) / np.log(2.0)):.2f} 个 octave —— "
        f"**不足以改善标度拟合**, 扩 L 的价值全在『16 不是墙』这一条。"
        f" 本函数**不做** v->infinity 外推。")
    record_guard('W5 c(L) 单调下降趋向 1/2', 'finite_size_scaling',
                 f"c: " + " -> ".join(f"{v:.6f}" for v in c_arr)
                 + f"; 单调={fss['c_mono']}, 末点离 1/2 距 "
                   f"{abs(c_arr[-1] - 0.5):.6f} < 首点 {abs(c_arr[0] - 0.5):.6f}",
                 bool(c_toward_half),
                 note='Ising CFT 的中心荷是 1/2, 这是解析锚点, 不是我拟合出来的。'
                      '单调性是关键: 只报末点值会被"恰好接近"蒙混过去。')
    record_guard('W5 E0/L 单调上升趋向 -4/pi', 'finite_size_scaling',
                 f"E0/L 末点 {e_arr[-1]:.6f} 偏离 -4/pi={E0_INF:.6f} "
                 f"{abs(e_arr[-1] - E0_INF):.2e}; 单调={fss['E0_over_L_mono']}",
                 bool(e_toward_limit),
                 note='-4/pi 是临界 TFIM 基态能量的**解析**热力学极限, 与本项目'
                      '任何拟合无关, 所以它是对 H 与标度律的第二重独立锚点。')
    record_guard('W5 与 spiral_loop 同 L 同数', 'finite_size_scaling',
                 f"重叠 L={ov}: dc_max={dc_max:.2e}, dxi_max={dxi_max:.2e} "
                 f"(要求 < 1e-6)",
                 bool(same_ok),
                 note='两条代码路径写了两遍同一套估计量 (exact_ground_state / '
                      'fit_central_charge / boundary_correlation_length / 半链 '
                      'SVD)。同 L 上若不同, 就是其中一条跑偏了 —— 这是**分歧'
                      '检测**, 不是两个独立测量。残余 ~1e-12 来自 eigsh 的 '
                      'ARPACK 随机起点 (没传 v0), 不是口径差。')
    record_guard('W5 逐 L Jordan-Wigner 独立核对', 'finite_size_scaling',
                 f"max|E0_eigsh - E0_jw| = {fss['jw_worst_dev']:.1e} < 1e-10 "
                 f"于 L={Ls}",
                 bool(jw_ok),
                 note='JW 闭合式与稀疏对角化是两条独立路径。eigsh 只回答"给定'
                      '这个 H 最小本征值是多少", 对 H 的位序约定/周期键/场符号'
                      '一句话都没有; JW 能测出 eigsh 测不出的东西。注意这条'
                      'spiral_loop 没做 —— 它是 W5 新增的。')
    record_guard('W5 诊断: MERA 那条腿只接受 2 的幂', 'finite_size_scaling',
                 f"quimb MERA 可用 L={fss['mera_admissible_L']}, "
                 f"不可用 L={fss['mera_inadmissible_L']}",
                 bool(fss['mera_admissible_L'] == [L for L in Ls if L & (L - 1) == 0]),
                 expect_pass=False,
                 note='**故意不计入通过率**: 这是 quimb 的接口约束 (qtn.MERA 要求 '
                      'L 为 2 的幂), 不是我们的物理结果, 拿它算通过率是给自己'
                      '发奖。登记它是为了钉住一条事实: 审计表 P2 那条"去掉 '
                      'max_L 上限"改不动天花板 —— 天花板不在算力上, 在末圈 '
                      'MERA 交叉校验上。')
    print(f"\n  ── v15·W5 有限尺寸标度 ──")
    print(f"      L = {Ls} (逐 L 单独求基态, 不经过尺度律)")
    print(f"      c(L)   : " + " -> ".join(f"{v:.6f}" for v in c_arr)
          + f"  (单调趋 1/2 = {fss['c_mono']})")
    print(f"      E0/L   : " + " -> ".join(f"{v:.6f}" for v in e_arr)
          + f"  (单调趋 -4/pi = {fss['E0_over_L_mono']})")
    print(f"      与 spiral_loop 同 L 对照: L={ov}, dc_max={dc_max:.2e}, "
          f"dxi_max={dxi_max:.2e} -> {'同数' if same_ok else '**分歧**'}")
    print(f"      逐 L JW 核对: 最大偏差 {fss['jw_worst_dev']:.1e}")
    print(f"      MERA 可用集 {fss['mera_admissible_L']} (其余 L 无 MERA 读数)")
    print(f"      **结论**: 16 不是这几个读数的墙 —— 卡在 16 的是末圈 MERA "
          f"要求 L 为 2 的幂;")
    print(f"                审计表 P2 那条方向要反过来读 (xi/L 在收敛, L 在发散)")
    return measured




# ===========================================================================
# v15 · W6 负对照台账 (逐层)
# ===========================================================================
# 审计表 P2 行: 「负对照系统化 (逐层) | 全局 | 可证伪框架 | 整体可信度↑」。
#
# 这张表**不新增任何检查**。它只做两件事:
#   1. 把散在各层的负对照/交叉检验按**七个物理阶段**归位 (此前没有层标,
#      只有 F1/F2a/F5/F6/F7 这套按"批次"编的号, 读者无法判断哪层被覆盖);
#   2. 如实报出**哪几层没有负对照** —— 这是负面事实, 不许用"整体可信"糊过去。
#
# 层号口径**与 F5 一致** (F5-L2 / F5-L3 / F5-L5 / F5-L6a / F5-L6b / F5-L7):
#   L1 空无基底等权叠加蕴潜能     L2 量子涨落扰动对称性破缺
#   L3 自指的动力学涌现           L4 自指画定边界投影出全息
#   L5 全息自然展开生成了时空     L6 时空提供梯度孕育显生命
#   L7 生命涌现意识自照见空无
#
# 什么才算"负对照" (判据比"有个检查"严得多): 它必须能**在结论不成立时失败**。
#   kind='推离'   把参数推离成立区, 要求框架**正确失败** (F2a 最强: 若临界性
#                 不是事实, c_fit 的峰就不会落在 h/J=1.0);
#   kind='零模型' 与零模型族比百分位, 双侧 [5%,95%] —— 两个方向的异常都要报
#                 (F6: 若 MERA 的负曲率只是随机图性质, 它会落进对照分布);
#   kind='二路径' 同一个量用两条来源不同的路径算出来对表 (F5: 若实现与它声称
#                 的解析式不符, 它就会失败);
#   kind='上界'   声称某个无量纲量 <= 1, 可以越界。
#
# **不算负对照的** (如实排除, 不冒充): 单纯的正性/有限性/非空检查 —— 例如
# "L4 纠缠熵为正"、"A 段产出有限值"。这类几乎不可能失败, 不具备可证伪性,
# 所以不列进本表。它们仍以普通守卫的身份留在通过率里, 只是不在这里充数。
#
# **也不引用"XX 负对照已运行"这类上报守卫** (v15 全流程实测修正, 2026-09-19):
# 它们在"该对照未执行"的**提前返回分支**里以 ok=False 注册, 对照真跑起来时
# 根本不会进 GUARDS (例: metric_hj_negative_control 在 hj 为空时 register 后
# 直接 return)。把这种名字写进本表, 会让完整性检查在"对照正常执行"这个**健康
# 情形下失败**、在"对照缺失"时反而通过 —— 判据整个反了。首次全流程实测正是
# 三条这样的引用被报成幻影。所以本表只引用**对照本身**的守卫。
_NEG_CTRL_TABLE = (
    ('L1', '空无基底等权叠加蕴潜能', 'B1 · 局部等权态转动推离', '推离',
     ('B1 推离: 转动 L1 局部态后重叠跌破 0.9',),
     '已补',
     'v16·B1 补上。此前 L1 **没有到下游的因果通路** (equal_state 整仓未被取用, '
     'dim_full 只被 print), 所以推离它不改变任何读数 —— 负对照造不出来, 这是'
     '"缺口"与"回头再做"的区别。B1 给 L1 建了第一条真实连线 '
     '(derive_L1_void_to_chain: 局部等权态张成 L 个自旋的域态, 再与 h/J=32 的'
     '精确基态比重叠), 推离才有可改变的对象。**口径限制**: 推离守卫的 overlap '
     '由被转动的态直接算出, 标定的是**灵敏度尺度**, 不是独立的物理证伪; 同层'
     '另有主判据 (单调上升 + h/J=32 时 > 0.99) 与维数守卫, 一并计分。'),

    ('L2', '量子涨落扰动对称性破缺', 'F2a · h/J 推离临界点', '推离',
     ('F2a 主判据: c_fit 峰在临界点',
      'F2a 次判据: S(L/2) 峰位在窗口内', 'F2a 区分力: 非临界点正确失败',
      'F2a 诊断: gap 单调 (不作判据)'),
     '若"临界模拟"不是事实, 把 h/J 推离 1.0 后 c_fit 的峰就不该落在 1.0, '
     '且框架应当在非临界点**正确失败** —— 后者比"某个数小"强得多。'
     '同层另有 F5-L2 (eigsh vs JW 闭合式) 与 F1 跨边界 twist (同族)。'),

    ('L2', '量子涨落扰动对称性破缺', 'F5-L2 · eigsh vs JW 闭合式', '二路径',
     ('F5-L2 基态能量与 JW 闭合式一致',),
     'eigsh 只回答"给定这个 H 最小本征值是多少", 对 H 的位序约定/周期键/'
     '横场符号一句话都没有; JW 是第二条独立路径, 能测出 eigsh 测不出的东西。'),

    ('L2', '量子涨落扰动对称性破缺', 'F1 · 跨边界 θ=π twist', '二路径',
     ('F1 同族: 跨边界条件已运行 (PBC vs θ=π twist)',
      'F1 twist 保持实矩阵 (可喂 float MERA)',
      'F1 边界条件确实换了态 (E0 与 S(L/2) 都变)',
      'F1 跨边界条件相对差 <= 5%'),
     '**同族, 不是独立族**: E0/L 与 S(n) 共用同一个 Calabrese-Cardy ansatz, '
     '换的只是态与边界条件。所以它检验"c 对边界条件稳不稳", 不检验 c 本身。'),

    ('L3', '自指的动力学涌现', 'W1 · 冻结权重对照', '推离',
     ('W1 可达 N 上变体 A 不秩1化',
      'W1 诊断: 冻结对照并非"永不收敛"'),
     '冻结权重是固定随机映射, 它**不该**表现出自指动力学的那种收敛。'
     '实测 16.0% 的 W 组合仍会收敛 —— 这一条恰恰是负面事实: v14 印的'
     '"冻结对照永不收敛"是错的, 它在非零比例上会收敛。'),

    ('L3', '自指的动力学涌现', 'F5-L3 · 幂迭代残差', '二路径',
     ('F5-L3 幂迭代残差达阈',),
     '断言残差达阈, 可失败。原判据写"与 Lanczos 100/200/500 步无关", '
     '但 eigsh 没有步数参数 —— 那条零分辨力, 已换掉。'),

    ('L3', '自指的动力学涌现', 'V6/V7 · 权重自由/归一变体秩1化', '推离',
     ('V6 权重自由变体秩1化', 'V7 权重归一变体秩1化'),
     '两个变体在**设计上就应当失败** (V6 尤其: 派生 N 之后变体 A 收敛到 '
     'x*=0, 权重失去方向)。它们以 expect_pass=False 登记, 不计入通过率。'),

    ('L4', '自指画定边界投影出全息',
     'W12 · 键维向下推离 (压到面积律下界以下)', '推离',
     ('L4 键维向下对照: 压到面积律下界以下下游退化',),
     '把 chi 压到面积律下界 exp(S) 以下, 要求下游 MERA 复现中心荷的能力'
     '**正确失败**。若界下与界上的 eps_c 一样小, 这条下界就不约束任何东西, '
     'derive_L4_bond_dimension 就是装饰 —— 那是对 L4 的否证。',
     '本层的 sanity 守卫 G2 ("L4 纠缠熵为正") 是**正性检查**, 几乎不可能失败, '
     '按本表的判据仍**不算**负对照 (它继续以普通守卫的身份留在通过率里, '
     '只是不在这里充数)。F1 的 MERA 重叠 >= 0.99 是交叉检验且已归到 L2 '
     '(它换的是边界条件)。面积律 -> 键维下界 (derive_L4_bond_dimension) '
     '这条派生原**没有任何对照**, W12 补上。'
     '**这条只验必要性方向**: 界不成立时下游应退化; **不验充分性** —— '
     '函数 docstring 本就写明它是"必要非充分条件"。'),

    ('L5', '全息自然展开生成了时空', 'F6 · 零模型对照族 (三类 x 3 实例)', '零模型',
     ('F6a 对照几何族已建立',
      'F6b MERA 体几何处于对照分布内 (非统计异常)',
      'F6c 诊断: MERA 并非比全部对照更负'),
     '单张同规模随机图只能回答"比随机图更负吗"; 三类对照 (度分布/聚类/'
     '树的成分) 各 3 实例给百分位, 用**双侧** [5%,95%] 判定 —— 两个方向的'
     '统计异常都报, 不能只报顺手的那个。本层是七层里负对照最强的一层。'),

    ('L5', '全息自然展开生成了时空', 'F5-L5 · Forman 在环 C_n 上为 0', '二路径',
     ('F5-L5 Forman 在环 C_n (n>=4) 上为 0',),
     '解析预期是 0, 可失败。**n=3 已排除** —— C_3 就是 K_3, augmented 的 '
     '+3 项把它顶到 +3; 排除理由写在守卫 note 里, 不是把不利点删掉。'),

    ('L6', '时空提供梯度孕育显生命', 'F5-L6 · 两条离散恒等式', '二路径',
     ('F5-L6a 离散拉普拉斯零和', 'F5-L6b Gray-Scott 精确离散平衡恒等式',
      'F5 诊断: Gray-Scott 的 u+v 不是守恒量'),
     '周期域上 Laplacian 零和 + 同一更新式的精确离散平衡。'
     '"u+v 不是守恒量"是**诊断项**: 它是负面事实 (Gray-Scott 没有守恒律), '
     '留着是防止有人把 u+v 当守恒量用。'),

    ('L6', '时空提供梯度孕育显生命', 'F7 · 特征波长的退化输入校验', '推离',
     ('F7 _char_scale 输入校验生效',),
     '喂空场/平坦场这类退化输入, 要求**不静默产出"看起来合理"的伪值**。'
     '这是唯一一条把"输入坏掉"当对照的 —— 其余负对照都假设输入是好的。'),

    ('L7', '生命涌现意识自照见空无', 'F5-L7 · 熵比与等权度上界', '上界',
     ('F5-L7 熵比与等权度满足上界',),
     '断言熵比 <= 1 且等权度 <= 1, 可越界。**本层是最薄的一层**: 只有这一条, '
     '而且是上界不是推离。W4 新增的 Jacobian 谱半径 (l7_rho_jac) 目前'
     '**只报告、无守卫** —— 登记在此, 不假装它是负对照。'),
)




def metric_l1_void_link(l1, push):
    """
    2.14 L1 空无 -> 链的下游连线 (v16·B1)

    答 v15·W11 诊断项点名的缺口: L1 此前**没有到下游的因果通路**, 所以
    "推离 L1 应当改变下游读数"这个负对照**造不出来** —— 推离一个没人取用的量
    不改变任何东西。B1 给 L1 建了第一条真实连线, 本函数把它的判据登记为
    **计分**守卫, 并把 W6 台账里 L1 那格 '缺' 补上。

    **口径边界** (v16_plan.md §2.4, 三句话不许并成一句):
      * "等权叠加大态 = h/J -> inf 的基态"是**数学事实**;
      * 把它叫做 "L1 到 L2 的因果连线"是**口径选择** —— 这里登记的是
        **可检验的接线**, 不是已证明的物理因果;
      * h/J=32 是有限值, 所以 overlap_inf 只是逼近 1, 差值如实报出。
    """
    L = int(l1['L'])
    curve = l1['overlap_curve']
    ov_inf = float(l1['overlap_inf'])
    main_ok = bool(l1['monotone']) and ov_inf > 0.99
    push_ok = push['theta_star'] is not None

    rec = record_metric(
        1, '14 L1 空无到链的下游连线 (重叠曲线)',
        '|<void|gs(h/J)>|^2, void = stage1_void 的等权局部态张成 L 个自旋',
        '给 L1 建第一条可检验的下游连线, 并让 dim_full 第一次被真消费 '
        '(v15 里它只被 print, 从不与 2**L 比对)',
        'spiral_model_v16.derive_L1_void_to_chain / l1_void_pushaway',
        'W6 台账 L1 格 = 缺; W11 诊断: L1 无到下游的因果通路, 负对照 0 条',
        {'dim_full': l1['dim_full'], 'dim_match': l1['dim_match'],
         'overlap_curve': [[h, o] for h, o in curve],
         'overlap_inf': ov_inf, 'one_minus_overlap_inf': 1.0 - ov_inf,
         'monotone': l1['monotone'],
         'pushaway_theta_star': push['theta_star'],
         'pushaway_overlap_at_star': push['overlap_at_star']},
        'L1 无任何下游连线 (equal_state 整仓未被取用), dim_full 只被 print',
        'overlap 对 h/J 单调上升, 且 h/J=32 时 > 0.99',
        bool(main_ok),
        note='**这条量的是接线, 不是因果**: "|+>^{tensor L} 是 h/J->inf 的基态"'
             '是数学事实; 真正被检验的是"L1 的态确实进了下游计算"这条路径 —— '
             'B1 之前它根本不存在。有限 h 的诚实标注: '
             f'1 - overlap_inf = {1.0 - ov_inf:.3e} (> 0)。')

    record_guard(
        'B1 主判据: L1 域态重叠对 h/J 单调上升且 h/J=32 时 > 0.99',
        'derive_L1_void_to_chain',
        'overlap = ' + " -> ".join(f"{h:g}:{o:.6f}" for h, o in curve)
        + f"; 单调上升={l1['monotone']}, h/J=32 时 {ov_inf:.6f} > 0.99",
        bool(main_ok),
        note='判据与 v16_plan.md §2.3 B1-1 一致。**有限 h**: h/J=32 不是无穷, '
             f'所以 overlap_inf = {ov_inf:.6f} < 1, 差 {1.0 - ov_inf:.3e} '
             '是 O((J/h)^2), 不是计算误差。')

    record_guard(
        'B1 维数: dim_full 与 2**L 相符 (首次被真消费)',
        'derive_L1_local_to_full',
        f"dim_full = {l1['dim_full']} == 2**{L} = {2 ** L}",
        bool(l1['dim_match']),
        note='v16·B1 之前 dim_full 只被 print (spiral_model_v16.main 的 [L1] '
             '打印行), 从不与实际链维数比对。B1 把它接到 derive_L1_void_to_chain '
             '张成的实际维数上 —— 这是 v15 审计表 L1 行第 3 条"张量积结构"'
             '由"否"转"是"的依据。')

    record_guard(
        'B1 推离: 转动 L1 局部态后重叠跌破 0.9',
        'l1_void_pushaway',
        '推离曲线 '
        + ", ".join(f"theta={t:g}:{o:.6f}" for t, o in push['curve'])
        + f"; 首个跌破 0.9 的 theta* = {push['theta_star']} "
          f"(overlap = {push['overlap_at_star']})",
        bool(push_ok),
        note='**这是 L1 的第一条负对照**, 填 W6 台账 L1 的缺口。检验的是'
             '"推离 L1 会改变下游读数" —— 若 L1 对下游无作用, 转它不该改变'
             '任何东西, 这条就失败。**对计划的有意偏离**: 计划写"绕随机轴", '
             '这里用**固定 y 轴** R_y —— 随机轴会让校准点每次运行都不同, '
             '不可复现; 固定轴同样能测出推离是否有效。**口径限制**: 本条的'
             'overlap 由被转动的态直接算出, 所以它标定的是**灵敏度尺度**, '
             '不是一条独立的物理证伪。')

    return rec




def metric_negative_control_ledger():
    """
    2.13 负对照台账 (逐层) —— 答审计表 P2 行「负对照系统化 (逐层)」。

    本函数**不新增任何检查**, 只把已有的负对照/交叉检验按七个物理阶段归位,
    并如实报出哪几层没有。所以它的产出是一张**覆盖表**, 不是一个新结论。

    两条判定分开:
      (a) **计分**: 台账与守卫注册表一致 —— 表里引用的每条守卫名都真实存在于
          GUARDS, 且七层各出现一次。这不是物理结论, 是**台账本身的完整性**:
          没有它, 台账可以写成宣传册 (引用不存在的守卫、或漏掉某层)。
      (b) **诊断** (expect_pass=False): 覆盖了几个层。**故意不计分** ——
          "逐层"是审计表提的目标, 不是模型的性质; 把它算进通过率等于拿一张
          待办清单给自己打分。v15 实测 L1/L4 无负对照、L7 仅一条上界, 这个
          缺口是那一次的**真实产出**; v16 后 L4 由 W12、L1 由 B1 补上, 覆盖
          7/7, 但**这仍只是一张覆盖表**, 不是新的物理结论。
    """
    layers = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7']
    layer_names = {r[0]: r[1] for r in _NEG_CTRL_TABLE}

    # 台账自身完整性: 七层都在, 且引用的守卫名全部真实存在
    missing_layers = [L for L in layers if L not in layer_names]
    phantom = sorted({g for r in _NEG_CTRL_TABLE for g in r[4]
                      if g not in {x['name'] for x in GUARDS}})
    ledger_ok = (not missing_layers) and (not phantom)

    # 逐层统计: 有几个层有 >=1 条负对照 (kind != '缺')
    per_layer = []
    for L in layers:
        recs = [r for r in _NEG_CTRL_TABLE if r[0] == L and r[5] != '缺']
        gnames = [g for r in recs for g in r[4]]
        grecs = [x for x in GUARDS if x['name'] in gnames]
        per_layer.append({
            'layer': L, 'name': layer_names.get(L, '?'),
            'n_controls': len(recs),
            'kinds': [r[3] for r in recs],
            'n_guards': len(grecs),
            'n_guard_failed': sum(1 for x in grecs if not x['ok']),
        })
    covered = [p for p in per_layer if p['n_controls'] > 0]
    gaps = [p['layer'] for p in per_layer if p['n_controls'] == 0]

    measured = {
        'nc_per_layer': {p['layer']: p['n_controls'] for p in per_layer},
        'nc_kinds_by_layer': {p['layer']: p['kinds'] for p in per_layer},
        'nc_guards_by_layer': {p['layer']: p['n_guards'] for p in per_layer},
        'nc_n_layers_covered': len(covered),
        'nc_n_layers_total': len(layers),
        'nc_gap_layers': gaps,
        'nc_phantom_guards': phantom,
        'nc_missing_layers': missing_layers,
        'nc_total_controls': sum(p['n_controls'] for p in per_layer),
        'nc_kind_counts': {k: sum(1 for r in _NEG_CTRL_TABLE if r[3] == k)
                           for k in ('推离', '零模型', '二路径', '上界')
                           if any(r[3] == k for r in _NEG_CTRL_TABLE)},
    }
    record_metric(
        2, '13 负对照台账 (逐层)',
        '七个物理阶段各自的负对照条数 / 类型 / 守卫存活情况',
        '把散在各层的负对照按阶段归位, 并**如实报出哪几层没有** —— '
        '答审计表 P2「负对照系统化 (逐层)」',
        'spiral_metric_v16._NEG_CTRL_TABLE 与 GUARDS 注册表对表; '
        '层号口径与 F5 一致 (F5-L2/L3/L5/L6a/L6b/L7)',
        '审计表 P2 行把"负对照系统化"登记为 🔶 未决; v14 的负对照按'
        '批次 (F1/F2a/F5/F6/F7) 编号, 没有层标, 无法判断哪层被覆盖',
        measured,
        'v14 无任何逐层负对照台账 (只有按批次编号的检查清单)',
        '台账引用的守卫名全部存在, 且七层各出现一次',
        bool(ledger_ok),
        note='**负对照的判据比"有个检查"严**: 它必须能在结论不成立时失败。'
             '所以单纯的正性/有限性检查 (如"L4 纠缠熵为正") 被**排除在外** —— '
             '它们仍以普通守卫留在通过率里, 只是不在这里充数。'
             f" 实测覆盖 {len(covered)}/{len(layers)} 层, "
             f"**无负对照的层: {gaps}**, 共 {measured['nc_total_controls']} 条对照 "
             f"(类型分布 {measured['nc_kind_counts']})。"
             ' 最强的是 L5 (F6 零模型族: 三类对照 x 3 实例, 双侧 [5%,95%]); '
             '最薄的是 L7 (仅一条上界, 且 W4 新增的 Jacobian 谱半径只报告无守卫)。'
             ' L4 的缺口已由 W12 的「推离」补上 (只补必要性方向); '
             'L1 的缺口由 v16·B1 补上 —— W11 当初查清了它造不出来的原因 '
             '(L1 到下游没有因果通路), B1 因此先给 L1 建了第一条真实连线, '
             '推离才有可改变的对象。**口径限制**: 那条推离的 overlap 由被转动'
             '的态直接算出, 标定的是灵敏度尺度, 不是独立的物理证伪。')
    record_guard('W6 台账与守卫注册表一致', 'negative_control_ledger',
                 f"台账引用的守卫名全部存在于 GUARDS (幻影引用 {phantom or '无'}), "
                 f"七层各出现一次 (缺失 {missing_layers or '无'})",
                 bool(ledger_ok),
                 note='这不是物理结论, 是**台账本身的完整性**: 没有它, 台账可以'
                      '写成宣传册 —— 引用不存在的守卫、或整层漏掉。'
                      '校验对象是 GUARDS 这个真实注册表, 不是表里自报的数。')
    record_guard('W6 诊断: 负对照逐层覆盖', 'negative_control_ledger',
                 f"有负对照的层: {[p['layer'] for p in covered]} "
                 f"= {len(covered)}/{len(layers)}; 缺口 {gaps}",
                 bool(len(covered) == len(layers)), expect_pass=False,
                 note='**故意不计入通过率**: "逐层"是审计表提的目标, 不是模型的'
                      '性质。把它算进通过率等于拿一张待办清单给自己打分。'
                      '实测缺口 (L1, 以及 W12 之前的 L4) 是真实产出, '
                      '不该被百分比掩掉。')

    print(f"\n  ── v15·W6 负对照台账 (逐层) ──")
    print(f"      {'层':<4}{'阶段':<24}{'对照数':<7}{'类型':<26}守卫")
    for p in per_layer:
        kinds = ','.join(p['kinds']) if p['kinds'] else '**无**'
        print(f"      {p['layer']:<4}{p['name']:<24}{p['n_controls']:<7}"
              f"{kinds:<26}{p['n_guards']}")
    print(f"      覆盖 {len(covered)}/{len(layers)} 层; 缺口 {gaps}")
    print(f"      台账完整性: 幻影引用 {phantom or '无'}, 缺失层 {missing_layers or '无'}")
    print(f"      判据: 只有能**在结论不成立时失败**的检查才算负对照 —— "
          f"正性/有限性检查已排除")
    return measured




# ===========================================================================
# v14 · F6 涌现几何的对照鲁棒性
# ===========================================================================
def _pctile_rank(value, sample):
    """value 在 sample 中的百分位 (sample 里 <= value 的比例 x 100)。"""
    s = np.asarray([x for x in sample if np.isfinite(x)], dtype=float)
    if s.size == 0 or not np.isfinite(value):
        return float('nan')
    return float(100.0 * np.mean(s <= value))




def metric_geometry_robustness(s5):
    """
    2.8 涌现几何的对照鲁棒性 —— MERA 体几何落在零模型对照分布的哪个位置。

    为什么需要这一项: 原来只有一个对照 (同规模随机图, 单个实例), 而结论
    直接读成"MERA 是负曲率"。单实例比较有两个问题 —— 它给不出"多负算异常"
    的尺度, 也没法控制"图接近树"这个前提。三类对照各控一个维度:
      gnm  同 |V| 同 |E|          -> 控规模, 不控度分布
      ws   平均度与 MERA 图对齐    -> 控度分布, 同时带高聚类
      tree 平衡二叉树              -> 控"接近树"这个前提 (零三角形)
    指标取 MERA 在这 9 个实例里的**百分位**, 于是"异常/不异常"变成一个有序
    量, 而不是一次单点比较的措辞。

    判据 (F6b) 是双边界 [5%, 95%]: 落在任一端都是统计异常 —— 更负说明几何
    比零模型更"双曲", 更不负说明它比零模型更平坦。两个方向都必须报, 不能
    只挑有利的那一侧说。

    诚实边界: 这是**零模型对照**, 不是同一物理系统的不同实现。百分位落在
    区间内只说明"MERA 的这类曲率读数在同类规模的图里不特别", 它**不**支持
    "MERA 就是全息对偶"; 反过来, 落在区间外也只说明这张图在那一个测度上
    偏了, 不构成对涌现几何整体主张的否证。
    """
    if not s5 or not s5.get('controls'):
        record_metric(2, '8 涌现几何的对照鲁棒性', 'MERA 在零模型对照分布中的百分位',
                      '涌现几何的曲率读数在同类规模图里是否异常',
                      'spiral_model_v16.geometry_controls()', '三类零模型对照族',
                      None, None, '未运行 (ctx 无对照族)', None,
                      note='未执行, 无法判定。')
        record_guard('F6 对照几何族已运行', 'collect_metrics_v16',
                     'ctx["s5"]["controls"] 非空', False)
        return None

    ctrls = s5['controls']
    mb = s5['mera_bulk']
    fams = sorted({c['family'] for c in ctrls})

    def _fam_vals(key, fam):
        return [c[key] for c in ctrls if c['family'] == fam]

    f_all = [c['forman_mean'] for c in ctrls]
    o_all = [c['or_mean'] for c in ctrls]
    f_mera, o_mera = float(mb['forman_mean']), float(mb['or_mean'])
    f_pct = _pctile_rank(f_mera, f_all)
    o_pct = _pctile_rank(o_mera, o_all)

    # 逐族均值 —— 让"哪一族把 MERA 拉到哪里"看得见, 而不是只留一个总分位
    by_fam = {f: {'forman_mean': float(np.mean(_fam_vals('forman_mean', f))),
                  'or_mean': float(np.nanmean(_fam_vals('or_mean', f))),
                  'n_triangles': int(np.mean(_fam_vals('n_triangles', f))),
                  'd_bar': float(np.mean(_fam_vals('d_bar', f)))}
              for f in fams}

    # "MERA 比全部对照更负"这个升级主张是否成立 (在两个测度上都要求成立)
    more_neg_all = bool(f_mera < min(f_all) and o_mera < min(o_all))

    measured = {
        'mera_forman_mean': f_mera, 'mera_or_mean': o_mera,
        'mera_d_bar': float(mb['d_bar']),
        'mera_n_triangles': int(mb['n_triangles']),
        'forman_pctile': f_pct, 'or_pctile': o_pct,
        'n_controls': len(ctrls), 'families': fams,
        'forman_ctrl_min': float(min(f_all)), 'forman_ctrl_max': float(max(f_all)),
        'or_ctrl_min': float(min(o_all)), 'or_ctrl_max': float(max(o_all)),
        'by_family': by_fam,
        'more_negative_than_all': more_neg_all,
        'pctile_band': [5.0, 95.0],
    }
    # 结论边界由**数据派生**成字符串, 不手写 —— 手写的边界会与数字脱钩
    if np.isfinite(f_pct) and np.isfinite(o_pct):
        verdict_txt = (
            f'MERA 的 Forman 均值 ({f_mera:+.3f}) 位于 {len(ctrls)} 个对照实例的'
            f'第 {f_pct:.0f} 百分位, Ollivier 均值 ({o_mera:+.4f}) 位于第 '
            f'{o_pct:.0f} 百分位 —— '
            + ('两个测度都落在 [5%, 95%] 内, 故结论是"MERA 体几何具有负曲率'
               '特征, 但在同类规模的零模型里不是统计异常"。'
               if (5.0 <= f_pct <= 95.0 and 5.0 <= o_pct <= 95.0)
               else '至少一个测度落在端点附近, 该测度上 MERA 相对零模型是'
                    '统计异常, 必须按方向如实报告 (更负 = 更双曲 / 更不负 = 更平坦)。')
            + f' "MERA 比全部对照更负" 这个升级主张实测为 {more_neg_all}。')
    else:
        verdict_txt = '百分位无法计算 (对照或 MERA 的曲率出现非有限值)。'

    record_metric(
        2, '8 涌现几何的对照鲁棒性', 'MERA 在零模型对照分布中的百分位',
        '涌现几何的曲率读数在同类规模图里是否异常 —— 把单点比较换成有序量',
        'spiral_model_v16.geometry_controls(): 三类零模型 x 3 实例, '
        '每图算 forman_ricci(triangles=False) 与 ollivier_ricci(alpha=0)',
        '三类零模型: gnm (同 |V| 同 |E|) / ws (配度小世界) / tree (平衡二叉树)',
        measured,
        '原先只有 1 个对照实例 (同规模随机图), 无百分位, 无族分解',
        'Forman 与 Ollivier 两个测度都落在对照分布的 [5%, 95%] 内',
        bool(5.0 <= f_pct <= 95.0 and 5.0 <= o_pct <= 95.0),
        note=verdict_txt + " 逐族均值: " + "; ".join(
            f"{f}: Forman {by_fam[f]['forman_mean']:+.3f} "
            f"(三角 {by_fam[f]['n_triangles']}, d_bar {by_fam[f]['d_bar']:.2f})"
            for f in fams) + "。"
            f"**诚实边界**: 这是零模型对照, 不是同一物理系统的不同实现; "
            f"百分位不特别**不**支持\"MERA 就是全息对偶\", 落在区间外也**不**"
            f"构成对涌现几何整体主张的否证。三角形数与平均度必须与曲率一起读 "
            f"—— 本工作用的是无三角形式 F=4-deg(u)-deg(v), 它只在图接近树时"
            f"与完整 Forman 等价, 而 MERA 图三角数 = {mb['n_triangles']}。")

    record_guard('F6a 对照几何族已建立', 'geometry_controls',
                 f"{len(ctrls)} 个实例, {len(fams)} 族 {fams}, 每族 3 个",
                 bool(len(ctrls) >= 9 and len(fams) >= 3),
                 note='族数决定能控几个维度 (度分布 / 聚类 / 树的成分); '
                      '实例数决定百分位的分辨率 (9 实例 -> 11% 一档)。')
    record_guard('F6b MERA 体几何处于对照分布内 (非统计异常)',
                 'curvature_report',
                 f"Forman 百分位 {f_pct:.0f}%, Ollivier 百分位 {o_pct:.0f}% "
                 f"(区间 [5%, 95%])",
                 bool(5.0 <= f_pct <= 95.0 and 5.0 <= o_pct <= 95.0),
                 note='**双边界**: 落在哪一端都是异常, 两个方向都必须报 —— '
                      '只报"比随机图更不负"而不报它是否落在端点, 等于用措辞'
                      '代替测量。')
    record_guard('F6c 诊断: MERA 并非比全部对照更负', 'geometry_controls',
                 f"Forman min(对照) = {min(f_all):+.3f} vs MERA {f_mera:+.3f}; "
                 f"Ollivier min(对照) = {min(o_all):+.4f} vs MERA {o_mera:+.4f} "
                 f"-> 两个测度同时更负 = {more_neg_all}",
                 more_neg_all, expect_pass=False,
                 note='**诊断项, 不计入通过率 (本项设计上就该不通过)**: '
                      '"升级"措辞 (MERA 的负曲率比零模型更强) 没有任何数据'
                      '支持, 如实登记为假。留着它是为了防止这个更强的主张'
                      '以后又被写回结论里。')

    # v15·W10: F6 的百分位读的是 MERA 图的曲率, 而那张图**不依赖态** ——
    # 实测换 seed 重建, 边集逐位相同 (见 spiral_model_v16 阶段五)。登记它是
    # 为了把读数的**口径**钉死: 百分位讲的是"二进制 MERA 的接线在同类规模图里
    # 特不特别", 不是"拟合出来的几何特不特别"。这两句话的分量差得很远。
    _bi = s5.get('bulk_seed_invariance') or {}
    if _bi:
        _dep = not bool(_bi.get('same'))
        # 键维这一问是 v15·B1 才加进产物 json 的。旧 json 没有这个键 ——
        # 那时**没测过**, 不能拿 None 当"依赖键维"填进台账 (那是假通过)。
        # 只有键在、且值为假, 才算测得"依赖"。
        _dep_chi = ('chi_same' in _bi) and not bool(_bi['chi_same'])
        record_guard(
            'F6d 诊断: mera_bulk 依赖态或键维 (不是纯线路布局)', 'mera_init',
            f"seed {_bi.get('seeds')} 重建的 MERA 图边集完全相同 = "
            f"{_bi.get('same')} (|E|={_bi.get('n_edges')}); "
            f"chi {_bi.get('chis')} 的边集完全相同 = {_bi.get('chi_same')} "
            f"-> 依赖态 = {_dep}, 依赖键维 = {_dep_chi}",
            bool(_dep or _dep_chi), expect_pass=False,
            note='**诊断项, 不计入通过率 (本项设计上就该不通过)**: 本项问的是'
                 '"这个读数是不是态/键维的函数", 两个实测答案都是**否**, 所以它'
                 '守不住, 如实登记为假。分四点: '
                 '(1) **为什么该问**: F6b/F6c 的百分位、L5 的曲率读数, 都建立在'
                 '"MERA 体几何"这张图上; 若这张图既不随态变、也不随键维变, 那些'
                 '数字就不是"涌现"出来的, 而是**线路布局的解析性质**。'
                 '(2) **实测·态**: 同样 L=32, chi=2, seed 0/1/7 三次构造, 图边集'
                 f"**逐位相同** (|E|={_bi.get('n_edges')}; L=16 时同样相同, "
                 '|E|=43, 由 v15·W10 一次性探测记录)。'
                 '(3) **实测·键维 (v15·B1 补)**: 同样 L=32, seed=0, '
                 f"chi {_bi.get('chis')} 三次构造, 边集"
                 f"**同样逐位相同** (chi_same={_bi.get('chi_same')})。这一问不是"
                 '修辞 —— L4 派生的键维 chi_dev=4 与 L5 实际构造用的 chi=2 **不同**, '
                 '若图随 chi 变, 那就是一条**没被声明的耦合**; 实测不变, 故该耦合'
                 '不存在 (chi 只定指标维度, 不定指标与张量的关联)。'
                 '(4) **代码锚点 + 解析性质**: spiral_model_v16.build_mera_graph '
                 '只读 mera.ind_map 的键值配对, 不看任何张量数值; mera_fit* 系列'
                 '只调 t.modify(data=...), **从不改接线**。所以该读数只是 '
                 '(L, 二进制布局) 的函数; 且 |V| = 2L-2、三角 = 0 都是二进制布局'
                 '的解析值, 不是拟合出来的。'
                 '**边界**: 这不等于"L5 无意义" —— MERA 的几何本就编码在接线里, '
                 '这是该架构的正常性质; 但它把 F6 的百分位从"物理结论"收紧为'
                 '"线路布局的性质", 上游结论的措辞必须按此改。')
        print(f"      [W10+B1] F6d 诊断: mera_bulk 依赖态 = {_dep} "
              f"(seed {_bi.get('seeds')} 边集相同 = {_bi.get('same')}), "
              f"依赖键维 = {_dep_chi} (chi {_bi.get('chis')} 边集相同 = "
              f"{_bi.get('chi_same')}) -> 该读数是线路布局的函数")

    print(f"\n  ── v14·F6 涌现几何的对照鲁棒性 ──")
    print(f"      MERA: Forman {f_mera:+.3f} (三角 {mb['n_triangles']}, "
          f"d_bar {mb['d_bar']:.2f}) | Ollivier {o_mera:+.4f}")
    print(f"      对照 (n={len(ctrls)}): Forman [{min(f_all):+.3f}, "
          f"{max(f_all):+.3f}] | Ollivier [{min(o_all):+.4f}, {max(o_all):+.4f}]")
    print(f"      百分位: Forman {f_pct:.0f}%, Ollivier {o_pct:.0f}% "
          f"-> {'非统计异常' if (5.0 <= f_pct <= 95.0 and 5.0 <= o_pct <= 95.0) else '存在端点异常'}")
    print(f"      升级主张 (比全部对照更负) = {more_neg_all}")
    return measured
