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
"""spiral_model_v16._model.core —— 由 spiral_model_v16.py 拆分。

派生层 (旋钮/因果链台账/L1..L6 派生) + 共用求解器 + 跨阶段复用的测量工具

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


# 本文件位于 v16/_model/ 下, 故须上溯一级才是 v16/ 目录 —— 与拆分前
# spiral_model_v16.py 的 dirname 等值 (不修正的话 _OUTPUT_DIR 会指向
# v16/_model/result, 产物落错地方)。
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# os.makedirs 原是模块级裸语句 (unit_name 认不出名字, 会被跳过) ——
# 并入本单元, 否则产物目录不建、存图时才炸。
_OUTPUT_DIR = _HERE+"/result"
os.makedirs(_OUTPUT_DIR, exist_ok=True)




# ============================================================================
# 派生层 · 因果链旋钮与派生映射
# ============================================================================
#
# 设计原则 (三条, 全部可检查):
#
#   1. 每个下游参数由上游**实测物理量**算出。推导式写在函数的 docstring 里,
#      并且实测值会被打印出来。不允许"看起来像派生"的注释配硬编码的值。
#
#   2. 真正不可约的外部输入集中到下面的 KNOBS 里, 打印时明确标注
#      「旋钮」而不是「派生」。不假装旋钮是派生出来的。
#
#   3. 上游量若退化 (熵为 0、谱太短、派生值超出稳定性上界), 就**拒绝运行
#      并说明原因**, 绝不静默退回硬编码默认值。
#
# 为什么需要第 3 条: 最容易骗人的地方就是"看起来在派生"。曾经算出关联长度
# xi = 19.314, 打印出来, 然后扔掉 —— docstring 却声称阶段六跑在"涌现几何"
# 上。一个从不流向任何地方的中间量, 比没有这个量更糟。所以用 record_link()
# 把每条链的输入/输出都记录下来, 谁没接线一眼可见。

KNOBS = {
    # ---- 尺度 (唯一的自由度: 整条链的尺寸) ----
    'L_chain': 16,        # 阶段二 自旋链长度。16=2^4: 须是 2 的幂 (MERA 库约束)
    'd_local': 2,         # 阶段一 局部维度: 空无 = 单个量子比特 (自旋 1/2)

    # ---- 派生层的阈值与裕度 ----
    'schmidt_tol': 1e-8,  # L3 有效 Schmidt 秩的截断阈值 (p_k > tol 才算数)
    'stability_limit': 0.25,   # 显式 Euler 稳定上界 dt*Du/h^2 <= 1/4

    # ---- 阶段六网格 (L6 的两条约束) ----
    'L_domain': 0.5,      # 旋钮: Gray-Scott 域长 (化学自身的长度单位, 见 L6)
    'n_lambda': 3.0,      # 旋钮: 域内应容纳的斑图波数 (来自 xi/L 的尺度不变性)
    'pts_per_wavelength': 12.0,  # 旋钮: 每个波长至少几个网格点才算"分辨"

    # ---- 阶段六化学参数: 不可约, 与几何无关 ----
    'F': 0.035,           # 旋钮: 进料率
    'k': 0.060,           # 旋钮: 移除率
    'dt': 1.0,            # 旋钮: 时间步 (与稳定性上界联立求得 N)
    'Du': 2e-5,           # 旋钮: u 扩散系数 (经典 Pearson 标度)
    'Dv': 1e-5,           # 旋钮: v 扩散系数

    # ---- 闭环 spiral_loop() ----
    'turns': 6,           # 旋钮: 最多转几圈
    'L0_loop': 8,         # 旋钮: 第 0 圈的系统尺寸
    'max_L_loop': 16,     # 旋钮: 尺寸上限。16=2^4 是 MERA 库的 2 的幂约束, 不是算力
    'scale_knob': 1.0,    # 旋钮: L_t = scale_knob * xi_{t-1}
    'loop_tol': 0.02,     # 收敛判据: 无量纲比值的相对变化 < 2%
}



# 因果链台账: 每条 (上游量 -> 下游参数) 的记录都进这里, 供打印/JSON/作图
CHAIN = []




def record_link(link, upstream, formula, derived, v10_manual, note=''):
    """
    记录一条因果链。link 形如 'L2', upstream 是上游物理量的描述。

    v10_manual 是同一个参数在手工接线时代的取值 (字段名保持不变以维持兼容)
    —— 把"派生值 vs 手工值"并排打印出来, 读者可以自己判断这条链是自洽还是
    自欺。二者不等, 说明派生确实改了取值, 而不是把原字面量换个说法抄一遍。
    """
    rec = {'link': link, 'upstream': upstream, 'formula': formula,
           'derived': derived, 'v10_manual': v10_manual, 'note': note,
           'same_as_manual': (derived == v10_manual)}
    CHAIN.append(rec)
    return rec




def print_chain_report():
    """把整条因果链打印成一张对照表。这是因果链层的主要可检查产物。"""
    print("\n" + "=" * 76)
    print("  因果链台账: 派生值 vs 手工值")
    print("=" * 76)
    for r in CHAIN:
        tag = "= 手工值" if r['same_as_manual'] else "!= 手工值"
        print(f"  [{r['link']}] {r['upstream']}")
        print(f"        公式: {r['formula']}")
        print(f"        派生 = {r['derived']!r}   (手工 = {r['v10_manual']!r})  {tag}")
        if r['note']:
            print(f"        注: {r['note']}")





# ---------- 八条链的派生函数 (L1..L6 在此实现, L7/L8 见 main 与 spiral_loop) ----------

def derive_L1_local_to_full(d_local, L):
    """
    L1 · 阶段一 -> 阶段二: 局部自由度张成全空间。

    空无 = 单个量子比特 (局部维度 d_local), 阶段二的 L 个自旋是整个空间。
        dim_full = d_local ** L

    手工接线时 stage1_void(4) 取维度 4, 与阶段二的 2 维自旋毫无关系 ——
    这条是八条链里最容易修也最明显的一条。
    """
    dim_full = int(d_local ** L)
    record_link('L1', f'空无的局部维度 d_local={d_local}, 链长 L={L}',
                'dim_full = d_local ** L', dim_full, 4,
                note='手工值 4 是随手取的; 现由局部维度与链长派生')
    return {'d_local': int(d_local), 'dim_full': dim_full}




def derive_L1_void_to_chain(equal_state, L, h_wide=(2, 4, 8, 16, 32), J=1.0):
    """
    L1 · 阶段一 -> 阶段二: 空无的等权叠加大态 = h/J -> inf 时的基态。

    v15·W11 诊断项点名的缺口是: `stage1_void` 返回的 equal_state 在整个仓库里
    **没有任何取值处**, 所以「推离 L1 应当改变下游读数」这个负对照**造不出来**
    —— 推离一个没人用的量不改变任何东西, 控制就是摆设。本函数是给 L1 建的第一条
    **真实下游连线**: 把局部等权态张成 L 个自旋的域态, 再与阶段二的精确基态做重叠。

    做法:
        1. |void> = equal_state 的 L 次张量幂, 取它的维数与 `dim_full` 比对
           —— 这是 v15 里**只被 print** 的 dim_full 第一次被真消费;
        2. 对 h/J in h_wide 求 tfi_periodic_sparse 的精确基态;
        3. overlap(h/J) = |<void | gs(h/J)>|^2。

    判据 (可证伪): overlap 对 h/J **单调上升**, 且 h/J=32 时 overlap_inf > 0.99。

    代价: len(h_wide) 次 L=16 精确对角化 (秒级, 远小于 MERA 拟合)。

    诚实边界 (v16_plan.md §2.4, **两句话不许并成一句**):
      * "|+>^{tensor L} 是 H(h) = -J Sum sz sz - h Sum sx 在 h/J -> inf 的基态"
        是**数学事实** (它是 Sum sx 的最大本征态, 该极限下非简并);
      * 但 h/J=32 是**有限值**, overlap_inf 只是**逼近** 1, 差值为 O((J/h)^2);
      * 把上述数学事实称作 "L1 到 L2 的因果连线" 是**口径选择**, 不是事实 ——
        它是一条**可检验的接线**, 不是已证明的物理因果。
    """
    L = int(L)
    eq = np.asarray(equal_state, dtype=complex)
    d_local = int(eq.size)
    dm = derive_L1_local_to_full(d_local, L)   # dim_full 的真实来源
    dim_full = int(dm['dim_full'])

    eq = eq / np.linalg.norm(eq)
    void_full = eq
    for _ in range(L - 1):
        void_full = np.kron(void_full, eq)

    dim_match = bool(void_full.size == dim_full == 2 ** L)

    # 维数不符时**不做重叠** —— 那个态根本不在自旋链的 Hilbert 空间里, 硬算
    # 只会让 vdot 抛形状错 (冒烟实测过)。overlap_inf 如实返回 nan, 不填数。
    overlap_curve, monotone, overlap_inf = [], False, float('nan')
    if dim_match:
        for ratio in h_wide:
            _E0, gs = exact_ground_state(L, J, float(ratio) * J)
            overlap_curve.append((float(ratio),
                                  float(abs(np.vdot(void_full, gs)) ** 2)))
        ovs = [o for _, o in overlap_curve]
        monotone = all(b >= a - 1e-12 for a, b in zip(ovs, ovs[1:]))
        overlap_inf = ovs[-1]

    print(f"  [L1] 空无 -> 链: d_local={d_local} 张成 {void_full.size} 维, "
          f"dim_full={dim_full}, 与 2^{L} 匹配={dim_match}")
    if dim_match:
        print("  [L1] |<void|gs(h/J)>|^2 = "
              + ", ".join(f"{r:g}:{o:.6f}" for r, o in overlap_curve)
              + f"  (对 h/J 单调上升={monotone})")
    else:
        print(f"  [L1] 张成维数 {void_full.size} != 2^{L}: 该态不在链的"
              f" Hilbert 空间里, **跳过重叠计算**, overlap_inf 如实为 nan")
    return {'d_local': d_local, 'dim_full': dim_full, 'dim_match': dim_match,
            'overlap_inf': overlap_inf, 'overlap_curve': overlap_curve,
            'monotone': monotone, 'L': L, 'void_full': void_full,
            'equal_state': eq}




def l1_void_pushaway(equal_state, L, thetas=(0.1, 0.3, 0.6), h_ratio=32.0,
                     J=1.0):
    """
    L1 的**推离负对照** —— 与 `derive_L1_void_to_chain` 配对的另一半。

    B1-1 说的是"不推离时重叠 -> 1"; 本函数说的是"推离之后重叠**必须掉下去**"。
    若 L1 的态对下游毫无作用, 转动它不会改变任何读数, 这条就失败 —— 这就是它
    的可证伪性所在。

    做法: 把局部等权态绕 y 轴转 theta, 张成 L 个自旋的域态, 量它与 h/J=32
    精确基态的重叠。theta 依次取 thetas, 取**第一个**使重叠 < 0.9 的作为校准
    点; 一个都到不了 0.9 就如实返回 None, **不调 theta 凑数**。

    对计划原文的一处**有意偏离**: 计划写"绕**随机**轴", 这里改成**固定 y 轴**
    (R_y = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]])。理由: 随机轴会让这条
    守卫每次运行给出不同的校准点, 不可复现; 固定轴同样能测出"推离是否有效",
    而且把校准写死成可复核的数。偏离本身记进守卫 note, 不掩盖。

    返回 {'theta_star', 'overlap_at_star', 'curve', 'none_below'}
    """
    L = int(L)
    eq = np.asarray(equal_state, dtype=complex)
    eq = eq / np.linalg.norm(eq)
    if eq.size != 2:
        raise ValueError(f'推离只在单比特局部态上有定义, 收到 size={eq.size}')
    _E0, gs = exact_ground_state(L, J, float(h_ratio) * J)

    curve = []
    for theta in thetas:
        c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
        psi = np.array([[c, -s], [s, c]], dtype=complex) @ eq
        void = psi
        for _ in range(L - 1):
            void = np.kron(void, psi)
        curve.append((float(theta), float(abs(np.vdot(void, gs)) ** 2)))

    theta_star, ov_star = None, None
    for theta, ov in curve:
        if ov < 0.9:
            theta_star, ov_star = theta, ov
            break

    print(f"  [L1] 推离 (h/J={h_ratio:g}): "
          + ", ".join(f"theta={t:g}:{o:.6f}" for t, o in curve)
          + f"  → theta*={theta_star} (overlap={ov_star})")
    return {'theta_star': theta_star, 'overlap_at_star': ov_star,
            'curve': curve, 'none_below': theta_star is None}




def derive_L2_entanglement_gap(p, tol=1e-15):
    """
    L2 · 阶段二 -> 阶段三-a: 纠缠谱给出幂迭代的收敛率。  ← 整条链的核心

    为什么这是数学而不是类比 (此处已按文献核实修正过一次措辞):
      把一个二分量子态写成矩阵 M (行/列标是两边的基), 则约化密度矩阵
          rho = M M^dagger,
      而 rho 的特征值**恰好**是 Schmidt 系数的平方 {s_k^2}。
      对 rho 做幂迭代, 收敛率 = 第二大 / 最大 = (s_1/s_0)^2 = p1/p0。
      这是初等线性代数, 严格成立。

    一处必须澄清的误用 (初稿曾写错, 已改):
      不能把上面这句说成"转移矩阵的特征值就是 {s_k^2}"。标准 MPS 的
      mixed transfer operator T(X) = sum_i A^i X A^{i†} 是 D^2 x D^2 的,
      正则形式下它的**主特征值恒为 1**, 而 {s_k^2} 是它的**主左不动点**
      (dominating left eigenvector), 不是特征值; 它的次主特征值给出的是
      关联长度 xi = -1/ln|lam2/lam1|, 与 Schmidt 谱无直接关系。
      所以本函数派生的是"对 rho = M M^dagger 做幂迭代的收敛率",
      不是任何转移矩阵定理。措辞按此写, 不夸大。

    所以阶段三-a 里"幂迭代收敛得多快"不该由人手指定, 而应由阶段二实测的
    Schmidt 谱决定。这是整条因果链里唯一一条有严格数学对应的连线。

    诚实边界: 临界系统的纠缠谱在 L -> inf 时能隙闭合 (p1/p0 -> 1),
    所以有限尺寸下取到的只是一个**被尺寸截断的**谱隙。这不是缺陷,
    恰是临界的标志 —— 闭环 spiral_loop() 里正是用相对谱隙 1 - p1/p0 -> 0
    来判临界。(注意 p1/p0 本身是**收敛比**不是谱隙, 它趋于 1 而非 0。)
    """
    p = np.asarray(p, dtype=float)
    p = p[p > tol]
    if len(p) < 2:
        raise RuntimeError(
            f"L2 退化: 约化密度矩阵只有 {len(p)} 个非零本征值, 无法定义谱隙。"
            f"拒绝派生 —— 阶段三-a 的收敛比必须有物理来源。")
    gap = float(p[1] / p[0])
    record_link('L2', f'约化密度矩阵谱 p = s^2 ({len(p)} 个非零值)',
                'gap = p[1] / p[0]', round(gap, 6), 0.5,
                note=f'p0={p[0]:.6f}, p1={p[1]:.6f}; '
                     f'临界态该值随 L 增大而 -> 1 (谱隙闭合)')
    return {'gap': gap, 'ngap': len(p), 'p0': float(p[0]), 'p1': float(p[1])}




def derive_L3_schmidt_rank(p, tol):
    """
    L3 · 阶段二 -> 阶段三-b/c: 有效 Schmidt 秩决定自指网络的宽度。

    临界态的 Schmidt 谱衰减慢, 有效秩随 L 增长 —— 这是真实可测的派生量。
    自指网络 x_{n+1} = tanh(g W x_n) 里的 W 是 N*N 的, 它的宽度 N 应当是
    "这个态实际需要多少个分量才能描述", 也就是有效 Schmidt 秩。

    诚实边界: 秩的数值依赖阈值 tol。这里显式传参并在日志里打印 tol,
    不藏一个 1e-8 在函数体里假装它是自然常数。
    """
    p = np.asarray(p, dtype=float)
    rank = int(np.count_nonzero(p > tol))
    rank = max(rank, 2)   # 至少要 2, 否则网络退化成一个标量
    record_link('L3', f'有效 Schmidt 秩 (p_k > {tol:g})',
                'N_selfref = #{k : p_k > tol}', rank, 64,
                note='手工值 64 与态的实际复杂度无关')
    return {'N_selfref': rank, 'schmidt_tol': float(tol)}




def derive_L4_bond_dimension(S, base=2):
    """
    L4 · 阶段二 -> 阶段四: 面积律给出键维下界。

    一维面积律说, 要精确表示纠缠熵为 S 的态, 键维至少要有
        chi >= exp(S)
    二进制 MERA 的键维必须是 2 的幂, 故取
        chi = base ** ceil(log_base(exp(S)))

    **这是必要非充分条件**: 它只保证能装下这么多纠缠, 不保证装下长程关联。
    所以派生值与实测最优配置吻合只能算自洽性证据, 不能算证明。

    手工接线时扫描 chi in {2, 4} 并报告 c(chi=4) 最好; 派生值若也落在 4,
    说明这条链是自洽的 —— 但脚本不因此宣称"证明了 chi=4 最优"。
    """
    if S <= 0:
        raise RuntimeError(
            f"L4 退化: 纠缠熵 S={S:.3e} <= 0, 面积律给不出正的下界。"
            f"拒绝派生 —— 一个无纠缠的态不能用来定键维。")
    chi_req = float(np.exp(S))
    chi = int(base ** int(np.ceil(np.log(chi_req) / np.log(base))))
    record_link('L4', f'半链纠缠熵 S(L/2) = {S:.4f}',
                f'chi = {base}**ceil(log_{base}(exp(S)))', chi, 4,
                note=f'面积律下界 exp(S) = {chi_req:.4f} -> 取 2 的幂得 {chi} '
                     f'(必要非充分)')
    return {'chi': chi, 'chi_lower_bound': chi_req}




def derive_L5_forman_from_degree(G, forman_measured, record=True):
    """
    L5 · 阶段四 -> 阶段五: MERA 张量图的平均度给出 Forman 曲率的解析预期。

    无权图、不含三角项时, Sreejith 加权式退化为
        F(e) = 4 - deg(u) - deg(v)
    由此有**两个**解析预期, 必须分清:

      (i)  正则粗式:   <F> = 4 - 2*d_bar        d_bar = 2E/V (图平均度)
                       只在 d-正则图上严格成立。

      (ii) 严格恒等式: <F> = 4 - 2*<deg>_edge
                       其中 <deg>_edge = mean_{(u,v) in E} (deg(u)+deg(v))/2
                       是**按边加权的端点度平均**。这个式子对任何无三角图
                       都精确成立 (它只是把 F(e) 逐边平均改写成度数形式),
                       所以它应当与实测到机器精度吻合。

    两者的差就是**度偏置 (degree bias)**: 非正则图上度数大的点被更多条边
    "选中", 于是 <deg>_edge > d_bar, 实测 <F> 比粗式给出的更负。

    MERA 体几何的 Forman 均值 = -2.242 需要说明它从哪里来, 也要解释为什么
    它不等于 4-2*d_bar。这条链补上"从哪里来", 并顺手把那个差归因清楚 ——
    是度偏置, 不是曲率实现有错。三角数一并报出, 因为 F(e)=4-deg(u)-deg(v)
    这个无三角形式本身需要 MERA 图接近树才成立。

    record=False 供**对照图**使用: 对照图不是因果链的一环, 把它们的度偏置
    写进因果链台账会污染那条链。但几何口径必须与 MERA 图完全一致, 所以
    复用同一个函数而不是另写一套。
    """
    n, m = G.number_of_nodes(), G.number_of_edges()
    d_bar = 2.0 * m / n if n else float('nan')
    # (i) 粗略式: 只在 d-正则图上才严格. 非正则图会系统性偏低, 见下.
    predicted = 4.0 - 2.0 * d_bar
    # (ii) 严格恒等式: 无三角项时 <F> = 4 - 2*<deg>_edge,
    #      其中 <deg>_edge 是**逐边端点度数的平均** (每条边贡献 deg(u)+deg(v),
    #      故除以 2E 再乘 2 —— 等价于按边加权, 度数高的点被数到更多次)。
    #      这个量与实测 Forman 均值应当**到机器精度**吻合, 因为它就是
    #      F(e)=4-deg(u)-deg(v) 逐边求平均的恒等改写。
    deg = dict(G.degree())
    deg_edge_mean = float(np.mean([deg[u] + deg[v] for u, v in G.edges()]) / 2.0)
    predicted_exact = 4.0 - 2.0 * deg_edge_mean
    # 三角数: MERA 近似树, 这个量应当很小 —— 它是上面 (b) 的直接证据
    triangles = sum(nx.triangles(G).values()) // 3
    # degeneracy (度偏置) 就是粗式与实测的差从哪来: 非正则图上
    # 度数大的点被更多条边"选中", 所以 <deg>_edge > d_bar, 于是
    # 实测 <F> 比 4-2*d_bar 更负。级数上 = 2*(<deg>_edge - d_bar)。
    bias = 2.0 * (deg_edge_mean - d_bar)
    if record:
        record_link('L5', f'张量图 V={n}, E={m} '
                          f'(平均度 d_bar={d_bar:.3f}, 逐边端点平均度={deg_edge_mean:.3f})',
                    '<F> = 4 - 2*d_bar (正则图式) | '
                    '<F> = 4 - 2*<deg>_edge (无三角严格式)',
                    round(predicted_exact, 4), round(forman_measured, 4),
                    note=f'实测 Forman 均值 = {forman_measured:.4f}; '
                         f'严格式预期 = {predicted_exact:.4f}; '
                         f'正则粗式预期 = {predicted:.4f}。'
                         f'粗式差 {abs(forman_measured - predicted):.4f} = 度偏置 2*(<deg>_edge-d_bar) '
                         f'= {bias:.4f} 起, 不是实现错误。'
                         f'图含 {triangles} 个三角形 (接近树, 故三角项影响小)')
    return {'d_bar': d_bar, 'forman_predicted': predicted,
            'deg_edge_mean': deg_edge_mean, 'forman_exact': predicted_exact,
            'bias': bias, 'triangles': triangles}




def derive_L6_grid(xi_over_L, knobs, xi=None):
    """
    L6 · 阶段五 -> 阶段六: 关联长度与稳定性上界共同定出网格。

    **先说清楚哪条能派生、哪条不能** —— 这是本条链最重要的部分。

    不能派生的: Gray-Scott 的**物理域长** L_domain 不能由 Ising 链的关联长度
    算出。两者是不同的物理 (一个是量子自旋关联, 一个是反应-扩散的扩散长度
    sqrt(Du/(F+k))), 把它们数值相等起来是量纲和物理的双重范畴错误。
    所以 L_domain, n_lambda, pts_per_wavelength 都是**显式旋钮**。

    真正能派生的是**网格数 N**, 它由两条约束夹出来:

      下界 (分辨要求): 每个斑图波长至少要 pts_per_wavelength 个网格点,
                       域内要容纳 n_lambda 个波长
                       N_req = ceil(pts_per_wavelength * n_lambda)

      上界 (稳定性定理): 显式 Euler 解 dt*Du/h^2 <= 1/4, 而 h = L_domain/N
                       解出 h >= sqrt(4*dt*Du), 即 N <= L_domain / sqrt(4*dt*Du)
                       N_cap = floor(L_domain / sqrt(4*dt*Du))
                       (注意是**除以** sqrt(4*dt*Du)。初稿误写成乘, 于是
                        N_cap=0 被本函数的守卫当场拦下 —— 守卫有效。)

      N = min(N_req, N_cap), 并报告是哪一条在起作用。

    若 N_cap < N_req, 两条约束冲突: 派生的域长大到无法在稳定步长下分辨。
    此时**拒绝运行**并说明, 而不是偷偷放宽稳定性。

    这一步的实际意义: N 若写死成 40, 稳定性守卫 (stab <= 0.25) 就只是
    "碰巧没被触发"。N 由稳定性上界派生之后, 守卫从运气变成了定理 ——
    它不可能被触发, 除非两条约束冲突 (那时是显式拒绝, 不是偷偷放行)。

    xi_over_L 是阶段五实测的无量纲比值 xi/L。它不直接进公式 (见上),
    但被记录下来, 因为它的尺度不变性正是 n_lambda 取值的依据。
    """
    L_dom = float(knobs['L_domain'])
    dt = float(knobs['dt'])
    Du = float(knobs['Du'])

    # 上界: 稳定性定理.  dt*Du/h^2 <= 1/4 且 h = L_dom/N  =>  N <= L_dom/sqrt(4*dt*Du)
    N_cap = int(np.floor(L_dom / np.sqrt(4.0 * dt * Du)))
    # 下界: 分辨要求
    N_req = int(np.ceil(knobs['pts_per_wavelength'] * knobs['n_lambda']))
    N = min(N_req, N_cap)
    bound_by = '分辨要求 (N_req)' if N_req <= N_cap else '稳定性上界 (N_cap)'

    if N < 8:
        raise RuntimeError(
            f"L6 拒绝运行: 派生网格 N={N} 太小 (N_req={N_req}, N_cap={N_cap}), "
            f"无法分辨斑图。这是派生值的真实冲突, 不偷偷放宽稳定性条件。")

    h = L_dom / N
    stab = dt * Du / h ** 2
    record_link('L6', f'域长 L_domain={L_dom} (旋钮), '
                      f'xi/L={xi_over_L:.4f} (阶段五实测, 无量纲)',
                'N = min(ceil(ppw*n_lambda), floor(L_domain/sqrt(4*dt*Du)))',
                N, 40,
                note=f'N_req={N_req} (分辨), N_cap={N_cap} (稳定), '
                     f'受限于 {bound_by}; 派生后 h={h:.6f}, '
                     f'dt*Du/h^2={stab:.4f} <= {knobs["stability_limit"]} '
                     f'由构造成立')
    return {'N': N, 'N_req': N_req, 'N_cap': N_cap, 'h': h, 'stab': stab,
            'bound_by': bound_by, 'L_domain': L_dom,
            'lambda_target': L_dom / knobs['n_lambda']}




# ============================================================================
# 第 0 节 · 工具: 周期 TF-Ising + 纠缠熵 + 中心荷拟合
# ============================================================================
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




def exact_ground_state(L, J=1.0, h=1.0):
    """稀疏 eigsh 求基态。**本函数没有 L<=16 这个限制** —— L=18 实测建 H 1.53s
    + eigsh 3.86s + 433MB, 且基态与 Jordan-Wigner 闭合式仍吻合 2.5e-14。
    16 这个数来自末圈 MERA 交叉校验要求 L 为 2 的幂, 与本函数无关。
    返回 (E0, gs)"""
    H = tfi_periodic_sparse(L, J, h)
    E, V = eigsh(H, k=1, which='SA')
    gs = V[:, 0]
    return float(E[0]), gs / np.linalg.norm(gs)




def jw_ground_energy(L, J=1.0, h=1.0):
    """
    周期 TF-Ising 的 Jordan-Wigner 解析基态能量 (NS 扇区, 动量 q_k = pi(2k+1)/L):
        E0 = -Sum_{k=0}^{L-1} sqrt( J^2 + h^2 - 2 J h cos(q_k) )

    这是与稀疏对角化**完全独立**的第二条路径, 所以它能测出 eigsh 测不出的东西:
      * H 的位序约定 (站点 i <-> 二进制位 L-1-i) 错了, 这里立刻对不上;
      * 周期边界那条键 ((-J) sigma^z_{L-1} sigma^z_0) 漏了或重复了, 也对不上;
      * 横场项的符号 (-h sigma^x) 反了, 同样对不上。
    eigsh 只回答"给定这个 H, 最小本征值是多少", 对 H 本身对不对一句话都没有。
    """
    k = np.arange(L)
    q = np.pi * (2.0 * k + 1.0) / L
    return float(-np.sum(np.sqrt(J ** 2 + h ** 2 - 2.0 * J * h * np.cos(q))))




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




# ============================================================================
# 阶段六 · 时空提供梯度孕育显生命 (涌现几何上的 Gray-Scott)
# ============================================================================
def _fit_length(dists, vals, max_d=None):
    """由 ln|值| 随距离的线性衰减拟合关联长度 xi:  值 ~ exp(-d/xi)"""
    d = np.asarray(dists, dtype=float)
    y = np.log(np.maximum(np.asarray(vals, dtype=float), 1e-300))
    if max_d is not None:
        m = d <= max_d
        d, y = d[m], y[m]
    if len(d) < 3 or np.ptp(d) == 0:
        return float('nan')
    slope, _ = np.polyfit(d, y, 1)
    return float(-1.0 / slope) if slope < 0 else float('inf')




def boundary_correlation_length(C, L):
    """
    边界关联几何的关联长度: 把 <sigma^z_i sigma^z_j> 按环上距离 d 分箱平均,
    再拟合 ln|C| ~ -d/xi。这是阶段五那套几何的特征尺度。
    """
    ds, vs = [], []
    for i in range(L):
        for j in range(i + 1, L):
            d = min(j - i, L - (j - i))
            ds.append(d)
            vs.append(abs(C[i, j]))
    ds = np.array(ds)
    vs = np.array(vs)
    binned = np.array([vs[ds == d].mean() for d in np.unique(ds)])
    return _fit_length(np.unique(ds), binned)
