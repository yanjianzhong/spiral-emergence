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
"""spiral_model_v16._model.stage5 —— 由 spiral_model_v16.py 拆分。

阶段五·全息自然展开生成了时空 (曲率几何)

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
from _model.core import derive_L5_forman_from_degree




# ============================================================================
# 阶段五 · 全息自然展开生成了时空
# ============================================================================
def all_pairs_distance(G):
    nodes = sorted(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    D = np.full((len(nodes), len(nodes)), np.inf)
    for s in nodes:
        for t, d in nx.single_source_shortest_path_length(G, s).items():
            D[idx[s], idx[t]] = d
    return nodes, idx, D




def ollivier_ricci(G, alpha=0.0):
    """
    Ollivier-Ricci 曲率:
        kappa(u,v) = 1 - W1(m_u, m_v) / d(u,v)
    m_u = 随机游走在 u 上的分布; W1 = 1-Wasserstein 距离 (最优传输), 用
    线性规划精确求解 (图很小, 无需近似)。

    alpha 是 idle 概率 (懒惰游走): m_u = (1-alpha)/deg 分布于邻居, alpha 留在
    u 自身。这个约定决定了数值 —— 超立方体在 alpha=0 时曲率为 0, 在
    alpha=1/(d+1) 时才是 2/(d+1)。因此本脚本显式暴露 alpha 并给出两组锚点,
    而不是引用一个来源不明的文献数字。
    """
    nodes, idx, D = all_pairs_distance(G)
    out = {}
    for u, v in G.edges():
        mu, mv = {}, {}
        for self_node, nbrs, tgt in ((u, G[u], mu), (v, G[v], mv)):
            nb = list(nbrs)
            for x in nb:
                tgt[x] = (1.0 - alpha) / len(nb)
            tgt[self_node] = tgt.get(self_node, 0.0) + alpha   # idle 留在自身
        ni, nj = sorted(mu), sorted(mv)
        cost = np.array([[D[idx[a], idx[b]] for b in nj] for a in ni])
        n_i, n_j = len(ni), len(nj)
        A_eq, b_eq = [], []
        for i_, a in enumerate(ni):                 # 行边缘 = m_u
            r = np.zeros(n_i * n_j)
            r[i_ * n_j:(i_ + 1) * n_j] = 1
            A_eq.append(r)
            b_eq.append(mu[a])
        for j_, b in enumerate(nj):                 # 列边缘 = m_v
            r = np.zeros(n_i * n_j)
            r[j_::n_j] = 1
            A_eq.append(r)
            b_eq.append(mv[b])
        res = linprog(cost.ravel(), A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                      bounds=(0, None), method='highs')
        out[(u, v)] = 1.0 - res.fun / D[idx[u], idx[v]]
    return out




def forman_ricci(G, edge_w=None, node_w=None, triangles=True):
    """
    Forman-Ricci 曲率 (Sreejith 等的加权式):
        F(e=uv) = w_e [ w_u/w_e + w_v/w_e
                        - Sum_{e'~u, e'!=e} w_u/sqrt(w_e w_e')
                        - Sum_{e'~v, e'!=e} w_v/sqrt(w_e w_e') ]
    单位权重的无权图退化为经典的 F(e) = 4 - deg(u) - deg(v)。
    triangles=True 时再加 3 * (含 e 的三角形数) * w_e (augmented 版本)。
    """
    node_w = node_w or {n: 1.0 for n in G}
    edge_w = edge_w or {tuple(sorted(e)): 1.0 for e in G.edges()}
    out = {}
    for u, v in G.edges():
        we = edge_w[tuple(sorted((u, v)))]
        wu, wv = node_w[u], node_w[v]
        s = wu / we + wv / we
        for e2 in G.edges(u):
            if set(e2) != {u, v}:
                s -= wu / np.sqrt(we * edge_w[tuple(sorted(e2))])
        for e2 in G.edges(v):
            if set(e2) != {u, v}:
                s -= wv / np.sqrt(we * edge_w[tuple(sorted(e2))])
        f = we * s
        if triangles:
            f += 3.0 * len(set(G.neighbors(u)) & set(G.neighbors(v))) * we
        out[(u, v)] = float(f)
    return out




def validate_curvature():
    """
    在三个有解析解的图上校验 (全部由本脚本手推, 不引用记忆中的文献数字):
      * 环 C_n (alpha=0):        kappa = 0
            m_u 在 {u-1, u+1} 各 1/2, 最优传输 W1 = 1 = d(u,v)
      * 完全图 K_n (alpha=0):    kappa = (n-2)/(n-1)
            N(u) = V\\{u}; 交集 V\\{u,v} 可零成本匹配 (n-2)/(n-1) 的质量,
            余下 1/(n-1) 走一步 -> W1 = 1/(n-1)
      * 超立方体 Q_d (alpha=1/(d+1)): kappa = 2/(d+1)
            所有 d+1 个源质量均为 1/(d+1), 可配对 a_i<->b_i (距离 1),
            总代价 (d-1)/(d+1) -> kappa = 1 - (d-1)/(d+1) = 2/(d+1)
      注: alpha=0 时 Q_d 的 kappa 恰为 0 (同样可手推)。两值的差异完全来自
          idle 约定, 所以约定必须显式声明, 而不是含糊地"引用文献"。
      * Forman (无权, 无三角): F(uv) = 4 - deg(u) - deg(v)
            环 C_n (n>=4) 每点度 2 -> F 恒为 0;
            P_6 内部边 -> 0; K_4 去掉三角项 -> -4+... = -2; 3-正则 -> 恒 -2。
            环的锚点把"度"这一项从正则图锚点里分离出来了 —— 3-正则锚点
            (-2) 与环锚点 (0) 都只依赖度, 但取值不同, 所以两者一起才能
            确认实现走的是 deg(u)+deg(v) 而不是别的正则组合。
    """
    res = {'cyclic': [], 'complete': [], 'hypercube': [], 'forman': {}}
    for n in (6, 8, 12):
        k = np.array(list(ollivier_ricci(nx.cycle_graph(n)).values()))
        res['cyclic'].append({'n': n, 'mean': float(k.mean()), 'exact': 0.0,
                              'max_abs': float(np.abs(k).max())})
    for n in (4, 6, 10):
        k = np.array(list(ollivier_ricci(nx.complete_graph(n)).values()))
        res['complete'].append({'n': n, 'mean': float(k.mean()),
                                'exact': (n - 2) / (n - 1)})
    for d in (3, 4, 5):
        Q = nx.hypercube_graph(d)
        k = np.array(list(ollivier_ricci(Q, alpha=1.0 / (d + 1)).values()))
        k0 = np.array(list(ollivier_ricci(Q, alpha=0.0).values()))
        res['hypercube'].append({'d': d, 'mean_lazy': float(k.mean()),
                                 'exact_lazy': 2.0 / (d + 1),
                                 'mean_plain': float(k0.mean()),
                                 'exact_plain': 0.0})
    p6 = forman_ricci(nx.path_graph(6))
    k4 = forman_ricci(nx.complete_graph(4), triangles=False)
    reg = nx.random_regular_graph(3, 10, seed=1)
    # 环 C_n 的 Forman 锚点。无三角版 F(uv) = 4 - deg(u) - deg(v), 环上每点度 2,
    # 所以每条边恒为 0 —— 这是 4-2-2 这条式子的直接后果。
    # **n=3 必须排除**: C_3 就是 K_3, 三条边两两相邻, 每条边都在一个三角形里,
    # augmented 项 +3*1 把它从 0 顶到 +3。写成"环恒为 0"而不过滤 n=3 就是错的。
    cyc = {n: float(np.mean(list(
        forman_ricci(nx.cycle_graph(n), triangles=False).values())))
        for n in (4, 6, 8, 12, 100)}
    res['forman'] = {
        'path_interior': float(p6[(1, 2)]),                 # 4 - 2 - 2 = 0
        'k4_no_tri': float(list(k4.values())[0]),           # 4 - 3 - 3 = -2
        'regular3_mean': float(np.mean(list(
            forman_ricci(reg, triangles=False).values()))),  # 3-正则恒为 -2
        'cycle_mean': cyc,
        'cycle_max_abs': max(abs(v) for v in cyc.values()),
        'cycle_min_n': min(cyc),
    }
    print("[阶段五] 曲率算子的解析锚点校验:")
    for r in res['cyclic']:
        print(f"  环 C_{r['n']}: {r['mean']:+.8f} (解析 0, max|.|={r['max_abs']:.1e})")
    for r in res['complete']:
        print(f"  完全图 K_{r['n']}: {r['mean']:.8f} (解析 {r['exact']:.8f})")
    for r in res['hypercube']:
        print(f"  超立方体 Q_{r['d']}: alpha=0 -> {r['mean_plain']:+.6f} (解析 0); "
              f"alpha=1/(d+1) -> {r['mean_lazy']:.6f} "
              f"(解析 {r['exact_lazy']:.6f})")
    print(f"  Forman: P_6 内部边={res['forman']['path_interior']:+.1f} (解析 0), "
          f"K_4 无三角={res['forman']['k4_no_tri']:+.1f} (解析 -2), "
          f"3-正则图={res['forman']['regular3_mean']:+.4f} (解析 -2), "
          f"环 C_n (n>=4) max|F|={res['forman']['cycle_max_abs']:.1e} (解析 0)")
    return res




def curvature_report(G, name):
    """
    一张图的 Forman / Ollivier 曲率分布汇总, 外加三角数与平均度。

    为什么必须一起报三角数与平均度: 本工作用的 Forman 是**无三角**形式
    F(e) = 4 - deg(u) - deg(v), 它只在图接近树时才与完整 Forman 等价。
    "这张图是不是接近树"这件事由三角形数回答, 只报曲率均值等于把前提藏起来。
    平均度给出正则粗式预期 4-2*d_bar, 与逐边严格式 4-2*<deg>_edge 并排,
    度偏置的来处(非正则)就摆在明面上。

    几何口径与 MERA 图**完全一致** (复用 derive_L5_forman_from_degree 且
    record=False) —— 对照图不进因果链台账, 但量必须是同一把尺子量出来的,
    否则"MERA 与对照图比曲率"这件事本身就不可比。
    """
    f = forman_ricci(G, triangles=False)
    try:
        k = ollivier_ricci(G)
    except Exception:
        k = {}
    fv = np.array(list(f.values()))
    kv = np.array(list(k.values())) if k else np.array([np.nan])
    geom = derive_L5_forman_from_degree(G, float(fv.mean()), record=False)
    rep = {'name': name, 'n_nodes': G.number_of_nodes(),
           'n_edges': G.number_of_edges(),
           'forman_mean': float(fv.mean()), 'forman_min': float(fv.min()),
           'forman_max': float(fv.max()),
           'forman_neg_frac': float((fv < 0).mean()),
           'or_mean': float(np.nanmean(kv)),
           'or_neg_frac': float(np.nanmean(kv < 0)) if k else float('nan'),
           'n_triangles': int(geom['triangles']),
           'd_bar': float(geom['d_bar']),
           'deg_edge_mean': float(geom['deg_edge_mean']),
           'forman_regular_pred': float(geom['forman_predicted']),
           'forman_exact_pred': float(geom['forman_exact']),
           'degree_bias': float(geom['bias'])}
    print(f"  {name}: |V|={rep['n_nodes']} |E|={rep['n_edges']} "
          f"d_bar={rep['d_bar']:.3f} 三角={rep['n_triangles']} | "
          f"Forman 均值={rep['forman_mean']:+.3f} "
          f"(负边占比 {rep['forman_neg_frac']:.2f}, 逐边严格式预期 "
          f"{rep['forman_exact_pred']:+.3f}) | "
          f"Ollivier 均值={rep['or_mean']:+.4f} "
          f"(负边占比 {rep['or_neg_frac']:.2f})")
    return rep, f, k




def geometry_controls(G_mera, n_each=3):
    """
    对照几何族 —— 三类零模型, 每类 n_each 个实例。

      gnm  同 |V| 同 |E| 的 Erdos-Renyi 随机图 (度分布不受约束)
      ws   Watts-Strogatz 小世界, 平均度与 MERA 图对齐 (配度, 高聚类)
      tree 平衡二叉树 (纯树: 零三角形, 直接对标"MERA 图接近树"这条前提)

    为什么要三类而不是一类: 单张同规模随机图只能回答"比随机图更负吗"这一个
    问题, 而"MERA 的负曲率有什么特点"至少有三个可分离的维度 —— 度分布、
    聚类(由三角形数度量)、树的成分。三类各控一个维度, 于是"MERA 落在对照
    分布里"这句话才有内容。

    实例数取 3: 单实例的曲率均值噪声在 0.1 量级, 3 个实例足以给出一个
    5%~95% 的粗略位置, 同时把 Ollivier (逐边解线性规划) 的总耗时控住。

    诚实边界: 这是**零模型对照**, 不是同一物理系统的不同实现。它回答的是
    "MERA 体几何在同类规模的图里算不算异常", 不是"MERA 就是全息对偶"。
    平衡树的节点数受限于 2^(h+1)-1, 与 |V| 只能接近, 故各图的实际规模
    一并报出而不是假定相等。
    """
    V, E = G_mera.number_of_nodes(), G_mera.number_of_edges()
    # WS 的度取 2*round(d_bar) 且取偶 —— 与 MERA 图的平均度对齐, 而不是写死 4
    k_ws = max(2, 2 * int(round(E / V))) if V else 2
    h_tree = max(1, int(np.log2(V + 1)) - 1)
    reps = []
    for fam in ('gnm', 'ws', 'tree'):
        for s in range(n_each):
            if fam == 'gnm':
                G = nx.gnm_random_graph(V, E, seed=100 + s)
            elif fam == 'ws':
                G = nx.watts_strogatz_graph(V, k_ws, 0.1, seed=200 + s)
            else:
                G = nx.balanced_tree(2, h_tree)
            rep, _, _ = curvature_report(G, f'对照 {fam} #{s}')
            rep['family'] = fam
            rep['instance'] = int(s)
            reps.append(rep)
    print(f"  对照族: gnm/ws/tree 各 {n_each} 个实例 "
          f"(MERA 图 |V|={V} |E|={E}, d_bar={2*E/V:.3f}; WS 取 k={k_ws}, "
          f"树高 {h_tree} -> {2**(h_tree+1)-1} 节点)")
    return reps
