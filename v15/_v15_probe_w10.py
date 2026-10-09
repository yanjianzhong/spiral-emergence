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
#
"""
v15·W10 侦察 —— 只读, 不写文件, 0 s 全流程。

问题: 审计表 P1-1「L5 涌现几何可测量指标」在 v15 全程未动, L4->L5 那堵墙
("桥梁为空") 一步都没走。本脚本试这座桥的一个具体候选:

    **类 RT 涌现距离** —— 从 MERA 张量网络的**最小割** (RT 面的离散对应物)
    导出边界弧的"面积" A_RT(n), 再看它能不能接上 L5 已有的**边界关联几何**
    (`boundary_correlation_graph(gs16, 16)`, 那条才依赖态)。

**两版废弃构造留在这里当教训 (都写错过, 都靠实测抓出来):**
  第一版: 把边界腿做成悬垂虚拟节点, 再收缩成两个超点。收缩会把**平行边并成
          一条**, 割值系统性少算, 给出 ceil(n/2) 的假线性。
  第二版: 腿把自己的张量用 INF 钉在源/汇侧。**任一张量同时挂着两侧的腿时
          不可行** —— 而这正是常态: quimb 的 MERA 把物理腿挂在 ndim=4 的
          **解纠缠器**上, 2 条腿共享一个张量 (实测底腿落在 8 个张量上, 全是
          4-腿张量)。于是 minimum_cut 悄悄返回退化划分, n=1 给出 0。
  第三版 (本版): 每条腿是**独立端点**, 腿->张量与张量->张量都可切 (各计 1),
          不做任何收缩。自检: 单条无向边隔开的割值 = 1, 正确。
          minimum_cut 在对称有向图上, 每条跨割无向边只计一次, 故割值即边数。

**顺序**: 先把各量量出来, 不在脚本里下"桥通了"的结论。
"""
import numpy as np
import networkx as nx

import spiral_model_v15 as G

INF = 10 ** 6


def edges_of(mera):
    Gr = G.build_mera_graph(mera)
    return frozenset(tuple(sorted(e)) for e in Gr.edges())


def tensor_of_leg(mera, i):
    """持有物理腿 k{i} 的张量 id。ind_map 的值是 oset, 不能下标。"""
    return int(next(iter(mera.ind_map[f'k{i}'])))


def arc_mincut(core, mera, L, n, want_cut=False):
    """
    RT 面 (第三版, 良定义): 弧 {0..n-1} 的腿接到 SRC, 其余接到 SNK (容量 INF);
    腿与张量之间、张量与张量之间容量都是 1 —— 处处可切, 不存在不可行。
    返回割边数 (跨割的无向边数)。
    """
    H = nx.DiGraph()
    for a, b in core.edges():
        H.add_edge(a, b, capacity=1.0)
        H.add_edge(b, a, capacity=1.0)
    for i in range(L):
        leg = ('L', i)
        t = tensor_of_leg(mera, i)
        H.add_edge(leg, t, capacity=1.0)
        H.add_edge(t, leg, capacity=1.0)
    for i in range(L):
        leg = ('L', i)
        term = 'SRC' if i < n else 'SNK'
        H.add_edge(term, leg, capacity=INF)
        H.add_edge(leg, term, capacity=INF)
    val, (S, _T) = nx.minimum_cut(H, 'SRC', 'SNK')
    if not want_cut:
        return val, None
    cut = [(a, b) for a, b in core.edges() if (a in S) != (b in S)]
    return val, cut


def main():
    print('=' * 88)
    print(' W10 侦察: 类 RT 涌现距离 (MERA 最小割 = RT 面的离散对应物)')
    print('=' * 88)

    # ---------------- 0. 先留证: MERA 图对 seed 的依赖 ----------------
    print('\n【0】MERA 图是不是"态"的函数? (决定 A_RT 能不能当涌现距离)')
    for L in (16, 32):
        es = []
        for seed in (0, 1, 7):
            m, _ = G.mera_init(L, 2, seed=seed)
            es.append(edges_of(m))
        print(f'  L={L}: seed 0/1/7 边集完全相同 = '
              f'{all(e == es[0] for e in es)}  (|E|={len(es[0])})')
    print('  -> build_mera_graph 只读 ind_map; mera_fit* 只调 t.modify(data=...)')
    print('     从不改接线。所以 A_RT 是 **(L, chi, 二进制布局) 的函数, 与态无关**。')

    # ---------------- 自检 ----------------
    T = nx.DiGraph()
    for _a, _b in (('SRC', 'u'), ('u', 'SRC'), ('SNK', 'v'), ('v', 'SNK')):
        T.add_edge(_a, _b, capacity=INF)
    T.add_edge('u', 'v', capacity=1.0)
    T.add_edge('v', 'u', capacity=1.0)
    _chk = nx.minimum_cut(T, 'SRC', 'SNK')[0]
    print(f'\n  [自检] 单条无向边隔开的割值 = {_chk}  (应为 1)')

    # ---------------- 1. RT 面积律 ----------------
    print('\n【1】A_RT(n) = 隔开 n 站点连续弧的最小割 —— ∝log n 还是 ∝n?')
    for L in (32, 16):
        m, _ = G.mera_init(L, 2, seed=0)
        core = G.build_mera_graph(m)
        ns = np.arange(1, L // 2 + 1)
        cuts = np.asarray([arc_mincut(core, m, L, int(n))[0] for n in ns],
                          dtype=float)
        print(f'\n  L={L}:  |V|={core.number_of_nodes()} '
              f'|E|={core.number_of_edges()}')
        print(f'    n        = {ns.tolist()}')
        print(f'    A_RT(n)  = {cuts.astype(int).tolist()}')
        for tag, x in (('log2 n', np.log2(ns)), ('n', ns.astype(float))):
            A = np.vstack([x, np.ones_like(x)]).T
            coef, *_ = np.linalg.lstsq(A, cuts, rcond=None)
            r2 = float(1.0 - (cuts - A @ coef).var() / cuts.var())
            print(f'    拟合 {tag:<7}: R^2 = {r2:.4f}   斜率 = {coef[0]:.4f}')
        d = np.diff(cuts)
        print(f'    单调: {bool((d >= 0).all())}   下降处 n = '
              f'{[int(ns[i+1]) for i in range(len(d)) if d[i] < 0]}')
        print(f'    A_RT(1)={int(cuts[0])}  A_RT(L/2)={int(cuts[-1])}  '
              f'log2(L/2)={float(np.log2(ns[-1])):.2f}')

    # ---------------- 2. 桥: 对得上边界关联几何吗 ----------------
    print('\n【2】桥: A_RT(n) 与 L5 已有的边界关联距离 D_corr(n) 对照 (L=16)')
    L = 16
    m, _ = G.mera_init(L, 2, seed=0)
    core = G.build_mera_graph(m)
    ns = np.arange(1, L // 2 + 1)
    cuts = np.asarray([arc_mincut(core, m, L, int(n))[0] for n in ns],
                      dtype=float)

    gs_ret = G.exact_ground_state(L, 1.0, 1.0)
    gs16 = gs_ret[1] if isinstance(gs_ret, tuple) else gs_ret
    _, C, _ = G.boundary_correlation_graph(gs16, L)
    cc = np.zeros(len(ns))
    for k, n in enumerate(ns):
        cc[k] = float(np.mean([abs(C[i, (i + n) % L]) for i in range(L)]))
    Dcorr = 1.0 - cc

    print(f'    {"n":>4} {"A_RT":>6} {"|C(n)|":>9} {"D_corr=1-|C|":>13}')
    for k, n in enumerate(ns):
        print(f'    {int(n):>4} {int(cuts[k]):>6} {cc[k]:>9.4f} {Dcorr[k]:>13.4f}')
    print(f'\n    A_RT   : 单调增? {bool((np.diff(cuts) >= 0).all())}, '
          f'n=8 处 / n=1 处 = {cuts[-1]/cuts[0]:.1f}x')
    print(f'    D_corr : 单调增? {bool((np.diff(Dcorr) >= 0).all())}, '
          f'n=8 处 / n=1 处 = {Dcorr[-1]/Dcorr[0]:.2f}x, '
          f'后四步增量 {np.diff(Dcorr)[-4:].round(4).tolist()}')
    print(f'\n    Pearson r( A_RT , D_corr )  = '
          f'{float(np.corrcoef(cuts, Dcorr)[0, 1]):+.4f}')
    print(f'    Pearson r( A_RT , log2 n )  = '
          f'{float(np.corrcoef(cuts, np.log2(ns))[0, 1]):+.4f}')
    print(f'    Pearson r( D_corr, log2 n ) = '
          f'{float(np.corrcoef(Dcorr, np.log2(ns))[0, 1]):+.4f}')
    print('\n    读法: A_RT 随 n 无界增长, D_corr 有界 (<=1) 且趋于饱和 ——')
    print('          若两者函数形式不同, 它们就不是同一条距离, 桥不闭合。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
