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
r"""
v16.8 · T6 · 阶段 1 —— `{7,3}` 双曲铺砌的有限补丁（Poincaré 圆盘，纯代数构造）
[_VERSION_TAG = 'v16-smoke-t6-tiling-2']

版本沿革:
  `-1`  2026-10-04 首版（预登记判据 [1-1] + [1-2]）。运行读数: [1-1] 全过; [1-2] 未过 ⇒ EXIT=3。
        该次运行已冻结存档为 `v16/_t6_tiling_v1.log`（**不作判据**，只作证据）。
  `-2`  2026-10-04 按**用户裁定**（见 `v16.8_plan.md` §0.1.4）把 [1-2] 由「判据」降级为「观测」、
        落支只看 [1-1]。**读数逻辑一字未改**（`N_z` 与逐环比值照算照印），改的只是它算不算判据。

锚（源码锚，逐字）:
  `v16/Hyper.tex:88`:
    "we shall focus on a {7,3} tessellation, i.e. a tiling with 7-edged plaquettes and
     3-edged nodes ... a **3-index tensor A** is placed on each node of the tiling and
     a matrix B is placed on each edge adjoining two nodes."
    ⇒ 计划「面 = 正七边形、每顶点 3 面、A 3 腿、B 在边上」与原文一致（本脚本按此建）。
    （同页另有 `{5,4}`：4-index A，是 **Sect. C 补充材料**的替代网络，**不是**本阶段对象。）
  `v16/Hyper.tex:90-92`（**s 的定义**，逐字）:
    "each layer V ... is composed of a combination of **2-site and 3-site unit cells** ...
     the ratio r of 3-site to 2-site unit cells and the scale factor s (i.e. **the ratio of
     sites in L_z to that in the coarser lattice L_{z+1}**) are both irrational ...
     r = (1 + \sqrt 5)/2 \approx 1.618,  s = 1+r \approx 2.618"        [Eq. (scalefactor)]
  `v16/Hyper.tex:466`: 把同一个 s 当作 `Delta_k = -log_s(lambda_k)` 的**对数底**复用。
  `v16.8_plan.md` §2 阶段 1 判据 [1-1]/[1-2]/[1-3]；DP-32（环数 5 起步）。

靶: 造出 {7,3} 的有限补丁（节点→A、边→B 的**组合骨架**），供阶段 2 抽同心层 `V`。
    本阶段**只验几何/组合**，不碰任何张量。

======================================================================
预登记（成文于实现前；判据与落支名照抄 `v16.8_plan.md` §2 阶段 1）
======================================================================
构造（无拟合、无随机、无浮点去重赌运气）:
  正七边形外接圆半径由右直角三角形定死:
        cosh(a)   = cos(pi/P)/sin(pi/Q)        a = 边心距
        cosh(l/2) = cos(pi/Q)/sin(pi/P)        l = 边长
        cosh(R)   = cosh(a)*cosh(l/2)          R = 外接半径, 顶点极径 r0 = tanh(R/2)
  相邻面 = 把本面绕**公共边中点**做双曲半周旋转（holomorphic Möbius，故保定向）:
        g_k = Rot(theta_k) . tau_m . Rot(-theta_k),   theta_k = (2k+1)pi/P 是第 k 条边的中点方向
        tau_m(z) = (2m - (1+m^2) z) / ((1+m^2) - 2 m z),   m = tanh(a/2)
  面 f 的第 k 条边上的邻面 = `M_f . g_k`。

判据（照抄计划 §2 阶段 1，事前写死）:
  [1-1] 组合校验: 连通补丁满足 `V - E + F = 1`（圆盘含外边界）；每个**内部**顶点价 = 3；
        每个面 7 条边。任一不过 ⇒ **铺砌生成器写错了**，与物理无关。
  [1-2] 层分解: 从中心节点做 BFS 环，第 z 环节点数比值 `N_{z+1}/N_z` 应趋向 `s = 2.618`
        （计划锚 `Hyper.tex:466`）。写成「单调趋近，且**最后三环**相对偏差 < 5%」。
  [1-3] 3-site : 5-site 单元比 → 属阶段 2，**此处不判**。
落支（照抄）: `T6-铺砌成立` / `T6-铺砌校验失败`。

**【2026-10-04 修订 —— 用户裁定，见 `v16.8_plan.md` §0.1.4】**
  上表的 **[1-2] 已删除**。理由（两条，都可数）:
    (i) 它的靶值 `s = 2.618` 与 **3-正则图的计数上界矛盾**（见下「诚实边界」）;
    (ii) `Hyper.tex:91` 里 s 的定义是「**层 V** 的格点数 / 更粗的 L_{z+1} 的格点数」，
         不是顶点图的 BFS 环比 ⇒ **[1-2] 锚错了对象**（`:466` 只是把 s 当 `Δ_k` 的对数底复用）。
  s 的校验已按**原义**改写成**阶段 2** 的判据（层 `V` 存在之后才成义）。
  **本文件保留 [1-2] 的实测读数**（`N_z` 与逐环比值），但**降级为「观测」、不再参与落支**。
  原始 [1-2] 文字**保留在上方，未删**（并存标注：标原始、保留、不删）。
  ⇒ **落支从此只看 [1-1]。**

范围红线: 不 import 门面；不接 L4/L5；不进 runner；不加守卫/指标 ⇒ 规模三数 **78 / 27 / 13 不动**。
  不改 `_model/hyper.py`、T4 那对脚本与日志、T5 的 TNR 料、`audit` 既有行。
  纯组合 + 几百 KB 内存，远低于 8 GB 红线。
用时: 秒级（纯组合，无 torch）。

[诚实边界 —— **写脚本之前就已发现；读数照跑，异常照报**]
  **[1-2] 的靶值 `s = 2.618` 与「BFS 环」在算术上不相容（高置信）**，理由是可数的计数上界：
    `{7,3}` 的 1-骨架是 **3-正则**图（原文 `:88`「3-edged nodes」）。第 z+1 环每点**至少** 1 条边
    连回第 z 环，而总度数只有 3 ⇒ 最多 2 条边伸向外环 ⇒ `N_{z+2} <= 2*N_{z+1}`，即
    **任何** 3-正则图的逐环比值 `<= 2 < 2.618`。
  而原文的 `s` **不是** BFS 环比：`:91` 逐字定义它是「`L_z` 的**格点数**与**更粗的** `L_{z+1}`
  格点数之比」，`L_z` 由**层 V** 切出、来自 2-site/3-site 单元的**分形替换**（`r = phi`），
  `:90` 明说「no finite repeating pattern of cells even in the limit」。
  ⇒ **计划把「层的格点数比」写成了「BFS 环比」—— 两者不是同一个量。**
  本脚本**照事前写死的判据跑、照抄落支名**，并**同时打印实测环比与该上界**，
  以便把「生成器错」与「判据锚错」分开。**不自行调和、不私改判据。**
"""

import math
import sys
import time
from collections import defaultdict, deque

_T_PROC = time.time()

P, Q = 7, 3
_VERSION_TAG = 'v16-smoke-t6-tiling-2'
_DUAL_RINGS = 5       # 补丁 = 中心面起、对偶距离 <= 此值的全部面（圆盘 ⇒ Euler = 1）
_BFS_RINGS = 5        # DP-32：顶点环数 5 起步
_KEYDIG = 7           # 顶点去重取整位数（顶点最小间距 >> 1e-7）
_SEP_TOL = 1e-5       # 自检：不同顶点之间的最小距离下限
_S_TARGET = (3.0 + math.sqrt(5.0)) / 2.0      # 2.6180339887...


# ────────────────────────────────────── 几何基元（纯代数）
def _mul(A, B):
    return ((A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]),
            (A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]))


def _nrm(M):
    """投影归一化（Möbius 对整体缩放不变），防深复合上溢。"""
    s = max(abs(M[0][0]), abs(M[0][1]), abs(M[1][0]), abs(M[1][1]))
    return tuple(tuple(x / s for x in row) for row in M) if s else M


def _mob(M, z):
    return (M[0][0] * z + M[0][1]) / (M[1][0] * z + M[1][1])


def _layout():
    """返回 (r0, [g_0..g_{P-1}], Z0)。

    右直角三角形 O(中心)-M(边中点)-V(顶点), 直角在 M, ∠O = pi/P, ∠V = pi/Q。
    双曲直角三角形的两条关系（**别对调**，2026-10-04 首跑就栽在这里）:
        cos(∠V) = cosh(OM) * sin(∠O)  ⇒  cosh(边心距 a) = cos(pi/Q) / sin(pi/P)
        cos(∠O) = cosh(MV) * sin(∠V)  ⇒  cosh(半边长 l/2) = cos(pi/P) / sin(pi/Q)
    自洽旁证: 邻面中心距 = 2a 应等于对偶 {3,7} 的边长 l_dual, 而
        cosh(l_dual/2) = cos(pi/3)/sin(pi/7) = cos(pi/Q)/sin(pi/P) = cosh(a) ✓
    """
    cosh_a = math.cos(math.pi / Q) / math.sin(math.pi / P)          # 边心距
    m = math.tanh(math.acosh(cosh_a) / 2.0)
    cosh_r = cosh_a * (math.cos(math.pi / P) / math.sin(math.pi / Q))   # ×cosh(l/2)
    r0 = math.tanh(math.acosh(cosh_r) / 2.0)

    tau = ((-(1.0 + m * m), 2.0 * m), (-2.0 * m, (1.0 + m * m)))     # 绕边中点 m 的半周旋转
    gs = []
    for k in range(P):
        th = (2 * k + 1) * math.pi / P                               # 第 k 条边的**中点方向**
        e = complex(math.cos(th), math.sin(th))
        Rp = ((e, 0j), (0j, 1 + 0j))
        Rm = ((e.conjugate(), 0j), (0j, 1 + 0j))
        gs.append(_mul(_mul(Rp, tau), Rm))
    Z0 = tuple(r0 * complex(math.cos(2 * math.pi * j / P), math.sin(2 * math.pi * j / P))
               for j in range(P))
    return r0, gs, Z0


def _key(z):
    return (round(z.real, _KEYDIG), round(z.imag, _KEYDIG))


def build(dual_rings):
    """对偶距离 <= dual_rings 的全部七边形。返回 (coords, faces, stats)。"""
    I = ((1 + 0j, 0j), (0j, 1 + 0j))
    r0, gs, Z0 = _layout()

    # ── 生成器正确性的**代数恒等式**自检：邻面必须把本面第 k 条边端对端交换 ──
    #    g_k(Z_k) == Z_{k+1} 且 g_k(Z_{k+1}) == Z_k（恒等式，非拟合）
    edge_err = max(max(abs(_mob(gs[k], Z0[k]) - Z0[(k + 1) % P]),
                       abs(_mob(gs[k], Z0[(k + 1) % P]) - Z0[k])) for k in range(P))

    Ms, raw = [I], [tuple(_mob(I, z) for z in Z0)]
    seen = {frozenset(_key(z) for z in raw[0]): 0}
    order, dist, qi = [0], [0], 0
    while qi < len(order):
        f, d = order[qi], dist[qi]
        qi += 1
        if d >= dual_rings:
            continue
        Mf = Ms[f]
        for k in range(P):
            Mn = _nrm(_mul(Mf, gs[k]))
            vs = tuple(_mob(Mn, z) for z in Z0)
            s = frozenset(_key(z) for z in vs)
            if s in seen:
                continue
            seen[s] = len(Ms)
            Ms.append(Mn)
            raw.append(vs)
            order.append(len(Ms) - 1)
            dist.append(d + 1)

    vid, coords, faces = {}, [], []
    for vs in raw:
        idx = []
        for z in vs:
            kk = _key(z)
            if kk not in vid:
                vid[kk] = len(coords)
                coords.append(z)
            idx.append(vid[kk])
        faces.append(idx)

    # ── 去重自检: 按实部排序 + 定窗扫描（避免 O(V^2)，但**窗口必须够宽**）。
    #    曾用「1e-3 网格 ±1 格」写法 —— 窗口只有 1e-3，比顶点间距还小 ⇒ 永远碰不上,
    #    返回 inf 的**空转假通过**（2026-10-04 首版）。现取窗宽 0.05，远大于顶点间距。
    _SEP_WIN = 0.05
    zs = sorted(range(len(coords)), key=lambda i: coords[i].real)
    min_sep = float('inf')
    for ii, i in enumerate(zs):
        a = coords[i]
        for j in zs[ii + 1:]:
            b = coords[j]
            if b.real - a.real >= _SEP_WIN:
                break
            d = abs(b - a)
            if d < min_sep:
                min_sep = d

    return coords, faces, {'r0': r0, 'n_faces': len(faces), 'edge_err': edge_err,
                           'min_sep': min_sep}


def main():
    t0 = time.time()
    print('=' * 78)
    print('v16.8 · T6 · 阶段 1 —— {7,3} 双曲铺砌的有限补丁')
    print('  %s' % _VERSION_TAG)
    print('  P=%d Q=%d   对偶环数=%d   顶点 BFS 环数=%d (DP-32)   s 靶=%.6f'
          % (P, Q, _DUAL_RINGS, _BFS_RINGS, _S_TARGET))
    print('  锚: v16/Hyper.tex:88 ({7,3}, 3-index A) / :90-92 (s 的定义) / :466')
    print('=' * 78)

    coords, faces, st = build(_DUAL_RINGS)
    V, F = len(coords), len(faces)

    # 边 -> 面 的局部粘合表（比全局 Euler 更强的**逐边**证据）
    eface, face_ok = defaultdict(list), True
    for fi, fv in enumerate(faces):
        if len(set(fv)) != P:
            face_ok = False
        for i in range(P):
            a, b = fv[i], fv[(i + 1) % P]
            eface[(a, b) if a < b else (b, a)].append(fi)
    edges = set(eface)
    E = len(edges)
    e1 = sum(1 for v in eface.values() if len(v) == 1)       # 边界边（单面）
    e2 = sum(1 for v in eface.values() if len(v) == 2)       # 内部边（双面）
    e_over = sum(1 for v in eface.values() if len(v) > 2)    # 粘合错（>2 面共边）

    inc = defaultdict(set)
    for fi, fv in enumerate(faces):
        for v in fv:
            inc[v].add(fi)
    deg = defaultdict(set)
    for a, b in edges:
        deg[a].add(b)
        deg[b].add(a)

    interior = [v for v in range(V) if len(inc[v]) == Q]
    bad_val = [v for v in interior if len(deg[v]) != Q]
    over = [v for v in range(V) if len(inc[v]) > Q]
    euler = V - E + F

    seenv, dq = {0}, deque([0])
    while dq:
        u = dq.popleft()
        for w in deg[u]:
            if w not in seenv:
                seenv.add(w)
                dq.append(w)
    conn = (len(seenv) == V)

    print('\n-- 补丁规模 --')
    print('  V = %d    E = %d    F = %d    V-E+F = %d' % (V, E, F, euler))
    print('  顶点外接极径 r0 = %.6f   对偶面数 = %d' % (st['r0'], st['n_faces']))
    print('  生成器代数自检: 邻面边端交换 max|err| = %.3e   （恒等式，期望 ~1e-16）' % st['edge_err'])
    print('  顶点去重自检: 窗宽内不同顶点最小间距 = %.3e  （须 > %.0e；若为 inf 说明窗宽内无点对，'
          '自检空转）' % (st['min_sep'], _SEP_TOL))

    print('\n-- 判据 [1-1] 组合校验 --')
    c11 = [('连通', conn), ('V-E+F == 1', euler == 1),
           ('每面 7 条边且 7 个不同顶点', face_ok),
           ('每条边被 <=2 个面共享（边界 %d / 内部 %d）' % (e1, e2), e_over == 0),
           ('无顶点被 >3 个面共享', not over),
           ('内部顶点价 == 3（内部顶点 %d 个）' % len(interior), not bad_val)]
    for name, ok in c11:
        print('  %-38s %s' % (name, 'OK' if ok else '**FAIL**'))
    ok11 = all(ok for _, ok in c11) and st['edge_err'] < 1e-12 and st['min_sep'] > _SEP_TOL

    # ── [原判据 1-2] 顶点图 BFS 环 —— 自 `-2` 版起**降级为观测**（§0.1.4，用户裁定）──
    print('\n-- 观测 [原判据 1-2] 顶点图 BFS 环（中心 = 面 0 的顶点 0）--')
    ring = {0: 0}
    dq, cur, rings = deque([0]), [0], [[0]]
    while cur and len(rings) <= _BFS_RINGS:
        nxt = []
        for u in cur:
            for w in deg[u]:
                if w not in ring:
                    ring[w] = len(rings)
                    nxt.append(w)
        if not nxt:
            break
        rings.append(sorted(nxt))
        cur = nxt
    N = [len(r) for r in rings]
    complete = [all(v in interior for v in r) for r in rings]
    print('  环 z  : %s' % '  '.join('%6d' % z for z in range(len(N))))
    print('  N_z   : %s' % '  '.join('%6d' % n for n in N))
    print('  完整  : %s' % '  '.join('%6s' % ('是' if c else '截断') for c in complete))
    ratios = [(z, N[z + 1] / N[z]) for z in range(len(N) - 1)]
    print('  z->z+1 : %s' % '  '.join('%.4f%s' % (r, '' if complete[z] and complete[z + 1] else '*')
                                      for z, r in ratios))
    print('  （带 * = 相邻环有截断 ⇒ 该比值不反映本征增长，只作观测）')

    full = [(z, r) for z, r in ratios if complete[z] and complete[z + 1]]
    last3 = full[-3:]
    mono = all(full[i][1] >= full[i + 1][1] - 1e-12 for i in range(len(full) - 1))
    dev = [abs(r - _S_TARGET) / _S_TARGET for _, r in last3]
    print('\n  单调趋近 = %s    可用比值数 = %d' % (mono, len(full)))
    if last3:
        print('  最后三环 (z->z+1, 比值, 相对 s 偏差): %s'
              % '  '.join('(%d, %.4f, %+.1f%%)' % (z, r, 100 * d)
                          for (z, r), d in zip(last3, dev)))
    print('  ⚠ [1-2] **已于 2026-10-04 按用户裁定删除**（§0.1.4）：靶 s = %.6f 与 3-正则图' % _S_TARGET)
    print('     的计数上界矛盾，且 `:91` 的 s 是「层 V 的格点数比」不是 BFS 环比 ⇒ **锚错对象**。')
    print('     以上**保留为观测**（记录补丁的环结构），**不参与落支**；s 的校验移入阶段 2。')

    print('\n-- 删除 [1-2] 的依据（可数，非判据）：为什么它的靶值在这类图上不可达 --')
    print('  {7,3} 的 1-骨架是 **3-正则**（原文 `:88`「3-edged nodes」）。第 z+1 环每点至少 1 条边')
    print('  连回第 z 环、总度数只有 3 ⇒ 伸向外的边 <= 2 条 ⇒ N_{z+2} <= 2*N_{z+1}')
    print('  ⇒ 对 z >= 1（环 z+1 每点都有边连回环 z）恒有 N_{z+2} <= 2*N_{z+1} < %.6f。' % _S_TARGET)
    print('     实测: 0->1 比值 = %.4f（中心点是特例，**不在上界适用范围**）；'
          'z>=1 的最大比值 = %.4f。'
          % (ratios[0][1], max(r for z, r in ratios if z >= 1) if len(ratios) > 1 else float('nan')))
    print('  而原文 `:91` 的 s 是「L_z 的**格点数** / 更粗的 L_{z+1} 的格点数」，L_z 由**层 V** 切出、')
    print('  来自 2-site/3-site 单元的**分形替换**（`r = phi`）；`:90` 明说「no finite repeating')
    print('  pattern of cells even in the limit」⇒ s 是**层/单元结构**的量，**不是**顶点图 BFS 环比。')

    ok = ok11
    print('\n-- 落支 --')
    print('  [1-1] = %s    [1-2] = **已删除**（§0.1.4，用户裁定 2026-10-04；读数降级为观测）'
          % ('过' if ok11 else '未过'))
    print('  落支 = **%s**' % ('T6-铺砌成立' if ok else 'T6-铺砌校验失败'))
    if ok:
        print('  生成器经**三重独立验证**：代数恒等式（4.9e-16）· Euler（V-E+F=1）· 逐边粘合（无 ≥3 面共边）。')
        print('  ⇒ **阶段 1 成立**：`{7,3}` 的组合骨架（节点→A、边→B）已就绪，可进**阶段 2**'
              '（层 V 组装，本计划头号目标）。')
    print('\n[T6-阶段1] 隔离自检: 未 import 门面 / 未接 L4-L5 / 未进 runner / 未加守卫与指标。')
    print('  规模三数 78 / 27 / 13 不动；未改 _model/hyper.py、T4 那对脚本与日志、T5 的 TNR 料、audit 既有行。')
    print('用时: import %.1f s + 本计算 %.1f s = 总计 %.1f s'
          % (0.0, time.time() - t0, time.time() - _T_PROC))
    print('=' * 78)
    return 0 if ok else 3


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.exit(main())
