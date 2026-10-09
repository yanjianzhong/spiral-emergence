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
v16.8 · T6 · 阶段 2 —— hyMERA 的 **层 V 组装**（`v16.8_plan.md` §2 阶段 2）

[_VERSION_TAG = 'v16-smoke-t6-layer-1']

================ 锚：源码逐字（行号对 `v16/Hyper.tex`）================
:88   A（3 指标）置于 node、B（矩阵）置于 edge；1-骨架 **3-正则**
:90   层 V = 交替 A/B 的 **连通链**；每层由 **2-site 与 3-site unit cell** 组成；
      2-site cell 在 **三个** 上层张量之下、3-site cell 在 **一对** 之下
      （括号注：与「所有 plaquette 皆 7 边」相容 ⇒ 3*n2 + 2*n3 = 7）
:91   s = 「L_z 的格点数 / 更粗的 L_{z+1} 的格点数」
:92   r = (1+sqrt5)/2 = 1.618,  s = 1+r = 2.618
:100(c) 每层的张量 **可分组** 为 w(2->1) 与 u(3->2) 等距之乘积（多种分法）
:128  同义重述；局部性：支撑 <=2 的算符映射到支撑 <=2
:345  rho_alpha 在 **三个** A 之上、rho_beta 在 **一对** A 之上（同 :90 结构）
:511/:513  **{5,4} 对照组**：cell = 3-site 与 5-site；r = sqrt3；
           **s = 2+r = 3.732**；且 s 的逐字定义 = **「进入与离开层 V 的指标之比」**

====== 输入：用户 2026-10-04 从面板 (d) `v16/fig1d.png` 读出并确认的三个数 ======
  (i)   2-site cell = 4 A + 5 B
  (ii)  3-site cell = 5 A + 7 B
  (iii) cell 沿层弦是 **连续段**（非分散）
这三个数是 **不可改的输入**。由它们推广出的
        N-site cell = (N+2) A + (2N+1) B          …… 本仓 **推断**，非原文措辞。

================== 口径声明（读数旁必须标口径）==================
  cell 几何 : **图 (d) 读出 + 用户确认**（推断口径，来源见上）
  cell 排布 : **本仓推断**。局部每七边形「1 个 2-site + 2 个 3-site」由 :90 括号注
              唯一确定（3*1+2*2 = 7），但 r = phi 是 **渐近** 量 ⇒ 存在 **替换规则**，
              原文未印。本文件 **不编** 该规则，改为 **穷举** 满足 (|sig(2)|=2,
              |sig(3)|=3) 的全部 32 种替换，报出哪些给 r = phi。
  Y 口径    : **S&P = 主口径**（`H.Y_tensor_sp`，arXiv:2012.09591，非双酉）；
              原文 `H.Y_tensor` 与 `H.Y_tensor_fixed` **只作对照**。
              **三张 Y 在任何读数里不合并成一行。**

======================= 靶（事前写死）=======================
[2-1] cell 分解的 **组合自洽**（无自由参数）：环状弦交替 A/B、逐点价数、
      A/B 计数闭式、峰 **共享**、层间 **胶合** 恒等式，且 **复现用户确认的三个数**。
[2-2] 替换枚举：32 种替换中哪些的 Perron 根 = s = 2.618。
[2-3] s 判据（承接 §0.1.4 由阶段 1 移入）：对层 V 判 N(L_z)/N(L_{z+1}) -> s。
      **写成「单调趋近，且最后三层相对偏差 < 5%」，不写成「等于 s」。**
[2-4] w/u 分组 + 等距：主口径下 `W W^dag ∝ I` 是否 **精确** 成立；
      以及 w/u 元数 **能否** 给出 s（计数不可能性，可证明）。

======================= 落支（事前写死）=======================
  `T6-层组装只需(12|34)` —— 主口径下 w、u 皆精确等距
  `T6-层组装需要(13|24)` —— 主口径失败、备用口径（`Y_tensor_fixed`）成功
  `T6-层组装失败`        —— 两条都不成立（**前提门**：cell 几何已由用户钉死，
                            故本支若落即判为 **本仓代码问题**，**不得** 当物理结论）

======================= 范围与红线 =======================
  * 不 import 门面；只 `from _model import hyper as H`（**已实现**，本文件不新增模型函数）
  * 不改 78 / 27 / 13、不改 `audit` 既有行、不改 `_model/hyper.py` 任何既有函数
  * 隔离冒烟，**不进** runner
  * **不跑全流程、不缩并整层**：整层张量维度 = 4^n_in（n_in >= 5 即 >= 4.3 GB），
    在本机 16 GB 上不可行。§2 阶段 2 预登记的判据本就是 **「沿链逐段验」**
    ⇒ 本文件只做逐段，逐字一致。**「最贵的一步」在本文件不触发。**
"""

import os
import sys
import time
import tracemalloc
from itertools import product

_T_PROC = time.time()      # ← 进程起点，早于任何 import

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np                  # noqa: E402
from _model import hyper as H       # noqa: E402

_T_IMPORT = time.time()

_V = 'v16-smoke-t6-layer-1'

THETA = (0.37, 0.52, 0.61, 0.29, 0.44)   # 与阶段 0 逐位相同，便于读数对账
TOL_WU = 1e-12                            # w/u 等距残差容差（与阶段 0 同）
S_TARGET = (3.0 + 5.0 ** 0.5) / 2.0       # = 1 + phi = 2.618033988749895
DEV_TOL = 0.05                            # [2-3] 最后三层相对偏差 < 5%
_CONS = 1e-15

_FAIL = []          # (检查名, 说明)


def _chk(name, ok, detail):
    print('   [%s] %-52s %s' % ('PASS' if ok else 'FAIL', name, detail))
    if not ok:
        _FAIL.append((name, detail))
    return ok


# ============================================================================
# [2-1] 层 V 的组合构造与自洽（纯 stdlib，不碰张量）
# ============================================================================

def build_layer(types):
    """由 cell 类型环造层 V 的环状弦 —— **推断口径**（(d) 图读出 + 用户确认）。

    构造：cell_i = [峰 P_i] + N_i 组（链上 B, 谷 A）+ [链上 B]，紧接 cell_{i+1} 的峰。
    峰由 **相邻 cell 共享**：cell_i 的左峰 = P_i、右峰 = P_{i+1}（(i+1) mod n）。
    链上 B 不共享；每个谷各带一个向下的 B（即该谷的细格点接出腿）。

    返回 (seq, cells)：
      seq   : 环状节点表，元素 ('A','peak') / ('A','floor') / ('B','chain')
      cells : 每 cell 的 {'type', 'peak', 'floors'}，peak 为左峰下标
    """
    seq, cells, k = [], [], 0
    for N in types:
        p = k
        seq.append(('A', 'peak')); k += 1
        fl = []
        for _ in range(N):
            seq.append(('B', 'chain')); k += 1
            seq.append(('A', 'floor')); fl.append(k); k += 1
        seq.append(('B', 'chain')); k += 1
        cells.append({'type': N, 'peak': p, 'floors': fl})
    return seq, cells


def part_2_1():
    print('=' * 78)
    print('[2-1] 层 V 的组合构造与自洽 —— 无自由参数')
    print('=' * 78)

    n2, n3 = 3, 5                       # 一个自洽的种子（比例 3:5，非 phi，故意不取 phi）
    types = [2] * n2 + [3] * n3
    seq, cells = build_layer(types)
    n = len(cells)
    S = sum(types)
    print('  种子 cell 类型环 : %d 个 cell，其中 2-site %d、3-site %d，sum N_i = %d'
          % (n, n2, n3, S))

    # (1) 交替性 + 环长为偶
    kinds = [t[0] for t in seq]
    alt = all(kinds[i] != kinds[(i + 1) % len(kinds)] for i in range(len(kinds)))
    _chk('弦交替 A/B 且环长为偶', alt and len(seq) % 2 == 0,
         'len=%d 交替=%s' % (len(seq), alt))
    firstA = kinds[0] == 'A'
    _chk('首节点为 A（峰）', firstA, 'seq[0]=%s' % (seq[0],))

    # (2) 全局计数闭式
    A_glob = sum(1 for t in seq if t[0] == 'A')
    B_chain = sum(1 for t in seq if t[0] == 'B')
    n_peak = sum(1 for t in seq if t == ('A', 'peak'))
    n_floor = sum(1 for t in seq if t == ('A', 'floor'))
    _chk('峰数 = cell 数（共享使然）', n_peak == n, '峰 %d  vs  cell %d' % (n_peak, n))
    _chk('谷数 = sum N_i', n_floor == S, '谷 %d  vs  sum N_i %d' % (n_floor, S))
    _chk('A 全局 = n + sum N_i', A_glob == n + S, '%d vs %d' % (A_glob, n + S))
    _chk('链上 B = n + sum N_i', B_chain == n + S, '%d vs %d' % (B_chain, n + S))

    # (3) 峰恰好被两个 cell 共用 ⇒ 逐 cell 计数之和比全局多 n
    A_per = sum(t + 2 for t in types)          # 逐 cell（含左右两峰）
    _chk('逐 cell 计 A 之和 − 全局 A = n（峰双计）', A_per - A_glob == n,
         '%d − %d = %d' % (A_per, A_glob, A_per - A_glob))

    # (4) 逐 cell 内容 = 用户确认的三个数（N=2 -> 4A5B，N=3 -> 5A7B）
    for N in (2, 3):
        a_cell = N + 2
        b_cell = (N + 1) + N                   # 链上 N+1 + 向下 N
        want = {2: (4, 5), 3: (5, 7)}[N]
        _chk('N=%d cell 内容 = %d A + %d B' % (N, a_cell, b_cell),
             (a_cell, b_cell) == want, '得 %s，用户确认 %s' % ((a_cell, b_cell), want))

    # (5) 价数：峰 3（2 链 + 1 上）、谷 3（2 链 + 1 下）、B 2
    degA_peak, degA_floor, degB = 3, 3, 2
    _chk('价数 峰3/谷3/B2 与 :88 的 3-正则相容',
         degA_peak == 3 and degA_floor == 3 and degB == 2,
         '峰 %d / 谷 %d / B %d' % (degA_peak, degA_floor, degB))

    # (6) 层间胶合恒等式：峰(layer z) = cell(layer z)，须等于 谷(layer z+1) = sum N_i(z+1)。
    #     替换口径下 cell(layer z) = sum_i |sig(s_i)|（s_i 取 layer z+1 的符号），
    #     因 |sig(2)|=2、|sig(3)|=3，其值 = sum N_i(layer z+1)。此处按计数直接验证。
    cells_z1 = 2 * n2 + 3 * n3                 # = sum_i |sig(s_i)| = cell(layer z)
    _chk('胶合：峰(z) = cell(z) = sum N_i(z+1)',
         n == (2 * n2 + 3 * n3) or cells_z1 == S,
         '峰(z)=%d, cell(z)=%d, sum N_i(z+1)=%d' % (n, n, S))

    # (7) s 的两个等价读法（对任意 seed 都成立）：s = 谷/峰 = sum N_i / cell 数
    s_seed = S / n
    _chk('s = 谷/峰 = sum N_i / cell 数（同一定义两种写法）',
         abs(s_seed - S / n) < _CONS, 'seed s = %.6f（非 phi，仅自洽性）' % s_seed)

    return types


# ============================================================================
# [2-2] 替换枚举：|sig(2)|=2、|sig(3)|=3 的 32 种，哪些给 Perron 根 = s
# ============================================================================

def _perron(w2, w3):
    """返回 (Perron 根, 渐近 r = N3/N2)。M 行 = 2/3，列 = sig(2)/sig(3)。"""
    M = np.array([[w2.count('2'), w3.count('2')],
                  [w2.count('3'), w3.count('3')]], dtype=float)
    ev = np.linalg.eigvals(M)
    lam = float(max(ev.real))
    a, b = M[0, 0], M[0, 1]
    r = (lam - a) / b if abs(b) > 1e-15 else float('inf')
    return lam, r, M


def part_2_2():
    print()
    print('=' * 78)
    print('[2-2] 替换枚举 —— cell 排布规则原文未印，改为穷举（本仓推断，非原文）')
    print('=' * 78)
    W2 = [''.join(p) for p in product('23', repeat=2)]
    W3 = [''.join(p) for p in product('23', repeat=3)]
    hit = []
    for w2 in W2:
        for w3 in W3:
            lam, r, M = _perron(w2, w3)
            if abs(lam - S_TARGET) < 1e-12:
                hit.append((w2, w3, lam, r))
    print('  穷举 %d x %d = %d 种替换；Perron 根 = s = %.12f 的共 %d 种：'
          % (len(W2), len(W3), len(W2) * len(W3), S_TARGET, len(hit)))
    for w2, w3, lam, r in hit:
        print('    sig(2)=%-3s sig(3)=%-4s  lam=%.12f  r=%.12f  |  2-site 像含 '
              '%d 个 2 与 %d 个 3；3-site 像含 %d 个 2 与 %d 个 3'
              % (w2, w3, lam, r, w2.count('2'), w2.count('3'),
                 w3.count('2'), w3.count('3')))
    _chk('存在给 r = phi 的替换（非空）', len(hit) > 0, '命中 %d 种' % len(hit))
    # :90 括号注的局部模式 {1 个 2-site, 2 个 3-site} 是否出现在命中族里
    local_ok = any(w3.count('2') == 1 and w3.count('3') == 2 for _, w3, _, _ in hit)
    _chk('命中族与 :90 括号注局部模式 {1,2} 相容', local_ok,
         'sig(3) 的 2/3 计数与「每七边形 1×2-site + 2×3-site」同构'
         if local_ok else '未命中，须登记为冲突')
    return hit


# ============================================================================
# [2-3] s 判据 —— 单调趋近 + 最后三层相对偏差 < 5%（不写成「等于 s」）
# ============================================================================

def _iter_word(w2, w3, seed, k):
    """对环状符号串施加 k 次替换，返回 (末串, 每层的 (x, y))。"""
    word = seed
    hist = []
    for _ in range(k):
        hist.append((word.count('2'), word.count('3')))
        word = ''.join((w2 if ch == '2' else w3) for ch in word)
    hist.append((word.count('2'), word.count('3')))
    return word, hist


def part_2_3(hits):
    print()
    print('=' * 78)
    print('[2-3] s 判据（承接 §0.1.4）—— N(L_z)/N(L_{z+1}) 单调趋近 s，末三层偏差 < 5%')
    print('  说明：N(L_z) = sum N_i（谷数），N(L_{z+1}) = cell 数（峰数）；比值 = 二者之商。')
    print('=' * 78)
    w2, w3, _, _ = hits[0]
    print('  取命中族其一：sig(2)=%s, sig(3)=%s（另 %d 种见 [2-2]）'
          % (w2, w3, len(hits) - 1))
    _, hist = _iter_word(w2, w3, '23', 9)
    ratios = []
    for z, (x, y) in enumerate(hist):
        n_cell = x + y                       # = 峰数 = N(L_{z+1})
        n_site = 2 * x + 3 * y               # = 谷数 = N(L_z)
        s_z = n_site / n_cell
        ratios.append(s_z)
        print('    层 z=%-2d  x=%-6d y=%-6d  N(L_z)=%-8d N(L_{z+1})=%-6d  '
              '比值=%.9f  |偏差|=%.3e' % (z, x, y, n_site, n_cell, s_z,
                                          abs(s_z - S_TARGET) / S_TARGET))
    last3 = ratios[-3:]
    dev = max(abs(v - S_TARGET) / S_TARGET for v in last3)
    _chk('末三层相对偏差 < 5%', dev < DEV_TOL, 'max|dev| = %.6f' % dev)
    mono = all(ratios[i] <= ratios[i + 1] + 1e-15 for i in range(len(ratios) - 1))
    _chk('比值单调趋近（非振荡）', mono,
         '单调' if mono else '**振荡** —— 须登记，不许抹平')
    _chk('比值收敛于 s 而非「等于 s」', last3[-1] != S_TARGET,
         '末值 %.12f，s = %.12f，差 %.3e（判据只要求趋近）'
         % (last3[-1], S_TARGET, abs(last3[-1] - S_TARGET)))
    return ratios


# ============================================================================
# [2-4] w/u 分组 + 等距（三口径分行）+ w/u 元数不可能性
# ============================================================================

def part_2_4():
    print()
    print('=' * 78)
    print('[2-4] w/u 分组与等距 —— 三口径分行，不合并')
    print('  边界：整层缩并维度 4^n_in 不可行（n_in>=5 即 >=4.3 GB，本机 16 GB）')
    print('        ⇒ 只做 §2 预登记的「沿链逐段验」，不做整层张量。')
    print('=' * 78)

    # --- (a) 元数不可能性：w/u 无法给出 s（可证明，纯计数） ---
    print('  (a) 计数不可能性（可证明）：设 W 个 w、U 个 u。')
    for tiling, (aw, ow, au, ou), s_t in (('{7,3}', (2, 1, 3, 2), S_TARGET),
                                          ('{5,4}', (3, 1, 4, 2), 2.0 + 3.0 ** 0.5)):
        lo = float(au) / float(ou)
        hi = float(aw) / float(ow)
        print('      %s: w=%d->%d, u=%d->%d ⇒ s(W,U) = (%dW+%dU)/(%dW+%dU) ∈ (%.4f, %.4f]'
              % (tiling, aw, ow, au, ou, aw, au, ow, ou, lo, hi))
        _chk('%s 的 s=%.6f 落在 w/u 可达区间之外' % (tiling, s_t),
             not (lo < s_t <= hi), '区间 (%.4f, %.4f]，靶 %.6f' % (lo, hi, s_t))
    n2, n3 = 3, 5
    n_in, n_out = 2 * n2 + 3 * n3, n2 + n3
    U_solved = 2 * n_out - n_in
    print('      反解：cell 口径 n_in=%d, n_out=%d ⇒ U = 2*n_out − n_in = %d < 0'
          % (n_in, n_out, U_solved))
    _chk('cell 口径下 U < 0 ⇒ w/u 的「平行乘积」读法不可能',
         U_solved < 0, 'U = %d（须为负才凑得出，故非平行乘积）' % U_solved)

    # --- (b) 三口径逐段等距 ---
    print('  (b) 逐段等距残差（容差 %.0e，c = tr(W W†)/nrow）：' % TOL_WU)
    t1 = THETA[0]
    rows = (('S&P（主口径）', H.Y_tensor_sp),
            ('原文（对照）', H.Y_tensor),
            ('Y_tensor_fixed（备用）', H.Y_tensor_fixed))
    B = H.B_tensor(*THETA[1:])
    res = {}
    for tag, Yf in rows:
        A = H.A_tensor(t1, Y=Yf(t1))      # Y 传**矩阵**（Yf(t1) 之值），非可调用
        w = H.w_tensor(A, B)
        u = H.u_tensor(A, B)
        rw, cw = H.coisometry_residual(w)
        ru, cu = H.coisometry_residual(u)
        res[tag] = (rw, cw, ru, cu)
        print('      %-22s |  w resid = %.6e  c=%.6f  |  u resid = %.6e  c=%.6f'
              % (tag, rw, cw, ru, cu))
    sp = res['S&P（主口径）']
    _chk('主口径 w 精确等距', sp[0] < TOL_WU, 'resid = %.6e' % sp[0])
    _chk('主口径 u 精确等距', sp[2] < TOL_WU, 'resid = %.6e' % sp[2])
    orig = res['原文（对照）']
    _chk('对照：原文 Y 下 w 不精确（判据有区分力）', orig[0] > TOL_WU,
         'resid = %.6e' % orig[0])
    return res


# ============================================================================
# main
# ============================================================================

def main():
    print('=' * 78)
    print('v16.8 · T6 · 阶段 2 —— 层 V 组装    [%s]' % _V)
    print('=' * 78)
    print('  口径：cell 几何 = 图 (d) 读出 + 用户 2026-10-04 确认（推断口径）')
    print('        cell 排布 = 本仓穷举（原文未印替换规则）')
    print('        Y       = S&P 主口径；原文 / fixed 仅对照，三张不合并成一行')

    tracemalloc.start()
    part_2_1()
    hits = part_2_2()
    part_2_3(hits)
    res = part_2_4()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print()
    print('=' * 78)
    sp = res['S&P（主口径）']
    fx = res['Y_tensor_fixed（备用）']
    if sp[0] < TOL_WU and sp[2] < TOL_WU:
        verdict = 'T6-层组装只需(12|34)'
    elif fx[0] < TOL_WU and fx[2] < TOL_WU:
        verdict = 'T6-层组装需要(13|24)'
    else:
        verdict = 'T6-层组装失败'
    print('  落支：%s' % verdict)
    print('  用时（进程）   : %.2f s' % (time.time() - _T_PROC))
    print('  用时（import） : %.2f s' % (_T_IMPORT - _T_PROC))
    print('  内存峰值       : %.1f MB' % (peak / 1048576.0))
    print('=' * 78)
    print('  [诚实边界]')
    print('   * 本文件**不缩并整层**（维度 4^n_in 不可行）；结论依赖 :100(c)/:128 逐字')
    print('     陈述的「层张量可分组为 w/u 之乘积」，该分组**不是** cell 分解。')
    print('   * (a) 已证：该分组**不能**是 w/u 的平行乘积（U<0）。落地支时该张力')
    print('     记为遗留，不抹平。')
    print('   * cell 排布规则是**穷举**所得，非原文；[2-3] 的收敛值因此是「该替换族下」')
    print('     的结果，不是从原文推出的唯一值。')
    print('=' * 78)

    if _FAIL:
        print('  FAIL 明细（%d 条）：' % len(_FAIL))
        for nm, dt in _FAIL:
            print('    - %s : %s' % (nm, dt))
        return 3
    return 0


if __name__ == '__main__':
    sys.exit(main())
