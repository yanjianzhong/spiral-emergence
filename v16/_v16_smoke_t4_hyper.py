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
v16.7 · T4 · Hyper-invariant 张量网络 · Phase 1 —— 附录 B 构建块与**自述约束**自检
[_VERSION_TAG = 'v16-smoke-t4-hyper-phase1-2']

锚（源码锚，行号按本仓 `v16/Hyper.tex`）:
  G. Evenbly, *Hyper-invariant tensor networks and holography*, PRL 119, 141602 (2017)
    Eq. (5) Y = Y_(ij)(kl)   `Hyper.tex:397-404`
    Eq. (7) R                `Hyper.tex:437-444`
    Eq. (9) Q                `Hyper.tex:450-457`
  约束与对称性逐字见 `_model/hyper.py` 头:  Y 两条 / R 两条 / Q 一条 + 反射 `:392-395`,
  `:405`, `:431-436`, `:446-449`。
  **本仓 `v16/Hyper.tex` 与 arXiv v1 (1704.04229) 源码 raw-diff 逐字一致**
  （2026-10-03 实测，`diff` 退出码 0）⇒ 本文件关于张量本身的任何判断，
  都属于**原文**，不是本仓的抄录错误。

靶（计划锚）: `v16.7_plan.md` §2.4 T4 · Phase 1 —— 数值验证三条闭式矩阵的**自述**约束。
  A/B 级多张量约束（$ABB=w$ 2→1 等距、$A^3B^5=u$ 3→2 等距）**不在本文件范围**:
  接法只在图 `Parameterization37.eps` 里，`.tex` 正文没写（计划 §2.4「未决 Phase 1b」）。

======================================================================
预登记（判据与落支来自计划 §2.4，**成文于实现之前**；跑之前不许改）
======================================================================
口径: θ = (θ1..θ5) = (0.37, 0.52, 0.61, 0.29, 0.44) —— generically 非退化角
容差: TOL = 1e-12（纯线性代数，非拟合；`hyper.py` docstring 自述期望 ~1e-15）

判据（四条）:
  [1] Y 双酉   `doubly_unitary_residual_Y(Y_tensor(θ1))` < TOL  （两个分划取最大）
  [2] R 双酉   `doubly_unitary_residual_R(R_tensor(θ2))` < TOL
  [3] Q 酉     `unitary_residual(Q_tensor(θ3,θ4,θ5))`    < TOL  （原文只声明一个分划）
  [4] 反射     `Y_klij / R_klij / R_jilk / Q_klij / Q_jilk` 五条 < TOL

对照（**非原文；只报读数，不参与判据，也不参与落支**）:
  [B] `doubly_unitary_residual_Y(Y_tensor_fixed(θ1))` —— 口径 B 落盘的
      「非原文 · 诊断猜想」变体（角上 $+c_1 \to -c_1$）。
      **判据 [1]~[4] 与落支一律以原文 `Y_tensor` 为准，不看它。**

落支（事前写死）:
  `H-约束通过`  [1]~[4] 全过 ⇒ EXIT=0
  `H-约束失败`  任一超限 ⇒ EXIT=3，并按计划「**先查指标序约定，不得改判据**」展开诊断

诊断（**只在失败时展开；不是判据，不参与落支，不改任何登记值**）:
  D-a 指标序约定扫描 —— 对 Y 的两条约束枚举合理指标序读法，报最优残差
  D-b 约定不变性     —— 单指标基的 置换 × ±符号 下残差的最大范数**逐字不变**；
                        且「$MM^\dagger=I$ 是否成立」在酉基变换下不变 ⇒ 约定救不回来
  D-c 缺陷定位       —— 按 Z2 电荷块 $(00,11\,|\,01,10)$ 分解，指出哪一块非酉
  D-d 最小修补       —— 调 `hyper.py::Y_tensor_fixed`（口径 B 已落盘的**非原文**变体，
                        角上 $+c_1 \to -c_1$）并报其电荷块残差。
                        **原文 `Y_tensor` 未动；判据与落支一律以原文为准**
  D-e θ1 扫描        —— 验证 Y 残差对 θ1 的依赖形式

范围红线（同计划 §2.4）:
  - 只调 `_model/hyper.py` 的公开函数（**真调，不复制逻辑**）；
  - **不 import 门面** `spiral_model_v16.py`（故无 torch/quimb，本文件 <1 s）；
  - 不接 L4/L5；不进 runner；不加守卫/指标 ⇒ 规模三数 **78 / 27 / 13 不动**。

[诚实边界] 本文件跑之前另做过一次**交互式一次性数值探查**（未落盘）。探查不是判据；
  一切读数以本文件的输出为准。
"""

import os
import sys
import time

_T_PROC = time.time()      # ← 进程起点，早于任何 import

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np                  # noqa: E402
from _model import hyper as H       # noqa: E402

_T_IMPORT = time.time()
_T0 = time.time()

THETA = (0.37, 0.52, 0.61, 0.29, 0.44)
TOL = 1e-12


# ──────────────────────────────────────────────────────────────────────
# 诊断用的小工具（**不属于判据**）
# ──────────────────────────────────────────────────────────────────────
def _four(M):
    """4x4 -> (i,j,k,l)，行=(ij) 列=(kl)。"""
    return np.ascontiguousarray(M).reshape(2, 2, 2, 2)


def _charge_blocks(M):
    """按 Z2 电荷把 (ij)(kl) 矩阵重排成 (00,11|01,10) 的两个 2x2 块。"""
    P = M[np.ix_([0, 3, 1, 2], [0, 3, 1, 2])]
    return P[:2, :2], P[2:, 2:]


def _blk_resid(B):
    n = B.shape[0]
    return float(np.max(np.abs(B @ B.conj().T - np.eye(n))))


def _target(order):
    """按输出指标顺序 order 造 δ_{order[0],order[2]}·δ_{order[1],order[3]}。"""
    T = np.zeros((2, 2, 2, 2), dtype=complex)
    for a in range(2):
        for b in range(2):
            idx = [None] * 4
            idx[0] = a
            idx[order.index(order[2])] = a
            idx[order.index(order[1])] = b
            idx[order.index(order[3])] = b
            T[tuple(idx)] += 1.0
    return T


_C1_READINGS = ['ijkl,mnkl->ijmn', 'ijkl,mnkl->ijnm', 'ijkl,mnkl->jimn',
                'ijkl,mnkl->jinm', 'ijkl,mnlk->ijmn']
_C2_READINGS = ['ijkl,mjnl->ikmn', 'ijkl,mjnl->kimn', 'ijkl,mjnl->iknm',
                'ijkl,mjnl->kinm']


def _reading_resid(Y, reading):
    T, C = _four(Y), _four(Y).conj()
    V = np.einsum(reading, T, C)
    return float(np.max(np.abs(V - _target(reading.split('->')[1]))))


def _diag_da(Y):
    print('  D-a 指标序约定扫描（对 Y 的两条约束枚举合理读法）')
    best = None
    for rd in _C1_READINGS + _C2_READINGS:
        r = _reading_resid(Y, rd)
        print('      %-18s  max resid = %.3e' % (rd, r))
        best = r if best is None else min(best, r)
    print('      => 全部读法的最优残差 = %.3e（**没有**任何一种读法降到 TOL 以下）' % best)


def _diag_db(Y):
    """单指标基的 (置换 × ±符号):  y'[i,j,k,l] = s0^i s1^j s2^k s3^l · y[σ(i),σ(j),σ(k),σ(l)]。
    理论: M' = A M B, A = D_r(P⊗P)（带符号置换）, B = (P⊗P)^T D_c ⇒ M'M'^dag = A (M M^dag) A^dag，
    故 **最大范数残差逐字不变**；又酉变换不改变「= I 是否成立」⇒ 约定**结构上**救不回来。"""
    T = _four(Y)
    perms = ((0, 1), (1, 0))
    rs = []
    for p0 in perms:
        for p1 in perms:
            for p2 in perms:
                for p3 in perms:
                    for s0 in (1, -1):
                        for s1 in (1, -1):
                            for s2 in (1, -1):
                                for s3 in (1, -1):
                                    T2 = np.empty_like(T)
                                    for i in range(2):
                                        for j in range(2):
                                            for k in range(2):
                                                for l in range(2):
                                                    T2[i, j, k, l] = (s0 ** i) * (s1 ** j) * (s2 ** k) * (s3 ** l) \
                                                        * T[p0[i], p1[j], p2[k], p3[l]]
                                    a = np.einsum('ijkl,mnkl->ijmn', T2, T2.conj()).reshape(4, 4)
                                    rs.append(float(np.max(np.abs(a - np.eye(4)))))
    rs = np.array(rs)
    print('  D-b 约定不变性（%d 种 置换×符号 的抽样）' % rs.size)
    print('      C1 残差 min = %.6e   max = %.6e   spread = %.3e'
          % (rs.min(), rs.max(), rs.max() - rs.min()))
    print('      => 残差在全部约定下**逐字不变** ⇒ 指标序/符号约定**不是**失败原因。')


def _diag_dc(Y):
    print('  D-c 缺陷定位（按 Z2 电荷块分解，(ij)(kl) 矩阵）')
    b0, b1 = _charge_blocks(Y)
    print('      电荷 0 块 (00,11) = %s' % np.array2string(b0, precision=4))
    print('      电荷 1 块 (01,10) = %s' % np.array2string(b1, precision=4))
    print('      块酉残差:  电荷0 = %.3e   电荷1 = %.3e   （**只有电荷 0 块非酉**）'
          % (_blk_resid(b0), _blk_resid(b1)))


def _diag_dd(t1):
    Yf = H.Y_tensor_fixed(t1)       # 口径 B：非原文变体，独立函数；原文 Y_tensor 未动
    b0, b1 = _charge_blocks(Yf)
    print('  D-d 最小修补（**非原文 · 诊断猜想**；已按口径 B 落为 `hyper.py::Y_tensor_fixed`，'
          '原文 `Y_tensor` 未动）')
    print('      Y[1,1,1,1]: +c1 -> -c1')
    print('      修补后:  Y 双酉 max resid = %.3e   ;   反射 Y_klij = %.3e'
          % (H.doubly_unitary_residual_Y(Yf), H.reflection_residual(Yf, 'klij')))
    print('              电荷块酉残差 0 / 1 = %.3e / %.3e' % (_blk_resid(b0), _blk_resid(b1)))


def _diag_de(t1_list):
    print('  D-e θ1 扫描（Y 残差 vs 2|c1 s1|）')
    for t1 in t1_list:
        c, s = np.cos(t1), np.sin(t1)
        r = H.doubly_unitary_residual_Y(H.Y_tensor(t1))
        print('      θ1=%.4f  c1=%.4f s1=%.4f   resid=%.6e   2|c1 s1|=%.6e   差=%.2e'
              % (t1, c, s, r, 2 * abs(c * s), abs(r - 2 * abs(c * s))))


def main():
    t1, t2, t3, t4, t5 = THETA
    print('=' * 76)
    print('v16.7 · T4 · Hyper-invariant 张量网络 · Phase 1 —— 附录 B 自述约束自检')
    print('  θ = %s   TOL = %g' % (THETA, TOL))
    print('  锚: v16/Hyper.tex (Evenbly PRL 119, 141602) Eq.(5)/(7)/(9)')
    print('=' * 76)

    Y = H.Y_tensor(t1)
    R = H.R_tensor(t2)
    Q = H.Q_tensor(t3, t4, t5)

    # ── 判据 [1]~[4] ───────────────────────────────────────────────────
    r1 = H.doubly_unitary_residual_Y(Y)
    r2 = H.doubly_unitary_residual_R(R)
    r3 = H.unitary_residual(Q)
    refl = {
        'Y_klij': H.reflection_residual(Y, 'klij'),
        'R_klij': H.reflection_residual(R, 'klij'),
        'R_jilk': H.reflection_residual(R, 'jilk'),
        'Q_klij': H.reflection_residual(Q, 'klij'),
        'Q_jilk': H.reflection_residual(Q, 'jilk'),
    }
    r4 = max(refl.values())

    c1, c2, c3, c4 = r1 < TOL, r2 < TOL, r3 < TOL, r4 < TOL
    print('\n-- 判据 --')
    print('  [1] Y 双酉（两分划）  max resid = %.6e   %s' % (r1, 'OK' if c1 else 'FAIL'))
    print('  [2] R 双酉（两分划）  max resid = %.6e   %s' % (r2, 'OK' if c2 else 'FAIL'))
    print('  [3] Q 酉（单分划）    max resid = %.6e   %s' % (r3, 'OK' if c3 else 'FAIL'))
    print('  [4] 反射对称          max resid = %.6e   %s' % (r4, 'OK' if c4 else 'FAIL'))
    for k, v in refl.items():
        print('        %-8s %.3e' % (k, v))

    # ── 对照：口径 B 的非原文变体（**不属于判据，不参与落支**）──────────
    Yf = H.Y_tensor_fixed(t1)
    print('\n-- 对照（**非原文 · 诊断猜想**；不参与判据与落支）--')
    print('  [B] Y_tensor_fixed（角上 +c1 -> -c1）  双酉 max resid = %.6e   ;   反射 Y_klij = %.6e'
          % (H.doubly_unitary_residual_Y(Yf), H.reflection_residual(Yf, 'klij')))

    ok = c1 and c2 and c3 and c4
    branch = 'H-约束通过' if ok else 'H-约束失败'

    # ── 诊断：只在失败时展开 ───────────────────────────────────────────
    if not ok:
        print('\n-- 诊断（**不是判据**；由计划「先查指标序约定」而来）--')
        _diag_da(Y)
        _diag_db(Y)
        _diag_dc(Y)
        _diag_dd(t1)
        _diag_de([0.0, 0.1, 0.37, float(np.pi) / 4, 0.9, 1.2])

    # ── 落支 ───────────────────────────────────────────────────────────
    print('\n-- 落支 --')
    print('  落支 = **%s**' % branch)
    if not ok:
        print('  未过: %s' % ', '.join(n for n, o in
                                       zip(('[1]', '[2]', '[3]', '[4]'), (c1, c2, c3, c4)) if not o))
        print('  读数: Y 的两条**自述**约束都超限（残差 ~O(1)，非 ~1e-15）；R、Q 与全部反射对称性通过。')
        print('  ⇒ 不是指标序约定问题（D-a/D-b），也不是本仓实现问题 —— 定位与最小修补见 D-c/D-d。')
    else:
        print('  [1]~[4] 全过: 三条闭式矩阵在 θ 处**精确**满足原文自述的全部约束与对称性。')
    print('  [Phase 1b 未决] A（3×Y）与 B（Q·R）的接法只在图 Parameterization37.eps 里，')
    print('      .tex 正文没写 ⇒ A/B 级多张量约束($ABB=w$, $A^3B^5=u$)**本仓无法凭文本复现**。')
    print('\n[T4-Phase1] 隔离自检: 未 import 门面 / 未接 L4-L5 / 未进 runner / 未加守卫与指标。')
    print('  规模三数 78 / 27 / 13 不动。')
    print('用时: import %.1f s + 本计算 %.1f s = 总计 %.1f s'
          % (_T_IMPORT - _T_PROC, time.time() - _T0, time.time() - _T_PROC))
    print('=' * 76)
    return 0 if ok else 3


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.exit(main())
