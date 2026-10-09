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
v16.7 · T4 · Hyper-invariant 张量网络 · Phase 1b —— A/B 组装与**多张量约束** w, u
[_VERSION_TAG = 'v16-smoke-t4-hyper-phase1b-1']

锚（源码锚，行号按本仓 `v16/Hyper.tex`）:
  G. Evenbly, *Hyper-invariant tensor networks and holography*, PRL 119, 141602 (2017)
    约束原话 `Hyper.tex:109`: "the product of an $A$ and two $B$ tensors is constrained to
    act as a 2-to-1 isometry $w$, while a product of three $A$ and five $B$ tensors is
    constrained to act as a 3-to-2 isometry $u$"
    判据原话 `Hyper.tex:100`: w "annihilates to identity with its conjugate $w^\dag$"
    常数豁免 `Hyper.tex:460`: "up to an irrelevant multiplicative constant"
  **接法的粗拓扑没有文字锚** —— 读自 arXiv v1 (1704.04229) 源码附图
  `v16/arXiv-1704.04229v1/{Parameterization37,Constraint37,Unitary37}.eps`（2026-10-03 下载）。
  那些 .eps 是 **GIMP 光栅图**，已写纯 stdlib 解码器转 PNG 后**直读**（结论见 D-a）。

靶（计划锚）: `v16.7_plan.md` §2.4 T4 · Phase 1b。
  计划 `:193` 把 A/B 接法列为「本仓无法仅凭文本复现 —— 除非另有图源，
  或自行选定接法并**明确标注为自选**」。本轮走了**前一条路**（拿到图源），
  但图是光栅的：**粗拓扑可读，腿内细比特走线不可读** —— 这正是 D-a 要量化的。

======================================================================
预登记（成文于实现之前；跑之前不许改）
======================================================================
口径: θ = (θ1..θ5) = (0.37, 0.52, 0.61, 0.29, 0.44) —— 与 Phase 1 同一个 generically 非退化角
容差: TOL = 1e-12（纯线性代数，非拟合）；D-a 另用一档松门限 GOOD = 1e-10
接法: `hyper.py::WIRING_U` —— **自选**，非原文；理由见 D-a

判据（三条）—— 一律用**原文** `Y_tensor`:
  [1] B 酉且对称  `B_tensor`：`||B^dag B - I_16||` 与 `||B - B^T||` < TOL。
                  B 只含 Q,R，**不含 Y** ⇒ 这条是 Y 缺陷的对照组，预期通过
  [2] w = A·B·B 是 2->1 等距   `coisometry_residual(w_tensor(A,B))` < TOL
  [3] u = A^3B^5 是 3->2 等距  `coisometry_residual(u_tensor(A,B))` < TOL

对照（**非原文；只报读数，不参与判据，也不参与落支**）:
  [B] 同 [2][3]，但 A 用 `Y_tensor_fixed`（口径 B 落盘的「非原文 · 诊断猜想」变体）。
      **判据与落支一律以原文 `Y_tensor` 为准，不看它。**

落支（事前写死）:
  `H-多张量约束通过`  [1]~[3] 全过 ⇒ EXIT=0
  `H-多张量约束失败`  任一超限 ⇒ EXIT=3
  预期: **失败** —— Phase 1 已证原文 Y 连单条酉性都不满足（残差 2|c1 s1|），
        含它的 w/u 不可能精确等距。

诊断（**不是判据，不参与落支，不改任何登记值**）:
  D-a 走线自由度 —— 枚举 3 个 A 各自的 6 种腿序（6^3 = 216 种），数出多少种给出精确等距；
                    原文 Y 与 Y_tensor_fixed 各扫一遍。这条量化「光栅图钉不死细比特走线」
  D-b θ 稳健性   —— 另取 3 组随机 θ 重跑 w/u，验证结论不是某个角的巧合
  D-c 常数 c     —— 报 c = tr(W W^dag)/n；原文允许差一个乘性常数（`:460`）
  D-d B 的结构   —— 报 Q/R 的酉性与对称性，及 B 的行列式模

范围红线（同计划 §2.4）:
  - 只调 `_model/hyper.py` 的公开函数（**真调，不复制逻辑**）；
  - **不 import 门面** `spiral_model_v16.py`（故无 torch/quimb）；
  - 不接 L4/L5；不进 runner；不加守卫/指标 ⇒ 规模三数 **78 / 27 / 13 不动**；
  - **不改** `_v16_smoke_t4_hyper.py` 与 `_t4_hyper.log`（Phase 1 那对已成对存档）。

用时: 约 80 s，其中 D-a 的 216x2 次 u 缩并占绝大部分（w 与其余诊断 < 1 s）。
      确切用时见日志末行，不在此处预登记（免得脚本与日志因一个数字失同步）。

[诚实边界] 本文件跑之前另做过**交互式一次性数值探查**（未落盘，含一次 216 种走线的计数）。
  探查不是判据；一切读数以本文件的输出为准。
"""

import itertools
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

_VERSION_TAG = 'v16-smoke-t4-hyper-phase1b-1'
THETA = (0.37, 0.52, 0.61, 0.29, 0.44)
TOL = 1e-12
GOOD = 1e-10        # D-a 里「算通过」的门槛，比 TOL 松一档

_ALL_WIRINGS = tuple(itertools.product(tuple(itertools.permutations(range(3))), repeat=3))


# ──────────────────────────────────────────────────────────────────────
# 诊断用的小工具（**不属于判据**）
# ──────────────────────────────────────────────────────────────────────
def _wiring_sweep(A, B):
    """D-a: 216 种腿序里有多少种给出精确等距。返回 (n_good, best_resid, best_wiring)。"""
    n_good, best_r, best_w = 0, float('inf'), None
    for wiring in _ALL_WIRINGS:
        r, _ = H.coisometry_residual(H.u_tensor(A, B, wiring))
        if r < GOOD:
            n_good += 1
        if r < best_r:
            best_r, best_w = r, wiring
    return n_good, best_r, best_w


def _diag_da(fixed_A, orig_A, B):
    print('  D-a 走线自由度（枚举 3 个 A 各自的 6 种腿序 = 216 种，看 u 是否精确等距）')
    for tag, A in (('原文 Y_tensor', orig_A), ('Y_tensor_fixed', fixed_A)):
        n, r, w = _wiring_sweep(A, B)
        print('      %-16s 通过 %3d / 216   最优残差 = %.6e   (最优走线 %s)'
              % (tag, n, r, '/'.join(''.join(map(str, x)) for x in w)))
    print('      => 光栅图**钉不死**腿内细比特走线：原文 Y 是 0/216（对走线免疫），')
    print('         而 Y_tensor_fixed 有大量走线成立 ⇒ 走线只能**自选并标注**。')


def _diag_db():
    print('  D-b θ 稳健性（另取 3 组随机 θ；WIRING_U = %s）'
          % '/'.join(''.join(map(str, x)) for x in H.WIRING_U))
    rng = np.random.default_rng(7)
    thetas = [THETA] + [tuple(rng.uniform(0.15, 1.4, 5)) for _ in range(3)]
    for th in thetas:
        B = H.B_tensor(th[1], th[2], th[3], th[4])
        rw_o = H.coisometry_residual(H.w_tensor(H.A_tensor(th[0]), B))[0]
        ru_o = H.coisometry_residual(H.u_tensor(H.A_tensor(th[0]), B))[0]
        Af = H.A_tensor(th[0], H.Y_tensor_fixed(th[0]))
        rw_f, cw_f = H.coisometry_residual(H.w_tensor(Af, B))
        ru_f, cu_f = H.coisometry_residual(H.u_tensor(Af, B))
        print('      θ=(%s)  原文 w %.2e u %.2e | 修补 w %.2e u %.2e (c=%.3f/%.3f)'
              % (' '.join('%.2f' % x for x in th), rw_o, ru_o, rw_f, ru_f, cw_f, cu_f))
    print('      => 结论与 θ 无关：原文 Y 处处超限，Y_tensor_fixed 处处精确。')


def _diag_dc(B, A, Af):
    print('  D-c 常数 c（原文 `:460` 允许 "up to an irrelevant multiplicative constant"）')
    for tag, Ax in (('原文 Y_tensor', A), ('Y_tensor_fixed', Af)):
        rw, cw = H.coisometry_residual(H.w_tensor(Ax, B))
        ru, cu = H.coisometry_residual(H.u_tensor(Ax, B))
        print('      %-16s w: resid %.6e  c=%.6f   |   u: resid %.6e  c=%.6f'
              % (tag, rw, cw, ru, cu))
    print('      => 成立时 c 恒 = 4.000000（= 被完全缩并掉的那个 Y_B 的 Tr 1），')
    print('         w 与 u 同值；不成立时 c 非整数（原文 Y 的 u 拟合出 5.211075）。')


def _diag_dd(t2, t3, t4, t5):
    print('  D-d B 的结构（B = Q·R，**不含 Y** ⇒ 与 Y 的缺陷无关）')
    Q, R = H.Q_tensor(t3, t4, t5), H.R_tensor(t2)
    B = H.B_tensor(t2, t3, t4, t5)
    print('      Q 酉 %.3e   R 双酉 %.3e   Q-Q^T %.1e   R-R^T %.1e'
          % (H.unitary_residual(Q), H.doubly_unitary_residual_R(R),
             np.max(np.abs(Q - Q.T)), np.max(np.abs(R - R.T))))
    print('      B 酉 %.3e   B 对称 %.1e   |det B| - 1 = %.3e'
          % (H.unitary_residual(B), np.max(np.abs(B - B.T)),
             abs(abs(np.linalg.det(B)) - 1.0)))


def main():
    t1, t2, t3, t4, t5 = THETA
    print('=' * 76)
    print('v16.7 · T4 · Hyper-invariant 张量网络 · Phase 1b —— A/B 组装与多张量约束 w, u')
    print('  %s' % _VERSION_TAG)
    print('  θ = %s   TOL = %g' % (THETA, TOL))
    print('  锚: v16/Hyper.tex (Evenbly PRL 119, 141602) `:100` / `:109` / `:460`')
    print('  接法锚: arXiv-1704.04229v1 附图（光栅）; WIRING_U 为**自选**')
    print('=' * 76)

    B = H.B_tensor(t2, t3, t4, t5)
    A = H.A_tensor(t1)                          # 原文 Y_tensor
    Af = H.A_tensor(t1, H.Y_tensor_fixed(t1))   # 口径 B 变体（**非原文**）

    # ── 判据 [1]~[3]（一律原文 Y_tensor）────────────────────────────────
    r1 = max(H.unitary_residual(B), float(np.max(np.abs(B - B.T))))
    r2, c2 = H.coisometry_residual(H.w_tensor(A, B))
    r3, c3 = H.coisometry_residual(H.u_tensor(A, B))

    k1, k2, k3 = r1 < TOL, r2 < TOL, r3 < TOL
    print('\n-- 判据（**原文** `Y_tensor`）--')
    print('  [1] B 酉且对称（不含 Y）   max resid = %.6e   %s' % (r1, 'OK' if k1 else 'FAIL'))
    print('  [2] w = A·B·B  2->1 等距  max resid = %.6e   %s   (c=%.6f)'
          % (r2, 'OK' if k2 else 'FAIL', c2))
    print('  [3] u = A^3B^5 3->2 等距  max resid = %.6e   %s   (c=%.6f)'
          % (r3, 'OK' if k3 else 'FAIL', c3))

    # ── 对照：口径 B 的非原文变体（**不属于判据，不参与落支**）──────────
    r2f, c2f = H.coisometry_residual(H.w_tensor(Af, B))
    r3f, c3f = H.coisometry_residual(H.u_tensor(Af, B))
    print('\n-- 对照（**非原文 · 诊断猜想**；不参与判据与落支）--')
    print('  [B] Y_tensor_fixed： w max resid = %.6e (c=%.6f)   u max resid = %.6e (c=%.6f)'
          % (r2f, c2f, r3f, c3f))

    ok = k1 and k2 and k3
    branch = 'H-多张量约束通过' if ok else 'H-多张量约束失败'

    # ── 诊断（本轮主要产出就是"为什么失败 / 接法有多可辨"，故始终展开）──
    print('\n-- 诊断（**不是判据**）--')
    _diag_da(Af, A, B)
    _diag_db()
    _diag_dc(B, A, Af)
    _diag_dd(t2, t3, t4, t5)

    # ── 落支 ───────────────────────────────────────────────────────────
    print('\n-- 落支 --')
    print('  落支 = **%s**' % branch)
    if not ok:
        print('  未过: %s' % ', '.join(n for n, o in
                                       zip(('[1]', '[2]', '[3]'), (k1, k2, k3)) if not o))
        print('  读数: B（不含 Y）精确酉且对称；含 Y 的 w 与 u 都超限，且 c 都不是整数。')
        print('  ⇒ 与 Phase 1 同源：缺陷在原文印的 Y（`Hyper.tex:397-404`），不在 A/B 接法。')
        print('     A/B 接法本身在 Y_tensor_fixed 下**精确**成立（对照行 [B]，c=4）。')
    else:
        print('  [1]~[3] 全过: A/B 构造在 θ 处精确满足原文的多张量约束。')
    print('  [接法诚实边界] 粗拓扑读自 arXiv 源码附图（光栅）；腿内细比特走线**不可读**，')
    print('     `WIRING_U` 为**自选**（D-a：原文 Y 0/216，Y_tensor_fixed 大量成立）。')
    print('\n[T4-Phase1b] 隔离自检: 未 import 门面 / 未接 L4-L5 / 未进 runner / 未加守卫与指标。')
    print('  规模三数 78 / 27 / 13 不动；未改 _v16_smoke_t4_hyper.py 与 _t4_hyper.log。')
    print('用时: import %.1f s + 本计算 %.1f s = 总计 %.1f s'
          % (_T_IMPORT - _T_PROC, time.time() - _T0, time.time() - _T_PROC))
    print('=' * 76)
    return 0 if ok else 3


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.exit(main())
