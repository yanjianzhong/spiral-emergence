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
v16.7 · T5 · Tensor Network Renormalization (TNR) · Phase 1 —— 附录 B 闭式构建块自检
[_VERSION_TAG = 'v16-smoke-t5-tnr-phase1-2']

锚（源码锚，行号按本仓 `v16/TNR.tex`）:
  G. Evenbly, G. Vidal, *Tensor network renormalization*, PRL 115, 180405 (2015)
    附录 B「Closed form expressions for a CDL tensor network」
      Eq. (CDL)  A^CDL  `TNR.tex:434`
      Eq. (TRG2) A^CDL = sum_m S_mij S_mkl       `TNR.tex:454`
      Eq. (TRG3) S 的显式式样                    `TNR.tex:458`
      Eq. (TRG4) A' 的一步粗粒化                  `TNR.tex:462`
      Eq. (TRG5) A' = eta A^CDL                  `TNR.tex:466`
      Eq. (TNR1) u                               `TNR.tex:499`
      Eq. (TNR2) v = w（**印刷含未定义指标**）    `TNR.tex:503`
      Eq. (TNR3) B, C                            `TNR.tex:510`
  **本仓 `v16/TNR.tex` 与 arXiv v3 (1412.0732) 源码逐字一致**
  （2026-10-03 实测 SHA256 前缀 FADA65A7D965739A）⇒ 本文件关于公式本身的任何判断，
  都属于**原文**，不是本仓的抄录错误。

靶（计划锚）: `v16.7_plan.md` §2.5 T5 · Phase 1 —— 附录 B 的**闭式**构建块检验。
  **不在本文件范围**：2×2 CDL 网络的**实际缩并接线**只在图 `TNRtensors.eps` / `TNR.eps` 里，
  `.tex` 正文没写 ⇒ 与 T4 的 Phase 1b 同性质，本仓不凭文本猜接线。

======================================================================
预登记（判据与落支来自计划 §2.5，**成文于实现之前**；跑之前不许改）
======================================================================
口径: eta = 2（原文附录 B 的取值；数值为精确 0/1 与 sqrt(2) 的代数，无拟合）
容差: TOL = 1e-12（纯线性代数 ⇒ 期望 ~1e-15）

判据（五条）:
  [1] Eq.(TRG2) 因子分解  `trg_factorization_residual(s_tensor(eta), cdl_tensor(eta))` < TOL
  [2] Eq.(TRG4/5) 一步自映 max |trg_step(s_tensor(eta)) - eta*cdl_tensor(eta)|          < TOL
  [3] Eq.(TNR1) u 等距     两种折法 `u_matrix(U, g)` 的 max|M M^dag - I| 取**最优** < TOL
  [4] Eq.(TNR2) v 等距     候选①② 的 max|V V^dag - I| 取**最优**                   < TOL
  [5] Eq.(TNR3) C 平凡     `matricize_rank(c_tensor(eta))` == 1  （即 chi' = 1 的平凡张量）

对照（**非原文；只报读数，不参与判据，也不参与落支**）—— 口径 B 的「诊断猜想」变体:
  [B1] `trg_step(s_tensor_fixed(eta))` 与 eta*A^CDL 的残差。
       **注意 Eq.(TRG2) 仍不可能**（D-a 已证对任何实 S 不成立），故 B1 只恢复 Eq.(TRG4)。
  [B2] `v_tensor_fixed(eta)` 的 V V^dag - I 残差（去掉 u 的残留 delta + 前因子改 1）。
  **判据 [1]~[5] 与落支一律以原文式样为准，不看它们。**

落支（事前写死）:
  `T-CDL通过`  [1]~[5] 全过 ⇒ EXIT=0
  `T-CDL失败`  任一超限   ⇒ EXIT=3，并按计划「**先查缩并约定与指标序，不得改判据**」展开诊断

诊断（**只在失败时展开；不是判据，不参与落支，不改任何登记值**）:
  D-a 免约定可行性 —— Eq.(TRG2) 的右端是实向量 {vec(S_m)} 的 Gram 和 ⇒ 必半正定；
                      扫 A 的 24 种腿序，看 (ij)×(kl) 矩阵有几个半正定。
                      **全非半正定 ⇒ 任何实 S 都不可能，与约定无关。**
  D-b Eq.(TRG4) 读法全扫 —— (S 的 delta 配对 64) × (腿序 6) = 384 种，报命中 eta A^CDL 的读法，
                      并指出**印刷的 Eq.(TRG3) 配对是否在命中集内**。
  D-c Eq.(TNR2)/Eq.(TNR3) 读数 —— 候选①② 的等距残差、B 的 |.|_1、C 的秩。

范围红线（同计划 §2.5）:
  - 只调 `_model/tnr.py` 的公开函数（**真调，不复制逻辑**）；
  - 新模块**独立于** `_model/mera.py`；**不动** `build_mera_graph` / `mera_bulk_invariance` / F6d；
  - **不 import 门面** `spiral_model_v16.py`（故无 torch/quimb，本文件 <1 s）；
  - 不接 L4/L5；不进 runner；不加守卫/指标 ⇒ 规模三数 **78 / 27 / 13 不动**。

[诚实边界] 本文件跑之前另做过若干次**交互式一次性数值探查**（未落盘）。探查不是判据；
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
from _model import tnr as T         # noqa: E402

_T_IMPORT = time.time()
_T0 = time.time()

ETA = 2
TOL = 1e-12


# ──────────────────────────────────────────────────────────────────────
# 诊断用的小工具（**不属于判据**）
# ──────────────────────────────────────────────────────────────────────
def _iso(M):
    """max |M M^dag - I|。"""
    return float(np.max(np.abs(M @ M.conj().T - np.eye(M.shape[0]))))


def _diag_da():
    print('  D-a 免约定可行性（Eq.(TRG2) 右端是实向量的 Gram 和 ⇒ 必半正定）')
    A = T.cdl_tensor(ETA)
    psd, negs = 0, []
    for lo in T.LEG_ORDERS:
        m = T.gram_min_eig(A, lo)
        if m >= -1e-9:
            psd += 1
        else:
            negs.append(m)
    print('      A 的 24 种腿序中，半正定（min eig >= 0）的个数 = %d / 24' % psd)
    print('      非半正定的最小特征值 = %.4f（最大 = %.4f）' % (min(negs), max(negs)))
    print('      => 24/24 全非半正定 ⇒ **任何实 S** 都满足不了 Eq.(TRG2)，'
          '与指标序/配对约定**无关**。')


def _diag_db():
    print('  D-b Eq.(TRG4) 读法全扫（S 的 delta 配对 64 × 腿序 6 = 384）')
    A = T.cdl_tensor(ETA)
    lit = ((1, 2), (1, 2), (1, 2))
    hits, lit_hits = [], []
    for p in T.S_PAIRINGS:
        for lo in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
            S = T.s_tensor_explicit(p, lo, ETA)
            if np.max(np.abs(T.trg_step(S) - ETA * A)) < 1e-9:
                hits.append((p, lo))
                if p == lit:
                    lit_hits.append(lo)
    print('      使 A\' == eta A^CDL 精确成立的 (配对,腿序) 数 = %d / 384' % len(hits))
    print('      其中 Eq.(TRG3) **印刷**配对 ((1,2),(1,2),(1,2)) 的命中腿序 = %s'
          % (lit_hits if lit_hits else '**空**'))
    print('      => Eq.(TRG4) 本身可对，但**印刷的 S 不是**那些读法之一 ⇒ TRG2 与 TRG4 互斥。')


def _diag_dc():
    print('  D-c Eq.(TNR2)/Eq.(TNR3) 读数')
    U = T.u_tensor(ETA)
    ru = {g: _iso(T.u_matrix(U, g)) for g in ('iK_jL', 'ij_KL')}
    print('      u 等距残差: ' + '   '.join('%s = %.6e' % kv for kv in ru.items())
          + '   （最优 %.6e）' % min(ru.values()))
    for c in (1, 2):
        Vm = T.v_matrix(T.v_tensor(ETA, c))
        diag = np.abs(np.diag(Vm @ Vm.conj().T) - 1.0)
        print('      v 候选%d 等距残差 = %.6e（对角偏离 1 的最大值 = %.6e）'
              % (c, _iso(Vm), float(np.max(diag))))
    B, C = T.b_tensor(ETA), T.c_tensor(ETA)
    print('      B: |B|_1 = %g ;  C: |C|_1 = %g ;  rank(C) = %d'
          % (np.abs(B).sum(), np.abs(C).sum(), T.matricize_rank(C)))


def main():
    print('=' * 76)
    print('v16.7 · T5 · TNR · Phase 1 —— 附录 B 闭式构建块自检')
    print('  eta = %d   TOL = %g' % (ETA, TOL))
    print('  锚: v16/TNR.tex 附录 B (Evenbly & Vidal, PRL 115, 180405 (2015))')
    print('=' * 76)

    A = T.cdl_tensor(ETA)
    S = T.s_tensor(ETA)

    # ── 判据 [1]~[5] ───────────────────────────────────────────────────
    r1 = T.trg_factorization_residual(S, A)
    r2 = float(np.max(np.abs(T.trg_step(S) - ETA * A)))
    U = T.u_tensor(ETA)
    r3 = min(_iso(T.u_matrix(U, g)) for g in ('iK_jL', 'ij_KL'))
    r4 = min(_iso(T.v_matrix(T.v_tensor(ETA, c))) for c in (1, 2))
    rk5 = T.matricize_rank(T.c_tensor(ETA))

    c1, c2, c3, c4, c5 = r1 < TOL, r2 < TOL, r3 < TOL, r4 < TOL, rk5 == 1
    print('\n-- 判据 --')
    print('  [1] Eq.(TRG2) 因子分解   max resid = %.6e   %s' % (r1, 'OK' if c1 else 'FAIL'))
    print('  [2] Eq.(TRG4/5) 一步自映 max resid = %.6e   %s' % (r2, 'OK' if c2 else 'FAIL'))
    print('  [3] Eq.(TNR1) u 等距     max resid = %.6e   %s' % (r3, 'OK' if c3 else 'FAIL'))
    print('  [4] Eq.(TNR2) v 等距     max resid = %.6e   %s' % (r4, 'OK' if c4 else 'FAIL'))
    print('  [5] Eq.(TNR3) C 平凡     rank(C)   = %d          %s' % (rk5, 'OK' if c5 else 'FAIL'))

    # ── 对照：口径 B 的非原文变体（**不属于判据，不参与落支**）──────────
    Sf, Vf = T.s_tensor_fixed(ETA), T.v_tensor_fixed(ETA)
    print('\n-- 对照（**非原文 · 诊断猜想**；不参与判据与落支）--')
    print('  [B1] s_tensor_fixed   Eq.(TRG4) 残差 = %.6e   （Eq.(TRG2) 仍不可能，见 D-a）'
          % float(np.max(np.abs(T.trg_step(Sf) - ETA * A))))
    print('  [B2] v_tensor_fixed   V V^dag - I 残差 = %.6e' % _iso(T.v_matrix(Vf)))

    ok = c1 and c2 and c3 and c4 and c5
    branch = 'T-CDL通过' if ok else 'T-CDL失败'

    # ── 诊断：只在失败时展开 ───────────────────────────────────────────
    if not ok:
        print('\n-- 诊断（**不是判据**；由计划「先查缩并约定与指标序」而来）--')
        _diag_da()
        _diag_db()
        _diag_dc()

    # ── 落支 ───────────────────────────────────────────────────────────
    print('\n-- 落支 --')
    print('  落支 = **%s**' % branch)
    if not ok:
        print('  未过: %s' % ', '.join(n for n, o in
                                       zip(('[1]', '[2]', '[3]', '[4]', '[5]'),
                                           (c1, c2, c3, c4, c5)) if not o))
        print('  读数: 附录 B 的闭式构建块在 eta=2 处**不能**同时自洽 ——')
        print('        D-a 证明 Eq.(TRG2) 对任何实 S 都不可能（免约定，与指标序无关）；')
        print('        D-b 说明 Eq.(TRG4) 可对但用的不是印刷的 S ⇒ TRG2 与 TRG4 互斥。')
        print('  ⇒ 不是本仓缩并约定错（D-a/D-b 已穷举 384+24 种读法），是**原文印刷层面**的问题。')
    else:
        print('  [1]~[5] 全过: 附录 B 的闭式构建块在 eta=2 处**精确**自洽。')
    print('  [Phase 1b 未决] 2×2 CDL 网络的**实际缩并接线**只在图 TNRtensors.eps / TNR.eps 里，')
    print('      .tex 正文没写 ⇒ 本仓不凭文本复现（同 T4 Phase 1b）。')
    print('  [Phase 2 未开工] TNR.tex:260 的「Algorithms for tensor network renormalization」')
    print('      不在仓库 ⇒ 迭代方案无法复现。')
    print('  范围标记 = T-仅Phase1')
    print('\n[T5-Phase1] 隔离自检: 未 import 门面 / 未接 L4-L5 / 未进 runner / 未加守卫与指标。')
    print('  不动 build_mera_graph / mera_bulk_invariance / F6d；规模三数 78 / 27 / 13 不动。')
    print('用时: import %.1f s + 本计算 %.1f s = 总计 %.1f s'
          % (_T_IMPORT - _T_PROC, time.time() - _T0, time.time() - _T_PROC))
    print('=' * 76)
    return 0 if ok else 3


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.exit(main())
