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
"""
v16.4 · 工作包 γ · D10 —— L1 的**下游读出**（隔离通路）
[_VERSION_TAG = 'v16-smoke-gamma-l1-downstream-1']

**为什么有这个脚本**（锚点四处, 逐字）:
  a. 现有推离读数是 |<R_y(theta)·psi^{tensor L} | gs_ED>|^2 —— 取 equal_state ->
     np.kron L 次 -> 与 ED 基态内积 (`_model/core.py:271-283`)。
  b. 它与 `derive_L1_void_to_chain` **共用同一构造** (`core.py:206-207` 的
     `np.kron` 递推), 差别只是局部因子转了一下 -> 审计判「同一条路径的自洽」。
  c. **下游根本不存在**: `void_state` 进 `stage2_critical_break` **只做维数断言**,
     物理读数一字不动 (`_model/stage12.py:87-90`; `:101-102` 逐字
     「维数相符本身不构成物理因果」)。
  d. L1 唯一负对照自述「此前 L1 **没有到下游的因果通路**」(`_metric/extras.py:584-588`)。
=> ① 要的不是「另找一个态来比内积」, 是**让 L1 的态物理地进入下游、由下游读数回答**。

======================================================================
**预登记（在跑任何数据之前写死；落支词条照此判定，不许事后改）**
======================================================================

* **DP-6(a) 按"构造"读**（计划默认）：① 要换的是**共用的构造**，不是函数调用。
* **DP-7 = 用户 2026-09-27 明确指示「请开始D10」** ⇒ 把条件 ① 拉回 v16.4；
  `spiral_v16_audit.md:348` 的「两条都不在 v16 范围内」须**并存订正**（标注原文、不删）。
* **DP-8 = 诊断**：本脚本**不进 runner、不加守卫/指标** ⇒ 规模三数 78 / 27 / 13 不动。
* **DP-9 = 读 `S(L/2)`**（唯一参与判据的下游量）。`E0/L` / `polarization` / `gap`
  只作**同页旁证**打印，**不参与落支判定**。
* **DP-10**：若落 `N-不敏感` ⇒ **维持 B**，不动任何既有读数。

**D10-a 通路（冻结）**
  1. L1 的局部态 `eq = stage1_void(2)['equal_state']`（d=2, 归一）。
  2. **不共用构造**：**不调用 `np.kron`**。改为把 `eq` 沿**每一根轴**广播后**逐点相乘**
     （L 个单轴 rank-1 因子的 Hadamard 积），展平得 `psi0`。通路里无 `kron` 调用。
  3. **阶段二算子**：`U(tau) = exp(-tau*H)`，`H = tfi_periodic_sparse(L, J, h)`，
     用 `scipy.sparse.linalg.expm_multiply` 作用（**不**碰 `stage2_critical_break` 的调用）。
  4. **tau 先钉死**：主点 `TAU_STAR = 1/|E0/L|`（无量纲, 无自由拟合）。
     旁证网格 `TAU_GRID = TAU_STAR * (0.25,0.5,1,2,4)` **只画曲线、不参与判定**。

**D10-b 证伪（冻结）**
  错态 `|->^{tensor L}`（局部 `[1,-1]/sqrt(2)`，先例 `_v16_smoke_cmera5.py:120`），
  同一条通路同样演化。
  **判定量**：`rel = |S_correct - S_wrong| / max(S_correct, S_wrong)`，在 `tau = TAU_STAR` 单点。

**落支（冻结）**
  * `N-下游`      : `rel > SENS_TOL` (1e-3)。
  * `N-不敏感`    : `rel <= SENS_TOL`。
  * `N-共用构造`  : D10-c 复核发现**通路里必须调用 `np.kron`**（AST 计数 > 0）。
  * **不论落哪一支，登记里必须写明这条限定**：逐点相乘展平后的 `psi0` 与
    `np.kron` 递推的结果**数值等同** ⇒ 「不共用构造」只在**构造写法**的意义上成立；
    实质差别在**读出量**（下游 `S(L/2)` vs 与 gs 的内积）。**不许把它读成"换了一条独立路径"。**

**A 段（自校验，先跑）**：真调 `stage2_critical_break(L, J, h)`（**不传 void_state**，
即生产路径），复现登记读数 `E0/L = -1.275287` / `S(L/2) = 0.7501` /
`Schmidt 极化 = 250.673` / `gap = 0.366151`（`_runall_v16.3_r1.log:54`）。
**四个数超出容差就不许往下跑**（D9 的笔误就是这样被挡住的）。
另加一条**读出量标定**：`half_entropy(gs)` 必须等于 `stage2` 自己算的 `entropy`
（否则本脚本的 `S(L/2)` 与登记的不是同一个量）。
"""

import os
import sys
import ast
import time
import contextlib
import io

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np            # noqa: E402
from scipy.sparse.linalg import expm_multiply   # noqa: E402

import spiral_model_v16 as G  # noqa: E402

# ── 生产口径（对齐 `_model/stage12.py` 的阶段二默认） ──────────────────────
L_SITE = 16
J = 1.0
H_FIELD = 1.0                 # h/J = 1, 临界点

# ── 预登记常量（跑之前冻结） ──────────────────────────────────────────────
E0_OVER_L_REG = -1.275287     # `_runall_v16.3_r1.log:54`
S_HALF_REG = 0.7501
POL_REG = 250.673
GAP_REG = 0.366151
# 容差 = 登记值里**打印精度最粗**的那个的半末位。pol 打印 3 位小数
# (`stage12.py:116` 的 `{pol:.3f}`) => 半末位 5e-4; 其余三个打印 4~6 位,
# 半末位更小 => 5e-4 是四个数能同时满足的**最紧**容差, 不是随手放宽。
A_TOL = 5e-4
SENS_TOL = 1e-3
TAU_STAR = 1.0 / abs(E0_OVER_L_REG)               # ≈ 0.784138
TAU_GRID = tuple(TAU_STAR * f for f in (0.25, 0.5, 1.0, 2.0, 4.0))


def _quiet(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*a, **kw)
    return out


def count_kron_calls(path):
    """D10-c：数**代码里**的 kron 调用数（AST）—— 注释/docstring 里的不算。"""
    tree = ast.parse(open(path, encoding='utf-8').read())
    return sum(1 for n in ast.walk(tree)
               if isinstance(n, ast.Attribute) and n.attr == 'kron')


def lift_no_kron(eq):
    """把局部态送到 L 个站点上 —— **不调用 np.kron**。

    做法：对每一根轴，把 eq 在该轴上广播成 rank-1 因子（其余轴为 1），
    再把 L 个这样的因子**逐点相乘**。这是 Hadamard 积，不是 kronecker 递推。
    诚实边界：结果与 `np.kron` 递推**数值等同**（见模块 docstring 的限定）。
    """
    T = np.ones((2,) * L_SITE, dtype=complex)
    for axis in range(L_SITE):
        shape = [1] * L_SITE
        shape[axis] = 2
        T = T * np.asarray(eq, dtype=complex).reshape(shape)
    return T.reshape(-1)


def half_entropy(psi):
    """S(L/2)：与 `stage2_critical_break` 同一算法（SVD of 二分矩阵）。"""
    v = np.asarray(psi, dtype=complex).ravel()
    half = L_SITE // 2
    s = np.linalg.svd(v.reshape(2 ** half, 2 ** (L_SITE - half)),
                      compute_uv=False)
    p = s ** 2
    p = p[p > 1e-15]
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def main():
    t0 = time.time()
    print('=' * 76)
    print('v16.4 · γ · D10 —— L1 的下游读出（隔离通路）')
    print(f'  L={L_SITE}, J={J}, h/J={H_FIELD}（生产阶段二口径）')
    print(f'  预登记: DP-9 读 S(L/2); DP-8 只作诊断; TAU_STAR={TAU_STAR:.6f}')
    print(f'  预登记落支判据: rel > {SENS_TOL:g} ⇒ N-下游; 否则 N-不敏感')
    print('=' * 76)

    # ── D10-c 自检：代码里不得有 kron 调用 ────────────────────────────────
    nk = count_kron_calls(os.path.abspath(__file__))
    print(f'\n[D10-c] 本脚本**代码**里的 kron 调用数（AST 计数）= {nk}')

    # ── A 段：复现登记值（生产路径，不传 void_state） ──────────────────────
    print('\n── A 段 · 复现登记的生产读数 ──')
    st2 = _quiet(G.stage2_critical_break, L_SITE, J, H_FIELD)
    got = {'E0/L': st2['E0'] / L_SITE, 'S(L/2)': st2['entropy'],
           'pol': st2['polarization'], 'gap': st2['gap']}
    reg = {'E0/L': E0_OVER_L_REG, 'S(L/2)': S_HALF_REG,
           'pol': POL_REG, 'gap': GAP_REG}
    ok_a = True
    for k in ('E0/L', 'S(L/2)', 'pol', 'gap'):
        d = abs(got[k] - reg[k])
        good = d <= A_TOL
        ok_a = ok_a and good
        print(f'  {k:8s} 实算 {got[k]:.6f}  登记 {reg[k]:.6f}  |diff|={d:.2e}  '
              f'{"OK" if good else "FAIL"}')

    # 读出量标定：本脚本的 S(L/2) 与 stage2 自己算的必须是同一个量
    s_cal = half_entropy(st2['gs'])
    d_cal = abs(s_cal - st2['entropy'])
    cal_ok = d_cal <= 1e-12
    ok_a = ok_a and cal_ok
    print(f'  标定: half_entropy(gs) = {s_cal:.12f}  vs stage2.entropy = '
          f'{st2["entropy"]:.12f}  |diff|={d_cal:.2e}  '
          f'{"OK" if cal_ok else "FAIL"}')
    print(f'  A 段结论: {"全部吻合" if ok_a else "**不吻合 ⇒ 不许往下跑**"}')
    if not ok_a:
        print('\n[A 段失败] 先查口径（L / J / h / 是否生产路径），停止。')
        return 3

    H = G.tfi_periodic_sparse(L_SITE, J, H_FIELD)

    # ── D10-a：L1 的态 →（不共用构造）→ 阶段二算子 → 下游 S(L/2) ──────────
    print('\n── D10-a · L1 局部态经阶段二算子后的下游读出 ──')
    v1 = _quiet(G.stage1_void, 2)
    eq = np.asarray(v1['equal_state'], dtype=complex)
    eq = eq / np.linalg.norm(eq)
    print(f'  L1 局部态: d={eq.size}, equal_state = {np.round(eq, 6).tolist()}')
    psi0 = lift_no_kron(eq)
    print(f'  通路构造: 逐点相乘展平, 维数 {psi0.size} = 2^{L_SITE}, '
          f'‖psi0‖={np.linalg.norm(psi0):.12f}')

    # 错态 |->^{tensor L}（先例 `_v16_smoke_cmera5.py:120`）
    minus = np.array([1.0, -1.0], dtype=complex) / np.sqrt(2.0)
    psi0_bad = lift_no_kron(minus)

    print(f'  τ* = {TAU_STAR:.6f}（= 1/|E0/L|, 预登记）')
    S_good = half_entropy(expm_multiply(-TAU_STAR * H, psi0))
    S_bad = half_entropy(expm_multiply(-TAU_STAR * H, psi0_bad))
    rel = abs(S_good - S_bad) / max(S_good, S_bad, 1e-30)
    print(f'  S(L/2) | τ* 正确态 = {S_good:.10f}')
    print(f'  S(L/2) | τ* 错态   = {S_bad:.10f}')
    print(f'  rel = |ΔS|/max = {rel:.6e}   (判据 > {SENS_TOL:g})')

    # 旁证：τ 网格（**不参与判定**）
    print('\n  ── 旁证（不参与判定）: τ 网格上的 S(L/2) ──')
    print(f'  {"τ":>10s} {"τ/τ*":>6s} {"正确态":>14s} {"错态":>14s} {"rel":>12s}')
    for tau in TAU_GRID:
        sg = half_entropy(expm_multiply(-tau * H, psi0))
        sb = half_entropy(expm_multiply(-tau * H, psi0_bad))
        r = abs(sg - sb) / max(sg, sb, 1e-30)
        print(f'  {tau:10.6f} {tau/TAU_STAR:6.2f} {sg:14.10f} {sb:14.10f} {r:12.4e}')

    # 旁证：两端（读出量的活口）
    print(f'\n  τ→0  端点: S(L/2) = {half_entropy(psi0):.10f}  (乘积态应为 0)')
    print(f'  τ→∞  端点: 登记基态 S(L/2) = {S_HALF_REG} —— 演化终态应当趋近它')
    gs = np.asarray(st2['gs'], dtype=complex).ravel()
    gs = gs / np.linalg.norm(gs)
    # 数值事实（首跑撞出来的, 登记在案）: exp(-tau*H) 是**虚时向上**传播,
    # 范数按 e^{tau*|E0|} 增长 ⇒ tau > ln(DBL_MAX)/|E0| = 709.78/20.404 = 34.79
    # 必然溢出双精度（τ=50 首跑即 nan, LAPACK gesdd 崩）。τ* 与整个 TAU_GRID
    # 都远低于此界。S(L/2) 对整体标度**不变**, 故这里归一化后再读重叠与 S。
    print(f'\n  数值事实: ‖exp(-τH)ψ0‖ ~ e^(τ·|E0|), |E0|={abs(st2["E0"]):.4f} '
          f'⇒ τ > {709.78/abs(st2["E0"]):.2f} 溢出双精度（与判据无关）')
    for tau in (10.0, 20.0, 30.0):
        v = expm_multiply(-tau * H, psi0)
        mx = float(np.max(np.abs(v)))
        # ⚠️ 守卫必须守在**最大元**上, 不能守 np.linalg.norm: 元素 ~1e177 时
        # 元素仍 isfinite, 但 norm 里的平方会溢出 => norm=inf => v/inf=0 =>
        # 会打印出一个**看起来合法的假读数** (S=0, 重叠=0)。首跑就踩了这个。
        if not (np.isfinite(mx) and mx > 0):
            print(f'    τ={tau:>6g}: max|v|={mx:.3e} 非有限 —— 未算（不报假读数）')
            continue
        u = v / mx
        u = u / np.linalg.norm(u)
        print(f'    τ={tau:>6g}: max|v|={mx:.3e}  |<gs|v/‖v‖>|='
              f'{abs(np.vdot(gs, u)):.10f}  S(L/2)={half_entropy(u):.10f}')

    # ── 落支 ────────────────────────────────────────────────────────────
    print('\n── 落支 ──')
    if nk > 0:
        branch = 'N-共用构造'
    elif rel > SENS_TOL:
        branch = 'N-下游'
    else:
        branch = 'N-不敏感'
    print(f'  落支 = **{branch}**')
    print('  ⚠️ 无论哪一支，登记必须带这条限定：逐点相乘展平后的 psi0 与 np.kron')
    print('     递推的结果**数值等同** ⇒「不共用构造」只在**构造写法**上成立；')
    print('     实质差别在**读出量**（下游 S(L/2) vs 与 gs 的内积）。')
    print('\n[D10] 隔离自检：本脚本未调用 stage2_critical_break(void_state=...)，'
          '未接主调用。')
    print(f'总用时 {time.time() - t0:.1f} s')
    print('=' * 76)
    return 0


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.exit(main())
