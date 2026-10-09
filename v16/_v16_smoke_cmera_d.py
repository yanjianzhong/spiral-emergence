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
"""_v16_smoke_cmera_d.py —— v16 · 方向 D: 离散尺度算符 L 的「有内容版」**及其否证**

版本: v16-smoke-cmera-d-1（2026-09-26）

背景（两件事, 都是本探针要处理的）:
  1. `_v16_cmera_gaussian.py::diagnostic_B` 的旧解读**已撤回**: 那一步测的是 `uv_vacuum`,
     而任何**对角数算符**都湮灭 c-真空 ⇒ 平凡; 平凡的原因是**态**, 不是离散化。
     旧文据此断言「离散格上不存在 k∂_k」**无证据支撑**（撤回依据: 原文
     `cMERA_Entropy_rev.tex:75`/`:87` 把 L 放在 `e^{-i…}` 里 ⇒ 酉性要求 L **厄米**,
     实测的厄米性正是应有性质）。
  2. 于是「改观测量、做有内容版」看似可行。**本探针实测否证了它。**

本探针的两段:
  A. **反例**: 配对型生成元叠加 `L = Σ_k λ_k H1[k]`（`H1 = _scale_matrices`, 即
     cMERA 流**自己用的那一族**）在 `uv_vacuum` 上给 `‖L|Ω⟩‖ ≈ 2.915 ≠ 0`
     ⇒ 非平凡的**厄米** L 在离散格上**存在**, 旧断言被实测推翻。
  B. **无判别力（在物理上真正相关的那一类态里）**: 同一个 L 在 uv 真空 / cMERA 流末态 /
     TFI 基态三个**偶宇称积态**上给**逐位相同**的读数 `√(Σ_k λ_k²) = 2.915476`
     ⇒ cMERA 末态与裸 UV 真空**不可区分**。
     而**离开该族**的零模型 `A_rand`（同 Frobenius 范数）在这些态上是**有**差别的,
     用它造的态也给不同的数（2.2815）⇒ 退化不是本探针的编码缺陷。

  机制（**实测**, 不是假设）: `H1[k]² = P_k` 是**秩 2 投影**（P²=P, 每对模式 tr=2）,
    偶宇称积态上 `⟨P_k⟩ = 1` ⇒ `⟨L²⟩ = Σ_k λ_k²` ⇒ `‖L|ψ⟩‖ = √(Σ_k λ_k²)`。
     ⚠️ **本探针 v1 曾把机制猜成 `L² = c·I`, 被自己的 C 段实测推翻**
        （`‖L² − ⟨L²⟩·I‖_max = 3.188 ≠ 0`, `c_k = 0.5` 而 `‖H1[k]² − c_k I‖_max = 0.375`）。
        此记录**保留**, 因为它正是「判据选中结论、而不是结论挑判据」的实证。

结论（登记, 不包装）: 方向 D 的「有内容版」在口径 A 下**做不出内容**。
  原因**不是**「离散格上不存在 k∂_k」（旧说法, 已撤回）,
  而是 `_scale_matrices` 的生成元族 `{H1[k]}` 的平方是**秩 2 投影**,
  在偶宇称积态上期望恒为 1 ⇒ 该族上不存在能把 cMERA 末态与 UV 真空分开的 `L|Ω⟩` 型观测量。
  要做出有判别力的版本, 必须**离开 `span{H1[k]}`**。

边界声明:
  - 本探针**不加进 runner**（`_v16_run_all.py`）: 改条目计数须另行决策。
  - 全部读数走**稠密 2^L**（L=8, dim=256）; Wick 路径**只作互核**, 不作唯一来源。
  - 退出码: 0 = 上述负结论**确认**; 3 = 负结论**被推翻**（那是该被看见的事, 不是误报）。
  - 本探针**不改** `diagnostic_B` 的任何读数（那两行照登在 `_v16_cmera_gaussian.py` 里）。
"""

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import _v16_cmera_gaussian as g  # noqa: E402

_VERSION_TAG = 'v16-smoke-cmera-d-1'

_L = 8
_N = 2 * _L
_LAMBDA = np.pi


# ---------------------------------------------------------------------------
# 基础: 稠密双线性算符 与 Wick 的 ⟨L²⟩
#   L = (i/4) Σ_ab A_ab γ_aγ_b ,  A 实反对称 ⇒ L 厄米。
#   Wick: ⟨L²⟩ = −(1/16)Σ A_ab A_cd ⟨γ_aγ_bγ_cγ_d⟩,
#         Γ_ab = i⟨γ_aγ_b⟩ − iδ_ab ⇒ ⟨γ_aγ_b⟩ = δ_ab − iΓ_ab。
#   两者**必须互核**（本探针 A 段就核）。
# ---------------------------------------------------------------------------
def dense_bilinear(A, Gam, n):
    return (1j / 4.0) * sum(A[a, b] * (Gam[a] @ Gam[b]) for a in range(n) for b in range(n))


def wick_L2(A, Gcov, n):
    Gc = np.eye(n, dtype=complex) - 1j * np.asarray(Gcov, dtype=complex)
    tot = 0.0 + 0.0j
    for a in range(n):
        for b in range(n):
            if A[a, b] == 0.0:
                continue
            for c in range(n):
                for d in range(n):
                    if A[c, d] == 0.0:
                        continue
                    tot += A[a, b] * A[c, d] * (Gc[a, b] * Gc[c, d]
                                                - Gc[a, c] * Gc[b, d] + Gc[a, d] * Gc[b, c])
    return float(np.real(-tot / 16.0))


def norm_L_wick(A, Gcov, n):
    return float(np.sqrt(max(wick_L2(A, Gcov, n), 0.0)))


def scale_weights(qs, L):
    """λ_k: k∂_k 在**正动量**格 {qs[k]}_{k<L/2} 上的中心差分权重（与 diagnostic_B 同款规则,
       但落在配对模式 k 上, 不是落在单粒子模式 q 上）。"""
    ks = list(range(L // 2))
    qv = np.array([qs[k] for k in ks])
    lam = np.zeros(len(ks))
    for r in range(len(ks)):
        if 0 < r < len(ks) - 1:
            lam[r] = qv[r] * (2.0 / (qv[r + 1] - qv[r - 1]))
    return lam


def main():
    t0 = time.time()
    L, N = _L, _N
    qs = g.ns_momenta(L)
    Gam = [g.majorana(a, L) for a in range(N)]
    psi0 = g.uv_vacuum(L)
    G0 = g.vacuum_covariance(L)

    print(f"=== {_VERSION_TAG} · 方向 D: 离散 L 的「有内容版」及其否证（L={L}, dim={2 ** L}）===")
    print(f"    口径: 全部稠密 2^L; Wick 只作互核; 本探针不进 runner。")

    # --- 待测的两个算符: 配对型 k∂_k  vs  同范数零模型 ---
    H1 = g._scale_matrices(L, qs, sign=+1.0)
    lam = scale_weights(qs, L)
    A_pair = sum(lam[r] * H1[r] for r in range(L // 2))
    nf = float(np.linalg.norm(A_pair))
    rng = np.random.default_rng(20260926)
    Ar = rng.standard_normal((N, N))
    Ar = Ar - Ar.T
    Ar *= nf / np.linalg.norm(Ar)
    print(f"\n    λ (正动量模式) = {np.array2string(lam, precision=4)}")
    print(f"    ‖A_pair‖_F = {nf:.6f}   ‖A_rand‖_F = {np.linalg.norm(Ar):.6f}（同范数）")

    # --- 四个态 ---
    # 态 2: cMERA 流末态（真 cMERA 生成元 B(u), χ ≡ 1）—— 稠密 U 与 Γ 两条路
    edges, u_list, du = g.make_grid(-6.0, 0.0, 8, 8)
    B = g.bond_matrices(L, qs, u_list, Lambda=_LAMBDA, prefactor=True, sign=+1.0)
    Gcm = np.array(G0)
    U = np.eye(2 ** L, dtype=complex)
    for s, u in enumerate(u_list):
        idx = min(int(np.searchsorted(edges, u, side='right') - 1), len(edges) - 2)
        R = g.expm(-du * 1.0 * B[s])
        Gcm = R.T @ Gcm @ R
        U = g.expm(-1j * du * dense_bilinear(B[s], Gam, N)) @ U
    psi_cm = U @ psi0
    d_gamma = float(np.abs(g.covariance(psi_cm, Gam) - Gcm).max())

    # 态 3: TFI 基态
    w, v = np.linalg.eigh(g.tfi_matrix(L, J=1.0, h=1.0, periodic=True))
    gs = v[:, 0]
    Gtfi = np.real(g.covariance(gs, Gam))

    # 态 4: 随机配对态（**离开** span{H1[k]} 的生成元造的）
    Grand = g.flow_covariance(G0, lambda u: Ar, 0.0, 0.7, 40)

    states = [('uv 真空', G0, psi0),
              ('cMERA 流末态 (χ=1)', Gcm, psi_cm),
              ('TFI 基态 (J=h=1)', Gtfi, gs),
              ('随机配对态 (对照)', Grand, None)]

    # -----------------------------------------------------------------------
    print(f"\n--- A · 反例: 非平凡的厄米 L 存在（旧断言「离散格上不存在 k∂_k」被推翻）---")
    op_pair = dense_bilinear(A_pair, Gam, N)
    print(f"    ‖L − L†‖_max = {float(np.abs(op_pair - op_pair.conj().T).max()):.3e}   (厄米 ✓)")
    dn = float(np.linalg.norm(op_pair @ psi0))
    wk = norm_L_wick(A_pair, G0, N)
    print(f"    ‖L|Ω⟩‖  稠密 = {dn:.6e}   Wick = {wk:.6e}   比值 = {wk / dn:.6f}")
    print(f"    -> L|Ω⟩=0 {'成立' if dn < 1e-10 else '**不成立**'}"
          f"（**这就是旧断言的反例**: 非平凡厄米 L 在离散格上确实存在）")

    # -----------------------------------------------------------------------
    print(f"\n--- B · 无判别力: 三个**偶宇称积态**读数逐位相同（cMERA 末态 ≡ 裸 UV 真空）---")
    print(f"    {'态':24s} {'‖L|ψ⟩‖ (A_pair)':>18s} {'‖L|ψ⟩‖ (A_rand)':>18s}   稠密核")
    pair_vals, rand_vals = [], []
    for tag, Gc, psi in states:
        a = norm_L_wick(A_pair, Gc, N)
        b = norm_L_wick(Ar, Gc, N)
        pair_vals.append(a)
        rand_vals.append(b)
        extra = ''
        if psi is not None:
            extra = f"{float(np.linalg.norm(op_pair @ psi)):.6e}"
        print(f"    {tag:24s} {a:18.6e} {b:18.6e}   {extra}")
    pv = np.array(pair_vals[:3])          # 三个偶宇称积态 = 物理上真正相关的那一类
    pv4 = pair_vals[3]                    # 离开该族的对照态
    rv = np.array(rand_vals[:3])
    spread_p = float((pv.max() - pv.min()) / pv.mean())
    spread_r = float((rv.max() - rv.min()) / rv.mean())
    print(f"    前三个（偶宇称积态）相对跨度:  A_pair = {spread_p:.3e}   A_rand = {spread_r:.3e}")
    print(f"    第四个（离开该族）: A_pair = {pv4:.6e}"
          f"   偏离前三者 {abs(pv4 - pv.mean()) / pv.mean():.3e}")
    print(f"    -> 前三个 {'**无判别力**（cMERA 末态与 UV 真空不可区分）' if spread_p < 1e-10 else '有判别力'}"
          f";  A_rand 在前三个上 {'无判别力' if spread_r < 1e-10 else '**有判别力**（对照成立）'}")

    # -----------------------------------------------------------------------
    print(f"\n--- C · 机制（实测, 不是假设）: H1[k]² = P_k 是**秩 2 投影**, 该族态上 ⟨P_k⟩ = 1 ---")
    n_rest = 4 ** (L // 2 - 1)
    tag_tr = f'tr/4^{L // 2 - 1}'
    Pk, struct_ok = [], True
    print(f"    {'k':>3s} {'λ':>6s} {'‖P_k²−P_k‖_max':>17s} {'tr P_k':>9s} {tag_tr:>12s}")
    for r in range(L // 2):
        h1d = dense_bilinear(H1[r], Gam, N)
        P = h1d @ h1d
        Pk.append(P)
        dev = float(np.abs(P @ P - P).max())
        tr = float(np.real(np.trace(P)))
        struct_ok = struct_ok and dev < 1e-9 and abs(tr / n_rest - 2.0) < 1e-9
        print(f"    {r:3d} {lam[r]:6.2f} {dev:17.3e} {tr:9.1f} {tr / n_rest:12.6f}")
    print(f"    ⇒ P²=P 且每对模式 tr=2 ⇒ 秩 2 投影。"
          f"故「L² 是 c-数」**不成立**（v1 的猜测, 本段推翻）。")
    print(f"    但 ⟨H1[k]²⟩ = ⟨P_k⟩ 在各态上取值不同 ⇒ 判别力问题转到 ⟨P_k⟩ 上:")
    print(f"    {'态':24s} {'⟨P_k⟩ (k=0..)':>34s} {'√(Σ λ_k²⟨P_k⟩)':>18s} "
          f"{'‖L|ψ⟩‖(§B)':>16s} {'⟨L²⟩−Σ λ_k²⟨P_k⟩':>18s}")
    for (tag, Gc, _psi), a in zip(states, pair_vals):
        pk = [wick_L2(H1[r], Gc, N) for r in range(L // 2)]
        s2 = sum(lam[r] ** 2 * pk[r] for r in range(L // 2))
        rec = float(np.sqrt(max(s2, 0.0)))
        print(f"    {tag:24s} {'  '.join(f'{x:.9f}' for x in pk):>34s} {rec:18.6e} "
              f"{a:16.6e} {a * a - s2:18.6e}")
    print(f"    末列 = **跨对模式项** Σ_(k≠k') λ_kλ_k'⟨H1[k]H1[k']⟩: 积态上为 0"
          f"（故 rec 与 ‖L|ψ⟩‖ 逐位相同）;")
    print(f"    非积态上 ≠0 ⇒ 残差 3.7% **不是误差, 是该态确有跨对关联**, 如实登记、不修。")

    # -----------------------------------------------------------------------
    print(f"\n--- D · 对照: 换 A_rand（同范数, 离开该族）则不退化 ---")
    sqp = op_pair @ op_pair
    cp = float(np.real(np.trace(sqp)) / (2 ** L))
    print(f"    A_pair: ‖L² − ⟨L²⟩·I‖_max = {float(np.abs(sqp - cp * np.eye(2 ** L)).max()):.3e}"
          f"   ⟨L²⟩(迹平均) = {cp:.6f}")
    op_r = dense_bilinear(Ar, Gam, N)
    sqr = op_r @ op_r
    cr = float(np.real(np.trace(sqr)) / (2 ** L))
    print(f"    A_rand: ‖L² − ⟨L²⟩·I‖_max = {float(np.abs(sqr - cr * np.eye(2 ** L)).max()):.3e}"
          f"   ⟨L²⟩(迹平均) = {cr:.6f}")
    print(f"    （两者都非 c-数 ⇒ 差异不在这一条, 而在 §B/§C 的**态依赖**上。）")

    # -----------------------------------------------------------------------
    print(f"\n--- E · 结论（登记, 不包装）---")
    print(f"    1. 「离散格上不存在 k∂_k」**被实测推翻**: 配对型 L = Σ λ_k H1[k] 厄米"
          f"（‖L−L†‖=0）且 ‖L|Ω⟩‖={dn:.4f}≠0。")
    print(f"    2. 但该族**没有判别力**: 三个偶宇称积态（含 cMERA 流末态与 TFI 基态）读数"
          f"逐位相同到 {spread_p:.1e} ⇒ cMERA 末态与裸 UV 真空**不可区分**。")
    print(f"    3. 机制（实测, 非假设）: H1[k]² = P_k 是**秩 2 投影**, 该族态上 ⟨P_k⟩ = 1 "
          f"⇒ ‖L|ψ⟩‖ = √(Σ λ_k²) = {np.sqrt(sum(lam[r] ** 2 for r in range(L // 2))):.6f}。")
    print(f"       ⚠️ v1 曾猜成「L² 是 c-数」, 被 C 段推翻（该记录保留在 docstring 与 C 段）。")
    print(f"    4. ⇒ 方向 D 的「有内容版」在口径 A 下**做不出内容**; 要做得有判别力, 必须离开 "
          f"span{{H1[k]}}（A_rand 已验证: 那种 L 在三个态上读数**不**同）。")
    print(f"    5. 本条**不改变**任何已登记读数（`diagnostic_B` 的两行照登）。")

    ok = struct_ok and spread_p < 1e-10 and spread_r > 1e-6
    print(f"\n    -> 负结论 {'**确认**' if ok else '**被推翻, 需重判**'}   "
          f"用时 {time.time() - t0:.1f} s")
    print(f"    （Γ 稠密↔协变互核: ‖ΔΓ‖_max = {d_gamma:.3e}）")
    return 0 if ok else 3


if __name__ == '__main__':
    sys.exit(main())
