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
_v16_smoke_cmera3.py —— v16 · 自由费米子高斯表述 + 自检

背景: 探针二判定"不通过"—— 现有构造(entangler 是通用单格点酉门)不含 cMERA 的尺度结构。
要"真标度算符", 格点上无良定义, 必须改到动量空间 / 高斯表述。本探针建这个表述并自检。

**自检策略(关键): 不依赖任何凭记忆写下的 Jordan-Wigner 公式。**
  显式构造 Majorana 算符 γ_a, 然后用对易子检验 H 是否**纯二次**:
      H = (i/4) Σ_ab h_ab γ_a γ_b   <=>   [H, γ_c] 仍落在 span{γ_a} 内
  这条判据只用到 γ_a 的定义, 用不到 JW 的位序/符号约定 —— 故可自证。
  约定本身也是被这条判据**选中**的, 不是我记住的 (见 run 记录的 v1 失败)。

Majorana 约定(由"场项同格点 / 耦合项跨键"反推得到):
  γ_{2i}   = (Π_{j<i} σx_j) σz_i
  γ_{2i+1} = (Π_{j<i} σx_j) σy_i
  ⇒  σx_i      = i γ_{2i} γ_{2i+1}            (同格点双线性)
      σz_i σ_{i+1} = i γ_{2i+1} γ_{2i+2}         (跨键双线性)

检验项:
  T0 稠密周期 H 的基态能量 == v15 的 Jordan-Wigner 解析值 (位序/符号自洽)
  T1 Majorana 代数 {γ_a, γ_b} = 2 δ_ab
  --- A 部分: 开链(边界项不存在, 应严格纯二次) ---
  T2 [H_open, γ_c] 在 span{γ} 内的相对残差
  T3 H_open 与由 h 重建的 H_rec 只差常数
  T4 开链基态的协方差矩阵 Γ 满足 Γ² = -I (纯高斯态)
  T5 由 Γ 算出的能量 == 开链 ED 能量
  --- B 部分: 周期链(报告性, 预期因宇称边界项而失败) ---
  T6 边界项假设: σz_{L-1}σz_0 =? c · i γ_{2L-1} γ_0 · P, P = Π_i σx_i
  T7 周期链 [H_per, γ_c] 相对残差 (与开链对比)

边界:
  - 本探针只回答"高斯表述建不建得起来、对不对", 不产出任何物理结论。
  - L=8 (2^8=256) 保证稠密矩阵可行。高斯表述的价值是**可以上大 L**
    (协方差矩阵只有 2L×2L), 但这要等本表述通过自检之后再说。
  - 若 T6 成立而 T7 失败: 周期 TFI 不是严格自由费米子, 而是**按宇称分扇区**自由。
    这是数学事实, 需在 B2 里显式选口径(开链 / 分扇区), 不能含糊过去。
"""

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'v15'))

from spiral_model_v15 import exact_ground_state, jw_ground_energy  # noqa: E402

_VERSION_TAG = 'v16-smoke-cmera3-2'

_I2 = np.eye(2, dtype=complex)
_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)


# ---------------------------------------------------------------------------
# 显式算符构造。站点 0 = 最高有效位 (cp. v15 位序, 由 T0 核验)
# ---------------------------------------------------------------------------
def _kron_chain(mats):
    out = np.array([[1.0 + 0j]])
    for m in mats:
        out = np.kron(out, m)
    return out


def op_at(op, i, L):
    """把 op 放在站点 i。"""
    mats = [_I2] * L
    mats[i] = op
    return _kron_chain(mats)


def majorana(a, L):
    """γ_{2i} = (Π_{j<i} σx_j) σz_i ; γ_{2i+1} = (Π_{j<i} σx_j) σy_i"""
    i, which = divmod(a, 2)
    return _kron_chain([_SX] * i + [(_SZ if which == 0 else _SY)] + [_I2] * (L - 1 - i))


def tfi_matrix(L, J=1.0, h=1.0, periodic=True):
    """稠密 TFI: H = -J Σ_i σz_i σz_{i+1} - h Σ_i σx_i。"""
    dim = 2 ** L
    H = np.zeros((dim, dim), dtype=complex)
    zs = [op_at(_SZ, i, L) for i in range(L)]
    xs = [op_at(_SX, i, L) for i in range(L)]
    bonds = [(i, (i + 1) % L) for i in range(L)] if periodic else [(i, i + 1) for i in range(L - 1)]
    for i, j in bonds:
        H -= J * (zs[i] @ zs[j])
    for i in range(L):
        H -= h * xs[i]
    return H


# ---------------------------------------------------------------------------
# 自由费米子判据: [H, γ_c] 是否落在 span{γ_a}
#   H = (i/4) Σ_ab h_ab γ_a γ_b  ⇒  [H, γ_c] = i Σ_a h_ac γ_a
# ---------------------------------------------------------------------------
def fermion_test(H, Gam, dim):
    """返回 (最差相对残差, h 矩阵)。"""
    n = len(Gam)
    hmat = np.zeros((n, n), dtype=complex)
    worst = 0.0
    for c in range(n):
        X = H @ Gam[c] - Gam[c] @ H
        coeffs = np.array([np.trace(Gam[a] @ X) / dim for a in range(n)])
        res = X - sum(coeffs[a] * Gam[a] for a in range(n))
        worst = max(worst, float(np.linalg.norm(res)) / max(float(np.linalg.norm(X)), 1e-30))
        hmat[:, c] = coeffs / 1j
    return worst, hmat


def reconstruct(hmat, Gam):
    n = len(Gam)
    H = np.zeros(Gam[0].shape, dtype=complex)
    for a in range(n):
        for b in range(n):
            H += 0.25j * hmat[a, b] * (Gam[a] @ Gam[b])
    return H


def build_h(L, J, h):
    """直接构造 H_til 的 h 矩阵 (2L×2L), 不经 2^L 空间。
       H_til = -J Σ_i iγ_{2i+1}γ_{2i+2} - h Σ_i iγ_{2i}γ_{2i+1} + J·iγ_{2L-1}γ_0
       单对项 H=(i/2) h_ab γ_aγ_b (a<b) ⇒  h_{2i+1,2i+2} = -2J
                                          h_{2i,2i+1}   = -2h
                                          h_{2L-1,0}    = +2J
    """
    H = np.zeros((2 * L, 2 * L))
    for i in range(L - 1):
        H[2 * i + 1, 2 * i + 2] = -2.0 * J
        H[2 * i + 2, 2 * i + 1] = +2.0 * J
    for i in range(L):
        H[2 * i, 2 * i + 1] = -2.0 * h
        H[2 * i + 1, 2 * i] = +2.0 * h
    H[2 * L - 1, 0] = +2.0 * J
    H[0, 2 * L - 1] = -2.0 * J
    return H


def gs_energy_from_h(Hh):
    """H=(i/4)Σh_ab γ_aγ_b; h 特征值为 ±iε_k ⇒ E_gs = -(1/4) Σ|Im λ|。"""
    return -0.25 * float(np.sum(np.abs(np.linalg.eigvals(Hh).imag)))


def covariance(gs, Gam):
    """Γ_ab = i<γ_a γ_b> - i δ_ab。"""
    n = len(Gam)
    G = np.zeros((n, n), dtype=complex)
    for a in range(n):
        for b in range(n):
            G[a, b] = 1j * np.vdot(gs, Gam[a] @ (Gam[b] @ gs)) - 1j * (a == b)
    return G


# ---------------------------------------------------------------------------
def main():
    print(f"=== v16 探针 · 自由费米子高斯表述自检  [{_VERSION_TAG}] ===")
    L, J, h = 8, 1.0, 1.0
    dim = 2 ** L
    print(f"L={L}, J={J}, h={h}, dim={dim}")

    Gam = [majorana(a, L) for a in range(2 * L)]
    n_gam = 2 * L

    # ---- T0: 位序/符号核验 (周期链 vs v15 解析 JW) ----
    print("\n[T0] 稠密周期 H 的基态能量 vs v15 Jordan-Wigner 解析值")
    H_per = tfi_matrix(L, J, h, periodic=True)
    H_open = tfi_matrix(L, J, h, periodic=False)
    e_dense = float(np.linalg.eigvalsh(H_per)[0])
    e_jw = jw_ground_energy(L, J, h)
    t0 = abs(e_dense - e_jw) < 1e-9
    print(f"    dense(周期) = {e_dense:.12f}   JW 解析 = {e_jw:.12f}"
          f"   偏差 = {abs(e_dense - e_jw):.2e}")
    print(f"    -> {'一致 (位序/符号自洽)' if t0 else '不一致'}")

    # ---- T1: Majorana 代数 ----
    print("\n[T1] Majorana 代数 {γ_a, γ_b} = 2 δ_ab")
    max_dev = 0.0
    for a in range(n_gam):
        for b in range(n_gam):
            ac = Gam[a] @ Gam[b] + Gam[b] @ Gam[a]
            max_dev = max(max_dev, float(np.abs(ac - 2.0 * (a == b) * np.eye(dim)).max()))
    t1 = max_dev < 1e-12
    print(f"    最大偏差 = {max_dev:.2e}   -> {'通过' if t1 else '不通过'}")

    # =======================================================================
    # A 部分: 开链 (无宇称边界项, 应严格纯二次)
    # =======================================================================
    print("\n--- A 部分: 开链 (L-1 条键, 无边界项) ---")

    print("\n[T2] 自由费米子判据: [H_open, γ_c] 落在 span{γ} 内的相对残差")
    res_open, hmat_open = fermion_test(H_open, Gam, dim)
    t2 = res_open < 1e-10
    print(f"    最差相对残差 = {res_open:.2e}   -> {'通过 (H 纯二次)' if t2 else '不通过'}")

    print("\n[T3] 由 h 重建 H_rec, 检验 H_open - H_rec 是否为常数")
    diff = H_open - reconstruct(hmat_open, Gam)
    off = diff - np.eye(dim) * (np.trace(diff) / dim)
    t3 = float(np.abs(off).max()) < 1e-9
    print(f"    非对角残差 = {np.abs(off).max():.2e}"
          f"   -> {'通过 (只差常数)' if t3 else '不通过'}")

    print("\n[T4] 开链基态协方差矩阵 Γ 是否纯高斯 (Γ² = -I)")
    gs0 = np.linalg.eigh(H_open)[1][:, 0]            # 开链基态用本文件的稠密 ED
    Gamma = covariance(gs0, Gam)
    dev_pure = float(np.abs(Gamma @ Gamma + np.eye(n_gam)).max())
    t4 = dev_pure < 1e-8
    print(f"    |Γ² + I| 最大元 = {dev_pure:.2e}   -> {'通过' if t4 else '不通过'}")

    print("\n[T5] 由 Γ 算出的能量 vs 开链 ED 能量")
    e0_dense = float(np.linalg.eigvalsh(H_open)[0])
    e_from_gamma = float(np.real(np.sum(hmat_open * Gamma)) / 4.0)
    t5 = abs(e_from_gamma - e0_dense) < 1e-9
    print(f"    开链 ED 能量    = {e0_dense:.12f}")
    print(f"    由 Γ 算出的能量 = {e_from_gamma:.12f}   偏差 = {abs(e_from_gamma - e0_dense):.2e}")
    print(f"    -> {'一致 (定量闭环)' if t5 else '不一致'}")

    # =======================================================================
    # B 部分: 周期链 (报告性)
    # =======================================================================
    print("\n--- B 部分: 周期链 (报告性) ---")

    print("\n[T6] 边界项假设: σz_{L-1}σz_0 =? c · i γ_{2L-1} γ_0 · P,  P = Π_i σx_i")
    P = np.eye(dim, dtype=complex)
    for i in range(L):
        P = P @ op_at(_SX, i, L)
    lhs = op_at(_SZ, L - 1, L) @ op_at(_SZ, 0, L)
    rhs = 1j * (Gam[2 * L - 1] @ Gam[0]) @ P
    # 允许整体符号 c = ±1
    c = float(np.real(np.trace(lhs @ rhs.conj().T))) / dim
    resid6 = float(np.abs(lhs - c * rhs).max())
    t6 = abs(abs(c) - 1.0) < 1e-9 and resid6 < 1e-9
    print(f"    最佳 c = {c:+.6f}   残差 = {resid6:.2e}"
          f"   -> {'成立 (边界项 = 双线性 × 宇称)' if t6 else '不成立'}")

    print("\n[T7] 周期链 [H_per, γ_c] 相对残差 (对比开链)")
    res_per, _ = fermion_test(H_per, Gam, dim)
    print(f"    周期链 最差相对残差 = {res_per:.2e}")
    print(f"    开链   最差相对残差 = {res_open:.2e}")
    print(f"    -> 周期链{'仍纯二次' if res_per < 1e-10 else '**不**纯二次 (与开链形成对照)'}")

    # ---- T8: 偶宇称扇区上恢复纯二次 (把周期链接回高斯表述) ----
    #   T6 给出 σz_{L-1}σz_0 = -iγ_{2L-1}γ_0·P, 而 H_per = H_open - J·σz_{L-1}σz_0
    #   ⇒ H_per = H_open + J·iγ_{2L-1}γ_0·P;  P=+1 扇区上即为纯二次的 H_til。
    print("\n[T8] 偶宇称扇区: H_til = H_open + J·i γ_{2L-1} γ_0 是否恢复纯二次")
    e_pbc_v15, gs_pbc = exact_ground_state(L, J, h)
    gs_pbc = np.asarray(gs_pbc, dtype=complex).reshape(-1)
    gs_pbc /= np.linalg.norm(gs_pbc)
    parity_exp = float(np.real(np.vdot(gs_pbc, P @ gs_pbc)))
    print(f"    [T8a] 周期基态的宇称 <P> = {parity_exp:+.12f}"
          f"   -> {'偶宇称扇区 (P=+1, 前提成立)' if abs(parity_exp - 1) < 1e-9 else '非偶扇区'}")

    H_til = H_open + 1j * J * (Gam[2 * L - 1] @ Gam[0])
    res_til, hmat_til = fermion_test(H_til, Gam, dim)
    print(f"    [T8b] H_til 最差相对残差 = {res_til:.2e}"
          f"   -> {'通过 (纯二次)' if res_til < 1e-10 else '不通过'}")

    e_til = float(np.linalg.eigvalsh(H_til)[0])
    print(f"    [T8c] H_til 基态能量 = {e_til:.12f}")
    print(f"          周期 JW 解析值 = {e_jw:.12f}   偏差 = {abs(e_til - e_jw):.2e}")
    t8c = abs(e_til - e_jw) < 1e-9
    print(f"          -> {'一致 (周期链已接回高斯表述)' if t8c else '不一致'}")
    t8 = t8c and res_til < 1e-10 and abs(parity_exp - 1) < 1e-9

    # =======================================================================
    # C 部分: 纯 Γ 表示 (2L×2L, 不经 2^L 希尔伯特空间) —— 这才是"高斯表述"的价值
    # =======================================================================
    print("\n--- C 部分: 纯 Γ 表示 (2L×2L) 与规模 ---")

    print("\n[T9] 纯 h 构造 (2L×2L) 与稠密路径的一致性")
    h_small = build_h(L, J, h)
    dev_h = float(np.abs(h_small - hmat_til).max())
    t9a = dev_h < 1e-12
    print(f"    [T9a] 与稠密路径 hmat_til 的最大元差 = {dev_h:.2e}"
          f"   -> {'一致' if t9a else '不一致'}")
    e_h_small = gs_energy_from_h(h_small)
    t9b = abs(e_h_small - e_til) < 1e-9
    print(f"    [T9b] L={L}: 由 h 的谱算得 E_gs = {e_h_small:.12f}")
    print(f"          稠密 H_til 基态能量   = {e_til:.12f}   偏差 = {abs(e_h_small - e_til):.2e}")
    print(f"          -> {'一致 (BdG 谱闭合)' if t9b else '不一致'}")

    print("\n[T9c] 规模: 只有 2L×2L, 无 2^L 希尔伯特空间")
    t9c = True
    for L_big in (32, 128, 512):
        t_a = time.time()
        hb = build_h(L_big, J, h)
        e_big = gs_energy_from_h(hb)
        dt = time.time() - t_a
        e_ref = jw_ground_energy(L_big, J, h)
        dev = abs(e_big - e_ref) / abs(e_ref)
        ok_big = dev < 1e-10
        t9c = t9c and ok_big
        print(f"    L={L_big:>4}: E_gs = {e_big:.10f}  JW = {e_ref:.10f}"
              f"  相对偏差 = {dev:.2e}  用时 = {dt * 1e3:.1f} ms"
              f"  {'OK' if ok_big else '**不符**'}")
    print(f"    内存: h 为 {2 * 512}×{2 * 512} float64 = {2 * 512 * 2 * 512 * 8 / 1e6:.1f} MB"
          f" (对比 L=32 稠密矢量 2^32 不可行)")
    t9 = t9a and t9b and t9c

    # ---- 判定 ----
    checks = {'T0 位序/符号': t0, 'T1 Majorana 代数': t1,
              'T2 纯二次(开链)': t2, 'T3 完整重建(开链)': t3,
              'T4 纯高斯(开链)': t4, 'T5 能量闭环(开链)': t5,
              'T6 边界项=双线性×宇称': t6, 'T8 周期接回高斯表述': t8,
              'T9 纯 h 表示(可上大 L)': t9}
    print("\n=== 判定 ===")
    for k, v in checks.items():
        print(f"    {k:<22}: {'通过' if v else '不通过'}")
    ok = all(checks.values())
    print(f"\n    高斯表述: {'可用 —— 开链严格成立, 周期链经宇称扇区亦成立' if ok else '未通过, 不得进入下一步'}")
    print(f"    周期链 T7(不设扇区) 残差 = {res_per:.2e} ⇒ 原始 H_per **不**纯二次;")
    print(f"        T8 用 H_til=H_open+J·iγ_{{2L-1}}γ_0 在偶扇区恢复, 能量与周期 JW 一致。")
    print("    => 周期 TFI 的自由费米子性**依赖宇称扇区**; 这是数学事实, 不是数值误差。")
    print("       B2 实施时必须显式声明用的是 H_til(偶扇区) 还是开链, 不能含糊过去。")
    print("    边界: 本探针不产出物理结论; 它是后续任何 cMERA 构造的**前提**。")
    return 0 if ok else 3


if __name__ == '__main__':
    sys.exit(main())
