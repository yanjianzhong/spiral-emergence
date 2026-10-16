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
AKLT (自旋 1 VBS) 链的精确 chi=2 MPS —— v16.9·T9 的**实体**；v16.10·P1 起兼作生产侧诊断项
=========================================================================================

本模块是什么
------------
AKLT 链有一组**解析已知**的性质，本仓把它当**外部**负对照系统用（不是本仓的模型对象）：

    ⟨H⟩ = −(2/3)·n_bonds 而 Σ_bonds⟨P_2⟩ = 0     （**注意 ⟨H⟩ ≠ 0**，见下）
    开链中点 S(L/2) = ln2                         （精确，与 L 无关）
    环单点 ρ = I_3/3 ⇒ S(1) = ln3                 （精确）
    环块谱 SU(2) **1+3** 简并                      （精确）
    max‖Σ_s A^{s†}A^s − I‖ = 0                    （精确等距）
    环 S(L/2) → ln4                               （渐近，单调升）
最关键的一条是**物理的**：AKLT 有 Haldane 能隙、纠缠遵守面积律、**没有 CFT 对应**。

v16.9·T9 用六把解析尺子 (O1–O6) 把"精确 chi=2 MPS 造对了"钉死，再把这份 S(L/2)
序列喂进**仓库自己那条非临界判据**（`_model/checks.py:345` 的
`gapped: |c_scaling| < 0.05`），并配三条负对照 N1/N2/N3。全部读数已登记在
`spiral_v16_audit.md` §9.15。

v16.10·P1 做了两件事
--------------------
1. **原语搬家**：`_v16_smoke_t9_aklt.py` 里那些张量 / 截割收缩 / 转移矩阵 / 拟合
   原语**整块搬到这里**（纯移动：同名同体，只去掉前导下划线）。冒烟脚本改为
   `from _model import aklt as AK` **真调，不复制逻辑**（与 `_model/hyper.py`、
   `_model/tnr.py` 同一个做法）。理由只有一条：生产侧也要这份读数，而
   **判据只能有一份**。
2. **新增 `diagnostic_readings()`**：把 T9 的四条读数（gapped 判据 + N1/N2/N3）
   连**判据**一起算出来，供 `_metric/main.py` 以 `record_guard(..., expect_pass=False)`
   登记成诊断项。冒烟脚本与生产流程**共用这一个函数** ⇒ 两边不可能分歧。

模块边界（同 `_model/hyper.py` / `_model/tnr.py`）
--------------------------------------------------
  · 只 import numpy；不 import 本包其它模块，不 import 任何 `_metric.*`；
  · 不调用 `record_*` / `metric_*`；不打印；不进 `_v16_run_all.py` 的 ENTRIES；
  · 判据常量（`C_GAPPED` / `N2_EPS` / `N1_SEP` / `TOL_N2_*`）是 `v16.9_plan.md` §2
    事前写死、`checks.py:345` 原文照抄的，本模块**只搬运不重定义**。

口径（读结论时不许省）
----------------------
  · AKLT **不是**本仓的模型对象。本模块的读数回答的是「仓库自家那条非临界判据在
    **别处**的系统上答对了吗」，**不是**对本仓任一物理结论的验证（T9 诚实边界 7）。
  · 四条读数**同源**（`source_id = 'T9_aklt'`）：同一次 AKLT 构造、同一套转移矩阵与
    截割收缩。**不得**写成「互相印证 / 独立佐证」（`_v16_claim_ledger.py` 的检查 (c)）。
  · `c_scaling` 是**复刻算式**（`checks.py:388-396` 那 8 行），不是调用既有拟合函数
    —— 既有那段内嵌在 `spectral_central_charge` 里、入参是 TFI 的 J/h（模型专属），
    对 AKLT 无法直接调用（T9 诚实边界 6）。
  · 第一号陷阱：⟨H⟩ = −(2/3)·n_bonds，**不是 0**。真正为零的是 Σ_bonds⟨P_2⟩。
    **任何写成「H|ψ⟩ = 0」的检查都会假红。**
"""

import numpy as np

__all__ = [
    'A_tensors', 'spin1', 'bond_ops', 'block_tensors', 'schmidt_from_cut', 'S_of',
    'cut_entropy', 'E', 'G', 'powers', 'ring_bond_sum', 'open_bond_sum',
    'bond_expect', 'ring_site_rdm', 'c_scaling', 'state_norm2', 'keep_only',
    'iso_residual', 'diagnostic_readings',
    'L_LIST', 'KC', 'LN2', 'LN3', 'LN4', 'P_THRESH',
    'TOL_PHYS', 'TOL_ISO', 'TOL_ASYM',
    'C_GAPPED', 'N2_EPS', 'TOL_N2_ISO', 'TOL_N2_P2', 'N1_SEP',
    'BL', 'BR', 'UP', 'DOWN',
]

# ══════════════════════════════════════════════════════════════════════════
# 0. 常量（判据 / 容差 / 窗口）—— 逐字搬自 `_v16_smoke_t9_aklt.py` v3
# ══════════════════════════════════════════════════════════════════════════
L_LIST = (8, 10, 12, 14, 16)
TOL_PHYS = 1e-12        # 「期望为数值 0」的物理量
TOL_ISO = 1e-14         # 「精确等距」
TOL_ASYM = 1e-6         # 「渐近」量

KC = 2.0 / np.sqrt(3.0)                      # 计划 §1.2b 的全局因子
LN2 = float(np.log(2.0))
LN3 = float(np.log(3.0))
LN4 = float(np.log(4.0))
P_THRESH = 1e-14                             # 谱截断（与规划期探针同一门限）

C_GAPPED = 0.05        # 阶段 2：`_model/checks.py:345` 的 `gapped` 判据**原文照抄**
N2_EPS = 0.05          # 阶段 3 N2 的注入强度（计划 §2 N2 行原文）
TOL_N2_ISO = 1e-3      # 阶段 3 N2/N3：注入后等距残差的**下限**（判据原文）
TOL_N2_P2 = 1e-4       # 阶段 3 N2：注入后 Σ⟨P_2⟩ 的**下限**（判据原文）
N1_SEP = 0.5           # 阶段 3 N1：与同取向未注入基线的**最小**分离（判据原文）

BL = np.array([1.0, 0.0], dtype=complex)     # 开链左边界 |↑⟩
BR = np.array([0.0, 1.0], dtype=complex)     # 开链右边界 |↓⟩
UP = np.array([1.0, 0.0], dtype=complex)     # |↑⟩（阶段 3 N2 的注入用）
DOWN = np.array([0.0, 1.0], dtype=complex)   # |↓⟩


# ══════════════════════════════════════════════════════════════════════════
# 1. 精确张量
# ══════════════════════════════════════════════════════════════════════════
def A_tensors():
    """AKLT (spin-1 VBS) 的精确 MPS 张量，**含全局因子 2/sqrt3**（计划 §1.2b）。

    A^s = (2/sqrt3)·P^s·M。不补这个因子会让「等距性正对照」自己判成负结果
    （sum_s A^{s†}A^s = (3/4)I，不是 I）。返回形状 (3, 2, 2)。
    """
    Pp = np.array([[1, 0], [0, 0]], dtype=complex)
    P0 = np.array([[0, 1], [1, 0]], dtype=complex) / np.sqrt(2.0)
    Pm = np.array([[0, 0], [0, 1]], dtype=complex)
    M = np.array([[0, 1], [-1, 0]], dtype=complex) / np.sqrt(2.0)
    return np.array([KC * (P @ M) for P in (Pp, P0, Pm)])


def spin1():
    """自旋 1 的三个分量，基 |+1⟩, |0⟩, |−1⟩（与 `A` 的物理指标 0,1,2 同序）。"""
    s2 = np.sqrt(2.0)
    Sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex) / s2
    Sy = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=complex) / s2
    Sz = np.diag([1.0, 0.0, -1.0]).astype(complex)
    return Sx, Sy, Sz


def bond_ops():
    """两体算符（9x9，基 |s,t⟩，索引 i = 3s + t；排法 `Om[输出, 输入] = ⟨s't'|O|st⟩`）。

    x  = S_i·S_{i+1}
    P2 = 总自旋 2 的投影 = x²/6 + x/2 + 1/3      （S_tot² = 4 + 2x ⇒ Lagrange 插值）
    h  = x + x²/3                                （计划 §2 O1 的**定义式**）
    返回 (x, P2, h)。
    """
    Sx, Sy, Sz = spin1()
    x = np.kron(Sx, Sx) + np.kron(Sy, Sy) + np.kron(Sz, Sz)
    I9 = np.eye(9, dtype=complex)
    P2 = x @ x / 6.0 + x / 2.0 + I9 / 3.0
    h = x + x @ x / 3.0
    return x, P2, h


# ══════════════════════════════════════════════════════════════════════════
# 2. MPS 截割收缩（唯一新算法，d 无关；计划 §5）
# ══════════════════════════════════════════════════════════════════════════
def block_tensors(A, n):
    """L[a, b, idx] = (A^{s_1}···A^{s_n})[a, b]，idx 小端枚举 3^n 个位形。"""
    dim = 3 ** n
    out = np.zeros((2, 2, dim), dtype=complex)
    for idx in range(dim):
        x, m = idx, np.eye(2, dtype=complex)
        for _ in range(n):
            m = m @ A[x % 3]
            x //= 3
        out[:, :, idx] = m
    return out


def schmidt_from_cut(Lt, Rt, periodic):
    """截割处的 Schmidt 值（降序）。秩 ≤ 4（周期）/ ≤ 2（开链）⇒ 只需 4x4 以内的 SVD。

    周期: psi = Σ_ab L^{ab} ⊗ R^{ba}     ⇒ U_{(ab)} = L^{ab},  W_{(ab)} = R^{ba}
    开链: psi = Σ_b (Σ_a bl_a L^{ab}) ⊗ (Σ_c R^{bc} br_c)
    对 U^T = Q_U R_U、W^T = Q_W R_W 做 QR ⇒ 谱 = svd(R_U R_W^T)。
    """
    if periodic:
        U = Lt.reshape(4, -1)
        W = np.transpose(Rt, (1, 0, 2)).reshape(4, -1)
    else:
        U = np.einsum('a,abz->bz', BL, Lt).reshape(2, -1)
        W = np.einsum('abz,b->az', Rt, BR).reshape(2, -1)
    _, Ru = np.linalg.qr(U.T)
    _, Rw = np.linalg.qr(W.T)
    return np.linalg.svd(Ru @ Rw.T, compute_uv=False)


def S_of(sv):
    """von Neumann 熵（自然对数），由 Schmidt 值算。归一化与 SVD 的整体标度无关。"""
    p = np.asarray(sv) ** 2
    p = p / p.sum()
    p = p[p > P_THRESH]
    return float(-np.sum(p * np.log(p)))


def cut_entropy(A, L, n, periodic):
    """S(n) = 前 n 个连续站点的 von Neumann 熵（**d 无关**路径，不造 3^L 稠密向量）。"""
    return S_of(schmidt_from_cut(block_tensors(A, n),
                                 block_tensors(A, L - n), periodic))


# ══════════════════════════════════════════════════════════════════════════
# 3. 转移矩阵（两点 / 单点算符的精确期望值）
# ══════════════════════════════════════════════════════════════════════════
def E(A):
    """E[(a,a'),(b,b')] = Σ_s A^s_{ab} · conj(A^s_{a'b'})。（4x4）"""
    return np.einsum('sab,sAB->aAbB', A, A.conj()).reshape(4, 4)


def G(A, Om):
    """两点算符插入的转移矩阵（4x4）。

    G[(a,a'),(d,d')] = Σ_{i,i'} Om[i',i]·B^i_{ad}·conj(B^{i'}_{a'd'})，B^i = A^s A^t。
    """
    B = np.einsum('sab,tbc->stac', A, A).reshape(9, 2, 2)
    return np.einsum('ji,iad,jAD->aAdD', Om, B, B.conj()).reshape(4, 4)


def powers(E, L):
    Ep = [np.eye(4, dtype=complex)]
    for _ in range(L):
        Ep.append(Ep[-1] @ E)
    return Ep


def ring_bond_sum(A, L, Om):
    """周期环: (Σ_{全部 L 个键} ⟨O_{i,i+1}⟩, ⟨ψ|ψ⟩) —— trace 形式，精确。"""
    # 注: 局部名用 Ek/Gk —— 顶层函数就叫 `E` / `G`，写 `E = E(A)` 会因
    # "赋值即使名字成为局部" 而 UnboundLocalError（搬家时踩到过）。
    Ek = E(A)
    Gk = G(A, Om)
    Ep = powers(Ek, L)
    tot = np.trace(Gk @ Ep[L - 2])                      # 环绕键 (L, 1)
    for i in range(1, L):                               # 键 (i, i+1), i = 1..L−1
        tot += np.trace(Ep[i - 1] @ Gk @ Ep[L - 1 - i])
    return complex(tot), complex(np.trace(Ep[L]))


def open_bond_sum(A, L, Om):
    """开链: (Σ_{L−1 个键} ⟨O_{i,i+1}⟩, ⟨ψ|ψ⟩)。"""
    Ek = E(A)
    Gk = G(A, Om)
    vL = np.outer(BL, BL.conj()).reshape(4)
    vR = np.outer(BR, BR.conj()).reshape(4)
    Ep = powers(Ek, L)
    tot = 0.0 + 0j
    for i in range(1, L):
        tot += vL @ Ep[i - 1] @ Gk @ Ep[L - 1 - i] @ vR
    return complex(tot), complex(vL @ Ep[L] @ vR)


def bond_expect(A, L, Om, periodic):
    """归一后的 Σ_键⟨O⟩（两个取向统一入口）。"""
    tot, nrm = ring_bond_sum(A, L, Om) if periodic else open_bond_sum(A, L, Om)
    return tot / nrm


def ring_site_rdm(A, L, site=1):
    """周期环单点约化密度矩阵 ρ_{s,s'} = ⟨ψ| |s⟩⟨s'|_site |ψ⟩ / ⟨ψ|ψ⟩。"""
    Ek = E(A)
    Ep = powers(Ek, L)
    nrm = np.trace(Ep[L])
    rho = np.zeros((3, 3), dtype=complex)
    for s in range(3):
        for sp in range(3):
            Qm = np.zeros((3, 3), dtype=complex)
            Qm[s, sp] = 1.0                             # |s⟩⟨s'|
            G1 = np.einsum('uv,vab,uAB->aAbB', Qm, A, A.conj()).reshape(4, 4)
            rho[s, sp] = np.trace(Ep[site - 1] @ G1 @ Ep[L - site]) / nrm
    return rho


def iso_residual(A):
    """max‖Σ_s A^{s†}A^s − I‖ —— O5 正对照 / N2 负对照**共用**的同一个检查器。"""
    return float(np.max(np.abs(np.einsum('sba,sbc->ac', A.conj(), A) - np.eye(2))))


# ══════════════════════════════════════════════════════════════════════════
# 4. 拟合与注入（阶段 2 / 阶段 3 的装置）
# ══════════════════════════════════════════════════════════════════════════
def c_scaling(L_list, s_half):
    """照 `v16/_model/checks.py:388-396` 的算式**逐行复刻**：S(L/2) = (c/3)·ln L + b。

    返回 (c_scaling, b, r2, yb)。既有那段拟合内嵌在 `spectral_central_charge` 里、
    入参是 TFI 的 J/h（模型专属），对 AKLT 无法直接调用 ⇒ 只能复刻算式（诚实边界 6）。
    `r2` 在零方差下分母为 0、不可读，**不作判据**。
    """
    xL = np.log(np.array(L_list, dtype=float))
    Ab = np.vstack([xL, np.ones_like(xL)]).T
    yb = np.array(s_half, dtype=float)
    cb, *_ = np.linalg.lstsq(Ab, yb, rcond=None)
    rb = yb - Ab @ cb
    den = float(((yb - yb.mean()) ** 2).sum())
    r2 = float('nan') if den == 0.0 else 1.0 - float(rb @ rb) / den
    return 3.0 * float(cb[0]), float(cb[1]), r2, yb


def state_norm2(A, L, periodic):
    """未归一的 ⟨ψ|ψ⟩：环取 tr(E^L)，开链取 v_L^T E^L v_R（与 `bond_expect` 同一范数）。"""
    Ep = powers(E(A), L)
    if periodic:
        return complex(np.trace(Ep[L]))
    vL = np.outer(BL, BL.conj()).reshape(4)
    vR = np.outer(BR, BR.conj()).reshape(4)
    return complex(vL @ Ep[L] @ vR)


def keep_only(A, k):
    """只留第 k 个物理分量的张量（阶段 3 N1 的候选注入；k = 0/1/2 ⇔ A^+/A^0/A^-）。"""
    Ak = np.zeros_like(A)
    Ak[k] = A[k]
    return Ak


# ══════════════════════════════════════════════════════════════════════════
# 5. 四条诊断读数（v16.10·P1：冒烟脚本与生产流程**共用**的唯一判据出处）
# ══════════════════════════════════════════════════════════════════════════
def diagnostic_readings(L_list=L_LIST):
    """T9 的四条诊断读数（gapped 判据 + N1/N2/N3），**连判据一起**算出来。

    返回 dict，键：
      'L_list'   → 参与拟合 / 对照的 L 列表
      'seq_open' / 'seq_ring' → 开链 / 环的中点 S(L/2) 序列（与 `L_list` 同序）
      'gapped'   → {'c_open','b_open','r2_open','c_ring','b_ring','r2_ring','ok','cond'}
                   `ok` = 两个取向都 |c_scaling| < C_GAPPED（`checks.py:345` 原文）
      'iso'      → O5 正对照的等距残差（未注入）
      'iso_n2'   → N2 注入后的等距残差（同一个 `iso_residual` 检查器）
      'n1'       → {'per_L': [(L, ‖ψ‖², S(L/2), 环基线, |ΔS|, ok_L), ...], 'ok', 'cond'}
      'n2'       → {'per': [(periodic, L, 基线Σ⟨P_2⟩, 注入后Σ⟨P_2⟩, ok_L), ...], 'ok', 'cond'}
      'n3'       → {'ok', 'cond'}
      'n_ok'     → 上述四条里 ok 的条数（0..4）

    **`ok` 的含义是「按设计工作」**（判据答对 / 对照被检出），不是「应当失败」：
      · N1/N2 的 `ok` 说的是**注入后被检出**（S 掉到 0、Σ⟨P_2⟩ 抬起来、等距残差 > 1e-3）；
      · N3 的 `ok` 要求**两条同时成立**：正对照真归零 **且** 负对照真报错 ——
        这正是"检查器既非橡皮图章、也非一律报错"的那条双向判据。

    代价：约 2–4 s（`block_tensors` 在 L=16 上做 3^8 次 2x2 乘）。生产侧每次全流程
    **只调一次**。
    """
    A = A_tensors()
    _, P2, _ = bond_ops()

    # ---- 两条 S(L/2) 序列（gapped 拟合与 N1 的基线共用同一份）----
    seq_ring_d = {L: cut_entropy(A, L, L // 2, True) for L in L_list}
    seq_ring = [seq_ring_d[L] for L in L_list]
    seq_open = [cut_entropy(A, L, L // 2, False) for L in L_list]

    # ---- gapped 判据（P2-1 开链 / P2-2 环）----
    c_o, b_o, r2_o, _ = c_scaling(L_list, seq_open)
    c_r, b_r, r2_r, _ = c_scaling(L_list, seq_ring)
    gapped_ok = bool(abs(c_o) < C_GAPPED and abs(c_r) < C_GAPPED)
    gapped_cond = (f"AKLT 两取向 |c_scaling| < {C_GAPPED} (`_model/checks.py:345` 原文): "
                   f"开链 {c_o:+.7f}, 环 {c_r:+.7f}")

    # ---- N1 键维向下 χ:2→1（只留 A^0；**只落在环上**，见冒烟脚本诚实边界 5）----
    A1 = keep_only(A, 1)
    n1_per, n1_ok = [], True
    for L in L_list:
        f2 = state_norm2(A1, L, True).real        # 有效性前置：零向量的熵无定义
        s = cut_entropy(A1, L, L // 2, True)
        base = seq_ring_d[L]
        sep = abs(s - base)
        ok_L = bool((f2 > 0.0) and (s < TOL_PHYS) and (sep > N1_SEP))
        n1_per.append((L, f2, s, base, sep, ok_L))
        n1_ok = n1_ok and ok_L
    n1_sep_min = min(e[4] for e in n1_per)
    n1_cond = (f"χ:2→1 (只留 A^0) 环态 L={L_list[0]}..{L_list[-1]}: ‖ψ‖²>0 且 "
               f"S(L/2) < {TOL_PHYS:g} 且与环基线差 > {N1_SEP} "
               f"(最小分离 {n1_sep_min:.6f}, {sum(1 for e in n1_per if e[5])}/{len(n1_per)} 格)")

    # ---- N2 非酉注入 B[1] += N2_EPS·|↓⟩⟨↑| ----
    B = A.copy()
    B[1] = B[1] + N2_EPS * np.outer(DOWN, UP)
    iso_n2 = iso_residual(B)
    n2_per, n2_ok = [], bool(iso_n2 > TOL_N2_ISO)
    for per in (True, False):
        for L in L_list:
            base = bond_expect(A, L, P2, per).real          # 未注入基线
            e_p2 = bond_expect(B, L, P2, per).real
            ok_L = bool((abs(base) < TOL_PHYS) and (e_p2 > TOL_N2_P2))
            n2_per.append((per, L, base, e_p2, ok_L))
            n2_ok = n2_ok and ok_L
    n2_p2_min = min(e[3] for e in n2_per)
    n2_cond = (f"B[1] += {N2_EPS}·|↓⟩⟨↑| 后: 等距残差 {iso_n2:.3e} > {TOL_N2_ISO:g} 且 "
               f"Σ⟨P_2⟩ 由 <{TOL_PHYS:g} 升到 > {TOL_N2_P2:g} "
               f"(最小 {n2_p2_min:.3e}, 2 取向 × {len(L_list)} L 全过)")

    # ---- N3 等距性**双向**配对 ----
    iso = iso_residual(A)
    n3_ok = bool((iso < TOL_ISO) and (iso_n2 > TOL_N2_ISO))
    n3_cond = (f"等距性双向配对: O5 正对照残差 {iso:.3e} < {TOL_ISO:g} 且 "
               f"N2 负对照残差 {iso_n2:.3e} > {TOL_N2_ISO:g}")

    gapped = {'c_open': c_o, 'b_open': b_o, 'r2_open': r2_o,
              'c_ring': c_r, 'b_ring': b_r, 'r2_ring': r2_r,
              'ok': gapped_ok, 'cond': gapped_cond}
    n1 = {'per_L': n1_per, 'ok': bool(n1_ok), 'cond': n1_cond}
    n2 = {'per': n2_per, 'ok': bool(n2_ok), 'cond': n2_cond}
    n3 = {'ok': n3_ok, 'cond': n3_cond}
    return {'L_list': tuple(L_list), 'seq_open': seq_open, 'seq_ring': seq_ring,
            'gapped': gapped, 'iso': iso, 'iso_n2': iso_n2,
            'n1': n1, 'n2': n2, 'n3': n3,
            'n_ok': int(gapped_ok) + int(n1['ok']) + int(n2['ok']) + int(n3['ok'])}
