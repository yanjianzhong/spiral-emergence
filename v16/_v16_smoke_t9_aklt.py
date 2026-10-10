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
v16.9 · T9 · AKLT 链负对照 —— 阶段 1 精确构造 + oracle O1–O6；
                          阶段 2 接代码台账（L2 面积律判别器）；阶段 3 负对照 N1–N3
[_VERSION_TAG = 'v16-smoke-t9-aklt-3']

锚（文档锚）:
  `v16.9_plan.md` §2 阶段 1 —— 判据与落支名**事前写死**在该节，本文件只执行
  `v16.9_plan.md` §1.2 —— 张量缺全局因子 `2/sqrt(3)`；`qtn.MPS` 在本环境不存在
  `v16.9_plan.md` §5    —— `d` 无关的截割熵函数（唯一新算法），规划期已与稠密 SVD 交叉核对
  `v16.9_plan.md` §4    —— 红线 2（判据不回头改）/ 红线 3（规划期读数非台账值）/ 红线 4（不改既有代码）
  `v16.9_pre.md:25-30`  —— `P^s`、`M` 的字面来源（pre 原文，一字未改）
  `v16.9_plan.md` §2 阶段 2 —— L2 判别器（**头号产物**）：做法、判据、落支名事前写死
  `v16.9_plan.md` §2 阶段 3 —— 负对照 N1–N3：注入方式与判据事前写死
  `v16/_model/checks.py:388-396` —— `c_scaling` 的拟合算式（本文件**逐行复刻这 8 行**）
  `v16/_model/checks.py:345`     —— `gapped: |c_scaling| < 0.05`（判据原文，照抄不改）
  `v16/_metric/solvers.py:51-62` —— 既有 `entanglement_curve`（**真调一次，一行不改**）

靶: 把 AKLT（自旋 1、有 Haldane 能隙、面积律纠缠、**无 CFT 对应**）的
  **精确 chi=2 MPS 基态**造成本仓可审计的对象，并用六个**解析已知**的 oracle 把它钉死。
  这是 v16.9 的主对象：现有 13 条负对照（`_metric/extras.py:583`）**全部**跑在
  临界横场 Ising 上，AKLT 是补这个缺口最省的正确选择（计划 §0.0）。
  阶段 2 把这份 S(L/2) 序列喂进**仓库自己那条非临界判据**（`_model/checks.py:345`
  的 `gapped: |c_scaling| < 0.05`）—— 这是 v16.9 的**头号产物**；
  阶段 3 配三条已标定的负对照 N1–N3；**登记在阶段 4**（本文件不登记）。

======================================================================
预登记（成文于实现之前；跑之前不许改）
======================================================================
张量（`v16.9_pre.md:25-30` 的 `P^s`、`M`，**补计划 §1.2b 的全局因子**）:
  A^s = (2/sqrt3) · P^s M,   s = +1, 0, −1   ⇔  物理指标 0, 1, 2
  P^+ = |↑⟩⟨↑|,  P^0 = (|↑⟩⟨↓| + |↓⟩⟨↑|)/sqrt2,  P^− = |↓⟩⟨↓|,  M = (|↑↓⟩ − |↓↑⟩)/sqrt2
  边界（开链）: bl = |↑⟩, br = |↓⟩ —— 两端各留一个自由自旋 1/2，取单态。
  约定: `A[s][a, b]`，`a` = 左虚指标，`b` = 右虚指标。

**第一号陷阱（计划 §2 O1 段逐字登记的措辞）**: ⟨H⟩ = −(2/3)·n_bonds，**不是 0**。
  真正为零的是**投影子之和** Σ_bonds⟨P_2⟩。
  **任何写成「H|ψ⟩ = 0」的检查都会假红。**

容差（三类，不混用）:
  TOL_PHYS = 1e-12  —— 用于**期望为数值 0** 的物理量（O1 / O1' / O2 / O3 / O4）
  TOL_ISO  = 1e-14  —— 用于**精确等距**（O5；规划期实测为精确 0.0）
  TOL_ASYM = 1e-6   —— 用于**渐近**量（O6 的 L=16 → ln4；不是精确等式）
  自检门限  _CONS = 1e-14 —— 只用于本文件内部的算法一致性（S 系列），**不作物理判据**

判据（O1–O6；L ∈ {8,10,12,14,16}，**开链与周期环各跑一遍**）:
  O1   ⟨H⟩ = −(2/3)·n_bonds                    |Δ| < TOL_PHYS
  O1'  Σ_bonds⟨P_2⟩ = 0                        |·| < TOL_PHYS
  O2   开链中点 S(L/2) = ln2（精确、与 L 无关） |Δ| < TOL_PHYS
  O3   环 n=1：ρ = I_3/3 ⇒ S = ln3             |Δ| < TOL_PHYS
  O4   环块谱 SU(2) **1+3** 简并               三重简并残差 < TOL_PHYS
  O5   max‖Σ_s A^{s†}A^s − I‖ = 0              < TOL_ISO
  O6   环 S(L/2) → ln4：单调升 且 |S(L=16) − ln4| < TOL_ASYM

落支（事前写死）:
  `T9-构造成立`  O1–O6 全过 ⇒ EXIT=0 ⇒ 可进阶段 2/3
  `T9-构造失败`  任一不过 ⇒ EXIT=3 ⇒ **停工上报，不改判据、不改容差**（计划 §4 红线 2）

阶段 2 的预登记（成文于实现之前；跑之前不许改）:
  做法: 用本文件的 S(L/2) 序列（L = 8,10,12,14,16，**开链 + 环两组**），
        照 `_model/checks.py:388-396` 的算式 S(L/2) = (c/3)·ln L + b 拟合取 c_scaling；
        再**显式调用一次**既有 `_metric.solvers.entanglement_curve`（d=3 输入），
        把它的失败**照实记录**（那个函数**一行不改**，计划 §4 红线 4）。
  C_GAPPED = 0.05 —— `checks.py:345` 原文照抄，**不是本文件选的数**
  判据:
    P2-1  开链 |c_scaling| < C_GAPPED
    P2-2  环   |c_scaling| < C_GAPPED
    P2-3  既有 `entanglement_curve` 对 d=3 输入**抛异常**（`2**n` 写死，计划 §1.4a）
  **R^2 明确不作判据**（计划 §2 阶段 2；零方差下不可读）—— 照实打印，不解读。
  落支:
    `T9-L2正确判gapped`  P2-1 与 P2-2 都过，且抛错被如实记录
    `T9-L2假临界`        任一 |c_scaling| >= C_GAPPED
    `T9-硬编码不成立`    未抛错（即那个函数其实能吃 d=3）⇒ 计划 §1.4 的锚定需订正

阶段 3 的预登记（成文于实现之前；跑之前不许改）:
  N1  键维向下 χ:2→1（积态，三个候选里只留一个张量）
      S(L/2) < TOL_PHYS **且** 与同取向未注入基线之差 > 0.5
  N2  非酉注入 B[1] += 0.05·|↓⟩⟨↑|
      max‖Σ_s B^{s†}B^s − I‖ > 1e-3 **且** Σ_bonds⟨P_2⟩ 由 < TOL_PHYS 升到 > 1e-4
  N3  等距性**双向**配对：O5 残差 < TOL_ISO **且** N2 残差 > 1e-3（两条须**同时**成立）
  落支:
    `T9-负对照三条齐`  N1–N3 全过 ⇒ EXIT=0
    `T9-负对照缺陷`    任一不过 ⇒ EXIT=3，**记下是哪条、照实报告，不调容差**

  实现期的**并存订正**（上面 N1 的预登记**原文保留、一字不删**；本条只记实测，
  见诚实边界 5 —— 第二轮跑出的「N1 全 OK」是**假阳**，该轮日志已冻结存档
  `_t9_aklt_run2_N1假阳_frozen_2026-10-10.log`）:
    - 照字面「只留 A^+」/「只留 A^-」⇒ **零向量**（‖ψ‖² 精确 0）。`_S_of` 对**空谱**
      **静默**返回 `-0.000e+00`，于是「S < 1e-12」被一个**无定义的量**满足。
    - 「只留 A^0」在**环**上合法（谱 = 单支，真积态），在**开链**上仍是零向量。
    - ⇒ N1 **只落在环上**，并加**有效性前置** `‖ψ‖² > 0`；开链一格
      **结构性不可实现**，登记为「**无对象**」而非「判据不过」（措辞同计划 §1.3）。
      判据本体（`S < TOL_PHYS` 且 与基线差 `> 0.5`）**一个字没改**。

范围红线（同计划 §2「本轮不做」+ §4）:
  - 既有函数 `entanglement_curve` / `fit_central_charge` / `boundary_correlation_graph`
    **一行都不改**（红线 4）—— 阶段 2 **只真调一次** `entanglement_curve` 并记录其失败；
    `fit_central_charge` / `boundary_correlation_graph` **不调**；
  - **不修** `entanglement_curve`（那是 DP-T9-3 的事）；不动 L5 的几何路径；
  - **不登记**（阶段 4）；**不进 runner**（DP-T9-4）；不加守卫 / 指标
    ⇒ 规模三数 **78 / 27 / 13 不动**；
  - **不造 3^L 稠密向量作为主路径**（L=20 需 51.96 GiB，超红线 1）；
    稠密态只在 L=6 自检与阶段 2 喂给 `entanglement_curve` 的那一次（L=8）上出现，
    主路径一律是 MPS 截割收缩 / 转移矩阵。

用时: 秒级（截割路径单次 L=16 全序列 < 5 s；见计划 §5 成本锚）。
      确切用时见日志末行，不在此处预登记。

[诚实边界]
  1. 本文件输出的**判据读数是本阶段的产出**。但计划 §1/§2 里印的 AKLT 数字标
     **`规划期实测`**（该口径名定义在计划 §0.1）—— 那是临时探针的输出，
     **不是任何已登记读数**，登记只发生在阶段 4（红线 3）。
     两者不一致时**以本文件为准**，并把不符照实写进计划 §0.1.3（红线 2：只报，不改判据）。
  2. **实现期发现（口径订正，已并存于计划 §0.1.3）**：计划 §2 O1 段的散文把算子恒等式
     印成 `H = (4/3)·ΣP_2 − (2/3)·n_bonds·I`；**正确恒等式是 `H = 2·ΣP_2 − (2/3)·n_bonds·I`**
     （`P_2 = x²/6 + x/2 + 1/3`、`h = x + x²/3` ⇒ `h = 2P_2 − (2/3)I`，`x = S_i·S_{i+1}`）。
     本文件把这件事做成**自检 S2**：预测残差**恰为 2/3**，且**只在 S_tot=2 子空间上不为零**——
     而 AKLT 态不占该子空间 ⇒ **两者在 AKLT 态上数值相同 ⇒ O1 的判据不受影响**。
     计划 §2 原文**一字未改**。
  3. 本文件**不**声称 pre 的物理错了：它只说张量缺一个全局因子（计划 §1.2b），
     而那个因子**不影响物理态**（只让张量不再等距）。
  4. **O4 首轮实现写死了单态的位置 ⇒ 假红，已改**。计划 §2 O4 的判据原文是
     「三重简并残差 < 1e-12」，**一字未改**；首轮把残差写成 `max(|p1−p2|,|p2−p3|)`，
     预设了「三重态在 1,2,3、单态在 0」。实测 **L=10 与 L=14 的单态落在末尾**，
     于是残差被读成 8.196e-03 / 9.141e-04 ⇒ 落 `T9-构造失败`。
     那一轮的完整日志**已冻结存档**：`_t9_aklt_run1_O4位置假设_frozen_2026-10-10.log`
     （与本文件首轮输出逐字相同，未删、未改）。
     改法：`_triple_resid` 在**两个候选三元组里取更紧的那个**，不预设位置。
     由此**暴露出一条计划没预见的事实**：四值里「哪一支在上」**随 L mod 4 交替**
     （L=8/12/16 单态在上；L=10/14 三重在上）—— 本文件照实标签，**不解读**。
  5. **N1 的注入措辞在实现期被订正（并存标注；计划 §2 原文一字未改）**：
     计划 §2 N1 的注入列逐字是「`A^{0}, A^{-}` 置零，只留 `A^{+}`」，§1.7a 同。
     **实测该写法给出零向量**：`A^{+} = (2/sqrt3)·P^{+}M = (2/sqrt3)·[[0, 1/sqrt2],[0, 0]]`
     是**幂零秩一** ⇒ `(A^{+})^L = 0`（L >= 2），开链态与环态**恒为零向量**，
     熵无从定义（`A^{-}` 同理，只有 `A^{0}` 非幂零）。
     本文件把**三个候选的 ‖ψ‖² 都打印出来**，N1 用**唯一合法的那一支**（只留 `A^{0}`），
     并把这件事照实写进计划 §0.1.4 作为**并存订正**。
     **第二轮（`v16-smoke-t9-aklt-2`）据此仍出了假阳**：它把「只留 A^0」喂给**两个取向**，
     而开链那一支恒为零向量；`_S_of` 里 `p = p / p.sum()` 在 `p.sum() == 0` 时得 `nan`，
     过滤后数组**为空**，`-np.sum(空)` 返回 `-0.0` ⇒ **零向量被静默判成「S = 0，通过」**，
     10 条 N1 子判据**全打 OK**（日志里同时漏出 `RuntimeWarning: invalid value in divide`）。
     该轮日志**已冻结存档**：`_t9_aklt_run2_N1假阳_frozen_2026-10-10.log`（未删、未改）。
     第三轮（`v16-smoke-t9-aklt-3`）的改法**只加一条有效性前置**：N1 判据加上 `‖ψ‖² > 0`，
     且**只落在环上**；开链一格登记为「**无对象**」。**`_S_of` 本身一行未改**
     （它属于既有代码，且其行为在合法态上并无错误）。
  6. **阶段 2 的 `c_scaling` 是「复刻算式」不是「调用既有拟合函数」**：既有那段拟合内嵌在
     `checks.py` 的 `spectral_central_charge` 里，且其入参是 TFI 的 J/h（**模型专属**，
     计划 §1.4c），对 AKLT **无法直接调用**。故本文件**逐行复刻**其 `:388-396` 的 8 行算式。
     **真调既有函数的那一条只有 `entanglement_curve`（P2-3）。**
  7. **本文件不声称「台账框架在非临界系统上已被验证」**：它只证明**一条**判据（`gapped`）
     在 AKLT 上**给出了事前已知的正确答案**，以及三条负对照在 AKLT 上**仍能失败**。
     七层覆盖、可复用性、「堵死审稿人」等结论**不在本文件范围内**。
"""

import os
import sys
import time

_T_PROC = time.time()      # ← 进程起点，早于任何 import

import numpy as np      # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:                                # 阶段 2 P2-3：**真调既有函数，不复制逻辑**
    from _metric.solvers import entanglement_curve as _entanglement_curve   # noqa: E402
    _EC_IMPORT_ERR = None
except Exception as _exc:                                    # pragma: no cover
    _entanglement_curve = None
    _EC_IMPORT_ERR = _exc

_T_IMPORT = time.time()
_T0 = time.time()

_VERSION_TAG = 'v16-smoke-t9-aklt-3'

L_LIST = (8, 10, 12, 14, 16)
TOL_PHYS = 1e-12        # 「期望为数值 0」的物理量
TOL_ISO = 1e-14         # 「精确等距」
TOL_ASYM = 1e-6         # 「渐近」量
_CONS = 1e-14           # 内部一致性自检（比 TOL_PHYS 严）

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
def _aklt_tensors():
    """AKLT (spin-1 VBS) 的精确 MPS 张量，**含全局因子 2/sqrt3**（计划 §1.2b）。

    A^s = (2/sqrt3)·P^s·M。不补这个因子会让「等距性正对照」自己判成负结果
    （sum_s A^{s†}A^s = (3/4)I，不是 I）。返回形状 (3, 2, 2)。
    """
    Pp = np.array([[1, 0], [0, 0]], dtype=complex)
    P0 = np.array([[0, 1], [1, 0]], dtype=complex) / np.sqrt(2.0)
    Pm = np.array([[0, 0], [0, 1]], dtype=complex)
    M = np.array([[0, 1], [-1, 0]], dtype=complex) / np.sqrt(2.0)
    return np.array([KC * (P @ M) for P in (Pp, P0, Pm)])


def _spin1():
    """自旋 1 的三个分量，基 |+1⟩, |0⟩, |−1⟩（与 `A` 的物理指标 0,1,2 同序）。"""
    s2 = np.sqrt(2.0)
    Sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex) / s2
    Sy = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=complex) / s2
    Sz = np.diag([1.0, 0.0, -1.0]).astype(complex)
    return Sx, Sy, Sz


def _bond_ops():
    """两体算符（9x9，基 |s,t⟩，索引 i = 3s + t；排法 `Om[输出, 输入] = ⟨s't'|O|st⟩`）。

    x  = S_i·S_{i+1}
    P2 = 总自旋 2 的投影 = x²/6 + x/2 + 1/3      （S_tot² = 4 + 2x ⇒ Lagrange 插值）
    h  = x + x²/3                                （计划 §2 O1 的**定义式**）
    返回 (x, P2, h)。
    """
    Sx, Sy, Sz = _spin1()
    x = np.kron(Sx, Sx) + np.kron(Sy, Sy) + np.kron(Sz, Sz)
    I9 = np.eye(9, dtype=complex)
    P2 = x @ x / 6.0 + x / 2.0 + I9 / 3.0
    h = x + x @ x / 3.0
    return x, P2, h


# ══════════════════════════════════════════════════════════════════════════
# 2. MPS 截割收缩（唯一新算法，d 无关；计划 §5）
# ══════════════════════════════════════════════════════════════════════════
def _block_tensors(A, n):
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


def _schmidt_from_cut(Lt, Rt, periodic):
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


def _S_of(sv):
    """von Neumann 熵（自然对数），由 Schmidt 值算。归一化与 SVD 的整体标度无关。"""
    p = np.asarray(sv) ** 2
    p = p / p.sum()
    p = p[p > P_THRESH]
    return float(-np.sum(p * np.log(p)))


def _cut_entropy(A, L, n, periodic):
    """S(n) = 前 n 个连续站点的 von Neumann 熵（**d 无关**路径，不造 3^L 稠密向量）。"""
    return _S_of(_schmidt_from_cut(_block_tensors(A, n),
                                   _block_tensors(A, L - n), periodic))


def _triple_resid(p):
    """1+3 结构里的「三重简并残差」：两个候选三元组 {p0,p1,p2} / {p1,p2,p3} 取更紧的那个，
    报其最大两两差。**不预设单态那一支在上还是在下** —— 实测两种都出现
    （首轮实现写死 `p[1]≈p[2]≈p[3]`，在 L=10/14 上假红；见计划 §0.1.3）。
    返回 (残差, 标签)。"""
    a = max(abs(p[0] - p[1]), abs(p[1] - p[2]), abs(p[0] - p[2]))
    b = max(abs(p[1] - p[2]), abs(p[2] - p[3]), abs(p[1] - p[3]))
    return (a, '三重在上') if a <= b else (b, '单态在上')


# ══════════════════════════════════════════════════════════════════════════
# 3. 转移矩阵（两点 / 单点算符的精确期望值）
# ══════════════════════════════════════════════════════════════════════════
def _E(A):
    """E[(a,a'),(b,b')] = Σ_s A^s_{ab} · conj(A^s_{a'b'})。（4x4）"""
    return np.einsum('sab,sAB->aAbB', A, A.conj()).reshape(4, 4)


def _G(A, Om):
    """两点算符插入的转移矩阵（4x4）。

    G[(a,a'),(d,d')] = Σ_{i,i'} Om[i',i]·B^i_{ad}·conj(B^{i'}_{a'd'})，B^i = A^s A^t。
    """
    B = np.einsum('sab,tbc->stac', A, A).reshape(9, 2, 2)
    return np.einsum('ji,iad,jAD->aAdD', Om, B, B.conj()).reshape(4, 4)


def _powers(E, L):
    Ep = [np.eye(4, dtype=complex)]
    for _ in range(L):
        Ep.append(Ep[-1] @ E)
    return Ep


def _ring_bond_sum(A, L, Om):
    """周期环: (Σ_{全部 L 个键} ⟨O_{i,i+1}⟩, ⟨ψ|ψ⟩) —— trace 形式，精确。"""
    E = _E(A)
    G = _G(A, Om)
    Ep = _powers(E, L)
    tot = np.trace(G @ Ep[L - 2])                       # 环绕键 (L, 1)
    for i in range(1, L):                               # 键 (i, i+1), i = 1..L−1
        tot += np.trace(Ep[i - 1] @ G @ Ep[L - 1 - i])
    return complex(tot), complex(np.trace(Ep[L]))


def _open_bond_sum(A, L, Om):
    """开链: (Σ_{L−1 个键} ⟨O_{i,i+1}⟩, ⟨ψ|ψ⟩)。"""
    E = _E(A)
    G = _G(A, Om)
    vL = np.outer(BL, BL.conj()).reshape(4)
    vR = np.outer(BR, BR.conj()).reshape(4)
    Ep = _powers(E, L)
    tot = 0.0 + 0j
    for i in range(1, L):
        tot += vL @ Ep[i - 1] @ G @ Ep[L - 1 - i] @ vR
    return complex(tot), complex(vL @ Ep[L] @ vR)


def _bond_expect(A, L, Om, periodic):
    """归一后的 Σ_键⟨O⟩（两个取向统一入口）。"""
    tot, nrm = _ring_bond_sum(A, L, Om) if periodic else _open_bond_sum(A, L, Om)
    return tot / nrm


def _ring_site_rdm(A, L, site=1):
    """周期环单点约化密度矩阵 ρ_{s,s'} = ⟨ψ| |s⟩⟨s'|_site |ψ⟩ / ⟨ψ|ψ⟩。"""
    E = _E(A)
    Ep = _powers(E, L)
    nrm = np.trace(Ep[L])
    rho = np.zeros((3, 3), dtype=complex)
    for s in range(3):
        for sp in range(3):
            Qm = np.zeros((3, 3), dtype=complex)
            Qm[s, sp] = 1.0                             # |s⟩⟨s'|
            G1 = np.einsum('uv,vab,uAB->aAbB', Qm, A, A.conj()).reshape(4, 4)
            rho[s, sp] = np.trace(Ep[site - 1] @ G1 @ Ep[L - site]) / nrm
    return rho


# ══════════════════════════════════════════════════════════════════════════
# 4. 稠密自检路径（**只在 L=6 上用**，3^6 = 729；不作主路径、不作判据）
# ══════════════════════════════════════════════════════════════════════════
def _dense_state(A, L, periodic):
    """稠密态向量。周期取 trace 形式；开链取 bl/br 边界。"""
    psi = np.zeros(3 ** L, dtype=complex)
    for idx in range(3 ** L):
        x, m = idx, np.eye(2, dtype=complex)
        for _ in range(L):
            m = m @ A[x % 3]
            x //= 3
        psi[idx] = np.trace(m) if periodic else (BL @ m @ BR)
    return psi


def _dense_cut_entropy(psi, L, n):
    """稠密 SVD 版的 S(n)（本仓既有 `entanglement_curve` 的 `d` 无关改写，仅作对照）。"""
    m = psi.reshape(3 ** n, 3 ** (L - n))
    return _S_of(np.linalg.svd(m, compute_uv=False))


def _dense_bond_exp(psi, Om, L, i, periodic):
    """稠密态上的 ⟨ψ|O_{i,i+1}|ψ⟩（未除范数）。i 从 0 起；周期 i = L−1 为环绕键。"""
    O4 = Om.reshape(3, 3, 3, 3)
    if periodic and i == L - 1:
        t = np.moveaxis(psi.reshape(3 ** (L - 1), 3), 1, 0).reshape(3, 3, -1)
        w = np.einsum('uvst,stb->uvb', O4, t).reshape(-1)
    else:
        t = psi.reshape(3 ** i, 3, 3, 3 ** (L - 2 - i))
        w = np.einsum('uvst,astb->auvb', O4, t).reshape(-1)
    return complex(np.vdot(t.reshape(-1), w))


# ══════════════════════════════════════════════════════════════════════════
# 5. 阶段 2 / 阶段 3 的装置（沿用上面的截割熵与转移矩阵，**不新增算法**）
# ══════════════════════════════════════════════════════════════════════════
def _c_scaling(L_list, s_half):
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


def _state_norm2(A, L, periodic):
    """未归一的 ⟨ψ|ψ⟩：环取 tr(E^L)，开链取 v_L^T E^L v_R（与 `_bond_expect` 同一范数）。"""
    Ep = _powers(_E(A), L)
    if periodic:
        return complex(np.trace(Ep[L]))
    vL = np.outer(BL, BL.conj()).reshape(4)
    vR = np.outer(BR, BR.conj()).reshape(4)
    return complex(vL @ Ep[L] @ vR)


def _keep_only(A, k):
    """只留第 k 个物理分量的张量（阶段 3 N1 的候选注入；k = 0/1/2 ⇔ A^+/A^0/A^-）。"""
    Ak = np.zeros_like(A)
    Ak[k] = A[k]
    return Ak


def _finish():
    """两个出口（阶段 1 就失败 / 三阶段都跑完）共用的收尾打印。"""
    print('\n[T9 隔离自检] 既有 `entanglement_curve` / `fit_central_charge` / '
          '`boundary_correlation_graph` **一行未改**；')
    print('  未修 `entanglement_curve`（DP-T9-3）；未登记（阶段 4）；未进 runner（DP-T9-4）；')
    print('  未加守卫与指标 ⇒ 规模三数 78 / 27 / 13 不动。')
    print('用时: import %.1f s + 本计算 %.1f s = 总计 %.1f s'
          % (_T_IMPORT - _T_PROC, time.time() - _T0, time.time() - _T_PROC))
    print('=' * 78)


# ══════════════════════════════════════════════════════════════════════════
def main():
    print('=' * 78)
    print('v16.9 · T9 · 阶段 1 —— AKLT 精确构造 + oracle 族 O1–O6')
    print('  %s' % _VERSION_TAG)
    print('  L = %s   取向 = 周期环 + 开链' % (list(L_LIST),))
    print('  TOL_PHYS = %g   TOL_ISO = %g   TOL_ASYM = %g' % (TOL_PHYS, TOL_ISO, TOL_ASYM))
    print('  锚: v16.9_plan.md §2 阶段 1（判据事前写死）/ §1.2 / §5')
    print('=' * 78)

    A = _aklt_tensors()
    x, P2, h = _bond_ops()
    I9 = np.eye(9, dtype=complex)

    # ── 0. 张量 ──────────────────────────────────────────────────────────
    print('\n-- 0. 张量（含计划 §1.2b 的全局因子 2/sqrt3）--')
    print('  A^s 形状 = %s   s = 0,1,2 ⇔ 物理 |+1⟩,|0⟩,|−1⟩' % (A.shape,))
    print('  2/sqrt(3) = %.16f   （不补它 ⇒ sum_s A†A = (3/4)I）' % KC)
    print('  局域维 d = 3；虚维 chi = 2；L 上限 = 16（截割路径 4x3^8 = 26,244 个系数）')

    # ── S1–S3 局域算符自检（不依赖任何登记值）────────────────────────────
    print('\n-- S1–S3 局域算符自检（内部一致性；**不作物理判据**）--')
    s1 = float(np.max(np.abs(h - (2.0 * P2 - (2.0 / 3.0) * I9))))
    s2 = float(np.max(np.abs(h - (4.0 / 3.0 * P2 - (2.0 / 3.0) * I9))))
    s3a = float(np.max(np.abs(P2 @ P2 - P2)))
    s3b = float(abs(np.trace(P2).real - 5.0))
    print('  S1  h == 2·P2 − (2/3)I                残差 %.3e   %s'
          % (s1, 'OK' if s1 < _CONS else 'FAIL'))
    print('  S2  h == (4/3)·P2 − (2/3)I            残差 %.6e   ← **计划 §2 原印系数**'
          % s2)
    print('      （预测值恰为 2/3：两式只在 x = S_i·S_{i+1} = 1 的 S_tot=2 子空间上不同，'
          'AKLT 态不占该子空间）')
    print('  S3  P2 幂等 ‖P2²−P2‖ = %.3e ；tr P2 = %.6f（S_tot=2 维数 = 5）'
          % (s3a, np.trace(P2).real))
    cons = [('S1', s1, _CONS), ('S2 对 2/3', abs(s2 - 2.0 / 3.0), _CONS),
            ('S3a', s3a, _CONS), ('S3b', s3b, _CONS)]

    # ── S4 稠密交叉核对（L=6）────────────────────────────────────────────
    print('\n-- S4 稠密交叉核对（L=6，**仅自检**；主路径是截割收缩 / 转移矩阵）--')
    Ld = 6
    s4 = []
    for per in (True, False):
        tag = '周期环' if per else '开链'
        psi = _dense_state(A, Ld, per)
        nn = np.vdot(psi, psi).real
        dmax = 0.0
        for n in range(1, Ld // 2 + 1):
            dmax = max(dmax, abs(_cut_entropy(A, Ld, n, per)
                                 - _dense_cut_entropy(psi, Ld, n)))
        e_dense = sum(_dense_bond_exp(psi, h, Ld, i, per)
                      for i in range(Ld if per else Ld - 1)) / nn
        e_tr = _bond_expect(A, Ld, h, per)
        s4.append(abs(e_dense.real - e_tr.real))
        print('  %-5s S(n) 截割 vs 稠密 SVD 最大差 %.3e ；⟨H⟩ 稠密 %.12f vs 转移 %.12f'
              % (tag, dmax, e_dense.real, e_tr.real))
        s4.append(dmax)

    # ── O5 等距性（精确）─────────────────────────────────────────────────
    print('\n-- O5 等距性：max‖Σ_s A^{s†}A^s − I‖ --')
    iso = float(np.max(np.abs(np.einsum('sba,sbc->ac', A.conj(), A) - np.eye(2))))
    print('  实测 %.3e   （期望精确 0；门限 %g）' % (iso, TOL_ISO))

    # ── O1 / O1′ 两个取向 × 5 个 L ───────────────────────────────────────
    print('\n-- O1 / O1′：⟨H⟩ = −(2/3)·n_bonds 与 Σ_bonds⟨P_2⟩ = 0 --')
    print('  **第一号陷阱：⟨H⟩ ≠ 0。** 真正为零的是 Σ_bonds⟨P_2⟩。')
    o1 = []
    base_p2 = {}                       # 未注入的 Σ⟨P_2⟩ 基线（阶段 3 N2 的对照）
    for per in (True, False):
        tag = '周期环' if per else '开链'
        print('  [%s]' % tag)
        for L in L_LIST:
            nb = L if per else L - 1
            e_h = _bond_expect(A, L, h, per).real
            e_p2 = _bond_expect(A, L, P2, per).real
            base_p2[(per, L)] = e_p2
            want = -(2.0 / 3.0) * nb
            d_h = abs(e_h - want)
            o1.append(('%s L=%d ⟨H⟩' % (tag, L), d_h, TOL_PHYS))
            o1.append(('%s L=%d Σ⟨P_2⟩' % (tag, L), abs(e_p2), TOL_PHYS))
            print('    L=%2d  n_bonds=%2d  ⟨H⟩ = %+.15f   靶 %+.15f   Δ = %.3e   '
                  'Σ⟨P_2⟩ = %+.3e  %s'
                  % (L, nb, e_h, want, d_h, e_p2,
                     'OK' if (d_h < TOL_PHYS and abs(e_p2) < TOL_PHYS) else 'FAIL'))

    # ── O2 开链中点 S(L/2) = ln2 ─────────────────────────────────────────
    print('\n-- O2 开链中点 S(L/2) = ln2（精确、与 L 无关）--')
    o2 = []
    for L in L_LIST:
        s = _cut_entropy(A, L, L // 2, False)
        d = abs(s - LN2)
        o2.append(('开链 L=%d S(L/2)' % L, d, TOL_PHYS))
        print('  L=%2d  S(L/2) = %.15f   ln2 = %.15f   Δ = %.3e  %s'
              % (L, s, LN2, d, 'OK' if d < TOL_PHYS else 'FAIL'))

    # ── O3 环 n=1：ρ = I_3/3 ⇒ S = ln3 ──────────────────────────────────
    print('\n-- O3 环 n=1：ρ = I_3/3 ⇒ S = ln3（两条独立路径）--')
    o3 = []
    print('  (a) 截割收缩（L=6,8；n=1 的互补块枚举 3^(L−1)，只在小 L 上做）')
    for L in (6, 8):
        s = _cut_entropy(A, L, 1, True)
        d = abs(s - LN3)
        o3.append(('环截割 L=%d S(1)' % L, d, TOL_PHYS))
        print('      L=%2d  S(1) = %.15f   ln3 = %.15f   Δ = %.3e  %s'
              % (L, s, LN3, d, 'OK' if d < TOL_PHYS else 'FAIL'))
    print('  (b) 单点 RDM 转移矩阵（L ∈ {8..16}，并直接核对 ρ = I_3/3）')
    for L in L_LIST:
        rho = _ring_site_rdm(A, L)
        dev = float(np.max(np.abs(rho - np.eye(3) / 3.0)))
        ev = np.linalg.eigvalsh(rho).real
        ev = ev[ev > P_THRESH]
        s = float(-np.sum(ev * np.log(ev)))
        d = abs(s - LN3)
        o3.append(('环RDM L=%d ρ 偏离 I/3' % L, dev, TOL_PHYS))
        o3.append(('环RDM L=%d S(1)' % L, d, TOL_PHYS))
        print('      L=%2d  max|ρ − I/3| = %.3e   S(1) = %.15f   Δ = %.3e  %s'
              % (L, dev, s, d, 'OK' if (dev < TOL_PHYS and d < TOL_PHYS) else 'FAIL'))

    # ── O6 环 S(L/2) 单调升 → ln4，O4 环块谱 1+3 简并 ────────────────────
    print('\n-- O6 环 S(L/2) 渐近 → ln4（单调升）；O4 环块谱 SU(2) 1+3 简并 --')
    seq = {L: _cut_entropy(A, L, L // 2, True) for L in L_LIST}
    o6 = []
    mono = all(seq[L_LIST[i]] < seq[L_LIST[i + 1]] for i in range(len(L_LIST) - 1))
    o6.append(('O6 单调升', 0.0 if mono else 1.0, 0.5))
    d16 = abs(seq[16] - LN4)
    o6.append(('O6 |S(16) − ln4|', d16, TOL_ASYM))
    for L in L_LIST:
        print('  L=%2d  S(L/2) = %.12f   (ln4 − S = %.3e)' % (L, seq[L], LN4 - seq[L]))
    print('  单调升 = %s ；|S(L=16) − ln4| = %.3e  （门限 %g，**渐近非精确**）'
          % (mono, d16, TOL_ASYM))
    print('  相邻差 dS = %s   （每加 2 格收缩约 9x ⇒ 指数收敛，'
          'xi ~ 1/ln3 ~ 0.91 格）' % (np.round(np.diff([seq[L] for L in L_LIST]), 12),))

    o4 = []
    for L in L_LIST:
        sv = _schmidt_from_cut(_block_tensors(A, L // 2),
                               _block_tensors(A, L - L // 2), True)
        p = sv ** 2
        p = p / p.sum()
        r, tag4 = _triple_resid(p)
        o4.append(('O4 L=%d 1+3 简并残差' % L, float(r), TOL_PHYS))
        print('  L=%2d  p = %s   三重简并残差 = %.3e  [%s]  %s'
              % (L, np.round(p, 10), r, tag4, 'OK' if r < TOL_PHYS else 'FAIL'))
    print('  判据只认「四值里恰有三个相等（残差 < %g）」；**哪一支在上不入判据**，'
          '但照实标签。' % TOL_PHYS)

    # ── 判据汇总 ────────────────────────────────────────────────────────
    print('\n-- 判据汇总（O1–O6；逐条列出不过项）--')
    allc = ([('O5 等距性', iso, TOL_ISO)]
            + [('S4/%s' % s4[i], s4[i + 1], _CONS) for i in range(0, len(s4), 2)]
            + [(n, v, t) for n, v, t in cons]
            + o1 + o2 + o3 + o4 + o6)
    bad = [(n, v, t) for n, v, t in allc if not (v < t)]
    print('  共 %d 条子判据；不过 %d 条。' % (len(allc), len(bad)))
    for n, v, t in bad:
        print('    FAIL  %-28s 实测 %.6e  门限 %.1e' % (n, v, t))

    ok1 = not bad
    b1 = 'T9-构造成立' if ok1 else 'T9-构造失败'
    print('  ⇒ 阶段 1 = **%s**' % b1)
    if ok1:
        print('  O1–O6 全过 ⇒ AKLT 的精确 chi=2 MPS 构造成立，六把解析尺子全部对上。')

    # ── 阶段 1 失败 ⇒ 阶段 2/3 的门控未满足，**不跑** ────────────────────
    if not ok1:
        print('\n-- 落支 --')
        print('  落支 = **T9-构造失败**')
        print('  未过项见上表。按计划 §2 阶段 1：**停工上报，不改判据、不改容差**（§4 红线 2）。')
        print('  阶段 2 / 阶段 3 的门控是「阶段 1 落 T9-构造成立」⇒ 本文件**不跑它们**。')
        _finish()
        return 3

    # ══════════════════════════════════════════════════════════════════════
    # 阶段 2 · 接代码台账：L2 面积律判别器（**头号产物**）
    # ══════════════════════════════════════════════════════════════════════
    print('\n' + '=' * 78)
    print('阶段 2 · 接代码台账：L2 面积律判别器 —— 照 `_model/checks.py:388-396` 拟合')
    print('=' * 78)
    seq_open = [_cut_entropy(A, L, L // 2, False) for L in L_LIST]
    seq_ring = [seq[L] for L in L_LIST]

    print('\n-- [P2-1 / P2-2] S(L/2) = (c/3)·ln L + b  的 c_scaling --')
    c_o, b_o, r2_o, _ = _c_scaling(L_LIST, seq_open)
    c_r, b_r, r2_r, _ = _c_scaling(L_LIST, seq_ring)
    p2 = []
    for tag, c, b, r2, y in (('开链', c_o, b_o, r2_o, seq_open),
                             ('环  ', c_r, b_r, r2_r, seq_ring)):
        nm = tag.strip()
        p2.append(('P2 %s |c_scaling|' % nm, abs(c), C_GAPPED))
        print('  %-4s S(L/2) = %s' % (nm, np.round(y, 15)))
        print('       c_scaling = %+.7f   b = %+.7f   R^2 = %s   |c| < %g ? %s'
              % (c, b, ('%.4f' % r2) if np.isfinite(r2) else 'nan', C_GAPPED,
                 'OK' if abs(c) < C_GAPPED else 'FAIL'))
    print('  **R^2 不作判据**（计划 §2 阶段 2：零方差下不可读）—— 照实打印，不解读。')
    print('  对照 `checks.py:338-340` 已登记的 Ising 值（同一条 `gapped` 判据的邻居）:')
    print('    h/J=0.5 -> c = 0.0007   ;   h/J=1.0 -> c = 0.4976   ;   h/J=2.0 -> c = -0.0050')

    print('\n-- [P2-3] 显式调用既有 `_metric.solvers.entanglement_curve`（d=3 输入）--')
    print('  既有实现 `v16/_metric/solvers.py:51-62` 里写着 `m = psi.reshape(2 ** n, 2 ** (L - n))`')
    ec_raised = False
    if _EC_IMPORT_ERR is not None:                              # pragma: no cover
        print('  **import 失败** %s: %s' % (type(_EC_IMPORT_ERR).__name__, _EC_IMPORT_ERR))
    else:
        psi8 = _dense_state(A, 8, False)
        print('  输入: AKLT **开链** L=8 的稠密态，长度 3^8 = %d（d = 3）' % psi8.size)
        try:
            out = _entanglement_curve(psi8, 8)
            print('  **未抛错** —— 返回 shape %s ⇒ 那个函数其实吃得下 d=3' % (np.shape(out),))
        except Exception as e:
            ec_raised = True
            print('  抛 %s: %s' % (type(e).__name__, e))
            print('  => `2**n` 硬编码确认 ⇒ 计划 §1.4a 的锚定成立。')
            print('     **这个失败是产物，不是待修的 bug**（计划 §4 红线 4 / DP-T9-3）。')
    p2.append(('P2 entanglement_curve 对 d=3 抛错', 0.0 if ec_raised else 1.0, 0.5))

    bad2 = [(n, v, t) for n, v, t in p2 if not (v < t)]
    if not ec_raised:
        b2 = 'T9-硬编码不成立'
    elif bad2:
        b2 = 'T9-L2假临界'
    else:
        b2 = 'T9-L2正确判gapped'
    print('\n-- 阶段 2 落支 --')
    print('  共 %d 条子判据；不过 %d 条。' % (len(p2), len(bad2)))
    for n, v, t in bad2:
        print('    FAIL  %-34s 实测 %.6e  门限 %.1e' % (n, v, t))
    print('  落支 = **%s**' % b2)
    if b2 == 'T9-L2正确判gapped':
        print('  ⇒ AKLT 被仓库**自己那条非临界判据**（`checks.py:345`）**正确判为 gapped**。')
        print('     **不是「此层不适用」，是「适用且答对」** —— 答案是事前已知的（面积律，c = 0）。')

    # ══════════════════════════════════════════════════════════════════════
    # 阶段 3 · 负对照三条 N1–N3
    # ══════════════════════════════════════════════════════════════════════
    print('\n' + '=' * 78)
    print('阶段 3 · 负对照三条 N1–N3（χ 下界 / 非酉注入 / 等距性双向配对）')
    print('=' * 78)
    bad3, _n3 = [], [0]

    def _judge(name, ok, detail):
        _n3[0] += 1
        print('    %-34s %s   %s' % (name, 'OK  ' if ok else 'FAIL', detail))
        if not ok:
            bad3.append(name)

    # ── N1 键维向下（χ:2→1 积态）────────────────────────────────────────
    print('\n-- N1 键维向下（χ:2→1 积态）--')
    print('  计划 §2 的注入列逐字是「A^0, A^- 置零，只留 A^+」。**先照字面把三个候选都算一遍**：')
    print('      候选      ‖ψ‖²(环 L=8)   ‖ψ‖²(开链 L=8)   Schmidt 谱(环 L=8, n=4)')
    for k, nm in ((0, 'A^+'), (1, 'A^0'), (2, 'A^-')):
        Ak = _keep_only(A, k)
        sv = np.asarray(_schmidt_from_cut(_block_tensors(Ak, 4), _block_tensors(Ak, 4), True))
        print('      只留 %s   %+.3e        %+.3e        %s'
              % (nm, _state_norm2(Ak, 8, True).real, _state_norm2(Ak, 8, False).real,
                 np.round(sv, 8)))
    print('  => `A^± = (2/sqrt3)·P^±·M` 是**幂零秩一** ⇒ `(A^±)^L = 0`（L >= 2）：')
    print('     「只留 A^+」与「只留 A^-」都给出**零向量**（‖ψ‖² 精确 0），熵**无定义**。')
    print('     **只有 `A^0` 给出非零态，且只在环上**：环 ‖ψ‖² = 6.097e-04、谱 = 单支 ⇒ 真积态（rank 1）。')
    print('     开链的边界是单态对 (bl, br) = (|↑⟩, |↓⟩)，而 `A^0 = (1/sqrt3)·diag(-1, 1)` 是**对角**的，')
    print('     `bl` 只走虚拟分量 0、`br` 只在分量 1 ⇒ **无重叠** ⇒ 开链恒为零向量。')
    print('     （探针实测：只留任何**单一个**张量，开链 ‖ψ‖² 都是 0；只有把 br 也改成 |↑⟩ 才非零。）')
    print('     ⇒ **N1 只落在环上**：开链一格**结构性不可实现**（「无对象」，不是「判据不过」）；')
    print('       并加一条**有效性前置** ‖ψ‖² > 0，使零向量**不可能**再被静默判过。')
    print('       此为**并存订正**，已照实记入计划 §0.1.4；计划 §2 原文**一字未改**。')
    A1 = _keep_only(A, 1)
    for L in L_LIST:
        f2 = _state_norm2(A1, L, True).real          # 有效性前置：零向量的熵无定义
        s = _cut_entropy(A1, L, L // 2, True)
        base = seq[L]
        sep = abs(s - base)
        _judge('N1 环 L=%2d' % L,
               (f2 > 0.0) and (s < TOL_PHYS) and (sep > N1_SEP),
               '‖ψ‖² = %.3e (>0) ; S(L/2) = %.3e (<1e-12) ; 与环基线 %.6f 的差 = %.6f (>0.5)'
               % (f2, s, base, sep))

    # ── N2 非酉注入 ─────────────────────────────────────────────────────
    print('\n-- N2 非酉注入 B[1] += %g·|↓⟩⟨↑| --' % N2_EPS)
    B = A.copy()
    B[1] = B[1] + N2_EPS * np.outer(DOWN, UP)
    iso_n2 = float(np.max(np.abs(np.einsum('sba,sbc->ac', B.conj(), B) - np.eye(2))))
    _judge('N2 等距残差 > 1e-3', iso_n2 > TOL_N2_ISO,
           'max‖Σ_s B^{s†}B^s − I‖ = %.6e' % iso_n2)
    for per in (True, False):
        tg = '环  ' if per else '开链'
        for L in L_LIST:
            base = base_p2[(per, L)]
            e_p2 = _bond_expect(B, L, P2, per).real
            _judge('N2 %s L=%2d' % (tg, L), (abs(base) < TOL_PHYS) and (e_p2 > TOL_N2_P2),
                   'Σ⟨P_2⟩ %+.3e → %+.6e  (基线 <1e-12 ; 注入后 >1e-4)' % (base, e_p2))

    # ── N3 等距性双向配对 ───────────────────────────────────────────────
    print('\n-- N3 等距性**双向**配对（O5 正对照 + N2 负对照）--')
    print('    正对照 O5 残差 = %.3e   （门限 %g，取自阶段 1）' % (iso, TOL_ISO))
    print('    负对照 N2 残差 = %.6e （门限 %g）' % (iso_n2, TOL_N2_ISO))
    _judge('N3 两条须同时成立', (iso < TOL_ISO) and (iso_n2 > TOL_N2_ISO),
           '⇒ 检查器既非橡皮图章（负对照真报错）、也非一律报错（正对照真归零）')

    b3 = 'T9-负对照三条齐' if not bad3 else 'T9-负对照缺陷'
    print('\n-- 阶段 3 落支 --')
    print('  共 %d 条子判据；不过 %d 条。' % (_n3[0], len(bad3)))
    for n in bad3:
        print('    FAIL  %s' % n)
    print('  落支 = **%s**' % b3)

    # ── 落支 ───────────────────────────────────────────────────────────
    ok_all = (b2 == 'T9-L2正确判gapped') and (b3 == 'T9-负对照三条齐')
    print('\n' + '=' * 78)
    print('-- 落支 --')
    print('  阶段 1 = **%s**' % b1)
    print('  阶段 2 = **%s**' % b2)
    print('  阶段 3 = **%s**' % b3)
    print('  EXIT = %d（0 = 三支全落正支）' % (0 if ok_all else 3))
    _finish()
    return 0 if ok_all else 3


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.exit(main())
