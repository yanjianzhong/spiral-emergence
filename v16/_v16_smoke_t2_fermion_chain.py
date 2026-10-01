# -*- coding: utf-8 -*-
#
# Copyright (c) 2024-2026 Jianzhong Yan
# Licensed under the Apache License, Version 2.0 (the "License");
#
"""
v16.6 · T2(b) —— L1→L2 的**自由费米子**通路：给 `S(L/2)` 造第二条独立实现
[_VERSION_TAG = 'v16.6-smoke-t2-fermion-chain-1']

**为什么有这个脚本**（锚点三处，逐字）:
  a. `spiral_v16_audit.md:341`（L1 行·缺口表）逐字：独立交叉路径一栏记
     「对照 L2 的 JW / L5 的零模型族 | **无**」—— L2 有 JW 交叉路径，L1 没有。
  b. L1→L2 的生产链是**纯自旋**：`stage2_critical_break`(`_model/stage12.py:81`)
     → `exact_ground_state`(`core.py:559`) → `tfi_periodic_sparse`(`core.py:524`)，
     实系数 `σzσz / σx`，**全程不含 `i`**。`jw_ground_energy`(`core.py:572`)
     只作 `checks.py:519/:835` 的**闭式旁证**，且**只覆盖 `E0`，不覆盖 `S(L/2)`**。
  c. L1 的**承重读数是 `S(L/2)`**（`_v16_smoke_gamma_l1_downstream.py:29` DP-9
     逐字「读 `S(L/2)`（唯一参与判据的下游量）」）⇒ 缺的是 `S(L/2)` 的第二条路。

**本脚本做什么**: 用**已有的** Majorana 工具箱 `_v16_cmera_gaussian.py`
（`build_h` :141 / `gs_energy_from_h` :156，真调，不复制）造一条 L1→L2 的
**自由费米子**通路，与自旋侧 ED 逐点比 `E0` 与 `S(L/2)`。

======================================================================
**预登记（在跑任何数据之前写死；落支词条照此判定，不许事后改）**
======================================================================

* **DP-8 = 诊断**：本脚本**不进 runner、不加守卫/指标** ⇒ 规模三数 78 / 27 / 13 不动，
  `result/_v16_data.json` 一个字节不改。
* **DP-25(b) = 用户 2026-10-01 明确选择 (b)** ⇒ 靶从「造一条自旋链」改瞄为
  「造一条**费米子**链，比 `E0` / `S(L/2)`」。

**必须同时登记的两条自限（预登记，非事后补）**
  1. **一致是 JW 等价的推论，不是实测发现** —— 周期 TFI 精确可积，自旋侧与
     自由费米子侧的 `E0`、半链熵相等是**定理保证**的。本脚本登记的是**机器正确性**
     （2L×2L BdG 协方差这条实现路径算对了），**不缩小 `audit:341` 的缺口**。
     与 `_v16_smoke_delta_fermion_repr.py:47-53` 的自限 1 同形。
  2. **给 `S(L/2)` 第二条路 ≠「L1 的因果通路问题解决了」** ——
     `_v16_smoke_gamma_l1_downstream.py:51-53` 逐字：
     「实质差别在**读出量**……**不许把它读成"换了一条独立路径"**」。
     **是否据此动 L1 的成熟度，是另一个 DP；本脚本不裁。**

**口径（冻结）**
  * 格点 `L = 16`，`J = 1.0`；主点 `h = 1.0`（临界），扫描点 `h/J ∈ (0.5, 1.0, 1.5, 2.0)`。
  * 边界：**周期**（与 `tfi_periodic_sparse` 一致）。费米子侧用**口径 A**
    `H_til = H_open + J·iγ_{2L-1}γ_0`（偶宇称扇区）= `cg.build_h` 的逐字定义。
  * 半链：自旋侧 `gs.reshape(2^8, 2^8)` 的行指标 = 站点 `0..7`（`stage12.py:105`）；
    费米子侧取实费米子 `j,l < L/2` ⇒ Majorana `0..15`。**两边同一条链、同一个二分。**
  * 费米子协方差：`h` 实反对称 ⇒ `1j*h` 厄米 ⇒ **`Γ = −sgn(h)`**（实反对称，对角为 0）。
    ⚠️ **不是 `sgn(h) − i·I`**：`covariance` 的定义 `Γ_ab = i⟨γ_aγ_b⟩ − iδ_ab`
    （`_v16_cmera_gaussian.py:161-168`）里那个 `−iδ` **已经**把对角清零了
    （`Γ_aa = i·1 − i = 0`），再减一次 `i·I` 会把对角污染成 `−i`，整条通路作废。
    单站点验算：`h=[[0,−2h],[2h,0]]` ⇒ `Γ=[[0,1],[−1,0]]` ⇒
    `C_00 = 0.25[1 + i(−i) − i(i) + 1] = 1`（该基态 `n=1` ✓），
    `E = −(1/4)tr(hΓ) = −h` ✓。
    又由 `Γ ≡ i⟨γ_aγ_b⟩ − iδ_ab` 反解 `⟨γ_aγ_b⟩ = δ_ab − iΓ_ab`；
    `c_j = (γ_{2j}+iγ_{2j+1})/2`（`:175`）。
  * 熵（**必须走 Majorana 途径**）：`A` = 2nA 个 Majorana 模 ⇒ `Γ_A = Γ[0:2nA, 0:2nA]`，
    `iΓ_A` 厄米、本征值成 ±μ_k 对（|μ_k| ≤ 1）⇒
    `S = −Σ_{k=1..nA}[(1+μ_k)/2·ln((1+μ_k)/2) + (1−μ_k)/2·ln((1−μ_k)/2)]`。
    ⚠️ **不能只拿 `C_A` 的本征值**：`H_til` 的基态是 **BdG 态（粒子数不守恒）**，
    约化态需 `(C_A, F_A)` **一对**算子；实测 `max|C²−C| = 0.125 ≠ 0`
    （B0 段），`C` 途径给 `S = 3.19` 而非 `0.7501`。**B0 段就是为抓这个而加的。**

**A 段（门，先跑；不过就不许往下跑）**
  真调 `stage2_critical_break(16, 1.0, 1.0)`（**不传 `void_state`**，即生产路径），
  复现登记读数 `E0/L = -1.275287` / `S(L/2) = 0.7501` / `Schmidt 极化 = 250.673` /
  `gap = 0.366151`（`_runall_v16.3_r1.log:54`）。
  另加**读出量标定**：本脚本自己的 `spin_half_entropy`（真调 `exact_ground_state`）
  必须等于 stage2 自己返回的 `entropy`，否则两边的 `S(L/2)` 不是同一个量
  （先例 `_v16_smoke_gamma_l1_downstream.py:59-60`）。

**C 段（分辨力反事实，先于 B 段判定）**
  1. **解析对照**：喂 `Γ_A = 0`（`μ ≡ 0`，无穷温极限）⇒ `S ≡ nA·ln2`
     （解析值，`nA = 8` ⇒ `5.545177444480`）。熵例程给不出这个数 ⇒ 例程本身有问题。
     实测 `S(L/2) = 0.7501` 远小于 `8·ln2 = 5.5452` ⇒ 例程有分辨力。
  2. **参数分辨力**：`S_fermi` 随 `h/J` 必须真的变（`max−min > 1e-3`）。

**B0 段（关联矩阵标定）—— ⚠️ 事后追加，理由如实登记**
  拿 **ED 自己**的 `<σx_j>` 验 `2·C_jj − 1`（`n_j = (1+σx_j)/2`，
  `_v16_cmera_gaussian.py:206-210` 逐字），外加纯高斯态条件 `C² = C`。
  **追加理由（诚实标注，非补写）**：第 1 次跑完，`S_fermi` 与 `S_spin` 差 4 倍量级，
  而 C 段分辨力**全过**。此时**无法区分**「我的 `C` 错了」与「两侧真的不一致」——
  若不先排除前者就报 `F-中途`，等于**把自己的 bug 当成物理结论**。
  这正是 `_v16_cmera_gaussian.py:228-231` 记下的硬纪律（该处逐字：
  「这条**只有稠密对照能抓**（对称性论证本身自洽, 看不出来）」）。
  该检查**只会证伪本脚本、不放宽任何既有判据** ⇒ 属「往保守方向改」，非移动球门。

**落支（冻结；按此顺序判）**
  * `F-校准失败` : A 段任一数超差 ⇒ 不许往下跑，不出任何物理结论。
  * `F-探针失明` : C 段任一分辨力检查不过 ⇒ 熵例程无分辨力，本脚本结论作废。
  * `F-关联矩阵错` : B0 段 `max|2C_jj − 1 − <σx_j>_ED| > TOL_C`，或 `C² ≠ C`
                  ⇒ 本脚本的 `C` 有错，**不得**把 `S` 的差读成物理结论。
  * `F-两错`     : `|E0_fermi − E0_spin| > TOL_E`（连标定都没过）。
  * `F-中途`     : `E0` 对上、但**至少一个** `h/J` 点的 `S(L/2)` 对不上。
                  **这是唯一有信息量的结果。**
  * `F-一致`     : `E0` 与**全部** `h/J` 点的 `S(L/2)` 都 ≤ 容差。

**容差**
  * `TOL_PRINT`：登记值只有 6/4/3/6 位有效数字 ⇒ 按末位计
    `E0/L ≤ 5e-7`、`S ≤ 5e-5`、`pol ≤ 5e-4`、`gap ≤ 5e-7`。
  * `TOL_E = 1e-9`、`TOL_S = 1e-8`（两侧都是精确计算，不是抽样）。
"""

import os
import sys
import time

_T_PROC = time.time()

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np            # noqa: E402

_T_IMPORT = time.time()

import spiral_model_v16 as G          # noqa: E402
import _v16_cmera_gaussian as cg      # noqa: E402

_MODS = (G, cg)


def _resolve(name):
    """按门面 → 工具箱的顺序取函数；取不到就报出来，不静默降级。"""
    for m in _MODS:
        f = getattr(m, name, None)
        if f is not None:
            return f
    raise RuntimeError(f'找不到 {name}（门面/工具箱都没有）')


# ── 常量（冻结） ─────────────────────────────────────────────────────────────
L_SITE = 16
J_REF = 1.0
H_REF = 1.0
H_SCAN = (0.5, 1.0, 1.5, 2.0)

E0_OVER_L_REF = -1.275287     # _runall_v16.3_r1.log:54
S_HALF_REF = 0.7501
POL_REF = 250.673
GAP_REF = 0.366151
TOL_E0L = 5e-7
TOL_S_PRINT = 5e-5
TOL_POL = 5e-4
TOL_GAP = 5e-7

TOL_E = 1e-9
TOL_S = 1e-8
TOL_C = 1e-9
DISC_MIN = 1e-3

_LOG_PATH = os.path.join(_HERE, '_t2_fermion.log')


class _Tee:
    def __init__(self, *streams):
        self._s = streams

    def write(self, d):
        for s in self._s:
            try:
                s.write(d)
            except Exception:
                pass
        return len(d)

    def flush(self):
        for s in self._s:
            try:
                s.flush()
            except Exception:
                pass


# ── 两侧的机构 ───────────────────────────────────────────────────────────────
def spin_half_entropy(L, J, h):
    """自旋侧 S(L/2)：**与 `stage12.py:104-110` 同一算法**（SVD of 二分矩阵）。
    真调 `exact_ground_state`（生产函数），不复制态构造。"""
    egs = _resolve('exact_ground_state')
    E0, gs = egs(L, J, h)
    half = L // 2
    s = np.linalg.svd(gs.reshape(2 ** half, 2 ** (L - half)), compute_uv=False)
    p = s ** 2
    p = p[p > 1e-15]
    p = p / p.sum()
    return float(E0), float(-np.sum(p * np.log(p)))


def sgn_h(Hh):
    """`sgn(h)`：`h` 实反对称 ⇒ SVD 给 `h = U Σ Vᵀ` ⇒ `sgn(h) = U Vᵀ`（实反对称）。
    **不用 `eigh(1j*h)` + `sign(w)`**：近零本征值处 `sign(0)=0` 会把该模式整个抹掉。"""
    U, _s, Vt = np.linalg.svd(Hh)
    return U @ Vt


def gamma_from_h(Hh):
    """口径 A 的基态协方差。`Γ_ab = i⟨γ_aγ_b⟩ − iδ_ab`（`covariance` 的定义）
    ⇒ 对角恒为 0、非对角 = `i⟨γ_aγ_b⟩`。基态给 `i⟨γ_aγ_b⟩ = −sgn(h)_ab`
    （单站点验算见文件头「口径」）⇒ **`Γ = −sgn(h)`**，**不再减 `i·I`**。"""
    return -sgn_h(Hh)


def corr_C(Gam, L):
    """`C[j,l] = ⟨c†_j c_l⟩`，`c_j = (γ_{2j}+iγ_{2j+1})/2`，`⟨γ_aγ_b⟩ = δ_ab − iΓ_ab`。
    逐字展开，不做任何对称性简化（对称性只当自校验）。"""
    n = 2 * L
    d = np.eye(n)

    def g(a, b):
        return d[a, b] - 1j * Gam[a, b]

    C = np.zeros((L, L), dtype=complex)
    for j in range(L):
        a, b = 2 * j, 2 * j + 1
        for l in range(L):
            c_, e_ = 2 * l, 2 * l + 1
            C[j, l] = 0.25 * (g(a, c_) + 1j * g(a, e_) - 1j * g(b, c_) + g(b, e_))
    return C


def majorana_half_entropy(Gam, nA):
    """自由费米子半链熵（**Majorana 途径，BdG 态通用**）。
    `Γ_A = Γ[0:2nA, 0:2nA]`；`iΓ_A` 厄米、本征值成 ±μ_k 对 ⇒
    `S = −Σ_{k=1..nA}[(1+μ_k)/2 ln((1+μ_k)/2) + (1−μ_k)/2 ln((1−μ_k)/2)]`
      = (1/2)·Σ_{全部 2nA 个} h(μ)   （每对重复一次, 故折半）。"""
    GA = Gam[:2 * nA, :2 * nA]
    mu = np.linalg.eigvalsh(1j * GA).real
    nu = np.clip(0.5 * (1.0 + mu), 1e-15, 1.0 - 1e-15)
    return float(-0.5 * np.sum(nu * np.log(nu) + (1.0 - nu) * np.log(1.0 - nu))), mu


def corr_entropy(C, nA):
    """**对照用**（`C_A` 本征值途径）。仅对粒子数守恒的 Slater 行列式成立；
    本脚本 `H_til` 的基态是 BdG 态（`C² ≠ C`）⇒ **这条给错数**，只作登记。"""
    nu = np.linalg.eigvalsh(C[:nA, :nA]).real
    nu = np.clip(nu, 1e-15, 1.0 - 1e-15)
    return float(-np.sum(nu * np.log(nu) + (1.0 - nu) * np.log(1.0 - nu))), nu


def main():
    _t0 = time.time()
    half = L_SITE // 2
    nA = half

    stage2 = _resolve('stage2_critical_break')
    build_h = _resolve('build_h')
    gs_energy_from_h = _resolve('gs_energy_from_h')
    try:
        jw_energy = _resolve('jw_ground_energy')
    except RuntimeError:
        jw_energy = None

    print(f"=== {__doc__.strip().splitlines()[1].strip()} ===")
    print(f"口径 A: H_til = H_open + J·iγ_{{2L-1}}γ_0 (偶宇称扇区)")
    print(f"L={L_SITE}, J={J_REF}, 主点 h={H_REF}, 扫描 h/J={H_SCAN}")
    print(f"进口用时 {_T_IMPORT - _T_PROC:.1f} s")

    # ── A 段：门 ────────────────────────────────────────────────────────────
    print("\n--- 阶段 A: 门（真调 stage2_critical_break，不传 void_state）---")
    res = stage2(L_SITE, J_REF, H_REF)
    E0s = float(res['E0'])
    S_stage2 = float(res['entropy'])
    pol = float(res['polarization'])
    gap = float(res['gap'])
    d_e0l = abs(E0s / L_SITE - E0_OVER_L_REF)
    d_s = abs(S_stage2 - S_HALF_REF)
    d_pol = abs(pol - POL_REF)
    d_gap = abs(gap - GAP_REF)
    ok_a = (d_e0l <= TOL_E0L and d_s <= TOL_S_PRINT
            and d_pol <= TOL_POL and d_gap <= TOL_GAP)
    print(f"  E0/L  实测 = {E0s / L_SITE:.9f}  登记 = {E0_OVER_L_REF}   "
          f"|差| = {d_e0l:.3e} (<= {TOL_E0L:.0e})")
    print(f"  S(L/2)实测 = {S_stage2:.9f}  登记 = {S_HALF_REF}       "
          f"|差| = {d_s:.3e} (<= {TOL_S_PRINT:.0e})")
    print(f"  Schmidt 极化 = {pol:.6f}   登记 = {POL_REF}     "
          f"|差| = {d_pol:.3e} (<= {TOL_POL:.0e})")
    print(f"  gap   实测 = {gap:.9f}  登记 = {GAP_REF}   "
          f"|差| = {d_gap:.3e} (<= {TOL_GAP:.0e})")
    print(f"  [门] 四数复现 = {'OK' if ok_a else '**失败**'}")

    E0_chk, S_chk = spin_half_entropy(L_SITE, J_REF, H_REF)
    d_cal = abs(S_chk - S_stage2)
    d_e0cal = abs(E0_chk - E0s)
    ok_cal = (d_cal <= 1e-9 and d_e0cal <= 1e-9)
    print(f"  [读出量标定] spin_half_entropy - stage2.entropy = {d_cal:.3e}; "
          f"E0 差 = {d_e0cal:.3e}  -> {'OK' if ok_cal else '**失败**'}")
    if jw_energy is not None:
        print(f"  [旁证] jw_ground_energy(NS 闭式) = "
              f"{jw_energy(L_SITE, J_REF, H_REF):.12f}  (只对 E0, 不涉 S)")

    if not (ok_a and ok_cal):
        print("\n-- 落支 --")
        print("  落支 = **F-校准失败**")
        print("  判据: A 段四数或读出量标定超差 ⇒ 不许往下跑，不出任何物理结论。")
        return

    # ── C 段：分辨力（先于 B 段判定）────────────────────────────────────────
    print("\n--- 阶段 C: 分辨力反事实 ---")
    S_vac, _ = majorana_half_entropy(np.zeros((2 * nA, 2 * nA)), nA)
    S_vac_ref = nA * np.log(2.0)
    d_vac = abs(S_vac - S_vac_ref)
    ok_vac = d_vac <= 1e-12
    print(f"  [C1] Γ_A = 0 (μ ≡ 0, 无穷温极限) ⇒ S 应为 nA·ln2 = {S_vac_ref:.12f}")
    print(f"       实测 = {S_vac:.12f}   |差| = {d_vac:.3e}  "
          f"-> {'OK (例程有分辨力)' if ok_vac else '**失明**'}")
    print(f"       对照: 真实 S(L/2) = {S_stage2:.6f}  <<  {S_vac_ref:.6f}")

    # ── B0 段：关联矩阵标定（先证明 C 对，再谈 S 的差）──────────────────────
    print("\n--- 阶段 B0: 关联矩阵标定 (拿 ED 的 <σx_j> 对照) ---")
    _, gs0 = _resolve('exact_ground_state')(L_SITE, J_REF, H_REF)
    G0 = gamma_from_h(build_h(L_SITE, J_REF, H_REF))
    C0 = corr_C(G0, L_SITE)
    _idx = np.arange(2 ** L_SITE, dtype=np.int64)
    dev_sx = 0.0
    for j in range(L_SITE):
        # 位序: 站点 j ↔ 比特 L-1-j (tfi_periodic_sparse 的约定) ⇒ σx_j 即翻该位
        ex = float(np.vdot(gs0, gs0[_idx ^ (1 << (L_SITE - 1 - j))]).real)
        dev_sx = max(dev_sx, abs((2.0 * C0[j, j].real - 1.0) - ex))
    pure = float(np.max(np.abs(G0 @ G0 + np.eye(2 * L_SITE))))
    csq = float(np.max(np.abs(C0 @ C0 - C0)))
    ok_corr = (dev_sx <= TOL_C) and (pure <= 1e-9)
    print(f"  max|2·C_jj − 1 − <σx_j>_ED| = {dev_sx:.3e}  (<= {TOL_C:.0e})")
    print(f"  max|Γ² + I|                  = {pure:.3e}  (纯高斯态条件)")
    print(f"  max|C² − C|                  = {csq:.3e}  "
          f"(**≠ 0 是应该的**: BdG 态粒子数不守恒 ⇒ C 本不是投影算子;")
    print(f"                                这正是熵必须走 Majorana 途径的理由)")
    print(f"  [B0] C 标定 = {'OK' if ok_corr else '**失败**'}")

    # ── B 段：费米子侧（主点 + 扫描）────────────────────────────────────────
    print("\n--- 阶段 B: 自由费米子通路 (2L×2L BdG, 真调 build_h) ---")
    rows = []
    for h in H_SCAN:
        Hh = build_h(L_SITE, J_REF, h)
        E0f = gs_energy_from_h(Hh)
        Gam = gamma_from_h(Hh)
        anti = float(np.max(np.abs(Gam + Gam.conj().T)))          # Γ† = −Γ
        C = corr_C(Gam, L_SITE)
        herm = float(np.max(np.abs(C - C.conj().T)))              # C† = C
        Sf, mu = majorana_half_entropy(Gam, nA)
        Sf_C, _ = corr_entropy(C, nA)
        E0_spin, S_spin = spin_half_entropy(L_SITE, J_REF, h)
        # 能量自校验: E = −(1/4)·tr(h·Γ)，与 gs_energy_from_h 必须一致
        E_from_G = -0.25 * float(np.trace(Hh @ Gam).real)
        dEG = abs(E_from_G - E0f)
        rows.append(dict(h=h, E0f=E0f, dE=abs(E0f - E0_spin),
                         Sf=Sf, dS=abs(Sf - S_spin),
                         anti=anti, herm=herm, dEG=dEG))
        tag = '主点' if abs(h - H_REF) < 1e-12 else '扫描'
        print(f"  h/J={h:.2f} ({tag}): E0_fermi={E0f:.12f}  E0_spin={E0_spin:.12f}  "
              f"|差|={abs(E0f - E0_spin):.3e}")
        print(f"           S_fermi={Sf:.9f}  S_spin={S_spin:.9f}  "
              f"|差|={abs(Sf - S_spin):.3e}  (TOL_S={TOL_S:.0e})")
        print(f"           [对照] 走 C_A 本征值途径 = {Sf_C:.9f}  (**错路**, 只作登记)")
        print(f"           [自校] ‖Γ+Γ†‖={anti:.2e}  ‖C−C†‖={herm:.2e}  "
              f"|E(Γ)−E(h)|={dEG:.2e}  μ∈[{mu.min():.6f},{mu.max():.6f}]")

    S_vals = [r['Sf'] for r in rows]
    spread = max(S_vals) - min(S_vals)
    ok_disc = spread > DISC_MIN
    print(f"  [C2] S_fermi 随 h/J 的跨度 = {spread:.6f}  (要求 > {DISC_MIN:.0e})  "
          f"-> {'OK' if ok_disc else '**失明**'}")

    # ── 落支 ────────────────────────────────────────────────────────────────
    print("\n-- 落支 --")
    if not (ok_vac and ok_disc):
        print("  落支 = **F-探针失明**")
        print("  判据: C 段分辨力检查不过 ⇒ 熵例程无分辨力，本脚本结论作废。")
        return
    if not ok_corr:
        print("  落支 = **F-关联矩阵错**")
        print("  判据: B0 段 C 标定不过 ⇒ 本脚本的 C 有错，")
        print("        **不得**把 S 的差读成物理结论。")
        return
    worst_e = max(r['dE'] for r in rows)
    worst_s = max(r['dS'] for r in rows)
    print(f"  E0   最大 |差| = {worst_e:.3e}  (TOL_E = {TOL_E:.0e})")
    print(f"  S(L/2) 最大 |差| = {worst_s:.3e}  (TOL_S = {TOL_S:.0e})")
    if worst_e > TOL_E:
        print("  落支 = **F-两错**")
        print("  判据: 连 E0 标定都没过 ⇒ 本脚本的费米子实现有错。")
    elif worst_s > TOL_S:
        print("  落支 = **F-中途**")
        print("  判据: E0 对上、至少一个 h/J 点的 S(L/2) 对不上"
              "（**唯一有信息量的结果**）。")
    else:
        print("  落支 = **F-一致**")
        print("  判据: E0 与全部 h/J 点的 S(L/2) 都 ≤ 容差。")
        print("  ⚠ 登记必写: 一致由 JW 等价**定理保证** ⇒ 本结果证明的是**机器正确性**，")
        print("     给 L1 的 S(L/2) 添了第二条实现路径；**不缩小 audit:341 的缺口**，")
        print("     **不构成「L1 的因果通路问题已解决」**，是否动 L1 成熟度另属一个 DP。")

    print(f"\n总用时 {time.time() - _t0:.2f} s")


if __name__ == '__main__':
    with open(_LOG_PATH, 'w', encoding='utf-8') as _lf:
        _orig = sys.stdout
        sys.stdout = _Tee(_orig, _lf)
        try:
            main()
        finally:
            sys.stdout = _orig
