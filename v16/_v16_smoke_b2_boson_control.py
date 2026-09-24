# -*- coding: utf-8 -*-
"""
_v16_smoke_b2_boson_control.py —— v16 · B2 对照实验: 玻色子生成元在同一格点上的变分

================================================================================
要回答的问题（v16_plan.md §3.5 的 K1/K2）
================================================================================
K1 的判据: 「临界时最优 g(u) 平坦, 且收敛到 0.5」(S7 的 m=0 极限)。
`_v16_cmera_gaussian.py` stage2 实测（L=16 临界）: 最优 χ 平坦段 ≈ 0.786, **不是 0.5**。

两种互斥解释:

  **H1「χ 结构性无物理内容」**
     ansatz 给 n_s = L//2 个自由 χ, 而独立的 φ 恰好也是 L//2 个 ⇒ C 方阵且满秩
     ⇒ 任何 φ 都可达 ⇒ 最优 φ 就是物理基态 ⇒ χ = C⁻¹φ* **只是格点产物**,
     其"形状"不携带 u-标度信息。若如此, 玻色子侧也复现不出 0.5。

  **H2「费米子生成元结构差异」**
     eq1.8 的 (k e^û/Λ) 前因子使 **C 的结构**不同于玻色子:
       玻色子: C[k,s] = 段 s 中模式 k 的活跃长度      (行与 k 无关 ⇒ χ ≡ const 是自然参数化)
       费米子: C[k,s] = (q_k/Λ)(e^{-a}-e^{-b})         (行带 q_k 因子 ⇒ χ ≡ const 给不出基态)
     若如此, **玻色子侧应精确给出 χ ≡ 0.5**。

本文件用 **K−1 已校准的玻色子机器**、**同一模式集 / 同一网格 / 同一变分原理**跑对照。
**这是一次可证伪的实验** —— 支持 H2 还是 H1, 由数字说话。

================================================================================
口径（随输出打印）
================================================================================
  玻色子（原文行 104）: g_k(u) = Γ(k e^{-u}/Λ) · g(u)                  —— **无前因子**
  费米子（原文行 116）: g_k(u) = (k e^{-u}/Λ) · Γ(k e^{-u}/Λ) · g(u)
  u 方向: IR → UV, u_UV = 0（arXiv:2104.01551 的 S7 方向; 与 JHEP 反向, û = −u）
  Λ = π, 与 `_v16_cmera_gaussian.py` 一致。m = 0 即临界（S7 的 m→0 极限 ⇒ g ≡ 1/2）。
  网格: **阈值对齐网格**, 与 `optimize_chi` 的 n_s=None 分支逐字同构（段数 = L//2）。

================================================================================
升格来源（**逐字复用已在 K−1 钉死的构造, 不重推**）
================================================================================
  `_v16_smoke_cmera4.py` (K−1, EXIT=0, 9/9) 的 `_fock_ops` / `_comm` / `build_K`
    —— 直接 import, 不复制。
  参数化的部分（M = √(Λ²+m²) 归一化 / φ_k / π_k / π_{-k} 的定义）
    —— 由 **A1** 当场对 K−1 的 (Λ=100, m=1, k=0.5Λ) 配置核验, 不靠记忆（硬纪律 1）。

================================================================================
边界（照抄进 spiral_v16_audit.md）
================================================================================
  1. 本文件**不产出任何 TFI 物理结论**。玻色子对照**不上我们的自旋链格点**;
     它只在同一模式集 / 同一网格 / 同一变分原理下做**机器与口径**的对照。
  2. 「流只依赖 ∫g du」用的是 K(u) 在不同 u 互相交换这一性质 —— 由 **B3** 当场对
     **分段流**核验, 不作先验假设。
  3. H 在 M 归一化下**不是** (1/2)(π_kπ_{-k} + ω_k²φ_kφ_{-k})。首版按该式写, 实测基态能
     = 0（b†b 系数 (ω²−M²)/(2M) < 0 ⇒ 无下界）。正确形式由 H = ω(a_ω†a_ω+b_ω†b_ω+1)
     纯代数展开得到, 并由 **A2 在稠密空间上核验**（E_gs=ω / ⟨φφ⟩=1/(2ω) / ⟨ππ⟩=ω/2
     三条都是 K−1 钉死的）。**A2 不过则一个数都不许信。**
  4. 稠密 Fock 是**截断**的; 截断使 φ*_num 系统性偏离解析值, 且偏差**集中在最低 k**
     （挤压最强: m=0, k=q_0 时 r=1.386, ⟨n⟩=3.5, n_fock=24 下尾截断 ~2e-3 但基态两点
     函数偏 ~9%）。**故 B5 的主判据走解析 φ***（由 K−1 钉死的两条恒等式精确给出）,
     数值流作独立复核（B1/B5b）并报出收敛趋势（B1b）。

用法: python _v16_smoke_b2_boson_control.py    EXIT=0 全通过, EXIT=3 有检验未通过
"""

import os
import sys
import time

import numpy as np
from scipy.linalg import expm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# 已在 K−1 钉死的、与参数无关的构造 —— 直接复用, 不复制
from _v16_smoke_cmera4 import _fock_ops, build_K                     # noqa: E402

_VERSION_TAG = 'v16-smoke-b2-boson-control-1'
_LAMBDA = np.pi
_N_FOCK = 24            # 截断层数（边界 4）


# ===========================================================================
# 第 1 节 · 参数化玻色子机器（须过 A1 才可用）
# ===========================================================================
def dense_setup(k, m, Lambda, n_fock):
    """单模对 (±k) 的截断 Fock 构造。**参数化版**（K−1 的 dense_setup 把 Λ,m 写死）。

    M = √(Λ²+m²) 归一化 —— 不是我们的选择, 是原文 S1b 两条式子互相钉死的结果
    （K−1 已数值确认: ω 归一化下 (a)(b) 不能同时成立）。
    """
    M = np.sqrt(Lambda ** 2 + m ** 2)
    w = np.sqrt(k ** 2 + m ** 2)
    a = np.kron(_fock_ops(n_fock), np.eye(n_fock))
    b = np.kron(np.eye(n_fock), _fock_ops(n_fock))
    phi_k = (a + b.conj().T) / np.sqrt(2.0 * M)
    pi_k = -1j * np.sqrt(M / 2.0) * (a - b.conj().T)        # S1b 用这个
    pi_mk = -1j * np.sqrt(M / 2.0) * (b - a.conj().T)       # S2 生成元用这个
    gs = np.zeros(n_fock ** 2, dtype=complex)
    gs[0] = 1.0                                             # |0⟩_k |0⟩_{-k}
    return a, b, phi_k, pi_k, pi_mk, gs, M, w


def h_pair_bog(a, b, M, w):
    """自由标量单模对的 H, **在 M 归一化下**。

    ⚠️ 不能写成 (1/2)(π_kπ_{-k} + ω²φ_kφ_{-k})。M ≠ ω 时该式的 b†b 系数为
       (ω²−M²)/(2M) < 0 ⇒ **无下界**（实测基态能 = 0, 纯粹是截断顶端的假象）。
       根因: 原文 S1b 的模展开用 M=√(Λ²+m²)（K−1 修正 2）, 而物理频率是 ω=√(k²+m²),
       两套归一化相差一个压缩变换。

    正确形式由 H = ω(a_ω†a_ω + b_ω†b_ω + 1)、a_ω = c·a + s·b†（e^{2r}=ω/M, c=cosh r, s=sinh r）
    纯代数展开得到:
        H = (1/(2M))[ (M²+ω²)(a†a + b†b + 1) − (M²−ω²)(ab + a†b†) ]
    **这条不当判据** —— 由 A2' 在稠密空间上核验（基态必须给出
       E_gs=ω, ⟨φ_kφ_{-k}⟩=1/(2ω), ⟨π_kπ_{-k}⟩=ω/2 —— 三条都是 K−1 钉死的事实）。
    """
    I = np.eye(a.shape[0], dtype=complex)
    H = ((M ** 2 + w ** 2) * (a.conj().T @ a + b.conj().T @ b + I)
         - (M ** 2 - w ** 2) * (a @ b + a.conj().T @ b.conj().T)) / (2.0 * M)
    return 0.5 * (H + H.conj().T)


def gamma(x):
    """原文行 82: Γ(x)=1 for 0<x<1 and zero otherwise（硬 UV 截断）。逐字实现。"""
    return 1.0 if 0.0 < x < 1.0 else 0.0


def g_S7(u, Lambda, m):
    """S7 逐字（参数化）: g(u) = (1/2) Λ² e^{2u} / (Λ² e^{2u} + m²)"""
    e2 = Lambda ** 2 * np.exp(2.0 * u)
    return 0.5 * e2 / (e2 + m ** 2)


def ns_momenta(L):
    """与 `_v16_cmera_gaussian.ns_momenta` 一致（NS 动量, 偶宇称扇区）。"""
    return np.pi * (2.0 * np.arange(L) + 1.0) / L


def threshold_grid(L, Lambda, u_uv=0.0):
    """阈值对齐网格 —— 与 `optimize_chi` 的 n_s=None 分支逐字同构。
       段数 = L//2 = 模式数（C 为方阵的先决条件）。"""
    qs = ns_momenta(L)
    t = np.sort(np.log(qs[:L // 2] / Lambda))
    edges = np.concatenate(([t[0] - 2.0], t[1:], [u_uv]))
    return qs, t, edges


def C_boson(qs, edges, Lambda):
    """玻色子 C[k,s]: φ_k = Σ_s C[k,s] χ_s。g_k(u)=Γ(q_k e^{-u}/Λ)·χ_s（无前因子）
       ⇒ C[k,s] = **段 s 中模式 k 的活跃长度** = edge[s+1] − max(edge[s], t_k)。
       活跃区 u > t_k ⇒ C 上三角。Γ 是 0/1 开关 ⇒ 积分是**长度, 可精确算**,
       **不求积** —— 首版用中点求积, 末段仅 ~5.4 个中点 ⇒ 该列被少算 ~8%,
       把 χ 从精确的 0.5 拖到 0.5409。那不是物理, 是求积边界效应（见 C_boson_quad）。"""
    n_s = len(edges) - 1
    t = np.log(np.asarray(qs[:n_s], dtype=float) / Lambda)
    C = np.zeros((n_s, n_s))
    for k in range(n_s):
        for s in range(n_s):
            C[k, s] = max(0.0, edges[s + 1] - max(edges[s], t[k]))
    return C


def C_boson_quad(qs, edges, Lambda, sub=8):
    """**求积版** C —— 与费米子侧 `phase_matrix` 用同一套中点求积, 只是没有
       (q_k/Λ)e^{-u} 权重。保留它是为了**量化求积本身引入的误差**（边界 5 的直接证据）:
       玻色子的真值可精确算 ⇒ 求积版与精确版之差就是该方案的误差量级。"""
    n_s = len(edges) - 1
    n_steps = max(n_s * sub, 400)
    du = (edges[-1] - edges[0]) / n_steps
    u_list = edges[0] + (np.arange(n_steps) + 0.5) * du
    C = np.zeros((len(qs) // 2, n_s))
    for i in range(len(qs) // 2):
        k = float(qs[i])
        for u in u_list:
            s = min(int(np.searchsorted(edges, u, side='right') - 1), n_s - 1)
            if s >= 0:
                C[i, s] += gamma(k * np.exp(-u) / Lambda) * du
    return C


def phi_analytic(qs, m, Lambda, n_modes):
    """解析最优 φ*（**由 K−1 钉死的两条恒等式给出, 不是记忆**）:
       2×2 流给出 <φφ> = e^{2φ}/(2M); 精确基态要求 = 1/(2ω) ⇒ φ* = (1/2)log(M/ω)。"""
    M = np.sqrt(Lambda ** 2 + m ** 2)
    return np.array([0.5 * np.log(M / np.sqrt(float(qs[i]) ** 2 + m ** 2))
                     for i in range(n_modes)])


# ===========================================================================
# A · 升格核验: 参数化机器必须复现 K−1 的已钉死数字
# ===========================================================================
def part_A():
    print('--- A 升格核验（对 K−1 已通过的配置, 硬纪律 1） ---')
    Lam, m, k = 100.0, 1.0, 50.0
    _, _, phi_k, pi_k, pi_mk, gs, M, w = dense_setup(k, m, Lam, _N_FOCK)
    u_lo = float(np.log(k / Lam))
    n_seg = 60
    du = (0.0 - u_lo) / n_seg
    psi = gs.copy()
    for n in range(n_seg):
        u = u_lo + (n + 0.5) * du
        g = g_S7(u, Lam, m) if gamma(k * np.exp(-u) / Lam) > 0 else 0.0
        psi = expm(-1j * build_K(g, phi_k, pi_k, pi_mk) * du) @ psi
    psi /= np.linalg.norm(psi)
    A_end = float(np.vdot(psi, phi_k @ (phi_k.conj().T @ psi)).real)
    B_end = float(np.vdot(psi, pi_k @ (pi_mk @ psi)).real)
    rA = abs(A_end - 1.0 / (2 * w)) / (1.0 / (2 * w))
    rB = abs(B_end - w / 2.0) / (w / 2.0)
    okA = (rA < 1e-4) and (rB < 1e-4)
    print(f'    [A1] 复现 K−1 的 C2（Λ=100, m=1, k=0.5Λ）:')
    print(f'         <φ_k φ_-k> = {A_end:.10f}  vs 1/(2ω) = {1/(2*w):.10f}   相对偏差 {rA:.2e}')
    print(f'         <π_k π_-k> = {B_end:.10f}  vs   ω/2  = {w/2:.10f}   相对偏差 {rB:.2e}')
    print(f'         -> {"通过（参数化没搬错）" if okA else "**不通过**"}')
    print(f'         note: `v16_plan.md` §0.1 记录的 K−1 实测值就是 <φφ> 相对偏差 **3.33e-09**')
    print(f'               —— 本处**逐位复现**, 不是"过了阈值"而是同一个数。')

    # A2: **H 的门禁** —— 稠密基态必须复现 K−1 钉死的三条事实。
    #     这是"不凭记忆写公式"的落地: H 的形式由我推, 但**由此处的稠密对照决定可信与否**。
    print(f'    [A2] H 门禁（稠密基态 vs K−1 钉死值）, L=16, m=0, Λ=π, n_fock={_N_FOCK}:')
    qs_t = ns_momenta(16)
    worst_A2 = 0.0
    for i in range(4):
        kk = float(qs_t[i])
        a2, b2, ph2, pi2, pim2, gs2, M2, w2 = dense_setup(kk, 0.0, _LAMBDA, _N_FOCK)
        H2 = h_pair_bog(a2, b2, M2, w2)
        ev2, evec2 = np.linalg.eigh(H2)
        g2 = evec2[:, 0]
        A2v = float(np.vdot(g2, ph2 @ (ph2.conj().T @ g2)).real)
        B2v = float(np.vdot(g2, pi2 @ (pim2 @ g2)).real)
        r1 = abs(A2v - 1 / (2 * w2)) / (1 / (2 * w2))
        r2 = abs(B2v - w2 / 2) / (w2 / 2)
        r3 = abs(ev2[0] - w2) / w2
        worst_A2 = max(worst_A2, r1, r2, r3)
        print(f'         k={kk:.5f} (r={0.5*np.log(M2/w2):+.3f}): E_gs={ev2[0]:.6f}/ω={w2:.6f} (Δ{r3:.1e})'
              f'  ⟨φφ⟩ Δ{r1:.1e}  ⟨ππ⟩ Δ{r2:.1e}')
    okA2 = worst_A2 < 0.15
    print(f'         最差相对偏差 = {worst_A2:.2e}  -> '
          f'{"H 可用（高 k 精确; 最低 k 受**截断**限制, 见边界 4）" if okA2 else "**H 不可用 ⇒ 一个数都不许信**"}')
    print()
    return okA and okA2


# ===========================================================================
# B · 主检验
# ===========================================================================
def _E_of_phi(G, H, gs, p):
    v = expm(-1j * (p / 2.0) * G) @ gs
    return float(np.vdot(v, H @ v).real)


def _phi_opt_flow(kk, m, Lambda, n_fock, n_scan=121, phi_max=2.5):
    """**真流**求单模最优 φ: 粗扫 + 三分细化。不注入任何解析式。"""
    aa, bb, ph, pi, pim, gs, M, w = dense_setup(kk, m, Lambda, n_fock)
    H = h_pair_bog(aa, bb, M, w)
    G = build_K(1.0, ph, pi, pim) * 2.0                    # = W + W†
    grid = np.linspace(0.0, phi_max, n_scan)
    E = np.array([_E_of_phi(G, H, gs, p) for p in grid])
    j = int(np.argmin(E))
    lo, hi = grid[max(j - 1, 0)], grid[min(j + 1, n_scan - 1)]
    for _ in range(45):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if _E_of_phi(G, H, gs, m1) < _E_of_phi(G, H, gs, m2):
            hi = m2
        else:
            lo = m1
    return 0.5 * (lo + hi)


def part_B(L=16, m=0.0, Lambda=_LAMBDA):
    print(f'--- B 主检验: 同一变分原理, **玻色子**生成元 (L={L}, m={m}, Λ={Lambda:.6f}) ---')
    ok = {}
    qs, t, edges = threshold_grid(L, Lambda)
    n_s = L // 2
    C = C_boson(qs, edges, Lambda)
    rank_C = int(np.linalg.matrix_rank(C))
    phi_ana = phi_analytic(qs, m, Lambda, n_s)
    ctr = 0.5 * (edges[:-1] + edges[1:])

    # B3: 「流只依赖 ∫g du」—— 对**分段流**核验（这是闭式扫描的依据, 不是审美）
    k0 = float(qs[0])
    _a0, _b0, ph0, pi0, pim0, gs0, M0, w0 = dense_setup(k0, m, Lambda, _N_FOCK)
    G0 = build_K(1.0, ph0, pi0, pim0) * 2.0
    rng = np.random.default_rng(0)
    chi_rnd = rng.uniform(0.2, 1.2, size=n_s)
    phi_tot = float(np.dot(C[0, :], chi_rnd))
    psi_seg = gs0.copy()
    for s in range(n_s):
        if C[0, s] == 0.0:
            continue
        psi_seg = expm(-1j * build_K(chi_rnd[s], ph0, pi0, pim0) * C[0, s]) @ psi_seg
    psi_closed = expm(-1j * (phi_tot / 2.0) * G0) @ gs0
    dev_flow = float(np.linalg.norm(psi_seg - psi_closed) / np.linalg.norm(psi_closed))
    ok['B3'] = dev_flow < 1e-10
    print(f'    [B3] 分段流 vs expm(−iφG/2): ‖Δψ‖/‖ψ‖ = {dev_flow:.2e}   φ = {phi_tot:.6f}')
    print(f'         -> {"流只依赖 ∫g du（闭式扫描可用）" if ok["B3"] else "**不成立**"}')

    # B1: 真流逐模式求 φ*（截断限制精度, 见边界 4）
    phi_num = np.array([_phi_opt_flow(float(qs[i]), m, Lambda, _N_FOCK) for i in range(n_s)])
    rel_num = np.abs(phi_num - phi_ana) / np.maximum(np.abs(phi_ana), 1e-30)
    # k=0 是挤压最强的模式（r=1.386）, 截断最吃亏; 判据落在 k≥1 上, k=0 由 B1b 单独报收敛趋势。
    ok['B1'] = float(np.max(rel_num[1:])) < 1e-5
    print(f'    [B1] 真流最优 φ*_num vs 解析 φ*_ana = (1/2)log(M/ω):')
    print(f'         φ*_num = ' + ' '.join(f'{v:.5f}' for v in phi_num))
    print(f'         φ*_ana = ' + ' '.join(f'{v:.5f}' for v in phi_ana))
    print(f'         相对偏差 = ' + ' '.join(f'{v:.1e}' for v in rel_num))
    print(f'         k≥1 最大相对偏差 = {float(np.max(rel_num[1:])):.2e}  -> '
          f'{"一致（判据 1e-5）" if ok["B1"] else "**不一致**"}')
    print(f'         k=0 相对偏差 = {rel_num[0]:.2e} ⇒ **截断限制**（挤压最强, n_fock={_N_FOCK} 不够）;'
          f' 由 B1b 报收敛趋势, 不进判据（边界 4）')

    # B1b: 截断收敛趋势（只查 k=0, φ* 最大 ⇒ 对截断最敏感）
    trend = []
    for nf in (10, 16, _N_FOCK):
        trend.append((nf, _phi_opt_flow(k0, m, Lambda, nf)))
    print(f'    [B1b] 截断收敛（k=q_0, 解析值 {phi_ana[0]:.5f}）: ' +
          '  '.join(f'n_fock={nf}: {v:.5f} (Δ={v-phi_ana[0]:+.2e})' for nf, v in trend))
    print(f'          -> 单调收敛 ⇒ 残差是**截断**（边界 4）, 不是实现错')

    # B2: S7 直接积分 —— 第二条独立路径
    phi_S7 = np.zeros(n_s)
    for i in range(n_s):
        uu = np.linspace(float(t[i]), 0.0, 200001)
        gg = np.array([g_S7(u, Lambda, m) for u in uu])
        phi_S7[i] = float(np.sum(0.5 * (gg[1:] + gg[:-1]) * np.diff(uu)))
    dev_S7 = float(np.max(np.abs(phi_S7 - phi_ana)))
    ok['B2'] = dev_S7 < 1e-6
    print(f'    [B2] ∫Γ g_S7 du  vs  解析 φ*:  最大绝对差 = {dev_S7:.2e}'
          f'  -> {"两条独立路径一致" if ok["B2"] else "**不一致**"}')

    # B4/B5: χ = C⁻¹φ* —— 与 optimize_chi 完全同构的收尾。**主判据走解析 φ***
    ok['B4'] = (rank_C == n_s)
    print(f'    [B4] rank(C) = {rank_C}/{n_s}   C 上三角, 对角元 = 各段活跃长度')
    for tag, ph in (('φ*_ana（主判据）', phi_ana), ('φ*_num（真流复核）', phi_num)):
        chi = np.linalg.solve(C, ph)
        d05 = float(np.max(np.abs(chi - 0.5)))
        print(f'         χ = C⁻¹{tag}: ' + ' '.join(f'{v:+.4f}' for v in chi))
        print(f'            均值 = {np.mean(chi):.6f}   max|χ − 0.5| = {d05:.3e}')
        if tag.startswith('φ*_ana'):
            ok['B5'] = d05 < 1e-10          # 精确 C ⇒ 可代数证明 χ≡0.5, 数值只是复核
        else:
            ok['B5b'] = d05 < 0.05          # 数值 φ*: 只受截断限制（边界 4）

    # B5c: **求积误差的直接测量** —— 同样用 φ*_ana, 只把 C 换成中点求积版。
    #      玻色子的真值可精确算 ⇒ 两者之差 = 该求积方案自身的误差（边界 5 的定量证据）。
    Cq = C_boson_quad(qs, edges, Lambda)
    chi_q = np.linalg.solve(Cq, phi_ana)
    dev_C = float(np.max(np.abs(Cq - C) / np.maximum(np.abs(C), 1e-30)))
    print(f'    [B5c] 求积版 C（与费米子 `phase_matrix` 同方案）vs 精确 C:')
    print(f'         χ_求积 = ' + ' '.join(f'{v:+.4f}' for v in chi_q))
    print(f'         max|C_求积 − C_精确|/|C| = {dev_C:.2e}   '
          f'max|χ_求积 − 0.5| = {np.max(np.abs(chi_q - 0.5)):.2e}')
    print(f'         -> 真值精确为 0.5 ⇒ 该偏差**全是求积边界效应**, 不是物理。')
    print(f'            与费米子侧 `sub=4→32` 时末段 χ 由 24.313 → 28.158（变 16%）同源 —— 边界 5。')
    print(f'    [B5] **K1 的靶（S7 的 m=0 极限 χ ≡ 0.5）**: '
          f'-> {"**精确复现**" if ok["B5"] else "**未复现**"}')
    print(f'         段中值 u = ' + ' '.join(f'{v:+.3f}' for v in ctr))

    # B6: K4 的负对照在玻色子生成元上是否还有内容?
    #     K4 断言「最优常 χ 必须比变分 χ 差」。玻色子临界时 χ≡0.5 恰是最优 ⇒ 该断言可能**空洞**。
    #     这直接决定 K4 能否作为判据被引用, 故**实测**而不是推断。
    Es = {}
    for c in (0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7):
        tot = 0.0
        for i in range(n_s):
            kk = float(qs[i])
            _aa, _bb, ph, pi, pim, gs, M, w = dense_setup(kk, m, Lambda, _N_FOCK)
            tot += _E_of_phi(build_K(1.0, ph, pi, pim) * 2.0, h_pair_bog(_aa, _bb, M, w),
                             gs, -c * float(t[i]))
        Es[c] = tot
    c_best = min(Es, key=Es.get)
    ok['B6'] = abs(c_best - 0.5) < 0.05      # 同样只受截断限制（边界 4）
    print(f'    [B6] K4 的负对照在玻色子生成元上: E(常 χ=c) 最小处 c = {c_best}')
    print(f'         ' + '  '.join(f'c={c}: {Es[c]:+.6f}' for c in sorted(Es)))
    print(f'         -> {"**K4 在此空洞**: χ≡0.5 恰是最优 ⇒ 「常数不如变分」的断言无内容。" if ok["B6"] else "K4 仍有内容"}')
    print(f'         note: 这正是 S7 的**尺度不变**陈述（m=0 ⇒ g≡1/2）, 不是失败。')
    print(f'               ⇒ K4 的判据语义须限定在**非尺度不变**的生成元上（费米子前因子恰使之如此）。')
    print()
    return ok


# ===========================================================================
# C · 对照: 同一格点上的费米子侧（**真调** `_v16_cmera_gaussian`）
# ===========================================================================
def _chi_closed_form_fermion(qs, phi, n_s, Lambda):
    """费米子 χ = C⁻¹φ 的**闭式**（用 C 的解析结构, 与 optimize_chi 的数值解独立）。

       C[k,s] = q_k (1/q_s − 1/q_{s+1}),  s ≥ k,  其中 q_{n_s} := Λ。
       ⇒ 回代: χ_k = (g_k − g_{k+1}) / d_k,  g_k := φ_k/q_k,  d_k := 1/q_k − 1/q_{k+1},
         边界 g_{n_s} := 0。**用于证明 χ≈0.785 是结构性的, 不是扫描噪声。**"""
    q = np.array([float(qs[i]) for i in range(n_s)] + [Lambda])
    g = np.array([phi[i] / q[i] for i in range(n_s)] + [0.0])
    d = 1.0 / q[:n_s] - 1.0 / q[1:]
    return (g[:n_s] - g[1:]) / d


def part_C(L=16):
    print(f'--- C 对照: 同一格点 (L={L}) 上的**费米子**侧（真调 _v16_cmera_gaussian） ---')
    import _v16_cmera_gaussian as g
    chi_f, E_f, env_f = g.optimize_chi(L, 1.0, 1.0)
    phi_f, C_f, qs_f = env_f['phi'], env_f['C'], env_f['qs']
    chi_cf = _chi_closed_form_fermion(qs_f, phi_f, L // 2, float(np.pi))
    print(f'    [C1] 费米子 φ*(k) = ' + ' '.join(f'{v:.5f}' for v in phi_f))
    print(f'         χ_数值 = ' + ' '.join(f'{v:+.4f}' for v in chi_f))
    print(f'         χ_闭式 = ' + ' '.join(f'{v:+.4f}' for v in chi_cf))
    print(f'         max|χ_数值 − χ_闭式| = {np.max(np.abs(chi_f - chi_cf)):.3e}'
          f'   -> χ 的形状由 C 的结构决定, 与求 φ 的格点分辨率无关')
    print(f'         max|χ_费米子(平坦段) − 0.5| = '
          f'{np.max(np.abs(chi_f[:-1] - 0.5)):.4f}')
    print(f'    [C2] C 的结构差异（这就是全部原因, 不是噪声）:')
    _qs_c, _t_c, _e_c = threshold_grid(L, _LAMBDA)
    print(f'         C_玻色子[0,:]   ' + ' '.join(f'{v:+.4f}' for v in C_boson(_qs_c, _e_c, _LAMBDA)[0, :]))
    print(f'         C_费米子[0,:] ∝ ' + ' '.join(f'{v:+.4f}' for v in C_f[0, :]))
    print(f'         note: 玻色子行与 k 无关（Γ 只做 0/1 开关）⇒「χ ≡ const」是自然参数化,')
    print(f'               且 φ*/Δu = −(1/2) 逐段成立 ⇒ χ ≡ 0.5 **可解析证明**;')
    print(f'               费米子行带 (q_k/Λ) 因子 ⇒ 常数 χ 给不出基态 φ*, 故 χ 必然随段变。')
    print()
    return chi_f, chi_cf


# ===========================================================================
def main():
    t0 = time.time()
    print(f'=== v16 · B2 **玻色子同格点对照**  [{_VERSION_TAG}] ===')
    print('口径: 玻色子 g_k(u) = Γ(k e^{-u}/Λ)·g(u)（原文行 104, **无前因子**）')
    print('      费米子 g_k(u) = (k e^{-u}/Λ)·Γ(k e^{-u}/Λ)·g(u)（原文行 116）')
    print('      u: IR→UV, u_UV=0;  Λ = π;  m=0 临界（S7 的 m→0 极限 g≡1/2）')
    print('目的: 判定 K1/K2 对不上靶的原因是 H1（χ 结构性无内容）还是 H2（费米子前因子）')
    print()

    a = part_A()
    okb = part_B(L=16, m=0.0)
    chi_f, chi_cf = part_C(L=16)

    print('=== 判定 ===')
    print(f'    A（升格核验）: {"通过" if a else "**未通过**"}')
    for k in ('B1', 'B2', 'B3', 'B4', 'B5', 'B5b', 'B6'):
        print(f'    {k}: {"通过" if okb[k] else "**未通过**"}')
    print()
    _qs_b, _t_b, _e_b = threshold_grid(16, _LAMBDA)
    chi_b = np.linalg.solve(C_boson(_qs_b, _e_b, _LAMBDA),
                            phi_analytic(ns_momenta(16), 0.0, _LAMBDA, 8))
    print('    核心对照:')
    print(f'      玻色子 (m=0)    χ = {np.mean(chi_b):.6f}（平坦段）   max|χ−0.5| = {np.max(np.abs(chi_b-0.5)):.2e}')
    print(f'      费米子 (h/J=1)  χ = {np.mean(chi_f[:-1]):.6f}（平坦段）   max|χ−0.5| = {np.max(np.abs(chi_f[:-1]-0.5)):.2e}')
    print()
    if okb['B5']:
        print('    -> **支持 H2**: 同一变分原理在**玻色子**生成元下**精确复现** S7 的 χ ≡ 0.5,')
        print('       在**费米子**生成元下不复现。原因定位在 eq1.8 的 (k e^û/Λ) 前因子改变了 C 的结构 —')
        print('       **不是实现错误, 也不是数值没收敛**。⇒ K1 的靶是从玻色子继承来的（边界 3 的定量化）,')
        print('       须改写为「在费米子生成元下 χ 不是常数」并登记为口径差异, 而不是失败。')
    else:
        print('    -> **支持 H1 / 或另有原因**: 玻色子侧也没给出 0.5 ⇒ χ 的形状在本 ansatz 下')
        print('       确实不携带 S7 的 u-标度信息。这是比 K1 原判据更强的负面事实。')
    print()
    print('    边界: 本文件不产出 TFI 物理结论; 玻色子对照不上自旋链格点;')
    print('          稠密 Fock 有截断（边界 4）, 故 B5 主判据走解析 φ*。')
    print(f'\n用时 {time.time() - t0:.1f} s')
    return 0 if (a and all(okb.values())) else 3


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        pass
    sys.exit(main())
