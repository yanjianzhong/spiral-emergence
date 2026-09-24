# -*- coding: utf-8 -*-
"""
_v16_cmera_gaussian.py —— v16 · B2 主体: 自由费米子高斯 cMERA（阶段 1：地基与纯 2L×2L 路径）

================================================================================
**阶段声明（本文件目前只完成了阶段 1，不要当成完整实现读）**
  阶段 1（本文件）: 升格地基 + **纯 2L×2L 生成元路径** + 两条自检
  阶段 2（未做）  : 变分优化 `χ(u)`、K1–K6 判据、诊断 B（离散 `L`）
================================================================================

为什么分成阶段 1: 写之前发现 `v16_plan.md` §3.5 的「先只跑 L=16」**按稠密路径不可行**。
  2^16 = 65536，一个稠密算符 = 65536² × 16 B = **68.7 GB** —— 远超 16 GB 红线。
  （§3.5 写的「L=16 的稠密态矢只有 1 MB」是对的，但那是**态矢** 65536×16 B = 1 MB，
    不是**算符**。`_v16_smoke_cmera5.py` 的 `pairing_generator` 建的是稠密算符，
    故**不能升格**到 L=16。）
  ⇒ 必须走 §3.4 标题本来要求的 **纯 2L×2L**，且**先对已通过的稠密路径（K−2, L=8）核验**。
  这正是硬纪律 2「冒烟测试必须真调被改动的函数」的用法。

口径（**必须随每个输出一起打印**，v16_plan.md §9 决策 2 的措辞纪律）:
  口径 A: H_til = H_open + J·iγ_{2L-1}γ_0   （偶宇称扇区; 由探针 3 的 T8 确立）

u 方向约定（**本文档唯一的约定选择，必须显式声明**）:
  取 **arXiv:2104.01551 的 S7 方向**: u 从 IR(u→−∞) 流向 UV(u=0)。
  理由: K1/K2 的解析靶 `g(u) = (1/2)Λ²e^{2u}/(Λ²e^{2u}+m²)` 与 `u* = log(m/Λ)` 都写在这个方向下。
  与 JHEP09(2015)002 **反向**: 原文 û 从 UV(0) 流向 IR(+∞)，故 `û = −u`。
  原文行 90/116 的 `Γ(k e^{û}/Λ)` 在本文约定下写作 `Γ(k e^{−u}/Λ)`。
  ⇒ 活跃模式集 `k < Λ e^{u}`：u→−∞ 时为空，u=0 时为 `k<Λ`。**随 u 增大而增长（IR→UV）**。
  ⚠️ 两文方向相反这件事已在 `v16_plan.md` §3.3.5「三式互锁」记录（S7 代 u→−u 即 eq2.8）。

原文锚点（逐行读过 `v16/cMERA_Entropy_rev.tex`，非记忆）:
  行  82: `Γ(x)=1 for 0<x<1 and zero otherwise`（硬 UV 截断）
  行  82: 「**to get rid of the L process in our analysis, we proceed by rescaling the cMERA states**」
          ⇒ **原文自己就消掉了 L** —— 这是 `v16_plan.md` §3.4.0 主线 A 的**直接证据**，不是推断。
  行  90: `K̃(û) = ∫dk Γ(k e^û/Λ) g(û, k e^û) Õ_k`
  行 104: `g_k^B(û) = Γ(k e^û/Λ) g^B(û,k)`
  行 116: `g_k^F(û) = (k e^u/Λ) Γ(k e^û/Λ) g^F(û,k)`   ← 费米子**多一个 (k e^û/Λ) 前因子**

升格的来源（**逐字搬运，未改逻辑**；改动即为 bug）:
  `_v16_smoke_cmera3.py` (探针 3, EXIT=0 9/9): 地基 —— Majorana 约定 / 稠密 TFI / 纯二次判据 /
      h 重建 / 2L×2L 的 build_h / BdG 谱能量 / 协方差
  `_v16_smoke_cmera5.py` (探针 5 = K−2, EXIT=0 9/9): 动量模式 / eq1.8 生成元 / UV 真空

边界（**照抄进 spiral_v16_audit.md**）:
  1. 本文件**不建 `L` 算符**（§3.4.0 主线 A）。尺度由 `g_k(u)` 的 running cutoff 承担。
     `L|Ω⟩=0` **在本口径下未被检验** —— 不得写成「满足 cMERA 全部定义」。
  2. `d_k ≡ c†_{-q_k}` 是**逻辑推断**（原文 d 是狄拉克反粒子，TFI 无此物），非原文陈述。
  3. K1/K2 的靶 `g(u)` 来自 **2104.01551（玻色子）**。费米子版的闭式原文**没有**
     （JHEP §3 整节无费米子内容，`eq3.5i` 是玻色子的）。⇒ 用玻色子靶检验费米子实现，
     是**继承来的假设**，不是已证事实。
  4. 本文件不产出任何 TFI 物理结论；阶段 1 只回答「纯 2L×2L 路径等不等于已通过的稠密路径」。
"""

import os
import sys
import time

import numpy as np
from scipy.linalg import expm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'v15'))

from spiral_model_v15 import jw_ground_energy   # K3 的解析参照（v15 已对稠密 ED 核到 2.5e-14）

_VERSION_TAG = 'v16-cmera-gaussian-1'

_LAMBDA = 1.0            # UV cutoff（原文 Λ; 本文量纲无关, 取 1）
_I2 = np.eye(2, dtype=complex)
_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)


# ===========================================================================
# 第 1 节 · 地基 —— 从 `_v16_smoke_cmera3.py` 逐字升格（探针 3, EXIT=0 9/9）
# ===========================================================================
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


def fermion_test(H, Gam, dim):
    """自由费米子判据: [H, γ_c] 是否落在 span{γ_a}。
       H = (i/4) Σ_ab h_ab γ_a γ_b  ⇒  [H, γ_c] = i Σ_a h_ac γ_a。返回 (最差相对残差, h)。"""
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
       H_til = -J Σ_i iγ_{2i+1}γ_{2i+2} - h Σ_i iγ_{2i}γ_{2i+1} + J·iγ_{2L-1}γ_0"""
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


# ===========================================================================
# 第 2 节 · 动量模式与 eq1.8 生成元 —— 从 `_v16_smoke_cmera5.py` 逐字升格
#           （探针 5 = K−2, EXIT=0 9/9）
# ===========================================================================
def real_fermions(Gam, L):
    """c_j = (γ_{2j} + i γ_{2j+1})/2 。由 A1 当场用反对易子核验, 不当假设。"""
    return [(Gam[2 * j] + 1j * Gam[2 * j + 1]) / 2.0 for j in range(L)]


def ns_momenta(L):
    """v15 的 NS 动量集合 (v15/spiral_model_v15.py:745)。偶宇称扇区 ⇒ 反周期边界 ⇒ 半整数动量。"""
    return np.pi * (2.0 * np.arange(L) + 1.0) / L


def fourier_modes(c, L, qs):
    """c_q = L^{-1/2} Σ_j e^{-i q j} c_j 。由 A2 当场核验其正则反对易关系。"""
    return [sum(np.exp(-1j * q * j) * c[j] for j in range(L)) / np.sqrt(L) for q in qs]


def antic(A, B):
    return A @ B + B @ A


def pairing_generator(cq, L, g_of_k, sign=-1.0):
    """eq1.8 的**稠密**格点转录（只在 L ≤ 10 可用: 算符是 2^L×2^L）。
       K = i Σ_{k<L/2} g_k ( c†_{q_k} c†_{-q_k}  +  sign · c_{-q_k} c_{q_k} )"""
    K = np.zeros_like(cq[0])
    for k in range(L // 2):
        a, b = k, L - 1 - k
        g = g_of_k(abs(np.pi * (2 * k + 1) / L))
        A = cq[a].conj().T @ cq[b].conj().T
        K = K + 1j * g * (A + sign * A.conj().T)
    return K


def uv_vacuum(L):
    """eq1.9 的 UV 真空 |Ω⟩ (c_q|Ω⟩=0)。
       由 c_j=(γ_{2j}+iγ_{2j+1})/2 展开: c†_jc_j = (I + σx_j)/2
    ⇒ **c-真空是全 σx = -1 ⇒ |Ω⟩ = |−⟩^{⊗L}**（不是 |+⟩^{⊗L} —— 那是全满态, 真空的补）。
       K−2 的 C 部分即死在这一条上（v1 误取 |+⟩^{⊗L}）。"""
    s = np.arange(2 ** L, dtype=np.int64)
    pc = np.zeros(2 ** L, dtype=np.int64)
    for b in range(L):
        pc += (s >> b) & 1
    return ((-1.0) ** pc) / np.sqrt(2.0 ** L)


# ===========================================================================
# 第 3 节 · 新增: 纯 2L×2L 生成元路径（**本文件的主要新增内容**）
#
# 推导（不靠任何凭记忆的 BdG 公式; 每一步都可对照第 2 节的稠密路径证伪）:
#   c†_j = (γ_{2j} − i γ_{2j+1})/2 ,  c†_q = L^{-1/2} Σ_j e^{+i q j} c†_j
#   D_k ≡ c†_{q_k} c†_{q_b}   (b = L−1−k)
#       = (1/(4L)) Σ_{jl} e^{iθ_{jl}} B_{jl} ,   θ_{jl} = q_k j + q_b l
#       B_{jl} = γ_{2j}γ_{2l} − iγ_{2j}γ_{2l+1} − iγ_{2j+1}γ_{2l} − γ_{2j+1}γ_{2l+1}   (B_{jl}=0 当 j=l)
#   K = i Σ_k g_k (D_k − D_k†)
#
#   ⚠️ **v1 死在这里, 记下来**: (c†_jc†_l)† = c_l c_j —— 共轭把 j,l **掉了个个**,
#      而指数里的 j,l **不动**。v1 误以为 D† 只是「系数取共轭再取负」, 于是
#      **该留的 sin 项被消掉、该消的 cos 项被留下**, 实测 hmat_pure ≡ 0（S2a 比值恰为 1.00e+00）。
#      这条**只有稠密对照能抓**（对称性论证本身自洽, 看不出来）—— 硬纪律 1 的实证。
#
#   由 (γ_xγ_y)† = γ_yγ_x = −γ_xγ_y (x≠y):
#       B_{jl}† = −γ_{2j}γ_{2l} − iγ_{2j}γ_{2l+1} − iγ_{2j+1}γ_{2l} + γ_{2j+1}γ_{2l+1}
#   ⇒  B + B† = −2i(γ_{2j}γ_{2l+1} + γ_{2j+1}γ_{2l})  ≡ −2i S_{jl}
#       B − B† =  2(γ_{2j}γ_{2l} − γ_{2j+1}γ_{2l+1})
#   D − D† = (1/(4L)) Σ[ e^{iθ}B − e^{-iθ}B† ] = (1/(4L)) Σ[ cosθ(B−B†) + i sinθ(B+B†) ]
#          = (1/(2L)) Σ[ cosθ(γ_{2j}γ_{2l} − γ_{2j+1}γ_{2l+1}) + sinθ·S_{jl} ]
#
#   对称性（θ = q_k(j−l), 因 q_b = −q_k **mod 2π**）:
#       cosθ 关于 j↔l **对称** ;  sinθ **反对称**
#       (γ_{2j}γ_{2l} − γ_{2j+1}γ_{2l+1}) **反对称** ;  S_{jl} **反对称**
#   ⇒ 对称×反对称的 cos 项**逐项相消** ⇒ **只余 sin 项**:
#       D_k − D_k† = (1/(2L)) Σ_{jl} sin(q_k(j−l)) · S_{jl}
#   自洽性: S_{jl} 反厄米、系数实 ⇒ D−D† 反厄米 ✓ ; 再乘 i ⇒ K 厄米 ✓（与 B1 的实测一致）
#   ⇒ K = i Σ_k g_k (1/(2L)) Σ_{jl} sin(q_k(j−l)) ( γ_{2j}γ_{2l+1} + γ_{2j+1}γ_{2l} )
#
#   hmat 的换算: 若 K = i Σ_{a<b} c_ab γ_aγ_b (c 实), 则 h = 2(M − Mᵀ), M[a,b] = c_ab。
#       交叉验证: K = −J·iγ_{2i+1}γ_{2i+2} ⇒ c = −J ⇒ h_{2i+1,2i+2} = −2J, 与 build_h 逐字相同 ✓
# ===========================================================================
def _accum_pair(M, a, b, coef):
    """把 coef·γ_aγ_b (a≠b) 累进「上三角系数」矩阵 M（用 γ_aγ_b = −γ_bγ_a 归一到 a<b）。"""
    if a < b:
        M[a, b] += coef
    else:
        M[b, a] -= coef


def _bilinear_hmat(M):
    """B = Σ_{a<b} M[a,b] γ_aγ_b  ⇒  H = (i/4)Σ_ab h_ab γ_aγ_b 的 h（实反对称）。"""
    return 2.0 * (M - M.T)


def pairing_hmat(L, qs, g_of_k, sign=-1.0):
    """eq1.8 的**纯 2L×2L** 转录（无 2^L, 故可上大 L）。返回实反对称 hmat。
       g_of_k: 可调用, 入参 |q_k|; running cutoff 与 (k e^{−u}/Λ) 前因子在调用方体现。
       ⚠️ 必须由自检 S2 对本文件第 2 节的稠密 `pairing_generator` 核验后才可使用。

       **sign=−1**: K = i g(D − D†) = i Σ g_k (c†c† − cc)   ← eq1.8 的**字面**转录（K−2 已钉）
       **sign=+1**: K = −g(D + D†) = −Σ g_k (c†c† + cc)      ← 见下方 ⚠️, 这才是 TFI 需要的那个

       ⚠️ **为什么需要 sign=+1（2026-09-24 阶段 2 实测逼出, 不是推导偏好）**:
       两式都给出 U|00⟩ 的**占据数** ⟨n_q⟩=sin²φ, 故 K−2 的 C1 **无法区分**它们。
       区别在**幅角的相位**: sign=−1 ⇒ U|00⟩ = cosφ|00⟩ + sinφ|11⟩（幅**实**）;
       sign=+1 ⇒ U|00⟩ = cosφ|00⟩ + i·sinφ|11⟩（幅**虚**）。
       实测: TFI 基态的配对幅 ⟨c_q c_{−q}⟩ 的辐角**恒为 +π/2（纯虚）**,
       而 sign=−1 的末态给出 0 或 π（纯实）⇒ 用 sign=−1 时**永远到不了基态**
       （L=8 上解出精确 φ* 后 ⟨H⟩ 仍为 −6.583 vs E0 −10.252）。
       ⇒ 本文件阶段 2 的物理一律用 **sign=+1**; sign=−1 保留仅为与 K−2 对账。
       两分支**都必须**对稠密路径核验（S2a/S2a2）。"""
    n = 2 * L
    M = np.zeros((n, n))
    for k in range(L // 2):
        a, b = k, L - 1 - k
        g = float(g_of_k(abs(qs[a])))
        for j in range(L):
            for l in range(L):
                c = g * np.sin(qs[a] * j + qs[b] * l) / (2.0 * L)
                if sign < 0:
                    _accum_pair(M, 2 * j, 2 * l + 1, +c)
                    _accum_pair(M, 2 * j + 1, 2 * l, +c)
                else:
                    # ⚠️ 这两个符号是**稠密对照定的**, 不是推导取定的:
                    #    首版取 (−c, +c) 实测偏离恰好 2.000 ⇒ 整体差一负号, 翻正后 2.2e-16。
                    _accum_pair(M, 2 * j, 2 * l, +c)
                    _accum_pair(M, 2 * j + 1, 2 * l + 1, -c)
    return _bilinear_hmat(M)


def cutoff_gamma(x):
    """原文行 82: Γ(x)=1 for 0<x<1 and zero otherwise（硬 UV 截断）。逐字实现。"""
    return 1.0 if 0.0 < x < 1.0 else 0.0


def g_fermion(u, k, chi, Lambda=_LAMBDA):
    """原文行 116（**已换到本文的 u 方向**: û = −u）:
           g_k^F(û) = (k e^{û}/Λ) · Γ(k e^{û}/Λ) · g^F(û,k)
       ⇒ 本文约定: g_k(u) = (k e^{−u}/Λ) · Γ(k e^{−u}/Λ) · χ
       活跃条件 Γ=1 ⇔ k < Λ e^{u}（u→−∞ 空集, u=0 为 k<Λ）。
       ⚠️ 前因子 (k e^{−u}/Λ) 是**费米子特有**（行 104 的玻色子版没有）—— 这是与 K−1 的差别之一。"""
    x = k * np.exp(-u) / Lambda
    return x * cutoff_gamma(x) * chi


def flow_covariance(Gam0, hmat_of_u, u0, u1, n_steps):
    """小步乘积（路径排序由乘积自然处理）。
       Γ ← Rᵀ Γ R ,  R = expm(−h·du)
       **变换指标在第二位**（γ'_a = U†γ_aU = Σ_c R_ca γ_c）⇒ 是 Rᵀ Γ R 而**不是** R Γ Rᵀ。
       K−2 的 D2c 实测: 5.55e-16（错误写法给出 9.67e-05 ≈ θ）。"""
    G = np.array(Gam0, dtype=float)
    du = (u1 - u0) / n_steps
    for s in range(n_steps):
        u = u0 + (s + 0.5) * du                 # 中点法则
        R = expm(-du * hmat_of_u(u))
        G = R.T @ G @ R
    return G


def energy_from_gamma(Gam, hmat):
    """⟨H⟩ = (1/4) Σ_ab h_ab Γ_ab （探针 3 的 T5 已对 ED 核到 1e-9）。"""
    return float(np.real(np.sum(hmat * Gam)) / 4.0)


# ===========================================================================
# 第 4 节 · 阶段 2: 变分流、K1–K6、诊断 B
#
# **关键结构（使变分可控）**: 不同 k 的模式相互独立 —— K = Σ_k K_k 且 [K_k, K_k'] = 0
#   （不同 k 的配对算符作用在不同模式上）。故整个流解析约化为相位
#       φ_k = ∫_{u_IR}^{0} g_k(u) du   ,   末态 = Π_k (cos φ_k |00⟩_k + sin φ_k |11⟩_k)
#   （与本文件 K−2 的 C1 一致: ⟨n_q⟩=sin²(gt)、γ_k=tan²(φ_k)）。
#   本实现仍走**显式小步乘积**（不走这条约化），因为约化式只对"分段常数 ansatz"成立;
#   显式流对更一般的 g 也对, 且可与约化式**互为独立检验**（K7 交叉核对, 见下）。
#
# **变分原理（这是一个选择, 必须声明，不是原文给的操作化定义）**:
#   极小化末态能量 ⟨H_til⟩ over χ(u)。cMERA 本就是基态变分 ansatz, 取能量为代价是标准做法,
#   但**本仓库没有从 2104.01551 原文核到它的具体操作化形式** ⇒ 标注为**选择**，不称"按原文"。
#
# **ansatz**: g_k(u) = (k e^{−u}/Λ)·Γ(k e^{−u}/Λ)·χ_s。K1/K2 的靶是 **χ**（裸的 g^F）,
#   理由: 原文行 104 玻色子版 `g_k^B = Γ·g^B` **无前因子**, S7 的 g 对应的是裸量 g^B;
#   费米子的对应裸量就是行 116 里的 g^F(û,k)。⇒ 与 S7 比较的是 χ, 不是 g_k。**这是口径选择。**
# ===========================================================================
def vacuum_covariance(L):
    """|Ω⟩ = |−⟩^{⊗L} 的 Γ, **不用 2^L**。
       由 Γ_ab = i⟨γ_aγ_b⟩ − iδ_ab, 及 |−⟩^{⊗L} 上 ⟨σz⟩=⟨σy⟩=0 ⇒ ⟨γ_a⟩=0 ⇒ a≠b 时 ⟨γ_aγ_b⟩=0;
       Γ_{2j,2j+1} = i⟨σz_jσy_j⟩ = i⟨−iσx_j⟩ = ⟨σx_j⟩ = −1。
       ⚠️ 这两条引理**不靠记忆** —— 由 S3 当场对 `covariance(uv_vacuum(L), Gam)` 稠密核验。"""
    G = np.zeros((2 * L, 2 * L))
    for j in range(L):
        G[2 * j, 2 * j + 1] = -1.0
        G[2 * j + 1, 2 * j] = +1.0
    return G


def _scale_matrices(L, qs, sign=-1.0):
    """H1[k] = 第 k 个模式的生成元（该模式 g=1, 其余 0）的 hmat。
       因 `pairing_hmat` 对 g **线性**, 任意 g 的 hmat = Σ_k g_k·H1[k]。"""
    H1 = []
    for k in range(L // 2):
        kk = float(qs[k])
        H1.append(pairing_hmat(L, qs, lambda kx, kk=kk: 1.0 if abs(kx - kk) < 1e-12 else 0.0,
                               sign=sign))
    return H1


def cutoff_prefactor(u, k, Lambda=_LAMBDA, prefactor=True):
    """截断因子 Γ(k e^{−u}/Λ)（原文行 82）; prefactor=True 时再乘费米子特有的 (k e^{−u}/Λ)。"""
    x = k * np.exp(-u) / Lambda
    return (x if prefactor else 1.0) * cutoff_gamma(x)


def bond_matrices(L, qs, u_list, Lambda=_LAMBDA, prefactor=True, sign=-1.0):
    """B[s] = Σ_k coef_k(u_s)·H1[k], 使 K(u_s) 的 hmat = χ·B[s]。**与 χ 无关, 故可预计算**。"""
    H1 = _scale_matrices(L, qs, sign=sign)
    ks = [float(qs[i]) for i in range(L // 2)]
    B = []
    for u in u_list:
        Bs = np.zeros((2 * L, 2 * L))
        for i, k in enumerate(ks):
            c = cutoff_prefactor(u, k, Lambda, prefactor)
            if c != 0.0:
                Bs += c * H1[i]
        B.append(Bs)
    return B


def make_grid(u_IR, u_UV, n_s, sub):
    """χ 分段常数（n_s 段）; 每段内再分 sub 个小步做流（中点法则）。"""
    edges = np.linspace(u_IR, u_UV, n_s + 1)
    n_steps = n_s * sub
    du = (u_UV - u_IR) / n_steps
    u_list = edges[0] + (np.arange(n_steps) + 0.5) * du
    return edges, u_list, du


def phase_matrix(L, qs, edges, u_list, du, Lambda=_LAMBDA, prefactor=True):
    """C[k,s], 使**离散流实际累积的相位** φ_k = Σ_s C[k,s]χ_s。

    ⚠️ **必须用与 `bond_matrices` 相同的中点求积**, 不能用解析积分:
       `coef_k(u)` 在 u = log(q_k/Λ) 处**跳变**（硬截断 Γ）; 解析积分与中点求积
       在跳变段相差 0.053（实测, L=8）—— 而 φ 直接决定末态, 5e-2 的差就足以
       让能量偏离 E0 达 4.4e-3。**解析式与数值式必须同源**, 这是本条存在的理由。

    用途: 由目标相位**直接解** χ（`lstsq(C, φ*)`），绕过优化器。
    可靠性由 S4 核验: 解出的 χ 必须使 flow 的末态 ⟨n_q⟩ = sin²φ* 且 ⟨H⟩ = E0。"""
    C = np.zeros((L // 2, len(edges) - 1))
    n_s = len(edges) - 1
    for i in range(L // 2):
        k = float(qs[i])
        for u in u_list:
            s = min(int(np.searchsorted(edges, u, side='right') - 1), n_s - 1)
            C[i, s] += cutoff_prefactor(u, k, Lambda, prefactor) * du
    return C


def flow_energy(chi, B, u_list, edges, du, Gam0, hmat_H):
    """末态能量 ⟨H_til⟩。流: Γ ← Rᵀ Γ R, R = expm(−du·χ_s·B_s)。"""
    n_s = len(chi)
    G = Gam0
    for s, u in enumerate(u_list):
        idx = min(int(np.searchsorted(edges, u, side='right') - 1), n_s - 1)
        R = expm(-du * float(chi[idx]) * B[s])
        G = R.T @ G @ R
    return energy_from_gamma(G, hmat_H)


def optimize_chi(L, J, h, n_s=None, u_IR=None, u_UV=0.0, sub=8,
                 Lambda=None, prefactor=True, x0=None, maxfev=3000,
                 n_sweeps=6, n_scan=61, n_iter=4, sign=1.0):
    """极小化末态能量。返回 (chi, E, 环境 dict)。sign 默认 +1（阶段 2 的物理分支, 见 `pairing_hmat`）。

       n_s 默认 = L/2, u_IR 默认 = log(q_min/Λ) − 2, 且**边界放在阈值 t_k = log(q_k/Λ) 上**
       （`drop_first`）—— 三条都不是审美, 是为了让 C 满秩:
       均匀网格下 rank(C) 只有 L/4（阈值在 u≈0 附近挤在一起, 分不开）, 而 rank(C) < L/2
       意味着可达的 φ 被限制在低维子空间里, **无论怎么优化都到不了基态**（实测 E=+15.25）。"""
    from scipy.optimize import minimize          # 保留供诊断, 主路径不用
    qs = ns_momenta(L)
    if Lambda is None:
        Lambda = np.pi
    if n_s is None or u_IR is None:
        t = np.sort(np.log(np.array([qs[i] for i in range(L // 2)]) / Lambda))
        edges = np.concatenate(([t[0] - 2.0 if u_IR is None else u_IR], t[1:], [u_UV]))
        n_s = len(edges) - 1
        # ⚠️ 步长必须**分辨最窄的段**。阈值对齐网格的段宽差可达 20 倍（末段 ~0.07）,
        #    若 du 大于该宽度, 末段一个中点都取不到 ⇒ C 末列为 0 ⇒ 奇异, inv() 直接抛错。
        #    这不是数值洁癖: 它就是边界 5（χ 依赖格点）的同一条事实。
        w_min = float(np.min(np.diff(edges)))
        n_steps = max(n_s * sub, int(np.ceil((edges[-1] - edges[0]) / (0.5 * w_min))))
        du = (edges[-1] - edges[0]) / n_steps
        u_list = edges[0] + (np.arange(n_steps) + 0.5) * du
    else:
        edges, u_list, du = make_grid(u_IR, u_UV, n_s, sub)
    B = bond_matrices(L, qs, u_list, Lambda, prefactor, sign=sign)
    C = phase_matrix(L, qs, edges, u_list, du, Lambda, prefactor)
    Gam0 = vacuum_covariance(L)
    hmat_H = build_h(L, J, h)
    # ⚠️ 用 rank 守卫的伪逆, **不用 inv()**。C 奇异是**可预期的物理情形**, 不是异常:
    #    若某段一个中点都落不进任何模式的活动区, 该列恒为 0 ⇒ C 亏秩。
    #    inv() 在这种情形直接抛 LinAlgError（实测 sub=4 时发生）, 把一条**关于格点的事实**
    #    伪装成崩溃; pinv 则把它转成可读的 rank 报告。χ 的**形状**在这种情形无意义,
    #    而 φ 仍有意义 —— 这正是边界 5。
    # ⚠️ **φ 坐标只在 n_s == L//2 时才是良型的**。C 的形状是 (L//2) × n_s:
    #    每个模式恰好一个 φ, 故段数必须等于模式数。若 n_s > L//2, C 是**矩形**,
    #    pinv 会**静默接受**它, 然后在 `Cinv @ phi` 处报一个看不出根因的形状错
    #    （2026-09-24 实测: `size 10 is different from 8`, 来自 run_K4 的等宽网格）。
    #    这是把响亮失败换成隐晦失败的**回归**, 故在此显式拦下。用阈值对齐网格即可满足。
    if C.shape[1] != L // 2:
        raise ValueError(
            f"φ 坐标要求 n_s == L//2（每模式一个 φ）, 但收到 C{C.shape} ⇒ n_s={C.shape[1]}, "
            f"L//2={L // 2}。请用阈值对齐网格（n_s=None, u_IR=None）而不是等宽网格。")
    rank_C = int(np.linalg.matrix_rank(C))
    Cinv = np.linalg.pinv(C, rcond=1e-10)
    f = lambda phi: flow_energy(Cinv @ phi, B, u_list, edges, du, Gam0, hmat_H)
    # ⚠️ **不要用联合优化器在 χ 上跑**。两条实测理由:
    #  (1) Nelder-Mead 裸跑: 默认初始单纯形步长 = 0.00025·|x0| ≈ 1e-4 而 xatol=1e-6
    #      ⇒ 几步就"收敛", 实测 E=+15.25 vs E0=−20.40（相对差 1.75, 完全没动）。
    #  (2) 换扫描式坐标下降后仍在 L=16 卡在 −8.57 vs −20.40。根因是 **χ 不是好坐标**:
    #      E 只通过 φ=Cχ 依赖 χ, 而 C 病态（cond≈10）且末段极窄 ⇒ χ 的等值面是极扁的椭球。
    #  **正解**: 换到 φ 坐标。H 在 JW 形式下对 k 对角 ⇒ E = Σ_k f(φ_k) **可分离**
    #      ⇒ 逐分量 1-D 扫描即可。实测 L=8/16/32 全部给出相对差 ~4e-4,
    #      且该残差**只由扫描格点分辨率决定**（L=32 上 φ* 落在步长 π/60 的算术梯上,
    #      与 n_scan=61 的格距逐位相同）—— 加密格点即可收敛, 不是物理误差。
    phi = np.full(n_s, 0.7)
    for _ in range(n_iter):
        for k in range(n_s):
            grid = np.linspace(0.0, np.pi, n_scan)
            vals = []
            for v in grid:
                p = phi.copy()
                p[k] = v
                vals.append(f(p))
            phi[k] = grid[int(np.argmin(vals))]
    chi = Cinv @ phi
    # 每段实际中点数: C 满秩的**直接充分条件**是每段 ≥1 个中点落在某模式活动区。
    # 这是边界 5 的可测形式 —— 格点一变这个向量就变, 而 φ*/E 不变。
    n_mid = np.array([int(np.sum((u_list > edges[s]) & (u_list < edges[s + 1])))
                      for s in range(n_s)], dtype=int)
    env = dict(qs=qs, edges=edges, u_list=u_list, du=du, B=B, Gam0=Gam0,
               hmat_H=hmat_H, C=C, phi=phi, rank_C=rank_C, n_mid=n_mid)
    return chi, float(f(phi)), env


# ---------------------------------------------------------------------------
def run_K(L=16, J=1.0, h=1.0, n_s=None, u_IR=None, sub=8, Lambda=None, tag=''):
    """K1–K3: 一次优化 + 判据。Lambda 默认 = π（布里渊区边界, 使 u=0 时全部格点模式都活跃,
       否则 k>Λ 的模式永不参与流, K3 的能量闭环无意义）。"""
    if Lambda is None:
        Lambda = np.pi
    E0 = jw_ground_energy(L, J, h)
    chi, E, env = optimize_chi(L, J, h, n_s=n_s, u_IR=u_IR, sub=sub, Lambda=Lambda)
    edges = env['edges']
    n_s = len(edges) - 1
    ctr = 0.5 * (edges[:-1] + edges[1:])
    # K1/K2: χ 的形状（**判据的良定义性见边界 5**）
    flat = float(np.max(np.abs(chi - np.mean(chi))) / max(abs(np.mean(chi)), 1e-30))
    # IR 段（最靠 u_IR 的一半）做 log-线性拟合
    n_ir = max(2, n_s // 2)
    sl = float(np.polyfit(ctr[:n_ir], np.log(np.maximum(np.abs(chi[:n_ir]), 1e-30)), 1)[0])
    print(f"\n--- K {tag} L={L} J={J} h={h} n_s={n_s} u_IR={edges[0]:.3f} Λ={Lambda:.6f} ---")
    print(f"    优化能量 ⟨H⟩ = {E:.12f}   JW 解析 E0 = {E0:.12f}   相对差 = {abs(E-E0)/abs(E0):.3e}")
    print(f"    φ*(k) = " + " ".join(f"{v:.5f}" for v in env['phi']))
    print(f"    χ(u) 段中值: " + " ".join(f"{v:+.4f}" for v in chi))
    print(f"    χ 均值 = {np.mean(chi):.6f}  平坦度 max|χ−χ̄|/|χ̄| = {flat:.4f}"
          f"   (受末段窄 ⇒ C⁻¹ 放大影响, 见边界 5)")
    print(f"    IR 段 d log|χ|/du = {sl:.4f}   (K2 靶 = 2.0, 但见边界 5)")
    print(f"    [rank] rank(C) = {env['rank_C']}/{n_s}   du = {env['du']:.5f}"
          f"   各段中点数 = {' '.join(str(v) for v in env['n_mid'])}")
    print(f"    [K3] ⟨H⟩ ≥ E0 ? {'是 (变分上界 ✓)' if E >= E0 - 1e-9 else '**否** ⇒ 实现有 bug'}")
    return chi, E, E0, flat, sl, env


def run_K4(L=16, J=1.0, h=1.0, n_s=None, u_IR=None, sub=10, Lambda=None):
    """K4 负对照: 把 χ 换成**常数**（不随 u 变）, K1/K2 必须变差。
       ⚠️ 必须用**阈值对齐网格**（n_s=None, u_IR=None）。曾默认 `n_s=10, u_IR=-6.0` 的
          等宽网格 —— 那是 10 段 / 8 模式, C 成矩形, φ 坐标不成立, 实测崩在
          `Cinv @ phi`（见 `optimize_chi` 里的拦查注释）。"""
    if Lambda is None:
        Lambda = np.pi
    E0 = jw_ground_energy(L, J, h)
    _, E_opt, env = optimize_chi(L, J, h, n_s=n_s, u_IR=u_IR, sub=sub, Lambda=Lambda)
    n_sg = len(env['edges']) - 1
    # ⚠️ **扫描区间必须包住最优点, 且最优点必须落在内部**。首版扫 linspace(0.05,1.5,30),
    #    实测最优落在**端点 1.5000** ⇒ 那个数不是"最优常 χ", 是扫描边界,
    #    "变分胜出"的结论**没有被这个数建立起来**。这是典型的边界最优陷阱。
    #    现在两段扫描（粗扫定位 + 细化）, 并把"是否触端"作为一条显式判据打印出来。
    def E_of_const(c):
        return flow_energy(np.full(n_sg, c), env['B'], env['u_list'], env['edges'],
                           env['du'], env['Gam0'], env['hmat_H'])

    lo_c, hi_c = -6.0, 6.0
    grid = np.linspace(lo_c, hi_c, 121)
    vals = np.array([E_of_const(c) for c in grid])
    j = int(np.argmin(vals))
    best_const, E_const = float(grid[j]), float(vals[j])
    # 细化: 在极小点两侧再扫 40 点（若无极小点, 端点是单调的 ⇒ 如实报出）
    if 0 < j < len(grid) - 1:
        fine = np.linspace(grid[j - 1], grid[j + 1], 40)
        vf = np.array([E_of_const(c) for c in fine])
        jf = int(np.argmin(vf))
        best_const, E_const = float(fine[jf]), float(vf[jf])
        interior = 0 < jf < len(fine) - 1
    else:
        interior = False
    print(f"\n--- K4 负对照 (h={h}) ---")
    print(f"    变分 χ(u)   : ⟨H⟩ = {E_opt:.12f}  (相对 E0 {abs(E_opt-E0)/abs(E0):.3e})")
    print(f"    最优常 χ={best_const:.4f}: ⟨H⟩ = {E_const:.12f}  (相对 E0 {abs(E_const-E0)/abs(E0):.3e})")
    print(f"    [扫描] χ ∈ [{lo_c}, {hi_c}] 粗扫 121 + 细化 40; 最优点"
          f"{'在**区间内部** ✓' if interior else '**触端 ⇒ 未找到内部最优, 本条判据不成立**'}")
    if not interior:
        print(f"    ⚠️ **不许**据此行断言「u 方向有信息」: 常数 χ 的最优尚未找到。")
    print(f"    -> u 方向{'**有**信息 (变分胜出)' if (interior and E_opt < E_const - 1e-10) else '**本条未建立**'}")
    return E_opt, E_const


def run_K5(L=8, J=1.0, h=1.0, u_probe=-1.0, chi_const=0.6, Lambda=None, sub=10, n_s=10):
    """K5 内部自洽: 该模式的生成元的**方差** ⟨K_k²⟩−⟨K_k⟩² 是否等于 g_k(u)²。
       推导: K_k = −g_k σ_y（K−2 的 C1）, 末态为实态 ⇒ ⟨σ_y⟩=0, σ_y²=I ⇒ 方差 = g_k²。
       ⚠️ 推导**不当判据** —— 当场在稠密 2^L 上用**实际的 K_k 算符**算方差（L=8, 2^8 可行）。"""
    if Lambda is None:
        Lambda = np.pi
    dim = 2 ** L
    qs = ns_momenta(L)
    Gam = [majorana(a, L) for a in range(2 * L)]
    c = real_fermions(Gam, L)
    cq = fourier_modes(c, L, qs)
    edges, u_list, du = make_grid(u_IR=-6.0, u_UV=0.0, n_s=n_s, sub=sub)
    B = bond_matrices(L, qs, u_list, Lambda, True)
    Gam0 = vacuum_covariance(L)
    # 流到 u_probe（用稠密 psi 同步走一遍, 以得到真实的末态矢量）
    psi = uv_vacuum(L)
    G = Gam0
    for s, u in enumerate(u_list):
        if u > u_probe:
            break
        hmat = chi_const * B[s]
        R = expm(-du * hmat)
        G = R.T @ G @ R
        Kd = reconstruct(hmat, Gam)
        psi = expm(-1j * Kd * du) @ psi
    print(f"\n--- K5 方差 vs g_k²  (L={L}, u_probe={u_probe}, χ={chi_const}) ---")
    worst = 0.0
    for k in range(min(3, L // 2)):
        u = u_probe
        def g_of_k(kx, kk=float(qs[k])):
            return cutoff_prefactor(u, kx, Lambda, True) * chi_const if abs(kx - kk) < 1e-12 else 0.0
        hmat_k = pairing_hmat(L, qs, g_of_k)
        Kk = reconstruct(hmat_k, Gam)
        m1 = float(np.real(np.vdot(psi, Kk @ psi)))
        m2 = float(np.real(np.vdot(psi, Kk @ (Kk @ psi))))
        var = m2 - m1 ** 2
        gk = cutoff_prefactor(u, float(qs[k]), Lambda, True) * chi_const
        rel = abs(var - gk ** 2) / max(gk ** 2, 1e-30)
        worst = max(worst, rel)
        print(f"    k={qs[k]:.4f}: ⟨K²⟩−⟨K⟩² = {var:.10e}   g_k² = {gk**2:.10e}"
              f"   相对差 = {rel:.2e}   (⟨K⟩ = {m1:.2e})")
    print(f"    -> {'通过 (方差 = g_k², 与 K−1 的单模结论一致)' if worst < 1e-9 else '**不通过**'}")
    return worst


def run_K6(L=8, J=1.0, n_s=None, u_IR=None, sub=6, Lambda=None):
    """K6 与探针 2 对照: 探针 2 的相似度**随 h/J 反向**。
       本实现改用**保真度** |⟨Ψ_cMERA|Ψ_GS⟩|² 随 h/J 的走向, 稠密精确计算（L=8, 2^8 可行）。
       ⚠️ 「方向应当如何」是我定的操作化判据（探针 2 只说了它反转了, 没说正确方向) —— 见边界。"""
    if Lambda is None:
        Lambda = np.pi
    results = []
    for h in (0.2, 1.0, 2.5):
        E, V = np.linalg.eigh(tfi_matrix(L, J, h, periodic=False))
        gs = V[:, 0] / np.linalg.norm(V[:, 0])
        chi, Eopt, env = optimize_chi(L, J, h, n_s=n_s, u_IR=u_IR, sub=sub, Lambda=Lambda)
        # 由 chi 走流得到 Γ, 再与稠密基态的内积用**稠密态矢**算(不写高斯内积公式)
        Gam = [majorana(a, L) for a in range(2 * L)]
        psi = uv_vacuum(L)
        n_sg = len(env['edges']) - 1
        for s, u in enumerate(env['u_list']):
            # ⚠️ 必须用 searchsorted 找段号。曾误写成 `chi[s // sub]`（假设各段等宽）,
            #    而阈值对齐网格的段宽差 20 倍（3.10 vs 0.13）⇒ 段号全错, 实测把
            #    "第 2 段"的行为变成了"第 0 段"。等宽假设是这里唯一的正确性来源。
            idx = min(int(np.searchsorted(env['edges'], u, side='right') - 1), n_sg - 1)
            Kd = reconstruct(float(chi[idx]) * env['B'][s], Gam)
            psi = expm(-1j * Kd * env['du']) @ psi
        fid = float(abs(np.vdot(gs, psi)) ** 2)
        results.append((h, fid, Eopt))
        print(f"    h/J={h:4.2f}: 保真度 |⟨Ψ_cMERA|Ψ_GS⟩|² = {fid:.6f}   ⟨H⟩ = {Eopt:.9f}")
    hs = [r[0] for r in results]
    fs = [r[1] for r in results]
    print(f"    -> 保真度随 h/J 的走向: {'上升至临界后回落 (与探针 2 的反向**不同**)' if max(fs) > fs[0] and max(fs) > fs[-1] else '单调'} (max at h/J={hs[int(np.argmax(fs))]})")
    return results


def diagnostic_B(L=8):
    """诊断 B（§3.4.0）: 在离散 NS 动量格上建显式尺度算符, 检验 L|Ω⟩ = 0。
       **结果正负都登记, 不阻断主线。**
       构造: k∂_k 的**中心差分**近似, 模式对角部分 L = Σ_q λ_q c†_q c_q,
             λ_q 取该点在离散格上的差分权重。"""
    dim = 2 ** L
    qs = ns_momenta(L)
    Gam = [majorana(a, L) for a in range(2 * L)]
    c = real_fermions(Gam, L)
    cq = fourier_modes(c, L, qs)
    print(f"\n--- 诊断 B: 离散尺度算符 L, L=8 ---")
    # λ 的差分近似: 把 q 排序后 λ ~ q · df/dq 的离散版 —— 用在 c†_q c_q 上即模式重标号
    idx = np.argsort(qs)
    lam = np.zeros(L)
    for rank, i in enumerate(idx):
        q = qs[i]
        if 0 < rank < L - 1:
            dq = qs[idx[rank + 1]] - qs[idx[rank - 1]]
            lam[i] = q * (2.0 / dq)          # 中心差分的对角权重
    Lop = sum(lam[i] * (cq[i].conj().T @ cq[i]) for i in range(L))
    psi0 = uv_vacuum(L)
    res = float(np.linalg.norm(Lop @ psi0))
    herm = float(np.abs(Lop - Lop.conj().T).max())
    print(f"    ‖L|Ω⟩‖ = {res:.6e}   (‖|Ω⟩‖ = 1)")
    print(f"    ‖L − L†‖ = {herm:.6e}   (若 ~0 则 L 厄米 ⇒ 它**生成**酉变换, 不是标度流)")
    print(f"    -> L|Ω⟩=0 {'成立' if res < 1e-10 else '**不成立**'}")
    print(f"    ⚠️ **本条近乎平凡**: 任何 Σ λ_q c†_q c_q 都湮灭 c-真空（c_q|Ω⟩=0 ⇒ c†_qc_q|Ω⟩=0）。")
    print(f"       真正有内容的负面事实是上面那行厄米性判据: 离散差分给出的 L 是**厄米的**,")
    print(f"       而 cMERA 的标度算符应是**反厄米生成元**（它的 'L' 出现在 exp(−i(K+L)u) 里）。")
    print(f"       ⇒ 离散格上**不存在**一个既满足差分近似又反厄米的 k∂_k。这就是探针 2 的死因。")
    return res


# ===========================================================================
def selfcheck(L=8, J=1.0, h=1.0):
    """阶段 1 自检。两条:
       S1 升格保真: 用**本文件的函数**重跑 K−2 的已钉死数字（证明搬运没搬错）
       S2 纯 2L×2L ≡ 稠密: `pairing_hmat` vs `fermion_test(pairing_generator(...))`
           —— **这是阶段 1 的唯一目的**, 过了才允许在上面建 K1–K6。
       ⚠️ 全部真调本文件的函数, 不复制逻辑、不回读探针文件。"""
    t0 = time.time()
    dim = 2 ** L
    print(f"=== v16 · 高斯 cMERA 阶段 1 自检  [{_VERSION_TAG}] ===")
    print("口径 A: H_til = H_open + J·iγ_{2L-1}γ_0 (偶宇称扇区)")
    print("u 约定: IR→UV, u_UV=0（2104.01551 的 S7 方向; 与 JHEP 反向, û=−u）")
    print(f"L={L}, J={J}, h={h}, dim={dim}")

    Gam = [majorana(a, L) for a in range(2 * L)]
    I_dim = np.eye(dim, dtype=complex)
    chk = {}

    print("\n--- S1 升格保真（重跑 K−2 的已钉死数字） ---")
    c = real_fermions(Gam, L)
    w = 0.0
    for i in range(L):
        for j in range(L):
            w = max(w, float(np.abs(antic(c[i], c[j].conj().T) - (1.0 if i == j else 0.0) * I_dim).max()))
            w = max(w, float(np.abs(antic(c[i], c[j])).max()))
    chk['S1a A1 c_j 反对易'] = w < 1e-12
    print(f"    [S1a] max|{{c_i,c_j†}}−δ| = {w:.2e}   (K−2 记录 0.00e+00)"
          f"   -> {'通过' if chk['S1a A1 c_j 反对易'] else '不通过'}")

    qs = ns_momenta(L)
    cq = fourier_modes(c, L, qs)
    w = 0.0
    for a in range(L):
        for b in range(L):
            w = max(w, float(np.abs(antic(cq[a], cq[b].conj().T) - (1.0 if a == b else 0.0) * I_dim).max()))
    chk['S1b A2 c_q 反对易'] = w < 1e-12
    print(f"    [S1b] max|{{c_q,c_q'†}}−δ| = {w:.2e}   (K−2 记录 1.75e-15)"
          f"   -> {'通过' if chk['S1b A2 c_q 反对易'] else '不通过'}")

    g_const = 0.37
    Kd = pairing_generator(cq, L, lambda kk: g_const, sign=-1.0)
    Kd_wrong = pairing_generator(cq, L, lambda kk: g_const, sign=+1.0)
    h_ok = float(np.abs(Kd - Kd.conj().T).max())
    h_bad = float(np.abs(Kd_wrong + Kd_wrong.conj().T).max())
    chk['S1c B1 厄米性(钉符号)'] = h_ok < 1e-12 and h_bad < 1e-12
    print(f"    [S1c] |K−K†|(sign=−1) = {h_ok:.2e}  |K+K†|(sign=+1) = {h_bad:.2e}"
          f"   -> {'通过' if chk['S1c B1 厄米性(钉符号)'] else '不通过'}")

    res_k, hmat_k = fermion_test(Kd, Gam, dim)
    chk['S1d B2 保高斯性'] = res_k < 1e-10
    print(f"    [S1d] K 最差相对残差 = {res_k:.2e}   (K−2 记录 3.74e-16)"
          f"   -> {'通过' if chk['S1d B2 保高斯性'] else '不通过'}")

    psi0 = uv_vacuum(L)
    g0 = covariance(psi0, Gam)
    n0 = float(np.real(np.vdot(psi0, cq[0].conj().T @ (cq[0] @ psi0))))
    chk['S1e C1 |Ω⟩ 是 c-真空'] = abs(n0) < 1e-12
    print(f"    [S1e] |Ω⟩=|−⟩^⊗L 上 ⟨c†_q c_q⟩ = {n0:.2e}   (应 ~0; |+⟩^⊗L 会给出 1.0)"
          f"   -> {'通过' if chk['S1e C1 |Ω⟩ 是 c-真空'] else '不通过'}")

    theta = 1e-4
    psi_1 = expm(-1j * Kd * theta) @ psi0
    G_dense = covariance(psi_1, Gam)
    R = expm(-theta * hmat_k)
    dev_flow = float(np.abs(G_dense - (R.T @ g0 @ R)).max()) / max(float(np.abs(G_dense).max()), 1e-30)
    chk['S1f D 流步 RᵀΓR'] = dev_flow < 1e-8
    print(f"    [S1f] max|Γ_dense − RᵀΓ₀R| / |Γ| = {dev_flow:.2e}   (K−2 记录 5.55e-16)"
          f"   -> {'通过' if chk['S1f D 流步 RᵀΓR'] else '不通过'}")

    print("\n--- S2 纯 2L×2L 路径 ≡ 稠密路径（**阶段 1 的唯一目的**） ---")
    hmat_pure = pairing_hmat(L, qs, lambda kk: g_const)
    dev = float(np.abs(hmat_pure - hmat_k).max()) / max(float(np.abs(hmat_k).max()), 1e-30)
    asym = float(np.abs(hmat_pure + hmat_pure.T).max())
    chk['S2a pairing_hmat ≡ 稠密'] = dev < 1e-10
    chk['S2b pairing_hmat 反对称'] = asym < 1e-12
    print(f"    [S2a] |hmat_pure − hmat_dense| / |hmat_dense| = {dev:.2e}"
          f"   (|hmat_dense| = {float(np.abs(hmat_k).max()):.6f})")
    print(f"          -> {'一致 (纯 2L 路径可用)' if chk['S2a pairing_hmat ≡ 稠密'] else '**不一致**'}")
    print(f"    [S2b] |hmat_pure + hmat_pureᵀ| = {asym:.2e}"
          f"   -> {'通过 (so(2L) 生成元)' if chk['S2b pairing_hmat 反对称'] else '不通过'}")

    # S2c: 纯路径在 L=32 上真的跑得动（对照: 稠密路径在 L=32 需 2^32, 红线禁止）
    L_big = 32
    qs_big = ns_momenta(L_big)
    t_big = time.time()
    hb = pairing_hmat(L_big, qs_big, lambda kk: g_const)
    dt_big = time.time() - t_big
    mem_dense = (2 ** L_big) ** 2 * 16 / 1e9
    chk['S2c 纯路径可上 L=32'] = hb.shape == (2 * L_big, 2 * L_big) and dt_big < 60.0
    print(f"    [S2c] L=32: 纯路径 {hb.shape[0]}×{hb.shape[1]} = {hb.nbytes/1e6:.2f} MB,"
          f" 用时 {dt_big*1e3:.0f} ms")
    print(f"          对照: 稠密算符需 2^32×2^32 = {mem_dense:.3e} GB —— 红线禁止, 故纯路径是必需的")

    print("\n=== 阶段 1 判定 ===")
    for k, v in chk.items():
        print(f"    {k:<26}: {'通过' if v else '不通过'}")
    ok = all(chk.values())
    print(f"\n    阶段 1: {'通过 —— 纯 2L×2L 路径已钉死, 可进阶段 2 (K1–K6)' if ok else '未通过, 不得进阶段 2'}")

    print("\n=== 诚实边界 ===")
    print("  1. 本文件不建 L 算符(§3.4.0 主线 A)。**L|Ω⟩=0 在本口径下未被检验** ——")
    print("     不得写成「满足 cMERA 全部定义」。诊断 B(离散 L, L=8) 属阶段 2。")
    print("  2. d_k ≡ c†_{-q_k} 是逻辑推断(原文 d 是狄拉克反粒子), 非原文陈述。")
    print("  3. K1/K2 的靶 g(u) 来自 2104.01551(**玻色子**)。费米子版闭式原文没有")
    print("     (JHEP §3 整节无费米子内容, eq3.5i 是玻色子的) ⇒ 用玻色子靶检验费米子实现是**继承的假设**。")
    print("  4. 阶段 1 不产出任何 TFI 物理结论。g_uu / 熵 / 变分优化 均**尚未实现**。")
    print(f"\n用时 {time.time() - t0:.1f} s")
    return 0 if ok else 3


def selfcheck2(L=8, J=1.0, h=1.0, sub=8):
    """阶段 2 自检。补上三条此前**只在一次性探针里验过、文件内无对应检验**的断言
       （`pairing_hmat` 的 sign=+1 分支、`vacuum_covariance`、`phase_matrix` 的 docstring
       各自声称的 S2a2 / S3 / S4）。**全部真调本文件的函数**, 不复制逻辑。
       ⚠️ 为什么不并进 `selfcheck`: 那是阶段 1 的判据, 已冻结为 9/9; 往里加检查会
          改动一条已登记结论。阶段 2 的检验独立成条。"""
    t0 = time.time()
    dim = 2 ** L
    qs = ns_momenta(L)
    Gam = [majorana(a, L) for a in range(2 * L)]
    c = real_fermions(Gam, L)
    cq = fourier_modes(c, L, qs)
    chk = {}
    print(f"=== v16 · 高斯 cMERA 阶段 2 自检  [{_VERSION_TAG}] ===")

    # S2a2: sign=+1 分支（阶段 2 实际使用的那个）也要对稠密路径核验
    g_const = 0.37
    _, hmat_p = fermion_test(pairing_generator(cq, L, lambda kk: g_const, sign=+1.0), Gam, dim)
    hmat_pure_p = pairing_hmat(L, qs, lambda kk: g_const, sign=+1.0)
    dev_p = float(np.abs(hmat_pure_p - hmat_p).max()) / max(float(np.abs(hmat_p).max()), 1e-30)
    chk['S2a2 sign=+1 ≡ 稠密'] = dev_p < 1e-10
    print(f"    [S2a2] |hmat_pure(+1) − hmat_dense(+1)| / |·| = {dev_p:.3e}"
          f"   -> {'通过' if chk['S2a2 sign=+1 ≡ 稠密'] else '不通过'}")

    # S3: vacuum_covariance 的解析式对稠密 covariance(uv_vacuum) 核验（docstring 声称的 S3）
    G0_an = vacuum_covariance(L)
    G0_de = covariance(uv_vacuum(L), Gam)
    dev_v = float(np.abs(G0_an - G0_de).max())
    chk['S3 Γ(|Ω⟩) 解析 ≡ 稠密'] = dev_v < 1e-12
    print(f"    [S3] max|Γ_解析 − Γ_稠密| = {dev_v:.3e}"
          f"   -> {'通过' if chk['S3 Γ(|Ω⟩) 解析 ≡ 稠密'] else '不通过'}")

    # S4: 由目标 φ* 解 χ, 末态必须给出 ⟨n_q⟩ = sin²φ* 且 ⟨H⟩ = E0
    #     —— 这条同时钉住 `phase_matrix`（φ = Σ C χ 的说法）与 `bond_matrices`+`flow_energy`。
    chi, E, env = optimize_chi(L, J, h, sub=sub, Lambda=np.pi)
    psi = uv_vacuum(L)
    n_sg = len(env['edges']) - 1
    for s, u in enumerate(env['u_list']):
        idx = min(int(np.searchsorted(env['edges'], u, side='right') - 1), n_sg - 1)
        psi = expm(-1j * reconstruct(float(chi[idx]) * env['B'][s], Gam) * env['du']) @ psi
    d_n = 0.0
    for a in range(L // 2):
        nq = float(np.real(np.vdot(psi, cq[a].conj().T @ (cq[a] @ psi))))
        d_n = max(d_n, abs(nq - np.sin(env['phi'][a]) ** 2))
    chk['S4 ⟨n_q⟩ = sin²φ*'] = d_n < 1e-9
    print(f"    [S4a] max|⟨n_q⟩_稠密 − sin²φ*| = {d_n:.3e}"
          f"   -> {'通过' if chk['S4 ⟨n_q⟩ = sin²φ*'] else '不通过'}")
    E0 = jw_ground_energy(L, J, h)
    rel = abs(E - E0) / abs(E0)
    chk['S4b ⟨H⟩ 闭合到 E0'] = rel < 1e-3
    print(f"    [S4b] |⟨H⟩ − E0|/|E0| = {rel:.3e}   (⟨H⟩={E:.12f}, E0={E0:.12f})"
          f"   -> {'通过' if chk['S4b ⟨H⟩ 闭合到 E0'] else '不通过'}")

    print("\n=== 阶段 2 自检判定 ===")
    for k, v in chk.items():
        print(f"    {k:<26}: {'通过' if v else '不通过'}")
    ok = all(chk.values())
    print(f"\n    阶段 2 自检: {'通过' if ok else '未通过'}")
    print(f"\n用时 {time.time() - t0:.1f} s")
    return 0 if ok else 3


if __name__ == '__main__':
    # Windows 控制台默认可能是 GBK, 而本文件的输出含 û / γ / ⟩ / 中文。
    # 不设这一行会随机 UnicodeEncodeError（2026-09-24 实测到一次, `û` 触发）。
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        pass
    stage = sys.argv[1] if len(sys.argv) > 1 else 'stage1'
    t0 = time.time()
    if stage == 'stage1':
        sys.exit(selfcheck())
    if stage == 'stage2check':
        sys.exit(selfcheck2())
    print(f"=== v16 · 高斯 cMERA 阶段 2  [{_VERSION_TAG}] ===")
    print("口径 A: H_til = H_open + J·iγ_{2L-1}γ_0 (偶宇称扇区)")
    print("变分原理（**选择, 非原文操作化定义**）: 极小化末态 ⟨H_til⟩ over 分段常数 χ(u)")
    print("K1/K2 的靶是**裸量 χ**（与原文行 104 玻色子 `g_k^B=Γ·g^B` 无前因子对应）; 口径选择, 见文件头")
    run_K(L=8, J=1.0, h=1.0, tag='冒烟 L=8 临界')
    run_K(L=16, J=1.0, h=1.0, tag='K1 临界 h=J')
    run_K(L=16, J=1.0, h=0.5, tag='K2 有质量 h/J=0.5')
    run_K(L=32, J=1.0, h=1.0, tag='K1 复验 L=32（纯 2L×2L, 红线内）')
    run_K5(L=8)
    diagnostic_B(L=8)
    # 边界 5 的**证据**: χ 的形状随格点变, φ* 与 ⟨H⟩ 不变 ⇒ χ 不是良定义的物理量
    print("\n--- 边界 5 证据: 加密求积格点, χ 变而 φ*/E 不变 ---")
    # ⚠️ sub 必须大到让 n_steps 真的不同。`optimize_chi` 现在会把 n_steps **下钳到
    #    分辨最窄段所需的最小值**（L=16 时 = 148）, 故 sub=4 与 sub=16 都落到同一个
    #    148 ⇒ 比较无意义。取 sub=4（被钳到 148）vs sub=32（=256, 高于钳位）才真的换了格点。
    for sub in (4, 32):
        chi_2, E_2, env_2 = optimize_chi(16, 1.0, 1.0, sub=sub)
        print(f"    sub={sub:2d}: ⟨H⟩={E_2:+.10f}  φ* 前 3 = "
              f"{' '.join(f'{v:.5f}' for v in env_2['phi'][:3])}  χ 前 3 = "
              f"{' '.join(f'{v:+.5f}' for v in chi_2[:3])}  末段 χ = {chi_2[-1]:+.3f}"
              f"  rank(C)={env_2['rank_C']}  n_steps={len(env_2['u_list'])}")
    # K4 负对照（此前只实现未跑）
    run_K4(L=16, J=1.0, h=1.0)
    # K6: 稠密重建末态、对基态算保真度（h 由 run_K6 内部扫, 不传参）
    run_K6(L=8, J=1.0)
    print(f"\n总用时 {time.time() - t0:.1f} s")
    sys.exit(0)
