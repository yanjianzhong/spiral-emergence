# -*- coding: utf-8 -*-
"""
_v16_smoke_cmera5.py —— v16 · 费米子 cMERA 生成元转录验证（K−1 的费米子对应物）

为什么需要这一步:
  K−1（`_v16_smoke_cmera4.py`）用**玻色子**基准把机器校准了。换费米子生成元之后,
  出错的位置会从"物理错"位移到"转录错" —— 而两者在输出上长得一样。
  本探针对应 `v16_plan.md` §9 风险 10 的防线①: 在进 K1–K6 之前先把生成元钉死。

生成元的来源（**逐行读过原文, 非记忆; 见 v16_plan.md §3.3.5**）:
  JHEP09(2015)002, 文件 v16/cMERA_Entropy_rev.tex
    eq1.8  (行 114): K̃_F = i∫dk( g_k^F c†_k d_k + g_k^{F*} c_k d†_k )
    eq1.8  (行 116): g_k^F = (k e^u/Λ)·Γ(k e^û/Λ)·g^F(û,k)        <- 多一个 (k e^u/Λ)
    eq1.9  (行 119): c_k|0⟩_F = d†_k|0⟩_F = 0                      <- d 是空穴算符(海是满的)
    eq2.10 (行 253): |Ψ_k⟩ = (1+γ_k)^{-1/2}(|0_c,0_d⟩ + γ_k^{1/2}|1_c,1_d⟩)
    eq2.12 (行 265): S_k = log(1+γ_k) - γ_k/(1+γ_k)·log γ_k
    eq2.16 (行 292): 相对熵 = 4 g_k(u)^2 du^2

格点识别（本文档**唯一的推断**, 必须显式声明）:
  TFI 是**实(Majorana)费米子链**, 没有狄拉克反粒子。原文 (c_k, d_k) 是"粒子/反粒子"两分量。
  我们的识别是:   d_k  ≡  c†_{-q_k}
  —— 把反粒子分量认成"同一支在 -q 的空穴"。三条支持(均可当场核验):
    ① 结构: |A|²+|B|²=1 与"只有两个 Fock 态"((c†d)²=0) 在 BCS/BdG 配对下同样成立;
    ② v15 的 NS 动量集合 q_k=π(2k+1)/L 在 q→-q 下封闭(v15/spiral_model_v15.py:743);
    ③ 原文自称这是 "opposite momenta (left-right moving modes)" 的纠缠(行 208)。
  ⚠️ 这是**逻辑推断**, 不是原文陈述。若后续 K1–K6 与它对不上, 先怀疑这一条。

符号的自证（**不靠记忆**）:
  eq1.8 里第二项写作 c_k d†_k, 而 c_k d†_k = -d†_k c_k。故 eq1.8 = i g(c†_k c†_{-k} - c_{-k} c_k),
  **减号**。若误写成加号: K 变反厄米、U 非酉、⟨n_q⟩ 按 sinh² 冲过 1 —— 可检测。
  B1(厄米性) 与 C1(⟨n⟩=sin²≤1) 合起来**把符号钉死**, 不依赖我记住哪个对。

玻色子 vs 费米子的判别性差别（C 部分专测, 可证伪）:
  玻色子: |A|²-|B|²=1 (双曲), γ = tanh²(gt)
  费米子: |A|²+|B|²=1 (三角), γ = tan²(gt)          <- 与 (c†d)²=0 直接绑定

边界:
  - 本探针**只**回答"eq1.8 转录到我们格点上对不对", 不产出任何 TFI 物理结论。
  - **不建 L 算符**: 尺度算符在有限格点上无良定义(v16_plan.md §3.0, 探针 2 的死因)。
    本探针把 u 依赖留给 g_k(u) 的 cutoff, 不动 L。L 的处置是独立的口径问题。
  - 全部核对走稠密 2^L (L=8, dim=256); 2L×2L 的费米子高斯表示只用于**对照**。
"""

import os
import sys
import time

import numpy as np
from scipy.linalg import expm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'v15'))

# 真调探针 3 的地基函数（不复制逻辑 —— v16_plan.md 硬纪律 2）
from _v16_smoke_cmera3 import (  # noqa: E402
    build_h, covariance, fermion_test, gs_energy_from_h, majorana,
    op_at, reconstruct, tfi_matrix, _I2, _SX, _SY, _SZ,
)
from spiral_model_v15 import jw_ground_energy  # noqa: E402

_VERSION_TAG = 'v16-smoke-cmera5-1'

_LAMBDA = 1.0          # UV cutoff（原文 Λ; 本探针量纲无关, 取 1）


# ---------------------------------------------------------------------------
# A. 动量模式: 全部当场稠密核验, 不预设任何 JW / Fourier 公式
# ---------------------------------------------------------------------------
def real_fermions(Gam, L):
    """c_j = (γ_{2j} + i γ_{2j+1})/2 。
    本式**不当假设** —— A1 当场用反对易子核验 (cp. 探针 3 的 T1 对 Majorana 代数的处理)。"""
    return [(Gam[2 * j] + 1j * Gam[2 * j + 1]) / 2.0 for j in range(L)]


def ns_momenta(L):
    """v15 的 NS 动量集合 (v15/spiral_model_v15.py:745, 已对稠密 ED 核到 2.5e-14)。
    偶宇称扇区 ⇒ 反周期边界 ⇒ 半整数动量; 且 q→-q 封闭: qs[L-1-k] = -qs[k]。"""
    return np.pi * (2.0 * np.arange(L) + 1.0) / L


def fourier_modes(c, L, qs):
    """c_q = L^{-1/2} Σ_j e^{-i q j} c_j 。A2 当场核验其正则反对易关系。"""
    return [sum(np.exp(-1j * q * j) * c[j] for j in range(L)) / np.sqrt(L) for q in qs]


def antic(A, B):
    return A @ B + B @ A


# ---------------------------------------------------------------------------
# B. eq1.8 的生成元（格点转录）
# ---------------------------------------------------------------------------
def pairing_generator(cq, L, g_of_k, sign=-1.0):
    """eq1.8 的格点转录（配对型, sign=-1 为原文正确符号, 见模块头「符号的自证」）。

        K = i Σ_{pairs} g_k ( c†_{q_k} c†_{-q_k}  +  sign · c_{-q_k} c_{q_k} )

    pairs 取 q_k 与 -q_k 配成一对, k 只跑一半 (k=0..L/2-1), 否则重复计数。
    g_of_k: 可调用, 入参为 |q_k|, 返回该对的 g。含原文的 (k e^u/Λ) 前因子时在此体现。
    """
    K = np.zeros_like(cq[0])
    for k in range(L // 2):
        a, b = k, L - 1 - k                      # qs[b] = -qs[a]
        g = g_of_k(abs(np.pi * (2 * k + 1) / L))
        A = cq[a].conj().T @ cq[b].conj().T      # c†_{q_k} c†_{-q_k}
        K = K + 1j * g * (A + sign * A.conj().T)
    return K


def uv_vacuum(L):
    """eq1.9 的 UV 真空 |Ω⟩ (即 c_q|Ω⟩=0)。
    探针 3 的约定给出 σx_i = i γ_{2i}γ_{2i+1}; 由 c_j=(γ_{2j}+iγ_{2j+1})/2 展开,
        c†_j c_j = (I + i γ_{2j}γ_{2j+1})/2 = (I + σx_j)/2    ⇒   n_j = (1+σx_j)/2
    ⇒ **c-真空是全 σx = -1 ⇒ |Ω⟩ = |−⟩^{⊗L}** (不是 |+⟩^{⊗L}!)。
    ⚠️ v1 误取 |+⟩^{⊗L} —— 那是全满态(n_j≡1), 恰好是真空的补, 使 C1/C2 的**参考态**错。
       实测 ⟨n_q(t)⟩ 与 cos²(gt) 逐点吻合, 证明是参考态错、**生成元与符号都对**。
    注: |−⟩^{⊗L} 与 B1 的 equal_state(|+⟩^{⊗L}) 是**同一族在两个极限**, 不是同一个态。"""
    s = np.arange(2 ** L, dtype=np.int64)
    pc = np.zeros(2 ** L, dtype=np.int64)
    for b in range(L):
        pc += (s >> b) & 1
    return ((-1.0) ** pc) / np.sqrt(2.0 ** L)


# ---------------------------------------------------------------------------
def main():
    t_start = time.time()
    print(f"=== v16 探针 · 费米子 cMERA 生成元转录验证  [{_VERSION_TAG}] ===")
    print("口径: H_til = H_open + J·iγ_{2L-1}γ_0 (偶宇称扇区, v16_plan.md §3.2 口径 A)")
    print("生成元来源: JHEP09(2015)002 eq1.8 (v16/cMERA_Entropy_rev.tex 行 114/116)")
    print("识别声明: d_k ≡ c†_{-q_k}  —— 逻辑推断, 非原文陈述, 见模块头")

    L, J, h = 8, 1.0, 1.0
    dim = 2 ** L
    print(f"\nL={L}, J={J}, h={h}, dim={dim}")

    Gam = [majorana(a, L) for a in range(2 * L)]
    I_dim = np.eye(dim, dtype=complex)

    # =======================================================================
    print("\n--- A 部分: 动量模式 (全部当场稠密核验) ---")

    print("\n[A1] c_j = (γ_{2j} + i γ_{2j+1})/2 的正则反对易关系")
    c = real_fermions(Gam, L)
    w_ab = 0.0
    for i in range(L):
        for j in range(L):
            w_ab = max(w_ab, float(np.abs(antic(c[i], c[j].conj().T)
                                            - (1.0 if i == j else 0.0) * I_dim).max()))
            w_ab = max(w_ab, float(np.abs(antic(c[i], c[j])).max()))
    t_a1 = w_ab < 1e-12
    print(f"    max |{{c_i,c_j†}} - δ_ij| 与 |{{c_i,c_j}}| = {w_ab:.2e}"
          f"   -> {'通过' if t_a1 else '不通过'}")

    print("\n[A2] c_q = L^(-1/2) Σ_j e^(-i q j) c_j 的反对易关系 (NS 动量集合)")
    qs = ns_momenta(L)
    cq = fourier_modes(c, L, qs)
    w_q = 0.0
    for a in range(L):
        for b in range(L):
            w_q = max(w_q, float(np.abs(antic(cq[a], cq[b].conj().T)
                                         - (1.0 if a == b else 0.0) * I_dim).max()))
    t_a2 = w_q < 1e-12
    print(f"    max |{{c_q,c_q'†}} - δ_qq'| = {w_q:.2e}   -> {'通过' if t_a2 else '不通过'}")
    print(f"    q→-q 封闭性: max_k |q_k + q_(L-1-k)| = "
          f"{float(np.abs(qs[:L // 2] + qs[L - 1:L // 2 - 1:-1]).max()):.2e}")

    print("\n[A3] 色散: H_til 在 NS 动量上是否对角化 (让代码判定口径, 不假设)")
    # build_h 的 2L×2L 表示已由探针 3 的 T9a 对稠密 hmat_til 核到 1e-12 内
    h_small = build_h(L, J, h)
    w_q_disp = np.sqrt(J ** 2 + h ** 2 - 2.0 * J * h * np.cos(qs))
    e_from_h = gs_energy_from_h(h_small)
    e_from_w = float(-np.sum(w_q_disp))
    t_a3 = abs(e_from_h - e_from_w) < 1e-10
    print(f"    -Σ_k ω(q_k) = {e_from_w:.12f}   E_gs(h) = {e_from_h:.12f}"
          f"   偏差 = {abs(e_from_h - e_from_w):.2e}")
    print(f"    v15 JW 解析值 = {jw_ground_energy(L, J, h):.12f}")
    print(f"    -> {'一致 (NS 色散正确)' if t_a3 else '**不一致**'}")

    # =======================================================================
    print("\n--- B 部分: 生成元 eq1.8 的结构核验 ---")

    g_const = 0.37
    K = pairing_generator(cq, L, lambda kk: g_const, sign=-1.0)
    K_wrong = pairing_generator(cq, L, lambda kk: g_const, sign=+1.0)

    print("\n[B1] 厄米性: 原文符号(-) 应厄米; 误写符号(+) 应反厄米")
    h_ok = float(np.abs(K - K.conj().T).max())
    h_bad = float(np.abs(K_wrong + K_wrong.conj().T).max())
    t_b1 = h_ok < 1e-12 and h_bad < 1e-12
    print(f"    |K - K†|          (sign=-1) = {h_ok:.2e}   <- 应 ~0")
    print(f"    |K + K†|          (sign=+1) = {h_bad:.2e}   <- 应 ~0 (反厄米)")
    print(f"    -> {'通过 (符号被钉死)' if t_b1 else '不通过'}")

    print("\n[B2] 保高斯性: [K, γ_c] 是否落在 span{γ} 内 (探针 3 的 fermion_test)")
    res_k, hmat_k = fermion_test(K, Gam, dim)
    res_bad, _ = fermion_test(K_wrong, Gam, dim)
    t_b2 = res_k < 1e-10
    print(f"    K(sign=-1) 最差相对残差 = {res_k:.2e}  -> {'通过 (纯二次)' if t_b2 else '不通过'}")
    print(f"    K(sign=+1) 最差相对残差 = {res_bad:.2e}  (也是二次的, 故 B1 才是判据)")

    print("\n[B3] 2L×2L 费米子高斯表示: reconstruct(hmat_k) 是否重建 K")
    re_k = reconstruct(hmat_k, Gam)
    dev_re = float(np.abs(re_k - K).max()) / max(float(np.abs(K).max()), 1e-30)
    t_b3 = dev_re < 1e-10
    print(f"    |reconstruct(K) - K| / |K| = {dev_re:.2e}   -> {'通过' if t_b3 else '不通过'}")
    asym = float(np.abs(hmat_k + hmat_k.T).max())
    print(f"    hmat_k 反对称性 |h + hᵀ| = {asym:.2e}  (so(2L) 生成元应反对称)")

    # =======================================================================
    print("\n--- C 部分: 三角 vs 双曲 (费米子的判别性特征) ---")

    print("\n[C1] U(t)|Ω⟩ 的占据数: 费米子应 sin²(gt); 玻色子才会 sinh²(gt)")
    psi0 = uv_vacuum(L)
    n0 = float(np.real(np.vdot(psi0, cq[0].conj().T @ (cq[0] @ psi0))))
    print(f"    |Ω⟩=|−⟩^⊗L 上的 ⟨c†_q c_q⟩ = {n0:.2e}  (eq1.9 的 c-真空, 应 ~0)")

    t_list = np.array([0.0, 0.3, 0.7, 1.1, 1.5])
    worst_sin, worst_sinh = 0.0, 0.0
    for t in t_list:
        psi_t = expm(-1j * K * t) @ psi0
        nq = float(np.real(np.vdot(psi_t, cq[0].conj().T @ (cq[0] @ psi_t))))
        d_sin = abs(nq - np.sin(g_const * t) ** 2)
        d_sinh = abs(nq - np.sinh(g_const * t) ** 2)
        worst_sin = max(worst_sin, d_sin)
        worst_sinh = max(worst_sinh, d_sinh)
        print(f"    t={t:4.1f}: <n_q> = {nq:.9f}   sin²(gt) = {np.sin(g_const*t)**2:.9f}"
              f"  |Δ|={d_sin:.2e}    sinh²(gt) = {np.sinh(g_const*t)**2:.9f}  |Δ|={d_sinh:.2e}")
    t_c1 = worst_sin < 1e-10
    print(f"    -> 最差: 对 sin² {worst_sin:.2e} / 对 sinh² {worst_sinh:.2e}"
          f"   => {'三角关系成立 (费米子)' if t_c1 else '不成立'}")

    print("\n[C2] eq2.10 的两态结构 + eq2.12 熵")
    t_s = 0.9
    psi_t = expm(-1j * K * t_s) @ psi0
    # 取一对 (q_k, -q_k) 的占据数: 配对结构要求两模相等
    occ = []
    for a in (0, L - 1):
        occ.append(float(np.real(np.vdot(psi_t, cq[a].conj().T @ (cq[a] @ psi_t)))))
    gamma_t = np.tan(g_const * t_s) ** 2
    P11 = gamma_t / (1.0 + gamma_t)
    t_c2a = abs(occ[0] - P11) < 1e-10 and abs(occ[1] - P11) < 1e-10
    print(f"    t={t_s}: γ = tan²(gt) = {gamma_t:.9f}   γ/(1+γ) = {P11:.9f}")
    print(f"    实测 <n_q0> = {occ[0]:.9f}   <n_-q0> = {occ[1]:.9f}   (两模相等 ⇒ 配对结构)")
    print(f"    |A|²+|B|² = cos²+sin² = {np.cos(g_const*t_s)**2 + np.sin(g_const*t_s)**2:.12f}"
          f"   -> {'三角归一 (eq1.15/1.16)' if t_c2a else '不成立'}")
    s_k = np.log(1 + gamma_t) - gamma_t / (1 + gamma_t) * np.log(gamma_t)
    # 稠密独立算: 该对子空间的 2x2 密度矩阵的 von Neumann 熵
    p_pair = np.array([1 / (1 + gamma_t), gamma_t / (1 + gamma_t)])
    s_dense = float(-np.sum(p_pair * np.log(p_pair)))
    t_c2 = t_c2a and abs(s_k - s_dense) < 1e-12
    print(f"    eq2.12 S_k = {s_k:.12f}   稠密 2x2 熵 = {s_dense:.12f}"
          f"   偏差 = {abs(s_k - s_dense):.2e}")

    # =======================================================================
    print("\n--- D 部分: K0 流步 —— 稠密 U vs 2L×2L 的 RΓRᵀ ---")

    g0 = covariance(psi0, Gam)
    dev_pure0 = float(np.abs(g0 @ g0 + np.eye(2 * L)).max())
    print(f"\n[D1] |Ω⟩ 的 Γ 纯高斯性 |Γ²+I| = {dev_pure0:.2e}  (应 ~0)")

    # 先钉住 [K, γ_c] = i Σ_a hmat[a,c] γ_a 这条关系本身(不靠约定), 再做流步比较
    lhs = K @ Gam[0] - Gam[0] @ K
    rhs = 1j * sum(hmat_k[a, 0] * Gam[a] for a in range(2 * L))
    dev_comm = float(np.abs(lhs - rhs).max()) / max(float(np.abs(lhs).max()), 1e-30)
    print(f"\n[D2a] |[K,γ_0] - iΣ_a hmat[a,0]γ_a| / |[K,γ_0]| = {dev_comm:.2e}")
    print(f"      hmat_k 虚部 max = {float(np.abs(hmat_k.imag).max()):.2e} (应 ~0);"
          f"  max|hmat_k| = {float(np.abs(hmat_k).max()):.6f}")

    theta = 1e-4
    psi_1 = expm(-1j * K * theta) @ psi0
    G_dense = covariance(psi_1, Gam)
    dG = G_dense - g0
    # 由 γ'_a = U†γ_aU = Σ_c R_ca γ_c  —— **变换指标在第二位**
    #   ⇒ Γ'_ab = Σ_cd R_ca Γ_cd R_db = (Rᵀ Γ R)_ab   (**不是** R Γ Rᵀ!)
    #   ⇒ 一阶 dΓ = θ(hΓ - Γh)                        [h 反对称]
    #   v1 误写 RΓRᵀ ⇒ 预测 -θ(hΓ+Γh); 在 hΓ≈-Γh 的真空上该预测≈0, 实测比值恰好 1.00, 由此定位。
    pred1 = theta * (hmat_k @ g0 - g0 @ hmat_k)
    dev1 = float(np.abs(dG - pred1).max()) / max(float(np.abs(dG).max()), 1e-30)
    print(f"[D2b] |dΓ_dense - θ(hΓ-Γh)| / |dΓ_dense| = {dev1:.2e}"
          f"   (|dΓ_dense| = {float(np.abs(dG).max()):.3e})")

    R = expm(-theta * hmat_k)                  # 由 U†γ_cU = Σ_a (I-θk)_ac γ_a 导出
    G_gauss = R.T @ g0 @ R                     # Γ' = Rᵀ Γ R
    dev_flow = float(np.abs(G_dense - G_gauss).max())
    scale = float(np.abs(G_dense).max())
    t_d = dev_flow / max(scale, 1e-30) < 1e-8
    print(f"[D2c] max|Γ_dense - R Γ0 Rᵀ| = {dev_flow:.2e}"
          f"   (|Γ| ~ {scale:.3f})  相对 = {dev_flow/scale:.2e}")
    print(f"      -> {'一致 (高斯路径可用)' if t_d else '**不一致**'}")

    # =======================================================================
    print("\n--- E 部分: eq2.16 相对熵 vs 4 g² du² (如实报告, 不调) ---")

    def rel_entropy(gam_a, gam_b):
        """S(ρ̄||ρ) 对两个对角 2x2 密度矩阵, 按 eq2.14 定义逐字计算。"""
        p = np.array([1 / (1 + gam_a), gam_a / (1 + gam_a)])
        q = np.array([1 / (1 + gam_b), gam_b / (1 + gam_b)])
        return float(np.sum(p * np.log(p / q)))

    def rel_entropy_boson(gam_a, gam_b):
        """eq2.15 的玻色子版: ρ_k = Σ_n γ^n(1-γ)|n⟩⟨n| (热分布, **无限维**)。
        S(ρ_a||ρ_b) = log((1-γ_a)/(1-γ_b)) + γ_a/(1-γ_a)·log(γ_a/γ_b)。
        加这条是为了**判别** E 的因子 2 是文献相对熵口径, 还是费米子特有。"""
        return float(np.log((1 - gam_a) / (1 - gam_b))
                     + gam_a / (1 - gam_a) * np.log(gam_a / gam_b))

    du = 1e-5
    pred = 4.0 * g_const ** 2 * du ** 2
    print(f"    参考: 文献 eq2.16/2.15 预测值 4g²du² = {pred:.6e}")
    for t_probe in (0.2, 0.8, 1.4):
        ga_f = np.tan(g_const * t_probe) ** 2
        gb_f = np.tan(g_const * (t_probe + du)) ** 2
        s_f = rel_entropy(gb_f, ga_f)
        ga_b = np.tanh(g_const * t_probe) ** 2
        gb_b = np.tanh(g_const * (t_probe + du)) ** 2
        s_b = rel_entropy_boson(gb_b, ga_b)
        print(f"    t={t_probe:4.1f}: 费米子 S = {s_f:.6e} (比值 {s_f/pred:.6f})"
              f"   玻色子 S = {s_b:.6e} (比值 {s_b/pred:.6f})")

    # =======================================================================
    checks = {
        'A1 c_j 反对易': t_a1, 'A2 c_q 反对易': t_a2, 'A3 NS 色散': t_a3,
        'B1 厄米性(钉符号)': t_b1, 'B2 保高斯性': t_b2, 'B3 2L×2L 重建': t_b3,
        'C1 三角 sin²(gt)': t_c1, 'C2 eq2.10/2.12': t_c2, 'D  K0 流步': t_d,
    }
    print("\n=== 判定 ===")
    for k, v in checks.items():
        print(f"    {k:<22}: {'通过' if v else '不通过'}")
    ok = all(checks.values())
    print(f"\n    生成元转录: {'可用 —— eq1.8 已在我们格点上钉死' if ok else '未通过, 不得进入 K1–K6'}")

    print("\n=== 诚实边界 ===")
    print("  1. d_k ≡ c†_{-q_k} 是**逻辑推断**(原文是狄拉克反粒子, TFI 无此物), 非原文陈述。")
    print("  2. 本探针**不建 L 算符**。有限格点上尺度算符无良定义(v16_plan.md §3.0, 探针 2 死因)。")
    print("     u 依赖只通过 g_k(u) 的 cutoff 与 (k e^u/Λ) 前因子进入; L 的处置是独立口径问题。")
    print("  3. E 部分的比值**只报数**, 不做判据 —— eq2.16 的 g 是否含 (k e^u/Λ) 前因子")
    print("     需要原文上下文才能定, 不在本探针的射程内。")
    print("  4. 本探针不产出任何 TFI 物理结论。")

    print(f"\n用时 {time.time() - t_start:.1f} s")
    return 0 if ok else 3


if __name__ == '__main__':
    sys.exit(main())
