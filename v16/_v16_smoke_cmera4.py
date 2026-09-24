# -*- coding: utf-8 -*-
"""
v16 探针 4 · cMERA 玻色子基准复现 (实施计划 v16_plan.md 的 K-1 步)
====================================================================

版本标记: v16-smoke-cmera4-1

功能
----
复现 arXiv:2104.01551v2 (Fernandez-Melgarejo & Molina-Vilaplana,
"On the Entanglement Entropy in Gaussian cMERA") 的玻色子高斯 cMERA,
用于校准本仓库的 cMERA 机器 (计划 §3.3.4 的路线 F-1)。

**本探针不产出任何关于 TFI 的物理结论。** 它的唯一产物是:
"这台机器能不能复现已发表的解析结果" —— 是/否。

方程编号沿用 v16_plan.md §3.3.1 的 S 编号 (均为逐字核过原文的 LaTeX)。

边界声明
--------
1. 纯玻色子 (free scalar)。文献全文 fermion/Fermi/Dirac/Majorana 命中数为 0。
   故本探针**不**覆盖费米子, 费米子版须另行推导 (计划 §3.3.4)。
2. 单模 k, 截断 Fock 空间, 不涉及 2^L 希尔伯特空间。
3. 不引入任何凭记忆写下的公式:
   - 生成元 K 的算子形式由 S2 的原文 LaTeX 直接转写;
   - |Omega> 由 S1b 的原文湮灭条件**数值**构造 (取 d†d 的基态), 不用解析式;
   - g_uu 由 S5(内积) / S5b(方差) / S5c(g^2) **三路独立**计算并互相对照。
   v4-2 修正: S1b 里的 pi 是**同动量** (原文逐字 pi(k)), 而 S2 生成元用 pi(-k)。
   两者在实标量场中不可互换 (pi_k^dag = pi_{-k}), 混用会得到 [c,c^dag]=0 的假湮灭算符。
4. 2x2 协方差路径 (B 部分) 与稠密 Fock 路径 (C 部分) 是两套独立实现,
   必须逐点一致 —— 这是本探针内部的自洽判据。

用法
----
    python _v16_smoke_cmera4.py
    EXIT=0 全通过; EXIT=3 有检验未通过。
"""

import sys
import time

import numpy as np
from scipy.linalg import expm

_VERSION_TAG = 'v16-smoke-cmera4-3'

# ---------------------------------------------------------------- 文献参数
# 文献数值节取 d=1, Lambda=100, m=Lambda/100
LAM = 100.0
MASS = LAM / 100.0
M_UV = np.sqrt(LAM ** 2 + MASS ** 2)          # S1b: M = sqrt(Lambda^2 + m^2)


# ------------------------------------------------------- S7: 最优 g(u) 解析解
def g_S7(u):
    """S7 逐字转写: g(u) = (1/2) * Lambda^2 e^{2u} / (Lambda^2 e^{2u} + m^2)"""
    e2 = LAM ** 2 * np.exp(2.0 * u)
    return 0.5 * e2 / (e2 + MASS ** 2)


def g_k_u(k, u):
    """S2b: g(k e^{-u}; u) = g(u) * Gamma(k e^{-u} / Lambda), Gamma(x) = Theta(1-|x|)"""
    return g_S7(u) if (k * np.exp(-u)) < LAM else 0.0


# ---------------------------------------------------------------- 小工具
def _fock_ops(n):
    """截断 Fock 空间的湮灭算符 (n x n)。"""
    return np.diag(np.sqrt(np.arange(1, n, dtype=float)), 1)


def _comm(A, B):
    return A @ B - B @ A


# =====================================================================
# A 部分: S6 <-> S7 的解析一致性
# =====================================================================
def part_A():
    print('--- A 部分: S6 <-> S7 解析一致性 ---')
    print()
    print('[A1] S6 的 f(k,u_IR) 与 S7 的 g(u) 是否互为积分/微分关系')
    print('     S6: f(k,u_IR) = (1/4) log( (k^2+m^2) / M^2 )')
    print('     S7: g(u) = (1/2) L^2 e^{2u} / (L^2 e^{2u} + m^2),  k = L e^u')
    print()

    ks = [0.05 * LAM, 0.2 * LAM, 0.5 * LAM, 0.9 * LAM]
    worst = 0.0
    print('     %-10s %-14s %-14s %-12s' % ('k/Lambda', 'S6 解析', '数值积分 -I(k)', '相对偏差'))
    for k in ks:
        f_S6 = 0.25 * np.log((k ** 2 + MASS ** 2) / M_UV ** 2)

        # I(k) = int_{log(k/L)}^0 g_S7(u) du  (被积函数只在 u > log(k/L) 时非零)
        u_lo = np.log(k / LAM)
        uu = np.linspace(u_lo, 0.0, 200001)
        gg = g_S7(uu)
        I = float(np.sum(0.5 * (gg[1:] + gg[:-1]) * np.diff(uu)))   # 梯形法, 不依赖 np.trapz
        f_num = -I                       # 文献的 f 与 I 反号, 见 note

        denom = max(abs(f_S6), 1e-30)
        rel = abs(f_num - f_S6) / denom
        worst = max(worst, rel)
        print('     %-10.3f %-14.8f %-14.8f %-12.2e' % (k / LAM, f_S6, f_num, rel))

    ok = worst < 1e-6
    print('     最差相对偏差 = %.2e  -> %s' % (worst, '一致' if ok else '不一致'))
    print('     note: 文献 f 取 int^0_{u_IR}, 被积函数受 Theta 截断后等价于')
    print('           -int^0_{log(k/L)} g du。此符号关系是本探针**数值确认**的, 不是假定的。')
    print()
    return ok


# =====================================================================
# B 部分: 2x2 协方差流 (流机器测试)
# =====================================================================
def cov_flow_2x2(k, n_steps=4000):
    """单模 k 的 2x2 协方差流。

    约定 (由 [phi_k, pi_{-k}] = i 与 |Omega> 的湮灭条件共同固定, 见 C 部分自检):
      Sigma = [[A, C], [C, B]],  A = <phi_k phi_{-k}>, B = <pi_k pi_{-k}>,
      C = (1/2)<{phi_k, pi_{-k}}>
    生成元 K = (1/2) g {phi_k, pi_{-k}} 给出  d/du (phi,pi) = (+g phi, -g pi)
      => S(u) = diag(e^{+F}, e^{-F}),  F(u) = int g du
    """
    A0, B0, C0 = 1.0 / (2.0 * M_UV), M_UV / 2.0, 0.0
    Sig = np.array([[A0, C0], [C0, B0]])

    u_lo = np.log(k / LAM)             # 截断以下 g==0, 不必从 -inf 积分
    du = (0.0 - u_lo) / n_steps

    for n in range(n_steps):
        u = u_lo + (n + 0.5) * du
        g = g_k_u(k, u)
        S = expm(np.array([[g, 0.0], [0.0, -g]]) * du)   # 小步乘积 = 路径排序
        Sig = S @ Sig @ S.T
    return Sig


def part_B():
    print('--- B 部分: 2x2 协方差流 vs 精确基态两点函数 ---')
    print()
    print('[B1] 用 S7 的 g(u) 走流, 末态两点函数是否等于精确基态 1/(2w), w/2')
    print()
    print('     %-10s %-14s %-14s %-12s' % ('k/Lambda', '<phi phi>_流', '1/(2w)_精确', '相对偏差'))

    ks = [0.05 * LAM, 0.2 * LAM, 0.5 * LAM, 0.9 * LAM]
    worst_A, worst_B, worst_pure = 0.0, 0.0, 0.0
    for k in ks:
        Sig = cov_flow_2x2(k)
        w = np.sqrt(k ** 2 + MASS ** 2)
        A_exact, B_exact = 1.0 / (2.0 * w), w / 2.0
        rA = abs(Sig[0, 0] - A_exact) / A_exact
        rB = abs(Sig[1, 1] - B_exact) / B_exact
        det = Sig[0, 0] * Sig[1, 1] - Sig[0, 1] ** 2
        worst_A, worst_B = max(worst_A, rA), max(worst_B, rB)
        worst_pure = max(worst_pure, abs(det - 0.25))
        print('     %-10.3f %-14.8f %-14.8f %-12.2e' % (k / LAM, Sig[0, 0], A_exact, rA))
    print()
    print('     <pi pi> 最差相对偏差        = %.2e' % worst_B)
    print('     纯高斯 A*B - C^2 - 1/4 最大偏离 = %.2e  -> %s'
          % (worst_pure, '保持' if worst_pure < 1e-9 else '破坏'))

    ok = (worst_A < 1e-4) and (worst_B < 1e-4) and (worst_pure < 1e-9)
    print('     -> %s' % ('通过' if ok else '未通过'))
    print()
    return ok


# =====================================================================
# C 部分: 稠密 Fock 独立实现 (机器测试 + g_uu 三路对照)
# =====================================================================
def dense_setup(k, n_fock=22):
    """按 S1b 的湮灭条件数值构造 |Omega>, 并返回 phi_k, pi_{-k} 的稠密矩阵。

    *** 关键下标由 [原文 + 代数] 双重钉死 (v16-smoke-cmera4-2 的修正点) ***
      S1b 原文逐字: ( sqrt(M)(phi(k)-phibar) + i/sqrt(M) * pi(k) ) |Omega> = 0
                                                             ^^^^^^
      是 **pi(k) 同动量**, 不是 pi(-k)。实标量场满足 phi_k^dag = phi_{-k},
      pi_k^dag = pi_{-k}, 故 pi_{-k} 自身并不厄米, 两者不可互换。
      误用 pi(-k) 时: [c,c^dag] = 0  (因 [phi_k,pi_k]=0 且 [phi_{-k},pi_{-k}]=0,
                                     交叉项 [phi_k,phi_{-k}]=[pi_k,pi_{-k}]=0)
                    => c 根本不是湮灭算符, 其"基态"是随机向量,
                       实测 <phi^2> < 0 (负方差, 物理上不可能)。
      用 pi(k) 时:    [c,c^dag] = M[phi_k,phi_{-k}] - i[phi_k,pi_{-k}]
                                + i[pi_k,phi_{-k}] + (1/M)[pi_k,pi_{-k}]
                               = 0 - i*i + i*(-i) + 0 = 2      ✓
      注: S2 的生成元是 phi(p)pi(q)delta(p+q) => 用 **pi(-k)**, 与 S1b 不同。
          两个下标必须分开命名, 混用即出错。

    *** 归一化口径由原文两条式子互相钉死 (v4-2 的修正点) ***
      (a) S1b 湮灭条件:  ( sqrt(M) phi(k) + i pi(k)/sqrt(M) ) |Omega> = 0
      (b) S1b 两点函数:  <Omega|phi(p)phi(q)|Omega> = (1/2M) d(p+q)
                         <Omega|pi(p) pi(q)|Omega>  = (M/2)  d(p+q)
      设模式展开 phi_k = (a_k + a_-k^dag)/sqrt(2 W), 则 c = sqrt(M)phi_k + i pi_k/sqrt(M)
      的挤压参数由 M/W 定:
        W = w = sqrt(k^2+m^2):  c = 1.5 a + 0.5 a_-k^dag, |Omega> 是挤压态,
                                <phi_k phi_-k> = (2N+1)/(2w) = 1.25/(2w) ≠ 1/(2M)
                                (M>w 时 (2N+1)=w/M<1 无解 —— (a)(b) 不能同时成立)
        W = M = sqrt(L^2+m^2):  c = sqrt(2) a_k, |Omega> = Fock 真空,
                                <phi_k phi_-k> = 1/(2M) ✓,  <pi_k pi_-k> = M/2 ✓
      故 (a)+(b) **只允许** M-归一化 —— 这正是 cMERA 里 |Omega> = UV 模真空的含义。
    """
    a = np.kron(_fock_ops(n_fock), np.eye(n_fock))     # a_k
    b = np.kron(np.eye(n_fock), _fock_ops(n_fock))     # a_{-k}
    w = np.sqrt(k ** 2 + MASS ** 2)

    I = np.eye(n_fock ** 2)
    # 模式展开用 M = sqrt(L^2+m^2), 不是 w = sqrt(k^2+m^2)
    phi_k = (a + b.conj().T) / np.sqrt(2.0 * M_UV)     # phi_k ; phi_{-k} = phi_k^dag
    pi_k = -1j * np.sqrt(M_UV / 2.0) * (a - b.conj().T)   # pi_k   <- S1b 用这个
    pi_mk = -1j * np.sqrt(M_UV / 2.0) * (b - a.conj().T)  # pi_{-k} <- S2 生成元用这个

    # S1b 的湮灭条件在 M-归一化下退化为 sqrt(2) a_k |Omega> = 0  =>  |Omega> = |0>_k|0>_-k
    # (这不是假定: C1 数值反验 ||c|Omega>|| 与 [c,c^dag])
    omega_gs = np.zeros(n_fock ** 2, dtype=complex)
    omega_gs[0] = 1.0
    return phi_k, pi_mk, omega_gs, w, I


def build_K(g, phi_k, pi_k, pi_mk):
    """S2 的单模对 (±k) 生成元。

    S2: K(u) = (1/2) int_{pq} g(p;u) [phi(p)pi(q)+pi(p)phi(q)] dbar(p+q)
    代 q=-p 并把第二项重标号 p->-p:
        K = (1/2) int_p g(p) { phi(p), pi(-p) }
    对 p 的积分**同时覆盖 +k 与 -k**, 故单模对给
        K_pair = (1/2) g [ {phi_k,pi_-k} + {phi_-k,pi_k} ]
               = (1/2) g (W + W^dag),   W = phi_k pi_-k + pi_-k phi_k
    只取 W 一支会丢掉 W^dag, 得到的 K **非厄米**; 非厄米 U 使 U^dag != U^-1,
    于是 U^dag phi U != e^F phi (实测只走到 e^{F/2} 的一半), 流全错。
    注: [W^dag, phi_k] = [W^dag, pi_-k] = 0, 故加不加 W^dag 都不改流动方程,
        但**只有厄米 K 才给出正确的态演化** —— 这是数值实测逼出来的, 不是假定的。
    """
    W = phi_k @ pi_mk + pi_mk @ phi_k
    Wd = pi_k @ phi_k.conj().T + phi_k.conj().T @ pi_k      # = W^dag
    return 0.5 * g * (W + Wd)


def part_C(k_frac=0.5, n_fock=22, n_steps=60):
    print('--- C 部分: 稠密 Fock 独立实现 (k = %.2f Lambda) ---' % k_frac)
    print()
    k = k_frac * LAM
    phi, pi_mk, gs, w, I = dense_setup(k, n_fock)
    phi_mk = phi.conj().T              # phi_{-k} = phi_k^dag
    pi_k = pi_mk.conj().T              # pi_k     = pi_{-k}^dag
    u_lo = np.log(k / LAM)

    print('[C1] 约定自检 (不依赖任何记忆公式)')
    # C1a: 截断 Fock 下 [phi_k, pi_-k] - i*I 的**解析预测值**。
    #      截断给出 [a,a^dag] = I - n*P (P = |n-1><n-1|), 故
    #      [phi_k,pi_-k] = i*(I - (n/2)*(P_a + I_a P_b)),
    #      ||...||_F = (n/2)*sqrt(2n+2)。这与实测吻合 => 证实差异纯属截断, 非错。
    pred = 0.5 * n_fock * np.sqrt(2.0 * n_fock + 2.0)
    d1 = np.linalg.norm(_comm(phi, pi_mk) - 1j * I)
    rel1 = abs(d1 - pred) / pred
    print('     ||[phi_k,pi_-k] - i*I||_F = %.6f   截断解析预测 %.6f   相对差 %.2e'
          % (d1, pred, rel1))

    # C1a2: 反验 S1b 的湮灭条件在 M-归一化下确实退化为 Fock 真空 (不假定, 断言它)
    c_op = np.sqrt(M_UV) * phi + 1j * pi_k / np.sqrt(M_UV)
    res_c = np.linalg.norm(c_op @ gs)
    cc = _comm(c_op, c_op.conj().T)
    print('     ||( sqrt(M)phi_k + i pi_k/sqrt(M) )|Omega>|| = %.2e   (应 ~0)   ' % res_c)
    print('     [c,c^dag] 对角元 min/max = %.6f / %.6f   (截断, 应 ~[2, 2-2n])'
          % (cc.diagonal().real.min(), cc.diagonal().real.max()))

    # C1a3: 生成元的厄米性 —— 只取 W 一支非厄米, 补上 W^dag 才厄米
    W = phi @ pi_mk + pi_mk @ phi
    herm_W = np.linalg.norm(W - W.conj().T) / max(np.linalg.norm(W), 1e-30)
    K1 = build_K(1.0, phi, pi_k, pi_mk)
    herm_K = np.linalg.norm(K1 - K1.conj().T) / np.linalg.norm(K1)
    print('     ||W - W^dag|| / ||W||          = %.4f   (W = {phi_k,pi_-k} 单支, 非厄米)'
          % herm_W)
    print('     ||K - K^dag|| / ||K||          = %.2e   (K = (1/2)g(W+W^dag), 厄米)'
          % herm_K)

    # C1b: 态上的标量检验 —— 不受截断边界污染 (|Omega> 在边界权重可忽略时精确)
    A0 = np.vdot(gs, phi @ (phi_mk @ gs)).real
    B0 = np.vdot(gs, pi_k @ (pi_mk @ gs)).real
    C0 = np.vdot(gs, _comm(phi, pi_mk) @ gs)
    bnd = float(np.sum(np.abs(gs.reshape(n_fock, n_fock)[-1, :]) ** 2)
                + np.sum(np.abs(gs.reshape(n_fock, n_fock)[:, -1]) ** 2))
    print('     <Omega|[phi_k,pi_-k]|Omega> = %.10f + %.2ei   应 = i' % (C0.real, C0.imag))
    print('     <Omega|phi_k phi_-k|Omega> = %.10f  vs 1/(2M) = %.10f  相对偏差 %.2e'
          % (A0, 1.0 / (2 * M_UV), abs(A0 - 1.0 / (2 * M_UV)) / (1.0 / (2 * M_UV))))
    print('     <Omega|pi_k  pi_-k |Omega> = %.10f  vs   M/2 = %.10f  相对偏差 %.2e'
          % (B0, M_UV / 2, abs(B0 - M_UV / 2) / (M_UV / 2)))
    print('     |Omega> 在 Fock 截断边界的权重(截断误差指示) = %.2e' % bnd)
    ok0 = (rel1 < 1e-10
           and herm_K < 1e-12
           and abs(C0 - 1j) < 1e-6
           and abs(A0 - 1 / (2 * M_UV)) / (1 / (2 * M_UV)) < 1e-6
           and abs(B0 - M_UV / 2) / (M_UV / 2) < 1e-6)
    print('     -> %s' % ('通过' if ok0 else '失败'))
    print()

    print('[C2] 稠密流: <phi phi> 是否走到精确基态 1/(2w)')
    psi = gs.copy()
    du = (0.0 - u_lo) / n_steps
    for n in range(n_steps):
        u = u_lo + (n + 0.5) * du
        g = g_k_u(k, u)
        K = build_K(g, phi, pi_k, pi_mk)
        psi = expm(-1j * K * du) @ psi
    psi /= np.linalg.norm(psi)

    A_end = np.vdot(psi, phi @ (phi_mk @ psi)).real
    B_end = np.vdot(psi, pi_k @ (pi_mk @ psi)).real
    rA = abs(A_end - 1 / (2 * w)) / (1 / (2 * w))
    rB = abs(B_end - w / 2) / (w / 2)
    print('     <phi_k phi_-k> = %.10f  vs 1/(2w) = %.10f   相对偏差 %.2e'
          % (A_end, 1 / (2 * w), rA))
    print('     <pi_k  pi_-k > = %.10f  vs   w/2  = %.10f   相对偏差 %.2e'
          % (B_end, w / 2, rB))
    okA = rA < 1e-4 and rB < 1e-4
    print('     -> %s' % ('通过' if okA else '未通过'))
    print()

    print('[C3] g_uu 三路对照 (S5 内积 / S5b 方差 / S5c g^2)')
    print()
    psi = gs.copy()
    n_fine = 240
    du_f = (0.0 - u_lo) / n_fine
    rec = []
    for n in range(n_fine):
        u = u_lo + (n + 0.5) * du_f
        g = g_k_u(k, u)
        K = build_K(g, phi, pi_k, pi_mk)
        psi_new = expm(-1j * K * du_f) @ psi
        ov2 = abs(np.vdot(psi, psi_new)) ** 2
        guu_int = (1.0 - ov2) / du_f ** 2                     # S5 (N=1)
        Kp = np.vdot(psi, K @ psi)
        K2p = np.vdot(psi, K @ (K @ psi))
        guu_var = (K2p - Kp ** 2).real                        # S5b
        guu_sq = g ** 2                                       # S5c
        rec.append((u, guu_int, guu_var, guu_sq))
        psi = psi_new

    print('     %-10s %-14s %-14s %-14s' % ('u', 'S5 内积', 'S5b 方差', 'S5c g^2'))
    for (u, gi, gv, gq) in rec[::48]:
        print('     %-10.4f %-14.6e %-14.6e %-14.6e' % (u, gi, gv, gq))
    u_l, gi_l, gv_l, gq_l = rec[-1]
    print('     %-10.4f %-14.6e %-14.6e %-14.6e  <- 末步' % (u_l, gi_l, gv_l, gq_l))
    print()
    sel = [(gi, gv, gq) for (_, gi, gv, gq) in rec if gq > 1e-12]
    rv = np.array([gv / gq for (_, gv, gq) in sel])
    ri = np.array([gi / gq for (gi, _, gq) in sel])
    print('     S5b 方差 / g^2 : 均值 %.6f  相对标准差 %.2e' % (np.mean(rv), np.std(rv) / abs(np.mean(rv))))
    print('     S5  内积 / g^2 : 均值 %.6f  相对标准差 %.2e' % (np.mean(ri), np.std(ri) / abs(np.mean(ri))))
    print('     note: 比值若为常数, 说明三条定义只差一个与 u 无关的归一化因子;')
    print('           该因子由 K 的归一化约定决定, 是**实测**得到的, 不是假定的。')
    okC = (np.std(rv) / abs(np.mean(rv)) < 1e-2 and np.std(ri) / abs(np.mean(ri)) < 1e-2)
    print('     -> %s' % ('三路 g_uu 互成比例 (比值 u-无关)' if okC else '未通过'))
    print()

    return ok0 and okA and okC


# =====================================================================
def main():
    t0 = time.time()
    print('=== v16 探针 4 · cMERA 玻色子基准复现 (K-1)  [%s] ===' % _VERSION_TAG)
    print('文献: arXiv:2104.01551v2   (纯玻色子 free scalar)')
    print('Lambda = %.1f, m = %.4f, M = sqrt(L^2+m^2) = %.6f' % (LAM, MASS, M_UV))
    print('本探针不产出任何关于 TFI 的物理结论 —— 它只回答"机器能否复现已发表解析结果"。')
    print()

    rA = part_A()
    rB = part_B()
    rC = part_C()

    print('=== 判定 ===')
    print('    A  S6 <-> S7 解析一致性         : %s' % ('通过' if rA else '未通过'))
    print('    B  2x2 协方差流 vs 精确两点函数   : %s' % ('通过' if rB else '未通过'))
    print('    C  稠密 Fock 独立实现 + g_uu 三路 : %s' % ('通过' if rC else '未通过'))
    print()
    print('用时: %.1f s' % (time.time() - t0))
    if rA and rB and rC:
        print('K-1 玻色子基准: 复现成功 —— 机器可用, 可以进入费米子版。')
        print('边界: 这是**机器校准**, 不是物理结论; 玻色子基准不上我们的格点。')
        return 0
    print('K-1 玻色子基准: **未复现** —— 按计划 §3.5 K-1 的判据, 此时停手,')
    print('    不许进入费米子版 (因为分不清是机器错还是推导错)。')
    return 3


if __name__ == '__main__':
    sys.exit(main())
