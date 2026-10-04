"""TNR (Evenbly & Vidal, PRL 115, 180405 (2015)) —— 附录 B 闭式构建块 (Phase 1).

锚: `v16/TNR.tex` 附录 B「Closed form expressions for a CDL tensor network」
  Eq. (CDL)  `:434`  CDL 张量 A^CDL:  A_ijkl = d_{i1 j2} d_{j1 k2} d_{k1 l2} d_{l1 i2}
  Eq. (TRG2) `:454`  (A^CDL)_ijkl = sum_m S_mij S_mkl
  Eq. (TRG3) `:458`  S_{(m1m2)(l1l2)(i1i2)} = d_{m1 l2} d_{l1 i2} d_{i1 m2}
  Eq. (TRG4) `:462`  A'_ijkl = sum_{m,n,o,p} S_imp S_jnm S_kon S_lpo
  Eq. (TRG5) `:466`  A' = eta A^CDL
  Eq. (TNR1) `:499`  u_ij(k1k2)(l1l2) = eta^{-1/2} d_{i k1} d_{k2 l2} d_{j l1}
  Eq. (TNR2) `:503`  v = w = ...                <-- 见下 3
  Eq. (TNR3) `:510`  B = eta^4 d_il d_jk ;  C = eta^{-1} d_ij d_kl

双指标约定 (原文): i = (i1 i2) 表示 i = i1 + eta (i2 - 1), 每个单指标跑 1..eta。
本仓内部一律 0-based: idx = i1 + eta * i2。

**已知原文缺陷 (2026-10-03 实测, 详见 `v16/_t5_tnr.log` D-a/D-b/D-c)**
  1. **Eq. (TRG2) 对任何实 S 都不可能成立**。把 A^CDL 按 (ij) x (kl) 展成 eta^2 x eta^2 矩阵,
     其最小特征值 < 0 (eta=2 时 = -2); 而 Eq.(TRG2) 右端 sum_m vec(S_m) vec(S_m)^T 是**实向量的
     Gram 和**, 必半正定。24 种腿序**全部**非半正定, 256 种 A 形态里只有 16 种半正定
     => 与指标序约定**无关**, 是印刷层面的不可能。
  2. Eq. (TRG4) 本身**可以**精确给出 eta A^CDL, 但只在 S 的 (delta 配对 x 腿序) 384 种读法里的
     **12 种**下成立, **不含** Eq.(TRG3) 印刷的配对 => TRG2 与 TRG4 **互斥**
     (768 种 (A,S) 组合无一同时满足)。
  3. Eq. (TNR2) 印刷版用到**未定义**的单指标 (同一式里定义了 k1,k2 却把另两个印成未定义的
     记号), 无法照抄。本模块**不替原文选**, 只提供两个候选:
       ①  v_ij(k1k2) = eta^{-1/2} d_{i k1} d_{j k2}
       ②  v_ij(k1k2) = eta^{-1/2} d_{i k2} d_{j k1}
     两者只差 k1 <-> k2 的约定, 由残差读数定夺 (判据与落支不预设哪个是原文)。

**非原文 · 诊断猜想**（口径 B，2026-10-03 落盘）—— 上面 3 条的**修正变体**，一律标注「不属于原文」，
判据与落支**一律以原文式样为准**，不得混同：
  `s_tensor_fixed` —— 换 delta 配对 ⇒ Eq.(TRG4) 精确成立；**但 Eq.(TRG2) 仍不可能**（已证）。
  `v_tensor_fixed` —— 去掉 u 的残留 delta + 前因子改 1 ⇒ V V^dag = I 精确成立。
  两者都**不声称是作者本意**。图 `TNRtensors.eps` 只能补出 v 的腿标 k1k2 / w 的腿标 k2k1
  （互镜像 ⇒ 候选①/②之别），**补不出归一化** ⇒ 图本身给不出自洽读法。

**范围红线** —— 本模块只做附录 B 的**闭式**构建块自检。2x2 CDL 网络的**实际缩并接线**
只在图 `TNRtensors.eps` / `TNR.eps` 里, `.tex` 正文没写 => **不在本模块范围**
(与 T4 的 Phase 1b 同性质: 不凭文本猜接线)。
不 import 其它 `_model` 模块; 不接 L4/L5; 不进 runner; 不加守卫与指标。
"""
import itertools

import numpy as np

__all__ = ['cdl_tensor', 's_tensor', 's_tensor_fixed', 's_tensor_explicit',
           'trg_factorization_residual', 'trg_step', 'gram_min_eig',
           'u_tensor', 'u_matrix', 'v_tensor', 'v_tensor_fixed', 'v_matrix',
           'b_tensor', 'c_tensor', 'matricize_rank']

LEG_ORDERS = tuple(itertools.permutations(range(4)))
S_PAIRINGS = tuple(itertools.product(itertools.product((1, 2), repeat=2), repeat=3))


def _sub(idx, k, eta):
    """双指标 idx=(i1 i2) 取单指标; k=1 -> i1, k=2 -> i2。"""
    return (idx % eta) if k == 1 else (idx // eta)


def cdl_tensor(eta=2):
    """Eq. (CDL): A_ijkl = d_{i1 j2} d_{j1 k2} d_{k1 l2} d_{l1 i2}; 每条腿维 eta^2。"""
    N = eta * eta
    A = np.zeros((N, N, N, N))
    for i, j, k, l in itertools.product(range(N), repeat=4):
        A[i, j, k, l] = float(_sub(i, 1, eta) == _sub(j, 2, eta)
                              and _sub(j, 1, eta) == _sub(k, 2, eta)
                              and _sub(k, 1, eta) == _sub(l, 2, eta)
                              and _sub(l, 1, eta) == _sub(i, 2, eta))
    return A


def s_tensor_explicit(pairing, leg_order=(0, 1, 2), eta=2):
    """通用读法。pairing = ((ka1,kb1),(kb2,kc2),(kc3,ka3)): 每条 delta 用哪个单指标;
    leg_order: S 三条腿的摆放序 (用于扫描约定)。"""
    N = eta * eta
    S = np.zeros((N, N, N))
    for a, b, c in itertools.product(range(N), repeat=3):
        S[a, b, c] = float(_sub(a, pairing[0][0], eta) == _sub(b, pairing[0][1], eta)
                           and _sub(b, pairing[1][0], eta) == _sub(c, pairing[1][1], eta)
                           and _sub(c, pairing[2][0], eta) == _sub(a, pairing[2][1], eta))
    return np.transpose(S, leg_order)


def s_tensor(eta=2):
    """Eq. (TRG3) **字面**读法: S_abc = d_{a1 b2} d_{b1 c2} d_{c1 a2}。"""
    return s_tensor_explicit(((1, 2), (1, 2), (1, 2)), (0, 1, 2), eta)


def s_tensor_fixed(eta=2):
    """**非原文 · 诊断猜想** —— 把 Eq. (TRG3) 的 delta 配对换成 ((1,1),(2,2),(1,2))。

    与 `s_tensor` 的差别**仅在每条 delta 用哪个单指标**。改后 Eq. (TRG4) 精确给出
    A' = eta A^CDL（残差 0）。
    **但 Eq. (TRG2) 仍然不可能** —— `gram_min_eig` 证明它对**任何实 S** 都不成立，
    换配对救不回来。故本变体只恢复 Eq.(TRG4)，**不**恢复 TRG2，也
    **不声称是原文作者的本意**；原文只印 `s_tensor`，两者不得混同。
    """
    return s_tensor_explicit(((1, 1), (2, 2), (1, 2)), (0, 1, 2), eta)


def trg_factorization_residual(S, A):
    """Eq. (TRG2): max |sum_m S_mij S_mkl - A|。"""
    return float(np.max(np.abs(np.einsum('mij,mkl->ijkl', S, S) - A)))


def trg_step(S):
    """Eq. (TRG4): A'_ijkl = sum_{m,n,o,p} S_imp S_jnm S_kon S_lpo。"""
    return np.einsum('imp,jnm,kon,lpo->ijkl', S, S, S, S)


def gram_min_eig(A, leg_order=(0, 1, 2, 3)):
    """**免约定**可行性判据: A 按 leg_order 前两腿 x 后两腿展成方阵, 对称化后取最小特征值。

    Eq.(TRG2) 的右端是实向量 {vec(S_m)} 的 Gram 和 => 必半正定 => 该值必须 >= 0,
    否则**任何实 S** 都满足不了 Eq.(TRG2), 与指标序/配对约定无关。
    """
    N = A.shape[0]
    M = np.transpose(A, leg_order).reshape(N * N, N * N)
    M = 0.5 * (M + M.T)
    return float(np.linalg.eigvalsh(M).min())


def u_tensor(eta=2):
    """Eq. (TNR1): u_ij(k1k2)(l1l2) = eta^{-1/2} d_{i k1} d_{k2 l2} d_{j l1}。
    腿维 (eta, eta, eta^2, eta^2)。"""
    N = eta * eta
    U = np.zeros((eta, eta, N, N))
    for i, j, K, L in itertools.product(range(eta), range(eta), range(N), range(N)):
        U[i, j, K, L] = (eta ** -0.5) * float(_sub(K, 1, eta) == i
                                              and _sub(L, 2, eta) == _sub(K, 2, eta)
                                              and _sub(L, 1, eta) == j)
    return U


def u_matrix(U, grouping='iK_jL'):
    """把 4 腿 u 折成方阵做等距检验。
       'iK_jL' -> 行 (i,(k1k2)), 列 (j,(l1l2));   'ij_KL' -> 行 (i,j), 列 ((k1k2),(l1l2))。"""
    eta, N = U.shape[0], U.shape[2]
    if grouping == 'iK_jL':
        return np.transpose(U, (0, 2, 1, 3)).reshape(eta * N, eta * N)
    if grouping == 'ij_KL':
        return U.reshape(eta * eta, N * N)
    raise ValueError('unknown grouping %r' % (grouping,))


def v_tensor(eta=2, candidate=1):
    """Eq. (TNR2) —— 印刷版含未定义指标, 故给两个候选 (本模块**不**替原文选):
       ①  v_ij(k1k2) = eta^{-1/2} d_{i k1} d_{j k2}
       ②  v_ij(k1k2) = eta^{-1/2} d_{i k2} d_{j k1}
    腿维 (eta, eta, eta^2), 自然折成 (ij) x (k1k2)。"""
    N = eta * eta
    V = np.zeros((eta, eta, N))
    if candidate not in (1, 2):
        raise ValueError('candidate must be 1 or 2')
    for i, j, K in itertools.product(range(eta), range(eta), range(N)):
        if candidate == 1:
            ok = (_sub(K, 1, eta) == i) and (_sub(K, 2, eta) == j)
        else:
            ok = (_sub(K, 2, eta) == i) and (_sub(K, 1, eta) == j)
        V[i, j, K] = (eta ** -0.5) * float(ok)
    return V


def v_tensor_fixed(eta=2):
    """**非原文 · 诊断猜想** —— Eq. (TNR2) 的 v：只用**两条** delta，前因子改 1。

    与 `v_tensor` 候选①② 的差别有两点：
      (a) 印刷式多带了 u 的一条 delta_{k2 l2}（那两个 l1,l2 在 v 里没有定义），本变体去掉；
      (b) 前因子 eta^{-1/2} -> 1。
    改后 V V^dag = I **精确成立**（残差 0）。两条 delta 的**连接次序**取候选①，
    与图 `TNRtensors.eps` 上 v 的腿标 k1k2 / w 的腿标 k2k1（互为镜像）一致；
    但**归一化不在图里**，由等距性定夺。
    **不声称是原文作者的本意**；原文只印 `v_tensor` 的式样，两者不得混同。
    """
    N = eta * eta
    V = np.zeros((eta, eta, N))
    for i, j, K in itertools.product(range(eta), range(eta), range(N)):
        V[i, j, K] = float(_sub(K, 1, eta) == i and _sub(K, 2, eta) == j)
    return V


def v_matrix(V):
    """(i,j,K) -> ((i,j), K) 方阵 (eta^2 x eta^2)。"""
    eta = V.shape[0]
    return V.reshape(eta * eta, V.shape[2])


def b_tensor(eta=2):
    """Eq. (TNR3): B = eta^4 d_il d_jk, 4 腿 (i,j,k,l), 每条腿维 eta。"""
    B = np.zeros((eta, eta, eta, eta))
    for i, j, k, l in itertools.product(range(eta), repeat=4):
        B[i, j, k, l] = (eta ** 4) * float(i == l and j == k)
    return B


def c_tensor(eta=2):
    """Eq. (TNR3): C = eta^{-1} d_ij d_kl —— 角张量。其 (ij) x (kl) 矩阵秩 1 ⇒ chi' = 1。"""
    C = np.zeros((eta, eta, eta, eta))
    for i, j, k, l in itertools.product(range(eta), repeat=4):
        C[i, j, k, l] = (eta ** -1.0) * float(i == j and k == l)
    return C


def matricize_rank(T, leg_order=None):
    """把 2n 腿等维张量折成方阵取秩 (用于 'chi=1' 式的平凡性判断)。d = 每条腿的维。"""
    n = T.ndim // 2
    d = int(round(float(T.size) ** (1.0 / T.ndim)))
    if leg_order is None:
        leg_order = tuple(range(T.ndim))
    return int(np.linalg.matrix_rank(np.transpose(T, leg_order).reshape(d ** n, d ** n)))
