"""Hyper-invariant 张量网络 (Evenbly, PRL 2017) —— 构建块与约束自检 (Phase 1).

锚: `v16/Hyper.tex` 附录 B
  Eq. (Yparam) `:397-404`  双酉张量 Y  (chi_tilde = 2, Z2 对称)
  Eq. (Rparam) `:437-444`  双酉张量 R
  Eq. (Qparam) `:450-457`  酉张量   Q
四条**自述**约束 (`:392-395`, `:431-435`, `:446-448`):
  Y: sum_kl Y_ijkl Y*_mnkl = d_im d_jn ;  sum_jl Y_ijkl Y*_mjnl = d_im d_kn
  R: sum_kl R_ijkl R*_mnkl = d_im d_jn ;  sum_jk R_ijkl R*_mjkn = d_im d_ln
  Q: sum_kl Q_ijkl Q*_mnkl = d_im d_jn
**自述**对称性 (`:405` Y, `:436` R, `:449` Q):
  Y_ijkl = Y_klij
  R_ijkl = R_klij , R_ijkl = R_jilk
  Q_ijkl = Q_klij , Q_ijkl = Q_jilk

**已知原文缺陷 (2026-10-03 实测)** —— Eq.(Yparam) 印刷的 `Y_tensor` **非酉**:
按 (ij)(kl) 分组当普通矩阵看, `Y Y^dag` 的 (1,4) 元 = 2 c1 s1 != 0
⇒ 它连**单条**酉性都不满足, 遑论双酉; 残差恒 = 2|c1 s1|。
`v16/Hyper.tex` 与 arXiv v1 (1704.04229) 源码 raw-diff 逐字一致 ⇒ 缺陷在**原文**,
不是本仓抄录错。R 与 Q 无此问题。
`Y_tensor_fixed` 是按**口径 B** 落盘的**非原文 · 诊断猜想**变体 (角上 +c1 -> -c1),
**不属于原文**; 引用时不得与 `Y_tensor` 混同, 判据与落支一律以 `Y_tensor` 为准。

**第三张 `Y` (2026-10-04 增, v16.8/T6)** —— `Y_tensor_sp` 是 **S&P 印的那张**:
角元 `sin θ1` -> `i sin θ1`, 右下仍 `+cos θ1`, 其余与 `Y_tensor` 逐元素相同。
来源 **M. Steinberg & J. Prior, Sci Rep 12, 532 (2022) / arXiv:2012.09591**, 源码级核对
(`spiral_v16_audit.md` §9.2 `:1607`); **不是 Evenbly 原文的 `Y`**, 也不是本仓自造的变体。
它的性质与另两张**都不同**: 只在 `(12|34)` 上酉, `(13|24)` 不酉 => **非双酉**, 但**精确**满足
原文的多张量约束 w/u (c=4); 反射对称仍成立。T6 把它定为**主口径**, `Y_tensor` 退为对照。
三张 `Y` 在任何读数里**不得合并成一行** (v16.8 计划 §1 硬规矩)。

**Phase 1b (2026-10-03)** —— 已组装 A/B 并验**多张量约束** (w = A·B·B 是 2->1 等距,
u = A^3B^5 是 3->2 等距; 正文原话 `Hyper.tex:109`, 判据 `:100` "annihilates to identity
with its conjugate w^dag", `:460` 允许 "up to an irrelevant multiplicative constant")。
**粗拓扑**读自 arXiv 源码附图 `v16/arXiv-1704.04229v1/`
{Parameterization37, Constraint37, Unitary37}.eps —— 本仓 `.tex` 正文确实没写接法。
但那些 .eps 是 **GIMP 光栅图**, 每条腿内 4 个细比特的**走线分辨不出**
⇒ `WIRING_U` 是**自选**值, 非原文 (实测: 216 种走线中 162 种给出精确等距)。

不 import 其它 `_model` 模块；不调用任何 `metric_*` / `record_*`；不进 runner。
"""
import numpy as np

__all__ = ['Y_tensor', 'Y_tensor_fixed', 'Y_tensor_sp', 'R_tensor', 'Q_tensor',
           'unitary_residual', 'doubly_unitary_residual_Y',
           'doubly_unitary_residual_R', 'reflection_residual',
           'WIRING_U', 'A_tensor', 'B_tensor', 'w_tensor', 'u_tensor',
           'coisometry_residual']

# 2-bit 指标序 (i,j) -> (j,i) 的置换: 00->0, 01->2, 10->1, 11->3
_SWAP_IJ = np.array([0, 2, 1, 3])


def Y_tensor(t1):
    """Eq. (Yparam): 4x4 矩阵, 行 = (ij), 列 = (kl), 每个单指标维 2."""
    c, s = np.cos(t1), np.sin(t1)
    return np.array([[c, 0.0, 0.0, s],
                     [0.0, s, 1j * c, 0.0],
                     [0.0, 1j * c, s, 0.0],
                     [s, 0.0, 0.0, c]], dtype=complex)


def Y_tensor_fixed(t1):
    """**非原文 · 诊断猜想** —— 把 Eq. (Yparam) 的右下角 `+c1` 改成 `-c1`。

    与 `Y_tensor` 的差别**仅此一项**。改后 Y 双酉与反射对称都精确成立
    (残差 ~1e-17 / 0), 即在**印刷结构内** (实 c1,s1 + 对称 M) 该改法唯一。
    但**不声称这是原文作者的本意**; 原文只印 `Y_tensor`, 两者不得混同。
    """
    c, s = np.cos(t1), np.sin(t1)
    return np.array([[c, 0.0, 0.0, s],
                     [0.0, s, 1j * c, 0.0],
                     [0.0, 1j * c, s, 0.0],
                     [s, 0.0, 0.0, -c]], dtype=complex)


def Y_tensor_sp(t1):
    """**非 Evenbly 原文** —— S&P (Sci Rep 12, 532 / arXiv:2012.09591) 印的 `Y`。

    与 `Y_tensor(t1)` 的差别**仅角元**: `sin θ1` -> `i sin θ1` (右下仍 `+cos θ1`)。
    锚: `v16/spiral_v16_audit.md` §9.2 `:1607` (源码级核对; 两版 arXiv 逐元素相同、进了正式版)。
    矩阵仍**对称** (M[0,3]=M[3,0], M[1,2]=M[2,1]) => 反射残差为 0。

    实测 (2026-10-04, `_v16_smoke_t6_admission.py`, θ1 = 0.37):
    `(12|34)` dev = 0, `(13|24)` dev = 6.742879e-01, 反射 = 0,
    w resid 1.776357e-15 / u resid 3.552714e-15, 两者 c 都 = 4。
    即: **非双酉**(与另两张都不同的第三种性质), 但**精确**满足原文的多张量约束。
    """
    c, s = np.cos(t1), np.sin(t1)
    return np.array([[c, 0.0, 0.0, 1j * s],
                     [0.0, s, 1j * c, 0.0],
                     [0.0, 1j * c, s, 0.0],
                     [1j * s, 0.0, 0.0, c]], dtype=complex)


def R_tensor(t2):
    """Eq. (Rparam)."""
    c, s = np.cos(t2), np.sin(t2)
    return np.array([[c, 0.0, 0.0, 1j * s],
                     [0.0, c, 1j * s, 0.0],
                     [0.0, 1j * s, c, 0.0],
                     [1j * s, 0.0, 0.0, c]], dtype=complex)


def Q_tensor(t3, t4, t5):
    """Eq. (Qparam)."""
    c3, s3 = np.cos(t3), np.sin(t3)
    c5, s5 = np.cos(t5), np.sin(t5)
    p = np.exp(1j * t4)
    return np.array([[c3, 0.0, 0.0, s3 * p],
                     [0.0, c5, 1j * s5, 0.0],
                     [0.0, 1j * s5, c5, 0.0],
                     [s3 * p, 0.0, 0.0, -c3 * p * p]], dtype=complex)


def unitary_residual(M):
    """max |M M^dag - I|.  精确线性代数 => 期望 ~1e-15."""
    n = M.shape[0]
    return float(np.max(np.abs(M @ M.conj().T - np.eye(n))))


def _four(M):
    """4x4 -> (2,2,2,2), 行=(ij) 拆成 (i,j), 列=(kl) 拆成 (k,l)."""
    return np.ascontiguousarray(M).reshape(2, 2, 2, 2)


def doubly_unitary_residual_Y(M):
    """Y 的两个分划: sum_kl(ij|kl) 与 sum_jl(ik|jl)."""
    T, C = _four(M), _four(M).conj()
    I = np.eye(4)
    a = np.einsum('ijkl,mnkl->ijmn', T, C).reshape(4, 4)
    b = np.einsum('ijkl,mjnl->ikmn', T, C).reshape(4, 4)
    return float(max(np.max(np.abs(a - I)), np.max(np.abs(b - I))))


def doubly_unitary_residual_R(M):
    """R 的两个分划: sum_kl(ij|kl) 与 sum_jk(il|jk)."""
    T, C = _four(M), _four(M).conj()
    I = np.eye(4)
    a = np.einsum('ijkl,mnkl->ijmn', T, C).reshape(4, 4)
    b = np.einsum('ijkl,mjkn->ilmn', T, C).reshape(4, 4)
    return float(max(np.max(np.abs(a - I)), np.max(np.abs(b - I))))


def reflection_residual(M, which):
    """which='klij': M_ijkl == M_klij (矩阵对称);  'jilk': M_ijkl == M_jilk."""
    if which == 'klij':
        return float(np.max(np.abs(M - M.T)))
    if which == 'jilk':
        return float(np.max(np.abs(M - M[np.ix_(_SWAP_IJ, _SWAP_IJ)])))
    raise ValueError('unknown reflection: %r' % (which,))


# ─── Phase 1b (2026-10-03): A / B 组装 + 多张量约束 w, u ───────────────────────
# 粗拓扑锚 (arXiv 源码附图, 已解码 PNG 直读; 本仓 .tex 正文没写接法):
#   Parameterization37(a)  A = 3 个 Y 的**张量积** (无缩并), 12 个细指标重组成 3 条腿;
#                          每条腿的 4 个比特里 2 个进一个 Y、2 个进另一个 Y (三角形)。
#   Parameterization37(c)  B = Q·R (Kronecker, 无缩并): 前 2 比特给 Q, 后 2 比特给 R。
#   Constraint37(a-b)      w = A·B·B;  u = 3 个 A 一排 + 2 条水平 B + 3 条竖直 B (共 5 个 B),
#                          外部腿 = A1、A3 的自由腿。
# 细比特走线: 光栅图分辨不出 ⇒ `WIRING_U` 是**自选**值 (见模块头 Phase 1b 段)。
# B 不含 Y ⇒ B 的酉性不受 Y 缺陷影响 (冒烟判据 [3])。

# u 里三个 A 的腿角色 -> A 的原生腿 (0=alpha, 1=beta, 2=gamma)。**自选**, 非原文。
#   A1: (自由腿a1, 水平m1, 竖直n1) / A2: (左m2, 右m3, 竖直n2) / A3: (水平m4, 自由腿a3, 竖直n3)
# 该值取"图上的几何读法" (A1 正立, A2 转 180°, A3 取 A1 的镜像); 仅此一读法无法由光栅图钉死。
WIRING_U = ((0, 2, 1), (2, 0, 1), (1, 0, 2))


def A_tensor(t1, Y=None):
    """图 Parameterization37(a): A[a,b,c], 由 3 个 Y **张量积** (无缩并) 组成。

    A 的 3 条腿各 4 个细比特; 腿内 2 比特共享一个 Y、另 2 比特共享另一个 Y:
        A[a,b,c] = Y_L(a_01,b_01) * Y_R(a_23,c_01) * Y_B(b_23,c_23)
    a_01 = a 的前 2 比特 (a // 2), a_23 = 后 2 比特 (a % 2)。
    A **不**在 3 条腿的置换下对称 (实测最大偏差 9.32e-01); 但 w 的残差实测与"哪条腿当
    自由腿"无关: 原文 Y 三种选法都 = 2.697152e+00, Y_tensor_fixed 三种都 < 1.8e-15。

    Y=None 时用原文 `Y_tensor(t1)`; 传 `Y_tensor_fixed(t1)` 得口径 B 的**诊断变体**。
    """
    T = _four(Y if Y is not None else Y_tensor(t1))
    return np.einsum('abcd,efgh,ijkl->abefcdijghkl', T, T, T).reshape(16, 16, 16)


def B_tensor(t2, t3, t4, t5):
    """图 Parameterization37(c): B[x,y] = Q(x_01,y_01) * R(x_23,y_23)。

    B 只由 Q, R 组成, **不含 Y** ⇒ 16x16, 精确酉 (残差 ~2e-16) 且精确对称
    (Q, R 各自对称) ⇒ B^dag B = B B^dag = I_16, 与 Y 的缺陷无关。
    """
    Q = _four(Q_tensor(t3, t4, t5))
    R = _four(R_tensor(t2))
    return np.einsum('abcd,efgh->abefcdgh', Q, R).reshape(16, 16)


def w_tensor(A, B):
    """w = A·B·B (Fig. Constraint37(a)): 2->1 等距, 展成 (16, 256) 矩阵。

    行 = A 的自由腿 (1 条腿, chi = 16), 列 = 两个 B 的自由腿 (2 条腿, chi^2 = 256)。
    `Hyper.tex:100` 的原话是 "annihilates to identity with its conjugate w^dag"
    ⇒ 验的是 W W^dag ∝ I, 不是 W^dag W (后者实测残差 9.66e-01, 不是这条约束)。
    """
    return np.einsum('amn,mx,ny->axy', A, B, B, optimize=True).reshape(16, 256)


def u_tensor(A, B, wiring=WIRING_U):
    """u = A^3 B^5 (Fig. Constraint37(b)): 3->2 等距, 展成 (256, 4096) 矩阵。

    行 = A1、A3 的自由腿 (2 条腿, chi^2 = 256), 列 = 3 条竖直 B 的自由腿 (3 条, chi^3 = 4096)。
    `wiring` 是每条腿内 4 个细比特的走线 —— **自选** (光栅图分辨不出)。
    """
    A1 = np.transpose(A, wiring[0])
    A2 = np.transpose(A, wiring[1])
    A3 = np.transpose(A, wiring[2])
    T1 = np.einsum('amn,mx->axn', A1, B, optimize=True)        # a, m2, n1
    T2 = np.einsum('axn,xyz->ayzn', T1, A2, optimize=True)     # a, m3, n2, n1
    T3 = np.einsum('ayzn,yw->awzn', T2, B, optimize=True)      # a, m4, n2, n1
    T4 = np.einsum('awzn,wbv->aznbv', T3, A3, optimize=True)   # a, n2, n1, a3, n3
    T5 = np.einsum('aznbv,zp->anbvp', T4, B, optimize=True)    # a, n1, a3, n3, x2
    T6 = np.einsum('anbvp,nq->abvpq', T5, B, optimize=True)    # a, a3, n3, x2, x1
    U = np.einsum('abvpq,vr->abpqr', T6, B, optimize=True)     # a, a3, x2, x1, x3
    return U.reshape(256, 4096)


def coisometry_residual(W):
    """返回 (max |W W^dag - c I|, c), c = tr(W W^dag) / nrow。

    c 即原文允许的 "irrelevant multiplicative constant" (`Hyper.tex:460`)。
    实测 w 与 u 的 c 都 = 4 = Tr(1_{chi_tilde^2}), 来自被完全缩并掉的那个 Y_B。
    纯线性代数 ⇒ 期望残差 ~1e-15。
    """
    n = W.shape[0]
    G = W @ W.conj().T
    c = float(np.trace(G).real / n)
    return float(np.max(np.abs(G - c * np.eye(n)))), c
