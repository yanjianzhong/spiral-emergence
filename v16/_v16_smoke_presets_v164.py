# -*- coding: utf-8 -*-
"""
v16.4 · 预设层攻坚冒烟 —— α 三条 (D5 / D6 / D7)
[_VERSION_TAG = 'v16-smoke-presets-v164-1']

**与既有两份冒烟的分工**:
  * `_v16_smoke_presets.py`        —— 只把 §5.4 的「给定」细分成「承重 / 不承重」, **不导出**;
  * `_v16_smoke_derive_presets.py` —— 真去导出。结论: `d_local` 的**奇偶**被初等导出,
    **偶数中取 2 仍是选择**; 复数域 L1 处不承重、下游由厄米性强制。两条同一根。
  * **本文件** —— v16.4 计划 §2 的 D5/D6/D7: 换 **不可约性** 口径重推 `d_local=2`,
    取证复数域在 `d=4` 档不承重, 复核随机测度是不是重言式。

口径 (三条, 全部可检查):
  * **不新增守卫、不新增指标、不改任何既有读数** ⇒ 规模三数 (守卫 78 / 指标 27 /
    负对照 13) 逐字不变, 计分仍 57。本文件是一份**登记**, 不是判定。
  * 每条检验配一个**活的反事实** (坏输入必须能翻红), 并明写它导不出来的那一半。
  * 结论**分层陈述**: 数学事实 / 代码事实 / 口径选择, 三者分开说。

**落支** (计划 §2.2 预登记; 结果出来只能往里填, 不能新建支):
  D5 → S-导出 / **S-部分** ⭐ / S-未导出 / S-不成题
  D6 → S-负 (代码已预判)
  D7 → S-重言 / S-导出但无收益

用法:
    python v16/_v16_smoke_presets_v164.py
"""

import inspect
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'v16'))

import numpy as np          # noqa: E402
import scipy.sparse as sp   # noqa: E402
import _v16_cmera_gaussian as cg   # noqa: E402
import spiral_model_v16 as G       # noqa: E402

L = 4                 # 站点数 (与既有两份冒烟同规模)
DIM = 2 ** L          # 物理载体维数: 阶段二链长 = 2^4 = 16
DIM4 = 4 ** L         # d_local=4 的载体维数 = 256
_FAILS = []


def check(label, cond, detail=''):
    tag = 'PASS' if cond else '**FAIL**'
    print(f"    [{tag}] {label}" + (f"  —— {detail}" if detail else ''))
    if not cond:
        _FAILS.append(label)


def realness(psi):
    """|Σ_i ψ_i²| —— **相位不变**判据: 等于 1 ⟺ ψ 在**整体相位**下是实的。
       (与 `_v16_smoke_derive_presets.py` 的 realness 同式, 便于两份对读。)"""
    p = np.asarray(psi).ravel()
    return float(abs(np.sum(p * p)))


# ============================================================================
# 中心化子 (commutant) 维数 —— 两套独立测法
# ============================================================================

def carrier(n_half, mult):
    """2·n_half 个 Majorana, 作用在 2^{n_half}·mult 上: γ_a ⊗ I_mult。
       mult=1 即**不可约**载体 (2^{n_half} 维); mult>1 是它的 mult 重直和。"""
    gam = [cg.majorana(a, n_half) for a in range(2 * n_half)]
    if mult == 1:
        return gam
    eye = np.eye(mult, dtype=complex)
    return [np.kron(g, eye) for g in gam]


def conjugate(gens, seed):
    """随机酉共轭 ⇒ 与标准载体同构的**一般**载体 (只差一个基底选取)。"""
    N = gens[0].shape[0]
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    U, _ = np.linalg.qr(H)
    return [U @ np.asarray(g) @ U.conj().T for g in gens]


def to_sparse(gens):
    """只对 Pauli 弦型载体用 —— 每条 γ 只有 N 个非零元, 乘积代价 O(N) 而非 O(N³)。"""
    return [sp.csr_matrix(np.asarray(g)) for g in gens]


def _monomial_sweep(gens, hook):
    """Gray 码遍历全部 2^n 个子集, 每步只乘一个 γ_a (因为 γ_a² = I)。
       第 step 步翻转的是 step 的最低置位。乘积与顺序只差一个整体符号 ±1,
       而本文件只用 |tr|², 故不受影响。"""
    n = len(gens)
    N = gens[0].shape[0]
    M = (sp.identity(N, format='csr', dtype=complex) if sp.issparse(gens[0])
         else np.eye(N, dtype=complex))
    for step in range(1 << n):
        if step:
            b = (step & -step).bit_length() - 1
            M = gens[b] @ M
        hook(M)


def commutant_dim_by_traces(gens):
    """测法一 · **群平均投影的迹** (主测法)。

    G = {i^k γ_S : k∈{0..3}, S⊆{0..n-1}} 是 n 个 γ 生成的有限**酉**群, 阶 4·2^n ——
    因为在 Clifford 关系下两个单项式之积 = (±1)·另一个单项式, 不带额外的 i。
    于是 P(X) = (1/|G|) Σ_{g∈G} g X g† 是到 comm(G) = comm(γ 生成的代数) 的投影, 而
    dim comm = tr P = (1/|G|) Σ_g |tr g|²。i^k 因子满足 |i^k|² = 1, 故

        dim comm = (1 / 2^n) · Σ_S |tr γ_S|²          (n = Majorana 个数)

    这里**逐个子集实算** tr γ_S —— 不是把公式当结论用; 公式本身在 A1 里由零空间
    测法独立对账, 在 A2 里由「P 落在中心化子里且幂等」再核一遍。
    """
    total = [0.0]

    def hook(M):
        total[0] += abs(complex(M.diagonal().sum())) ** 2
    _monomial_sweep(gens, hook)
    return total[0] / float(1 << len(gens))


def commutant_dim_by_nullspace(gens):
    """测法二 · **直接零空间** (只在小维数上可用)。

    [X, γ_a] = 0 对 a 逐个成立 ⟺ X 在全部 γ 的中心化子里。把 X 展平成 N² 维向量,
    第 a 条约束的系数矩阵是 (γ_a^T ⊗ I − I ⊗ γ_a); 堆叠后取零空间维数。
    N=256 时该矩阵是 524288×65536 (密集 274 GB) ⇒ 本测法只在 N ≤ 8 上跑,
    用途是**验证测法一**, 不是替代它。
    """
    n, N = len(gens), gens[0].shape[0]
    eye = np.eye(N, dtype=complex)
    K = np.vstack([np.kron(np.asarray(g).T, eye) - np.kron(eye, np.asarray(g))
                   for g in gens])
    return int(N * N - np.linalg.matrix_rank(K, tol=1e-9))


def commutant_project(X, gens):
    """P(X) = (1/2^n) Σ_S γ_S X γ_S† —— 到 comm(γ 生成的代数) 的**正交投影**。

    γ_S 的 ±1 符号在两侧各出现一次而抵消, 故只用 2^n 个单项式 (完整群 {±γ_S}
    是它的 2 倍, 平均值相同)。用于核验 P 确实**落在**中心化子里且**幂等**。"""
    N = X.shape[0]
    acc = np.zeros((N, N), dtype=complex)
    cnt = [0]

    def hook(M):
        Md = M.toarray() if sp.issparse(M) else M
        acc[:] += Md @ X @ Md.conj().T
        cnt[0] += 1
    _monomial_sweep(gens, hook)
    return acc / float(cnt[0])


def commutant_projector_trace(gens):
    """P 的**算子迹** tr P = Σ_{ij} (P(E_ij))_{ij} —— 逐基矢实算。

    这是继「迹公式」与「零空间」之后的**第三个**独立算法, 只在小维数上跑得动
    (代价 N² 次投影); 它的用途是给测法一再上一道锁。"""
    N = gens[0].shape[0]
    tot = 0.0 + 0j
    for i in range(N):
        for j in range(N):
            E = np.zeros((N, N), dtype=complex)
            E[i, j] = 1.0
            tot += commutant_project(E, gens)[i, j]
    return float(tot.real)


def clifford_residual(gens):
    """γ_a² = I 与 {γ_a, γ_b} = 2δ_ab 的最差残差 —— 载体合法性的**唯一**入口。"""
    n = len(gens)
    worst = 0.0
    for a in range(n):
        ga = np.asarray(gens[a])
        worst = max(worst, float(np.abs(ga @ ga - np.eye(ga.shape[0])).max()))
        for b in range(a + 1, n):
            worst = max(worst, float(np.abs(cg.antic(ga, np.asarray(gens[b]))).max()))
    return worst


# ============================================================================
# A · D5 (α1 · 不可约性口径)
# ============================================================================

def part_A():
    t0 = time.time()
    print("\n  A · D5 (α1) —— 把「偶数中取 2」的排除理由换成**不可约性**")
    print(f"      L={L}: 物理载体 2^L={DIM} 维, d_local=4 的载体 4^L={DIM4} 维")

    # --- A1 小例对账: 两套独立测法必须给出同一个数 ----------------------
    print("\n    A1 · 测法对账 (小维数, 两套测法都跑得动)")
    for lbl, nh, mu in [('2 个 Majorana, 2 维', 1, 1),
                        ('2 个 Majorana, 4 维', 1, 2),
                        ('4 个 Majorana, 8 维', 2, 2)]:
        g = carrier(nh, mu)
        d_tr = commutant_dim_by_traces(g)
        d_ns = commutant_dim_by_nullspace(g)
        check(f'{lbl}: 迹公式 == 零空间 == mult² = {mu ** 2}',
              abs(d_tr - mu ** 2) < 1e-9 and d_ns == mu ** 2,
              f'迹公式 {d_tr:g}, 零空间 {d_ns}, 载体残差 {clifford_residual(g):.1e}')

    # 活的反事实: 把一条 γ 换成重复的 γ_0 ⇒ 不再是 Clifford 集,
    # 而**载体合法性检查必须当场翻红** (否则 A1 的数字无意义)。
    bad = carrier(1, 2)
    bad[1] = bad[0]
    w_bad = clifford_residual(bad)
    check('A1 **活的反事实**: 重复一条 γ ⇒ Clifford 残差立刻 O(1)',
          w_bad > 1.0, f'残差 = {w_bad:.2f} ⇒ A1 的合法性闸有分辨力')

    # --- A2 投影子结构核验 --------------------------------------------
    print("\n    A2 · 投影子结构核验 (2 个格点 × 重数 2 ⇒ 8 维, 共 16 项)")
    g8 = carrier(2, 2)
    rng = np.random.default_rng(11)
    X = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
    PX = commutant_project(X, g8)
    P2X = commutant_project(PX, g8)
    comm_w = max(float(np.abs(PX @ np.asarray(ga) - np.asarray(ga) @ PX).max())
                 for ga in g8)
    check('P(X) 落在中心化子里 (与全部 4 个 γ **对易**)', comm_w < 1e-10,
          f'最差 ‖[P(X), γ_a]‖ = {comm_w:.2e} (用对易子, 不是反对易子)')
    check('P 是投影 (P² = P)', float(np.abs(P2X - PX).max()) < 1e-10,
          f'max|P(P(X)) − P(X)| = {float(np.abs(P2X - PX).max()):.2e}')
    tr_P = commutant_projector_trace(g8)
    check('P 的**算子迹** == 迹公式 == 零空间 == 4 (三种算法对账)',
          abs(tr_P - 4.0) < 1e-9 and abs(commutant_dim_by_traces(g8) - 4.0) < 1e-9,
          f'tr P = {tr_P:.6f}; 迹公式 = {commutant_dim_by_traces(g8):g}; '
          f'零空间 = {commutant_dim_by_nullspace(g8)}')

    # --- A3 物理载体: 不可约 ------------------------------------------
    print(f"\n    A3 · 物理载体 (d_local=2): 8 个 Majorana 作用在 2^L={DIM} 维上")
    g_phys = carrier(L, 1)
    d_phys = commutant_dim_by_traces(g_phys)
    check('物理载体的中心化子维数 = 1 ⇒ **不可约**', abs(d_phys - 1.0) < 1e-9,
          f'dim comm = {d_phys:g}, dim = {DIM}, 载体残差 {clifford_residual(g_phys):.1e}')

    # --- A4 D5-a: L=4 上的 2L 载体必可约 ------------------------------
    print(f"\n    A4 · D5-a: 2L=8 个 Majorana 作用在 4^L={DIM4} 维上的**一切**载体")
    g4 = carrier(L, DIM4 // DIM)              # 即既有 A5 的 γ_a ⊗ I₁₆
    d_g4 = commutant_dim_by_traces(to_sparse(g4))
    want4 = float((DIM4 // DIM) ** 2)
    check(f'标准载体 γ_a ⊗ I₁₆: dim comm = {want4:.0f} = (2^L)² > 1 ⇒ 可约',
          abs(d_g4 - want4) < 1e-6,
          f'dim comm = {d_g4:g}, 重数 = {DIM4 // DIM} = 2^L')
    g4r = conjugate(g4, 20260926)
    d_g4r = commutant_dim_by_traces(g4r)
    check('随机酉共轭后的载体: 同一个读数 ⇒ 与载体取法无关',
          abs(d_g4r - want4) < 1e-4 and clifford_residual(g4r) < 1e-12,
          f'dim comm = {d_g4r:.6f}, 载体残差 {clifford_residual(g4r):.1e}')

    # --- A5 D5-b: 维数账 ----------------------------------------------
    print("\n    A5 · D5-b 维数账 (对账, 零机时)")
    mult = DIM4 // DIM
    check('Cl(2L) 不可约表示维数 = 2^L; 4^L = 2^L · 2^L ⇒ 重数 = 2^L = 16',
          DIM4 == DIM * mult and mult == 2 ** L,
          f'4^{L} = {DIM4} = {DIM} · {mult}; dim comm = (2^L)² = {mult ** 2} = 4^L')
    check('实测 dim comm 与维数账逐位对上', abs(d_g4 - mult ** 2) < 1e-6,
          f'实测 {d_g4:g} == 账 {mult ** 2}')

    # --- A6 D5-c: 不可约的 d=4 载体 -----------------------------------
    print(f"\n    A6 · D5-c: **不可约**的 d=4 载体 (每格点两对 ⇒ 4L={4 * L} 个 Majorana)")
    g_irr4 = carrier(2 * L, 1)                # 16 个 Majorana, 2^8 = 256 维
    d_irr4 = commutant_dim_by_traces(to_sparse(g_irr4))
    check('16 个 Majorana 在 256 维载体上的中心化子维数 = 1 ⇒ 不可约',
          abs(d_irr4 - 1.0) < 1e-9,
          f'dim comm = {d_irr4:g}, dim = {g_irr4[0].shape[0]}; '
          f'载体残差 {clifford_residual(g_irr4):.1e}')
    check('⇒ 「每格点恰好一对 Majorana」是那个**必需的**给定', d_irr4 < d_g4,
          f'每格点一对 ⇒ 只能到 2^L 维 (不可约); 要 4^L 维且不可约 ⇒ 每格点**两对**')

    # --- A7 D5-c′: Clifford 真空 vs L1 空无 (只作诊断, 不登记) ---------
    print("\n    A7 · D5-c′ (pre2 思路 A) —— Clifford 真空与 L1 空无的重叠")
    print("      口径: 该比较只能在 **d_local=2 载体**上做 —— L1 的空无态是 2^L 维,")
    print(f"            而 A6 的不可约 d=4 载体是 {DIM4} 维, 两者不在同一个空间里。")
    cf = cg.real_fermions([np.asarray(g) for g in g_phys], L)
    omega = cg.uv_vacuum(L)
    vac_w = max(float(np.abs(c @ omega).max()) for c in cf)
    check('c_j 确实湮灭 uv_vacuum ⇒ 它就是 JW 费米子真空 (实测, 非假设)',
          vac_w < 1e-12, f'max_j |c_j|Ω⟩| = {vac_w:.2e}')
    cc_w = max(float(np.abs(cg.antic(cf[i], cf[j].conj().T)
                            - (1.0 if i == j else 0.0) * np.eye(DIM)).max())
               for i in range(L) for j in range(L))
    check('且 {c_i, c_j†} = δ_ij', cc_w < 1e-12, f'最差残差 {cc_w:.2e}')
    s1 = G.stage1_void(2)
    r1 = G.derive_L1_void_to_chain(s1['equal_state'], L)
    ov = float(abs(np.vdot(omega, r1['void_full'])) ** 2)
    check('**诊断打印** (DP-6 未定前不登记守卫): 重叠 = 0', ov < 1e-24,
          f'|⟨Ω|void_full⟩|² = {ov:.3e}。pre2 (v16.4_pre2.md:103) 预期 ≈ 1; '
          '计划 §1.6 订正丙预期 ≪ 1 —— 实测**恰为 0**: |Ω⟩ = |−⟩^{⊗L} 与 '
          'L1 的 |+⟩^{⊗L} 逐位正交。**订正丙成立, 且比预期更硬**')

    # --- A8 D5-d: 复核既有 A5 反事实 ----------------------------------
    print("\n    A8 · D5-d: 复核既有 A5 反事实的构造 (读码 + 实测)")
    I2 = np.eye(DIM)                          # `_v16_smoke_derive_presets.py:123`
    check('A5 里的 I2 = np.eye(DIM) 是 16×16, **不是** 2×2',
          I2.shape == (DIM, DIM) and DIM == 16,
          f'shape = {I2.shape} (L={L} ⇒ DIM = 2^{L} = {DIM}) ⇒ G4 = γ_a ⊗ I₁₆')
    check('该载体实测 dim comm = 256 ⇒ 它**本身就是可约载体**',
          abs(d_g4 - want4) < 1e-6,
          f'dim comm = {d_g4:g} ⇒ 它不是「定理不排除 d=4」的证据: 对**奇偶引理**'
          '成立, 对**不可约性定理**恰是被排除的那一类')

    print("\n      A 的判定 (D5 · α1 · 不可约性口径):")
    print(f"        · **实测**: L=4 上 2L=8 个 Majorana 在 4^L={DIM4} 维上的**任何**载体,")
    print(f"          中心化子维数 = {d_g4:.0f} (> 1) ⇒ **必可约** (随机酉共轭给出同一读数)。")
    print(f"        · **实测**: 同一批 8 个 Majorana 在 2^L={DIM} 维上, 中心化子维数 = 1 ⇒ 不可约。")
    print(f"        · **实测**: 不可约的 d=4 载体要 4L={4 * L} 个 Majorana = 每格点**两对**。")
    print(f"        · **维数账** (D5-b): 重数 = {mult} = 2^L, dim comm = (2^L)² = 4^L = {DIM4} ✓")
    print("        · **落支 = S-部分** ⭐: `d_local=2` 由「奇偶 + 不可约 + 每格点一对」导出。")
    print("          「每格点恰好一对 Majorana」**等价于 JW 的 site↔mode 对应 = 费米子表述选择**")
    print("          ⇒ 选择的**个数仍为 1**, 天花板不变。变的只是排除理由:")
    print("          从「一条最小性(选择)」换成「一条定理 + 那个表述选择」。")
    print("        · 顺带: 审计 §5.5 那句「排除它靠…的最小性」须按 v16.3 纪律**并存订正**。")
    print(f"    (A 段用时 {time.time() - t0:.1f} s)")


# ============================================================================
# B · D6 (α2 · 复数域在 d_local=4 下是否承重)
# ============================================================================

def part_B():
    print("\n  B · D6 (α2) —— `d_local=4` 是否迫使 L1 的态变复")
    print("      代码预判: 否。本段只做**取证**, 不去「再找一条路」。")

    s4 = G.stage1_void(4)
    eq4 = s4['equal_state']
    r4 = realness(eq4)
    check('B1 `stage1_void(4)` 的 equal_state 是实的 (相位不变判据 |Σψ²| = 1)',
          abs(r4 - 1.0) < 1e-12,
          f'|Σψ²| = {r4:.16f}, d = {eq4.size}, equal_state = ones(4)/2, '
          f'|Im|max = {float(np.abs(eq4.imag).max()):.1e}')

    # 活的反事实: 加**相对**相位 ⇒ 判据必须翻红。
    ph = np.where(np.arange(eq4.size) % 2 == 0, 1.0 + 0j, np.exp(1j * 0.7))
    rc = realness(ph * eq4)
    check('B1 **活的反事实**: 加相对相位后判据立刻 < 1', rc < 0.999,
          f'|Σ(ψ·e^(iφ_j))²| = {rc:.6f} ⇒ 该判据对"复"有分辨力')

    # Q 的承重性检验是**按签名**做的, 与维度无关 ⇒ d=4 档同样成立。
    sigs = {'derive_L1_void_to_chain': G.derive_L1_void_to_chain,
            'l1_void_pushaway': G.l1_void_pushaway,
            'stage2_critical_break': G.stage2_critical_break}
    no_q = all(not any(('unit' in p.lower()) or (p == 'Q') for p in
                       inspect.signature(fn).parameters) for fn in sigs.values())
    check('B2 链上三个函数的签名里都没有 Q/unitary (该检验与维度无关)',
          no_q, '、'.join(sigs))

    def _cf(unitary, equal_state, L=4):   # noqa: ARG001
        return None
    check('B2 **活的反事实**: 签名里加上 unitary, 同一条检验会翻红',
          any('unit' in p.lower() for p in inspect.signature(_cf).parameters),
          f'反事实签名 = {list(inspect.signature(_cf).parameters)}')

    print("\n      B 的判定 (D6):")
    print("        · **落支 = S-负**: `d_local=4` 不迫使 L1 态变复 —— equal_state 在")
    print("          d=4 档仍是 `ones(4)/2` (数值上是实的); 唯一真复的对象 Q 仍无消费者,")
    print("          且那条承重性检验是**按签名**做的、与维度无关。")
    print("        · ⇒ α2 **给不出** `d_local=2` 的独立支撑。这是**确定性负结论**,")
    print("          本段的产出就是把它登记下来, **不是**去另找一条路。")


# ============================================================================
# C · D7 (α3 · 随机测度唯一性)
# ============================================================================

def part_C():
    print("\n  C · D7 (α3) —— 导出「随机测度」是否缩小缺口")

    for d in (2, 3, 4):
        psi = np.ones(d, dtype=complex) / np.sqrt(d)
        rho = np.outer(psi, psi.conj())
        ev = np.linalg.eigvalsh(rho).real
        p = ev[ev > 1e-15]
        s_vn = float(-np.sum(p * np.log(p)))
        rho_mix = np.eye(d, dtype=complex) / d
        ev_m = np.linalg.eigvalsh(rho_mix).real
        s_mix = float(-np.sum(ev_m * np.log(ev_m)))
        s_shannon = float(np.log(d))
        check(f'C d={d}: ones({d})/√{d} 是**纯态** ⇒ S(ρ) = 0, 不是 von Neumann 熵的极大点',
              abs(s_vn) < 1e-12 and abs(s_mix - np.log(d)) < 1e-12,
              f'S(纯态) = {s_vn:.1e}; 而 S(I/d) = {s_mix:.6f} = ln {d}; '
              f'概率矢量的 Shannon 熵 = {s_shannon:.6f} = ln {d} (这才是「均匀 ⇒ 最大」)')

    # 活的反事实: 把概率矢量推离均匀 ⇒ Shannon 熵必须严格下降。
    p_uni, p_skew = np.ones(4) / 4.0, np.array([0.7, 0.1, 0.1, 0.1])
    s_uni = float(-np.sum(p_uni * np.log(p_uni)))
    s_skew = float(-np.sum(p_skew * np.log(p_skew)))
    check('C **活的反事实**: 推离均匀 ⇒ Shannon 熵严格下降 (该判据有分辨力)',
          s_skew < s_uni, f'{s_skew:.6f} < {s_uni:.6f}')

    print("\n      C 的判定 (D7):")
    print("        · **一个口径事实**: 本仓库的「最大熵」指**概率矢量的 Shannon 熵** ——")
    print("          `negctrl.py:290` 逐字「熵 <= ln N (均匀分布最大熵)」;")
    print("          `stage67.py:481` 逐字「probs = 1/N 与 entropy = ln N 是同一件事的换写…」")
    print("          「它们是「空无」这个名字的**定义**, 不是关于系统的观测」。")
    print("        · **实测**: `ones(d)/√d` 是**纯态**, S(ρ) ≡ 0 —— 它是 von Neumann 熵的")
    print("          **极小**点, 不是极大点 (极大的那一个是 I/d, 实测 = ln d)。")
    print("        · **落支 = S-重言**: 「均匀分布极大化 Shannon 熵」与「Haar 是唯一酉不变")
    print("          测度」都是**现成定理的并列**, 没有导出任何新东西。")
    print("        · 且审计 §5.5 已判 #2 与 #3「都只活在 stage1_void 的 QR 块里, 且**都不承重**」")
    print("          ⇒ 导出一条**不承重**的给定**不缩小缺口**。**本条不动缺口。**")


def main():
    t0 = time.time()
    print("=== v16.4 · 预设层攻坚冒烟 (α: D5 / D6 / D7) ===")
    print("    口径: 不新增守卫/指标, 不改任何既有读数 ⇒ 规模三数 78/27/13 逐字不变。")
    part_A()
    part_B()
    part_C()
    print("\n  === α 三条的最终地位 (取代 §5.5 的「偶数中取 2 仍是选择」) ===")
    print("    #1 d_local=2  : 落 **S-部分** —— 由「奇偶 + 不可约 + 每格点一对」导出;")
    print("                    最后那条 == 费米子表述选择 ⇒ **选择个数仍为 1**。")
    print("    #2 复数域     : 落 **S-负** —— d=4 不迫使 L1 变复, α2 无独立支撑。")
    print("    #3 随机测度   : 落 **S-重言** —— 导出的是**不承重**的给定, 缺口不缩小。")
    print("    ⇒ α 命中 L1 升级**条件 ②**、不碰 ① ⇒ **即便 α 全成功, L1 仍记 B**。")
    print("      **本文件不宣称预设层缺口已修复。**")
    print(f"    (总用时 {time.time() - t0:.1f} s)")

    print(f"\n=== 结果: "
          f"{'全部通过' if not _FAILS else '**失败 ' + str(len(_FAILS)) + ' 项**'}"
          f" ===")
    for f in _FAILS:
        print(f"    · {f}")
    return 0 if not _FAILS else 1


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.exit(main())
