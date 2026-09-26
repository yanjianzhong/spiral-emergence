# -*- coding: utf-8 -*-
"""
v16 · S2 冒烟 —— L4: 三分块熵机器 + 互信息的单调性 (MMI)  [_VERSION_TAG = 'v16-smoke-l4-mmi-1']

**这是 v16.2_plan.md §5.2 的 S2**, 不进 runner, 不动任何登记值, 不碰生产路径。

要回答的问题 (背景见 plan 订正③):
  L4 现在的判据只有 `纠缠熵 KL 散度` (`solvers.py:217-259`, 与精确对角化比谱),
  它**至今是 8 条孤儿之一** (audit §6.3)。本冒烟试一条**真能失败**的第二判据。

**为什么不是强次可加性 (SSA)**: `I(A:C) <= I(A:B) + I(B:C)` 是一条**定理**, 对所有量子态
恒成立 => 它**在物理上不可能失败**, 只能抓实现 bug。拿它当判据 = 把恒真式当判据。

**能失败的是互信息的单调性 (MMI)**:
    I3(A:B:C) = S(A) + S(B) + S(C) - S(AB) - S(AC) - S(BC) + S(ABC)  <=  0
它对**有几何对偶的全息态**成立, 对**一般量子态可被违反** (GHZ 是反例)。
L4 的主张是「自指画定边界, 投影显全息」, 其接口正是 RT => **MMI 是"存在几何对偶"的
必要条件**。测它 = 测 L4 的主张能不能立。

⚠️ **天花板 (与结果同页, 不许省)**:
  1. MMI 成立只是**必要**条件, 不是充分条件。通过 != RT 被验证, 更 != "涌现时空"
     (后者是 B2 的硬边界明令禁止的宣称, `spiral_model_v16.py:52-53`)。
  2. 本工作尺度上"空间区域"这个概念本身是弱的: 实测 `xi ~ 19.3 > L = 16`, 即
     **整条环落在一个关联长度之内**。此尺度上谈"分离的区域"是**格点尺度**的, 不是几何的。
  3. => 无论 I3 是正是负, 判词只能是「**在 L=16 的可及尺度上**, 该态的纠缠结构**是否**
     满足几何对偶的必要条件」, **不能**写成「L4 被验证/被证否」。
  4. **成熟度不动**。L4 是 C (audit `:75`), 升 B 靠的是这层物理结论本身的强度, 不是多一条判据。

三项验收:
  1. **reshape 轴序自校验**: 用本模块的块熵 helper 重算**已登记过的量** ——
     `S(前 8 站)` 必须与 `entanglement_curve(gs, 16)[7]` **逐位一致** (同一 gs, 同一 SVD)。
     不过 => 轴序错, **全部作废**。
  2. **SSA 自校验** (身份 = 自校验, **不是判据**): 必须成立; 不成立 => 实现错。
     报告中 SSA 与 MMI **分列两行并各标身份** —— 否则"SSA 通过"会被读成 L4 有证据。
  3. **MMI 主读数**: I3 的值与符号, 以及 I3(w) for w = 1..4 的**趋势**。

**不预设 I3 的符号** —— `xi > L` 意味着所有块强关联, 是否满足 MMI 事先不可知。
写了"预期 I3 < 0"就是预设结论。

退出码: 0 = 三项都完成; 3 = **轴序自校验或 SSA 失败**(实现错, 结论作废); 2 = 脚本自身错误。

用法:
    python v16/_v16_smoke_l4_mmi.py
"""

import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'v16'))

import numpy as np  # noqa: E402
import spiral_model_v16 as G  # noqa: E402
import spiral_metric_v16 as MM  # noqa: E402

_VERSION_TAG = 'v16-smoke-l4-mmi-1'

L = 16
J, H = 1.0, 1.0              # 临界点: 与 `_metric/main.py` 的 exact_ground_state(L,1,1) 同参数
SSC_TOL = 1e-9               # 自校验的数值容差 (SSA 是精确的, 只留浮点余量)

_FAILS = []


def check(label, cond, detail=''):
    tag = 'PASS' if cond else '**FAIL**'
    print(f"    [{tag}] {label}" + (f"  —— {detail}" if detail else ''))
    if not cond:
        _FAILS.append(label)


def block_entropy(psi, L, blocks):
    """
    任意**连续站点块并集**的冯诺依曼熵。

    blocks: [(start, n), ...], 站点按 0..L-1 编号。
    站点约定**沿用现有代码** (`core.py:630-631` 的 `boundary_correlation_graph` 逐字:
    "站点 0 = 最高有效位 (与 tfi_periodic_sparse 的约定一致)")。
    在 `psi.reshape([2]*L)` 里 axis 0 就是最高有效位 => 连续站点 <-> 连续 axis,
    **不需要转置** —— 这正是"能直接 reshape"的前提, 由验收 1 逐位校验。

    与 `core.py:590` / `solvers.py:51` 的 `entanglement_curve` 是同一个二分定义
    (把 blocks 的并集当作 A, 其余当作 B), 区别是 A 不必是前缀、也不必连续。
    """
    t = psi.reshape([2] * L)
    axes = []
    for s, n in blocks:
        axes.extend(range(s, s + n))
    rest = [a for a in range(L) if a not in axes]
    m = np.transpose(t, axes + rest).reshape(2 ** len(axes), -1)
    sv = np.linalg.svd(m, compute_uv=False)
    p = sv ** 2
    p = p[p > 1e-15]
    return float(-np.sum(p * np.log(p)))


def main():
    t0 = time.time()
    print(f"=== {_VERSION_TAG} · L4: 三分块熵 + 互信息的单调性 (MMI), L={L} ===")
    print(f"    口径: 真调 exact_ground_state({L},{J:g},{H:g}) 与 entanglement_curve; "
          f"不进 runner, 不改登记值。")

    tb = time.time()
    E0, gs = G.exact_ground_state(L, J, H)
    print(f"\n--- 0 · 精确基态 ---")
    print(f"    E0 = {E0:.12f} (JW 解析 {G.jw_ground_energy(L, J, H):.12f}), "
          f"dim = {2 ** L}, 建态用时 {time.time() - tb:.2f} s")

    # =====================================================================
    print(f"\n--- 1 · reshape 轴序自校验 (必须先过) ---")
    ec = MM.entanglement_curve(gs, L)          # n = 1..8, 索引 7 即 n=8
    s_prefix8 = block_entropy(gs, L, [(0, 8)])
    print(f"    entanglement_curve(gs,{L})[7] = {ec[7]:.15f}   (前 8 站平分)")
    print(f"    本模块 block_entropy([(0,8)])  = {s_prefix8:.15f}")
    d_ax = abs(s_prefix8 - ec[7])
    check("S(前 8 站) 与 entanglement_curve[7] 逐位一致", d_ax < 1e-12,
          f"|diff| = {d_ax:.3e}")

    # 顺带把整条曲线都核一遍 (免费, 且能抓住只在特定 n 上暴露的轴序错)
    d_all = max(abs(block_entropy(gs, L, [(0, n)]) - ec[n - 1])
                for n in range(1, L // 2 + 1))
    check("整条 entanglement_curve (n=1..8) 都对得上", d_all < 1e-12,
          f"max|diff| = {d_all:.3e}")

    # =====================================================================
    print(f"\n--- 2 · 块熵表 + SSA 自校验 (身份: 自校验, **不是判据**) ---")
    print(f"    {'w':>3s} | {'S(A)':>10s} {'S(B)':>10s} {'S(C)':>10s} {'S(AB)':>10s} "
          f"{'S(AC)':>10s} {'S(BC)':>10s} {'S(ABC)':>10s} {'S(D)':>10s} | "
          f"{'S(ABC)-S(D)':>11s} {'SSA 余量':>11s}")

    rows = []
    for w in (1, 2, 3, 4):
        A, B, C = (0, w), (w, w), (2 * w, w)
        D = (3 * w, L - 3 * w)
        sA = block_entropy(gs, L, [A])
        sB = block_entropy(gs, L, [B])
        sC = block_entropy(gs, L, [C])
        sAB = block_entropy(gs, L, [A, B])
        sAC = block_entropy(gs, L, [A, C])
        sBC = block_entropy(gs, L, [B, C])
        sABC = block_entropy(gs, L, [A, B, C])
        sD = block_entropy(gs, L, [D])
        # MMI
        i3 = sA + sB + sC - sAB - sAC - sBC + sABC
        # SSA: I(A:C) <= I(A:B) + I(B:C)   (等价于 I(A:C|B) >= 0)
        i_ac = sA + sC - sAC
        i_ab = sA + sB - sAB
        i_bc = sB + sC - sBC
        ssa_margin = i_ab + i_bc - i_ac
        rows.append(dict(w=w, sA=sA, sB=sB, sC=sC, sAB=sAB, sAC=sAC, sBC=sBC,
                         sABC=sABC, sD=sD, i3=i3, i_ac=i_ac, i_ab=i_ab, i_bc=i_bc,
                         ssa=ssa_margin, purity=sABC - sD))
        print(f"    {w:3d} | {sA:10.6f} {sB:10.6f} {sC:10.6f} {sAB:10.6f} "
              f"{sAC:10.6f} {sBC:10.6f} {sABC:10.6f} {sD:10.6f} | "
              f"{sABC - sD:11.2e} {ssa_margin:11.2e}")

    mp = max(abs(r['purity']) for r in rows)
    check("纯态自校验: S(ABC) == S(D) (整体是纯态)", mp < 1e-12, f"max|diff| = {mp:.3e}")

    mssa = min(r['ssa'] for r in rows)
    check("SSA 成立 (违反即实现错, **不是物理判据**)", mssa > -SSC_TOL,
          f"最小余量 = {mssa:.3e}")

    # 平移不变性: 周期临界 TFI 的基态平移不变 => 等宽块的熵应相等
    dev_block = max(max(abs(r['sA'] - r['sB']), abs(r['sB'] - r['sC']),
                        abs(r['sAB'] - r['sBC'])) for r in rows)
    check("平移不变自校验: 等宽块的熵相等", dev_block < 1e-9, f"max|diff| = {dev_block:.3e}")

    # =====================================================================
    print(f"\n--- 3 · MMI 主读数 (身份: **判据**) ---")
    print(f"    {'w':>3s} | {'I3(A:B:C)':>13s} {'符号':>5s} | {'I(A:B)':>10s} "
          f"{'I(B:C)':>10s} {'I(A:C)':>10s}")
    for r in rows:
        sgn = '<=0' if r['i3'] <= 0 else '>0'
        print(f"    {r['w']:3d} | {r['i3']:13.8f} {sgn:>5s} | "
              f"{r['i_ab']:10.6f} {r['i_bc']:10.6f} {r['i_ac']:10.6f}")

    signs = {('<=0' if r['i3'] <= 0 else '>0') for r in rows}
    seq = ', '.join(f"{r['i3']:.6f}" for r in rows)
    print(f"\n    I3 的符号集合: {sorted(signs)}")
    print(f"    I3(w) 序列 = [{seq}]")
    if signs == {'<=0'}:
        print(f"    => **在 L={L} 的可及尺度上, 几何对偶的必要条件通过**。")
        print(f"       但这是**必要**条件 —— 不构成「RT 被验证」, 更不是「涌现时空」(天花板 1)。")
    elif signs == {'>0'}:
        print(f"    => **该尺度上必要条件不成立** (负结果, 照登)。")
        print(f"       成因是尺度: xi ~ 19.3 > L = {L}, 整条环在一个关联长度内 (天花板 2);")
        print(f"       **不等于**「RT 被证否」—— 它只说这一尺度上谈几何区域是弱的。")
    else:
        print(f"    => **符号随 w 变号**, 按 w 逐条登记, 不合并成一句话。")
    trend = rows[-1]['i3'] - rows[0]['i3']
    print(f"    趋势 I3(w=4) - I3(w=1) = {trend:+.8f}")
    print(f"    (计划里可证伪的形态是「随 w 增大趋于 0」; 不出现就照登「未观察到」)")

    print(f"\n--- 天花板 (与结果同页) ---")
    print(f"    xi ~ 19.3 > L = {L} 实测 (`spiral_model_v16.py:274/309-310`)")
    print(f"    => 判词范围: 「在 L={L} 的可及尺度上是否满足必要条件」;")
    print(f"       不得写成「L4 被验证/被证否」, 不得宣称涌现时空。")
    print(f"    成熟度不动: L4 仍是 C (audit :75), 升 B 不靠多一条判据。")
    print(f"    SSA 一行是**自校验**, MMI 一行才是判据 —— 两者不可互换引用。")

    ok = not _FAILS
    print(f"\n=== 结果: 自校验 {'通过' if ok else '**失败 => 结论作废**'}; "
          f"累计用时 {time.time() - t0:.1f} s ===")
    if _FAILS:
        print(f"    失败项: {_FAILS}")
    return 0 if ok else 3


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:                      # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"\n**脚本自身错误**: {type(exc).__name__}: {exc}")
        sys.exit(2)
