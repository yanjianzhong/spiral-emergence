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
"""
v16 · 预设层的**真导出**尝试 —— `d_local=2` 与复数域
[_VERSION_TAG = 'v16-smoke-derive-presets-1']

**与 `_v16_smoke_presets.py` 的分工**: 那一份只把「给定」细分成「承重 / 不承重」,
**明确不导出任何预设**; 本文件是它的下一步 —— 用户 2026-09-25 指定: **真的去导出**。
口径照旧: **不新增守卫或指标**(那会动 27 指标 / 78 守卫的分母), 不重跑全流程,
不改任何阈值。每条检验配一个**活的反事实**, 并**明写导不出来的那一半**。

────────────────────────────────────────────────────────────────────────
**结论先行**(全文要证的只有这三句, 其余都是支撑):

  A. `d_local` 的**奇偶**可以被导出, 而且是**初等的** —— 不需要 Clifford 代数:
     取 A=γ₀, B=γ₁, 由 A²=B²=I、{A,B}=0 ⇒ 对任意 v∈ker(A−I) 有 A(Bv)=−Bv,
     而 B 可逆 ⇒ B 把 ker(A−I) **双射**到 ker(A+I) ⇒ n₊=n₋ ⇒ **dim 必为偶数**。
     又 dim = d_local^L ⇒ **d_local 必为偶数** ⇒ `d_local=3` **不是"代码没支持",
     是数学上不可能**(3^L 为奇数, 引理给出直接矛盾)。
  A'.**但偶数里取 2 仍是选择**(v16.4·D5 订正: 理由已换): `d_local=4` 上确能造出 2L 个 Majorana
     (本文件实测), 但那个载体 `γ_a ⊗ I₁₆` **可约**(dim comm = 256 > 1) ⇒ **被定理排除**, 非最小性。
  B. 复数域**不由 L1 导出, 也不能由 L1 导出**: L1 的链上态与 L2 的 TFI 基态实测
     都是**实的**(相位不变判据 |Σψ²| = 1) ⇒ L1→L2 的全部读数在**实** Hilbert 空间
     就能复现。那个 `i` 是在**下游**被**厄米性**逼出来的: Majorana 双线性 γ_aγ_b
     实测**反厄米**, 而 H 必须厄米 ⇒ 必须有 `i`(实测 iγ_aγ_b 厄米)。口径 A 的边界项
     逐字就是 `J·iγ_{2L-1}γ_0`, 而 `build_h` 存的那个 2L×2L 矩阵**是实的** ——
     `i` 正是把它变成哈密顿量的那个因子(代码自己的实现见 `reconstruct`)。
     注: 本实现在**两处**出现 `i`, 地位不同 —— (a) 载体的 `γ_{2i+1}` 自带 i
     (JW 的 σy 约定; 实测纯虚; **属基底选取, 本文件不导出它**); (b) 双线性形式的
     前因子 `0.25j`(**必需**: 抽掉后 H 立刻反厄米, 实测)。B2a 之所以能一般化,
     是因为 `(γ_aγ_b)† = γ_bγ_a` 只用到 `γ†=γ`, 与基底选取无关。

**必须写明的边界(不许夸大的两处)**:
  ① A 是**条件命题**: 前提是「L2 用 Majorana 表述」。自旋表述(L2 的 TFI 自己)用
     **实**系数的 σzσz / σx 就厄米, **根本不需要 `i`**。所以 B 导出的不是
     「宇宙必须是复的」, 而是「**一旦选费米子表述, `i` 就是被强制的**」。
  ② A 与 B 的可导出部分**共用同一个前提** —— 那个表述选择。⇒ 两条预设并没有被
     **分别**导出, 它们**一起**随一个选择进来。真正剩下的"给定"是**这个选择本身**
     加上 A' 的最小性。这比 §5.4 原先记的「三条各自给定」要小, 但**不是零**。

用法:
    python v16/_v16_smoke_derive_presets.py
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'v16'))

import numpy as np  # noqa: E402
import _v16_cmera_gaussian as cg  # noqa: E402
import spiral_model_v16 as G  # noqa: E402

L = 4                 # 站点数; dim = 2**L = 16, 稠密 Majorana 完全跑得动
DIM = 2 ** L
_FAILS = []


def check(label, cond, detail=''):
    tag = 'PASS' if cond else '**FAIL**'
    print(f"    [{tag}] {label}" + (f"  —— {detail}" if detail else ''))
    if not cond:
        _FAILS.append(label)


def realness(psi):
    """|Σ_i ψ_i²| —— **相位不变**判据: 等于 1 ⟺ ψ 在**整体相位**下是实的。
       整体相位不改变 Σψ²; 任何**相对**相位都会把它拉小于 1。"""
    p = np.asarray(psi).ravel()
    return float(abs(np.sum(p * p)))


def part_A():
    print("\n  A · 导出 d_local=2 —— 能导到哪一步, 如实划界")
    print(f"      载体: cg.majorana(a, L) 给出的 2L={2 * L} 个算符, 作用在 dim=2^{L}={DIM} 上")

    Gam = [cg.majorana(a, L) for a in range(2 * L)]

    # --- A0/A1 先把前提本身量出来, 不假设 ----------------------------
    # 实测结构: 本实现的 JW 约定是 γ_{2i} 实、γ_{2i+1} 纯虚 (σy 自带 i)。
    # **这条只登记结构, 不是结论** —— 载体那个 i 是基底的选取, 本文件不导它。
    re_even = max(float(np.abs(np.asarray(Gam[2 * i]).imag).max()) for i in range(L))
    im_odd = max(float(np.abs(np.asarray(Gam[2 * i + 1]).real).max()) for i in range(L))
    check('A0 载体结构(实测): γ_2i 为实、γ_2i+1 为纯虚 (JW 的 σy 约定)',
          re_even == 0.0 and im_odd == 0.0,
          f'max|Im γ_2i| = {re_even:.1e}, max|Re γ_2i+1| = {im_odd:.1e} '
          '⇒ 此为基底选取, 与 B2 要导的那个 i 分开记账')

    cl_w = 0.0
    for a in range(2 * L):
        cl_w = max(cl_w, float(np.abs(Gam[a] @ Gam[a] - np.eye(DIM)).max()))
        for b in range(a + 1, 2 * L):
            cl_w = max(cl_w, float(np.abs(cg.antic(Gam[a], Gam[b])).max()))
    check('A1 载体合法: γ_a²=I 且 {γ_a,γ_b}=2δ_ab (2L 个, 两两)',
          cl_w < 1e-12, f'L={L}, dim={DIM}, 最差残差 {cl_w:.2e}')

    # --- A2 引理: 反对易的两个对合 ⇒ 两特征子空间维数相等 ------------
    A, B = Gam[0], Gam[1]
    ev = np.linalg.eigvalsh((A + A.conj().T) / 2.0)
    n_p, n_m = int(np.sum(ev > 1e-9)), int(np.sum(ev < -1e-9))
    check('A2 引理(实测): n₊ = n₋ ⇒ dim 必为偶数',
          n_p == n_m and (n_p + n_m) == DIM,
          f'n₊={n_p}, n₋={n_m}, dim={DIM}={n_p + n_m}; '
          f'tr γ₀ = {float(np.trace(A).real):+.2e} (=0 是同一件事)')

    # --- A3 落到本项目: d_local=3 ⇒ dim 为奇数 ⇒ 不可能 --------------
    check(f'A3 d_local=3 ⇒ dim = 3^{L} = {3 ** L} 为**奇数** ⇒ 引理直接矛盾',
          (3 ** L) % 2 == 1,
          f'3^{L} = {3 ** L} (奇); 而 A2 要求 dim 为偶 ⇒ 该空间上不存在 γ₀, γ₁')

    # 模算术佐证(更强的形式): Clifford 代数 Cl_{2L} 的不可约表示维数是 2**L,
    # 任何表示都是它的直和 ⇒ 2^L | dim ⇒ 2^L | d_local^L ⇒ d_local 为偶数。
    # 注意这一条**断言**了教科书事实(Cl_{2L} 的不可约性), 与 A2 的实测分开记账。
    odd_bad = {d: (d ** L) % (2 ** L) for d in (3, 5, 7, 9)}
    even_ok = {d: (d ** L) % (2 ** L) for d in (2, 4, 6, 8)}
    check('A4 模算术佐证: 2^L ∤ d^L 对一切奇数 d',
          all(v != 0 for v in odd_bad.values()), f'{odd_bad}')
    check("A4' **活的反事实**: 同一条算术对一切偶数 d 都整除 ⇒ 无分辨力",
          all(v == 0 for v in even_ok.values()),
          f'{even_ok} ⇒ 排除的是**奇数**; 若不加 A5, A3 会被误读成"排除了 d≠2"')

    # --- A5 反事实: d_local=4 上**同样**能造出合法载体 ---------------
    I2 = np.eye(DIM)
    dim4 = 4 ** L
    G4 = [np.kron(g, I2) for g in Gam]           # 2L 个算符, 作用在 4^L 上
    w4 = 0.0
    for a in range(2 * L):
        w4 = max(w4, float(np.abs(G4[a] @ G4[a] - np.eye(dim4)).max()))
        for b in range(a + 1, 2 * L):
            w4 = max(w4, float(np.abs(cg.antic(G4[a], G4[b])).max()))
    check('A5 **活的反事实**: d_local=4 上 γ⊗I 仍是合法载体',
          w4 < 1e-12 and dim4 % (2 ** L) == 0,
          f'dim=4^{L}={dim4}, 最差残差 {w4:.2e} ⇒ 引理**不排除** d_local=4, '
          '故"取 2"靠最小性, 不靠这条定理')

    print("\n      A 的判定:")
    print("        · **已导出**: d_local 必为偶数 (3, 5, 7, … 数学上不可能)")
    print("        · **仍是选择**: 偶数中取 2 —— 靠「每格点恰好一对 Majorana」的最小性;")
    print("          A5 实测 d_local=4 不被排除。")
    print("        · 前提: L2 选用 Majorana 表述(自旋表述本身不需要它, 见 B)。")


def part_B():
    print("\n  B · 导出复数域 —— 导出点不在 L1, 而在下游的厄米性")
    print("      B1 先证「L1 处不承重」, B2 再定位「那个 i 从哪来」")

    # --- B1 L1 与 L2 的态都是实的 ----------------------------------
    s1 = G.stage1_void(2)
    r1 = G.derive_L1_void_to_chain(s1['equal_state'], L)
    vf = r1['void_full']
    check('B1a L1 链上态 void_full 是**实的**(相位不变判据 |Σψ²| = 1)',
          abs(realness(vf) - 1.0) < 1e-12,
          f'|Σψ²| = {realness(vf):.16f}, dim={vf.size}')

    _E0, gs = G.exact_ground_state(L, 1.0, 32.0)     # L1→L2 那条链的比对标态
    check('B1b L2 的 TFI 基态(h/J=32)也是**实的**(同一判据)',
          abs(realness(gs) - 1.0) < 1e-12,
          f'|Σψ²| = {realness(gs):.16f} ⇒ L1→L2 的全部重叠读数在**实** '
          'Hilbert 空间里即可复现')

    # 活的反事实: 给同一个基态加一个**相对**相位(不是整体相位), 判据必须翻红。
    ph = np.where(np.arange(DIM) % 2 == 0, 1.0 + 0j, np.exp(1j * 0.7))
    rc = realness(ph * gs)
    check('B1c **检验是活的**: 加相对相位后判据立刻 < 1',
          rc < 0.999,
          f'|Σ(ψ·e^(iφ_j))²| = {rc:.6f} < 1 ⇒ 该判据对"复"有分辨力, B1a/b 不是空判')

    # --- B2 那个 i 被厄米性强制 ------------------------------------
    Gam = [cg.majorana(a, L) for a in range(2 * L)]
    ah_w = 0.0
    for a in range(2 * L):
        for b in range(2 * L):
            if a != b:
                X = Gam[a] @ Gam[b]
                ah_w = max(ah_w, float(np.abs(X + X.conj().T).max()))
    check('B2a Majorana 双线性 γ_aγ_b (a≠b) 一律**反厄米**',
          ah_w < 1e-12,
          f'max‖γ_aγ_b + (γ_aγ_b)†‖ = {ah_w:.2e} (全部 {2 * L * (2 * L - 1)} 对)')

    Y01 = 1j * (Gam[0] @ Gam[1])
    check('B2b 乘上 i 之后才**厄米** ⇒ 那个 `i` 是厄米性逼出来的',
          float(np.abs(Y01 - Y01.conj().T).max()) < 1e-12,
          f'‖iγ₀γ₁ − (iγ₀γ₁)†‖ = {float(np.abs(Y01 - Y01.conj().T).max()):.2e}; '
          '而 γ₀γ₁ 反厄米 ⇒ 不乘 i 就不是哈密顿量')

    # 口径 A 的边界项: 代码里真实存在, 且它**正是**那个 i 的用处。
    hb = cg.build_h(L, 1.0, 1.0)
    check('B2c 口径 A 的边界项在代码里真实存在 (build_h 的 (2L−1, 0) 元 = +2J)',
          float(hb[2 * L - 1, 0]) == 2.0 and float(hb[0, 2 * L - 1]) == -2.0,
          f'h[{2 * L - 1},0] = {float(hb[2 * L - 1, 0]):+.1f}, '
          f'h[0,{2 * L - 1}] = {float(hb[0, 2 * L - 1]):+.1f} (J=h=1) '
          '—— 口径 A 的横幅不是空话')

    check('B2d build_h 返回的矩阵**是实的且反对称** ⇒ `i` 不在矩阵里',
          (not np.iscomplexobj(hb)) and float(np.abs(hb + hb.T).max()) == 0.0,
          f'dtype={hb.dtype}, ‖h + hᵀ‖ = 0 ⇒ i 住在双线性形式的前因子里')

    Htil = cg.reconstruct(hb, Gam)      # 代码自己的实现: 0.25j·Σ h_ab γ_aγ_b
    herm = float(np.abs(Htil - Htil.conj().T).max())
    check('B2e 用实矩阵 h 重建出的 H_til **厄米** (走代码自己的 reconstruct)',
          herm < 1e-12, f'‖H_til − H_til†‖ = {herm:.2e}')

    Hbad = Htil / 1j                    # 抽掉那个 i (= 0.25·Σ h_ab γ_aγ_b)
    anto = float(np.abs(Hbad + Hbad.conj().T).max())
    check('B2f **活的反事实**: 抽掉那个 i 立刻**反厄米**',
          anto < 1e-12, f'‖H_til/i + (H_til/i)†‖ = {anto:.2e} ⇒ 抽掉 i 就不是哈密顿量')

    # 第二处强制: 湮灭算符 c_j=(γ_2j+iγ_2j+1)/2 必须**不自伴**。
    # 口径注意: 实测 c_0 的**矩阵是实的**(|Im c_0| = 0 —— 因为 iγ_1 在 σy 约定下
    # 是实的), 所以准确说法是「c_j ≠ c_j†」, **不是**「c_j 的元素是复数」。
    cf = cg.real_fermions(Gam, L)
    ic = float(np.abs(np.asarray(cf[0]).imag).max())
    d_self = float(np.abs(cf[0] - cf[0].conj().T).max())
    c_noi = (Gam[0] + Gam[1]) / 2.0            # 抽掉 i 的反事实构造
    d_noi = float(np.abs(c_noi - c_noi.conj().T).max())
    check('B2g 湮灭算符要那个 i: c_j ≠ c_j† (缺了 i 就退化成自伴的 Majorana)',
          d_self > 0.0 and d_noi == 0.0,
          f'‖c_0 − c_0†‖ = {d_self:.3f} > 0 (而 |Im c_0|max = {ic:.1e} —— '
          f'矩阵本身是**实**的); 抽掉 i: ‖c′ − c′†‖ = {d_noi:.1e} = 0 '
          '⇒ c′ 自伴, 是 Majorana 而非湮灭算符')

    cc = float(np.abs(cg.antic(cf[0], cf[0].conj().T) - np.eye(DIM)).max())
    check('B2h 且该构造满足正则反对易 {c_i,c_i†}=I ⇒ 这个 i 用对了',
          cc < 1e-12, f'最差残差 {cc:.2e}')

    print("\n      B 的判定:")
    print("        · **L1 处不承重**: B1a/B1b 实测两个态都是实的, 判据有分辨力(B1c)。")
    print("        · **下游由厄米性强制**: B2a/B2b 是那个 i, B2c~f 把它定位到口径 A 的")
    print("          H_til 与代码实现, B2g/h 是第二处(湮灭算符不自伴, 缺 i 即退化)。")
    print("        · **边界**: 自旋表述(σzσz / σx, 实系数)根本不需要 i ⇒ 导出的是")
    print("          「**一旦选费米子表述, i 被强制**」, 不是「宇宙必须是复的」。")


def main():
    print("=== v16 · 预设层「真导出」尝试 (d_local=2 / 复数域) ===")
    print("    与 _v16_smoke_presets.py 的分工: 那一份**不导出**, 本文件**去导出**。")
    print("    口径: 不新增守卫/指标(不动 27/78 分母), 不改阈值, 不重跑全流程。")
    part_A()
    part_B()
    print("\n  === 两条预设的最终地位 (取代 §5.4 的「给定且承重 / 给定且不承重」) ===")
    print("    #1 d_local=2 : 奇数值**已被导出排除**; 偶数中取 2 **仍是选择**")
    print("    #2 复数域    : L1 处**不承重**; 下游由 H_til 的**厄米性强制** ——")
    print("                   但其前提(费米子表述)本身是选择 ⇒ 与 #1 **同一个根**")
    print("    ⇒ 三条'给定'收敛为 **一个**选择(表述选择) + 它的两个推论 + 一条最小性。")
    print("      缺口比 §5.5 记的小, 但**不是零**; 本文件不宣称缺口已修复。")

    print(f"\n=== 结果: "
          f"{'全部通过' if not _FAILS else '**失败 ' + str(len(_FAILS)) + ' 项**'}"
          f" ===")
    for f in _FAILS:
        print(f"    · {f}")
    return 0 if not _FAILS else 1


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.exit(main())
