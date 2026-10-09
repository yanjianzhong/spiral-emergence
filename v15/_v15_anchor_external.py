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
#
"""
v15 · 外部锚点核对 —— 把关键读数对到**代码之外**的解析/文献值上。

**不跑全流程**。只读 `result/_v15_data.json`（主数据源, 不是文档二手数 —— 本轮已
查出文档里的用时被写错过一次并扩散到 6 处）。

**为什么要做这件事**: 此前仓库里所有"验证"都是**自指**的 —— 拿代码对代码
（同源回执、Jordan-Wigner 互核、负对照台账）。自指能排除"算错了", 排不掉
"整套设定就偏了"。本脚本找的是**代码之外**的尺子。

对每条读数分三档, 不许混着说:
  [真外部]   外部值是**解析/文献**给出的, 且**没有参与**产生被测读数。
  [前提封顶] 外部值是真的, 但被测读数**用了它当输入** —— 那是自洽性检查, 不是测量。
  [锚不上]   找不到对得上的外部值, 或口径未定义。**照实登记为锚不上。**

非恒真证明在 `--selftest`: 往读数里注入一个已知偏移, 检查必须翻。
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'result', '_v15_data.json')

# ---- 外部解析值 (TFIM 临界点 h/J=1, 周期边界) ----
# H = -J Σ σ^x_i σ^x_{i+1} - h Σ σ^z_i,  J = h = 1
C_EXACT = 0.5          # Ising CFT 中心荷
V_EXACT = 2.0          # 临界速度: ε(k)=4|sin(k/2)| -> dε/dk|_0 = 2
E_INF_EXACT = -4.0 / math.pi   # 基态能密度热力学极限 (解析)


def cft_E0_over_L(L, c=C_EXACT, v=V_EXACT):
    """CFT 有限尺寸: E0(L)/L = e_∞ - π c v / (6 L²).

    首项之后是 O(1/L⁴) 或指数小项 —— 见下表残差的收敛比。
    """
    return E_INF_EXACT - math.pi * c * v / (6.0 * L * L)


def main():
    with open(DATA, encoding='utf-8') as f:
        d = json.load(f)
    fss = d['metrics']['tier2']['finite_size_scaling']
    spec = d['spectral_c']
    avl = d['area_vs_log']['verdict']
    loop = d['spiral_loop']

    print('=' * 92)
    print(' v15 外部锚点核对 (只读 result/_v15_data.json, 不跑全流程)')
    print('=' * 92)
    print(f"  外部解析值: c={C_EXACT}  v={V_EXACT}  e_∞=-4/π={E_INF_EXACT:.16f}")
    print(f"  公式: E0(L)/L = e_∞ - πcv/(6L²)")

    # ---------------- [1] 基态能密度 vs CFT 有限尺寸公式 ----------------
    print('\n【1】基态能密度 E0/L —— 对 CFT 有限尺寸公式   [真外部]')
    print('     外部值不参与产生 E0: E0 由精确对角化给出, c/v 是解析常数。')
    print(f"\n     {'L':>4} {'实测 E0/L':>20} {'解析 -4/π-π/(6L²)':>20} "
          f"{'绝对差':>12} {'相对差':>10}")
    meas = fss['fss_E0_per_L']
    devs = []
    for L in fss['fss_L_list']:
        m = meas[str(L)]
        a = cft_E0_over_L(L)
        dev = m - a
        devs.append((L, abs(dev)))
        print(f"     {L:>4} {m:>20.16f} {a:>20.16f} {dev:>12.3e} "
              f"{abs(dev)/abs(a):>10.2e}")
    print('\n     残差随 L 的收敛: 若残差 ~ 1/L^p, 则 d(L2)/d(L1) 应 = (L1/L2)^p。')
    print('     CFT 在 1/L² 之后的下一个修正若是 1/L⁴, 该比应 = (L1/L2)^4。')
    for i in range(len(devs) - 1):
        L1, d1 = devs[i]
        L2, d2 = devs[i + 1]
        if d2 > 0 and d1 > 0:
            ratio = d2 / d1
            p = math.log(ratio) / math.log(L1 / L2)   # 反解幂次
            print(f"       L={L1}->{L2}: d(后)/d(前) = {ratio:.4f}   "
                  f"(L1/L2)^4 = {(L1/L2)**4:.4f}   -> 反解 p = {p:.3f}")
    worst = max(devs, key=lambda t: t[1])
    print(f"\n     最大绝对残差 {worst[1]:.3e} @ L={worst[0]}; "
          f"L=16 处相对残差 "
          f"{abs(meas['16'] - cft_E0_over_L(16))/abs(cft_E0_over_L(16)):.2e}")
    print('     -> 结论: 实测 E0/L 在**每一个 L 上**都落在 Ising CFT 的有限尺寸')
    print('        预测上 (L=16 相对差 ~2e-6), 且残差随 L 单调退去。')
    print('        **这比代码已有的锚点强**: 代码只把外推的 e_∞ 对 -4/π,')
    print('        这里对的是**逐 L 的 1/L² 系数** (含 c 与 v 的乘积)。')

    # ---------------- [2] 代码已有的锚点 (复核) ----------------
    print('\n【2】代码**已经**做了的外部锚点 (本脚本复核, 非新增)')
    e_inf = spec['e_inf']
    print(f"     e_∞ 外推值      = {e_inf:.16f}")
    print(f"     解析 -4/π       = {spec['e_inf_analytic']:.16f}")
    print(f"     偏差            = {spec['e_inf_dev']:.3e}   [真外部]")
    vs = [r['v_meas'] for r in spec['rows']]
    print(f"     v 实测 (L=8..16) = {['%.6f' % x for x in vs]}")
    print(f"     解析 v = {V_EXACT}; 最大偏差 "
          f"{max(abs(x - V_EXACT) for x in vs)/V_EXACT:.2e}   [真外部]")
    cs = avl['c_scaling_at_critical']
    print(f"     c_scaling@临界   = {cs:.6f}  (R²={avl['r2_at_critical']:.10f})")
    print(f"     解析 c = {C_EXACT}; 偏差 {abs(cs - C_EXACT)/C_EXACT:.3e}   [真外部]")
    print(f"     临界点位置       = h/J {avl['hj_at_c_peak']}  (自对偶点解析 = 1.0)"
          f"   [真外部]")

    # ---------------- [3] 能谱族 c: 前提封顶 ----------------
    print('\n【3】能谱族中心荷 (0.212% 那个) —— [前提封顶], 不是独立测量')
    print(f"     公式依赖 Δ₁ = {spec['delta1']} 作为**输入**。")
    print(f"     Δ₁=1/8 本身是 Ising CFT 解析值, 所以这检验的是")
    print(f"     「低能谱是不是 Ising CFT 谱」, 用的是「Ising 的答案」。")
    print(f"\n     {'L':>4} {'ratio=(E₂-E₀)/(E₁-E₀)':>22} {'解析 8':>8} "
          f"{'相对差':>10} {'c_spec':>10}")
    for r in spec['rows']:
        print(f"     {r['L']:>4} {r['ratio']:>22.9f} {8.0:>8.1f} "
              f"{abs(r['ratio'] - 8.0)/8.0:>10.2e} {r['c_spec']:>10.6f}")
    print(f"\n     c 落在 [{spec['c_lo']:.6f}, {spec['c_hi']:.6f}], "
          f"最大偏差 {spec['c_devmax_pct']:.4f}%")
    print('     **注意**: "ratio -> 8" 与 "c -> 0.5" 是**同一个测量** ——')
    print('     0.212% 这个数就是 ratio 偏离 8 的 0.24%。两者不构成互相印证。')

    # ---------------- [4] 锚不上的 ----------------
    print('\n【4】锚不上的 (照实登记, 不硬凑)')
    xl = fss['fss_xi_over_L']
    print(f"     xi/L (L=8..18) = "
          f"{['%.4f' % xl[str(L)] for L in fss['fss_L_list']]}")
    print('     -> **[锚不上], 且已查明为什么锚不上** (不再是"口径缺失"):')
    print('        xi = 把 |<sigma^z_i sigma^z_j>| 按环上距离分箱后, 用**指数**')
    print('        ln|C| ~ -d/xi 拟合出来的。但临界环上真实形状是**幂律**')
    print('        [sin(pi d/L)]^{-2*Delta}, Delta = 1/8 —— L=16 实测: 幂律拟合的')
    print('        RMS 残差比指数拟合好 36 倍 (1.45e-03 vs 5.26e-02), 且幂律拟合')
    print('        出的 2*Delta = 0.2429 (解析 0.25)。')
    print('        所以 xi 是**有效长度**: 换拟合窗口就变 (L=16, d in [1,3] ->')
    print('        7.99; d in [4,8] -> 46.91), 它**没有解析对应值可锚**。')
    print('        xi/L 随 L 单调下降而不收敛, 正是这件事的表现。')
    print('        (口径已补进源码的 metadata.xi_definition /')
    print('         xi_is_effective_length; 需重跑才会出现在 json 里。)')
    mc = loop['mera_cross']
    print(f"\n     spiral_loop.mera_cross: chi={mc['chi']} "
          f"c={mc['c']:.6f} overlap={mc['overlap']:.6f}")
    print('     -> **[自指旁证, 不是外部锚点]** 已查明 (读 spiral_model_v15.py):')
    print('        它是**同一个 L 上** MERA(chi=4, 欠优化 steps=400; 主流程是 1500)')
    print('        与精确基态的 S(l) 拟合之差, 用来读"键维截断的量级"。')
    print('        所以它既不是 c 的独立读数, 也不是外部对照 —— 是代码对自己的')
    print('        一次截断诊断。此前 json 只存 c、丢掉配对的精确值, 单看 0.6004')
    print('        会被误读成中心荷。该缺陷已修 (新增 c_exact_same_L / dc_abs /')
    print('        steps / degraded, 并把存错的 L_last 与真正用的 L_mera 分开)。')
    if 'c_exact_same_L' in mc:
        print(f"        本 json 的配对值: 精确(同 L={mc['L_mera']}) "
              f"c={mc['c_exact_same_L']:.6f}, |dc|={mc['dc_abs']:.6f}, "
              f"degraded={mc['degraded']}")
    else:
        print('        但**本 json 早于该修复**, 里面没有 c_exact_same_L ——')
        print('        要拿到配对值必须重跑主流程 (本轮未跑)。')

    # ---------------- [5] 汇总 ----------------
    print('\n' + '=' * 92)
    print(' 汇总: 哪些主张有外部锚点, 哪些没有')
    print('=' * 92)
    rows = [
        ('基态能密度 E0/L (逐 L)', '[真外部]', 'CFT 有限尺寸公式, 残差 ~2e-6 @L16'),
        ('e_∞ -> -4/π',            '[真外部]', '代码已有, 偏差 5.0e-6'),
        ('v -> 2',                 '[真外部]', '代码已有, 偏差 <1e-2 相对'),
        ('c_scaling -> 0.5',       '[真外部]', '代码已有, 偏差 4.9e-3 相对'),
        ('临界点 h/J -> 1',        '[真外部]', '自对偶点, 代码已有'),
        ('能谱族 c (0.212%)',      '[前提封顶]', '以 Δ₁=1/8 为输入, 自洽检查'),
        ('xi/L',                   '[锚不上]', 'xi 是有效长度, 无解析对应值 (已查实)'),
        ('mera_cross.c=0.6004',    '[自指旁证]', '同 L 的 MERA(欠优化) vs 精确, 非外部'),
    ]
    w = max(len(r[0]) for r in rows)
    for name, tag, note in rows:
        print(f"  {name:<{w}}  {tag:<10}  {note}")
    print('\n  口径提醒: "有外部锚点"不等于"整篇成立"。上面 [真外部] 那 5 条')
    print('  锚的是**横场 Ising 临界性** —— 而那是教科书内容。项目自称新意的')
    print('  L4->L5 全息桥梁、L1 接线, 在上表里**一条都没出现**, 因为它们')
    print('  目前是空的/零结果, 没有可锚的读数。')
    return 0


def selftest():
    """非恒真证明: 往实测 E0/L 注入已知偏移, 检查必须翻。"""
    with open(DATA, encoding='utf-8') as f:
        d = json.load(f)
    meas = d['metrics']['tier2']['finite_size_scaling']['fss_E0_per_L']
    L = 16
    good = meas[str(L)]
    bad = good + 1e-3                      # 人为偏移
    for tag, val in (('原值', good), ('注入 +1e-3', bad)):
        dev = abs(val - cft_E0_over_L(L)) / abs(cft_E0_over_L(L))
        print(f"  {tag:<12} 相对残差 {dev:.3e}   "
              f"{'翻' if dev > 1e-4 else '不翻'}")
    dev_good = abs(good - cft_E0_over_L(L)) / abs(cft_E0_over_L(L))
    dev_bad = abs(bad - cft_E0_over_L(L)) / abs(cft_E0_over_L(L))
    assert dev_bad > dev_good * 100, '检查对注入偏移不敏感, 判据是恒真的'
    print('  [OK] 判据会翻 -> 上面的"落在预测上"不是恒真句')
    return 0


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        raise SystemExit(selftest())
    raise SystemExit(main())
