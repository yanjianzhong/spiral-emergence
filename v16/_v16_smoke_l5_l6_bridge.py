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
v16 · L5->L6 空桥尝试 (§5.4)  [_VERSION_TAG = 'v16-smoke-l5l6-1']

**先说清"打通"的判定, 免得事后把"没打通"说成"打通了一半"。**

一条"桥"要成立, 必须同时满足两条 —— 缺一条都不算:
  (i)  L6 的某个入参真的由 L5 的某个读数决定;
  (ii) 改动 L5 的读数, L6 的输出**必须跟着变**（可证伪的形式）。

本文件量的是 (ii)。(i) 已经被 `derive_L6_grid` 的 docstring 用**范畴论证**否掉了
—— 那是一条比任何实验都强的理由, 本文件不重复论证, 只把它转述清楚:

    不能派生的: Gray-Scott 的物理域长 L_domain 不能由 Ising 链的关联长度算出。
    两者是不同的物理 (一个是量子自旋关联, 一个是反应-扩散的扩散长度
    sqrt(Du/(F+k))), 把它们数值相等起来是量纲和物理的双重范畴错误。

**但代码里留了一处没被判定的"依据"**: `KNOBS['n_lambda']` 的注释写着
「旋钮: 域内应容纳的斑图波数 (来自 xi/L 的尺度不变性)」。本文件量这句话。

判定(三段, 全部可证伪):
  (a) 扫描 xi_over_L（含真实读数 1.2071）⇒ L6 的全部输出**逐位相同** ⇒ 无影响。
  (b) 反事实构造：一个**真的把 xi/L 接进 N_req** 的 L6 ⇒ 同一条扫描会翻红
      ⇒ 证明 (a) 那条检验有分辨力, 不是恒真。
  (c) 对照组：n_lambda / L_domain 的扰动**确实**改变输出 ⇒ 证明这个函数
      不是"整个是死的", 死掉的只有 xi/L 那一路。

**为什么这是"打不通"而不是"技术不足"（本文件最重要的结论）**:
  L5 产出的是**张量网络几何**（MERA 图的 Forman 均值 / 平均度）与**边界关联长度**;
  L6 需要的是**物理空间里的网格**。要让前者决定后者, 唯一的语义是把 MERA 图的
  几何**等同于物理空间** —— 而那正是 v16 audit §三 B2 硬边界明令禁止的
  「不宣称涌现时空」。**故"打不通"是本项目自身边界的必然结论, 不是没做到。**

用法:
    python v16/_v16_smoke_l5_l6_bridge.py
"""

import inspect
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'v16'))

import numpy as np  # noqa: E402
import spiral_model_v16 as G  # noqa: E402

# 冻结基线里的真实读数（v16/_baseline/stdout.log:159 —— "xi/L = 1.2071"）。
XI_OVER_L_REAL = 1.2071
_SWEEP = (0.01, 0.1, 0.5, XI_OVER_L_REAL, 2.0, 10.0, 100.0)
_FAILS = []


def check(label, cond, detail=''):
    tag = 'PASS' if cond else '**FAIL**'
    print(f"    [{tag}] {label}" + (f"  —— {detail}" if detail else ''))
    if not cond:
        _FAILS.append(label)


def _key(res):
    """L6 输出的可比指纹（逐位比较, 不用容差）。"""
    return (res['N'], res['N_req'], res['N_cap'], res['bound_by'],
            repr(res['h']), repr(res['stab']), repr(res['lambda_target']))


def _cf_grid(xi_over_L, knobs):
    """**反事实构造**: 一个真的把 xi/L 接进公式的 L6。

    只用于证明上面那条"逐位相同"的检验**有分辨力** —— 若 xi/L 真的进公式,
    同一族取值会给出**不同**的 N。它不参与任何物理判定。
    """
    return int(np.ceil(knobs['pts_per_wavelength'] * knobs['n_lambda']
                       * float(xi_over_L)))


def main():
    print("=== v16 · L5->L6 空桥尝试 (§5.4) ===")
    print("    判据: 改动 L5 的读数, L6 的输出必须跟着变 —— 否则桥不存在")

    print("\n  (a) 扫描 xi_over_L, 真调 derive_L6_grid")
    base = G.derive_L6_grid(XI_OVER_L_REAL, G.KNOBS)
    keys = {}
    for v in _SWEEP:
        keys[v] = _key(G.derive_L6_grid(v, G.KNOBS))
    uniq = set(keys.values())
    check('xi_over_L 扫描下 L6 输出逐位相同', len(uniq) == 1,
          f"{len(_SWEEP)} 个取值 -> {len(uniq)} 种输出")
    check('真实读数 xi/L=1.2071 也在其中 (不是只扫了玩具值)',
          XI_OVER_L_REAL in keys,
          f"xi/L={XI_OVER_L_REAL} -> N={keys[XI_OVER_L_REAL][0]}, "
          f"bound_by={keys[XI_OVER_L_REAL][3]}")
    check('**结论**: xi/L 对 L6 的算法**零影响**', len(uniq) == 1,
          "⇒ KNOBS 注释里那句「来自 xi/L 的尺度不变性」在本函数的算术里不存在")

    print("\n  (b) 检验是活的 —— 反事实构造下同一条扫描必须翻红")
    cf = {v: _cf_grid(v, G.KNOBS) for v in _SWEEP}
    check('反事实(真用了 xi/L)下, 同一族取值给出不同结果',
          len(set(cf.values())) > 1,
          f"反事实 N: {sorted(set(cf.values()))} ⇒ 该扫描有分辨力")

    print("\n  (c) 对照组 —— 这些旋钮**确实**改变输出 (函数不是整个死的)")
    k2 = dict(G.KNOBS)
    k2['n_lambda'] = G.KNOBS['n_lambda'] * 2.0
    r_nl = G.derive_L6_grid(XI_OVER_L_REAL, k2)
    check('n_lambda 加倍 ⇒ N_req 变 (它只有这一条通路)',
          r_nl['N_req'] != base['N_req'],
          f"N_req {base['N_req']} -> {r_nl['N_req']}")
    k3 = dict(G.KNOBS)
    k3['L_domain'] = G.KNOBS['L_domain'] * 2.0
    r_ld = G.derive_L6_grid(XI_OVER_L_REAL, k3)
    check('L_domain 加倍 ⇒ N_cap / h 变',
          r_ld['N_cap'] != base['N_cap'] or r_ld['h'] != base['h'],
          f"N_cap {base['N_cap']} -> {r_ld['N_cap']}, "
          f"h {base['h']:.6f} -> {r_ld['h']:.6f}")

    print("\n  (d) 结构性检验 —— L5 的几何在签名上就进不了 L6")
    _p = list(inspect.signature(G.derive_L6_grid).parameters)
    check('derive_L6_grid 的签名里没有图/几何/Forman',
          not any(s in ' '.join(_p).lower()
                  for s in ('graph', 'forman', 'd_bar', 'degree')),
          f"参数 = {_p}")
    check('L5 的派生量确实是几何 (Forman), 与 L6 的入参不同域',
          callable(G.derive_L5_forman_from_degree),
          'derive_L5_forman_from_degree 产出 Forman 均值; '
          'derive_L6_grid 只收标量旋钮')

    print("\n  === 结论: L5->L6 **打不通**, 且不是技术不足 ===")
    print("    · 直接桥: 由范畴论证关闭 (量纲 + 物理双重错误), 理由已在")
    print("      derive_L6_grid 的 docstring 里 —— 这条论证比任何实测都强。")
    print("    · 间接'依据': KNOBS['n_lambda'] 的注释声称依据 xi/L 的尺度不变性,")
    print("      实测零影响 ⇒ **注释与事实不符**, 属 B3 该抓的 claim↔代码不符。")
    print("    · 唯一的'打通'语义是把 MERA 图几何当作物理空间 —— 那正是 B2 硬边界")
    print("      禁止的「不宣称涌现时空」⇒ 打不通是**本项目边界的必然结论**。")
    print("    · 按用户的停止规则: **停在这里**, 不改判据、不硬接。")

    print(f"\n=== 结果: "
          f"{'全部通过' if not _FAILS else '**失败 ' + str(len(_FAILS)) + ' 项**'}"
          f" ===")
    for f in _FAILS:
        print(f"    · {f}")
    return 0 if not _FAILS else 1


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.exit(main())
