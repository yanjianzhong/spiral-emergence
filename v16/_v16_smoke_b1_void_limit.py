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
v16 · B1 冒烟测试 —— L1 下游连线 [_VERSION_TAG = 'v16-smoke-b1-1']

**真调** `derive_L1_void_to_chain()` 与 `stage2_critical_break()` (硬纪律 2:
不许把逻辑抄一遍再断言)。四项:

  (1) 判据: overlap_curve 对 h/J 单调上升, 且 h/J=32 时 > 0.99。
  (2) 可证伪: 喂一个**维数不对**的局部态, dim_match 必须翻成 False
      —— 证明它不是写死的 True。
  (3) **纯增量**: 传 / 不传 void_state 时阶段二的物理读数必须**逐位相同**。
      这是 B1-2 的核心声明, 也是最可能悄悄破掉 v15 冻结读数的地方。
  (4) 断言是活的: 维数不符的 void_state 必须真的抛 AssertionError。

另附一段"W11 源码扫描判据的复现" —— 如实标注它**不是**守卫本身,
真值以 v16 全流程日志里 W11 那一行为准。

用法:
    python v16/_v16_smoke_b1_void_limit.py
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'v16'))

import numpy as np  # noqa: E402
import spiral_model_v16 as G  # noqa: E402

L = 16
_FAILS = []


def check(label, cond, detail=''):
    tag = 'PASS' if cond else '**FAIL**'
    print(f"    [{tag}] {label}" + (f"  —— {detail}" if detail else ''))
    if not cond:
        _FAILS.append(label)


def main():
    print("=== v16 · B1 冒烟测试 (L1 下游连线) ===")

    s1 = G.stage1_void(G.KNOBS['d_local'])
    print("\n  (1)(2) 真调 derive_L1_void_to_chain(equal_state, L=16)")
    r = G.derive_L1_void_to_chain(s1['equal_state'], L)

    check('dim_full == 2**L', r['dim_full'] == 2 ** L,
          f"{r['dim_full']} vs {2 ** L}")
    check('dim_match 为真', r['dim_match'] is True)
    check('overlap_curve 有 5 个点', len(r['overlap_curve']) == 5,
          str([f"{a:g}" for a, _ in r['overlap_curve']]))
    check('单调上升 (判据 1/2)', r['monotone'] is True,
          "  ".join(f"{a:g}:{b:.6f}" for a, b in r['overlap_curve']))
    check('overlap_inf > 0.99 (判据 2/2)', r['overlap_inf'] > 0.99,
          f"{r['overlap_inf']:.8f}")
    check('**有限 h 的诚实标注**: overlap_inf 严格小于 1',
          r['overlap_inf'] < 1.0, f"1 - overlap_inf = {1 - r['overlap_inf']:.3e}")

    print("\n  (2) 可证伪 —— 维数不对时 dim_match 必须失败")
    bad = np.ones(3, dtype=complex) / np.sqrt(3)
    rb = G.derive_L1_void_to_chain(bad, 4, h_wide=(8,))
    check('d_local=3, L=4: dim_match 翻成 False',
          rb['dim_match'] is False,
          f"dim_full={rb['dim_full']}, 张成={3 ** 4}")
    # 首版这里直接崩在 vdot 的形状错上 —— 函数没报告不符而是硬算。实测逼出的
    # 缺陷, 已修 (维数不符则跳过重叠)。这条断言把它钉住。
    check('维数不符时**跳过**重叠 (不崩) 且 overlap_inf 如实为 nan',
          rb['overlap_curve'] == [] and np.isnan(rb['overlap_inf'])
          and rb['monotone'] is False,
          f"curve={rb['overlap_curve']} inf={rb['overlap_inf']}")

    print("\n  (3) **纯增量** —— 传/不传 void_state, 阶段二读数逐位相同")
    # 先量**基准噪声**: eigsh 的迭代起点不由本仓库控制, 连调两次本身就有抖动。
    # 不先量它, 就会把这层噪声误记成"接线造成的差异"(首版正是这么误判的)。
    _e1, _ = G.exact_ground_state(L, 1.0, 1.0)
    _e2, _ = G.exact_ground_state(L, 1.0, 1.0)
    _noise = abs(_e1 - _e2)
    print(f"      · 基准噪声: 同一个 exact_ground_state 连调两次, "
          f"|dE0| = {_noise:.3e}, 相对 {_noise / abs(_e1):.1e}")

    # 把 exact_ground_state 记忆化, 让 a/b 两次调用**拿到同一个基态** ——
    # 这样两边唯一的差别就只剩 void_state 这一条接线, 抖动被彻底隔离。
    _cache, _orig = {}, G.exact_ground_state

    def _memo(L_, J_, h_):
        k = (L_, J_, h_)
        if k not in _cache:
            _cache[k] = _orig(L_, J_, h_)
        return _cache[k]

    G.exact_ground_state = _memo
    try:
        a = G.stage2_critical_break(L=L)
        b = G.stage2_critical_break(L=L, void_state=r['void_full'])
    finally:
        G.exact_ground_state = _orig

    for k in ('E0', 'entropy', 'polarization', 'gap'):
        check(f'{k} 逐位相同', a[k] == b[k], f"{a[k]!r} vs {b[k]!r}")
    check('gs 逐位相同', bool(np.array_equal(a['gs'], b['gs'])))
    check('probabilities 逐位相同',
          bool(np.array_equal(a['probabilities'], b['probabilities'])))
    check('返回字典的键集未变 (无新增字段)',
          set(a) == set(b), f"{sorted(set(b) - set(a))}")

    print("\n  (4) 断言是活的 —— 维数不符必须抛 AssertionError")
    try:
        G.stage2_critical_break(L=L, void_state=np.ones(4, dtype=complex))
        check('维数不符抛出 AssertionError', False, '没有抛')
    except AssertionError as e:
        check('维数不符抛出 AssertionError', True, str(e))

    print("\n  (5) 推离负对照 —— 真调 l1_void_pushaway")
    pu = G.l1_void_pushaway(s1['equal_state'], L)
    check('有 theta 使重叠跌破 0.9 (theta_star 非 None)',
          pu['theta_star'] is not None,
          f"theta*={pu['theta_star']}, overlap={pu['overlap_at_star']}")
    check('theta* 处重叠确实 < 0.9', pu['overlap_at_star'] < 0.9,
          f"{pu['overlap_at_star']:.6f}")
    check('推离越多重叠越低 (曲线随 theta 单调下降)',
          all(b < a for (_, a), (_, b) in zip(pu['curve'], pu['curve'][1:])),
          "  ".join(f"{t:g}:{o:.6f}" for t, o in pu['curve']))
    # 交叉校验: theta=0 时推离函数必须退回 B1-1 的同一个物理量。若两条路径
    # 各算各的, 这里就会对不上 —— 这是"两个函数真的在量同一个东西"的证据。
    pu0 = G.l1_void_pushaway(s1['equal_state'], L, thetas=(0.0,))
    check('theta=0 与 derive_L1_void_to_chain 的 overlap_inf 一致',
          abs(pu0['curve'][0][1] - r['overlap_inf']) < 1e-12,
          f"{pu0['curve'][0][1]:.10f} vs {r['overlap_inf']:.10f}")
    check('**口径限制**如实标注: 推离的 overlap 由被转动的态直接算出',
          True, '标定灵敏度尺度, 不是独立的物理证伪 (见守卫 note)')

    print("\n  (附) W11 源码扫描判据的**复现** (非守卫本身)")
    # 拆分后 model 的实体在 v16/_model/ 子包里, 门面只剩 re-export —— 只扫门面
    # 会扫到 0 处, 这条会假失败。所以门面与子包一起扫, 扫描面与拆分前等值
    # (与 spiral_metric_v16 里 W11 守卫自身的改法一致)。
    _src = [os.path.join(_ROOT, 'v16', 'spiral_model_v16.py')]
    _mdl = os.path.join(_ROOT, 'v16', '_model')
    if os.path.isdir(_mdl):
        _src += [os.path.join(_mdl, f) for f in sorted(os.listdir(_mdl))
                 if f.endswith('.py')]
    lines = []
    for _p in _src:
        with open(_p, encoding='utf-8') as f:
            lines += f.read().splitlines()
    eq_lines = [l for l in lines if 'equal_state' in l]
    consumed = any(("['equal_state']" in l) or ('.equal_state' in l)
                   for l in eq_lines)
    check("复现: 扫描到 equal_state 且被下游取用", consumed,
          f"{len(eq_lines)} 处出现, consumed={consumed}")

    print(f"\n=== 结果: "
          f"{'全部通过' if not _FAILS else '**失败 ' + str(len(_FAILS)) + ' 项**'}"
          f" ===")
    for f in _FAILS:
        print(f"    · {f}")
    return 0 if not _FAILS else 1


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.exit(main())
