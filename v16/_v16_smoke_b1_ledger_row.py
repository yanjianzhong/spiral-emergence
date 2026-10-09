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
v16 · B1 冒烟测试(二) —— 台账那一格的**运行时耦合** [_VERSION_TAG = 'v16-smoke-b1-2']

只验一件事, 但它值一次全流程: W6 台账 (`_NEG_CTRL_TABLE`) 的 L1 行现在**指名**
引用一条守卫 `'B1 推离: 转动 L1 局部态后重叠跌破 0.9'`。而
`metric_negative_control_ledger()` 有一条 phantom 检查:

    phantom = {g for r in _NEG_CTRL_TABLE for g in r[4] if g not in GUARDS 的名字}

**名字只要差一个字, 这条就在全流程末尾判失败** —— 而那时已经烧掉 1200 s。
所以现在就**真调** `metric_l1_void_link()` 把守卫注册进去, 再**真调**
`metric_negative_control_ledger()`, 看 L1 那格到底算不算数。

注意本脚本是**孤立**运行: 只有 B1 的三条守卫在册, 所以 ledger 的 phantom 名单
里会出现其它层的守卫名 (它们在本进程里没注册)。这是孤立运行的性质, 不是缺陷 ——
断言只针对 **B1 的名字**与 **L1 那一格**。

用法:
    python v16/_v16_smoke_b1_ledger_row.py
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'v16'))

import spiral_metric_v16 as M  # noqa: E402
import spiral_model_v16 as G  # noqa: E402

L = 16
_FAILS = []


def check(label, cond, detail=''):
    tag = 'PASS' if cond else '**FAIL**'
    print(f"    [{tag}] {label}" + (f"  —— {detail}" if detail else ''))
    if not cond:
        _FAILS.append(label)


def main():
    print("=== v16 · B1 冒烟测试(二): 台账 L1 格的运行时耦合 ===")

    s1 = G.stage1_void(G.KNOBS['d_local'])
    l1 = G.derive_L1_void_to_chain(s1['equal_state'], L, h_wide=(2, 8, 32))
    push = G.l1_void_pushaway(l1['equal_state'], L)

    print("\n  (1) 真调 metric_l1_void_link() 注册守卫/指标")
    n_m, n_g = len(M.METRICS), len(M.GUARDS)
    m = M.metric_l1_void_link(l1, push)
    new_g = M.GUARDS[n_g:]
    check('注册了 1 条指标', len(M.METRICS) - n_m == 1, m['name'])
    check('注册了 3 条守卫', len(new_g) == 3,
          " | ".join(g['name'] for g in new_g))
    check('三条都是**计分**守卫 (expect_pass=True)',
          all(g['expect_pass'] for g in new_g))
    check('三条都通过 (ok=True)', all(g['ok'] for g in new_g),
          str([g['name'] for g in new_g if not g['ok']]))

    print("\n  (2) 台账 L1 行指名的守卫**确实在 GUARDS 里** (phantom 检查的实质)")
    rows = [r for r in M._NEG_CTRL_TABLE if r[0] == 'L1']
    check('_NEG_CTRL_TABLE 恰有一行 L1', len(rows) == 1)
    row = rows[0]
    check('L1 行不再是"缺"', row[5] != '缺', f"状态 = {row[5]!r}")
    check('L1 行的类型是"推离"', row[3] == '推离', repr(row[3]))
    registered = {g['name'] for g in M.GUARDS}
    missing = [g for g in row[4] if g not in registered]
    check('L1 行引用的守卫名**逐字**存在 (否则全流程末尾判失败)',
          missing == [], f"引用={list(row[4])}, 缺失={missing}")

    print("\n  (3) 真调 metric_negative_control_ledger(), 看 L1 那格")
    led = M.metric_negative_control_ledger()
    # ledger 的返回值不是记录本身, 故从**真实注册表**里取它那条 metric 的
    # measured —— 不猜返回结构, 直接读它登记进去的东西。
    msr = [x for x in M.METRICS
           if x['name'] == '13 负对照台账 (逐层)'][-1]['measured']
    check('L1 已被计为"有对照" (n_controls == 1)',
          msr['nc_per_layer']['L1'] == 1, str(msr['nc_per_layer']))
    check('L1 已**不在**缺口名单里', 'L1' not in msr['nc_gap_layers'],
          f"缺口 = {msr['nc_gap_layers']}")
    check('L1 那格数到 1 条存活守卫', msr['nc_guards_by_layer']['L1'] == 1,
          str(msr['nc_guards_by_layer']))
    check('覆盖层数 (孤立运行下口径自洽: 覆盖 + 缺口 = 7)',
          msr['nc_n_layers_covered'] + len(msr['nc_gap_layers']) == 7,
          f"覆盖 {msr['nc_n_layers_covered']}/7, 缺口 {msr['nc_gap_layers']}")
    check('B1 的守卫名**没有**出现在 phantom 名单里',
          all(g not in msr['nc_phantom_guards'] for g in row[4]),
          f"phantom 共 {len(msr['nc_phantom_guards'])} 条 (孤立运行, "
          f"其它层的守卫未注册 —— 属正常)")
    check('L1 那格的类型是"推离"',
          msr['nc_kinds_by_layer']['L1'] == ['推离'],
          str(msr['nc_kinds_by_layer']['L1']))

    print("\n  (4) claim 列指向的指标名与真实注册名一致")
    from _v16_claim_ledger import _NEG_CTRL_CLAIM
    claim = _NEG_CTRL_CLAIM.get(('L1', row[2]))
    check('台账 L1 行在 _NEG_CTRL_CLAIM 里有登记', claim is not None,
          f"键 = ('L1', {row[2]!r})")
    check('claim 列指向的指标名 == 真实注册的指标名', claim == m['name'],
          f"{claim!r} vs {m['name']!r}")

    print(f"\n=== 结果: "
          f"{'全部通过' if not _FAILS else '**失败 ' + str(len(_FAILS)) + ' 项**'}"
          f" ===")
    for f in _FAILS:
        print(f"    · {f}")
    return 0 if not _FAILS else 1


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.exit(main())
