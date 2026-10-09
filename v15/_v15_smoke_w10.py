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
v15·W10 冒烟 —— 只读, 不写文件, 不跑全流程。

验两件事:
  (1) 模型侧 stage-5 新块: 三次 mera_init 后**全局 torch RNG 状态被存还**,
      边集比较给出预期结果。
  (2) 指标侧 F6d 登记: 用**真实** MERA 图 + 合成对照族调
      metric_geometry_robustness, 确认 F6d 入账且 expect_pass=False。

对照族的数值是合成的 (只为让函数走到 F6d), 所以 F6a/F6b/F6c 的读数
在这一跑里**没有物理含义**, 只看它们不报错、F6d 正确入账。
"""
import numpy as np
import torch

import spiral_model_v15 as G
import spiral_metric_v15 as M


def main():
    print('=' * 84)
    print(' W10 冒烟')
    print('=' * 84)

    # ---------------- (1) 模型侧新块 ----------------
    print('\n【1】模型侧 stage-5 新块 (逐字逻辑 + RNG 存还)')
    torch.manual_seed(12345)
    _before = torch.get_rng_state().clone()
    _probe_draw = torch.randn(4)

    torch.manual_seed(12345)
    _rng_bak = torch.get_rng_state()
    _bulk_e = {_s: frozenset(tuple(sorted(e)) for e in
                             G.build_mera_graph(
                                 G.mera_init(32, 2, seed=_s)[0]).edges())
               for _s in (0, 1, 7)}
    torch.set_rng_state(_rng_bak)
    _bulk_same = len(set(_bulk_e.values())) == 1
    _after = torch.randn(4)

    print(f"  边集完全相同 = {_bulk_same}   |E| = {len(_bulk_e[0])}")
    print(f"  RNG 状态已存还: {bool(torch.equal(_before, _rng_bak))}")
    print(f"  存还后抽样与未受扰动的抽样一致: "
          f"{bool(torch.equal(_probe_draw, _after))}")
    assert _bulk_same, '边集应完全相同'
    assert torch.equal(_probe_draw, _after), 'RNG 状态没存还干净'
    # 反证: 三次构造**确实推进过** RNG 状态 (否则存还是空操作, 上面测不到东西)。
    # 注意不能靠 manual_seed 重放来证 —— 那只会给出相同抽样, 什么也证不了。
    torch.manual_seed(12345)
    _s0 = torch.get_rng_state()
    for _s in (0, 1, 7):
        G.mera_init(32, 2, seed=_s)
    _advanced = not bool(torch.equal(torch.get_rng_state(), _s0))
    print(f"  [反证] 不做存还时 RNG 状态会被推进: {_advanced}")
    assert _advanced, '构造没推进 RNG, 则存还测试是空操作'

    # ---------------- (2) 指标侧 F6d ----------------
    print('\n【2】指标侧 metric_geometry_robustness 的 F6d 登记')
    _m32, _ = G.mera_init(32, 2, seed=0)
    _Gm = G.build_mera_graph(_m32)
    rep_mera, _, _ = G.curvature_report(_Gm, 'smoke')

    rng = np.random.default_rng(0)
    fams = ('gnm', 'ws', 'tree')
    ctrls = [{'family': f,
              'forman_mean': float(rep_mera['forman_mean'] + rng.normal(0, 0.2)),
              'or_mean': float(rep_mera['or_mean'] + rng.normal(0, 0.05)),
              'n_triangles': int(rng.integers(0, 30)),
              'd_bar': float(rng.uniform(2, 4))}
             for f in fams for _ in range(3)]

    # v15·B1: chi 那一问是后加的键。这里**故意**喂进去, 让冒烟真的走到
    # `_dep_chi` 那条分支 —— 否则新代码在冒烟里是死路径, 等于没冒烟。
    s5 = {'mera_bulk': rep_mera, 'controls': ctrls,
          'bulk_seed_invariance': {'same': bool(_bulk_same),
                                   'n_edges': int(len(_bulk_e[0])),
                                   'seeds': [0, 1, 7],
                                   'chis': [2, 4, 8], 'chi_same': True}}

    n0 = len(M.GUARDS)
    M.metric_geometry_robustness(s5)
    new = M.GUARDS[n0:]
    print(f"\n  新增守卫 {len(new)} 条:")
    for g in new:
        ep = g.get('expect_pass', True)
        print(f"    {g['name']:<44} real_ok={g['ok']!s:<5} "
              f"expect_pass={ep!s:<5} 计分={g['ok'] if ep else '— (诊断)'}")

    f6d = [g for g in new if g['name'].startswith('F6d')]
    assert f6d, 'F6d 没入账'
    d = f6d[0]
    assert d.get('expect_pass') is False, 'F6d 必须是诊断项 (不入分母)'
    assert d['ok'] is False, 'F6d 的判据是"依赖态", 实测应为 False'
    assert '0, 1, 7' in d['condition'] and '91' in d['condition'], \
        f"F6d 的 condition 没带实测值: {d['condition']}"
    # v15·B1: 键维那一支必须**真被走到** —— 'None' 说明字典没喂进去, 该冒烟
    # 对新代码是死路径, 等于没验。
    assert '2, 4, 8' in d['condition'] and 'None' not in d['condition'], \
        f"F6d 的 condition 没覆盖键维分支 (chi 分支是死路径?): {d['condition']}"
    print('\n  [OK] F6d 入账, expect_pass=False, condition 带实测值 (非硬编码)')

    # ---------------- (3) 分母没变 ----------------
    print('\n【3】通过率分母核对')
    scored = [g for g in M.GUARDS if g.get('expect_pass', True)]
    print(f"  计分守卫 {len(scored)} 条 / 总守卫 {len(M.GUARDS)} 条 "
          f"(F6d 是诊断, 不计入)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
