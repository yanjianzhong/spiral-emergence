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
v15·W12 冒烟 —— 不跑全流程。

验三件事:
  (1) **全局名解析检查**: 我被改过的三个函数里, 每一个 `is_global` 的名字都必须
      真的存在于模块命名空间或 builtins 里。这**专门针对上一轮那个 bug**
      (`metric_consistency_checks` 里用了它没有的 `_HERE`, 全流程崩在 NameError,
      而冒烟因为是**复制的逻辑**而没照出来)。
      附**非恒真证明**: 往临时副本注入一个未定义的 `_HERE`, 检查器必须报出它。
  (2) 真调 `v13_tier1_runs` (小预算) -> 返回的 `chi_down` 必须存在且自洽。
      **不复制逻辑, 调真函数**。
  (3) `chi_sweep` 里必须有 `chi_dev` 这个键 (新守卫要按它取界上的 eps_c)。

对照族的数值是合成的, 只为让函数走到分支, 无物理含义。
"""
import ast
import builtins
import os
import symtable
import tempfile

import spiral_model_v15 as G

HERE = os.path.dirname(os.path.abspath(__file__))
TARGETS = ('v13_tier1_runs', 'collect_metrics_v15', 'metric_chi_bottleneck')


def unresolved_globals(path, func_names):
    """返回 {函数名: [在该函数里当全局用、但模块里并不存在的名字]}。

    做法: 用 symtable 取函数的符号表, 挑出 `is_global()` 的名字, 再对照模块级
    被绑定的名字与 builtins。
    symtable 单独用不够 —— 它**不检查存在性**, `_HERE` 同样被标成 global。
    """
    with open(path, encoding='utf-8') as f:
        src = f.read()
    top = symtable.symtable(src, path, 'exec')
    bound = {s.get_name() for s in top.get_symbols()
             if s.is_assigned() or s.is_imported() or s.is_namespace()}
    child = {c.get_name(): c for c in top.get_children()}
    out = {}
    for fn in func_names:
        tab = child[fn]          # 注意: top.lookup(fn) 只给 Symbol, 不是表
        miss = sorted({s.get_name() for s in tab.get_symbols()
                       if s.is_global()
                       and s.get_name() not in bound
                       and not hasattr(builtins, s.get_name())})
        out[fn] = miss
    return out


def main():
    print('=' * 84)
    print(' W12 冒烟')
    print('=' * 84)

    # ---------------- (1) 全局名解析 ----------------
    print('\n【1】全局名解析检查 (专治上一轮的 NameError)')
    for mod, names in (('spiral_model_v15.py', ('v13_tier1_runs',)),
                       ('spiral_metric_v15.py', ('collect_metrics_v15',
                                                 'metric_chi_bottleneck'))):
        p = os.path.join(HERE, mod)
        res = unresolved_globals(p, names)
        for fn, miss in res.items():
            print(f"  {mod}::{fn:<26} 未解析的全局名 = {miss or '无'}")
            assert not miss, f'{fn} 用了模块里不存在的名字: {miss}'

    # 非恒真证明: 注入一个未定义的全局名, 检查器必须报出来
    tmpd = tempfile.mkdtemp(prefix='_v15_w12_')
    try:
        dst = os.path.join(tmpd, 'spiral_metric_v15.py')
        with open(os.path.join(HERE, 'spiral_metric_v15.py'),
                  encoding='utf-8') as f:
            src = f.read()
        # 在 collect_metrics_v15 体首插一行, 引用一个模块里没有的名字
        tree = ast.parse(src)
        tgt = next(n for n in tree.body
                   if isinstance(n, ast.FunctionDef)
                   and n.name == 'collect_metrics_v15')
        tgt.body.insert(0, ast.Expr(value=ast.Name(id='_SMOKE_UNDEFINED',
                                                   ctx=ast.Load())))
        ast.fix_missing_locations(tree)
        with open(dst, 'w', encoding='utf-8') as f:
            f.write(ast.unparse(tree))
        bad = unresolved_globals(dst, ('collect_metrics_v15',))
        print(f"  [非恒真] 注入 `_SMOKE_UNDEFINED` 后报出 = "
              f"{bad['collect_metrics_v15']}")
        assert '_SMOKE_UNDEFINED' in bad['collect_metrics_v15'], \
            '检查器抓不到未定义全局名, 判据是恒真的'
        print('  [OK] 检查器会翻 -> 上面的"无"是真的无')
    finally:
        import shutil
        shutil.rmtree(tmpd, ignore_errors=True)

    # ---------------- (2) 真调 v13_tier1_runs (小预算) ----------------
    print('\n【2】真调 v13_tier1_runs —— 不复制逻辑, 调真函数')
    L = 8
    s2 = G.stage2_critical_break(L=L)
    c_exact = float(G.fit_central_charge(s2['gs'], L)['c'])
    steps_bak, lr_bak = G.V13_STEPS, G.V13_LR_SEEDS
    G.V13_STEPS, G.V13_LR_SEEDS = 5, ((0.03, (0,)),)
    try:
        v13, _psi = G.v13_tier1_runs(s2['gs'], L, 4, c_exact, verbose=True)
    finally:
        G.V13_STEPS, G.V13_LR_SEEDS = steps_bak, lr_bak

    cd = v13.get('chi_down')
    print(f"  chi_down = {cd}")
    assert 'chi_down' in v13, '返回 dict 里没有 chi_down —— 模型侧接线没生效'
    assert cd is not None, 'chi_down 是 None (chi=4 时不该发生)'
    assert cd['chi'] == 4 // 2, f"chi_down 应为 chi//2, 实得 {cd['chi']}"
    assert cd['eps_c'] > 0 and cd['overlap'] > 0, 'chi_down 的读数不合理'
    print(f"  [OK] chi_down.chi = {cd['chi']} (= chi//2, 严格在 exp(S) 之下)")

    # ---------------- (3) chi_sweep 的键 ----------------
    print('\n【3】新守卫要按 chi_dev 取界上的 eps_c, 键必须存在')
    print(f"  chi_sweep 的键 = {sorted(v13['chi_sweep'])}")
    assert 4 in v13['chi_sweep'], 'chi_sweep 里没有 chi_dev 这个键'
    print('  [OK] 界上读数可取')

    # ---------------- (4) 口径 ----------------
    print('\n【4】口径核对')
    print('  本包新增 **1 条计分守卫** (kind=推离) -> 计分守卫 53 -> 54,')
    print('  负对照台账 L4 由「缺」变为有 1 条; 全流程每跑一次 +~38 s (chi=2 那一跑)。')
    print('  **最终数以全流程实测为准** —— 本冒烟不产生通过率。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
