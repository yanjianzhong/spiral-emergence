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
v15·W12 一次性探针 —— L4 缺口的负对照: 把键维压到面积律下界**以下**。

**不跑全流程**。只重放 `main()` 里 L4 那一段的 setup, 然后做一个单变量对照。

L4 台账登记的缺口是「面积律 -> 键维下界 (`derive_L4_bond_dimension`) 这条派生
**没有任何对照**」。本探针构造的是 **「推离」型**负对照 (与 `_NEG_CTRL_TABLE`
的 kind 之一同名):

    把 chi 压到面积律下界 exp(S) 以下 -> 下游 (MERA 复现中心荷 c) 应当退化。

**两种结果都有信息, 不分好坏**:
  - 若界下的 eps_c 明显变大 -> 这条界**确实在约束**下游, 负对照**成立**;
  - 若界下的 eps_c 与界上一样小 -> 这条界**不约束任何东西**, L4 的派生是装饰。
    那是对 L4 的**否证**, 照实登记, 不是"探针失败"。

单变量控制: 同种子 (0) / 同 lr (0.03) / 同预算 (1500 步), **只改 chi**。
chi=4 = 派生的下界上取 2 的幂; chi=8 = 既有的向上对照; **chi=2 = 新的向下对照**
(2 < exp(S) = 2.1171, 严格在界下)。chi=2 与 chi=4 都在本探针里**当场重跑**,
不引用旧日志 —— 否则环境差异无法与 chi 的效应分开。
"""
import time

import spiral_model_v15 as G

L = 16
SEED = 0
LR = 0.03

# 旧日志的锚点 (引自 _v15_run.log 的 print 行, 用来核对本探针重放的是同一套 setup)
LOG = {'entropy': 0.7501, 'chi_lower_bound': 2.1171, 'c_exact': 0.5072,
       'eps_c_chi4_seed0': 0.03027, 'eps_c_chi8_ctrl': 0.01879}


def run_chi(gs, chi, c_exact):
    """与 v13_tier1_runs 里逐字同一条路径: main() 的种子/lr/预算, 只改 chi。"""
    mera, raws = G.mera_init(L, chi, seed=SEED)
    psi, ov, cstep, _curve = G.mera_fit_v13(
        mera, raws, gs, L, steps=G.V13_STEPS, lr=LR, c_exact=c_exact)
    fm = G.fit_central_charge(psi, L)
    eps = abs(fm['c'] - c_exact) / c_exact
    return {'chi': chi, 'eps_c': float(eps), 'c': float(fm['c']),
            'overlap': float(ov), 'step': int(cstep)}


def main():
    print('=' * 84)
    print(' W12 探针: L4 面积律下界的「推离」负对照 (chi 压到界下)')
    print('=' * 84)

    # ---------- (1) 重放 main() 的 L4 setup, 并对旧日志锚点核账 ----------
    print('\n【1】重放 setup 并核对旧日志锚点')
    s2 = G.stage2_critical_break(L=L)
    S = float(s2['entropy'])
    l4 = G.derive_L4_bond_dimension(S)
    c_exact = float(G.fit_central_charge(s2['gs'], L)['c'])
    print(f"  S(L/2)          = {S:.4f}       (旧日志 {LOG['entropy']})")
    print(f"  exp(S) 下界     = {l4['chi_lower_bound']:.4f}    "
          f"(旧日志 {LOG['chi_lower_bound']})")
    print(f"  派生 chi        = {l4['chi']}")
    print(f"  精确 c          = {c_exact:.4f}     (旧日志 {LOG['c_exact']})")
    assert abs(S - LOG['entropy']) < 5e-4, 'S 与旧日志对不上, 不是同一套 setup'
    assert abs(l4['chi_lower_bound'] - LOG['chi_lower_bound']) < 5e-4, \
        '面积律下界与旧日志对不上'
    assert abs(c_exact - LOG['c_exact']) < 5e-4, '精确 c 与旧日志对不上'
    assert int(l4['chi']) == 4, '派生 chi 应为 4'

    # ---------- (2) 非空性: chi=2 必须**严格在下界以下** ----------
    print('\n【2】非空性检查 (没有这一步, "压到界下"可能是空话)')
    below = float(l4['chi_lower_bound']) > 2.0
    print(f"  下界 exp(S) = {l4['chi_lower_bound']:.4f} > 2 -> "
          f"chi=2 严格在界下 = {below}")
    assert below, 'chi=2 并不在界下, 这条对照无从谈起'

    # ---------- (3) 两个 chi 当场重跑 (单变量: 只改 chi) ----------
    print('\n【3】单变量对照 (同种子/同 lr/同预算, 只改 chi)')
    rows = []
    for chi in (4, 2):
        t0 = time.time()
        r = run_chi(s2['gs'], chi, c_exact)
        r['sec'] = time.time() - t0
        rows.append(r)
        rel = '界上' if chi >= l4['chi_lower_bound'] else '**界下**'
        print(f"  chi={chi} ({rel})  ov={r['overlap']:.6f} @step{r['step']:4d}  "
              f"c={r['c']:.6f}  eps_c={r['eps_c']:>8.3%}  ({r['sec']:.0f}s)")

    # 复现性回执: 本探针的 chi=4 应重现旧日志
    e4 = rows[0]['eps_c']
    print(f"\n  [复现性] chi=4 seed0 本次 eps_c={e4:.3%} vs 旧日志 "
          f"{LOG['eps_c_chi4_seed0']:.3%}  "
          f"(差 {abs(e4 - LOG['eps_c_chi4_seed0']):.3%})")

    # ---------- (4) Schmidt 截断参照 (零优化成本, 定性归因用) ----------
    print('\n【4】Schmidt 逐切割截断参照 (非严格界, 只作定性归因)')
    for chi in (2, 4):
        print(f"  chi={chi}: eps_c = {G._schmidt_floor(s2['gs'], L, chi):.3%}")

    # ---------- (5) 判定 ----------
    e2 = rows[1]['eps_c']
    thr = G.KNOBS.get('eps_c_target', 0.05)
    degraded = e2 > max(e4, LOG['eps_c_chi8_ctrl']) * 1.5
    print('\n【5】判定')
    print(f"  chi=8 (界上, 旧日志) eps_c = {LOG['eps_c_chi8_ctrl']:.3%}")
    print(f"  chi=4 (界上, 本次)   eps_c = {e4:.3%}")
    print(f"  chi=2 (界下, 本次)   eps_c = {e2:.3%}   "
          f"(阈值 {thr:.0%}, 超阈 = {e2 > thr})")
    print(f"  -> 压到界下后下游**退化** = {degraded}")
    if degraded:
        print('  [结论] 面积律下界确实在约束下游 —— 「推离」型负对照**成立**。')
    else:
        print('  [结论] 压到界下下游**没有**退化 —— 这条界不约束任何东西,')
        print('         即 L4 的「面积律 -> 键维」派生在本读数下是**装饰**。')
        print('         这是对 L4 的否证, 照实登记。')
    print('\n  判据口径: 本探针判的是"下游是否退化"; eps_c 阈值取产品代码自己的')
    print(f'  KNOBS[eps_c_target]={thr}。判据与结论分离, 不为任何一边调阈值。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
