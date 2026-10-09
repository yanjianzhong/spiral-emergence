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
v15·B1 冒烟: 调**真函数** mera_bulk_invariance, 验四件事。

项目纪律: 冒烟必须调用真函数 —— 复制一份逻辑出来跑, 只能验算法, 验不了接线。
故本脚本 import 的就是主程序里那个函数, 不重写。

验:
  1. 新键在不在 (chi_same / chis / n_nodes / nodes_match_binary_mera)
  2. chi 不变性 = True  (本轮要补进 F6d note 的那一条)
  3. 非恒真对照: 换 L, |E| 必须变 (否则这个比较机制没有分辨力, 测什么都报 True)
  4. RNG 状态守恒: 函数体内存还了全局 torch RNG (否则污染下游)

不跑全流程 —— 只构造不收缩。L=32 在笔记本上秒级。
"""
import torch

from spiral_model_v15 import mera_bulk_invariance

print("=== 1. 真函数 @ L=32 ===")
_rng_before = torch.get_rng_state().clone()
d = mera_bulk_invariance(32)
_rng_after = torch.get_rng_state()
for k, v in d.items():
    print(f"  {k:>28} = {v}")

print()
print("=== 2. 新键齐不齐 ===")
need = ['chis', 'chi_same', 'chi_invariant', 'n_nodes',
        'nodes_match_binary_mera', 'expected_nodes_2L_minus_2',
        'same', 'seed_invariant', 'seeds', 'n_edges']
missing = [k for k in need if k not in d]
print(f"  缺键: {missing if missing else '无'}")

print()
print("=== 3. 断言 ===")
checks = [
    ("chi 不变 (chi_same)",        d['chi_same'] is True),
    ("chi 不变 (chi_invariant)",   d['chi_invariant'] is True),
    ("态不变 (seed_invariant)",    d['seed_invariant'] is True),
    ("|V| = 2L-2",                 d['nodes_match_binary_mera'] is True),
    ("|V| = 62",                   d['n_nodes'] == 62),
]
for name, ok in checks:
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}")

print()
print("=== 4. 非恒真对照: 换 L, |E| 必须变 ===")
d16 = mera_bulk_invariance(16)
print(f"  L=16: |E|={d16['n_edges']}, |V|={d16['n_nodes']}")
print(f"  L=32: |E|={d['n_edges']}, |V|={d['n_nodes']}")
print(f"  两者 |E| 不同 = {d16['n_edges'] != d['n_edges']} "
      f"(必须 True, 否则比较机制无分辨力)")

print()
print("=== 5. RNG 状态守恒 ===")
print(f"  调用前后 torch RNG 状态相同 = {torch.equal(_rng_before, _rng_after)}")
