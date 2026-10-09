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
B1 一次性探测: MERA 体几何图 G 是否依赖键维 chi?

背景: L5 的体几何建在 `mera_init(32, 2, seed=0)` 上 (spiral_model_v15.py:3485),
       chi 是写死的 2; 而 L4 派生的键维是 chi_dev = 4 (面积律 exp(S)=2.1171)。
       若 G 与 chi 无关, 这个不一致就是无害的 (chi 只定指标维度, 不定接线);
       若 G 随 chi 变, 则 L5 用的不是 L4 派生的那个量。

判据: 逐个 (L, chi) 建图, 比边集 (frozenset of sorted tuples)。
      边集逐位相同 -> G 与 chi 无关。

不跑全流程, 只构造张量网络对象, 不收缩成稠密态。
"""
from spiral_model_v15 import build_mera_graph, mera_init


def edges(L, chi, seed=0):
    mera, _ = mera_init(L, chi, seed=seed)
    G = build_mera_graph(mera)
    return (frozenset(tuple(sorted(e)) for e in G.edges()),
            G.number_of_nodes(), G.number_of_edges(), mera.num_tensors)


print(f"{'L':>4} {'chi':>4} {'|V|':>5} {'|E|':>5} {'num_tensors':>12}")
ref = {}
for L in (8, 16, 32):
    for chi in (2, 4, 8):
        e, V, E, nt = edges(L, chi)
        print(f"{L:>4} {chi:>4} {V:>5} {E:>5} {nt:>12}")
        ref.setdefault(L, {})[chi] = e

print()
for L, d in ref.items():
    base = d[2]
    same = {c: (e == base) for c, e in d.items()}
    print(f"L={L:>2}: 边集与 chi=2 相同 -> {same}  "
          f"=> {'G 与 chi 无关' if all(same.values()) else '**G 依赖 chi**'}")

# 对照: G 是否依赖 L? (应当依赖 —— 张量数不同)
print()
print(f"L 依赖对照: L=8 vs L=16 边集相同 = "
      f"{ref[8][2] == ref[16][2]} (应当为 False, 否则本探测无分辨力)")

# 对照: 张量数是否等于二进制 MERA 的 2L-2?
print()
for L in (8, 16, 32):
    _, _, _, nt = edges(L, 2)
    print(f"L={L:>2}: num_tensors={nt}, 2L-2={2*L-2}, 相符={nt == 2*L-2}")
