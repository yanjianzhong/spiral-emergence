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
_v16_smoke_cmera2.py —— v16 · B2 已知答案探针: cMERA 的尺度不变性

探针一(_v16_smoke_cmera.py)只证明了"能构造/能优化/能算 g_uu/资源可接受",
**没有**证明造出来的东西是 cMERA。本探针补这一条。

已知答案(来自 cMERA 的可证伪定义性质):
  cMERA 的"尺度"由标度变换算符 L(u) 承担。对**临界(无质量)**系统,
  最优 cMERA 是**尺度不变**的 —— K(u)、L(u) 与 u 无关, 这正是其径向度规为 AdS 的由来。
  对**有质量**系统, 尺度不变性破缺, 生成元随 u 变化, 变化起始尺度 u* ~ log(1/m)。

TFI 经 Jordan-Wigner 是自由费米子: h=J 临界(eps(k)=4|sin(k/2)|, 无质量),
h!=J 有质量 m = 2|J-h|。故:
  h/J = 1.0  ->  逐层生成元应高度相似(尺度不变)
  h/J = 0.5  ->  相似度应显著更低

判据(可证伪):
  (a) 两个 h/J 都要先**收敛**(能量差远小于未收敛时的 0.3~0.8%)
  (b) sim(h=1.0) 显著大于 sim(h=0.5), 且 sim(h=1.0) 接近 1
  若 (b) 不成立 -> 本构造不含 cMERA 的尺度结构, B2 的 cMERA 路线需重写构造。

边界:
  - 只比**同奇偶层**(brickwork 奇偶层键位不同, 跨奇偶不可比)。
  - 生成元 A 有参数化冗余(A 只取 P 的反厄米部分), 故比的是 A 本身(门是规范不变的)。
  - 本探针仍不产出物理结论; 它只回答"这个构造有没有 cMERA 的尺度结构"。
"""

import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'v15'))

from _v16_smoke_cmera import CMERA, _gen_and_unitary, tfi_energy  # noqa: E402
from spiral_model_v15 import jw_ground_energy                      # noqa: E402

_VERSION_TAG = 'v16-smoke-cmera2-1'
_PARITY_GAP = 2      # 只比同奇偶层(brickwork 键位相同)


def optimize(L, h, n_layers, steps, seed=0, lr=0.02, verbose=True):
    """优化到能量收敛, 返回 (模型, E_init, E_final, 用时)。"""
    model = CMERA(L, n_layers, seed=seed)
    opt = torch.optim.Adam(model.params(), lr=lr)
    t0 = time.time()
    with torch.no_grad():
        psi, _ = model.forward()
        e_init = float(tfi_energy(psi, L, 1.0, h))
    for _ in range(steps):
        opt.zero_grad()
        psi, _ = model.forward()
        e = tfi_energy(psi, L, 1.0, h)
        e.backward()
        opt.step()
    dt = time.time() - t0
    with torch.no_grad():
        psi, _ = model.forward()
        e_fin = float(tfi_energy(psi, L, 1.0, h))
    if verbose:
        print(f"    E {e_init:+.6f} -> {e_fin:+.6f}   ({dt:.1f}s, {steps} 步)")
    return model, e_init, e_fin, dt


def layer_generators(model):
    """每层的生成元 A 拼成实向量(实部+虚部), 用于层间比较。"""
    L = model.L
    out = []
    with torch.no_grad():
        for k in range(model.n_layers):
            parts = []
            for m, _pair in enumerate(model._bonds(k)):
                A, _ = _gen_and_unitary(model.p2[k, m], 4)
                parts.append(torch.cat([A.real.reshape(-1), A.imag.reshape(-1)]))
            for i in range(L):
                A, _ = _gen_and_unitary(model.p1[k, i], 2)
                parts.append(torch.cat([A.real.reshape(-1), A.imag.reshape(-1)]))
            out.append(torch.cat(parts))
    return out


def similarity(vecs, gap=_PARITY_GAP):
    """同奇偶层的生成元余弦相似度剖面。"""
    sims = []
    for k in range(len(vecs) - gap):
        a, b = vecs[k], vecs[k + gap]
        denom = float(a.norm() * b.norm())
        sims.append(float(torch.dot(a, b)) / denom if denom > 0 else float('nan'))
    return sims


def main():
    print(f"=== v16 已知答案探针 · cMERA 尺度不变性  [{_VERSION_TAG}] ===")
    L, N_LAYERS, STEPS = 8, 8, 1200

    results = {}
    for h in (1.0, 0.5):
        print(f"\n[h/J = {h}]  L={L} 层数={N_LAYERS}")
        model, e0, e1, dt = optimize(L, h, N_LAYERS, STEPS)
        e_exact = jw_ground_energy(L, 1.0, h)
        gap = abs(e1 - e_exact) / abs(e_exact)
        sims = similarity(layer_generators(model))
        results[h] = {'e_fin': e1, 'e_exact': e_exact, 'gap': gap,
                      'sims': sims, 'mean_sim': float(np.mean(sims)), 'seconds': dt}
        print(f"    精确(JW) = {e_exact:+.6f}   收敛差距 = {gap * 100:.4f}%")
        print(f"    层间相似度 = [{', '.join(f'{v:+.3f}' for v in sims)}]")
        print(f"    平均相似度 = {np.mean(sims):.4f}")

    # ---- 已知答案判定 ----
    s_crit = results[1.0]['mean_sim']
    s_mass = results[0.5]['mean_sim']
    conv_ok = max(results[h]['gap'] for h in results) < 1e-3
    scale_ok = (s_crit > s_mass) and (s_crit > 0.9)

    print("\n=== 已知答案判定 ===")
    print(f"    收敛充分 (<0.1%)            : {'通过' if conv_ok else '不通过'}"
          f"  (最大差距 {max(results[h]['gap'] for h in results) * 100:.4f}%)")
    print(f"    临界尺度不变 > 有质量        : {'通过' if scale_ok else '不通过'}")
    print(f"      sim(h/J=1.0) = {s_crit:.4f}   sim(h/J=0.5) = {s_mass:.4f}")
    print("\n    若'不通过': 本构造不含 cMERA 的尺度结构, 应重写构造"
          "(entangler 需约束为真正的标度变换算符), 而不是继续加层/加步数。")
    return 0 if (conv_ok and scale_ok) else 3


if __name__ == '__main__':
    sys.exit(main())
