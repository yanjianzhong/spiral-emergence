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
v15·W9 侦察脚本 —— 只读, 0 s 全流程。

目的: 给 A 段的 `lam_pk` 补一条**真正独立**的佐证 —— Gray-Scott 的
Turing 线性稳定性分析给出的最不稳定波长 λ_c。它与谱峰 argmax、与自相关
尺度都没有共同来源, 所以是"第三条路"。

方程 (Pearson 形式的 Gray-Scott):
    ∂u/∂t = Du ∇²u − u v² + F (1 − u)
    ∂v/∂t = Dv ∇²v + u v² − (F + k) v

**方向说明 (照 W9 计划)**: 若 λ_c 与 lam_pk 对不上, 那是**结果**, 不是失败。

用法: python _v15_probe_w9.py
"""
import json
import os
import numpy as np

import spiral_model_v15 as G

HERE = os.path.dirname(os.path.abspath(__file__))


def steady_states(F, k):
    """
    齐次定态。v != 0 支: F u² − F u + (F+k)² = 0。
    判别式 < 0 => 只有平凡定态 (1, 0)。
    """
    disc = F ** 2 - 4.0 * F * (F + k) ** 2
    out = {'disc': disc, 'cond_F_ge_4Fk2': F >= 4.0 * (F + k) ** 2}
    if disc < 0:
        return None, out
    r = np.sqrt(disc)
    us = [(F + r) / (2 * F), (F - r) / (2 * F)]
    pts = [(float(u), float((F + k) / u)) for u in us if u > 0]
    return pts, out


def jacobian(F, k, u, v):
    """反应项 Jacobian。f = −uv² + F(1−u), g = uv² − (F+k)v。"""
    fu = -v * v - F
    fv = -2.0 * u * v
    gu = v * v
    gv = 2.0 * u * v - (F + k)
    return np.array([[fu, fv], [gu, gv]])


def dispersion(F, k, Du, Dv, u, v, kmax=400.0, nk=200000):
    """
    线性化的色散关系。∇² -> −k², 所以 J(k) = J + k²·diag(−Du, −Dv)。
    返回 (k 网格, 最大增长率 σ(k), det(k))。
    """
    J = jacobian(F, k, u, v)
    ks = np.linspace(0.0, kmax, nk)
    f_u, f_v = J[0, 0], J[0, 1]
    g_u, g_v = J[1, 0], J[1, 1]
    k2 = ks ** 2
    a = f_u - Du * k2
    d = g_v - Dv * k2
    tr = a + d
    det = a * d - f_v * g_u
    disc = tr ** 2 - 4.0 * det
    sq = np.sqrt(np.abs(disc).astype(complex))
    sig = (tr + sq) / 2.0
    return ks, sig.real, det


def turing_threshold(F, k, Du, Dv, u, v):
    """Turing 必要条件 (两式都要满足)。"""
    J = jacobian(F, k, u, v)
    f_u, f_v, g_u, g_v = J[0, 0], J[0, 1], J[1, 0], J[1, 1]
    c1 = Du * g_v + Dv * f_u
    c2 = c1 ** 2 - 4.0 * Du * Dv * (f_u * g_v - f_v * g_u)
    return {'Du*g_v + Dv*f_u': c1, 'c1^2 - 4 Du Dv det(J)': c2,
            'turing_ok': bool(c1 > 0 and c2 > 0)}


def report(tag, F, k, Du, Dv, h=None, L_domain=None):
    print('=' * 84)
    print(f' {tag}')
    print(f'   F={F}, k={k}, Du={Du}, Dv={Dv}')
    print('=' * 84)
    pts, info = steady_states(F, k)
    print(f'  判别式 F² − 4F(F+k)² = {info["disc"]:+.6e}')
    print(f'  4(F+k)² = {4*(F+k)**2:.6f}   F = {F:.6f}   '
          f'差 F − 4(F+k)² = {F - 4*(F+k)**2:+.6e}')
    if pts is None:
        print('  -> **无非平凡齐次定态** (判别式 < 0), 只有平凡定态 (u,v)=(1,0)。')
        print('     Turing 分析没有可线性化的非平凡定点。')
        J0 = jacobian(F, k, 1.0, 0.0)
        ev0 = np.linalg.eigvals(J0)
        print(f'     平凡定态 J = [[{J0[0,0]:+.4f}, {J0[0,1]:+.4f}], '
              f'[{J0[1,0]:+.4f}, {J0[1,1]:+.4f}]]')
        print(f'     特征值 = {np.round(ev0, 6)}  -> max Re = {ev0.real.max():+.4f}')
        print('     (平凡定态 u=1 处 v 的增长率 = f 的系数 = F − ... 见上)')
        return None
    res = None
    for (u, v) in pts:
        print(f'\n  定态 (u*, v*) = ({u:.6f}, {v:.6f})')
        J = jacobian(F, k, u, v)
        ev = np.linalg.eigvals(J)
        print(f'    J = [[{J[0,0]:+.4f}, {J[0,1]:+.4f}], '
              f'[{J[1,0]:+.4f}, {J[1,1]:+.4f}]]')
        print(f'    特征值 = {np.round(ev, 6)}   max Re = {ev.real.max():+.6f}')
        print(f'    均匀态稳定 (max Re < 0): {bool(ev.real.max() < 0)}')
        th = turing_threshold(F, k, Du, Dv, u, v)
        for kk, vv in th.items():
            print(f'    {kk} = {vv}')
        if not th['turing_ok']:
            print('    -> **不满足 Turing 条件**, 无图灵失稳。')
            continue
        ks, sig, det, = dispersion(F, k, Du, Dv, u, v)
        i = int(np.argmax(sig))
        kc = float(ks[i])
        lam_c = 2.0 * np.pi / kc
        band = ks[det < 0]
        print(f'    σ_max = {sig[i]:+.6e} @ k_c = {kc:.4f} /域长')
        print(f'    -> **λ_c = {lam_c:.6f} 域长单位**')
        if band.size:
            k1, k2 = float(band.min()), float(band.max())
            print(f'    失稳带 k ∈ [{k1:.2f}, {k2:.2f}] -> '
                  f'λ ∈ [{2*np.pi/k2:.4f}, {2*np.pi/k1:.4f}] 域长单位')
        if h:
            print(f'    h = {h:.6f} 域长/格 -> λ_c = **{lam_c/h:.3f} 格**')
            if L_domain is not None:
                print(f'    域内波长数 L/λ_c = {L_domain/lam_c:.3f}')
        res = {'k_c': kc, 'lam_c': lam_c, 'sigma_max': float(sig[i]),
               'u': u, 'v': v}
    return res


def main():
    print('#' * 84)
    print('# v15·W9: Gray-Scott Turing 线性稳定性 —— 给 lam_pk 的第三条独立路径')
    print('#' * 84)

    kn = G.KNOBS
    F, k, Du, Dv = kn['F'], kn['k'], kn['Du'], kn['Dv']
    L_domain = kn['L_domain']
    h = L_domain / 36.0            # N_derived = 36, 见 result/_v15_data.json

    print('\n【对照读数 —— 引自 result/_v15_data.json, 非本脚本算出】')
    print(f'  模型侧 lam_pk   = 36 / n_lambda_model(4.0757913411) = '
          f'{36/4.0757913411:.4f} 格')
    sp = 0.17821782178217824
    print(f'  模型侧 spacing  = {sp:.6f} 域长 = {sp/h:.4f} 格 '
          f'(stage6_life, 与 lam_pk 差 '
          f'{abs(sp/h - 36/4.0757913411)/(36/4.0757913411):.1%})')
    print(f'  参考侧 lam_pk 中位 = 28.7475 格;  4*xi_ac 中位 = 58 格')
    print('  (A-char_scale 守卫报的 46.6% 缺口 —— **那是参考侧的**)')

    print()
    r_mod = report('模型侧 (spiral_model_v15.KNOBS, 受控参数)',
                   F, k, Du, Dv, h=h, L_domain=L_domain)

    print()
    zc = np.load(os.path.join(HERE, 'data', '_v13_cache', 'clip_gs.npz'),
                 allow_pickle=True)
    cpar = json.loads(str(zc['meta'])).get('params', {})
    print('=' * 84)
    print(' 参考侧 (归档 CLIP GS) 的 meta.params 原文')
    print('=' * 84)
    print('  ' + json.dumps(cpar, ensure_ascii=False))

    def pick(names):
        low = {kk.lower(): kk for kk in cpar}
        for n in names:
            if n in low:
                return cpar[low[n]]
        return None

    fr, kr = pick(('f',)), pick(('k',))
    dur = pick(('du', 'd_u', 'diffu', 'diff_u', 'dud'))
    dvr = pick(('dv', 'd_v', 'diffv', 'diff_v', 'dvd'))
    print(f'\n  解析到: f={fr}  k={kr}  Du={dur}  Dv={dvr}')
    if fr is None or kr is None:
        print('  -> **归档 meta 里没有 f/k**, 参考侧 Turing 分析缺输入, 只能登记缺口。')
        return
    if dur is None or dvr is None:
        print('  -> **归档 meta 未给扩散系数**。用模型侧的 Du/Dv 做一次"若同一'
              '标度"的试算 —— 这是**假设**, 不是归档事实, 必须标注。')
        dur, dvr = Du, Dv
    print()
    report('参考侧 (归档 f,k; 扩散系数见上, 无格子物理尺寸 -> 只报域长单位)',
           float(fr), float(kr), float(dur), float(dvr))


if __name__ == '__main__':
    main()
