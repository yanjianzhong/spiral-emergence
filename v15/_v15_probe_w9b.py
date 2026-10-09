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
v15·W9 侦察 之二 —— 只读, 不写文件。

W9 的第一步 (_v15_probe_w9.py) 得到: 模型侧与归档侧的 Gray-Scott 参数都
**没有非平凡齐次定态** (判别式 F² − 4F(F+k)² < 0), 所以 Turing 线性稳定性
**给不出 λ_c** —— "第三条独立路径"在解析上不存在。

本脚本回答下一个问题: 那 lam_pk 到底偏在哪?
做法: 在**同一个模型场**上, 把所有能用的尺度估计量并排摆出来。

    lam_pk        Δ²(k) 的 argmax       (二值场, A 段的实际口径)
    P(k) 峰       功率谱本身的 argmax    (二值场 / 未二值化 delta)
    零交叉间距    v 沿 x 的实空间周期    (原始 v 场, stage6_life 自己的量)
    4*xi_ac       自相关首次过零 x4      (与谱峰无关的估计量)

**注意顺序**: 这里只是把数摆出来, 不在脚本里下"谁对"的结论。
"""
import os

import numpy as np

import spiral_metric_v15 as M
import spiral_model_v15 as G

HERE = os.path.dirname(os.path.abspath(__file__))
N, L, STEPS = 36, 0.5, 40000


def pk_argmax_wavelength(delta, box, deconv=False):
    """P(k) **本身**的 argmax 对应波长 —— 与 _char_scale 取 Δ² 的 argmax 相对照。"""
    pk = M._band_power(delta, box, 2, deconv=deconv)
    if len(pk['k']) == 0:
        return float('nan'), float('nan'), pk
    i = int(np.argmax(pk['P']))
    return float(2 * np.pi / pk['k'][i]), float(pk['k'][i]), pk


def d2_argmax_wavelength(delta, box, deconv=False):
    """Δ²(k) 的 argmax —— 逐字复刻 _char_scale 的取法。"""
    pk = M._band_power(delta, box, 2, deconv=deconv)
    if len(pk['k']) == 0:
        return float('nan'), pk
    i = int(np.argmax(M._delta2(pk['k'], pk['P'], 2)))
    return float(2 * np.pi / pk['k'][i]), pk


def zero_cross_spacing(v, L):
    """逐字复刻 stage6_life 的斑图波长取法 (v 沿 x 的零交叉间隔 x2)。"""
    N = v.shape[0]
    zc = np.mean([len(np.where(np.diff(np.sign(v[i] - v[i].mean())))[0])
                  for i in range(N)])
    return float(2.0 * L / zc) if zc > 0 else float('inf')


def bandwidth_stats(delta, box, deconv=False):
    """
    谱峰的**相对带宽**与**单色性** —— 回答"宽带"这个成因是实测还是推断。

    Δ²(k) 是 _char_scale 实际取 argmax 的量, 所以宽度就在 Δ² 上量。
    壳是离散的 (k 是壳内 |k| 均值), 宽度也以壳为单位, 因此 Δk 有量化台阶,
    比值只当量级读。

    单色性用参与比 (ΣΔ²)² / Σ(Δ²)²: 全部功率落在一个壳 -> 1;
    均匀铺在 n 个壳 -> n。它比"半高宽"稳, 不受峰形影响。
    """
    pk = M._band_power(delta, box, 2, deconv=deconv)
    if len(pk['k']) == 0:
        return None
    d2 = np.asarray(M._delta2(pk['k'], pk['P'], 2), dtype=float)
    d2 = np.where(np.isfinite(d2), d2, 0.0)
    if d2.max() <= 0:
        return None
    i = int(np.argmax(d2))
    above = d2 >= 0.5 * d2[i]
    lo = hi = i
    while lo - 1 >= 0 and above[lo - 1]:
        lo -= 1
    while hi + 1 < len(d2) and above[hi + 1]:
        hi += 1
    kpk = float(pk['k'][i])
    # 半高区只占一个壳时 hi == lo, 宽度不能报 0 —— 那是"无限锐"的假象。
    # 一个壳本身就有宽度, 所以按"跨了几个壳"算, 再乘壳宽。
    dk_sh = float(pk['k'][1] - pk['k'][0]) if len(pk['k']) > 1 else 0.0
    dk = float((hi - lo + 1) * dk_sh)
    den = float((d2 ** 2).sum())
    return {'k_pk': kpk, 'lam_pk': 2 * np.pi / kpk, 'dk': dk, 'rel': dk / kpk,
            'n_above': int(above.sum()), 'n_shells': int(len(d2)),
            'n_span': int(hi - lo + 1),
            'participation': float(d2.sum() ** 2 / den) if den > 0
            else float('nan')}


def _bw_row(name, delta, box):
    st = bandwidth_stats(delta, box)
    if st is None:
        print(f'  {name:<34} 谱为空')
        return None
    print(f'  {name:<34} {st["lam_pk"]:>8.4f} {st["k_pk"]:>9.4f} '
          f'{st["dk"]:>8.4f} {st["rel"]:>8.1%} {st["participation"]:>7.1f} '
          f'{st["n_above"]:>4}/{st["n_shells"]:<4}')
    return st


def main():
    print('=' * 88)
    print(' W9 侦察二: 同一个模型场上, 各尺度估计量并排')
    print('=' * 88)

    s6 = G.stage6_life(F=G.KNOBS['F'], k=G.KNOBS['k'], N=N, L=L,
                       steps=STEPS, dt=G.KNOBS['dt'],
                       Du=G.KNOBS['Du'], Dv=G.KNOBS['Dv'])
    if not s6.get('ok'):
        print(f"  stage6_life 拒绝运行: {s6.get('reason')}")
        return 1
    h = s6['h']
    print(f"\n  【复现性核对】实跑 N={N}, L={L}, steps={STEPS}")
    print(f"    stab            = {s6['stab']:.6f}   (JSON 记 0.10368)")
    print(f"    contrast        = {s6['contrast']:.6f}  (JSON 记 0.349076)")
    print(f"    active_frac     = {s6['active_frac']:.6f}  (JSON 记 0.637346)")
    print(f"    spacing         = {s6['spacing']:.10f}  (JSON 记 "
          f"0.1782178218)")
    _sp_json = 0.17821782178217824
    print(f"    spacing 逐位一致: {s6['spacing'] == _sp_json}")
    if s6['spacing'] != _sp_json:
        print('    !! 没有复现出 A 段那个场, 下面的对比不能当 A 段读数用')

    v = np.asarray(s6['v'], dtype=np.float64)
    delta_raw = M._model_field(s6)          # v/<v> - 1
    b_mod = M._binarize_median(delta_raw)   # A 段的实际输入
    box = float(max(b_mod.shape))
    delta_bin = b_mod.astype(np.float64) - b_mod.mean()

    print(f'\n  场: {b_mod.shape[0]}^2, 域长 {L}, h={h:.6f}, '
          f'占空比 phi={float(b_mod.mean()):.4f}')

    rows = []

    # 1) A 段的实际口径
    dmod = M.structural_descriptors(b_mod)
    rows.append(('lam_pk  (D2 argmax, 二值场)  <- A 段口径', dmod['lam_pk']))

    # 2) P(k) 本身的峰
    lam_p_raw, k_p_raw, _ = pk_argmax_wavelength(delta_raw, box)
    rows.append(('P(k) argmax (未二值化 delta)', lam_p_raw))
    lam_p_bin, k_p_bin, _ = pk_argmax_wavelength(delta_bin, box)
    rows.append(('P(k) argmax (二值场)', lam_p_bin))
    lam_d2_raw, _ = d2_argmax_wavelength(delta_raw, box)
    rows.append(('D2 argmax (未二值化 delta)', lam_d2_raw))

    # 3) 实空间
    rows.append(('零交叉间距 (原始 v 场)  <- stage6_life 口径',
                 zero_cross_spacing(v, L) / h))

    # 4) 自相关
    xi_ac = M._ac_len(b_mod)
    rows.append(('xi_ac (二值场自相关首次过零)', xi_ac))
    rows.append(('4 * xi_ac', 4.0 * xi_ac))

    print(f'\n  {"估计量":<42} {"格":>9} {"域长":>10} {"域内波长数":>11}')
    print('  ' + '-' * 76)
    lam_pk_cells = dmod['lam_pk']
    for name, cells in rows:
        if not np.isfinite(cells) or cells <= 0:
            print(f'  {name:<42} {cells!s:>9}')
            continue
        print(f'  {name:<42} {cells:>9.4f} {cells*h:>10.6f} '
              f'{L/(cells*h):>11.4f}')

    print(f'\n  【与 A 段记录对照】')
    print(f'    JSON n_lambda_model = 4.0757913411 -> lam_pk = '
          f'{36/4.0757913411:.4f} 格')
    print(f'    本脚本 lam_pk       = {lam_pk_cells:.4f} 格')
    print(f'    JSON xi_ac_model    = 4.0  -> 4*xi_ac = 16 格')
    print(f'    本脚本 xi_ac        = {xi_ac:.4f} -> 4*xi_ac = '
          f'{4*xi_ac:.4f} 格')

    print(f'\n  【估计量之间的比值 (以 lam_pk 为分母)】')
    for name, cells in rows:
        if not np.isfinite(cells) or cells <= 0:
            continue
        print(f'    {name:<42} / lam_pk = {cells/lam_pk_cells:>8.4f}  '
              f'({(cells/lam_pk_cells - 1)*100:+.1f}%)')

    print(f'\n  【A 段设计目标】L_domain/n_lambda = {L}/3 = {L/3:.6f} 域长 '
          f'= {L/3/h:.4f} 格')
    print(f'    lambda_ratio (JSON) = 1.0693069 -> spacing / 目标 = '
          f'{s6["spacing"]/(L/3):.6f}')

    # ------------------------------------------------------------------
    # 侦察三: 上一个探测只能推断"这组参数在 Turing 区外 => 斑图是宽带的"。
    # 推断不是实测。这一节直接量谱宽, 让成因要么落实, 要么被推翻。
    # ------------------------------------------------------------------
    print('\n' + '=' * 88)
    print(' W9 侦察三: "宽带"是实测还是推断 —— 谱峰相对带宽与单色性')
    print('=' * 88)
    print('  量在 Δ²(k) 上 ( _char_scale 实际取 argmax 的量)。半高宽在离散壳上量,')
    print('  有量化台阶, 比值只当量级读; 参与比 (=1 单色 / =n 均匀) 不受峰形影响。\n')
    print(f'  {"场":<34} {"λ峰(格)":>8} {"k峰":>9} {"Δk":>8} {"Δk/k":>8} '
          f'{"参与比":>7} {"半高壳":>7}')
    print('  ' + '-' * 84)

    _bw_row('模型侧 二值场  <- A 段口径', b_mod.astype(float) - b_mod.mean(),
            box)
    _bw_row('模型侧 未二值化 v delta', delta_raw, box)

    # 参考侧: 与 A 段同一条管线 (bool_ 末帧), 逐种子量再报中位
    _zc = np.load(os.path.join(HERE, 'data', '_v14_cache', 'gs_seeds.npz'),
                  allow_pickle=True)
    _seeds = sorted(int(k.split('_')[1]) for k in _zc.files
                    if k.startswith('bool_'))
    _bws, _lams = [], []
    for s in _seeds:
        _b = M._late_frame(_zc[f'bool_{s}'])
        _bx = float(max(_b.shape))
        _st = bandwidth_stats(_b.astype(float) - float(_b.mean()), _bx)
        if _st is not None:
            _bws.append(_st)
            _lams.append(M.structural_descriptors(_b)['lam_pk'])
    if _bws:
        _md = lambda a: float(np.median(a))  # noqa: E731
        print(f'  {"参考侧 " + str(len(_bws)) + " 个种子 (中位)":<34} '
              f'{_md([s["lam_pk"] for s in _bws]):>8.4f} '
              f'{_md([s["k_pk"] for s in _bws]):>9.4f} '
              f'{_md([s["dk"] for s in _bws]):>8.4f} '
              f'{_md([s["rel"] for s in _bws]):>8.1%} '
              f'{_md([s["participation"] for s in _bws]):>7.1f} '
              f'{_md([s["n_above"] for s in _bws]):>4.0f}/'
              f'{_bws[0]["n_shells"]:<4}')
    print(f'\n  参考侧 lam_pk 中位 = {np.median(_lams):.4f} 格 '
          f'(JSON 记 28.7475)')

    print('\n  【读法】参与比 ≈1 => 谱几乎是单色, 有一条被选定的波长;')
    print('          参与比 >> 1 且半高壳成片 => 宽带, 没有单一波长。')
    print('          若两侧都是宽带, 则"谱峰 vs 实空间周期对不上 45%"是这类场的'
          '性质, 不是某一个估计量的偏 —— 成因落实。')

    # ------------------------------------------------------------------
    # 侦察四: 上一节的读法预期"宽带", 若实测谱很窄, 则宽带这条解释被推翻,
    # 成因仍空着。下一条最便宜的候选是**形态**: 斑块/斑点状而非条纹状时,
    # 逐行过均值点的次数与谱峰波长脱钩。逐行游程统计可以直接分辨。
    # ------------------------------------------------------------------
    print('\n' + '=' * 88)
    print(' W9 侦察四: 若"宽带"被推翻, 成因是否在**形态** (条纹 vs 斑块)')
    print('=' * 88)

    def run_stats(bb):
        """二值场逐行游程统计。理想条纹 (周期 λ) -> 每行 2N/λ 段, 段长齐 λ/2。"""
        _N = int(bb.shape[0])
        bi = bb.astype(np.int8)
        per_row, allr = [], []
        for i in range(_N):
            idx = np.where(np.diff(bi[i]) != 0)[0] + 1
            seg = np.diff(np.concatenate([[0], idx, [_N]])).astype(float)
            per_row.append(len(seg))
            allr.append(seg)
        a = np.concatenate(allr)
        return (float(np.mean(per_row)), float(a.mean()),
                float(a.std() / a.mean()) if a.mean() > 0 else float('nan'))

    _lp = 8.8326
    print(f'  {"场":<30} {"每行段数":>9} {"理想":>8} {"均段长":>8} {"理想":>8} '
          f'{"段长CV":>8} {"占空比":>8}')
    print('  ' + '-' * 82)
    _r = run_stats(b_mod)
    print(f'  {"模型侧 二值场":<30} {_r[0]:>9.2f} {2*36/_lp:>8.2f} '
          f'{_r[1]:>8.2f} {_lp/2:>8.2f} {_r[2]:>8.3f} '
          f'{float(b_mod.mean()):>8.4f}')
    for s in _seeds[:3]:
        _b = M._late_frame(_zc[f'bool_{s}'])
        _st = bandwidth_stats(_b.astype(float) - float(_b.mean()),
                              float(max(_b.shape)))
        _rr = run_stats(_b)
        _l = M.structural_descriptors(_b)['lam_pk']
        print(f'  {"参考 seed " + str(s):<30} {_rr[0]:>9.2f} {2*_b.shape[0]/_l:>8.2f} '
              f'{_rr[1]:>8.2f} {_l/2:>8.2f} {_rr[2]:>8.3f} '
              f'{float(_b.mean()):>8.4f}')
    print(f'\n  读法: 段长 CV ≈ 0 且每行段数≈理想 => 规则条纹;')
    print('        CV 大 或 段数远低于理想 => 斑块/缺陷多, 逐行过均值点与谱峰'
          '波长本来就不是一回事。')
    print('  注: 参考侧占空比恒为 0.5000 (中位数二值化的恒等式), 那一列在参考侧'
          '没有信息量, 只看模型侧。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
