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
v15·W8 侦察脚本 —— 只读, 不改任何产品代码。

目的: 实测 `T3-4b` 为什么在 Reading 侧报 "0/3 份序列做了自相关滞后检验"。

`spiral_metric_v15.tier3_structural_ladder` 里 Reading 分支的调用是
    _la = _series_lag_agreement(_s2, sp['period_frames_band_argmax'])
而 `_series_lag_agreement` 有三处 `return None`。本脚本逐条打印是哪一个拦下的,
并给出该分支下**可用的样本量**，供判断"该不该拦"。

用法: python _v15_probe_w8.py
"""
import os
import numpy as np

import spiral_metric_v15 as M

HERE = os.path.dirname(os.path.abspath(__file__))
READING = os.path.join(HERE, 'data', '_v13_cache', 'bz_reading.npz')


def where_none(series, period_px):
    """逐条复刻 _series_lag_agreement 的早退条件, 报出拦在哪一条。"""
    s = np.asarray(series, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] < 2 or s.shape[0] < 16:
        return 'A: 形状/长度不足', {}
    x, y = s[:, 0], s[:, 1]
    lag = int(round(float(period_px)))
    if lag < 2:
        return 'B: lag < 2', {'lag': lag}
    m = x > 3.0 * lag
    info = {'lag': lag, 'n_total': int(s.shape[0]),
            'skip_3lag': int(3 * lag), 'n_after_skip': int(m.sum()),
            'need_12lag': int(12 * lag),
            'periods_after_skip': float(m.sum()) / lag,
            'pairs_at_lag': int(m.sum()) - lag}
    if int(m.sum()) < 12 * lag:
        return 'C: 跳过后样本 < 12*lag', info
    return '通过 (未早退)', info


def main():
    print('=' * 78)
    print(' W8 侦察: T3-4b 在 Reading 侧为何 0/3')
    print('=' * 78)
    zr = np.load(READING, allow_pickle=True)
    lum_keys = [k for k in zr.files if k.startswith('lum_')]
    print(f'缓存: {READING}')
    print(f'lum_ 序列: {lum_keys}\n')

    for k in lum_keys:
        y = np.asarray(zr[k], dtype=float)
        nm = k[4:]
        sp = M._series_period(y)
        print('-' * 78)
        print(f'[{nm}]  n={len(y)}')
        if sp is None:
            print('  _series_period 返回 None -> 连 per 都不进, 直接跳过')
            continue
        pf = sp['period_frames']
        pba = sp['period_frames_band_argmax']
        print(f"  verdict        = {sp['verdict']}")
        print(f"  带内argmax周期 = {pba:.1f} 帧  (k={sp['k_band_argmax']})")
        print(f"  检出后主周期   = {pf}")

        # 产品代码里的实际调用 (Reading 分支照抄)
        s2 = np.column_stack([np.arange(len(y), dtype=float), y])
        la = M._series_lag_agreement(s2, pba)
        why, info = where_none(s2, pba)
        print(f'  -> 实际调用返回: {"None" if la is None else "有行"}')
        print(f'  -> 拦截点: {why}')
        if info:
            for kk, vv in info.items():
                print(f'       {kk:>20} = {vv}')

    # 对照: BZ 侧的实际调用 (用的是 dt_median_px, 序列来自 soton 缓存)
    print('\n' + '=' * 78)
    print(' 对照: BZ 侧同一函数的调用条件')
    print('=' * 78)
    soton = os.path.join(HERE, 'data', '_v13_cache', 'bz_soton.npz')
    zs = np.load(soton, allow_pickle=True)
    sk = [f for f in zs.files if f.startswith('series_')]
    print(f'soton 缓存里的 series_ 键: {sk}')
    for f in sk:
        arr = np.asarray(zs[f])
        print(f'  {f}: shape={arr.shape}, x 范围 [{arr[:, 0].min():.1f}, '
              f'{arr[:, 0].max():.1f}]')


if __name__ == '__main__':
    main()
