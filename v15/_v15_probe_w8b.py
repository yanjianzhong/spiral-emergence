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
v15·W8 侦察脚本 之二 —— 只读。

`_series_lag_agreement` 在 Reading 侧被 `12*lag` 拦下 (见 _v15_probe_w8.py)。
本脚本回答**下一个**问题: 若把这道门槛按统计量重新表述, 读数会长什么样?

**顺序很重要**: 先看 ac 的真实形状, 再定判据。绝不用"哪个判据能过"来倒推。

同时打印: 不同 skip 下的可用对数 / 周期数, 以及自相关在多个滞后上的取值。
"""
import os
import numpy as np

import spiral_metric_v15 as M

HERE = os.path.dirname(os.path.abspath(__file__))
READING = os.path.join(HERE, 'data', '_v13_cache', 'bz_reading.npz')


def ac_curve(y, lag, skip):
    """去线性趋势后, 在若干滞后上算自相关。skip 是掐掉的头部点数。"""
    yy = np.asarray(y, dtype=float)[skip:]
    idx = np.arange(len(yy), dtype=float)
    yy = yy - np.polyval(np.polyfit(idx, yy, 1), idx)

    def ac(L):
        if L <= 0 or L >= len(yy) - 2:
            return float('nan')
        a, b = yy[:-L], yy[L:]
        if a.std() <= 0 or b.std() <= 0:
            return float('nan')
        return float(np.corrcoef(a, b)[0, 1])

    out = {}
    for name, L in (('T/2', max(1, lag // 2)), ('T', lag),
                    ('1.5T', int(round(1.5 * lag))), ('2T', 2 * lag)):
        out[name] = (L, ac(L))
    return out, len(yy), len(yy) - lag


def main():
    zr = np.load(READING, allow_pickle=True)
    lum_keys = [k for k in zr.files if k.startswith('lum_')]
    print('=' * 92)
    print(' W8 侦察二: 放宽 skip 后, 自相关实际长什么样')
    print(' 说明: 只在下述若干 skip 下**看数**, 不在此处决定采用哪个')
    print('=' * 92)

    for k in lum_keys:
        y = np.asarray(zr[k], dtype=float)
        nm = k[4:]
        sp = M._series_period(y)
        lag = int(round(sp['period_frames_band_argmax']))
        nd = M._leading_dark(y)
        print('-' * 92)
        print(f'[{nm}]  n={len(y)}  lag(带内argmax)={lag} 帧  '
              f'头部暗帧 nd={nd} 帧')
        for skip_name, skip in (('0', 0),
                                (f'nd={nd}', int(nd)),
                                ('nd+16', int(nd) + 16),
                                ('3*lag', 3 * lag)):
            acs, n_used, pairs = ac_curve(y, lag, skip)
            if n_used < lag + 10:
                print(f'  skip={skip_name:>7}: 可用 {n_used} 点 < lag+10, 不可算')
                continue
            cells = '  '.join(f'{kk}={vv:+.4f}(L={L})' for kk, (L, vv) in acs.items())
            print(f'  skip={skip_name:>7}: 用 {n_used:>5} 点, T 处对数 {pairs:>5}, '
                  f'周期数 {n_used/lag:.2f}')
            print(f'              {cells}')

        # 该序列的**独立**周期数上限 (完全不掐)
        print(f'  -> 全长仅 {len(y)/lag:.2f} 个 lag 周期 '
              f'(n/lag = {len(y)}/{lag})')

    print('\n' + '=' * 92)
    print(' 对照: BZ 侧 (空间序列) 的同一组量')
    print('=' * 92)
    soton = os.path.join(HERE, 'data', '_v13_cache', 'bz_soton.npz')
    zs = np.load(soton, allow_pickle=True)
    for f in [x for x in zs.files if x.startswith('series_')]:
        arr = np.asarray(zs[f])
        print(f'  {f}: n={arr.shape[0]}, x∈[{arr[:,0].min():.0f}, '
              f'{arr[:,0].max():.0f}]')


if __name__ == '__main__':
    main()
