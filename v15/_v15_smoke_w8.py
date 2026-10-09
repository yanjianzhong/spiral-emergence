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
v15·W8 冒烟测试 —— 只读, 0 s 全流程。

复刻 `tier3_structural_ladder` 的 Reading 分支 (含 trim_scan / 自相关交叉检验)
逐字执行, 核对:
  1. 三条 Reading 序列现在**都**产出自相关滞后行 (T3-4b 的通路打通)
  2. 三条的判定是什么 (预期: 未确认 —— 极大不在 T)
  3. BZ 侧调用走默认参数, 返回的 ac 值与改动前**逐位相同**

第 3 条用一个"旧实现"复本来比对, 而不是靠眼睛看。
"""
import json
import os
import numpy as np

import spiral_metric_v15 as M

HERE = os.path.dirname(os.path.abspath(__file__))
READING = os.path.join(HERE, 'data', '_v13_cache', 'bz_reading.npz')
SOTON = os.path.join(HERE, 'data', '_v13_cache', 'bz_soton.npz')


def _old_series_lag_agreement(series, period_px):
    """改动前的实现, 逐字复刻, 只用来核对 BZ 侧读数没动。"""
    s = np.asarray(series, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] < 2 or s.shape[0] < 16:
        return None
    x, y = s[:, 0], s[:, 1]
    lag = int(round(float(period_px)))
    if lag < 2:
        return None
    m = x > 3.0 * lag
    if int(m.sum()) < 12 * lag:
        return None
    yy = y[m]
    idx = np.arange(len(yy))
    yy = yy - np.polyval(np.polyfit(idx, yy, 1), idx)

    def ac(L):
        if L <= 0 or L >= len(yy) - 2:
            return float('nan')
        a, b = yy[:-L], yy[L:]
        if a.std() <= 0 or b.std() <= 0:
            return float('nan')
        return float(np.corrcoef(a, b)[0, 1])

    lags = {'T/2': max(1, lag // 2), 'T': lag, '1.5T': int(round(1.5 * lag)),
            '2T': 2 * lag}
    acs = {k: ac(v) for k, v in lags.items()}
    ac_T, ac_half = acs['T'], acs['T/2']
    if not np.isfinite(ac_T) or not np.isfinite(ac_half):
        return None
    ratio = float(ac_T / ac_half) if ac_half > 0 else float('inf')
    finite = {k: v for k, v in acs.items() if np.isfinite(v)}
    best = max(finite, key=finite.get)
    if best == 'T':
        verdict = '独立确认 (滞后 T 处是严格自相关极大)'
    elif ac_T >= 0.95 * finite[best]:
        verdict = f'弱确认 (极大在 {best}, 但 ac(T) 与它相差 <5%)'
    else:
        verdict = f'未确认 (极大在 {best}, 不在 T)'
    return {'ac': acs, 'ac_T': ac_T, 'ac_half': float(ac_half),
            'ratio_T_over_half': ratio, 'lag_best': best, 'lag_T_px': lag,
            'verdict': verdict, 'confirmed': bool(best == 'T'),
            'n_used': int(m.sum())}


def main():
    ok = True

    # ---------------- 1. Reading 侧: 通路是否打通 ----------------
    print('=' * 84)
    print(' W8 冒烟 1/2: Reading 侧自相关交叉检验 (新调用点)')
    print('=' * 84)
    zr = np.load(READING, allow_pickle=True)
    lum_keys = [k for k in zr.files if k.startswith('lum_')]
    per, lag_rows_read = {}, {}
    for k in lum_keys:
        y_read = np.asarray(zr[k], dtype=float)
        sp = M._series_period(y_read)
        if sp is None:
            continue
        nm = k[4:]
        per[nm] = sp
        nd = M._leading_dark(y_read)
        # ---- 与产品代码 Reading 分支的调用点逐字一致 ----
        _s2 = np.column_stack([np.arange(nd, len(y_read), dtype=float),
                               y_read[nd:]])
        _la = M._series_lag_agreement(_s2, sp['period_frames_band_argmax'],
                                      skip_periods=0.0, min_lag_cycles=4.0)
        if _la is not None:
            lag_rows_read[nm] = _la
        print(f'  {nm:>32}: nd={nd:>3}  行={"有" if _la is not None else "None"}'
              + (f'  n_used={_la["n_used"]}  周期数='
                 f'{_la["n_used"]/_la["lag_T_px"]:.2f}' if _la else ''))

    print(f'\n  T3-4b 判据 bool(_lagr) = bool({len(lag_rows_read)} 行) -> '
          f'{"通过" if lag_rows_read else "未通过"}')
    if len(lag_rows_read) != len(per):
        ok = False
        print(f'  !! 期望 {len(per)} 行, 实得 {len(lag_rows_read)} 行')

    print('\n  逐条判定与实测:')
    for nm, v in sorted(lag_rows_read.items()):
        print(f'    {nm:>32}: {v["verdict"]}')
        print(f'        ac(T/2)={v["ac"]["T/2"]:+.4f}  ac(T)={v["ac_T"]:+.4f}  '
              f'ac(1.5T)={v["ac"]["1.5T"]:+.4f}  ac(2T)={v["ac"]["2T"]:+.4f}  '
              f'比={v["ratio_T_over_half"]:+.3f}')
    n_conf = sum(1 for v in lag_rows_read.values() if v['confirmed'])
    print(f'\n  确认周期份数 = {n_conf}/{len(lag_rows_read)} '
          f'(T3-4b 守的是"检验做过", 确认份数不是它的判据)')

    # ---------------- 2. BZ 侧: 读数是否逐位不变 ----------------
    print('\n' + '=' * 84)
    print(' W8 冒烟 2/2: BZ 侧读数与改动前逐位比对')
    print('=' * 84)
    zs = np.load(SOTON, allow_pickle=True)
    meta_s = json.loads(str(zs['meta']))
    sec_per_px = 1.0 / (meta_s['calib']['px_per_frame']
                        / meta_s['calib']['sec_per_frame'])
    n_cmp, max_dev, n_rows = 0, 0.0, 0
    # 逐字复刻产品代码 :2381-2395: nm 取自 st_* 键, dt_median_px 来自
    # **peaks_{nm}** (作者标注的波峰列), 序列才是 series_{nm}。两者不是一回事。
    for nm in [k[3:] for k in zs.files if k.startswith('st_')]:
        if f'peaks_{nm}' not in zs.files or f'series_{nm}' not in zs.files:
            continue
        ps = M._peak_period_stats(zs[f'peaks_{nm}'], sec_per_px)
        if ps is None:
            print(f'  {nm}: _peak_period_stats 返回 None, 产品代码里也会跳过')
            continue
        arr = np.asarray(zs[f'series_{nm}'])
        lag_px = ps['dt_median_px']
        new = M._series_lag_agreement(arr, lag_px)
        old = _old_series_lag_agreement(arr, lag_px)
        same_none = (new is None) == (old is None)
        print(f'  {nm}: n={arr.shape[0]:>6}  lag={lag_px:.1f}px  '
              f'新={"None" if new is None else "有行"}  '
              f'旧={"None" if old is None else "有行"}  '
              f'{"一致" if same_none else "!! 不一致"}')
        if not same_none:
            ok = False
            continue
        if new is None:
            continue
        n_rows += 1
        for kk in ('ac_T', 'ac_half', 'ratio_T_over_half', 'n_used'):
            if kk == 'n_used':
                dev = abs(new[kk] - old[kk])
            else:
                dev = abs(float(new[kk]) - float(old[kk]))
            max_dev = max(max_dev, dev)
            n_cmp += 1
        for kk in ('T/2', 'T', '1.5T', '2T'):
            dev = abs(float(new['ac'][kk]) - float(old['ac'][kk]))
            max_dev = max(max_dev, dev)
            n_cmp += 1
        if new['verdict'] != old['verdict']:
            ok = False
            print(f'    !! verdict 变了: {old["verdict"]} -> {new["verdict"]}')
    print(f'\n  比对 {n_rows} 条 BZ 序列, {n_cmp} 个标量, '
          f'最大偏差 = {max_dev:.3e}')
    if max_dev != 0.0:
        ok = False
        print('  !! BZ 侧读数发生了变化, 应逐位相同')

    print('\n' + '=' * 84)
    print(f' W8 冒烟结论: {"全部通过" if ok else "有未通过项 (见上)"}')
    print('=' * 84)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
