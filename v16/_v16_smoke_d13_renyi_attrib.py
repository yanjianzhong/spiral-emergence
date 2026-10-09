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
v16.5 · 工作包 D13 —— Rényi 族的失败，真的是「L=16 的有限尺寸压不住」吗？
[_VERSION_TAG = 'v16-smoke-d13-renyi-attrib-1']

**靶心（逐字，`v16.5_plan.md` §3.1）**
  `_model/checks.py:163-164` 登记:
    「alpha >= 2 后系统性飘高: Rényi 熵的次领头修正比 alpha=1 大得多,
      **L=16 的有限尺寸压不住**。」
  这是一条**有限尺寸归因** ⇒ 可用更大的 L 直接证伪。

**pre 的误解（本脚本要澄清的）**
  `v16.5_pre.md` A2 说「不再用单一 n 值做拟合，而是扫描多个 Rényi 指数 n」
  —— 这件事**已经实现**：`renyi_alpha_scan` 就是**多 alpha × 多 n**
  (`RENYI_ALPHAS` 七个 alpha、`RENYI_NS` 七个 n)。pre 描述的"新策略"正是现存代码。

**本脚本不做的事（预登记，`v16.5_plan.md` §3.3）**
  1. **不改门槛** —— 用代码自己的 `RENYI_DEV_TRUE_MAX=0.05` / `RENYI_STD_MAX=0.02`
     (`checks.py:91,94`)。pre 的「偏差 < 1%」比代码严 5 倍，**是自己加的门槛，不采用**。
  2. **不动分母** —— 即使新 L 上 `validated=True`，生产登记仍是 L=16 的 `False`
     (`expect_pass=False` 诊断项)。规模三数 78 / 57 / 13 不动（DP-17 默认）。
  3. **不进 runner**（DP-14 默认）。

======================================================================
**预登记（跑之前写死；判据与落支不许事后改）**
======================================================================

口径: `J=1.0, h=1.0`（`F1_L`/`F1_C_TRUE` 同格点）。`L` 是 `renyi_alpha_scan` 的**入参**。

阶段 A · **先复现**（D9 教训：先复现登记值再扫参数，不许跳）
  `renyi_alpha_scan(exact_ground_state(16,1,1)[1], L=16)` 必须复现 `checks.py:159-160`:
    逐 alpha 的 c = 0.53819 / 0.50838 / 0.51114 / 0.53184 / 0.56138 / 0.56820 / 0.55991
    mean = 0.53986, std = 0.02252, dev = 7.97%
  容差: 逐 alpha 5e-5（5 位小数）；dev 1e-4（3 位有效）。
  **不复现 ⇒ 落 `R-复现失败`，`EXIT=3`，不进阶段 B。**

阶段 B · 扩 L
  L=18、L=20 依次跑（ED 侧无 16 上限；本次不碰 MERA）。
  单点用时 > T_EXT_MAX 秒 ⇒ 停在该点并**如实登记**。L=22 不做（全仓无锚）。

落支（**按此顺序判，先中先得**）:
  `R-复现失败`  阶段 A 不过 ⇒ 不报任何归因结论
  `R-不收缩`    `dev_new > RENYI_DEV_TRUE_MAX (0.05)` 在新 L 上
                ⇒ 登记归因**错**：不是有限尺寸，是 alpha 依赖本身
  `R-收缩`      `dev_new <= 0.05` 且 `std_new < RENYI_STD_MAX (0.02)`
                ⇒ 归因成立，但**本版只登记、不改分母**
  `R-部分`      `dev_new < dev_16` 但 `std_new >= 0.02` ⇒ 只登记
  `R-其他`      以上都不中 ⇒ 如实登记，不强行归类

**升级预期：无。L2 维持 A。**
"""

import os
import sys
import time

_T_PROC = time.time()   # ← 必须早于门面 import

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np      # noqa: E402

import spiral_model_v16 as G   # noqa: E402

_T_IMPORT = time.time()
_T0 = time.time()

# ── 登记值（逐字抄自 `_model/checks.py:157-160`）─────────────────────
L_REF = 16
REF_ALPHAS = (0.50, 0.75, 1.50, 2.00, 3.00, 5.00, 8.00)
REF_C = (0.53819, 0.50838, 0.51114, 0.53184, 0.56138, 0.56820, 0.55991)
REF_MEAN = 0.53986
REF_STD = 0.02252
REF_DEV = 0.0797

TOL_ALPHA = 5e-5
TOL_DEV = 1e-4
T_EXT_MAX = 300.0       # 秒；超过则停（plan §5.5 的停机规则）

L_EXT = (18, 20)


def main():
    print('=' * 76)
    print('v16.5 · D13 —— Rényi 失败归因「L=16 的有限尺寸压不住」检验')
    print('  J=1.0, h=1.0；不进 runner；不动分母')
    print('  代码自己的门槛: DEV_TRUE_MAX=%.2f, STD_MAX=%.2f'
          % (G.RENYI_DEV_TRUE_MAX, G.RENYI_STD_MAX))
    print('=' * 76)

    # ── 阶段 A：复现登记值 ────────────────────────────────────────────
    print('\n-- 阶段 A: 复现登记值 (L=16) --')
    t = time.perf_counter()
    _E0, gs16 = G.exact_ground_state(L_REF, 1.0, 1.0)
    ref = G.renyi_alpha_scan(gs16, L=L_REF)
    t_ref = time.perf_counter() - t
    got = {r['alpha']: r['c'] for r in ref['alphas']}
    dev_alpha = max(abs(got[a] - c) for a, c in zip(REF_ALPHAS, REF_C))
    dev_mean = abs(ref['mean'] - REF_MEAN)
    dev_std = abs(ref['std'] - REF_STD)
    dev_dev = abs(ref['dev_vs_true'] - REF_DEV)
    ok_a = (dev_alpha <= TOL_ALPHA and dev_mean <= TOL_ALPHA
            and dev_std <= TOL_ALPHA and dev_dev <= TOL_DEV)
    print('  复现自检（对 checks.py:159-160）:')
    print('    max|c(alpha) - 登记| = %.3e  (<= %g)  %s'
          % (dev_alpha, TOL_ALPHA, 'OK' if dev_alpha <= TOL_ALPHA else 'FAIL'))
    print('    |mean - 0.53986|     = %.3e  %s'
          % (dev_mean, 'OK' if dev_mean <= TOL_ALPHA else 'FAIL'))
    print('    |std  - 0.02252|     = %.3e  %s'
          % (dev_std, 'OK' if dev_std <= TOL_ALPHA else 'FAIL'))
    print('    |dev  - 0.0797 |     = %.3e  (<= %g)  %s'
          % (dev_dev, TOL_DEV, 'OK' if dev_dev <= TOL_DEV else 'FAIL'))
    if not ok_a:
        print('\n-- 落支 --')
        print('  落支 = **R-复现失败** ⇒ 不报任何归因结论，EXIT=3')
        print('\n用时：门面 import %.1f s + 计算 %.1f s = 总计 %.1f s'
              % (_T_IMPORT - _T_PROC, time.time() - _T0, time.time() - _T_PROC))
        print('=' * 76)
        return 3

    # ── 阶段 B：扩 L ─────────────────────────────────────────────────
    print('\n-- 阶段 B: 扩 L --')
    res = {L_REF: ref}
    for L in L_EXT:
        t = time.perf_counter()
        _e, gs = G.exact_ground_state(L, 1.0, 1.0)
        r = G.renyi_alpha_scan(gs, L=L)
        dt = time.perf_counter() - t
        res[L] = r
        print('  L=%d: 用时 %.2f s   mean=%.5f  std=%.5f  dev=%.4f%%  '
              'validated=%s'
              % (L, dt, r['mean'], r['std'], 100.0 * r['dev_vs_true'],
                 r['validated']))
        if dt > T_EXT_MAX:
            print('  => 用时 %.2f s > T_EXT_MAX=%.1f s ⇒ 停，不再往上' % (dt, T_EXT_MAX))
            break

    # ── 落支 ─────────────────────────────────────────────────────────
    new_Ls = [L for L in L_EXT if L in res]
    print('\n-- 逐 L 汇总 --')
    print('  %3s %10s %10s %10s %6s' % ('L', 'mean', 'std', 'dev%', 'valid'))
    for L in [L_REF] + new_Ls:
        r = res[L]
        print('  %3d %10.5f %10.5f %10.4f %6s'
              % (L, r['mean'], r['std'], 100.0 * r['dev_vs_true'],
                 r['validated']))

    dev16 = res[L_REF]['dev_vs_true']
    dev_new = min(res[L]['dev_vs_true'] for L in new_Ls)
    std_new = min(res[L]['std'] for L in new_Ls)
    print('\n  新 L 上最小的 dev = %.4f%%（L=16 是 %.4f%%）；最小的 std = %.5f'
          % (100.0 * dev_new, 100.0 * dev16, std_new))

    print('\n-- 落支 --')
    if dev_new > G.RENYI_DEV_TRUE_MAX:
        branch = 'R-不收缩'
        why = ('dev 在新 L 上仍 > %.2f ⇒ 登记归因「L=16 的有限尺寸压不住」**错**；'
               '真实原因是 alpha 依赖本身' % G.RENYI_DEV_TRUE_MAX)
    elif dev_new <= G.RENYI_DEV_TRUE_MAX and std_new < G.RENYI_STD_MAX:
        branch = 'R-收缩'
        why = ('dev 收到 <= %.2f 且 std < %.2f ⇒ 归因成立，'
               '**但本版只登记、不改分母**（DP-17）'
               % (G.RENYI_DEV_TRUE_MAX, G.RENYI_STD_MAX))
    elif dev_new < dev16:
        branch = 'R-部分'
        why = 'dev 收缩但 std 仍 >= %.2f ⇒ 只登记' % G.RENYI_STD_MAX
    else:
        branch = 'R-其他'
        why = '以上都不中 ⇒ 如实登记，不强行归类'
    print('  落支 = **%s**' % branch)
    print('  %s' % why)
    if branch == 'R-不收缩':
        print('  ⇒ 须并存订正 `checks.py:163-164` 的 docstring（原文一字不删，追加一行）')
    print('  **升级预期：无。L2 维持 A**；生产登记仍是 L=16 的 validated=False')
    print('\n[D13] 隔离自检：未调用 metric_* / 未碰 MERA；未动既有读数与分母。')
    print('用时：门面 import %.1f s + 计算 %.1f s = 总计 %.1f s'
          % (_T_IMPORT - _T_PROC, time.time() - _T0, time.time() - _T_PROC))
    print('=' * 76)
    return 0


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.exit(main())
