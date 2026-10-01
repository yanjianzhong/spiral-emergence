# -*- coding: utf-8 -*-
#
# Copyright (c) 2024-2026 Jianzhong Yan
# Licensed under the Apache License, Version 2.0 (the "License");
#
"""
v16.5 · 工作包 D12 —— 能谱族的「O(1/L) 有限尺寸修正」这条**归因**是真的吗？
[_VERSION_TAG = 'v16-smoke-d12-spectrum-attrib-1']

**靶心（逐字，`v16.5_plan.md` §2.1）**
  `_model/checks.py:632-634` 登记:
    「读数**不随 L 单调收敛** (L=8..16 落在 0.49894..0.50034, L=16 反而最差),
      残差是 **O(1/L) 有限尺寸修正**, 不是数值误差」。
  同时 `_runall_v16.3_r1.log:418` 登记「比值 …**单调趋 8**」。

**为什么值得查（【代码事实】）**
  实测 c_spec 在 L>=10 上**单调远离 0.5**，每 ΔL=2 约 -0.00033（见下表），
  这是**线性漂移**。O(1/L) 修正**应当随 L 衰减**，与数据不符。
  ⇒ 用外插**预测**未参与拟合的 L 点，可以证伪这条归因。

**本脚本不做的事（预登记，`v16.5_plan.md` §2.3）**
  1. **不做 v->infinity 外推** —— `checks.py:489-491` 逐字已登记「**不做**」。
  2. **不声称"测到 c=1/2"** —— `checks.py:490-491` 逐字。
  3. **不碰 MERA** —— `spectral_central_charge` 纯 ED (`tfi_periodic_sparse` + `eigsh`),
     本函数**没有** MERA 那条腿, `L_list` 默认就含 10/12/14 三个非 2 的幂。
  4. **不调用 `metric_spectral_central_charge`** —— 那会碰已计分读数。

**隔离声明**：不进 runner、不加守卫/指标 ⇒ 规模三数 78 / 27 / 13 不动（DP-14 默认）。

======================================================================
**预登记（跑之前写死；判据与落支不许事后改）**
======================================================================

口径: `J=1.0, h=1.0`（`spectral_central_charge` 的默认，与登记读数同格点）。

阶段 A · **先复现**（D9 教训：先复现登记值再扫参数，不许跳）
  `spectral_central_charge(L_list=(8,10,12,14,16))` 必须逐位复现日志:
    ratio  = 7.9231 / 7.9508 / 7.9658 / 7.9748 / 7.9807     (容差 5e-5 = 半个末位)
    c_spec = 0.50034 / 0.49994 / 0.49960 / 0.49928 / 0.49894  (同上)
    e_inf  = -1.2732445447（对解析 -4/pi 偏差 5.0e-06）
  **不复现 ⇒ 落 `F-复现失败`，`EXIT=3`，不进阶段 B。**

阶段 B · 扩 L（**L 上限由 DP-16 定**）
  先跑到 **L=18**（代码自带实测锚：`checks.py:482` 逐字「L=18 实测建 H 1.53s +
  eigsh 3.86s + 峰值 433MB」）。若 L=18 的 (建 H + eigsh) 用时 **<= T_EXT_MAX 秒**
  再跑 **L=20**（`_metric/extras.py:397` 逐字「L=18/20 实测均可」）；
  否则**停在 18 并如实登记**。**L=22 一律不做**（全仓无锚）。

阶段 C · 归因判据（**两个模型都在登记数据上拟合，再去预测未参与拟合的新 L 点**）
  拟合集 = 登记的 4 点 (L=10,12,14,16)。两个模型:
    `lin`:  c = a + b*L
    `inv`:  c = a + b/L
  对新 L 点算预测误差 `err_lin` / `err_inv`。

落支（**按此顺序判，先中先得**）:
  `F-复现失败`  阶段 A 不过 ⇒ 不报任何归因结论
  `F-非单调`    `ratio` 在新 L 上不再增大（打破「单调趋 8」）
  `F-线性`      `err_lin <= PRED_TOL` 且 `|c_new-0.5| > |c_16-0.5|`
                ⇒ 登记归因「O(1/L)」**错**，须按 DP-15 并存订正
                （原文一字不删，追加一行）
  `F-1/L`       `|c_new-0.5| < |c_16-0.5|`（且随 L 下降）
                ⇒ 登记归因成立 ⇒ **零结果，不改任何文字**
  `F-其他`      以上都不中（plan 未预登记的第四种组合）⇒ **如实登记，不强行归类**

**升级预期：无。** 无论落哪支，L2 维持 A —— 本任务判的是**归因真假**，
不是提高拟合质量（`checks.py:492-496` 已量化杠杆臂只宽 17%，不足以改善标度拟合）。
"""

import os
import sys
import time

_T_PROC = time.time()   # ← 必须早于门面 import：否则会漏掉 torch/quimb 的 ~10 s

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np      # noqa: E402

import spiral_model_v16 as G   # noqa: E402

_T_IMPORT = time.time()
_T0 = time.time()

# ── 登记值（逐字抄自 `_runall_v16.3_r1.log:412-416`）──────────────────
L_REF = (8, 10, 12, 14, 16)
REF_RATIO = (7.9231, 7.9508, 7.9658, 7.9748, 7.9807)
REF_C = (0.50034, 0.49994, 0.49960, 0.49928, 0.49894)
REF_E_INF = -1.2732445447
REF_E_INF_DEV = 5.0e-06

L_EXT_FIRST = 18
L_EXT_SECOND = 20
TOL_REPRO = 5e-5        # 日志只给 5 位小数 ⇒ 半个末位
TOL_REPRO_EINF = 5e-10  # e_inf 给了 10 位
T_EXT_MAX = 8.0         # 秒；新 L 单点 (建 H + eigsh) 超过就停，不硬跑
PRED_TOL = 1e-5         # 线性模型预测与实测之差（§2.4 预先写死）

# ── 拟合集：登记的 4 点，**不含**任何新点（这是外插检验的关键）──────
L_FIT = (10, 12, 14, 16)


def _peak_rss_mb():
    """峰值 RSS(MB)；psutil 不可用时退化为 None（不编数字）。"""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024.0 / 1024.0
    except Exception:
        return None


def _branch(rows_by_L, new_Ls):
    """按预登记顺序判落支。rows_by_L: {L: row}; new_Ls: 实际跑成的新 L 列表。"""
    if not new_Ls:
        return 'F-未扩展', 'L 扩展未执行（用时或资源超限）'

    # 1. ratio 单调性（靶: log:418「单调趋 8」）
    seq = [rows_by_L[L]['ratio'] for L in L_REF] + [rows_by_L[L]['ratio']
                                                    for L in new_Ls]
    mono = all(seq[i] < seq[i + 1] for i in range(len(seq) - 1))
    if not mono:
        return 'F-非单调', 'ratio 在新 L 上不再增大 ⇒「单调趋 8」这条登记被打破'

    # 2/3. 两模型在**登记 4 点**上拟合，预测新 L
    xf = np.array(L_FIT, dtype=float)
    cf = np.array([rows_by_L[L]['c_spec'] for L in L_FIT])
    lin = np.polyfit(xf, cf, 1)                    # c = a + b*L
    inv = np.polyfit(1.0 / xf, cf, 1)              # c = a + b/L
    xn = np.array(new_Ls, dtype=float)
    meas = np.array([rows_by_L[L]['c_spec'] for L in new_Ls])
    err_lin = float(np.max(np.abs(np.polyval(lin, xn) - meas)))
    err_inv = float(np.max(np.abs(np.polyval(inv, 1.0 / xn) - meas)))

    dev16 = abs(rows_by_L[16]['c_spec'] - 0.5)
    grows = all(abs(rows_by_L[L]['c_spec'] - 0.5) > dev16 for L in new_Ls)
    shrinks = all(abs(rows_by_L[L]['c_spec'] - 0.5) < dev16 for L in new_Ls)

    print('\n-- 阶段 C: 归因判据 --')
    print('  拟合集 = 登记 4 点 L=%s（新点**未参与拟合**）' % (L_FIT,))
    print('  线性模型 c=a+b*L : a=%.6e b=%.6e' % (lin[1], lin[0]))
    print('  倒数模型 c=a+b/L : a=%.6f b=%.6f' % (inv[1], inv[0]))
    for i, L in enumerate(new_Ls):
        print('    L=%2d  实测 c=%.6f  线性预测=%.6f  倒数预测=%.6f'
              % (L, meas[i], np.polyval(lin, L), np.polyval(inv, 1.0 / L)))
    print('  err_lin = %.3e   err_inv = %.3e   (PRED_TOL=%g)'
          % (err_lin, err_inv, PRED_TOL))
    print('  |c-0.5|: L=16 是 %.3e；新点%s'
          % (dev16, '全部增大' if grows else ('全部减小' if shrinks else '有增有减')))

    if err_lin <= PRED_TOL and grows:
        return 'F-线性', ('线性外插命中 ⇒ 登记归因「O(1/L)」**错**'
                          '（O(1/L) 应随 L 衰减，实测是线性漂移）')
    if shrinks:
        return 'F-1/L', '|c-0.5| 随 L 回落 ⇒ 登记归因成立，**零结果**'
    return 'F-其他', '两模型都不中，且 |c-0.5| 不单调回落 ⇒ 如实登记，不强行归类'


def main():
    print('=' * 76)
    print('v16.5 · D12 —— 能谱族的「O(1/L) 有限尺寸修正」归因检验')
    print('  J=1.0, h=1.0；纯 ED，**不碰 MERA**；不进 runner')
    print('=' * 76)

    # ── 阶段 A：复现登记值 ────────────────────────────────────────────
    print('\n-- 阶段 A: 复现登记值 (L=8..16) --')
    t = time.perf_counter()
    ref = G.spectral_central_charge(L_list=L_REF)
    t_ref = time.perf_counter() - t
    rows_by_L = {r['L']: r for r in ref['rows']}

    dev_ratio = max(abs(rows_by_L[L]['ratio'] - r)
                    for L, r in zip(L_REF, REF_RATIO))
    dev_c = max(abs(rows_by_L[L]['c_spec'] - c)
                for L, c in zip(L_REF, REF_C))
    dev_einf = abs(ref['e_inf'] - REF_E_INF)
    ok_a = (dev_ratio <= TOL_REPRO and dev_c <= TOL_REPRO
            and dev_einf <= TOL_REPRO_EINF)
    print('\n  复现自检（对 _runall_v16.3_r1.log:412-416）:')
    print('    max|ratio - 登记| = %.3e  (<= %g)  %s'
          % (dev_ratio, TOL_REPRO, 'OK' if dev_ratio <= TOL_REPRO else 'FAIL'))
    print('    max|c_spec - 登记| = %.3e  (<= %g)  %s'
          % (dev_c, TOL_REPRO, 'OK' if dev_c <= TOL_REPRO else 'FAIL'))
    print('    |e_inf - 登记|     = %.3e  (<= %g)  %s   (登记偏差 %.1e)'
          % (dev_einf, TOL_REPRO_EINF,
             'OK' if dev_einf <= TOL_REPRO_EINF else 'FAIL', ref['e_inf_dev']))
    if not ok_a:
        print('\n-- 落支 --')
        print('  落支 = **F-复现失败** ⇒ 不报任何归因结论，EXIT=3')
        print('  (D9 教训: 先复现登记值再扫参数，不许跳)')
        print('\n用时：门面 import %.1f s + 计算 %.1f s = 总计 %.1f s'
              % (_T_IMPORT - _T_PROC, time.time() - _T0, time.time() - _T_PROC))
        print('=' * 76)
        return 3

    # ── 阶段 B：扩 L（用时门控）──────────────────────────────────────
    print('\n-- 阶段 B: 扩 L --')
    new_Ls = []
    for L in (L_EXT_FIRST, L_EXT_SECOND):
        t = time.perf_counter()
        ext = G.spectral_central_charge(L_list=(L,))
        dt = time.perf_counter() - t
        rss = _peak_rss_mb()
        rows_by_L[L] = ext['rows'][0]
        new_Ls.append(L)
        print('  L=%d: 建 H + eigsh 用时 %.2f s%s   ratio=%.4f  c_spec=%.5f'
              % (L, dt, '' if rss is None else ('  峰值 RSS %.0f MB' % rss),
                 rows_by_L[L]['ratio'], rows_by_L[L]['c_spec']))
        if dt > T_EXT_MAX:
            print('  => 用时 %.2f s > T_EXT_MAX=%.1f s ⇒ **停在 L=%d，不再往上**'
                  % (dt, T_EXT_MAX, L))
            break
    print('  实际扩展到的 L: %s（L=22 一律不做：全仓无锚）' % (new_Ls,))

    # ── 阶段 C + 落支 ────────────────────────────────────────────────
    branch, why = _branch(rows_by_L, new_Ls)

    print('\n-- 逐 L 汇总（登记点 + 新点）--')
    print('  %3s %10s %10s %12s' % ('L', 'ratio', 'c_spec', '|c-0.5|'))
    for L in list(L_REF) + new_Ls:
        r = rows_by_L[L]
        print('  %3d %10.4f %10.5f %12.3e'
              % (L, r['ratio'], r['c_spec'], abs(r['c_spec'] - 0.5)))

    print('\n-- 落支 --')
    print('  落支 = **%s**' % branch)
    print('  %s' % why)
    if branch == 'F-线性':
        print('  ⇒ 按 DP-15 **并存订正** `checks.py:632-634` 的 docstring')
        print('     （原文一字不删，追加一行；逐行等长替换，不移位）')
    elif branch == 'F-1/L':
        print('  ⇒ 登记归因成立，**不改任何文字**（零结果如实登记）')
    print('  **升级预期：无。L2 维持 A**（本任务判归因，不提高拟合质量）')
    print('\n[D12] 隔离自检：未调用 metric_spectral_central_charge / 未碰 MERA；'
          '未动既有读数。')
    print('用时：门面 import %.1f s + 计算 %.1f s = 总计 %.1f s'
          % (_T_IMPORT - _T_PROC, time.time() - _T0, time.time() - _T_PROC))
    print('=' * 76)
    return 0


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.exit(main())
