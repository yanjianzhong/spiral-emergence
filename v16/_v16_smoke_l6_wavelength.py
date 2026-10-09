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
v16 · S1 冒烟 —— L6 斑图波长: 独立估计量互核 + 分辨率/种子 2×2 + 第三估计量
[_VERSION_TAG = 'v16-smoke-l6-lambda-2']

**v2 (v16.3 · §7.4 S4 / 方向 D3)**: 新增 **D 段** —— 第三个估计量 (环形 ACF 首个
过零点) + 三估计量对照 + **落盘 JSON**。**A/B/C 三段一行未改** (§4.3 明令):
A 段的 `SPACING_LOGGED = 0.1782` 复现断言与 C 段的 2×2 读数**一个都不动**。
本段同样**不加守卫、不动分母**, 且**禁止"多数票"** —— 不许拿第三个去裁定前两个谁对。

**这是 v16.2_plan.md §5.1 的 S1** (v2 追加 v16.3_plan.md §4 的 D3), 不进 runner
(`_v16_run_all.py`), 不动任何登记值, 不碰生产路径。
真调 `stage6_life` / `derive_L6_grid` / `_char_scale_checked`。

要回答的问题 (背景见 plan 订正②):
  `stage6_life` 返回的 `spacing` (`stage67.py:362`) **至今没有任何守卫读它** ——
  `_v16_run.log:166` 印过 `斑图波长~0.1782`, 但那只是个**被打印的数**。
  本冒烟要把它变成**被两个独立估计量夹住的量**, 并分离一个此前未被识别的混杂因子。

四段:
  A. **生产复现**(自校验, 必须先过): 用与 `_model/main.py` **逐字相同**的调用跑一次,
     `spacing` 必须复现 `0.1782` (`_v16_run.log:166`, 打印精度 4 位)。
     ⚠️ 不过则整个冒烟的结论作废 —— 那说明测的不是生产那个量 (A 段"同方程不同盒子"
     栽过的正是这种坑)。同时**实测** `stage6_life` 的单次用时: 该用时从未被单独登记过。
  B. **两个独立估计量**:
       甲 = `spacing / h`  —— 沿 x 行的零交叉计数 (`stage67.py:360-362`, 域长单位)
       乙 = `_char_scale_checked(v - v̄, box=N, deconv=False)[0]` —— Δ²(k) 谱峰 (格)
     单位口径 (**本次逐行核过, 这是 pre.md 要求"显式定死"的那一条**):
       `_kgrids(shape, box)` 取 `d = box/n` (`registry.py:322`) ⇒ box 传格数时 k 的单位
       是 rad/格; `tier3_desc.py:130` 传的正是 `box = float(max(b.shape))` ⇒ **乙 的单位是格**。
       而 `spacing = 2L/zc` 是**域长**单位 ⇒ 甲 = `spacing/h` (h = L/N) 才可比。
      `deconv=False` 与 `tier3_desc.py:137` 对**连续场**的选择一致 (`registry.py:392-393`:
       "模型侧是连续场, 不能除 —— 对连续场除一个 CIC 窗等于凭空注入一个错误的谱形")。
  C. **分辨率 × 种子 2×2 析因**。`stage67.py:342-346` 的初值是**按格点写的**固定 4 格方块,
     其**物理**边长 `4h = 4*L_domain/N` 随 N 变 ⇒ 换 N 同时换掉了分辨率和种子尺寸,
     这是一个**混杂因子**, 与 A 段当年栽过的"盒子大小 vs 参数相区"同型。
     2×2 = {N=36, 48} × {种子格点固定, 物理固定} ⇒ 把两者分开。
     扫描**只在本冒烟里另起调用**, 不改 `stage6_life` 的初值规则 (那会改生产读数)。
  D. **(v2 新增 · v16.3 §4 D3) 第三个估计量**: 丙 = **环形自相关的首个过零点**。
     **硬约束**(§4.2, 已锚定): 乙的壳边界在物理 k 空间上是 `k = j*2*pi/L`, 与 N 无关
     ⇒ "λ乙 的物理值与 N 无关"是**构造使然的恒等式, 不是证据**。第三估计量**若也从
     FFT 壳取峰, 会继承同一个恒等式** ⇒ 丙 **必须来自不依赖壳量化的量**。
     故丙用**环形直接求和**算 ACF (`np.roll` 逐滞后点积), **不走 FFT**;
     定义域本就是周期的 (PDE 用 `np.roll` 离散), 环形 ACF 是**正确对象**, 不是权宜。
     正弦约定: 对波长 λ 的正弦, `ACF ∝ cos(2*pi*m/λ)`, 首个过零点在 `m = λ/4`
     ⇒ **丙 = 4*m0** (格; 换域长乘 `h`)。
     **预登记判据 (§4.3)**: 令 `r甲 = |丙-甲|/甲`, `r乙 = |丙-乙|/乙`
     (全部换算到**域长**再比, 见 C 段的单位警告)。
       * **T1**: `min(r甲,r乙) <= 0.5 * max(r甲,r乙)` (至少 2:1 分离) ⇒ **只登记丙站哪边**;
       * **T2**: 否则 (含无过零点 => 不可判) ⇒ 登记「**三个估计量互不一致
         ⇒ λ 相关的任何结论作废**」—— **这才是这条方向真正的产出**。
     **禁止**: ① 多数票 (不许拿丙裁定甲/乙谁对); ② 加守卫 (任何"给 `lambda_target`
     比值补守卫"都是为通过率造指标); ③ 改 A/B/C 段既有断言。
     顺带补订正 5 的洞: 本段把三估计量结果**写进新文件**
     `v16/output/l6_lambda_three_estimators.json` (**不覆盖任何已登记产物**)。
     ⚠️ 路径口径: 计划 §4.2 逐字写的是 `v16/output/`; 仓库既有产物目录是
     `v16/result/` (`_model/core.py:53` 的 `_OUTPUT_DIR`)。本冒烟**按计划新开
     `v16/output/`**, 不往 `result/` 里塞非登记产物 (该目录由 runner 消费)。

诚实边界 (写在这里, 不许在报告里省):
  * 本冒烟**不判定** L6 的物理主张。`V8 生命斑图涌现` 仍是 L6 七条守卫里**唯一**扛
    物理主张的那一条 (audit `:101`); 这里做的是**测量质量**, 不是物理判定。
  * **不预设通过阈值**。仓库先例 `_v16_run.log:610` 的同类互核结果是 46.6% 的**系统性
    不一致** ⇒ 本项事先不知道会不会对上。先跑出实际偏差, 再由 DP-2 决定它够不够格当判据。
  * `spacing` 的估计量是**沿 x 行的零交叉计数**, 对迷宫/斑点斑图, 逐行散布可能很大;
    故一并报 `zc` 的**行间标准误**, 否则"λ 变了"与"估计量噪声大"分不开。

退出码: 0 = 三段都完成且 A 段复现成功; 3 = **A 段复现失败**(结论作废, 需查);
        2 = 本脚本自身错误。

用法:
    python v16/_v16_smoke_l6_wavelength.py
"""

import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'v16'))

import numpy as np  # noqa: E402
import spiral_model_v16 as G  # noqa: E402
import spiral_metric_v16 as MM  # noqa: E402

_VERSION_TAG = 'v16-smoke-l6-lambda-2'   # v2: 追加 D 段 (v16.3 §4 D3); A/B/C 未改

# A 段的自校验靶: `_v16_run.log:166` 逐字 "斑图波长~0.1782 (域长 0.5)" (打印精度 .4f)
SPACING_LOGGED = 0.1782
SPACING_TOL = 5e-5            # .4f 的半个末位

# C 段: 物理固定种子所锚的参考边长 = 生产配置 (N=36) 下 4 格的物理尺寸
_SEED_REF_CELLS = 4

# D 段 (v2): 落盘目标 —— 计划 §4.2 逐字指定的 `v16/output/`, **新文件**
_OUT_JSON = os.path.join(_ROOT, 'v16', 'output', 'l6_lambda_three_estimators.json')

_FAILS = []


def check(label, cond, detail=''):
    tag = 'PASS' if cond else '**FAIL**'
    print(f"    [{tag}] {label}" + (f"  —— {detail}" if detail else ''))
    if not cond:
        _FAILS.append(label)


# ---------------------------------------------------------------------------
# 与 stage6_life 的**逐字复刻**, 唯一区别是把种子边长参数化。
# 复刻的合法性不靠"看着像", 靠 A 段与真函数**逐位比对**来证 (见 §A2)。
# ---------------------------------------------------------------------------
def rd_run(N, L, seed_cells, F, k, steps, dt, Du, Dv):
    h = L / N
    stab = dt * Du / h ** 2
    if stab > 0.25:
        return {'ok': False, 'stab': stab, 'h': h}

    u = np.ones((N, N))
    v = np.zeros((N, N))
    c = N // 2
    lo = c - seed_cells // 2
    u[lo:lo + seed_cells, lo:lo + seed_cells] = 0.5
    v[lo:lo + seed_cells, lo:lo + seed_cells] = 0.25
    for _ in range(steps):
        lu = (np.roll(u, 1, 0) + np.roll(u, -1, 0) +
              np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4 * u) / h ** 2
        lv = (np.roll(v, 1, 0) + np.roll(v, -1, 0) +
              np.roll(v, 1, 1) + np.roll(v, -1, 1) - 4 * v) / h ** 2
        uv2 = u * v ** 2
        u = u + dt * (Du * lu - uv2 + F * (1 - u))
        v = v + dt * (Dv * lv + uv2 - (F + k) * v)

    finite = bool(np.isfinite(u).all() and np.isfinite(v).all())
    return {'ok': True, 'u': u, 'v': v, 'h': h, 'stab': stab, 'finite': finite,
            'seed_cells': seed_cells, 'seed_phys': seed_cells * h}


def spacing_and_se(v, N, L):
    """与 `stage67.py:360-362` 同款的零交叉估计量, 外加行间标准误。"""
    zcs = np.array([len(np.where(np.diff(np.sign(v[i] - v[i].mean())))[0])
                    for i in range(N)], dtype=float)
    zc = float(zcs.mean())
    se = float(zcs.std(ddof=1) / np.sqrt(N)) if N > 1 else 0.0
    spacing = float(2.0 * L / zc) if zc > 0 else float('inf')
    # λ 与 zc 成反比 ⇒ 相对误差直接传递
    rel_se = float(se / zc) if zc > 0 else float('nan')
    return spacing, zc, se, rel_se


def spectral_lambda(v, N):
    """独立估计量乙: Δ²(k) 谱峰波长, 单位格。"""
    d = v - v.mean()          # 去掉 k=0 直流分量 (Δ²=k²P 在 k=0 处为 0, 故不改判据)
    lam, pk, valid = MM._char_scale_checked(d, float(N), deconv=False)
    i = int(np.argmax(MM._delta2(pk['k'], pk['P'], 2))) if len(pk['k']) else -1
    k_pk = float(pk['k'][i]) if i >= 0 else float('nan')
    k_fund = float(pk['k_fund']) if len(pk['k']) else float('nan')
    return float(lam), bool(valid), k_pk, k_fund


# ---------------------------------------------------------------------------
# D 段 (v2 新增) · 第三估计量丙: **环形自相关的首个过零点**
# 为什么必须是这个: 乙的壳边界在物理 k 空间里是 `k = j*2*pi/L` (与 N 无关),
# 那是 FFT 盒子的构造使然 ⇒ 任何"从壳取峰"的估计量都继承该恒等式, 当不了独立证据。
# 本函数**不走 FFT**: 逐滞后直接点积 (环形, `np.roll`) ⇒ 与壳无关。
# 定义域是周期的 (PDE 用 `np.roll` 离散) ⇒ 环形自相关是正确对象。
# 口径与甲对齐: 场取**逐行/逐列去均值后的起伏** (`v[i]-v[i].mean()`, 同 `spacing_and_se`)。
# ---------------------------------------------------------------------------
def acf_first_zero(v, N, max_lag=None):
    """
    返回 (m0, m_min, r):
      m0    = 首个过零点 (r 由正变非正) 的滞后, 格; 无过零 => None
      m_min = 首个局部极小的滞后, 格; 无 => None
      r     = 归一化 ACF, r[0] = 1, 长度 max_lag+1
    行、列两个方向各算一条再平均 —— 只用行会与甲共用一个方向的偏差。
    """
    M = max_lag if max_lag is not None else N // 2
    acc = np.zeros(M + 1)
    cnt = 0
    for axis in (0, 1):
        d = v - v.mean(axis=axis, keepdims=True)
        for i in range(N):
            x = d[i, :] if axis == 0 else d[:, i]
            nrm = float(np.dot(x, x))
            if nrm <= 0.0:
                continue
            for m in range(M + 1):
                acc[m] += float(np.dot(x, np.roll(x, -m))) / nrm
            cnt += 1
    if cnt == 0:
        return None, None, None
    r = acc / cnt
    m0 = next((m for m in range(1, M + 1) if r[m] <= 0.0), None)
    m_min = next((m for m in range(1, M)
                  if r[m] < r[m - 1] and r[m] <= r[m + 1]), None)
    return m0, m_min, r


def main():
    t0 = time.time()
    K = G.KNOBS
    print(f"=== {_VERSION_TAG} · L6 斑图波长: 独立估计量互核 + 分辨率/种子 2×2 "
          f"+ 第三估计量 (D 段, v16.3 §4 D3) ===")
    print(f"    口径: 真调 stage6_life / derive_L6_grid / _char_scale_checked; 不进 runner。")

    # --- 生产网格: N 由 derive_L6_grid 派生, 不写死 ---------------------------
    # xi_over_L 不参与 N 的公式 (core.py:485-486 逐字), 它只进台账; 传 1.0 是占位。
    l6 = G.derive_L6_grid(1.0, K)
    N0, L_dom = int(l6['N']), float(l6['L_domain'])
    print(f"\n--- 0 · 生产配置 (由 derive_L6_grid 派生, 非写死) ---")
    print(f"    N={N0} (N_req={l6['N_req']}, N_cap={l6['N_cap']}, 受限于 {l6['bound_by']}), "
          f"L_domain={L_dom}, h={l6['h']:.6f}, dt*Du/h^2={l6['stab']:.4f}")
    print(f"    lambda_target = L_domain/n_lambda = {l6['lambda_target']:.4f}"
          f"  (⚠️ 旋钮靶, 不是预言: n_lambda={K['n_lambda']}, "
          f"pts_per_wavelength={K['pts_per_wavelength']} 都是自由旋钮)")

    # =====================================================================
    print(f"\n--- A · 生产复现 (自校验, 必须先过) ---")
    ta = time.time()
    s6 = G.stage6_life(F=K['F'], k=K['k'], N=N0, L=L_dom,
                       dt=K['dt'], Du=K['Du'], Dv=K['Dv'])
    t_prod = time.time() - ta
    print(f"    → 单次 stage6_life 用时 **{t_prod:.2f} s** "
          f"(该用时此前**从未被单独登记过**, 本行即其首次实测)")

    if not s6.get('ok'):
        print(f"    **FAIL** stage6_life 拒绝运行: {s6.get('reason')}")
        return 2

    sp_prod = float(s6['spacing'])
    print(f"\n    A1 生产读数: spacing={sp_prod:.6f} (域长), h={s6['h']:.6f}, "
          f"contrast={s6['contrast']:.3f}, active={s6['active_frac']:.3f}, "
          f"emerged={s6['emerged']}")
    check(f"spacing 复现登记值 {SPACING_LOGGED} (打印精度 .4f)",
          abs(sp_prod - SPACING_LOGGED) <= SPACING_TOL,
          f"|{sp_prod:.6f} - {SPACING_LOGGED}| = {abs(sp_prod - SPACING_LOGGED):.2e}")

    # A2: 复刻件与真函数逐位比对 —— 这是 C 段用复刻件做扫描的**唯一许可证**
    rep = rd_run(N0, L_dom, _SEED_REF_CELLS, K['F'], K['k'], s6['steps'],
                 K['dt'], K['Du'], K['Dv'])
    dv = float(np.abs(rep['v'] - s6['v']).max())
    du = float(np.abs(rep['u'] - s6['u']).max())
    print(f"\n    A2 复刻件 vs 真函数 (同一 N={N0}, 同一 4 格种子): "
          f"max|dv|={dv:.3e}, max|du|={du:.3e}")
    check("复刻件与 stage6_life 逐位一致 (C 段扫描的许可证)",
          dv == 0.0 and du == 0.0,
          "逐位相同" if (dv == 0.0 and du == 0.0) else "**有差异 => C 段的复刻件不可用**")

    # =====================================================================
    print(f"\n--- B · 两个独立估计量 (N={N0}) ---")
    sp_a, zc, se_zc, rel_se = spacing_and_se(s6['v'], N0, L_dom)
    lam_a = sp_a / s6['h']
    lam_b, valid_b, k_pk, k_fund = spectral_lambda(s6['v'], N0)
    print(f"    甲 (零交叉): spacing={sp_a:.6f} 域长 = {lam_a:.4f} 格"
          f"   [zc={zc:.3f} +- {se_zc:.3f} 行间SE => 相对 {rel_se * 100:.2f}%]")
    print(f"    乙 (D2谱峰): {lam_b:.4f} 格   valid={valid_b}   "
          f"k_peak={k_pk:.6f} rad/格, k_fund={k_fund:.6f} => 峰在第 "
          f"{k_pk / k_fund:.2f} 壳 (壳宽 k_fund, **量化偏差的来源**)")
    dev = abs(lam_a - lam_b) / lam_b if lam_b > 0 else float('nan')
    print(f"    相对偏差 |甲-乙|/乙 = **{dev * 100:.2f}%**")
    print(f"    ⚠️ **不预设阈值**: 仓库先例 (`_v16_run.log:610`) 同类互核是 46.6% 的"
          f"系统性不一致 => 本项能不能对上事先不知道, 此行只登记实测。")
    print(f"    参照: 甲 / (pts_per_wavelength={K['pts_per_wavelength']}) = "
          f"{lam_a / K['pts_per_wavelength']:.4f}"
          f"  (~1 说明 N 就是按 12 格/波长构造的, 不是物理预言)")

    # =====================================================================
    print(f"\n--- C · 分辨率 x 种子 2x2 析因 ---")
    print(f"    混杂因子: 种子是**按格点**写的固定 4 格 => 物理边长 4h 随 N 变。")
    print(f"    物理固定 = 锚住生产 (N=36) 下的物理边长 "
          f"{_SEED_REF_CELLS * L_dom / 36:.6f}, 反解该 N 下的格数。")

    rows = []
    for N in (36, 48):
        h = L_dom / N
        kk_phys = max(2, int(round((_SEED_REF_CELLS * L_dom / 36) / h)))
        for mode, kk in (('格点固定', _SEED_REF_CELLS), ('物理固定', kk_phys)):
            r = rd_run(N, L_dom, kk, K['F'], K['k'], s6['steps'],
                       K['dt'], K['Du'], K['Dv'])
            if not r.get('ok'):
                rows.append((N, mode, kk, None, None, None, None, None, None, r['stab']))
                continue
            sp, zc_, se_, rse = spacing_and_se(r['v'], N, L_dom)
            lam = sp / r['h']          # 格 —— ⚠️ 含一个 N 因子, **不可跨 N 直接比**
            lb, vd, _, _ = spectral_lambda(r['v'], N)
            lb_phys = lb * r['h']
            contrast = float(r['v'].max() - r['v'].min())
            act = float((r['v'] > 0.1).mean())
            rows.append((N, mode, kk, lam, sp, lb, lb_phys, rse, contrast, r['stab']))
            print(f"    N={N} {mode}({kk}格, 物理边长 {kk * r['h']:.5f}): "
                  f"λ甲={sp:.5f} 域长 ({lam:.4f} 格), λ乙={lb_phys:.5f} 域长 ({lb:.4f} 格), "
                  f"zc 相对SE={rse * 100:.2f}%, contrast={contrast:.3f}, "
                  f"dt*Du/h^2={r['stab']:.4f}")

    print(f"\n    {'N':>4s} {'种子':>10s} {'格数':>5s} {'λ甲(域长)':>12s} {'λ乙(域长)':>12s} "
          f"{'λ甲(格)':>10s} {'zc相对SE':>10s} {'contrast':>9s}")
    for N, mode, kk, lam, sp, lb, lbp, rse, ct, stab in rows:
        if sp is None:
            print(f"    {N:4d} {mode:>10s} {kk:5d}   拒绝运行 (dt*Du/h^2={stab:.4f})")
        else:
            print(f"    {N:4d} {mode:>10s} {kk:5d} {sp:12.5f} {lbp:12.5f} "
                  f"{lam:10.4f} {rse * 100:9.2f}% {ct:9.3f}")

    by = {(N, m): (sp, lbp, rse)
          for N, m, kk, lam, sp, lb, lbp, rse, ct, st in rows}
    kk48p = [kk for N, m, kk, *_ in rows if N == 48 and m == '物理固定']
    g36c, g36p = by.get((36, '格点固定')), by.get((36, '物理固定'))
    g48c, g48p = by.get((48, '格点固定')), by.get((48, '物理固定'))
    print(f"\n    读法 (四格齐了才下结论, 缺一格不下):")
    print(f"      ⚠️ 比较**必须在物理单位(域长)上做** —— λ甲_格 = spacing/h 里天然含一个 N")
    print(f"         因子, 拿「格」跨 N 直接比会把单位换算读成物理效应(首版就栽在这里)。")
    print(f"      N=36 两格**必然重合**(物理固定的反解格数就是 4) => 该行不构成证据, "
          f"它只是复刻件的第二次自校验。")
    if None not in (g36c, g36p, g48c, g48p):
        sp36c, _, se36 = g36c
        sp48c, _, se48 = g48c
        sp48p, _, _ = g48p
        d_res = (sp48c - sp36c) / sp36c * 100.0     # 只动分辨率 (种子规则固定)
        d_seed = (sp48p - sp48c) / sp48c * 100.0    # 只动种子 (N 固定)
        se_comb = float(np.hypot(se36, se48)) * 100.0
        print(f"      只动 N (36->48, 种子格点固定): spacing {sp36c:.6f} -> {sp48c:.6f}"
              f"  = {d_res:+.2f}%")
        print(f"      只动种子 (4 格->{kk48p[0] if kk48p else '?'} 格, N=48): "
              f"{sp48c:.6f} -> {sp48p:.6f}  = {d_seed:+.2f}%")
        print(f"      估计量自身噪声底 (两跑合并行间SE): {se_comb:.2f}%"
              f"  => |分辨率效应| / 噪声 = {abs(d_res) / se_comb:.2f}")
        if abs(d_res) > 2.0 * se_comb:
            print(f"      => 超出噪声 2 倍 ⇒ 可读出「**分辨率依赖**」")
        else:
            print(f"      => **在噪声底附近, 不能读作检测到分辨率依赖**;")
            print(f"         但同样**不能**说「分辨率无关」—— 那是把'未检测到'当成'无效应'。")
    else:
        print(f"      ⚠️ 四格未齐, 按计划**不下结论**。")
    print(f"\n      ⚠️ **λ乙 不能用来判分辨率无关性**: 它的壳边界在物理 k 空间里是 "
          f"k = j*2*pi/L, **与 N 无关**")
    print(f"         (FFT 的壳随盒子一起缩放) ⇒ λ乙_物理 必然 N 无关。那是**恒等式**, "
          f"不是证据; 拿它当「分辨率无关」的佐证就是假绿。")

    print(f"\n    诚实边界: 本段只检验『斑图波长由化学参数定, 不由几何定』这句话"
          f"(`_v16_run.log:169` / `main.py:261-262` 一直在印, **至今无守卫**)。"
          f"\n    它**不**判定 L6 的物理主张 —— `V8` 仍是那 7 条里唯一扛物理主张的守卫。")

    # =====================================================================
    # D 段 (v2 新增 · v16.3 §7.4 S4 / §4 方向 D3): 第三个估计量
    # =====================================================================
    print(f"\n--- D · 第三估计量: 环形 ACF 首个过零点 (v16.3 §4 D3, **避开壳量化**) ---")
    print(f"    为什么不能用 FFT 取峰: 壳边界在物理 k 空间是 k = j*2*pi/L, **与 N 无关**")
    print(f"    (FFT 盒随盒子一起缩放) => 任何'从壳取峰'的估计量都继承这个恒等式,")
    print(f"    当不了独立证据 (C 段末尾已记该陷阱)。故丙**不走 FFT**, 逐滞后直接点积。")

    m0, m_min, acf_r = acf_first_zero(s6['v'], N0)
    lam_c_cells = 4.0 * m0 if m0 is not None else float('nan')   # 正弦约定 λ = 4*m0
    lam_c_dom = lam_c_cells * s6['h']
    n_cross = int(np.sum(np.diff(np.sign(acf_r)) != 0)) if acf_r is not None else 0
    print(f"    ACF 诊断: 首个过零滞后 m0 = {m0 if m0 is not None else 'n/a'} 格, "
          f"首个极小 m_min = {m_min if m_min is not None else 'n/a'} 格, "
          f"滞后 <= {N0 // 2} 内符号翻转次数 = {n_cross}")
    if acf_r is not None:
        show = min(len(acf_r), 13)
        print(f"    r(m) 前 {show} 项 = " + ' '.join(f"{v:.3f}" for v in acf_r[:show]))
    print(f"    (m0 与 m_min 是**原始诊断量**, 不参与判据; 只有 丙 = 4*m0 是估计量)")
    if m0 is None:
        print(f"    ⚠️ 在 N/2 内**无过零点** => 丙 不可判 (照登, 不许换定义凑一个数出来)。")

    # 全部换算到**域长**再比 (C 段已记: λ甲_格 天然含一个 N 因子, 不可跨 N 直比)
    lam_a_dom = sp_a                      # 甲 本就是域长 (2L/zc)
    lam_b_dom = lam_b * s6['h']           # 乙 是格, 乘 h 换域长
    print(f"\n    三估计量 (统一到**域长**; 括号内为格):")
    print(f"      甲 零交叉计数 : {lam_a_dom:.6f}  ({lam_a:.4f} 格)   "
          f"zc={zc:.3f} +- {se_zc:.3f} => 相对SE {rel_se * 100:.2f}%")
    print(f"      乙 D2谱峰     : {lam_b_dom:.6f}  ({lam_b:.4f} 格)   "
          f"valid={valid_b}, k_peak/k_fund = {k_pk / k_fund:.2f} 壳")
    if m0 is None:
        print(f"      丙 ACF过零    : **n/a** (无过零点)")
    else:
        print(f"      丙 ACF过零    : {lam_c_dom:.6f}  ({lam_c_cells:.4f} 格)   "
              f"m0={m0} 格 (*4 是正弦约定, 见文件头)")
    print(f"      甲 vs 乙 相对偏差 = {dev * 100:.2f}%  (v16.2 已登记值)")

    verdict_d, r_a, r_b = '不可判', float('nan'), float('nan')
    if m0 is not None:
        r_a = abs(lam_c_dom - lam_a_dom) / lam_a_dom
        r_b = abs(lam_c_dom - lam_b_dom) / lam_b_dom
        sep = min(r_a, r_b) / max(r_a, r_b) if max(r_a, r_b) > 0 else float('nan')
        print(f"\n    T1/T2 判定 (口径见文件头, **先于数据写死**):")
        print(f"      r甲 = |丙-甲|/甲 = {r_a * 100:.2f}%,  r乙 = |丙-乙|/乙 = {r_b * 100:.2f}%")
        print(f"      分离度 min/max = {sep:.3f}  (门槛 <= 0.5, 即至少 2:1)")
        if sep <= 0.5:
            verdict_d = 'T1-甲' if r_a < r_b else 'T1-乙'
            near = '甲' if r_a < r_b else '乙'
            print(f"      => **T1 成立**: 丙 落在**{near}**一侧 (只登记它站哪边, 不加守卫)。")
        else:
            verdict_d = 'T2'
            print(f"      => **T2 成立**: 丙 **两侧都不挨** (分离不足 2:1) ⇒ 登记为")
            print(f"         「**三个估计量互不一致 => λ 相关的任何结论作废**」。")
        if min(r_a, r_b) < rel_se:
            print(f"      ⚠️ **弱读警告**: 较小偏差 {min(r_a, r_b) * 100:.2f}% **低于甲自身的行间SE**"
                  f" {rel_se * 100:.2f}%,")
            print(f"         故 T1 的「落在某一侧」须弱读为「与噪声不可分」,")
            print(f"         **不可**读成「已确认同源」。")
    # ---- 约定敏感性 (**事后追加**, 见下声明) --------------------------------
    # 首版按预登记约定 lambda = 4*m0 判出上方的落支。但 4*m0 只对**纯正弦**成立
    # (ACF ∝ cos(2*pi*m/lambda) => 首个过零在 lambda/4)。周期型信号的**标准**读法是
    # ACF 的**首个次级极大**位置 (ACF 与信号同周期)。两者对本场是否一致, 是一件事
    # **必须先查**的事: 若不一致, 落支就是约定造出来的, 不是数据给的。
    # ⚠️ **本诊断不参与判据**: 不改上方判词、不改阈值、不改任何预登记口径;
    #    它只回答「丙 的落支是不是约定产物」。**事后追加, 照实声明。**
    m2 = None
    if acf_r is not None:
        m2 = next((m for m in range(2, N0 // 2)
                   if acf_r[m] > acf_r[m - 1] and acf_r[m] >= acf_r[m + 1]), None)
    print(f"\n    · 约定敏感性 (事后追加的诊断, **不参与判据**):")
    if m2 is None:
        print(f"      ACF 在 N/2 内无次级极大 => 本诊断不可判, 不影响上方判词。")
    else:
        lam_c2 = float(m2) * s6['h']
        ra2 = abs(lam_c2 - lam_a_dom) / lam_a_dom
        rb2 = abs(lam_c2 - lam_b_dom) / lam_b_dom
        print(f"      同一份 ACF, 换用**标准**约定「首个次级极大」(m={m2} 格):")
        print(f"        λ丙' = {lam_c2:.6f} 域长 ({m2} 格)  vs  预登记约定 4*m0 = "
              f"{lam_c_dom:.6f} (={lam_c_dom / s6['h']:.0f} 格)")
        print(f"        r甲' = {ra2 * 100:.2f}%,  r乙' = {rb2 * 100:.2f}%  "
              f"=> 落在**{'甲' if ra2 < rb2 else '乙'}**一侧 (分离度 "
              f"{min(ra2, rb2) / max(ra2, rb2):.3f})")
        if m0 is None:
            print(f"      (预登记约定无过零点 => 无 m0 可比, 本诊断只作单侧登记。)")
        elif (ra2 < rb2) != (r_a < r_b):
            print(f"      ⚠️ **落支随约定翻转**: 预登记约定判 {verdict_d}, 标准约定判 "
                  f"**T1-{'甲' if ra2 < rb2 else '乙'}**。")
            print(f"         => 丙 **没有判别力** —— 它没在甲/乙之间裁决, 裁决的是**约定**。")
            print(f"         实质结论按 §4.3 的 T2 精神读: 「**三个估计量互不一致 =>")
            print(f"         λ 相关的任何结论作废**」, 尽管按预登记字面规则落在 T1。")
        else:
            print(f"      两约定同侧 => 落支**不是**约定产物, 上方判词可用。")

    print(f"\n    ⚠️ **禁止多数票** ( §4.3 明令): 本段**不**用丙去裁定甲与乙谁对;")
    print(f"       它只回答'丙站哪边'或'三者互不一致'。")
    print(f"    ⚠️ 本段**不加守卫、不动分母**: L6 仍是 C, `V8` 仍是唯一扛物理主张的守卫。")

    # --- 落盘 (补订正 5 的洞: 此前读数只在 scrollback 里) --------------------
    blob = {
        "version_tag": _VERSION_TAG,
        "source": {"script": "v16/_v16_smoke_l6_wavelength.py",
                   "section": "D (v16.3 §7.4 S4 / §4 D3)",
                   "n": int(N0), "h": float(s6['h']), "L_domain": float(L_dom)},
        "estimators": {
            "jia": {"lambda_domain": float(lam_a_dom), "lambda_cells": float(lam_a),
                    "zc": float(zc), "zc_se": float(se_zc), "zc_rel_se": float(rel_se)},
            "yi": {"lambda_domain": float(lam_b_dom), "lambda_cells": float(lam_b),
                   "valid": bool(valid_b), "k_peak": float(k_pk), "k_fund": float(k_fund),
                   "shell_index": float(k_pk / k_fund) if k_fund else None},
            "bing": {"lambda_domain": None if m0 is None else float(lam_c_dom),
                     "lambda_cells": None if m0 is None else float(lam_c_cells),
                     "m0_cells": m0, "m_min_cells": m_min, "sign_flips": int(n_cross)},
        },
        "cross_checks": {"jia_vs_yi_rel": float(dev),
                         "spacing_reproduced": float(sp_prod),
                         "spacing_logged": float(SPACING_LOGGED)},
        "adjudication": {"r_jia": None if m0 is None else float(r_a),
                         "r_yi": None if m0 is None else float(r_b),
                         "verdict": verdict_d},
        # 事后追加的诊断 (不参与判据): 同一份 ACF 换用「首个次级极大」约定再看一次。
        # 声明: 该项在首版 D 段输出之后追加; 它**不改** verdict 字段与任何阈值。
        "sensitivity_posthoc": {
            "added_after_first_d_run": True,
            "convention": "first_secondary_maximum (ACF 与信号同周期; 标准读法)",
            "m2_cells": m2,
            "lambda_domain": None if m2 is None else float(m2) * float(s6['h']),
            "r_jia": None if m2 is None else float(abs(float(m2) * s6['h'] - lam_a_dom) / lam_a_dom),
            "r_yi": None if m2 is None else float(abs(float(m2) * s6['h'] - lam_b_dom) / lam_b_dom),
            "verdict_flips": None if (m2 is None or m0 is None) else bool(
                (abs(float(m2) * s6['h'] - lam_a_dom) / lam_a_dom
                 < abs(float(m2) * s6['h'] - lam_b_dom) / lam_b_dom) != (r_a < r_b)),
            "note": "若 verdict_flips 为真, 则 D3 的实质产出应按 §4.3 T2 精神读: 三估计量互不一致。",
        },
        "caveats": {"no_majority_vote": True, "no_shell_quantization": True,
                    "no_guard": True, "no_denominator_change": True,
                    "lambda_b_physical_n_independent_is_identity": True},
    }
    os.makedirs(os.path.dirname(_OUT_JSON), exist_ok=True)
    with open(_OUT_JSON, 'w', encoding='utf-8') as fh:
        json.dump(blob, fh, ensure_ascii=False, indent=2)
    print(f"\n    落盘: {os.path.relpath(_OUT_JSON, _ROOT)}  "
          f"(**新文件**, 不覆盖任何已登记产物; 无时间戳字段以保逐字节可复现)")
    print(f"    ⚠️ 路径口径: 计划 §4.2 写 `v16/output/`; 仓库既有产物目录是 "
          f"`v16/result/` (core.py:53), 本冒烟**不往 result/ 里塞非登记产物**。")

    ok = not _FAILS
    print(f"\n=== 结果: A 段自校验 {'通过' if ok else '**失败 => 结论作废**'}; "
          f"累计用时 {time.time() - t0:.1f} s"
          f" (其中 stage6_life 单次 {t_prod:.2f} s) ===")
    if _FAILS:
        print(f"    失败项: {_FAILS}")
    return 0 if ok else 3


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:                      # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"\n**脚本自身错误**: {type(exc).__name__}: {exc}")
        sys.exit(2)
