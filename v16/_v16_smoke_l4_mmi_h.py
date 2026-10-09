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
v16.3 · S2 冒烟 —— L4: 给 `G8` 补检验力 (离开临界点的 h 扫描)
[_VERSION_TAG = 'v16-smoke-l4-mmi-h-1']

**这是 v16.3_plan.md §7.2 的 S2**, **不进 runner** (`_v16_run_all.py`), 不动任何登记值,
不碰生产路径, **不加守卫、不动分母** (DP-2)。

要回答的问题 (v16.3_plan.md §1.1 Q1):
  `G8` 登记的 `I3(w) = [0.194068, 0.216633, 0.224846, 0.229723]` (w=1..4) **全部 > 0**,
  条件串「全部 <= 0」因此未通过 (`_runall.log:956`)。这是**负结果**。
  但它在**临界点**上取的 (`exact_ground_state(16, 1, 1)`) —— 而临界环上的关联是**幂律**,
  `xi` 只是**有效长度不是关联长度** (`_v16_data.json:74791`), 换拟合窗口 `xi` 摆动 ~6 倍。
  于是那条负结果**可能只是在退化区取样**, 而不是「这个态真的违反 MMI」。
  本冒烟换一个自由度 (`h`, 离开临界点) 去测**同一个判据**, 看它到底有没有检验力。

**为什么这是「补检验力」而不是「再测一遍」**: `h = 1.0` 那一档是**对照组** ——
它必须复现登记值, 因此**不携带新信息**; 信息全在 `h != 1` 的四档里。
判词只能挂在 `h != 1` 的态上, **不可外推回临界点** (§2.4)。

**ξ 的多窗口是必须的, 不是锦上添花** (§2.3 D1-c): `boundary_correlation_length` 写死
「全部 d」, 而 `_v16_data.json:74791` 已实测同一个态换窗口 `xi` 从 7.99 变到 46.91。
所以**单窗口的 `xi` 不足以支撑 `xi/L << 1` 的断言**; 本冒烟每档报三个窗口。

预登记判据 (**先写死, 再看数**; 全文见 §2.3, 此处是它的可执行转录):
  * **NO-GO**: 任一档**全窗** `xi >= L` => **该档作废**, 不许当作「有检验力」的档。
    作废档**照登** (标 `[作废]`), 只是不参与 P1/P2 判定 —— 不许删。
  * **P1**: 在 `xi/L <= 0.25` 的档上仍 `I3 > 0` 全部 w => **有检验力的负结果**。
    登记为「该态无几何对偶的 MMI 证据」。
  * **P2**: `I3` 转 `<= 0` => 必要条件在该态成立。**必须与实测 `xi/L` 同页**,
    并显式声明**不可外推回临界点**。
  * **P3**: 符号随 `h` **非单调** => 「MMI 符号在本模型上不是稳健量」(负面结论, 照登)。
  * **P4**: 五档全部 `xi >= L` => 「在 `L=16` 上无法把 `xi` 压到 `L` 以下」,
    如实报告所试的 `h` 网格。**不许**把「没做成」写成「路径已探明」。
  * **P1/P2/P3 可同时成立**; 报告**须逐条列出哪几条成立**, 不得只报有利的那条。

P3 的**口径**(计划未给可执行定义, 故在此**先于数据写死**):
  令 `s(档) in {'-','+','+-'}` = (全部 <= 0 / 全部 > 0 / 混合)。若 `s` 沿 `h` 的序列
  **变号 >= 2 次** (存在 `i<j<k` 使 `s_i != s_j` 且 `s_j != s_k`) => 判 P3。
  定在「变号两次」而不是「变号一次」, 是因为一次变号可能只反映跨过一个相变区;
  写在这里是为了**防止结果出来之后再挑口径**。有效档 < 3 时 P3 不可判, 照实写「不可判」。

**防 p-hacking 的结构保证** (§2.3 合法性条款): `xi` **先测**, `I3` **后测**。
本脚本把这条**做进代码结构**, 不靠自觉:
  1. `--xi-only` 只跑 §0 / §A-ξ / §B1 (一行 `I3` 都不算), 供「看 ξ 之后调网格」用;
  2. 完整运行里 §B1 (全部 ξ) 与 §B2 (全部 I3) 是**两个分离的循环**, 中间有显式冻结线;
  3. `I3` 一旦打印, 网格冻结 —— 要改就重跑 `--xi-only` 再定, **不许看完 I3 改网格**。

**超出计划 §7.2 规格的一处增补 (必须写明, 且它没有改动任何判据)**:
  P1/P2/P3 三条判据**全部**由 `I3` 的**符号**决定。一个符号若其实是 eigsh 的数值
  噪声, 那 P1/P2/P3 就是**假绿 / 假红**(同罪)。故 §B2b 给每条符号判定配一把尺子 ——
  **每档额外 N_REPEAT 次独立 eigsh, 查符号是否不变**; §C 的联合表另加一列 `min|I3|`,
  把「符号」与「量级」摆在同一页上。
  这与 `_v16_smoke_l6_wavelength.py:114` 给 `spacing` 加**行间标准误**是同一个动作:
  没有它, 「符号变了」与「噪声翻转」分不开。
  **本段是看过 §C 的联合表之后才加的, 计划 §7.2 的 S2 规格里没有它** —— 之所以加,
  是因为首轮表里 `h=2.0` 档的决定性符号落在 `|I3| = 1.0e-7` 上 (比同档最大 |I3|
  小 5 个数量级)。判据 P1/P2/P3 的**文字口径一字未动**。

数据的**上游锚点** (逐条核过):
  * `G8` 登记值      —— `_runall.log:956`;
  * `xi = 19.314`    —— `_v16_run.log:429` (L=16, h=1);
  * `xi` 多窗口       —— `_v16_data.json:74791` (`d in [1,3] -> 7.99, d in [4,8] -> 46.91`);
  * `I3`/`xi` 的估计量 —— 生产同源, 见下方「估计量来源声明」。

**估计量来源声明**: `mmi_tripartite` 是本冒烟**直接调生产实现** (`_metric/solvers.py:108`,
`from _metric.solvers import`), **不是复刻件** —— `G8` 在 `_metric/main.py:191` 调的正是它,
所以 §A 的复现是「同一个函数、同一个态」的复现, 不是「看着像」。
门面 `spiral_metric_v16` **没有** re-export 它 (那是 `G8` 自己的内部依赖),
本冒烟**不去改门面的 API** (加 re-export 就是动生产)。
ξ 侧一律走门面 (`G.boundary_correlation_length` / `G._fit_length`),
**唯一的复刻件是分箱那 9 行** (`binned_corr`), 它由 §A1 与真函数**逐位比对**发许可证。

⚠️ **天花板 (与结果同页, 不许省)** —— 沿用 `_v16_smoke_l4_mmi.py:20-27`:
  1. MMI 成立只是**必要**条件 —— 通过 != RT 被验证, 更 != 「涌现时空」
     (B2 硬边界 `spiral_model_v16.py:52-53`)。
  2. **本冒烟的一切读数都挂在 `h != 1` 的态上**, 与 `G8` 在临界点上的登记值
     **互不影响**: 这里 `I3 <= 0` **不等于**临界点上 `I3 <= 0`; 反之亦然 (§2.4)。
  3. 成熟度**不动**: L4 仍是 C (audit `:75`)。D1 不给 L4 任何成熟度 (§2.4)。
  4. 本脚本**不注册任何守卫 / 指标条目**: `metric_correlation_exponent` 内部会调
     `record_metric`, 那只是**本进程内**的 list append, **不落盘、不进 runner**。

退出码 (沿用 v16.2 两条冒烟): `0 = 通过`; `3 = 复现/自校验失败` (A 段复现失败,
或任一档的 SSA/纯度自校验失败, 或任一档的 I3 符号在重复 eigsh 下翻转 ——
三者都是「结论作废」, 不是物理结论); `2 = 脚本自身错误`。
⚠️ `3` 是「预期内的失败路径」, **不是 runner 意义的回归** (memory: `v16-single-runner`)。

用法:
    python v16/_v16_smoke_l4_mmi_h.py              # 完整三段
    python v16/_v16_smoke_l4_mmi_h.py --xi-only    # 只测 ξ (供定网格; 不算 I3)
"""

import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'v16'))

import numpy as np  # noqa: E402
import spiral_model_v16 as G  # noqa: E402
import spiral_metric_v16 as MM  # noqa: E402
from _metric.solvers import mmi_tripartite  # noqa: E402

_VERSION_TAG = 'v16-smoke-l4-mmi-h-1'

# --- 生产配置 (与 `_metric/main.py` 的 exact_ground_state(16,1,1) 同参数) ---------
L = 16
J = 1.0
H_GRID = (1.0, 1.2, 1.5, 2.0, 3.0)   # DP-1 已决; h=1.0 是**对照组**, 其余是待测档

# --- A 段的自校验靶 (全部来自已登记读数, 不是新算的) ------------------------------
I3_LOGGED = (0.194068, 0.216633, 0.224846, 0.229723)   # `_runall.log:956` (6 位小数)
XI_LOGGED_FULL = 19.314                                # `_v16_run.log:429` (3 位小数)
XI_LOGGED_W13 = 7.99                                   # `_v16_data.json:74791` (2 位小数)
XI_LOGGED_W48 = 46.91                                  # 同上
TOL_I3 = 1e-6          # 登记值 6 位小数 => 半个末位; §8 D1-b 定的是「逐位或 1e-6 内」
TOL_XI_FULL = 5e-4     # 19.314 是 3 位小数 => 半个末位
TOL_XI_WIN = 5e-3      # 7.99 / 46.91 是 2 位小数 => 半个末位

# --- ξ 的窗口 (§2.3 D1-c): (lo, hi), None 表示全窗 -------------------------------
XI_WINDOWS = ((1, 3), (4, 8), (None, None))
XI_WINDOW_KEYS = ('1-3', '4-8', 'full')
XI_POWER_THRESHOLD = 0.25   # 「有检验力」的 ξ/L 门槛 (计划 §2.3 预登记)
N_REPEAT = 3                # 每档的**独立** eigsh 重复次数 (测符号可重复性; 见 §B2b)

_FREEZE_TAG = '=' * 76

_FAILS = []


def check(label, cond, detail=''):
    tag = 'PASS' if cond else '**FAIL**'
    print(f"    [{tag}] {label}" + (f"  —— {detail}" if detail else ''))
    if not cond:
        _FAILS.append(label)


# ---------------------------------------------------------------------------
# ξ 的分箱: `boundary_correlation_length` (`core.py:681-690`) 的**逐字复刻**。
# 复刻的合法性不靠「看着像」, 靠 §A1 与真函数 `boundary_correlation_length(C, L)`
# **逐位比对**来证 (全窗那条必须相等); 不过 => 多窗口不可用。
# 为什么需要复刻: 真函数把分箱结果**内部**交给 `_fit_length`, 不把 (d, y) 交出来,
# 而多窗口必须在 (d, y) 上按区间切片。**改真函数去暴露它 = 动生产**, 不做。
# ---------------------------------------------------------------------------
def binned_corr(C, L):
    """按环上距离 d = min(j-i, L-(j-i)) 分箱平均 |C_ij|。返回 (唯一 d 升序, 分箱均值)。"""
    ds, vs = [], []
    for i in range(L):
        for j in range(i + 1, L):
            d = min(j - i, L - (j - i))
            ds.append(d)
            vs.append(abs(C[i, j]))
    ds = np.array(ds)
    vs = np.array(vs)
    ud = np.unique(ds)
    return ud, np.array([vs[ds == d].mean() for d in ud])


def xi_multi(C, L):
    """
    三个窗口的 ξ。**拟合一律走生产的 `G._fit_length`**, 本函数只做区间切片 ——
    切片与 `_fit_length` 的 `max_d` 做的是同一件事 (`core.py:665-667`),
    只是 `max_d` 只有上界、给不出 `[4,8]` 这种下界窗口, 故此处自己切。
    """
    d, y = binned_corr(C, L)
    out = {}
    for (lo, hi), key in zip(XI_WINDOWS, XI_WINDOW_KEYS):
        if lo is None:
            out[key] = float(G._fit_length(d, y))
        else:
            m = (d >= lo) & (d <= hi)
            out[key] = float(G._fit_length(d[m], y[m]))
    return out, d, y


def sign_state(i3s):
    """档的符号状态: '-' 全部 <= 0 / '+' 全部 > 0 / '+-' 混合。"""
    if all(v <= 0 for v in i3s):
        return '-'
    if all(v > 0 for v in i3s):
        return '+'
    return '+-'


def measure_xi(h):
    """只测 ξ 侧的量 (不碰 I3) —— 这是「ξ 先测」那条纪律的代码形态。"""
    tb = time.time()
    E0, gs = G.exact_ground_state(L, J, h)
    t_state = time.time() - tb
    _, C, _ = G.boundary_correlation_graph(gs, L)
    xw, d, y = xi_multi(C, L)
    ce = MM.metric_correlation_exponent(gs, L)   # 生产估计量; 进程内 append, 不落盘
    return dict(h=h, E0=E0, gs=gs, C=C, xi=xw, d=d, y=y, t_state=t_state,
                two_delta=float(ce['two_delta']), eta_naive=float(ce['eta_naive']))


def print_xi_table(rows):
    print(f"    {'h':>5s} | {'xi[1-3]':>9s} {'xi[4-8]':>9s} {'xi[全窗]':>9s} "
          f"{'E0':>14s} | {'2D(CFT拟合)':>12s} {'eta朴素':>8s} | {'建态用时':>8s}")
    for r in rows:
        x = r['xi']
        print(f"    {r['h']:5.2f} | {x['1-3']:9.4f} {x['4-8']:9.4f} {x['full']:9.4f} "
              f"{r['E0']:14.9f} | {r['two_delta']:12.6f} {r['eta_naive']:8.4f} | "
              f"{r['t_state']:7.2f}s")


def main():
    xi_only = '--xi-only' in sys.argv
    t0 = time.time()
    print(f"=== {_VERSION_TAG} · L4: 给 G8 补检验力 (h 扫描), L={L}, J={J:g} ===")
    print(f"    口径: 生产同源估计量 mmi_tripartite + boundary_correlation_length/_fit_length;")
    print(f"    不进 runner, 不加守卫, 不动分母 (DP-2), 不改任何登记值。"
          f"{'  [--xi-only: 只测 xi, 不算 I3]' if xi_only else ''}")

    # =====================================================================
    print(f"\n--- 0 · 生产配置回显 ---")
    print(f"    L={L} (固定), J={J:g} (固定), h 网格 = {list(H_GRID)}")
    print(f"    对照档 h=1.0 即 `_metric/main.py` 里 G8 的生产配置 "
          f"exact_ground_state({L}, {J:g}, 1.0)")
    print(f"    I3 估计量: mmi_tripartite (G8 亲自调它, `_metric/main.py:191`)")
    print(f"    xi 估计量: boundary_correlation_graph -> boundary_correlation_length "
          f"-> _fit_length")
    print(f"    多窗口: {XI_WINDOW_KEYS}  (依据 `_v16_data.json:74791`: 换窗口 xi 变 ~6 倍)")

    # =====================================================================
    # §A · 对照档 h=1.0 —— 复现登记值。**这一段不携带新信息**, 它是后面所有档的许可证。
    # =====================================================================
    print(f"\n--- A · 对照档 h=1.0: 复现 G8 登记值 (自校验, 必须先过) ---")
    tA = time.time()
    a = measure_xi(1.0)
    print(f"    xi(全窗) = {a['xi']['full']:.6f}   "
          f"(登记 `_v16_run.log:429`: {XI_LOGGED_FULL})")
    check(f"xi(全窗, L={L}, h=1.0) 复现登记值 {XI_LOGGED_FULL}",
          abs(a['xi']['full'] - XI_LOGGED_FULL) <= TOL_XI_FULL,
          f"|{a['xi']['full']:.6f} - {XI_LOGGED_FULL}| = "
          f"{abs(a['xi']['full'] - XI_LOGGED_FULL):.2e}")

    # A1: 分箱复刻件与**真函数**逐位比对 —— 这是多窗口用复刻件的**唯一许可证**
    xi_real = float(G.boundary_correlation_length(a['C'], L))
    d_fit = abs(a['xi']['full'] - xi_real)
    print(f"\n    A1 分箱复刻件(全窗) = {a['xi']['full']:.12f}")
    print(f"       真边界关联长度     = {xi_real:.12f}   "
          f"[boundary_correlation_length(C, L)]")
    check("分箱复刻件与 boundary_correlation_length 逐位一致 (多窗口的许可证)",
          d_fit < 1e-12, f"|diff| = {d_fit:.3e}")

    # A2: 多窗口复现 `_v16_data.json:74791` 的两个实测值
    print(f"\n    A2 多窗口 (登记对照 `_v16_data.json:74791`: "
          f"d in [1,3] -> 7.99, d in [4,8] -> 46.91)")
    check(f"xi(d in [1,3]) 复现 {XI_LOGGED_W13}",
          abs(a['xi']['1-3'] - XI_LOGGED_W13) <= TOL_XI_WIN,
          f"实算 {a['xi']['1-3']:.4f}")
    check(f"xi(d in [4,8]) 复现 {XI_LOGGED_W48}",
          abs(a['xi']['4-8'] - XI_LOGGED_W48) <= TOL_XI_WIN,
          f"实算 {a['xi']['4-8']:.4f}")
    ratio = a['xi']['4-8'] / a['xi']['1-3'] if a['xi']['1-3'] else float('nan')
    print(f"    => 同一态同一份 C, 换窗口 xi 在 {a['xi']['1-3']:.2f} ~ {a['xi']['4-8']:.2f} "
          f"之间摆 {ratio:.2f} 倍 ——")
    print(f"       **单窗口的 xi 撑不起 `xi/L << 1` 的断言**。")

    # A3: I3 复现 —— 这一条是 G8 登记值本身 (已知量, 不是新信息)
    mm = None
    if xi_only:
        print(f"\n    A3 I3 复现: **--xi-only 模式跳过** "
              f"(不算 I3, 这是防 p-hacking 的结构保证)")
    else:
        mm = mmi_tripartite(a['gs'], L)
        print(f"\n    A3 I3 主读数 (登记 `_runall.log:956`)")
        print(f"       {'w':>3s} | {'I3 实算':>13s} {'I3 登记':>13s} {'|diff|':>10s}")
        for w, got, want in zip((1, 2, 3, 4), mm['i3'], I3_LOGGED):
            print(f"       {w:3d} | {got:13.8f} {want:13.8f} {abs(got - want):10.2e}")
        dev = max(abs(g - w) for g, w in zip(mm['i3'], I3_LOGGED))
        check("I3(w=1..4) 复现 G8 登记值 (1e-6 内)", dev <= TOL_I3,
              f"max|diff| = {dev:.3e}")
        check("I3 自校验 (SSA / 纯度) 成立",
              mm['ssa_min'] > -1e-9 and mm['purity_max'] < 1e-12,
              f"ssa_min = {mm['ssa_min']:.3e}, purity_max = {mm['purity_max']:.3e}"
              f"  (**自校验不是判据**: SSA 是定理, 物理上不可能失败)")

    if _FAILS:
        print(f"\n    **A 段复现失败 => 按纪律先查复现, 不继续** (v16.2 §5.1)")
        print(f"    失败项: {_FAILS}")
        return 3
    print(f"    A 段用时 {time.time() - tA:.2f} s")

    # =====================================================================
    # §B1 · ξ 侧全档 —— **先把所有档的 ξ 测完并打印**, 再算任何 I3。
    # =====================================================================
    print(f"\n--- B1 · xi 侧 (先测; 此时一行 I3 都还没有算出来) ---")
    rows = [a]
    for h in H_GRID[1:]:
        print(f"    ... h={h:g} 建态 + 测 xi")
        rows.append(measure_xi(h))
    print()
    print_xi_table(rows)
    print(f"    ⚠️ 2D 是**共形形式**的拟合 (`metric_correlation_exponent`, 用 (pi/L)/sin(pi r/L))。")
    print(f"       它作为 Delta_sigma = 1/8 的读数**只在临界点 (h=1) 有物理意义**;")
    print(f"       h != 1 时该态有能隙, 这个拟合值**不是标度维** —— 照登,")
    print(f"       但不得当成「2D 随 h 变化」的物理结论。")

    if xi_only:
        print(f"\n{_FREEZE_TAG}")
        print(f"  [--xi-only] 未计算任何 I3。可据此调网格, 然后跑完整版;")
        print(f"  一旦完整版的 I3 打印出来, 网格即冻结 (v16.3_plan §2.3)。")
        print(f"{_FREEZE_TAG}")
        print(f"\n=== 结果: xi 侧完成 (未测 I3); 累计用时 {time.time() - t0:.1f} s ===")
        return 0

    print(f"\n{_FREEZE_TAG}")
    print(f"  ***** 网格冻结线 —— 以下开始算 I3, 网格自此冻结 (v16.3_plan §2.3) *****")
    print(f"  冻结的网格: h = {list(H_GRID)}")
    print(f"  看过 I3 之后再改网格 = p-hacking; 作废档照登, 不许删。")
    print(f"{_FREEZE_TAG}")

    # =====================================================================
    # §B2 · I3 侧 —— 与 §B1 是**分离的循环**, 这是「ξ 先测、I3 后测」的代码形态。
    # =====================================================================
    print(f"\n--- B2 · I3 侧 (后测; 网格已冻结) ---")
    for r in rows:
        if r['h'] == 1.0:
            m2 = mm                       # 对照档复用 A3 的结果, 不重算
        else:
            m2 = mmi_tripartite(r['gs'], L)
            # 自校验: 不成立 => 该档的 I3 不可用 (实现错, 不是物理结论)
            check(f"h={r['h']:g} 的 I3 自校验 (SSA/纯度) 成立",
                  m2['ssa_min'] > -1e-9 and m2['purity_max'] < 1e-12,
                  f"ssa_min = {m2['ssa_min']:.3e}, purity_max = {m2['purity_max']:.3e}")
        r['i3'] = list(m2['i3'])
        r['trend'] = float(m2['trend'])
        r['state'] = sign_state(r['i3'])
        r['min_abs_i3'] = float(np.abs(r['i3']).min())
        r['xi_over_L'] = r['xi']['full'] / L
        r['valid'] = r['xi_over_L'] < 1.0        # NO-GO: 全窗 xi >= L => 作废

    # =====================================================================
    # §B2b · I3 符号的可重复性 —— **读数可信度, 不是判据** (见文件头「超出规格」段)。
    # 做法: 每档额外 N_REPEAT 次**独立** eigsh (eigsh 不传 v0 => 随机起点),
    # 逐 w 比较符号状态是否不变。符号若会翻转 => P1/P2/P3 的判定作废 (退 3)。
    # 对照档 h=1.0 不做重复: 它是 A 段的复现靶, 且符号是所有档里最钝的 (|I3| ~ 0.2)。
    # =====================================================================
    print(f"\n--- B2b · I3 符号的可重复性 (读数可信度, **不是判据**; "
          f"每档额外 {N_REPEAT} 次独立 eigsh) ---")
    print(f"    {'h':>5s} | {'符号(主)':>8s} {'符号(重复)':>16s} | {'逐w极差':>10s} "
          f"{'min|I3|':>10s} | {'符号可重复':>10s}")
    for r in rows:
        if r['h'] == 1.0:
            continue
        reps = []
        for _ in range(N_REPEAT):
            _, gs_rep = G.exact_ground_state(L, J, r['h'])
            reps.append(list(mmi_tripartite(gs_rep, L)['i3']))
        allv = np.array([r['i3']] + reps)
        states = [sign_state(v) for v in allv]
        same = all(s == states[0] for s in states)
        spread = float((allv.max(0) - allv.min(0)).max())
        r['spread'], r['sign_same'] = spread, same
        print(f"    {r['h']:5.2f} | {states[0]:>8s} {','.join(states[1:]):>16s} | "
              f"{spread:10.2e} {r['min_abs_i3']:10.2e} | {'是' if same else '**否**':>10s}")
        check(f"h={r['h']:g} 的 I3 符号在 {N_REPEAT} 次独立 eigsh 下不变",
              same, f"逐 w 极差 = {spread:.2e}")
    print(f"    ⇒ 符号只要能重复, P1/P2/P3 的符号判定就**不是噪声**;")
    print(f"      **但量级仍必须单列** (§C 的 `min|I3|` 列) —— 一个 `+1e-7` 的符号与")
    print(f"      一个 `+7e-3` 的符号在数值上都可重复, 在物理上**不是同一件事**。")

    # =====================================================================
    # §C · 联合表 —— ξ/L 与 I3 **同页** (v16.2 §4.2 纪律)
    # =====================================================================
    print(f"\n--- C · 联合表 (xi 与 I3 同页) + 判据判定 ---")
    print(f"    NO-GO 口径: **全窗** xi >= L ({L}) => 该档作废 (照登, 标 [作废])")
    print()
    print(f"    {'h':>5s} | {'xi[1-3]':>8s} {'xi[4-8]':>8s} {'xi[全窗]':>9s} {'xi/L':>7s} "
          f"{'档':>6s} | {'I3(w=1)':>12s} {'I3(w=2)':>12s} {'I3(w=3)':>12s} {'I3(w=4)':>12s} "
          f"{'符号':>5s} {'趋势':>11s}")
    for r in rows:
        x = r['xi']
        st = '' if r['valid'] else '[作废]'
        print(f"    {r['h']:5.2f} | {x['1-3']:8.4f} {x['4-8']:8.4f} {x['full']:9.4f} "
              f"{r['xi_over_L']:7.4f} {st:>6s} | "
              + ' '.join(f"{v:12.3e}" for v in r['i3'])
              + f" {r['state']:>5s} {r['trend']:+11.3e}")

    # 量级 —— 符号离开量级就是空的 (见文件头「超出规格」段)
    print(f"\n    量级 (与符号**同页**; 缺了它, 一个 +1e-7 会被读成与 +7e-3 等价):")
    for r in rows:
        print(f"      h={r['h']:4.2f}: min|I3| = {r['min_abs_i3']:.2e},  "
              f"max|I3| = {max(abs(v) for v in r['i3']):.2e}"
              f"{'   [作废档]' if not r['valid'] else ''}")

    print(f"\n    三窗口 xi/L —— 用于 §11 风险 3 (跨阈值时必须显式指出):")
    for r in rows:
        trio = [r['xi'][k] / L for k in XI_WINDOW_KEYS]
        cross = (min(trio) <= XI_POWER_THRESHOLD) != (max(trio) <= XI_POWER_THRESHOLD)
        flag = ('  ⚠️ **三窗口跨越 0.25 阈值** => '
                '「有检验力」这个判断本身依赖窗口选择') if cross else ''
        print(f"      h={r['h']:4.2f}: xi/L = [{trio[0]:.4f}, {trio[1]:.4f}, {trio[2]:.4f}]"
              f"  (全窗{'' if r['valid'] else ' [作废]'}){flag}")

    # ---------------- 判据落支 ----------------
    valid = [r for r in rows if r['valid']]
    powered = [r for r in valid if r['xi_over_L'] <= XI_POWER_THRESHOLD]
    print(f"\n    --- 判据落支 (P1/P2/P3 可同时成立, 逐条列, 不得只报有利的那条) ---")
    print(f"    有效档 = {[r['h'] for r in valid]}   "
          f"(作废档 = {[r['h'] for r in rows if not r['valid']]}, 照登)")
    print(f"    有检验力档 (xi/L <= {XI_POWER_THRESHOLD}) = {[r['h'] for r in powered]}")

    hits = []
    if not valid:
        print(f"\n    => **P4 成立**: 五档全部 `xi >= L` —— "
              f"「在 L={L} 上无法把 xi 压到 L 以下」。")
        print(f"       如实报告所试网格: h = {list(H_GRID)}, J = {J:g}, L = {L}。")
        print(f"       **不许**把「没做成」写成「路径已探明」。")
        hits.append('P4')
    else:
        p1 = [r for r in powered if r['state'] == '+']
        if p1:
            print(f"\n    => **P1 成立** (h = {[r['h'] for r in p1]}): 在 xi/L <= "
                  f"{XI_POWER_THRESHOLD} 的档上仍 I3 > 0 全部 w")
            for r in p1:
                print(f"       h={r['h']:4.2f}: xi/L(全窗) = {r['xi_over_L']:.4f}, "
                      f"三窗口 xi/L = "
                      f"{['%.4f' % (r['xi'][k] / L) for k in XI_WINDOW_KEYS]}, "
                      f"min|I3| = {r['min_abs_i3']:.2e}, "
                      f"符号可重复 = {'是' if r['sign_same'] else '**否**'}")
            print(f"       => 「**该态无几何对偶的 MMI 证据**」—— 这是**有检验力的负结果**,")
            print(f"          比 v16.2 临界点上那条强得多 (那条所属的档本身按 NO-GO 作废)。")
            print(f"          `min|I3|` 与三窗口 xi/L 都印在上面: 符号要**离噪声足够远**、")
            print(f"          档位要**三个窗口都不越 0.25**, 这条负结果才立得住。")
            hits.append('P1')
        elif not powered:
            print(f"\n    => **P1 不可判**: 有效档里**没有档**达到 xi/L <= "
                  f"{XI_POWER_THRESHOLD}")
            print(f"       (第四种情况 —— 计划未列, 照实登记该情况本身, 不硬套 P1/P2)")
            hits.append('P1-不可判')

        p2 = [r for r in valid if r['state'] == '-']
        if p2:
            print(f"\n    => **P2 成立** (h = {[r['h'] for r in p2]}): I3(w) <= 0 全部 w ——")
            print(f"       必要条件在**这些态**上成立。同页实测档位:")
            for r in p2:
                print(f"         h={r['h']:4.2f}: xi/L(全窗) = {r['xi_over_L']:.4f}, "
                      f"xi[1-3]/L = {r['xi']['1-3'] / L:.4f}, "
                      f"2D拟合 = {r['two_delta']:.6f}, "
                      f"min|I3| = {r['min_abs_i3']:.2e}")
            print(f"       ⚠️ **量级附注**: 若某档的 `min|I3|` 已到 `1e-8` 量级, 那一条")
            print(f"          只能读作「**I3 ~ 0**」, **不是**「明显 <= 0」—— 即该 w 上")
            print(f"          「必要条件成立」是**平凡**的, 支撑 P2 的是其余那些大分量的 w。")
            print(f"       ⚠️ **不可外推回临界点** —— 这是「在另一个态上做同一个检验」,")
            print(f"          不是同一个结论。G8 在 h=1.0 上的登记值**一个数字都不动** (§2.4)。")
            hits.append('P2')

        # P3: 符号沿 h 非单调 (口径见文件头, 先于数据写死)
        seq = [r['state'] for r in valid]
        flips = sum(1 for i in range(len(seq) - 1) if seq[i] != seq[i + 1])
        print(f"\n    P3 口径 (先写死): 符号状态沿 h 的序列 = {seq}, 变号 {flips} 次; "
              f"判据 = 变号 >= 2 次")
        if len(valid) < 3:
            print(f"    => **P3 不可判**: 有效档只有 {len(valid)} 个 (< 3), 照实写「不可判」。")
        elif flips >= 2:
            print(f"    => **P3 成立**: 「MMI 符号在本模型上不是稳健量」(负面结论, 照登)。")
            mins = ', '.join('h=%g: %.2e' % (r['h'], r['min_abs_i3']) for r in valid)
            print(f"       ⚠️ **必须与量级同页**: 各有效档 min|I3| = {mins}")
            print(f"       若决定性那次变号发生在 `min|I3|` 很小的档上, 该变号是**临界式的**")
            print(f"       —— 数值上可重复 (§B2b) 不等于物理上量级可忽略。")
            print(f"       措辞应为「**符号不是稳健量**」, **不是**「MMI 被强烈违反」。")
            hits.append('P3')
        else:
            print(f"    => P3 不成立 (变号 {flips} 次 < 2): 符号沿 h "
                  f"{'不变' if flips == 0 else '单调变号一次'}, 本判据不触发。")

    print(f"\n    **落支汇总: {hits if hits else '无判据触发'}**")

    # ---------------- 天花板 (与结果同页) ----------------
    print(f"\n--- 天花板 (与结果同页, 不许省) ---")
    print(f"    1. MMI 成立只是**必要**条件 —— 通过 != RT 被验证, 更 != 「涌现时空」")
    print(f"       (B2 硬边界 `spiral_model_v16.py:52-53`)。")
    print(f"    2. **不可外推回临界点**: 本冒烟的一切读数挂在 h != 1 的态上;")
    print(f"       这里 I3 <= 0 **不等于**临界点上 I3 <= 0, 反之亦然。")
    print(f"       G8 的登记值 (`_runall.log:956`) **一个数字都不动** (§2.4)。")
    print(f"    3. 成熟度**不动**: L4 仍是 C (audit :75)。D1 不给 L4 任何成熟度。")
    print(f"    4. 本冒烟**未加任何守卫 / 指标条目**, 分母 78 / 27 / 13 不变 (DP-2)。")
    print(f"    5. SSA 与 MMI **不可互换引用**: 前者是自校验 (定理), 后者才是判据。")

    ok = not _FAILS
    print(f"\n=== 结果: 自校验 {'通过' if ok else '**失败 => 结论作废**'}; "
          f"累计用时 {time.time() - t0:.1f} s ===")
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
