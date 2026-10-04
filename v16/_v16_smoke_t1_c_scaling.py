# -*- coding: utf-8 -*-
#
# Copyright (c) 2024-2026 Jianzhong Yan
# Licensed under the Apache License, Version 2.0 (the "License");
#
"""
v16.7 · 工作包 T1 —— L2 中心荷的「第三把尺子」：跨 L 标度
[_VERSION_TAG = 'v16-smoke-t1-c-scaling-1']

**为什么是这一条（而不是 `v16.7_pre.md` 的 #1）**
  pre #1 说「L2 与 L4 是**两条完全独立的路径**」。锚定后**不成立**：
    「L2 的 c」 = `fit_central_charge(exact_gs, L=16)`（`_model/core.py:606`）
    「L4 的 c」 = `fit_central_charge(MERA_state, L=16)`（**同一个函数**）
  同函数、同 L、同物理对象（L=16 临界 TF-Ising 基态），唯一差别是态的**表示**
  （exact vs MERA 近似）⇒ 二者之差**按定义就是** `_metric/layer1.py:55` 已登记的
  `eps_c`。**不是跨层交叉验证，是重言。**（见 `v16/v16.7_plan.md` §0.1 锚 2。）

**真正缺的**（同计划 §0.1 锚 3）：现有两族**都只在单个 L 上** ——
  族A 纠缠谱族：`fit_central_charge(精确基态, L=16)`，吃 S(n) 对 **n** 的依赖；
  族B 能谱族  ：`spectral_central_charge`（`_model/checks.py:599`），吃 H 头三本征值。
  **没有任何一条**用 `S(L/2)` 对 **L** 的标度（PBC 下 `S(L/2) = (c/3)ln(L/π)+const`）。
  本脚本做这条 —— 结构上真正独立。这是 pre #1「想要的那种独立性」在本仓**唯一
  物理上成立的形式**。

**装置（隔离冒烟，不进 runner、不加守卫/指标）**
  真调门面导出的 `exact_ground_state` / `entanglement_curve`，**不复制逻辑**。
  `L ∈ {8, 12, 16, 18, 20}`（沿用 v16.5·D12 已跑通的 L 集），
  对 `S(L/2)` vs `ln L` 做线性最小二乘 ⇒ `c = 3 × slope`。

**本条相对计划的唯一附加（负对照；明确不参与落支）**
  在 `L ∈ {8, 12, 16}` 上另取 gapped 点 `h = 2.0` 跑**同一把尺子**。
  有隙相是面积律 ⇒ `S(L/2)` 饱和 ⇒ slope ≈ 0 ⇒ 该尺子**必须**给出 `c ≈ 0`。
  若它照样给出 ≈ 0.5，说明本尺子**没有分辨力**，T1 的任何读数都不可用。
  预登记判据：`|c_control| < 0.025`（= 0.5 的 5%）即"尺子会动"。
  这一条**不进落支判据**（落支阶梯在计划 §2.1 已冻结）。

**预登记 · 阶段 A 复现门控**（不过则落 `C-中途`，EXIT=3，不许继续）
  L=16：`E0/L  = -1.2752871546722913`                 （容差 5e-7）
        `S(L/2) = 0.7500553972738794`                 （容差 5e-5）
        `fit_central_charge['c'] = 0.5072169931065545`（族A；容差 1e-9）
  能谱族（族B，`spectral_central_charge()` 默认 L_list）：
        `c_devmax_pct = 0.21212661081063677`           （容差 5e-5）
        `c_L16        = 0.4989393669459468`            （容差 1e-9）
  以上五条**逐字取自本仓既有登记或本会话实测**，不新造数字。

**预登记 · 落支阶梯（计划 §2.1 冻结，不许事后改判据）**
  5% 判据：`|c_third − c_族| / c_族 ≤ 0.05`
  `C-三点一致`      与族A、族B **都**在 5% 内
  `C-与一族不符`    只与其中一族在 5% 内
  `C-与两族都不符`  与两族**都**超 5%
  `C-中途`          阶段 A 门控不过，或某点超 60 s

**两条必须一同登记的自限（计划 §2.1 写死）**
  1. 三点标度 + 有限尺寸 ⇒ **外推力极弱**。PBC 下 `S(L/2) = (c/3)ln(L/π)+const`
     带对数修正项 ⇒ 本读数**只能报一致性，不能报外推**。
  2. 若落 `C-与两族都不符`，**不许**直接读成「整条链的因果叙事要重写」
     —— 必须先排除「对数修正项被当成主项」这一实现风险。

**本脚本不做的事**
  1. 不进 runner（DP-14 默认）。
  2. **不调用任何 `metric_*` / `record_metric` / `record_guard`** —— 隔离，不污染指标登记。
  3. 不加跨 L 的组合 `fit_central_charge`（DP-29 默认：只做最小可证伪的那一条）。
  4. 不动任何既有登记读数、不动分母、不改评级（附A：本版**零升级**）。
"""

import os
import sys
import time

_T_PROC = time.time()   # ← 早于门面 import：否则漏掉 torch/quimb 的 ~10-40 s

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np      # noqa: E402

import spiral_model_v16 as G   # noqa: E402

_T_IMPORT = time.time()
_T0 = time.time()

try:
    import psutil
    _PROC = psutil.Process(os.getpid())

    def _rss_mb():
        return _PROC.memory_info().rss / 1024.0 ** 2
except Exception:           # noqa: BLE001
    def _rss_mb():
        return float('nan')

# ── 登记值（逐条注明出处；不许引用无锚数字）────────────────────────
E0L_REF = -1.2752871546722913      # 本会话实测复现， -1.275287
S_HALF_REF = 0.7500553972738794    # 同上 0.7501
C_A_REF = 0.5072169931065545       # 族A；对上 _runall_v16.3_r1.log:136 的 c=0.5072
C_B_REF = 0.4989393669459468       # 族B；_model/checks.py:664 的 c_spec @ L=16
DEVMAX_B_REF = 0.21212661081063677  # 族B； 0.212%

TOL_E0L = 5e-7
TOL_S = 5e-5
TOL_C = 1e-9
TOL_CB = 1e-9
TOL_DEVMAX = 5e-5

L_GATE = 16
L_LIST = (8, 12, 16, 18, 20)

# 负对照（附加，不参与落支）
L_CTRL = (8, 12, 16)
H_CTRL = 2.0
TOL_CTRL = 0.025        # |c_control| < 0.5 的 5% ⇒ "尺子会动"

TOL_BRANCH = 0.05       # 计划 §2.1 冻结的 5% 判据
T_ONE_MAX = 60.0        # 秒；单点超过即停（计划 §5 停机点）
MEM_STOP_MB = 1024.0    # MB；超过即停（计划 §5 停机点）

_LOG_PATH = os.path.join(_HERE, '_t1_c_scaling.log')

# 门面是否导出这四个；不导出则 `_model` 直取（同一个函数对象，不是复制）
_exact_gs = getattr(G, 'exact_ground_state', None)
_ent_curve = getattr(G, 'entanglement_curve', None)
_fit_cc = getattr(G, 'fit_central_charge', None)
_spec_cc = getattr(G, 'spectral_central_charge', None)
_IMPORT_NOTE = '门面 (spiral_model_v16)'
if _exact_gs is None or _ent_curve is None or _fit_cc is None or _spec_cc is None:
    from _model.core import (exact_ground_state as _exact_gs,      # noqa: E402
                             entanglement_curve as _ent_curve,
                             fit_central_charge as _fit_cc)
    from _model.checks import spectral_central_charge as _spec_cc  # noqa: E402
    _IMPORT_NOTE = '_model 直取（门面未全导出）'


class _Tee:
    def __init__(self, *streams):
        self._s = streams

    def write(self, d):
        for s in self._s:
            try:
                s.write(d)
            except Exception:      # noqa: BLE001
                pass
        return len(d)

    def flush(self):
        for s in self._s:
            try:
                s.flush()
            except Exception:      # noqa: BLE001
                pass


def _point(L, h=1.0):
    """真调门面函数取该点的 E0 与 S(n) 曲线。**不复制任何逻辑**。"""
    t = time.perf_counter()
    E0, gs = _exact_gs(L, 1.0, h)
    S = np.asarray(_ent_curve(gs, L), dtype=float)
    dt = time.perf_counter() - t
    # 注: RSS 取在 `exact_ground_state` **返回之后**，H 已被释放
    #     ⇒ 这是**下界**，不是峰值。停机条件据此仍成立（只可能更保守）。
    return {'L': int(L), 'h': float(h), 'E0': float(E0), 'gs': gs, 'S': S,
            'dt': dt, 'rss': _rss_mb()}


def _fit_slope(Ls, ys):
    """`ys` vs `ln Ls` 的最小二乘；返回 (slope, intercept, rms)。"""
    xs = np.log(np.asarray(Ls, dtype=float))
    y = np.asarray(ys, dtype=float)
    A = np.vstack([xs, np.ones_like(xs)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    rms = float(np.sqrt(np.mean((y - A @ coef) ** 2)))
    return float(coef[0]), float(coef[1]), rms


def main():
    print('=' * 78)
    print('v16.7 · T1 —— L2 中心荷的「第三把尺子」：S(L/2) 随 ln L 的标度')
    print('  装置: exact_ground_state + entanglement_curve（真调，不复制逻辑）')
    print('  不进 runner; 不调 metric_*; 不加组合拟合（DP-29）; 预期零升级')
    print('  函数来源: %s' % _IMPORT_NOTE)
    print('=' * 78)

    # ── 阶段 A：复现门控 ────────────────────────────────────────────
    print('\n-- 阶段 A: 复现门控 (L=%d) --' % L_GATE)
    a = _point(L_GATE)
    e0l = a['E0'] / L_GATE
    s_half = float(a['S'][-1])
    c_a = float(_fit_cc(a['gs'], L_GATE)['c'])
    b = _spec_cc()                      # 族B；默认 L_list=(8,10,12,14,16)
    c_b = float(b['c_L16'])
    dev_b = float(b['c_devmax_pct'])

    gate = [
        ('E0/L', e0l, E0L_REF, TOL_E0L),
        ('S(L/2)', s_half, S_HALF_REF, TOL_S),
        ('族A c = fit_central_charge@L=16', c_a, C_A_REF, TOL_C),
        ('族B c_L16', c_b, C_B_REF, TOL_CB),
        ('族B c_devmax_pct', dev_b, DEVMAX_B_REF, TOL_DEVMAX),
    ]
    fails = []
    for name, got, ref, tol in gate:
        ok = abs(got - ref) <= tol
        if not ok:
            fails.append(name)
        print('  %-34s 实测 %+.12f   登记 %+.12f   |差| %.2e  (<= %g)  %s'
              % (name, got, ref, abs(got - ref), tol, 'OK' if ok else '**FAIL**'))
    print('  用时 %.2f s' % a['dt'])

    if fails:
        print('\n-- 落支 --')
        print('  门控未过: %s' % ', '.join(fails))
        print('  落支 = **C-中途** ⇒ 不报任何标度结论，EXIT=3')
        print('\n用时：门面 import %.1f s + 计算 %.1f s = 总计 %.1f s'
              % (_T_IMPORT - _T_PROC, time.time() - _T0, time.time() - _T_PROC))
        print('=' * 78)
        return 3

    # ── 阶段 B：S(L/2) 随 ln L 的标度 ───────────────────────────────
    print('\n-- 阶段 B: S(L/2) 随 L 的标度, L ∈ %s --' % (list(L_LIST),))
    pts, stopped = [], None
    for L in L_LIST:
        p = _point(L)
        pts.append(p)
        print('  L=%2d  E0/L=%+.9f  S(L/2)=%.10f  (%d 个 n, n_max=%d)  '
              'RSS=%.0f MB  用时 %.2f s'
              % (L, p['E0'] / L, p['S'][-1], len(p['S']), len(p['S']),
                 p['rss'], p['dt']))
        if p['dt'] > T_ONE_MAX:
            stopped = 'L=%d 用时 %.2f s > %.1f s' % (L, p['dt'], T_ONE_MAX)
            break
        if p['rss'] == p['rss'] and p['rss'] > MEM_STOP_MB:
            stopped = 'L=%d RSS %.0f MB > %.0f MB' % (L, p['rss'], MEM_STOP_MB)
            break

    if stopped:
        print('\n  ⇒ **停机**：%s（计划 §5 停机点）' % stopped)
        print('\n-- 落支 --')
        print('  落支 = **C-中途** ⇒ 扫描不完整，不报标度结论，EXIT=3')
        print('\n用时：门面 import %.1f s + 计算 %.1f s = 总计 %.1f s'
              % (_T_IMPORT - _T_PROC, time.time() - _T0, time.time() - _T_PROC))
        print('=' * 78)
        return 3

    Ls = [p['L'] for p in pts]
    ys = [float(p['S'][-1]) for p in pts]
    slope, inter, rms = _fit_slope(Ls, ys)
    c3 = 3.0 * slope

    print('\n  %-4s %-10s %-16s %-12s' % ('L', 'ln L', 'S(L/2)', '残差'))
    for L, y in zip(Ls, ys):
        pred = slope * np.log(L) + inter
        print('  %-4d %-10.6f %-16.10f %+.3e' % (L, np.log(L), y, y - pred))
    print('  拟合: S(L/2) = %.6f * ln L + %.6f   (rms = %.3e)' % (slope, inter, rms))
    print('  ⇒ c_third = 3 * slope = **%.10f**' % c3)

    # ── 负对照（附加；不参与落支）────────────────────────────────────
    print('\n-- 负对照（附加，不参与落支）: gapped h=%.1f，同一把尺子应变哑 --' % H_CTRL)
    ctr = []
    for L in L_CTRL:
        p = _point(L, H_CTRL)
        ctr.append(p)
        print('  L=%2d  E0/L=%+.9f  S(L/2)=%.10f  用时 %.2f s'
              % (L, p['E0'] / L, p['S'][-1], p['dt']))
    sc, ic, rmsc = _fit_slope([p['L'] for p in ctr], [float(p['S'][-1]) for p in ctr])
    c_ctrl = 3.0 * sc
    ctrl_ok = abs(c_ctrl) < TOL_CTRL
    print('  ⇒ c_control = %.6f  (rms = %.3e)   判据 |c_control| < %g  %s'
          % (c_ctrl, rmsc, TOL_CTRL, 'OK（尺子会动）' if ctrl_ok else '**BAD（尺子不动）**'))

    # ── 落支 ─────────────────────────────────────────────────────────
    print('\n-- 落支（计划 §2.1 冻结判据，5%%）--')
    devA = abs(c3 - C_A_REF) / C_A_REF
    devB = abs(c3 - C_B_REF) / C_B_REF
    okA, okB = devA <= TOL_BRANCH, devB <= TOL_BRANCH
    print('  族A（纠缠谱族, fit_central_charge@L=16） = %.10f   |差| %.3f%%  %s'
          % (C_A_REF, 100.0 * devA, 'IN ' if okA else 'OUT'))
    print('  族B（能谱族,   spectral_central_charge L=16）= %.10f   |差| %.3f%%  %s'
          % (C_B_REF, 100.0 * devB, 'IN ' if okB else 'OUT'))
    print('  [背景] 族A 与族B **自身**相差 %.3f%%（同 L=16，不同数据与函数形式）'
          % (100.0 * abs(C_A_REF - C_B_REF) / C_B_REF))

    if okA and okB:
        branch = 'C-三点一致'
        why = '第三把尺子与两族既有读数**都**落在 5% 内'
    elif okA or okB:
        branch = 'C-与一族不符'
        why = ('只与%s落在 5% 内' % ('族A' if okA else '族B'))
    else:
        branch = 'C-与两族都不符'
        why = '与两族**都**超出 5%'
    print('  落支 = **%s**' % branch)
    print('  %s' % why)

    print('\n  【必须同页登记的自限 1】三点标度 + 有限尺寸 ⇒ **外推力极弱**；')
    print('    PBC 下 S(L/2)=(c/3)ln(L/π)+const 带对数修正项 ⇒')
    print('    本读数**只能报一致性，不能报外推**。')
    print('  【必须同页登记的自限 2】若落 `C-与两族都不符`，**不许**直接读成')
    print('    「整条链的因果叙事要重写」—— 必须先排除「对数修正项被当成主项」')
    print('    这一实现风险。')
    if branch == 'C-与两族都不符':
        print('    ⇒ **本支已触发**：下一步走 DP-26（是否改写 README 叙事），'
              '且先做对数修正项排查。')
    else:
        print('    ⇒ 本支未触发。')
    if not ctrl_ok:
        print('  ⚠ 负对照 BAD ⇒ 本尺子分辨力存疑，上面的落支**须打折读**。')

    print('\n[T1] 隔离自检：未调用 metric_* / 未碰 MERA / 未进 runner / '
          '未动既有读数与分母。')
    print('用时：门面 import %.1f s + 计算 %.1f s = 总计 %.1f s'
          % (_T_IMPORT - _T_PROC, time.time() - _T0, time.time() - _T_PROC))
    print('=' * 78)
    return 0


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    with open(_LOG_PATH, 'w', encoding='utf-8') as _lf:
        _orig = sys.stdout
        sys.stdout = _Tee(_orig, _lf)
        try:
            _rc = main()
        finally:
            sys.stdout = _orig
    sys.exit(_rc)
