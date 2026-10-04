# -*- coding: utf-8 -*-
#
# Copyright (c) 2024-2026 Jianzhong Yan
# Licensed under the Apache License, Version 2.0 (the "License");
#
"""
v16.6 · 工作包 T1 —— L7 的「B 支参数无关的解析预期」真的存在吗？
[_VERSION_TAG = 'v16-smoke-t1-l7-jac-analytic-1']

**靶心（两条独立锚点，互相印证）**
  `_metric/negctrl.py:464-474`（活代码里的**逐字原文**，比 audit 转述更全）:
    「(3) **升级条件 (留给以后)**: 若 B 支出现**参数无关**的解析预期
      (例如从自指映射的结构本身推出谱半径上界), 再升级成计分守卫;
      在那之前它只是"报告"。」
  独立写下的同一件事:
    「B 支虽有分辨力, 却是同一套自指约定在**特定参数下的产物, 参数一变就可能塌回 1.0**
      ⇒ 算进分母等于用自己挑的参数换一个百分点。」

**代码锚点（T0 取证，已逐行读 —— 不是推断）**
  `_model/main.py:119`  生产调用 `selfref_coupled(N=N_self, steps=8000)`
                        —— **不传 g、不传 seed** ⇒ 默认 g=1.0, seed=0（口径已核实）。
  `_model/stage67.py:502-516`  `_rho_jac`: J = g·diag(1-x*^2)·W*（:515）; 返回 max|eig(J)|（:516）。
  `_model/stage67.py:463-464`  docstring: 变体 B 每步归一使 ||x*||=1, 才有真不动点 W* = x* x*^T。
  `_model/stage3.py:367`       **该归一确实由代码强制**（每步 `xn/||xn||`）⇒ 推导前提成立。
  `_model/stage3.py:369-370`   **但 W* 不是精确的 x*x*^T**: EMA + 谱归一
                                 W <- W + eta*(x x^T - W);  W <- W / ||W||_2
                               只在 sigma1/sigma2 大时**近似**秩 1（代码阈值 RANK_TOL=1e3, :403）。
  `_model/stage3.py:455-457`   返回 '_W'=norm['W']、'_x'=norm['x']、'_g'=g。
  `_model/stage67.py:546-554`  返回 'rho_jac_B' 与 'rho_is_identity_B'。
  `result/_v16_data.json:75463` 登记值 **全精度**: "l7_rho_jac_B": 0.9411764705657704。

**T0 阶段得出的解析推论（本脚本要**实测**判定, 不许推断）**
  在**精确**不动点上 W* x* = x*, 自洽方程化为
      x* = tanh(g x*) / ||tanh(g x*)||
  逐分量即 tanh(g x*_i)/x*_i = c。因 t -> tanh(g t)/t 在 t>0 上**严格单调**,
  ⇒ **|x*_i| 在其支撑集上必须全相等**。设支撑集大小 m, 则 Sum x*_i^4 = 1/m, 于是
      **rho_B = g * (1 - Sum x*_i^4) = g * (1 - 1/m)**
  **旁证（跑之前就有的）**: 登记值 0.9411764705657704 与 1 - 1/17 = 0.9411764705882353
  相差 **-2.2465e-11**（相对 3.8e-10）⇒ m = 17 已被**现有登记数据**在 11 位有效数字上证实。
  ⇒ **本脚本的预登记预测: 落 `J-恒等式`**。
  于是 T1 剩下的**唯一**信息量在腿 B: 这个解析式**是否参数无关**（那才是升级条件本身）。
  **预测归预测 —— 结论一律由实测出。**

**本脚本不做的事（预登记）**
  1. **不改门槛去迁就结论**: 容差由**实测**的秩 1 缺陷量反推（见阶段 B）, 不写死 1e-10
     —— 1e-10 只有**精确**秩 1 才可能满足, 写死它等于把 `J-不成立` 做成一条
     **结构上不可达**的支（v16.5·D12 的 PRED_TOL 教训: 门槛写错就照记, 不许移动球门）。
  2. **不动分母、不改评级**: 即使落 `J-解析成立`, 本版也只登记, 交用户拍板（DP-18）。
  3. **不进 runner**（DP-14 默认）。
  4. **不碰 MERA、不碰 metric_***（隔离声明, 见文末）。

======================================================================
**预登记（跑之前写死; 判据与落支不许事后改）**
======================================================================

口径: `N = 17`（生产派生宽度; `_model/main.py:115` `N_self = l3['N_selfref']`）、
      `steps = 8000`、`eta = 0.05`、`g = 1.0`、`seed = 0`（后三者即默认, `main.py:119` 未传）。

阶段 A · **先复现**（D9 教训: 先复现登记值再扫参数, 不许跳）
  `stage7_consciousness(selfref_coupled(N=17))['rho_jac_B']`
  必须复现 `result/_v16_data.json:75463` 的全精度登记值 **0.9411764705657704**。
  容差 `TOL_REPRO = 1e-9`（登记值全精度, 故可收紧; 只留浮点/BLAS 级余量）。
  **不复现 ⇒ 落 `J-复现失败`, EXIT=3, 不进阶段 B。**
  （若不复现, 只**附带打印** seed=1/2 的值供诊断, **不据此改判**。）

阶段 B · 闭式、等权度与秩 1 缺陷（腿 A = 生产参数单点）
  从**真实返回的** `coupled['_x']` 算:
    S4       = Sum_i x*_i^4
    rho_pred = g * (1 - S4)
    supp     = {i : |x*_i| > 1e-8};   m = |supp|
    eqw_dev  = max_{i in supp} | |x*_i| - 1/sqrt(m) |
    defect   = ||W* - x* x*^T||_2 / ||W*||_2      （秩 1 缺陷, 无量纲）
    sratio   = coupled['norm']['sigma_ratio']     （真实返回的 sigma1/sigma2）
  自检 (i)（**复现自检**, 不是复制生产逻辑）: 按 `_rho_jac` 的写法从 _W/_x 重算
    max|eig(g·diag(1-x*^2)·W*)|, 与**返回的** rho_jac_B 比 —— 证明"我对代码的读法"没错。
  容差: `TOL_CLOSED = max(1e-9, 10 * defect)`  ← **由实测缺陷量反推**, 不用固定值。
        理由: 闭式只在精确秩 1 下严格成立, 缺陷有多大, 容差就放多大。

阶段 C · 参数相关性（腿 B —— **升级条件点名的那一条**）
  对预登记网格 `G_SCAN x SEED_SCAN` 逐点重跑阶段 B 的算式:
    `G_SCAN = (0.8, 1.0, 1.2, 1.5)`、`SEED_SCAN = (0, 1, 2)`。
  单点用时 > `T_EXT_MAX` 秒 ⇒ **停在该点并如实登记**, 不硬跑。

落支（**按此顺序判, 先中先得**）:
  `J-复现失败`  阶段 A 不过 ⇒ 不报任何结论
  `J-不成立`    腿 A 上 |rho_B - rho_pred| > TOL_CLOSED
                ⇒ 要么 W* 偏离秩 1 太多, 要么推导错 ⇒ 如实登记, 不强行归类
  `J-恒等式`    闭式成立 **且** 全程 `eqw_dev <= 1e-6`（x* 在其支撑集上等权）
                ⇒ rho_B = g(1 - 1/m) 是**支撑集大小的换写** ⇒ **不是独立测量**
                ⇒ **零升级**; 由"担心"升级为"**实测**"
  `J-参数依赖`  闭式在腿 A 成立, 但在 `G_SCAN x SEED_SCAN` 的**某些点**上不成立
                （或 x* 的等权性随参数翻转）⇒ 坐实"特定参数下的产物" ⇒ **零升级**
  `J-解析成立`  闭式在**腿 A 与腿 B 全部点**成立 **且** `eqw_dev > 1e-6` **全程**
                ⇒ 升级条件「参数无关的解析预期」**满足** ⇒ **L7 C→B 候选**, 交用户拍板（DP-18）

**升级预期：低**（登记值已把 m=17 钉在 11 位有效数字上）。但**无论落哪支, 结论都由实测出。**
"""

import os
import sys
import time

_T_PROC = time.time()   # ← 必须早于门面 import：否则会漏掉 torch/quimb 的 ~10-40 s

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np      # noqa: E402

import spiral_model_v16 as G   # noqa: E402

_T_IMPORT = time.time()
_T0 = time.time()

# ── 登记值（全精度, 逐字抄自 `result/_v16_data.json:75463`）──────────
RHO_B_REF = 0.9411764705657704
TOL_REPRO = 1e-9        # 登记值是全精度 ⇒ 只留浮点/BLAS 级余量
TOL_EQW = 1e-6          # "在其支撑集上等权"的判据
SUPP_TOL = 1e-8         # 支撑集判据 |x*_i| > SUPP_TOL

N_REF = 17
STEPS = 8000
ETA = 0.05
SEED_REF = 0
G_REF = 1.0

G_SCAN = (0.8, 1.0, 1.2, 1.5)
SEED_SCAN = (0, 1, 2)
T_EXT_MAX = 300.0       # 秒；腿 B 单点超过就停并如实登记

# 门面是否导出这两个函数（不导出则走 _model 直取 —— 同一个函数对象，不是复制）
_selfref = getattr(G, 'selfref_coupled', None)
_stage7 = getattr(G, 'stage7_consciousness', None)
_IMPORT_NOTE = '门面'
if _selfref is None or _stage7 is None:
    from _model.stage3 import selfref_coupled as _selfref          # noqa: E402
    from _model.stage67 import stage7_consciousness as _stage7     # noqa: E402
    _IMPORT_NOTE = '_model 直取（门面未导出）'


def _diag(coupled, g):
    """从**真实返回的** _W/_x 算闭式所需的一切。不重跑动力学。"""
    W = coupled['_W'].detach().numpy()
    xs = coupled['_x'].detach().numpy()
    S4 = float(np.sum(xs ** 4))
    rho_pred = float(g * (1.0 - S4))
    supp = np.abs(xs) > SUPP_TOL
    m = int(np.count_nonzero(supp))
    uniq = 1.0 / np.sqrt(m)
    eqw_dev = float(np.max(np.abs(np.abs(xs[supp]) - uniq))) if m else float('nan')
    Wx = np.outer(xs, xs)
    defect = float(np.linalg.norm(W - Wx, 2) / np.linalg.norm(W, 2))
    sratio = float(coupled['norm']['sigma_ratio'])
    return {'W': W, 'x': xs, 'S4': S4, 'rho_pred': rho_pred, 'm': m,
            'eqw_dev': eqw_dev, 'defect': defect, 'sratio': sratio}


def _leg(g, seed):
    """跑一次真实通路：selfref_coupled -> stage7_consciousness，取真实读数。"""
    t = time.perf_counter()
    coupled = _selfref(N=N_REF, steps=STEPS, eta=ETA, g=g, seed=seed)
    s7 = _stage7(coupled)
    dt = time.perf_counter() - t
    d = _diag(coupled, g)
    d['rho_B'] = float(s7['rho_jac_B'])
    d['is_id_B'] = bool(s7['rho_is_identity_B'])
    d['dt'] = dt
    return d


def main():
    print('=' * 76)
    print('v16.6 · T1 —— L7 的「B 支参数无关的解析预期」真的存在吗？')
    print('  N=%d, steps=%d, eta=%g; g/seed 走默认（口径同 _model/main.py:119）'
          % (N_REF, STEPS, ETA))
    print('  不进 runner; 不动分母; 预登记预测: 落 `J-恒等式`')
    print('  函数来源: %s' % _IMPORT_NOTE)
    print('=' * 76)

    # ── 阶段 A：复现登记值 ────────────────────────────────────────────
    print('\n-- 阶段 A: 复现登记值 (g=%.1f, seed=%d) --' % (G_REF, SEED_REF))
    a = _leg(G_REF, SEED_REF)
    dev_repro = abs(a['rho_B'] - RHO_B_REF)
    print('  rho_jac_B 实测 = %.16f' % a['rho_B'])
    print('  登记 (data.json:75463) = %.16f    |差| = %.3e  (<= %g)  %s'
          % (RHO_B_REF, dev_repro, TOL_REPRO,
             'OK' if dev_repro <= TOL_REPRO else 'FAIL'))
    print('  用时 %.2f s' % a['dt'])
    if dev_repro > TOL_REPRO:
        print('\n  [诊断, 不改判] 换 seed 只打印，不据此改判落支:')
        for s in (1, 2):
            try:
                b = _leg(G_REF, s)
                print('    seed=%d: rho_jac_B=%.16f  |差|=%.3e'
                      % (s, b['rho_B'], abs(b['rho_B'] - RHO_B_REF)))
            except Exception as e:      # noqa: BLE001
                print('    seed=%d: 失败 %s' % (s, e))
        print('\n-- 落支 --')
        print('  落支 = **J-复现失败** ⇒ 不报任何归因结论，EXIT=3')
        print('\n用时：门面 import %.1f s + 计算 %.1f s = 总计 %.1f s'
              % (_T_IMPORT - _T_PROC, time.time() - _T0, time.time() - _T_PROC))
        print('=' * 76)
        return 3

    # ── 阶段 B：闭式 / 等权度 / 秩 1 缺陷（腿 A）─────────────────────
    print('\n-- 阶段 B: 闭式与等权度（腿 A）--')
    W, xs = a['W'], a['x']
    J = G_REF * (1.0 - xs ** 2)[:, None] * W
    rho_check = float(np.max(np.abs(np.linalg.eigvals(J))))
    tol_closed = max(1e-9, 10.0 * a['defect'])
    dev_closed = abs(a['rho_B'] - a['rho_pred'])
    print('  Sum x*_i^4          = %.12f' % a['S4'])
    print('  支撑集大小 m         = %d   (判据 |x*_i| > %g)' % (a['m'], SUPP_TOL))
    print('  等权度 eqw_dev       = %.3e   (判据 <= %g 即"支撑集上等权")'
          % (a['eqw_dev'], TOL_EQW))
    print('  秩 1 缺陷 defect     = %.3e   sigma1/sigma2 = %.3e'
          % (a['defect'], a['sratio']))
    print('  rho_B 返回值         = %.12f' % a['rho_B'])
    print('  rho_pred = g(1-S4)   = %.12f' % a['rho_pred'])
    print('  |rho_B - rho_pred|   = %.3e   (TOL_CLOSED = %.3e = 10*defect)'
          % (dev_closed, tol_closed))
    print('  [自检 i] 按 _rho_jac 重算 max|eig(J)| = %.12f   |差| = %.3e'
          % (rho_check, abs(rho_check - a['rho_B'])))
    print('  [旁证]   g(1 - 1/m)   = %.16f   vs 登记值 差 %.3e'
          % (G_REF * (1.0 - 1.0 / a['m']), abs(G_REF * (1.0 - 1.0 / a['m']) - RHO_B_REF)))

    ok_closed_A = dev_closed <= tol_closed

    # ── 阶段 C：参数相关性（腿 B）────────────────────────────────────
    print('\n-- 阶段 C: 参数相关性（腿 B, g x seed = %d x %d）--'
          % (len(G_SCAN), len(SEED_SCAN)))
    rows = []
    stopped = None
    for g in G_SCAN:
        for s in SEED_SCAN:
            r = _leg(g, s)
            tcl = max(1e-9, 10.0 * r['defect'])
            dc = abs(r['rho_B'] - r['rho_pred'])
            ok = dc <= tcl
            rows.append((g, s, r, dc, tcl, ok))
            print('  g=%.1f seed=%d: rho_B=%.6f rho_pred=%.6f |差|=%.2e tol=%.2e %s'
                  '  m=%d eqw_dev=%.2e defect=%.2e  用时 %.2f s'
                  % (g, s, r['rho_B'], r['rho_pred'], dc, tcl,
                     'OK ' if ok else 'BAD', r['m'], r['eqw_dev'],
                     r['defect'], r['dt']))
            if r['dt'] > T_EXT_MAX:
                stopped = (g, s)
                print('  => 用时 %.2f s > T_EXT_MAX=%.1f s ⇒ **停在此点，不再往上**'
                      % (r['dt'], T_EXT_MAX))
                break
        if stopped:
            break

    n_bad = sum(0 if r[5] else 1 for r in rows)
    eqw_max = max(r[2]['eqw_dev'] for r in rows)
    m_set = sorted(set(r[2]['m'] for r in rows))
    print('\n  腿 B 共 %d 点；闭式不成立的点 = %d；全程最大 eqw_dev = %.3e；m 取值 = %s'
          % (len(rows), n_bad, eqw_max, m_set))

    # ── 落支 ─────────────────────────────────────────────────────────
    print('\n-- 落支 --')
    if not ok_closed_A:
        branch = 'J-不成立'
        why = ('腿 A 上闭式就差（|rho_B-rho_pred| > TOL_CLOSED）'
               '⇒ W* 偏离秩 1 太多，或推导错 ⇒ 如实登记，不强行归类')
    elif eqw_max <= TOL_EQW:
        branch = 'J-恒等式'
        why = ('闭式成立且 x* 在其支撑集上等权 ⇒ rho_B = g(1-1/m) 是'
               '**支撑集大小的换写**，不是独立测量 ⇒ **零升级**')
    elif n_bad > 0:
        branch = 'J-参数依赖'
        why = ('闭式在腿 A 成立，但腿 B 的 %d/%d 点不成立'
               '⇒ "特定参数下的产物" ⇒ **零升级**'
               % (n_bad, len(rows)))
    else:
        branch = 'J-解析成立'
        why = ('闭式在腿 A 与腿 B **全部点**成立且 x* 全程非等权'
               '⇒ 升级条件「参数无关的解析预期」**满足** ⇒ **L7 C→B 候选**（DP-18）')
    print('  落支 = **%s**' % branch)
    print('  %s' % why)
    if stopped:
        print('  ⚠️ 腿 B 在 (g=%.1f, seed=%d) 处因用时停跑 —— 扫描范围**不完整**，'
              '落支按已跑点判，须同页读。' % stopped)
    if branch == 'J-恒等式':
        print('  ⇒ 那句"特定参数下的产物"由**担心**升级为**实测**；')
        print('     是否给 B 支补 `is_identity=True` 标注见 DP-19（本版只登记，不改代码）。')
    print('  **升级预期：低**。L7 维持 C 除非落 `J-解析成立` 且用户批 DP-18。')
    print('\n[T1] 隔离自检：未调用 metric_* / 未碰 MERA；未动既有读数与分母。')
    print('用时：门面 import %.1f s + 计算 %.1f s = 总计 %.1f s'
          % (_T_IMPORT - _T_PROC, time.time() - _T0, time.time() - _T_PROC))
    print('=' * 76)
    return 0


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.exit(main())
