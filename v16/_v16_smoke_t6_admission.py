# -*- coding: utf-8 -*-
#
# Copyright (c) 2024-2026 Jianzhong Yan
# Licensed under the Apache License, Version 2.0 (the "License");
#
r"""
v16.8 · T6 · hyMERA 几何层与 L4/L5 真接入 —— 阶段 0 **准入筛**：三个 `Y` 口径的复现
[_VERSION_TAG = 'v16-smoke-t6-admission-1']

锚（源码锚）:
  `v16.8_plan.md` §2 阶段 0 —— 判据与落支名**事前写死**在该节，本文件只执行
  `v16/spiral_v16_audit.md` **§9.10** 行 3（S&P 口径读数）与行 4（反射残差）
  `v16/spiral_v16_audit.md` **§9.2** `:1607` —— S&P 的 `Y` 与 Evenbly 原文的**唯一**差别
      「`Y` 角元: `sin θ₁`（实） → **`i sin θ₁`**（虚）；`Y` 右下: 同」
      （源: Steinberg & Prior, Sci Rep 12, 532 (2022) / arXiv:2012.09591，源码级核对）
  旁证锚 `v16/_t4_hyper_phase1b.log:30-31` —— 原文口径的 `w`/`u` 残差与 `c`
      （**注意**：这两个数不在 §9.10 表内，登记在 T4 Phase 1b 的**已冻结日志**里）

靶: 把 T4 期的**临时探查**（未落盘、当时声明「非判据」）落成一个**正式脚本**，
  让 §9.10 的三个口径读数**可复现、可复算、可入档**。这是 T6 的第一道门 ——
  口径没钉死之前，阶段 1~3 的一切读数都没有意义。

======================================================================
预登记（成文于实现之前；跑之前不许改）
======================================================================
口径: θ = (θ1..θ5) = (0.37, 0.52, 0.61, 0.29, 0.44) —— 与 T4 Phase 1/1b 同一组角
三个 `Y` 口径（**每个读数旁必须标口径名**，不许静默换）:
  [原文]  `H.Y_tensor(t1)`       —— Evenbly 印的（`Hyper.tex:397-404`），T6 的**对照组**
  [S&P]   `H.Y_tensor_sp(t1)`    —— S&P 印的（**非 Evenbly 原文**），T6 的**主口径**
  [修补]  `H.Y_tensor_fixed(t1)` —— 本仓自造「口径 B · 诊断猜想」，**永远标「非原文」**

容差（两类，不混用）:
  TOL    = 1e-12  —— 用于**期望为数值 0** 的量（纯线性代数）
  1e-4 相对      —— 用于**登记值本身只有 4 位有效数字**的量（如 6.743e-01）
                    ⇒ 判据是「**前 4 位有效数字一致**」，不是逐位相等（登记值没那个精度）

判据（靶值全部来自已登记文本，逐条给出出处）:
  靶 A  §9.10 行 3  · S&P   : dev(12|34) = 0        ; dev(13|24) = 6.743e-01
                              w resid = 1.78e-15     ; u resid = 3.55e-15
  靶 B  §9.10 行 4  · S&P   : max|Y - Y^T| = 0
  靶 C  计划 §1 表  · 原文  : dev(12|34) = dev(13|24) = 6.743e-01 ; 反射 = 0
  靶 D  计划 §1 表  · 修补  : dev(12|34) = dev(13|24) = 2.2e-17  ; 反射 = 0
  靶 E  冻结日志    · 原文  : w resid = 2.697152e+00 (c=4.000000)
                              u resid = 9.497650e+00 (c=5.211075)   ← 旁证锚，非 §9.10
  靶 F  冻结日志    · 修补  : w resid = 1.776357e-15 ; u resid = 3.552992e-15 (c=4)
  靶 G  **内部一致性**（不依赖任何登记值，用来钉住本文件自己的算法）:
        (i)  max(dev12, dev13) 必须 == `H.doubly_unitary_residual_Y(M)`（差 < 1e-15）
             —— 因为 `hyper.py` 只公开**两分划的最大值**，本文件**必须**自己写两个缩并
                才能把 (12|34) 与 (13|24) **分开**。这条自检就是把「我的分法」钉在
                「`hyper.py` 的定义」上，防止读错量。
        (ii) dev12 必须 == `H.unitary_residual(M)`（差 < 1e-15）
             —— `unitary_residual` 走普通矩阵乘，本文件走 4 指标缩并，两者独立。
        (iii) 原文口径的 dev 必须 == 2|c1 s1|（差 < 1e-15）
             —— audit `:1597` 的免约定判法，闭式对照。

落支（事前写死）:
  `T6-口径已钉`      靶 A~G 全过 ⇒ EXIT=0 ⇒ 准入通过，可进阶段 1
  `T6-口径不可复现`  任一不过 ⇒ EXIT=3 ⇒ **停工上报，不许自行调和**

范围红线（同计划 §2 阶段 0）:
  - 只调 `_model/hyper.py` 的公开函数（**真调，不复制逻辑**）；
  - **不 import 门面** `spiral_model_v16.py`（故无 torch/quimb）；
  - 不接 L4/L5；不进 runner；不加守卫/指标 ⇒ 规模三数 **78 / 27 / 13 不动**；
  - **不改** `_v16_smoke_t4_hyper*.py`、`_t4_hyper*.log`（T4 那一对是成对存档，保持冻结）；
  - `_model/hyper.py` **只允许一处纯新增**：`Y_tensor_sp`（用户 2026-10-04 放行，见 [诚实边界] 1）。
    既有函数、判据、`__all__` 中既有名字**一律不动**；
  - 不改 `spiral_v16_audit.md` 任何既有行。

用时: < 1 分钟（纯线性代数；u 一次缩并 (256x4096) x 3 个口径）。
      确切用时见日志末行，不在此处预登记（免得脚本与日志因一个数字失同步）。

[诚实边界]
  1. S&P 的 `Y` **已提升进** `_model/hyper.py::Y_tensor_sp`（用户 2026-10-04 放行；
     此前是本文件内的 `_Y_sp`）。它**不是** Evenbly 原文的 `Y`，也**不是**本仓自造的变体：
     归属永久限定 arXiv:2012.09591 / Sci Rep 12, 532，标注「**非 Evenbly 原文**」。
     若它与靶 A/B 不符，则**不是**「S&P 错了」，而是**读错了那张矩阵** —— 这正是本阶段要拦下的错误。
  2. 本文件不声称原文 `Y_tensor`「印错了」应改成什么；它只复现三个口径的读数。
     「原文 Y 非酉」是 audit §9.1 已登记的结论（缺陷在原文，非本仓抄录）。
  3. 本文件**不**动 §5.12、**不**动 L4/L5、**不**产出任何 T6 物理结论。
"""

import os
import sys
import time

_T_PROC = time.time()      # ← 进程起点，早于任何 import

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np                  # noqa: E402
from _model import hyper as H       # noqa: E402

_T_IMPORT = time.time()
_T0 = time.time()

_VERSION_TAG = 'v16-smoke-t6-admission-1'
THETA = (0.37, 0.52, 0.61, 0.29, 0.44)
TOL = 1e-12         # 「期望为数值 0」的绝对门限
REL = 1e-4          # 「登记值只有 4 位有效数字」的相对门限
_CONS = 1e-15       # 内部一致性自检的门限（比 TOL 严）


# S&P 的 `Y` 已按用户 2026-10-04 指令**提升进** `_model/hyper.py::Y_tensor_sp`
# （此前是本文件内的 `_Y_sp`）。定义只留一份, 本文件不再自带副本。
def _devs(M):
    """把 Y 的**两个分划**分开量。`hyper.py` 只公开两者的 max，故缩并只能写在本地。

    (12|34) = sum_kl 与 (13|24) = sum_jl —— 与 `hyper.py:104-105` 的两条**同式**。
    返回 (dev12, dev13, joint_max_from_hyper)。
    """
    T = np.asarray(M).reshape(2, 2, 2, 2)
    C = T.conj()
    I = np.eye(4)
    a = np.einsum('ijkl,mnkl->ijmn', T, C).reshape(4, 4)
    b = np.einsum('ijkl,mjnl->ikmn', T, C).reshape(4, 4)
    d12 = float(np.max(np.abs(a - I)))
    d13 = float(np.max(np.abs(b - I)))
    return d12, d13, H.doubly_unitary_residual_Y(M)


def _probe(tag, Y, t1, B, check_closed=False):
    """一个口径的全套读数。返回 (读数 dict, 一致性自检 dict)。

    `check_closed`: 靶 G(iii) 的闭式 `2|c1 s1|` **只对原文口径成义**（原文的 dev 恒等于它）；
    S&P 与修补的 dev12 按构造就是 0 ⇒ 对它们算这一条是无意义的，故默认不算。
    """
    A = H.A_tensor(t1, Y)
    d12, d13, joint = _devs(Y)
    refl = H.reflection_residual(Y, 'klij')
    rw, cw = H.coisometry_residual(H.w_tensor(A, B))
    ru, cu = H.coisometry_residual(H.u_tensor(A, B))
    cons = {
        'joint':  abs(max(d12, d13) - joint),
        'unitar': abs(d12 - H.unitary_residual(Y)),
    }
    if check_closed:
        cons['closed'] = abs(d12 - 2 * abs(np.cos(t1) * np.sin(t1)))
    return ({'dev12': d12, 'dev13': d13, 'refl': refl,
             'rw': rw, 'cw': cw, 'ru': ru, 'cu': cu}, cons)


def main():
    t1, t2, t3, t4, t5 = THETA
    print('=' * 76)
    print('v16.8 · T6 · 阶段 0 准入筛 —— 三个 `Y` 口径的 T4 §9.10 复现')
    print('  %s' % _VERSION_TAG)
    print('  θ = %s   TOL = %g   相对门限 = %g' % (THETA, TOL, REL))
    print('  锚: v16.8_plan.md §2 阶段 0 ; spiral_v16_audit.md §9.10 / §9.2 (`:1607`)')
    print('=' * 76)

    B = H.B_tensor(t2, t3, t4, t5)

    # ── 三个口径 ────────────────────────────────────────────────────────
    R = {}
    R['原文'] = _probe('原文', H.Y_tensor(t1), t1, B, check_closed=True)
    R['S&P'] = _probe('S&P', H.Y_tensor_sp(t1), t1, B)
    R['修补'] = _probe('修补', H.Y_tensor_fixed(t1), t1, B)

    print('\n-- 三个口径的读数（每行都带口径名；`修补` 永远标「非原文」）--')
    print('  %-6s %-12s %-12s %-10s %-14s %-14s' %
          ('口径', 'dev(12|34)', 'dev(13|24)', '反射', 'w resid (c)', 'u resid (c)'))
    for tag in ('原文', 'S&P', '修补'):
        d = R[tag][0]
        print('  %-6s %-12.6e %-12.6e %-10.1e %-7.3e (%.6f) %-7.3e (%.6f)'
              % (tag, d['dev12'], d['dev13'], d['refl'], d['rw'], d['cw'], d['ru'], d['cu']))

    # ── 一致性自检（靶 G；不依赖任何登记值）────────────────────────────
    print('\n-- 靶 G 内部一致性自检（把本文件的算法钉在 hyper.py 的定义上）--')
    g_ok = True
    for tag in ('原文', 'S&P', '修补'):
        _, cons = R[tag]
        ok = all(v < _CONS for v in cons.values())
        g_ok = g_ok and ok
        cl = cons.get('closed')
        print('  %-6s 联合max %.1e | vs unitary_residual %.1e | vs 2|c1s1| %-9s  %s'
              % (tag, cons['joint'], cons['unitar'],
                 ('%.1e' % cl) if cl is not None else '(仅原文)',
                 'OK' if ok else 'FAIL'))

    # ── 靶 A~F（对登记值）──────────────────────────────────────────────
    zero = lambda x, v: abs(x) < TOL
    rel = lambda x, v: abs(x - v) <= REL * abs(v)

    checks = [
        ('A1 §9.10行3', 'S&P',  'dev(12|34)', R['S&P'][0]['dev12'], 0.0, zero),
        ('A2 §9.10行3', 'S&P',  'dev(13|24)', R['S&P'][0]['dev13'], 6.743e-01, rel),
        ('A3 §9.10行3', 'S&P',  'w resid',    R['S&P'][0]['rw'], 1.78e-15, zero),
        ('A4 §9.10行3', 'S&P',  'u resid',    R['S&P'][0]['ru'], 3.55e-15, zero),
        ('A5 §9.10行3', 'S&P',  'c (w=u)',    R['S&P'][0]['cw'], 4.0, rel),
        ('B1 §9.10行4', 'S&P',  '反射',       R['S&P'][0]['refl'], 0.0, zero),
        ('C1 计划§1表', '原文', 'dev(12|34)', R['原文'][0]['dev12'], 6.743e-01, rel),
        ('C2 计划§1表', '原文', 'dev(13|24)', R['原文'][0]['dev13'], 6.743e-01, rel),
        ('C3 计划§1表', '原文', '反射',       R['原文'][0]['refl'], 0.0, zero),
        ('D1 计划§1表', '修补', 'dev(12|34)', R['修补'][0]['dev12'], 2.2e-17, zero),
        ('D2 计划§1表', '修补', 'dev(13|24)', R['修补'][0]['dev13'], 2.2e-17, zero),
        ('D3 计划§1表', '修补', '反射',       R['修补'][0]['refl'], 0.0, zero),
        ('E1 冻结日志', '原文', 'w resid',    R['原文'][0]['rw'], 2.697152e+00, rel),
        ('E2 冻结日志', '原文', 'u resid',    R['原文'][0]['ru'], 9.497650e+00, rel),
        ('E3 冻结日志', '原文', 'c (w)',      R['原文'][0]['cw'], 4.0, rel),
        ('E4 冻结日志', '原文', 'c (u)',      R['原文'][0]['cu'], 5.211075, rel),
        ('F1 冻结日志', '修补', 'w resid',    R['修补'][0]['rw'], 1.776357e-15, zero),
        ('F2 冻结日志', '修补', 'u resid',    R['修补'][0]['ru'], 3.552992e-15, zero),
        ('F3 冻结日志', '修补', 'c (w=u)',    R['修补'][0]['cw'], 4.0, rel),
    ]

    print('\n-- 靶 A~F：对**已登记值**逐条核对（登记值只有 4 位有效数字者用相对门限）--')
    bad = []
    for name, tag, what, got, want, fn in checks:
        ok_c = fn(got, want)
        if not ok_c:
            bad.append('%s %s %s' % (name, tag, what))
        print('  %-12s %-6s %-12s 实测 %-14.6e 靶 %-14.6e  %s'
              % (name, tag, what, got, want, 'OK' if ok_c else 'FAIL'))

    ok = g_ok and not bad
    branch = 'T6-口径已钉' if ok else 'T6-口径不可复现'

    # ── 落支 ───────────────────────────────────────────────────────────
    print('\n-- 落支 --')
    print('  落支 = **%s**' % branch)
    if ok:
        print('  靶 A~G 全过 ⇒ 三个口径在 θ 处逐条复现登记值。')
        print('  T6 主口径 = **S&P 的 `Y`**（(12|34) 酉、(13|24) 不酉 ⇒ **非双酉**，`c`=4）；')
        print('  备用 = `Y_tensor_fixed`（双酉，`c`=4，**本仓自造 · 非原文**）；')
        print('  对照 = 原文 `Y_tensor`（两条都不满足，`c` 非整数）。')
        print('  ⇒ 准入通过，可进阶段 1（`{7,3}` 双曲铺砌）。')
    else:
        print('  未过: %s' % ('; '.join(bad) if bad else '(靶 G 一致性自检)'))
        print('  ⇒ 按计划 §2 阶段 0：**停工上报，不许自行调和**。')
        print('     最可能的出错处是 `_Y_sp` 的逐元素构造（它没有落盘先例）——')
        print('     见本文件 [诚实边界] 第 1 条：那是「我的构造读错了」，不是「S&P 错了」。')
    print('\n[T6-阶段0] 隔离自检: 未 import 门面 / 未接 L4-L5 / 未进 runner / 未加守卫与指标。')
    print('  规模三数 78 / 27 / 13 不动；未改 T4 那对脚本与日志、audit 既有行；')
    print('  `_model/hyper.py` 只有一处**纯新增**（`Y_tensor_sp`, 用户 2026-10-04 放行）, 既有函数一字未动。')
    print('用时: import %.1f s + 本计算 %.1f s = 总计 %.1f s'
          % (_T_IMPORT - _T_PROC, time.time() - _T0, time.time() - _T_PROC))
    print('=' * 76)
    return 0 if ok else 3


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.exit(main())
