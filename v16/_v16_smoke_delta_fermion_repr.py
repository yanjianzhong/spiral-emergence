# -*- coding: utf-8 -*-
#
# Copyright (c) 2024-2026 Jianzhong Yan
# Licensed under the Apache License, Version 2.0 (the "License");
#
"""
v16.4 · 工作包 δ · δ-a —— JW 约定（序 / 边界扇区）**承重吗**？
[_VERSION_TAG = 'v16-smoke-delta-fermion-repr-a-1']

**靶心（三处逐字，见 `v16.4_plan.md` §工作包 δ）**:
  a. 「每格点恰好一对 Majorana」**等价于 JW 的 site<->mode 对应，即费米子表述选择**
     （计划 §1.3）
  b. ② 的原文是**程序性**的：「预设层缺口被当作**审计对象**处理，而不是当作事实背景」
     （`spiral_v16_audit.md:347`）—— 而 α 已经做了这件事（计划 §1.2 逐字「α 命中的是条件 ②」）
  c. 现有 JW 通路**已存在**且走 **NS 扇区**，其 docstring 自己列了「周期边界那条键漏了或
     重复了，也对不上」（`_model/core.py:572`）
=> δ 就是把 (b) 从「做了动作」推到「到达状态」（DP-11 默认 = 按**实质**读）。

======================================================================
**预登记（跑之前写死；判据与落支不许事后改）**
======================================================================

口径: `L=4, J=1, h=1`（同 D5 格点）。δ-a 只用 16 维，**不碰 `4^L=256``**。
产出: **诊断**，不进 runner、不加守卫/指标 ⇒ 规模三数 78 / 27 / 13 不动（DP-8 同款）。

三个能量:
  `E_ED` = 自旋侧 `exact_ground_state(4,1,1)`             (`core.py:559`)
  `E_NS` = `jw_ground_energy(4,1,1)`, NS 扇区 `q_k=pi(2k+1)/L`  (`core.py:572`，现有)
  `E_R`  = 同一闭式换**周期扇区** `q_k=2*pi*k/L`           (**本文件新写**)

判据（四条，全部跑之前写死）:
  1. 一致性      `|E_NS - E_ED| <= 1e-9`    —— v16 已登记「吻合到 10 位」，此处是**复现**
                                                （D9 教训：先复现再扫）
  2. 扇区承重    `|E_R  - E_ED|  > 1e-9`    —— **牙齿判据**；相等 ⇒ 检验没牙齿
  3. 序约定重言  `max|P H P^T - H| <= 1e-12` —— `P` = 位点反序置换。
                 靶是 `core.py:533-534` 逐字「(v9 的 tfi_hamiltonian 用的是相反位序;
                 二者由环的反射对称性联系, 本征值完全相同, 已实测一致到 7e-15)」
                 —— 它注册的是**本征值**级，本条在**算子**级，更强。
  4. 活反事实    打坏 H（去掉绕回键 `-J sz_{L-1} sz_0`）后基态能量 `!= E_ED`

落支:
  `δ-重言` ⭐    1~4 全过 ⇒ JW 约定的**序**可自由变更而读数不变；唯一非平凡的部分
                 （**边界扇区**）由物理定（只有 NS 对上基态）⇒ **它不是可自由选的自由度**
  `δ-坐实`      存在**合法**约定给出不同读数
  `δ-检验失效`  2 或 4 不过 ⇒ **不报结论**

**必须同时登记的两条自限（预登记，非事后补）**:
  1. **本脚本的「不变」是构造性重言，不是实测发现** —— 1/3 读的都是**自旋侧**量，而自旋侧
     **根本不含 JW** ⇒ 它**不可能**变。⇒ 本项价值只在于把「JW 约定不承重」从"显然"变成
     **登记在案**，与 D7（`S-重言`）同形：**登记一条重言式，不缩小缺口**。
  2. **本脚本的设计到不了 `δ-坐实` 支** —— 要落那一支需要一条**合法**约定给出不同读数，
     而上面第 1 条说明自旋侧读数**结构上**不可能因 JW 约定而变。⇒ 该支由本脚本**不可达**，
     **如实打印，不许假装扫过了整个约定空间**。
"""

import os
import sys
import time

_T_PROC = time.time()   # ← 进程起点。**必须早于门面 import**：只从 main() 起算会把
                        #   torch/quimb 的 ~10 s 漏掉，印出一个看着像坏掉的「0.0 s」。

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np      # noqa: E402
import scipy.sparse as sp   # noqa: E402

import spiral_model_v16 as G   # noqa: E402

_T_IMPORT = time.time()
_T0 = time.time()

L_SITE = 4
J = 1.0
H_FIELD = 1.0
TOL_CONSIST = 1e-9
TOL_REFLECT = 1e-12


def _broken_H_dense():
    """打坏版：**逐句照抄** `tfi_periodic_sparse`（`core.py:536-554`）的 kron 嵌套顺序，
    唯一差别是键只取 `range(L-1)` ⇒ 绕回键 `-J sz_{L-1} sz_0` 被去掉。
    **这是"检验有没有牙齿"的活反事实。**"""
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    sz = np.array([[1.0, 0.0], [0.0, -1.0]])
    I2 = np.eye(2)

    Hx = None
    for i in range(L_SITE):
        b = L_SITE - 1 - i
        m = np.kron(np.eye(2 ** b), np.kron(sx, np.eye(2 ** (L_SITE - 1 - b))))
        Hx = m if Hx is None else Hx + m
    H = (-H_FIELD) * Hx
    for (a, b) in [(i, (i + 1) % L_SITE) for i in range(L_SITE - 1)]:   # ← 少了最后一个
        mat = 1
        for j in range(L_SITE):
            mat = np.kron(mat, sz if j in (a, b) else I2)
        H = H + (-J) * mat
    return (H + H.T) / 2


def _reflect_perm():
    """位点反序置换 P：在 `core.py:529-534` 的约定（site i <-> 位 L-1-i）下，
    site i -> L-1-i 就是**位串整体反序**。"""
    D = 2 ** L_SITE
    rev = np.array([int(format(b, '0%db' % L_SITE)[::-1], 2) for b in range(D)])
    P = np.zeros((D, D))
    P[rev, np.arange(D)] = 1.0
    return P


def _S_half(psi):
    half = L_SITE // 2
    s = np.linalg.svd(np.asarray(psi).reshape(2 ** half, 2 ** (L_SITE - half)),
                      compute_uv=False)
    p = s ** 2
    p = p[p > 1e-15]
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def main():
    print('=' * 76)
    print('v16.4 · δ-a —— JW 约定（序 / 边界扇区）承重吗？')
    print('  L=%d, J=%g, h=%g（同 D5 格点；只用 16 维，不碰 256）' % (L_SITE, J, H_FIELD))
    print('=' * 76)

    # ── 三个能量 ────────────────────────────────────────────────────────
    E_ED, gs = G.exact_ground_state(L_SITE, J, H_FIELD)
    E_NS = G.jw_ground_energy(L_SITE, J, H_FIELD)
    q_r = 2.0 * np.pi * np.arange(L_SITE) / L_SITE
    E_R = float(-np.sum(np.sqrt(J ** 2 + H_FIELD ** 2
                                 - 2.0 * J * H_FIELD * np.cos(q_r))))
    print('\n-- 三个能量 --')
    print('  E_ED (自旋侧稀疏对角化)        = %.12f' % E_ED)
    print('  E_NS (JW, NS 扇区 pi(2k+1)/L)  = %.12f   差 %.3e'
          % (E_NS, abs(E_NS - E_ED)))
    print('  E_R  (JW, 周期扇区 2*pi*k/L)   = %.12f   差 %.3e'
          % (E_R, abs(E_R - E_ED)))

    # ── 判据 1 / 2 ──────────────────────────────────────────────────────
    c1 = abs(E_NS - E_ED) <= TOL_CONSIST
    c2 = abs(E_R - E_ED) > TOL_CONSIST
    print('\n  [1] 一致性   |E_NS-E_ED|=%.3e <= %g  %s'
          % (abs(E_NS - E_ED), TOL_CONSIST, 'OK' if c1 else 'FAIL'))
    print('      （复现 v16 登记的「吻合到 10 位」）')
    print('  [2] 扇区承重 |E_R -E_ED|=%.3e >  %g  %s   （牙齿判据）'
          % (abs(E_R - E_ED), TOL_CONSIST, 'OK' if c2 else 'FAIL'))
    print('      => 只有 NS 对上基态 => 边界扇区**由物理定**，不是可自由选的')

    # ── 判据 3：序约定重言 ──────────────────────────────────────────────
    Hm = np.asarray(G.tfi_periodic_sparse(L_SITE, J, H_FIELD).todense())
    P = _reflect_perm()
    dev = float(np.max(np.abs(P @ Hm @ P.T - Hm)))
    c3 = dev <= TOL_REFLECT
    print('\n  [3] 序约定   max|P H P^T - H| = %.3e <= %g  %s'
          % (dev, TOL_REFLECT, 'OK' if c3 else 'FAIL'))
    print('      靶: core.py:533-534 逐字「二者由环的反射对称性联系, 本征值完全相同,')
    print('      已实测一致到 7e-15」—— 那条是**本征值**级，本条是**算子**级，更强。')

    # ── 判据 4：活反事实（打坏 H） ──────────────────────────────────────
    Eb = float(np.linalg.eigvalsh(_broken_H_dense())[0])
    c4 = abs(Eb - E_ED) > TOL_CONSIST
    print('\n  [4] 活反事实 打坏 H（去掉绕回键）后 E0 = %.12f   差 %.3e  %s'
          % (Eb, abs(Eb - E_ED), 'OK' if c4 else 'FAIL'))
    print('      => 打坏了就不相等 => **这套检验有牙齿**（不是"怎么算都一样"）')

    # ── 重言的自证：读出量根本不看 JW ───────────────────────────────────
    print('\n-- 重言的自证：下游读出根本不看 JW --')
    print('  S(L/2) 由**自旋侧**基态算出 = %.10f' % _S_half(gs))
    print('  它不含任何 JW 约定输入 => 换 JW 约定**在结构上**不可能改变它。')

    # ── 落支 ────────────────────────────────────────────────────────────
    print('\n-- 落支 --')
    failed = [n for n, ok in zip(('[1]', '[2]', '[3]', '[4]'), (c1, c2, c3, c4))
              if not ok]
    if failed:
        branch = 'δ-检验失效'
        print('  落支 = **%s**（未过：%s）=> 不报任何结论'
              % (branch, ', '.join(failed)))
    else:
        branch = 'δ-重言'
        print('  落支 = **%s**' % branch)
        print('  => JW 约定的**序**可自由变更而读数不变；唯一非平凡的部分')
        print('     （**边界扇区**）由物理定（只有 NS 对上基态）')
        print('     => **它不是可自由选的自由度**。')
    print('  [自限 1] 本条「不变」是**构造性重言**（自旋侧不含 JW）=> 与 D7 `S-重言`')
    print('     同形，**登记一条重言式，不缩小缺口**。')
    print('  [自限 2] 本脚本**到不了 `δ-坐实` 支** —— 那需要一条合法约定给出不同读数，')
    print('     而自旋侧读数结构上不可能因 JW 约定而变。**该支未扫，如实登记。**')
    print('\n[δ-a] 隔离自检：未调用 stage2_critical_break / 未接主调用；未动既有读数。')
    print('用时：门面 import %.1f s + 本计算 %.1f s = 总计 %.1f s'
          % (_T_IMPORT - _T_PROC, time.time() - _T0, time.time() - _T_PROC))
    print('=' * 76)
    return 0


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.exit(main())
