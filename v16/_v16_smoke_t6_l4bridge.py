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
"""v16-smoke-t6-l4bridge-1 —— 阶段 3（接 L4）：**桥 1 可行性判据**冒烟。

来源锚（只读，不改）：
  v16/v16.8_plan.md:414-426   阶段 3 判据：桥 1 = 从 hyMERA 造出 rho(R)，用**本仓同一个**
                              `fit_central_charge` 拟合**同样的** L=16，再与 MERA 的 c 比。
  v16/_metric/solvers.py:53-70  `entanglement_curve` / `fit_central_charge` 真实现
                              —— 本脚本**真调，不复制**（阶段 3 是唯一被授权的例外）。
  v16/_model/hyper.py:170-183 `A_tensor` 三腿各 16 维 ⇒ 网络腿局域维 chi=16。
  v16.8_plan.md §0.1.5        阶段 1 补丁 V=2240 E=2856 F=617（**引用登记值，不重算**）。

靶（事前写死）：
  [3-1] 桥 1 的两个前提能否**同时**满足 ——
        (a) 「同函数」：`fit_central_charge` 是否只接受长度 2^L 的向量（局域维 2 硬编码）；
        (b) 「同 L=16」：hyMERA 的腿局域维（chi）与边界腿数是否允许一条 16 点的 2 维链。
  [3-2] 桥 2（备）由 Delta_k 反推须假设 CFT；假设唯一性不可判定 ⇒ 按计划「不做」。

落支（事前写死，DP-31）：
  `T6-L4桥成立`  —— 两个前提同时满足，且 hyMERA 的 c 落在 MERA 的 c 既有散布内；
  `T6-L4桥落空`  —— 任一前提不满足（**预登记为可接受结果，不追加投入**）。

边界声明：
  - 本脚本**不改** 78 / 27 / 13；**不改** 既有 log / audit / `_model/hyper.py` / 既有判据。
  - **不跑全流程**：只真调 `fit_central_charge` 一次（L=16 的 2^16 向量）作正对照，
    其余全是形状/异常级检查，秒级。
  - 本步**不涉及任何 Y**，故不存在「三张 Y 合并成一行」的问题。
  - 桥 1 若不可搭，**不另起炉灶**（把 L 换成边界腿数 = 换了 L，违反判据「同函数、同 L 才能比」）。
"""
import os
import sys
import time

_T_PROC = time.time()

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np                       # noqa: E402
from _model import hyper as H            # noqa: E402
from _metric.solvers import fit_central_charge   # noqa: E402  ← 真调，不复制

_T_IMPORT = time.time()

THETA = (0.37, 0.52, 0.61, 0.29, 0.44)
L_MERA = 16                              # 与 MERA 侧**同一个** L
_STAMP = '2026-10-04'
_FAIL = []
_LINES = []


def say(s):
    print(s)
    sys.stdout.flush()
    _LINES.append(s)


def _rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024.0 / 1024.0
    except Exception:
        return float('nan')


def part_3_1a():
    """(a) 同函数：局域维 2 是否硬编码在 fit_central_charge 里。"""
    say('--- [3-1a] 同函数：fit_central_charge 接受的向量长度 ---')

    # 正对照：长度 2^16 的**真态**（MERA 侧用的就是这种），L=16
    psi16 = np.ones(2 ** L_MERA, dtype=complex)
    psi16 /= np.linalg.norm(psi16)
    try:
        r = fit_central_charge(psi16, L_MERA)
        say('  长度 2^16, L=16  ->  通过, c=%.6f  rms=%.3e  (n=%d)'
            % (r['c'], r['rms'], len(r['ns'])))
        ok_a = True
    except Exception as e:                                    # pragma: no cover
        say('  长度 2^16, L=16  ->  **失败** %s: %s' % (type(e).__name__, e))
        _FAIL.append('3-1a 正对照失败')
        ok_a = False

    # 反对照：**同一长度** 65536 的向量，按「4 个 16 维腿」解释（hyMERA 的腿就是这个维数）
    chi = H.A_tensor(*THETA[:1], Y=H.Y_tensor_sp(THETA[0])).shape[0]
    psi_chi = np.ones(chi ** 4, dtype=complex)
    psi_chi /= np.linalg.norm(psi_chi)
    say('  hyMERA 腿局域维 chi=%d, 4 条腿 => 向量长度 %d（与 2^16 同长度）' % (chi, chi ** 4))
    try:
        r = fit_central_charge(psi_chi, 4)
        say('  长度 %d, L=4   ->  **竟然通过**, c=%.6f  => 局域维 2 未硬编码' % (chi ** 4, r['c']))
        ok_a = False
    except Exception as e:
        say('  长度 %d, L=4   ->  被拒: %s: %s' % (chi ** 4, type(e).__name__, e))
        say('  => `entanglement_curve` 把态 reshape 成 (2^n, 2^(L-n))，**局域维 2 硬编码**。')

    if not ok_a:
        _FAIL.append('3-1a')
    return ok_a, chi


def part_3_1b(chi):
    """(b) 同 L=16：hyMERA 的腿局域维与边界几何是否允许 16 点 2 维链。"""
    say('--- [3-1b] 同 L=16：hyMERA 的腿局域维与边界 ---')
    say('  A_tensor 输出形状 = %s  => chi = %d  (hyper.py:170-183)'
        % (str(H.A_tensor(*THETA[:1], Y=H.Y_tensor_sp(THETA[0])).shape), chi))
    say('  阶段 1 补丁（§0.1.5 登记值，**引用不重算**）：V=2240 E=2856 F=617')
    say('  => hyMERA 的边界是**环**，每条边界腿一个位点，局域维 = chi = %d' % chi)
    say('  => 造出的边界态 = 「N 个 %d 维位点」，**永远不会**是「16 个 2 维位点」。' % chi)
    ok_b = (chi == 2)
    if not ok_b:
        say('  => chi=%d != 2：桥 1 的「同 L=16」前提**结构上不可满足**。' % chi)
        _FAIL.append('3-1b')
    return ok_b


def part_3_2():
    """[3-2] 桥 2（备）—— 按计划不做，只登记理由。"""
    say('--- [3-2] 桥 2（备）：由 Delta_k 反推 c ---')
    say('  需要「hyMERA 的谱 = 某 CFT 的谱」这一**假设**；该假设的唯一性无法在本仓内判定。')
    say('  按 v16.8_plan.md:414-426「若假设不唯一则不做」=> **不做**，不登记为缺口。')


def main():
    say('=' * 72)
    say('v16-smoke-t6-l4bridge-1  阶段 3 桥 1 可行性判据   stamp=%s' % _STAMP)
    say('L_MERA = %d（与 MERA 侧同一个 L）；THETA = %s' % (L_MERA, THETA))
    say('import 用时 %.1f s' % (_T_IMPORT - _T_PROC))
    say('=' * 72)

    ok_a, chi = part_3_1a()
    ok_b = part_3_1b(chi)
    part_3_2()

    say('-' * 72)
    if ok_a and ok_b:
        say('桥 1 前提**同时满足** => 需继续做拟合并与 MERA 的 c 散布比（本脚本未含，另行开工）。')
        say('落支 = (未定)')
    else:
        say('桥 1 前提**不能同时满足** => 结构上不可搭；DP-31：落空即登记，**不追加投入**。')
        say('落支 = T6-L4桥落空')
    say('失败项 = %s' % (_FAIL if _FAIL else '无'))
    say('用时 %.2f s   RSS %.1f MB' % (time.time() - _T_PROC, _rss_mb()))
    say('=' * 72)

    with open(os.path.join(_HERE, '_t6_l4bridge.log'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(_LINES) + '\n')

    return 0 if (ok_a and ok_b) else 3


if __name__ == '__main__':
    sys.exit(main())
