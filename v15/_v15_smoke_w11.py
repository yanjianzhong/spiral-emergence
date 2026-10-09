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
v15·W11 冒烟 —— 只读源文件 + 一个临时副本, 不跑全流程。

验三件事:
  (1) L1 因果通路扫描: 在真实 spiral_model_v15.py 上给出 False (无消费者)。
  (2) **非恒真证明**: 往临时副本里注入一行假消费者, 扫描必须翻成 True。
      没有这一步, (1) 的 False 可能只是扫描永远返回 False。
  (3) l7_rho_jac_B 诊断: 用 result/_v15_data.json 里的实录值驱动守卫逻辑,
      确认 ok=True 且 expect_pass=False (登记但不计分)。

l7 的两个值在下面以**字面量**给出并注明出处 —— 本脚本不去解析那个 7 万行的
JSON, 因为它验的是**守卫逻辑**, 不是数据。
"""
import os
import shutil
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

# 实录值, 引自 result/_v15_data.json (l7_rho_jac / l7_rho_jac_B / l7_rho_is_identity)
C7_REAL = {'rho_jac': 1.0, 'rho_jac_B': 0.9411764705657704,
           'rho_is_identity': True}


def scan_equal_state(path):
    """与 spiral_metric_v15.py 里新守卫**逐字相同**的扫描逻辑。"""
    with open(path, encoding='utf-8') as f:
        lines = [l for l in f.read().splitlines() if 'equal_state' in l]
    consumed = any(("['equal_state']" in l) or ('.equal_state' in l)
                   for l in lines)
    return lines, consumed


def main():
    print('=' * 84)
    print(' W11 冒烟')
    print('=' * 84)

    src = os.path.join(HERE, 'spiral_model_v15.py')

    # ---------------- (1) 真实源码 ----------------
    print('\n【1】真实 spiral_model_v15.py 上的扫描')
    lines, consumed = scan_equal_state(src)
    print(f'  含 equal_state 的行数 = {len(lines)}')
    for l in lines:
        print(f'    {l.strip()[:96]}')
    print(f'  被下游取用 = {consumed}   -> 守卫判据 (有因果通路) = {consumed}')
    assert lines, '一行都没扫到, 说明扫描本身失效'
    assert consumed is False, '真实源码里 equal_state 不该有消费者'

    # ---------------- (2) 非恒真证明 ----------------
    print('\n【2】非恒真证明: 注入一个假消费者, 扫描必须翻')
    tmpd = tempfile.mkdtemp(prefix='_v15_w11_')
    try:
        dst = os.path.join(tmpd, 'spiral_model_v15.py')
        shutil.copyfile(src, dst)
        with open(dst, 'a', encoding='utf-8') as f:
            f.write("\n\n# [冒烟注入] 假装下游接上了\n"
                    "_FAKE = stage1_void(2)['equal_state']\n")
        l2, c2 = scan_equal_state(dst)
        print(f'  注入后: 行数 {len(lines)} -> {len(l2)}, '
              f'被下游取用 {consumed} -> {c2}')
        assert c2 is True, '注入假消费者后扫描没翻, 判据是恒假的'
        print('  [OK] 判据会翻 -> 非恒真')
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)
        print(f'  临时副本已删 (目录存在 = {os.path.exists(tmpd)})')

    # ---------------- (3) l7_rho_jac_B 诊断 ----------------
    print('\n【3】l7_rho_jac_B 诊断守卫逻辑 (实录值)')
    c7 = C7_REAL
    rhoA, rhoB = float(c7['rho_jac']), float(c7['rho_jac_B'])
    b_res = abs(rhoB - 1.0)
    ok = bool(b_res > 1e-9)
    print(f'  A 支 rho_jac   = {rhoA:.10f}  (rho_is_identity='
          f'{c7["rho_is_identity"]})')
    print(f'  B 支 rho_jac_B = {rhoB:.10f}  |rho_B - 1| = {b_res:.3e}')
    print(f'  守卫判据 ok = {ok},  expect_pass = False (登记但不计分)')
    assert c7['rho_is_identity'] is True and rhoA == 1.0, \
        'A 支应恒为 1.0 —— 若这里变了, "A 不可守"的论证要重写'
    assert ok is True, 'B 支应非恒等'
    print('  [OK] A 支恒等 (不可守) / B 支非恒等 (可守但弱) 都与论证一致')

    # ---------------- (4) 分母口径 ----------------
    print('\n【4】口径核对')
    print('  本包新增守卫 2 条, **全部 expect_pass=False** ——')
    print('  计分守卫数不变, 通过率分母不变。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
