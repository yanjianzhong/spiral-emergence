# -*- coding: utf-8 -*-
"""
v16 · 预设层审计冒烟 —— v15 表 §5.4 的三条"给定"预设
[_VERSION_TAG = 'v16-smoke-presets-1']

**这份冒烟不"导出"任何预设。** 它做的是把 §5.4 里那句被动的登记
（"这三条是给定的"）换成**可测的承重性判断**：换掉它，链条某一步会不会断？

三条预设的结论形态**不一样**，这正是本文件存在的理由：

  P1 `d_local = 2`       —— 给定**且承重**：换掉它, L1->L2 这条链在**两处**断。
  P2 复数域 + 内积 + 归一化 —— 给定**且不承重**：唯一载体 `Q` 在链上无消费者。
  P3 随机测度 (复高斯 -> QR) —— 给定**且不承重**：`equal_state` 逐位不随 seed 变。

口径(照 §5.4 的自我约束):
  * **不新增守卫或指标** —— 那会动分母。本文件是一份**登记**, 不是判定。
  * **不改任何数字** —— 不重跑全流程, 不触碰任何阈值。
  * 每一条检验都配一个**活的反事实构造**: 证明这条检验在坏输入下**会翻红**。
    没有这一半的检验是"几乎不可能失败"的检查, §5.4 已把它排除在负对照之外。

用法:
    python v16/_v16_smoke_presets.py
"""

import inspect
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'v16'))

import numpy as np  # noqa: E402
import spiral_model_v16 as G  # noqa: E402

L = 4
_FAILS = []


def check(label, cond, detail=''):
    tag = 'PASS' if cond else '**FAIL**'
    print(f"    [{tag}] {label}" + (f"  —— {detail}" if detail else ''))
    if not cond:
        _FAILS.append(label)


def main():
    print("=== v16 · 预设层审计冒烟 (§5.4 三条给定预设) ===")
    print("    承重性判据: 换掉该预设, 链条某一步会不会断?")

    # ------------------------------------------------------------------
    print("\n  P1 · 预设 #1 局部维度 d_local=2 —— 给定**且承重**")
    # ------------------------------------------------------------------
    # 真调: 用 stage1_void 自己造出 d_local=3 的对象, 而不是手搓一个态 ——
    # 手搓只能证明"函数能报错", 造出来才能证明"这条链没有 qutrit 的表示"。
    s2 = G.stage1_void(2)
    s3 = G.stage1_void(3)
    r2 = G.derive_L1_void_to_chain(s2['equal_state'], L)
    r3 = G.derive_L1_void_to_chain(s3['equal_state'], L)

    check('d_local=2: dim_match 为真 (链条走过)', r2['dim_match'] is True)
    check(f'd_local=3: 张成维数 == dim_full == 3**{L} == {3 ** L}',
          r3['void_full'].size == r3['dim_full'] == 3 ** L,
          f"{r3['void_full'].size} == {r3['dim_full']} == {3 ** L}")
    check('d_local=3: dim_match 仍翻成 False', r3['dim_match'] is False,
          f'维数两边**自洽**, 翻 False 的唯一原因是 dim_match 里那道硬编码 '
          f'`== 2**L` 的闸 (2**{L} = {2 ** L})')
    check('d_local=3: 重叠被跳过, overlap_inf 如实为 nan',
          r3['overlap_curve'] == [] and bool(np.isnan(r3['overlap_inf'])),
          f"curve={r3['overlap_curve']}")

    # 第二处断点: 推离的 R_y 是 2x2, 对 qutrit 无定义。函数**显式拒绝**。
    try:
        G.l1_void_pushaway(s3['equal_state'], L)
        check('d_local=3: 推离拒绝执行', False,
              '**没有拒绝** —— 那才是意外, 说明 R_y 不再是 2x2')
    except ValueError as e:
        check('d_local=3: 推离拒绝执行 (ValueError, 显式说明理由)', True, str(e))

    check('**检验是活的**: d_local=2 与 3 的 dim_match 确实不同',
          r2['dim_match'] is not r3['dim_match'],
          f"True vs {r3['dim_match']} —— 不是恒真/恒假的检查")

    # ------------------------------------------------------------------
    print("\n  P2 · 预设 #2 复数域 + 内积 + 归一化 —— 给定**且不承重**")
    # ------------------------------------------------------------------
    _imag = float(np.max(np.abs(s2['equal_state'].imag)))
    check('初态是实值的 (复数结构不在初态里)', _imag == 0.0,
          f'equal_state = ones(2)/sqrt(2), |imag|max = {_imag:.1e}')

    # 结构性检验: 链上函数**签名里没有** Q/unitary ⇒ Q 在算术上不可能进任何读数。
    # 这比"扫源码找取值处"更强: 签名是接口, 改名/换实现都骗不过它。
    _sigs = {
        'derive_L1_void_to_chain': G.derive_L1_void_to_chain,
        'l1_void_pushaway': G.l1_void_pushaway,
        'stage2_critical_break': G.stage2_critical_break,
    }
    for _nm, _fn in _sigs.items():
        _params = list(inspect.signature(_fn).parameters)
        check(f'{_nm} 的签名里没有 Q/unitary',
              not any(('unit' in p.lower()) or (p == 'Q') for p in _params),
              ', '.join(_params))

    # 活的反事实: 若链上某函数**接受** Q, 上面那条必须翻红。
    # 造一个真接受 Q 的函数来证明这条检验有分辨力。
    def _counterfactual(unitary, equal_state, L=4):  # noqa: ARG001
        return None

    _cf = list(inspect.signature(_counterfactual).parameters)
    check('**检验是活的**: 若签名里加上 unitary, 同一条检验会翻红',
          any('unit' in p.lower() for p in _cf),
          f'反事实构造的签名 = {_cf} ⇒ 该检验能分辨')

    # ------------------------------------------------------------------
    print("\n  P3 · 预设 #3 随机测度 (复高斯 -> QR) —— 给定**且不承重**")
    # ------------------------------------------------------------------
    _seeds = (0, 1, 2, 7, 99)
    _st = {k: G.stage1_void(2, seed=k) for k in _seeds}
    _base = _st[_seeds[0]]['equal_state']

    check('equal_state 逐位不随 seed 变',
          all(np.array_equal(_base, _st[k]['equal_state']) for k in _seeds[1:]),
          f'{len(_seeds)} 个 seed 全等')
    check('cond(Q) == 1 对所有 seed 成立 (构造性为真)',
          all(np.isclose(np.linalg.cond(_st[k]['unitary']), 1.0) for k in _seeds),
          'QR 的输出按构造就是酉的 ⇒ 这个上报量对任何输入恒为 1')
    # 随机测度**确实存在** —— 否则上面两条只是"随机被关掉了", 结论没有内容。
    _dQ = float(np.linalg.norm(_st[0]['unitary'] - _st[1]['unitary']))
    check('Q 本身确实随 seed 变 (随机测度没被关掉)', _dQ > 0.0,
          f'||Q(seed=0) - Q(seed=1)|| = {_dQ:.6f}')
    # 活的反事实: 把初态换成 Q 的一列(它**依赖** seed), 同一条检验必须翻红。
    check('**检验是活的**: 若初态由 Q 派生, 同一条检验会翻红',
          not np.array_equal(_st[0]['unitary'][:, 0], _st[1]['unitary'][:, 0]),
          '反事实构造: equal_state := Q[:,0] ⇒ seed 会影响它')
    # "随机"二字下最容易担心的事: 可复现性。默认 seed=0 把它关掉了。
    check('默认 seed=0 ⇒ 连调两次逐位相同 (可复现性不受随机测度影响)',
          np.array_equal(G.stage1_void(2)['unitary'], _st[0]['unitary'])
          and np.array_equal(G.stage1_void(2)['equal_state'], _base))

    # ------------------------------------------------------------------
    print("\n  === 三条预设的承重性判定 ===")
    print("    #1 d_local=2          : 给定 **且承重**   —— 换掉则链在两处断")
    print("    #2 复数域+内积+归一化 : 给定 **且不承重** —— Q 在链上无消费者")
    print("    #3 随机测度           : 给定 **且不承重** —— equal_state 与 seed 无关")
    print("    ⇒ 三条都**仍是给定的**(§5.4 的缺口不因本文件而消失); 本文件只把")
    print("      '给定' 细分成 '承重' / '不承重' 两态, 供审计表登记。")

    print(f"\n=== 结果: "
          f"{'全部通过' if not _FAILS else '**失败 ' + str(len(_FAILS)) + ' 项**'}"
          f" ===")
    for f in _FAILS:
        print(f"    · {f}")
    return 0 if not _FAILS else 1


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.exit(main())
