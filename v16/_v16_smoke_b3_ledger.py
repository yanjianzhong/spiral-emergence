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
v16 · B3 冒烟测试 —— `_v16_claim_ledger.claim_test_ledger()`  [_VERSION_TAG = 'v16-smoke-b3-2']

两件事, 都不许用"源码看着对"代替:

**第一部分 · 负向测试**: (a)(b)(c)(d)(e) 每一条都要能被**人为构造的坏输入**
触发。做法是**真的调** `claim_test_ledger(...)` 并注入坏登记项 —— 不是把它的
逻辑抄一遍再断言 (硬纪律 2: 冒烟测试必须真调被改动的函数)。

**第二部分 · 真实清单覆盖**: (d) 说"推不出层的会报出来"。但注册**只在运行时**
发生, 全流程要 1245 s; 若等到那时才发现规则漏了一条, 代价就是一次全跑。
故此处从 `v15/_v15_run.log` (v15 全流程的完整登记转储) 重建 26 条指标 +
74 条守卫的**真实名字与 where/method**, 再喂给 (d) —— 现在就量出覆盖率。

边界: 日志转储**早于** B3-1 的字段增补, 所以重建出的登记项没有
`claim_id`/`source_id`。第一部分用注入数据覆盖那两个字段; 第二部分只能覆盖
"层号 + 名字 + where"这几列。这个差别在下面如实标注, 不掩盖。

用法:
    python v16/_v16_smoke_b3_ledger.py
"""

import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _v16_claim_ledger import (claim_test_ledger,  # noqa: E402
                               _NEG_CTRL_CLAIM, _SAME_SOURCE_REGISTRY)

_LOG = os.path.join(_ROOT, 'v15', '_v15_run.log')
_FAILS = []


def check(label, cond, detail=''):
    tag = 'PASS' if cond else '**FAIL**'
    print(f"    [{tag}] {label}" + (f"  —— {detail}" if detail else ''))
    if not cond:
        _FAILS.append(label)


# ===========================================================================
# 基线 fixture
# ===========================================================================
# ⚠️ 基线**必须自带** (c) 与 (e) 要求的那批名字, 否则基线自己就会挂:
#   (c) 要求 `_SAME_SOURCE_REGISTRY` 的每个 source_id 都真的存在,
#   (e) 要求 `_NEG_CTRL_CLAIM` 的每个值都能解析到一条真指标。
# 首版基线只有 2 条指标, 于是 (c)(e) 连带 (a) 一起判失败 —— **那是 fixture 的
# 缺陷, 不是检查的缺陷**; 下面直接从两张登记表**反推**基线该有哪些名字。
# C5b 那三条读数**分属两个注册表**: 指标侧 1 条, 守卫侧 2 条。首版把它们全塞
# 进指标侧, 于是 (d) 报出两条"未归层" —— 那是 fixture 的错位, 不是规则漏了
# 名字 (W3b 的层号规则在 `_GUARD_RULES` 里, 指标侧本就不该有 W3b 前缀)。
_W3B = _SAME_SOURCE_REGISTRY['W3b_spectrum']
_W3B_METRIC = '10 谱学中心荷 (能谱族)'
_W3B_GUARDS = tuple(n for n in _W3B if n != _W3B_METRIC)
# (e) 要求登记表里的每个主张都解析得到 ⇒ 这些名字必须真的落在**指标**侧
_CLAIMED = sorted({c for c in _NEG_CTRL_CLAIM.values() if c})

# v16·B1 新增的主张。第二部分用的是 **v15 日志转储**, 里面不可能有它们 ——
# 在那里对它们断言"必须解析得到"是把 fixture 的时限错当成检查的缺陷。
_B1_CLAIMS = {'14 L1 空无到链的下游连线 (重叠曲线)'}

# 两条"有名有姓"的指标, 用来在 (a) 里造出"一个被指认、一个没有"的对照
_M_MATCHED = {'tier': 2, 'name': '5 临界性负对照 (h/J 扫描)',
              'method': 'spiral_model_v15.hj_scan()', 'passed': True,
              'claim_id': None, 'source_id': None}
_M_ORPHAN = {'tier': 1, 'name': '中心荷误差',
             'method': 'fit_central_charge(MERA 态, L=16)', 'passed': True,
             'claim_id': None, 'source_id': None}


def _baseline():
    """完整到 (b)(c)(d)(e) 全过的登记集。"""
    named = {_M_MATCHED['name']: _M_MATCHED, _M_ORPHAN['name']: _M_ORPHAN}
    names = list(named) + [n for n in _CLAIMED if n not in named] \
        + [_W3B_METRIC]
    metrics = []
    for n in names:
        metrics.append(dict(named[n]) if n in named else
                       {'tier': 1, 'name': n, 'method': '', 'passed': True,
                        'claim_id': None,
                        'source_id': 'W3b_spectrum' if n == _W3B_METRIC
                        else None})
    guards = [{'name': 'F2a 主判据: c_fit 峰在临界点', 'where': 'hj_scan',
               'tests_claims': (), 'source_id': None}]
    guards += [{'name': n, 'where': 'spectral_central_charge',
                'tests_claims': (), 'source_id': 'W3b_spectrum'}
               for n in _W3B_GUARDS]
    return metrics, guards


def _fake_table(rows=None):
    """按 `_NEG_CTRL_CLAIM` 的 13 个 (层, 条目名) 造一张同形状的表。

    只填 (e) 用到的 t[0]=层 与 t[2]=条目名, 其余位保留真实形状。
    """
    src = _NEG_CTRL_CLAIM if rows is None else rows
    return tuple((L, '层名', lab, '推离', (), None) for (L, lab) in src)


def _run(m, g, t):
    return claim_test_ledger(metrics=m, guards=g, table=t, verbose=False)


def part1_negative():
    print("\n[第一部分] 负向测试 —— 每条检查都必须能被坏输入触发")

    mb, gb = _baseline()
    base = _run(mb, gb, _fake_table())
    print("  · 基线 (健康输入):")
    check('基线 (b) 无幻影', base['phantom'] == [], str(base['phantom']))
    check('基线 (c) 无伪独立', base['same_source_problems'] == [],
          str(base['same_source_problems']))
    check('基线 (d) 无未归层', base['unattributed'] == [],
          str(base['unattributed']))
    check('基线 (e) claim 列完备',
          not (base['nc_unregistered'] or base['nc_phantom']
               or base['nc_unknown_claim']),
          f"未登记={base['nc_unregistered']} 幻影={base['nc_phantom']} "
          f"主张不存在={base['nc_unknown_claim']}")
    check('基线 (a) **只报不判** —— 有孤儿但 ok 仍为真',
          '中心荷误差' in base['orphans'] and base['ok'] is True,
          f"n_orphans={len(base['orphans'])} ok={base['ok']}")

    print("  · (a) 孤儿主张: 有判定但无守卫指认 —— 必须出现在名单里")
    check('(a) 无守卫指认的被列出', '中心荷误差' in base['orphans'],
          str(base['orphans']))
    check('(a) 被 where 指认的不在名单里',
          _M_MATCHED['name'] not in base['orphans'])

    print("  · (b) 幻影引用: tests_claims 指向不存在的主张")
    g = [dict(gb[0], tests_claims=('99.9',))]
    r = _run(mb, g, _fake_table())
    check('(b) 幻影被报出且 ok=False',
          r['phantom'] == ['99.9'] and r['ok'] is False, str(r['phantom']))

    print("  · (b) 真实引用: claim_id 存在时按 id 指认, 孤儿消失")
    m = [dict(x) for x in mb]
    for x in m:
        if x['name'] == '中心荷误差':
            x['claim_id'] = '1.1'
    g = [dict(gb[0], tests_claims=('1.1',))]
    r = _run(m, g, _fake_table())
    check('(b) 无幻影且该条孤儿清零',
          r['phantom'] == [] and '中心荷误差' not in r['orphans'],
          f"phantom={r['phantom']}")

    print("  · (c) 伪独立 —— 三种失败都要报")
    m = [dict(x) for x in mb]
    for x in m:
        if x['name'] == '中心荷误差':
            x['source_id'] = 'SX'
    r = _run(m, gb, _fake_table())
    check('(c) 未登记同源被报出',
          any(w == '未登记同源' for _, w, _ in r['same_source_problems'])
          and r['ok'] is False, str(r['same_source_problems']))

    m = [dict(x) for x in mb if x['name'] in _W3B][:1]
    r = _run(m, [], _fake_table())
    check('(c) 登记成员不符被报出',
          any(w == '登记成员不符' for _, w, _ in r['same_source_problems'])
          and r['ok'] is False, str(r['same_source_problems']))

    m = [{'tier': 1, 'name': n, 'method': '', 'passed': True,
          'claim_id': None, 'source_id': 'W3b_spectrum'} for n in _W3B]
    r = _run(m, [], _fake_table())
    check('(c) 与登记表逐成员一致时通过',
          r['same_source_problems'] == [], str(r['same_source_problems']))

    # 要把**两个注册表**里的 W3b 读数都撤掉: 守卫侧那两条也带 source_id,
    # 只撤指标侧的话该组仍有 2 个成员, 触发的是"登记成员不符"而不是本条的路径。
    m = [dict(x) for x in mb if x['name'] not in _W3B]
    r = _run(m, [], _fake_table())
    check('(c) 登记表指向不存在的数据源被报出',
          any(w == '登记表指向不存在的数据源'
              for _, w, _ in r['same_source_problems']) and r['ok'] is False,
          str(r['same_source_problems']))

    print("  · (d) 未归层项 —— 名字与 where 都命中不了规则")
    m = mb + [{'tier': 1, 'name': 'ZZ 无规则指标', 'method': 'nope_zz',
               'passed': True, 'claim_id': None}]
    g = gb + [{'name': 'ZZ 无规则守卫', 'where': 'nope_zz',
               'tests_claims': (), 'source_id': None}]
    r = _run(m, g, _fake_table())
    check('(d) 两条未归层都被列名报出且 ok=False',
          r['unattributed'] == [('metric', 'ZZ 无规则指标'),
                                ('guard', 'ZZ 无规则守卫')]
          and r['ok'] is False, str(r['unattributed']))

    print("  · (e) 台账 claim 列的四种情况")
    t = _fake_table() + (('L9', '层名', 'ZZ · 未登记对照', '推离', (), None),)
    r = _run(mb, gb, t)
    check('(e) 未登记的对照条目被报出且 ok=False',
          r['nc_unregistered'] == ['L9/ZZ · 未登记对照'] and r['ok'] is False,
          str(r['nc_unregistered']))

    rows = [k for k in _NEG_CTRL_CLAIM if k[0] == 'L7']
    r = _run(mb, gb, _fake_table([k for k in _NEG_CTRL_CLAIM if k not in rows]))
    check('(e) 登记表里的幻影条目被报出且 ok=False',
          len(r['nc_phantom']) == 1 and r['ok'] is False, str(r['nc_phantom']))

    # 这一条**改不动表**: `nc_unknown_claim` 比的是登记表的**值**能不能在指标侧
    # 解析到。所以触发它要**撤掉那条指标**, 而不是往表里塞别的名字。
    gone = _NEG_CTRL_CLAIM[('L2', 'F2a · h/J 推离临界点')]
    m = [x for x in mb if x['name'] != gone]
    r = _run(m, gb, _fake_table())
    check('(e) 登记表指向不存在的主张被报出且 ok=False',
          gone in r['nc_unknown_claim'] and r['ok'] is False,
          str(r['nc_unknown_claim']))

    # v16·B1 之后 `_NEG_CTRL_CLAIM` 里已无 None (L1 的缺口补上了), 所以这条
    # 从"检查 L1 留缺"改成**注入式**: 临时加一条值为 None 的登记, 验证它进
    # nc_gap 但**不**让 ok 翻假 (留缺是如实报告, 不是失败)。
    check('(e) 实际登记表已无留缺 (B1 补上了 L1)', base['nc_gap'] == [],
          str(base['nc_gap']))
    _NEG_CTRL_CLAIM[('L1', 'ZZ 注入留缺')] = None
    try:
        r = _run(mb, gb, _fake_table())
        check('(e) 值为 None 的条目进 nc_gap 且**不算失败**',
              'L1/ZZ 注入留缺' in r['nc_gap'] and r['ok'] is True,
              str(r['nc_gap']))
    finally:
        del _NEG_CTRL_CLAIM[('L1', 'ZZ 注入留缺')]


# ===========================================================================
# 第二部分: 用 v15 全流程日志里的**真实清单**量 (d) 的覆盖率
# ===========================================================================
# 两个正则都**必须锚到行尾**: 首版漏了锚, 于是守卫正则吞下了一条内容里含
# 「通过率」的 `注:` 续行, 数出 75 条 (真值 74)。这就是"必须真跑"的原因 ——
# 只看源码看不出这个。
_RE_METRIC = re.compile(r'^  (?P<t>\d+)\.(?P<i>\d+) (?P<name>.+?) {2,}\S')
_RE_METHOD = re.compile(r'^ +方法: (?P<m>.+)$')
_RE_GUARD = re.compile(r'^ {6}(?P<body>.+?) {2,}(?P<st>通过|未通过)'
                       r'\s*(?:\[[^\]]*\])?\s*$')


def load_real_inventory():
    """从日志转储重建登记项清单; 字段名与记录结构对齐 v15 注册表。"""
    with open(_LOG, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    metrics, guards = [], []
    cur = None
    for ln in lines:
        mm = _RE_METHOD.match(ln)
        if mm and cur is not None:
            cur['method'] = mm.group('m')
            continue
        m = _RE_METRIC.match(ln)
        if m:
            cur = {'tier': int(m.group('t')), 'name': m.group('name').strip(),
                   'method': '', 'claim_id': None, 'source_id': None,
                   'passed': None if ln.rstrip().endswith('[诊断]') else True}
            metrics.append(cur)
            continue
        gm = _RE_GUARD.match(ln)
        if gm:
            body = re.sub(r'\s+', ' ', gm.group('body')).strip()
            head, _, tail = body.rpartition(' ')
            where = tail if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*'
                                         r'(\(.*\))?', tail) else '?'
            guards.append({'name': head.strip() if where != '?' else body,
                           'where': where, 'tests_claims': (),
                           'source_id': None})
    return metrics, guards


def part2_coverage():
    print("\n[第二部分] (d) 对**真实清单**的覆盖率 "
          "(源: v15/_v15_run.log; 该转储早于 B3-1 字段增补 ⇒ 无 claim_id/source_id)")
    metrics, guards = load_real_inventory()
    print(f"  · 重建: 指标 {len(metrics)} 条, 守卫 {len(guards)} 条")
    check('重建条数与 v15 全流程日志一致 (26 / 74)',
          len(metrics) == 26 and len(guards) == 74,
          f"实测 {len(metrics)} / {len(guards)}")

    res = claim_test_ledger(metrics=metrics, guards=guards,
                            table=_fake_table(), verbose=False)
    check('(d) 真实清单**零未归层**', res['unattributed'] == [],
          str(res['unattributed']))
    check('(b) 真实清单无幻影引用 (日志转储不含 tests_claims)',
          res['phantom'] == [])
    # ⚠️ 这一段用的是 **v15 转储**, 而 v16·B1 之后 `_NEG_CTRL_CLAIM` 里多了一条
    # B1 新增的主张 —— 它在 v15 日志里**当然不存在**。所以下面把 B1 新增的那条
    # 单列出来, 并对**其余**主张断言可解析。v16 全流程日志里这条会解析得到。
    unknown = [c for c in res['nc_unknown_claim'] if c not in _B1_CLAIMS]
    check('(e) v15 时代的主张在真实清单里全部解析得到', unknown == [],
          str(unknown))
    check('(e) 未解析的**只有** B1 新增的那一条 (v15 转储不可能有它)',
          set(res['nc_unknown_claim']) <= _B1_CLAIMS,
          f"未解析={res['nc_unknown_claim']}, B1 新增={sorted(_B1_CLAIMS)}")

    scored = sum(p['n_total'] for p in res['per_layer'])
    print(f"\n  · 口径匹配率实测 (分母 = 有 pass/fail 判定的 tier-1/2 指标 = {scored})")
    for p in res['per_layer']:
        if p['n_total'] == 0:
            print(f"        {p['layer']}: **无计分指标**  "
                  f"(同层守卫 {p['n_guard']} 条)")
        else:
            print(f"        {p['layer']}: {p['n_matched']}/{p['n_total']}"
                  f"   (同层守卫 {p['n_guard']} 条)")
    print(f"    => 有守卫指认 {scored - len(res['orphans'])}/{scored}; "
          f"其余 {len(res['orphans'])} 条**没有任何既有守卫指认**:")
    for n in res['orphans']:
        print(f"        · {n}")
    print("    注 1: 本段用**日志转储**, 故 C5b 的 claim_id 指认 (真实运行时才有)"
          " 不在此体现 —— 此处匹配率是**下界**。")
    print("    注 2: 匹配口径是「某守卫的 where 出现在该指标的 method 里」。"
          "对不上**不等于缺陷** —— v15 里指标的 pass/fail 本就是它与自己目标的"
          "比较, 守卫是**另一个**注册表; 这条数字量的是「有多少条主张另有独立"
          "守卫», 不是「有多少条主张被检验过」。")
    return res


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    print("=== v16 · B3 冒烟测试 ===")
    part1_negative()
    part2_coverage()
    print(f"\n=== 结果: "
          f"{'全部通过' if not _FAILS else '**失败 ' + str(len(_FAILS)) + ' 项**'}"
          f" ===")
    for f in _FAILS:
        print(f"    · {f}")
    sys.exit(0 if not _FAILS else 1)
