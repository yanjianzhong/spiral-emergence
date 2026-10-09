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
_v16_freeze_check.py —— v15 → v16 的**冻结核对**（按名字，不按逐位相等）
================================================================================
【为什么不能按逐位相等核对】

`v16_plan.md` §0.1 的「B1 实测登记」已经把原因量出来了: `exact_ground_state`
**连调两次**, `E0` 就有 `3.553e-15` 的抖动（相对 `1.7e-16`）—— eigsh 的迭代起点
不由本仓库控制。所以「v16 与 v15 读数逐位相同」是一个**永远无法成立的判据**，
拿它做冻结核对只会得到假红。

**正确的判据**（§0.1 逐字）:「**原始 54 条守卫无一翻成未通过**」——
不是「总数仍是 54」。B1 新增计分守卫 ⇒ 分母必然上涨;
把总数不变当冻结, 等于禁止一切新增守卫。

【本脚本做什么】

拿两次**全流程运行日志**（`v15/_v15_run.log` 与 `v16/_v16_run.log`）的
**注册项名字 + 判定**做差集, 回答四个问题:

  1. v15 的计分守卫, 在 v16 里**有没有翻成未通过**?（这才是「冻结」）
  2. v16 新增了哪些计分守卫 / 诊断项?
  3. v15 的诊断项, 判定有没有变?
  4. 指标是否原样?

【为什么不读 `_v16_data.json`】

JSON 里的注册表是**运行时**的, 但它不带「这条在 v15 里是什么」——
差集必须由两份日志的**名字**对齐。日志是唯一的共同证据面。

【本脚本自己撞过的一条缺陷（首版, 已修, 照登）】

首版的守卫正则把 `where` 写成了 `[A-Za-z_][A-Za-z0-9_]*`, 于是
`_v15_run.log:910` 那条

    G7 最优检查点非首步 mera_fit_v13(overlap 选择)           通过

**整行匹配不上** —— 它的 `where` 是 `mera_fit_v13(overlap 选择)`, **括号内含空格**。
后果不是崩溃而是**静默少数一条**: 脚本会报「原始 53 条无一翻成未通过」,
而 `_v15_run.log:1090` 自己印的是 `通过率 = 54/54`。
⇒ 凭空造出一个「53 vs 54」的差额, 且看上去像是**真发现了什么**。

**修法不是放宽 `where`**（它可能还有别的形状, 例如带参数、带下标）—— 而是
**不再切分**: 以整行 body 作为键。`where` 只在**打印**时尝试切出来供人读,
切不出就整条显示。另加一条**自校验**: 解析出的计分数/诊断数必须等于
**日志自己打印的那个分母**（`通过率 = N/M` 与 `另有诊断项 K 项`）,
不等就报 `[自校验失败]` 并退 2 —— 让「解析器少数了一条」**无法伪装成科学结论**。

【本脚本撞过的第二条缺陷（首次对真 v16 日志运行, 已修, 照登）】

用整行 body 作键**修好了少数**, 却把**判据换严了**: body = 名字 + `where`,
而 `where` 是**注册它的函数名**。v15 → v16 的拆分把 `collect_metrics_v15`
改名为 `collect_metrics_v16`, 于是

    v15:  G6 最坏轨迹也达标  collect_metrics_v15   通过
    v16:  G6 最坏轨迹也达标  collect_metrics_v16   通过

被判成「v15 的 1 条计分守卫在 v16 里**找不到**」并 `EXIT=3`。**这是一次假回归**:
`v16_plan.md` §0.1 的判据逐字是「**按名字**做」, 而这里按的是「名字 + 函数名」。
函数名随模块改名而变是**预期的**, 名字才是守卫的身份。

⇒ **键改为名字**（`_WHERE_RE` 切不出就退回整行 body）, 另把 `where` 的变化
**单列报出**供人工看, **不计入回归**。再加一条歧义检查: 若两条守卫**同名**,
名字键会互相覆盖 ⇒ 直接报出来退 2, 不静默取一个。
**教训与第一条同源**: 判据写得比声明的更严, 得到的不是更安全, 而是**假红**;
假红和假绿一样会把结论带偏。

【边界（如实登记）】

  - 本脚本**只做名字与判定的比对**, 不比对条件/注/读数 —— 那些会随浮点抖动,
    且「条件」里常含末位浮点数。名字是稳定的, 读数是漂的。
  - 名字重复时以**整行 body** 为键, 重复会被 `dict` 覆盖 —— 故下面显式
    报出「解析条数 vs 去重条数」的差, **不静默取一个**。
  - 本脚本**不修改任何文件**, 只读。
  - **退出码口径**: 只有**计分侧**的变化算回归(`3`)。诊断项（`expect_pass=False`）
    本就是留档的负面结果, 翻正是**预期内的证据**（§2.3 点名 W11 应翻成 True）
    —— 故单列报出供人工确认, **不进回归数**。

用法::

    python _v16_freeze_check.py
    python _v16_freeze_check.py --v15 <path> --v16 <path>
"""

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# 守卫段的行: `      G1 L2 谱退化 derive_L2_entanglement_gap         通过`
#               `      V6 权重自由变体秩1化 selfref_coupled   未通过  [诊断项, 不计入分母]`
# 缩进 4~8 空格（注册行是 6）—— 「条件:」/「注:」续行缩进 13, 借此排除。
_GUARD_RE = re.compile(
    r'^\s{4,8}(?P<body>.+?)\s+'
    r'(?P<verdict>通过|未通过)'
    r'(?P<diag>\s*\[诊断项, 不计入分母\])?\s*$'
)

# 指标行: `  1.1 中心荷误差   eps_c = |c_MERA - c_exact| / c_exact   [达标]`
_METRIC_RE = re.compile(
    r'^\s{2}(?P<idx>\d+\.\d+)\s+(?P<label>\S.*?)\s{2,}'
    r'(?P<method>.+?)\s*\[(?P<verdict>达标|未达标)\]\s*$'
)

# 日志自己印的两个分母, 用作**自校验**的锚点。
_SELF_TALLY_RE = re.compile(r'通过率\s*=\s*(\d+)\s*/\s*(\d+)')
_SELF_DIAG_RE = re.compile(r'另有诊断项\s*(\d+)\s*项')

# 展示用: 从 body 尾部切出 `where`（函数名, 可带参数/下标）。
_WHERE_RE = re.compile(r'^(?P<name>.+?)\s+(?P<where>[A-Za-z_][\w.]*(?:\(.*\))?)\s*$')


def _split(name_where):
    """把 body 拆成 (名字, where) 仅供打印; 拆不出就把 where 记为 '?'。"""
    m = _WHERE_RE.match(name_where)
    if m:
        return m.group('name'), m.group('where')
    return name_where, '?'


def parse_guards(path):
    """返回 (注册表 dict, 自校验 dict)。

    键 = 整行 body（**不切 where** —— 见文件头「本脚本自己撞过的一条缺陷」）。
    只在 `守卫通过率` 段与日志自印的 `通过率 = N/M` 之间取行。
    """
    lines = open(path, encoding='utf-8', errors='replace').read().splitlines()
    hdr = next((i for i, l in enumerate(lines) if '守卫通过率 (可证伪检查)' in l), None)
    if hdr is None:
        return None, {'error': '找不到「守卫通过率 (可证伪检查)」段头'}
    end = next((i for i, l in enumerate(lines)
                if i > hdr and l.strip().startswith('通过率 = ')), None)
    if end is None:
        return None, {'error': '找不到 `通过率 = N/M` 汇总行'}

    out, raw = {}, 0
    for lineno in range(hdr + 1, end):
        m = _GUARD_RE.match(lines[lineno])
        if not m:
            continue
        raw += 1
        out[m.group('body')] = dict(verdict=m.group('verdict'),
                                    diag=bool(m.group('diag')),
                                    line=lineno + 1)

    # 自校验: 与日志**自己印的**分母对齐。
    tally = _SELF_TALLY_RE.search(lines[end])
    n_scored_self = int(tally.group(1)) if tally else None
    n_total_self = int(tally.group(2)) if tally else None
    diag_self = None
    for l in lines[end:end + 4]:
        d = _SELF_DIAG_RE.search(l)
        if d:
            diag_self = int(d.group(1))
            break

    n_scored = sum(1 for v in out.values() if not v['diag'])
    n_diag = sum(1 for v in out.values() if v['diag'])
    # 名字键的歧义检查: 两条守卫同名 ⇒ 名字键会互相覆盖, 差集不可信。
    _names = {}
    for k in out:
        _names.setdefault(_split(k)[0], []).append(k)
    dup = {nm: ks for nm, ks in _names.items() if len(ks) > 1}
    checks = dict(n_scored=n_scored, n_diag=n_diag, n_parsed=raw,
                  n_unique=len(out), n_scored_self=n_scored_self,
                  n_total_self=n_total_self, n_diag_self=diag_self,
                  dup=dup)
    checks['ok'] = (n_scored == n_scored_self and n_diag == diag_self
                    and raw == len(out))
    return out, checks


def _by_name(recs):
    """把 body 键的注册表换成**名字键**（判据按名字, 见文件头第二条缺陷）。"""
    out = {}
    for bodykey, rec in recs.items():
        nm, where = _split(bodykey)
        out[nm] = dict(rec, body=bodykey, where=where)
    return out


def parse_metrics(path):
    """返回 {名字: dict(verdict, method, line)}。名字形如 `1.1 中心荷误差`。"""
    out = {}
    with open(path, encoding='utf-8', errors='replace') as fh:
        for lineno, line in enumerate(fh, 1):
            m = _METRIC_RE.match(line)
            if not m:
                continue
            name = f"{m.group('idx')} {m.group('label').strip()}"
            out[name] = dict(verdict=m.group('verdict'),
                             method=m.group('method'), line=lineno)
    return out


def _split_g(g):
    return ({k: v for k, v in g.items() if not v['diag']},
            {k: v for k, v in g.items() if v['diag']})


def _tag(checks):
    """把自校验结果摊成一行, 供两个日志各印一条。"""
    if 'error' in checks:
        return f"!! {checks['error']}"
    ok = '✓' if checks['ok'] else '✗ **自校验失败**'
    return (f"{ok} 解析 计分{checks['n_scored']}/诊断{checks['n_diag']}"
            f" (行 {checks['n_parsed']}, 去重 {checks['n_unique']})"
            f" vs 日志自印 计分{checks['n_scored_self']}"
            f"/总{checks['n_total_self']}/诊断{checks['n_diag_self']}")


def main():
    ap = argparse.ArgumentParser(description='v15 → v16 冻结核对（按名字）')
    ap.add_argument('--v15', default=os.path.join(ROOT, 'v15', '_v15_run.log'))
    ap.add_argument('--v16', default=os.path.join(HERE, '_v16_run.log'))
    args = ap.parse_args()

    for p in (args.v15, args.v16):
        if not os.path.exists(p):
            print(f"!! 找不到日志: {p}")
            return 2

    g15, c15 = parse_guards(args.v15)
    g16, c16 = parse_guards(args.v16)
    m15, m16 = parse_metrics(args.v15), parse_metrics(args.v16)

    print('=' * 78)
    print('  v15 → v16 冻结核对（按名字, 不按逐位相等）')
    print('=' * 78)
    print(f"  v15 日志: {args.v15}")
    print(f"    {_tag(c15)}")
    print(f"  v16 日志: {args.v16}")
    print(f"    {_tag(c16)}")

    # 解析器自校验先行 —— 解析不可信时, 后面所有结论都不可信, 直接停。
    if not (c15.get('ok') and c16.get('ok')):
        print()
        print('  ✗ **解析器自校验失败**: 数与日志自印的分母不一致。')
        print('    这不是科学结论, 是**工具坏了** —— 先修解析器, 别读下面的差集。')
        return 2

    # 名字键的歧义会让差集**静默取一个**, 与「少数一条」是同一类错误 ⇒ 直接停。
    for tag, chk in (('v15', c15), ('v16', c16)):
        if chk['dup']:
            print()
            print(f"  ✗ **{tag} 日志里有同名守卫**: {list(chk['dup'])}")
            print("    名字键会互相覆盖, 差集不可信 —— 先给守卫起唯一名, 别读下面的结论。")
            return 2

    s15, d15 = _by_name(_split_g(g15)[0]), _by_name(_split_g(g15)[1])
    s16, d16 = _by_name(_split_g(g16)[0]), _by_name(_split_g(g16)[1])

    print()
    print(f"  注册规模:          {'v15':>8} {'v16':>8}")
    print(f"    计分守卫        {len(s15):>8} {len(s16):>8}")
    print(f"    诊断守卫        {len(d15):>8} {len(d16):>8}")
    print(f"    守卫合计        {len(g15):>8} {len(g16):>8}")
    print(f"    指标            {len(m15):>8} {len(m16):>8}")

    # ---- 1. 冻结判据 ----
    print()
    print('-' * 78)
    print('  【判据】v15 的计分守卫, 在 v16 里有无翻成「未通过」')
    print('-' * 78)
    missing, flipped, wherechg = [], [], []
    for nm, rec in sorted(s15.items()):
        if nm not in s16:
            missing.append(nm)
            continue
        if s16[nm]['verdict'] != rec['verdict']:
            flipped.append((nm, rec['verdict'], s16[nm]['verdict']))
        if rec['where'] != s16[nm]['where']:
            wherechg.append((nm, rec['where'], s16[nm]['where']))
    if missing:
        print(f"  ⚠️  v16 日志里**找不到**的 v15 计分守卫 {len(missing)} 条:")
        for nm in missing:
            print(f"        {nm}  [{s15[nm]['where']}]")
    else:
        print(f"  ✓ v15 的 {len(s15)} 条计分守卫, 按名字**一条不少**")
    if flipped:
        print(f"  ✗ **判定翻转 {len(flipped)} 条**（这就是回归）:")
        for nm, a, b in flipped:
            print(f"        {nm}  [{s16[nm]['where']}]  {a} -> {b}")
    else:
        print(f"  ✓ **原始 {len(s15)} 条无一翻成未通过** ⇒ 冻结成立")
    if wherechg:
        print(f"  ℹ 另有 {len(wherechg)} 条的 `where`（注册函数名）变了 —— 名字与判定"
              f"都没变, **不计入回归**:")
        for nm, a, b in wherechg:
            print(f"        {nm}  {a} -> {b}")

    # ---- 2. v16 新增 ----
    print()
    print('-' * 78)
    print('  v16 新增的注册项（v15 日志里没有的名字）')
    print('-' * 78)
    new_sc = sorted(k for k in s16 if k not in s15)
    new_dg = sorted(k for k in d16 if k not in d15)
    print(f"  新增计分守卫 {len(new_sc)} 条:")
    for nm in new_sc:
        print(f"        {nm}  [{s16[nm]['where']}]  {s16[nm]['verdict']}"
              f"  (行 {s16[nm]['line']})")
    print(f"  新增诊断项 {len(new_dg)} 条:")
    for nm in new_dg:
        print(f"        {nm}  [{d16[nm]['where']}]  {d16[nm]['verdict']}"
              f"  (行 {d16[nm]['line']})")

    # ---- 3. v15 诊断项的判定有没有变 ----
    print()
    print('-' * 78)
    print(f"  v15 的 {len(d15)} 条诊断项在 v16 里的判定")
    print('-' * 78)
    dchg = []
    for nm, rec in sorted(d15.items()):
        if nm not in d16:
            dchg.append((nm, rec['verdict'], '（v16 里找不到）'))
        elif d16[nm]['verdict'] != rec['verdict']:
            dchg.append((nm, rec['verdict'], d16[nm]['verdict']))
    if dchg:
        # ⚠️ 诊断项翻面**不是回归** —— 它们本就是留档的负面结果, 翻正是「可证伪
        #    守卫被代码改动翻面」的实证（`v16_plan.md` §2.3 点名 W11 应翻成 True）。
        #    故此处**逐条报出供人工确认**, 但**不计入回归数**（见末尾 `bad`）。
        print(f"  判定发生变化的 {len(dchg)} 条（**不计入回归**, 需人工确认是否预期）:")
        for nm, a, b in dchg:
            w = (d16[nm]['where'] if nm in d16 else d15[nm]['where'])
            print(f"        {nm}  [{w}]  {a} -> {b}")
    else:
        print(f"  ✓ v15 的 {len(d15)} 条诊断项, 判定**逐条未变**")

    # ---- 4. 指标 ----
    print()
    print('-' * 78)
    print('  指标')
    print('-' * 78)
    mflip = [(k, m15[k]['verdict'], m16[k]['verdict'])
             for k in sorted(m15)
             if k in m16 and m16[k]['verdict'] != m15[k]['verdict']]
    mmiss = [k for k in sorted(m15) if k not in m16]
    mnew = [k for k in sorted(m16) if k not in m15]
    print(f"  v15 指标 {len(m15)} 条 / v16 指标 {len(m16)} 条")
    if mmiss:
        print(f"  ⚠️ v16 里找不到的 v15 指标 {len(mmiss)} 条: {mmiss}")
    if mflip:
        print(f"  ✗ 判定翻转 {len(mflip)} 条:")
        for k, a, b in mflip:
            print(f"        {k}  {a} -> {b}")
    else:
        print("  ✓ 共有指标判定**无一翻转**")
    if mnew:
        print(f"  新增指标 {len(mnew)} 条: {mnew}")

    # ---- 5. 通过率 ----
    print()
    print('-' * 78)
    print('  通过率')
    print('-' * 78)
    for tag, g, s in (('v15', g15, s15), ('v16', g16, s16)):
        npass = sum(1 for v in s.values() if v['verdict'] == '通过')
        ndiag_bad = sum(1 for v in g.values()
                        if v['diag'] and v['verdict'] == '未通过')
        rate = (100.0 * npass / len(s)) if s else float('nan')
        print(f"  {tag}: 计分 {npass}/{len(s)} = {rate:.2f}%"
              f"   (诊断 {sum(1 for v in g.values() if v['diag'])} 条, "
              f"其中未通过 {ndiag_bad} 条)")

    # 退出码口径: **只有计分侧的变化算回归**。诊断项(`expect_pass=False`)是留档的
    # 负面结果, 其翻面是**预期内的证据**（§2.3 点名 W11）, 单列供人看, 不进 bad。
    bad = len(flipped) + len(missing) + len(mflip) + len(mmiss)
    print()
    print(f"  计分侧变化: {bad} 项（翻转 {len(flipped)} + 缺失 {len(missing)}"
          f" + 指标翻转 {len(mflip)} + 指标缺失 {len(mmiss)}）")
    print(f"  诊断侧变化: {len(dchg)} 项（单列, 不计入回归）")
    print(f"  结论: {'冻结成立' if bad == 0 else f'**有 {bad} 项需要看**'}")
    return 0 if bad == 0 else 3


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        pass
    sys.exit(main())
