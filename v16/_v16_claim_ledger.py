# -*- coding: utf-8 -*-
"""
v16 · B3 —— claim ↔ test 口径对齐账本  [_VERSION_TAG = 'v16-claim-ledger-1']

答的是什么
----------
v15 的 `record_metric`(主张) 与 `record_guard`(检验) 之间**没有任何字段互相指认**:
主张登记在 METRICS, 检验登记在 GUARDS, 负对照台账 `_NEG_CTRL_TABLE` 又只按
**物理阶段**分组。于是"这条读数有没有人检验"和"这条检验守的是哪条主张"
在代码里都问不出来。

最硬的实例是 C5b (`v15/spiral_v15_paper_outline.md:188-195`):
`ratio→8` 与 `c→0.5` 被登记成两条计分守卫, 后者的 note 还写着"独立佐证" ——
但两者来自**同一次** `eigsh(H,k=3)` 的三个本征值, 且共用同一个写死的
`Delta1=1/8` 前提。按负对照台账自己的判据("能在结论不成立时失败"),
它们**不是两条独立证据**。

本模块把这件事**机器化**: 不新增物理, 只做口径对齐。

层号怎么来 (为什么是规则而不是逐条手写, 以及规则的依据)
----------------------------------------------------
注册只在**运行时**发生 (`import` 后 `len(METRICS)==0`), 源码 grep 只能看到
61/74 条 `record_guard`。所以"逐调用点手写层号"会**静默漏掉 13 条**。
故层号走规则推导, 依据取自 **v15 全流程日志里登记项自己的字段** (不是命名的印象):

- 守卫: 用 `where` —— 它就是**登记这条守卫的函数名**, 客观可查。
  日志实测 74 条守卫共 30 种 where 取值。
- 指标: 用 `method` —— 里面写着算这条读数的函数名。
- **名字前缀优先**于 where/method, 因为 F5-L2/L5/L6/L7 四条共用
  `where='consistency_checks'` 之类的并集函数, 只有名字能区分它们。

推不出层的项**不是被跳过**, 而是由硬检查 (d) **列名报出**。
实测覆盖: 26/26 指标 + 74/74 守卫。

边界 (必须随结论一起读)
------------------------
- 本模块**不判定任何物理结论**, 只判"登记口径是否自洽"。
- 层号是**数据/检验分层**, **不是** OA/HS 分层 (`v16_plan.md` §4.4:
  OA 层算符/透明区间/融合规则在整个 v15 仓库里不存在, 不做)。
- `EXT` 是审计表**第三层(宏观结构对标)**那一族 —— 它对的是**外部数据**,
  **不属于** L1..L7 这七个物理阶段, 故单列, 不硬塞进任何一层。
- `X` 是**跨层/台账自身完整性**, 同样不冒充某一层。

运行
----
    python _v16_claim_ledger.py     # 需与 v15 全流程同进程, 否则注册表是空的
"""

import os
import sys

_VERSION_TAG = 'v16-claim-ledger-1'

_LAYERS = ('L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'EXT', 'X')

# ===========================================================================
# 指标层号规则 —— name 前缀优先, 未命中再看 method 子串
# ===========================================================================
_METRIC_RULES = (
    # v16·B1: L1 的第一条下游连线。此前 L1 在**指标侧一条都没有** ——
    # 这正是 W11 诊断的缺口在口径匹配率上的表现 (L1 那行曾是"无计分指标")。
    ('14 L1 空无到链', 'L1'),
    # tier-1 (内部自洽性): 主体是 MERA 自身的一致性 ⇒ L4; 1.3 走 spiral_loop ⇒ L5
    ('中心荷误差', 'L4'), ('共形不变性', 'L4'), ('MERA 一致性', 'L4'),
    ('全程稳健性', 'L4'), ('预算漂移', 'L4'), ('chi 瓶颈归因', 'L4'),
    ('谱隙闭合残差', 'L5'),
    # tier-2 (微观物理对标)
    ('因果链拟合', 'L3'), ('Jordan 块收敛率', 'L3'), ('9 自指变体', 'L3'),
    ('关联衰减指数', 'L2'), ('5 临界性', 'L2'), ('6 跨边界', 'L2'),
    ('纠缠熵 KL', 'L4'),
    ('8 涌现几何', 'L5'), ('10 谱学中心荷', 'L5'),
    ('11 面积律', 'L5'), ('12 有限尺寸标度', 'L5'),
    # 跨层束: 一致性检查束横跨 L2/L3/L5/L6; 台账是 MGMT
    ('7 一致性检查束', 'X'), ('13 负对照台账', 'X'),
    # tier-3 (宏观结构对标): 对外部数据, 不属七层
    ('III-', 'EXT'),
)

# ===========================================================================
# 守卫层号规则 —— name 前缀优先, 未命中再看 where 子串
# ===========================================================================
# 顺序即优先级。带空格的 'V1 ' / 'W1 ' 不会误吞 'V11' / 'W11'; 反过来说,
# 'W11 诊断: L1 …' 必须排在 'W11' **之前**, 否则那条会被错归到 L7。
_GUARD_RULES = (
    # F5 束逐层自报, 必须排在最前 (四条共用同一个 where)
    ('F5-L2', 'L2'), ('F5-L3', 'L3'), ('F5-L5', 'L5'),
    ('F5-L6', 'L6'), ('F5-L7', 'L7'),
    ('F5 诊断: Gray-Scott', 'L6'),
    ('W11 诊断: L1', 'L1'),
    # 单字母族
    ('G1 ', 'L2'), ('G2 ', 'L4'), ('G3 ', 'L6'), ('G4 ', 'L6'),
    ('G5 ', 'L5'), ('G6 ', 'X'), ('G7 ', 'X'), ('G8 ', 'L4'),
    ('V1 ', 'L5'), ('V2 ', 'L5'), ('V3 ', 'L5'), ('V4 ', 'L5'), ('V5 ', 'L5'),
    ('V6 ', 'L3'), ('V7 ', 'L3'), ('V8 ', 'L6'),
    ('W1 ', 'L3'), ('W3a', 'L5'), ('W3b', 'L5'), ('W5 ', 'L5'),
    ('W6', 'X'), ('W11', 'L7'),
    ('F1 ', 'L2'), ('F2a', 'L2'), ('F6', 'L5'), ('F7 ', 'L6'),
    ('L4 键维向下', 'L4'),
    ('A-', 'EXT'), ('T3-', 'EXT'), ('III-', 'EXT'),
)

# where / method 子串兜底 (名字规则都没命中时用)
_WHERE_RULES = (
    ('stage1_void', 'L1'),
    # v16·B1 新增的三条 L1 守卫各自的 where (都是 spiral_model_v16 的真函数名)。
    # 放在这里而不是靠名字前缀, 是因为它们与 F5 束不同: 名字前缀 'B1' 只说明
    # 属于哪个工作包, 不说明层号; 层号由 where 决定。
    ('derive_L1_void_to_chain', 'L1'), ('l1_void_pushaway', 'L1'),
    ('derive_L1_local_to_full', 'L1'),
    ('derive_L2', 'L2'), ('hj_scan', 'L2'), ('cross_boundary_twist', 'L2'),
    ('selfref', 'L3'),
    ('derive_L4', 'L4'), ('v13_tier1_runs', 'L4'), ('mera_init', 'L4'),
    ('mera_fit_v13', 'L4'),
    ('spiral_loop', 'L5'), ('validate_curvature', 'L5'),
    ('spectral_central_charge', 'L5'), ('area_vs_log_law', 'L5'),
    ('finite_size_scaling', 'L5'), ('geometry_controls', 'L5'),
    ('curvature_report', 'L5'),
    ('derive_L6', 'L6'), ('stage6_life', 'L6'),
    ('gray_scott_invariants', 'L6'), ('derive_L6_grid', 'L6'),
    ('stage7_consciousness', 'L7'),
    ('tier3_structural_ladder', 'EXT'), ('structural_descriptors', 'EXT'),
    ('collect_metrics_v16', 'X'), ('negative_control_ledger', 'X'),
)

# ===========================================================================
# 同源登记表 —— 一个 source_id 下**不允许有成员漏登记**
# ===========================================================================
# C5b: 三条读数共用同一次 `eigsh(H, k=3)` 的三个本征值 + 同一个写死的
# `Delta1 = 1/8` 前提 ⇒ 必须标成同源, 任何文档里**不得**写成
# 「互相印证 / 独立佐证」。
_SAME_SOURCE_REGISTRY = {
    'W3b_spectrum': ('10 谱学中心荷 (能谱族)',
                     'W3b v 无关比值单调趋 8',
                     'W3b 谱学 c 与 0.5 相容到 1%'),
}

# ===========================================================================
# B3-3 —— `_NEG_CTRL_TABLE` 的 claim 列
# ===========================================================================
# **为什么不是在 v15 里给 13 个元组各加第 8 个元素**: 那些元组是**参差的**
# (6 或 7 个元素, L4 行多一个补充说明), 所以位置下标 `r[7]` 会在 6 元组上
# IndexError; 要"加列"就得**逐行改 13 条冻结记录**。故把这一列**外置**成
# 按 (层, 条目名) 索引的登记表 —— 不删任何既有条目, 且同样的完备性检查
# 由 (e) 承担 (未登记 / 幻影 / 指向不存在的主张, 三种都报)。
# 值 = **这条对照在为哪条主张提供可证伪性**。None = 如实留缺。
# v16·B1 后 L1 有对照了, 故 None 只剩"注入式留缺"这一条测试路径, **实际为 0 条**。
_NEG_CTRL_CLAIM = {
    ('L1', 'B1 · 局部等权态转动推离'): '14 L1 空无到链的下游连线 (重叠曲线)',
    ('L2', 'F2a · h/J 推离临界点'): '5 临界性负对照 (h/J 扫描)',
    ('L2', 'F5-L2 · eigsh vs JW 闭合式'):
        '7 一致性检查束 (数值路径 <-> 解析路径)',
    ('L2', 'F1 · 跨边界 θ=π twist'): '6 跨边界条件中心荷交叉检验 (同族)',
    ('L3', 'W1 · 冻结权重对照'): '9 自指变体秩1化的 N 依赖 (相图)',
    ('L3', 'F5-L3 · 幂迭代残差'): '7 一致性检查束 (数值路径 <-> 解析路径)',
    ('L3', 'V6/V7 · 权重自由/归一变体秩1化'): '9 自指变体秩1化的 N 依赖 (相图)',
    ('L4', 'W12 · 键维向下推离 (压到面积律下界以下)'): '中心荷误差',
    ('L5', 'F6 · 零模型对照族 (三类 x 3 实例)'): '8 涌现几何的对照鲁棒性',
    ('L5', 'F5-L5 · Forman 在环 C_n 上为 0'):
        '7 一致性检查束 (数值路径 <-> 解析路径)',
    ('L6', 'F5-L6 · 两条离散恒等式'):
        '7 一致性检查束 (数值路径 <-> 解析路径)',
    ('L6', 'F7 · 特征波长的退化输入校验'): 'III-A1 描述子距离',
    ('L7', 'F5-L7 · 熵比与等权度上界'):
        '7 一致性检查束 (数值路径 <-> 解析路径)',
}


def _layer_of(name, hint, rules):
    """名字前缀优先, 再看 where/method 子串; 都命中不了返回 None (= 未归层)。"""
    for pref, L in rules:
        if name.startswith(pref):
            return L
    hay = hint or ''
    for sub, L in _WHERE_RULES:
        if sub in hay:
            return L
    return None


def layer_of_metric(name, method=None):
    return _layer_of(name, method, _METRIC_RULES)


def layer_of_guard(name, where=None):
    return _layer_of(name, where, _GUARD_RULES)


def _live_registries():
    """取 v16 的活注册表。注册只在运行时发生, 故必须与全流程同进程调用。"""
    try:
        import spiral_metric_v16 as M
    except ImportError:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, os.path.join(root, 'v16'))
        import spiral_metric_v16 as M
    return M.METRICS, M.GUARDS, M._NEG_CTRL_TABLE


def claim_test_ledger(metrics=None, guards=None, table=None, verbose=True):
    """
    四条检查 (a)(b)(c)(d) + B3-3 的完备性检查 (e)。参数可注入 ⇒ 冒烟测试能用
    **人为构造的坏输入**逐条触发 (硬纪律 2: 真调本函数, 不复刻它的逻辑)。

    (a) 无孤儿主张   tier-1/2 且有 pass/fail 判定的指标, 至少被 1 条守卫指认。
                     指认 = claim_id 出现在某守卫 tests_claims 里,
                     或某守卫的 where 出现在该指标的 method 里 (规则兜底)。
    (b) 无幻影引用   tests_claims 里每个 id 都真实存在于 METRICS 的 claim_id。
    (c) 无伪独立     source_id 相同的读数必须与 _SAME_SOURCE_REGISTRY
                     **逐成员**一致; 登记表也不得指向不存在的数据源。
    (d) 无未归层项   每条注册项都要能推出层号 —— 推不出的**列名报出**, 不静默。
    (e) 负对照台账 claim 列完备: 每条对照条目都有登记。(L1 的 None 是**如实留缺**,
                     单独计数, 不算失败 —— 它正是 B1 要填的缺口。)

    返回 dict; `ok` = (b)(c)(d) 的合取。**(a) 只报不判** —— 口径匹配率是一张
    待办清单, 不是模型的性质; 把它算进通过率等于拿未做完的清单给自己打分
    (同 v15 `metric_negative_control_ledger` 对"逐层覆盖"的处理)。
    """
    if metrics is None or guards is None or table is None:
        _m, _g, _t = _live_registries()
        metrics = _m if metrics is None else metrics
        guards = _g if guards is None else guards
        table = _t if table is None else table

    referenced = {c for g in guards for c in g.get('tests_claims', ())}

    # ---- (a) 孤儿主张: 只报不判 -------------------------------------------
    orphans = []
    for m in metrics:
        if m.get('tier') not in (1, 2) or m.get('passed') is None:
            continue
        cid = m.get('claim_id')
        if cid and cid in referenced:
            continue
        meth = m.get('method') or ''
        if any((g.get('where') or '') and g['where'] in meth for g in guards):
            continue
        orphans.append(m['name'])

    # ---- (b) 幻影引用: 判 ------------------------------------------------
    known_ids = {m.get('claim_id') for m in metrics if m.get('claim_id')}
    phantom = sorted(c for c in referenced if c not in known_ids)

    # ---- (c) 伪独立: 判 --------------------------------------------------
    groups = {}
    for r in list(metrics) + list(guards):
        sid = r.get('source_id')
        if sid:
            groups.setdefault(sid, []).append(r['name'])
    same_source_problems = []
    for sid, names in sorted(groups.items()):
        declared = _SAME_SOURCE_REGISTRY.get(sid)
        if declared is None:
            same_source_problems.append((sid, '未登记同源', sorted(names)))
        elif set(declared) != set(names):
            same_source_problems.append((sid, '登记成员不符', sorted(names)))
    for sid, declared in sorted(_SAME_SOURCE_REGISTRY.items()):
        if sid not in groups:
            same_source_problems.append((sid, '登记表指向不存在的数据源',
                                         sorted(declared)))

    # ---- (d) 未归层项: 判 ------------------------------------------------
    unattributed = []
    for m in metrics:
        if layer_of_metric(m['name'], m.get('method')) is None:
            unattributed.append(('metric', m['name']))
    for g in guards:
        if layer_of_guard(g['name'], g.get('where')) is None:
            unattributed.append(('guard', g['name']))

    # ---- (e) 负对照台账的 claim 列 --------------------------------------
    real_entries = {(t[0], t[2]) for t in table}
    unknown_claim = sorted({c for c in _NEG_CTRL_CLAIM.values()
                            if c and c not in {m['name'] for m in metrics}})
    nc_unregistered = sorted(f"{L}/{lab}" for (L, lab) in real_entries
                             if (L, lab) not in _NEG_CTRL_CLAIM)
    nc_phantom = sorted(f"{L}/{lab}" for (L, lab) in _NEG_CTRL_CLAIM
                        if (L, lab) not in real_entries)
    nc_gap = sorted(f"{L}/{lab}" for (L, lab), c in _NEG_CTRL_CLAIM.items()
                    if c is None)
    nc_ok = (not unknown_claim) and (not nc_unregistered) and (not nc_phantom)

    # ---- 逐层口径匹配率 ---------------------------------------------------
    per_layer = []
    for L in _LAYERS:
        ms = [m for m in metrics
              if m.get('tier') in (1, 2) and m.get('passed') is not None
              and layer_of_metric(m['name'], m.get('method')) == L]
        matched = [m for m in ms if m['name'] not in orphans]
        per_layer.append({'layer': L, 'n_matched': len(matched),
                          'n_total': len(ms),
                          'n_guard': sum(1 for g in guards
                                         if layer_of_guard(g['name'],
                                                           g.get('where')) == L)})

    ok = (not phantom) and (not same_source_problems) \
        and (not unattributed) and nc_ok

    if verbose:
        print(f"\n=== v16 · B3 claim↔test 口径账本  [{_VERSION_TAG}] ===")
        print(f"    注册规模: 指标 {len(metrics)} 条, 守卫 {len(guards)} 条, "
              f"负对照 {len(table)} 条")
        print(f"    [(a) 孤儿主张] {len(orphans)} 条 / 分母 "
              f"{sum(p['n_total'] for p in per_layer)} 条 (**只报不判**）")
        for n in orphans:
            print(f"        · 无守卫指认: {n}")
        print(f"    [(b) 幻影引用] {phantom or '无'} -> "
              f"{'通过' if not phantom else '**失败**'}")
        print(f"    [(c) 伪独立]   {len(same_source_problems)} 条 -> "
              f"{'通过' if not same_source_problems else '**失败**'}")
        for sid, why, names in same_source_problems:
            print(f"        · [{sid}] {why}: {names}")
        print(f"    [(d) 未归层项] {len(unattributed)} 条 -> "
              f"{'通过' if not unattributed else '**失败**'}")
        for kind, n in unattributed:
            print(f"        · ({kind}) {n}")
        print(f"    [(e) 台账 claim 列] 未登记 {len(nc_unregistered)} / "
              f"幻影 {len(nc_phantom)} / 指向不存在的主张 {len(unknown_claim)}"
              f" -> {'通过' if nc_ok else '**失败**'}")
        for x in nc_unregistered:
            print(f"        · 未登记: {x}")
        for x in nc_phantom:
            print(f"        · 幻影条目: {x}")
        for x in unknown_claim:
            print(f"        · 主张不存在: {x}")
        if nc_gap:
            print(f"        · **如实留缺** (由 B1 填): {nc_gap}")
        print("    口径匹配率（分子 = 被既有守卫指认到的, "
              "分母 = 有 pass/fail 判定的 tier-1/2 指标）:")
        for p in per_layer:
            if p['n_total'] == 0:
                print(f"        {p['layer']}: **无计分指标** "
                      f"(同层守卫 {p['n_guard']} 条)")
            else:
                print(f"        {p['layer']}: {p['n_matched']}/{p['n_total']}"
                      f"   (同层守卫 {p['n_guard']} 条)")
        print(f"    同源登记: {len(_SAME_SOURCE_REGISTRY)} 组")
        for sid, names in sorted(_SAME_SOURCE_REGISTRY.items()):
            print(f"        [{sid}] {len(names)} 条读数 **同源**"
                  f" ⇒ 不得写成「互相印证/独立佐证」")
        print(f"    => (b)(c)(d)(e) 合计: {'通过' if ok else '**失败**'}")

    return {'orphans': orphans, 'phantom': phantom,
            'same_source_problems': same_source_problems,
            'unattributed': unattributed, 'per_layer': per_layer,
            'nc_unregistered': nc_unregistered, 'nc_phantom': nc_phantom,
            'nc_unknown_claim': unknown_claim, 'nc_gap': nc_gap,
            'ok': ok, 'version_tag': _VERSION_TAG}


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.exit(0 if claim_test_ledger()['ok'] else 3)
