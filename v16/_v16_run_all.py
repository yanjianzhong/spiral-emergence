# -*- coding: utf-8 -*-
#
# Copyright (c) 2024-2026 Jianzhong Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""
_v16_run_all.py —— v16 全部冒烟脚本的**唯一 runner**
================================================================================
【本文件解决什么】

在此之前, v16 的 11 个 `_v16_*.py` **全靠手工 `python xxx.py` + 重定向** 留一份
`_*.log`。全仓没有任何 runner（2026-09-25 用 Glob 核实: 无 Makefile / .sh /
.ps1 / .bat / .cmd / CI 配置）。后果是一条具体的、可指名的缝:

  **「曾经绿过」与「现在仍绿」之间没有桥。**
  已知答案（`22/22`、`16/16`、`26/26`、`9/9`、`stage1 9/9`、`stage2check 4/4`）
  只存在于 `v16_plan.md` 的状态表里, 没有任何东西复核它们是否还成立。
  而恰恰就在 2026-09-25, `stage2check` 被复核时抓到一个真缺陷（S2a2 偏离 √2,
  根因是 `sign` 参数在两个函数里同名不同义）—— 这个缺陷从写下那天起就在,
  **状态表上它一直显示为绿**。

【设计上的三条硬约束（都是实测逼出来的, 不是偏好）】

1. **每个目标起独立子进程, 且必须如此。**
   `_v16_smoke_b1_void_limit.py` 猴补 `G.exact_ground_state`; `_v16_smoke_b1_ledger_row.py`
   的 docstring 自陈「**孤立**运行…这是孤立运行的性质, 不是缺陷」。所以同进程串行
   **不是中性**的 —— 它会静默改变读数。
2. **必须用 Python 驱动, 不能用 PowerShell。**
   PS 5.1 对原生 exe 的 `2>&1` 会把 stderr 包成 `NativeCommandError` 块写进日志
   （`_s2.log:61-77` 实证）, 污染解析。
3. **`EXIT=n  用时 X s` 那行由本 runner 接管。**
   它此前是 shell 外部追加的, 格式不稳定（`EXIT=0  用时 64.5 s` /
   `EXIT=0  WALL=17.6s` / 裸 `EXIT=0` / `_s2b.log` 里干脆没有）。

【计分口径 —— 照抄 `_metric/registry.py` 的 `guard_pass_rate()` 精神】

**该通过的**（`expect_pass=True`）与**只想看看的**（`expect_pass=False`）分开,
分母只算前者。这不是洁癖: `_v16_smoke_cmera2.py`（探针 2）是**已知故意的**
`EXIT=3`（`v16_plan.md:1005` 要求它「必须在此有独立条目」）。

把预期失败算进失败率, 和把真实负面结果算成「没通过」一样, 都是**让通过率失真**。
v15 的 `record_guard(..., expect_pass=False)` 就是为这件事立的。

退出码: `0` 全部符合登记预期 / `3` 有**本该通过**的项不符（回归）/ `2` runner 自身错误。

【未解决的诚实边界】

  - 子进程 stdout 是管道 ⇒ 子进程侧默认**块缓冲**。本 runner 用 `-u` 让子进程
    无缓冲, 所以能实时流式看到；但**不保证**逐行及时（取决于子进程自己的缓冲行为）。
  - 本 runner **不改任何脚本的行为**, 也不碰 cache（`_v13/_v14/_v15_cache` 只读）。
"""

import argparse
import os
import re
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, '_runall.log')

# 子进程输出的数字对, 例如 `22/22`。用于 `expect_counts` 断言。
_COUNT_RE = re.compile(r'(\d+)\s*/\s*(\d+)')


def _e(name, argv, tier='fast', expect_pass=True, expect_exit=0,
       expect_counts=None, must_contain=(), timeout_s=600, note=''):
    """登记一条。`name` 是 `--only` 用的唯一键; `argv` 是完整命令行（不含解释器）。"""
    return dict(name=name, argv=list(argv), tier=tier, expect_pass=bool(expect_pass),
                expect_exit=int(expect_exit), expect_counts=expect_counts,
                must_contain=tuple(must_contain), timeout_s=int(timeout_s),
                note=note, inproc=False)


# ---------------------------------------------------------------------------
# 注册表
#
# `expect_counts` / `must_contain` 里的数字来自 `v16_plan.md` §0.1 的**已登记读数**,
# 不是「跑一遍抄下来」—— runner 的用途正是把那张表变成可执行的断言。
#
# ⚠️ **登记依据（2026-09-25 逐行核实过 11 个脚本, 不是推断）**:
#   plan §0.1 里那些 `22/22` / `16/16` / `26/26` / `9/9` 是**脚本内部 `check()` 的
#   调用次数**, **不是**输出里的字符串 —— 大多数脚本根本不打印 `n_pass/n_total`。
#   所以断言必须落在**脚本真的打出来的那句判定**上（`must_contain`）, 而不是
#   「最后一条 `n/m`」。两者的差别不是风格问题: 按后者写, 三条 `=== 结果 ===` 脚本
#   会**全部假失败**（`b1_void_limit` 输出里一条 `n/m` 都没有）。
#   `expect_counts` 只在**确实打印了 `(n/m)` 汇总**的地方用（目前只有
#   `_v16_cmera_gaussian.py` 的 `run_K` / `stage2` 两个动词）。
# ---------------------------------------------------------------------------
ENTRIES = [
    # ---- 冒烟探针族 ----
    _e('cmera', ['_v16_smoke_cmera.py'], timeout_s=1800,
       note='离散化 cMERA 可行性（torch）。**只断言退出码**: 它的四条判定逐行打印, 没有'
            '汇总句。⚠️ 其中 **Q4 是「资源可接受 (<600 s)」** —— 一个**墙钟依赖**的判据, '
            '机器负载高时可能因非物理原因退 3。它红了先看 Q4 那一行, 别急着当物理回归。'),
    _e('cmera2', ['_v16_smoke_cmera2.py'], expect_pass=False, expect_exit=3,
       timeout_s=1800,
       note='**探针 2 尺度不变性 —— 在 v16 口径 A 下不适用, 故 EXIT=3 是设计如此**'
            '（v16_plan.md:1005 要求独立条目）。**根因已钉死, 不是"未修好"**: 口径 A 没有'
            '「层」/逐层生成元 ⇒ 探针 2 的观测量**不存在**; 且离散 L 是**厄米**的, 而 cMERA '
            '的标度算符必须是**反厄米**生成元 ⇒ 离散格上不存在同时满足两者的 k∂_k'
            '（厄米性判据逐行打印在 diagnostic_B 的输出里）。'
            '预期失败, 不进分母。⚠️ 「单列」不等于「不检查」: 若它某天变成 EXIT=0, '
            '那是**已知负结果翻正**, runner 会红 —— 那是该被看见的事, 不是误报。'),
    _e('cmera3', ['_v16_smoke_cmera3.py'], timeout_s=900,
       must_contain=('高斯表述: 可用',),
       note='自由费米子高斯表述自检（9 条 checks）。判定句逐字来自其 L316。'),
    _e('cmera4', ['_v16_smoke_cmera4.py'], timeout_s=900,
       must_contain=('K-1 玻色子基准: 复现成功',),
       note='探针 4 · cMERA 玻色子基准复现（K−1）。实测 65.4 s。'
            '其自身边界: 「这是**机器校准**, 不是物理结论」。'),
    _e('cmera5', ['_v16_smoke_cmera5.py'], timeout_s=600,
       must_contain=('生成元转录: 可用',),
       note='费米子 cMERA 生成元转录验证（9 条 checks）。实测 4.5 s。'),
    # ⚠️ **`expect_exit=3` 而不是 0**（2026-09-25 读代码核实后改的, 原先错写 0）。
    #    出口是 L458 `return 0 if (a and all(okb.values())) else 3`, 而 `okb` **包含**
    #    `B1` 与 `B5b`（L436-437 逐字 `for k in ('B1',...,'B5b','B6'): ... okb[k]`）。
    #    `_b2c.log` 里这两条**都是「未通过」** —— 且都是**已登记的**负面结果:
    #    B1 = 稠密 Fock 截断（边界 4）、B5b = 求积边界效应（边界 5）。故 `all(...)` 必为
    #    False ⇒ **`EXIT=3` 是设计如此**。这与 `cmera2` 是同一类: **头条结论为正, 退出码非零**。
    #    `expect_pass=False` ⇒ 不进 `n_pass/n_expect` 比率（避免把一个已登记的负面子结果
    #    算成「没通过」而粉饰通过率）; 科学结论由 `must_contain=('支持 H2',)` 断言。
    _e('b2_boson_control', ['_v16_smoke_b2_boson_control.py'], tier='slow',
       expect_pass=False, expect_exit=3,
       timeout_s=3600,
       must_contain=('支持 H2',),
       note='**B2 玻色子同格点对照 —— 判定 K1/K2 负面结果性质的唯一仪器。**'
            '实测 1292.0 s（`_b2c.log`）。这个时长是稠密 Fock 对照的固有代价, '
            'v16_plan.md:169-172 记的 21× 是**用错路径的估计**, 不是代码慢 ⇒ 不重划。'
            'tier=slow: 默认不跑。'),

    # ---- B1 / B3 台账族 ----
    #
    # ⚠️ **这三个脚本都不打印 `n/m` 通过计数**（2026-09-25 逐行核实, 见文件头「登记依据」）。
    #    plan 里记的 `22/22` / `16/16` / `26/26` 是 **`check()` 的调用次数**, 不是输出里的字符串。
    #    它们真正的判定句是三行同构的 `=== 结果: 全部通过 ===`。断言那句才是对的。
    #    曾按「最后一条 `n/m` = 通过计数」写 `expect_counts`, 会让三条**全部假失败**:
    #    `b1_void_limit` 输出里一条 `n/m` 都没有; `b1_ledger_row` 最后一条是诊断 `覆盖 {n}/7`;
    #    `b3_ledger` 最后一条是 `=> 有守卫指认 {X}/{scored}`。
    _e('b1_void_limit', ['_v16_smoke_b1_void_limit.py'], timeout_s=900,
       must_contain=('=== 结果: 全部通过 ===',),
       note='B1 冒烟(一) L1 下游连线。`check()` 22 次（plan 记的 22/22 指这个, 非输出字符串）。'
            '**猴补 `G.exact_ground_state`（记忆化）⇒ 必须独立进程**（其 memo 靠 '
            '`_ForwardingFacade.__setattr__` 写回 `_model.*` 才生效）。'),
    _e('b1_ledger_row', ['_v16_smoke_b1_ledger_row.py'], timeout_s=900,
       must_contain=('=== 结果: 全部通过 ===',),
       note='B1 冒烟(二) 台账 L1 格运行时耦合。`check()` 16 次。docstring 自陈须**孤立**运行: '
            '「只有 B1 的三条守卫在册…这是孤立运行的性质, 不是缺陷」⇒ 独立进程是硬要求。'),
    _e('b3_ledger', ['_v16_smoke_b3_ledger.py'], timeout_s=900,
       must_contain=('=== 结果: 全部通过 ===',),
       note='B3 台账。`check()` 24 次（plan 记的 26/26 指注册规模, 非输出字符串）。'
            '只读 `v15/_v15_run.log`（冻结的外部参照, plan :87）—— 不写盘。'),

    # ---- 高斯 cMERA 主体（逐动词, 与 stage2 同一组参数 ⇒ 读数可逐位对照）----
    _e('gaussian.stage1', ['_v16_cmera_gaussian.py', 'stage1'], timeout_s=600,
       must_contain=('阶段 1: 通过', '口径 A: H_til'),
       note='**冻结判据** 9/9。实测 3.4 s。任何改动都不得扰动它。'),
    _e('gaussian.stage2check', ['_v16_cmera_gaussian.py', 'stage2check'], timeout_s=900,
       must_contain=('阶段 2 自检: 通过', '4.203e-15', '口径 A: H_til'),
       note='阶段 2 自检 4/4。`4.203e-15` 是 S2a2 的已登记读数（2026-09-25 修 √2 后）—— '
            '钉住它是为了「修完之后没再漂」。实测 76.2 s。'),
    _e('gaussian.run_K4', ['_v16_cmera_gaussian.py', 'run_K4'], timeout_s=900,
       must_contain=('口径 A: H_til', '判定: 通过'),
       note='K4 负对照: 把 χ 换成**常数**, K1/K2 必须变差。断言两条 —— (i) 口径 A 横幅'
            '(audit §三 硬边界: 口径须出现在每处输出的第一行; run_K4 走 _stage2_preamble); '
            '(ii) `判定: 通过` 这个汇总句。'
            '⚠️ 判据语义边界(已写进 run_K4 的 docstring): 它只在生成元**非尺度不变**时才构成'
            '负对照; 本文件走变分原理, 故适用。'
            '⚠️ **用时存疑**: 2026-09-25 单独实测 **58.3 s**, 而先前登记的 run_K4 用时是 '
            '**173.4 s**(差 3.0×), 二者各自的 L 上下文**未核对** ⇒ 按审计 §七·7 纪律: '
            '**只登记, 不解释、不复测**。'
            '本条目是**用户 2026-09-25 点名加入**的(此前 audit §七·11 登记为「支持的动词、'
            '但未擅自加入注册表」)。'),
    _e('gaussian.run_K5', ['_v16_cmera_gaussian.py', 'run_K5'], tier='slow',
       timeout_s=1800,
       must_contain=('非空模 11 个', '口径 A: H_til'),
       note='K5 方差 = g_k²。**断言的是「非空模 11 个」这个结构整数**, 不是最差相对差 '
            '`6.04e-16` —— 后者按行打印、末位随浮点抖动, 钉死它会制造**假红**; '
            '而非空模数是判据的一半（「非空模 ≥ 2」）, 且是整数。'
            '⚠️ **用时存疑**: 2026-09-25 单独实测该动词 **694.1 s**, 而 `v16_plan.md:134` 登记的 '
            'L=8 K5 是 **91.8 s**（差 7.6×）。读数逐位相同（`6.04e-16` / 非空模 11）⇒ 不是算错, '
            '是**那 91.8 s 的来源不明**（可能是 stage2 里按小节切分的错配）。**此差额未解决, 未复测**。'),
    _e('gaussian.run_K6', ['_v16_cmera_gaussian.py', 'run_K6'], tier='slow',
       timeout_s=1800,
       must_contain=('口径 A: H_til',),
       note='K6 稠密重建保真度。实测 284.4 s。**不设字符串断言**: 它的判定是逐行打印的'
            '保真度与配对差, 没有稳定的汇总句 —— 只断言退出码, 这一条弱于其它条目, 如实登记。'),
    _e('gaussian.run_K', ['_v16_cmera_gaussian.py', 'run_K'], tier='slow',
       timeout_s=5400,
       expect_counts=(4, 4),
       must_contain=('口径 A: H_til',),
       note='K1/K2/K3 四配置组（含 L=32 复验）—— stage2 里最贵的部分。'
            '**用时此前从未单独测过**（plan :134 的「run_K 14.1 s」是 L=8 冒烟单点的耗时, '
            '不是这个四配置动词的）。'),
    _e('gaussian.stage2', ['_v16_cmera_gaussian.py', 'stage2'], tier='slow',
       timeout_s=7200,
       expect_counts=(7, 7),
       must_contain=('阶段 2 判定聚合', '口径 A: H_til'),
       note='**阶段 2 全流程。** 此前它的判定全部被丢弃、退出码硬编码 0；'
            '现在聚合后真的会退 3。判定聚合 7 条 = K3×4 + K5 + K4 + K6。'
            '实测 ≥2466.0 s（`_s2b.log` 是下界, 其内容早于当前源码: K5 表头不同、无 K6-b）。'),
    _e('gaussian.diagnostic_B', ['_v16_cmera_gaussian.py', 'diagnostic_B'], tier='slow',
       timeout_s=1800,
       must_contain=('口径 A: H_til',),
       note='诊断 B（离散 L, L=8）。**非门禁**: 其 docstring 明写「结果正负都登记, 不阻断主线」⇒ 恒退 0。'),

    # ---- 预设层 / 空桥 冒烟（audit §5.4 / §5.5）----
    #
    # ⚠️ 本段两条是 2026-09-25 新增的;**注册条目数 18 -> 20 -> 22** —— 与同日"高斯主体"段
    #    新增的 `gaussian.run_K4` 合起来, 本轮共 +2（`run_K4` 与 `derive_presets`）。审计里
    #    登记过的 runner 条目计数需要同步改。这些条目**不加守卫也不加指标**(那会动分母
    #    27/77), 只把原先只在 markdown 里的判断变成可执行、可复跑的检查。
    _e('presets', ['_v16_smoke_presets.py'], timeout_s=600,
       must_contain=('=== 结果: 全部通过 ===',),
       note='预设层审计 (§5.4 的三条"给定"预设)。把「给定」细分成「承重 / 不承重」: '
            '#1 d_local=2 **承重**(换掉则链在两处断: dim_match 里硬编码的 ==2**L 闸, '
            '与 l1_void_pushaway 的 2x2 R_y 显式 ValueError); #2 复数域 / #3 随机测度 '
            '**不承重**(唯一载体 Q 在链上无消费者, equal_state 与 seed 无关)。'
            '每条检验都配一个**活的反事实构造**。'
            '⚠️ 本条目"只细分、**不导出**任何预设"的读法**已被 `derive_presets` 取代** —— '
            '保留它是因为它仍是一条独立可跑的检查, 不是因为它还是现行口径。'),
    _e('l5_l6_bridge', ['_v16_smoke_l5_l6_bridge.py'], timeout_s=600,
       must_contain=('=== 结果: 全部通过 ===',),
       note='L5->L6 空桥 (§5.4)。**结论是「打不通」, 且这是本项目自身边界的必然结论**, '
            '不是技术不足 —— 唯一的「打通」语义是把 MERA 图几何当作物理空间, 而 B2 硬边界'
            '禁止「宣称涌现时空」。实测同时抓出一处 claim<->代码不符: KNOBS 里 n_lambda 的'
            '注释自称依据 xi/L 的尺度不变性, 而 xi/L 扫描对该函数**零影响**。'),
    _e('derive_presets', ['_v16_smoke_derive_presets.py'], timeout_s=600,
       must_contain=('=== 结果: 全部通过 ===',),
       note='预设层的**真导出**尝试（用户 2026-09-25 点名: 「去真的导出 d_local=2 或复数域」）。'
            '**它取代了此前「只把给定细分成承重/不承重」的读法** —— 那一读法被用户否掉。'
            '结论两条: (A) `d_local` 的**奇偶**被**初等**导出 —— γ₀,γ₁ 是反对易的对合 ⇒ '
            '`n₊=n₋` ⇒ dim 必为偶 ⇒ **d_local 必为偶**, 故 `d_local=3` 不是"代码没支持"'
            '而是**数学上不可能**; **但偶数中取 2 仍是选择**(A5 实测 `d_local=4` 上 γ⊗I 仍'
            '合法, 不被这条定理排除)。(B) 复数域 **L1 处不承重**(L1 链上态与 TFI 基态实测都'
            '是实的, 相位不变判据 |Σψ²| = 1), 那个 `i` 是在**下游**由 `H_til` 的**厄米性**'
            '强制(γ_aγ_b 实测反厄米, 全部 56 对) —— 但其前提(**费米子表述**)本身是选择 ⇒ '
            '**与 (A) 同一个根**。'
            '⇒ 三条"给定"收敛为**一个**选择 + 它的两个推论 + 一条最小性; 缺口比 §5.5 记的'
            '小, **但不是零** —— 本条目**不宣称缺口已修复**。**不加守卫/指标**(不动 27/77)。'),

    # ---- 主张台账 ----
    _e('claim_ledger.standalone', ['_v16_claim_ledger.py'], expect_pass=False, expect_exit=3,
       timeout_s=600,
       note='**已知必退 3, 且这是对的。** `METRICS`/`GUARDS` 是进程内全局, 独立进程里为空 ⇒ '
            '四条检查必然报「无孤儿主张」以外的空判。其 docstring 自陈需与全流程**同进程**。'
            '真正有效的调用在 `--full` 的 `full_pipeline+claim_ledger` 条目里。'
            '登记于此是为了让「孤立跑 claim_ledger 得到红」不再是困惑, 而是**已登记的事实**。'),
]

# `--full` 追加的**同进程**条目: 先跑 v16 全流程注册指标/守卫, 再在**同一个子进程**内
# 调 `claim_test_ledger()`。这正是用户 2026-09-25 选定的形式。
INPROC_SCRIPT = r'''
import sys, os
sys.path.insert(0, sys.argv[1])
import spiral_model_v16 as M
try:
    M.main()
except SystemExit as exc:
    if exc.code not in (0, None):
        print(f"!! 全流程 sys.exit({exc.code}) —— 不继续做同进程审计")
        sys.exit(exc.code)
from _v16_claim_ledger import claim_test_ledger
res = claim_test_ledger()
sys.exit(0 if res['ok'] else 3)
'''


def _entries_for(full):
    out = [e for e in ENTRIES if full or e['tier'] == 'fast']
    if full:
        e = _e('full_pipeline+claim_ledger',
               ['-c', INPROC_SCRIPT, HERE], tier='full', timeout_s=10800,
               note='--full 专有: `spiral_model_v16.main()` 全流程 **与** `claim_test_ledger()` '
                    '在**同一个子进程**内先后执行（同进程是硬要求, 见 claim_ledger docstring）。'
                    '放在一个子进程里而非 runner 自己进程里, 是为了不污染 runner、且崩溃不带走 runner。')
        e['inproc'] = True
        out.append(e)
    return out


# ---------------------------------------------------------------------------
# 执行
# ---------------------------------------------------------------------------

def _verdict(e, rc, out, timed_out):
    """返回 (符合预期?, 说明)。"""
    if timed_out:
        return False, f"**超时**（>{e['timeout_s']} s, 已杀）"
    if rc != e['expect_exit']:
        return False, f"退出码 {rc} ≠ 登记 {e['expect_exit']}"
    for s in e['must_contain']:
        if s not in out:
            return False, f"输出里没有登记过的判定句 {s!r}"
    if e['expect_counts'] is not None:
        hits = _COUNT_RE.findall(out)
        if not hits:
            return False, f"输出里找不到 `n/m` 计数（登记 {e['expect_counts']}）"
        got = (int(hits[-1][0]), int(hits[-1][1]))
        if got != tuple(e['expect_counts']):
            return False, f"计数 {got[0]}/{got[1]} ≠ 登记 {e['expect_counts'][0]}/{e['expect_counts'][1]}"
    return True, '符合登记预期'


def _run(e, logf):
    logf.write('\n' + '=' * 78 + '\n')
    logf.write(f"[{e['name']}]  {' '.join(e['argv'])}\n")
    if e['note']:
        logf.write(f"  note: {e['note']}\n")
    logf.write('-' * 78 + '\n')
    logf.flush()

    argv = [sys.executable, '-u'] + e['argv']
    t0 = time.time()
    chunks = []
    try:
        proc = subprocess.Popen(argv, cwd=HERE, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
    except OSError as exc:
        logf.write(f"!! 起进程失败: {exc}\n")
        return False, f"起进程失败: {exc}", 0.0

    # ⚠️ 超时必须由**独立看门狗**保证, 不能在读循环里判 `time.time() - t0`:
    #    读循环只在**收到一行输出时**才醒来, 所以子进程若静默挂死（无输出, 例如
    #    卡在某次 eigsh / 被 OOM 前的抖动）, 判据永远轮不到执行, `timeout_s`
    #    形同虚设 —— 整个 runner 会被一个哑进程挂住。
    killed = []
    wd = threading.Timer(e['timeout_s'], lambda: (killed.append(True), proc.kill()))
    wd.start()
    try:
        for raw in proc.stdout:
            line = raw.decode('utf-8', errors='replace').rstrip('\r\n')
            chunks.append(line)
            logf.write(line + '\n')
            logf.flush()
    finally:
        wd.cancel()
    rc = proc.wait()
    dt = time.time() - t0
    timed_out = bool(killed)

    out = '\n'.join(chunks)
    ok, why = _verdict(e, rc, out, timed_out)
    # 这一行此前由 shell 外部追加, 格式不稳定 —— 现在由 runner 接管。
    logf.write(f"EXIT={rc}  用时 {dt:.1f} s  ->  {why}\n")
    logf.flush()
    return ok, why, dt


def main():
    ap = argparse.ArgumentParser(description='v16 全部冒烟脚本的 runner')
    ap.add_argument('--full', action='store_true',
                    help='含 slow 层（b2c 1292 s、stage2 ≥2466 s 等）与同进程全流程审计')
    ap.add_argument('--only', default=None,
                    help='只跑逗号分隔的登记名（见 --list）')
    ap.add_argument('--list', action='store_true', help='列出注册表后退出')
    args = ap.parse_args()

    # ⚠️ `--only` 与 `--list` 都在**全表**上工作, 不受 tier 默认影响。
    #    这条是 2026-09-25 **实跑抓出来的缺陷**, 不是设计偏好: 原先两者都只用 tier 过滤后的
    #    表, 于是 `--only b2_boson_control`（tier=slow）被判「未登记」并退 2 —— 而它明明
    #    是登记过的; 更糟的是 `--list` 也看不到 slow 的名字, 用户**无从发现**该怎么点。
    #    口径: **显式点名压过 tier 默认** —— tier 只在「没点名」时决定跑哪些。
    all_entries = _entries_for(full=True)
    entries = _entries_for(args.full)

    if args.list:
        print(f"{'name':<26} {'tier':<5} {'expect':<7} {'exit':<4} {'timeout':>8}  note")
        for e in all_entries:
            exp = '通过' if e['expect_pass'] else '预期失败'
            print(f"{e['name']:<26} {e['tier']:<5} {exp:<7} {e['expect_exit']:<4} "
                  f"{e['timeout_s']:>7}s  {e['note'][:60]}")
        return 0

    if args.only:
        want = [s.strip() for s in args.only.split(',') if s.strip()]
        known = {e['name'] for e in all_entries}
        unknown = [s for s in want if s not in known]
        if unknown:
            print(f"!! 未登记的 --only 名字: {unknown}")
            print(f"   已登记: {sorted(known)}")
            return 2
        entries = [e for e in all_entries if e['name'] in want]

    sel = (f"--only {args.only}" if args.only
           else ('--full' if args.full else 'fast 层'))
    print(f"=== v16 runner  [{len(entries)} 条, {sel}] ===")
    print(f"日志: {LOG}")
    for e in entries:
        print(f"  [{e['tier']}] {e['name']:<26} {'预期失败' if not e['expect_pass'] else ''}")

    t_all = time.time()
    rows = []
    with open(LOG, 'w', encoding='utf-8') as logf:
        logf.write(f"=== v16 runner  {time.strftime('%Y-%m-%d %H:%M:%S')}  "
                   f"{'--full' if args.full else 'fast'}, {len(entries)} 条 ===\n")
        for e in entries:
            print(f"\n>>> {e['name']} ...", flush=True)
            ok, why, dt = _run(e, logf)
            print(f"<<< {e['name']}: {why}  ({dt:.1f} s)", flush=True)
            rows.append((e, ok, why, dt))
        total = time.time() - t_all

        # 计分: 形状照抄 guard_pass_rate() —— 预期失败单列, 不进分母。
        exp = [(e, ok, why, dt) for (e, ok, why, dt) in rows if e['expect_pass']]
        diag = [(e, ok, why, dt) for (e, ok, why, dt) in rows if not e['expect_pass']]
        n_pass = sum(1 for (_, ok, _, _) in exp if ok)
        n_expect = len(exp)
        tally = {'n_pass': n_pass, 'n_expect': n_expect,
                 'rate': (n_pass / n_expect) if n_expect else float('nan'),
                 'n_diag': len(diag),
                 'n_diag_notok': sum(1 for (_, ok, _, _) in diag if not ok)}

        logf.write('\n' + '=' * 78 + '\n=== 汇总 ===\n')
        logf.write(f"  该通过的: {tally['n_pass']}/{tally['n_expect']}"
                   f"  ({tally['rate'] * 100:.1f}%)\n")
        for (e, ok, why, dt) in exp:
            logf.write(f"    {'✓' if ok else '✗'} {e['name']:<26} {dt:7.1f}s  {why}\n")
        logf.write(f"  预期失败(单列, 不进分母): {tally['n_diag']} 条, "
                   f"其中未按预期失败 {tally['n_diag_notok']} 条\n")
        for (e, ok, why, dt) in diag:
            logf.write(f"    {'✓(如预期)' if ok else '!!未如预期'} {e['name']:<26} "
                       f"{dt:7.1f}s  {why}\n")
        logf.write(f"\n总用时 {total:.1f} s\n")

    print('\n' + '=' * 78)
    print(f"  该通过的: {tally['n_pass']}/{tally['n_expect']}  ({tally['rate'] * 100:.1f}%)")
    for (e, ok, why, dt) in exp:
        print(f"    {'✓' if ok else '✗'} {e['name']:<26} {dt:7.1f}s  {why}")
    print(f"  预期失败(单列): {tally['n_diag']} 条, 未如预期 {tally['n_diag_notok']} 条")
    for (e, ok, why, dt) in diag:
        print(f"    {'✓(如预期)' if ok else '!!未如预期'} {e['name']:<26} {dt:7.1f}s  {why}")
    print(f"\n总用时 {total:.1f} s   日志: {LOG}")

    # 退出码口径（**「单列」不等于「不检查」**——这两件事必须分清）:
    #   预期失败的项不进 `n_pass/n_expect` 这个**比率**（否则通过率被粉饰）,
    #   但它仍与**自己的登记值**比对。若把它整个放过, `cmera2` 从 EXIT=3 悄悄变成
    #   EXIT=0（= 已知负结果翻正, 这正是科学上最该被看见的一类变化）会无人察觉。
    n_bad = (n_expect - n_pass) + tally['n_diag_notok']
    return 0 if n_bad == 0 else 3


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        pass
    sys.exit(main())
