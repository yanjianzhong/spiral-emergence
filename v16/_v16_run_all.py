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

【日志归档（2026-09-26 新增）】

`open(LOG, 'w')` 是**截断写** —— 上一次的记录会被静默覆盖。仓库此前靠**手工**改名保历史
（`_runall_fast_2026-09-25.log` / `_runall_full_2026-09-25.log` /
`_runall_only_before_22_2026-09-25.log` 三个都是这么来的）。现在这一步由 runner 接管:
开新日志前, 先把已存在的 `_runall.log` 改名为 `_runall_<YYYYmmdd-HHMMSS>.log`。

  - 归档失败（磁盘满 / 权限）**退码 2, 不静默继续** —— 静默继续就等于又覆盖了一次。
  - **只归档, 不清理**。清理分支失败的方式是**删证据**; 目录膨胀是可以事后处理的小事。
    两者不对称, 所以不设保留份数上限。
  - `--list` 与参数错误**不触发归档**（所有提前 return 都排在归档之前）。
  - 归档文件的 **mtime 是它被写下的时刻**, 不是归档时刻 —— `os.replace` 保留原 mtime。
    实测: `_runall_20260926-082034.log` 的 mtime 显示 `19:00:55`（前一天）, 而名字里是
    `082034`。**名字记归档时刻, mtime 记内容时刻, 两者本就不同, 不是缺陷。**

【运行环境快照（2026-09-26 新增）—— 只登记, 不复测、不解释】

为什么: v16 留下三条「读数一致、用时对不上」的差额（`997.3 vs 1292.0` /
`694.1 vs 91.8` / `1879 vs 1318.2`）。差额本身按纪律**只登记不复测**; 本条要做的是让
**下一次跑的时候环境证据自己就在日志里**, 否则差额永远无法归因。

  - 日志开头一块环境头（时间戳 / `platform` / 逻辑核数 / Python 版本）;
  - **每一条登记项**的 `EXIT=` 行尾附该条的**全机 CPU 忙占比** —— 三条差额都是**逐条**的,
    整场一个数归因不到具体条目。
  - 口径: 忙占比 = `(Δ(内核+用户) − Δ空闲) / Δ(内核+用户)`, 源为 `kernel32!GetSystemTimes`
    的**整机**累计计数。⚠️ Windows 的 `kernel` 累计**已含** idle, 写错会把忙占比算成接近 100%。
  - 它记的是「**这台机器当时在忙什么**」（含其它进程）, **不是子进程自身占用**。
  - **不因负载高低改判任何东西**: 不重跑、不标「干净」、不解释历史差额。
  - 非 Windows 平台 `windll` 不存在 ⇒ 一律写「不适用」, **不假装有数**。

  **实测标定（2026-09-26, 本机 8 逻辑核 / Python 3.13.9）**: 8 核满载 3 s 窗口读
  `100.0%`, 卸载后空转 3 s 读 `18.4%` ⇒ 公式与「`kernel` 已含 idle」的口径**正确**
  （若重复计数, 满载也只能读到约 50%）。
  ⚠️ **分辨率边界（实测, 不是推断）**: 本机**背景负载本身就在 18~26% 之间波动**,
  **单核**满载（≈ +12.5pp）的贡献被淹没 —— 实测「空转 2 s = 25.7%」与
  「单核烧 2 s = 22.4%」**无法区分**（后者甚至更低）。所以本字段只够判
  「**这台机器当时是否被打满**」, **不足以归因 20~30% 级别的用时差**。
  非 Windows 分支（`windll` 不存在 ⇒ 「不适用」）**代码已写但未实测**（本机是 Windows）。
"""

import argparse
import ctypes
import os
import platform
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
            '（v16_plan.md:1005 要求独立条目）。**根因是「观测量不存在」, 不是"未修好"**: '
            '口径 A 没有「层」/逐层生成元 ⇒ 探针 2 要算的余弦相似度**无对象**。'
            '⚠️ **2026-09-26 撤回一处旧表述**（旧文写「离散 L 厄米, 而 cMERA 的标度算符'
            '必须反厄米 ⇒ 离散格上不存在 k∂_k」）: 原文 `cMERA_Entropy_rev.tex:75`/`:87` '
            '把 L 放在 `e^{-i∫(K+L)du}` / `e^{-iûL}` 里 ⇒ 酉性要求 L **厄米** ⇒ 实测的厄米性'
            '（diagnostic_B：‖L−L†‖=2.482534e-16）**正是应有性质, 不是矛盾**。'
            '那次检验**平凡**的真原因是**态**：测的是 `uv_vacuum`, 被所有 c_q 湮灭 ⇒ '
            '任何**对角数算符**都给 0。有内容版见 `_v16_smoke_cmera_d.py`。'
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
    # ⚠️ 本段原两条是 2026-09-25 新增的;**注册条目数 18 -> 20 -> 22 -> 23** —— 与同日"高斯主体"段
    #    新增的 `gaussian.run_K4` 合起来, 2026-09-25 那轮共 +2（`run_K4` 与 `derive_presets`）;
    #    v16.4·D5–D7 再 +1（`presets_v164`, DP-4 默认 = 注册）。审计里登记过的 runner 条目计数
    #    需要同步改 —— **本文件注释先改到 23; 审计 `14.` 与 README/说明/CHANGELOG 的 fast 档计数
    #    由 v16.4·R-4 一并同步（这笔欠账已写进 v16.4_plan.md §0.1 的进度指针）**。这些条目
    #    **不加守卫也不加指标**(那会动分母 27/78), 只把原先只在 markdown 里的判断变成可执行、可复跑的检查。
    #    ⇐ **2026-10-04（第四轮, 23 -> 28）**: 上句「本文件注释先改到 **23**」是 **2026-09-25/09-27 那几轮的时态**,
    #       按本仓「带日期的存档不改 / append 新结论」**有意保留不改**。**现行 = 28 条**
    #       （v16.8·T6 阶段 5 后按用户裁定「全进: fast 4 + slow 1」新增 5 条, 见上方 T6 专题块）。
    #       同轮已同步: README / `spiral_v16_说明.md` / 审计 `14.`。**仍未加守卫/指标 ⇒ 27/78 依旧不动**。
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
            '而是**数学上不可能**; **但偶数中取 2 仍是选择**(v16.4·D5 订正 2026-09-27: 原文写'
            '「A5 实测 γ⊗I 不被排除」—— 该载体可约(dim comm=256) ⇒ **已被这条定理排除**)。(B) 复数域 **L1 处不承重**(L1 链上态与 TFI 基态实测都'
            '是实的, 相位不变判据 |Σψ²| = 1), 那个 `i` 是在**下游**由 `H_til` 的**厄米性**'
            '强制(γ_aγ_b 实测反厄米, 全部 56 对) —— 但其前提(**费米子表述**)本身是选择 ⇒ '
            '**与 (A) 同一个根**。'
            '⇒ 三条"给定"收敛为**一个**选择 + 它的两个推论(那条「最小性」已由 v16.4·D5 证明与同一表述选择是一回事, 不另计); 缺口比 §5.5 记的'
            '小, **但不是零** —— 本条目**不宣称缺口已修复**。**不加守卫/指标**(不动 27/78)。'),
    _e('presets_v164', ['_v16_smoke_presets_v164.py'], timeout_s=600,
       must_contain=('=== 结果: 全部通过 ===',),
       note='v16.4·α（D5/D6/D7）的收口冒烟。**它订正的是上一条 `derive_presets` 的结论 (A)** —— '
            '那里自陈「**偶数中取 2 仍是选择**」, 本条把那个"选择"换成一条**定理**: '
            '`Cl(2L) ≅ M_{2^L}(C)` 不可约表示唯一 ⇒ 2L 个 Majorana 在 `4^L = 2^{2L}` 上的'
            '**任何**载体都是 `2^L` 份不可约表示的直和 ⇒ 必可约（实测 L=4: dim comm = 256, '
            '随机酉共轭给同一读数; 同一批 γ 在 `2^L=16` 维上 dim comm = 1 ⇒ 不可约）。'
            '要不可约的 `d=4` 载体得 `4L` 个 Majorana = **每格点两对**。'
            '**⇒ 落支 S-部分: 排除理由从「一条最小性(选择)」换成「一条定理 + 那个表述选择」, '
            '选择的个数仍为 1, 天花板不变**（那条表述选择 = JW 的 site↔mode 对应 = 费米子表述, '
            '与「复数域」共用, 即 `derive_presets` 已登记的同一个根）。'
            'D6 落 **S-负**(`d_local=4` 不迫使 L1 变复), D7 落 **S-重言**(均匀极大化 Shannon 熵 '
            '与 Haar 唯一酉不变都是现成定理的并列)。'
            '⚠️ 三条都**不给 d_local=2 的升级条件 ① 补任何东西** ⇒ **本条目不宣称预设层缺口已修复, '
            'L1 仍记 B**。中心化子维数用**三种独立算法**对账（迹公式 / 直接零空间 / 算子迹）。'
            '**不加守卫/指标**(不动 27/78)。'),

    # ---- v16.8·T6 专题（hyMERA 的几何层与 L4/L5 真接入）----
    #
    # ⚠️ **2026-10-04 新增 5 条（runner 条目数 23 → 28）** —— 依据 `v16.8_plan.md` §4 红线 6:
    #    「隔离冒烟**不进 runner、不加守卫/指标** —— 直到**阶段 5 的登记完成后**才谈是否进」。
    #    阶段 5 登记（该计划 §0.1.13）完成后由**用户裁定: 全进（fast 4 + slow 1）**
    #    （fast = admission / tiling / layer / l4bridge; slow = l5measure）。
    #    **一律不加守卫 / 指标**（同 `presets` / `derive_presets` / `presets_v164` 的先例）
    #    ⇒ 规模三数 **78 / 27 / 13 一个不动**。
    #    断言取各脚本 stdout 的**落支句**（确定性结构句, 非浮点末位）, 与 T6 各阶段
    #    登记的落支名逐字一致 —— 落支名变了就是该被看见的事。
    #    ⚠️ `t6_l4bridge` 与 `t6_l5measure` **自写固定名日志**（`open(..., 'w')` **截断写**）
    #    ⇒ 每跑一次就重写 `_t6_l4bridge.log` / `_t6_l5measure.log`。原始冻结证据已按用户
    #    2026-10-04 裁定**复制存档**为 `_t6_l4bridge_frozen_2026-10-04.log` /
    #    `_t6_l5measure_frozen_2026-10-04.log`（与原件 SHA256 逐字相同:
    #    `693E970A…` / `B5D42636…`）⇒ 自此**原文件名表示「最近一次运行」**;
    #    计划 §0.1.9 / §0.1.12 引用的原始读数以 frozen 存档为准。这不是缺陷, 是登记在案的
    #    **截断写**, 与运行期 `LOG` 同一类问题（见文件头【日志归档】）。
    _e('t6_admission', ['_v16_smoke_t6_admission.py'], timeout_s=600,
       must_contain=('落支 = **T6-口径已钉**',),
       note='v16.8·T6 阶段 0 准入筛: 三口径逐条复现 T4 §9.10 表（θ 与容差均写死, 无自由参数）。'
            '实测 0.8 s。判据句里的 `**` 是脚本自己打的 Markdown 强调, 逐字对上。'),
    _e('t6_tiling', ['_v16_smoke_t6_tiling.py'], timeout_s=600,
       must_contain=('落支 = **T6-铺砌成立**',),
       note='v16.8·T6 阶段 1 `{7,3}` 双曲铺砌有限补丁（纯 stdlib + `complex`, **不 import numpy**）。'
            '实测 0.1 s。⚠️ 同目录的 `_t6_tiling_v1.log` 是判据 [1-2] 删除**之前**的首跑'
            '（EXIT=3）, **不作判据、只作证据** —— 两者别混同。'),
    _e('t6_layer', ['_v16_smoke_t6_layer.py'], timeout_s=600,
       must_contain=('落支：T6-层组装只需(12|34)',),
       note='v16.8·T6 阶段 2 层 `V` 组装 —— 该计划的**头号目标**。断言取 `落支` 行'
            '（注意是**全角冒号**）。实测 1.11 s。已登记的两处缺口: 层 V 是**抽象 cell 弦**、'
            '非从补丁几何抽出; **整层缩并未做**。'),
    _e('t6_l4bridge', ['_v16_smoke_t6_l4bridge.py'], timeout_s=900,
       expect_pass=False, expect_exit=3,
       must_contain=('落支 = T6-L4桥落空',),
       note='v16.8·T6 阶段 3 接 L4 的桥 1 —— **预登记的「判据不成立」路径, EXIT=3 是设计如此**'
            '（DP-31 允许落空）。与 `cmera2` / `b2_boson_control` 同类: **「单列」不等于「不检查」** '
            '—— 若某天桥 1 成立（EXIT 变 0）, 那是**已登记结果翻正**, runner 会红。'
            '桥 1 两前提**结构上不可同时满足**（稠密态向量接口 vs `L=1393 != 16`）。'
            '⚠️ import torch + quimb ⇒ 约 36.8 s 是 **import 代价**, 总实测 37.1 s。'
            '⚠️ **本条目会截断重写 `_t6_l4bridge.log`**（原始捕获见 frozen 存档）。'),
    _e('t6_l5measure', ['_v16_smoke_t6_l5measure.py'], tier='slow', timeout_s=1800,
       must_contain=('落支 = T6-L5测度仍零',),
       note='v16.8·T6 阶段 4 接 L5 —— 8 几何测度 × 2 对象, 判据**冻结于开跑前**'
            '（`v16.8_plan.md` §0.1.11）。**种子全写死**（`SEED_MERA=0` / `SWAP_SEED=7` / '
            '`seed=100+s,200+s`）⇒ 确定性, 落支句**不 flaky**。实测 258.4 s。'
            'tier=slow: 默认不跑。⚠️ 日志里的「完成 … 次交换」打印的是**图对象**'
            '（`nx.double_edge_swap` 不返回次数）—— 已登记的 bug, 不影响判据。'
            '⚠️ **本条目会截断重写 `_t6_l5measure.log`**（原始捕获见 frozen 存档）。'),

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


class _FILETIME(ctypes.Structure):
    """`GetSystemTimes` 的出参: 64 位计数拆成高/低两个 32 位。"""
    _fields_ = [('dwLowDateTime', ctypes.c_uint32),
                ('dwHighDateTime', ctypes.c_uint32)]


def _cpu_times():
    """读全机累计 `(idle, kernel, user)` 计数（100 ns 单位）; 取不到则 `None`。"""
    if not hasattr(ctypes, 'windll'):
        return None
    idle, kern, user = _FILETIME(), _FILETIME(), _FILETIME()
    try:
        ok = ctypes.windll.kernel32.GetSystemTimes(
            ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user))
    except OSError:
        return None
    if not ok:
        return None

    def _q(ft):
        return (ft.dwHighDateTime << 32) | ft.dwLowDateTime
    return _q(idle), _q(kern), _q(user)


def _cpu_busy_text(t0, t1):
    """两次采样之间的全机 CPU 忙占比, 形如 `  [全机 CPU 忙 12.3%, 8 核]`。

    ⚠️ `kernel` 累计**已含** idle（Windows 口径）, 故 `total = kernel + user`,
    `忙 = total − idle`。写成 `total = idle + kernel + user` 会把忙占比算成接近 100%。
    """
    if t0 is None or t1 is None:
        return '  [全机 CPU 忙: 不适用]'
    d_idle = t1[0] - t0[0]
    d_total = (t1[1] + t1[2]) - (t0[1] + t0[2])
    if d_total <= 0:
        return '  [全机 CPU 忙: 不适用]'
    return (f"  [全机 CPU 忙 {100.0 * (d_total - d_idle) / d_total:.1f}%, "
            f"{os.cpu_count()} 核]")


def _env_header():
    """本次运行的环境头, 写在日志开头。**只登记**, 不参与任何判定。"""
    now = time.time()
    return '\n'.join([
        '',
        '--- 运行环境（只登记, 不参与任何判定）---',
        f"    本机时间 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}"
        f"   (UTC {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(now))})",
        f"    platform {platform.platform()}",
        f"    CPU 逻辑核 {os.cpu_count()}   Python {sys.version.split()[0]}",
        '    全机 CPU 忙占比 = (Δ(内核+用户) − Δ空闲) / Δ(内核+用户), 源为',
        '      kernel32!GetSystemTimes 的**整机**累计计数 —— 含**其它进程**的活动,',
        '      记的是「这台机器当时在忙什么」, 不是子进程自身占用。逐条见其 EXIT 行。',
    ]) + '\n'


def _run(e, logf):
    logf.write('\n' + '=' * 78 + '\n')
    logf.write(f"[{e['name']}]  {' '.join(e['argv'])}\n")
    if e['note']:
        logf.write(f"  note: {e['note']}\n")
    logf.write('-' * 78 + '\n')
    logf.flush()

    argv = [sys.executable, '-u'] + e['argv']
    t0 = time.time()
    cpu0 = _cpu_times()      # 与环境头同一口径: 整机累计计数, 采样窗口 = 本条进程的存活期
    chunks = []
    try:
        # ⚠️ 子进程 stdout 的编码**由启动环境决定**: 环境里没有 `PYTHONIOENCODING` 时,
        #    Windows 上就是 `locale.getpreferredencoding()`(本机 cp936); 而下面按 **utf-8**
        #    解码 ⇒ 没自带 `sys.stdout.reconfigure(encoding='utf-8')` 的脚本整段变乱码,
        #    中文 `must_contain` 随之**假红**(2026-09-26 在 `_v16_smoke_cmera4.py` 上实测:
        #    子进程 EXIT=0、A/B/C 全"通过"、判定句确实打印了, 却因乱码被判"没有登记过的判定句")。
        #    这里**强制**给子进程一个 utf-8 的 stdout, 使 runner **不再依赖调用者的环境**。
        #    只改标准流编码, **不设** `PYTHONUTF8` —— 那会连带改 `open()` 的默认编码,
        #    可能改变读 GBK 数据文件的脚本的行为; 本行**不动任何读数**。
        child_env = dict(os.environ, PYTHONIOENCODING='utf-8')
        proc = subprocess.Popen(argv, cwd=HERE, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, env=child_env)
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
    # 尾部附该条的**全机** CPU 忙占比（只登记, 不改判; 见文件头【运行环境快照】）。
    logf.write(f"EXIT={rc}  用时 {dt:.1f} s  ->  {why}"
               f"{_cpu_busy_text(cpu0, _cpu_times())}\n")
    logf.flush()
    return ok, why, dt


def _archive_prev_log():
    """把已存在的 `LOG` 改名为带时间戳的归档, 返回归档路径; 无旧日志则返回 `None`。

    为什么需要: `open(LOG, 'w')` 是**截断写**, 上一次的记录会被静默覆盖。仓库此前靠
    **手工**改名保历史, 这一步把它交给 runner, 使「曾经绿过」与「现在仍绿」之间的桥
    不再依赖人记得改名。详见文件头【日志归档】。

    边界: **只归档, 不清理**。调用方负责把 `OSError` 转成退码 2 —— 归档失败必须让本次
    运行**停下来**, 否则又是一次静默覆盖。
    """
    if not os.path.exists(LOG):
        return None
    ts = time.strftime('%Y%m%d-%H%M%S')
    dst = os.path.join(HERE, f"_runall_{ts}.log")
    n = 1
    while os.path.exists(dst):
        # 同一秒内连跑两次时不覆盖既有归档 —— 本函数存在的理由就是「别丢证据」,
        # 它自己不能成为丢证据的那一环。
        dst = os.path.join(HERE, f"_runall_{ts}_{n}.log")
        n += 1
    os.replace(LOG, dst)
    return dst


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

    # 归档排在两个提前 return（`--list` / 未登记的 `--only`）之后: 它们不该动日志。
    try:
        prev_log = _archive_prev_log()
    except OSError as exc:
        print(f"!! 归档旧日志失败, 拒绝覆盖 {LOG}: {exc}")
        return 2

    sel = (f"--only {args.only}" if args.only
           else ('--full' if args.full else 'fast 层'))
    print(f"=== v16 runner  [{len(entries)} 条, {sel}] ===")
    print(f"日志: {LOG}")
    if prev_log:
        print(f"     上一份已归档 -> {os.path.basename(prev_log)}")
    for e in entries:
        print(f"  [{e['tier']}] {e['name']:<26} {'预期失败' if not e['expect_pass'] else ''}")

    t_all = time.time()
    rows = []
    with open(LOG, 'w', encoding='utf-8') as logf:
        logf.write(f"=== v16 runner  {time.strftime('%Y-%m-%d %H:%M:%S')}  "
                   f"{'--full' if args.full else 'fast'}, {len(entries)} 条 ===\n")
        if prev_log:
            logf.write(f"    (上一份日志已归档: {os.path.basename(prev_log)})\n")
        logf.write(_env_header())
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
