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
"""v16 的 metric 实现子包 —— 对外门面是 spiral_metric_v16.py。

本包不由任何人直接 import, 只由 spiral_metric_v16.py 做 re-export。对外请用门面:
那是唯一稳定的 API, 本包内部怎么切可以随时再动。

模块地图 (自下而上; 列出的是该模块负责的顶层单元数):
    registry    31 个   指标/守卫注册表 + 打印与 JSON + 维度无关场工具 + 跨层共用常量
    solvers     14 个   共用求解器 + 既有指标函数 (第一层 1.3~1.5 / 第二层 2.1~2.4)
    layer1       5 个   v14 第一层·分布层 (多种子 / 稳健性 / 预算 / chi 瓶颈)
    tier3_desc  14 个   第三层·结构描述子与谱工具
    tier3        1 个   第三层·结构对结构梯子 (A/B/C 段 + 流程守卫)
    negctrl      3 个   负对照 T3 段 (F2a / F1 / F5)
    extras       9 个   W1·W3a/W3b·W5·L1 负对照·W6 台账·几何稳健性
    main         2 个   编排入口 collect_metrics_v16 + 20 面板作图 plot_metrics_v16

模块头部有一段自动推导的跨模块 import, 那就是它真正依赖的名字;
import 方向自下而上, 无环。

拆分原则 —— 沿**原有的 `# ===` 分节边界**切, 不重新组织论证结构:
中文注释在这里是资产 (它们记的是"为什么"), 所以随所属单元整块搬走,
不删、不改、不摘要。每个顶层单元按源码行号逐字切片, 函数体未改一字。
全文件仅有的改动是几处靠 __file__ 推导的路径和 W11 的源码扫描范围,
每一处都在代码旁边写明了原委。
"""
