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
"""v16 的 model 实现子包 —— 对外门面是 spiral_model_v16.py。

本包不由任何人直接 import, 只由 spiral_model_v16.py 做 re-export。对外请用门面:
那是唯一稳定的 API, 本包内部怎么切可以随时再动。

模块地图 (自下而上; 列出的是该模块负责的顶层单元数):
    core        22 个   派生层 (旋钮/因果链台账/L1..L6 派生) + 共用求解器 + 跨阶段复用的测量工具
    stage12      9 个   阶段一·空无 / 阶段二·对称性破缺 + v14·F2a h/J 负对照扫描
    stage3       9 个   阶段三·自指的动力学涌现
    mera        16 个   阶段四·自指画定边界投影出全息 + 第一层的物理迭代 (v13 检查点选择)
    stage5       6 个   阶段五·全息自然展开生成了时空 (曲率几何)
    stage67      4 个   阶段六·时空孕育显生命 (Gray-Scott) + 闭环螺旋 + 阶段七·意识自照见
    checks      18 个   F1 跨边界条件中心荷 + v15·W3a/W5/W3b + 任务A + F5 一致性检查束
    main         2 个   主程序与总图 (plot_all)

模块头部有一段自动推导的跨模块 import, 那就是它真正依赖的名字;
import 方向自下而上, 无环。

拆分原则 —— 沿**原有的 `# ===` 分节边界**切, 不重新组织论证结构:
中文注释在这里是资产 (它们记的是"为什么"), 所以随所属单元整块搬走,
不删、不改、不摘要。每个顶层单元按源码行号逐字切片, 函数体未改一字。
全文件仅有的改动是几处靠 __file__ 推导的路径和 W11 的源码扫描范围,
每一处都在代码旁边写明了原委。
"""
