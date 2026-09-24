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
spiral_metric_v16.py
====================
指标层 (体检报告) —— 与 spiral_model_v16.py 配对, 由它 import。

【本文件解决什么】
  把"打印一个数字"变成"可证伪的判定"。每个量绑上
      定义 / 计算方法 / 参照(基线) / 目标 / 实测 / 判定
  于是"达标与否"是一个 bool, 而 bool 是可以被证伪的。
  v11 说"曲率锚点全部到机器精度"——那句话没有任何代码能判定它错。
  两块基石: METRICS (指标注册表) + GUARDS (可证伪守卫注册表)。

【为什么单独一个文件, 而不是改 spiral_metric.py?】
  已发布的体检报告必须**保持可复现**。spiral_metric.py 被多个版本共用,
  在里面动一行, 所有历史数字的前提就变了。所以本文件是它的
  **自包含冻结副本 + 本层新增** —— 文件里那段 `from spiral_metric import (...)`
  已整体注释掉, 保留下来是作为**出处留痕**, 不是活的依赖。
  代价是两份可能分歧; 两个文件都是冻结快照, 所以这个代价可接受,
  但读者必须知道它存在。

【文件结构 (按出现顺序)】
  1 注册表与打印       record_metric / record_guard / guard_pass_rate /
                       _jsonable / print_health_report
  2 维度无关辅助函数   _band_power / _xi_from_fft / _char_scale /
                       _shape_residual / _mode_census / _delta2 / _first_zero
  3 既有指标函数       1.3 谱隙闭合 / 1.4 MERA 一致性 / 2.1 因果链 RMSE /
                       2.2 纠缠熵 KL / 2.3 关联衰减指数 / 2.4 Jordan 块收敛率
                       (+ exact_ground_state / tfi_periodic_sparse /
                        boundary_correlation_graph 这三个共用求解器)
  4 v14 第一层分布层   metric_central_charge_error_seeds /
                       metric_conformal_invariance_seeds /
                       metric_design_robustness / metric_budget_diagnostic /
                       metric_chi_bottleneck
  5 第三层描述子与梯子 structural_descriptors / descriptor_distance /
                       _series_lag_agreement / tier3_structural_ladder (A/B/C)
  6 v14 负对照         F2a h/J 扫描 / F1 跨边界 twist / F5 一致性检查束
  7 v15 新增           metric_spectral_central_charge (W3b 能谱族第二把尺子) /
                       _NEG_CTRL_TABLE + metric_negative_control_ledger (W6 台账)
  8 编排与作图         collect_metrics_v16 (总入口) / plot_metrics_v16 (20 面板)

【报告口径 (v15 实测: 指标 18/18, 守卫 54/54)】
  指标 18/18 达标 = 第一层 1.1~1.5 (5 项) + 第二层 2.1~2.13 (13 项)。
      第三层 3.1~3.6 **全部是诊断项** (passed=None), 不进分母。
      另有第一层诊断项 1.6 预算漂移 / 1.7 chi 瓶颈归因。
  守卫 54/54 = **计分守卫**; 另有 20 条诊断项, 其中 **17 条设计上就该失败**
      (如 V6: 派生 N 之后变体 A 收敛到 x*=0, 权重失去方向)。
      由 record_guard(..., expect_pass=False) 在**注册时**显式标出, 排除出分母。
      不分开计的后果是把**真实的负面结果**算成"没通过", 通过率就成了粉饰。

第一层的两处改动 (都是"诊断驱动", 不是"为达标而调参"):
  1.1/1.2 从单点值 -> 多种子分布, 预算 600 -> 1500 步。
      依据见 _v13_diag_conv: eps_c 在 step=600 处仍陡降, 11.45% 是预算产物。
  新增 1.5 全程稳健性: 结论必须在**所有**轨迹上成立 (种子 x 学习率),
      不是在中位数上成立。这是防止"挑一条好看的曲线"的唯一硬约束。
  新增诊断 1.6/1.7: chi 扫描 —— 直接检验基线 note 里"主因是有限键维截断"这句
      断言。实测 chi=4 在充分优化下达标, 所以那句话是错的, 这里如实记录。

第三层 (v14 起从 **3D CDM 场对场** -> **2D 斑图结构对结构**), 三段梯子:
  A 段 同方程不同盒子: 阶段六 Gray-Scott (36^2) <-> CLIP Gray-Scott (100^2, 3 种子)。
       参考侧是**数值解**, 所以这一段不受成像与标定污染, 而且有 3 个种子 ——
       第一次有了**种子间散布**可以当噪声底 (场对场口径恰恰没有这个)。
       **但不是干净的"同方程不同盒子"**: 实测 f/k 落在 Pearson 相图的不同相区
       (模型 F=0.035, k=0.060 斑点相 / 参考 f=0.01, k=0.042 蠕虫相), 这不是
       同一个动力学区, "同一 PDE"只在方程形式意义上成立 —— 单这一条就足以使
       D_struct **无法单独归因给盒子大小**, 这是本段不设达标线的直接理由。
       **已撤回的一条**: 此前并列的第二条混杂因子"盒子内波长数差一个量级"
       在 v14 撤回 —— 旧值来自坏参考场 (归档 bool 场的 lam_pk=1.45 格恰是二维
       离散谱角点伪像); 换成自建种子后两侧 n_lam 几乎相同。但也不能反过来说
       "盒子大小无影响": n_lam 是 kpk_k1 的换写、同样取自离散壳, 不是独立测量。
  B 段 跨系统: CLIP Gray-Scott <-> Southampton BZ 时空图 / Reading 时序。
       交付物是"哪些描述子能迁移、哪些不能", 不是"模型复现了实验"。
  C 段 判别力: 跨系统距离 / 系统内散布。若这个比值 ~1, 那 B 段无论什么结果
       都是无信息的 —— 它把"0/4 达标"从"失败"变成"该判据本就分不开"。

**轴语义的边界 (贯穿 B 段)**: Southampton 的图是 space x time, 不是二维空间斑图。
两个轴的含义不同, 把它按二维场算描述子只是"用同一套尺子量", 不构成
"实验斑图与数值斑图形态一致"的证据。所有 B 段数字一律 passed=None。

【v15 新增的两块 (详见各自函数头)】
  W3b 能谱族 (`metric_spectral_central_charge`): 低能能谱比值 (E2-E0)/(E1-E0) -> 8
      给出 c, 这是**独立于纠缠谱族**的第二把尺子 (Rényi 族实测偏差 7.97%, 已
      降级为诊断)。实测 c 落在 [0.49894, 0.50034], 最大偏差 0.212%。
      **口径警告**: "ratio->8" 与 "c->0.5" 是**同一次测量** (0.212% 就是 ratio
      偏离 8 的量), 不许写成"两族互相印证"; 且读数**不随 L 单调收敛** (L=16
      最差, O(1/L) 有限尺寸修正), 只能说"相容到 0.21%", 不能说"外推到 0.5"。
  W6 负对照台账 (`_NEG_CTRL_TABLE` + `metric_negative_control_ledger`):
      把散在各层的对照按**七个物理阶段**归位 (此前只有 F1/F2a/F5/F6/F7 这套按
      批次编的号, 读者无法判断哪层被覆盖)。实测 12 条覆盖 **6/7 层**, 缺口
      `['L1']` (W12 之前是 5/7, 缺口 `['L1','L4']`)。
      配一条**台账完整性守护**: 交叉核对台账引用的守卫名是否真实存在于 GUARDS、
      七层是否各出现一次。这不是物理结论, 是**台账自身的完整性** —— 没有它,
      台账可以写成宣传册 (引用不存在的守卫、或漏掉某层)。
      **判据曾整个判反** (v15 首次全流程实测抓到 3 条幻影引用): 有 3 个条目写入
      的是"XX 对照已运行"这类**上报守卫**, 而它们在"该对照未执行"的提前返回
      分支里才注册 —— 于是完整性检查在"对照正常执行"(健康情形)下**失败**、
      在"对照缺失"时反而**通过**。已改为只引用**对照本身**的守卫。
      覆盖层数**故意不计分** (expect_pass=False): "逐层"是审计表提的目标, 不是
      模型的性质; 把它算进通过率等于拿一张待办清单给自己打分, 会把真实缺口
      (L1 无负对照) 用一个百分比掩掉。

【依赖】 numpy / scipy / networkx (无 torch, 本层不训练)
【被谁 import】 spiral_model_v16.py: collect_metrics_v16 / plot_metrics_v16 /
              metrics_for_json
"""

import os
import json

import networkx as nx
import scipy.sparse as sp
import numpy as np

from scipy import ndimage
from scipy.sparse.linalg import eigsh

# ============================================================================
# 拆分后的 re-export: 对外 API 与拆分前完全一致
# ============================================================================
from _metric.registry import (
    METRICS, GUARDS, _TIER_TITLE, _TIER_SHORT, record_metric, record_guard, guard_pass_rate, _jsonable, _deep_jsonable, _fmt_val, print_health_report, _cic_power_window, _kgrids, _band_power, _delta2, _trapz, _xi_from_fft, _first_zero, _mode_census, _model_field, _char_scale, _shape_residual, CACHE_DIRNAME, V14_CACHE_DIRNAME, V15_CACHE_DIRNAME, V13_A_BASELINE, PANEL_SEEDS, EPS_T, RC_T, PROM_MIN, PROM_WIN,
)
from _metric.solvers import (
    entanglement_curve, fit_central_charge, _schmidt_spectrum, metric_gap_closure, metric_mera_consistency, metric_chain_rmse, metric_entropy_kl, metric_jordan_peak, metric_correlation_exponent, exact_ground_state, boundary_correlation_graph, tfi_periodic_sparse, _char_scale_checked, metrics_for_json,
)
from _metric.layer1 import (
    metric_central_charge_error_seeds, metric_conformal_invariance_seeds, metric_design_robustness, metric_budget_diagnostic, metric_chi_bottleneck,
)
from _metric.tier3_desc import (
    _KEYS, _late_frame, _binarize_median, _ac_len, _ds, structural_descriptors, descriptor_distance, _norm_pk_curve, _peak_period_stats, _series_lag_agreement, _log_bin, _leading_dark, _window_drift_ratio, _series_period,
)
from _metric.tier3 import (
    tier3_structural_ladder,
)
from _metric.negctrl import (
    metric_hj_negative_control, metric_f1_cross_boundary, metric_consistency_checks,
)
from _metric.extras import (
    _NEG_CTRL_TABLE, metric_selfref_N_dependence, metric_spectral_central_charge, metric_area_vs_log_law, metric_finite_size_scaling, metric_l1_void_link, metric_negative_control_ledger, _pctile_rank, metric_geometry_robustness,
)
from _metric.main import (
    collect_metrics_v16, plot_metrics_v16,
)


# ---------------------------------------------------------------------------
# 门面赋值转发 —— 恢复"拆分前一个模块"的可打桩语义
#
# 理由同 spiral_model_v16 门面: 拆分后同一个名字在 _metric/ 里有多份独立绑定,
# 只改门面那份等于没改 —— 补丁被**静默吞掉**, 打桩看着成功、实际没生效, 数值上
# 只表现为 ~1e-15 的抖动。这里把赋值同步写回每一份真正持有该名字的子模块。
#
# 只在**赋值**时才走这条路: 正常运行没有代码给门面属性赋值, 而且
# `from ... import` 走 STORE_NAME 直接写 __dict__, 不经过 __setattr__ ——
# 所以对主流程惰性, 不影响已验证的数值结果。
# ---------------------------------------------------------------------------
_ModType = __import__('types').ModuleType   # 刻意不写 `import sys/types`: 原件里
                                            # 两者都没有, 多写会往门面的公开命名
                                            # 空间里塞名字, 而门面的契约是"对外可见
                                            # 的名字与拆分前逐一相同"。


class _ForwardingFacade(_ModType):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        for _n, _m in list(__import__('sys').modules.items()):
            if _n == '_metric' or _n.startswith('_metric.'):
                if name in vars(_m):
                    setattr(_m, name, value)


__import__('sys').modules[__name__].__class__ = _ForwardingFacade

