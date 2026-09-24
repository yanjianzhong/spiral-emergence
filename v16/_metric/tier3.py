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
"""spiral_metric_v16._metric.tier3 —— 由 spiral_metric_v16.py 拆分。

第三层·结构对结构梯子 (A/B/C 段 + 流程守卫)

**纯移动产物**: 每个顶层单元按源码行号逐字切片, 函数体未改一字;
改动只在靠 __file__ 推导的路径与 W11 源码扫描范围两处 (见门面说明)。
"""

import os
import sys
import json
import platform
import time
import warnings
import contextlib
import io

import numpy as np
import scipy.sparse as sp
from scipy import ndimage   # 只有 tier3 用到, 但宽 import 才挡得住"漏 import"这类事故
from scipy.sparse.linalg import eigsh
from scipy.optimize import linprog
import networkx as nx

import matplotlib
matplotlib.use('Agg')      # 原文件里 matplotlib 一进来就是 Agg (无窗口); 见门面
import matplotlib.pyplot as plt

import torch
import quimb.tensor as qtn

# ---- 跨模块依赖 (自动推导, 仅列本模块实际用到的名字) ----
from _metric.registry import (
    CACHE_DIRNAME, PANEL_SEEDS, PROM_MIN, V13_A_BASELINE, V14_CACHE_DIRNAME, V15_CACHE_DIRNAME, _band_power, _delta2, _model_field, _shape_residual, record_guard, record_metric,
)
from _metric.solvers import _char_scale_checked
from _metric.tier3_desc import (
    _KEYS, _ac_len, _binarize_median, _ds, _late_frame, _leading_dark, _norm_pk_curve, _peak_period_stats, _series_lag_agreement, _series_period, _window_drift_ratio, descriptor_distance, structural_descriptors,
)




# ---------------------------------------------------------------------------
# 第三层编排: 三段梯子
# ---------------------------------------------------------------------------
def tier3_structural_ladder(s6, data_dir, clip_seeds=(0, 5, 10),
                            model_params=None):
    """
    第三层 —— 从"宇宙学对标"降级为"方法学展示"。

    降级的具体含义 (不是换个措辞就算降级):
      - 对标对象: 3D CDM -> 2D 反应-扩散 / 斑图系统 (CLIP Gray-Scott)。
      - 对标方式: 场对场 -> 结构对结构 (6 个无量纲描述子 + 归一化谱形)。
      - **裁决方式**: 原先对第三层设了物理达标线并得出 "0/4 达标", 现在取消
        达标线, 第三层只报数字与边界。理由不是"想让它好看", 而是: 参考数据
        的混杂因子 (相区参数不同、时空图轴语义不同、实验影像分辨率过低;
        原列的"盒子内波长数量级差"**已被 v14 撤回** —— 旧值来自坏参考
        场的谱角点伪像, 见下方【混杂因子】小节) 使得任何一个数字都无法单独归因给物理, 因此在
        这个区间做达标裁决本身没有信息量 —— "0/4" 就是这种无信息量判决的
        产物。判定项改为**流程守卫** (数据能不能用、流程有没有产出有限值),
        那才是这一层真正能支撑的结论。

    model_params: {'F':..., 'k':...} —— 用来在报告里点明模型与参考的相区差异。
    """
    print("\n" + "=" * 76)
    print("  第三层 · 结构对结构 (2D 反应-扩散 / 斑图)")
    print("=" * 76)
    cache = os.path.join(data_dir, CACHE_DIRNAME)
    v14cache = os.path.join(data_dir, V14_CACHE_DIRNAME)
    need = {'clip': 'clip_gs.npz', 'soton': 'bz_soton.npz',
            'reading': 'bz_reading.npz', 'v14seed': 'gs_seeds.npz'}
    files = {k: os.path.join(cache, v) for k, v in need.items()}
    files['v14seed'] = os.path.join(v14cache, need['v14seed'])
    missing = [k for k, p in files.items() if not os.path.isfile(p)]
    if missing:
        print(f"  缓存缺失 {missing} (归档侧位于 {cache}, 自建侧位于 {v14cache})"
              f" -> 第三层如实报告为『跳过』")
        record_metric(
            3, '第三层数据源', 'data/_v13_cache/*.npz 与 data/_v14_cache/gs_seeds.npz 是否存在',
            '第三层全部指标依赖离线预处理缓存; 归档侧 (_v13_cache) 是随仓库附带的'
            '既存产物, **A 段自建参考侧**由 spiral_v14_prepare.py 生成, v15 另加'
            '一层同单位受控参考侧, 由 v15/spiral_v15_prepare.py 生成。',
            f'os.path.isfile({cache}/*.npz) 且 os.path.isfile({v14cache}/gs_seeds.npz)',
            'CLIP / Southampton / Reading + v14 自建 GS 种子',
            f'缺失 {missing} -> 跳过', '全部缓存齐备', '缓存齐备', False,
            note='这是不可计算, 不是"不达标"。**本仓库里可运行的生成脚本只有两个**: '
                 'spiral_v14_prepare.py (自建参考侧) 与 v15/spiral_v15_prepare.py '
                 '(同单位受控侧)。归档侧 (_v13_cache) 的生成脚本 '
                 '**spiral_v13_prepare.py 不在本仓库里** —— 它是随数据附带的既存'
                 '缓存, 不要去跑一个不存在的脚本。缺哪侧补哪侧。')
        return {'skipped': True, 'missing': missing}

    print(f"  缓存: 归档侧 {cache}, 自建侧 {v14cache}")
    print("  **方法论说明**: 对标对象已从 3D CDM 换成 2D 反应-扩散/斑图系统,")
    print("     对标方式从『场对场』换成『结构对结构』—— 比的是无量纲描述子,")
    print("     不是场值。本层是方法学展示, 不构成任何『复现了实验』的论断。")

    out = {'skipped': False}

    # ---------------------------------------------------------------- A 段
    print("\n" + "-" * 76)
    print("  A 段 · 同方程同相区: 阶段六 Gray-Scott (36^2) <-> 自建 Gray-Scott (100^2)")
    print("-" * 76)
    print("  【v14·F3 目标重述】**本段不再声称测量『盒子大小的影响』**。")
    print("  原目标是『同方程不同盒子』, 但实测两个混杂因子使 D_struct 无法")
    print("  单独归因给盒子: (1) 双方 f/k 落在 Pearson 相图的不同相区。原列的第 (2) 条")
    print("  『盒子内波长数不同』**已被 v14 撤回** —— 旧值来自坏参考场的谱角点伪像,")
    print("  见下方【混杂因子】小节。v14 把目标收缩到本段真正能支撑的那一句话 ——")
    print("  **『这套无量纲描述子能不能把模型侧与参考侧的斑图家族区分开』**, 并以")
    print("  参考侧自身的种子间散布为噪声底。盒子大小仍是已知混杂因子, 按任务书")
    print("  §5c 退出本段, 不再作为可测结论。")
    print("  **参考侧换了来源 (v14·F3)**: 不再直接搬归档的 bool 场, 改用")
    print("  spiral_v14_prepare.py 自建的种子 (同一条 v>中位数 二值化规则, 与模型侧")
    print("  一致)。理由见本段末尾的诊断项 —— 归档 bool 场与归档浮点场对不上。")
    print("  **为什么两侧用同一条二值化规则是必须的**: 描述子 phi (占空比) 直接")
    print("  参与距离; 若一侧按中位数二值化 (phi 恒等于 0.5) 而另一侧用别的阈值,")
    print("  测到的距离里就混进了『阈值选择』这个与物理无关的自由度。")

    d_mod = _model_field(s6) if 'v' in s6 else None
    if d_mod is None:
        print("  ctx 里没有阶段六的 v 场 -> A 段跳过")
        return {'skipped': True, 'reason': 'no s6.v'}
    b_mod = _binarize_median(d_mod)
    dmod = structural_descriptors(b_mod)
    L_mod = float(s6['h'] * b_mod.shape[0])
    print(f"  模型侧: 盒子 {b_mod.shape[0]}^2 (物理域长 {L_mod:.3f}), "
          f"占空比 {dmod['phi']:.4f}, 特征波长 {dmod['lam_pk']:.4f} 格")

    zc = np.load(files['v14seed'], allow_pickle=True)
    seeds = sorted(int(k.split('_')[1]) for k in zc.files
                   if k.startswith('bool_'))
    d_ref, b_ref = {}, {}
    for s in seeds:
        b = _late_frame(zc[f'bool_{s}'])
        b_ref[s] = b
        d_ref[s] = structural_descriptors(b)
    print(f"  参考侧: {len(seeds)} 个自建种子 {seeds} "
          f"(v>每帧中位数, 与模型侧同规则)")
    for s in seeds[:3]:
        print(f"            seed {s:>2}: 盒子 {b_ref[s].shape[0]}^2, "
              f"占空比 {d_ref[s]['phi']:.4f}, "
              f"特征波长 {d_ref[s]['lam_pk']:.4f} 格")
    print(f"            … 其余 {max(len(seeds) - 3, 0)} 个同规则, 逐项值见下")

    # 种子间散布 = 噪声底 (A 段全部判据的分母)
    pair_d = [descriptor_distance(d_ref[seeds[i]], d_ref[seeds[j]])[0]
              for i in range(len(seeds)) for j in range(i + 1, len(seeds))]
    scatter = float(np.mean(pair_d)) if pair_d else float('nan')

    d_clip_med = {k: float(np.median([d_ref[s][k] for s in seeds])) for k in _KEYS}
    d_a, per_a = descriptor_distance(dmod, d_clip_med)

    xr, _yr = _norm_pk_curve(b_ref[seeds[0]])
    yr = np.mean(np.stack([_norm_pk_curve(b_ref[s])[1] for s in seeds]), axis=0)
    xm, ym = _norm_pk_curve(b_mod)
    r_p, _gx, _ga, _gb = _shape_residual(xr, yr, xm, ym, n_grid=24)

    print("\n  描述子 (模型 vs 参考中位数):")
    for k in _KEYS:
        print(f"    {k:>8}: 模型 {dmod[k]:>11.4f}   参考 {d_clip_med[k]:>11.4f}   "
              f"逐项距离 {per_a[k]:.4f}")
    print(f"  合成描述子距离 D_struct = {d_a:.4f}")
    print(f"  参考侧种子间散布 (噪声底) = {scatter:.4f}  ({len(pair_d)} 个种子对)")
    print(f"  归一化谱形残差 R_P = {r_p:.4f}")

    # 【已发布基线的并列报告 (v14·F3(d))】
    # 已发布基线的 A1 报的是 D_struct=0.9331 / scatter=0.4544 / 比值=2.05。那个
    # 口径的参考侧是归档里直接搬来的 bool 场。换成自建参考侧后必须把旧值一并列出,
    # 否则读者会以为数字自己变了。参考侧的更换理由见本段末尾的诊断项。
    lc = np.load(files['clip'], allow_pickle=True)
    _ls = [s for s in clip_seeds if f'bool_{s}' in lc.files] or sorted(
        int(k.split('_')[1]) for k in lc.files if k.startswith('bool_'))
    _ldref = {s: structural_descriptors(_late_frame(lc[f'bool_{s}']))
              for s in _ls}
    _lpair = [descriptor_distance(_ldref[_ls[i]], _ldref[_ls[j]])[0]
              for i in range(len(_ls)) for j in range(i + 1, len(_ls))]
    _lsc = float(np.mean(_lpair)) if _lpair else float('nan')
    _lmed = {k: float(np.median([_ldref[s][k] for s in _ls])) for k in _KEYS}
    _lda, _lper = descriptor_distance(dmod, _lmed)
    _lratio = float(_lda / _lsc) if _lsc > 0 else float('nan')
    print("\n  【已发布基线 (参考侧 = 归档 bool 场直接搬)】")
    print(f"    D_struct = {_lda:.4f}   种子散布 = {_lsc:.4f} ({len(_lpair)} 对)   "
          f"比值 = {_lratio:.3f}")
    print(f"    复核 spiral_metric_v16.V13_A_BASELINE: "
          f"{V13_A_BASELINE['D_struct']:.4f}/{V13_A_BASELINE['scatter']:.4f}/"
          f"{V13_A_BASELINE['ratio']:.3f} -> "
          f"{'一致' if abs(_lda - V13_A_BASELINE['D_struct']) < 5e-3 else '**不一致**'}")
    ratio_a = float(d_a / scatter) if scatter > 0 else float('nan')
    print("\n  【口径变化的量化】:")
    for _k, _a, _b in (('D_struct', _lda, d_a), ('scatter', _lsc, scatter),
                       ('ratio', _lratio, ratio_a)):
        _ch = abs(_b - _a) / abs(_a) if _a else float('nan')
        print(f"    {_k:<10} {_a:>8.4f} -> {_b:>8.4f}   变化 {_ch:>8.1%}"
              f"{'  **超过 20%**' if _ch > 0.20 else ''}")
    print("    逐项距离 (模型 vs 参考中位数):")
    for _k in _KEYS:
        print(f"      {_k:>8}: {_lper[_k]:>7.4f} -> {per_a[_k]:>7.4f}")

    # 【混杂因子】
    # (1) 参数相区不同 (仍然成立)。模型阶段六用 F=0.035, k=0.060 (Pearson 斑点相);
    #     参考侧 (归档与自建种子都一样) 用 f=0.01, k=0.042 (蠕虫/自复制相)。
    #     这不是同一个动力学区, 所以"同一 PDE"只在方程形式意义上成立。
    # (2) 盒子内波长数 —— **v14 撤回这一条**。v13 报的"两者差一个量级"用的是
    #     归档 bool 场的 lam_pk (1.45 格, 恰是二维离散谱角点), 那是个伪像。
    #     换成自建种子后两侧的 n_lam 几乎相同。但也不能反过来说"盒子大小无影响":
    #     n_lam 就是 kpk_k1 = 盒子边长/lam_pk 的换写, 而 lam_pk 取自离散壳,
    #     所以它不是独立测量, 只能说明"两个盒子相对而言装了差不多多个波长"。
    n_lam_mod = b_mod.shape[0] / max(dmod['lam_pk'], 1e-12)
    ref_lam = float(np.median([d_ref[s]['lam_pk'] for s in seeds]))
    ref_box = b_ref[seeds[0]].shape[0]
    n_lam_ref = ref_box / max(ref_lam, 1e-12)
    mp = model_params or {}
    par_mod = (f"F={mp['F']}, k={mp['k']}" if 'F' in mp and 'k' in mp else '未知')
    try:
        cpar = json.loads(str(zc['meta'])).get('params', {})
        par_ref = f"f={cpar.get('f')}, k={cpar.get('k')}"
    except Exception:
        par_ref = '未知'
    print(f"\n  【混杂因子】参数相区: 模型 {par_mod}  vs  参考 {par_ref}  (仍然成立)")
    print(f"              盒子内波长数: 模型 {n_lam_mod:.2f}  vs  参考 {n_lam_ref:.2f}"
          f"  (v14 撤回『差一个量级』: 旧值来自坏参考场)")

    ratio_a = float(d_a / scatter) if scatter > 0 else float('nan')
    print(f"  D_struct / 种子散布 = {ratio_a:.2f}  "
          f"(<1 不可区分, ~1-2 弱可区分, >3 可区分)")

    record_metric(
        3, 'III-A1 描述子距离', 'D_struct = RMS_i |a_i-b_i| / sqrt(a_i^2+b_i^2)',
        '阶段六 GS 斑图与 CLIP GS 斑图的 6 个无量纲结构描述子之间的距离, '
        '以参考侧自身的种子间散布为噪声底。**这是方法学展示, 不设达标判定**: '
        '两个混杂因子 (相区参数不同、盒子尺寸与网格不同) 使这个距离'
        '无法单独归因给"盒子大小", 因此按物理目标判达标/不达标是没有信息量的。',
        f'structural_descriptors() 在二值化 (两侧同为 v>中位数) 的末帧上; '
        f'参考侧 {len(seeds)} 个 v14 自建种子取中位数',
        f'Gray-Scott 参考家族: 归档 bool 场 (3 个, 已废) 与自建'
        f' {len(seeds)} 个种子并列报告',
        {'D_struct': d_a, 'scatter': scatter, 'ratio_over_scatter': ratio_a,
         'per_key': per_a, 'n_lambda_model': n_lam_mod,
         'n_lambda_ref': n_lam_ref, 'params_model': par_mod,
         'params_ref': par_ref,
         'v13_published': {'D_struct': _lda, 'scatter': _lsc, 'ratio': _lratio},
         'n_ref_seeds': len(seeds)},
        None, None, None,
        note=(f'D_struct = {d_a:.4f}, 参考侧种子间散布 = {scatter:.4f}, '
              f'比值 {ratio_a:.2f}。**读法**: 比值 >1 说明模型斑图在描述子意义上'
              f'可被区分于参考家族, 但只到 {ratio_a:.2f} 倍, 属于弱可区分 —— '
              f'不足以声称"是同一族", 也不足以声称"是不同的斑图类型"。'
              f'**F3 的口径变更 (必读)**: 已发布基线值 '
              f'D_struct={V13_A_BASELINE["D_struct"]:.4f} / '
              f'scatter={V13_A_BASELINE["scatter"]:.4f} / '
              f'比值={V13_A_BASELINE["ratio"]:.3f}。本版换参考侧后为 '
              f'{d_a:.4f} / {scatter:.4f} / {ratio_a:.3f}。'
              f'**参考侧为什么换**: 旧参考侧是从归档搬来的 bool 场, 实测'
              f'与同一归档的浮点场任何简单二值化都对不上 (一致率≈0.5), '
              f'其生成管线不在归档中; 且三个"种子"实为同一个硬编码 IC。'
              f'所以那个 scatter 不是种子间散布, 而是一条不可复现管线的抖动。'
              f'**边界**: 换掉参考侧后 D_struct 降了 '
              f'{abs(d_a-_lda)/_lda:.0%}, 这不等于"模型变好了"—— 它说明旧值里'
              f'有相当一部分距离来自坏参考场, 而不是来自模型与参考的物理差异。'
              f'已发布基线把这一类比较按"达标/不达标"裁决, 得出的 "0/4 达标" 正是'
              f'在这个区间里做的无信息量判决。本版按用户要求把第三层正式'
              f'降级为方法学展示: 报数字、报混杂因子、不设物理达标线。'))
    out['A1'] = {'D_struct': d_a, 'scatter': scatter,
                 'ratio_over_scatter': ratio_a, 'per_key': per_a,
                 'n_lambda_model': n_lam_mod, 'n_lambda_ref': n_lam_ref,
                 'params_model': par_mod, 'params_ref': par_ref,
                 'passed': None}

    record_metric(
        3, 'III-A2 归一化谱形残差', 'R_P = RMS(log P_a - log P_b), 各自归一到积分 1',
        '两条功率谱在 k/k_pk 坐标下的对数形状残差, 振幅已归一化掉, 只剩形状。'
        '横轴以各自谱峰波长为单位, 所以与盒子大小无关 —— 这是 A 段里受'
        '混杂因子 (2) 影响最小的一项。',
        '_band_power(...,dim=2) -> Delta^2(k) -> _shape_residual(n_grid=24)',
        'CLIP Gray-Scott 基准 (同方程形式, 不同相区参数与盒子)',
        r_p, None, None, None,
        note=(f'R_P = {r_p:.4f}。这是对已发布基线那条结果 R_P = 1.277 的直接追问: '
              f'那个数当时是拿 GS 去比 3D CDM 晕谱, 既可能反映"GS 是锁模环"'
              f'这个真实性质, 也可能只是"2D 斑图 vs 3D 密度场"维度错配的产物。'
              f'本版换成同方程形式的二维参考重做, 谱形残差仍然不小, 说明'
              f'它不是纯维度错配造成的; 但因相区参数也不同, 仍不能归因给盒子。'
              f'与 A1 一样, 只作展示, 不设达标线。'))
    out['A2'] = {'R_P': r_p, 'passed': None}

    # 【F3 验收判据 (a)(c): 参考侧换代 + 两侧同规则】
    # 旧口径那条 `0.02 < 占空比 < 0.98` 的守卫在本口径下**恒真**: 两侧都走
    # "a > median(a)", 按定义恰好一半为真, 占空比恒等于 0.5。一条恒真的守卫
    # 没有信息量, 所以换成两条真判据:
    #   (1) 参考侧必须来自 v14 自建缓存 (不是归档搬来的 bool 场) —— 这是 F3 的
    #       核心改动, 必须能在守卫里体现;
    #   (2) 每个参考种子必须在**连续** u 场上健康。判据不能落在二值场上, 因为
    #       二值化把任何场都劈成 50/50 —— 死场也会"通过"。连续 u 场实测:
    #       死场 std≈0.04, 活斑图 std≈0.29, 所以门槛 0.15 有物理间隔。
    _ref_from = 'v14 自建种子 (spiral_v14_prepare.py)'
    record_guard('A-参考侧来源是 v14 自建种子', 'tier3_structural_ladder',
                 f"缓存 {os.path.basename(files['v14seed'])}, {len(seeds)} 个种子 "
                 f"{seeds}, 键前缀全部是 bool_/u_/v_",
                 bool(seeds and all(f'u_{s}' in zc.files and f'v_{s}' in zc.files
                                    for s in seeds)),
                 note='**F3(a) 的验收判据**。旧口径的参考侧是 spiral_v13_prepare.py '
                      '从归档直接搬来的 bool 场, 而归档那条生成管线经实测'
                      '(见下诊断项 A-归档 bool 场与归档浮点场一致) 无法复现。本版改成自建, '
                      '每个种子同时存二值场 (bool_)、连续 u 场尾窗 (u_)、'
                      '末帧浮点 v 场 (v_), 后两者是健康判据与交叉核对的依据。')
    _ustd = {}
    for s in seeds:
        if f'u_{s}' in zc.files:
            _ustd[s] = float(np.asarray(zc[f'u_{s}'])[-1].std())
    _ustd_min = min(_ustd.values()) if _ustd else float('nan')
    _ustd_str = ' / '.join(f"{s}: {_ustd[s]:.4f}" for s in sorted(_ustd))
    record_guard('A-参考种子在连续 u 场上健康', 'tier3_structural_ladder',
                 f"末帧连续 u 场 std 最小值 = {_ustd_min:.4f} >= 0.15  "
                 f"(逐个: {_ustd_str or '未测量'})",
                 bool(_ustd and _ustd_min >= 0.15),
                 note='**F3(a) 的验收判据, 也是本版的关键方法差别**。'
                      '旧口径用二值场占空比当健康判据, 但 "a > median(a)" 恒给出 '
                      '恰好 50% —— 一个完全没有结构的死场也会得满分 0.5, '
                      '那是**假阳性**。所以判据必须落在二值化之前的连续场上。'
                      '实测: 未成核的死场 u 场 std≈0.04 (均值≈0.95, 场近乎均匀), '
                      '活斑图 std≈0.29。门槛 0.15 落在两者之间, 有约 7 倍间隔。')

    # =========================================== v15·W2 同单位受控参考侧 (A3)
    # 【这一节回答什么】上面的 A1/A2 拿模型侧去比归档 CLIP 家族, 而那个参考侧与
    # 模型侧处在**两套单位制与两个相区**:
    #     模型侧  h = L_domain/N     = 0.5/36      = 0.013889 每格 (F,k=0.035,0.060)
    #     参考侧  h = PHY_SIZE/MESH  = 447.21/128  = 3.4939   每格 (f,k=0.01,0.042)
    # 每格物理长度差 252 倍, 而核心读数 lam_pk 的单位是"格" —— 所以 A1 那条
    # D_struct 无法单独归因给物理。W2 按锁定决策走"方案 B·同单位受控": 参考侧改用
    # 与模型侧**逐字相同**的 (Du,Dv,F,k,L_domain), 只变分辨率 N=36/72/144, dt 随
    # h^2 同比缩以保持 stab 相同。两侧因此是**同一物理**。
    # **于是这一节不能用来回答"模型像不像外部数据"** —— 那需要独立数据源。它回答
    # 的是另外两件真能回答的事:
    #   (i)  模型侧那个读数是否落在参考侧自身的初值散布内 (是否只是一次普通抽样);
    #   (ii) 分辨率变化带来的距离是否超出该散布 (读数是否收敛)。
    v15file = os.path.join(data_dir, V15_CACHE_DIRNAME, 'gs_v15seed.npz')
    _v15bad, zv15, _c15, N15 = True, None, {}, []
    if os.path.isfile(v15file):
        zv15 = np.load(v15file, allow_pickle=True)
        try:
            _c15 = json.loads(str(zv15['meta'])).get('contract', {})
        except Exception:
            _c15 = {}
        N15 = [N for N in _c15.get('N_list', ()) if f'bool_{N}_0' in zv15.files]
        _v15bad = not N15

    if _v15bad:
        print(f"\n  【v15·W2 同单位受控】缓存缺失或不可用 -> 本小节跳过 "
              f"(不影响上面的 A1/A2)")
        print(f"      生成: python v15/spiral_v15_prepare.py  ->  {v15file}")
        record_guard('A-同单位受控参考侧可用', 'tier3_structural_ladder',
                     f"{V15_CACHE_DIRNAME}/gs_v15seed.npz 是否存在且含 bool_ 场",
                     False, expect_pass=False,
                     note='**W2 的离线产物缺失**。本项不是"不达标", 而是**不可计算**; '
                          '按 expect_pass=False 登记为诊断项, 不计入通过率分母 —— '
                          '把"没跑"记成"没通过"是两件不同的事实, 不能混。')
        out['A3'] = {'skipped': True, 'reason': 'no v15 cache'}
    else:
        _seeds15 = sorted(int(k.rsplit('_', 1)[1]) for k in zv15.files
                          if k.startswith(f'bool_{N15[0]}_'))
        print(f"\n  【v15·W2 同单位受控】参考侧 = 与模型侧逐字同参数的受控重算")
        print(f"      L={_c15.get('L_domain')}, Du={_c15.get('Du')}, "
              f"Dv={_c15.get('Dv')}, F={_c15.get('F')}, k={_c15.get('k')}, "
              f"T_phys={_c15.get('T_phys')}")
        print(f"      二值化 = {_c15.get('binarize')}; "
              f"积分器 = {_c15.get('integrator')}")
        print(f"      分辨率 N ∈ {N15} (模型侧本轮实际派生 N = {b_mod.shape[0]}), "
              f"种子 {_seeds15}")
        print(f"      **两侧是同一物理** -> 本节检验分辨率收敛与混杂量化, "
              f"**不是**与外部数据对标。")

        # (a) 硬核对: 与模型侧同分辨率的 seed 0 复刻了模型侧那个**确定性**初值,
        #     所以它必须与模型侧本轮的 s6['v'] 逐点一致, 差应当**严格为 0**。
        #     开销为零: 模型运行本身就产出 s6['v'], 不需要重跑任何东西。
        _Nmod = b_mod.shape[0]
        _iddiff = float('nan')
        if f'v_{_Nmod}_0' in zv15.files and 'v' in s6:
            _iddiff = float(np.max(np.abs(
                np.asarray(zv15[f'v_{_Nmod}_0'], dtype=float)
                - np.asarray(s6['v'], dtype=float))))
        record_guard(
            'A-同单位受控: 参考侧复刻了模型侧的积分器与初值',
            'tier3_structural_ladder',
            f"v_{_Nmod}_0 vs 模型侧 s6['v'] 最大绝对差 = {_iddiff:.3e} (要求严格 0)",
            bool(_iddiff == 0.0),
            note='**W2 的硬核对, 也是整个受控比较的前提**。参考侧用显式欧拉 + '
                 '5 点 np.roll 拉普拉斯、周期边界, 参数与初值都与 stage6_life 逐字'
                 '相同, 所以同分辨率下它与模型侧必须是**同一个浮点序列**, 差严格'
                 '为 0 —— 不是"小于某容差"。若本项不为 0, 说明"同方程"这个前提'
                 '没做到, 后面所有受控比较都不成立。开销为零。')

        # (b) 描述子与统计。距离**逐种子算**, 不取中位数 —— 取中位数时若种子里
        #     含与模型侧同初值的那个, 逐描述子中位数会正好落在模型值上, 距离
        #     恒为 0.0000, 那是循环论证。
        _r15 = {}
        for N in N15:
            for s in sorted(int(k.rsplit('_', 1)[1]) for k in zv15.files
                            if k.startswith(f'bool_{N}_')):
                _r15[(N, s)] = structural_descriptors(
                    _late_frame(zv15[f'bool_{N}_{s}']))
        _byN = {N: sorted(s for (n, s) in _r15 if n == N) for N in N15}
        print(f"\n      模型侧 vs 受控参考 (逐种子):")
        _sc15, _mdis, _dd15 = {}, {}, {}
        for N in N15:
            _ss = _byN[N]
            _pd = [descriptor_distance(_r15[(N, i)], _r15[(N, j)])[0]
                   for _i, i in enumerate(_ss) for j in _ss[_i + 1:]]
            _sc15[N] = float(np.mean(_pd)) if _pd else float('nan')
            _dd15[N] = {s: descriptor_distance(dmod, _r15[(N, s)])[0] for s in _ss}
            # seed 0 就是模型侧那个确定性初值 (见上面 (a) 的硬核对), 它与模型侧
            # 的距离恒为 0。算进均值会把结果朝 0 拉 —— 那是循环, 必须排除。
            _rnd = [_dd15[N][s] for s in _ss if s != 0]
            _mdis[N] = float(np.mean(_rnd)) if _rnd else float('nan')
            print(f"        N={N:3d}: 逐种子 {[round(_dd15[N][s], 4) for s in _ss]}"
                  f"  非同一初值均值 {_mdis[N]:.4f} | 同 N 种子散布 "
                  f"{_sc15[N]:.4f} | 比值 {_mdis[N] / _sc15[N]:.2f}")

        # 下面两处也一律排除 seed 0: 它是模型侧自己那个场, 留在中位数里会让
        # N=36 这一列朝模型侧塌陷 (实测未排除时 w50 的"模型/参考"恰好印成
        # 1.000 —— 那不是巧合, 就是循环)。
        _byN_ns = {N: [s for s in _byN[N] if s != 0] for N in N15}
        _med15 = {N: {k: float(np.median([_r15[(N, s)][k] for s in _byN_ns[N]]))
                      for k in _KEYS} for N in N15}
        _cross = {}
        print(f"\n      跨 N 距离 (同一物理, 只变分辨率; 各 N 的中位数之间, "
              f"排除 seed 0):")
        for _i, _a in enumerate(N15):
            for _b in N15[_i + 1:]:
                _d, _ = descriptor_distance(_med15[_a], _med15[_b])
                _cross[(_a, _b)] = _d
                print(f"        N={_a:3d} vs N={_b:3d}: D = {_d:.4f}")

        _w50 = {N: float(np.median([_r15[(N, s)]['w50'] for s in _byN_ns[N]]))
                for N in N15}
        print(f"\n      w50 的分辨率依赖 (中位数, 排除 seed 0; 及与模型侧之比):")
        for N in N15:
            print(f"        N={N:3d}: {_w50[N]:.5f}   模型/参考 = "
                  f"{dmod['w50'] / _w50[N]:.3f}")

        # (c) 描述子集的名义维度被高估
        _spread = {k: float(max(d[k] for d in _r15.values())
                            - min(d[k] for d in _r15.values())) for k in _KEYS}
        _const = [k for k in _KEYS if _spread[k] < 1e-9]
        record_guard(
            'A-同单位受控: 6 个描述子里只有 4 个在判别',
            'tier3_structural_ladder',
            f"跨 {len(_r15)} 个受控参考场取值极差为 0 的描述子 = {_const} "
            f"({len(_KEYS) - len(_const)}/{len(_KEYS)} 维有效)",
            bool(not _const), expect_pass=False,
            note='**诊断项 (不计入分母; 本项设计上就该不通过)**。受控口径下有两个'
                 '描述子**恒为常数**, 对任何距离贡献恰好 0: '
                 '(1) phi 恒 0.5 —— "a > median(a)" 的必然结果 (上方已有专条); '
                 '(2) kpk_k1 恒等于 盒子/lam_pk; 两侧盒子与**物理波长**都相同, '
                 '于是它等于 L/lam_phys = 盒内波长数, **本来就与分辨率无关**。'
                 f'实测 {len(_r15)} 个场 (3 个分辨率 x 3 个种子) 的 kpk_k1 全部是 '
                 f'{_r15[(N15[0], _byN[N15[0]][0])]["kpk_k1"]:.4f}。'
                 '**后果**: 有效判别维度只有 4 维 (v1_lam / v2_lam2 / w50 / xi_lam), '
                 '而 structural_descriptors 的 docstring 声称给 6 个 —— 名义维度'
                 '被高估了三分之一, 读 D_struct 时应按 4 维理解。这不是 bug '
                 '(那两个量本来就该是常数), 但把它当成"6 维结构比较"是夸大。')

        # (d) w50 的分辨率依赖 —— 描述子集的真实缺陷
        _w50_mono = all(_w50[N15[i]] > _w50[N15[i + 1]]
                        for i in range(len(N15) - 1))
        # 旧口径 (A1) 的参考侧 w50, 与模型侧之比 —— 用 A1 实际报的那个参考侧算,
        # 不引用归档 CLIP 的数字 (那与本段当前口径不是同一个参考侧)。
        _w50_refA1 = float(np.median([d_ref[s]['w50'] for s in seeds]))
        record_guard(
            'A-同单位受控: w50 不是分辨率无关的',
            'tier3_structural_ladder',
            "w50 中位数 " + " -> ".join(f"N={N}: {_w50[N]:.5f}" for N in N15)
            + f" (排除 seed 0); 随 N 单调下降 = {_w50_mono}",
            bool(not _w50_mono), expect_pass=False,
            note='**诊断项 (不计入分母; 本项设计上就该不通过)** —— 这是 W2 量到的'
                 '**描述子集的真实缺陷**, 它回头解释了旧口径里的一部分距离。'
                 'w50 = "承载 50% 功率的模数 / 总非零模数"。一个**固定物理**的斑图'
                 '占住固定数量的模, 而总模数 ∝ N^2, 所以 w50 应当 ∝ 1/N^2 —— 它虽然'
                 '无量纲, 却**不是分辨率无关的**, 与 docstring 里"一律除以特征波长'
                 '或除以总模数, 剩下的才是形状"的说法不符。'
                 f'实测每级约 ÷{_w50[N15[0]] / _w50[N15[1]]:.2f} 与 '
                 f'÷{_w50[N15[1]] / _w50[N15[2]]:.2f} (N 每级 x2, 1/N^2 应给 x4)。'
                 '**指数定不准, 不要引它**: 排除同一个初值后每个 N 只剩 2 个种子, '
                 '中位数就是两个数的平均, 3.2 与 4 的差别在这点样本下没有意义。'
                 '站得住的只有**单调下降**这一条 (守卫条件查的就是它)。'
                 f'**对旧口径的后果**: A1 报的参考侧 w50 = {_w50_refA1:.5f}, '
                 f'模型侧 = {dmod["w50"]:.5f}, 之比 {dmod["w50"] / _w50_refA1:.2f} —— '
                 '其中混着**纯网格分辨率**造成的成分 (旧参考侧盒子 100 格, 模型侧 36 '
                 '格), 与物理无关。受控后那一项被消掉 (同 N 下比值回到 1 附近), 但'
                 '读者仍不该把 w50 当作"形状"量引用。')

        # (e) 旧口径的距离主要来自混杂 —— 一条关于已发布基线的负面事实
        _ctrl_mean = float(np.mean([_mdis[N] for N in N15]))
        _scat_mean = float(np.mean([_sc15[N] for N in N15]))
        record_guard(
            'A-同单位受控: 旧口径的距离主要来自混杂',
            'tier3_structural_ladder',
            f"旧口径 (参考侧 = 归档 CLIP bool 场) D_struct = {_lda:.4f}; "
            f"受控口径 (vs 同物理参考) 非同一初值均值 = {_ctrl_mean:.4f}; "
            f"受控侧种子散布均值 = {_scat_mean:.4f}",
            bool(_ctrl_mean >= _lda), expect_pass=False,
            note='**诊断项 (不计入分母; 本项设计上就该不通过)** —— 本项登记的是一条'
                 '关于**已发布基线**的负面事实, 不是关于模型的。'
                 f'对比的三方: 已发布基线 D_struct = {_lda:.4f} (参考侧是归档 CLIP '
                 f'bool 场, 即 A1 上方"口径变化"表里那一列); A1 现口径 = {d_a:.4f} '
                 f'(参考侧换成 v14 自建种子); 受控口径 = {_ctrl_mean:.4f}。'
                 '把参数相区、单位制、二值化协议、网格分辨率四者控制住之后, 模型侧'
                 f'与"和它同物理的参考侧"之间的距离从 {_lda:.4f} 降到 '
                 f'{_ctrl_mean:.4f}, 并且**落在参考侧自身的种子散布 '
                 f'({_scat_mean:.4f}) 之内** —— 即模型侧那个场只是同一分布里的'
                 '一次普通抽样, 没有超出的结构差异。**结论方向必须说准**: '
                 '这不是"模型变好了", 而是**旧口径那条距离里的大部分来自混杂因子** '
                 '(相区参数不同、CLIP 侧 phi 0.64~0.92 而非 0.5、盒子与网格不同), '
                 '不是来自模型与参考的物理差异。旧口径把它按"达标/不达标"裁决得出的'
                 ' "0/4 达标", 正建立在这条不可归因的距离上。')

        print(f"\n      -> 受控后模型侧落在自身初值散布内; 旧口径 {_lda:.4f} 的"
              f"大部分距离来自混杂, 不是物理差异。")
        out['A3'] = {
            'skipped': False, 'bit_identity_maxdiff': _iddiff,
            'dist_model_vs_ref': {int(N): _mdis[N] for N in N15},
            'scatter_within_N': {int(N): _sc15[N] for N in N15},
            'ratio_over_scatter': {int(N): _mdis[N] / _sc15[N] for N in N15},
            'cross_N_distance': {f'{a}-{b}': v for (a, b), v in _cross.items()},
            'w50_median': {int(N): _w50[N] for N in N15},
            'constant_descriptors': _const, 'old_D_struct': _lda,
            'contract': {k: _c15.get(k) for k in
                         ('L_domain', 'Du', 'Dv', 'F', 'k', 'T_phys', 'N_ref',
                          'integrator', 'binarize')}}
        record_metric(
            3, 'III-A3 同单位受控距离',
            'D_struct(模型侧 vs 同物理受控参考), 逐种子',
            '把参考侧换成与模型侧**逐字同参数同单位**的受控重算 '
            '(v15/spiral_v15_prepare.py, N=36/72/144, dt 随 h^2 同比缩), 于是两侧是'
            '同一物理。衡量两件事: (i) 模型侧读数是否落在参考侧自身的初值散布内; '
            '(ii) 分辨率变化带来的距离是否超出该散布。**这是 A 段唯一可归因的比较**。',
            'data/_v15_cache/gs_v15seed.npz; descriptor_distance 逐种子算, '
            '并排除与模型侧同初值的 seed 0 (否则循环)',
            '与模型侧逐字同参数的 Gray-Scott 受控重算 —— 不是独立数据源',
            out['A3'], None, None, None,
            note=('模型侧 vs 受控参考 (非同一初值) 的距离/散布比值: '
                  + ', '.join(f'N={N}: {_mdis[N] / _sc15[N]:.2f}' for N in N15)
                  + ' —— 全部 <1, 即落在噪声底内。'
                  '**边界 (必读)**: 受控参考侧与模型侧是**同一物理**, 所以它'
                  '**不能**回答"模型像不像外部实验数据"; 那个问题需要独立数据源, '
                  '而 A 段现有的两个参考源 (归档 bool 场、v14 自建种子) 都带着'
                  '相区与单位混杂。本项只支撑一句话: 在控制住混杂之后, 模型侧的'
                  '结构描述子读数落在其自身初值散布之内。'))

    # ---------------------------------------------------------------- A 段诊断
    # 【v14·F3 实测到的参考侧缺陷 —— 这一节改写了已发布基线的判读】
    # 已发布基线的诊断说"参考侧种子 5/10 是近乎均匀的退化场"。实现 F3 时把
    # 归档翻到底, 发现那个判读本身建立在一条坏管线上:
    #   (甲) 归档的三个**浮点** u 场都健康且几乎相同 (mean 0.6417/0.6415/0.6412,
    #        std 0.2930/0.2935/0.2950) —— 因为 data_make.py 的 IC 是硬编码的,
    #        np.random.seed 是死代码, 三个"种子"其实同出一个 IC。
    #   (乙) 旧口径实际用的归档 **bool** 场占空比 0.6378/0.9181/0.9125, 与归档浮点场
    #        的任何简单二值化都对不上 (一致率见下), 其 100^2/300 帧管线不在归档里。
    #   (丙) 于是 scatter 根本不是"种子间散布", 它测的是一条来路不明的管线的抖动;
    #        而 lam_pk 中位数 1.45 格正是那种场里 1 格台阶造成的谱角点伪像。
    # 本版的处置: 参考侧换成自建种子 (spiral_v14_prepare.py), 两侧同一条
    # v>中位数 规则; 旧数字作为已发布基线并列报出 (见上); 本节的诊断项如实记录
    # 归档那条管线不可复现这个缺口, 不假装它被修好了。
    fld = {'模型': b_mod}
    for s in seeds:
        fld[f'参考 seed {s}'] = b_ref[s]
    fld_info = {}
    for nm, bb in fld.items():
        ph = float(np.mean(bb))
        fld_info[nm] = {'phi': ph, 'minority': min(ph, 1.0 - ph),
                        'xi_ac': _ac_len(bb), 'box': int(bb.shape[0]),
                        'img': _ds(bb).tolist()}
    ref_minor = [fld_info[f'参考 seed {s}']['minority'] for s in seeds]
    print("\n  【A 段诊断 · 两侧场与 lam_pk 的可信度】")
    print(f"    模型侧 盒子 {b_mod.shape[0]}  占空比 {dmod['phi']:.4f}  "
          f"自相关长度 {fld_info['模型']['xi_ac']:.1f} 格")
    print(f"    参考侧 {len(seeds)} 个种子: 占空比全是 0.5000 "
          f"(中位数二值化的恒等式), 自相关长度 "
          f"{min(fld_info[f'参考 seed {s}']['xi_ac'] for s in seeds):.0f}~"
          f"{max(fld_info[f'参考 seed {s}']['xi_ac'] for s in seeds):.0f} 格")

    # ---- lam_pk 的可信度交叉核对 -------------------------------------------
    # _char_scale 取 Δ²(k) 的 argmax。argmax 不要求峰有内部性, 所以坏场会让它
    # 落在网格能表示的**最短**波长 (二维离散谱角点 |n|=(N/2,N/2), λ=√2 格) 上。
    # 这个担心早被记录过。实测: 在**自建种子**上它没犯这个错, 而且与两条
    # 独立尺度估计吻合; 但归档 bool 场上的 λ=1.45 格确实就是角点。
    _lam_corner = float(ref_box / np.hypot(ref_box / 2, ref_box / 2))
    _spec, _ac4 = {}, {}
    for s in seeds:
        if f'v_{s}' not in zc.files:      # 只有 bool_ 时无法做这项交叉核对
            _spec[s], _ac4[s] = float('nan'), float('nan')
            continue
        vf = np.asarray(zc[f'v_{s}'], dtype=np.float64)
        _z = vf - vf.mean()
        _pk = _band_power(_z, float(vf.shape[0]), 2, deconv=False)
        _d2 = _delta2(_pk['k'], _pk['P'], 2)
        _spec[s] = float(2 * np.pi / _pk['k'][int(np.argmax(_d2))])
        _ac4[s] = 4.0 * fld_info[f'参考 seed {s}']['xi_ac']
    _pk_dev = np.array([abs(d_ref[s]['lam_pk'] - _spec[s]) / _spec[s]
                        for s in seeds])
    _ac_dev = np.array([abs(d_ref[s]['lam_pk'] - _ac4[s]) / _ac4[s]
                        for s in seeds])
    _pk_m = float(np.nanmedian(_pk_dev)) if np.isfinite(_pk_dev).any() else float('nan')
    _ac_m = float(np.nanmedian(_ac_dev)) if np.isfinite(_ac_dev).any() else float('nan')
    print(f"    lam_pk (二值场的 Δ² 峰) vs 未二值化 v 场的 Δ² 峰: "
          f"中位偏差 {_pk_m:.1%}  (最大 {np.nanmax(_pk_dev):.1%})  <- 同估计量换输入, 弱一致")
    print(f"    lam_pk vs 与谱峰无关的自相关尺度 4*xi_ac: "
          f"中位偏差 {_ac_m:.1%}  (最大 {np.nanmax(_ac_dev):.1%})  "
          f"<- **不一致 (系统性偏大), lam_pk 缺独立佐证**")
    _lam_corner_hit = abs(ref_lam - _lam_corner) / _lam_corner < 0.05
    print(f"    二维谱角点波长 = {_lam_corner:.3f} 格; 本段 lam_pk 中位数 = "
          f"{ref_lam:.4f} 格 -> {'**仍在角点上**' if _lam_corner_hit else '不在角点上'}")
    _lam_leg = float(np.median([_ldref[s]['lam_pk'] for s in _ls]))
    print(f"    对照归档 bool 场: lam_pk 中位数 = {_lam_leg:.4f} 格 -> "
          f"{'**落在角点上 (伪像)**' if abs(_lam_leg - _lam_corner) / _lam_corner < 0.05 else '不在角点上'}")

    # 归档溯源数字 (由 spiral_v14_prepare.archive_provenance() 一次性量出)
    _prov = None
    _provf = os.path.join(v14cache, 'archive_provenance.json')
    if os.path.exists(_provf):
        try:
            with open(_provf, encoding='utf-8') as _fh:
                _prov = json.load(_fh)
        except Exception as _e:
            print(f"    [溯源] 读取 {_provf} 失败: {_e}")
    if _prov and _prov.get('seeds'):
        _leg_agree = [float(r['agree_u_med']) for r in _prov['seeds']]
        _leg_fill = [float(r['bool_fill']) for r in _prov['seeds']]
        _leg_stdc = [float(r['float_u_std_c']) for r in _prov['seeds']]
        _leg_u = [float(r['float_u_mean']) for r in _prov['seeds']]
        _leg_vmed = [float(r['agree_v_med']) for r in _prov['seeds']]
        _leg_s = [int(r['seed']) for r in _prov['seeds']]
    else:
        _leg_agree = [float('nan')]
        _leg_fill = _leg_stdc = _leg_u = _leg_vmed = [float('nan')]
        _leg_s = []
    _leg_str = ' / '.join(
        f"seed {s}: {a:.4f}" for s, a in zip(_leg_s, _leg_agree)) or '未测量'
    _leg_all = ' / '.join(
        f"seed {s}: float_u mean={m:.4f} std={t:.4f}, bool 占空比={fl:.4f}, "
        f"一致率 u>med={a:.4f} v>med={vm:.4f}"
        for s, m, t, fl, a, vm in zip(_leg_s, _leg_u, _leg_stdc, _leg_fill,
                                      _leg_agree, _leg_vmed)) or '未测量 (缺少 archive_provenance.json)'
    print(f"    归档溯源: {_leg_all}")

    # ---- 参考侧连通性 (仍保留: 它描述自建种子的斑图形态) --------------------
    _conn = {}
    for _s in seeds:
        _c = b_ref[_s] if float(np.mean(b_ref[_s])) < 0.5 else ~b_ref[_s]
        _nb = (ndimage.convolve(_c.astype(int), np.ones((3, 3)), mode='wrap')
               - _c.astype(int))
        _lab, _nc = ndimage.label(_c, structure=np.ones((3, 3)))
        _sz = np.bincount(_lab.ravel())[1:]
        _conn[int(_s)] = {'n_comp': int(_nc),
                          'mean_size': float(_sz.mean()) if _sz.size else 0.0,
                          'max_size': int(_sz.max()) if _sz.size else 0,
                          'iso_frac': float(((_nb == 0) & _c).sum()
                                            / max(_c.sum(), 1))}
    _cm = b_mod if float(np.mean(b_mod)) < 0.5 else ~b_mod
    _nbm = (ndimage.convolve(_cm.astype(int), np.ones((3, 3)), mode='wrap')
            - _cm.astype(int))
    _labm, _ncm = ndimage.label(_cm, structure=np.ones((3, 3)))
    _szm = np.bincount(_labm.ravel())[1:]
    print(f"    少数相连通块: 模型 {_ncm} 块 (平均 "
          f"{_szm.mean() if _szm.size else 0:.1f} 格); 参考侧 "
          f"{min(_conn[int(s)]['n_comp'] for s in seeds)}~"
          f"{max(_conn[int(s)]['n_comp'] for s in seeds)} 块, 平均 "
          f"{np.mean([_conn[int(s)]['mean_size'] for s in seeds]):.1f} 格; "
          f"孤立单格占比 最大 "
          f"{max(_conn[int(s)]['iso_frac'] for s in seeds):.1%}")
    _lam_str = '/'.join(f"{d_ref[s]['lam_pk']:.4f}" for s in seeds)
    _conn_str = '  '.join(
        f"seed {int(s)}: {_conn[int(s)]['n_comp']} 块, 平均 "
        f"{_conn[int(s)]['mean_size']:.1f} 格, 孤立单格 "
        f"{_conn[int(s)]['iso_frac']:.1%}" for s in seeds)
    _iso_str = ' / '.join(f"{_conn[int(s)]['iso_frac']:.1%}" for s in seeds)

    # ---- 归档管线不可复现: 如实记录缺口 (诊断项) ---------------------------
    record_guard(
        'A-归档 bool 场与归档浮点场一致', 'tier3_structural_ladder',
        f"归档 bool 场与归档浮点场 (u>中位数) 的逐格一致率: {_leg_str}  "
        f"-> 最高 {max(_leg_agree) if _leg_agree else float('nan'):.4f} < 0.90",
        bool(np.isfinite(_leg_agree).all() and max(_leg_agree) >= 0.90),
        expect_pass=False,
        note='**F3 实测的归档缺口 (诊断项, 不计入分母; 本项设计上就该不通过)**。'
             '旧口径的 A 段参考侧是 spiral_v13_prepare.py 直接从归档搬来的 bool 场 '
             '(源码注释: "源文件本身已是 bool, 直接搬")。本版把这批 bool 场与同一'
             f'归档里对应的**浮点**场逐格比对: {_leg_all} —— 与 (u>中位数) 的一致率'
             '在 0.5 附近 (与随机猜测无异), 与 (v>中位数) 更低。也就是说: '
             '**归档里那批 bool 场不是同目录下浮点场的任何简单二值化**, '
             '生成它们的那条 100^2/300 帧管线不在归档里, 无法复现。'
             '另一条独立事实: 三个浮点场的统计量几乎相同 (见上), 因为归档 '
             'data_make.py 的初始条件是硬编码的, 其中 np.random.seed 与 '
             'random.seed 设了却从未被使用 —— 三个"种子"实际同出一个 IC。'
             '**后果**: 那个 scatter 不是种子间散布, 它测的是一条来路不明的'
             f'管线的抖动; 而 lam_pk 中位数 {_lam_leg:.4f} 格恰是二维离散谱角点 '
             f'({_lam_corner:.4f} 格), 是那种场里 1 格台阶的伪像。'
             '本版的处置: 参考侧换成自建种子, 旧数字作为已发布基线并列报出, '
             '**不声称旧值被"修正"**, 而是明确标注它建立在一个不可复现的缓存上。'
             + ('' if _prov else ' **注意: 未读到 archive_provenance.json, '
                                 '上面的一致率为占位 NaN, 本诊断项未能判定**; '
                                 '跑 python spiral_v14_prepare.py 可补齐。'))
    print(f"    -> 缺口记录完毕; A1 的旧数字保留为已发布基线, 不作修改。")

    # ---- 恒等式暴露: phi 在两侧都是常数 -----------------------------------
    record_guard(
        'A-phi 携带场的结构信息', 'tier3_structural_ladder',
        f"模型侧 phi={dmod['phi']:.6f}, 参考侧 phi 落在 "
        f"[{min(d_ref[s]['phi'] for s in seeds):.6f}, "
        f"{max(d_ref[s]['phi'] for s in seeds):.6f}] —— 全部贴住 0.5",
        bool(abs(dmod['phi'] - 0.5) > 1e-3
             or any(abs(d_ref[s]['phi'] - 0.5) > 1e-3 for s in seeds)),
        expect_pass=False,
        note='**两侧同规则带来的必然结果 (诊断项, 不计入分母; 本项设计上就该不通过)**。'
             '_binarize_median 取 "a > median(a)", 参考侧又是逐帧取中位数, 所以 '
             'phi 恒在 0.5 附近 —— 实测参考侧 12 个种子的 phi 全部是 0.5000 '
             '(其中一个 0.4999, 是偶数格点抽到并列值时的舍入, 不是结构信号), '
             '模型侧 phi 与参考中位数逐项距离恰好 0.0000 (见上节 per_key)。'
             '**所以 phi 在本口径下不携带任何场的结构信息**, 它只是一条恒等式。'
             '旧口径的实际后果更糟: 模型侧走中位数 (phi≡0.5), 参考侧走归档 bool 场 '
             '(phi 0.6378/0.9181/0.9125), 于是一个**恒量**去比一个**变量** —— '
             '逐项距离里 phi 一项就贡献 0.3964 (见上节已发布基线), '
             '而那条 D_struct 是 0.9331, 即**四成以上的"距离"来自这条恒等式**。'
             '换成两侧同规则后 phi 贡献恰好 0, 恒等式因此变得无害。'
             '**诚实边界**: 这不代表 phi 是个好描述子 —— 它在本口径下没有信息量, '
             '只是"无害"; 保留它是为了不静默改变描述子集合的维度(4 个 vs 5 个)。'
             '若将来要给 phi 注入信息, 必须改用与二值化无关的阈值口径, '
             '而那会同时改变两侧, 属于另一次口径变更。')

    # ---- lam_pk 的量化与佐证 (F3(b): 照实报缺口, 不动 _char_scale) --------
    # v15·W9: 把**同一个缺口**在模型侧也量一遍。守卫报的 46.6% 是参考侧的
    # (口径见上, 分母是 4*xi_ac)。模型侧用同一口径算出来是多少, 决定了这个
    # 缺口是"归档数据怪"还是"这类场本来如此" —— 两者对结论的分量完全不同。
    _lam_mod = float(dmod['lam_pk'])
    _ac4_mod = 4.0 * float(fld_info['模型']['xi_ac'])
    _mod_dev = (abs(_lam_mod - _ac4_mod) / _ac4_mod if _ac4_mod > 0
                else float('nan'))

    # 注意本项的名称: 判据是"有独立佐证", 实测**不通过** —— 这是 v14 新量到的
    # 事实, 比"只是量化"更严重, 照实写。
    record_guard(
        'A-char_scale 的 lam_pk 有独立佐证', 'tier3_structural_ladder',
        f"lam_pk vs 未二值化 v 场的 Δ² 峰: 中位偏差 "
        f"{np.nanmedian(_pk_dev):.1%}; vs 自相关尺度 4*xi_ac: 中位偏差 "
        f"{np.nanmedian(_ac_dev):.1%}; 且 {len(seeds)} 个种子只给出 "
        f"{len(set(round(d_ref[s]['lam_pk'], 4) for s in seeds))} 个不同的 lam_pk",
        bool(np.nanmedian(_pk_dev) < 0.05 and np.nanmedian(_ac_dev) < 0.05),
        expect_pass=False,
        note='**F3(b) 的诚实缺口 (诊断项, 不计入分母; 本项设计上就该不通过), '
             '按用户决定不改 _char_scale**。审计曾要求先给 _char_scale 加'
             '"峰必须在谱内部"的判据再动 F3。实测后把这个诊断**改写了**, '
             '因为量到的比原判更不利 —— 分三层说清:'
             f'(1) **角点伪像的说法被否掉一半**: 旧参考侧观测到的 λ={_lam_leg:.4f} 格'
             f'确实落在二维离散谱角点 {_lam_corner:.4f} 格上 (|n|=(N/2,N/2), '
             '网格能表示的**最短**波长), 但那是**归档 bool 场自身**的 1 格台阶'
             f'造成的; 换成自建种子后 lam_pk 中位数变成 {ref_lam:.4f} 格, '
             '离角点很远。所以角点伪像不是 _char_scale 的固有 bug, 它取决于输入场。'
             f'(2) **但 lam_pk 缺少独立佐证 —— 这是新量到的负面事实**: '
             f'与"同一个估计量作用在未二值化的 v 场"相比, 中位偏差 '
             f'{np.nanmedian(_pk_dev):.1%} (最大 {np.nanmax(_pk_dev):.0%}), '
             '属于弱一致; 而与**真正独立**的自相关尺度 4*xi_ac 相比, 中位偏差 '
             f'{np.nanmedian(_ac_dev):.1%} (最大 {np.nanmax(_ac_dev):.0%}) —— '
             '不一致, 且是**系统性**的: 自相关尺度恒比 lam_pk 大 (实测中位数 '
             f'{float(np.median([_ac4[s] for s in seeds])):.1f} 格 vs '
             f'{ref_lam:.1f} 格), 不是随机散布。已发布说明里"两条独立路线互相'
             '印证"那句话在自建参考家族上**不成立**, 此处撤回。'
             f'(3) **而且它是量化的**: _char_scale 取 Δ²(k) 在**离散壳**上的 argmax, '
             f'所以 lam_pk 只能取离散值 —— 本段 {len(seeds)} 个种子只给出 '
             f'{len(set(round(d_ref[s]["lam_pk"], 4) for s in seeds))} 个不同的 '
             f'lam_pk。模型侧与参考侧的 kpk_k1 因此都等于 {n_lam_mod:.4f}, '
             '这不是"两个盒子装了同样多的波长"这条物理结论; 而 n_lam ≡ kpk_k1 '
             '≡ 盒子边长/lam_pk 只是同一个量的换写, 本身也不是独立测量。'
             '**为什么不改**: 修它需要一个有物理依据的取峰规则, 而 v14 试过的'
             '替换方案 (在 λ>=k 的壳里取 P(k) 最大者) 在合成条纹上验证通过 '
             '(7.692 vs 真值 8 / 16.667 vs 真值 16), 但在真实场上退化到盒子尺度的'
             '低频团块 (三个参考种子都给 46.841), 稳定性不达标 —— 拿一个未验证的'
             '估计量换掉一个已知有缺陷的估计量, 风险大于收益。故按用户决定'
             '照实报缺口。'
             '(4) **第三条佐证路径也不通, 但原因查清了 (v15·W9)**: 本来最独立的'
             '候选是 Gray-Scott 的 Turing 线性稳定性给出的 λ_c —— 它与谱峰 argmax、'
             '与自相关尺度都没有共同来源。实测这条路**不存在**: 两组参数的判别式 '
             'F²−4F(F+k)² 都 < 0, 没有非平凡齐次定态, 无 λ_c 可算 (详见下一项诊断)。'
             '同时排掉两个更省事的解释: 缺口**不是估计量造成的** —— 在模型场上 '
             'Δ²-argmax 与 P(k)-argmax、二值场与原始场, 四条谱路给出的波长**逐位'
             '相同** (都是 8.8326 格); 缺口**也不是"谱太宽、没有单一波长"造成的** '
             '—— 模型侧谱峰半高只占 1 个壳, 原始场的谱参与比只有 1.8/25, 是很窄的峰。'
             f'缺口在模型侧**同样复现**: 以 4*xi_ac 为分母, 模型侧 {_mod_dev:.1%} '
             f'vs 参考侧 {np.nanmedian(_ac_dev):.1%}。剩下的候选是"空间排布不规则"'
             '(模型侧逐行段长 CV≈1.1, 接近指数分布, 而规则条纹应≈0), 但**这条没有'
             '直接测**, 只是最省事的推断, 不得当结论引用。'
             '**边界**: 本段的 lam_pk 与 n_lam 只能按"某条离散壳上的'
             '一个读数"读, 既不保证是斑图波长, 也没有独立佐证; 涉及"盒子内波长数"'
             '的结论一律降级为量级陈述, 不能作为定量结论引用。')
    print(f"    -> F3(b) 缺口已如实登记: lam_pk 量化 "
          f"({len(set(round(d_ref[s]['lam_pk'], 4) for s in seeds))} 个不同值 / "
          f"{len(seeds)} 个种子), 且缺独立佐证 "
          f"(vs 自相关中位偏差 {np.nanmedian(_ac_dev):.0%}); 未改 _char_scale。")

    # ------------------------------------------- v15·W9 Turing 路的排除 (诊断)
    # W9 的目标是给 lam_pk 补**第三条**独立佐证: Gray-Scott 的 Turing 线性
    # 稳定性给出的最不稳定波长 λ_c —— 它与谱峰 argmax、与自相关尺度都没有
    # 共同来源, 是三路里最独立的一条。
    # 实测结果: 这条路**不存在**。两组参数都没有非平凡齐次定态, λ_c 无从算起。
    # 登记为诊断项 (不计入分母), 不假装这条佐证存在, 也不因为不通就改判据。
    def _disc_gsc(Fv, kv):
        """非平凡齐次定态存在性判别式 F² − 4F(F+k)²。**与扩散系数无关** ——
        所以归档 meta 缺 Du/Dv 不影响这条结论。"""
        return Fv ** 2 - 4.0 * Fv * (Fv + kv) ** 2

    try:
        _cp = json.loads(str(zc['meta'])).get('params', {})
    except Exception:
        _cp = {}
    _ok_m = ('F' in mp) and ('k' in mp)
    _ok_r = ('f' in _cp) and ('k' in _cp)
    _Fm = float(mp['F']) if _ok_m else float('nan')
    _km = float(mp['k']) if _ok_m else float('nan')
    _Fr = float(_cp['f']) if _ok_r else float('nan')
    _kr = float(_cp['k']) if _ok_r else float('nan')
    _d_m = _disc_gsc(_Fm, _km) if _ok_m else float('nan')
    _d_r = _disc_gsc(_Fr, _kr) if _ok_r else float('nan')
    # 临界 k (判别式恰为 0): 4(F+k)² = F -> k_crit = sqrt(F)/2 − F。
    # 报出来是为了说明两侧**只是刚出区**, 不是"差得远"。
    _kc_m = (float(np.sqrt(_Fm)) / 2.0 - _Fm) if _ok_m else float('nan')
    _kc_r = (float(np.sqrt(_Fr)) / 2.0 - _Fr) if _ok_r else float('nan')
    _tur = bool(np.isfinite(_d_m) and np.isfinite(_d_r)
                and _d_m > 0 and _d_r > 0)
    record_guard(
        'A-lam_pk 的第三条佐证路径存在 (Turing 线性稳定性)',
        'tier3_structural_ladder',
        f"非平凡齐次定态判别式 F²−4F(F+k)²: 模型 {_d_m:+.3e} "
        f"(F={_Fm}, k={_km}, 临界 k={_kc_m:.6f}); 参考 {_d_r:+.3e} "
        f"(f={_Fr}, k={_kr}, 临界 k={_kc_r:.6f}) —— 两侧都 < 0, "
        f"只有平凡定态 (u,v)=(1,0), 无 λ_c 可算",
        _tur,
        expect_pass=False,
        note='**这是缺口的原因之一被查清, 不是一条失败的检验** (诊断项, 不计入'
             '分母; 本项设计上就该不通过)。上游那项诊断 '
             '「A-char_scale 的 lam_pk 有独立佐证」记的是"缺口存在", 本项记的是'
             '"为什么补不上"。分三点:'
             '(1) **为什么这条路本来是对的选法**: Turing 线性稳定性从反应-扩散'
             '方程本身出发, 与谱峰 argmax 用的同一个 Δ²(k)、与自相关首次过零'
             '都没有共同来源 —— 它本是三路里唯一真正独立的。'
             '(2) **为什么它不存在**: 非平凡齐次定态要求 v≠0 支 F u² − F u + '
             '(F+k)² = 0 有实根, 即 F >= 4(F+k)²。两组参数都差一点: 模型侧 '
             '4(F+k)² 超出 F 约 3.1%, 参考侧约 8.2%; 换成临界 k 看, 模型侧 '
             f'k={_km} 比 k_crit={_kc_m:.6f} 高 '
             f'{(_km/_kc_m - 1) if _ok_m else float("nan"):.2%}, 参考侧 '
             f'k={_kr} 比 k_crit={_kc_r:.6f} 高 '
             f'{(_kr/_kc_r - 1) if _ok_r else float("nan"):.2%}。'
             '判别式**与扩散系数无关**, 所以归档 meta 不给 Du/Dv 不影响这条结论; '
             '更新式已逐字核对 spiral_model_v16.stage6_life。'
             '(3) **不要把它读成"所以斑图是宽带的"**: 这个推断被实测否掉了 —— '
             '模型侧谱峰半高只占 1 个离散壳, 原始 v 场的谱参与比只有 1.8/25。'
             '同一项侦察还否掉了"估计量取法造成缺口": 四条谱路 (Δ² / P(k) × '
             '二值 / 原始) 在同一场上给出**逐位相同**的波长。缺口在模型侧同样'
             f'复现 ({_mod_dev:.1%} vs 参考侧 {np.nanmedian(_ac_dev):.1%}), '
             '说明它是这类场的性质, 不是归档数据的怪癖。'
             '**边界**: 本条只否证, 不给出替代佐证 —— lam_pk 至今仍然没有独立'
             '佐证, 上游那项诊断的降级结论不因本条而放宽。')
    print(f"    -> W9: Turing 路已排除 (判别式 模型 {_d_m:+.2e} / 参考 {_d_r:+.2e}, "
          f"两侧都无非平凡齐次定态); lam_pk 仍无独立佐证 —— 缺口原因已登记。")

    # ------------------------------------------------------------ F7 输入校验
    # 只验校验**生效**, 不动取峰规则。五种合成输入各走一遍包装, 看 valid 标位
    # 是否与事实一致 —— 这条若能通过, 说明退化输入不会再静默产出可用数字。
    _probe = [('空场', np.zeros(0)),
              ('全零', np.zeros((36, 36))),
              ('全一', np.ones((36, 36))),
              ('常数场', np.full((36, 36), 3.7)),
              ('周期条纹', (np.sin(np.arange(36)[:, None] * 2 * np.pi / 9.0)
                            * np.ones((1, 36))))]
    _f7 = {}
    for _nm, _arr in _probe:
        try:
            _l7, _p7, _v7 = _char_scale_checked(_arr, 36.0)
            _f7[_nm] = {'lam': float(_l7), 'valid': bool(_v7), 'raised': False}
        except ValueError as _e7:
            _f7[_nm] = {'lam': float('nan'), 'valid': False, 'raised': True,
                        'msg': str(_e7)}
    _f7_txt = ', '.join(f"{k}: {'valid' if v['valid'] else 'invalid'}"
                        f"{' (报错)' if v['raised'] else ''}"
                        for k, v in _f7.items())
    record_guard('F7 _char_scale 输入校验生效', 'structural_descriptors',
                 f"空场报错, 全零/全一/常数场标 invalid, 周期条纹标 valid "
                 f"-> {_f7_txt}",
                 bool(not _f7['空场']['valid'] and not _f7['全零']['valid']
                      and not _f7['全一']['valid']
                      and not _f7['常数场']['valid']
                      and _f7['周期条纹']['valid']),
                 note='守的是"退化输入不再静默产出可用数字"。原来的调用方守卫是'
                      'isfinite(lam) and lam>0, 而平坦场的 lam 恰好有限且为正 '
                      '(= box), 那个守卫接不住 —— 它不是失效, 是本就不为这件事'
                      '写的。**这条不改取峰规则**: 只加校验与标位, argmax 口径'
                      '一字未动, 因为换掉它需要一个有物理依据的替代规则, 而'
                      '试过的替代方案在真实场上稳定性不达标。'
                      f"实测: 常数场 lam={_f7['常数场']['lam']:.4f} (无效值), "
                      f"周期条纹 (周期 9 格 / 盒子 36 格) lam="
                      f"{_f7['周期条纹']['lam']:.4f} 格。"
                      '**诚实边界**: valid=False 只说"这次输入不满足定波长的'
                      '前提", 不说 lam 一定是错的; valid=True 也不说 lam 有'
                      '独立佐证 —— 那一条由上方 A 段诊断项回答。')

    def _fin(x):
        """JSON 里不要出现 NaN: 未测量/未定义一律写 null。"""
        x = float(x)
        return x if np.isfinite(x) else None

    out['A1'].update({
        'lam_ref_cells': ref_lam,
        'lam_corner_cells': _lam_corner,
        'lam_spec_cells': {str(int(s)): _fin(_spec[s]) for s in seeds},
        'lam_ac_cells': {str(int(s)): _fin(_ac4[s]) for s in seeds},
        'lam_pk_spec_dev_median': _fin(np.nanmedian(_pk_dev)),
        'lam_pk_ac_dev_median': _fin(np.nanmedian(_ac_dev)),
        'lam_legacy_cells': _lam_leg,
        'legacy_agree': {str(int(s)): _fin(a)
                         for s, a in zip(_leg_s, _leg_agree)},
        'legacy_provenance': (_prov.get('seeds') if _prov else None),
        'minority_ref': ref_minor,
        'xi_ac_ref': [fld_info[f'参考 seed {s}']['xi_ac'] for s in seeds],
        'xi_ac_model': fld_info['模型']['xi_ac'],
        'conn_ref': {str(int(s)): _conn[int(s)] for s in seeds},
        'conn_model': {'n_comp': int(_ncm),
                       'mean_size': float(_szm.mean()) if _szm.size else 0.0,
                       'max_size': int(_szm.max()) if _szm.size else 0,
                       'iso_frac': float(((_nbm == 0) & _cm).sum()
                                         / max(_cm.sum(), 1))},
        'degenerate_ref_seeds': [],
        'n_ref_seeds': len(seeds),
        'binarize_rule': 'v > median (两侧同规则, F3 的核心改动)',
        'v13_baseline': {'D_struct': _lda, 'scatter': _lsc, 'ratio': _lratio},
        'calib_change': {'D_struct': float(abs(d_a - _lda) / _lda),
                         'scatter': float(abs(scatter - _lsc) / _lsc),
                         'ratio': float(abs(ratio_a - _lratio) / _lratio)}})

    # ---------------------------------------------------------------- B 段
    print("\n" + "-" * 76)
    print("  B 段 · 时序: Southampton BZ 波峰 / Reading 自振荡水凝胶")
    print("-" * 76)
    print("  **为什么不做斑图形态对比**")
    print("    第一版运行把 Southampton 的 .tif 时空图当二维斑图算描述子, 结果")
    print("    figS1C 的『特征尺度』= 14883.95 格 —— 逼近整条时间轴的宽度。")
    print("    诊断 (_v13_diag_tier3.py) 查明原因: 那些 .tif 是**图版渲染图**,")
    print("    可分离背景 (行均值 x 列均值) 占了方差的 68%~98.8%; 去掉背景后")
    print("    三张图的谱峰全部落在频带边缘 (即整条时间轴), 不存在干净的波列。")
    print("    也就是说: 在那些图上算出的形态描述子度量的是图版布局, 不是斑图。")
    print("    故弃用该路径, B 段只保留**与轴语义无关**的一维时序量。")
    print("  **为什么不做『逐帧波峰位置内部性约束』**:")
    print("    波峰标记是**原始作者给出的数值列** (缓存键 peaks_*), 不是本工作")
    print("    从图像里提取的 —— 所以『重新跑一次波峰提取、再要求峰位离边界 5%』")
    print("    这一步在代码路径上不存在。对作者给出的标记做位置过滤, 只会丢掉")
    print("    真实存在的边界峰。该约束要防的『提取算法把边界伪影当峰』在这里")
    print("    没有对应的实现可防。")

    zs = np.load(files['soton'], allow_pickle=True)
    meta_s = json.loads(str(zs['meta']))
    st_names = [k[3:] for k in zs.files if k.startswith('st_')]

    # 标定: 源文档 0_Process_steps.txt 写 "15 pixels = 2.5 seconds"。
    # 波峰标记 CSV 的 X 列就是时空图的 x 轴, 单位是**像素**, 不是帧。
    # 第一版把这个单位弄错了两次 (先 x2.5 得 225 s, 又 x15 得 1350 s),
    # 正确的换算是 2.5/15 = 1/6 s per px -> fig3 的 90 px 间隔 = 15.0 s。
    px_per_sec = (meta_s['calib']['px_per_frame']
                  / meta_s['calib']['sec_per_frame'])
    sec_per_px = 1.0 / px_per_sec

    peak_rows = []
    lag_rows = {}
    for nm in st_names:
        k = f'peaks_{nm}'
        if k not in zs.files:
            continue
        ps = _peak_period_stats(zs[k], sec_per_px)
        if ps is None:
            continue
        peak_rows.append((nm, ps))
        print(f"  Southampton {nm:>6} 波峰: {ps['n_peaks']} 个, "
              f"间隔中位数 {ps['dt_median_px']:.1f} px = {ps['period_s']:.1f} s, "
              f"间隔 CV = {ps['dt_cv']:.3f}")
        # 独立交叉检验 (需要强度序列, 若缓存里没有就跳过)
        sk = f'series_{nm}'
        if sk in zs.files:
            la = _series_lag_agreement(zs[sk], ps['dt_median_px'])
            if la is not None:
                lag_rows[nm] = la
                print(f"           独立检验 (强度序列自相关): "
                      f"ac(T/2)={la['ac']['T/2']:.3f}, ac(T)={la['ac']['T']:.3f}, "
                      f"ac(1.5T)={la['ac']['1.5T']:.3f}, ac(2T)={la['ac']['2T']:.3f} "
                      f"-> 比值 {la['ratio_T_over_half']:.2f}, {la['verdict']}")
    if peak_rows:
        cvs = np.array([p['dt_cv'] for _n, p in peak_rows])
        pers = np.array([p['period_s'] for _n, p in peak_rows])
        record_metric(
            3, 'III-B1 BZ 振荡周期与规整度',
            'CV = std(dt) / mean(dt), 波峰间隔的变异系数',
            'BZ 时空图里波峰时刻的一维时序统计。**这个量不依赖任何轴语义类比** '
            '—— 波峰时刻本就是一维时序, 与"把时空图当二维斑图"无关, 所以它'
            '不受上面那条图版污染的影响 (波峰标记是原始作者从数据里提出来的'
            '数值列, 不经过 .tif 渲染)。',
            f'_peak_period_stats(): 相邻波峰间隔的中位数与变异系数; '
            f'标定 15 px = 2.5 s -> {sec_per_px:.4f} s/px',
            'Southampton D0363 (Sci. Rep. 2018) 三份独立记录',
            {'n_records': len(peak_rows),
             'dt_cv': cvs.tolist(), 'period_s': pers.tolist(),
             'period_spread': float((pers.max() - pers.min()) / pers.mean()),
             'lag_agreement': {n: {'ac': v['ac'], 'ratio_T_over_half':
                                   v['ratio_T_over_half'],
                                   'lag_best': v['lag_best'],
                                   'confirmed': v['confirmed']}
                               for n, v in lag_rows.items()}},
            None, None, None,
            note=(f'三个记录的周期中位数 {np.round(pers,1).tolist()} s, '
                  f'最大/最小 = {pers.max()/pers.min():.2f}。'
                  f'**不设达标线**: 第一版曾用 "CV < 0.5" 判规整度, 但 figS1C 的 '
                  f'CV = {cvs.max():.3f} (>0.5) 是**那条记录的真实性质** '
                  f'(不规则爆发式波形), 不是流程失败 —— 把记录的性质当成流程的'
                  f'达标项是判据错配。三个记录的周期差异本身是有信息的: '
                  f'不同记录/不同驱动条件下的 BZ 振荡周期确实不同。'
                  + (f' **独立交叉检验** (强度序列在滞后 T 处的自相关, 与波峰'
                     f'标记是两套独立提取): '
                     + '; '.join(
                         f"{n}: ac(T)/ac(T/2)={v['ratio_T_over_half']:.2f}, "
                         f"极大在 {v['lag_best']}"
                         for n, v in lag_rows.items())
                     + f'。检验结论与 CV 的排序一致 (CV 越小越确认), 三个量'
                       f'(周期/CV/滞后峰) 互为佐证 —— 这正是本层作为"方法学'
                       f'展示"要交付的东西。'
                     if lag_rows else
                     ' 强度序列不在缓存里, 交叉检验跳过。')))
        out['B1'] = {'dt_cv': cvs.tolist(), 'period_s': pers.tolist(),
                     'names': [n for n, _ in peak_rows],
                     'lag_agreement': {n: v['verdict'] for n, v in lag_rows.items()},
                     'passed': None}

    # Reading: 时间序列主周期 (机械压缩下的自振荡水凝胶)
    zr = np.load(files['reading'], allow_pickle=True)
    lum_keys = [k for k in zr.files if k.startswith('lum_')]
    per = {}
    trim_scan = {}
    lag_rows_read = {}
    drift_ratio_read = {}
    for k in lum_keys:
        y_read = np.asarray(zr[k], dtype=float)
        sp = _series_period(y_read)
        if sp is None:
            continue
        nm = k[4:]
        per[nm] = sp
        pf = sp['period_frames']
        print(f"  Reading {nm:>32}: {sp['verdict']}  漂移占比 "
              f"{sp['drift_frac']:.1%}, 带内最大在 k={sp['k_band_argmax']} "
              f"(周期 {sp['period_frames_band_argmax']:.1f} 帧 = "
              f"{sp['period_frames_band_argmax']/len(zr[k])*100:.0f}% 全长), "
              f"突出度 {sp['prominence']:.0f}"
              + (f" -> 主周期 {pf:.1f} 帧" if pf else ""))

        # 【裁剪敏感性 —— 预热暗帧】
        # 三条序列的**头部**都是相机预热的暗帧 (亮度 ~25.6 vs 稳态 84~115),
        # 是一段硬阶跃。阶跃是宽频的, 会给谱灌进高频功率, 从而把"漂移占比"
        # 这个数**压小** —— 也就是说上面报出的 9.9%/15.3%/16.1% 是被低估的。
        # 处理方式: **不改主口径** (改了就是在动已经报出去的数), 而是把裁剪后
        # 的重算结果并排存下来, 并加一条守卫要求结论在两种处理下都成立。
        nd = _leading_dark(y_read)
        trials = []
        for t in sorted({0, nd, nd + 16}):
            st = _series_period(y_read[t:])
            if st is not None:
                trials.append({'trim': int(t), 'detected': bool(st['detected']),
                               'drift_frac': st['drift_frac'],
                               'k_band_argmax': st['k_band_argmax'],
                               'prominence': st['prominence']})
        trim_scan[nm] = {'n_dark': int(nd), 'trials': trials}

        # 【前后窗口漂移比 —— 量化预热段的影响, 只诊断不裁剪】
        dr, s_head, s_tail = _window_drift_ratio(y_read)
        drift_ratio_read[nm] = dr

        # 【独立交叉检验 —— 自相关滞后】
        # 谱方法是第一条路径, 但它对不规则波形是**弱**的: 波峰间隔的变异系数
        # 一大会把谱线自己抹平, 这正是"未检出"的成因。自相关是第二条独立路径 ——
        # 它不要求波形严格周期, 只问"隔一个周期回到自己吗"。
        # 候选滞后取带内 argmax 周期 (它在未检出的序列上照样有值), 于是两法
        # **不一致本身就构成证据**: 谱峰与自相关都对不上 => "未检出"不是谱方法
        # 偏保守, 而是真的没有可确认的周期。
        # 两个门槛按**本域**给, 不用默认值 —— 理由见 _series_lag_agreement 的说明。
        # skip: 默认的 "跳 3 个周期" 是 BZ 空间序列"斑图形成前的混合段", Reading 是
        #   物理实验, 没有这一段; 照搬会跳掉 810/2161 = 37% 的记录, 反而把自相关
        #   洗白 (Reference 的 ac(T/2) 从 +0.401 掉到 +0.036, 见 W8 侦察)。这里只掐
        #   掉**实测到的**相机预热暗帧 (nd, 上面 trim_scan 已算过), 口径与它一致。
        # min_lag_cycles: 默认 12 在这里结构性不可达 (全长只有 8.0 个周期), 改成 4 ——
        #   少于 4 个独立周期就谈不上"隔一个周期回到自己"。
        _s2 = np.column_stack([np.arange(nd, len(y_read), dtype=float),
                               y_read[nd:]])
        _la = _series_lag_agreement(_s2, sp['period_frames_band_argmax'],
                                    skip_periods=0.0, min_lag_cycles=4.0)
        if _la is not None:
            lag_rows_read[nm] = _la

        print(f"            裁剪敏感性: 头部暗帧 {nd} 帧, 前后窗口漂移比 "
              f"{dr:.2f} (前段斜率 {s_head:+.3e} / 后段 {s_tail:+.3e}); "
              + ' | '.join(f'trim={t["trim"]}: 漂移 {t["drift_frac"]:.1%}, '
                           f'{"检出" if t["detected"] else "未检出"}'
                           for t in trials))
        if _la is not None:
            print(f"            自相关交叉检验 (候选滞后 {_la['lag_T_px']} 帧, "
                  f"{_la['n_used']} 点): ac(T)={_la['ac_T']:+.3f}, "
                  f"ac(T/2)={_la['ac_half']:+.3f}, "
                  f"比={_la['ratio_T_over_half']:+.2f} -> {_la['verdict']}")
    if per:
        detected = {n: bool(s['detected']) for n, s in per.items()}
        # 注里那两个数从实测算出来, 不写死 —— 否则改了参数注就成了假话
        prom_txt = ' / '.join(f"{s['prominence']:.0f}" for s in per.values())
        _trim_txt = '; '.join(
            f"{n}: " + ' -> '.join(f"{t['drift_frac']:.0%}" for t in v['trials'])
            for n, v in sorted(trim_scan.items()))
        base = next((n for n in per if 'Reference' in n), None)
        # 只有**全部**序列都检出周期时, 比值才有意义
        ratios = {}
        if base and all(detected.values()):
            ratios = {n: float(s['period_frames'] / per[base]['period_frames'])
                      for n, s in per.items()}
        record_metric(
            3, 'III-B2 Reading 驱动响应', '主周期(受驱) / 主周期(参考)',
            'BZ 自振荡水凝胶在周期性机械压缩下的逐帧平均亮度主周期, 相对无驱动'
            '参考序列的比值。机械压缩是外驱、自振荡是内禀 —— 比值直接量'
            '"外驱有没有把内禀振荡拉过去"(entrainment)。'
            '**本次运行的结论是"未检出", 因此不给比值** —— 见 note。',
            '_series_period(): 线性去趋势 + 谱域高通 (置零 k<8) + Hann 窗 '
            '+ 带内局部极大 + 突出度判定; '
            f'{len(per)} 个序列, 每个 {len(zr[lum_keys[0]])} 帧',
            'Reading (Geher-Herczegh PhD 2023) 三份记录: 参考 / 2min / 20min 压缩',
            {'verdict': {n: s['verdict'] for n, s in per.items()},
             'detected': detected,
             'drift_frac': {n: s['drift_frac'] for n, s in per.items()},
             'band_argmax_period_frames':
                 {n: s['period_frames_band_argmax'] for n, s in per.items()},
             'prominence': {n: s['prominence'] for n, s in per.items()},
             'ratios_vs_reference': ratios,
             'trim_sensitivity': trim_scan},
            None, None, None,
            note=('**三条序列全部未检出周期**。第一版运行曾报三条的主周期都是 '
                  '2161.0 帧 = 100.0% 全长 —— 那不是发现: 三条长度都是 2161, '
                  '谱峰也都在 k=1, 即整条记录。诊断查明是慢漂移 (凝胶干燥/'
                  '照明漂移) 主导: 仅去线性趋势后 k<8 仍占 51%~78% 的功率。'
                  '改用谱域高通后, 三条的带内最大值都恰好落在频带边缘, '
                  '说明剩余谱是**红的** (功率随 k 单调下降), 根本没有振荡峰。'
                  '检出判据因此要求峰是"带内**内部**的局部极大"且突出度 >= '
                  f'{PROM_MIN:.0f} 倍局部背景; 该阈值按零假设标定过 (纯红噪声 '
                  '300 次抽样突出度最大 113, 而 T=25~200 帧的正弦信号最小 2309)。'
                  '物理上说得通: 这些凝胶存在相位不均匀的螺旋波, 全局平均亮度'
                  '会把反相的局部振荡抵消掉, 只剩漂移 —— 因此这一项的正确结论'
                  '是"该观测量检不出周期", 而不是给它编一个周期。'
                  f' **注意**三条的突出度 {prom_txt} 其实**都超过**了 '
                  f'{PROM_MIN:.0f} 这个阈值 —— 挡下检出的是"边缘极大不是峰"'
                  '这一条, 不是突出度不够。'
                  ' **另一处必须说明的污染**: 三条序列的头部都是相机预热的暗帧 '
                  '(亮度 ~25.6 vs 稳态 84~115), 是一段硬阶跃。阶跃谱是宽频的, '
                  '会给整条谱灌高频功率, 从而把漂移占比**压小** —— 也就是说'
                  '上面报出的那三个漂移数是被**低估**的下界。按"掐掉头部 N 帧"'
                  f'重算 (trim = 0 / 暗帧数 / 暗帧数+16): {_trim_txt}。'
                  '**但结论不受影响**: 全部 (序列 x 裁剪) 组合都未检出, '
                  '见 trim_sensitivity 与守卫 T3-5。'))
        out['B2'] = {'verdict': {n: s['verdict'] for n, s in per.items()},
                     'detected': detected, 'ratios': ratios, 'passed': None,
                     'lag_agreement_read': {
                         n: {'lag_T_frames': int(v['lag_T_px']),
                             'ac_T': _fin(v['ac_T']),
                             'ac_half': _fin(v['ac_half']),
                             'ratio_T_over_half': _fin(v['ratio_T_over_half']),
                             'lag_best': v['lag_best'],
                             'n_used': int(v['n_used']),
                             'confirmed': bool(v['confirmed']),
                             'verdict': v['verdict']}
                         for n, v in sorted(lag_rows_read.items())},
                     'drift_ratio_head_tail': {
                         n: _fin(v) for n, v in sorted(drift_ratio_read.items())},
                     'trim_sensitivity': trim_scan}

    # ---------------------------------------------------------------- C 段
    print("\n" + "-" * 76)
    print("  C 段 · 判别力: 这套描述子到底分不分得开两种斑图族?")
    print("-" * 76)
    # 分子取 A 段的模型 <-> 参考距离 (而不是第一版用的 CLIP <-> Southampton)。
    # 原因: 那条路径建立在被否掉的"把 .tif 图版当二维斑图"之上, 分子已被污染。
    # 用 A 段的距离反而更切题 —— 它问的正是"模型斑图与参考斑图是不是同一族"。
    d_cross = float(d_a)
    dprime = float(d_cross / scatter) if scatter > 0 else float('nan')
    print(f"  跨族距离 (模型 vs 参考) = {d_cross:.4f}")
    print(f"  族内散布 (参考种子间)   = {scatter:.4f}")
    print(f"  判别力 d' = {dprime:.2f}   (>3 可区分; 1~2 弱可区分; <1 不可区分)")
    record_metric(
        3, 'III-C1 描述子判别力', "d' = D_struct(模型 vs 参考) / D_struct(参考种子间)",
        '这套结构描述子把「阶段六 GS 斑图」从「CLIP GS 斑图」里分出来的能力, '
        '以参考自身的种子间散布为单位。它决定 A 段那个距离有没有信息量。',
        '分子 = A 段模型-参考描述子距离; 分母 = 参考侧 3 个种子的描述子距离均值',
        '自设判据 —— 无文献标准; 阈值 3 取自"跨族差异应远大于同族采样噪声"',
        {'D_cross': d_cross, 'scatter': scatter, 'd_prime': dprime},
        None, None, None,
        note=(f'd\' = {dprime:.2f}。**这一项是对已发布基线那条"0/4 达标"的重新解读**。'
              f'已发布基线用四个阈值型判据对第三层做了裁决并得出"全部未达标"; '
              f'而这里量出描述子的判别力只有约 {dprime:.1f} 倍噪声底 —— 也就是说'
              f'两个斑图族在这个描述子空间里**只是弱可区分**。弱可区分的判据'
              f'用来做达标裁决, 结果必然对阈值敏感、对采样敏感, 于是"未达标"'
              f'这个结论里究竟有多少是"模型不像", 有多少是"判据问不出问题", '
              f'分不开。把判别力量出来, 才能说清那条"失败"里哪一部分'
              f'是物理、哪一部分是工具。这就是降级为方法学展示后应当交付的东西。'))
    out['C1'] = {'D_cross': d_cross, 'scatter': scatter, 'd_prime': dprime,
                 'passed': None}
    # 【v14·F3】clip_seeds 的语义变了: 它现在是 v14 自建种子的编号, 不是归档里
    # 的 0/5/10; n_clip_frames 也从"归档 clip 的 300 帧"变成"自建种子保留的
    # 尾窗帧数 (TAIL=100)"。两个键名都保留以维持调用方兼容, 但语义由
    # ref_source / n_ref_frames / legacy_clip_seeds 三个新键明确标出。
    _legacy_clip_seeds = _ls
    out['ref_meta'] = {'clip_seeds': seeds, 'soton_names': st_names,
                       'n_clip_frames': int(zc[f'bool_{seeds[0]}'].shape[0])
                       if seeds else 0,
                       'ref_source': _ref_from,
                       'n_ref_frames': int(zc[f'bool_{seeds[0]}'].shape[0])
                       if seeds else 0,
                       'n_ref_seeds': len(seeds),
                       'legacy_clip_seeds': _legacy_clip_seeds,
                       'cache_dir': cache,
                       'v14_cache_dir': v14cache,
                       'binarize_rule': 'v > median (两侧同规则)',
                       'soton_image_path_dropped': True}

    # ---------------------------------------------------------------- 流程守卫
    # 第三层降级为方法学展示后, 能设判定的是"流程有没有产出可用于判断的量",
    # 而不是"物理像不像"。以下三条都是可证伪的流程断言。
    finite_a = bool(np.isfinite(d_a) and np.isfinite(r_p))
    record_guard('T3-1 A 段产出有限值', 'tier3_structural_ladder',
                 f"D_struct={d_a:.4f}, R_P={r_p:.4f} 均为有限",
                 finite_a,
                 note='参考侧或模型侧任一退化 (全 0/全 1 场) 都会让描述子变成 '
                      'NaN/inf; 这条守的是"这套尺子还量得出数"。')
    n_peaks_ok = len(peak_rows) == len(st_names)
    record_guard('T3-2 波峰时序全部可用', 'tier3_structural_ladder',
                 f"{len(peak_rows)}/{len(st_names)} 份记录产出了波峰间隔统计",
                 n_peaks_ok,
                 note='波峰标记是原始作者的数值列, 不经过 .tif 图版渲染; '
                      '这条守的是 B 段唯一"硬"的量确实拿到了。')
    det_ok = bool(per) and all('verdict' in s for s in per.values())
    record_guard('T3-3 时序周期判定给出结论', 'tier3_structural_ladder',
                 f"{len(per)} 份序列都有明确的检出/未检出结论",
                 det_ok,
                 note='**"未检出"也算结论**。这条守的是不会出现"报了个数但没说 '
                      '它可不可信"的情况 —— 第一版正是那样报出了 2161.0 帧。')
    lag_ok = bool(lag_rows) and len(lag_rows) == len(peak_rows)
    record_guard('T3-4 波峰周期通过独立交叉检验', 'tier3_structural_ladder',
                 f"{len(lag_rows)}/{len(peak_rows)} 份记录的波峰周期在强度序列上"
                 f"做了自相关滞后检验",
                 lag_ok,
                 note='守的是"两个独立提取都对得上"这件事**被检验过**。注意它守'
                      '的不是"检验通过" —— 记录本身不规则 (CV 大) 时检验本就不'
                      '该通过, 那是物理结论。判据与结论分离, 否则又回到"用阈值'
                      '裁决物理"的错法。')
    _tr = [t for v in trim_scan.values() for t in v['trials']]
    record_guard('T3-5 未检出结论不依赖预热段',
                 'tier3_structural_ladder',
                 f"{len(_tr)} 种 (序列 x 裁剪) 组合全部未检出; "
                 f"头部暗帧 {[v['n_dark'] for v in trim_scan.values()]} 帧",
                 bool(_tr) and not any(t['detected'] for t in _tr),
                 note='守的是"这个结论不是预处理假象"。三条序列的头部是相机预热'
                      '的暗帧 (亮度 25.6 vs 稳态 84~115), 一段硬阶跃; 它给谱灌进'
                      '宽频功率, 把漂移占比**压小**。这条守的是: 去掉预热段、以及'
                      '再去掉一段上升沿之后, "未检出"依然成立 —— 否则这个结论就'
                      '只是那 16 帧暗帧的产物。')

    # T3-4b: Reading 侧的第二条独立路径 (谱方法 vs 自相关滞后)。
    # 与 T3-4 同一个模式: 守"检验被做过", 不守"检验通过" —— 这里检验本就
    # **不该通过**, 三条序列的谱都判"未检出", 自相关若也确认不了周期, 那是
    # 两条路径互相印证这个负面结论, 而不是失败。
    _lagr = list(lag_rows_read.values())
    _conf_r = [r for r in _lagr if r['confirmed']]
    _lags_txt = ', '.join(f"{n}: {v['lag_T_px']}帧 ac(T)={v['ac_T']:+.2f} "
                          f"比={v['ratio_T_over_half']:+.2f}"
                          for n, v in sorted(lag_rows_read.items()))
    record_guard('T3-4b Reading 谱峰经自相关交叉检验',
                 'tier3_structural_ladder',
                 f"{len(_lagr)}/{len(per)} 份 Reading 序列在强度序列上做了自相关"
                 f"滞后检验 (候选滞后取带内 argmax 周期), 其中 {len(_conf_r)} 份"
                 f"确认周期",
                 bool(_lagr),
                 note='守的是"Reading 侧也有第二条独立路径", 不是"它确认了周期"。'
                      '谱方法对不规则波形是弱的 (波峰间隔 CV 大时谱线被自身抹平), '
                      '自相关不要求严格周期, 所以两法**都指不出周期、这个一致'
                      '本身就构成证据**: 谱峰与自相关都没给出可确认的周期 '
                      '=> "未检出"不是谱方法偏保守。判据与结论分离 —— 让阈值'
                      '裁决物理是已经犯过一次的错法。'
                      ' (v15·W8: 自相关在掐掉头部预热暗帧后计算, 口径同 '
                      'trim_sensitivity; 见调用点的门槛说明。)'
                      f'实测: {_lags_txt or "无可用序列"}。')

    # T3-5b: 前后窗口漂移比 (量化预热段影响, 只诊断不裁剪)
    _dr = {n: v for n, v in drift_ratio_read.items() if np.isfinite(v)}
    _dr_max = max(_dr.values()) if _dr else float('nan')
    _dr_txt = ', '.join(f'{n}: {v:.2f}' for n, v in sorted(_dr.items()))
    record_guard('T3-5b 前后窗口漂移比已量化 (不作裁剪依据)',
                 'tier3_structural_ladder',
                 f"前 10% / 后 10% 最小二乘斜率之比 max = {_dr_max:.2f} > 2 "
                 f"(逐个: {_dr_txt or '未测量'})",
                 bool(_dr) and _dr_max < 2.0, expect_pass=False,
                 note='**诊断项, 不计入通过率 (本项设计上就该不通过)**: 头部暗帧'
                      '是一段硬阶跃, 它给整条序列灌进一个单边趋势, 于是前窗口的'
                      '斜率被这个趋势主导。这条把"漂移比有多大"量化出来。'
                      '**它不驱动任何裁剪** —— 自动掐掉前段等于改变已经报出去的'
                      '口径; 裁剪与否由 trim_scan 的敏感性扫描裁决 (见 T3-5)。'
                      '判据与动作分开, 是为了不让一个阈值悄悄改掉读数。')

    # 作图用的紧凑数组 (只在内存里传, 不进 json —— 见 spiral_v13_plots 的说明)
    out['_plot'] = {
        'A': {'keys': list(_KEYS), 'seeds': seeds,
              'mod': [dmod[k] for k in _KEYS],
              'ref': [[d_ref[s][k] for k in _KEYS] for s in seeds],
              'pk': {'xm': xm.tolist(), 'ym': ym.tolist(),
                     'xr': xr.tolist(), 'yr': yr.tolist()},
              'D': d_a, 'scatter': scatter, 'R_P': r_p,
              # 模型侧 + 参考侧**前 3 个**种子的缩略图, 供四张面板 (十七~二十)。
              # 这四张图把参考侧的实际形态亮出来, 让读者自己看它是不是斑图。
              # 换自建种子后参考侧全部健康 (少数相恒 0.5), 故这里的用途是
              # "呈现参考家族的实际形态"而非"指认退化样本"—— 面板只放 3 个,
              # 是因为网格只有 4 列 x 1 行可用 (第十三~十六 占了 [3,:]); 全 12 个在
              # result/_v16_data.json 的 A1.ref_panel 里。
              'fields': {k: fld_info[k] for k in
                         ['模型'] + [f'参考 seed {s}' for s in seeds[:PANEL_SEEDS]]},
              'panel_seeds': [int(s) for s in seeds[:PANEL_SEEDS]],
              'n_seeds_total': len(seeds),
              'ref_minor': ref_minor,
              'lam_mod': float(dmod['lam_pk']), 'lam_ref': ref_lam,
              'lam_corner_cells': _lam_corner,
              'lam_legacy_cells': _lam_leg,
              'lam_pk_ac_dev_median': _fin(np.nanmedian(_ac_dev)),
              'lam_spec_cells': {str(int(s)): _fin(_spec[s]) for s in seeds},
              'legacy_agree': {str(int(s)): _fin(a)
                               for s, a in zip(_leg_s, _leg_agree)},
              'n_lambda_model': n_lam_mod, 'n_lambda_ref': n_lam_ref,
              'conn_ref': {str(int(s)): _conn[int(s)] for s in seeds},
              'conn_model': {'n_comp': int(_ncm),
                             'mean_size': float(_szm.mean()) if _szm.size else 0.0,
                             'max_size': int(_szm.max()) if _szm.size else 0,
                             'iso_frac': float(((_nbm == 0) & _cm).sum()
                                               / max(_cm.sum(), 1))}},
        'B': {'names': [n for n, _ in peak_rows],
              'peaks': {nm: {'dt_cv': ps['dt_cv'], 'period_s': ps['period_s'],
                             'n': ps['n_peaks'],
                             'lag_verdict': lag_rows.get(nm, {}).get('verdict'),
                             'lag_ratio': lag_rows.get(nm, {}).get('ratio_T_over_half'),
                             # 完整的自相关字典 (四个滞后各一个系数), 供"独立
                             # 交叉检验"面板直接画出来 —— 只报 verdict 字符串
                             # 的话, 读者无法核对"极大到底落在哪个滞后"。
                             'ac': (lag_rows.get(nm) or {}).get('ac'),
                             'lag_best': (lag_rows.get(nm) or {}).get('lag_best')}
                        for nm, ps in peak_rows},
              'reading': {n: {'lum': np.asarray(zr['lum_' + n])[::4].tolist(),
                              'detected': bool(s['detected']),
                              'drift_frac': s['drift_frac'],
                              'verdict': s['verdict'],
                              # 未检出结论所依据的谱与阈值, 见 _series_period
                              'power': s['power'],
                              'power_thresh': s['power_thresh'],
                              'prominence': s['prominence'],
                              'k_band_argmax': s['k_band_argmax'],
                              'min_cycles': s['min_cycles'],
                              'n_frames': s['n_frames'],
                              'n_dark': trim_scan[n]['n_dark'],
                              'trim_trials': trim_scan[n]['trials']}
                          for n, s in per.items()},
              'ratios': ratios},
        'C': {'d_prime': dprime, 'D_cross': d_cross, 'scatter': scatter},
    }

    print("\n  【第三层总诚实边界 · 必读】")
    print("    1. 本层已从『宇宙学对标』降级为『方法学展示』: 对标对象是 2D 反应-")
    print("       扩散/斑图系统, 方式是结构对结构。不构成任何『复现了观测』的论断。")
    print("    2. **第三层不设物理达标线**。混杂因子 (相区参数不同 / 时空图轴语义")
    print("       不同) 使任何数字都无法单独归因给物理, 在该区间做达标裁决没有")
    print("       信息量。判定项改为流程守卫 T3-1~5。原列的『盒子内波长数差一个")
    print("       量级』**已被 v14 撤回** (旧值来自坏参考场的谱角点伪像), v15·W2 的")
    print("       同单位受控侧更是逐字同参数, 故它不再列为混杂因子。")
    print("    3. Southampton 的 .tif **图版**已被弃用 (可分离背景占方差 68~98.8%,")
    print("       在其上算描述子等于量图版布局); 只保留波峰标记这条数值列。")
    print("    4. Reading 三条序列**全部未检出周期**, 这是结论而非缺失 —— 全局")
    print("       平均亮度在有螺旋波的凝胶上会抵消掉反相振荡, 只剩漂移。")
    print("    5. 模型侧是**确定性**的锁模格子, 没有模型集成; A 段的误差棒来自")
    print("       **参考侧**的 3 个种子, 不是模型侧。")
    return out
