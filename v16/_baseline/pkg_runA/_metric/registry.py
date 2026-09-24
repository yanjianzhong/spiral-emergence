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
"""spiral_metric_v16._metric.registry —— 由 spiral_metric_v16.py 拆分。

指标/守卫注册表 + 打印与 JSON + 维度无关场工具 + 跨层共用常量

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



# from spiral_metric import (
#     record_metric, record_guard, guard_pass_rate,
#     fit_central_charge,
#     # ---- 既有的指标函数, 原样复用 (不改定义, 不改阈值) ----
#     metric_gap_closure, metric_mera_consistency,
#     metric_chain_rmse, metric_entropy_kl,
#     metric_correlation_exponent, metric_jordan_peak,
#     print_health_report,
#     # ---- 维度无关的辅助函数 (第三层改用得上) ----
#     _band_power, _delta2, _char_scale, _xi_from_fft, _first_zero,
#     _mode_census, _shape_residual,
#     # ---- 只被作图用到的两个 (2.3 关联衰减指数面板要在图上重画那条拟合线) ----
#     exact_ground_state, boundary_correlation_graph,
# )


# ============================================================================
# v14 · 指标注册表 + 守卫注册表  (体检报告的两块基石)
# ============================================================================
#
# 为什么要有注册表, 而不是把数字直接 print 出来?
#   因为"打印一个数字"是不可证伪的。v11 说"曲率锚点全部到机器精度",
#   这句话没有任何代码能判定它错。注册表把每个数字绑上
#      定义 / 计算方法 / 参照(基线) / 目标 / 实测
#   于是"达标与否"变成一个 bool, 而 bool 是可以被证伪的。
#
# 与 CHAIN 的分工:
#   CHAIN  记录 阶段之间怎么连接 (上游物理量 -> 下游参数)
#   METRICS 记录 连接之后对不对 (实测 vs 基线 vs 目标)
#   GUARDS 记录 哪些前提必须成立 (可证伪的守卫, 汇成一个通过率)

METRICS = []   # 三层指标 (tier 1/2/3)


GUARDS = []    # 可证伪守卫 (通过率 = 应当通过者中通过的比例)




def record_metric(tier, name, symbol, definition, method, reference,
                  measured, baseline, target, passed, note='',
                  source_id=None, claim_id=None):
    """
    记录一条指标。

    tier      1=内部自洽性  2=微观物理对标  3=宏观结构/Surrogate 对标
    name      中文指标名          symbol    符号或公式
    definition 定义 (一句话说清"这个数在量什么")
    method    计算方法 (用哪个函数/哪条公式算出来的)
    reference 参照物 —— 第1层是 v11 基线, 第2层是解析解, 第3层是数据来源
    measured  实测值      baseline 基线值      target 目标 (字符串, 便于写不等式)
    passed    True/False, 或 None 表示"只能诊断, 不给数值判定"

    --- 以下两个字段是 v16·B3 新增, 默认值使既有调用**行为一字不动** ---
    source_id 数据来源标识 (str 或 None)。默认 None = 未声明来源。
              非 None 表示"本条读数出自这个源"; 若同一 source_id 下有 >=2 条读数,
              它们就是**同源**, 必须在 `_v16_claim_ledger._SAME_SOURCE_REGISTRY`
              里显式登记, 否则 `claim_test_ledger` 检查 (c) 判失败。
    claim_id  主张编号 (str 或 None), 供 `record_guard(tests_claims=...)` 指认。
              默认 None = 本条暂未被任何守卫按 id 指认 (由检查 (a) 报出孤儿)。
    """
    rec = {
        'tier': int(tier),
        'name': name,
        'symbol': symbol,
        'definition': definition,
        'method': method,
        'reference': reference,
        'measured': _jsonable(measured),
        'baseline': _jsonable(baseline),
        'target': target,
        'passed': None if passed is None else bool(passed),
        'note': note,
        'source_id': source_id,
        'claim_id': claim_id,
    }
    METRICS.append(rec)
    return rec




def record_guard(name, where, condition, ok, expect_pass=True, note='',
                 tests_claims=(), source_id=None):
    """
    记录一条守卫。守卫 = 一个**可证伪**的检查: 条件明确, 且可能失败。

    expect_pass=False 表示"设计上就应当失败"的诊断项 —— 例如 v11 阶段三-c
    的权重自由变体本来就不该秩 1 化。这类项**不计入通过率**:
    否则通过率可以靠塞进注定要失败的项做低, 也可以靠删掉它们做高。
    一个诚实的通过率必须先把"该通过的"和"只想看看的"分开。

    --- 以下两个字段是 v16·B3 新增, 默认值使既有调用**行为一字不动** ---
    tests_claims 本守卫守的是哪几条主张, 元素是 `record_metric` 登记的
                 `claim_id`。默认 () = 未按 id 指认 (此时由 `_v16_claim_ledger`
                 的 where↔method 规则兜底, 见检查 (a))。
    source_id    本守卫这条读数自己的数据来源标识, 语义同 `record_metric`。
    """
    rec = {'name': name, 'where': where, 'condition': condition,
           'ok': bool(ok), 'expect_pass': bool(expect_pass), 'note': note,
           'tests_claims': tuple(tests_claims), 'source_id': source_id}
    GUARDS.append(rec)
    return rec




def guard_pass_rate():
    """应当通过的守卫里, 实际通过的比例。诊断项单列, 不混入分母。"""
    exp = [g for g in GUARDS if g['expect_pass']]
    diag = [g for g in GUARDS if not g['expect_pass']]
    n_pass = sum(1 for g in exp if g['ok'])
    return {'n_pass': n_pass, 'n_expect': len(exp),
            'rate': (n_pass / len(exp)) if exp else float('nan'),
            'n_diag': len(diag),
            'n_diag_notok': sum(1 for g in diag if not g['ok'])}




def _jsonable(v):
    """把 numpy 标量转成原生 Python, 供 json.dump 使用。"""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v




def _deep_jsonable(v):
    """
    `_jsonable` 只管最外层 —— 指标函数的返回值里嵌着 numpy 数组
    (例如 P(k) 的分壳结果、形态学泛函的曲线), 直接塞进 json.dump 会在
    第 4 层抛 "Object of type ndarray is not JSON serializable"。
    这里递归下去, 叶子仍交给 `_jsonable`。
    """
    v = _jsonable(v)          # ndarray -> list 也在这里发生
    if isinstance(v, dict):
        return {str(k): _deep_jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_deep_jsonable(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)




def _fmt_val(v):
    """体检报告里的数值格式: 极小/极大自动切科学计数法。"""
    if v is None:
        return '—'
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return '是' if v else '否'
    if isinstance(v, float):
        if not np.isfinite(v):
            return 'nan' if np.isnan(v) else ('inf' if v > 0 else '-inf')
        a = abs(v)
        if a != 0.0 and (a < 1e-3 or a >= 1e5):
            return f'{v:.3e}'
        return f'{v:.4f}'
    return str(v)




_TIER_TITLE = {
    1: '第一层 · 内部自洽性 (模型对自己)',
    2: '第二层 · 微观物理对标 (模型对解析解)',
    3: '第三层 · 宏观结构 / Surrogate 对标 (模型对外部数据)',
}



_TIER_SHORT = {
    1: '内部自洽性',
    2: '微观物理对标',
    3: '宏观结构对标',
}




def print_health_report():
    """体检报告主表: 三层指标逐条列出, 最后给达标汇总与守卫通过率。"""
    print("\n" + "=" * 76)
    print("  体检报告 · 三层量化指标 (15)")
    print("=" * 76)

    for tier in (1, 2, 3):
        rows = [m for m in METRICS if m['tier'] == tier]
        if not rows:
            continue
        print(f"\n  ── {_TIER_TITLE[tier]} ──")
        for i, m in enumerate(rows, 1):
            if m['passed'] is None:
                verdict = '[诊断]'
            else:
                verdict = '[达标]' if m['passed'] else '[未达标]'
            print(f"\n  {tier}.{i} {m['name']}   {m['symbol']}   {verdict}")
            print(f"      定义: {m['definition']}")
            print(f"      方法: {m['method']}")
            print(f"      参照: {m['reference']}")
            print(f"      实测 = {_fmt_val(m['measured'])}   "
                  f"基线 = {_fmt_val(m['baseline'])}   目标: {m['target']}")
            if m['note']:
                print(f"      诚实边界: {m['note']}")

    # ---- 守卫段 ----
    print(f"\n  ── 守卫通过率 (可证伪检查) ──")
    for g in GUARDS:
        tag = '通过' if g['ok'] else '未通过'
        mark = '' if g['expect_pass'] else '  [诊断项, 不计入分母]'
        print(f"      {g['name']:6s} {g['where']:34s} {tag}{mark}")
        if g['condition']:
            print(f"             条件: {g['condition']}")
        if g['note']:
            print(f"             注: {g['note']}")
    gpr = guard_pass_rate()
    print(f"\n      通过率 = {gpr['n_pass']}/{gpr['n_expect']} = {gpr['rate']:.1%}"
          f"   (另有诊断项 {gpr['n_diag']} 项, 其中未通过 "
          f"{gpr['n_diag_notok']} 项 —— 那是设计上就该失败的对照)")

    # ---- 汇总 ----
    print("\n  ── 汇总 ──")
    tot_ok = tot = 0
    for tier in (1, 2, 3):
        rows = [m for m in METRICS if m['tier'] == tier]
        if not rows:
            continue
        ok = sum(1 for m in rows if m['passed'])
        n = sum(1 for m in rows if m['passed'] is not None)
        diag = len(rows) - n
        tot_ok += ok
        tot += n
        extra = f"  (另有诊断项 {diag} 项)" if diag else ""
        print(f"      第{tier}层 ({_TIER_SHORT[tier]}): 达标 {ok}/{n}{extra}")
    print(f"      合计: 达标 {tot_ok}/{tot}")



def _cic_power_window(kgrids, dx):
    """
    CIC 赋值的**功率**窗 |W(k)|^2 = prod_a sinc^2(k_a dx / 2)。

    要除以的就是它本身, 只除一次 —— CIC 把密度与核 W_cic 卷积, 于是在功率谱上
    出现的是 |W_cic|^2 = prod sinc^2(k_a dx/2)。再平方一次(除以窗的平方)
    会静默地把高 k 抬起来, 曲线看上去照样"很正常"。这是本函数唯一值得注释的地方。

    约定的坑: numpy 的 np.sinc(u) = sin(pi u)/(pi u), 所以 sinc(k dx/2) 必须写成
    np.sinc(k*dx/(2*pi))。写成 np.sinc(k*dx/2) 同样是静默错误。

    形状: 返回的数组与 rfftn 的输出同形 (最后一轴可以是半轴)。
    """
    shape = tuple(len(k) for k in kgrids)
    w = np.ones(shape, dtype=np.float64)
    for a, kg in enumerate(kgrids):
        sh = [1] * len(kgrids)
        sh[a] = len(kg)
        w = w * np.sinc(kg.reshape(sh) * dx / (2 * np.pi)) ** 2
    return w



def _kgrids(shape, box, real_last=False):
    """各轴的角波数网格 (rad / Mpc/h)。real_last=True 时最后一轴取 rfft 的半轴。"""
    ks = []
    for a, n in enumerate(shape):
        d = box / n
        if real_last and a == len(shape) - 1:
            ks.append(np.fft.rfftfreq(n, d=d) * 2 * np.pi)
        else:
            ks.append(np.fft.fftfreq(n, d=d) * 2 * np.pi)
    return ks



def _band_power(delta, box, dim, deconv=True):
    """
    各向同性功率谱 P(k), 按 |k| 球壳分箱。

    归一化约定 (写死在这里, 因为它是最容易静默错一个因子 6 的地方):
        P(k) = |delta_k|^2 * box^dim / N^(2*dim)
    使得  int d^dim k / (2 pi)^dim  P(k)  =  sigma^2。
    球壳用 **该壳内实际 |k| 的均值** 作为横坐标, 不用箱中心 —— 低 k 处球壳很稀疏
    (基频壳只有 6 个模), 用箱中心会制造假散布。
    """
    ng = delta.shape
    ntot = int(np.prod(ng))
    fk = np.fft.rfftn(delta)
    ks = _kgrids(ng, box, real_last=True)
    kfull = _kgrids(ng, box, real_last=False)
    p = np.abs(fk) ** 2 * box ** dim / ntot ** 2
    if deconv:
        p = p / _cic_power_window(ks, box / ng[0])
    kmag = np.sqrt(sum((k ** 2).reshape([len(k) if a == b else 1
                                         for b in range(dim)])
                       for a, k in enumerate(ks))).ravel()
    pb = p.ravel()
    kf = 2 * np.pi / box
    edges = np.arange(0.5 * kf, kmag.max() + kf, kf)
    idx = np.digitize(kmag, edges) - 1
    keep = (idx >= 0) & (idx < len(edges) - 1)
    idx, kmag, pb = idx[keep], kmag[keep], pb[keep]
    n_modes = np.bincount(idx, minlength=len(edges) - 1)
    kmean = np.bincount(idx, weights=kmag, minlength=len(edges) - 1) / np.maximum(n_modes, 1)
    pmean = np.bincount(idx, weights=pb, minlength=len(edges) - 1) / np.maximum(n_modes, 1)
    m = n_modes > 0
    return {'k': kmean[m], 'P': pmean[m], 'n_modes': n_modes[m],
            'k_fund': float(kf), 'k_nyq': float(np.pi * ng[0] / box)}




def _delta2(k, P, dim):
    """无量纲带功率 Delta^2(k) = k^dim P(k) / (2 pi)^dim * S_(dim-1)。"""
    s = 4 * np.pi if dim == 3 else (2 * np.pi if dim == 2 else 2.0)
    return k ** dim * P / (2 * np.pi) ** dim * s




def _trapz(y, x):
    """梯形积分 (自己写: numpy 2.x 把 np.trapz 改名了, 不引版本分支)。"""
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))




def _xi_from_fft(delta, box, deconv=True):
    """
    由 FFT 自相关 (Wiener-Khinchin) 得两点相关 xi(r)。

    对全盒做各向同性平均有偏 (积分约束): xi_est(r) = xi_true(r) - (1/V) int xi dV。
    本函数减掉盒内的 xi 均值做积分约束修正, 并返回修正前的值以便对照。

    deconv: 参考侧是 CIC 赋值的粒子场, 要除以赋值窗; 模型侧是连续场, **不能**除
    —— 对连续场除一个 CIC 窗等于凭空注入一个错误的谱形。
    """
    shape = delta.shape
    ntot = int(np.prod(shape))
    fk = np.fft.fftn(delta)
    ks = _kgrids(shape, box)
    ps = np.abs(fk) ** 2
    if deconv:
        ps = ps / _cic_power_window(ks, box / shape[0])
    ac = np.real(np.fft.ifftn(ps)) / ntot

    # 周期的最近像距离 (每轴各自按自己的长度 reshape 后再相加, 否则广播成 1D)
    rmag = np.zeros(shape, dtype=np.float64)
    for a, n in enumerate(shape):
        i = np.arange(n)
        r1 = np.minimum(i, n - i) * (box / n)
        sh = [1] * len(shape)
        sh[a] = n
        rmag = rmag + (r1 ** 2).reshape(sh)
    rmag = np.sqrt(rmag)
    dx = box / shape[0]
    nb = max(int(min(shape) / 2) - 1, 2)
    edges = np.arange(0.0, nb * dx + dx, dx)
    idx = np.digitize(rmag.ravel(), edges) - 1
    keep = (idx >= 0) & (idx < len(edges) - 1)
    idx = idx[keep]
    cnt = np.bincount(idx, minlength=len(edges) - 1)
    got = np.bincount(idx, weights=ac.ravel()[keep], minlength=len(edges) - 1)
    rmean = (np.bincount(idx, weights=rmag.ravel()[keep], minlength=len(edges) - 1)
             / np.maximum(cnt, 1))
    xmean = got / np.maximum(cnt, 1)
    m = cnt > 0
    xi_raw = xmean[m]
    return {'r': rmean[m], 'xi_raw': xi_raw,
            'xi': xi_raw - float(np.mean(xi_raw)),
            'V': float(box ** len(shape))}




def _first_zero(r, y):
    """xi(r) 的第一个过零点 (线性内插)。没有过零则返回 nan。"""
    s = np.sign(y)
    for i in range(len(s) - 1):
        if s[i] > 0 and s[i + 1] <= 0:
            f = y[i] / (y[i] - y[i + 1])
            return float(r[i] + f * (r[i + 1] - r[i]))
    return float('nan')




def _mode_census(delta):
    """
    「这个场有几个独立自由度?」—— 把 |delta_k|^2 按模从大到小排, 数出承载
    50% / 90% 功率各需多少个**模**。

    为什么要按模而不是按**球壳**: 壳平均会把锁模藏起来。一个 36x36 的锁模格子
    只有几个壳有功率, 看起来"壳数不多但也不算少"; 而按模一数就露馅 ——
    683 个非零模里个位数承载了一半功率。壳数不等于自由度数。
    """
    fk = np.fft.rfftn(np.asarray(delta, dtype=np.float64)).copy()
    fk.ravel()[0] = 0.0                      # 去掉 DC (减均值后本应为 0)
    p = (np.abs(fk) ** 2).ravel()
    p = p[p > 0]
    if len(p) == 0:
        return 0, 0, 0
    cum = np.cumsum(np.sort(p)[::-1]) / p.sum()
    return (int(np.searchsorted(cum, 0.50) + 1),
            int(np.searchsorted(cum, 0.90) + 1), int(len(p)))




def _model_field(s6):
    """阶段六 v 场 -> 归一化涨落 delta = v/<v> - 1, 以及它的二维 P(k)/xi/空洞。"""
    v = np.asarray(s6['v'], dtype=np.float64)
    vb = float(v.mean())
    return v / vb - 1.0 if vb > 0 else v - v.mean()




def _char_scale(delta, box, deconv=True):
    """场的特征尺度: 谱峰处的波长 2 pi / k_peak。"""
    pk = _band_power(delta, box, 2, deconv=deconv)
    if len(pk['k']) == 0:
        return float('nan'), pk
    i = int(np.argmax(_delta2(pk['k'], pk['P'], 2)))
    return float(2 * np.pi / pk['k'][i]), pk



def _shape_residual(x_a, y_a, x_b, y_b, n_grid=16):
    """
    两条曲线在公共 x 区间上的对数形状残差, 各自归一到"在 ln x 上积分为 1"。
    归一化掉振幅之后比较的就只剩**形状**。区间无交集时返回 nan。
    """
    lo, hi = max(x_a.min(), x_b.min()), min(x_a.max(), x_b.max())
    if not (hi > lo) or len(x_a) < 2 or len(x_b) < 2:
        return float('nan'), np.zeros(0), np.zeros(0), np.zeros(0)
    g = np.exp(np.linspace(np.log(lo), np.log(hi), n_grid))
    la = np.interp(np.log(g), np.log(x_a), np.log(y_a))
    lb = np.interp(np.log(g), np.log(x_b), np.log(y_b))
    w = np.log(hi) - np.log(lo)
    la = la - _trapz(la, np.log(g)) / w
    lb = lb - _trapz(lb, np.log(g)) / w
    return float(np.sqrt(np.mean((la - lb) ** 2))), g, la, lb





CACHE_DIRNAME = '_v13_cache'


# v14·F3: 参考侧 Gray-Scott 自建种子 (由 spiral_v14_prepare.py 生成)。
# A 段参考侧若直接搬归档里的 bool 场, 实测那批 bool 场与归档浮点场对不上
# (一致率≈0.5, 见 tier3_structural_ladder 的 A 段诊断), 其管线不在归档中,
# 无法复现; 三个"种子"其实还同出一个硬编码 IC。所以参考侧换自建种子。
V14_CACHE_DIRNAME = '_v14_cache'


# v15·W2: A 段"同单位受控"参考侧。与 v14 侧的差别是**参考侧改用与模型侧逐字
# 相同的参数与单位**, 只变网格分辨率 —— 见 v15/spiral_v15_prepare.py 的文件头。
# 它是可选的: 缺失时 A 段的 A1/A2 照常运行, 只有新的受控小节报"跳过"。
V15_CACHE_DIRNAME = '_v15_cache'


V13_A_BASELINE = {'D_struct': 0.9331, 'scatter': 0.4544, 'ratio': 2.053}


PANEL_SEEDS = 3       # "参考侧实际场"面板只放前 3 个种子 (网格只有 4 列 x 1 行)


EPS_T = 0.05          # 1.1 的目标 (与已发布基线同一阈值, 没有放宽)


RC_T = 1e-2           # 1.2 的目标 (与已发布基线同一阈值)



# 一维时序周期检出的标定常数 —— 数值来自对零假设的实测 (见 _series_period)
PROM_MIN = 300.0      # 峰相对局部背景的最小突出度; 标定见 _series_period 注释


PROM_WIN = 40         # 算局部背景时峰两侧各取多少根谱线




def _model_field(s6):
    """阶段六 v 场 -> 归一化涨落 delta = v/<v> - 1, 以及它的二维 P(k)/xi/空洞。"""
    v = np.asarray(s6['v'], dtype=np.float64)
    vb = float(v.mean())
    return v / vb - 1.0 if vb > 0 else v - v.mean()
