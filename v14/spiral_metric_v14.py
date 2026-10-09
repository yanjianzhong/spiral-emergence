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
"""
指标层 —— 第一层用分布口径 + 第三层为「方法学展示」

为什么单独一个文件, 而不是改 spiral_metric.py?
    已发布的体检报告必须**保持可复现**。spiral_metric.py 被多个版本共用,
    在里面动一行, 所有历史数字的前提就变了。所以这里用 import 的方式复用
    它的注册表与全部维度无关的辅助函数 (功率谱/关联长度/形态学), 只**新增**
    本层的指标定义。那 12 项既有指标与 31 条守卫一个字节都没动。

第一层的两处改动 (都是"诊断驱动", 不是"为达标而调参"):
  1.1/1.2 从单点值 -> 多种子分布, 预算 600 -> 1500 步。
      依据见 _v13_diag_conv: eps_c 在 step=600 处仍陡降, 11.45% 是预算产物。
  新增 1.5 全程稳健性: 结论必须在**所有**轨迹上成立 (种子 x 学习率),
      不是在中位数上成立。这是防止"挑一条好看的曲线"的唯一硬约束。
  新增诊断 1.D: chi 扫描 —— 直接检验基线 note 里"主因是有限键维截断"这句断言。
      实测 chi=4 在充分优化下达标, 所以那句话是错的, 这里如实记录。

第三层 (从 3D CDM 场对场 -> 2D 斑图结构对结构), 三段梯子:
  A 段 同方程不同盒子: 阶段六 Gray-Scott (36^2) <-> CLIP Gray-Scott (100^2, 3 种子)。
       参考侧是**数值解**, 所以这一段不受成像与标定污染, 而且有 3 个种子 ——
       第一次有了**种子间散布**可以当噪声底 (场对场口径恰恰没有这个)。
       **但不是干净的"同方程不同盒子"**: 实测 f/k 落在 Pearson 相图的不同相区
       (模型 F=0.035, k=0.060 / 参考 f=0.01, k=0.042), 盒子内波长数还差一个
       量级。两个混杂因子使 D_struct **无法单独归因给盒子大小** —— 这是本段
       不设达标线的直接理由, 不是措辞上的退让。
  B 段 跨系统: CLIP Gray-Scott <-> Southampton BZ 时空图 / Reading 时序。
       交付物是"哪些描述子能迁移、哪些不能", 不是"模型复现了实验"。
  C 段 判别力: 跨系统距离 / 系统内散布。若这个比值 ~1, 那 B 段无论什么结果
       都是无信息的 —— 它把"0/4 达标"从"失败"变成"该判据本就分不开"。

**轴语义的边界 (贯穿 B 段)**: Southampton 的图是 space x time, 不是二维空间斑图。
两个轴的含义不同, 把它按二维场算描述子只是"用同一套尺子量", 不构成
"实验斑图与数值斑图形态一致"的证据。所有 B 段数字一律 passed=None。
"""
import os
import json

import networkx as nx
import scipy.sparse as sp
import numpy as np

from scipy import ndimage  
from scipy.sparse.linalg import eigsh

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
                  measured, baseline, target, passed, note=''):
    """
    记录一条指标。

    tier      1=内部自洽性  2=微观物理对标  3=宏观结构/Surrogate 对标
    name      中文指标名          symbol    符号或公式
    definition 定义 (一句话说清"这个数在量什么")
    method    计算方法 (用哪个函数/哪条公式算出来的)
    reference 参照物 —— 第1层是 v11 基线, 第2层是解析解, 第3层是数据来源
    measured  实测值      baseline 基线值      target 目标 (字符串, 便于写不等式)
    passed    True/False, 或 None 表示"只能诊断, 不给数值判定"
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
    }
    METRICS.append(rec)
    return rec


def record_guard(name, where, condition, ok, expect_pass=True, note=''):
    """
    记录一条守卫。守卫 = 一个**可证伪**的检查: 条件明确, 且可能失败。

    expect_pass=False 表示"设计上就应当失败"的诊断项 —— 例如 v11 阶段三-c
    的权重自由变体本来就不该秩 1 化。这类项**不计入通过率**:
    否则通过率可以靠塞进注定失败的项做低, 也可以靠删掉它们做高。
    一个诚实的通过率必须先把"该通过的"和"只想看看的"分开。
    """
    rec = {'name': name, 'where': where, 'condition': condition,
           'ok': bool(ok), 'expect_pass': bool(expect_pass), 'note': note}
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
    print("  体检报告 · 三层量化指标 (14)")
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

def entanglement_curve(psi, L):
    """S(n) = -Tr rho_A ln rho_A, A = 前 n 个连续站点 (n = 1..L//2)"""
    psi = np.asarray(psi).reshape(-1)
    psi = psi / np.linalg.norm(psi)
    out = []
    for n in range(1, L // 2 + 1):
        m = psi.reshape(2 ** n, 2 ** (L - n))
        s = np.linalg.svd(m, compute_uv=False)
        p = s ** 2
        p = p[p > 1e-15]
        out.append(float(-np.sum(p * np.log(p))))
    return np.array(out)

def fit_central_charge(psi, L):
    """
    周期 CFT 的 Calabrese-Cardy 公式:
        S(n) = (c/3) ln[ (L/pi) sin(pi n / L) ] + const
    对 [ln(...), 1] 做线性最小二乘, 斜率 -> c = 3 * slope。
    返回的 rms 是"这个公式是否成立"的残差, 不是 c 的误差棒。
    """
    ns = np.arange(1, L // 2 + 1)
    x = np.log((L / np.pi) * np.sin(np.pi * ns / L))
    y = entanglement_curve(psi, L)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    rms = float(np.sqrt(np.mean((y - A @ coef) ** 2)))
    return {'c': 3.0 * coef[0], 'const': coef[1], 'rms': rms,
            'x': x, 'S': y, 'ns': ns}


# ============================================================================
# v14 · 指标层 I/II: 内部自洽性 + 微观物理对标
# ============================================================================
#
# 每个函数算完就 record_metric(...) 落一条账, 并返回自己的明细 dict 供打印/JSON。
# 三条约定:
#   1. 目标都是本项目自设的**工程阈值**, 不是文献标准 —— 报告里如实标注。
#   2. **只报告不追目标**: 没达标时给的是差距诊断, 不是调参重算。
#   3. 参照物分两类: 第 1 层的参照是 v11 的实测基线 (单次运行, 无误差棒);
#      第 2 层的参照是**解析解** (c=1/2, eta=1/4, k*=(N-1)/|ln lam|, 精确基态谱)。


def _schmidt_spectrum(psi, L, tol=1e-12):
    """
    半链二分的**归一化 Schmidt 谱** p_k = s_k^2 / sum(s_k^2)。

    与 entanglement_curve 用同一个二分 (前 n 个连续站点), 区别是这里要整个谱,
    不是它的熵。tol 截断是熵/KL 计算的常规做法 (否则 ln 0 发散); 截断本身是一条
    诚实边界, 所以把被丢掉的分量数一并返回, 让调用方可以如实报告。
    """
    n = L // 2
    M = psi.reshape(2 ** n, 2 ** (L - n))
    s = np.linalg.svd(M, compute_uv=False)
    p_all = s ** 2
    keep = p_all > tol
    p = p_all[keep]
    return p / p.sum(), int((~keep).sum())


def metric_gap_closure(rows):
    """
    1.3 谱隙闭合残差 —— 逼近临界时相对谱隙 1-r 是否按 1/L 收拢?

    定义: r = p1/p0 是约化密度矩阵前两个本征值之比 (= 幂迭代收敛率)。
      临界时 r -> 1 (谱隙闭合)。记 g(L) = 1 - r(L), 拟合 g(L) = a/L + b。
      指标取该拟合的残差 RMS; 并单独检查 a 的符号 (a > 0 才是"在闭合")。
    诚实边界: 只有 3 个 L (8/12/16), 拟一条直线只剩 1 个自由度 ——
      残差 RMS 很小是**弱证据**, 不是强证据。这条写进 note。
    """
    Ls = np.array([r['L'] for r in rows], dtype=float)
    g = np.array([1.0 - r['gap'] for r in rows], dtype=float)
    a, b = np.polyfit(1.0 / Ls, g, 1)
    resid = float(np.sqrt(np.mean((g - (a / Ls + b)) ** 2)))
    closing = bool(a > 0)
    target = 5e-3
    ok = bool(resid < target and closing)
    record_metric(
        1, '谱隙闭合残差', 'RMS of  g(L) = a/L + b,  g = 1 - p1/p0',
        '相对谱隙 g = 1 - p1/p0 随系统尺寸 L 的闭合行为对 1/L 律的偏离。'
        '临界性的标志是 g -> 0; 这里量的是它收拢得规不规律。',
        'spiral_loop() 每圈实测的 gap = p1/p0, 对 L in {8,12,16} 拟合 a/L + b',
        '解析对照: 临界点处谱隙随 1/L 闭合',
        resid, None, f'< {target:.0e} 且 a > 0 (方向正确)', ok,
        note=(f'拟合得 a = {a:.4f} (a > 0 表示确实在闭合), b = {b:.4f}; '
              f'g 的六个实测值 = {np.round(g, 4).tolist()} '
              f'(spiral_loop 跑了 6 圈, 但 L 被 max_L=16 夹住, 末 3 圈与第 3 圈同值)。'
              f'**只有 3 个点拟一条直线, 仅 1 个自由度** —— 残差小是弱证据。'))
    return {'resid': resid, 'a': float(a), 'b': float(b),
            'g': g.tolist(), 'closing': closing, 'passed': ok}


def metric_mera_consistency(iso, r2_cone):
    """
    1.4 MERA 一致性 —— 张量网络自身的约束满足得怎么样?

    定义: 两个子量同时成立才算达标
      (a) 张量一致性 err = max(等距误差 ||W^dag W - I||, 酉性误差 ||U^dag U - I||)
      (b) 因果锥线性度 R^2_linear > R^2_log (体是体积律, 而 S(n) 只是面积律)
    为什么 (a) 要卡到 1e-12: 等距性是 MERA 的**定义性质**, 由 QR 构造保证,
      所以它应当到机器精度。v11 只把它 print 出来 ("6.66e-16") 而没有判定 ——
      这正是一个"有数字无判定"的洞。
    """
    err = float(max(iso['unitary_err'], iso['isometry_err']))
    lin, lg = float(r2_cone['linear']), float(r2_cone['log'])
    ok = bool(err < 1e-12 and lin > 0.95 and lin > lg)
    record_metric(
        1, 'MERA 一致性', 'max(||W^dag W - I||, ||U^dag U - I||)  与  R^2_linear > R^2_log',
        '张量网络自身必须满足的两条约束: (a) 解纠缠器酉、等距张量等距, 且到机器精度;'
        '(b) 因果锥张量数随 n 线性增长 (体是体积律), 优于对数增长。',
        f'mera_isometry_check(): {iso["n_unitary"]} 个酉 + {iso["n_isometry"]} 个等距; '
        f'mera_causal_cone(): 线性拟合 R^2={lin:.4f} vs 对数拟合 R^2={lg:.4f}',
        '解析对照: 等距性是 MERA 的定义性质, 不是拟合结果',
        err, 6.66e-16, '< 1e-12 且 R^2_linear > 0.95 且 线性 > 对数', ok,
        note=('v11 只把这两个误差 print 出来, 没有判定 —— v12 补成可证伪的 bool。'
              '这里能说的是"张量性质成立", **不是**"MERA 拟合得好"; '
              '拟合质量由 1.1 的中心荷误差负责。'))
    return {'err': err, 'r2_linear': lin, 'r2_log': lg, 'passed': ok}


def metric_chain_rmse(s3a, l2_phys=None):
    """
    2.1 因果链拟合 RMSE —— 理论收敛比 与 实测幂迭代率 的均方根偏差。

    定义: 对每组受控谱隙 (gap, eps), 理论预言收敛比 = |lam2/lam1| (由矩阵特征值
      直接算出), 实测收敛比 = 幂迭代末段步长比的中位数。RMSE = sqrt(mean((t-m)^2))。
    参照物是**解析量** |lam2/lam1|, 不是拟合值 —— 这是这条指标之所以叫"对标"的原因。
    """
    pairs = [(float(s['rate_theory']), float(s['rate_meas']))
             for s in s3a['gap_sweep']]
    good = [(t, m) for t, m in pairs if np.isfinite(t) and np.isfinite(m)]
    rmse = float(np.sqrt(np.mean([(t - m) ** 2 for t, m in good])))
    target = 1e-2
    ok = rmse < target
    phys_note = ''
    if l2_phys:
        phys_note = (f'物理算例 (谱隙由纠缠谱派生 gap={l2_phys["gap"]:.6f}): '
                     f'理论 {l2_phys["rate_theory"]:.6f} vs 实测 '
                     f'{l2_phys["rate_meas"]:.6f}, 偏差 '
                     f'{abs(l2_phys["rate_theory"] - l2_phys["rate_meas"]):.2e}。')
    record_metric(
        2, '因果链拟合 RMSE', 'RMSE(rate_theory, rate_meas)',
        '受控谱隙扫描下, 幂迭代的理论收敛比 |lam2/lam1| 与实测收敛比之间的均方根偏差。'
        '量的是 L2 这条因果链"理论预言能不能被数值复现"。',
        f'selfref_general_matrix() 的 gap_sweep: {len(good)} 组 (gap, eps) 各取一对 (理论, 实测)',
        '解析对照: |lam2/lam1| (矩阵特征值直接算出, 非拟合)',
        rmse, 7.612e-3, f'< {target:.0e}', ok,
        note=(phys_note + '基线 7.612e-3 由 v11 的 gap_sweep 四组 '
              '(0.9000/0.8955, 0.5000/0.4878, 0.2000/0.1926, 0.0500/0.0473) '
              '直接算出, 与本次实测同源 —— 这是复现, 不是改进。'
              '逐组偏差: 4.49e-3 / 1.22e-2 / 7.41e-3 / 2.74e-3, '
              '与各组步数 (260/45/22/12) 不对应, 故这里不给出偏差来源的解释。'))
    return {'rmse': rmse, 'pairs': good, 'passed': ok}


def metric_entropy_kl(psi_mera, gs, L):
    """
    2.2 纠缠熵 KL 散度 —— MERA 的 Schmidt 谱与精确谱差多远?

    定义: D_KL(p || q) = sum_k p_k ln(p_k / q_k), p = MERA 的归一化 Schmidt 谱,
      q = 精确基态的。另外报对称化的 Jensen-Shannon 散度 (它天然有限)。
    为什么用谱而不是只用熵: 熵是一个数, 谱是一个分布 —— 两个态可以有相同的熵
      却完全不同的谱。KL 对分布形状敏感, 这是它比"熵差多少"更强的地方。
    诚实边界: 两条谱都在 tol=1e-12 处截断过, 所以这里报的 KL 是**截断后的**,
      是真实 KL 的一个下界; 被丢掉的分量数一并报告。
    """
    p, n_drop_p = _schmidt_spectrum(psi_mera, L)
    q, n_drop_q = _schmidt_spectrum(gs, L)
    n = max(len(p), len(q))
    pp = np.zeros(n); pp[:len(p)] = p
    qq = np.zeros(n); qq[:len(q)] = q
    mask = (pp > 0) & (qq > 0)
    kl = float(np.sum(pp[mask] * np.log(pp[mask] / qq[mask])))
    m = 0.5 * (pp + qq)
    # 每一项只在**自己**非零处求和: 若按 mm=(m>0) 统一取支撑, pp 为 0 而 qq 非零的
    # 位置上会出现 0*log(0) = nan —— 而 JS 之所以要和 KL 并排报, 正是因为它在
    # 支撑不匹配时**仍然有限**。用 mm 会让这个性质凭空消失。
    mp, mq = pp > 0, qq > 0
    js = 0.5 * float(np.sum(pp[mp] * np.log(pp[mp] / m[mp]))) \
        + 0.5 * float(np.sum(qq[mq] * np.log(qq[mq] / m[mq])))
    target, target_js = 1e-2, 1e-3
    ok = bool(kl < target and js < target_js)
    record_metric(
        2, '纠缠熵 KL 散度', 'D_KL(p || q),  p,q = 归一化 Schmidt 谱',
        'MERA 态的 Schmidt 谱 p 相对精确基态谱 q 的 KL 散度; 另报对称化的 JS。'
        '比"熵差多少"更强: 熵相同而谱不同的两个态, 熵判据看不出来, KL 能。',
        f'半链二分 SVD, 两条谱均在 1e-12 处截断后重归一化 (L={L})',
        '解析对照: 精确对角化基态的 Schmidt 谱',
        kl, None, f'< {target:.0e} nats 且 JS < {target_js:.0e}', ok,
        note=(f'JS = {js:.3e}。谱在 1e-12 处截断: MERA 丢 {n_drop_p} 个分量, '
              f'精确态丢 {n_drop_q} 个 —— 所以这个 KL 是真实值的**下界**。'
              f'支撑长度: MERA {len(p)}, 精确 {len(q)}, 共同 {int(mask.sum())} '
              f'(KL 只在这个交集上求和)。**反直觉但属实**: 精确基态在 L=16 '
              f'时已是有效低秩 (最小的保留分量 s={np.sqrt(q[-1]):.2e}), 而 chi=4 的 '
              f'MERA 反而拖出一条更长的弱尾 —— 熵匹配不代表尾部匹配。'))
    return {'kl': kl, 'js': js, 'n_p': len(p), 'n_q': len(q),
            'n_common': int(mask.sum()), 'n_drop_p': n_drop_p,
            'n_drop_q': n_drop_q, 'passed': ok}



def metric_jordan_peak(transient):
    """
    2.4 Jordan 块收敛率 —— 退化矩阵的瞬态峰值位置能不能被解析预言?

    定义: A = lam*I + N_N (单个 N 阶 Jordan 块, |lam| < 1)。谱隙 |lam2/lam1| = 1,
      所以**谱分析预言"不收敛"**; 而 ||A^k||_2 会先暴涨再衰减。
      峰值位置的解析估计: k* = (N-1) / |ln lam|
        (推导: ||A^k|| 由 Toeplitz 矩阵最大元 C(k,N-1)|lam|^(k-N+1) 主导,
         对 k 求极值得 (N-1)/k + ln|lam| = 0)
      指标 = |k_peak_measured - k*| / k*。

    为什么这个指标有分量: 它是一条**真正有预言力的**解析结果 —— 不是"数值和理论一致"
      这种事后说法, 而是先算出 47.5 再去看数值落在哪里。实测 (N=6, lam=0.9):
      ||A^k||_2 峰值在 k=49, 状态范数峰值在 k=48。误差 1%~3%。
    """
    N = int(transient['N'])
    lam = float(transient['lam'])
    k_meas = int(transient['k_peak'])
    k_norm_meas = int(transient['k_norm_peak'])
    k_theory = float((N - 1) / abs(np.log(lam))) if 0 < lam < 1 else float('nan')
    rel = float(abs(k_meas - k_theory) / k_theory) if np.isfinite(k_theory) else float('nan')
    target = 0.05
    ok = bool(np.isfinite(rel) and rel < target)
    record_metric(
        2, 'Jordan 块收敛率', 'k* = (N-1)/|ln lam|  vs  实测 ||A^k||_2 峰值位置',
        f'{N} 阶单个 Jordan 块 (lam={lam}) 的幂次范数 ||A^k||_2 的峰值位置, '
        '与解析估计 k* = (N-1)/|ln lam| 的相对误差。'
        '它 одновременно证明谱隙判据在此失效 (|lam2/lam1| = 1 却仍有结构) '
        '且瞬态峰值位置是可预言的。',
        'selfref_general_matrix() 的 transient: 扫 k=1..300 取 ||A^k||_2 的 argmax',
        '解析对照: k* = (N-1)/|ln lam| (对 Toeplitz 主元求极值导出)',
        rel, None, f'相对误差 < {target:.0%}', ok,
        note=(f'实测: ||A^k||_2 峰值 k={k_meas} (解析 {k_theory:.2f}), '
              f'状态范数 ||A^k x0|| 峰值 k={k_norm_meas}。'
              f'谱半径 rho={transient["spectral_radius"]:.4f} < 1, '
              f'峰值放大 {transient["peak_ratio"]:.1f} 倍, '
              f'峰值处 ||A^k||_2/rho^k = {transient["peak_over_rho_k"]:.3g} —— '
              f'谱半径把范数低估了这么多倍, 这就是非正规矩阵的"假收敛"陷阱。'))
    return {'k_meas': k_meas, 'k_theory': k_theory, 'k_norm_meas': k_norm_meas,
            'rel': rel, 'passed': ok}

def metric_correlation_exponent(gs, L):
    """
    2.3 关联衰减指数 —— 临界 Ising 的关联函数指数 eta 对不对?

    定义: 临界 1+1 维 Ising 普适类里 spin 算符的标度维 Delta_sigma = 1/8,
      等时关联函数 |C(r)| = A * [(pi/L)/sin(pi r/L)]^(2*Delta_sigma), 即 eta = 2*Delta = 1/4。
      指标 = 拟合出的 2*Delta 相对 1/4 的相对误差。

    为什么**必须**用共形形式而不是朴素幂律 log|C| ~ -eta log r:
      本脚本实测过 —— 朴素幂律给 eta = 0.205 (18% 误差), 共形形式给 2*Delta = 0.2426
      (3% 误差)。有限环上的 sin 修正不可忽略, 朴素幂律是在用错误的函数形式做拟合。
      朴素幂律的斜率仍然并排打印出来, 作为"选错形式的代价"的对照, 不掩盖。
    """
    _, C, _ = boundary_correlation_graph(gs, L)
    r = np.arange(1, L // 2)
    y = np.abs(C[0, r])
    f = (np.pi / L) / np.sin(np.pi * r / L)   # CFT 的周期像, 随 r 单调下降
    two_delta = float(np.polyfit(np.log(f), np.log(y), 1)[0])   # |C| = A * f^(2*Delta)
    eta_naive = float(-np.polyfit(np.log(r), np.log(y), 1)[0])
    exact = 0.25
    rel = float(abs(two_delta - exact) / exact)
    target = 0.10
    ok = rel < target
    record_metric(
        2, '关联衰减指数', 'eta = 2*Delta_sigma,  |C(r)| = A*[(pi/L)sin(pi r/L)]^(2*Delta)',
        '临界 TF-Ising 基态的等时自旋关联函数 C(r) = <sigma^z_0 sigma^z_r> 的衰减指数,'
        '用共形不变性给出的周期形式拟合 (不是朴素幂律)。',
        f'boundary_correlation_graph(gs,{L}) 取 C[0,r], r=1..{L//2 - 1}, '
        f'对 ln|C| 与 ln[(pi/L)/sin(pi r/L)] 做线性拟合',
        '解析对照: Delta_sigma = 1/8 -> eta = 1/4 (2D Ising / c=1/2 普适类)',
        two_delta, exact, f'相对误差 < {target:.0%}', ok,
        note=(f'朴素幂律 log|C| ~ -eta log r 给 eta = {eta_naive:.4f}, '
              f'相对误差 {abs(eta_naive - exact) / exact:.1%} —— '
              f'远差于共形形式。这正说明有限环上不能用朴素幂律。'
              f'MERA 态 (非精确态) 的同量在下方"差距诊断"里给出。'))
    return {'two_delta': two_delta, 'eta_naive': eta_naive, 'exact': exact,
            'rel': rel, 'passed': ok}

def exact_ground_state(L, J=1.0, h=1.0):
    """稀疏 eigsh 求基态。L<=16 时 2^L 可控。返回 (E0, gs)"""
    H = tfi_periodic_sparse(L, J, h)
    E, V = eigsh(H, k=1, which='SA')
    gs = V[:, 0]
    return float(E[0]), gs / np.linalg.norm(gs)

def boundary_correlation_graph(psi, L, keep_per_node=3):
    """
    从 (MERA 或精确) 态的边界关联构造涌现几何:
        C_ij = <sigma^z_i sigma^z_j>,   d_ij = 1 - |C_ij|
    每个节点保留距离最近的 keep_per_node 条边 —— 稀疏化, 使图连通且规模可控。
    站点 0 = 最高有效位 (与 tfi_periodic_sparse 的约定一致)。
    """
    psi = np.asarray(psi).reshape(-1)
    psi = psi / np.linalg.norm(psi)
    idx = np.arange(2 ** L)
    bits = (idx[:, None] >> np.arange(L - 1, -1, -1)) & 1
    z = 1.0 - 2.0 * bits
    prob = psi ** 2
    C = np.einsum('s,si,sj->ij', prob, z, z)

    Dm = 1.0 - np.abs(C)
    np.fill_diagonal(Dm, 0.0)
    G = nx.Graph()
    G.add_nodes_from(range(L))
    for i in range(L):
        added = 0
        for j in np.argsort(Dm[i]):
            if j == i:
                continue
            G.add_edge(i, int(j), weight=float(Dm[i, j]))
            added += 1
            if added >= keep_per_node:
                break
    return G, C, Dm


def tfi_periodic_sparse(L, J=1.0, h=1.0):
    """
    周期边界 (PBC) 横场 Ising 稀疏哈密顿量:
        H = -J Sum_i sigma^z_i sigma^z_{i+1} - h Sum_i sigma^x_i

    位序约定 (关键, 与 quimb MERA 的输出指标 k0..k_{L-1} 对齐):
        站点 i 对应二进制位 (L-1-i), 即 k0 是最高有效位。这样 quimb 收缩
        出的稠密矢量 reshape(-1) 后, 前 n 个分量恰好对应前 n 个连续站点,
        纠缠熵 S(n) 的块划分才与物理一致。
        (v9 的 tfi_hamiltonian 用的是相反位序; 二者由环的反射对称性联系,
         本征值完全相同, 已实测一致到 7e-15。)
    """
    sx = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=float))
    sz = sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=float))

    Hx = None
    for i in range(L):
        b = L - 1 - i
        m = sp.kron(sp.eye(2 ** b, format='csr'),
                    sp.kron(sx, sp.eye(2 ** (L - 1 - b), format='csr')),
                    format='csr')
        Hx = m if Hx is None else Hx + m

    H = (-h) * Hx
    for (a, b) in [(i, (i + 1) % L) for i in range(L)]:
        mat = 1
        for j in range(L):
            mat = sp.kron(mat, sz if j in (a, b) else sp.eye(2, format='csr'),
                          format='csr')
        H = H + (-J) * mat
    return ((H + H.T) / 2).tocsr()



CACHE_DIRNAME = '_v13_cache'
# v14·F3: 参考侧 Gray-Scott 自建种子 (由 spiral_v14_prepare.py 生成)。
# A 段参考侧若直接搬归档里的 bool 场, 实测那批 bool 场与归档浮点场对不上
# (一致率≈0.5, 见 tier3_structural_ladder 的 A 段诊断), 其管线不在归档中,
# 无法复现; 三个"种子"其实还同出一个硬编码 IC。所以参考侧换自建种子。
V14_CACHE_DIRNAME = '_v14_cache'
V13_A_BASELINE = {'D_struct': 0.9331, 'scatter': 0.4544, 'ratio': 2.053}
PANEL_SEEDS = 3       # "参考侧实际场"面板只放前 3 个种子 (网格只有 4 列 x 1 行)
EPS_T = 0.05          # 1.1 的目标 (与已发布基线同一阈值, 没有放宽)
RC_T = 1e-2           # 1.2 的目标 (与已发布基线同一阈值)

# 一维时序周期检出的标定常数 —— 数值来自对零假设的实测 (见 _series_period)
PROM_MIN = 300.0      # 峰相对局部背景的最小突出度; 标定见 _series_period 注释
PROM_WIN = 40         # 算局部背景时峰两侧各取多少根谱线


def _char_scale_checked(delta, box, deconv=False):
    """
    _char_scale 的输入校验包装 —— 返回 (lam, pk, valid)。

    为什么需要: _char_scale 内部是 `lam = 2*pi / k[argmax(Delta^2(k))]`, 那个
    argmax **不要求峰有内部性**, 也不检查输入场是否退化。两种退化输入会给出
    看起来正常的数字:
      * **空数组** —— _band_power 里的 np.fft.rfftn 直接抛 ValueError, 崩在
        它自己的 len==0 守卫之前, 报错信息与"场为空"这件事没有可读的关联;
      * **平坦场** —— 频谱恒为零, argmax 退化为第 0 个 bin, 于是返回
        lam = box, 一个**有限且看起来合理**的伪值。调用方原本的
        `isfinite(lam) and lam > 0` 守卫接不住它 —— 那不是守卫失效, 是那个
        守卫本来就不是为这件事写的。

    本包装只加两件事: 空场给出可读的报错, 退化场把 valid 标成 False。
    **取峰规则一个字节都没改**。换掉 argmax 需要一个有物理依据的替代规则;
    已试过的替换方案 (在 lambda >= k 的壳里取 P(k) 最大者) 在合成条纹上验证
    通过、在真实场上退化到盒子尺度的低频团块, 稳定性不达标。拿一个未验证的
    估计量换掉一个已知有缺陷的估计量, 风险大于收益, 所以缺口如实保留。

    诚实边界: valid=False 只说"这一次的输入不满足定波长的前提", 不说 lam 一定
    是错的; valid=True 也不说 lam 有独立佐证 (这一条由 A 段的诊断项回答)。
    """
    d = np.asarray(delta, dtype=np.float64)
    if d.size == 0:
        raise ValueError('_char_scale_checked: 输入场为空, 无法定特征波长; '
                         '(原路径会在 _band_power 的 rfftn 处抛 ValueError)')
    if not np.isfinite(d).all():
        return float('nan'), None, False
    lam, pk = _char_scale(delta, box, deconv=deconv)
    # 平坦性判据必须**相对**, 不能用 `std() > 0`。常数场取浮点非整数时 (如 3.7),
    # 1296 个元素的成对求和均值并不位等于 3.7, 偏差在 1 ulp 量级 —— std 于是算
    # 出 ~1e-16 而不是 0, 恰好从 `> 0` 里漏过去。这与上面批评原守卫的是同一类
    # 毛病: 判据的量级没跟被判对象对齐。改用"中心化后是否还有相对波动"。
    dc = d - d.mean()
    scale = max(float(np.abs(d).max()), 1e-300)
    valid = bool(float(np.abs(dc).max()) > 1e-12 * scale
                 and np.isfinite(lam) and lam > 0)
    return lam, pk, valid


def metrics_for_json(obj):
    """
    递归转成 json 可序列化的原生类型; 同时**丢掉所有下划线开头的键**。

    为什么要丢: 第三层会把作图用的紧凑数组挂在 `_plot` 下 (时空图的抽样像、
    功率谱曲线、逐帧亮度序列)。那些是几百 KB 的数组, 进 json 既撑大文件又
    没有审计价值 (它们是已经算出来的指标的原始素材, 不是指标本身)。
    命名约定 `_` 前缀 = 内存内传递, 不落盘。
    """
    if isinstance(obj, dict):
        return {str(k): metrics_for_json(v) for k, v in obj.items()
                if not str(k).startswith('_')}
    if isinstance(obj, (list, tuple)):
        return [metrics_for_json(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ===========================================================================
# 第一层 (分布口径)
# ===========================================================================
def metric_central_charge_error_seeds(runs, chi, L, baseline=0.1145):
    """
    1.1 中心荷误差 —— 从"一条轨迹的单点值"改成"多条轨迹的分布"

    11.45% 不是 chi=4 的表示极限, 而是一条**未收敛轨迹的瞬时值**:
    同预算下换种子/换学习率, 这个数在 1%~60% 之间跳 (见 _v13_diag_conv)。
    所以单点报法缺的不是精度, 是**方差** —— 一个没有误差棒的单点值
    无法区分"表示能力不够"和"这次没跑好"。

    判据仍然卡在最坏情况 (max < 5%), 不是中位数 —— 中位数达标但某条轨迹
    60% 的话, "chi=4 能做到 5%" 这句话就不成立。
    """
    e = np.array([r['eps_c'] for r in runs], dtype=float)
    med, lo, hi = float(np.median(e)), float(e.min()), float(e.max())
    ok = bool(hi < EPS_T)
    lrs = sorted({r['lr'] for r in runs})
    record_metric(
        1, '中心荷误差', 'eps_c = |c_MERA - c_exact| / c_exact',
        'MERA 态按 CFT 形式拟合出的中心荷 c, 相对同 L 精确值的偏差。'
        '报多条轨迹的分布 (中位数与全距), 判据卡最大值。',
        f'fit_central_charge(MERA 态, L={L}) vs 精确基态同 L 拟合; '
        f'chi={chi}, {len(runs)} 条轨迹 (lr={" / ".join(str(x) for x in lrs)}), '
        f'每条取 overlap 最高检查点',
        '解析对照: c=1/2 的 CFT (Ising 普适类)',
        {'median': med, 'min': lo, 'max': hi, 'all': e.tolist()},
        baseline, f'所有轨迹 < {EPS_T:.0%}', ok,
        note=(f'基线 {baseline:.4f} 是单条 600 步轨迹的瞬时值, 无误差棒。'
              f'本层实测 {len(runs)} 条轨迹全距 [{lo:.2%}, {hi:.2%}] —— '
              f'那个基线落在本分布的右尾, 说明它是**未收敛**而不是**表示极限**。'
              f'判据用 max 而非 median: 只要有一条轨迹做不到, "chi={chi} 能做到"'
              f'就不成立。目标 5% 未放宽。'))
    return {'median': med, 'min': lo, 'max': hi, 'passed': ok}


def metric_conformal_invariance_seeds(runs, rc_exact):
    """
    1.2 共形不变性 —— 同样改成分布, 判据卡最大。

    精确基态同量 (rc_exact) 是有限尺寸造成的残差地板; 它不随优化预算变,
    所以"MERA 相对地板高多少倍"这个说法依然成立, 依然写进 note。
    """
    r = np.array([x['R_conf'] for x in runs], dtype=float)
    med, lo, hi = float(np.median(r)), float(r.min()), float(r.max())
    ok = bool(hi < RC_T)
    record_metric(
        1, '共形不变性', 'R_conf = rms(CFT拟合) / mean(S(n))',
        'S(n) 对 CFT 形式 S(n) = (c/3)ln[(L/pi)sin(pi n/L)] + const 的归一化'
        '拟合残差。量的是"形式对不对", 与"c 的数值准不准"是两件事。',
        f'fit_central_charge(psi, L)["rms"] / mean(["S"]); {len(runs)} 条轨迹, '
        f'同 L 下同时算 MERA 与精确基态',
        '解析对照: 共形场论的 Cardy-Calabrese 公式本身',
        {'median': med, 'min': lo, 'max': hi, 'all': r.tolist()},
        None, f'所有轨迹 < {RC_T:.0%}', ok,
        note=(f'精确基态同量 = {rc_exact:.3e}, 是有限尺寸造成的残差地板; '
              f'实测全距 [{lo:.3e}, {hi:.3e}]。'
              f'基线未达标值 0.0102 与本分布的右尾同量级, 同样是预算问题而非形式问题。'
              f'目标 1e-2 未放宽。'))
    return {'median': med, 'min': lo, 'max': hi, 'R_conf_exact': rc_exact,
            'passed': ok}


def metric_design_robustness(runs, chi, L):
    """
    1.5 全程稳健性 —— 结论必须在**每一条**轨迹上同时成立

    这是本层唯一新增的达标项, 它防的是一个具体的作弊路径:
    多跑几个种子, 然后把中位数报出来当结论。若 3 条轨迹里 2 条达标、1 条
    是 60%, 中位数可能仍然"达标", 但"chi=4 能复现 c=1/2"这句话是假的。

    所以判据写成: 每一条轨迹 (种子 x 学习率) 都要达标。这不是新目标,
    就是把 1.1 与 1.2 的阈值同时施加到每一条轨迹上。
    """
    eps = np.array([r['eps_c'] for r in runs], dtype=float)
    rc = np.array([r['R_conf'] for r in runs], dtype=float)
    i_worst = int(np.argmax(eps / EPS_T + rc / RC_T))
    ok = bool(eps.max() < EPS_T and rc.max() < RC_T)
    record_metric(
        1, '全程稳健性', f'max over {len(runs)} 条轨迹 (eps_c, R_conf) < ({EPS_T:.0%}, {RC_T:.0%})',
        '1.1 与 1.2 的阈值同时施加到每一条独立轨迹 (不同随机种子 x 不同学习率), '
        '取最坏情况。防止"多跑几个种子然后报中位数"把失败轨迹平均掉。',
        f'chi={chi}, L={L}, 各轨迹独立初始化 (种子) 与独立优化器 (学习率)',
        '无外部参照 —— 这是自设的稳健性要求, 不是文献标准',
        {'max_eps_c': float(eps.max()), 'max_R_conf': float(rc.max()),
         'n_runs': len(runs),
         'worst_run': {'lr': runs[i_worst]['lr'], 'seed': runs[i_worst]['seed'],
                       'eps_c': runs[i_worst]['eps_c'],
                       'R_conf': runs[i_worst]['R_conf']}},
        None, f'max eps_c < {EPS_T:.0%} 且 max R_conf < {RC_T:.0%}', ok,
        note=(f'最坏轨迹: lr={runs[i_worst]["lr"]}, seed={runs[i_worst]["seed"]}, '
              f'eps_c={eps[i_worst]:.4%}, R_conf={rc[i_worst]:.3e}。'
              f'这一项比 1.1/1.2 严格 —— 它不可能在 1.1/1.2 失败时通过。'))
    return {'max_eps_c': float(eps.max()), 'max_R_conf': float(rc.max()),
            'n_runs': len(runs), 'passed': ok}


def metric_budget_diagnostic(curve, steps, chi):
    """
    1.D 预算漂移 (诊断项, passed=None) —— 报出"还没收敛多少", 不假装已收敛

    诚实边界必须是可读的数字, 不是一句"大致收敛了"。这里给出
    eps_c(steps) 相对 eps_c(steps/2) 的相对变化。若它明显是负的,
    说明报出的 eps_c 是**上界**而不是最优值。

    为什么这反而是好事: 一个达标的上界比一个达标的最优值更强。
    "chi=4 最终能做到 <5%" 需要最优值达标; 而"chi=4 在 1500 步时
    已经 <5%, 且还在下降"直接堵死"你只是没跑够"这条反驳。
    """
    cs = {int(s): (e, r) for (s, e, r, _o) in curve}
    keys = sorted(cs)
    near = lambda target: min(keys, key=lambda s: abs(s - target))
    k_h, k_f = near(int(steps // 2)), near(int(steps))
    e_h, r_h = cs[k_h]
    e_f, r_f = cs[k_f]
    drift = float((e_f - e_h) / e_h) if e_h > 0 else float('nan')
    record_metric(
        1, '预算漂移', 'drift = (eps_c(N) - eps_c(N/2)) / eps_c(N/2)',
        '训练预算翻倍时中心荷误差的相对变化。负值 = 还在下降, 说明报出的值'
        '是上界; 接近 0 = 已进平台, 说明报出的值就是该 chi 的最优。'
        '这是一个诊断项, 不给通过判定 —— 它的作用是把"收敛了多少"变成可读数字。',
        f'训练过程中每 100 步记录一次; 取 step={k_h} 与 step={k_f} 两点',
        '无外部参照 —— 这是对本次运行自身的收敛性描述',
        {'drift': drift, 'eps_c_half': float(e_h), 'eps_c_full': float(e_f),
         'step_half': k_h, 'step_full': k_f,
         'R_conf_half': float(r_h), 'R_conf_full': float(r_f)},
        None, '仅诊断, 不设阈值', None,
        note=(f'chi={chi}, N={steps}: eps_c 从 {e_h:.3%} (step {k_h}) 到 '
              f'{e_f:.3%} (step {k_f}), 相对变化 {drift:+.1%}。'
              f'{"仍在下降, 故报出值是上界 (对达标有利, 不是不利)" if drift < -0.05 else "已接近平台"}; '
              f'但**不能**声称已收敛到该 chi 的最优值。'))
    return {'drift': drift, 'eps_c_half': float(e_h), 'eps_c_full': float(e_f)}


def metric_chi_bottleneck(chi_sweep, chi_used, floor_chi, baseline=0.1145):
    """
    1.D2 chi 瓶颈诊断 (诊断项, passed=None) —— 基线 note 那句断言到底对不对

    基线 note 写"残差的主因是有限键维截断 (chi=4)"。这是一个**经验断言**,
    但它从没被检验过 —— 因为没有第二个 chi 的结果可比。

    【判定规则为什么是最小充分性, 而不是"实测 vs 参照线的比值"】
    第一版用的是 ratio = eps_c(chi) / Schmidt 截断参照, ratio < 2 就判"表示能力"。
    这个规则在真实数据上直接给反了: 实测 eps_c(chi=4) = 3.03%, 而参照线 = 5.93%,
    ratio = 0.51 < 2 -> 判成"表示能力受限"。但实测值**低于**参照线, 恰恰说明
    参照线不是有效约束 (逐切割 Schmidt 截断本就不是严格下界, 见下), 因此它
    什么也证明不了; 拿它做判据是把一个无效参照当成了标尺。

    改用**受控比较**, 只看同预算下换 chi 的效果:
      chi_gain   = eps_c(chi=4) / eps_c(chi=8)   同种子同步数同 lr, 只有 chi 变
      optim_gain = baseline(chi=4, 600步) / eps_c(chi=4, 1500步)  同 chi, 只有预算变
    再叠加一条不需要任何比较的事实: **chi=4 在同预算下已经达标**。
    一个已经达标的设置不可能是当前瓶颈 —— 这条比任何比值都硬。

    诚实边界 (两个增益并不对称, 必须写明):
      - chi_gain 是严格受控的 (单变量)。
      - optim_gain 把"更多步数 + 最佳检查点选择 + 种子"捆在一起, 是合并数字,
        不能单独归因给步数; 它和 chi_gain 的比较只用于判断**量级**谁更大。
      - 参照线 (逐切割保留前 chi 个精确 Schmidt 分量) **不是严格下界**: 同一个态
        不可能在所有切割点同时取到该 rank, 且 rank-chi 态的熵可在 [0, ln chi] 内
        任意取值。它只作为"表示能力的大致量级"附列, 不参与判定。
    """
    chis = sorted(chi_sweep)
    measured = {c: chi_sweep[c]['eps_c'] for c in chis}
    ctrl = next((c for c in chis if c > chi_used), None)
    chi_gain = (float(measured[chi_used] / measured[ctrl])
                if ctrl is not None and measured[ctrl] > 0 else float('nan'))
    optim_gain = float(baseline / measured[chi_used]) if measured[chi_used] > 0 \
        else float('nan')
    passes = bool(measured[chi_used] < EPS_T)

    if passes:
        verdict = (f'chi 不是瓶颈: chi={chi_used} 同预算下已达标 '
                   f'({measured[chi_used]:.2%} < {EPS_T:.0%})')
    elif chi_gain > optim_gain:
        verdict = '表示能力是主因 (换 chi 的增益大于换预算)'
    else:
        verdict = '预算/优化是主因 (换预算的增益大于换 chi)'

    monotone = all(measured[chis[i]] >= measured[chis[i + 1]]
                   for i in range(len(chis) - 1))
    ratio = float(measured[chi_used] / floor_chi) if floor_chi > 0 else float('nan')
    record_metric(
        1, 'chi 瓶颈归因', '受控比较: 换 chi 的增益 vs 换预算的增益',
        '判定基线 note 的"主因是有限键维截断"这句断言是否成立。'
        '做法: (a) 同种子同步数只改 chi, 量出 chi_gain; '
        '(b) 同 chi 只改预算, 量出 optim_gain; 两者比量级。'
        '另加一条无需比较的硬事实: chi=4 同预算下是否已达标。',
        f'chi in {chis}; 控制组 chi={ctrl}; 每条轨迹取最佳检查点 (按 overlap 选)',
        f'单条 600 步轨迹的 eps_c = {baseline:.4f} (无误差棒) 作为预算对照',
        {'measured': measured, 'chi_gain': chi_gain, 'optim_gain': optim_gain,
         'chi_used': chi_used, 'chi_ctrl': ctrl, 'passes_at_chi_used': passes,
         'monotone_in_chi': monotone,
         'floor_chi_used': floor_chi, 'ratio_over_floor_supporting_only': ratio,
         'verdict': verdict},
        None, '仅诊断, 不设阈值', None,
        note=(f'同种子同步数, chi {chi_used}->{ctrl}: eps_c '
              f'{measured[chi_used]:.3%} -> {measured[ctrl]:.3%}, '
              f'chi_gain = {chi_gain:.2f}x。同 chi={chi_used}, 预算 '
              f'600->1500 步: {baseline:.2%} -> {measured[chi_used]:.3%}, '
              f'optim_gain = {optim_gain:.2f}x。'
              f'chi 单调性: {"成立" if monotone else "不成立"}。'
              f'**结论**: {verdict}。提 chi 只能再降 {chi_gain:.2f}x, '
              f'而基线的 {baseline:.2%} 与本层的 {measured[chi_used]:.3%} 差 '
              f'{optim_gain:.2f}x。注: optim_gain 把步数/检查点选择/种子捆在一起, '
              f'是合并数字; chi_gain 才是单变量受控量。'
              f'Schmidt 截断参照 {floor_chi:.3%} 仅作量级附列, 非严格下界, '
              f'不参与判定。'))
    return {'chi_gain': chi_gain, 'optim_gain': optim_gain, 'verdict': verdict,
            'monotone': monotone, 'measured': measured}


# ===========================================================================
# 第三层 · 结构对结构
# ===========================================================================
def _late_frame(mod, t=None):
    """(T, N, N) 序列取最后一帧 (最接近稳态斑图), 也支持已经是 (N, N) 的输入。"""
    a = np.asarray(mod)
    if a.ndim == 3:
        a = a[-1 if t is None else t]
    return a


def _binarize_median(a):
    """与 spiral_v13_prepare.py 完全相同的二值化规则 (中位数阈值)。

    两侧用同一个规则是必须的: 若模型侧用中位数而参考侧用固定阈值,
    测到的差异里就混进了"阈值选择"这个与物理无关的自由度。
    """
    a = np.asarray(a, dtype=np.float64)
    return a > np.median(a)


def _ac_len(b):
    """场的自相关长度 (格): 径向平均自相关首次过零的滞后。

    这是**与谱峰无关**的尺度估计 —— 正因为 _char_scale 靠 argmax, 它会被
    二值场的高频噪声壳骗到 (归档 bool 场就是例子, 见 A 段诊断项
    A-归档参考侧管线不可复现)。自相关是零阶量, 不挑壳, 所以可以拿来交叉核对
    lam_pk 是不是伪影。
    """
    z = np.asarray(b, dtype=np.float64)
    z = z - z.mean()
    f = np.fft.fft2(z)
    ac = np.fft.fftshift(np.fft.ifft2(f * np.conj(f)).real)
    c = ac.shape[0] // 2
    prof = ac[c, c:c + max(ac.shape[0] // 2, 1)]
    if prof.size == 0 or prof[0] <= 0:
        return float('nan')
    prof = prof / prof[0]
    neg = np.flatnonzero(prof < 0)
    return float(neg[0]) if neg.size else float(prof.size)


def _ds(b, k=4):
    """块平均降采样, 只用于作图 (把场画成缩略图), 不参与任何量化。"""
    a = np.asarray(b, dtype=np.float64)
    n = a.shape[0] // k
    if n < 1:
        return a
    return a[:n * k, :n * k].reshape(n, k, n, k).mean(axis=(1, 3))


def structural_descriptors(b, lam=None):
    """
    一个二维周期二值场的 6 个**无量纲**结构描述子。

    为什么全部无量纲: 模型侧盒子 36, 参考侧 100 (CLIP) 或 450x4517
    (Southampton 时空图), 尺度和分辨率都不同。任何带量纲的量 (边界总长、
    关联长度绝对值) 直接比就是在比盒子大小。所以一律除以该场自己的特征波长
    lambda_pk, 或者除以总模数 —— 剩下的才是"形状"。

    六个量:
      phi     占空比              (面积率)
      v1_lam  边界长度 x lam       (界面密度)
      v2_lam2 Euler 示性数 x lam^2 (连通域/空洞的净数)
      kpk_k1  谱峰所在模 / 基频    (斑图的锁模位置)
      w50     承载 50% 功率所需模数 / 总非零模数 (自由度集中度)
      xi_lam  关联长度 / lam
    """
    b = np.asarray(b, dtype=bool)
    box = float(max(b.shape))
    z = b.astype(np.float64)
    phi = float(z.mean())

    # 特征波长: 用连续化后的场 (z - <z>) 的谱峰, 与 _char_scale 同一约定。
    # 走校验包装: 平坦场在那里被标成 invalid (原路径会静默返回 lam = box)
    delta = z - z.mean()
    lam_pk, pk, pk_valid = _char_scale_checked(delta, box, deconv=False)
    if not pk_valid:
        lam_pk = float(box)
    if lam is None:
        lam = lam_pk

    # Minkowski V1/V2 在唯一的二值阈值上 (二值场没有别的阈值可选)
    dh = np.abs(z - np.roll(z, -1, 1)).sum()
    dv = np.abs(z - np.roll(z, -1, 0)).sum()
    v1 = float((dh + dv) / float(b.size))                   # 单位: 1/格
    p00 = b.astype(np.int8)
    p01 = np.roll(b, -1, 1).astype(np.int8)
    p10 = np.roll(b, -1, 0).astype(np.int8)
    p11 = np.roll(np.roll(b, -1, 0), -1, 1).astype(np.int8)
    s = p00 + p01 + p10 + p11
    v2 = 0.25 * float(np.sum((s == 1).astype(np.float64)
                             - (s == 3).astype(np.float64))) / float(b.size)

    w50, _w90, n_modes = _mode_census(delta)
    xi = _xi_from_fft(delta, box, deconv=False)
    xi0 = _first_zero(xi['r'], xi['xi'])
    if not np.isfinite(xi0) or xi0 <= 0:
        # 无过零点时退到 xi(r) 的积分尺度, 仍是尺度无关的形式
        r, y = xi['r'], xi['xi']
        xi0 = (float(np.trapezoid(np.clip(y, 0.0, None), r) / max(y[0], 1e-12))
               if len(r) > 2 and y[0] > 0 else float(box))

    return {'phi': phi, 'v1_lam': float(v1 * lam), 'v2_lam2': float(v2 * lam ** 2),
            'kpk_k1': float(box / lam) if lam > 0 else float('nan'),
            'w50': float(w50) / max(n_modes, 1), 'xi_lam': float(xi0 / lam),
            'lam_pk': float(lam), 'w50_n': int(w50), 'n_modes': int(n_modes)}


_KEYS = ('phi', 'v1_lam', 'v2_lam2', 'kpk_k1', 'w50', 'xi_lam')


def descriptor_distance(da, db):
    """
    两个描述子向量之间的距离: 逐项 d_i = |a-b| / sqrt(a^2+b^2), 取 RMS。

    为什么用这个归一化 (与 spiral_metric._mf_distance 同一约定): 它对称、
    有界于 sqrt(2), 且分母恒非负 —— V2 (Euler 示性数密度) 会在零点附近穿过,
    用 "|a-b|/|a|" 会爆出几百上千的假距离。它对"两个量都接近 0"的情形
    自动给出接近 0 的差异, 这正是我们想要的 (都接近 0 就是都接近 0)。
    """
    d = []
    for k in _KEYS:
        a, b = float(da[k]), float(db[k])
        den = float(np.hypot(a, b))
        d.append(abs(a - b) / den if den > 0 else 0.0)
    d = np.array(d, dtype=float)
    return float(np.sqrt(np.mean(d ** 2))), dict(zip(_KEYS, d.tolist()))


def _norm_pk_curve(b, n_bins=24):
    """
    归一化功率谱形状: 横轴 k/k_pk, 纵轴 Delta^2 归一到 ln k 上积分为 1。
    这样比较的只剩**形状**, 与盒子大小、分辨率、振幅都无关。
    """
    box = float(max(b.shape))
    delta = np.asarray(b, dtype=np.float64)
    delta = delta - delta.mean()
    pk = _band_power(delta, box, 2, deconv=False)
    if len(pk['k']) < 3:
        return np.zeros(0), np.zeros(0)
    k, P = pk['k'], pk['P']
    k_pk = k[int(np.argmax(_delta2(k, P, 2)))]
    if not np.isfinite(k_pk) or k_pk <= 0:
        return np.zeros(0), np.zeros(0)
    x = k / k_pk
    y = np.clip(_delta2(k, P, 2), 1e-30, None)
    lx = np.log(x)
    w = lx.max() - lx.min()
    if w <= 0:
        return np.zeros(0), np.zeros(0)
    return x, y / (np.trapezoid(y, lx) / w)


def _peak_period_stats(peaks, sec_per_px):
    """
    Southampton 波峰标记 (时刻, 幅值) -> 振荡周期的统计量。

    这是 B 段里**唯一不依赖轴语义类比**的量: 波峰时刻是一维时序, 与
    "把时空图当二维斑图"无关, 也不经过 .tif 图版渲染 (它是原始作者从
    数据里提出的数值列)。返回相邻峰间隔的中位数、变异系数 (CV) 与秒。

    sec_per_px 必须由源文档标定给出: 0_Process_steps.txt 写
    "15 pixels = 2.5 seconds", 即 1/6 s per px。这里的 X 列单位是**像素**
    (取值直到 25297, 与时空图宽度 27100 同量级), 不是帧 —— 第一版把
    这个单位弄错过两次, 所以参数名里直接把单位写死。
    """
    t = np.sort(np.asarray(peaks, dtype=float)[:, 0])
    if len(t) < 4:
        return None
    dt = np.diff(t)
    dt = dt[dt > 0]
    if len(dt) < 3:
        return None
    med = float(np.median(dt))
    return {'n_peaks': int(len(t)), 'dt_median_px': med,
            'dt_cv': float(np.std(dt) / np.mean(dt)),
            'period_s': med * sec_per_px,
            'span_px': float(t.max() - t.min())}


def _series_lag_agreement(series, period_px):
    """
    独立交叉检验: 波峰间隔给出的周期 T, 在**原始强度序列**上对得上吗?

    为什么会需要这个: 波峰标记和强度序列是两套**独立**的提取 —— 前者是原始
    作者在时空图上标的峰位, 后者是在固定空间位置上量的强度随时间的变化。
    如果两者都对, 强度序列在滞后 ≈ T 处的自相关应当出现极大 (隔一个周期,
    波形回到自己), 且明显高于滞后 T/2 处 (反相, 应当低)。

    它同时解释了"为什么谱方法反而检不出周期": 谱峰要求波形**严格周期**,
    而这里波峰间隔的变异系数 CV 从 0.27 到 0.90 不等 —— CV 越大, 谱线被
    自身的不规则性抹得越平。所以三个量 (CV / 自相关滞后峰 / 谱突出度)
    是同一件事的三个侧面, 它们的**一致性**才是可以交付的结论。

    返回: ac (各滞后的自相关系数) / ratio (ac_T / ac_{T/2}) /
          是否为严格滞后峰 -> 三档判定。
    """
    s = np.asarray(series, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] < 2 or s.shape[0] < 16:
        return None
    x, y = s[:, 0], s[:, 1]
    lag = int(round(float(period_px)))
    if lag < 2:
        return None
    m = x > 3.0 * lag                      # 跳过混合之前的那一段
    if int(m.sum()) < 12 * lag:
        return None
    yy = y[m]
    idx = np.arange(len(yy))
    yy = yy - np.polyval(np.polyfit(idx, yy, 1), idx)

    def ac(L):
        if L <= 0 or L >= len(yy) - 2:
            return float('nan')
        a, b = yy[:-L], yy[L:]
        if a.std() <= 0 or b.std() <= 0:
            return float('nan')
        return float(np.corrcoef(a, b)[0, 1])

    lags = {'T/2': max(1, lag // 2), 'T': lag, '1.5T': int(round(1.5 * lag)),
            '2T': 2 * lag}
    acs = {k: ac(v) for k, v in lags.items()}
    ac_T = acs['T']
    ac_half = acs['T/2']
    if not np.isfinite(ac_T) or not np.isfinite(ac_half):
        return None
    ratio = float(ac_T / ac_half) if ac_half > 0 else float('inf')
    finite = {k: v for k, v in acs.items() if np.isfinite(v)}
    best = max(finite, key=finite.get)
    if best == 'T':
        verdict = '独立确认 (滞后 T 处是严格自相关极大)'
    elif ac_T >= 0.95 * finite[best]:
        verdict = f'弱确认 (极大在 {best}, 但 ac(T) 与它相差 <5%)'
    else:
        verdict = f'未确认 (极大在 {best}, 不在 T)'
    return {'ac': acs, 'ac_T': ac_T, 'ac_half': float(ac_half),
            'ratio_T_over_half': ratio, 'lag_best': best,
            'lag_T_px': lag, 'verdict': verdict,
            'confirmed': bool(best == 'T'),
            'n_used': int(m.sum())}


def _log_bin(k, p, n_bin=12):
    """
    把 (k, P) 按对数间隔分箱取均值 —— **只用于画图, 判定绝不用它**。

    单次实现的周期图自由度只有 2, 方差极大: 1080 根谱线叠在 log-log 上是一团
    噪声, 完全看不出"功率随 k 单调下降"这个唯一重要的形状。分箱把方差压下去,
    代价是**抹掉窄峰** —— 所以它只能展示趋势。判定 (带内局部极大 + 突出度)
    一律在未分箱的原始谱上做, 见 _series_period。
    """
    k = np.asarray(k, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(k) & np.isfinite(p) & (k > 0) & (p > 0)
    if int(m.sum()) < n_bin:
        return k[m], p[m]
    kk, pp = k[m], p[m]
    edges = np.logspace(np.log10(kk.min()), np.log10(kk.max()), n_bin + 1)
    idx = np.clip(np.digitize(kk, edges) - 1, 0, n_bin - 1)
    kb, pb = [], []
    for bidx in range(n_bin):
        sel = idx == bidx
        if sel.any():
            kb.append(float(kk[sel].mean()))
            pb.append(float(pp[sel].mean()))
    return np.asarray(kb), np.asarray(pb)


def _leading_dark(y, frac=0.5):
    """
    头部连续"暗帧"的个数 (相机预热 / 快门未开)。

    为什么需要这个数: Reading 三条序列的**前 16 帧**亮度约 25.6, 而稳态是
    84~115 —— 这是一段硬阶跃, 不是物理。阶跃的谱是宽频的, 会给整条谱灌进
    高频功率, 从而把"漂移占比"**压小**。实测: 参考序列不去预热段漂移占比
    9.9%, 去掉后 4.1%; 而再往后去掉上升沿则升到 ~88%。也就是说照原样报出的
    漂移占比是**被低估**的, 不是高估。

    阈值取 0.5 x 中位数 (自标定, 不写死绝对值): 三条序列的稳态亮度不同
    (115 / 90 / 84), 固定的绝对阈值会漏掉最暗的那条。
    """
    yy = np.asarray(y, dtype=np.float64)
    if yy.size == 0:
        return 0
    med = float(np.median(yy))
    if med <= 0:
        return 0
    thr = frac * med
    i = 0
    while i < yy.size and yy[i] < thr:
        i += 1
    return int(i)


def _window_drift_ratio(y, frac=0.1):
    """
    前 frac 段与后 frac 段各自的最小二乘斜率之比的绝对值。

    为什么需要这个数: 头部暗帧是一段硬阶跃, 它给整条序列灌进一个**单边**
    趋势。前后窗口的斜率比就是"这个单边趋势占了多大比重"的一个直接读数 ——
    比值远大于 1 说明前段的漂移是阶跃造成的, 而不是物理漂移。

    返回 (比值, 前段斜率, 后段斜率); 段长不足或后段斜率退化到 0 时返回 nan。

    诚实边界: 这个数**只作诊断**。它不去自动裁剪序列 —— 自动裁掉前段等于
    改变已经报出去的口径。序列是否裁剪由 trim_scan 的敏感性扫描裁决, 本函数
    只负责把"漂移比有多大"这件事量化出来。
    """
    yy = np.asarray(y, dtype=np.float64)
    n_w = int(round(frac * yy.size))
    if yy.size < 20 or n_w < 8:
        return float('nan'), float('nan'), float('nan')
    sl = lambda v: float(np.polyfit(np.arange(v.size), v, 1)[0])  # noqa: E731
    s_head, s_tail = sl(yy[:n_w]), sl(yy[-n_w:])
    if not (np.isfinite(s_head) and np.isfinite(s_tail)) or abs(s_tail) < 1e-30:
        return float('nan'), s_head, s_tail
    return float(abs(s_head) / abs(s_tail)), s_head, s_tail


def _series_period(lum, n_keep=6, min_cycles=8):
    """
    一维时序 (逐帧平均亮度) 的主周期 —— 带**居中的检出判定**, 检不出就报检不出。

    第一版直接把谱峰当主周期, 结果三条序列全报 2161.0 帧 = 100.0% 全长, 即谱峰
    恒在 k=1。那不是发现, 是残余趋势: 线性去趋势后 std 只从 8.66 降到 8.38,
    k=1 仍占 51% 功率, 说明主导成分是**非线性慢漂移** (凝胶干燥/照明漂移),
    线性拟合吃不掉它, Hann 窗又把它压成一根高谱线。把趋势读成周期是这类分析
    最典型的自欺, 所以这里做三件事:

      1. 高通: 除线性拟合外, 再减去窗口为 n/min_cycles 的滑动平均, 压掉慢漂移。
      2. 限带: 只在 k >= min_cycles 的谱线里找峰。周期必须短于全长的 1/8,
         否则"周期"和"整段记录"无法区分。
      3. 判定: 同时要求 (a) 带外漂移功率占比 < drift_max, (b) 峰功率占比
         >= peak_min。任一不满足就 detected=False, 周期置 None。

    谱峰仍做**抛物线插值**。这不是装饰: 2161 帧的记录里周期 ~137 帧对应第 15.8
    根谱线, 取整会锁到第 16 根 = 135.1 帧, 引入 -1.3% 量化误差; B3 报的是三个
    等长序列的**周期比值**, 量化误差在三者上不同, 会污染比值。

    返回字段: detected / period_frames / drift_frac / peak_power_frac / verdict。
    """
    y = np.asarray(lum, dtype=np.float64)
    n = len(y)
    if n < 4 * min_cycles:
        return None
    idx = np.arange(n)
    y = y - np.polyval(np.polyfit(idx, y, 1), idx)

    # 漂移占比: 在**高通之前**的谱上量, 否则高通会把它一起减掉、量不到
    F0 = np.fft.rfft(y)
    p0 = np.abs(F0) ** 2
    p0[0] = 0.0
    ks = np.arange(len(F0))
    tot0 = float(p0[1:].sum())
    if tot0 <= 0:
        return None
    drift = float(p0[(ks >= 1) & (ks < min_cycles)].sum() / tot0)

    # 高通: 直接把 k < min_cycles 的谱线置零。用谱域而不是滑动平均 ——
    # 减滑动平均的频响是**梳状**的 (在 k = n/w 的整数倍处有零点), 会在零点之间
    # 造出假的峰; 谱域置零的搜索带就是通带, 没有这个问题。
    F2 = F0.copy()
    F2[:min_cycles] = 0.0
    yh = np.fft.irfft(F2, n)
    f = np.abs(np.fft.rfft(yh * np.hanning(n)))
    p = f ** 2
    p[0] = 0.0
    tot = float(p[1:].sum())
    if tot <= 0:
        return None

    p_band = p.copy()
    p_band[ks < min_cycles] = 0.0
    k0 = int(np.argmax(p_band))
    peak_frac = float(p[k0] / tot)

    # 局部背景: 峰附近 (不含峰本身) 的中位数功率。用中位数而不是均值, 免得
    # 背景被峰自己抬高。
    lo = max(min_cycles, k0 - PROM_WIN)
    hi = min(len(p), k0 + PROM_WIN + 1)
    bg_vals = np.concatenate([p[lo:k0], p[k0 + 1:hi]])
    bg = float(np.median(bg_vals)) if len(bg_vals) else 0.0
    prominence = float(p[k0] / bg) if bg > 0 else float('inf')

    # 【检出判定的核心, 两步】
    # (a) 峰必须是**带内内部的局部极大**。只要求"带内最大"不够: 亮度谱是红的
    #     (功率随 k 单调下降), 于是带内最大值总落在带的最低端, 即频带边缘。
    #     把边缘当峰, 换个 min_cycles 就换一个"周期" —— 那是把滤波器边界读成
    #     了物理量。真振荡必须在带内部拱起一个局部极大。
    # (b) 峰必须**高出局部背景** PROM_MIN 倍。这一条不可省: 红噪声的谱很散
    #     (单次实现的周期图方差极大), 天然会在带内冒出一堆局部极大; 只看
    #     "有没有局部极大"必然假阳性 —— 实测纯红噪声 300 次里就有 226 次冒峰。
    #
    # PROM_MIN 与 PROM_WIN 不是拍出来的, 是按零假设标定的:
    #   纯红噪声 (随机游走+漂移) 300 次抽样, prominence 中位 23, p99 85, 最大 113;
    #   正弦+30% 噪声+漂移, T=25~200 帧, 20 次抽样里最小 2309。
    #   两者之间有约 20 倍的空隙, 取 300 落在空隙中间 -> 假阳性 < 1/300,
    #   而 T>=25 帧的信号 100% 检出。标定脚本见 _v13_diag_tier3.py。
    interior = bool(min_cycles < k0 < len(p) - 1)
    local_max = bool(p[k0] > p[k0 - 1] and p[k0] > p[k0 + 1])
    detected = bool(interior and local_max and prominence >= PROM_MIN)
    if detected:
        verdict = '检出周期'
    elif not interior or not local_max:
        verdict = '谱为红噪声, 无带内峰 (边缘极大不是峰)'
    else:
        verdict = (f'峰不显著 (突出度 {prominence:.0f} < {PROM_MIN}), 未检出')

    def refine(k0_):
        """在谱线 k0 处对 |F| 做三点抛物线定峰, 返回小数谱线位置。"""
        if k0_ <= 1 or k0_ + 1 >= len(f):
            return float(k0_)
        a, b, c = f[k0_ - 1], f[k0_], f[k0_ + 1]
        den = a - 2.0 * b + c
        return float(k0_) + (0.5 * (a - c) / den if den != 0 else 0.0)

    k_ref = refine(k0)
    order = ks[p > 0][np.argsort(p[p > 0])[::-1]][:n_keep]
    return {'detected': detected,
            'period_frames': float(n / k_ref) if detected and k_ref > 0 else None,
            'k_refined': k_ref if detected else None,
            'k_band_argmax': k0,
            'period_frames_band_argmax': float(n / k0),
            'top_periods_frames': [float(n / refine(int(t))) for t in order],
            'drift_frac': drift, 'peak_power_frac': peak_frac,
            'prominence': prominence, 'prom_min': PROM_MIN,
            'min_cycles': int(min_cycles), 'n_frames': n,
            # 判定所依据的那条谱本身。作图时要把它画出来 —— "未检出"是个结论,
            # 只有把谱亮出来 (带内最大值落在频带边缘、且低于突出度阈值),
            # 读者才能自己复核这个结论, 而不是被告知一个 verdict 字符串。
            # power[i] 对应谱线 k = i+1 (已丢掉 k=0 的直流项)。
            'power': p[1:].tolist(),
            'power_bg': bg,
            'power_thresh': float(bg * PROM_MIN),
            'verdict': verdict}


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
        的混杂因子 (相区参数不同、盒子内波长数量级差、时空图轴语义不同、
        实验影像分辨率过低) 使得任何一个数字都无法单独归因给物理, 因此在
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
            '第三层全部指标依赖离线预处理缓存; 归档侧由 spiral_v13_prepare.py 生成, '
            '**A 段参考侧**另需自建种子, 由 spiral_v14_prepare.py 生成。',
            f'os.path.isfile({cache}/*.npz) 且 os.path.isfile({v14cache}/gs_seeds.npz)',
            'CLIP / Southampton / Reading + v14 自建 GS 种子',
            f'缺失 {missing} -> 跳过', '全部缓存齐备', '缓存齐备', False,
            note='这是不可计算, 不是"不达标"。请先运行 python spiral_v13_prepare.py '
                 '与 python spiral_v14_prepare.py')
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
    print("  单独归因给盒子: (1) 双方 f/k 落在 Pearson 相图的不同相区; (2) 盒子内")
    print("  波长数不同。v14 把目标收缩到本段真正能支撑的那一句话 ——")
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
    print(f"    复核 spiral_metric_v14.V13_A_BASELINE: "
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
             '照实报缺口。**边界**: 本段的 lam_pk 与 n_lam 只能按"某条离散壳上的'
             '一个读数"读, 既不保证是斑图波长, 也没有独立佐证; 涉及"盒子内波长数"'
             '的结论一律降级为量级陈述, 不能作为定量结论引用。')
    print(f"    -> F3(b) 缺口已如实登记: lam_pk 量化 "
          f"({len(set(round(d_ref[s]['lam_pk'], 4) for s in seeds))} 个不同值 / "
          f"{len(seeds)} 个种子), 且缺独立佐证 "
          f"(vs 自相关中位偏差 {np.nanmedian(_ac_dev):.0%}); 未改 _char_scale。")

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
        _s2 = np.column_stack([np.arange(len(y_read), dtype=float), y_read])
        _la = _series_lag_agreement(_s2, sp['period_frames_band_argmax'])
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
                      '自相关不要求严格周期, 所以两法**不一致本身就构成证据**: '
                      '谱峰与自相关都没给出可确认的周期 => "未检出"不是谱方法'
                      '偏保守。判据与结论分离 —— 让阈值裁决物理是已经犯过一次的'
                      f'错法。实测: {_lags_txt or "无可用序列"}。')

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
              # result/_v14_data.json 的 A1.ref_panel 里。
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
    print("    2. **第三层不设物理达标线**。三个混杂因子 (相区参数不同 / 盒子内")
    print("       波长数差一个量级 / 时空图轴语义不同) 使任何数字都无法单独归因")
    print("       给物理, 在该区间做达标裁决没有信息量。判定项改为流程守卫 T3-1~5。")
    print("    3. Southampton 的 .tif **图版**已被弃用 (可分离背景占方差 68~98.8%,")
    print("       在其上算描述子等于量图版布局); 只保留波峰标记这条数值列。")
    print("    4. Reading 三条序列**全部未检出周期**, 这是结论而非缺失 —— 全局")
    print("       平均亮度在有螺旋波的凝胶上会抵消掉反相振荡, 只剩漂移。")
    print("    5. 模型侧是**确定性**的锁模格子, 没有模型集成; A 段的误差棒来自")
    print("       **参考侧**的 3 个种子, 不是模型侧。")
    return out


# ===========================================================================
# 指标层编排
# ===========================================================================
def _model_field(s6):
    """私有副本 —— 与 spiral_metric._model_field 逐字相同。

    不复用 import 是因为它没有被下划线以外的方式导出; 复制一行逻辑比
    跨模块访问私有名更干净。若将来上游定义变了, 这里会**静默分歧**,
    这是已知代价, 由说明文档记录。
    """
    v = np.asarray(s6['v'], dtype=np.float64)
    vb = float(v.mean())
    return v / vb - 1.0 if vb > 0 else v - v.mean()


# ===========================================================================
# v14 · F2a 负对照 (h/J 扫描) 的指标与守卫
# ===========================================================================
def metric_hj_negative_control(hj):
    """
    2.5 临界性负对照 —— 把 h/J 推离 1.0, 看指标是否塌缩。

    这是 v14 的生死检验。判据**已修正** (任务书 Task 3 的原判据物理上不成立,
    详见 spiral_model_v14.hj_scan 的函数头):
      主判据: argmax c_fit == h/J=1.0。含义是 Calabrese-Cardy ansatz 只在临界
              点成立 —— 实测 0.5072, 而 0.5 处 0.0646、1.5 处 0.1317。
      次判据: argmax S(L/2) 落在 [0.9, 1.05]。有限尺寸熵峰偏向 h/J<1 是
              **已声明**的效应 (stage2_critical_break docstring), 故给窗口。
      区分力: 非临界点上框架必须**正确失败** —— 审计 §五.2「一个能在非临界点
              正确失败的框架, 比一个永远输出 3% 的框架可靠」。要求
              |c-0.5|/0.5 > 20%。
      gap=p1/p0 是**单调量**, 无内部极值, 降级为诊断, **不作判据**。
    """
    if not hj:
        record_metric(2, '5 临界性负对照 (h/J 扫描)', 'argmax c_fit @ h/J=1',
                      '把 h/J 推离 1.0, 检验临界指标是否塌缩',
                      'spiral_model_v14.hj_scan()',
                      '横场 Ising 热力学临界点 h/J=1', None, None,
                      '未运行 (ctx 无 hj)', None,
                      note='负对照未执行, 无法判定。')
        record_guard('F2 负对照已运行', 'collect_metrics_v14',
                     'ctx["hj"] 存在', False)
        return None

    v = hj['verdict']
    pts = hj['points']
    measured = {
        'hj_at_c_peak': v['hj_at_c_peak'],
        'c_peak': float(max(q['c_fit'] for q in pts)),
        'c_at_hj_first': float(pts[0]['c_fit']),
        'c_at_hj_last': float(pts[-1]['c_fit']),
        'hj_at_s_peak': v['hj_at_s_peak'],
        'gap_monotone': v['gap_monotone_decreasing'],
        'discrim': v['discrim'],
    }
    record_metric(
        2, '5 临界性负对照 (h/J 扫描)', 'argmax c_fit @ h/J=1',
        '把 h/J 推离 1.0, 检验临界指标是否塌缩 —— 临界模拟是不是事实',
        'spiral_model_v14.hj_scan(): 逐点精确对角化 + Calabrese-Cardy 拟合',
        '横场 Ising 热力学临界点 h/J=1 (解析已知)',
        measured,
        '基线只在 h/J=1.0 单点检验, 无负对照',
        'c_fit 峰位 = 1.0 且 S 峰位 in [0.9,1.05] 且非临界点 |c-0.5|/0.5 > 20%',
        v['passed'],
        note=f"**判据已修正**: 任务书原判据 (gap 在 h/J=1.0 取极小值) 物理上"
             f"不成立 —— gap 对 h/J 单调递减 ({pts[0]['gap']:.4f} -> "
             f"{pts[-1]['gap']:.4f}), 曲线无内部极值, 照原判据会**伪失败**。"
             f"改用 c_fit 峰 (0.5072 @ h/J=1.0, 首点 {pts[0]['c_fit']:.4f}, "
             f"末点 {pts[-1]['c_fit']:.4f}) + S 峰位 ({v['hj_at_s_peak']:.2f}) "
             f"+ 区分力。诚实边界: S 峰位漂移 {v['hj_at_s_peak']-1.0:+.2f} 是"
             f"有限尺寸效应 (L=16 熵峰偏向 h/J<1), 已声明;")

    # ---- 三条可证伪守卫 ----
    record_guard('F2a 主判据: c_fit 峰在临界点', 'hj_scan',
                 f"argmax c_fit = h/J {v['hj_at_c_peak']:.2f} (要求 = 1.0)",
                 v['ok_c_peak_at_critical'],
                 note='Calabrese-Cardy ansatz 只在临界点成立 —— 比"某个数小"强。')
    record_guard('F2a 次判据: S(L/2) 峰位在窗口内', 'hj_scan',
                 f"argmax S = h/J {v['hj_at_s_peak']:.2f} "
                 f"(窗口 {v['s_window']})",
                 v['ok_s_peak_in_window'],
                 note='窗口而非单点: 有限尺寸熵峰偏向 h/J<1 是预期效应。')
    record_guard('F2a 区分力: 非临界点正确失败', 'hj_scan',
                 f"h/J in {sorted(v['discrim'])} 上 |c-0.5|/0.5 > "
                 f"{v['discrim_min']:.0%}",
                 v['ok_discriminating'],
                 note='审计 §五.2: 能在非临界点正确失败的框架才可靠。')
    record_guard('F2a 诊断: gap 单调 (不作判据)', 'hj_scan',
                 f"gap 对 h/J 单调递减 = {v['gap_monotone_decreasing']}",
                 v['gap_monotone_decreasing'], expect_pass=False,
                 note='这条**故意不计入通过率**: 单调是物理预期 (GHZ->乘积态), '
                      '不是"通过"。它的作用是防止有人再把 gap 当临界判据用。')

    print(f"\n  ── v14·F2a 负对照 (h/J 扫描) ──")
    print(f"      c_fit 峰位 h/J={v['hj_at_c_peak']:.2f} "
          f"(c={measured['c_peak']:.4f}) | S 峰位 h/J={v['hj_at_s_peak']:.2f} "
          f"| 区分力 {'PASS' if v['ok_discriminating'] else 'FAIL'}")
    print(f"      裁决: {'临界性成立' if v['passed'] else '未通过'}")
    return measured


# ===========================================================================
# v14 · F1 跨边界条件中心荷交叉检验的指标与守卫
# ===========================================================================
def metric_f1_cross_boundary(f1):
    """
    2.6 跨边界条件中心荷交叉检验。

    **分层标注** (审计 §三之补 F1① 的降级要求, 必须写在指标名里):
      同族   —— PBC vs θ=π Z2 twist。换的是**边界条件与态**, 不换估计量;
                两边都用同一个 Calabrese-Cardy log 拟合。
      独立族 —— **未建立**。已试的 Rényi 指数扫描实测验证失败 (L=16 上
                c(alpha) 标准差 0.0225、均值对 0.5 偏 7.97%), 如实登记为诊断项,
                不冒充第二把尺子。

    两个偏差并列报告 (审计 F1② 要求, 5% 标准不放宽):
      对 c_exact(L) —— 消掉有限尺寸偏差后的纯方法误差;
      对真值 0.5    —— 把有限尺寸偏差也算进去的端到端误差。
    只报后者会掩盖有限尺寸, 只报前者会让读者误以为到达了精确值。
    """
    if not f1:
        record_metric(2, '6 跨边界条件中心荷交叉检验 (同族)',
                      '|c_twist - c_PBC| / c_PBC',
                      '换边界条件后中心荷读数是否稳定', 'spiral_model_v14'
                      '.cross_boundary_twist()',
                      '临界 Ising c=0.5', None, None,
                      '未运行 (ctx 无 f1)', None,
                      note='未执行, 无法判定。')
        record_guard('F1 跨边界条件已运行', 'collect_metrics_v14',
                     'ctx["f1"] 存在', False)
        return None

    sd = f1['sides']
    c_p = sd['pbc']['c_representative']
    c_t = sd['twist']['c_representative']
    bd, ry = f1['boundary'], f1['renyi']
    dt, de = f1['dev_vs_true'], f1['dev_vs_c_exact']

    # 边界条件真的换了态: E0 与 S(L/2) **都**必须变 (只变一个说明实现有问题)
    bc_changed = bool(bd['E0_twist'] != bd['E0_pbc']
                      and bd['S_half_twist'] != bd['S_half_pbc'])
    devs_reported = bool(
        all(k in dt for k in ('pbc', 'twist'))
        and all(k in de for k in ('pbc', 'twist'))
        and all(np.isfinite([dt['pbc'], dt['twist'], de['pbc'], de['twist']])))

    measured = {
        'c_pbc': c_p, 'c_twist': c_t,
        'c_exact_pbc': sd['pbc']['c_exact'], 'c_exact_twist': sd['twist']['c_exact'],
        'rel_diff': f1['rel_diff'],
        'dev_vs_exact_pbc': de['pbc'], 'dev_vs_exact_twist': de['twist'],
        'dev_vs_true_pbc': dt['pbc'], 'dev_vs_true_twist': dt['twist'],
        'c_std_pbc': sd['pbc']['c_std'], 'c_std_twist': sd['twist']['c_std'],
        'min_overlap': f1['min_overlap'],
        'renyi_mean': ry['mean'], 'renyi_std': ry['std'],
        'renyi_dev_vs_true': ry['dev_vs_true'], 'renyi_validated': ry['validated'],
    }
    record_metric(
        2, '6 跨边界条件中心荷交叉检验 (同族)', '|c_twist - c_PBC| / c_PBC',
        '换边界条件 (PBC -> θ=π Z2 twist) 后中心荷读数是否稳定 —— '
        '**同族**交叉检验 (换态与边界条件, 不换估计量)',
        'spiral_model_v14.cross_boundary_twist(): 精确对角化 + 检查点选择协议 MERA '
        f'(每边 {len(f1["seeds"])} 种子 x {f1["steps"]} 步, 取 eps_c 中位数轨迹)',
        '临界 Ising c=0.5 (体中心荷不应随边界条件改变)',
        measured,
        '基线只有 PBC 一个边界条件, 无跨边界条件检验',
        f'相对差 <= {f1["rel_max"]:.0%}, 且两个偏差 (vs c_exact(L) 与 vs 0.5) 并列报告',
        f1['passed'],
        note=f"**同族**标注: ansatz 仍是同一个 Calabrese-Cardy log 式, 换的只是"
             f"边界条件与态 —— 它检验稳健性, **不是**独立族估计量。"
             f"独立族**: **未建立**。已试 Rényi 指数扫描 (alpha in "
             f"{[r['alpha'] for r in ry['alphas']]}), 实测均值 c="
             f"{ry['mean']:.5f}, std={ry['std']:.5f}, 对 0.5 偏差 "
             f"{ry['dev_vs_true']:.2%} -> 验证不通过, 降为诊断项。"
             f"诚实边界: 这是受 L<=16 天花板限制的结果, 已如实声明; "
             f"未伪造第二把尺子。")

    # ---- 可证伪守卫 ----
    record_guard('F1 同族: 跨边界条件已运行 (PBC vs θ=π twist)',
                 'cross_boundary_twist',
                 f"两侧均有 {len(f1['seeds'])} 条轨迹, L={f1['L']}, chi={f1['chi']}",
                 bool(sd['pbc']['runs'] and sd['twist']['runs']))
    record_guard('F1 twist 保持实矩阵 (可喂 float MERA)',
                 'cross_boundary_twist',
                 f"θ=π twist 的 H 为实矩阵 = {bd['twist_h_is_real']}",
                 bd['twist_h_is_real'],
                 note='连续 U(1) twist 给复矩阵, 那样就得改张量网络 dtype —— 本项'
                      '刻意选 Z2 (θ=π), 只为不引入复张量网络。')
    record_guard('F1 边界条件确实换了态 (E0 与 S(L/2) 都变)',
                 'cross_boundary_twist',
                 f"E0/L: {bd['E0_pbc']/f1['L']:+.6f} vs "
                 f"{bd['E0_twist']/f1['L']:+.6f}; S(L/2): "
                 f"{bd['S_half_pbc']:.5f} vs {bd['S_half_twist']:.5f}",
                 bc_changed,
                 note='防"伪对照": 若两者相同, 说明 twist 没生效, 这条会抓住。')
    record_guard('F1 MERA 全部轨迹重叠 >= 0.99', 'cross_boundary_twist',
                 f"min overlap = {f1['min_overlap']:.5f} (两侧共 "
                 f"{len(f1['seeds']) * 2} 条轨迹)",
                 f1['min_overlap'] >= 0.99,
                 note='重叠低说明是**优化没跑到位**, 不是物理 —— 实测朴素协议 '
                      '(600 步/单种子) 在 PBC 侧只到 0.98829, 会把跨边界条件差'
                      '夸大成 5.98%。')
    record_guard('F1 跨边界条件相对差 <= 5%', 'cross_boundary_twist',
                 f"|c_twist - c_PBC|/c_PBC = {f1['rel_diff']:.3%} "
                 f"({c_t:.5f} vs {c_p:.5f})",
                 f1['passed'],
                 note='审计要求 5% 标准**不放宽**。')
    record_guard('F1 两个偏差并列报告 (vs c_exact(L) 与 vs 0.5)',
                 'cross_boundary_twist',
                 f"vs c_exact(L): PBC {de['pbc']:.3%} / twist {de['twist']:.3%}; "
                 f"vs 0.5: PBC {dt['pbc']:.3%} / twist {dt['twist']:.3%}",
                 devs_reported,
                 note='审计 F1②: 只报一个偏差会分别掩盖有限尺寸或方法误差。')
    record_guard('F1 诊断: Rényi 扫描不作独立族估计量 (L<=16 实测失效)',
                 'renyi_alpha_scan',
                 f"c(alpha) std = {ry['std']:.5f} (门槛 "
                 f"{0.02}), 均值对 0.5 偏差 = {ry['dev_vs_true']:.2%}",
                 ry['validated'], expect_pass=False,
                 note='这条**故意不计入通过率**: 它是负面结果的可复现证据。'
                      '结论是"本工作未建立独立族估计量", 而不是"通过了"。')

    print(f"\n  ── v14·F1 跨边界条件中心荷 (同族) ──")
    print(f"      PBC c={c_p:.5f} | θ=π twist c={c_t:.5f} | "
          f"相对差 {f1['rel_diff']:.3%} "
          f"{'PASS' if f1['passed'] else 'FAIL'}")
    print(f"      [独立族] Rényi 扫描 std={ry['std']:.5f} "
          f"-> {'可作估计量' if ry['validated'] else '验证失败, 未建立独立族估计量'}")
    return measured


# ===========================================================================
# v14 · F5 一致性检查束的指标与守卫
# ===========================================================================
def metric_consistency_checks(f5):
    """
    2.7 一致性检查束 —— 五条, 每条都用两条独立路径对同一个量。

    与第二层其它指标的分工: 其它指标对标的是**物理真值** (c=0.5, 2Δ=0.25,
    临界点 h/J=1), 检验物理结论; 这一项对标的是**解析式**, 检验实现本身。
    它通过与否不改变任何物理结论 —— 改的是"这些数字是怎么算出来的"这件事
    的可信度, 以及"解析路径没被绕过"这件事有没有被验过。

    七条判据, 每条都刻意不做成"某个数很小"(那种判据没有内容, 任何实现都能
    让它很小), 而是"两条路径给出同一个数", 容差取各自噪声的量级:
      L2   tol 1e-10  eigsh 与 JW 闭合式都只差舍入
      L3   tol 1e-12  幂迭代自身阈 1e-13, 残差应在同量级
      L5   tol 1e-12  环上 4-2-2=0 是恒等式
      L6a  tol 1e-13  零和是严格换写, 只剩抵消误差
      L6b  tol 1e-12  平衡恒等式两侧是同一更新式的算术
      L7   上界       熵 <= ln N (均匀分布最大熵) 且 |<等权, D*>| <= 1
    删掉的三条 (L2 步数 / L3 Jordan 峰 / L4 酉性) 的理由见 consistency_checks。
    """
    if not f5:
        record_metric(2, '7 一致性检查束 (数值路径 <-> 解析路径)', '5 条各自的双路径偏差',
                      '实现与它声称的解析式是否一致', 'spiral_model_v14'
                      '.consistency_checks()',
                      'JW 闭合式 / 幂迭代残差 / Forman 恒等式 / '
                      '离散 Laplacian 零和 / 平衡恒等式 / 熵上界',
                      None, None, '未运行 (ctx 无 f5)', None,
                      note='未执行, 无法判定。')
        record_guard('F5 一致性检查已运行', 'collect_metrics_v14',
                     'ctx["f5"] 存在', False)
        return None

    c2, c3, c5, c6, c7 = f5['L2'], f5['L3'], f5['L5'], f5['L6'], f5['L7']
    measured = {
        'l2_max_abs_dev': c2['max_abs_dev'], 'l2_n_points': c2['n_points'],
        'l3_max_residual': c3['max_residual'], 'l3_n_case': c3['n_case'],
        'l5_cycle_max_abs': c5['cycle_max_abs'],
        'l5_cycle_min_n': c5['cycle_min_n'],
        'l6_lap_zero_rel_max': c6['lap_zero_rel_max'],
        'l6_balance_abs_err_max': c6['balance_abs_err_max'],
        'l6_drift_first_step': c6['drift_first_step'],
        'l7_entropy_ratio': c7['entropy_ratio'],
        'l7_equal_weight': c7['equal_weight'],
        'l7_defined': c7['defined'], 'l7_bound_ok': c7['bound_ok'],
    }
    record_metric(
        2, '7 一致性检查束 (数值路径 <-> 解析路径)', '5 条各自的双路径偏差',
        '实现与它声称的解析式是否一致 —— 测的是**实现**, 不是物理结论',
        'spiral_model_v14.consistency_checks(): eigsh vs JW 闭合式 / 幂迭代残差 '
        '/ Forman 环恒等式 / 离散 Laplacian 零和 / 精确平衡恒等式 / 熵上界',
        'Jordan-Wigner 闭合式, 4-deg(u)-deg(v), 周期域上 Laplacian 零和, '
        '同一更新式的离散平衡',
        measured,
        '既有实现完全没有这类对照 —— 三份检查里两份恒真、一份是范畴错误',
        'L2/L3/L5/L6a/L6b < 1e-12 (L2 < 1e-10), L7 两条上界成立',
        bool(c2['max_abs_dev'] < 1e-10 and c3['max_residual'] < 1e-12
             and c5['cycle_max_abs'] < 1e-12
             and c6['lap_zero_rel_max'] < 1e-13
             and c6['balance_abs_err_max'] < 1e-12
             and c7['bound_ok']),
        note=f"**五条各自的两条路径**: L2 用 Jordan-Wigner 闭合式 "
             f"E0=-Sum sqrt(J^2+h^2-2Jh cos(pi(2k+1)/L)) 对 eigsh, "
             f"{c2['n_points']} 个 (L,h) 点最大偏差 {c2['max_abs_dev']:.2e} —— "
             f"这条是**真的独立路径**, 它能测出 H 的位序、周期键、横场符号错, "
             f"而 eigsh 对 H 本身对不对一句话都没有。L3 断言幂迭代残差 "
             f"{c3['max_residual']:.2e} ({c3['n_case']} 个算例) —— "
             f"原判据写的是「与 Lanczos 100/200/500 步无关」, 但 eigsh 没有"
             f"步数参数, 换成 maxiter 后 30 以上逐位相同, 零分辨力。"
             f"L5 环 C_n (n>={c5['cycle_min_n']}) 的 Forman max|F|="
             f"{c5['cycle_max_abs']:.2e} (解析 0); **n=3 已排除** —— C_3 就是 "
             f"K_3, augmented 的 +3 项把它顶到 +3。"
             f"L6 两条: 离散 Laplacian 零和相对残差 "
             f"{c6['lap_zero_rel_max']:.2e}, 精确平衡恒等式 abs 误差 "
             f"{c6['balance_abs_err_max']:.2e}。"
             f"L7 熵比 {c7['entropy_ratio']:.4f} <= 1 且等权度 "
             f"{c7['equal_weight']:.4f} <= 1。"
             f"**诚实边界**: L6 的两条是**格式性质** —— 保证代码按写下的方程在"
             f"算, 不保证那个方程描述的是生命; 首步 <u+v> 漂移 "
             f"{c6['drift_first_step']:+.3e} 就是「u+v 守恒」这个伪不变量的反例 "
             f"(源汇项 F(1-u) 与 (F+k)v 不相消)。"
             f"L7 上界在塌缩支恒取等号, "
             f"本项 defined={c7['defined']}, 那一支不带信息量。")

    record_guard('F5-L2 基态能量与 JW 闭合式一致', 'consistency_checks',
                 f"{c2['n_points']} 个 (L,h) 点 max|eigsh - JW| = "
                 f"{c2['max_abs_dev']:.2e} < 1e-10",
                 bool(c2['max_abs_dev'] < 1e-10),
                 note='这是唯一一条真的独立路径: 它同时验 H 的构造、位序约定与'
                      '周期边界。')
    record_guard('F5-L3 幂迭代残差达阈', 'consistency_checks',
                 f"{c3['n_case']} 个算例 max ||A x - <x,A x> x|| = "
                 f"{c3['max_residual']:.2e} < 1e-12",
                 bool(c3['max_residual'] < 1e-12),
                 note='判据是残差, 不是步数 —— 步数没有解析预期, 残差有。')
    record_guard('F5-L5 Forman 在环 C_n (n>=4) 上为 0', 'validate_curvature',
                 f"n = {sorted(int(k) for k in c5['cycle_mean'])}, "
                 f"max|F| = {c5['cycle_max_abs']:.2e} < 1e-12",
                 bool(c5['cycle_max_abs'] < 1e-12),
                 note='判据是恒等式 4-deg(u)-deg(v); **n=3 必须排除** (C_3=K_3, '
                      'augmented 项给 +3)。')
    record_guard('F5-L6a 离散拉普拉斯零和', 'gray_scott_invariants',
                 f"周期域上 max |Sum Lap u| / (4<u>/h^2) = "
                 f"{c6['lap_zero_rel_max']:.2e} < 1e-13",
                 bool(c6['lap_zero_rel_max'] < 1e-13),
                 note='测 stencil 系数与 roll 的包裹方向, 与守恒律无关。')
    record_guard('F5-L6b Gray-Scott 精确离散平衡恒等式', 'gray_scott_invariants',
                 f"max |d<u+v> - dt[F(1-<u>) - (F+k)<v>]| = "
                 f"{c6['balance_abs_err_max']:.2e} < 1e-12",
                 bool(c6['balance_abs_err_max'] < 1e-12),
                 note='**替代"u+v 质量守恒"这个错误判据**: 源汇项不相消, 照原'
                      '判据会伪失败 (实测首步漂移已超其容差 62 倍)。这条用'
                      '更新前的场算, 两侧是同一更新式的算术, 检验整个 RHS 装配。')
    record_guard('F5-L7 熵比与等权度满足上界', 'stage7_consciousness',
                 f"熵比 {c7['entropy_ratio']:.4f} <= 1 且等权度 "
                 f"{c7['equal_weight']:.4f} <= 1",
                 bool(c7['bound_ok']),
                 note='上界: 熵 <= ln N (均匀分布最大熵) 与 Cauchy-Schwarz。'
                      '**只在非塌缩支有内容** —— 塌缩支 probs=1/N 是约定, 两条'
                      '恒取等号; 该项 defined='
                      f"{c7['defined']}, 如实标注。")
    record_guard('F5 诊断: Gray-Scott 的 u+v 不是守恒量', 'stage6_life',
                 f"首步 <u+v> 漂移 = {c6['drift_first_step']:+.3e} != 0 "
                 f"(源汇项 F(1-<u>) - (F+k)<v> 不相消)",
                 bool(abs(c6['drift_first_step']) < 1e-6), expect_pass=False,
                 note='**故意不计入通过率**: 这是负面事实的可复现证据 —— '
                      '"u+v 守恒到 1e-6"物理上不成立, 只有恰好已在稳态时才成立。'
                      '留着它是为了防止这个错误判据以后又被加回来。')

    print(f"\n  ── v14·F5 一致性检查 (数值路径 <-> 解析路径) ──")
    print(f"      L2 JW 闭合式  max|eigsh-JW| = {c2['max_abs_dev']:.2e} | "
          f"L3 幂迭代残差 max = {c3['max_residual']:.2e} | "
          f"L5 环 Forman max|F| = {c5['cycle_max_abs']:.2e}")
    print(f"      L6 Laplacian 零和 = {c6['lap_zero_rel_max']:.2e}, "
          f"平衡恒等式 = {c6['balance_abs_err_max']:.2e} | "
          f"L7 熵比 {c7['entropy_ratio']:.4f}, 等权度 {c7['equal_weight']:.4f}"
          f"{'' if c7['defined'] else ' (塌缩支, 约定值)'}")
    return measured


# ===========================================================================
# v14 · F6 涌现几何的对照鲁棒性
# ===========================================================================
def _pctile_rank(value, sample):
    """value 在 sample 中的百分位 (sample 里 <= value 的比例 x 100)。"""
    s = np.asarray([x for x in sample if np.isfinite(x)], dtype=float)
    if s.size == 0 or not np.isfinite(value):
        return float('nan')
    return float(100.0 * np.mean(s <= value))


def metric_geometry_robustness(s5):
    """
    2.8 涌现几何的对照鲁棒性 —— MERA 体几何落在零模型对照分布的哪个位置。

    为什么需要这一项: 原来只有一个对照 (同规模随机图, 单个实例), 而结论
    直接读成"MERA 是负曲率"。单实例比较有两个问题 —— 它给不出"多负算异常"
    的尺度, 也没法控制"图接近树"这个前提。三类对照各控一个维度:
      gnm  同 |V| 同 |E|          -> 控规模, 不控度分布
      ws   平均度与 MERA 图对齐    -> 控度分布, 同时带高聚类
      tree 平衡二叉树              -> 控"接近树"这个前提 (零三角形)
    指标取 MERA 在这 9 个实例里的**百分位**, 于是"异常/不异常"变成一个有序
    量, 而不是一次单点比较的措辞。

    判据 (F6b) 是双边界 [5%, 95%]: 落在任一端都是统计异常 —— 更负说明几何
    比零模型更"双曲", 更不负说明它比零模型更平坦。两个方向都必须报, 不能
    只挑有利的那一侧说。

    诚实边界: 这是**零模型对照**, 不是同一物理系统的不同实现。百分位落在
    区间内只说明"MERA 的这类曲率读数在同类规模的图里不特别", 它**不**支持
    "MERA 就是全息对偶"; 反过来, 落在区间外也只说明这张图在那一个测度上
    偏了, 不构成对涌现几何整体主张的否证。
    """
    if not s5 or not s5.get('controls'):
        record_metric(2, '8 涌现几何的对照鲁棒性', 'MERA 在零模型对照分布中的百分位',
                      '涌现几何的曲率读数在同类规模图里是否异常',
                      'spiral_model_v14.geometry_controls()', '三类零模型对照族',
                      None, None, '未运行 (ctx 无对照族)', None,
                      note='未执行, 无法判定。')
        record_guard('F6 对照几何族已运行', 'collect_metrics_v14',
                     'ctx["s5"]["controls"] 非空', False)
        return None

    ctrls = s5['controls']
    mb = s5['mera_bulk']
    fams = sorted({c['family'] for c in ctrls})

    def _fam_vals(key, fam):
        return [c[key] for c in ctrls if c['family'] == fam]

    f_all = [c['forman_mean'] for c in ctrls]
    o_all = [c['or_mean'] for c in ctrls]
    f_mera, o_mera = float(mb['forman_mean']), float(mb['or_mean'])
    f_pct = _pctile_rank(f_mera, f_all)
    o_pct = _pctile_rank(o_mera, o_all)

    # 逐族均值 —— 让"哪一族把 MERA 拉到哪里"看得见, 而不是只留一个总分位
    by_fam = {f: {'forman_mean': float(np.mean(_fam_vals('forman_mean', f))),
                  'or_mean': float(np.nanmean(_fam_vals('or_mean', f))),
                  'n_triangles': int(np.mean(_fam_vals('n_triangles', f))),
                  'd_bar': float(np.mean(_fam_vals('d_bar', f)))}
              for f in fams}

    # "MERA 比全部对照更负"这个升级主张是否成立 (在两个测度上都要求成立)
    more_neg_all = bool(f_mera < min(f_all) and o_mera < min(o_all))

    measured = {
        'mera_forman_mean': f_mera, 'mera_or_mean': o_mera,
        'mera_d_bar': float(mb['d_bar']),
        'mera_n_triangles': int(mb['n_triangles']),
        'forman_pctile': f_pct, 'or_pctile': o_pct,
        'n_controls': len(ctrls), 'families': fams,
        'forman_ctrl_min': float(min(f_all)), 'forman_ctrl_max': float(max(f_all)),
        'or_ctrl_min': float(min(o_all)), 'or_ctrl_max': float(max(o_all)),
        'by_family': by_fam,
        'more_negative_than_all': more_neg_all,
        'pctile_band': [5.0, 95.0],
    }
    # 结论边界由**数据派生**成字符串, 不手写 —— 手写的边界会与数字脱钩
    if np.isfinite(f_pct) and np.isfinite(o_pct):
        verdict_txt = (
            f'MERA 的 Forman 均值 ({f_mera:+.3f}) 位于 {len(ctrls)} 个对照实例的'
            f'第 {f_pct:.0f} 百分位, Ollivier 均值 ({o_mera:+.4f}) 位于第 '
            f'{o_pct:.0f} 百分位 —— '
            + ('两个测度都落在 [5%, 95%] 内, 故结论是"MERA 体几何具有负曲率'
               '特征, 但在同类规模的零模型里不是统计异常"。'
               if (5.0 <= f_pct <= 95.0 and 5.0 <= o_pct <= 95.0)
               else '至少一个测度落在端点附近, 该测度上 MERA 相对零模型是'
                    '统计异常, 必须按方向如实报告 (更负 = 更双曲 / 更不负 = 更平坦)。')
            + f' "MERA 比全部对照更负" 这个升级主张实测为 {more_neg_all}。')
    else:
        verdict_txt = '百分位无法计算 (对照或 MERA 的曲率出现非有限值)。'

    record_metric(
        2, '8 涌现几何的对照鲁棒性', 'MERA 在零模型对照分布中的百分位',
        '涌现几何的曲率读数在同类规模图里是否异常 —— 把单点比较换成有序量',
        'spiral_model_v14.geometry_controls(): 三类零模型 x 3 实例, '
        '每图算 forman_ricci(triangles=False) 与 ollivier_ricci(alpha=0)',
        '三类零模型: gnm (同 |V| 同 |E|) / ws (配度小世界) / tree (平衡二叉树)',
        measured,
        '原先只有 1 个对照实例 (同规模随机图), 无百分位, 无族分解',
        'Forman 与 Ollivier 两个测度都落在对照分布的 [5%, 95%] 内',
        bool(5.0 <= f_pct <= 95.0 and 5.0 <= o_pct <= 95.0),
        note=verdict_txt + " 逐族均值: " + "; ".join(
            f"{f}: Forman {by_fam[f]['forman_mean']:+.3f} "
            f"(三角 {by_fam[f]['n_triangles']}, d_bar {by_fam[f]['d_bar']:.2f})"
            for f in fams) + "。"
            f"**诚实边界**: 这是零模型对照, 不是同一物理系统的不同实现; "
            f"百分位不特别**不**支持\"MERA 就是全息对偶\", 落在区间外也**不**"
            f"构成对涌现几何整体主张的否证。三角形数与平均度必须与曲率一起读 "
            f"—— 本工作用的是无三角形式 F=4-deg(u)-deg(v), 它只在图接近树时"
            f"与完整 Forman 等价, 而 MERA 图三角数 = {mb['n_triangles']}。")

    record_guard('F6a 对照几何族已建立', 'geometry_controls',
                 f"{len(ctrls)} 个实例, {len(fams)} 族 {fams}, 每族 3 个",
                 bool(len(ctrls) >= 9 and len(fams) >= 3),
                 note='族数决定能控几个维度 (度分布 / 聚类 / 树的成分); '
                      '实例数决定百分位的分辨率 (9 实例 -> 11% 一档)。')
    record_guard('F6b MERA 体几何处于对照分布内 (非统计异常)',
                 'curvature_report',
                 f"Forman 百分位 {f_pct:.0f}%, Ollivier 百分位 {o_pct:.0f}% "
                 f"(区间 [5%, 95%])",
                 bool(5.0 <= f_pct <= 95.0 and 5.0 <= o_pct <= 95.0),
                 note='**双边界**: 落在哪一端都是异常, 两个方向都必须报 —— '
                      '只报"比随机图更不负"而不报它是否落在端点, 等于用措辞'
                      '代替测量。')
    record_guard('F6c 诊断: MERA 并非比全部对照更负', 'geometry_controls',
                 f"Forman min(对照) = {min(f_all):+.3f} vs MERA {f_mera:+.3f}; "
                 f"Ollivier min(对照) = {min(o_all):+.4f} vs MERA {o_mera:+.4f} "
                 f"-> 两个测度同时更负 = {more_neg_all}",
                 more_neg_all, expect_pass=False,
                 note='**诊断项, 不计入通过率 (本项设计上就该不通过)**: '
                      '"升级"措辞 (MERA 的负曲率比零模型更强) 没有任何数据'
                      '支持, 如实登记为假。留着它是为了防止这个更强的主张'
                      '以后又被写回结论里。')

    print(f"\n  ── v14·F6 涌现几何的对照鲁棒性 ──")
    print(f"      MERA: Forman {f_mera:+.3f} (三角 {mb['n_triangles']}, "
          f"d_bar {mb['d_bar']:.2f}) | Ollivier {o_mera:+.4f}")
    print(f"      对照 (n={len(ctrls)}): Forman [{min(f_all):+.3f}, "
          f"{max(f_all):+.3f}] | Ollivier [{min(o_all):+.4f}, {max(o_all):+.4f}]")
    print(f"      百分位: Forman {f_pct:.0f}%, Ollivier {o_pct:.0f}% "
          f"-> {'非统计异常' if (5.0 <= f_pct <= 95.0 and 5.0 <= o_pct <= 95.0) else '存在端点异常'}")
    print(f"      升级主张 (比全部对照更负) = {more_neg_all}")
    return measured


def collect_metrics_v14(ctx, KNOBS, _HERE, v13):
    """
    体检报告编排。

    与 spiral_metric.collect_metrics 的关系:
      - **原样复用** 1.3 谱隙闭合 / 1.4 MERA 一致性 / 第二层全部四项 —— 它们的
        物理前提没变, 定义和阈值一个都没动。
      - **替换** 1.1 / 1.2 为多种子分布版, 并新增 1.5 全程稳健性与两条诊断项。
      - **替换** 第三层为 tier3_structural_ladder (2D 斑图结构对结构)。
      - 守卫登记块是那一段的**逐字副本**。之所以复制而不抽出共用函数:
        改动 spiral_metric.py 会让已发布的历史数字不可复现。代价是两份可能
        分歧 —— 两个文件都是冻结快照, 所以这个代价是可接受的; 但读者必须
        知道它存在。

    v13 (参数名保持不变以维持兼容) =
        {'runs': [...], 'curve': [...], 'steps': int, 'lrs': [...],
         'chi_sweep': {chi: {'eps_c':..., 'R_conf':..., 'overlap':...}},
         'floor_chi': float}
    """
    s2, s3a, s3c = ctx['s2'], ctx['s3a'], ctx['s3c']
    s4, s5, s6, loop = ctx['s4'], ctx['s5'], ctx['s6'], ctx['loop']
    l6, L_chain, chi_dev = ctx['l6'], ctx['L_chain'], ctx['chi_dev']
    psi_main, gs_exact = ctx['psi_main'], s2['gs']

    runs = v13['runs']
    print("\n" + "=" * 76)
    print("  量化指标层 (体检报告)")
    print("=" * 76)
    print("  第一层: 1.1/1.2 用多种子分布; 1.5 全程稳健性 (最坏轨迹)")
    print("  第三层: 3D CDM 场对场 -> 2D 反应-扩散 结构对结构 (方法学展示)")
    print(f"  预算: {v13['steps']} 步 x {len(runs)} 条轨迹 "
          f"(lr={' / '.join(str(x) for x in v13['lrs'])}, chi={chi_dev})")
    print("  原则: 目标阈值 eps_c < 5%, R_conf < 1e-2, **未放宽**。")

    # ---------------- 守卫: 先登记 (既有登记块的逐字副本) ----------------
    print("\n  ── 守卫登记 (可证伪, 汇成第一层最后一项的通过率) ──")
    record_guard('G1 L2 谱退化', 'derive_L2_entanglement_gap',
                 f"非零 Schmidt 值个数 {len(s2['probabilities'])} >= 2",
                 len(s2['probabilities']) >= 2, expect_pass=True,
                 note='此处原是硬 raise; 现补成可读的判定。')
    record_guard('G2 L4 纠缠熵为正', 'derive_L4_bond_dimension',
                 f"S(L/2) = {s2['entropy']:.6f} > 0",
                 float(s2['entropy']) > 0, note='此处原是硬 raise。')
    record_guard('G3 L6 网格足够', 'derive_L6_grid',
                 f"N = {int(l6['N'])} >= 8", int(l6['N']) >= 8,
                 note=f"此处原是硬 raise; 派生 N={l6['N']} "
                      f"(N_req={l6['N_req']}, N_cap={l6['N_cap']})。")
    record_guard('G4 Gray-Scott 数值稳定', 'stage6_life',
                 f"dt*Du/h^2 = {s6['stab']:.4f} <= {KNOBS['stability_limit']}",
                 bool(s6.get('ok')) and s6['stab'] <= KNOBS['stability_limit'],
                 note='失稳不会给出"更剧烈的斑图", 只会给出棋盘状假斑图。')

    vd = loop['verdicts']
    record_guard('V1 中心荷趋于 1/2', 'spiral_loop',
                 f"c_last = {loop['rows'][-1]['c']:.4f}, |c-0.5| < 0.05",
                 vd['c_toward_half'])
    record_guard('V2 尺度比 xi/L 稳定', 'spiral_loop',
                 f"ptp(xi/L) < 15% * mean  (末圈 xi/L = "
                 f"{loop['rows'][-1]['xi_over_L']:.4f})", vd['ratio_stable'],
                 note='只有无量纲量才配当收敛判据 —— xi 的绝对值随 L 发散是'
                      '临界的定义, 不是失败。')
    record_guard('V3 相对谱隙闭合', 'spiral_loop',
                 f"1-r: {1 - loop['rows'][0]['gap']:.4f} -> "
                 f"{1 - loop['rows'][-1]['gap']:.4f} (下降)",
                 vd['relgap_closing'])
    record_guard('V4 尺度被 max_L 截断', 'spiral_loop',
                 f"L 触顶 {KNOBS['max_L_loop']}", vd['scale_clamped'],
                 note='诚实为真: 去掉上限 L 会一直涨 —— 那正是临界性, 不是闭环失败。')
    record_guard('V5 无量纲量逐圈收敛', 'spiral_loop',
                 f"第 {vd['converged_at_turn']} 圈起变化 < {KNOBS['loop_tol']:.0%}",
                 vd['converged_at_turn'] is not None)
    record_guard('V6 权重自由变体秩1化', 'selfref_coupled',
                 f"变体A sigma1/sigma2 = {s3c['free']['sigma_ratio']:.3e} > 1e3",
                 bool(s3c['free']['sigma_ratio'] > 1e3), expect_pass=False,
                 note='**设计上就该失败**: 派生 N 之后变体 A 收敛到 x*=0, '
                      '权重失去方向。这是诊断项, 不计入通过率 —— 否则会假装'
                      '把一条真实的负面结果算成"没通过"。')
    record_guard('V7 权重归一变体秩1化', 'selfref_coupled',
                 f"变体B sigma1/sigma2 = {s3c['norm']['sigma_ratio']:.3e} > 1e3",
                 bool(s3c['norm']['sigma_ratio'] > 1e3))
    record_guard('V8 生命斑图涌现', 'stage6_life',
                 f"对比度 {s6['contrast']:.3f} > 0.2 且活化面积 "
                 f"{s6['active_frac']:.3f} > 0.05", bool(s6['emerged']))

    curv_val = s5['validation']
    c_max = 0.0
    for r in curv_val['cyclic']:
        c_max = max(c_max, float(r['max_abs']))
    for r in curv_val['complete']:
        c_max = max(c_max, abs(float(r['mean']) - float(r['exact'])))
    for r in curv_val['hypercube']:
        c_max = max(c_max, abs(float(r['mean_plain']) - float(r['exact_plain'])),
                    abs(float(r['mean_lazy']) - float(r['exact_lazy'])))
    cf = curv_val['forman']
    c_max = max(c_max, abs(float(cf['path_interior'])),
                abs(float(cf['k4_no_tri']) + 2.0),
                abs(float(cf['regular3_mean']) + 2.0),
                abs(float(cf['cycle_max_abs'])))
    record_guard('G5 曲率解析锚点', 'validate_curvature',
                 f"max|实测 - 解析| = {c_max:.2e} < 1e-6", c_max < 1e-6,
                 note='这条曾只把 max|.| 打印出来, 没有任何判定 —— '
                      '这是一个"有数字无判定"的洞, 现补成 bool。')

    # 新增守卫: 结论的稳健性前提
    record_guard('G6 最坏轨迹也达标', 'collect_metrics_v14',
                 f"{len(runs)} 条轨迹 max eps_c = "
                 f"{max(r['eps_c'] for r in runs):.3%} < {EPS_T:.0%}",
                 max(r['eps_c'] for r in runs) < EPS_T,
                 note='若这条为假, 1.1 的"达标"只是在某个种子上运气好。')
    record_guard('G7 最优检查点非首步',
                 'mera_fit_v13(overlap 选择)',
                 f"全部轨迹的选定 step > 0",
                 all(r['chosen_step'] > 0 for r in runs),
                 note='最佳检查点选择必须真的移动过; 若全停在 step 0, '
                      '说明选择逻辑坏掉, 报出的会是初态的随机值。')

    # ---------------- 第一层 ----------------
    print("\n  ── 第一层 · 内部自洽性 ──")
    fit_exact = fit_central_charge(gs_exact, L_chain)
    m11 = metric_central_charge_error_seeds(runs, chi_dev, L_chain)
    m12 = metric_conformal_invariance_seeds(
        runs, fit_exact['rms'] / float(np.mean(fit_exact['S'])))
    m13 = metric_gap_closure(loop['rows'])
    m14 = metric_mera_consistency(s4['isometry'], s4['cone']['r2_L32'])
    m15 = metric_design_robustness(runs, chi_dev, L_chain)
    m1d = metric_budget_diagnostic(v13['curve'], v13['steps'], chi_dev)
    m1e = metric_chi_bottleneck(v13['chi_sweep'], chi_dev, v13['floor_chi'])
    print(f"      中心荷误差 {m11['median']:.4f} [{m11['min']:.4f}, {m11['max']:.4f}] "
          f"| 共形不变性 {m12['median']:.3e} [{m12['min']:.3e}, {m12['max']:.3e}] "
          f"(精确态地板 {m12['R_conf_exact']:.3e})")
    print(f"      谱隙闭合残差 {m13['resid']:.2e} (a={m13['a']:.4f}) "
          f"| MERA 一致性 err={m14['err']:.2e}, R²_lin={m14['r2_linear']:.4f}")
    print(f"      全程稳健性 max eps_c={m15['max_eps_c']:.3%}, "
          f"max R_conf={m15['max_R_conf']:.3e} ({m15['n_runs']} 条轨迹)")

    # ---------------- 第二层 (口径原样) ----------------
    print("\n  ── 第二层 · 微观物理对标 (与解析解硬连接) ──")
    m21 = metric_chain_rmse(s3a, s3a['physical'])
    m22 = metric_entropy_kl(psi_main, gs_exact, L_chain)
    m23 = metric_correlation_exponent(gs_exact, L_chain)
    m24 = metric_jordan_peak(s3a['transient'])
    print(f"      因果链 RMSE {m21['rmse']:.3e} | 纠缠谱 KL {m22['kl']:.3e} "
          f"(JS {m22['js']:.3e}) | 关联指数 2Δ={m23['two_delta']:.4f} "
          f"(精确 0.25, 朴素幂律 {m23['eta_naive']:.4f}) | Jordan 峰值 "
          f"k={m24['k_meas']} vs {m24['k_theory']:.2f}")

    # ---------------- v14·F2a 负对照 (h/J 扫描) ----------------
    # 放在第二层: 它检验的是"阶段二的临界性是不是物理事实", 属于对解析解对标。
    m25 = metric_hj_negative_control(ctx.get('hj'))

    # ---------------- v14·F1 跨边界条件中心荷 (同族) ----------------
    # 也放在第二层: 与 F2 同属"对解析解对标", 检验的是读数的稳健性。
    m26 = metric_f1_cross_boundary(ctx.get('f1'))

    # ---------------- v14·F5 一致性检查束 (实现 <-> 解析式) ----------------
    # 也放在第二层: 与 F1/F2 同属"对标解析物", 但对标的是**解析式**而非物理真值。
    m27 = metric_consistency_checks(ctx.get('f5'))

    # ---------------- v14·F6 涌现几何的对照鲁棒性 ----------------
    # 也放在第二层: 它的参照物是三类解析定义的零模型族, 不是实验数据。
    m28 = metric_geometry_robustness(ctx.get('s5'))

    # ---------------- 第三层 ----------------
    m3 = tier3_structural_ladder(s6, os.path.join(_HERE, 'data'),
                                 model_params={'F': KNOBS['F'],
                                               'k': KNOBS['k']})

    # ---------------- 体检报告 ----------------
    print_health_report()
    return {'tier1': {'central_charge_error': m11, 'conformal_invariance': m12,
                      'gap_closure': m13, 'mera_consistency': m14,
                      'design_robustness': m15,
                      'budget_diagnostic': m1d, 'chi_bottleneck': m1e,
                      'curve': v13['curve'],
                      'curves_by_seed': v13.get('curves_by_seed', {}),
                      'budget': {'steps': v13['steps'], 'lrs': v13['lrs'],
                                 'chi': chi_dev,
                                 'floor_chi': v13['floor_chi'],
                                 'floor_chi_ctrl': v13['floor_chi_ctrl'],
                                 'chi_ctrl': v13['chi_ctrl']},
                      'runs_summary': [
                          {k: r[k] for k in ('lr', 'seed', 'eps_c', 'R_conf',
                                             'overlap', 'chosen_step')}
                          for r in runs]},
            'tier2': {'chain_rmse': m21, 'entropy_kl': m22,
                      'correlation_exponent': m23, 'jordan_peak': m24,
                      'hj_negative_control': m25,
                      'f1_cross_boundary': m26,
                      'consistency_checks': m27,
                      'geometry_robustness': m28},
            'tier3': m3,
            'guards': guard_pass_rate()}


# ===========================================================================
# 指标层的 20 面板图
# ===========================================================================
def plot_metrics_v14(metrics, s2, s3a, out_png):
    """
    指标层专属的 20 面板图, 与 15 面板物理图完全分离。

    既有 plot_metrics 读 tier1[...]['eps_c'] 与 tier3['pk'] —— 本层的指标
    结构变了 (1.1/1.2 变成分布, 第三层换了物理对象), 所以那 4 个面板在新结构下
    会直接抛 KeyError。这里不是"顺手重画", 是本层真的需要自己的面板。

    为什么从 9 格加到 20 格:
      原先是 3x3 的 9 格 (第一层 1 + 第二层 3 + 第三层 4 + 守卫 1)。本层第一层
      自己就要 4 格 (1.1/1.2 分布、1.6 预算漂移、1.7 chi 归因), 照 9 格的预算,
      第二层和第三层各让出两格 —— 结果 2.3 关联衰减指数与 2.4 Jordan 块
      被挤掉了。那两张图**各自是一条独立断言的唯一可视化**, 而且第二层一字未改,
      正是"没有回归"的证据。所以正确的做法是加格子而不是砍:

       1  第一层达标度条形 (5 项, 含「全程稳健性」)
       2  收敛曲线 eps_c(step): 各 seed + 中位数 + 600 步位置       <- 核心新证据
       3  多轨迹分布 vs 目标线 + 单点基线                            <- 核心新证据
       4  chi 归因: 实测 eps_c(chi) vs Schmidt 截断参照
       --- 第二层 (原样保留, 4/4) ---
       5  因果链拟合 RMSE: 理论 |lam2/lam1| vs 实测收敛比
       6  纠缠谱 (MERA 代表态 vs 精确基态)
       7  关联衰减指数: 共形形式 vs 朴素幂律
       8  Jordan 块瞬态增长: 谱半径<1 但范数先涨
       --- 第三层 (降级为方法学展示) ---
       9  A 段: 描述子逐项距离 vs 种子间散布 (噪声底)
      10  A 段: 归一化谱形 (k/k_pk) 模型 vs CLIP 三种子
      11  A 段: 两个混杂因子 (波长数 / 相区参数) —— 为什么距离不能被归因
      12  B1 段: Southampton 波峰周期与 CV
      13  B1 段: 独立交叉检验 —— 强度序列自相关在四个滞后上的系数
      14  B2 段: Reading 功率谱 —— "未检出"这个结论所依据的谱与阈值
      15  B2 段: Reading 逐帧亮度 —— 谱为什么是红的 (慢漂移主导)
      16  守卫通过率
       --- A 段参考侧实际场 ---
      17  A 段参考侧 seed 0 的实际场
      18  A 段参考侧 seed 1 的实际场   (自建种子家族, 少数相恒为 0.5)
      19  A 段参考侧 seed 2 的实际场
      20  A 段模型侧 36² 的实际场 (对照)

    13~15 三格把"报一个 verdict"变成"把判决依据亮出来"。一个"未检出周期"的
    结论, 读者有权看到那条谱和那条阈值线自己复核。

    17~20 四格把参考侧的实际场摆出来, 用途是让读者**看见**参考家族是不是斑图。
    A1 曾以这 3 个种子的 lam_pk 中位数为据声称"两个盒子内波长数差一个量级",
    实测那条论断来自坏参考场的谱角点伪像, 已撤回 —— 把场摆出来, 这个缺口
    本是可见的, 只是当时没往归档管线上想。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    def _short(s, n=20):
        s = str(s).replace('_', ' ')
        return s if len(s) <= n else s[:n - 1] + '…'

    # 网格按"每条独立断言至少占一格"排: 第一层 4 格 (分布口径), 第二层含
    # 2.3/2.4 —— 那两张图各自是一条独立断言的唯一可视化 (2.3 是"共形形式 vs
    # 朴素幂律"的对照, 2.4 是"谱半径<1 但范数先涨"这个非正规性陷阱), 二者
    # 一旦被别的面板挤掉, 对应的断言就只剩一个数字。第二层**一字未改**, 正是
    # "没有回归"的证据。所以不是砍掉它们, 是把格子加大。
    fig, axes = plt.subplots(5, 4, figsize=(25, 25))  # 5x4: 含 A 段第 5 行
    axes = np.asarray(axes)
    t1 = metrics['tier1']
    bud = t1['budget']

    # ---- 1. 第一层达标度条形 ----
    ax = axes[0, 0]
    dr = t1['design_robustness']
    items = [
        ('中心荷误差\n(最坏轨迹)', t1['central_charge_error']['max'], 5e-2,
         t1['central_charge_error']['passed']),
        ('共形不变性\n(最坏轨迹)', t1['conformal_invariance']['max'], 1e-2,
         t1['conformal_invariance']['passed']),
        ('全程稳健性\n(1.1 且 1.2 同时)', max(dr['max_eps_c'] / 5e-2,
                                        dr['max_R_conf'] / 1e-2), 1.0,
         dr['passed']),
        ('MERA 一致性', t1['mera_consistency']['err'], 1e-12,
         t1['mera_consistency']['passed']),
        ('谱隙闭合残差', t1['gap_closure']['resid'], 5e-3,
         t1['gap_closure']['passed']),
    ]
    names = [i[0] for i in items]
    ratios = [float(i[1]) / i[2] for i in items]
    cols = ['seagreen' if i[3] else 'crimson' for i in items]
    ax.barh(names, ratios, color=cols)
    ax.axvline(1.0, color='k', ls='--', lw=1.2)
    ax.set_xscale('log')
    ax.set_xlabel('实测 / 目标  (对数轴; < 1 即达标)')
    ax.set_title('一 · 内部自洽性\n条形长度 = 实测值相对目标; 前两项取最坏轨迹')
    for i, r in enumerate(ratios):
        ax.text(r, i, f"  {r:.2g}x", va='center', fontsize=8)

    # ---- 2. 收敛曲线 ----
    ax = axes[0, 1]
    for seed, cv in sorted(t1.get('curves_by_seed', {}).items(),
                           key=lambda kv: int(kv[0])):
        if not cv:
            continue
        cv = np.asarray(cv, dtype=float)
        ax.plot(cv[:, 0], cv[:, 1] * 100, lw=0.9, alpha=0.55,
                label=f'seed {seed}')
    med = np.asarray(t1['curve'], dtype=float)
    if med.size:
        ax.plot(med[:, 0], med[:, 1] * 100, 'k-', lw=2.3,
                label=f'中位数 (以上 {len(t1.get("curves_by_seed", {}))} 条)')
    ax.axhline(5.0, color='crimson', ls='--', lw=1.3, label='目标 5%')
    ax.axhline(11.45, color='gray', ls=':', lw=1.3, label='已发布基线 11.45%')
    ax.axvline(600, color='gray', lw=0.9, alpha=0.7)
    ax.annotate('已发布基线止步于此\n(600 步)', xy=(600, 30), fontsize=8,
                color='gray')
    ax.set_yscale('log')
    ax.set_xlabel('变分步数')
    ax.set_ylabel('eps_c  [%]')
    n_cv = len(t1.get('curves_by_seed', {}))
    other_lr = (f'lr={bud["lrs"][1]} 的 {len(t1["runs_summary"]) - n_cv} 条'
                f'未逐步记录, 只出现在面板三\n' if len(bud['lrs']) > 1 else '')
    ax.set_title(f'二 · 收敛曲线 (lr={bud["lrs"][0]}, chi={bud["chi"]}, '
                 f'{n_cv} 个种子)  ★核心证据\n'
                 + other_lr
                 + f'预算翻倍时 eps_c 相对变化 = '
                   f'{t1["budget_diagnostic"]["drift"]:+.1%}')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    # ---- 3. 多轨迹分布 ----
    ax = axes[0, 2]
    for lr, mk in zip(bud['lrs'], ('o', 's')):
        rs = [r for r in t1['runs_summary'] if r['lr'] == lr]
        ax.semilogy([r['eps_c'] for r in rs], [r['R_conf'] for r in rs], mk,
                    ms=9, alpha=0.85, label=f'lr={lr} ({len(rs)} 条)')
        for r in rs:
            ax.annotate(f"s{r['seed']}", (r['eps_c'], r['R_conf']),
                        fontsize=7, xytext=(3, 3), textcoords='offset points')
    ax.axvline(5e-2, color='crimson', ls='--', lw=1.3)
    ax.axhline(1e-2, color='crimson', ls='--', lw=1.3)
    ax.axvline(0.1145, color='gray', ls=':', lw=1.3)
    ax.axhline(t1['conformal_invariance']['R_conf_exact'], color='steelblue',
               ls='-.', lw=1.2, label='有限尺寸地板')
    ax.set_xscale('log')
    ax.set_xlabel('eps_c  (目标 < 5%)')
    ax.set_ylabel('R_conf  (目标 < 1e-2)')
    ax.set_title('三 · 多轨迹分布 (判据卡最坏轨迹)\n'
                 '虚线 = 目标; 点线 = 已发布基线 eps_c')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    # ---- 4. chi 归因 ----
    ax = axes[0, 3]
    chis = [bud['chi'], bud['chi_ctrl']]
    meas = [t1['chi_bottleneck']['measured'][str(c)] if str(c)
            in t1['chi_bottleneck']['measured']
            else t1['chi_bottleneck']['measured'][c] for c in chis]
    floors = [bud['floor_chi'], bud['floor_chi_ctrl']]
    xpos = np.arange(len(chis))
    ax.bar(xpos - 0.18, [m * 100 for m in meas], 0.36, color='steelblue',
           label='MERA 实测')
    ax.bar(xpos + 0.18, [f * 100 for f in floors], 0.36, color='lightgray',
           label='Schmidt 截断参照 (非严格界)')
    ax.axhline(5.0, color='crimson', ls='--', lw=1.3, label='目标 5%')
    ax.set_xticks(xpos)
    ax.set_xticklabels([f'chi={c}' for c in chis])
    ax.set_ylabel('eps_c  [%]')
    ax.set_title('四 · chi 瓶颈归因 (同种子同预算, 只改 chi)\n'
                 f'{t1["chi_bottleneck"]["verdict"]}')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, axis='y')

    # ---- 5. 因果链拟合 RMSE (2.1) ----
    ax = axes[1, 0]
    cr = metrics['tier2']['chain_rmse']
    pr = np.atleast_2d(np.asarray(cr.get('pairs', []), dtype=float))
    if pr.size and pr.shape[1] == 2:
        th, me = pr[:, 0], pr[:, 1]
        lo = float(min(th.min(), me.min()))
        hi = float(max(th.max(), me.max()))
        pad = 10 ** (0.08 * (np.log10(hi) - np.log10(lo) + 1e-9))
        ax.loglog([lo / pad, hi * pad], [lo / pad, hi * pad], 'k--', lw=1.3,
                  label='理想: 实测 = 理论 (y=x)')
        ax.loglog(th, me, 'o', ms=10, mfc='none', mec='navy', mew=2,
                  label=f'{len(th)} 组受控谱隙')
        for t, m in zip(th, me):
            ax.annotate(f'{abs(t - m):.1e}', (t, m), fontsize=7, color='navy',
                        xytext=(6, -3), textcoords='offset points')
        ax.set_xlim(lo / pad, hi * pad)
        ax.set_ylim(lo / pad, hi * pad)
    ax.set_xlabel('理论收敛比  |lam2/lam1|  (矩阵特征值直接算出)')
    ax.set_ylabel('实测收敛比 (幂迭代末段步长比中位数)')
    ax.set_title(f'五 · 微观物理 · 因果链拟合 RMSE (2.1)\n'
                 f'RMSE = {cr["rmse"]:.3e}  (目标 < 1e-2, 与已发布基线同阈值)')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, which='both')

    # ---- 6. 纠缠谱 KL (2.2) ----
    ax = axes[1, 1]
    kl = metrics['tier2']['entropy_kl']
    spar = np.asarray(s2['spectrum'])
    ax.semilogy(np.arange(1, len(spar) + 1), np.maximum(spar, 1e-18), 'o-',
                ms=3, label=f'精确基态 Schmidt 谱 ({len(spar)} 个奇异值)')
    ax.set_xlabel('Schmidt 指标 n')
    # 注: s2['spectrum'] 是**奇异值** s_n 本身 (spiral_model_v13.py:638 的 SVD 输出,
    # 没有平方)。第一版这里写成 s_i^2 是笔误, 竖轴与横轴对不上 —— 已改回 s_n。
    ax.set_ylabel('奇异值  s_n')
    ax.set_title(f'六 · 微观物理 · 纠缠熵 KL (2.2)\n'
                 f'KL(MERA‖精确) = {kl["kl"]:.3e} nats, JS = {kl["js"]:.3e}\n'
                 f'支撑 MERA {kl["n_p"]} / 精确 {kl["n_q"]}, '
                 f'共同 {kl["n_common"]} (KL 只在交集上求和)')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    # ---- 7. 关联衰减指数 (2.3) ----
    # 为什么必须画出: 这是"共形形式 (π/L)/sin(πr/L) vs 朴素幂律 r"这个对照的
    # 唯一可视化, 而已发布基线的"朴素幂律给 η=0.205、共形形式给 0.2426"正是
    # 那条诚实边界的依据。只报一个数字看不出形式选错有多大代价。
    ax = axes[1, 2]
    ce = metrics['tier2']['correlation_exponent']
    L16 = 16
    _, gs16 = exact_ground_state(L16, 1.0, 1.0)
    _, C16, _ = boundary_correlation_graph(gs16, L16)
    rs = np.arange(1, L16 // 2 + 1)
    y16 = np.abs(np.array([C16[0, r] for r in rs]))
    f16 = (np.pi / L16) / np.sin(np.pi * rs / L16)
    good = y16 > 1e-12
    ax.loglog(f16[good], y16[good], 'ko', ms=5, label='|C(r)| 精确 (L=16)')
    if good.sum() >= 2:
        sl, ic = np.polyfit(np.log(f16[good]), np.log(y16[good]), 1)
        ff = np.linspace(np.log(f16[good].min()), np.log(f16[good].max()), 50)
        ax.loglog(np.exp(ff), np.exp(ic + sl * ff), 'r-', lw=1.5,
                  label=f'共形形式拟合 斜率={sl:.4f}\n(→ 2Δ={ce["two_delta"]:.4f}, '
                        f'精确 0.25, 相对误差 {ce["rel"]:.1%})')
    ax.loglog(rs, y16, 'b--', lw=1.0,
              label=f'朴素幂律对照 η={ce["eta_naive"]:.4f}')
    ax.set_title('七 · 微观物理 · 关联衰减指数 (2.3)\n'
                 '共形形式 vs 朴素幂律 (后者在有 PBC 的环上必然偏,\n'
                 '这是形状效应不是数值误差)')
    ax.set_xlabel('f = (π/L)/sin(πr/L)  [实心点],   r  [虚线]')
    ax.set_ylabel('|⟨σᶻ₀σᶻ_r⟩|')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, which='both')
    # loglog 上 matplotlib 会给次刻度也标数字 (范围 2e-3~7e0), 在 625 px 宽的
    # 格子里这些标签互相压在一起完全读不出来。只保留主刻度标签。
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # ---- 8. Jordan 块瞬态增长 (2.4) ----
    # 为什么必须恢复: 这是"谱半径 0.90 < 1 却先涨 272 倍"这个非正规性陷阱的
    # 唯一可视化。只看谱半径会得出"收敛"的结论, 而范数曲线直接反驳它。
    ax = axes[1, 3]
    tr = s3a['transient']
    jp = metrics['tier2']['jordan_peak']
    kk = np.arange(1, len(tr['growth']) + 1)
    ax.semilogy(kk, tr['growth'], 'o-', ms=3, color='navy',
                label=f"‖A^k‖₂ 实测峰在 k={jp['k_meas']}")
    ax.axvline(jp['k_theory'], color='crimson', ls='--', lw=1.5,
               label=f"理论 k* = (N−1)/|ln λ| = {jp['k_theory']:.2f}\n"
                     f"相对误差 {jp['rel']:.2%} (目标 < 5%)")
    ax.set_title(f"八 · 微观物理 · Jordan 块瞬态增长 (2.4)\n"
                 f"N={tr['N']}, λ={tr['lam']}, 谱半径 "
                 f"{tr['spectral_radius']:.2f} < 1 但范数先涨后落\n"
                 f"— 非正规性的直接证据, 不是数值错误")
    ax.set_xlim(0, 1.06 * max(len(kk), float(jp['k_theory'])))
    ax.set_xlabel('迭代步 k')
    ax.set_ylabel('‖A^k‖₂')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    # ---- 9. A 段描述子距离 vs 噪声底 ----
    ax = axes[2, 0]
    a = metrics['tier3'].get('_plot', {}).get('A')
    if a:
        per = metrics['tier3']['A1']['per_key']
        ks = list(a['keys'])
        ax.bar(ks, [per[k] for k in ks], color='steelblue')
        ax.axhline(a['scatter'], color='darkorange', ls='--', lw=1.5,
                   label=f'种子间散布 (噪声底) = {a["scatter"]:.4f}')
        ax.axhline(3 * a['scatter'], color='crimson', ls='--', lw=1.3,
                   label=f'判据上限 3x = {3 * a["scatter"]:.4f}')
        ax.set_ylabel('逐项距离 d_i')
        ax.set_title(f'九 · 第三层 A 段 · 描述子距离 (36^2 vs 100^2)\n'
                     f'D_struct = {a["D"]:.4f}  (不设达标线, 见十一)')
        ax.legend(fontsize=7)
        ax.tick_params(axis='x', rotation=30)
    else:
        ax.set_title('九 · A 段\n(第三层跳过)')
    ax.grid(alpha=0.25, axis='y')

    # ---- 10. A 段归一化谱形 ----
    ax = axes[2, 1]
    if a:
        pk = a['pk']
        ax.loglog(pk['xr'], pk['yr'], 'o-', ms=3, color='darkorange',
                  label='CLIP GS (3 种子均值)')
        ax.loglog(pk['xm'], pk['ym'], 's--', ms=3, color='steelblue',
                  label='阶段六 GS (36^2)')
        ax.set_xlabel('k / k_peak')
        ax.set_ylabel('Delta^2 (归一到积分 1)')
        ax.set_title(f'十 · 第三层 A 段 · 归一化谱形\n'
                     f'R_P = {a["R_P"]:.4f}   (已发布基线同量是 1.277,\n'
                     f'当年是拿 2D GS 比 3D CDM 晕谱)')
        ax.legend(fontsize=7)
    else:
        ax.set_title('十 · A 段谱形\n(第三层跳过)')
    ax.grid(alpha=0.25, which='both')

    # ---- 11. A 段: 两个尺度估计器对不上 -> 混杂因子 (2) 已撤回 ----
    # 【本格已改写】原版画的是"模型 4.1 vs 参考 68.9 个波长",
    # 用来支撑混杂因子 (2)。后查明参考侧那个 68.9 是 _char_scale 的
    # argmax 抓到二维离散谱**角点** (λ=√2=1.414 格, 网格能表示的最短波长)
    # 造成的伪像 —— 1 格的二值台阶, 不是斑图波长。
    # 所以这一格改成把两个**互相独立**的尺度估计器并排画出来:
    #   lam_pk = 谱峰 argmax    (会被高频壳骗, 参考侧失效)
    #   xi_ac  = 自相关首过零  (零阶量, 不挑壳, 可作交叉核对)
    # 模型侧两者差 2.2 倍 (正常), 参考侧差 29 倍 (估计器失效)。这比原来的
    # 单估计器柱图诚实: 它把"为什么这个数是伪像"直接画出来了。
    ax = axes[2, 2]
    a1 = metrics['tier3'].get('A1')
    if a1 and a.get('lam_mod') is not None:
        _lm, _lr = float(a['lam_mod']), float(a['lam_ref'])
        _lc = float(a.get('lam_corner_cells', float('nan')))
        _xm = float(a1.get('xi_ac_model', float('nan')))
        _xr_list = [float(v) for v in (a1.get('xi_ac_ref') or []) if v == v]
        _xr = float(np.median(_xr_list)) if _xr_list else float('nan')
        _w = 0.36
        ax.bar([-_w / 2, 1 - _w / 2], [_lm, _lr], _w, color='steelblue',
               label='lam_pk (谱峰 argmax)')
        ax.bar([_w / 2, 1 + _w / 2], [_xm, _xr], _w, color='seagreen',
               label='xi_ac (自相关首次过零)')
        ax.axhline(_lc, ls='--', lw=1.2, color='crimson')
        # 角点线**不写图内文字**: 这张图几乎每个位置都有柱子, 文字压上去
        # 不是被柱子盖住就是被右边缘切掉 (成图上两种都实测遇到过)。
        # 改成一个 Line2D 图例项, 信息不丢, 也不和任何东西重叠。
        _cnr = plt.Line2D([], [], ls='--', lw=1.2, color='crimson',
                          label=f'网格角点 {_lc:.3f} 格 (最短可表示波长)')
        _h, _l = ax.get_legend_handles_labels()
        ax.legend(_h + [_cnr], _l + [_cnr.get_label()], fontsize=6.5,
                  loc='upper left', framealpha=0.92)
        for _xi, _v in zip([-_w / 2, _w / 2, 1 - _w / 2, 1 + _w / 2],
                           [_lm, _xm, _lr, _xr]):
            ax.text(_xi, _v * 1.06, f'{_v:.2f}', ha='center', va='bottom',
                    fontsize=8)
        ax.set_yscale('log')
        # 留出顶部余量, 否则最高那根柱顶到坐标框、数值标签飘到框外
        ax.set_ylim(bottom=max(0.3, _lc * 0.22),
                    top=max(_lm, _xr) * 1.9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f'模型 36^2\n{a1["params_model"]}',
                            f'参考 100^2\n{a1["params_ref"]}'], fontsize=7.5)
        ax.set_ylabel('特征尺度 (格)')
        ax.set_title(f'十一 · 第三层 A 段 · 两个尺度估计器对不上\n'
                     f'模型 {_lm / max(_xm, 1e-9):.1f}x (正常), 参考 '
                     f'{_xr / max(_lr, 1e-9):.0f}x ⇒ 参考侧 lam_pk 是伪像\n'
                     f'混杂因子 (2) 已撤回, 只剩 (1) 相区参数不同')
        ax.grid(alpha=0.25, axis='y', which='both')
    else:
        ax.set_title('十一 · A 段混杂因子\n(第三层跳过)')

    # ---- 12. B1 段: Southampton 波峰周期 (数值列, 非图版) ----
    ax = axes[2, 3]
    b = metrics['tier3'].get('_plot', {}).get('B')
    if b and b.get('peaks'):
        names = sorted(b['peaks'])
        pers = [b['peaks'][n]['period_s'] for n in names]
        cvs = [b['peaks'][n]['dt_cv'] for n in names]
        cols = ['#2b7bba' if c < 0.5 else '#d1653a' for c in cvs]
        ax.bar(range(len(names)), pers, color=cols)
        for i, (p, c) in enumerate(zip(pers, cvs)):
            ax.text(i, p, f'{p:.1f}s\nCV={c:.2f}', ha='center', va='bottom',
                    fontsize=7)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel('波峰间隔中位数 (s)')
        ax.set_ylim(0, max(pers) * 1.35)
        ax.set_title('十二 · 第三层 B1 段 · BZ 波峰周期\n'
                     '来源 = 作者的波峰标记数值列 (不经 .tif 图版)\n'
                     '橙 = CV>0.5, 是该记录的真实性质而非流程失败')
        ax.grid(alpha=0.25, axis='y')
    else:
        ax.set_title('十二 · B 段\n(第三层跳过)')

    # ---- 13. B1 段的独立交叉检验 (两层提取是否指向同一个 T) ----
    # 这是第三层作为"方法学展示"真正要交付的东西: 波峰间隔统计与强度序列
    # 自相关是两套**互相独立**的提取 (前者来自作者标注的峰位, 后者来自原始
    # 强度列, 不经过任何波峰检测)。把四个滞后的系数全画出来, 读者才能自己
    # 核对"极大落在哪个滞后", 而不是接受一句 verdict。
    ax = axes[3, 0]
    lagk = ['T/2', 'T', '1.5T', '2T']
    if b and b.get('peaks'):
        names = sorted(b['peaks'])
        pal = ['#2b7bba', '#d1653a', '#3f8f5f', '#8a5fb0']
        xs = np.arange(len(lagk))
        w = 0.8 / max(len(names), 1)
        any_ac = False
        for i, nm in enumerate(names):
            pk = b['peaks'][nm]
            acs = pk.get('ac')
            if not acs:
                continue
            any_ac = True
            vals = [float(acs.get(k, np.nan)) for k in lagk]
            ax.bar(xs + (i - (len(names) - 1) / 2) * w, vals, w,
                   color=pal[i % len(pal)],
                   label=f"{nm}  (CV={pk['dt_cv']:.2f}, 极大在 {pk.get('lag_best')})")
        if any_ac:
            ax.set_xticks(xs)
            ax.set_xticklabels(lagk)
        ax.axhline(0.0, color='k', lw=0.8)
    ax.set_xlabel('滞后 (以波峰间隔中位数 T 为单位)')
    ax.set_ylabel('强度序列自相关系数 ac(τ)')
    ax.set_title('十三 · 第三层 B1 段 · 独立交叉检验\n'
                 '波峰标记 vs 强度序列自相关 —— 两套独立提取\n'
                 '三个记录的"极大位置"排序与 CV 排序完全一致')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=0.25, axis='y')

    # ---- 14. B2 段: Reading 功率谱 —— "未检出"是个结论, 把证据画出来 ----
    ax = axes[3, 1]
    rd = (b or {}).get('reading', {})
    if rd:
        pal = ['#2b7bba', '#d1653a', '#3f8f5f', '#8a5fb0']
        kmin = None
        for i, (nm, v) in enumerate(sorted(rd.items())):
            pw = np.asarray(v['power'], dtype=float)
            kk = np.arange(1, len(pw) + 1)
            kmin = v['min_cycles']
            c = pal[i % len(pal)]
            kb, pb = _log_bin(kk, pw, 14)
            ax.loglog(kb, pb, lw=1.3, color=c,
                      label=f"{_short(nm)}: 带内最大在 k={v['k_band_argmax']}, "
                            f"突出度 {v['prominence']:.0f}")
            ax.axhline(v['power_thresh'], color=c, ls=':', lw=1.0, alpha=0.85)
            ax.plot([v['k_band_argmax']], [pw[v['k_band_argmax'] - 1]], 'v',
                    color=c, ms=10)
        if kmin:
            ax.axvline(kmin, color='k', ls='--', lw=1.2,
                       label=f'频带下界 k={kmin} (以下被高通置零)')
            ax.set_xlim(kmin * 0.75, 600)
        ax.set_xlabel('谱线 k  (周期 = 2161 / k 帧)')
        ax.set_ylabel('Hann 窗后 |F|²  (对数分箱显示, 判定用未分箱谱)')
        ax.set_title('十四 · 第三层 B2 段 · Reading 功率谱 (高通后)\n'
                     '▼ = 带内最大值, 三条都恰好落在频带**下界** k=8 上;\n'
                     '点线 = 各自的突出度阈值 (局部背景中位数 x 300)\n'
                     '三条都**越过**了点线 —— 检不出不是因为峰不够突出,\n'
                     '而是因为 k=8 不是"带内内部的局部极大"。带下界随\n'
                     'min_cycles 变, 把边缘当峰等于把滤波器边界读成物理量')
        ax.legend(fontsize=6.5, loc='lower left')
    else:
        ax.set_title('十四 · B2 段 Reading 谱\n(第三层跳过)')
    ax.grid(alpha=0.25, which='both')

    # ---- 15. B2 段: Reading 逐帧亮度 + 预热段裁剪敏感性 ----
    ax = axes[3, 2]
    if rd:
        pal = ['#2b7bba', '#d1653a', '#3f8f5f', '#8a5fb0']
        nd_max = 0
        lo, hi = np.inf, -np.inf
        for i, (nm, v) in enumerate(sorted(rd.items())):
            y = np.asarray(v['lum'], dtype=float)
            x = np.arange(len(y)) * 4.0            # 已按 ::4 降采样
            nd_max = max(nd_max, int(v.get('n_dark', 0)))
            lo, hi = min(lo, float(y.min())), max(hi, float(y.max()))
            ax.plot(x, y, lw=0.9, color=pal[i % len(pal)], label=_short(nm))
        if nd_max:
            ax.axvspan(0, float(nd_max), color='k', alpha=0.10)
        span = max(hi - lo, 1e-9)
        ax.set_ylim(lo - 0.05 * span, hi + 0.30 * span)   # 顶部留白给图例
        ax.set_xlabel('帧  (三条长度均为 2161; 显示按 ::4 降采样)')
        ax.set_ylabel('逐帧平均亮度 (绝对灰度)')
        ax.set_title('十五 · 第三层 B2 段 · Reading 逐帧亮度\n'
                     '左端灰带 = 相机预热暗帧 (25.6 vs 稳态 84~115),\n'
                     '一段硬阶跃: 它给谱灌宽频功率, 把漂移占比**压小**')
        lines = ['裁剪敏感性  (漂移占比; trim = 掐掉头部帧数)']
        tot = tot_det = 0
        for nm, v in sorted(rd.items()):
            tr = v.get('trim_trials', [])
            tot += len(tr)
            tot_det += sum(1 for t in tr if t['detected'])
            fr = '  '.join(f"{t['trim']}:{t['drift_frac']:.0%}" for t in tr)
            lines.append(f"{_short(nm, 13):<13} {fr}")
        lines.append(f'合计检出 {tot_det}/{tot} => 结论不依赖预处理')
        ax.text(0.985, 0.02, '\n'.join(lines), transform=ax.transAxes,
                ha='right', va='bottom', fontsize=6.5,
                # family 必须是**列表**: 单调 'monospace' 会解析到 DejaVu Sans
                # Mono, 它没有 CJK 字形, 中文标签会全变成方块 (□□) —— 而这张
                # 表的标题正是中文。matplotlib >= 3.6 按字形逐个回退, 所以
                # 数字仍用等宽体保持列对齐, 中文回退到全局 CJK 字体。
                family=['monospace'] + list(plt.rcParams['font.sans-serif']),
                bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.92))
        ax.legend(fontsize=6, loc='upper right', framealpha=0.92)
    else:
        ax.set_title('十五 · B2 段 Reading 序列\n(第三层跳过)')
    ax.grid(alpha=0.25)

    # ---- 16. 守卫通过率 ----
    ax = axes[3, 3]
    g = metrics['guards']
    ax.bar(['应当通过', '诊断项\n(设计上就该失败)'],
           [g['rate'] * 100, (g['n_diag'] - g['n_diag_notok'])
            / max(g['n_diag'], 1) * 100],
           color=['seagreen', 'gray'])
    ax.set_ylim(0, 105)
    ax.axhline(100, color='k', ls='--', lw=1.0)
    ax.set_ylabel('通过率 [%]')
    ax.set_title(f'十六 · 守卫通过率\n{g["n_pass"]}/{g["n_expect"]} = '
                 f'{g["rate"]:.1%} (诊断项 {g["n_diag"]} 条单列, 不计入分母;\n'
                 f'分母 31→22 的拆解见 spiral_v13_说明.md §5)')
    for i, v in enumerate([g['rate'] * 100,
                           (g['n_diag'] - g['n_diag_notok'])
                           / max(g['n_diag'], 1) * 100]):
        ax.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=9)

    # ---- 17~20. A 段参考侧/模型侧实际场 (v14·F3 改写) ----
    # 这四张图让读者**看见**参考家族 (自建种子) 与模型侧的实际形态, 并同时
    # 看到两者的自相关长度 —— 参考侧是不是斑图, 自己看即可, 不必只听结论。
    # 参考侧只放前 PANEL_SEEDS 个 (网格只有 4 列 x 1 行可用), 全部 12 个种子的
    # 描述子在 result/_v14_data.json 的 A1 里。
    _fld = a.get('fields') if a else None
    if _fld:
        _order = [f'参考 seed {s}' for s in (a.get('panel_seeds') or [])] + ['模型']
        _order = [n for n in _order if n in _fld] + \
                 [n for n in _fld if n not in _order]
        _rom = ['十七', '十八', '十九', '二十']
        for _i, _nm in enumerate(_order[:len(_rom)]):
            _r, _c = 4 + _i // 4, _i % 4
            _ax = axes[_r, _c]
            _fi = _fld[_nm]
            _img = np.asarray(_fi['img'], dtype=float)
            _ax.imshow(_img, cmap='gray_r', vmin=0, vmax=1,
                       interpolation='nearest', origin='lower',
                       extent=(0, _fi['box'], 0, _fi['box']))
            _cn = (a.get('conn_model') if _nm.startswith('模型') else
                   (a.get('conn_ref', {}) or {}).get(
                       _nm.replace('参考 seed ', ''), None))
            _tag = (f', 少数相 {_cn["n_comp"]} 块, 平均 {_cn["mean_size"]:.0f} 格'
                    if _cn else '')
            _ax.set_title(f'{_rom[_i]} · A 段实际场\n{_nm}   {_fi["box"]}²  '
                          f'占空比 {_fi["phi"]:.4f} (中位数阈恒等式)',
                          fontsize=8.5)
            _ax.text(0.5, -0.10, f'自相关长度 {_fi["xi_ac"]:.1f} 格 = '
                                 f'{_fi["xi_ac"] / _fi["box"]:.2f} 盒长{_tag}',
                     transform=_ax.transAxes, ha='center', va='top', fontsize=7.5)
            # **不设 xlabel**: 它与下面那行说明文字都以 axes 的 x=0.5 居中,
            # 必然重叠 (成图上实测到的排版缺陷: 黑色的 '格' 压在
            # 红色说明文字上)。单位由 ylabel 和说明文字里的"格"给出。
            _ax.set_ylabel('格', fontsize=8)
            _ax.tick_params(labelsize=7)
        _lm = a.get('lam_mod', float('nan'))
        _lr = a.get('lam_ref', float('nan'))
        _lc = a.get('lam_corner_cells', float('nan'))
        _lleg = a.get('lam_legacy_cells', float('nan'))
        fig.text(0.5, 0.012,
                 f'A 段诊断 (v14·F3 改写, 诊断项不计入分母): '
                 f'参考侧已换为自建种子 (共 {a.get("n_seeds_total", "?")} 个, '
                 f'图中前 3 个), 两侧同用 (v > 中位数) 二值化。'
                 f'lam_pk 模型 {_lm:.2f} 格 vs 参考中位数 {_lr:.2f} 格 -> '
                 f'"盒子内波长数" {a.get("n_lambda_model", float("nan")):.1f} vs '
                 f'{a.get("n_lambda_ref", float("nan")):.1f} (但该量是盒子边长/lam_pk '
                 f'的换写, 不是独立测量)。\n'
                 f'**两条照着事实报**: (i) 旧口径用的归档 bool 场与同一归档的浮点场'
                 f'任何简单二值化都对不上 (逐格一致率≈0.5), 其生成管线不在归档里 '
                 f'-> 那个 lam_pk {_lleg:.2f} 格落在二维离散谱角点 '
                 f'{_lc:.3f} 格 (网格能表示的**最短**波长), 是那种场的 1 格台阶伪像, '
                 f'也是"波长数差一个量级"那条混杂因子的来源; '
                 f'(ii) _char_scale 取 Δ²(k) 在**离散壳**上的 argmax, 所以 lam_pk 只'
                 f'能取离散值, 且与**独立**的自相关尺度 4*xi_ac 系统性不一致 '
                 f'(中位偏差 {a.get("lam_pk_ac_dev_median", float("nan")):.0%}) —— '
                 f'它在本段给出的读数既不保证是斑图波长, 也没有独立佐证, '
                 f'不能当连续波长读。**未改 _char_scale**, 缺口如实记录。',
                 ha='center', va='bottom', fontsize=8.5,
                 bbox=dict(boxstyle='round', fc='#f5f8ff', ec='steelblue',
                           alpha=0.95))

    fig.suptitle('v14 量化指标层 · 20 面板 · 第一层分布口径 (多种子 + 预算) · '
                 '第二层原样 (4/4, 含 2.3/2.4) · '
                 '第三层降级 (2D 斑图 结构对结构, 不设达标线) · '
                 'F3: 参考侧换自建种子',
                 fontsize=14, y=0.996)
    fig.tight_layout(rect=(0, 0.035, 1, 0.982))
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    print(f"\n  指标图已保存: {out_png}")
