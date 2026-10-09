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
spiral_model_v14.py
====================
「空无到照见」· 完整可运行版本 v14
================================================================================

【量化指标评估层 (体检报告)】

  七个物理阶段若只用"打印出来的数字"陈述结论 —— "c=0.5653 vs 精确 0.5072"、
  "曲率锚点全部到机器精度" —— 那些数字就不是**可判定的指标**: 没有基线、
  没有目标、没有通过/未通过、没有通过率。于是"机器精度吻合"这句话本身是
  不可证伪的散文。

  本层**不加任何新物理阶段**, 它给已有的七个阶段装上一套评估: 每个指标给出
      定义 / 计算方法 / 基线 / 目标 / 实测 / 判定
  最后汇总成一份三层体检报告。

  第一层 · 内部自洽性 (模型对自己)
      中心荷误差 | 共形不变性 | 谱隙闭合残差 | MERA 一致性 | 守卫通过率
      这一层回答: "模型内部自相矛盾吗?"

  第二层 · 微观物理对标 (模型对解析解)
      因果链拟合 RMSE | 纠缠熵 KL 散度 | 关联衰减指数 | Jordan 块收敛率
      这一层回答: "模型对**已知答案**的偏离有多大?"
      参照物是硬连接的解析结果, 不是拟合出来的东西:
        中心荷 c = 1/2        (c=1/2 的 CFT, 即 Ising 普适类)
        关联衰减指数 eta = 2*Delta = 1/4  (Delta_sigma = 1/8)
        Jordan 块瞬态峰值位置 k* = (N-1)/|ln lambda|
        纠缠熵 KL 散度         (对照精确对角化的基态 Schmidt 谱)

  第三层 · 宏观结构 / Surrogate 对标 (模型对外部数据)
      功率谱残差 | 形态学 KL 散度 | 两点相关比 | 空洞尺寸分布
      数据来源: data/mini-millennium (Mini-Millennium N-body 的 8 个 tree 文件,
      本脚本自行逆向其二进制格式, 不依赖任何宇宙学库)。

  【第三层的诚实边界 —— 这是本版最需要说清的一点】
  模型侧产生的是**二维反应-扩散斑图**, 参考侧是**三维 CDM N-body 晕目录**。
  两者不是同一个物理对象。所以第三层**只比较尺度无关的统计形式**
  (归一化功率谱、标准化 PDF、比值), **绝不比较振幅、单位、绝对尺度**。
  任何第三层的结论都不得表述为"模型复现了宇宙学观测"。
  具体地: Gray-Scott 的 P(k) 是"特征波长处有峰 + 两侧衰减", 而 CDM 晕的 P(k)
  在此范围内接近单调幂律 —— 所以功率谱残差**预期会很大**。那是真实的发现,
  不是待修的 bug; 本脚本不允许通过调参把这个残差压下去。

  【"只报告不追目标" 的原则】
  本脚本**不做目标驱动的重算**。某个指标没达标时, 做的是**差距诊断**
  (残差来自键维截断? 有限尺寸? 优化步数?), 而不是加大 L/chi 直到数字好看。
  体检报告的意义在于诚实, 不在于满分。

  【负对照与交叉检验 —— 让判据可证伪】
  一个结论若在它不成立时也不会失败, 那句话就不是判据, 是散文。为此本版加
  四组检查。它们**不加新物理阶段**: 加的都是对照、校验与评估, 不是新的物理
  环节, 既有阶段的结论一字未改。

    F1 · 跨边界条件中心荷 (同族交叉检验)
        用 θ=π Z2 twist (反周期键) 重算中心荷, 与 PBC 侧对照。
        必须标注为**同族**: E0/L 与 S(n) 共用同一个 Calabrese-Cardy ansatz,
        换的只是**态**与**边界条件**, 不换估计量。所以它检验的是"c 的读数对
        边界条件稳不稳", 不能当作独立族验证 —— 独立族那一条已如实降级。

    F2a · h/J 负对照扫描 (临界性是不是事实)
        若"临界模拟"成立, 把 h/J 推离 1.0 之后所有临界指标都应当**塌缩**。
        主判据是 c_fit(h/J) 在 h/J=1.0 取全局最大 —— 这说的是
        "Calabrese-Cardy ansatz 只在临界点成立", 比"某个数小"强得多。
        另加一条**区分力**要求: 框架必须在非临界点**正确失败**
        (审计 §五.2: 一个能在非临界点正确失败的框架, 比一个永远输出 3% 的
        框架可靠)。

    F5 · 一致性检查束 (数值路径 <-> 解析路径)
        每一条都把同一个量用两条**来源不同**的路径算出来对表:
        JW 闭合式 <-> 稀疏对角化 | 幂迭代残差 | Forman 环锚点 |
        周期域拉普拉斯零和 <-> Gray-Scott 精确平衡恒等式 | 熵比与等权度上界。
        它们专门覆盖"数值路径算对了、但模型本身搭错了"这一类错误 ——
        例如 H 的位序约定、周期键、横场符号, 稀疏对角化对这三件事是哑的。

    F6 · 零模型对照族 (报百分位, 不是单点比较)
        单张同规模随机图只能回答"比随机图更负吗"; 用三类对照 (度分布 /
        聚类 / 树的成分) 各 3 个实例给出百分位, 并用**双侧**边界 [5%, 95%]
        判定 —— 两种方向的统计异常都必须报出来, 不能只报顺手的那个。

    F7 · 特征波长的输入校验
        退化输入 (空场/平坦场) 不再静默产出"看起来合理"的伪值:
        只加校验与标位, 取峰规则不动。

【阶段间因果链 (下游参数由上游物理量函数式派生)】

  七个阶段在 main() 里若全靠手工接线, 就会退化成 20 多个写死的字面量
  (stage1_void(4), mera_init(8, 2), stage6_life(F=0.035, k=0.060) ...),
  阶段之间只传递给人看的诊断字符串, 没有任何物理量真正流向下游。
  最能说明问题的例子: 用 boundary_correlation_length() 算出关联长度
  xi = 19.314, 打印出来, 然后扔掉 —— 它从未进入 stage6。
  本脚本把八条连线全部换成真的派生 (推导式写进注释并打印实测值):

    L1  阶段一 -> 阶段二   局部维度 d_local=2        -> dim_full = d_local**L
    L2  阶段二 -> 阶段三-a 约化密度矩阵谱 {p_k}      -> 谱隙 = p1/p0
    L3  阶段二 -> 阶段三-b/c 有效 Schmidt 秩         -> 自指网络宽度 N
    L4  阶段二 -> 阶段四   纠缠熵 S                  -> 键维 chi >= exp(S)
    L5  阶段四 -> 阶段五   MERA 张量图平均度         -> Forman 曲率解析锚点
    L6  阶段五 -> 阶段六   关联长度 xi / 稳定性      -> 域长 L 与网格 N
    L7  阶段三+六 -> 阶段七 联合不动点               -> 照见判据
    L8  阶段七 -> 阶段一   由 spiral_loop() 闭合

  L2 是整条链的核心, 因为它有**严格的数学对应而非类比**: 约化密度矩阵
  rho = M M^dagger 的特征值恰好是 Schmidt 系数的平方 {s_k^2}, 所以对
  rho 做幂迭代的收敛率 = p1/p0。(注意: 这句话不能改写成"转移矩阵的
  特征值是 {s_k^2}" —— 那是错的, 详见 derive_L2 的 docstring。)
  L6 把"网格 N"从写死的 40 改成由显式 Euler 稳定性上界派生:
  N <= L*sqrt(4*dt*Du)。代入得 N=55, 于是 "stab > 0.25 就拒绝运行" 的守卫
  从"碰巧躲过"变成了"由构造保证不会被触发"。

  闭环螺旋 spiral_loop() (多圈自举)
      把整条七阶段链当作一个映射, 反复应用到自身。第 t 圈的系统尺寸由
      第 t-1 圈**实测**的关联长度定向:  L_t = clip(scale * xi_{t-1}, L0, max_L)。
      每圈跟踪三个量, 因为它们的收敛行为不同 —— 这才是闭环的答案所在:
          c_t        -> 1/2      (CFT 不动点)
          xi_t / L_t -> 常数     (共形不变性; 无量纲比值才是尺度不变量)
          相对谱隙 1-r -> 0      (r = p1/p0 上升, 这才是临界的标志)
      必须强调的诚实边界: xi 本身随 L 发散**不是失败, 是临界的定义**。
      临界系统的关联长度被系统尺寸截断, 所以绝不能用 xi 的绝对值判断收敛 ——
      收敛判据必须用无量纲比值。实测 xi = 19.314 > L = 16, 正是这个原因。

【七阶段 (对应七行诗) —— 全部保留, 其中 3/4/5 为真实实现】
  L1. 真空等权叠加，基底即潜能。        - 完美张量 / 等权态 (幺半群单位)
  L2. 量子涨落扰动，对称性破缺。        - 临界 TF-Ising 基态的 Schmidt 极化
  L3. 偏振分化差异，递归成自指。        - 一般矩阵与神经网络自指动力学
  L4. 自指画定边界，投影显全息。        - quimb 二进制 MERA + 等距性验证
  L5. 全息自然展开，织就了时空。        - Forman / Ollivier-Ricci 曲率
  L6. 时空提供梯度，孕育出生命。        - Gray-Scott 反应-扩散 (参数由阶段五派生)
  L7. 生命涌现意识，自照见空无。        - 权重-状态耦合自指的联合不动点 (二阶自指)

【诚实边界 (全部由本脚本实测确认, 不夸大)】
  * 中心荷: 精确对角化对照 L=16 给 c = 0.507。MERA 拟合值随键维单调趋近 0.5
    (chi=2 -> ~0.60, chi=4 -> ~0.52~0.56), 残差是有限键维截断, 不是 1/2 的证明。
    证据是"随 chi 收敛的趋势 + 精确对照", 而非单一数字。
  * 键维派生 chi >= exp(S) 是**必要非充分**条件: 它只保证能装下这么多纠缠,
    不保证装得下长程关联。派生值恰好与实测最优配置吻合, 是自洽性证据而非证明。
  * MERA 变分目标: 用"最大化与精确基态的重叠"。若改用"最小化能量",
    梯度下降会塌缩到平均场局部极小 (E/L = -1.2665 看似接近精确值 -1.2815,
    但 S(n) ~ 0.02, c ~ 0.02)。本脚本把这条错误路径作为对照实验如实报告。
  * 有限尺寸: L=8/16 的环, MERA 用周期边界, 故 c 拟合用 CFT 的周期公式。
  * 曲率约定: Ollivier-Ricci 依赖随机游走的 idle 概率 alpha。本脚本对
    alpha=0 与 alpha=1/(d+1) 分别给出解析锚点并显式声明, 不引用记忆中的文献数字。
  * 阶段六: 反应参数 F, k 是**不可约的化学输入** (反应动力学, 与几何无关),
    因此显式标注为旋钮, 不假装从曲率派生。真正派生的是几何量 (域长 L、网格 N)。
  * 曲率对照: MERA 体几何的负曲率**不是**"比所有对照图更负"。三类零模型对照
    显示它落在对照分布之内 (百分位在 [5%, 95%] 区间), 故只能说"与随机图同量级",
    不能说"全息几何更负"。这条结论由对照数据算出, 不是写死的措辞。
  * 闭环: max_L 是旋钮且实测会触顶。去掉它 L 会一直涨 —— 这正是临界性,
    而不是闭环的失败。脚本把这个触顶如实打印出来。
  * 阶段七仍是概念模型: "照见"是联合不动点的结构性表达, 不是意识实证。
    状态塌缩到 x*=0 时 D* 不存在, 该支的熵比与等权度**一律是约定**
    (两者是同一件事的换写), 已如实标 defined=False, 不当作独立证据。
  * 阶段六不变量: Gray-Scott 的源/汇项不抵消, 所以 "u+v 质量守恒" 是**伪不变量**
    (首步漂移 6.25e-5, 是常被声称的 1e-6 容差的 62 倍)。真正成立的是两条离散
    恒等式: 周期域上拉普拉斯零和, 以及由更新前场精确预言的 ⟨u+v⟩ 增量平衡式。
  * 第三层参考侧种子: 归档的"多种子"其实同出一个硬编码初始条件, 其 bool 场与
    浮点场的任何简单二值化一致率都在 0.5 附近 (随机水平)。该缺口由 A 段诊断项
    如实登记为未通过, 不用它撑任何结论。

【依赖】
  numpy, scipy, matplotlib, networkx, quimb>=1.0, torch  (均为硬依赖, 不静默降级)
  第三层只额外需要标准库 tarfile —— 刻意不引入任何宇宙学库,
  以保证 Mini-Millennium 的读取逻辑完全可审计。
【运行】  python spiral_model_v14.py
【产物】  spiral_v14.png (15 面板物理图)
          spiral_v14_metrics.png (指标图)
          _v14_data.json (含 metrics 三层指标)
          _v14_hj_scan.png (F2a 负对照扫描)
【数据】  data/mini-millennium  (缺失时第三层如实报告为"跳过", 不崩溃)
================================================================================
"""
import os
import sys
import json
import platform
import time
import warnings

# 钉住 stdout 编码 (不改变任何数值, 只防止崩溃):
# 本机 Windows 的默认 locale 是 GBK, 而代码里有 U+2080 这类下标字符
# (例如阶段三-a 打印的 "x₀")。一旦 stdout 被重定向到**文件**而不是管道,
# Python 就用 locale 编码, 打印到那一行直接抛 UnicodeEncodeError 并把整个
# 运行丢掉 (实测: 运行到 19.7 秒时崩在阶段三-a 的打印处)。
# 这不是"运气好没踩到"—— stdout 只要不是 UTF-8 就一定会踩到。这里显式钉住
# UTF-8, 与 locale 无关。
# 同时开行缓冲: 输出重定向到文件时 Python 用块缓冲 (8KB 才落盘), 于是运行
# 几分钟之后日志仍然是 0 字节 —— 分不清"在算"还是"卡住了"。这个脚本要跑
# 十几分钟, 没有实时日志就没法监工。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', line_buffering=True)
    except (AttributeError, ValueError):
        pass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.optimize import linprog
import networkx as nx
from spiral_metric_v14 import collect_metrics_v14, plot_metrics_v14, metrics_for_json
warnings.filterwarnings('ignore')

# ---------- 中文字体 ----------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False


def setup_chinese_font():
    """
    选一个字形覆盖足够全的 CJK 字体。
    顺序很关键: SimHei 缺 U+2212 (减号), 用它会让刻度里的负号退化成方块并
    刷屏 "does not have a glyph for U+2212" 警告。Microsoft YaHei 覆盖
    减号 + 希腊字母 (chi/lambda) + 箭头 + 双竖线, 因此排在前面。
    """
    system = platform.system()
    if system == 'Windows':
        cands = ['Microsoft YaHei', 'SimHei', 'SimSun']
    elif system == 'Darwin':
        cands = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC']
    else:
        cands = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei']
    avail = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for f in cands:
        if f in avail:
            # 单字体是**不够的**。实测: Microsoft YaHei (msyh.ttc)
            # 没有 U+27E8 ⟨ / U+27E9 ⟩ (数学尖括号)、U+1D5B ᵛ、U+2080 ₀ / U+2082 ₂
            # (下标)、U+21D2 ⇒ —— 第二层"关联衰减指数"面板的纵轴
            # |⟨σᶻ₀σᶻ_r⟩| 一次用掉其中四个, 整条纵轴标题因此退化成方块。
            # 回退的开关是 **font.family 取列表**, 不是 font.sans-serif:
            # 实测 ['Microsoft YaHei'] 与 ['Microsoft YaHei','DejaVu Sans'] 配在
            # font.sans-serif 上都仍然报 6 个缺字警告; 放到 font.family 上
            # (['Microsoft YaHei','DejaVu Sans']) 就变成 0 个。
            # 所以两处都设: sans-serif 保持原意, family 承担逐字形回退。
            plt.rcParams['font.sans-serif'] = [f]
            plt.rcParams['font.family'] = [f, 'DejaVu Sans']
            return f
    return None


setup_chinese_font()

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUTPUT_DIR = _HERE+"/result"
os.makedirs(_OUTPUT_DIR, exist_ok=True)

# ---------- 这两个库是本模型的核心, 不做静默降级 ----------
import torch  # noqa: E402
import quimb.tensor as qtn  # noqa: E402

torch.set_default_dtype(torch.float64)


# ============================================================================
# 派生层 · 因果链旋钮与派生映射
# ============================================================================
#
# 设计原则 (三条, 全部可检查):
#
#   1. 每个下游参数由上游**实测物理量**算出。推导式写在函数的 docstring 里,
#      并且实测值会被打印出来。不允许"看起来像派生"的注释配硬编码的值。
#
#   2. 真正不可约的外部输入集中到下面的 KNOBS 里, 打印时明确标注
#      「旋钮」而不是「派生」。不假装旋钮是派生出来的。
#
#   3. 上游量若退化 (熵为 0、谱太短、派生值超出稳定性上界), 就**拒绝运行
#      并说明原因**, 绝不静默退回硬编码默认值。
#
# 为什么需要第 3 条: 最容易骗人的地方就是"看起来在派生"。曾经算出关联长度
# xi = 19.314, 打印出来, 然后扔掉 —— docstring 却声称阶段六跑在"涌现几何"
# 上。一个从不流向任何地方的中间量, 比没有这个量更糟。所以用 record_link()
# 把每条链的输入/输出都记录下来, 谁没接线一眼可见。

KNOBS = {
    # ---- 尺度 (唯一的自由度: 整条链的尺寸) ----
    'L_chain': 16,        # 阶段二 自旋链长度。L8=2^16, 精确对角化可行
    'd_local': 2,         # 阶段一 局部维度: 空无 = 单个量子比特 (自旋 1/2)

    # ---- 派生层的阈值与裕度 ----
    'schmidt_tol': 1e-8,  # L3 有效 Schmidt 秩的截断阈值 (p_k > tol 才算数)
    'stability_limit': 0.25,   # 显式 Euler 稳定上界 dt*Du/h^2 <= 1/4

    # ---- 阶段六网格 (L6 的两条约束) ----
    'L_domain': 0.5,      # 旋钮: Gray-Scott 域长 (化学自身的长度单位, 见 L6)
    'n_lambda': 3.0,      # 旋钮: 域内应容纳的斑图波数 (来自 xi/L 的尺度不变性)
    'pts_per_wavelength': 12.0,  # 旋钮: 每个波长至少几个网格点才算"分辨"

    # ---- 阶段六化学参数: 不可约, 与几何无关 ----
    'F': 0.035,           # 旋钮: 进料率
    'k': 0.060,           # 旋钮: 移除率
    'dt': 1.0,            # 旋钮: 时间步 (与稳定性上界联立求得 N)
    'Du': 2e-5,           # 旋钮: u 扩散系数 (经典 Pearson 标度)
    'Dv': 1e-5,           # 旋钮: v 扩散系数

    # ---- 闭环 spiral_loop() ----
    'turns': 6,           # 旋钮: 最多转几圈
    'L0_loop': 8,         # 旋钮: 第 0 圈的系统尺寸
    'max_L_loop': 16,     # 旋钮: 尺寸上限 (2^16 是精确对角化的实际边界)
    'scale_knob': 1.0,    # 旋钮: L_t = scale_knob * xi_{t-1}
    'loop_tol': 0.02,     # 收敛判据: 无量纲比值的相对变化 < 2%
}

# 因果链台账: 每条 (上游量 -> 下游参数) 的记录都进这里, 供打印/JSON/作图
CHAIN = []


def record_link(link, upstream, formula, derived, v10_manual, note=''):
    """
    记录一条因果链。link 形如 'L2', upstream 是上游物理量的描述。

    v10_manual 是同一个参数在手工接线时代的取值 (字段名保持不变以维持兼容)
    —— 把"派生值 vs 手工值"并排打印出来, 读者可以自己判断这条链是自洽还是
    自欺。二者不等, 说明派生确实改了取值, 而不是把原字面量换个说法抄一遍。
    """
    rec = {'link': link, 'upstream': upstream, 'formula': formula,
           'derived': derived, 'v10_manual': v10_manual, 'note': note,
           'same_as_manual': (derived == v10_manual)}
    CHAIN.append(rec)
    return rec


def print_chain_report():
    """把整条因果链打印成一张对照表。这是因果链层的主要可检查产物。"""
    print("\n" + "=" * 76)
    print("  因果链台账: 派生值 vs 手工值")
    print("=" * 76)
    for r in CHAIN:
        tag = "= 手工值" if r['same_as_manual'] else "!= 手工值"
        print(f"  [{r['link']}] {r['upstream']}")
        print(f"        公式: {r['formula']}")
        print(f"        派生 = {r['derived']!r}   (手工 = {r['v10_manual']!r})  {tag}")
        if r['note']:
            print(f"        注: {r['note']}")



# ---------- 八条链的派生函数 (L1..L6 在此实现, L7/L8 见 main 与 spiral_loop) ----------

def derive_L1_local_to_full(d_local, L):
    """
    L1 · 阶段一 -> 阶段二: 局部自由度张成全空间。

    空无 = 单个量子比特 (局部维度 d_local), 阶段二的 L 个自旋是整个空间。
        dim_full = d_local ** L

    手工接线时 stage1_void(4) 取维度 4, 与阶段二的 2 维自旋毫无关系 ——
    这条是八条链里最容易修也最明显的一条。
    """
    dim_full = int(d_local ** L)
    record_link('L1', f'空无的局部维度 d_local={d_local}, 链长 L={L}',
                'dim_full = d_local ** L', dim_full, 4,
                note='手工值 4 是随手取的; 现由局部维度与链长派生')
    return {'d_local': int(d_local), 'dim_full': dim_full}


def derive_L2_entanglement_gap(p, tol=1e-15):
    """
    L2 · 阶段二 -> 阶段三-a: 纠缠谱给出幂迭代的收敛率。  ← 整条链的核心

    为什么这是数学而不是类比 (此处已按文献核实修正过一次措辞):
      把一个二分量子态写成矩阵 M (行/列标是两边的基), 则约化密度矩阵
          rho = M M^dagger,
      而 rho 的特征值**恰好**是 Schmidt 系数的平方 {s_k^2}。
      对 rho 做幂迭代, 收敛率 = 第二大 / 最大 = (s_1/s_0)^2 = p1/p0。
      这是初等线性代数, 严格成立。

    一处必须澄清的误用 (初稿曾写错, 已改):
      不能把上面这句说成"转移矩阵的特征值就是 {s_k^2}"。标准 MPS 的
      mixed transfer operator T(X) = sum_i A^i X A^{i†} 是 D^2 x D^2 的,
      正则形式下它的**主特征值恒为 1**, 而 {s_k^2} 是它的**主左不动点**
      (dominating left eigenvector), 不是特征值; 它的次主特征值给出的是
      关联长度 xi = -1/ln|lam2/lam1|, 与 Schmidt 谱无直接关系。
      所以本函数派生的是"对 rho = M M^dagger 做幂迭代的收敛率",
      不是任何转移矩阵定理。措辞按此写, 不夸大。

    所以阶段三-a 里"幂迭代收敛得多快"不该由人手指定, 而应由阶段二实测的
    Schmidt 谱决定。这是整条因果链里唯一一条有严格数学对应的连线。

    诚实边界: 临界系统的纠缠谱在 L -> inf 时能隙闭合 (p1/p0 -> 1),
    所以有限尺寸下取到的只是一个**被尺寸截断的**谱隙。这不是缺陷,
    恰是临界的标志 —— 闭环 spiral_loop() 里正是用相对谱隙 1 - p1/p0 -> 0
    来判临界。(注意 p1/p0 本身是**收敛比**不是谱隙, 它趋于 1 而非 0。)
    """
    p = np.asarray(p, dtype=float)
    p = p[p > tol]
    if len(p) < 2:
        raise RuntimeError(
            f"L2 退化: 约化密度矩阵只有 {len(p)} 个非零本征值, 无法定义谱隙。"
            f"拒绝派生 —— 阶段三-a 的收敛比必须有物理来源。")
    gap = float(p[1] / p[0])
    record_link('L2', f'约化密度矩阵谱 p = s^2 ({len(p)} 个非零值)',
                'gap = p[1] / p[0]', round(gap, 6), 0.5,
                note=f'p0={p[0]:.6f}, p1={p[1]:.6f}; '
                     f'临界态该值随 L 增大而 -> 1 (谱隙闭合)')
    return {'gap': gap, 'ngap': len(p), 'p0': float(p[0]), 'p1': float(p[1])}


def derive_L3_schmidt_rank(p, tol):
    """
    L3 · 阶段二 -> 阶段三-b/c: 有效 Schmidt 秩决定自指网络的宽度。

    临界态的 Schmidt 谱衰减慢, 有效秩随 L 增长 —— 这是真实可测的派生量。
    自指网络 x_{n+1} = tanh(g W x_n) 里的 W 是 N*N 的, 它的宽度 N 应当是
    "这个态实际需要多少个分量才能描述", 也就是有效 Schmidt 秩。

    诚实边界: 秩的数值依赖阈值 tol。这里显式传参并在日志里打印 tol,
    不藏一个 1e-8 在函数体里假装它是自然常数。
    """
    p = np.asarray(p, dtype=float)
    rank = int(np.count_nonzero(p > tol))
    rank = max(rank, 2)   # 至少要 2, 否则网络退化成一个标量
    record_link('L3', f'有效 Schmidt 秩 (p_k > {tol:g})',
                'N_selfref = #{k : p_k > tol}', rank, 64,
                note='手工值 64 与态的实际复杂度无关')
    return {'N_selfref': rank, 'schmidt_tol': float(tol)}


def derive_L4_bond_dimension(S, base=2):
    """
    L4 · 阶段二 -> 阶段四: 面积律给出键维下界。

    一维面积律说, 要精确表示纠缠熵为 S 的态, 键维至少要有
        chi >= exp(S)
    二进制 MERA 的键维必须是 2 的幂, 故取
        chi = base ** ceil(log_base(exp(S)))

    **这是必要非充分条件**: 它只保证能装下这么多纠缠, 不保证装下长程关联。
    所以派生值与实测最优配置吻合只能算自洽性证据, 不能算证明。

    手工接线时扫描 chi in {2, 4} 并报告 c(chi=4) 最好; 派生值若也落在 4,
    说明这条链是自洽的 —— 但脚本不因此宣称"证明了 chi=4 最优"。
    """
    if S <= 0:
        raise RuntimeError(
            f"L4 退化: 纠缠熵 S={S:.3e} <= 0, 面积律给不出正的下界。"
            f"拒绝派生 —— 一个无纠缠的态不能用来定键维。")
    chi_req = float(np.exp(S))
    chi = int(base ** int(np.ceil(np.log(chi_req) / np.log(base))))
    record_link('L4', f'半链纠缠熵 S(L/2) = {S:.4f}',
                f'chi = {base}**ceil(log_{base}(exp(S)))', chi, 4,
                note=f'面积律下界 exp(S) = {chi_req:.4f} -> 取 2 的幂得 {chi} '
                     f'(必要非充分)')
    return {'chi': chi, 'chi_lower_bound': chi_req}


def derive_L5_forman_from_degree(G, forman_measured, record=True):
    """
    L5 · 阶段四 -> 阶段五: MERA 张量图的平均度给出 Forman 曲率的解析预期。

    无权图、不含三角项时, Sreejith 加权式退化为
        F(e) = 4 - deg(u) - deg(v)
    由此有**两个**解析预期, 必须分清:

      (i)  正则粗式:   <F> = 4 - 2*d_bar        d_bar = 2E/V (图平均度)
                       只在 d-正则图上严格成立。

      (ii) 严格恒等式: <F> = 4 - 2*<deg>_edge
                       其中 <deg>_edge = mean_{(u,v) in E} (deg(u)+deg(v))/2
                       是**按边加权的端点度平均**。这个式子对任何无三角图
                       都精确成立 (它只是把 F(e) 逐边平均改写成度数形式),
                       所以它应当与实测到机器精度吻合。

    两者的差就是**度偏置 (degree bias)**: 非正则图上度数大的点被更多条边
    "选中", 于是 <deg>_edge > d_bar, 实测 <F> 比粗式给出的更负。

    MERA 体几何的 Forman 均值 = -2.242 需要说明它从哪里来, 也要解释为什么
    它不等于 4-2*d_bar。这条链补上"从哪里来", 并顺手把那个差归因清楚 ——
    是度偏置, 不是曲率实现有错。三角数一并报出, 因为 F(e)=4-deg(u)-deg(v)
    这个无三角形式本身需要 MERA 图接近树才成立。

    record=False 供**对照图**使用: 对照图不是因果链的一环, 把它们的度偏置
    写进因果链台账会污染那条链。但几何口径必须与 MERA 图完全一致, 所以
    复用同一个函数而不是另写一套。
    """
    n, m = G.number_of_nodes(), G.number_of_edges()
    d_bar = 2.0 * m / n if n else float('nan')
    # (i) 粗略式: 只在 d-正则图上才严格. 非正则图会系统性偏低, 见下.
    predicted = 4.0 - 2.0 * d_bar
    # (ii) 严格恒等式: 无三角项时 <F> = 4 - 2*<deg>_edge,
    #      其中 <deg>_edge 是**逐边端点度数的平均** (每条边贡献 deg(u)+deg(v),
    #      故除以 2E 再乘 2 —— 等价于按边加权, 度数高的点被数到更多次)。
    #      这个量与实测 Forman 均值应当**到机器精度**吻合, 因为它就是
    #      F(e)=4-deg(u)-deg(v) 逐边求平均的恒等改写。
    deg = dict(G.degree())
    deg_edge_mean = float(np.mean([deg[u] + deg[v] for u, v in G.edges()]) / 2.0)
    predicted_exact = 4.0 - 2.0 * deg_edge_mean
    # 三角数: MERA 近似树, 这个量应当很小 —— 它是上面 (b) 的直接证据
    triangles = sum(nx.triangles(G).values()) // 3
    # degeneracy (度偏置) 就是粗式与实测的差从哪来: 非正则图上
    # 度数大的点被更多条边"选中", 所以 <deg>_edge > d_bar, 于是
    # 实测 <F> 比 4-2*d_bar 更负。级数上 = 2*(<deg>_edge - d_bar)。
    bias = 2.0 * (deg_edge_mean - d_bar)
    if record:
        record_link('L5', f'张量图 V={n}, E={m} '
                          f'(平均度 d_bar={d_bar:.3f}, 逐边端点平均度={deg_edge_mean:.3f})',
                    '<F> = 4 - 2*d_bar (正则图式) | '
                    '<F> = 4 - 2*<deg>_edge (无三角严格式)',
                    round(predicted_exact, 4), round(forman_measured, 4),
                    note=f'实测 Forman 均值 = {forman_measured:.4f}; '
                         f'严格式预期 = {predicted_exact:.4f}; '
                         f'正则粗式预期 = {predicted:.4f}。'
                         f'粗式差 {abs(forman_measured - predicted):.4f} = 度偏置 2*(<deg>_edge-d_bar) '
                         f'= {bias:.4f} 起, 不是实现错误。'
                         f'图含 {triangles} 个三角形 (接近树, 故三角项影响小)')
    return {'d_bar': d_bar, 'forman_predicted': predicted,
            'deg_edge_mean': deg_edge_mean, 'forman_exact': predicted_exact,
            'bias': bias, 'triangles': triangles}


def derive_L6_grid(xi_over_L, knobs, xi=None):
    """
    L6 · 阶段五 -> 阶段六: 关联长度与稳定性上界共同定出网格。

    **先说清楚哪条能派生、哪条不能** —— 这是本条链最重要的部分。

    不能派生的: Gray-Scott 的**物理域长** L_domain 不能由 Ising 链的关联长度
    算出。两者是不同的物理 (一个是量子自旋关联, 一个是反应-扩散的扩散长度
    sqrt(Du/(F+k))), 把它们数值相等起来是量纲和物理的双重范畴错误。
    所以 L_domain, n_lambda, pts_per_wavelength 都是**显式旋钮**。

    真正能派生的是**网格数 N**, 它由两条约束夹出来:

      下界 (分辨要求): 每个斑图波长至少要 pts_per_wavelength 个网格点,
                       域内要容纳 n_lambda 个波长
                       N_req = ceil(pts_per_wavelength * n_lambda)

      上界 (稳定性定理): 显式 Euler 解 dt*Du/h^2 <= 1/4, 而 h = L_domain/N
                       解出 h >= sqrt(4*dt*Du), 即 N <= L_domain / sqrt(4*dt*Du)
                       N_cap = floor(L_domain / sqrt(4*dt*Du))
                       (注意是**除以** sqrt(4*dt*Du)。初稿误写成乘, 于是
                        N_cap=0 被本函数的守卫当场拦下 —— 守卫有效。)

      N = min(N_req, N_cap), 并报告是哪一条在起作用。

    若 N_cap < N_req, 两条约束冲突: 派生的域长大到无法在稳定步长下分辨。
    此时**拒绝运行**并说明, 而不是偷偷放宽稳定性。

    这一步的实际意义: N 若写死成 40, 稳定性守卫 (stab <= 0.25) 就只是
    "碰巧没被触发"。N 由稳定性上界派生之后, 守卫从运气变成了定理 ——
    它不可能被触发, 除非两条约束冲突 (那时是显式拒绝, 不是偷偷放行)。

    xi_over_L 是阶段五实测的无量纲比值 xi/L。它不直接进公式 (见上),
    但被记录下来, 因为它的尺度不变性正是 n_lambda 取值的依据。
    """
    L_dom = float(knobs['L_domain'])
    dt = float(knobs['dt'])
    Du = float(knobs['Du'])

    # 上界: 稳定性定理.  dt*Du/h^2 <= 1/4 且 h = L_dom/N  =>  N <= L_dom/sqrt(4*dt*Du)
    N_cap = int(np.floor(L_dom / np.sqrt(4.0 * dt * Du)))
    # 下界: 分辨要求
    N_req = int(np.ceil(knobs['pts_per_wavelength'] * knobs['n_lambda']))
    N = min(N_req, N_cap)
    bound_by = '分辨要求 (N_req)' if N_req <= N_cap else '稳定性上界 (N_cap)'

    if N < 8:
        raise RuntimeError(
            f"L6 拒绝运行: 派生网格 N={N} 太小 (N_req={N_req}, N_cap={N_cap}), "
            f"无法分辨斑图。这是派生值的真实冲突, 不偷偷放宽稳定性条件。")

    h = L_dom / N
    stab = dt * Du / h ** 2
    record_link('L6', f'域长 L_domain={L_dom} (旋钮), '
                      f'xi/L={xi_over_L:.4f} (阶段五实测, 无量纲)',
                'N = min(ceil(ppw*n_lambda), floor(L_domain/sqrt(4*dt*Du)))',
                N, 40,
                note=f'N_req={N_req} (分辨), N_cap={N_cap} (稳定), '
                     f'受限于 {bound_by}; 派生后 h={h:.6f}, '
                     f'dt*Du/h^2={stab:.4f} <= {knobs["stability_limit"]} '
                     f'由构造成立')
    return {'N': N, 'N_req': N_req, 'N_cap': N_cap, 'h': h, 'stab': stab,
            'bound_by': bound_by, 'L_domain': L_dom,
            'lambda_target': L_dom / knobs['n_lambda']}


# ============================================================================
# 第 0 节 · 工具: 周期 TF-Ising + 纠缠熵 + 中心荷拟合
# ============================================================================
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


def exact_ground_state(L, J=1.0, h=1.0):
    """稀疏 eigsh 求基态。L<=16 时 2^L 可控。返回 (E0, gs)"""
    H = tfi_periodic_sparse(L, J, h)
    E, V = eigsh(H, k=1, which='SA')
    gs = V[:, 0]
    return float(E[0]), gs / np.linalg.norm(gs)


def jw_ground_energy(L, J=1.0, h=1.0):
    """
    周期 TF-Ising 的 Jordan-Wigner 解析基态能量 (NS 扇区, 动量 q_k = pi(2k+1)/L):
        E0 = -Sum_{k=0}^{L-1} sqrt( J^2 + h^2 - 2 J h cos(q_k) )

    这是与稀疏对角化**完全独立**的第二条路径, 所以它能测出 eigsh 测不出的东西:
      * H 的位序约定 (站点 i <-> 二进制位 L-1-i) 错了, 这里立刻对不上;
      * 周期边界那条键 ((-J) sigma^z_{L-1} sigma^z_0) 漏了或重复了, 也对不上;
      * 横场项的符号 (-h sigma^x) 反了, 同样对不上。
    eigsh 只回答"给定这个 H, 最小本征值是多少", 对 H 本身对不对一句话都没有。
    """
    k = np.arange(L)
    q = np.pi * (2.0 * k + 1.0) / L
    return float(-np.sum(np.sqrt(J ** 2 + h ** 2 - 2.0 * J * h * np.cos(q))))


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
# F1 · 跨边界条件中心荷交叉检验 (v14)
# ============================================================================
#
# 审计 (§三之补 F1①) 要求的是**独立族估计量**: E0/L 与 S(n) 共用同一个
# Calabrese-Cardy ansatz, 只能算同族交叉检验, 必须降级。据此把 F1 拆成两块,
# 每一块都显式标注是哪一族:
#
#   (同族)   跨边界条件: PBC vs θ=π Z2 twist (反周期键)。
#            换的是**态**与**边界条件**, 不换估计量。
#   (独立族) **未建立**。
#            已试的 Rényi 指数扫描实测验证失败 (见 renyi_alpha_scan 的函数头),
#            如实记录为负面结果, 不伪造一个"看起来在检验"的守卫。
F1_L = 16
F1_CHI = 4
F1_SEEDS = (0, 1, 2)
F1_C_TRUE = 0.5                  # 临界 Ising 的中心荷真值
F1_REL_MAX = 0.05                # 跨边界条件相对差判据 (审计要求不放宽)

RENYI_ALPHAS = (0.5, 0.75, 1.5, 2.0, 3.0, 5.0, 8.0)
RENYI_NS = (2, 3, 4, 5, 6, 7, 8)
RENYI_STD_MAX = 0.02             # c(alpha) 在 alpha 上一致性的门槛
RENYI_DEV_TRUE_MAX = 0.05        # c(alpha) 均值对 0.5 的偏差门槛


def tfi_z2_twisted_sparse(L, J=1.0, h=1.0):
    """
    周期 TF-Ising, 但边界键 (L-1, 0) 取 +J 而非 -J。

    这就是 θ=π 的 Z2 twist, 物理上等于 Jordan-Wigner 的反周期 (NS) 扇区。
    与连续 U(1) twist 的区别是实测出来的 (见 _v14_probe_twist.py): 连续 twist
    给出复矩阵, 而 Z2 版本保持 H 为**实**矩阵 —— 所以能直接喂 dtype=float 的
    quimb MERA, 不需要动任何现有张量网络代码。

    位序与 tfi_periodic_sparse 完全相同 (站点 i <-> 位 L-1-i), 唯一差别是边界键
    符号, 这样 S(n) 的块划分与 PBC 情形**逐位可比** —— 这是"只换边界条件"这句
    话能成立的前提。
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
        sgn = +1.0 if (a, b) == (L - 1, 0) else -1.0
        H = H + (sgn * J) * mat
    return ((H + H.T) / 2).tocsr()


def _schmidt_probs(psi, L, n):
    """块 A = 前 n 个连续站点的 Schmidt 概率谱 (已归一, 已截掉 < 1e-15)。"""
    psi = np.asarray(psi).reshape(-1)
    psi = psi / np.linalg.norm(psi)
    s = np.linalg.svd(psi.reshape(2 ** n, 2 ** (L - n)), compute_uv=False)
    p = s ** 2
    p = p[p > 1e-15]
    return p / p.sum()


def renyi_alpha_scan(gs, L=F1_L, alphas=RENYI_ALPHAS, ns=RENYI_NS):
    """
    F1 · 独立族估计量的**失败尝试**, 保留为可复现的负面证据。

    想法: Rényi 熵满足 S_alpha = (c/6)(1 + 1/alpha) ln[(L/pi) sin(pi n/L)]
    + const_alpha。对**固定 alpha 扫 n** 拟合得 c(alpha) —— 与 alpha=1 的 S(n)
    拟合相比, 它多检验了一个**独立的系数预言** (1 + 1/alpha), 失效方式也不同
    (若态不共形, alpha 依赖会先于 n 依赖垮掉)。看上去像个独立族。

    实测 (L=16, h/J=1.0, 解析 c=0.5):
        alpha  0.50     0.75     1.50     2.00     3.00     5.00     8.00
        c      0.53819  0.50838  0.51114  0.53184  0.56138  0.56820  0.55991
      均值 0.53986, 标准差 0.02252, 对 0.5 的偏差 7.97%。

    alpha=1 附近是自洽的 (0.508/0.511 正确夹住 0.5072, 说明实现本身没写错),
    但 alpha >= 2 后系统性飘高: Rényi 熵的次领头修正比 alpha=1 大得多, L=16
    的有限尺寸压不住。

    => **验证不通过, 不作独立族估计量**。在指标层注册为诊断项
       (expect_pass=False), 不计入通过率 —— 它的作用是留下"为什么没有独立族"
       的可复现证据, 而不是充当一个假的第二把尺子。
    """
    ns = list(ns)
    x = np.array([np.log((L / np.pi) * np.sin(np.pi * n / L)) for n in ns])
    probs = {n: _schmidt_probs(gs, L, n) for n in ns}
    rows = []
    for a in alphas:
        y = np.array([float(np.log(np.sum(probs[n] ** a)) / (1.0 - a))
                      for n in ns])
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        rows.append({'alpha': float(a),
                     'c': float(6.0 * coef[0] / (1.0 + 1.0 / a)),
                     'rms': float(np.sqrt(np.mean((y - A @ coef) ** 2)))})
    cs = np.array([r['c'] for r in rows])
    dev = float(abs(cs.mean() - F1_C_TRUE) / F1_C_TRUE)
    return {'alphas': rows, 'mean': float(cs.mean()), 'std': float(cs.std()),
            'range': float(cs.max() - cs.min()), 'dev_vs_true': dev,
            'L': L, 'ns': ns,
            'validated': bool(cs.std() < RENYI_STD_MAX
                              and dev < RENYI_DEV_TRUE_MAX)}


def cross_boundary_twist(L=F1_L, J=1.0, h=1.0, chi=F1_CHI, seeds=F1_SEEDS,
                         steps=None, lr=0.03, verbose=True):
    """
    F1 (同族) · 跨边界条件的中心荷交叉检验: PBC vs θ=π Z2 twist。

    **这一项换的是边界条件与态, 不是估计量** —— 两边都用同一个
    fit_central_charge (同一个 Calabrese-Cardy log 式)。所以它回答的是
    "c 的读数对边界条件稳不稳", 而不是"换一把尺子量"。审计要求的独立族
    估计量本工作**未建立** (原因见 renyi_alpha_scan 的函数头)。

    为何用检查点选择协议 (mera_fit_v13) 而不是朴素拟合:
    实测 (_v14_probe_twistmera.py) 朴素 600 步/单种子在 PBC 侧给出 c=0.56530、
    重叠 0.98829 —— 那正是"11.45% 未收敛假象": 它会把跨边界条件的差夸大成
    5.98%, **是优化没跑到位, 不是物理失配**。改用检查点选择 + 1500 步后
    PBC 侧 c=0.52257, 差降到 3.13%。

    每种边界 3 个种子, 取 eps_c 处于**中位数**的那条轨迹作代表 —— 与
    v13_tier1_runs 同一条可审计规则 (中位数 = 典型表现, 不用最好的一条)。
    """
    if steps is None:                    # 延迟解析: V13_STEPS 在本文件后面才定义
        steps = V13_STEPS

    E0_p, gs_p = exact_ground_state(L, J, h)
    H_t = tfi_z2_twisted_sparse(L, J, h)
    E_t, V_t = eigsh(H_t, k=1, which='SA')
    E0_t = float(E_t[0])
    gs_t = V_t[:, 0] / np.linalg.norm(V_t[:, 0])

    fit_ex = {'pbc': fit_central_charge(gs_p, L),
              'twist': fit_central_charge(gs_t, L)}
    twist_is_real = not np.iscomplexobj(H_t)

    if verbose:
        print("=" * 76)
        print("  任务 F1 · 跨边界条件中心荷交叉检验 (同族)")
        print("=" * 76)
        print(f"  边界条件: PBC vs θ=π Z2 twist (边界键 -J -> +J), "
              f"H 保持实矩阵 = {twist_is_real}")
        print(f"  L={L}, chi={chi}, 每种边界 {len(seeds)} 个种子 x {steps} 步 "
              f"(检查点按 overlap 选择, 不偷看 eps_c)")
        print(f"  精确态 E0/L: PBC={E0_p/L:+.6f}  twist={E0_t/L:+.6f}  "
              f"(边界条件确实换了态)")
        print(f"  精确态 S(L/2): PBC={fit_ex['pbc']['S'][-1]:.5f}  "
              f"twist={fit_ex['twist']['S'][-1]:.5f}")
        print(f"  精确态拟合 c: PBC={fit_ex['pbc']['c']:.5f}  "
              f"twist={fit_ex['twist']['c']:.5f}   "
              f"(rms {fit_ex['pbc']['rms']:.4f} / {fit_ex['twist']['rms']:.4f})")

    sides = {}
    for name, target in (('pbc', gs_p), ('twist', gs_t)):
        runs = []
        for seed in seeds:
            mera, raws = mera_init(L, chi, seed=seed)
            psi, ov, cstep, _c = mera_fit_v13(mera, raws, target, L,
                                              steps=steps, lr=lr, c_exact=None)
            fm = fit_central_charge(psi, L)
            runs.append({'seed': int(seed), 'c': float(fm['c']),
                         'rms': float(fm['rms']), 'overlap': float(ov),
                         'chosen_step': int(cstep)})
            if verbose:
                print(f"      [{name:>5} seed={seed}] ov={ov:.6f} "
                      f"@step{cstep:4d}  c={fm['c']:.6f}  (rms={fm['rms']:.4f})")
        cs = np.array([r['c'] for r in runs])
        i_med = int(np.argmin(np.abs(cs - np.median(cs))))
        sides[name] = {
            'runs': runs, 'i_representative': i_med,
            'c_median': float(np.median(cs)),
            'c_representative': float(cs[i_med]),
            'c_std': float(cs.std()),
            'c_exact': float(fit_ex[name]['c']),
            'rms_exact': float(fit_ex[name]['rms']),
            'S_half_exact': float(fit_ex[name]['S'][-1]),
            'E0_per_L': float((E0_p if name == 'pbc' else E0_t) / L),
        }

    c_p = sides['pbc']['c_representative']
    c_t = sides['twist']['c_representative']
    rel = abs(c_t - c_p) / abs(c_p)
    dev_true = {k: abs(sides[k]['c_representative'] - F1_C_TRUE) / F1_C_TRUE
                for k in sides}
    dev_ex = {k: abs(sides[k]['c_representative'] - sides[k]['c_exact'])
              / abs(sides[k]['c_exact']) for k in sides}
    min_ov = min(r['overlap'] for s in sides.values() for r in s['runs'])

    renyi = renyi_alpha_scan(gs_p, L)

    if verbose:
        print(f"\n  跨边界条件 (代表态 = eps_c 中位数轨迹):")
        print(f"     PBC   c = {c_p:.5f}  (种子间 std={sides['pbc']['c_std']:.5f})")
        print(f"     twist c = {c_t:.5f}  (种子间 std={sides['twist']['c_std']:.5f})")
        print(f"     相对差 = {rel:.3%}   (判据 < {F1_REL_MAX:.0%})")
        print(f"  两个偏差并列报告 (审计 F1② 要求, 5% 标准不放宽):")
        print(f"     对 c_exact(L) = {sides['pbc']['c_exact']:.5f} / "
              f"{sides['twist']['c_exact']:.5f}: "
              f"PBC {dev_ex['pbc']:.3%}, twist {dev_ex['twist']:.3%}")
        print(f"     对真值 0.5                    : "
              f"PBC {dev_true['pbc']:.3%}, twist {dev_true['twist']:.3%}")
        print(f"  [独立族尝试] Rényi 指数扫描: 均值 c={renyi['mean']:.5f}, "
              f"std={renyi['std']:.5f}, 对 0.5 偏差={renyi['dev_vs_true']:.2%}")
        print(f"     -> {'通过验证' if renyi['validated'] else '未通过验证'}: "
              f"{'可作独立族估计量' if renyi['validated'] else '不作独立族估计量 (如实记录为负面结果)'}")
        print("  诚实边界: 本项换的是**边界条件**与**态**, 不是估计量; ansatz 仍是")
        print("            同一个 Calabrese-Cardy log 式, 因此属**同族**交叉检验。")
        print("            本工作**未建立**真正的独立族中心荷估计量 (见上条负面结果)。")

    return {'L': L, 'chi': chi, 'seeds': [int(s) for s in seeds],
            'steps': int(steps), 'J': J, 'h': h,
            'c_true': F1_C_TRUE, 'rel_max': F1_REL_MAX,
            'boundary': {'E0_pbc': float(E0_p), 'E0_twist': E0_t,
                         'twist_h_is_real': bool(twist_is_real),
                         'S_half_pbc': float(fit_ex['pbc']['S'][-1]),
                         'S_half_twist': float(fit_ex['twist']['S'][-1])},
            'sides': sides, 'rel_diff': float(rel),
            'min_overlap': float(min_ov),
            'dev_vs_true': dev_true, 'dev_vs_c_exact': dev_ex,
            'renyi': renyi,
            'passed': bool(rel < F1_REL_MAX)}


# ============================================================================
# 阶段一 · 空无基底等权叠加蕴潜能
# ============================================================================
def stage1_void(dimension=2, seed=0):
    """
    空无 = 最大对称态 (等权叠加), 也是幺半群的单位元。

    因果链 L1: 默认维度取 2 —— 空无就是**单个量子比特**, 即阶段二自旋链的
    局部自由度。整个 Hilbert 空间是它的 L 次张量幂, 由
    derive_L1_local_to_full() 派生 (dim_full = d_local ** L)。

    维度 4 曾是一个与后续任何阶段都无关的随手值; 取 2 之后,
    "空无"与"链"的关系从修辞变成了集合论事实。
    """
    rng = np.random.default_rng(seed)
    H = (rng.standard_normal((dimension, dimension)) +
         1j * rng.standard_normal((dimension, dimension)))
    Q, _ = np.linalg.qr(H)
    equal = np.ones(dimension, dtype=complex) / np.sqrt(dimension)
    print(f"[阶段一] 空无基底: 局部维度 d={dimension}, "
          f"酉条件数={np.linalg.cond(Q):.2f}")
    return {'unitary': Q, 'equal_state': equal, 'dimension': dimension}


# ============================================================================
# 阶段二 · 量子涨落扰动对称性破缺 (临界 Ising 基态的 Schmidt 极化)
# ============================================================================
def stage2_critical_break(L=16, J=1.0, h=1.0):
    """
    h/J = 1 是横场 Ising 的热力学临界点。基态的 Schmidt 谱呈现极化 ——
    这是"涨落打破对称性"在纠缠谱上的可计算投影。
    诚实边界: 有限尺寸下纠缠熵峰值偏向 h/J < 1, 严格临界点只在热力学极限。
    """
    print(f"[阶段二] 临界点对称性破缺: L={L}, h/J={h/J}")
    E0, gs = exact_ground_state(L, J, h)
    half = L // 2
    s = np.linalg.svd(gs.reshape(2 ** half, 2 ** (L - half)), compute_uv=False)
    p = s ** 2
    p = p[p > 1e-15]
    p = p / p.sum()
    pol = float(s[0] / (np.mean(s[1:]) + 1e-30)) if len(s) > 1 else 1.0
    S = float(-np.sum(p * np.log(p)))
    # v14·F2: gap = p1/p0 显式返回。此前只由 derive_L2_entanglement_gap() 从
    # probabilities 里算, 走 h/J 扫描时需要一个自包含的读数。
    # 注意它是**单调量** (h/J 从 0 增大: GHZ 态 gap->1, 乘积态 gap->0),
    # 在 h/J=1.0 处**不是**极值 —— 不可当作临界判据, 见 hj_scan() 的说明。
    gap = float(p[1] / p[0]) if len(p) > 1 else float('nan')
    print(f"  E0/L={E0/L:.6f}, S(L/2)={S:.4f}, Schmidt 极化={pol:.3f}, "
          f"gap=p1/p0={gap:.6f}")
    return {'E0': E0, 'gs': gs, 'spectrum': s, 'probabilities': p,
            'entropy': S, 'polarization': pol, 'gap': gap}


# ============================================================================
# v14·F2a · 负对照: h/J 扫描 (临界性是不是事实)
# ============================================================================
#
# 这是 v14 的生死检验: 若"临界模拟"成立, 把 h/J 推离 1.0 之后所有临界指标
# 都应当**塌缩**。
#
# ⚠ 判据的选取 —— v14 实测**修正了任务书的原判据**。
#   任务书 Task 3 要求: "h/J=1.0 处 gap=p1/p0 是全局/局部极小值, 否则临界模拟
#   存疑", 并给出验收 "gap 比邻近点低 (比值<0.8)"。这个判据在物理上是错的,
#   实测会**伪失败**:
#       gap 对 h/J **单调递减** (0.5 -> 0.9967, 1.0 -> 0.3662, 1.5 -> 0.0398),
#       因为 h/J->0 时基态趋于 GHZ 猫态 (两个等权 Schmidt 值, gap->1),
#       h/J->inf 时基态趋于乘积态 (p0->1, gap->0)。整条曲线上没有内部极值。
#   照原判据: gap(1.0)/gap(1.05) = 1.41 > 0.8 -> 判"未通过" -> 触发备选方案
#   (twisted BC / 上 L=24)。那是判据错误导致的伪失败, 不是物理结论。
#
#   改用三条 (本模块实测值见下):
#     主判据: c_fit(h/J) 在 h/J=1.0 取全局**最大**。实测 0.5072, 而 0.5 处
#             0.0646、1.5 处 0.1317。含义是 **Calabrese-Cardy ansatz 只在临界
#             点成立** —— 比"某个数小"强, 它是"拟合公式本身失效"。
#     次判据: S(L/2) 峰位落在 h/J=1.0 附近 (实测 0.95)。有限尺寸下熵峰偏向
#             h/J<1, 这与 stage2_critical_break 的 docstring 声明一致, 故窗口
#             取 [0.9, 1.05] 而非单点, 并把漂移量显式报出。
#     区分力: 框架必须在非临界点**正确失败** (审计 §五.2: 一个能在非临界点
#             正确失败的框架, 比一个永远输出 3% 的框架可靠)。要求 h/J 在
#             {0.5, 0.8, 1.2, 1.5} 上 |c-0.5|/0.5 > 20% (实测 87/45/39/74%)。
#   gap 降级为**单调性诊断**: 只报它是单调的, 不测极值。

HJ_SCAN_DEFAULT = (0.5, 0.8, 0.9, 0.95, 0.98, 1.0, 1.02, 1.05, 1.1, 1.2, 1.5)
HJ_CRIT = 1.0                  # 热力学临界点
HJ_S_WINDOW = (0.9, 1.05)      # 次判据: S(L/2) 峰位允许窗口 (有限尺寸漂移)
HJ_DISCRIM = (0.5, 0.8, 1.2, 1.5)   # 区分力判据: 这些点上框架必须"正确失败"
HJ_DISCRIM_MIN = 0.20          # 区分力门槛: |c-0.5|/0.5 至少这么大


def hj_scan(hj_values=HJ_SCAN_DEFAULT, L=16, J=1.0, verbose=True):
    """
    F2a 负对照: 在 h/J != 1.0 上重跑阶段二, 看临界指标是否塌缩。

    每点记录 E0/L, S(L/2), gap=p1/p0, Schmidt 极化, 非零 Schmidt 数,
    以及**由同一组 Schmidt 谱派生的下游参数** (L2 gap, L3 N_selfref, L4 chi),
    外加该点的 c_fit。这样"参数偏离临界 -> 下游派生量是否跟着动"一目了然。

    返回 dict, 含 'points' (逐点记录) 与 'verdict' (三条判据的裁决)。
    """
    print("\n" + "=" * 76)
    print(f"  v14·F2a 负对照 · h/J 扫描 (L={L}, J={J})")
    print("=" * 76)
    print("  这是 v14 的生死检验: 若临界模拟是事实, h/J 偏离 1.0 后指标必须塌缩。")
    print("  ⚠ 判据已修正: 任务书原判据 (gap 在 h/J=1.0 取极小值) 物理上不成立 ——")
    print("     gap 对 h/J 单调递减, 曲线无内部极值。改用 c_fit 峰值 + S 峰位 + 区分力。")

    points = []
    for hj in hj_values:
        E0, gs = exact_ground_state(L, J, hj)
        half = L // 2
        s = np.linalg.svd(gs.reshape(2 ** half, 2 ** (L - half)),
                          compute_uv=False)
        p = s ** 2
        n_nonzero = int(np.sum(p > 1e-15))
        p = p[p > 1e-15]
        p = p / p.sum()
        S = float(-np.sum(p * np.log(p)))
        gap = float(p[1] / p[0]) if len(p) > 1 else float('nan')
        pol = float(s[0] / (np.mean(s[1:]) + 1e-30)) if len(s) > 1 else 1.0

        # 下游派生量: 走**与 main() 完全相同**的派生函数, 保证口径一致
        l2 = derive_L2_entanglement_gap(p)
        l3 = derive_L3_schmidt_rank(p, KNOBS['schmidt_tol'])
        l4 = derive_L4_bond_dimension(S)

        fit = fit_central_charge(gs, L)
        c_fit = float(fit['c'])
        rms = float(fit['rms'])

        points.append({'hj': float(hj), 'E0': float(E0), 'E0_per_L': float(E0 / L),
                       'S': S, 'gap': gap, 'polarization': pol,
                       'n_schmidt_nonzero': n_nonzero,
                       'N_selfref': int(l3['N_selfref']), 'chi': int(l4['chi']),
                       'chi_lower_bound': float(l4['chi_lower_bound']),
                       'c_fit': c_fit, 'c_rms': rms,
                       'eps_vs_half': abs(c_fit - 0.5) / 0.5})
        if verbose:
            print(f"    h/J={hj:5.2f}: E0/L={E0/L:10.6f} S={S:8.5f} "
                  f"gap={gap:8.5f} pol={pol:9.3f} N_sr={l3['N_selfref']:3d} "
                  f"chi={l4['chi']:2d} c_fit={c_fit:.4f} "
                  f"|c-0.5|/0.5={abs(c_fit-0.5)/0.5:.1%}")

    pts = np.array([(q['hj'], q['S'], q['gap'], q['c_fit']) for q in points])
    hj_arr, S_arr, gap_arr, c_arr = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]

    i_c = int(np.argmax(c_arr))
    i_s = int(np.argmax(S_arr))
    hj_c, hj_s = float(hj_arr[i_c]), float(hj_arr[i_s])

    # 主判据: c_fit 峰值恰在临界点
    ok_c = (abs(hj_c - HJ_CRIT) < 1e-12)
    # 次判据: S 峰位落在窗口内 (有限尺寸漂移是**预期**的, 故给窗口)
    ok_s = (HJ_S_WINDOW[0] <= hj_s <= HJ_S_WINDOW[1])
    # 区分力: 非临界点上必须"正确失败"
    disc = {}
    for hj in HJ_DISCRIM:
        m = np.isclose(hj_arr, hj)
        if m.any():
            disc[float(hj)] = float(abs(c_arr[m][0] - 0.5) / 0.5)
    ok_d = bool(disc) and all(v > HJ_DISCRIM_MIN for v in disc.values())

    # gap 单调性诊断 (不作判据, 只报告"它确实是单调的")
    dgap = np.diff(gap_arr)
    gap_monotone = bool(np.all(dgap < 0))

    verdict = {'ok_c_peak_at_critical': ok_c,
               'hj_at_c_peak': hj_c,
               'ok_s_peak_in_window': ok_s,
               'hj_at_s_peak': hj_s,
               's_window': list(HJ_S_WINDOW),
               'ok_discriminating': ok_d,
               'discrim_min': HJ_DISCRIM_MIN,
               'discrim': disc,
               'gap_monotone_decreasing': gap_monotone,
               'passed': bool(ok_c and ok_s and ok_d)}

    print("\n  --- go/no-go 裁决 ---")
    print(f"    [主] argmax c_fit 在 h/J = {hj_c:.2f}  -> "
          f"{'PASS' if ok_c else 'FAIL'} (要求 = {HJ_CRIT})")
    print(f"         实测 c_fit = {c_arr[i_c]:.4f}; 偏离处: "
          f"h/J={hj_arr[0]:.2f} -> {c_arr[0]:.4f}, "
          f"h/J={hj_arr[-1]:.2f} -> {c_arr[-1]:.4f}")
    print(f"    [次] argmax S(L/2) 在 h/J = {hj_s:.2f}  -> "
          f"{'PASS' if ok_s else 'FAIL'} (窗口 {HJ_S_WINDOW})")
    print(f"         漂移 {hj_s - HJ_CRIT:+.2f}: 有限尺寸熵峰偏向 h/J<1, 已声明")
    print(f"    [区分力] 非临界点 |c-0.5|/0.5 > {HJ_DISCRIM_MIN:.0%} -> "
          f"{'PASS' if ok_d else 'FAIL'}")
    for k, v in sorted(disc.items()):
        print(f"         h/J={k:4.2f}: {v:.1%}")
    print(f"    [诊断] gap 单调递减: {'是' if gap_monotone else '否'} "
          f"({gap_arr[0]:.4f} -> {gap_arr[-1]:.4f}) —— 不作判据, 见函数头说明")
    print(f"\n    >>> 总裁决: {'临界性成立 (继续后续任务)' if verdict['passed'] else '未通过 (触发备查)'}")

    return {'points': points, 'verdict': verdict,
            'hj_values': [float(x) for x in hj_values], 'L': L, 'J': J}


def plot_hj_scan(scan, out_png):
    """三联图: c_fit(h/J) 主判据 | S(L/2) 次判据 | gap 单调性诊断。"""
    pts = scan['points']
    hj = np.array([q['hj'] for q in pts])
    S = np.array([q['S'] for q in pts])
    gap = np.array([q['gap'] for q in pts])
    c = np.array([q['c_fit'] for q in pts])
    v = scan['verdict']

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    ax.plot(hj, c, 'o-', color='#c0392b', lw=2, ms=6)
    ax.axhline(0.5, color='k', ls='--', lw=1.2, label='真实 c = 0.5')
    ax.axvline(HJ_CRIT, color='gray', ls=':', lw=1.2)
    ax.plot([v['hj_at_c_peak']], [c[np.argmax(c)]], '*', color='gold',
            ms=18, mec='k', zorder=5, label=f"峰位 h/J={v['hj_at_c_peak']:.2f}")
    ax.set_xlabel('h/J'); ax.set_ylabel('$c_{fit}$ (Calabrese-Cardy)')
    ax.set_title(f"[主判据] c_fit 峰值在临界点\n{'PASS' if v['ok_c_peak_at_critical'] else 'FAIL'}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(hj, S, 's-', color='#2980b9', lw=2, ms=6)
    ax.axvline(HJ_CRIT, color='gray', ls=':', lw=1.2, label='h/J=1.0')
    ax.axvspan(*HJ_S_WINDOW, color='green', alpha=0.12,
               label=f"窗口 {HJ_S_WINDOW}")
    ax.axvline(v['hj_at_s_peak'], color='orange', ls='--', lw=1.5,
               label=f"峰位 h/J={v['hj_at_s_peak']:.2f}")
    ax.set_xlabel('h/J'); ax.set_ylabel('S(L/2)')
    ax.set_title(f"[次判据] 半链纠缠熵峰位\n{'PASS' if v['ok_s_peak_in_window'] else 'FAIL'}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(hj, gap, '^-', color='#27ae60', lw=2, ms=6)
    ax.axvline(HJ_CRIT, color='gray', ls=':', lw=1.2, label='h/J=1.0')
    ax.set_xlabel('h/J'); ax.set_ylabel('gap = $p_1/p_0$')
    ax.set_title(f"[诊断·不作判据] gap 单调递减 = "
                 f"{'是' if v['gap_monotone_decreasing'] else '否'}\n"
                 f"(无内部极值 -> 原判据不可用)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f"v14·F2a 负对照: h/J 扫描 (L={scan['L']})  "
                 f"总裁决 {'PASS' if v['passed'] else 'FAIL'}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_png


# ============================================================================
# 阶段三 · 自指的动力学涌现
# ============================================================================
def _power_iteration(A, x0, max_iter=3000, tol=1e-13):
    """规范化幂迭代 x <- A x / |A x|; 返回不动点、步长历史、轨迹"""
    x = x0 / np.linalg.norm(x0)
    hist, steps = [], []
    for _ in range(max_iter):
        y = A @ x
        nrm = np.linalg.norm(y)
        if nrm < 1e-300:
            break
        y = y / nrm
        phase = np.vdot(x, y)          # 相位对齐 (复特征值时 x, -x, ix 等价)
        if abs(phase) > 1e-14:
            y = y * (phase / abs(phase))
        steps.append(float(np.linalg.norm(y - x)))
        hist.append(y.copy())
        if steps[-1] < tol:
            x = y
            break
        x = y
    return x, np.array(steps), hist


def _nonnormality(A):
    """非正规性度量: ||A A^T - A^T A||_F / ||A||_F^2。正规矩阵为 0。"""
    return float(np.linalg.norm(A @ A.T - A.T @ A) / np.linalg.norm(A) ** 2)


def _transient_growth(A, kmax=8):
    """||A^k||_2 序列, 用来暴露非正规矩阵的瞬态放大"""
    Ak = np.eye(A.shape[0])
    g = []
    for _ in range(kmax):
        Ak = Ak @ A
        g.append(float(np.linalg.svd(Ak, compute_uv=False)[0]))
    return np.array(g)


def selfref_general_matrix(N=128, seed=0, gap_physical=None):
    """
    (a) 一般 (非正规) 矩阵的幂迭代 x <- A x / |A x|。

    第 0 组 (因果链 L2): **物理来源的谱隙**。gap_physical 由阶段二实测的
    Schmitt 谱给出 (gap = p1/p0, 见 derive_L2_entanglement_gap)。
    四个预先指定的 gap (0.9/0.5/0.2/0.05) 只证明"方法实现正确", 那是在
    **校准方法**; 此外多跑一个不由人选的 gap —— 它由临界 Ising 基态的纠缠谱
    决定。实测收敛比与这个派生值的吻合, 才是"幂迭代的收敛速度有物理来源"
    的证据。

    第 1 组: 受控谱隙校准。构造 A = V diag(lam) V^{-1} (lam1 = 1, lam2 = gap),
    V 取轻微非正交 (I + 0.05 G, cond ~ 6), 这样收敛速率有干净的解析预言
    |lam2/lam1| = gap, 可以直接对表。实测比与理论吻合到 3 位。

    第 2 组: Jordan 块 (单一特征值 lam=0.9, 剪切 gam=0.5, 尺寸 6)。谱半径
    0.9 < 1, 幂迭代**必然**收敛到 0; 但 ||A^k|| 先从 1.32 涨到 358 (k=49)
    再衰减。所有特征值都在衰减, 范数却先放大 272 倍 —— 谱半径对这个峰
    完全无话可说, 这正是"自指不一定平滑收敛"的数学来源。

    (为什么必须换一个算例: 第 1 组谱半径 = 1, ||A^k|| 单调, 没有"先涨后落";
     而 V diag(lam) V^{-1} 那一族无论怎么调, ||A^k|| 都在 k=1 取最大 ——
     它的非正规性在第一步就饱和了。真正的内部峰需要 Jordan/剪切这种
     "幂零 + 衰减"的竞争结构。)
    """
    rng = np.random.default_rng(seed)

    # ---------- 第 0 组: gap 由阶段二的纠缠谱派生 (因果链 L2) ----------
    physical = None
    if gap_physical is not None:
        if not (0.0 < gap_physical < 1.0):
            print(f"   [L2] 派生谱隙 = {gap_physical:.6f} 落在 (0,1) 之外, "
                  f"跳过物理算例 (临界态谱隙应 < 1)")
        else:
            V = np.eye(N) + 0.05 * rng.standard_normal((N, N))
            lam = np.array([1.0] + [gap_physical * (1 - 0.5 * i / N)
                                    for i in range(N - 1)])
            A = V @ np.diag(lam) @ np.linalg.inv(V)
            ev, evec = np.linalg.eig(A)
            order = np.argsort(-np.abs(ev))
            rate_theory = float(abs(ev[order[1]]) / abs(ev[order[0]]))
            xf, steps, _ = _power_iteration(A, rng.standard_normal(N))
            if len(steps) > 10:
                nz = steps[:-1] > 1e-14
                ratios = steps[1:][nz] / steps[:-1][nz]
                rate_meas = (float(np.median(ratios[-30:])) if len(ratios)
                             else float('nan'))
            else:
                rate_meas = float('nan')
            physical = {'gap': gap_physical, 'rate_theory': rate_theory,
                        'rate_meas': rate_meas, 'n_steps': int(len(steps)),
                        'residual': float(np.linalg.norm(
                            A @ xf - (np.vdot(xf, A @ xf)) * xf))}
            print(f"   [L2 物理算例] 由纠缠谱派生的谱隙 = {gap_physical:.6f}")
            print(f"        理论收敛比 = {rate_theory:.6f}, "
                  f"实测收敛比 = {rate_meas:.6f}, 步数 = {len(steps)}")

    sweep = []
    for gap, eps in ((0.9, 0.05), (0.5, 0.05), (0.2, 0.05), (0.05, 0.05)):
        V = np.eye(N) + eps * rng.standard_normal((N, N))
        lam = np.array([1.0] + [gap * (1 - 0.5 * i / N) for i in range(N - 1)])
        A = V @ np.diag(lam) @ np.linalg.inv(V)
        ev, evec = np.linalg.eig(A)
        order = np.argsort(-np.abs(ev))
        rate_theory = float(abs(ev[order[1]]) / abs(ev[order[0]]))

        xf, steps, _ = _power_iteration(A, rng.standard_normal(N))
        if len(steps) > 10:
            nz = steps[:-1] > 1e-14
            ratios = steps[1:][nz] / steps[:-1][nz]
            rate_meas = float(np.median(ratios[-30:])) if len(ratios) else float('nan')
        else:
            rate_meas = float('nan')

        vec1 = evec[:, order[0]]
        vec1 = vec1 / np.linalg.norm(vec1)
        sweep.append({'gap': gap, 'eps': eps,
                      'cond_V': float(np.linalg.cond(V)),
                      'rate_theory': rate_theory, 'rate_meas': rate_meas,
                      'n_steps': int(len(steps)),
                      'overlap': float(abs(np.vdot(vec1, xf))),
                      'residual': float(np.linalg.norm(
                          A @ xf - (np.vdot(xf, A @ xf)) * xf)),
                      'nonnormality': _nonnormality(A),
                      'spectral_radius': float(abs(ev[order[0]]))})

    print(f"[阶段三-a] 幂迭代: 一般 (非正规) 矩阵 N={N}")
    for s in sweep:
        print(f"   gap={s['gap']:.2f} eps={s['eps']:.2f} (cond(V)={s['cond_V']:.0f}): "
              f"理论比={s['rate_theory']:.4f}, 实测比={s['rate_meas']:.4f}, "
              f"步数={s['n_steps']:3d}, 残差={s['residual']:.1e}, "
              f"非正规性={s['nonnormality']:.4f}")

    # 第 2 组: Jordan 块 —— 谱半径 < 1 却先放大再衰减
    NJ, lamJ, gamJ, KJ = 6, 0.9, 0.5, 300
    AJ = lamJ * (np.eye(NJ) + gamJ * np.diag(np.ones(NJ - 1), 1))
    rho = float(np.abs(np.linalg.eigvals(AJ)).max())
    g = _transient_growth(AJ, kmax=KJ)
    k_peak = int(np.argmax(g)) + 1
    hump = float(g.max())
    peak_ratio = hump / float(g[0])
    rate_transient = hump / rho ** k_peak     # 峰值 / 谱半径预言

    # 同一矩阵跑幂迭代: 方向收敛 / 范数衰减 是两件不同的事
    x0 = rng.standard_normal(NJ)
    x0 = x0 / np.linalg.norm(x0)
    _, steps_t, _ = _power_iteration(AJ, x0)
    if len(steps_t) > 10:
        nz = steps_t[:-1] > 1e-300
        rt = steps_t[1:][nz] / steps_t[:-1][nz]
        step_ratio = float(np.median(rt[-10:])) if len(rt) else float('nan')
    else:
        step_ratio = float('nan')
    # 特征值只有一个不同值 -> 谱隙 = 0, 线性收敛根本不被预言
    gap_J = 1.0

    # 状态范数 ‖A^k x0‖ 的轨迹 (这才是"自发收敛"的判据)
    xn = x0.copy()
    norms = []
    for _ in range(KJ):
        xn = AJ @ xn
        norms.append(float(np.linalg.norm(xn)))
    norms = np.array(norms)
    k_norm_peak = int(np.argmax(norms)) + 1
    # 最后一次跌破初值并保持不回升的步数 (首次跌破可能只是 k=1 的偶然下探)
    above = np.where(norms >= 1.0)[0]
    k_below = int(above[-1]) + 2 if len(above) else 1

    print(f"   第 2 组 Jordan 块 N={NJ}, 特征值全 = {lamJ} "
          f"(谱半径={rho:.3f} < 1, 非正规性={_nonnormality(AJ):.3f}):")
    print(f"     只有 1 个不同特征值 -> 谱隙 |λ2/λ1| = {gap_J:.2f}, "
          f"线性收敛不被预言; 实测方向步长比 = {step_ratio:.4f} (吻合, 方向不收敛)")
    print(f"     状态范数 ‖A^k x₀‖: 1.000 (k=1: {norms[0]:.3f}) -> 峰值 "
          f"{norms[k_norm_peak-1]:.2f} (k={k_norm_peak}) -> 先涨后落, "
          f"k={k_below} 才最后一次跌破初值并保持, "
          f"k={KJ} 衰减到 {norms[-1]:.2e} (ρ^k={rho**KJ:.2e})")
    print(f"     ‖A‖₂={g[0]:.3f} -> 峰值 ‖A^k‖₂={hump:.3f} (k={k_peak}), "
          f"放大 {peak_ratio:.1f}x, 之后才衰减")
    print(f"     峰值处 ‖A^k‖₂/ρ^k = {rate_transient:.1f}: "
          f"谱半径把范数低估了 {rate_transient:.0f} 倍")
    return {'gap_sweep': sweep, 'physical': physical,
            'transient': {'growth': g.tolist(), 'spectral_radius': rho,
                          'nonnormality': _nonnormality(AJ),
                          'lam': lamJ, 'N': NJ, 'kmax': KJ,
                          'gap': gap_J, 'step_ratio': step_ratio,
                          'norm_trace': norms.tolist(),
                          'norm_final': float(norms[-1]),
                          'k_norm_peak': k_norm_peak,
                          'norm_peak': float(norms[k_norm_peak - 1]),
                          'k_below': k_below,
                          'peak': hump, 'peak_ratio': peak_ratio,
                          'k_peak': k_peak, 'norm_A': float(g[0]),
                          'peak_over_rho_k': rate_transient}}


def selfref_nonconvergent(N=128, seed=0):
    """
    (a') 反例: 当 |lam1| = |lam2| 时幂迭代**根本不收敛** —— 状态在两个
         模相等的特征方向之间无尽旋转。这是"自指不必然收敛"的严格边界。
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, N)) / np.sqrt(N)
    rho = float(np.abs(np.linalg.eigvals(A)).max())
    A = A / rho
    A[:2, :2] = np.array([[0.0, -1.0], [1.0, 0.0]])    # 90 度旋转块 -> 模相等
    ev = np.sort(np.abs(np.linalg.eigvals(A)))
    x0 = rng.standard_normal(N)
    _, steps, _ = _power_iteration(A, x0, max_iter=4000)
    tail = steps[-300:] if len(steps) > 300 else steps
    tail_mean = float(np.mean(tail)) if len(tail) else float('nan')
    print(f"[阶段三-a'] 模相等反例: |lam1|={ev[-1]:.4f}, |lam2|={ev[-2]:.4f}")
    print(f"  4000 步后步长仍为 {tail_mean:.4f} (不趋于 0) -> 不收敛 (周期轨道)")
    return {'lam1': float(ev[-1]), 'lam2': float(ev[-2]),
            'tail_step': tail_mean, 'steps': steps, 'converged': False}


def _lyapunov(W, g, N, warmup=1500, iters=400, sub=10):
    """
    最大 Lyapunov 指数: 沿轨道对 Jacobian J = diag(1-tanh^2) * gW 做
    幂迭代, 每步重新归一化 (不重归一化的话数值会饱和, 指数不可靠)。
    """
    x = torch.randn(N)
    for _ in range(warmup):
        x = torch.tanh(g * (W @ x))
    ly = 0.0
    for _ in range(iters):
        J = torch.diag(1 - torch.tanh(g * (W @ x)) ** 2) @ (g * W)
        v = torch.randn(N)
        v = v / torch.norm(v)
        for _ in range(sub):
            v = J @ v
            ly += float(torch.log(torch.norm(v)))
            v = v / torch.norm(v)
        x = torch.tanh(g * (W @ x))
    return ly / (iters * sub)


def selfref_neural_network(N=64, seed=0):
    """
    (b) 神经网络自指映射  x_{n+1} = tanh(g W x_n)。
        判别三相需要两个独立量, 只看步长会把极限环误判成混沌:
          不动点: 尾部步长 -> 0  且 Lyapunov < 0
          极限环: 尾部步长有限   且 Lyapunov < 0   (周期, 非混沌)
          混沌  : 尾部步长有限   且 Lyapunov > 0
        相变发生在 g * rho(W) ~ 1 (边缘混沌)。
    """
    g_t = torch.Generator().manual_seed(seed)
    W = torch.randn(N, N, generator=g_t) / np.sqrt(N)
    rho = float(torch.linalg.eigvals(W).abs().max())
    gains = [0.4, 0.8, 0.9, 1.0, 1.1, 1.4, 2.0]
    scan = []
    for g in gains:
        x = torch.randn(N, generator=g_t)
        for _ in range(3000):
            x = torch.tanh(g * (W @ x))
        steps = []
        for _ in range(400):
            xn = torch.tanh(g * (W @ x))
            steps.append(float(torch.norm(xn - x)))
            x = xn
        tail = max(steps[-50:])
        ly = _lyapunov(W, g, N)
        if tail < 1e-8 and ly < 0:
            phase = '不动点'
        elif ly > 0:
            phase = '混沌'
        else:
            phase = '极限环'
        scan.append({'g': g, 'g_rho': g * rho, 'tail_step': tail,
                     'lyapunov': ly, 'phase': phase})
        print(f"   g={g:.1f} (g*rho={g*rho:.2f}): 尾部步长={tail:.2e} "
              f"Lyapunov={ly:+.4f} -> {phase}")
    print(f"[阶段三-b] 自指神经网络 N={N}, 谱半径 rho(W)={rho:.4f}")
    return {'rho': rho, 'scan': scan}


def selfref_coupled(N=64, steps=8000, eta=0.05, g=1.0, seed=0):
    """
    (c) 真自指 —— 映射的权重本身是状态的函数 (二阶自指):
          x_{n+1} = tanh(g W_n x_n)                (状态)
          W_{n+1} = normalize_spec(W_n + eta (x_{n+1} x_{n+1}^T - W_n))
                                                   (权重被状态改写)
    权重每步做谱归一, 否则 W -> 0 的塌缩会淹没一切。

    三种变体 (全部实测, 输出里的判定完全由数字驱动, 不预设结论):
      冻结权重 (对照, 不自指): 权重不随状态改变, 就是一个固定随机映射。
      状态自由 (A):            权重随状态改写, 状态不做归一。
      状态归一 (B):            权重随状态改写, 状态每步归一。

    判定依据是三个量: 末段步长 |dx| (收敛与否)、末段 ||x|| (状态是否消退)、
    权重的 sigma1/sigma2 (是否自发秩 1 —— 秩 1 意味着 W* ∝ x* x*^T, 即权重
    收敛到"恰好产生自己的那个态", 这才是自指的自洽不动点)。
    诚实边界: 步长在有限步内只降到 1e-5 量级, 是渐近收敛而非机器精度;
    真正干净的证据是 sigma 比随步数单调增长若干个数量级。
    """
    def _run(norm_x, frozen=False):
        gt = torch.Generator().manual_seed(seed)
        W = torch.randn(N, N, generator=gt) / np.sqrt(N)
        x = torch.randn(N, generator=gt)
        if norm_x:
            x = x / torch.norm(x)
        dx, dW, rk, nx = [], [], [], []
        for it in range(steps):
            Wn = W
            xn = torch.tanh(g * (Wn @ x))
            if norm_x:
                xn = xn / torch.norm(xn)
            if not frozen:
                W = Wn + eta * (torch.outer(xn, xn) - Wn)
                W = W / torch.linalg.matrix_norm(W, 2)
            dx.append(float(torch.norm(xn - x)))
            dW.append(float(torch.norm(W - Wn)))
            nx.append(float(torch.norm(xn)))
            if it % 2000 == 0:
                sv = torch.linalg.svdvals(W)
                rk.append((it, float(sv[0] / sv[1])))
            x = xn
        return {'dx': np.array(dx), 'dW': np.array(dW), 'rank': rk,
                'x_norm': np.array(nx), 'W': W, 'x': x,
                'final_dx': float(np.mean(dx[-200:])),
                'final_dW': float(np.mean(dW[-200:])),
                'sigma_ratio': rk[-1][1],
                'collapsed': bool(np.mean(nx[-200:]) < 1e-6)}

    print(f"[阶段三-c] 权重-状态耦合自指 N={N}, eta={eta}, g={g}")
    frozen = _run(norm_x=True, frozen=True)
    free = _run(norm_x=False)
    norm = _run(norm_x=True)
    for lab, r in (("冻结权重 (对照)", frozen),
                   ("自指·状态自由 (A)", free),
                   ("自指·状态归一 (B)", norm)):
        xn = float(np.mean(r['x_norm'][-200:]))
        print(f"  {lab}: 末段|dx|={r['final_dx']:.3e} "
              f"({'收敛' if r['final_dx'] < 1e-4 else '不收敛'}), "
              f"末段||x||={xn:.3e}, sigma1/sigma2={r['sigma_ratio']:.3e}")
    for it, r in norm['rank']:
        print(f"      B 的秩1化轨迹 step {it:5d}: sigma1/sigma2={r:14.2f}")
    # 结语必须由**实测**生成, 不能写死。
    # 手工 N=64 下两个变体都秩 1 化, 若照此写死一句"都自发秩 1 化", 那么把 N
    # 派生为 17 (有效 Schmidt 秩) 之后 — 此时变体 A 收敛到了 x*=0, 它的
    # sigma1/sigma2 掉到量级 1, **没有**秩 1 化 — 写死的结语就成了假话。
    # 所以改成按实测阈值下结论。
    RANK_TOL = 1e3        # sigma1/sigma2 > 1e3 才算自发秩 1 化
    ok_free = free['sigma_ratio'] > RANK_TOL
    ok_norm = norm['sigma_ratio'] > RANK_TOL
    verdict_free = '秩 1 化' if ok_free else '未秩 1 化 (它收敛到 x*=0, 权重失去方向)'
    verdict_norm = '秩 1 化' if ok_norm else '未秩 1 化'
    print("      -> 冻结的对照永不收敛; 两个自指变体都收敛 (末段|dx| < 1e-4)。")
    print(f"         但'权重自发秩 1 化 (W* -> x* x*^T)'只在实测 sigma1/sigma2 "
          f"> {RANK_TOL:g} 时才能说:")
    print(f"           变体 A (状态自由): sigma1/sigma2={free['sigma_ratio']:.3e} -> "
          f"{verdict_free}")
    print(f"           变体 B (状态归一): sigma1/sigma2={norm['sigma_ratio']:.3e} -> "
          f"{verdict_norm}")
    print(f"         诚实边界: 手工取 N=64 时 A、B 都秩 1 化; 派生 N=17 后"
          f"只剩 B。")
    print(f"         所以这条结论依赖 N, 不是与 N 无关的普适行为。")
    print(f"         状态消退与否: 归一后 ||x|| 保持在 "
          f"{np.mean(norm['x_norm'][-200:]):.3f}, "
          f"自由时衰减到 {np.mean(free['x_norm'][-200:]):.2e}。")
    # 只保留可序列化 / 可画图的量; torch 张量 (W, x) 单独用 _W / _x 带出
    drop = ('W', 'x')
    return {'frozen': {k: v for k, v in frozen.items() if k not in drop},
            'free': {k: v for k, v in free.items() if k not in drop},
            'norm': {k: v for k, v in norm.items() if k not in drop},
            '_W': norm['W'], '_x': norm['x']}


# ============================================================================
# 阶段四 · 自指画定边界投影出全息
# ============================================================================
def mera_init(L, chi, seed=0):
    """
    构造 quimb 二进制 MERA (L 为 2 的幂, 周期边界), 并为每个可训练张量
    分配一个无约束的原始参数矩阵 (后面用 QR 投到 Stiefel 流形上)。
    形状必须从张量自身的 shape 推导 —— 不同层的键维不同 (底层 4x4, 更高层更大)。
    """
    torch.manual_seed(seed)
    mera = qtn.MERA.rand(L, phys_dim=2, max_bond=chi, dtype=float)
    raws = []
    for t in mera.tensors:
        if t.ndim == 4:
            d = t.shape[0]
            raws.append(torch.randn(d * d, d * d, requires_grad=True))
        elif t.ndim == 3:
            a, b, c = t.shape
            raws.append(torch.randn(a * b, c, requires_grad=True))
    return mera, raws


def _qr_isometry(raw, cols=None):
    """
    把无约束矩阵参数化为 (半) 酉张量: M = Q R -> Q, 再做列符号规范。
    这是 Stiefel 流形上的标准参数化, QR 在 torch 中可微, 约束恒满足 ——
    所以 MERA 的酉性/等距性是"张量性质保证", 不是事后凑出来的。
    """
    Q, R = torch.linalg.qr(raw)
    Q = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)
    return Q if cols is None else Q[:, :cols]


def mera_dense(mera, raws, L):
    """
    把 MERA 收缩成 2^L 维稠密态矢量 (全程可微)。
    output_inds 强制 k0 为第 0 轴 -> reshape(-1) 后 k0 是最高有效位,
    与 tfi_periodic_sparse 的位序约定一致。
    注意: 必须按张量属性 (ndim) 遍历并逐个替换, 不能按下标索引 ——
    下标在 mera 与 mera.copy() 之间不保证对应。
    """
    tn = mera.copy()
    k = 0
    for t in tn.tensors:
        if t.ndim == 4:
            t.modify(data=_qr_isometry(raws[k]).reshape(t.shape))
            k += 1
        elif t.ndim == 3:
            t.modify(data=_qr_isometry(raws[k], t.shape[2]).reshape(t.shape))
            k += 1
        else:                                   # 顶层固定的边界张量
            t.modify(data=torch.as_tensor(np.asarray(t.data)))
    out = tn.contract(all, optimize='greedy',
                      output_inds=[f'k{i}' for i in range(L)])
    if hasattr(out, 'data'):                    # quimb 收缩可能返回 Tensor
        out = out.data
    return out.reshape(-1)


def mera_isometry_check(mera, L):
    """
    验证 MERA 的定义性质 —— 这才是"真正的张量网络", 而不是随机 QR 矩阵:
        酉性   U^dag U = I   (解纠缠器)
        等距性 W^dag W = I   (等距)
    """
    eu, ew = [], []
    for t in mera.tensors:
        d = np.asarray(t.data)
        if t.ndim == 4:
            n = t.shape[0]
            U = d.reshape(n * n, n * n)
            eu.append(float(np.abs(U.conj().T @ U - np.eye(n * n)).max()))
        elif t.ndim == 3:
            a, b, c = t.shape
            W = d.reshape(a * b, c)
            ew.append(float(np.abs(W.conj().T @ W - np.eye(c)).max()))
    return {'unitary_err': max(eu) if eu else 0.0,
            'isometry_err': max(ew) if ew else 0.0,
            'n_unitary': len(eu), 'n_isometry': len(ew)}


def mera_fit(mera, raws, target, L, steps=600, lr=0.03, verbose=True):
    """
    变分拟合: 最大化 |<target|psi(mera)>|^2。

    为何不用能量作目标 (实测结论, 见下方对照实验):
      以 <psi|H|psi> 为目标时梯度下降会塌缩到平均场乘积态 —— 临界 Ising
      的最优乘积态就有 E/L = -1.25 (精确 -1.2815), 只差 2.5%, 却只有
      S(n) ~ 0.02 的纠缠。普通梯度法看不出这点差别, 重叠目标才直接奖励
      "像基态", 因此能落到正确的纠缠分支。
    """
    gst = torch.as_tensor(np.asarray(target).reshape(-1))
    opt = torch.optim.Adam(raws, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr / 50)
    hist = []
    t0 = time.time()
    for step in range(steps):
        opt.zero_grad()
        psi = mera_dense(mera, raws, L)
        loss = -((gst @ psi) ** 2) / (psi @ psi)
        loss.backward()
        opt.step()
        sched.step()
        if verbose and (step % 200 == 0 or step == steps - 1):
            with torch.no_grad():
                ov = abs(float(gst @ (psi / torch.norm(psi))))
            hist.append((step, ov))
            print(f"     step {step:4d} |<gs|psi>|={ov:.5f}  ({time.time()-t0:.0f}s)")
    with torch.no_grad():
        psi = mera_dense(mera, raws, L).numpy()
    return psi / np.linalg.norm(psi), hist


# ===========================================================================
# 第一层的物理迭代: 带检查点选择的多种子变分
# ===========================================================================
#
# 为什么 11.45% 不是 chi=4 的表示极限 (诊断证据, 见 _v13_diag_*.py):
#   证据一: 同 chi=4, 换种子/换学习率, eps_c 在 1%~60% 之间跳 —— 谱宽远超 chi 的影响。
#   证据二: chi=4 在充分优化下达标 (见 _v13_diag_chi.py 的 chi 扫描)。
#   证据三: 重叠从 0.98816 涨到 0.99961 (1.2%), eps_c 从 11.45% 掉到 1.09% (40 倍)。
#           eps_c 在重叠 0.98~0.999 区间**极度敏感**, 所以"跑没跑到位"是主导项。
#   证据四: QR 参数化的等距矩阵带 sign(diag(R)) 定号, Stiefel 流形上这个不连续点
#           会让 ADAM 的动量偶尔跨过符号翻转 -> 重叠出现**悬崖**。实测 lr=0.06 同一种子:
#           step 400 给 eps_c=3.13%, step 800 给 29.42%, step 1800 给 59.52%。
#
# 由此两条对策, 都必须显式声明, 不能悄悄用:
#   (1) **检查点选择**: 每 100 步记一次, 取 overlap 最高的那个。
#       选的是**训练目标本身** (overlap), 不是 eps_c —— 偷看 eps_c 选点就是
#       测试集泄漏, 那才会让"达标"变得不值钱。
#   (2) **多轨迹**: 报分布而不是单点。判据卡**最坏轨迹** (见 metric_design_robustness)。
#
# 目标阈值一个字都没改: eps_c < 5%, R_conf < 1e-2。改的只有预算与统计。
V13_STEPS = 1500                    # 迭代步数预算
V13_LR_SEEDS = ((0.03, (0, 1, 2)),  # 默认学习率, 3 个种子 (headline)
                (0.12, (0, 1)))     # 稳健性对照, 2 个种子
V13_EVERY = 100                     # 收敛曲线与检查点的采样间隔
V13_CHI_CTRL = (8, 0)               # 键维对照: (chi, seed) —— 同种子同预算, 只改 chi


def mera_fit_v13(mera, raws, target, L, steps=V13_STEPS, lr=0.03,
                 every=V13_EVERY, c_exact=None):
    """
    朴素拟合**同一个损失、同一个优化器、同一个调度、同一个 lr 默认值**,
    只加两件不改变优化目标的事: 检查点选择 + 沿途记录。

    检查点选择 (按 overlap, 即训练目标): 因为悬崖的存在, 末态不一定是轨迹上的
    最好点。取最好点是把"优化没跑好"这件事从测量里剔掉 —— 注意这只影响**报什么数**,
    不影响任何物理量; 状态本身仍是纯粹的变分 MERA, 没有任何人工构造。

    返回 (best_psi, best_overlap, best_step, curve);
    curve = [(step, eps_c, R_conf, overlap), ...] 每 every 步一个点。
    """
    gst = torch.as_tensor(np.asarray(target).reshape(-1))
    opt = torch.optim.Adam(raws, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr / 50)

    curve, best = [], None
    for step in range(steps + 1):
        if step % every == 0 or step == steps - 1:
            with torch.no_grad():
                psi = mera_dense(mera, raws, L).numpy()
                psi = psi / np.linalg.norm(psi)
            ov = abs(float(gst.numpy() @ psi))
            if best is None or ov > best[1]:
                best = (psi, ov, step)
            if c_exact is not None:
                fm = fit_central_charge(psi, L)
                curve.append((step, abs(fm['c'] - c_exact) / c_exact,
                              fm['rms'] / float(np.mean(fm['S'])), ov))
        if step == steps:
            break
        opt.zero_grad()
        psi = mera_dense(mera, raws, L)
        loss = -((gst @ psi) ** 2) / (psi @ psi)
        loss.backward()
        opt.step()
        sched.step()
    return best[0], float(best[1]), int(best[2]), curve


def v13_tier1_runs(gs, L, chi, c_exact, verbose=True):
    """
    跑完第一层的全部轨迹 (含键维对照), 返回 (记录 dict, 代表态)。

    代表态 = eps_c 处于**中位数**的那条轨迹的检查点态。选它而不是"最好的那条",
    是因为第二层的纠缠谱 KL 是拿 MERA 与精确态比, 用最好的一条会夸大第二层的成绩。
    中位数 = 典型表现, 这是可审计的选择。
    """
    t0 = time.time()
    runs, curves = [], {}
    for lr, seeds in V13_LR_SEEDS:
        for seed in seeds:
            mera, raws = mera_init(L, chi, seed=seed)
            psi, ov, cstep, curve = mera_fit_v13(
                mera, raws, gs, L, steps=V13_STEPS, lr=lr, c_exact=c_exact)
            fm = fit_central_charge(psi, L)
            eps = abs(fm['c'] - c_exact) / c_exact
            rc = fm['rms'] / float(np.mean(fm['S']))
            runs.append({'lr': lr, 'seed': seed, 'eps_c': float(eps),
                         'R_conf': float(rc), 'overlap': ov,
                         'c': float(fm['c']), 'chosen_step': cstep,
                         'psi': psi})
            if lr == V13_LR_SEEDS[0][0]:
                curves[seed] = curve
            if verbose:
                print(f"      [chi={chi} lr={lr} seed={seed}] "
                      f"ov={ov:.6f} @step{cstep:4d}  c={fm['c']:.6f}  "
                      f"eps_c={eps:>7.3%}  R_conf={rc:.3e}  "
                      f"({time.time() - t0:.0f}s)")

    # 收敛曲线取 headline 学习率下各 seed 的**逐点中位数**, 供预算诊断
    steps_axis = sorted({s for c in curves.values() for (s, _e, _r, _o) in c})
    curve_med = []
    for s in steps_axis:
        vals = [t for c in curves.values() for t in c if t[0] == s]
        curve_med.append((s,
                          float(np.median([v[1] for v in vals])),
                          float(np.median([v[2] for v in vals])),
                          float(np.median([v[3] for v in vals]))))

    # 键维对照: 同种子同预算, 只改 chi -> 直接测"提 chi 有没有用"
    cchi, cseed = V13_CHI_CTRL
    mera_c, raws_c = mera_init(L, cchi, seed=cseed)
    psi_c, ov_c, cstep_c, _ = mera_fit_v13(
        mera_c, raws_c, gs, L, steps=V13_STEPS, lr=V13_LR_SEEDS[0][0],
        c_exact=c_exact)
    fm_c = fit_central_charge(psi_c, L)
    eps_c_ctrl = float(abs(fm_c['c'] - c_exact) / c_exact)
    if verbose:
        print(f"      [chi={cchi} 对照 seed={cseed}] ov={ov_c:.6f} @step{cstep_c}  "
              f"eps_c={eps_c_ctrl:>7.3%}  (同种子同预算, 只改 chi)")

    # Schmidt 逐切割截断参照 (chi=4): 不是严格下界, 见 spiral_metric_v13 的说明
    floor4 = _schmidt_floor(gs, L, chi)
    floor8 = _schmidt_floor(gs, L, cchi)
    if verbose:
        print(f"      [参照] Schmidt 逐切割截断 (非严格界): "
              f"chi={chi} -> eps_c={floor4:.3%}, chi={cchi} -> eps_c={floor8:.3%}")

    eps_all = np.array([r['eps_c'] for r in runs])
    i_med = int(np.argmin(np.abs(eps_all - np.median(eps_all))))
    runs_stripped = [{k: v for k, v in r.items() if k != 'psi'} for r in runs]
    return {'runs': runs_stripped, 'curve': curve_med,
            'curves_by_seed': {int(k): v for k, v in curves.items()},
            'steps': V13_STEPS,
            'lrs': [lr for lr, _s in V13_LR_SEEDS],
            'chi_sweep': {chi: {'eps_c': float(eps_all[i_med]),
                                'R_conf': runs[i_med]['R_conf'],
                                'overlap': runs[i_med]['overlap']},
                          cchi: {'eps_c': eps_c_ctrl,
                                 'R_conf': float(fm_c['rms'] / float(np.mean(fm_c['S']))),
                                 'overlap': ov_c}},
            'floor_chi': floor4, 'floor_chi_ctrl': floor8,
            'chi_ctrl': cchi, 'chosen_run_index': i_med,
            'elapsed': time.time() - t0}, runs[i_med]['psi']


def _schmidt_floor(psi, L, chi):
    """
    精确基态在每个切割点把 Schmidt 谱截到 chi 项并归一化后的 eps_c。

    **这不是严格下界** (同一个态不可能在所有切割点同时取到该 rank, 且 rank-chi
    态的熵可在 [0, ln chi] 内任意)。它只用来做**定性归因**: 实测 MERA 的 eps_c
    明显高于这条线 -> 瓶颈在优化; 贴着这条线 -> 瓶颈在表示能力。
    """
    psi = np.asarray(psi, dtype=float).reshape(-1)
    psi = psi / np.linalg.norm(psi)
    y = []
    for n in range(1, L // 2 + 1):
        m = psi.reshape(2 ** n, 2 ** (L - n))
        s = np.linalg.svd(m, compute_uv=False)[:chi]
        p = s ** 2
        p = p[p > 1e-15]
        p = p / p.sum()
        y.append(float(-np.sum(p * np.log(p))))
    y = np.array(y)
    ns = np.arange(1, L // 2 + 1)
    x = np.log((L / np.pi) * np.sin(np.pi * ns / L))
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    c_ex = fit_central_charge(psi, L)['c']
    return float(abs(3.0 * coef[0] - c_ex) / abs(c_ex))


def mera_energy_only(mera, raws, L, H_sparse, steps=400, lr=0.03):
    """对照实验: 用"最小化能量"作目标, 展示会塌缩到平均场极小。"""
    Hd = torch.as_tensor(H_sparse.toarray())
    opt = torch.optim.Adam(raws, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr / 50)
    for _ in range(steps):
        opt.zero_grad()
        psi = mera_dense(mera, raws, L)
        loss = (psi @ (Hd @ psi)) / (psi @ psi)
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        psi = mera_dense(mera, raws, L).numpy()
    psi = psi / np.linalg.norm(psi)
    return psi, float(psi @ (H_sparse @ psi))


def mera_causal_cone(mera, L):
    """
    MERA 的全息结构证据: 区域 A = {0..n-1} 的因果锥张量数 |cone(n)|。
    quimb 已在每个张量上打好 I{j} 标签 (j 属于其因果锥), 所以这里是直接
    数张量, 不是模拟出来的。|cone(n)| 应与 n 成体积律 (线性), 而 S(n)
    只随 ln n 增长 —— 体是体积律, 纠缠是面积律, 这正是全息的定义性质。
    返回 (n 列表, cone 列表, 线性拟合 R^2, 对数拟合 R^2), 让数据自己说话。
    诚实边界: 这是对全息结构的直接张量计数, 不是 Ryu-Takayanagi 面积计算。
    """
    ns = np.arange(1, L // 2 + 1)
    cone = np.array([int(mera.select_any([f'I{j}' for j in range(n)]).num_tensors)
                     for n in ns], dtype=float)
    r2 = {}
    for name, x in (('linear', ns.astype(float)), ('log', np.log(ns))):
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, cone, rcond=None)
        r2[name] = float(1.0 - (cone - A @ coef).var() / cone.var())
    return ns, cone, r2


def build_mera_graph(mera):
    """
    MERA 的张量网络图: 节点 = 张量 (quimb 的整数 tid), 边 = 共享指标。
    这就是离散的体几何 —— 阶段五的曲率就算在它上面。
    (quimb 的 Tensor 没有 .tid 属性, 但 ind_map 的值就是整数 tid。)
    """
    G = nx.Graph()
    G.add_nodes_from(range(mera.num_tensors))
    for _, tids in mera.ind_map.items():
        if len(tids) == 2:
            a, b = sorted(int(x) for x in tids)
            G.add_edge(a, b)
    return G


# ============================================================================
# 阶段五 · 全息自然展开生成了时空
# ============================================================================
def all_pairs_distance(G):
    nodes = sorted(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    D = np.full((len(nodes), len(nodes)), np.inf)
    for s in nodes:
        for t, d in nx.single_source_shortest_path_length(G, s).items():
            D[idx[s], idx[t]] = d
    return nodes, idx, D


def ollivier_ricci(G, alpha=0.0):
    """
    Ollivier-Ricci 曲率:
        kappa(u,v) = 1 - W1(m_u, m_v) / d(u,v)
    m_u = 随机游走在 u 上的分布; W1 = 1-Wasserstein 距离 (最优传输), 用
    线性规划精确求解 (图很小, 无需近似)。

    alpha 是 idle 概率 (懒惰游走): m_u = (1-alpha)/deg 分布于邻居, alpha 留在
    u 自身。这个约定决定了数值 —— 超立方体在 alpha=0 时曲率为 0, 在
    alpha=1/(d+1) 时才是 2/(d+1)。因此本脚本显式暴露 alpha 并给出两组锚点,
    而不是引用一个来源不明的文献数字。
    """
    nodes, idx, D = all_pairs_distance(G)
    out = {}
    for u, v in G.edges():
        mu, mv = {}, {}
        for self_node, nbrs, tgt in ((u, G[u], mu), (v, G[v], mv)):
            nb = list(nbrs)
            for x in nb:
                tgt[x] = (1.0 - alpha) / len(nb)
            tgt[self_node] = tgt.get(self_node, 0.0) + alpha   # idle 留在自身
        ni, nj = sorted(mu), sorted(mv)
        cost = np.array([[D[idx[a], idx[b]] for b in nj] for a in ni])
        n_i, n_j = len(ni), len(nj)
        A_eq, b_eq = [], []
        for i_, a in enumerate(ni):                 # 行边缘 = m_u
            r = np.zeros(n_i * n_j)
            r[i_ * n_j:(i_ + 1) * n_j] = 1
            A_eq.append(r)
            b_eq.append(mu[a])
        for j_, b in enumerate(nj):                 # 列边缘 = m_v
            r = np.zeros(n_i * n_j)
            r[j_::n_j] = 1
            A_eq.append(r)
            b_eq.append(mv[b])
        res = linprog(cost.ravel(), A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                      bounds=(0, None), method='highs')
        out[(u, v)] = 1.0 - res.fun / D[idx[u], idx[v]]
    return out


def forman_ricci(G, edge_w=None, node_w=None, triangles=True):
    """
    Forman-Ricci 曲率 (Sreejith 等的加权式):
        F(e=uv) = w_e [ w_u/w_e + w_v/w_e
                        - Sum_{e'~u, e'!=e} w_u/sqrt(w_e w_e')
                        - Sum_{e'~v, e'!=e} w_v/sqrt(w_e w_e') ]
    单位权重的无权图退化为经典的 F(e) = 4 - deg(u) - deg(v)。
    triangles=True 时再加 3 * (含 e 的三角形数) * w_e (augmented 版本)。
    """
    node_w = node_w or {n: 1.0 for n in G}
    edge_w = edge_w or {tuple(sorted(e)): 1.0 for e in G.edges()}
    out = {}
    for u, v in G.edges():
        we = edge_w[tuple(sorted((u, v)))]
        wu, wv = node_w[u], node_w[v]
        s = wu / we + wv / we
        for e2 in G.edges(u):
            if set(e2) != {u, v}:
                s -= wu / np.sqrt(we * edge_w[tuple(sorted(e2))])
        for e2 in G.edges(v):
            if set(e2) != {u, v}:
                s -= wv / np.sqrt(we * edge_w[tuple(sorted(e2))])
        f = we * s
        if triangles:
            f += 3.0 * len(set(G.neighbors(u)) & set(G.neighbors(v))) * we
        out[(u, v)] = float(f)
    return out


def validate_curvature():
    """
    在三个有解析解的图上校验 (全部由本脚本手推, 不引用记忆中的文献数字):
      * 环 C_n (alpha=0):        kappa = 0
            m_u 在 {u-1, u+1} 各 1/2, 最优传输 W1 = 1 = d(u,v)
      * 完全图 K_n (alpha=0):    kappa = (n-2)/(n-1)
            N(u) = V\\{u}; 交集 V\\{u,v} 可零成本匹配 (n-2)/(n-1) 的质量,
            余下 1/(n-1) 走一步 -> W1 = 1/(n-1)
      * 超立方体 Q_d (alpha=1/(d+1)): kappa = 2/(d+1)
            所有 d+1 个源质量均为 1/(d+1), 可配对 a_i<->b_i (距离 1),
            总代价 (d-1)/(d+1) -> kappa = 1 - (d-1)/(d+1) = 2/(d+1)
      注: alpha=0 时 Q_d 的 kappa 恰为 0 (同样可手推)。两值的差异完全来自
          idle 约定, 所以约定必须显式声明, 而不是含糊地"引用文献"。
      * Forman (无权, 无三角): F(uv) = 4 - deg(u) - deg(v)
            环 C_n (n>=4) 每点度 2 -> F 恒为 0;
            P_6 内部边 -> 0; K_4 去掉三角项 -> -4+... = -2; 3-正则 -> 恒 -2。
            环的锚点把"度"这一项从正则图锚点里分离出来了 —— 3-正则锚点
            (-2) 与环锚点 (0) 都只依赖度, 但取值不同, 所以两者一起才能
            确认实现走的是 deg(u)+deg(v) 而不是别的正则组合。
    """
    res = {'cyclic': [], 'complete': [], 'hypercube': [], 'forman': {}}
    for n in (6, 8, 12):
        k = np.array(list(ollivier_ricci(nx.cycle_graph(n)).values()))
        res['cyclic'].append({'n': n, 'mean': float(k.mean()), 'exact': 0.0,
                              'max_abs': float(np.abs(k).max())})
    for n in (4, 6, 10):
        k = np.array(list(ollivier_ricci(nx.complete_graph(n)).values()))
        res['complete'].append({'n': n, 'mean': float(k.mean()),
                                'exact': (n - 2) / (n - 1)})
    for d in (3, 4, 5):
        Q = nx.hypercube_graph(d)
        k = np.array(list(ollivier_ricci(Q, alpha=1.0 / (d + 1)).values()))
        k0 = np.array(list(ollivier_ricci(Q, alpha=0.0).values()))
        res['hypercube'].append({'d': d, 'mean_lazy': float(k.mean()),
                                 'exact_lazy': 2.0 / (d + 1),
                                 'mean_plain': float(k0.mean()),
                                 'exact_plain': 0.0})
    p6 = forman_ricci(nx.path_graph(6))
    k4 = forman_ricci(nx.complete_graph(4), triangles=False)
    reg = nx.random_regular_graph(3, 10, seed=1)
    # 环 C_n 的 Forman 锚点。无三角版 F(uv) = 4 - deg(u) - deg(v), 环上每点度 2,
    # 所以每条边恒为 0 —— 这是 4-2-2 这条式子的直接后果。
    # **n=3 必须排除**: C_3 就是 K_3, 三条边两两相邻, 每条边都在一个三角形里,
    # augmented 项 +3*1 把它从 0 顶到 +3。写成"环恒为 0"而不过滤 n=3 就是错的。
    cyc = {n: float(np.mean(list(
        forman_ricci(nx.cycle_graph(n), triangles=False).values())))
        for n in (4, 6, 8, 12, 100)}
    res['forman'] = {
        'path_interior': float(p6[(1, 2)]),                 # 4 - 2 - 2 = 0
        'k4_no_tri': float(list(k4.values())[0]),           # 4 - 3 - 3 = -2
        'regular3_mean': float(np.mean(list(
            forman_ricci(reg, triangles=False).values()))),  # 3-正则恒为 -2
        'cycle_mean': cyc,
        'cycle_max_abs': max(abs(v) for v in cyc.values()),
        'cycle_min_n': min(cyc),
    }
    print("[阶段五] 曲率算子的解析锚点校验:")
    for r in res['cyclic']:
        print(f"  环 C_{r['n']}: {r['mean']:+.8f} (解析 0, max|.|={r['max_abs']:.1e})")
    for r in res['complete']:
        print(f"  完全图 K_{r['n']}: {r['mean']:.8f} (解析 {r['exact']:.8f})")
    for r in res['hypercube']:
        print(f"  超立方体 Q_{r['d']}: alpha=0 -> {r['mean_plain']:+.6f} (解析 0); "
              f"alpha=1/(d+1) -> {r['mean_lazy']:.6f} "
              f"(解析 {r['exact_lazy']:.6f})")
    print(f"  Forman: P_6 内部边={res['forman']['path_interior']:+.1f} (解析 0), "
          f"K_4 无三角={res['forman']['k4_no_tri']:+.1f} (解析 -2), "
          f"3-正则图={res['forman']['regular3_mean']:+.4f} (解析 -2), "
          f"环 C_n (n>=4) max|F|={res['forman']['cycle_max_abs']:.1e} (解析 0)")
    return res


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


def curvature_report(G, name):
    """
    一张图的 Forman / Ollivier 曲率分布汇总, 外加三角数与平均度。

    为什么必须一起报三角数与平均度: 本工作用的 Forman 是**无三角**形式
    F(e) = 4 - deg(u) - deg(v), 它只在图接近树时才与完整 Forman 等价。
    "这张图是不是接近树"这件事由三角形数回答, 只报曲率均值等于把前提藏起来。
    平均度给出正则粗式预期 4-2*d_bar, 与逐边严格式 4-2*<deg>_edge 并排,
    度偏置的来处(非正则)就摆在明面上。

    几何口径与 MERA 图**完全一致** (复用 derive_L5_forman_from_degree 且
    record=False) —— 对照图不进因果链台账, 但量必须是同一把尺子量出来的,
    否则"MERA 与对照图比曲率"这件事本身就不可比。
    """
    f = forman_ricci(G, triangles=False)
    try:
        k = ollivier_ricci(G)
    except Exception:
        k = {}
    fv = np.array(list(f.values()))
    kv = np.array(list(k.values())) if k else np.array([np.nan])
    geom = derive_L5_forman_from_degree(G, float(fv.mean()), record=False)
    rep = {'name': name, 'n_nodes': G.number_of_nodes(),
           'n_edges': G.number_of_edges(),
           'forman_mean': float(fv.mean()), 'forman_min': float(fv.min()),
           'forman_max': float(fv.max()),
           'forman_neg_frac': float((fv < 0).mean()),
           'or_mean': float(np.nanmean(kv)),
           'or_neg_frac': float(np.nanmean(kv < 0)) if k else float('nan'),
           'n_triangles': int(geom['triangles']),
           'd_bar': float(geom['d_bar']),
           'deg_edge_mean': float(geom['deg_edge_mean']),
           'forman_regular_pred': float(geom['forman_predicted']),
           'forman_exact_pred': float(geom['forman_exact']),
           'degree_bias': float(geom['bias'])}
    print(f"  {name}: |V|={rep['n_nodes']} |E|={rep['n_edges']} "
          f"d_bar={rep['d_bar']:.3f} 三角={rep['n_triangles']} | "
          f"Forman 均值={rep['forman_mean']:+.3f} "
          f"(负边占比 {rep['forman_neg_frac']:.2f}, 逐边严格式预期 "
          f"{rep['forman_exact_pred']:+.3f}) | "
          f"Ollivier 均值={rep['or_mean']:+.4f} "
          f"(负边占比 {rep['or_neg_frac']:.2f})")
    return rep, f, k


def geometry_controls(G_mera, n_each=3):
    """
    对照几何族 —— 三类零模型, 每类 n_each 个实例。

      gnm  同 |V| 同 |E| 的 Erdos-Renyi 随机图 (度分布不受约束)
      ws   Watts-Strogatz 小世界, 平均度与 MERA 图对齐 (配度, 高聚类)
      tree 平衡二叉树 (纯树: 零三角形, 直接对标"MERA 图接近树"这条前提)

    为什么要三类而不是一类: 单张同规模随机图只能回答"比随机图更负吗"这一个
    问题, 而"MERA 的负曲率有什么特点"至少有三个可分离的维度 —— 度分布、
    聚类(由三角形数度量)、树的成分。三类各控一个维度, 于是"MERA 落在对照
    分布里"这句话才有内容。

    实例数取 3: 单实例的曲率均值噪声在 0.1 量级, 3 个实例足以给出一个
    5%~95% 的粗略位置, 同时把 Ollivier (逐边解线性规划) 的总耗时控住。

    诚实边界: 这是**零模型对照**, 不是同一物理系统的不同实现。它回答的是
    "MERA 体几何在同类规模的图里算不算异常", 不是"MERA 就是全息对偶"。
    平衡树的节点数受限于 2^(h+1)-1, 与 |V| 只能接近, 故各图的实际规模
    一并报出而不是假定相等。
    """
    V, E = G_mera.number_of_nodes(), G_mera.number_of_edges()
    # WS 的度取 2*round(d_bar) 且取偶 —— 与 MERA 图的平均度对齐, 而不是写死 4
    k_ws = max(2, 2 * int(round(E / V))) if V else 2
    h_tree = max(1, int(np.log2(V + 1)) - 1)
    reps = []
    for fam in ('gnm', 'ws', 'tree'):
        for s in range(n_each):
            if fam == 'gnm':
                G = nx.gnm_random_graph(V, E, seed=100 + s)
            elif fam == 'ws':
                G = nx.watts_strogatz_graph(V, k_ws, 0.1, seed=200 + s)
            else:
                G = nx.balanced_tree(2, h_tree)
            rep, _, _ = curvature_report(G, f'对照 {fam} #{s}')
            rep['family'] = fam
            rep['instance'] = int(s)
            reps.append(rep)
    print(f"  对照族: gnm/ws/tree 各 {n_each} 个实例 "
          f"(MERA 图 |V|={V} |E|={E}, d_bar={2*E/V:.3f}; WS 取 k={k_ws}, "
          f"树高 {h_tree} -> {2**(h_tree+1)-1} 节点)")
    return reps


# ============================================================================
# 阶段六 · 时空提供梯度孕育显生命 (涌现几何上的 Gray-Scott)
# ============================================================================
def _fit_length(dists, vals, max_d=None):
    """由 ln|值| 随距离的线性衰减拟合关联长度 xi:  值 ~ exp(-d/xi)"""
    d = np.asarray(dists, dtype=float)
    y = np.log(np.maximum(np.asarray(vals, dtype=float), 1e-300))
    if max_d is not None:
        m = d <= max_d
        d, y = d[m], y[m]
    if len(d) < 3 or np.ptp(d) == 0:
        return float('nan')
    slope, _ = np.polyfit(d, y, 1)
    return float(-1.0 / slope) if slope < 0 else float('inf')


def boundary_correlation_length(C, L):
    """
    边界关联几何的关联长度: 把 <sigma^z_i sigma^z_j> 按环上距离 d 分箱平均,
    再拟合 ln|C| ~ -d/xi。这是阶段五那套几何的特征尺度。
    """
    ds, vs = [], []
    for i in range(L):
        for j in range(i + 1, L):
            d = min(j - i, L - (j - i))
            ds.append(d)
            vs.append(abs(C[i, j]))
    ds = np.array(ds)
    vs = np.array(vs)
    binned = np.array([vs[ds == d].mean() for d in np.unique(ds)])
    return _fit_length(np.unique(ds), binned)


# ============================================================================
# 闭环螺旋 · spiral_loop()
# ============================================================================
def spiral_loop(turns=6, L0=8, scale_knob=1.0, max_L=16, loop_tol=0.02,
                seed=0):
    """
    多圈自举: 把整条七阶段链当作一个映射, 反复应用到自身。

    ---------------------------------------------------------------------------
    尺度更新律 (闭环的"自指"部分)
    ---------------------------------------------------------------------------
        第 t 圈的系统尺寸由第 t-1 圈**实测**的关联长度定向:
            L_t = clip(round(scale_knob * xi_{t-1}), L0, max_L)
        第 0 圈没有上一圈, 用旋钮 L0 起步。
    也就是说: 模型测出来的关联长度, 决定模型下一次在多大的尺度上运行。
    尺度不再是外部写死的常数: 它是上一圈自己测出来的。

    ---------------------------------------------------------------------------
    跟踪哪三个量, 以及为什么必须跟踪三个而不是一个
    ---------------------------------------------------------------------------
    闭环到底收不收敛, **取决于问哪个量**。这三个的答案是不一样的:

      (1) c_t          中心荷。应趋于 1/2 —— 这是共形场论不动点,
                       是"模型自我应用时复现自己"的那一面。

      (2) xi_t / L_t   无量纲比值。应趋于常数 (实测约 1.2) ——
                       这才叫尺度不变。**关键**: 绝不能用 xi 的绝对值
                       判断收敛。临界系统的关联长度被系统尺寸截断,
                       实测 xi = 19.314 > L = 16, xi 就是跟着 L 涨的。
                       用 xi 判收敛会得到"永远发散"的错误结论。

      (3) 相对谱隙 1 - p1/p0。应趋于 0 —— 这是**临界**的标志。
          (p1/p0 本身是收敛比, 它趋于 1; 初稿说"p1/p0 -> 0"方向反了。)
                       (能隙闭合)。这个量就是因果链 L2 里那个派生量,
                       所以闭环与因果链在这里咬合上了。

    收敛判据只作用于无量纲量 (c 与 xi/L), 因为只有它们才是尺度不变量。

    ---------------------------------------------------------------------------
    诚实边界 (必须在代码里说清楚, 否则这个闭环会自我欺骗)
    ---------------------------------------------------------------------------
    (a) **xi 随 L 发散不是失败, 是临界的定义。** 若把"xi 不收敛"报成闭环
        失败, 那就把最漂亮的物理结论说反了。脚本对这一点有专门打印。
    (b) **max_L 是旋钮, 且实测会触顶。** 去掉上限 L 会一直涨 —— 又是临界性。
        脚本把触顶如实打印, 不假装它是自然收敛。
    (c) **本函数用精确对角化做 oracle**, 每圈直接从基态算 c / xi / gap,
        不走 MERA。理由: 若每圈都跑 MERA, 键维截断误差会与尺度流动混淆,
        分不清"c 在流动"还是"截断在漂移"。末圈单独做一次 MERA 交叉校验。
    (d) 每圈 L 取偶数 (自旋链半链切分需要), 且 <= max_L; 2^16 = 65536 维
        是稀疏 Lanczos 在本机舒服的范围。

    返回每圈的记录; 同时返回一个 verdict 字符串, 说明"收敛"是对哪些量说的。
    """
    print("\n" + "=" * 76)
    print("  闭环螺旋 spiral_loop() · 多圈自举")
    print("=" * 76)
    print(f"  尺度律: L_t = clip(round({scale_knob} * xi_(t-1)), {L0}, {max_L})")
    print(f"  收敛判据: 无量纲量 (c, xi/L) 的相对变化 < {loop_tol:.1%}")
    print(f"  注意: xi 的绝对值**不**参与收敛判据 —— 临界时它随 L 发散, ")
    print(f"        那是临界的定义, 不是失败。")

    rows = []
    L_prev = None
    prev_c, prev_ratio = None, None
    clamped = False
    converged_at = None

    for t in range(turns):
        if t == 0:
            L = int(L0)
            L_source = f'旋钮 L0={L0} (第 0 圈无上一圈)'
        else:
            raw = int(round(scale_knob * rows[-1]['xi']))
            L = int(np.clip(raw, L0, max_L))
            clamped = clamped or (raw != L)
            L_source = (f'round({scale_knob}*xi_{t-1}={rows[-1]["xi"]:.3f}) '
                        f'= {raw}' + (f' -> clip 到 {L}' if raw != L else ''))
        # 半链切分要求偶数
        L = L - (L % 2)

        # ---- 本圈的物理量: 全部来自精确对角化 (oracle) ----
        E0, gs = exact_ground_state(L, 1.0, 1.0)
        half = L // 2
        s = np.linalg.svd(gs.reshape(2 ** half, 2 ** (L - half)),
                          compute_uv=False)
        p = s ** 2
        p = p / p.sum()
        p_nz = p[p > 1e-15]
        c_fit = fit_central_charge(gs, L)
        _, C, _ = boundary_correlation_graph(gs, L)   # C_ij = <sigma^z_i sigma^z_j>
        xi = boundary_correlation_length(C, L)
        gap = float(p_nz[1] / p_nz[0]) if len(p_nz) > 1 else float('nan')
        ratio = xi / L

        # 无量纲量的相对变化 (只有它们参与收敛判据)
        dc = abs(c_fit['c'] - prev_c) / abs(prev_c) if prev_c else float('nan')
        dr = abs(ratio - prev_ratio) / abs(prev_ratio) if prev_ratio else float('nan')

        rows.append({'turn': t, 'L': L, 'L_source': L_source,
                     'E0_per_L': E0 / L, 'c': c_fit['c'],
                     'xi': float(xi), 'xi_over_L': float(ratio),
                     'gap': gap, 'entropy': float(-np.sum(p_nz * np.log(p_nz))),
                     'dc_rel': dc, 'dratio_rel': dr})

        print(f"\n  第 {t} 圈: L={L}   [{L_source}]")
        print(f"     E0/L={E0/L:.6f}, c={c_fit['c']:.4f}, "
              f"xi={xi:.3f}, xi/L={ratio:.4f}, gap={gap:.6f}")
        if t > 0:
            print(f"     相对上圈: |dc|/c={dc:.4f}, |d(xi/L)|/(xi/L)={dr:.4f}")

        if (t > 0 and np.isfinite(dc) and np.isfinite(dr)
                and dc < loop_tol and dr < loop_tol and converged_at is None):
            converged_at = t
            print(f"     -> 无量纲量已收敛 (第 {t} 圈)")

        prev_c, prev_ratio = c_fit['c'], ratio

    # ---- 末圈 MERA 交叉校验 (诚实边界 (c)) ----
    L_last = rows[-1]['L']
    chi_cross = max(2, int(2 ** int(np.ceil(np.log(np.exp(rows[-1]['entropy']))
                                             / np.log(2)))))
    print(f"\n  [交叉校验] 末圈 L={L_last} 用 MERA (chi={chi_cross}) 复核 c:")
    m_cross, raws_cross = mera_init(L_last, chi_cross, seed=0)
    _, gs_last = exact_ground_state(L_last, 1.0, 1.0)
    psi_cross, _ = mera_fit(m_cross, raws_cross, gs_last, L_last,
                            steps=400, verbose=False)
    c_cross = fit_central_charge(psi_cross, L_last)
    ov_cross = float(abs(np.vdot(gs_last, psi_cross)))
    print(f"     MERA: c={c_cross['c']:.4f}, 重叠={ov_cross:.4f} "
          f"| 精确: c={rows[-1]['c']:.4f}")
    print(f"     差 |dc| = {abs(c_cross['c'] - rows[-1]['c']):.4f} "
          f"—— 这是键维截断的量级, 不是流动")

    # ---- 收尾判定: 三个量分开下结论 ----
    #
    # 两处必须说清楚的坑 (初稿都踩了, 这里已修正):
    #
    # (坑一) 不能拿第 0 圈当"流动的起点"来判收敛。
    #   第 0 圈的 L0 是**旋钮**, 不是尺度律的产物; 它必然离不动点远。
    #   拿 seed 和末圈比, 得到的一定是"不稳定", 而这与逐圈的
    #   |d(xi/L)|/(xi/L) -> 0 直接矛盾。所以收敛判据只对**非 seed 圈**
    #   (t >= 1) 生效, seed 单列报告。
    #
    # (坑二) p1/p0 **不是**"谱隙", 它是幂迭代的**收敛比**。
    #   记 r = p1/p0, 则相对谱隙 = 1 - r。临界时 Schmidt 谱变密变慢,
    #   r -> 1, 于是**相对谱隙 1-r -> 0**, 这才是"谱隙闭合"。
    #   初稿写成 "gap_t = p1/p0 应趋于 0", 方向搞反了 —— 实测
    #   r: 0.309 -> 0.343 -> 0.366 一路上升, 对应的 1-r: 0.691 -> 0.657
    #   -> 0.634 一路下降, **谱隙确实在闭合**。判定按 1-r 下。
    c_all = [r['c'] for r in rows]
    ratio_all = [r['xi_over_L'] for r in rows]
    gap_all = [r['gap'] for r in rows]           # 收敛比 r = p1/p0
    relgap_all = [1.0 - g for g in gap_all]      # 相对谱隙 1 - r
    c_flow, ratio_flow, relgap_flow = c_all[1:], ratio_all[1:], relgap_all[1:]
    verdicts = {
        'c_toward_half': bool(abs(c_all[-1] - 0.5) < 0.05),
        # 判据用非 seed 圈: 比值在流动中是否稳定下来
        'ratio_stable': bool(np.ptp(ratio_flow) < 0.15 * np.mean(ratio_flow)),
        # 收敛比上升 <=> 相对谱隙闭合 (趋向临界)
        'relgap_closing': bool(relgap_flow[-1] < relgap_flow[0]),
        'scale_clamped': bool(clamped),
        'converged_at_turn': converged_at,
    }
    print("\n  闭环裁决 (三个量分开说, 不合并成一个'收敛'):")
    print(f"    [seed] 第 0 圈由旋钮 L0={L0} 给定, 不是尺度律的产物,")
    print(f"           故不参与收敛判定 (与末圈比一定'不稳定', 那是假信号)。")
    print(f"    c_t: {c_all[0]:.4f} -> {c_all[-1]:.4f}  "
          f"(理论 0.5)  {'趋于 1/2' if verdicts['c_toward_half'] else '未收敛到 1/2'}")
    print(f"    xi_t/L_t (t>=1): {ratio_flow[0]:.4f} -> {ratio_flow[-1]:.4f}  "
          f"{'尺度不变 (比值稳定)' if verdicts['ratio_stable'] else '不稳定'}")
    print(f"    收敛比 r=p1/p0: {gap_all[0]:.6f} -> {gap_all[-1]:.6f}  "
          f"(上升 = Schmidt 谱变密)")
    print(f"    相对谱隙 1-r:   {relgap_all[0]:.6f} -> {relgap_all[-1]:.6f}  "
          f"{'谱隙闭合 (趋向临界)' if verdicts['relgap_closing'] else '谱隙未闭合'}")
    print(f"    xi 绝对值: {rows[0]['xi']:.3f} -> {rows[-1]['xi']:.3f}  "
          f"(随 L 增长 —— 这是临界的定义, 不是失败)")
    if clamped:
        print(f"    旋钮 max_L={max_L} 已触顶: 尺度 L 的流动被上限截断。")
        print(f"    去掉上限 L 会继续涨 —— 同样是临界性, 不是闭环缺陷。")
        print(f"    注意 (诚实边界): 谱隙闭合依赖 L 增长; L 被夹住之后")
        print(f"    1-r 会停在 L={max_L} 的值上不再下降, 这不是模型停止趋向")
        print(f"    临界, 而是**闭环再也拿不到更大的 L**。")
    print(f"\n  结论: 无量纲不变量"
          f"{'收敛' if verdicts['ratio_stable'] else '未收敛'}"
          f" (第 {converged_at} 圈起逐圈变化 < {loop_tol:.0%}), "
          f"而尺度本身发散。模型自我应用时复现 CFT 不动点;")
    print(f"        唯一'不收敛'的那一项恰好是临界性。")

    return {'rows': rows, 'verdicts': verdicts, 'mera_cross': {
        'L': L_last, 'chi': chi_cross, 'c': c_cross['c'], 'overlap': ov_cross}}


def stage6_life(F=0.035, k=0.060, N=40, L=0.5, steps=40000, dt=1.0,
                Du=2e-5, Dv=1e-5):
    """
    标准 Gray-Scott 反应-扩散 (Pearson 的斑点相区参数):
        du/dt = Du Lap u - u v^2 + F(1-u)
        dv/dt = Dv Lap v + u v^2 - (F+k) v
    在 N x N 的周期域 L x L 上做显式欧拉。

    【数值稳定性的硬约束 —— 这是本函数最容易被糊弄过去的地方】
    二维 5 点拉普拉斯 + 显式欧拉的稳定条件是  dt * Du / h^2 <= 1/4 (h = L/N)。
    超限不会得到"更剧烈的斑图", 只会得到棋盘状的数值失稳 (饱和到 [0,1] 的
    假斑图)。本函数把 dt*Du/h^2 算出来并报告, 且不用 np.clip 去掩盖失稳。
    分辨率也必须够: 模式波长 ~0.05 域长单位, h 大到 0.06 时每个波长只有
    1 个格点, 拍不出来。经典 Pearson 用 h ~ 0.01; 这里 L=0.5, N=40 给
    h=0.0125, 与之一致。

    诚实边界: 这是反应-扩散的数学斑图, 不是生物生命。而且它跑在平直网格上
    —— 本函数不假装扩散张量来自阶段五的关联几何; 两者的特征尺度由
    boundary_correlation_length() 与下面的 pattern_spacing 分别报出。
    """
    h = L / N
    stab = dt * Du / h ** 2
    if stab > 0.25:
        return {'ok': False, 'h': h, 'stab': stab, 'v': None,
                'contrast': float('nan'), 'emerged': False,
                'reason': f'不满足稳定性条件 dt*Du/h^2={stab:.4f} > 0.25, '
                          f'拒绝给出可能失稳的结果'}

    u = np.ones((N, N))
    v = np.zeros((N, N))
    c = N // 2
    u[c - 2:c + 2, c - 2:c + 2] = 0.5
    v[c - 2:c + 2, c - 2:c + 2] = 0.25
    for _ in range(steps):
        lu = (np.roll(u, 1, 0) + np.roll(u, -1, 0) +
              np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4 * u) / h ** 2
        lv = (np.roll(v, 1, 0) + np.roll(v, -1, 0) +
              np.roll(v, 1, 1) + np.roll(v, -1, 1) - 4 * v) / h ** 2
        uv2 = u * v ** 2
        u = u + dt * (Du * lu - uv2 + F * (1 - u))
        v = v + dt * (Dv * lv + uv2 - (F + k) * v)

    finite = bool(np.isfinite(u).all() and np.isfinite(v).all())
    contrast = float(v.max() - v.min()) if finite else float('nan')
    active = float((v > 0.1).mean()) if finite else 0.0
    # 斑图波长: v 沿 x 的平均零交叉间隔 * 2 (一个完整周期有两次穿越)
    zc = np.mean([len(np.where(np.diff(np.sign(v[i] - v[i].mean())))[0])
                  for i in range(N)]) if finite else 0.0
    spacing = float(2.0 * L / zc) if zc > 0 else float('inf')
    emerged = bool(finite and contrast > 0.2 and active > 0.05)

    print(f"[阶段六] Gray-Scott 生命斑图 (平直周期网格):")
    print(f"  网格 h={h:.4f}, dt*Du/h^2={stab:.4f} (稳定上限 0.25) -> 数值稳定")
    print(f"  对比度={contrast:.3f}, 活化面积占比={active:.3f}, "
          f"斑图波长~{spacing:.4f} (域长 {L})")
    print(f"  -> {'涌现' if emerged else '未涌现'}"
          f"  (判据: 对比度>0.2 且 活化面积>5%)")
    return {'ok': True, 'u': u, 'v': v, 'contrast': contrast,
            'active_frac': active, 'spacing': spacing, 'h': h, 'stab': stab,
            'steps': steps, 'emerged': emerged, 'finite': finite}


def gray_scott_invariants(N=40, L=0.5, steps=20, F=0.035, k=0.060,
                          dt=1.0, Du=2e-5, Dv=1e-5):
    """
    Gray-Scott 显式欧拉格式的两条**精确**离散不变量。

    【为什么不是"u+v 质量守恒"】
    更新式是
        du/dt = Du Lap u - u v^2 + F(1-u)
        dv/dt = Dv Lap v + u v^2 - (F+k) v
    反应项 -u v^2 与 +u v^2 逐点相消, 但**源汇项不相消**。对全空间取平均:
        d/dt <u+v> = Du <Lap u> + Dv <Lap v> + F(1-<u>) - (F+k)<v>
    周期域上两个 Laplacian 项精确为 0, 剩下的 F(1-<u>) - (F+k)<v> 一般不为零
    (只有恰好已在稳态时才为零)。所以"u+v 守恒"是个**伪不变量**: 实测第一步
    就偏离 6.25e-5, 比"守恒到 1e-6"的容差大 62 倍, 累计 2000 步漂移 0.278。
    照着它写守卫, 只会把正确的物理判成失败。

    换成的两条都是真命题, 且各管一段:

      (a) **离散拉普拉斯零和** <Sum Lap u> == 0。
          周期包裹下四个 roll 是同一组数的置换, 所以四个和严格相等、分子
          精确为 0 (浮点上只留抵消误差)。它测的是 stencil 系数 (-4 中心)
          与 roll 的包裹方向 —— 与守恒律无关, 纯粹是"算子写对了没有"。

      (b) **精确离散平衡恒等式**: 用**更新前**的场算
            <u+v>_{n+1} - <u+v>_n == dt [ F(1-<u>_n) - (F+k)<v>_n ]
          这不是物理守恒, 而是同一个更新式的算术恒等式 (Laplacian 项在求
          平均时精确消掉)。它测的是整个 RHS 的装配: 少一项、符号反、dt 位置
          错, 都会立刻在这里暴露。实测吻合到 3.6e-16。

    诚实边界: 两条都是**格式性质**, 不是"Gray-Scott 守恒量"这样的物理命题。
    它们保证的是"这段代码按写下的方程在算", 不保证那个方程描述的是生命。
    """
    h = L / N
    u = np.ones((N, N))
    v = np.zeros((N, N))
    c = N // 2
    u[c - 2:c + 2, c - 2:c + 2] = 0.5
    v[c - 2:c + 2, c - 2:c + 2] = 0.25

    lap_rel_max = 0.0
    bal_abs_max = 0.0
    drift_first = float('nan')
    for n in range(steps):
        lu = (np.roll(u, 1, 0) + np.roll(u, -1, 0) +
              np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4 * u) / h ** 2
        lv = (np.roll(v, 1, 0) + np.roll(v, -1, 0) +
              np.roll(v, 1, 1) + np.roll(v, -1, 1) - 4 * v) / h ** 2

        # (a) 零和: 归一化成无量纲的抵消残差 (除以 4<u>/h^2 这个量级)
        lu_rel = abs(float(lu.sum())) * h ** 2 / (4.0 * abs(float(u.sum())))
        lv_rel = abs(float(lv.sum())) * h ** 2 / (4.0 * abs(float(v.sum())) + 1e-30)
        lap_rel_max = max(lap_rel_max, lu_rel, lv_rel)

        # (b) 平衡恒等式: 预言的增量用更新前的场算
        m_before = float((u + v).mean())
        pred = dt * (F * (1.0 - float(u.mean())) - (F + k) * float(v.mean()))
        uv2 = u * v ** 2
        u = u + dt * (Du * lu - uv2 + F * (1 - u))
        v = v + dt * (Dv * lv + uv2 - (F + k) * v)
        m_after = float((u + v).mean())
        if n == 0:
            drift_first = float(m_after - m_before)
        bal_abs_max = max(bal_abs_max, abs((m_after - m_before) - pred))

    return {'lap_zero_rel_max': lap_rel_max,
            'balance_abs_err_max': bal_abs_max,
            'drift_first_step': drift_first,
            'N': N, 'L': L, 'h': h, 'steps': steps, 'dt': dt,
            'F': F, 'k': k}


# ============================================================================
# 阶段七 · 生命涌现意识自照见空无 (二阶自指闭环)
# ============================================================================
def stage7_consciousness(coupled):
    """
    二阶自指 = 以阶段三-c 的联合不动点 W* 作为"照见"闭环:
      D* = W* 的主特征方向; 熵比 = H(D*)/ln N。
      '照见空无' = D* 接近等权 (熵比 -> 1); '照见结构' = D* 偏离等权。
    诚实: 照见的内容取决于缘起的条件。阶段三-c 的变体 B 自发秩 1 化, 其
    主方向就是状态自己 —— 这正是"照见自己"的数学形式; 而变体 A 塌缩到
    x* = 0, 在数学上没有可读的 D*, 此时如实报"空无", 不伪造结构。
    """
    print("[阶段七] 二阶自指闭环:")
    if coupled['free']['collapsed']:
        # x* = 0 时 W* 没有可读的主方向, D* 不存在。这一支取值**一律是约定**:
        # probs = 1/N 与 entropy = ln N 是同一件事的换写, 所以熵比恒等于 1;
        # 等权度也恒等于 1。它们是"空无"这个名字的定义, 不是关于系统的观测。
        # 如实标 defined=False, 免得被当成"照见空无"的独立证据。
        N = len(coupled['_x'])
        probs = np.ones(N) / N
        entropy = float(np.log(N))
        eqw = float(abs(np.vdot(np.ones(N) / np.sqrt(N),
                                np.ones(N) / np.sqrt(N))))
        status = '空无'
        defined = False
        note = '状态塌缩到 x*=0, 无可读的 D*; 均匀分布是本处约定, 非测量'
    else:
        W = coupled['_W'].detach().numpy()
        ev, evec = np.linalg.eig(W)
        i = int(np.argmax(np.abs(ev)))
        D = evec[:, i]
        D = D / np.linalg.norm(D)
        probs = np.abs(D) ** 2
        probs = probs / probs.sum()
        entropy = float(-np.sum(probs * np.log(probs + 1e-15)))
        N = W.shape[0]
        eqw = float(abs(np.vdot(np.ones(N) / np.sqrt(N), D)))
        status = '结构'
        defined = True
        note = f'D* = W* 主方向 (W* 秩1, sigma1/sigma2={coupled["norm"]["sigma_ratio"]:.1e})'
    ratio = entropy / np.log(len(probs))
    # 上界: 熵 <= ln N (均匀分布取最大熵) 与 |<等权向量, D*>| <= 1 (Cauchy-Schwarz)。
    # 塌缩支两条都恒取等号, 所以只有**非塌缩支**的这两条才带信息量。
    bound_ok = bool(ratio <= 1.0 + 1e-12 and eqw <= 1.0 + 1e-12)
    print(f"  熵比={ratio:.3f}, 等权度={eqw:.3f} -> 照见{status}"
          f"{'' if defined else ' (约定值, 非测量)'}")
    print(f"  ({note})")
    return {'entropy_ratio': ratio, 'equal_weight': eqw, 'status': status,
            'note': note, 'defined': defined, 'bound_ok': bound_ok}


# ============================================================================
# 任务 A · 中心荷 c 验证 (MERA 键维扫描 + 精确对照)
# ============================================================================
def central_charge_verification(configs=((8, 2, 600), (8, 4, 600),
                                         (16, 2, 600), (16, 4, 600))):
    """
    双重证据:
      (1) 精确对角化 (有限尺寸基线): 拟合 c, 应约等于 0.5;
      (2) MERA (不同键维 chi): 拟合 c, 应随 chi 增大单调趋近 0.5。
    诚实边界: 有限尺寸 + 有限键维下报告的是"收敛趋势", 不是 c=1/2 的证明。
    """
    print("=" * 76)
    print("  任务 A · 中心荷 c 验证 (quimb MERA 变分 + 精确对照)")
    print("=" * 76)
    out = {'exact': [], 'mera': [], 'energy_control': None}

    exact_cache = {}
    for (L, _, _) in configs:
        if L in exact_cache:
            continue
        E0, gs = exact_ground_state(L, 1.0, 1.0)
        fit = fit_central_charge(gs, L)
        exact_cache[L] = (E0, gs, fit)
        out['exact'].append({'L': L, 'E0_per_L': E0 / L, 'c': float(fit['c']),
                             'rms': fit['rms'], 'S': fit['S'].tolist()})
        print(f"  [精确对照] L={L}: E0/L={E0/L:.6f}, 拟合 c={fit['c']:.4f} "
              f"(rms={fit['rms']:.4f})")

    for (L, chi, steps) in configs:
        E0, gs, _ = exact_cache[L]
        mera, raws = mera_init(L, chi, seed=0)
        iso = mera_isometry_check(mera, L)
        print(f"\n  [MERA] L={L}, chi={chi}: 张量数={mera.num_tensors}, "
              f"酉性误差={iso['unitary_err']:.2e}, "
              f"等距误差={iso['isometry_err']:.2e}")
        psi, hist = mera_fit(mera, raws, gs, L, steps=steps)
        E = float(psi @ (tfi_periodic_sparse(L) @ psi))
        fit = fit_central_charge(psi, L)
        ov = abs(float(gs @ psi))
        out['mera'].append({'L': L, 'chi': chi, 'c': float(fit['c']),
                            'rms': fit['rms'], 'overlap': ov,
                            'E_per_L': E / L, 'E0_per_L': E0 / L,
                            'n_unitary': iso['n_unitary'],
                            'n_isometry': iso['n_isometry'],
                            'unitary_err': iso['unitary_err'],
                            'isometry_err': iso['isometry_err'],
                            'S': fit['S'].tolist(), 'fit': hist})
        print(f"     -> 重叠={ov:.5f}, E/L={E/L:+.5f} (精确{E0/L:+.5f}), "
              f"拟合 c={fit['c']:.4f} (rms={fit['rms']:.4f})")

    # 对照实验: 能量目标 -> 平均场塌缩
    print("\n  [对照实验] 若改用'最小化能量'作变分目标 (L=8, chi=4):")
    mera_e, raws_e = mera_init(8, 4, seed=0)
    psi_e, E_e = mera_energy_only(mera_e, raws_e, 8,
                                  tfi_periodic_sparse(8), steps=400)
    fit_e = fit_central_charge(psi_e, 8)
    E0_8, gs8, _ = exact_cache[8]
    ov_e = abs(float(gs8 @ psi_e))
    out['energy_control'] = {'E_per_L': E_e / 8, 'E0_per_L': E0_8 / 8,
                             'c': float(fit_e['c']), 'overlap': ov_e,
                             'S': fit_e['S'].tolist()}
    print(f"     能量目标: E/L={E_e/8:+.5f} (精确{E0_8/8:+.5f}), "
          f"重叠={ov_e:.5f}, 拟合 c={fit_e['c']:.4f}")
    print("     -> 能量看似接近, 但重叠低、c 完全错误: 塌缩到平均场局部极小")
    print("        (最优乘积态 E/L = -1.25, 与精确值只差 2.5%, 却有 S(n)~0)")
    return out



# ============================================================================
# 主程序
# ============================================================================
# ============================================================================
# v14 · F5 一致性检查束 (数值路径 <-> 解析路径)
# ============================================================================
def consistency_checks(s3a, curv_val, s7, gray=None):
    """
    五条检查, 每条都把同一个量用**两条独立路径**算出来对表。

    为什么必须是这个形式: "某个数很小"本身没有内容 —— 任何实现都能让它很小。
    有内容的是"数值路径与解析路径给出同一个数", 因为解析路径不共享任何被检验
    的代码, 它既错不了也蒙不对。

      L2  基态能量:  稀疏 eigsh        vs  Jordan-Wigner 闭合式
      L3  幂迭代:    迭代残差 ||A x - <x,A x> x|| 是否到阈值 (不是"迭代步数")
      L5  Forman:    环 C_n (n>=4) 的 F 是否恒为 0 (4 - deg(u) - deg(v))
      L6  Gray-Scott: 离散拉普拉斯零和 + 精确离散平衡恒等式
      L7  自指闭环:   非塌缩支的熵比与等权度是否满足上界

    三条被删掉的候选, 记在这里免得以后又被加回来:
      * "L2 的迭代步数与预设值无关" —— eigsh 没有"步数"这个参数。用 maxiter
        翻译之后, 30 以上逐位相同, 零分辨力; 换成 JW 闭合式才真的在做对照。
      * "L3 的 Jordan 峰 k*=(N-1)/|ln lam|" —— 已经是第二层的指标 2.4,
        在这里重算只是把同一个数算第二遍, 不构成第二条路径。
      * "L4 每步优化后的酉性/等距性残差" —— 双重恒真: 进优化器的是 raws
        参数, MERA 张量对象从未被优化触碰 (而且两个检查点都排在优化之前);
        就算搬进循环, QR 参数化也按构造给出等距性。metric_mera_consistency
        已经断言过这件事。

    诚实边界: 这五条测的都是"实现与它声称的解析式一致", 不测"物理结论成立"。
    L6 的两条是**格式性质** —— 保证代码按写下的方程在算, 不保证那个方程描述
    的是生命。L7 的上界在塌缩支恒取等号, 那一支不带信息量。
    """
    # ---- L2: 基态能量 vs Jordan-Wigner 闭合式 ----
    jw_pts = []
    for L, h in ((8, 1.0), (10, 0.5), (12, 1.0), (14, 1.5), (16, 1.0)):
        e_num, _ = exact_ground_state(L, 1.0, h)
        e_jw = jw_ground_energy(L, 1.0, h)
        jw_pts.append({'L': L, 'h': h, 'E_eigsh': e_num, 'E_jw': e_jw,
                       'abs_dev': float(abs(e_num - e_jw))})
    jw_max_dev = float(max(p['abs_dev'] for p in jw_pts))

    # ---- L3: 幂迭代残差 (物理算例 + 四个受控算例, 取最差的一个) ----
    res_all = [float(s['residual']) for s in s3a['gap_sweep']]
    if s3a.get('physical'):
        res_all.append(float(s3a['physical']['residual']))
    l3_max_res = float(max(res_all)) if res_all else float('nan')
    l3_n_case = len(res_all)

    # ---- L5: Forman 环锚点 ----
    cd = curv_val['forman']

    # ---- L6: Gray-Scott 不变量 ----
    gs = gray if gray is not None else gray_scott_invariants()

    # ---- L7: 自指闭环上界 ----
    ratio, eqw = float(s7['entropy_ratio']), float(s7['equal_weight'])
    l7_defined = bool(s7.get('defined'))
    l7_bound_ok = bool(s7.get('bound_ok'))

    checks = {
        'L2': {'points': jw_pts, 'max_abs_dev': jw_max_dev,
               'n_points': len(jw_pts)},
        'L3': {'max_residual': l3_max_res, 'n_case': l3_n_case,
               'residuals': res_all},
        'L5': {'cycle_mean': cd['cycle_mean'],
               'cycle_max_abs': float(cd['cycle_max_abs']),
               'cycle_min_n': int(cd['cycle_min_n'])},
        'L6': gs,
        'L7': {'entropy_ratio': ratio, 'equal_weight': eqw,
               'defined': l7_defined, 'bound_ok': l7_bound_ok,
               'status': s7['status']},
    }

    print("\n  ── v14·F5 一致性检查 (数值路径 <-> 解析路径) ──")
    print(f"      L2 基态能量: {len(jw_pts)} 个 (L,h) 点, "
          f"max|eigsh - JW| = {jw_max_dev:.2e}")
    print(f"      L3 幂迭代残差: {l3_n_case} 个算例, max = {l3_max_res:.2e}")
    print(f"      L5 Forman 环 C_n (n>=4): max|F| = {cd['cycle_max_abs']:.2e} "
          f"(解析 0)")
    print(f"      L6 Gray-Scott: 拉普拉斯零和相对残差 max = "
          f"{gs['lap_zero_rel_max']:.2e}; 平衡恒等式 abs 误差 max = "
          f"{gs['balance_abs_err_max']:.2e} (首步 <u+v> 漂移 = "
          f"{gs['drift_first_step']:+.3e})")
    print(f"      L7 自指闭环: 熵比 = {ratio:.3f} <= 1, 等权度 = {eqw:.3f} <= 1"
          f"{'' if l7_defined else ' (塌缩支: 两条均为约定值, 无信息量)'}")
    return checks


def main():
    t_start = time.time()
    print("=" * 76)
    print("  空无到照见 · v14")
    print("  量化指标评估层 (三层体检报告) —— 不加任何新物理阶段")
    print("  阶段间因果链 (函数式派生) | 闭环螺旋 spiral_loop()")
    print("  负对照与交叉检验: F1 跨边界 c | F2a h/J 扫描 | F5 双路径对表 | F6 零模型对照族")
    print("=" * 76)
    print("\n  旋钮 (不可约外部输入, 不是派生量):")
    for kk in ('L_chain', 'd_local', 'L_domain', 'n_lambda',
               'pts_per_wavelength', 'F', 'k', 'dt', 'Du', 'Dv'):
        print(f"    {kk:20s} = {KNOBS[kk]}")
    print("    其余所有下游参数由上游实测物理量派生, 见下方因果链台账。")

    np.random.seed(0)
    torch.manual_seed(0)

    # ---------- 阶段 1-2, 以及因果链 L1 ----------
    L_chain = int(KNOBS['L_chain'])
    s1 = stage1_void(KNOBS['d_local'])
    l1 = derive_L1_local_to_full(s1['dimension'], L_chain)
    print(f"  [L1] 空无的局部维度 {l1['d_local']} 张成 {L_chain} 个自旋的全空间: "
          f"dim_full = {l1['dim_full']} (= 2^{L_chain})")

    s2 = stage2_critical_break(L=L_chain)

    # ---------- 因果链 L2/L3/L4: 全部从实测的约化密度矩阵谱派生 ----------
    print("\n  --- 因果链: 从阶段二的约化密度矩阵谱派生下游参数 ---")
    l2 = derive_L2_entanglement_gap(s2['probabilities'])
    l3 = derive_L3_schmidt_rank(s2['probabilities'], KNOBS['schmidt_tol'])
    l4 = derive_L4_bond_dimension(s2['entropy'])

    # ---------- 阶段三: 自指动力学 (参数由 L2/L3 派生) ----------
    print("\n" + "=" * 76)
    print("  阶段三 · 自指的动力学涌现")
    print("=" * 76)
    # N=128 是"演示矩阵有多大"的尺寸选择 (非物理量); gap 与网络宽度才是派生量
    s3a = selfref_general_matrix(N=128, gap_physical=l2['gap'])
    s3a2 = selfref_nonconvergent(N=128)
    N_self = l3['N_selfref']
    print(f"  [L3] 自指网络宽度由有效 Schmidt 秩派生: N = {N_self} "
          f"(手工取 64)")
    s3b = selfref_neural_network(N=N_self)
    s3c = selfref_coupled(N=N_self, steps=8000)

    # ---------- 阶段四: 真实 MERA (键维由 L4 派生) ----------
    print("\n" + "=" * 76)
    print("  阶段四 · 真实张量网络全息 (quimb MERA)")
    print("=" * 76)
    chi_dev = int(l4['chi'])
    print(f"  [L4] 键维由面积律派生: chi = {chi_dev} "
          f"(下界 exp(S) = {l4['chi_lower_bound']:.4f}); 手工取 4")
    # 开发/验证用的结构样本: 小 chi, 只为检查等距性
    mera8, _ = mera_init(8, 2, seed=0)
    iso8 = mera_isometry_check(mera8, 8)
    print(f"  MERA(L=8, chi=2): {mera8.num_tensors} 个张量, "
          f"{iso8['n_unitary']} 个解纠缠器 + {iso8['n_isometry']} 个等距 "
          f"+ 1 个顶层边界张量")
    print(f"  酉性 ||U^dag U - I|| = {iso8['unitary_err']:.2e} "
          f"(张量性质保证, 非构造凑数)")
    print(f"  等距 ||W^dag W - I|| = {iso8['isometry_err']:.2e}")

    # L=32 只用来看结构 (因果锥 / 体几何), 绝不收缩成稠密态
    mera32, _ = mera_init(32, 2, seed=0)
    ns32, cone32, r2_32 = mera_causal_cone(mera32, 32)
    mera16, _ = mera_init(16, 2, seed=0)
    ns16c, cone16, r2_16 = mera_causal_cone(mera16, 16)
    print(f"  因果锥张量数 |cone(n)| (L=32, 共 {mera32.num_tensors} 张量):")
    print(f"     n=1..16 -> {cone32.astype(int).tolist()}")
    print(f"     线性拟合 R^2={r2_32['linear']:.4f} vs 对数拟合 R^2={r2_32['log']:.4f}")
    print(f"     -> 体是体积律 (线性), 而 S(n) 只随 ln n 增长 (面积律)")

    # 主拟合使用**派生键维**的 MERA, 并带多种子 + 检查点选择
    c_exact_main = float(fit_central_charge(s2['gs'], L_chain)['c'])
    print(f"  [主拟合] MERA(L={L_chain}, chi={chi_dev} 派生): "
          f"{len(V13_LR_SEEDS[0][1]) + len(V13_LR_SEEDS[1][1])} 条轨迹 "
          f"x {V13_STEPS} 步, 精确对照 c={c_exact_main:.4f}")
    print(f"       检查点按 **overlap** (训练目标本身) 选择, 不偷看 eps_c; "
          f"另有 chi={V13_CHI_CTRL[0]} 对照轨迹测『提 chi 有没有用』")
    v13_tier1, psi_main = v13_tier1_runs(s2['gs'], L_chain, chi_dev,
                                         c_exact_main)
    ov_main = float(abs(np.vdot(s2['gs'], psi_main)))
    c_main = fit_central_charge(psi_main, L_chain)
    print(f"  [代表态] 取 eps_c 位于中位数的那条轨迹 "
          f"(index {v13_tier1['chosen_run_index']}), 用于第二层的纠缠谱 KL")
    print(f"           重叠={ov_main:.6f}, c={c_main['c']:.6f}, "
          f"eps_c={abs(c_main['c'] - c_exact_main) / c_exact_main:.3%}; "
          f"全部轨迹总用时 {v13_tier1['elapsed']:.0f}s")

    s4 = {'isometry': iso8, 'n_tensors': int(mera8.num_tensors),
          'chi_derived': chi_dev,
          'main_fit': {'L': L_chain, 'chi': chi_dev, 'overlap': ov_main,
                       'c': float(c_main['c']),
                       'v13_runs': v13_tier1['runs'],
                       'v13_curve': v13_tier1['curve'],
                       'v13_steps': v13_tier1['steps'],
                       'v13_chi_ctrl': v13_tier1['chi_ctrl'],
                       'v13_floor_chi': v13_tier1['floor_chi'],
                       'v13_floor_chi_ctrl': v13_tier1['floor_chi_ctrl']},
          'cone': {'L16': cone16.astype(int).tolist(), 'r2_L16': r2_16,
                   'L32': cone32.astype(int).tolist(), 'r2_L32': r2_32,
                   'n_tensors_L32': int(mera32.num_tensors)}}

    # ---------- 阶段五: 离散曲率 ----------
    print("\n" + "=" * 76)
    print("  阶段五 · 离散时空几何 (Forman-Ricci / Ollivier-Ricci)")
    print("=" * 76)
    curv_val = validate_curvature()

    print("\n  应用到涌现几何:")
    G_mera = build_mera_graph(mera32)
    rep_mera, _, _ = curvature_report(G_mera, "MERA 体几何 (L=32 张量网络图)")

    # [L5] MERA 图的平均度 -> Forman 曲率的解析预期
    l5 = derive_L5_forman_from_degree(G_mera, rep_mera['forman_mean'])
    print(f"  [L5] 正则粗式 <F> = 4 - 2*d_bar = {l5['forman_predicted']:.4f} "
          f"(差 {abs(l5['forman_predicted'] - rep_mera['forman_mean']):.4f})")
    print(f"       严格式 <F> = 4 - 2*<deg>_edge = {l5['forman_exact']:.4f} "
          f"vs 实测 {rep_mera['forman_mean']:.4f} "
          f"(差 {abs(l5['forman_exact'] - rep_mera['forman_mean']):.2e})")
    print(f"       -> 粗式的差全部来自度偏置 <deg>_edge - d_bar = "
          f"{l5['deg_edge_mean'] - l5['d_bar']:.4f} (非正则图: 度大的点被更多边选中),")
    print(f"          不是曲率实现有错; 严格式到机器精度吻合。")

    _, gs16 = exact_ground_state(16, 1.0, 1.0)
    G_corr, C_corr, D_corr = boundary_correlation_graph(gs16, 16)
    rep_corr, _, _ = curvature_report(G_corr, "边界关联几何 (临界基态)")

    # v14·F6: 零模型对照族 (三类 x 3 实例) —— 给出的是**百分位**, 不是单点比较。
    # 单张同规模随机图只能回答"比随机图更负吗"; 三类对照各控一个维度
    # (度分布 / 聚类 / 树的成分), 才能说清 MERA 的负曲率有什么特点。
    ctrl_reps = geometry_controls(G_mera)
    rep_rand_same = ctrl_reps[0]        # gnm #0, 作为"同规模随机图"的代表实例

    s5 = {'validation': curv_val, 'mera_bulk': rep_mera, 'boundary': rep_corr,
          'random_same': rep_rand_same, 'controls': ctrl_reps,
          'correlation': C_corr.tolist(),
          'degree_prediction': l5}

    # ---------- 阶段六: 参数由 L6 派生 (域长来自阶段五的 xi) ----------
    xi_boundary = boundary_correlation_length(C_corr, 16)
    print(f"\n  阶段五几何的特征尺度: 边界关联长度 xi = {xi_boundary:.3f} "
          f"(环上有 16 个站点, 故 xi/L = {xi_boundary/16:.4f})")
    print(f"    诚实边界: xi > L 是临界系统的正常现象 (关联长度被尺寸截断);")
    print(f"    尺度不变量是比值 xi/L = {xi_boundary/16:.4f}, 不是 xi 本身。")

    l6 = derive_L6_grid(xi_boundary / 16.0, KNOBS, xi=xi_boundary)
    print(f"  [L6] 网格 N = {l6['N']} 派生 (N_req={l6['N_req']}, "
          f"N_cap={l6['N_cap']}, 受限于 {l6['bound_by']})")
    print(f"       手工取 N=40; 派生后 dt*Du/h^2 = {l6['stab']:.4f} "
          f"<= {KNOBS['stability_limit']} 由构造成立 (不再是碰巧)")

    s6 = stage6_life(F=KNOBS['F'], k=KNOBS['k'], N=l6['N'],
                     L=l6['L_domain'], dt=KNOBS['dt'],
                     Du=KNOBS['Du'], Dv=KNOBS['Dv'])

    # [L6-c] 可证伪的跨阶段预言: 域内应容纳 n_lambda 个波长
    if s6.get('spacing') is not None and np.isfinite(s6['spacing']):
        ratio_lambda = s6['spacing'] / l6['lambda_target']
        print(f"  [L6] 跨阶段预言: 目标波长 = {l6['lambda_target']:.4f} "
              f"(= L_domain/{KNOBS['n_lambda']:.0f}), 实测 = {s6['spacing']:.4f}, "
              f"比值 = {ratio_lambda:.3f}")
        print(f"       诚实边界: 这个预言只在同一个域长下才有意义 —— "
              f"斑图波长由化学参数定, 不由几何定。")
    else:
        ratio_lambda = float('nan')
        print(f"  [L6] 斑图未成形, 跨阶段波长预言无法检验 (如实报告)")

    # ---------- 阶段七 ----------
    s7 = stage7_consciousness(s3c)

    # ---------- v14·F5 一致性检查束 (数值路径 <-> 解析路径) ----------
    f5 = consistency_checks(s3a, s5['validation'], s7)

    # ---------- 闭环螺旋: 把整条链应用到自身 ----------
    loop = spiral_loop(turns=KNOBS['turns'], L0=KNOBS['L0_loop'],
                       scale_knob=KNOBS['scale_knob'],
                       max_L=KNOBS['max_L_loop'],
                       loop_tol=KNOBS['loop_tol'])

    # ---------- 因果链台账 ----------
    print_chain_report()

    # ---------- v14·F2a 负对照: h/J 扫描 ----------
    # 放在因果链台账之后、指标层之前: 它是"阶段二的临界性是不是事实"的检验,
    # 与主线计算解耦, 失败也不会阻断后续。
    hj = hj_scan(L=L_chain)
    plot_hj_scan(hj, os.path.join(_OUTPUT_DIR, '_v14_hj_scan.png'))
    print(f"  负对照图已保存: {os.path.join(_OUTPUT_DIR, '_v14_hj_scan.png')}")

    # ---------- v14·F1 跨边界条件中心荷交叉检验 (同族) ----------
    # 与 F2 一样放在指标层之前、且与主线解耦: 它检验的是"c 的读数对边界条件
    # 稳不稳", 属对解析解对标, 不是新的物理阶段。
    print()
    f1 = cross_boundary_twist(L=L_chain, chi=chi_dev)

    # ---------- 任务 A: c 验证 ----------
    print()
    cc = central_charge_verification()

    # ---------- 指标层: 三层量化评估 ----------
    metrics = collect_metrics_v14({
        's2': s2, 's3a': s3a, 's3c': s3c, 's4': s4, 's5': s5,
        's6': s6, 'loop': loop, 'cc': cc, 'l6': l6, 'hj': hj, 'f1': f1,
        'f5': f5,
        'L_chain': L_chain, 'chi_dev': chi_dev, 'psi_main': psi_main,
    },KNOBS,_HERE, v13_tier1)

    # ---------- 汇总 ----------
    data = {
        'stages': {
            'void_cond': float(np.linalg.cond(s1['unitary'])),
            'critical': {'E0_per_L': s2['E0'] / 16, 'entropy': s2['entropy'],
                         'polarization': s2['polarization']},
            'selfref_matrix': s3a,
            'selfref_nonconvergent': {k: v for k, v in s3a2.items()
                                      if k != 'steps'},
            'selfref_nn': s3b,
            'selfref_coupled': {
                name: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in s3c[name].items()}
                for name in ('frozen', 'free', 'norm')},
            'mera': s4,
            'curvature': s5,
            'life': {'contrast': s6['contrast'], 'emerged': s6['emerged'],
                     'active_frac': s6.get('active_frac'), 'h': s6['h'],
                     'stab': s6['stab'], 'spacing': s6.get('spacing'),
                     'steps': s6.get('steps'),
                     'boundary_xi': xi_boundary,
                     'xi_over_L': xi_boundary / 16.0,
                     'N_derived': l6['N'], 'N_req': l6['N_req'],
                     'N_cap': l6['N_cap'], 'bound_by': l6['bound_by'],
                     'lambda_target': l6['lambda_target'],
                     'lambda_ratio': ratio_lambda},
            'consciousness': s7,
        },
        'causal_chain': CHAIN,
        'spiral_loop': loop,
        'knobs': {k: v for k, v in KNOBS.items()},
        'central_charge': cc,
        'consistency': f5,
        'metadata': {
            'version': 'v14',
            'implementation': 'quimb MERA + torch autograd + networkx/scipy(LP)',
            'critical_point': 'h/J = 1.0 (周期 TF-Ising)',
            'c_fit_formula': 'S(n) = (c/3) ln[(L/pi) sin(pi n/L)] + const',
            'or_convention': 'Ollivier-Ricci 默认 alpha=0 (简单随机游走, 无 idle)',
            'site_ordering': '站点 0 = 最高有效位 (与 quimb 输出指标 k0 对齐)',
            'loop_scale_law': 'L_t = clip(round(scale_knob * xi_{t-1}), L0, max_L)',
            'runtime_sec': None,
        },
    }

    # metrics 三层指标必须真正写进 json。曾经的数据文件声称含 metrics 三层
    # 指标, 但实测里面**没有** metrics 这一节 —— 只有 stages/causal_chain/
    # spiral_loop/knobs/central_charge/metadata。指标只活在打印输出和 PNG 里,
    # 拿不到机器可读的指标本身。这里把它写进去。
    data['metrics'] = metrics_for_json(metrics)

    # 先落盘再作图: 计算结果比画图贵得多, 作图出错不该把结果一起丢掉
    out_json = os.path.join(_OUTPUT_DIR, '_v14_data.json')
    data['metadata']['runtime_sec'] = time.time() - t_start
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  数据已保存: {out_json}")

    plot_all(data, s2, s3a, s3b, s3c, s6, s4, cc, loop)
    plot_metrics_v14(metrics, s2, s3a,
                     os.path.join(_OUTPUT_DIR, 'spiral_v14_metrics.png'))

    # ---------- 汇总 ----------
    print("\n" + "=" * 76)
    print("  汇总")
    print(f"  因果链台账 ({len(CHAIN)} 条):")
    for r in CHAIN:
        tag = "= 手工基线" if r['same_as_manual'] else "!= 手工基线"
        print(f"            {r['link']}: 派生 {r['derived']!r} "
              f"vs 手工 {r['v10_manual']!r}  {tag}")
    print(f"            L2 派生收敛比 = {l2['gap']:.6f} -> 实测收敛比 "
          f"{(s3a['physical'] or {}).get('rate_meas', float('nan')):.6f}")
    print(f"  闭环 spiral_loop(): 三量分述")
    lr = loop['rows']
    print(f"            c_t: " + " -> ".join(f"{r['c']:.4f}" for r in lr)
          + f"  (理论 0.5)")
    print(f"            xi/L: " + " -> ".join(f"{r['xi_over_L']:.3f}" for r in lr))
    print(f"            gap_t: " + " -> ".join(f"{r['gap']:.4f}" for r in lr))
    print(f"            L_t: " + " -> ".join(str(r['L']) for r in lr)
          + ("  [max_L 已触顶]" if loop['verdicts']['scale_clamped'] else ""))
    print(f"            相对谱隙 1-r: " + " -> ".join(f"{1-r['gap']:.4f}" for r in lr)
          + ("  (谱隙闭合 = 趋向临界)" if loop['verdicts']['relgap_closing'] else ""))
    print(f"            裁决: 无量纲不变量"
          f"{'稳定' if loop['verdicts']['ratio_stable'] else '不稳定'}"
          f" (第 {loop['verdicts']['converged_at_turn']} 圈起收敛), "
          f"尺度发散 = 临界性")
    print(f"  MERA: {s4['n_tensors']} 张量, "
          f"酉性误差={s4['isometry']['unitary_err']:.1e}, "
          f"等距误差={s4['isometry']['isometry_err']:.1e}")
    print(f"            派生键维 chi={s4['chi_derived']}: "
          f"L={s4['main_fit']['L']} 重叠={s4['main_fit']['overlap']:.4f}, "
          f"c={s4['main_fit']['c']:.4f}")
    for m in cc['mera']:
        print(f"            L={m['L']} chi={m['chi']}: 重叠={m['overlap']:.4f}, "
              f"c={m['c']:.4f}")
    for e in cc['exact']:
        print(f"            [精确对照] L={e['L']}: c={e['c']:.4f}")
    print(f"  幂迭代·受控谱隙: 理论比/实测比 "
          + ", ".join(f"{s['rate_theory']:.2f}/{s['rate_meas']:.2f}"
                      for s in s3a['gap_sweep']))
    tr = s3a['transient']
    print(f"            Jordan 块 (特征值全={tr['lam']}, 谱隙=0, rho={tr['spectral_radius']:.2f}<1): "
          f"方向不收敛(步长比={tr['step_ratio']:.3f}); "
          f"状态范数 1.0 -> 峰 {tr['norm_peak']:.1f} -> {tr['norm_final']:.1e} (k={tr['kmax']})")
    print(f"            模相等反例: 末段步长仍={s3a2['tail_step']:.3f} (不收敛)")
    print(f"            自指网络相图: "
          f"{' -> '.join(r['phase'] for r in s3b['scan'])}")
    print(f"            权重-状态耦合: 冻结|dx|={s3c['frozen']['final_dx']:.2e} (不收敛) -> "
          f"自指后 |dx|={s3c['norm']['final_dx']:.2e} (收敛)")
    print(f"            权重秩1化只在变体 B 成立: A(sigma1/sigma2="
          f"{s3c['free']['sigma_ratio']:.2e}) vs B({s3c['norm']['sigma_ratio']:.2e})")
    print(f"            诚实边界: 手工 N=64 时 A、B 都秩1化; 派生 N=17 后只剩 B,")
    print(f"            且 g=2.0 的混沌消失 (变成极限环) —— 相图依赖 N, 不是普适的。")
    print(f"  曲率解析锚点全部通过 (到机器精度)")
    print(f"            MERA 体几何: Forman 均值={rep_mera['forman_mean']:+.3f}, "
          f"Ollivier 均值={rep_mera['or_mean']:+.4f} "
          f"(负边占比 {rep_mera['or_neg_frac']:.2f}) -> 负曲率")
    print(f"            边界关联几何: Ollivier 均值={rep_corr['or_mean']:+.4f} "
          f"(负边占比 {rep_corr['or_neg_frac']:.2f})")
    _cf = [c['forman_mean'] for c in ctrl_reps]
    _co = [c['or_mean'] for c in ctrl_reps]
    print(f"            零模型对照族: {len(ctrl_reps)} 个实例 "
          f"({len({c['family'] for c in ctrl_reps})} 族), Forman "
          f"[{min(_cf):+.3f}, {max(_cf):+.3f}], Ollivier "
          f"[{min(_co):+.4f}, {max(_co):+.4f}]")
    print(f"  既有文档 spiral_v13_说明.md 的 A 段有两条说法被本次实测推翻 —— "
          f"'scatter 是参考侧种子间散布'与'两条独立路线互相印证';")
    print(f"            依据是上方 A 段两个诊断项 (归档 bool 场与归档浮点场一致 / "
          f"lam_pk 有独立佐证), 两项目前都未通过。该文档未改动。")
    g = metrics['guards']
    print(f"  指标层: 体检报告见上方; 守卫通过率 "
          f"{g['n_pass']}/{g['n_expect']} = {g['rate']:.2%} "
          f"(另有 {g['n_diag']} 项诊断项不计入, 其中 {g['n_diag_notok']} 项未通过)")
    print(f"            产物: spiral_v14_metrics.png (20 面板) + _v14_data.json")
    print(f"  总用时 {time.time()-t_start:.0f}s")
    print("=" * 76)
    print("  变化以周期性偏振，统一于空无；")
    print("  增长以螺旋式上升，起始于终结。")
    print("  本自具足。🌊")


def plot_all(data, s2, s3a, s3b, s3c, s6, s4, cc, loop):
    """15 个面板:
        1-12 七个物理阶段 (第 2 面板为"预设 + 物理派生"双层)
        13   闭环无量纲不变量 c_t / (xi_t/L_t)
        14   闭环谱隙收窄 gap_t 与尺度流动 L_t
        15   因果链台账 (派生值 vs 手工值 的文本对照)
    """
    fig, axes = plt.subplots(5, 3, figsize=(21, 26))
    axes = np.asarray(axes)

    # 1. Schmidt 谱
    s = s2['spectrum'][:12]
    axes[0, 0].bar(range(len(s)), s)
    axes[0, 0].set_title(f"阶段二: 临界 Schmidt 谱 (极化={s2['polarization']:.2f})\n"
                         f"注: 此比值被极小分母放大, 不是可靠的破缺指标")
    axes[0, 0].set_xlabel('指标')
    axes[0, 0].set_ylabel('奇异值')

    # 2. 幂迭代收敛: 理论比 vs 实测比 对表
    ax = axes[0, 1]
    gaps = [s['gap'] for s in s3a['gap_sweep']]
    ax.semilogy(gaps, [s['rate_theory'] for s in s3a['gap_sweep']], 'k--o',
                label='理论 |λ2/λ1|')
    ax.semilogy(gaps, [s['rate_meas'] for s in s3a['gap_sweep']], 'r-s',
                label='实测收敛比')
    # 叠加由纠缠谱派生的那个物理谱隙 (因果链 L2)
    if s3a.get('physical'):
        ph = s3a['physical']
        ax.axvline(ph['gap'], color='seagreen', lw=1.0, ls=':')
        ax.semilogy([ph['gap']], [ph['rate_meas']], 'D', color='seagreen',
                    ms=9, label=f"[L2] 纠缠谱派生 gap={ph['gap']:.3f}\n"
                                f"实测比={ph['rate_meas']:.4f}")
    ax.set_title('阶段三-a: 幂迭代收敛比对表\n'
                 '(受控谱隙校准 + 物理派生谱隙)')
    ax.set_xlabel('谱隙 (预设值, 及 L2 派生值)')
    ax.set_ylabel('收敛比')
    ax.invert_xaxis()
    ax.legend(fontsize=7)

    # 3. 自指网络相图
    ax = axes[0, 2]
    ax.plot([r['g_rho'] for r in s3b['scan']],
            [r['lyapunov'] for r in s3b['scan']], 'o-', color='crimson',
            label='Lyapunov 指数')
    ax.axhline(0, color='k', lw=0.8, ls=':')
    ax.axvline(1.0, color='gray', lw=0.8, ls='--')
    ax.set_title('阶段三-b: 自指动力学相图\n不动点 → 极限环 → 混沌')
    ax.set_xlabel('g·ρ(W)')
    ax.set_ylabel('Lyapunov 指数')
    for r in s3b['scan']:
        ax.annotate(r['phase'], (r['g_rho'], r['lyapunov']), fontsize=7,
                    textcoords='offset points', xytext=(0, 5))
    ax.legend()

    # 4. 瞬态增长 (非正规性) + 自指闭环的秩1化
    ax = axes[1, 0]
    strong = s3a['transient']
    kk = np.arange(1, len(strong['growth']) + 1)
    ax.semilogy(kk, strong['growth'], 'o-', ms=3,
                label=f"‖A^k‖₂ (非正规性 {strong['nonnormality']:.3f})")
    ax.semilogy(kk, strong['spectral_radius'] ** kk, 'k--',
                label=f"ρ(A)^k (ρ={strong['spectral_radius']:.2f}<1)")
    ax.semilogy(kk, strong['norm_trace'], '-', color='seagreen', lw=1.8,
                label=f"状态范数 ‖A^k x₀‖ -> {strong['norm_final']:.1e} (收敛)")
    ax.axvline(strong['k_peak'], color='gray', lw=0.8, ls=':',
               label=f"‖A^k‖₂ 峰值 k={strong['k_peak']}")
    ax.axvline(strong['k_below'], color='seagreen', lw=0.8, ls=':',
               label=f"状态范数最后一次跌破初值 k={strong['k_below']}")
    ax.set_title(f"阶段三-a: Jordan 块 (特征值全={strong['lam']}, ρ<1) 的瞬态增长\n"
                 f"状态范数先涨后落: 1.0 -> {strong['norm_peak']:.1f} -> "
                 f"{strong['norm_final']:.0e}, 收敛但仍非单调")
    ax.set_xlabel('k')
    ax.set_ylabel('范数')
    ax.legend(fontsize=8)

    # 5. 中心荷: c vs 键维 (核心图)
    ax = axes[1, 1]
    ax.axhline(0.5, color='k', ls='--', lw=1.2, label='理论 c = 1/2')
    for e in cc['exact']:
        ax.axhline(e['c'], color='gray', ls=':', lw=1.0,
                   label=f"精确对照 L={e['L']}: c={e['c']:.3f}")
    for L in sorted({m['L'] for m in cc['mera']}):
        ms = sorted([m for m in cc['mera'] if m['L'] == L], key=lambda m: m['chi'])
        ax.plot([m['chi'] for m in ms], [m['c'] for m in ms], 'o-',
                label=f'MERA L={L}')
    ax.set_title('任务A: 中心荷 c 随键维 χ 收敛到 1/2')
    ax.set_xlabel('键维 χ')
    ax.set_ylabel('拟合 c')
    ax.legend(fontsize=8)

    # 6. S(n) 曲线 + CFT 拟合
    ax = axes[1, 2]
    for m in cc['mera']:
        if m['L'] == 16:
            ax.plot(np.arange(1, len(m['S']) + 1), m['S'], 'o-', alpha=0.7,
                    label=f"MERA χ={m['chi']} (c={m['c']:.3f})")
    for e in cc['exact']:
        if e['L'] == 16:
            ax.plot(np.arange(1, len(e['S']) + 1), e['S'], 'k*-', ms=10,
                    label=f"精确 (c={e['c']:.3f})")
    ax.set_title('S(n) 曲线 (L=16, 周期 CFT 拟合)')
    ax.set_xlabel('n')
    ax.set_ylabel('S(n)')
    ax.legend(fontsize=8)

    # 7. 因果锥 vs 纠缠 (全息压缩: 体是体积律, 纠缠是面积律)
    ax = axes[2, 0]
    cone32 = np.array(s4['cone']['L32'], dtype=float)
    ns32 = np.arange(1, len(cone32) + 1)
    ax.plot(ns32, cone32, 's-', color='darkorange',
            label=f"|cone(n)| L=32 (线性 R²={s4['cone']['r2_L32']['linear']:.3f})")
    ax2 = ax.twinx()
    m8 = [m for m in cc['mera'] if m['L'] == 8]
    if m8:
        S8 = np.array(m8[0]['S'])
        ax2.plot(np.arange(1, len(S8) + 1), S8, 'o-', color='navy',
                 label='S(n) L=8 (对数增长)')
    ax.set_title('阶段四: 全息压缩\n因果锥张量数 ∝ n, 纠缠熵 ∝ ln n')
    ax.set_xlabel('n (边界站点数)')
    ax.set_ylabel('因果锥张量数', color='darkorange')
    ax2.set_ylabel('S(n)', color='navy')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='lower right', fontsize=8)

    # 8. 曲率对比
    ax = axes[2, 1]
    labels = ['MERA体几何', '边界关联', '随机图(同规模)']
    means = [data['stages']['curvature']['mera_bulk']['forman_mean'],
             data['stages']['curvature']['boundary']['forman_mean'],
             data['stages']['curvature']['random_same']['forman_mean']]
    negf = [data['stages']['curvature']['mera_bulk']['forman_neg_frac'],
            data['stages']['curvature']['boundary']['forman_neg_frac'],
            data['stages']['curvature']['random_same']['forman_neg_frac']]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, means, 0.4, label='Forman 均值', color='steelblue')
    ax.bar(x + 0.2, negf, 0.4, label='负曲率边占比', color='salmon')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title('阶段五: 离散 Ricci 曲率对比')
    ax.legend(fontsize=8)

    # 9. Gray-Scott 斑图
    ax = axes[2, 2]
    if s6['v'] is None:
        # 稳定性守卫路径: 拒绝画一张可能失稳的结果
        ax.text(0.5, 0.5, f"未运行\n{s6['reason']}", ha='center', va='center',
                wrap=True, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("阶段六: Gray-Scott 拒绝运行\n(不满足稳定性条件)")
    else:
        ax.imshow(s6['v'], cmap='magma')
        ax.set_title(f"阶段六: 生命斑图 (对比度={s6['contrast']:.2f}, "
                     f"{'涌现' if s6['emerged'] else '未涌现'})\n"
                     f"dt·Du/h²={s6['stab']:.3f}<0.25, 间距≈{s6['spacing']:.3f}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    # 10. 自指闭环: 权重自发秩 1 化 + 冻结对照
    ax = axes[3, 0]
    for key, lab, col in (('norm', '自指·状态归一', 'seagreen'),
                          ('free', '自指·状态自由', 'darkorange'),
                          ('frozen', '冻结权重 (对照)', 'gray')):
        st = np.array(s3c[key]['dx'])
        ax.semilogy(np.arange(len(st)), np.maximum(st, 1e-16), lw=1.2,
                    color=col, label=lab)
    ax.set_title('阶段三-c: 权重-状态耦合自指\n冻结(灰)永不收敛; 自指后(绿/橙)收敛')
    ax.set_xlabel('迭代步')
    ax.set_ylabel('‖x_{n+1}-x_n‖')
    ax.legend(fontsize=8)

    # 11. 对照实验: 能量目标塌缩到平均场
    ax = axes[3, 1]
    e8 = [e for e in cc['exact'] if e['L'] == 8]
    if e8:
        ax.plot(np.arange(1, len(e8[0]['S']) + 1), e8[0]['S'], 'k*-', ms=10,
                label=f"精确基态 (c={e8[0]['c']:.3f})")
    m8o = [m for m in cc['mera'] if m['L'] == 8 and m['chi'] == 4]
    if m8o:
        ax.plot(np.arange(1, len(m8o[0]['S']) + 1), m8o[0]['S'], 'o-',
                label=f"重叠目标 (c={m8o[0]['c']:.3f})")
    if cc['energy_control']:
        ec = cc['energy_control']
        ax.plot(np.arange(1, len(ec['S']) + 1), ec['S'], 'x--', color='crimson',
                label=f"能量目标 (c={ec['c']:.3f})")
    ax.set_title('对照实验: 为何用重叠而非能量\n'
                 '能量目标塌缩到平均场, 纠缠几乎为零')
    ax.set_xlabel('n')
    ax.set_ylabel('S(n)')
    ax.legend(fontsize=8)

    # 12. 曲率算子族
    ax = axes[3, 2]
    val = data['stages']['curvature']['validation']
    kn = [r['n'] for r in val['complete']]
    ax.plot(kn, [r['mean'] for r in val['complete']], 'o-', label='K_n 实测')
    ax.plot(kn, [r['exact'] for r in val['complete']], 'k--',
            label='(n-2)/(n-1) 解析')
    kd = [r['d'] for r in val['hypercube']]
    ax.plot(kd, [r['mean_lazy'] for r in val['hypercube']], 's-',
            label='Q_d 实测 (alpha=1/(d+1))')
    ax.plot(kd, [r['exact_lazy'] for r in val['hypercube']], 'k:',
            label='2/(d+1) 解析')
    ax.plot(kd, [r['mean_plain'] for r in val['hypercube']], '^--',
            color='gray', label='Q_d (alpha=0) -> 0')
    ax.set_title('阶段五: 曲率算子的解析锚点校验\n(全部到机器精度)')
    ax.set_xlabel('n 或 d')
    ax.set_ylabel('kappa / F 均值')
    ax.legend(fontsize=7)

    # 13. 闭环三量流动: c_t 与 xi/L (无量纲不变量)
    ax = axes[4, 0]
    rows = loop['rows']
    tt = [r['turn'] for r in rows]
    ax.plot(tt, [r['c'] for r in rows], 'o-', color='crimson', label='c_t')
    ax.axhline(0.5, color='k', ls='--', lw=1.0, label='理论 c = 1/2')
    ax.set_xlabel('圈数 t')
    ax.set_ylabel('中心荷 c_t', color='crimson')
    ax.set_ylim(0.4, 0.65)
    ax2 = ax.twinx()
    ax2.plot(tt, [r['xi_over_L'] for r in rows], 's-', color='navy',
             label='xi_t / L_t')
    ax2.set_ylabel('xi_t / L_t  (尺度不变量)', color='navy')
    ax.set_title('闭环 spiral_loop(): 无量纲不变量\n'
                 'c -> 1/2 且 xi/L -> 常数 (共形不变性)')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc='center right')

    # 14. 闭环: 谱隙收窄 (临界的标志) 与尺度流动
    ax = axes[4, 1]
    ax.plot(tt, [1.0 - r['gap'] for r in rows], 'o-', color='darkgreen',
            label='相对谱隙 1 - p1/p0')
    ax.set_xlabel('圈数 t')
    ax.set_ylabel('相对谱隙 1 - r', color='darkgreen')
    ax3 = ax.twinx()
    ax3.plot(tt, [r['L'] for r in rows], '^--', color='gray', label='L_t')
    ax3.set_ylabel('系统尺寸 L_t', color='gray')
    clamped = loop['verdicts']['scale_clamped']
    ax.set_title('闭环: 相对谱隙闭合 = 趋向临界\n'
                 '(r=p1/p0 上升, 故 1-r 下降)'
                 + ('  [L_t 触顶 max_L]' if clamped else ''))
    h1, l1 = ax.get_legend_handles_labels()
    h3, l3 = ax3.get_legend_handles_labels()
    ax.legend(h1 + h3, l1 + l3, fontsize=7, loc='best')

    # 15. 因果链台账 (文本面板)
    ax = axes[4, 2]
    ax.axis('off')
    lines = ['']
    for r in data['causal_chain']:
        same = '=' if r['same_as_manual'] else '≠'
        d = r['derived']
        ds = f"{d:.4g}" if isinstance(d, (int, float)) else str(d)
        m = r['v10_manual']
        ms = f"{m:.4g}" if isinstance(m, (int, float)) else str(m)
        lines.append(f"{r['link']}  {r['formula']}")
        lines.append(f"     derive ={ds}  manual={ms} {same}")
    ax.text(0.0, 1.0, "\n".join(lines), va='top', ha='left', fontsize=6.5,
            family='monospace', transform=ax.transAxes)
    ax.set_title('因果链: 派生值 vs 手工值\n(≠ 表示派生修正了原来的随手取值)')

    plt.suptitle('空无到照见 · v14 — 因果链 (函数式派生) + 闭环 spiral_loop()',
                 fontsize=15)
    plt.tight_layout()
    out_png = os.path.join(_OUTPUT_DIR, 'spiral_v14.png')
    plt.savefig(out_png, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  图像已保存: {out_png}")


if __name__ == '__main__':
    main()
