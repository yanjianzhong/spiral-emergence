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
spiral_model_v16.py
====================
「空无到照见」· 完整可运行版本 v16
================================================================================

【v16 是什么 —— 先读这一节, 再看下面从 v15 起累积的各节】

  v16 是 v15 之上的**增量**: 三个工作包 B1 / B2 / B3。三者**都不加新物理阶段**,
  改的是"连线、口径、审计" —— 七阶段本身一字未动, `v15/` 已冻结 (v16 不 import
  v15 的任何代码, 两个版本目录各自自包含)。

    B1 · L1 的第一条真实下游连线 (本文件 + spiral_metric_v16.py)
        缺口: `derive_L1_local_to_full` 算出的 dim_full **只被 print**,
        stage2 的 void_state 永远是 None ⇒ 推离 L1 不改变任何读数 ⇒
        W11 "L1 有到下游的因果通路" 与 W6 "负对照逐层覆盖" 两条诊断项
        **必然失败** —— 这是"造不出对照", 与"还没做"是两回事。
        做法: `derive_L1_void_to_chain` 把局部等权态张成 L 个自旋的域态, 再与
        h/J=32 的精确基态比重叠, 给出重叠曲线; `l1_void_pushaway` 提供 L1 的
        第一条负对照 (**固定 y 轴 R_y, 不是计划里的"随机轴"** —— 随机轴会让
        校准点每次运行都不同, 不可复现)。
        实测: 重叠对 h/J 单调上升且 h/J=32 时 > 0.99 (主判据); 转动 L1 局部态
        后重叠跌破 0.9 (推离); dim_full 与 2**L 相符 (**首次被真消费**);
        上述两条诊断项由"未通过"翻为"通过"; 负对照台账由 6/7 补到 **7/7**。
        **口径限制 (三句话不许并成一句)**: "等权大态 = h/J->inf 的基态"是
        **数学事实**; 把它叫"L1 到 L2 的因果连线"是**口径选择**; 推离守卫的
        overlap 由**被转动的态直接算出**, 标定的是**灵敏度尺度**, 不是独立的
        物理证伪。**L1 的成熟度仍记 B, 不升级** —— "缺口补上了"补的是**理由**,
        不是结论。B1 也**不修复**预设层审计缺口: 五条预设里只有 2 条
        ("等权初态"、"张量积结构")转为被消费, 其余 3 条仍是给定的。

    B2 · Gaussian cMERA (独立可执行体, **本文件不 import 它**)
        `_v16_cmera_gaussian.py`: 把"几何从态里算出来"放在 `2L x 2L` 协方差
        矩阵上做, **不构造 `2^L` 稠密向量**。**口径 A**: `H_til = H_open +
        J·iγ_{2L-1}γ_0` (偶宇称扇区), 每一条输出的第一行都必须打印它。
        三个探针里 **探针 2 已知 `EXIT=3`** —— 记作**不适用**, 不是"未修好":
        cMERA 尺度不变性的观测量是"逐层生成元的余弦相似度", 而口径 A 没有
        "层"、没有逐层生成元 ⇒ 原问题**无观测量可算**。玻色子对照判据 H2:
        费米子 χ = 0.788697 **不是常数 0.5**, 这是**口径差**不是 bug, 不许"修"。
        硬边界: 不修改既有 L5 路径 (`build_mera_graph` / `mera_bulk_invariance`
        / F6d 一字未动)、**不宣称"涌现时空"**、不宣称"MERA 实现引力对偶"。

    B3 · claim <-> test 口径对齐账本 (审计工具)
        `_v16_claim_ledger.py`: 检查 (a) 孤儿主张 (b) 幻影引用 (c) 伪独立
        (d) 未归层 (e) 台账 claim 列完整性; **(a) 只报不判**, 进判定的只有
        (b)(c)(d)(e)。**必须与全流程同进程跑** —— 它读的是**进程内**的
        METRICS / GUARDS (理由见下方"修直的口径"第 7 条)。
        同进程实测: 指标 27 / 守卫 77 / 负对照 13; 孤儿主张 8 条 / 分母 19 条
        (2026-09-26 由 9 条更新: `中心荷误差` 被既有守卫 `G6` 指认, 即 C-1 的接线);
        (b)(c)(d)(e) 全通过; L1 的口径匹配率由"无计分指标"填成 **1/1**
        (合计 9/18 -> 10/19 -> **11/19**)。

  【本版实测总账 (源 = `v16/_v16_run.log`, 2026-09-25)】
      计分守卫 **57/57 = 100.0%** (另有诊断项 20 项, 其中 **15** 项设计上就该
      失败; v15 是 17 项 —— 翻过来的那 2 项正是 B1 瞄准的 W11 与 W6);
      指标 **19/19** (第一层 6/6 + 第二层 13/13; 第三层 6 项全是诊断项);
      全流程总用时 **1879 s** —— **超出 `v16_plan.md` 登记的 1500 s 上界**
      (v15 基线 1245 s ⇒ 1.51 倍)。**如实登记, 不解释成"正常的"。**
      冻结核对 (`_v16_freeze_check.py`): v15 的 54 条计分守卫**按名字一条不少、
      无一翻成未通过** ⇒ 冻结成立; 只有 `G6 最坏轨迹也达标` 的**注册函数名**
      变了 (名字与判定都没变, 单列不计入回归)。判据是**按名字**而不是逐位相等:
      `exact_ground_state` 连调两次 E0 就有 3.553e-15 的抖动 (相对 1.7e-16),
      eigsh 的迭代起点不由本仓库控制 ⇒ "逐位相同"是**永远无法成立**的判据。

  【唯一的 runner】`v16/_v16_run_all.py` (不是本文件)
      `fast` 层 15 条 / `slow` 层 6 条 / `full` 层含同进程
      `full_pipeline+claim_ledger`。退出码: **0 = 全部如登记, 3 = 回归,
      2 = runner 自身错误**。已知负结果 (`cmera2` / `b2_boson_control` /
      `claim_ledger.standalone`) 登记为 `expect_pass=False, expect_exit=3`。
      ⚠️ runner 的 `open(LOG, 'w')` 是**截断写** ⇒ 日志不另存就会被下一次运行
      抹掉 (本版已手工归档 `_runall_*` 若干份)。

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

  第三层 · 结构对结构 (模型对外部斑图) —— v14 起已从"3D CDM 场对场"换成
      **2D 反应-扩散斑图结构对结构**, 三段梯子 (详见 spiral_metric_v16.py 头部):
        A 段 同方程不同盒子: 阶段六 Gray-Scott (36^2) <-> CLIP Gray-Scott (100^2, 3 种子)
        B 段 跨系统: CLIP Gray-Scott <-> Southampton BZ 时空图 / Reading 时序
        C 段 判别力: 跨系统距离 / 系统内散布 (比值 ~1 则 B 段无信息)
      交付物是**描述子距离**与"哪些描述子能迁移", 不是"模型复现了实验"。
      数据来源:
        # 1、Southampton · BZ 油滴网络时空图（Figure_3/4/6.zip、Figure_S1/S2/S3.zip,D0363_readme.txt）
        # 2. Reading · BZ 自振荡水凝胶延时影像（T_Geher-Herczegh_PhD_Exp_data.zip，1.57GB）
        # 3，CLIP 反应-扩散基准（gray_scott_data.tar.gz，442MB）
      本脚本只读上述数据集的**导出产物**, 不依赖任何宇宙学库)。

  【第三层的诚实边界 —— 这是本版最需要说清的一点】
  第三层**明确降级为方法学展示, 不设物理达标线**: 全部 6 项 (3.1~3.6) 都是
  诊断项, passed=None。理由有两条, 都是实测出来的, 不是措辞上的退让:

  (1) **A 段不是干净的"同方程不同盒子"**。实测两侧 f/k 落在 Pearson 相图的
      不同相区 (模型 F=0.035, k=0.060 斑点相 / 参考 f=0.01, k=0.042 蠕虫相),
      这不是同一个动力学区, 所以"同一 PDE"只在方程形式意义上成立。
      **注意**: 此前还并列过第二条混杂因子"盒子内波长数差一个量级" ——
      **v14 已撤回** (旧值来自坏参考场: 归档 bool 场的 lam_pk=1.45 格恰是二维
      离散谱角点伪像; 换成自建种子后两侧 n_lam 几乎相同)。但**也不能反过来
      说"盒子大小无影响"**: n_lam 是 kpk_k1 的换写、同样取自离散壳, 不是独立
      测量。参数相区这一条**仍然成立**, 单它一条就足以使 D_struct 无法单独
      归因给盒子大小。
  (2) **B 段的轴语义对不上**。Southampton 的图是 space x time, 不是二维空间
      斑图; 两个轴含义不同, 按二维场算描述子只是"用同一套尺子量", 不构成
      "实验斑图与数值斑图形态一致"的证据。
      这一条曾把口径写反 (见下方【v15 修直的三处口径】)。

  A 段参考侧在 v15 又换了一次: 换成由 `v15/spiral_v15_prepare.py` 生成的、与模型
  侧**逐字同参数同单位**的受控重算 (只变网格分辨率), 取代此前跨来源的比较。

  【"只报告不追目标" 的原则】
  本脚本**不做目标驱动的重算**。某个指标没达标时, 做的是**差距诊断**
  (残差来自键维截断? 有限尺寸? 优化步数?), 而不是加大 L/chi 直到数字好看。
  体检报告的意义在于诚实, 不在于满分。

  【负对照与交叉检验 —— 让判据可证伪】
  一个结论若在它不成立时也不会失败, 那句话就不是判据, 是散文。为此 v14 加
  五组检查, v15 再补六项 (见本段末尾)。它们**不加新物理阶段**: 加的都是
  对照、校验与评估, 不是新的物理环节, 既有阶段的结论一字未改。

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

  【v15 新增的六项 (W1~W12 里的 W3a/W3b/W4/W5/W6/W12)】

    W3b · 谱学中心荷 (能谱族) —— v15 唯一真正的**第二把尺子**
        低能能谱比值 (E2-E0)/(E1-E0) -> 8 给出 c, **独立于纠缠谱族**
        (Rényi 族实测偏差 7.97%, 已如实降级为诊断项)。实测 c 落在
        [0.49894, 0.50034], 最大偏差 0.212%。
        **口径警告 (必须连着读)**: "ratio->8" 与 "c->0.5" 是**同一次测量**
        (0.212% 就是 ratio 偏离 8 的量), **不许**写成"能谱族与 Rényi 族
        互相印证" —— 正确写法是"能谱族**独立于**纠缠谱族"。且该读数**不随 L
        单调收敛** (L=16 反而最差, 残差是 O(1/L) 有限尺寸修正), 只能说
        "相容到 0.21%", **不能**说"外推到 0.5"。

    W3a · 面积律 vs 对数律 (h/J 扫描)
        新增 h/J in {0.5, 1.0, 1.5} 扫描: 临界点给对数律, 偏离临界给面积律,
        实测 0.0007 / -0.0187 (完全不涨或轻微为负)。用的是**同一条 S 数据**,
        故只算自洽, 不是交叉检验。

    W4 · L7 的 Jacobian 谱半径 (上界对照)
        B 支 l7_defined_B = True 且谱半径 0.9412 != 1 —— 从"约定值"变成
        "已定义且非退化"。A 支恒等不可守, 故 l7_rho_jac_B 登记为**弱守卫**,
        不计分。

    W5 · 有限尺寸标度 (L=8..18)
        逐 L 给 c(L)/xi(L)/S(L/2)/E0/L, 逐 L 与 JW 闭合式核对。**更正了
        max_L 的归因** (详见下方诚实边界): 卡在 16 的不是算力, 是末圈 MERA
        要求 L 为 2 的幂。

    W6 · 逐层负对照台账 (`spiral_metric_v16._NEG_CTRL_TABLE`)
        把散在各层的对照按**七个物理阶段**归位 (此前只有 F1/F2a/F5/F6/F7
        这套按批次编的号, 读者无法判断哪层被覆盖)。v15 实测 12 条覆盖 6/7 层,
        缺口 `['L1']` (W12 之前是 5/7, 缺口 `['L1','L4']`);
        **v16·B1 补上 L1 那格 ⇒ 13 条覆盖 7/7 层, 缺口 `[]`**。
        (补的时候 L1 行的第 2~6 列被改写: `'缺'` -> `'已补'` 等; 其余 12 行
         **逐元素未动, 列结构未变** —— 表仍是 6/7 元素参差, 所以不加第 8 列。)
        配一条**台账完整性守护**: 交叉核对台账引用的守卫名是否真实存在于
        GUARDS、七层是否各出现一次。这不是物理结论, 是**台账自身的完整性** ——
        没有它, 台账可以写成宣传册。
        首次全流程实测抓到 **3 条幻影引用** (见下方"修直的三处口径")。

    W12 · 键维向下推离 (L4 台账上**唯一**的负对照)
        把 chi 压到面积律下界 exp(S) 以下 (chi=2), 要求下游复现中心荷的能力
        **正确失败**: 实测 eps_c 3.027% -> 19.595%, 退化 6.47x。
        **只验必要性方向** (界不成立时下游退化), **不验充分性** ——
        derive_L4_bond_dimension 的 docstring 本就写明它是"必要非充分条件"。

  【计分 / 诊断分离 (v15 的口径修正)】
  守卫分两类, 由 record_guard(..., expect_pass=) 在**注册时**显式设置:
      计分守卫 57 条 (v16 实测通过率 = 57/57; v15 是 54 条, 新增 3 条全是 B1)
      诊断项   20 条, 其中 15 条**设计上就该失败** (如 V6: 派生 N 之后变体 A
               收敛到 x*=0, 权重失去方向), 排除出分母
  不分开计的后果是把**真实的负面结果**算成"没通过", 通过率就变成了粉饰。
  指标同理: **19/19 达标** = 第一层 6/6 (1.1~1.5 + **v16·B1 新增的 1.8**;
  另有诊断项 1.6 预算漂移 / 1.7 chi 瓶颈归因) + 第二层 2.1~2.13 (13 项);
  第三层 3.1~3.6 全部是诊断项 (passed=None), 不进分母。
  **v15 的 54 条计分守卫一条不少、判定无一翻转** (见上方"冻结核对")。

【阶段间因果链 (下游参数由上游物理量函数式派生)】

  七个阶段在 main() 里若全靠手工接线, 就会退化成 20 多个写死的字面量
  (stage1_void(4), mera_init(8, 2), stage6_life(F=0.035, k=0.060) ...),
  阶段之间只传递给人看的诊断字符串, 没有任何物理量真正流向下游。
  最能说明问题的例子: 用 boundary_correlation_length() 算出关联长度
  xi = 19.314, 打印出来, 然后扔掉 —— 它从未进入 stage6。
  本脚本把八条连线全部换成真的派生 (推导式写进注释并打印实测值):

    L1  阶段一 -> 阶段二   局部维度 d_local=2        -> dim_full = d_local**L
       (v16·B1 起 dim_full **真被消费**; 此前它只被 print, 从不与 2**L 比对)
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
  L1. 真空等权叠加，基底即潜能。        - 完美张量 / 等权态 (幺半群单位; v16·B1 起
                                          被下游真消费 —— 重叠曲线见上方 B1)
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
  * 有限尺寸: 主链走 L=8/16 的环, MERA 用周期边界, 故 c 拟合用 CFT 的周期公式。
    (v15·W5 起标度扫描扩到 L=8…18; 注意杠杆臂 L 8->18 在全 log2 轴上只跨
     1.17 个 octave, 说 L 区间时"L 的比值"与"octave 数"要分开说。)
  * 曲率约定: Ollivier-Ricci 依赖随机游走的 idle 概率 alpha。本脚本对
    alpha=0 与 alpha=1/(d+1) 分别给出解析锚点并显式声明, 不引用记忆中的文献数字。
  * 阶段六: 反应参数 F, k 是**不可约的化学输入** (反应动力学, 与几何无关),
    因此显式标注为旋钮, 不假装从曲率派生。真正派生的是几何量 (域长 L、网格 N)。
  * 曲率对照: MERA 体几何的负曲率**不是**"比所有对照图更负"。三类零模型对照
    显示它落在对照分布之内 (百分位在 [5%, 95%] 区间), 故只能说"与随机图同量级",
    不能说"全息几何更负"。这条结论由对照数据算出, 不是写死的措辞。
  * 闭环: max_L 是旋钮且实测会触顶。去掉它 L 会一直涨 —— 这正是临界性,
    而不是闭环的失败。脚本把这个触顶如实打印出来。**触顶的成因不是算力上限**
    (v15·W5 实测更正): 每圈主结果走精确对角化, L=18 实测仅建 H 1.53s + eigsh
    3.86s + 433MB; 卡住 16 的是**末圈 MERA 交叉校验**要求 L 为 2 的幂 (quimb 库
    约束)。因此"闭环能否真自组织到临界"在本工作条件下**无法回答** —— 那是因为
    **存在**上限这件事本身, 不是因为 16 这个数, 更不是因为笔记本算不动。
  * 阶段七仍是概念模型: "照见"是联合不动点的结构性表达, 不是意识实证。
    状态塌缩到 x*=0 时 D* 不存在, 该支的熵比与等权度**一律是约定**
    (两者是同一件事的换写), 已如实标 defined=False, 不当作独立证据。
  * 阶段六不变量: Gray-Scott 的源/汇项不抵消, 所以 "u+v 质量守恒" 是**伪不变量**
    (首步漂移 6.25e-5, 是常被声称的 1e-6 容差的 62 倍)。真正成立的是两条离散
    恒等式: 周期域上拉普拉斯零和, 以及由更新前场精确预言的 ⟨u+v⟩ 增量平衡式。
  * 第三层参考侧种子: 归档的"多种子"其实同出一个硬编码初始条件, 其 bool 场与
    浮点场的任何简单二值化一致率都在 0.5 附近 (随机水平)。该缺口由 A 段诊断项
    如实登记为未通过, 不用它撑任何结论。
  * 中心荷的第二把尺子 (v15·W3b): 能谱族与纠缠谱族的**口径必须分开说**,
    见上文 W3b 那条的"口径警告"。一句话: 0.212% 不许写成"两族互相印证"。
  * L5 的涌现几何读数 (v15·W10 结论, 是**负面事实**): `mera_bulk` 实测
    **不依赖态** (seed 0/1/7 边集相同), 也**不依赖键维** (chi in {2,4,8} 边集
    逐位相同) ⇒ 该读数只是 (L, 二进制布局) 的**纯解析函数**。
    **线上没有上游派生的量在流动** —— 这与 L1 的缺口是**不同类型**: L1 是
    缺工具, L5 是这条候选路径本身不携带上游信息。故 L5 仍记 C。
  * v16·B1 的三条守卫 (主判据 / 推离 / 维数) **都不构成"L1 的物理结论已被独立
    验证"**: 它们量的是**接线通不通**。推离守卫尤其只标定**灵敏度尺度**。
    这层限制写在 `metric_l1_void_link` 与三条守卫的 note 里, 也在 v16_plan.md
    §2.4 的边界 (1) 明写。**L1 成熟度维持 B, 不升级。**
  * v16·B2 (cMERA) 的读数**不进本文件的任何判定**: 它自成一个可执行体, 与主
    流程零耦合。费米子 chi = 0.788697 与 0.5 的差别是**口径差**, 不是 bug。
  * v16·B3 的账本**只在与全流程同进程时有效**: 独立进程跑会退 3, 而退 3 的
    成因是两张**静态登记表**在空注册表下引用不到运行时对象、报出假失败
    (见下方"修直的口径"第 7 条) —— **不是**"空判"。

【修直的口径 (逐条留痕, 便于与历史数字对账) —— v15 四条 + v16 三条】
  1. max_L 归因: 源码注释曾把 L<=16 归因于精确对角化边界 —— 错。见上方诚实
     边界里那条 (真因是末圈 MERA 要求 L 为 2 的幂)。
  2. "种间 spread": 散点统计曾被误标为"种间散布", 实为同一条曲线上的点。
  3. "相互印证": 两条路径曾声称"相互印证", 实则为**同一次测量**的两种读法
     (即 0.212% 与 "ratio->8")。已改成"能谱族独立于纠缠谱族"。
  4. 台账完整性判据**判反**: 3 个账本条目写入的是"XX 对照已运行"这类**上报
     守卫**, 而它们在"该对照未执行"的提前返回分支里才注册 —— 于是完整性检查
     在"对照正常执行"(健康情形)下**失败**、在"对照缺失"时反而**通过**。
     首次全流程实测正是这 3 条被报成幻影引用。已改为只引用**对照本身**的守卫。
  5. `KNOBS['n_lambda']` 的注释曾自称"来自 xi/L 的尺度不变性", **v16·B3 实测
     证伪** (xi_over_L 扫描对 L6 的全部输出**零影响**) ⇒ 注释已改为"自由旋钮,
     **不由 L5 派生**"。**残留未清**: `derive_L6_grid` 的 docstring 里还留着
     同一句话 (它与注释是两处, 不是一处)。
  6. 负对照台账 L1 行: v16·B1 补 L1 那格时**改写了该行第 2~6 列** (`'缺'` ->
     `'已补'` 等), 其余 12 行逐元素未动、**列结构未变** (仍 6/7 元素参差 ⇒
     不加第 8 列)。故"未改 13 个冻结元组"这种说法**不成立**, 准确说法是
     "未改**列结构**; L1 行被改写"。
  7. `_v16_claim_ledger.py` 独立跑退 3 的成因**不是**"空注册表 ⇒ 四条检查退化
     成空判": 进判定的只有 (b)(c)(d)(e), **(a) 只报不判**; 空注册表下 (b)(d)
     确实空判通过, 但 `_SAME_SOURCE_REGISTRY` 与 `_NEG_CTRL_CLAIM` 是**静态
     字面量**, 引用不到运行时对象 ⇒ (c) 报 1 条、**(e) 报 8 条假失败** ⇒ 退 3。
     所以该脚本**必须**与全流程同进程使用。

【源码拆分 (v16) —— 本文件是门面, 实体在 `_model/`】
  实体已拆到 `v16/_model/` 的 9 个模块 (core / stage12 / stage3 / mera / stage5 /
  stage67 / checks / main / __init__); 指标层同理拆到 `v16/_metric/`。
  本文件与 `spiral_metric_v16.py` 只剩三件事: 版权头 + 本说明 + 一次性 re-export
  (外加文件末尾的门面赋值转发)。拆分前的单文件原件冻结在 `v16/_baseline/orig/`,
  作为**出处留痕**, 不是活的依赖; 另存 `_baseline/pkg_runA/` 一份按包布局的副本。
  **对外可见的名字与拆分前逐一相同** —— 拆分的验收判据就是这个。
  门面赋值转发: 拆分前 `spiral_model_v16.exact_ground_state = f` 一处赋值就改掉
  所有调用点; 拆分后同一个名字在 `_model/` 里有**多份**独立绑定 (exact_ground_state
  就有 5 份), 只改门面那份等于没改 —— 补丁被**静默吞掉**: 打桩看着成功、数值上
  只表现为 ~1e-15 的抖动。实测 `_v16_smoke_b1_void_limit.py` 的 memo 就是这么
  失效的 (6 条逐位断言假失败)。所以 `__setattr__` 把赋值同步写回每一份真正持有
  该名字的子模块。只在**赋值**时走这条路 —— 正常运行没有代码给门面属性赋值,
  而且 `from ... import` 走 STORE_NAME 直接写 __dict__, 不经过 __setattr__。

【依赖】
  numpy, scipy, matplotlib, networkx, quimb>=1.0, torch  (均为硬依赖, 不静默降级)
【运行】  cd v16 && python spiral_model_v16.py
            ⇒ 全流程 (实测 1879 s)。产物: result/spiral_v16_metrics.png (20 面板,
              由指标层 plot_metrics_v16 出) + result/spiral_v16.png (因果链总图)
              + result/_v16_data.json
          cd v16 && python _v16_run_all.py [--full | --only <名字>]
            ⇒ **唯一 runner** (分层与退出码见上; 跑冒烟只用它)
          (仓库根下 v14/ v15/ v16/ 并存, 各版本自包含; 需在版本目录内运行,
           它会 import 同目录的 spiral_metric_v16.py)

================================================================================
"""

import os
import sys
import json
import platform
import time
import warnings
import contextlib   # v15·W1: selfref_N_dependence 要吞掉被扫描调用的行内打印
import io           # v15·W1: 同上 (StringIO 承接)

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
from spiral_metric_v16 import (collect_metrics_v16,
                                    plot_metrics_v16, metrics_for_json)
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

# ---------- 这两个库是本模型的核心, 不做静默降级 ----------
import torch  # noqa: E402
import quimb.tensor as qtn  # noqa: E402

torch.set_default_dtype(torch.float64)


# ============================================================================
# 拆分后的 re-export: 对外 API 与拆分前完全一致
# ============================================================================
from _model.core import (
    KNOBS, CHAIN, record_link, print_chain_report, tfi_periodic_sparse, exact_ground_state, jw_ground_energy, entanglement_curve, fit_central_charge, derive_L1_local_to_full, derive_L1_void_to_chain, l1_void_pushaway, derive_L2_entanglement_gap, derive_L3_schmidt_rank, derive_L4_bond_dimension, derive_L5_forman_from_degree, derive_L6_grid, boundary_correlation_graph, boundary_correlation_length, _fit_length, _HERE, _OUTPUT_DIR,
)
from _model.stage12 import (
    HJ_SCAN_DEFAULT, HJ_CRIT, HJ_S_WINDOW, HJ_DISCRIM, HJ_DISCRIM_MIN, stage1_void, stage2_critical_break, hj_scan, plot_hj_scan,
)
from _model.stage3 import (
    _power_iteration, _nonnormality, _transient_growth, selfref_general_matrix, selfref_nonconvergent, _lyapunov, selfref_neural_network, selfref_coupled, selfref_N_dependence,
)
from _model.mera import (
    V13_STEPS, V13_LR_SEEDS, V13_EVERY, V13_CHI_CTRL, mera_init, _qr_isometry, mera_dense, mera_isometry_check, mera_fit, mera_fit_v13, v13_tier1_runs, _schmidt_floor, mera_energy_only, mera_causal_cone, build_mera_graph, mera_bulk_invariance,
)
from _model.stage5 import (
    all_pairs_distance, ollivier_ricci, forman_ricci, validate_curvature, curvature_report, geometry_controls,
)
from _model.stage67 import (
    spiral_loop, stage6_life, gray_scott_invariants, stage7_consciousness,
)
from _model.checks import (
    F1_L, F1_CHI, F1_SEEDS, F1_C_TRUE, F1_REL_MAX, RENYI_ALPHAS, RENYI_NS, RENYI_STD_MAX, RENYI_DEV_TRUE_MAX, tfi_z2_twisted_sparse, _schmidt_probs, renyi_alpha_scan, cross_boundary_twist, area_vs_log_law, finite_size_scaling, spectral_central_charge, central_charge_verification, consistency_checks,
)
from _model.main import (
    main, plot_all,
)


if __name__ == '__main__':
    main()


# ---------------------------------------------------------------------------
# 门面赋值转发 —— 恢复"拆分前一个模块"的可打桩语义
#
# 拆分前整个实现是一个模块, `spiral_model_v16.exact_ground_state = f` 一处赋值
# 就改掉了所有调用点。拆分后同一个名字在 _model/ 里有**多份**独立绑定
# (exact_ground_state 就是 5 份: checks / core / main / stage12 / stage67 各
# `from .core import` 一份), 门面这份没有任何人调用 —— 于是补丁被**静默吞掉**:
# 打桩看着成功了, 实际没生效, 数值上只表现为 ~1e-15 的抖动。实测
# _v16_smoke_b1_void_limit.py 的 memo 就是这么失效的 (6 条逐位断言假失败)。
# 静默失效比直接报错难查得多, 所以这里把赋值同步写回每一份真正持有该名字的
# 子模块, 恢复拆分前的语义。
#
# 只在**赋值**时才走这条路: 正常运行没有任何代码给门面属性赋值, 而且
# `from ... import` 走 STORE_NAME 直接写 __dict__, 根本不经过 __setattr__ ——
# 所以对主流程惰性, 不影响已验证的数值结果。
# ---------------------------------------------------------------------------
_ModType = __import__('types').ModuleType   # 刻意不写 `import types`: 那会往门面的
                                            # 公开命名空间里多塞一个 `types`, 而门面
                                            # 的契约是"对外可见的名字与拆分前逐一
                                            # 相同"。


class _ForwardingFacade(_ModType):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        for _n, _m in list(__import__('sys').modules.items()):
            if _n == '_model' or _n.startswith('_model.'):
                if name in vars(_m):
                    setattr(_m, name, value)


__import__('sys').modules[__name__].__class__ = _ForwardingFacade
