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
L2 → CFT 显式算符对应（v16.10 理论标注）
========================================

本模块/文件仅作理论标注，不包含任何数值拟合代码。
所有对应关系基于已发表文献，非本仓独立发现。

1. 离散哈密顿量
---------------
H_Ising = -J ∑ σ_i^z σ_{i+1}^z - h ∑ σ_i^x,  h/J = 1, L ≤ 16, PBC
临界点对应无扰动 Ising CFT (c = 1/2)。

2. 映射路径
-----------
H_Ising  --(Jordan-Wigner)--> 自由马约拉纳费米子
         --(低能极限)--> 连续费米场
         --(玻色化)--> Ising CFT

3. 算符对应表
-------------
离散算符                     →  CFT 初级算符       权重 (h, hbar)   物理角色
σ_i^z σ_{i+1}^z              ↔  ε(x)              (1/2, 1/2)       能量密度
σ_i^x                        ↔  σ(x)              (1/16, 1/16)     自旋/序参量
σ_i^z (单点)                 ↔  ψ(x), ψ̄(x)        -                费米场
恒等                         ↔  𝟙                 (0, 0)           真空

4. 纠缠谱与初级算符的对应
-------------------------
基态约化密度矩阵本征值 λ_n 与 CFT 权重的关系（Calabrese-Cardy 公式）：

    λ_n ∼ exp[ -(2πv/L) (h_n + hbar_n) · ℓ_eff ]
    其中 ℓ_eff = (L/π) sin(πℓ/L)

能级标识（按 λ 降序）：
    第1支 → |𝟙⟩,          (0, 0)
    第2支 → |σ⟩ (简并),   (1/16, 1/16)
    第3支 → |ε⟩,          (1/2, 1/2)
    更高支 → descendant (∂σ, ∂ε, ...)

5. 纠缠熵校验公式（仅标注，不跑新拟合）
----------------------------------------
S(ℓ) = (c/3) log[(L/π) sin(πℓ/L)] + log g
Ising CFT: c = 1/2, g = 1.

6. 口径红线
-----------
- 本节是"从离散哈密顿量到连续 CFT 的算符对应"标注。
- F1/W3b 是同族/同次测量的变体，不包装为显式映射。
- 不声称本仓首次建立此对应，不跑新 DMRG/MERA 计算。

参考文献
--------
[1] Calabrese & Cardy, JSTAT 2004 P06002.
[2] Di Francesco, Mathieu & Sénéchal, Conformal Field Theory, 1997, Ch.7.
"""