## v16.10 新增：L2→CFT 显式算符对应（理论标注）

---

>**声明**：本节为从离散哈密顿量到连续 CFT 的算符对应标注，基于 Calabrese & Cardy (2004)、Di Francesco et al. (1997) 等已知理论，**非本仓数值拟合产物**。F1/W3b 等同族变体不包装为"显式映射"。

---

### 2.1 离散端：哈密顿量

$$
H_{\text{Ising}} = -J\sum_{i=1}^{L} \sigma_i^z \sigma_{i+1}^z - h\sum_{i=1}^{L} \sigma_i^x, \quad h/J = 1,\; L \leq 16,\; \text{PBC}
$$

临界点 $h/J=1$ 对应无扰动 Ising CFT（$c=1/2$）。

---

### 2.2 连续端：Ising CFT 初级算符

| 初级算符 | 共形权重 $(h,\bar{h})$ | 维度 $\Delta = h+\bar{h}$ | 物理角色 |
|---|---|---|---|
| $\mathbb{1}$ | $(0,0)$ | 0 | 恒等/真空 |
| $\varepsilon$ | $(1/2, 1/2)$ | 1 | 能量密度 |
| $\sigma$ | $(1/16, 1/16)$ | $1/8$ | 自旋/序参量 |

---

### 2.3 显式算符对应（核心表）

| 离散算符（格点） | → | CFT 初级算符 | 备注 |
|---|---|---|---|
| $\sigma_i^z \sigma_{i+1}^z$ | ↔ | $\varepsilon(x)$ | 能量密度，Jordan-Wigner 后对应费米子双线性 $\sim i\psi\partial\psi$ |
| $\sigma_i^x$ | ↔ | $\sigma(x)$ | 自旋翻转，非局域费米子弦算符 $e^{i\pi\sum_{j<i}n_j}\psi_i$ |
| $\sigma_i^z$（单点） | ↔ | 费米场 $\psi(x), \bar{\psi}(x)$ | 经 Jordan-Wigner：$\sigma_i^z \sim i\psi_i \bar{\psi}_i$ |
| 恒等 | ↔ | $\mathbb{1}$ | — |

**映射路径**：$H_{\text{Ising}} \xrightarrow{\text{JW}} \text{自由马约拉纳费米子} \xrightarrow{\text{低能}} \text{连续费米场} \xrightarrow{\text{玻色化}} \text{Ising CFT}$。

---

### 2.4 纠缠谱与初级算符的对应

基态约化密度矩阵本征值 $\lambda_n$ 与 CFT 共形权重的关系（Calabrese-Cardy）：

$$
\lambda_n \sim \exp\left(-\frac{2\pi v}{L} (h_n + \bar{h}_n) \cdot \ell_{\text{eff}}\right)
$$

其中 $h_n, \bar{h}_n$ 为初级或 descendant 的权重，$\ell_{\text{eff}} = \frac{L}{\pi}\sin\frac{\pi \ell}{L}$。

**显式标识**（对 $L \leq 16$ 的基态）：

| 纠缠谱能级（按 $\lambda$ 降序） | 对应 CFT 态 | $(h,\bar{h})$ |
|---|---|---|
| 第 1 支 | $\|\mathbb{1}\rangle$ | $(0,0)$ |
| 第 2 支 | $\|\sigma\rangle$（简并） | $(1/16,1/16)$ |
| 第 3 支 | $\|\varepsilon\rangle$ | $(1/2,1/2)$ |
| 更高支 | descendant |$\|\partial\sigma\rangle, \|\partial\varepsilon\rangle$ 等， $(h+1,\bar{h})$ 等 |

---

### 2.5 引用 Calabrese-Cardy 公式

纠缠熵的 CFT 预测（PBC，子区域长度 $\ell$）：

$$
S(\ell) = \frac{c}{3}\log\left(\frac{L}{\pi}\sin\frac{\pi\ell}{L}\right) + \log g
$$

对 Ising CFT：$c = 1/2$，$g = 1$（无拓扑简并）。此公式用于**校验**离散基态纠缠熵是否落在 CFT 预期带内，但 **v16.10 不做新拟合**，仅标注此对应关系。

**引用文档**：Calabrese & Cardy, JSTAT 2004 P06002; Di Francesco, Mathieu & Sénéchal, CFT 1997 Ch.7.

---

### 代码注释落点 

详见 [`_model/cft_mapping.py`](_model/cft_mapping.py) 顶部注释块。





