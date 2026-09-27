# -*- coding: utf-8 -*-
"""
v16.4 · D8 (β-L4) 冒烟 —— **非酉注入**负对照 (隔离路径)
[_VERSION_TAG = 'v16-smoke-beta-l4-nonunitary-1']

**这是 v16.4_plan.md §3.1 的 D8**。**不进 runner**（DP-β-1 默认 = 不执行「计入诊断
守卫」）⇒ **不加守卫、不加指标、不动分母 78/27/13**，不改任何登记值。

---

## 要回答的问题

L4 台账上**只有一条**负对照：W12 键维**向下**推离（`_metric/main.py:234-250`）。
`negctrl.py` 里**没有任何**破坏酉性/等距性的控制（计划 §1.4 逐条核过）。
本冒烟补的就是这块空白：**把某个解纠缠器 U 换成 U' = U + ε·N（N 随机实矩阵，
不做 QR 重投影），等距残差必然上升 —— 那么下游读数会不会跟着退化？**

## 硬约束：隔离路径（这是本脚本存在的**唯一**理由）

最接近「酉性」的既有检查是 `metric_mera_consistency`（`_metric/solvers.py:217-242`），
它是**指标 1.4**，方向**相反** —— 它**断言** `err < 1e-12`（`solvers.py:230` 逐字
`ok = bool(err < 1e-12 and lin > 0.95 and lin > lg)`）。**在主流程注入非酉扰动会把它
翻红 ⇒ 指标读数变化 ⇒ 违反冻结。**

⇒ 本脚本**自建一份 MERA**，**绝不**接进 `stage2_critical_break` / `mera_fit` /
`v13_tier1_runs` 的任何调用点。它**不 import 任何生产入口**（不 import
`spiral_model_v16` / `spiral_metric_v16`），只从 `_model.mera` / `_model.core` 取
**构造与度量函数**。可核检查（计划 §9 风险 3）：本脚本跑完后全流程日志里
`MERA 一致性` 必须与 v16.3 逐字相同 —— 因为它根本没碰那条路径。

⚠️ **与历史上被删掉的候选 `"L4 每步优化后的酉性/等距性残差"`
（`_model/checks.py:822-825`）不是同一件事**（计划 §1.4 要求显式写明）：
那条被删的理由是「双重恒真 —— 进优化器的是 `raws` 参数，MERA 张量对象从未被优化触碰；
就算搬进循环，QR 参数化也按构造给出等距性」。**该理由不适用于外部注入** ——
本脚本注入的是**张量数据本身**，QR 参数化**看不到**它，等距性**不再**由构造成立。
这正是它能成为一条真检验的原因。

---

## 预登记判据（**先写死，再看数**；计划 §3.1 的可执行转录）

| 落支 | 条件 | 含义 |
|---|---|---|
| **N1 有检验力** | 存在 ε 使 `err > 1e-12`，**且**下游读数退化 | 负对照成立 |
| **N2 无检验力** | `err` 升上去了但下游读数**不变** | 如实记「下游对等距性不敏感」—— 这是对 L4 唯一那条负对照的**边界补充**，**不是成功**，不得包装 |
| **N0 装置失效** | **没有任何** ε 使 `err > 1e-12` | **计划未列，故在此先于数据写死。** 含义是**注入没生效**（装置坏了），既不是 N1 也不是 N2，**不许**归到 N2 |

**「下游读数退化」的可执行定义（计划未给，故在此先于数据写死）**：
`eps_c(ε) > 1.5 × eps_c(0)`。
**1.5 这个数不是新发明的** —— 它沿用本仓库既有的退化判据口径
（`_metric/main.py:229` 逐字 `(判据 > 1.5x)`）。定在这里是为了**防止结果出来之后再挑阈值**。

**判据只挂在 `eps_c` 上**（计划 §3.1 给的是「`eps_c` 或 `R²_linear`」二选一）。
`R²_linear` 与 `overlap` 一律**照登在表里**，但**不参与判定** —— 理由见下方「口径声明」。

## 口径声明（必须与结果同页）

1. **`R²_linear` 对张量数据免疫**：`mera_causal_cone`（`_model/mera.py:404-421`）
   数的是 `mera.select_any([f'I{j}'...]).num_tensors` —— **纯拓扑计数**，
   不读任何张量数值。故它对**一切** ε 必然给出**逐位相同**的读数。
   本脚本**实测**这一点并照登，**不靠推断**。⇒ 它在计划给的两个候选里
   **结构性地不可能**成为灵敏读数。这不是本冒烟的发现，是这条读数的定义。
2. **`eps_c(0)` 的绝对值不可与 v16.3 登记值比**：生产用 `V13_STEPS=1500`
   × 5 条轨迹 + 检查点选择（`_model/mera.py:193-204`），本冒烟用 `FIT_STEPS`
   单条轨迹。**本脚本只报「同一个预算下 ε=0 与 ε>0 的比」，不报绝对精度**，
   也**不宣称**复现了 1.1 的任何登记值。
3. **注入位置是选择，不是派生**：本脚本注入**第 0 个**解纠缠器（`ndim==4` 的张量里
   下标 0），写死在这里。不事后换位置。
4. **N 是随机实矩阵**（`np.random.default_rng(SEED_N)`，固定种子），与计划 §3.1 一致。

## 许可证：为什么本脚本的收缩可信

`mera_dense`（`_model/mera.py:87-110`）从 `raws` 经 `_qr_isometry` **重投影**后收缩，
所以它对张量数据**免疫** —— 注入写进张量也不影响它。本脚本因此必须自备一条
「直接吃张量数据」的收缩（`contract_with_data`）。**它不被信任，除非 §A3 证明它与
生产 `mera_dense` 在 ε=0 时逐位一致。** A3 不过 ⇒ 全脚本读数作废（退 3）。

## 退出码（沿用 v16.2/v16.3 冒烟惯例）

`0 = 自校验通过`（**注意：这不代表 N1**，落支可能是 N2 —— 那是**结论**不是失败）；
`3 = 自校验失败`（A 段任一条不过，或落 N0）⇒ **结论作废**；
`2 = 脚本自身错误`。
⚠️ `3` 是「预期内的失败路径」，**不是 runner 意义的回归**（memory: `v16-single-runner`）。

用法:
    python v16/_v16_smoke_beta_l4_nonunitary.py
"""

import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'v16'))

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402

# ⚠️ **隔离的代价, 必须自己复刻**: 生产门面 `spiral_model_v16.py:472` 有一行
#     `torch.set_default_dtype(torch.float64)` —— 那是**进程级全局**。
#     本脚本刻意不 import 门面, 于是 `torch.randn` 退回 float32, 而 quimb
#     `MERA.rand(dtype=float)` 给的是 float64 张量 ⇒ cotengra 的 torch 后端
#     报 `both inputs should have same dtype`。
#     **不写这一行, 本冒烟根本跑不起来**; 写了之后 dtype 才与生产一致。
#     先例: `_v16_smoke_cmera.py:36` 是同一个动作。
torch.set_default_dtype(torch.float64)

from _model.core import exact_ground_state, fit_central_charge   # noqa: E402
from _model.mera import (                                     # noqa: E402
    _qr_isometry, mera_causal_cone, mera_dense, mera_fit_v13,
    mera_init, mera_isometry_check,
)

_VERSION_TAG = 'v16-smoke-beta-l4-nonunitary-1'

# --- 配置 -----------------------------------------------------------------------
L = 8            # 收缩成稠密态, 必须小; 生产用 L=16 (KNOBS['L_chain'])
CHI = 4          # 与生产 chi_dev 同 (L4 派生: _model/main.py:129)
SEED = 0         # mera_init 的种子 (固定)
FIT_STEPS = 800  # **不是**生产值 (生产 V13_STEPS=1500); 本冒烟单条轨迹, 见口径 2
SEED_N = 20260927   # 注入噪声 N 的种子 (固定)

# --- 预登记的 ε 网格 (先写死, 不看结果调) ---------------------------------------
EPS_GRID = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1)

ERR_FLOOR = 1e-12       # 计划 §3.1 的 err 门槛 (与指标 1.4 的 solvers.py:230 同值)
DEGRADE_RATIO = 1.5     # 「退化」阈值; 沿用 _metric/main.py:229 的 > 1.5x 口径
INJECT_WHICH = 0        # 注入第几个解纠缠器 (ndim==4 的下标); 见口径 3

_FREEZE_TAG = '=' * 76

_FAILS = []


def check(label, cond, detail=''):
    tag = 'PASS' if cond else '**FAIL**'
    print(f"    [{tag}] {label}" + (f"  —— {detail}" if detail else ''))
    if not cond:
        _FAILS.append(label)


# ---------------------------------------------------------------------------
# 就地物化 + 直接吃张量数据的收缩。**这两件是注入能生效的唯一原因**。
# ---------------------------------------------------------------------------
def materialize(mera, raws):
    """把 QR 投影后的参数**就地**写进 mera 的每个张量。

    与 `mera_dense`（`_model/mera.py:95-105`）内部的写法**逐字相同**,
    区别只有一个: 写进 `mera` **本身**而非 `mera.copy()` —— 这样后面的注入与
    `mera_isometry_check` 才看得到同一批数据。
    """
    with torch.no_grad():
        k = 0
        for t in mera.tensors:
            if t.ndim == 4:
                t.modify(data=_qr_isometry(raws[k]).reshape(t.shape))
                k += 1
            elif t.ndim == 3:
                t.modify(data=_qr_isometry(raws[k], t.shape[2]).reshape(t.shape))
                k += 1
            else:                                   # 顶层固定的边界张量
                t.modify(data=torch.as_tensor(np.asarray(t.data)))
    return mera


def contract_with_data(mera, L):
    """与 `mera_dense` 的**收缩段**（`_model/mera.py:106-110`）逐字相同,
    但直接吃 `mera` 里已就地的张量数据, **不再从 raws 重投影**。

    合法性**不靠**「看着像」—— 由 §A3 与真函数 `mera_dense` 逐位比对发许可证。
    """
    tn = mera.copy()
    out = tn.contract(all, optimize='greedy',      # noqa: F821  (内置 all, 与 mera.py 同)
                      output_inds=[f'k{i}' for i in range(L)])
    if hasattr(out, 'data'):
        out = out.data
    with torch.no_grad():
        v = out.reshape(-1)
        v = v.detach() if hasattr(v, 'detach') else v
        return np.asarray(v, dtype=float)


def inject(mera, eps, N, U=None, which=INJECT_WHICH):
    """把第 `which` 个解纠缠器换成 `U + eps*N`, **不做 QR 重投影**。

    ⚠️ **`U` 必须显式传入才算「单点 ε」**。省略 `U` 时以**当前**张量数据为基准
    （并把那个基准作为第二个返回值交出）—— 若在同一张量上连着调用, 读到的
    **已经是被污染的那个**, 扰动会**累积**成 `U_orig + (eps_prev+eps)·N`。
    本脚本首轮就是这样错的 (由 §A4 的复位检查抓出来, 不是自己发现的)。
    正确用法: **基准只取一次**, 之后每个 ε 都把它传进来。
    """
    t = [x for x in mera.tensors if x.ndim == 4][which]
    with torch.no_grad():
        if U is None:
            U = t.data.clone()
        t.modify(data=U + eps * torch.as_tensor(N))
    return t, U


def err_of(mera):
    """等距残差 (与指标 1.4 的 `max(unitary_err, isometry_err)` 同一个量)。"""
    iso = mera_isometry_check(mera, L)
    return float(max(iso['unitary_err'], iso['isometry_err']))


def readout(mera, c_exact):
    """下游读数。只算, 不判定 —— 判定在 §C。"""
    psi = contract_with_data(mera, L)
    psi = psi / np.linalg.norm(psi)
    fm = fit_central_charge(psi, L)
    _, _, r2 = mera_causal_cone(mera, L)
    return {'err': err_of(mera),
            'c': float(fm['c']),
            'eps_c': float(abs(fm['c'] - c_exact) / c_exact),
            'rms': float(fm['rms']),
            'r2_linear': float(r2['linear']),
            'psi': psi}


def main():
    t0 = time.time()
    print(f"=== {_VERSION_TAG} · D8 (β-L4) 非酉注入负对照, L={L}, chi={CHI} ===")
    print(f"    口径: 自建隔离 MERA (**不 import 任何生产入口**), 不进 runner,")
    print(f"          不加守卫, 不动分母 78/27/13, 不改任何登记值 (DP-β-1)。")

    # =====================================================================
    print(f"\n--- 0 · 隔离与配置回显 ---")
    print(f"    L={L}, chi={CHI}, seed={SEED}, FIT_STEPS={FIT_STEPS} "
          f"(生产 V13_STEPS=1500, **本冒烟不比绝对值**, 见口径 2)")
    print(f"    ε 网格 (先写死) = {list(EPS_GRID)}")
    print(f"    注入对象 = 第 {INJECT_WHICH} 个解纠缠器 (ndim==4); 噪声 seed = {SEED_N}")
    print(f"    判据: err > {ERR_FLOOR:g}  **且**  eps_c(ε) > {DEGRADE_RATIO} x eps_c(0)")
    print(f"    隔离声明: 本脚本不 import spiral_model_v16 / spiral_metric_v16,")
    print(f"              不调用 stage2_critical_break / mera_fit / v13_tier1_runs。")

    # =====================================================================
    print(f"\n--- A · 自校验 (全部先过, 否则结论作废) ---")
    tA = time.time()
    E0, gs = exact_ground_state(L)
    c_exact = float(fit_central_charge(gs, L)['c'])
    print(f"    A0 精确基态: L={L}, E0/L = {E0 / L:+.9f}, "
          f"c_exact(同 L 拟合) = {c_exact:.6f}")

    mera, raws = mera_init(L, CHI, seed=SEED)
    e_init = err_of(mera)
    iso_raw = mera_isometry_check(mera, L)
    check("mera_init 的初始张量已是等距/酉 (QR 构造保证, 机器精度)",
          e_init < 1e-12,
          f"err = {e_init:.2e} ({iso_raw['n_unitary']} 酉 + "
          f"{iso_raw['n_isometry']} 等距 + 1 顶层边界)")

    print(f"    ... 拟合 {FIT_STEPS} 步 (单条轨迹)")
    tf = time.time()
    psi_fit, ov_fit, cstep, _ = mera_fit_v13(
        mera, raws, gs, L, steps=FIT_STEPS, c_exact=None)
    t_fit = time.time() - tf
    fm_fit = fit_central_charge(psi_fit, L)
    eps_fit = abs(fm_fit['c'] - c_exact) / c_exact
    print(f"    A1 拟合完成: overlap = {ov_fit:.6f} @step{cstep}, "
          f"eps_c(拟合态) = {eps_fit:.6f}")
    print(f"       **用时 {t_fit:.1f} s** (只报脚本自己的计时, 不报 CPU 时间)")

    materialize(mera, raws)
    e_mat = err_of(mera)
    check("物化 (QR 投影写进张量) 后仍等距/酉 —— 物化本身零扰动",
          e_mat < 1e-12, f"err = {e_mat:.2e}")

    # 锁住上面那条 dtype 失效模式: 全部张量必须同 dtype, 否则 cotengra 的 torch
    # 后端会在收缩时报 `both inputs should have same dtype` (本脚本首轮就是这样崩的)。
    _dts = {np.asarray(t.data).dtype for t in mera.tensors}
    check("全部张量同 dtype (float64) —— 锁住「隔离不复刻默认 dtype」这个失效模式",
          _dts == {np.dtype('float64')},
          f"dtype 集合 = {sorted(str(d) for d in _dts)}, "
          f"torch 默认 = {torch.get_default_dtype()}")

    # A3 是本脚本的**许可证**: 自备收缩必须在 ε=0 时与生产 mera_dense 逐位一致。
    psi_mine = contract_with_data(mera, L)
    psi_mine = psi_mine / np.linalg.norm(psi_mine)
    psi_prod = mera_dense(mera, raws, L)
    with torch.no_grad():
        psi_prod = np.asarray(psi_prod.detach(), dtype=float)
    psi_prod = psi_prod / np.linalg.norm(psi_prod)
    d_prod = float(np.abs(psi_mine - psi_prod).max())
    print(f"\n    A3 自备收缩 vs 生产 mera_dense (ε=0, 同一个 MERA):")
    print(f"       自备 contract_with_data  max|Δψ| = {d_prod:.3e}")
    check("自备收缩与生产 mera_dense 逐位一致 (< 1e-12) —— 注入实验的许可证",
          d_prod < 1e-12, f"max|Δψ| = {d_prod:.3e}")

    # A3b: 拟合态本身也要在同一条路径上对账 (上一比用的是未拟合的 raws)
    psi_mine_f = contract_with_data(mera, L)     # materialize 时 raws 已拟合
    d_fit = float(np.abs(psi_mine_f / np.linalg.norm(psi_mine_f)
                         - (lambda p: p / np.linalg.norm(p))(
                             np.asarray(mera_dense(mera, raws, L).detach(), dtype=float))
                         ).max())
    check("拟合态上两条路径同样逐位一致 (同一张量的两处对账)",
          d_fit < 1e-12, f"max|Δψ| = {d_fit:.3e}")

    # A4 活的反事实: 证明「err 这把尺子」对非酉扰动真的有分辨力。
    rng = np.random.default_rng(SEED_N)
    t_first = [x for x in mera.tensors if x.ndim == 4][INJECT_WHICH]
    N = rng.standard_normal(tuple(t_first.shape))
    # **基准只取这一次**; 后面每个 ε (含复位) 都显式传它 ⇒ 每个点是单点 ε。
    _, U_base = inject(mera, 0.0, N)
    inject(mera, 1e-1, N, U_base)
    e_bad = err_of(mera)
    check("**活的反事实**: ε=0.1 的注入使 err 立刻 >> 1e-12 (这把尺子有分辨力)",
          e_bad > 1e-6, f"err(0.1) = {e_bad:.3e}")
    inject(mera, 0.0, N, U_base)             # 复位: 从**基准**重算, 不是从当前值
    e_reset = err_of(mera)
    check("复位后 err 回到机器精度 (ε=0 从基准重算是恒等操作)", e_reset < 1e-12,
          f"err(0) = {e_reset:.2e}")

    # A5 锁住上面那个**累积**失效模式: 从同一个基准连做两次不同的 ε, 第二次的结果
    # 必须与「单独做第二次」逐位相同。不锁的话, ε 扫描会把累积量当成单点 ε。
    inject(mera, 1e-3, N, U_base)
    e_seq = err_of(mera)
    inject(mera, 1e-2, N, U_base)            # 中间隔一次不同 ε, 再回到 1e-2
    e_again = err_of(mera)
    inject(mera, 1e-3, N, U_base)
    e_single = err_of(mera)
    check("同一基准下重做同一个 ε 给同一个 err —— 锁住「扰动累积」失效模式",
          abs(e_single - e_seq) < 1e-15 * max(1.0, e_seq),
          f"err(1e-3) 单做 = {e_seq:.17e}, 隔一次 1e-2 后重做 = {e_single:.17e}")
    inject(mera, 0.0, N, U_base)

    if _FAILS:
        print(f"\n    **A 段自校验失败 => 按纪律先查装置, 不进入 ε 扫描**")
        print(f"    失败项: {_FAILS}")
        return 3
    print(f"    A 段用时 {time.time() - tA:.2f} s (其中拟合 {t_fit:.1f} s)")

    # =====================================================================
    print(f"\n{_FREEZE_TAG}")
    print(f"  ***** 网格冻结线 —— 以下开始 ε 扫描, 网格自此冻结 (v16.4_plan §3.1) *****")
    print(f"  冻结的网格: ε = {list(EPS_GRID)}")
    print(f"  看过读数之后再改网格 = p-hacking。")
    print(f"{_FREEZE_TAG}")

    print(f"\n--- B · ε 扫描 (每个点都从**同一个基准 U_base** 重算, 不累积) ---")
    rows = []
    for eps in EPS_GRID:
        inject(mera, eps, N, U_base)
        r = readout(mera, c_exact)
        r['eps'] = eps
        r['overlap'] = float(abs(np.vdot(gs, r['psi'])))
        rows.append(r)
    inject(mera, 0.0, N, U_base)

    eps0 = rows[0]['eps_c']
    print(f"    eps_c(ε=0) = {eps0:.6f}  ← 本预算下的**基准** (绝对值不可比生产, 见口径 2)")
    print(f"    退化阈值 = {DEGRADE_RATIO} x {eps0:.6f} = {DEGRADE_RATIO * eps0:.6f}")
    print()
    print(f"    {'eps':>8s} | {'err':>10s} {'>1e-12':>7s} | {'eps_c':>10s} {'比值':>8s} "
          f"{'退化':>5s} | {'overlap':>9s} | {'R^2_linear':>11s}")
    for r in rows:
        over = r['err'] > ERR_FLOOR
        deg = r['eps_c'] > DEGRADE_RATIO * eps0
        print(f"    {r['eps']:8.0e} | {r['err']:10.2e} {'是' if over else '否':>7s} | "
              f"{r['eps_c']:10.6f} {r['eps_c'] / eps0:8.3f} {'是' if deg else '否':>5s} | "
              f"{r['overlap']:9.6f} | {r['r2_linear']:11.6f}")

    # R²_linear 的「对数据免疫」是**实测**出来的, 不是推断出来的 (口径 1)
    r2s = sorted({r['r2_linear'] for r in rows})
    print(f"\n    口径 1 的实测核验: R²_linear 在 {len(rows)} 个 ε 上取值集合 = {r2s}")
    check("R²_linear 对张量数据**逐位免疫** (纯拓扑计数, 见口径 1) —— "
          "故它不可能成为灵敏读数",
          len(r2s) == 1, f"取值个数 = {len(r2s)}")

    # =====================================================================
    print(f"\n--- C · 落支判定 (N1 / N2 / N0) ---")
    over_eps = [r for r in rows if r['err'] > ERR_FLOOR]
    deg_eps = [r for r in over_eps if r['eps_c'] > DEGRADE_RATIO * eps0]
    print(f"    err > {ERR_FLOOR:g} 的 ε          = {[r['eps'] for r in over_eps]}")
    print(f"    其中 eps_c 退化的 ε  = {[r['eps'] for r in deg_eps]}")

    if not over_eps:
        print(f"\n    => **落支 N0 (装置失效)**: 没有任何 ε 使 err > {ERR_FLOOR:g}。")
        print(f"       含义是**注入没生效** —— 既不是 N1 也不是 N2, **不许归到 N2**。")
        print(f"       A4 已证这把尺子有分辨力, 故 N0 只可能来自注入路径本身出错。")
        branch = 'N0'
    elif deg_eps:
        w = deg_eps[-1]
        print(f"\n    => **落支 N1 (有检验力)**: 负对照成立。")
        print(f"       最小退化 ε = {deg_eps[0]['eps']:g} "
              f"(err = {deg_eps[0]['err']:.2e}, "
              f"eps_c = {deg_eps[0]['eps_c']:.6f} = "
              f"{deg_eps[0]['eps_c'] / eps0:.2f}x 基准)")
        print(f"       ε = {w['eps']:g} 时 eps_c = {w['eps_c']:.6f} "
              f"({w['eps_c'] / eps0:.2f}x), overlap 从 "
              f"{rows[0]['overlap']:.6f} 掉到 {w['overlap']:.6f}")
        print(f"       ⚠️ **边界**: N1 说的是「注入能被这套读数看见」,**不是**")
        print(f"          「L4 的物理主张被验证」; 且下游退化只在**本预算**的 MERA 上测的。")
        branch = 'N1'
    else:
        emax = max(r['err'] for r in rows)
        pmax = max(r['eps_c'] for r in rows)
        print(f"\n    => **落支 N2 (无检验力)**: err 升上去了但下游读数不变。")
        print(f"       最大 err = {emax:.2e}, 而 eps_c 最大 {pmax:.6f} "
              f"= {pmax / eps0:.3f}x 基准 (< {DEGRADE_RATIO}x 阈值)")
        print(f"       ⇒ 如实记「**下游对等距性不敏感**」—— 这是对 L4 唯一那条负对照的")
        print(f"         **边界补充**, **不是成功**, 不得包装。")
        branch = 'N2'

    # ---------------- 天花板 (与结果同页, 不许省) ----------------
    print(f"\n--- 天花板 (与结果同页, 不许省) ---")
    print(f"    1. 本冒烟**不改指标 1.4** (`solvers.py:230` 的 `err < 1e-12` 一个字符不动):")
    print(f"       它走的是**隔离副本**, 生产路径一次都没被调用。可核检查 = 全流程日志里")
    print(f"       `MERA 一致性` 与 v16.3 逐字相同 (计划 §9 风险 3)。")
    print(f"    2. **绝对精度不可比**: 单条轨迹 x {FIT_STEPS} 步 vs 生产 1500 步 x 5 轨迹")
    print(f"       + 检查点选择。本脚本只报**同一预算内的比值** (口径 2)。")
    print(f"    3. **负对照成立 != 升级**: 依 `memory/l1-maturity-held-at-b.md` 逐字")
    print(f"       「负对照缺口补上了 **≠** 该升级」, 与 `README.md` 对 L4 逐字")
    print(f"       「故 L4 成熟度仍记 C, 不因补了一条对照而升级」⇒ **L4 仍记 C**。")
    print(f"    4. 本脚本**未加任何守卫/指标条目**, 分母 78 / 27 / 13 不变 (DP-β-1)。")
    print(f"    5. A3/A3b 只证了「ε=0 时两条收缩路径一致」;**ε>0 时生产 `mera_dense`")
    print(f"       给不出同一个态** (它从 raws 重投影, 看不见注入) ⇒ 本脚本**无法**用")
    print(f"       生产函数交叉核验 ε>0 的态, 只能靠 A3 的连续性外推。这条限制照登。")

    ok = (not _FAILS) and branch != 'N0'
    print(f"\n=== 结果: 落支 {branch}; 自校验 {'通过' if not _FAILS else '**失败**'}; "
          f"累计用时 {time.time() - t0:.1f} s ===")
    if _FAILS:
        print(f"    失败项: {_FAILS}")
    return 0 if ok else 3


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    try:
        sys.exit(main())
    except Exception as exc:                      # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"\n**脚本自身错误**: {type(exc).__name__}: {exc}")
        sys.exit(2)
