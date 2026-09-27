# -*- coding: utf-8 -*-
"""
v16.3 · S3 冒烟 —— L4: 临界点上把 `I3` 接到 L 扫描 (方向 D2)
[_VERSION_TAG = 'v16-smoke-l4-mmi-L-1']

**这是 v16.3_plan.md §7.3 的 S3 (§3 方向 D2)**, **不进 runner** (`_v16_run_all.py`),
不动任何登记值, 不碰生产路径, **不加守卫、不动分母** (DP-2)。

要回答的问题 (§1.1 Q1 的第二条腿):
  D1 换 `h`(离开临界点)去给 `G8` 补检验力; **本冒烟换 `L`**(留在临界点)。
  两条腿**回答的不是同一个问题**, 故 §3.4 明令**各自一张表、各自一个结论段**。

**与 D1 的分工 (§3.4 混杂警告)**: D1 动 `xi/L`(改 `h`, `L` 固定); 本冒烟动 `L`
(`h` 固定, `xi/L` 随之变)。**两者绝不能合并成一张表** —— 否则 `xi` 与 `L` 混杂,
就是 v16.2 的 S1 刚踩过的那类坑(分辨率与种子混杂, 靠 2x2 析因才分离出来)。

**先做的一件事: 核生产者/消费者的 L 网格是否一致** (§3.1)。代码级结论:
  * 生产者 `finite_size_scaling(L_list=(8,10,12,14,16,18), J=1.0, h=1.0)`
    (`_model/checks.py:455`) 的管线在 `:504-528`;
  * 消费者 `metric_finite_size_scaling(fss, loop=None)` (`_metric/extras.py:386`)
    在 `:422` 只做 `rows = fss['rows']`, **自己不跑任何对角化** => 网格**由构造相同**;
  * 已登记 `_v16_data.json` 的 `fss_L_list` = `[8,10,12,14,16,18]` (行 75630-75637),
    与生产者的 `L_list` **逐项相同** => **网格一致, 无需对齐**。
  本冒烟据此**自建那三行循环** (§2.2 订正 3(c) 的方法), **不改 `checks.py`** ——
  `checks.py:502` 明文承诺「读数与 `spiral_loop` **逐字同源**, 故同 L 必须同数」,
  那是一条**逐位**契约, 动它的风险远大于收益。

**由订正 1 改写的预登记判据 (§3.3)**: **不给方向性预测**。
  `xi/L` 随 `L` **单调下降且不收敛** (登记: 1.6648 -> 1.1616, `_v16_data.json:74791`),
  既非常数、也无已知极限 => pre.md 原拟的「由共形不变性预测 `I3` 与 `L` 无关」
  **不能写**。改为登记三条**可判的结构**。计划给了这三条的名称但**没给可执行口径**,
  故在此**先于数据写死**(防止结果出来之后再挑口径):

  参考分量一律取 **`w = 1`** —— 它是唯一在整条 L 网格上都合法的宽度
  (合法性 `3w < L`, 见下)。

  * **S1 同源**: 令 `a_L = |I3(L, w=1)|`, `b_L = xi(L)/L`。若 `a_L` 与 `b_L`
    在 L 上**各自单调**, 且**同向** (同增或同减) => 「`I3` 的量级由 `xi/L` 支配,
    而非由 `L` 支配」=> 与 D1 的机理**同源**, 两条方向**必须合并解读**,
    **不许当作两条独立证据**。
  * **S2 尺度稳定的违反**: 在 `L in {12,14,16,18}` 上 (i) `I3` 的**每个分量符号相同**
    (全 `>0` 或全 `<0`), 且 (ii) 量级稳定 `max|I3(L,w=1)| / min|I3(L,w=1)| <= 3`。
  * **S3 有限尺寸效应**: `I3(L, w=1)` 在 L 网格上**变号** =>
    **v16.2 的 `G8` 负结果不能外推** (这条会**削弱** `G8`, 照登)。
  * 三条**可同时成立、也可全不成立**; 报告**逐条列出**, 不得只报有利的那条。

**宽度合法性 (必须先写死, 否则会拿越界的块去算)**: `mmi_tripartite` 的三块是
`A=(0,w) B=(w,w) C=(2w,w)`, 余块 `D=(3w, L-3w)`。要求 `3w < L` (否则 D 为空、
`w=4` 在 `L<=12` 上直接越界)。故每个 L 的合法宽度集合是
`[w for w in (1,2,3,4) if 3*w < L]` (**必须显式传 `widths=`, 不能用默认的 `(1,2,3,4)`**):
  `L=8 -> {1,2}`, `L=10/12 -> {1,2,3}`, `L=14/16/18 -> {1,2,3,4}`。
**非法格子一律印 `—`, 不许插值、不许拿小 w 冒充** —— 那会让"跨 L 比较"变成假的。
⚠️ 推论: **`趋势 = I3(4) - I3(1)` 只在 `L >= 14` 上有定义**;
`L < 14` 行的趋势列用 `I3(w_max) - I3(1)` 并明写 `w_max`, 不得混排。

**数据的上游锚点 (逐条核过)**:
  * `G8` 登记值 (`L=16` 行必须复现)  —— `_runall.log:956`;
  * `xi/L` 逐 L 登记值                  —— `_v16_data.json` 的 `fss_xi_over_L` (行 75646-75653);
  * `c` 逐 L 登记值                     —— 同文件的 `fss_c_by_L` (行 75638-75645);
  * `xi/L` 不收敛的说法                 —— `_v16_data.json:74791`。

**估计量来源声明**: `I3` 用**生产实现** `mmi_tripartite` (`_metric/solvers.py:108`,
`from _metric.solvers import`; `G8` 在 `_metric/main.py:191` 调的正是它);
`xi` 走门面 `G.boundary_correlation_length` / `G._fit_length`;
`2D` 走门面 `MM.metric_correlation_exponent`。
**门面 `spiral_metric_v16` 没有 re-export `mmi_tripartite`, 本冒烟不去改门面 API。**

⚠️ **天花板 (与结果同页, 不许省)**:
  1. MMI 成立只是**必要**条件 —— 通过 != RT 被验证, 更 != 「涌现时空」
     (B2 硬边界 `spiral_model_v16.py:52-53`)。
  2. 本冒烟**全程在临界点 `h=1.0` 上**, 与 D1 的 `h != 1` 读数**互不影响**;
     两条方向的读数**不许混进同一张表** (§3.4)。
  3. **`2D` 的 L 依赖是"附带产出"**: 若 `2D` 在 `L=12..18` 上稳定趋近 `0.25`,
     那是**临界性本身**的一条正面读数, **与 MMI 无关**, 须**分列**,
     不许混进 `I3` 的结论里 (§3.3)。
  4. 成熟度**不动**: L4 仍是 C (audit `:75`)。
  5. 本脚本**不注册任何守卫 / 指标条目**; `METRICS` 的进程内 append **不落盘、不进 runner**。

退出码: `0 = 通过`; `3 = 复现/自校验失败` (L=16 行没复现 `G8`, 或 `xi/L` 对不上登记表,
或任一行的 SSA/纯度自校验失败 —— 都是「结论作废」); `2 = 脚本自身错误`。
⚠️ `3` 是「预期内的失败路径」, **不是 runner 意义的回归** (memory: `v16-single-runner`)。

用法:
    python v16/_v16_smoke_l4_mmi_L.py              # 完整三段
    python v16/_v16_smoke_l4_mmi_L.py --xi-only    # 只跑 xi/c/2D 侧 (不算 I3)
"""

import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'v16'))

import numpy as np  # noqa: E402
import spiral_model_v16 as G  # noqa: E402
import spiral_metric_v16 as MM  # noqa: E402
from _metric.solvers import mmi_tripartite  # noqa: E402

_VERSION_TAG = 'v16-smoke-l4-mmi-L-1'

# --- 生产配置: 与 `_model/checks.py:455` 的 finite_size_scaling 逐项相同 -----------
L_GRID = (8, 10, 12, 14, 16, 18)
J = 1.0
H = 1.0                      # 临界点 (D2 全程固定; D1 才动 h)

WIDTHS_ALL = (1, 2, 3, 4)
REF_W = 1                    # S1/S2/S3 的参考分量 (唯一在整条网格上合法的 w)
S2_L_SET = (12, 14, 16, 18)  # S2 的判据窗
S2_RATIO_MAX = 3.0           # S2 的「量级稳定」门槛 (本冒烟口径, 先写死)

WINDOW_KEYS = ('1-3', '4-8', 'full')

# --- 对齐靶 (全部来自已登记读数, 不是新算的) --------------------------------------
I3_LOGGED_L16 = (0.194068, 0.216633, 0.224846, 0.229723)   # `_runall.log:956`
XI_OVER_L_LOGGED = {                                        # `_v16_data.json` 行 75646-75653
    8: 1.6647944526783813, 10: 1.4689297809618247, 12: 1.348620566318015,
    14: 1.266716370469456, 16: 1.2070977917904357, 18: 1.1616085336867574,
}
C_LOGGED = {                                                # `_v16_data.json` 行 75638-75645
    8: 0.51340550819704, 10: 0.5109642433126494, 12: 0.5093138864893327,
    14: 0.5081210456096795, 16: 0.507216993106539, 18: 0.5065070735816783,
}
TOL_I3 = 1e-6        # 登记值 6 位小数
TOL_XI_OL = 1e-9     # 登记值给到 16 位; 留 1e-9 给 eigsh 的随机起点抖动 (~1e-12 量级)
TOL_C = 1e-9         # 同上
TOL_JW = 1e-10       # 与 `checks.py:545` 的 jw_worst 口径一致
N_REPEAT = 2         # 参考分量符号的可重复性检查 (见 §B2b)

_FREEZE_TAG = '=' * 76

_FAILS = []


def check(label, cond, detail=''):
    tag = 'PASS' if cond else '**FAIL**'
    print(f"    [{tag}] {label}" + (f"  -- {detail}" if detail else ''))
    if not cond:
        _FAILS.append(label)


def widths_valid(L):
    """合法宽度: 三块 A=(0,w) B=(w,w) C=(2w,w) 之后余块 D=(3w, L-3w) 必须非空 => 3w < L。"""
    return [w for w in WIDTHS_ALL if 3 * w < L]


def binned_corr(C, L):
    """`boundary_correlation_length` (`core.py:681-690`) 的分箱**逐字复刻** (见 D1 冒烟同款说明)。"""
    ds, vs = [], []
    for i in range(L):
        for j in range(i + 1, L):
            d = min(j - i, L - (j - i))
            ds.append(d)
            vs.append(abs(C[i, j]))
    ds = np.array(ds)
    vs = np.array(vs)
    ud = np.unique(ds)
    return ud, np.array([vs[ds == d].mean() for d in ud])


def xi_multi(C, L):
    """
    三窗口 xi。拟合一律走生产的 `G._fit_length`; 本函数只做区间切片
    (`max_d` 只有上界, 给不出 `[4,8]` 这种下界窗口)。
    ⚠️ 小 L 上长窗**合法点数可能 < 3** (`_fit_length` 要求 >= 3 点),
    此时返回 NaN —— **照登 NaN, 不许插值**。
    """
    d, y = binned_corr(C, L)
    out = {}
    for (lo, hi), key in zip(((1, 3), (4, 8), (None, None)), WINDOW_KEYS):
        if lo is None:
            out[key] = float(G._fit_length(d, y))
        else:
            m = (d >= lo) & (d <= hi)
            out[key] = float(G._fit_length(d[m], y[m])) if int(m.sum()) >= 3 else float('nan')
    return out


def monotone_dir(v):
    """序列的单调方向: +1 递增 / -1 递减 / 0 非单调(或全等)。"""
    dv = np.diff(np.asarray(v, dtype=float))
    if np.all(dv > 0):
        return 1
    if np.all(dv < 0):
        return -1
    return 0


def sign_state(i3s):
    """'-' 全部 <= 0 / '+' 全部 > 0 / '+-' 混合。"""
    if all(v <= 0 for v in i3s):
        return '-'
    if all(v > 0 for v in i3s):
        return '+'
    return '+-'


def measure_xi(L):
    """xi/c/2D 侧 —— 不碰 I3 (「xi 先测」纪律的代码形态)。"""
    tb = time.time()
    E0, gs = G.exact_ground_state(L, J, H)
    t_state = time.time() - tb
    Ejw = G.jw_ground_energy(L, J, H)
    cfit = G.fit_central_charge(gs, L)
    _, C, _ = G.boundary_correlation_graph(gs, L)
    xi = float(G.boundary_correlation_length(C, L))
    ce = MM.metric_correlation_exponent(gs, L)   # 进程内 append, 不落盘
    return dict(L=L, E0=E0, gs=gs, xi=xi, xi_over_L=xi / L, c=float(cfit['c']),
                jw_dev=abs(E0 - Ejw), xi_multi=xi_multi(C, L), t_state=t_state,
                two_delta=float(ce['two_delta']), eta_naive=float(ce['eta_naive']))


def main():
    xi_only = '--xi-only' in sys.argv
    t0 = time.time()
    print(f"=== {_VERSION_TAG} · L4: 临界点上把 I3 接到 L 扫描, "
          f"h={H:g}, J={J:g} ===")
    print(f"    口径: 生产同源估计量 mmi_tripartite + boundary_correlation_length/"
          f"_fit_length + metric_correlation_exponent;")
    print(f"    不进 runner, 不加守卫, 不动分母 (DP-2), 不改任何登记值。"
          f"{'  [--xi-only: 只测 xi/c/2D, 不算 I3]' if xi_only else ''}")
    print(f"    ⚠️ **本表与 D1 的 h 扫描表是两张表** (§3.4) —— 不许合并、不许互引为独立证据。")

    # =====================================================================
    print(f"\n--- 0 · 生产配置回显 + 生产者/消费者网格一致性 ---")
    print(f"    L 网格 = {list(L_GRID)}  (逐项同 `_model/checks.py:455` 的 L_list)")
    print(f"    消费者 `metric_finite_size_scaling` (`_metric/extras.py:422`) 只做")
    print(f"      `rows = fss['rows']`, **自己不跑对角化** => 网格由构造相同。")
    print(f"    已登记 `fss_L_list` = [8, 10, 12, 14, 16, 18] (`_v16_data.json` 行 75630)")
    print(f"      => 与生产者 L_list **逐项相同** => **网格一致, 无需对齐**。")
    print(f"    合法宽度 w (3w < L): " + ", ".join(
        f"L={L}->{widths_valid(L)}" for L in L_GRID))
    print(f"    S1/S2/S3 的参考分量 w = {REF_W} (唯一在整条网格上合法)")

    # =====================================================================
    # §A · 对齐校验 (必须先过) —— L=16 行复现 G8, 逐 L 复现 xi/L 与 c
    # =====================================================================
    print(f"\n--- A · 对齐校验 (自校验, 必须先过) ---")
    print(f"    ... 逐 L 建态 + 测 xi/c")
    rows = [measure_xi(L) for L in L_GRID]

    print(f"\n    A1 逐 L 的 xi/L 与 c 对齐已登记表 (`_v16_data.json` 行 75638-75653)")
    print(f"       {'L':>3s} | {'xi':>11s} {'xi/L 实算':>20s} {'xi/L 登记':>20s} "
          f"{'|diff|':>9s} | {'c 实算':>10s} {'|diff|':>9s} | {'|E0-E_jw|':>10s}")
    for r in rows:
        L = r['L']
        print(f"       {L:3d} | {r['xi']:11.5f} {r['xi_over_L']:20.16f} "
              f"{XI_OVER_L_LOGGED[L]:20.16f} {abs(r['xi_over_L'] - XI_OVER_L_LOGGED[L]):9.2e} | "
              f"{r['c']:10.6f} {abs(r['c'] - C_LOGGED[L]):9.2e} | {r['jw_dev']:10.2e}")
    dxi = max(abs(r['xi_over_L'] - XI_OVER_L_LOGGED[r['L']]) for r in rows)
    dc = max(abs(r['c'] - C_LOGGED[r['L']]) for r in rows)
    djw = max(r['jw_dev'] for r in rows)
    check("逐 L 的 xi/L 对齐已登记 fss_xi_over_L (对不上 => 停, 查复现)",
          dxi <= TOL_XI_OL, f"max|diff| = {dxi:.3e}")
    check("逐 L 的 c 对齐已登记 fss_c_by_L", dc <= TOL_C, f"max|diff| = {dc:.3e}")
    check("逐 L 的 Jordan-Wigner 独立核对 (查的是 H 的构造, 不是 eigsh 的收敛)",
          djw <= TOL_JW, f"max|E0 - E_jw| = {djw:.3e}")

    print(f"\n    A2 多窗口 xi 逐 L (小 L 上长窗合法点数 < 3 => NaN, **照登不插值**)")
    print(f"       {'L':>3s} | {'xi[1-3]':>10s} {'xi[4-8]':>10s} {'xi[全窗]':>10s} | "
          f"{'xi[1-3]/L':>10s} {'xi[4-8]/L':>10s} {'xi[全窗]/L':>10s}")
    for r in rows:
        x = r['xi_multi']
        print(f"       {r['L']:3d} | {x['1-3']:10.4f} {x['4-8']:10.4f} {x['full']:10.4f} | "
              f"{x['1-3'] / r['L']:10.4f} {x['4-8'] / r['L']:10.4f} {x['full'] / r['L']:10.4f}")

    if xi_only:
        print(f"\n    A3 I3 复现: **--xi-only 模式跳过** (不算 I3)")
    else:
        print(f"\n    A3 L=16 行复现 G8 登记值 (`_runall.log:956`) —— **本方向的对照段**")
        r16 = [r for r in rows if r['L'] == 16][0]
        m16 = mmi_tripartite(r16['gs'], 16, widths=WIDTHS_ALL)
        r16['mm'] = m16
        print(f"       {'w':>3s} | {'I3 实算':>13s} {'I3 登记':>13s} {'|diff|':>10s}")
        for w, got, want in zip(WIDTHS_ALL, m16['i3'], I3_LOGGED_L16):
            print(f"       {w:3d} | {got:13.8f} {want:13.8f} {abs(got - want):10.2e}")
        dev16 = max(abs(g - w) for g, w in zip(m16['i3'], I3_LOGGED_L16))
        check("L=16 行的 I3(w=1..4) 复现 G8 登记值 (1e-6 内)", dev16 <= TOL_I3,
              f"max|diff| = {dev16:.3e}")
        check("L=16 行的 I3 自校验 (SSA / 纯度) 成立",
              m16['ssa_min'] > -1e-9 and m16['purity_max'] < 1e-12,
              f"ssa_min = {m16['ssa_min']:.3e}, purity_max = {m16['purity_max']:.3e}")

    if _FAILS:
        print(f"\n    **对齐/复现失败 => 按纪律先查复现, 不继续** (v16.2 §5.1)")
        print(f"    失败项: {_FAILS}")
        return 3
    print(f"    A 段用时 {time.time() - t0:.2f} s")

    # =====================================================================
    # §B1 · xi 侧已在上方打印完 (A1/A2); 此处只补 2D
    # =====================================================================
    print(f"\n--- B1 · 2D (幂律/共形拟合, 附带产出) ---")
    print(f"    {'L':>3s} | {'2D':>10s} {'eta朴素':>10s} | {'|2D-0.25|':>10s}")
    for r in rows:
        print(f"    {r['L']:3d} | {r['two_delta']:10.6f} {r['eta_naive']:10.5f} | "
              f"{abs(r['two_delta'] - 0.25):10.6f}")
    print(f"    ⚠️ 这是**临界性本身**的读数, **与 MMI 无关** —— 必须与 I3 的结论**分列** (§3.3 附带产出)。")

    if xi_only:
        print(f"\n{_FREEZE_TAG}")
        print(f"  [--xi-only] 未计算任何 I3 (L=16 的 G8 复现也未做)。")
        print(f"{_FREEZE_TAG}")
        print(f"\n=== 结果: xi/c/2D 侧完成; 累计用时 {time.time() - t0:.1f} s ===")
        return 0

    print(f"\n{_FREEZE_TAG}")
    print(f"  ***** L 网格冻结线 —— 以下算 I3 *****")
    print(f"  冻结的网格: L = {list(L_GRID)}, h = {H:g} (临界点), J = {J:g}")
    print(f"{_FREEZE_TAG}")

    # =====================================================================
    # §B2 · I3 侧
    # =====================================================================
    print(f"\n--- B2 · I3 侧 ---")
    for r in rows:
        L = r['L']
        ws = widths_valid(L)
        r['widths'] = ws
        if L == 16:
            m = r['mm']                      # 复用 A3 的结果, 不重算
        else:
            m = mmi_tripartite(r['gs'], L, widths=ws)
        check(f"L={L} 的 I3 自校验 (SSA/纯度) 成立",
              m['ssa_min'] > -1e-9 and m['purity_max'] < 1e-12,
              f"ssa_min = {m['ssa_min']:.3e}, purity_max = {m['purity_max']:.3e}")
        # 按合法宽度铺成 4 槽; 非法槽位 None => 印 '-'
        slot = dict(zip(ws, m['i3']))
        r['i3'] = [slot.get(w) for w in WIDTHS_ALL]
        r['state'] = sign_state(m['i3'])
        r['ref_i3'] = slot[REF_W]            # S1/S2/S3 的参考分量 (w=1)
        r['w_max'] = ws[-1]
        r['trend'] = slot[ws[-1]] - slot[REF_W] if len(ws) > 1 else float('nan')

    # =====================================================================
    # §B2b · 参考分量符号的可重复性 —— **读数可信度, 不是判据**
    # S1/S2/S3 三条判据全部由 I3 的符号决定; 符号若其实是 eigsh 噪声, 判定就是假绿/假红。
    # 做法同 D1 冒烟 `_v16_smoke_l4_mmi_h.py` 的 §B2b (计划 §7.2 S2 规格之外的一处增补)。
    # =====================================================================
    print(f"\n--- B2b · 参考分量 (w={REF_W}) 符号的可重复性 "
          f"(读数可信度, **不是判据**; 每 L 额外 {N_REPEAT} 次独立 eigsh) ---")
    print(f"    {'L':>3s} | {'逐分量极差':>11s} | {'参考符号可重复':>14s}")
    for r in rows:
        if r['L'] == 16:
            print(f"    {r['L']:3d} | {'(A3 已测, 见上)':>11s}")
            r['ref_sign_same'] = True
            continue
        reps = []
        for _ in range(N_REPEAT):
            _, gs_rep = G.exact_ground_state(r['L'], J, H)
            reps.append(list(mmi_tripartite(gs_rep, r['L'], widths=r['widths'])['i3']))
        allv = np.array([r['ref_i3']] + [v[0] for v in reps])   # 参考分量 = 第 0 槽 (w=1)
        same = all(v > 0 for v in allv) or all(v <= 0 for v in allv)
        spread = float(allv.max() - allv.min())
        r['ref_sign_same'] = same
        print(f"    {r['L']:3d} | {spread:11.2e} | {'是' if same else '**否**':>14s}")
        check(f"L={r['L']} 的参考分量 I3(w={REF_W}) 符号在 {N_REPEAT} 次独立 eigsh 下不变",
              same, f"逐分量极差 = {spread:.2e}")

    # =====================================================================
    # §C · 联合表 (xi/L 与 I3 **同页**) + S1/S2/S3 判定
    # =====================================================================
    print(f"\n--- C · 联合表 (xi/L 与 I3 同页) + S1/S2/S3 判定 ---")
    print(f"    非法宽度槽位印 `-` (3w >= L, 越界/退化的定义域外), **不插值、不冒充**。")
    print(f"    趋势 = I3(w_max) - I3(1); w_max 逐行标注 (L>=14 时 w_max=4, 与计划一致)。")
    print()
    print(f"    {'L':>3s} | {'xi/L':>9s} {'xi[1-3]/L':>10s} {'xi[4-8]/L':>10s} "
          f"{'2D':>8s} | {'I3(w=1)':>12s} {'I3(w=2)':>12s} {'I3(w=3)':>12s} "
          f"{'I3(w=4)':>12s} | {'符号':>5s} {'w_max':>5s} {'趋势':>12s}")
    for r in rows:
        cells = []
        for v in r['i3']:
            cells.append('       n/a  ' if v is None else f"{v:12.3e}")
        print(f"    {r['L']:3d} | {r['xi_over_L']:9.4f} "
              f"{r['xi_multi']['1-3'] / r['L']:10.4f} "
              f"{r['xi_multi']['4-8'] / r['L']:10.4f} {r['two_delta']:8.4f} | "
              + ' '.join(cells)
              + f" | {r['state']:>5s} {r['w_max']:5d} {r['trend']:+12.3e}")

    print(f"\n    量级 (与符号**同页**; 缺了它, 一个 +1e-7 会被读成与 +7e-3 等价):")
    for r in rows:
        vals = [abs(v) for v in r['i3'] if v is not None]
        print(f"      L={r['L']:2d}: min|I3| = {min(vals):.2e},  max|I3| = {max(vals):.2e},  "
              f"参考 |I3(w={REF_W})| = {abs(r['ref_i3']):.2e}")

    # ---------------- S1 / S2 / S3 ----------------
    a = [abs(r['ref_i3']) for r in rows]        # |I3(L, w=1)|
    b = [r['xi_over_L'] for r in rows]          # xi/L
    print(f"\n    --- S1/S2/S3 判定 (口径见文件头, 先于数据写死) ---")
    print(f"    a_L = |I3(L, w={REF_W})| = {['%.3e' % v for v in a]}")
    print(f"    b_L = xi/L            = {['%.4f' % v for v in b]}")
    da, db = monotone_dir(a), monotone_dir(b)
    print(f"    a_L 单调方向 = {da:+d} (0 = 非单调),  b_L 单调方向 = {db:+d}")
    s1 = (da != 0 and db != 0 and da == db)
    if s1:
        print(f"    => **S1 成立 (同源)**: a_L 与 b_L **各自单调且同向** =>")
        print(f"       「**I3 的量级由 xi/L 支配, 而非由 L 支配**」。")
        print(f"       ⚠️ 与 D1 的机理**同源** => 两条方向**必须合并解读**,")
        print(f"          **不许当作两条独立证据** (§3.3)。")
    else:
        print(f"    => S1 不成立: "
              f"{'两序列并非各自单调' if (da == 0 or db == 0) else '两序列单调方向相反'}。")

    sub = [r for r in rows if r['L'] in S2_L_SET]
    states = sorted({r['state'] for r in sub})
    refs = [abs(r['ref_i3']) for r in sub]
    ratio = max(refs) / min(refs)
    print(f"\n    S2 窗 L in {list(S2_L_SET)}: 符号集合 = {states}, "
          f"max/min |I3(w={REF_W})| = {ratio:.3f} (门槛 <= {S2_RATIO_MAX})")
    s2 = (len(states) == 1 and '+-' not in states and ratio <= S2_RATIO_MAX)
    if s2:
        print(f"    => **S2 成立 (尺度稳定的违反)**: 符号不变且量级稳定。")
    else:
        why = []
        if len(states) != 1:
            why.append(f"符号不唯一 ({states})")
        if ratio > S2_RATIO_MAX:
            why.append(f"量级不稳 ({ratio:.2f} > {S2_RATIO_MAX})")
        print(f"    => S2 不成立: {'; '.join(why)}。")

    refs_all = [r['ref_i3'] for r in rows]
    s3 = (max(refs_all) > 0) and (min(refs_all) <= 0)
    print(f"\n    S3: I3(L, w={REF_W}) 的符号集合 = "
          f"{sorted({'<=0' if v <= 0 else '>0' for v in refs_all})}")
    if s3:
        print(f"    => **S3 成立 (有限尺寸效应)**: I3 在 L 网格上**变号** =>")
        print(f"       **v16.2 的 G8 负结果不能外推** —— 这条会**削弱** G8, 照登。")
    else:
        print(f"    => S3 不成立: I3(L, w={REF_W}) 在整条网格上符号不变。")

    hit = [n for n, ok_ in (('S1', s1), ('S2', s2), ('S3', s3)) if ok_]
    print(f"\n    **落支汇总: {hit or '无判据触发'}**")

    # ---------------- 附带产出 (必须分列) ----------------
    print(f"\n--- 附带产出 · 2D 的 L 依赖 (**与 MMI 无关, 必须分列**) ---")
    late = [r for r in rows if r['L'] >= 12]
    print(f"    L >= 12 的 2D = {['%.6f' % r['two_delta'] for r in late]}")
    print(f"    对解析值 0.25 的偏差 = {['%.2e' % abs(r['two_delta'] - 0.25) for r in late]}")
    d0 = abs(late[0]['two_delta'] - 0.25)
    d1 = abs(late[-1]['two_delta'] - 0.25)
    print(f"    单调方向 = {monotone_dir([r['two_delta'] for r in late]):+d} "
          f"(对 0.25 的距离: L={late[0]['L']} 的 {d0:.2e} -> L={late[-1]['L']} 的 {d1:.2e}, "
          f"{'在减小' if d1 < d0 else '未减小'})")
    print(f"    若它稳定趋近 0.25, 那是**临界性本身**的正面读数, **不是** MMI 的证据 (§3.3)。")

    # ---------------- 天花板 (与结果同页) ----------------
    print(f"\n--- 天花板 (与结果同页, 不许省) ---")
    print(f"    1. MMI 成立只是**必要**条件 —— 通过 != RT 被验证, 更 != 「涌现时空」")
    print(f"       (B2 硬边界 `spiral_model_v16.py:52-53`)。")
    print(f"    2. 本表是 **L 扫描表 (h=1.0 固定)**; D1 那张是 **h 扫描表 (L=16 固定)**。")
    print(f"       **两张表不许合并**, 也不许互引为独立证据; 若 S1 判同源, 只在**结论段**合并说明。")
    print(f"    3. 成熟度**不动**: L4 仍是 C (audit :75)。")
    print(f"    4. 本冒烟**未加任何守卫 / 指标条目**, 分母 78 / 27 / 13 不变 (DP-2)。")
    print(f"    5. `2D` 一列属**附带产出**, 与 `I3` 分列; 小 L 的多窗口 xi 印 NaN 是定义域所限。")

    ok = not _FAILS
    print(f"\n=== 结果: 自校验 {'通过' if ok else '**失败 => 结论作废**'}; "
          f"累计用时 {time.time() - t0:.1f} s ===")
    if _FAILS:
        print(f"    失败项: {_FAILS}")
    return 0 if ok else 3


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:                      # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"\n**脚本自身错误**: {type(exc).__name__}: {exc}")
        sys.exit(2)
