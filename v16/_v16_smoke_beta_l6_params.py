# -*- coding: utf-8 -*-
"""
v16.4 · D9 (β-L6) 冒烟 —— `(F,k)` 随机化负对照 (**V8 的判据能失败吗?**)
[_VERSION_TAG = 'v16-smoke-beta-l6-params-1']

**这是 v16.4_plan.md §3.2 的 D9**。**不进 runner**（DP-β-1 默认 = 不执行「计入诊断
守卫」）⇒ **不加守卫、不加指标、不动分母 78/27/13**，不改任何登记值。

---

## 要回答的问题

L6 的 7 条同层守卫中**只有 `V8 生命斑图涌现` 一条扛物理主张**（审计 §一），
而两条 L6 负对照**都是格式性质** —— `negctrl.py:355-358` 自述逐字：
「L6 的两条是**格式性质** —— 保证代码按写下的方程在算, 不保证那个方程描述的是生命」。
⇒ β-L6 会是 L6 上**第一条推离物理判据**的对照，填的是**空白**，不是重复（计划 §1.4）。

`F`/`k` 在生产里是**写死的默认值**（`stage67.py:314`：`F=0.035, k=0.060`），
全仓无「随机化参数」匹配。本冒烟就是在 `(F,k)` 平面上取一族点，**逐点问同一句话**：
`emerged` 会不会变成 `False`？

## 判据落点（锚死，不是推断）

`stage67.py:363` 逐字：
    emerged = bool(finite and contrast > 0.2 and active > 0.05)

## 预登记判据（**先写死，再看数**；计划 §3.2 的可执行转录）

| 落支 | 条件 | 含义 |
|---|---|---|
| **N3 有检验力** | 存在一族 `(F,k)` 使 `emerged=False` | 负对照成立（V8 的判据**能**失败） |
| **N4 不敏感** | 所有取点都 `emerged=True` | 记「V8 的判据对参数不敏感」 |
| **N5 仅有数值失稳** | **新增，计划未列，故在此先于数据写死**：全部 `emerged=False` 的点都是 `finite=False`（数值发散），**没有一个**是 `finite=True` 却被 `contrast`/`active` 判掉 | 判据本身**未被检验** —— 失败来自 `finite` 那一项，不是来自物理量。既不是 N3 也不是 N4，**不许归到 N3** |

**N3 的口径（计划未给，故在此先于数据写死）**：N3 只数 **`finite=True` 且
`emerged=False`** 的点（判据被 `contrast`/`active` 判掉）；`finite=False` 的点
**单列**为 `[数值失稳]`，**不计入 N3** —— 否则「算炸了」会被记成「判据有检验力」。

## 两个必须先登记的口径（计划 §3.2 点名要求）

1. **`steps` 砍量 = 零（这是实测结论，不是省略）**：`stage6_life` 默认 `steps=40000`
   （`stage67.py:314`），生产 `_model/main.py:251-253` **不传 `steps`** ⇒ 生产值就是
   `40000`。DP-β-2 的默认是「砍到单点 < 30 s」—— 本脚本 §A 实测**单点 8.4 s**，
   **已经 < 30 s** ⇒ **按 DP-β-2 的默认条件，不需要砍**。
   ⇒ **本冒烟的读数与 v16.3 登记值直接可比**（这一点必须写明，因为它是可比性的来源）。
2. **`(F,k)` 取点是选择，不是派生** —— 写死在这里，**规则格点**（不是手挑的点），
   且**含有生产默认点 `(0.035, 0.060)`**：
       `F_GRID = (0.010, 0.020, 0.030, 0.035, 0.045, 0.055)`
       `K_GRID = (0.045, 0.050, 0.055, 0.060, 0.065)`
   ⇒ 6 x 5 = 30 点。
   ⚠️ **声明**：本脚本**没有**去重新推导 Pearson 相图的边界 —— 取这个格点的理由是
   「跨度够大 + 是规则格点（无事后挑选的余地）+ 含生产默认点」。
   本冒烟**只**主张「在**这个声明过的格点**上判据是否会失败」，
   **不主张**它覆盖了 Pearson 相图的哪个相区。

## 其余配置全部钉在生产值上（只动 `F`/`k` 两个自由度）

`N=36`（L4/L6 派生的值，**不是** `stage6_life` 的默认 40；依据 `_runall_v16.3_r1.log:306`
逐字「派生 = 36（手工 = 40）」与 `:307`「h=0.013889, dt*Du/h^2=0.1037」）、
`L=0.5`、`dt=1.0`、`Du=2e-5`、`Dv=1e-5`、`steps=40000`。

## 退出码（沿用 v16.2/v16.3 冒烟惯例）

`0 = 自校验通过`（**这不代表 N3**，落支可能是 N4 —— 那是**结论**不是失败）；
`3 = 自校验失败`（A 段复现不过，或稳定性闸不过）⇒ **结论作废**；
`2 = 脚本自身错误`。
⚠️ `3` 是「预期内的失败路径」，**不是 runner 意义的回归**（memory: `v16-single-runner`）。

用法:
    python v16/_v16_smoke_beta_l6_params.py
"""

import contextlib
import io
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'v16'))

import numpy as np                                            # noqa: E402

from _model.stage67 import stage6_life                        # noqa: E402

_VERSION_TAG = 'v16-smoke-beta-l6-params-1'

# --- 生产配置 (全部钉死; 只放 F / k 两个自由度) ----------------------------------
N = 36           # 派生值 (`_runall_v16.3_r1.log:306`); stage6_life 的默认是 40, 不用
L_DOMAIN = 0.5
DT = 1.0
DU = 2e-5        # `core.py:98` 逐字 `'Du': 2e-5`
DV = 1e-5        # `core.py:99` 逐字 `'Dv': 1e-5`
#   ⚠️ D9 首跑 A 段复现失败 (`v16/_beta_l6_params.log`), 根因就是这一行: 本脚本初版误写
#   `DV = 2e-5`。等扩散 (Dv = Du) 没有 Turing 失稳 ⇒ 实测 contrast=0.000, active=0.000,
#   emerged=False —— A 段自校验**正是为此存在**, 它把笔误挡在了扫描之前。
STEPS = 40000    # 生产值 (`_model/main.py:251-253` 不传 steps ⇒ 默认 40000) ⇒ 砍量 = 0

F_PROD, K_PROD = 0.035, 0.060

# --- 预登记格点 (规则格点, 含生产默认点; 见口径 2) --------------------------------
F_GRID = (0.010, 0.020, 0.030, 0.035, 0.045, 0.055)
K_GRID = (0.045, 0.050, 0.055, 0.060, 0.065)

# --- V8 的登记读数 (自校验靶; 全部来自 `_runall_v16.3_r1.log:192-196`) ------------
V8_CONTRAST = 0.349        # 4 位小数
V8_ACTIVE = 0.637          # 3 位小数
V8_SPACING = 0.1782        # 4 位小数
V8_H = 0.0139              # 4 位小数 (log :193)
V8_STAB = 0.1037           # 4 位小数 (log :193)
TOL_V8 = 5e-4              # 各值半个末位

CONTRAST_MIN = 0.2         # `stage67.py:363` 逐字
ACTIVE_MIN = 0.05          # `stage67.py:363` 逐字

_FREEZE_TAG = '=' * 76

_FAILS = []


def check(label, cond, detail=''):
    tag = 'PASS' if cond else '**FAIL**'
    print(f"    [{tag}] {label}" + (f"  —— {detail}" if detail else ''))
    if not cond:
        _FAILS.append(label)


def life(F, k, quiet=True):
    """跑一个 `(F,k)` 点。`quiet=True` 时吞掉 `stage6_life` 自己的打印 (30 个点会淹掉表)。"""
    buf = io.StringIO()
    ctx = contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext()
    t0 = time.time()
    with ctx:
        r = stage6_life(F=F, k=k, N=N, L=L_DOMAIN, dt=DT, Du=DU, Dv=DV, steps=STEPS)
    r = dict(r)
    r['elapsed'] = time.time() - t0
    r['stdout'] = buf.getvalue()
    return r


def main():
    t0 = time.time()
    print(f"=== {_VERSION_TAG} · D9 (β-L6) `(F,k)` 随机化负对照 ===")
    print(f"    口径: 不进 runner, 不加守卫, 不动分母 78/27/13, 不改任何登记值 (DP-β-1)。")
    print(f"    判据落点: stage67.py:363  `emerged = finite and contrast > "
          f"{CONTRAST_MIN} and active > {ACTIVE_MIN}`")

    # =====================================================================
    print(f"\n--- 0 · 配置回显 (只动 F/k 两个自由度) ---")
    print(f"    N={N} (**派生值**, 不是 stage6_life 默认的 40; `_runall_v16.3_r1.log:306`)")
    print(f"    L={L_DOMAIN}, dt={DT:g}, Du={DU:g}, Dv={DV:g}, steps={STEPS}")
    print(f"    ⇒ **DP-β-2 的砍量 = 0**: 生产值就是 steps={STEPS} (生产不传 steps), "
          f"而 §A 实测单点 8.4 s 已 < 30 s ⇒")
    print(f"       **不需要砍** ⇒ 本冒烟读数与 v16.3 登记值**直接可比** (口径 1)。")
    print(f"    (F,k) 格点 = {len(F_GRID)} x {len(K_GRID)} = {len(F_GRID) * len(K_GRID)} 点; "
          f"F = {list(F_GRID)}")
    print(f"                                        k = {list(K_GRID)}")
    print(f"    含生产默认点 ({F_PROD}, {K_PROD}) = {F_PROD in F_GRID and K_PROD in K_GRID}")

    # =====================================================================
    print(f"\n--- A · 复现 V8 登记值 (自校验, 必须先过) ---")
    tA = time.time()
    a = life(F_PROD, K_PROD, quiet=False)
    print(f"    A1 稳定性闸: dt*Du/h^2 = {a['stab']:.4f} <= 0.25 "
          f"(登记 `_runall_v16.3_r1.log:193`: {V8_STAB})")
    check("稳定性闸由构造成立 (h 与 dt 与生产同) —— 数值失稳不是本冒烟的变量",
          a['ok'] and a['stab'] <= 0.25,
          f"ok = {a['ok']}, stab = {a['stab']:.4f}, h = {a['h']:.6f}")
    check(f"h 复现 {V8_H}", abs(a['h'] - V8_H) <= TOL_V8, f"实算 {a['h']:.6f}")
    check(f"对比度复现登记值 {V8_CONTRAST}",
          abs(a['contrast'] - V8_CONTRAST) <= TOL_V8,
          f"实算 {a['contrast']:.4f} vs 登记 {V8_CONTRAST}")
    check(f"活化面积复现登记值 {V8_ACTIVE}",
          abs(a['active_frac'] - V8_ACTIVE) <= TOL_V8,
          f"实算 {a['active_frac']:.4f} vs 登记 {V8_ACTIVE}")
    check(f"斑图波长复现登记值 {V8_SPACING}",
          abs(a['spacing'] - V8_SPACING) <= TOL_V8,
          f"实算 {a['spacing']:.4f} vs 登记 {V8_SPACING} (`_runall_v16.3_r1.log:194`)")
    check("生产默认点 emerged=True (与 V8 登记「通过」一致)", bool(a['emerged']),
          f"contrast={a['contrast']:.4f}>{CONTRAST_MIN} 且 "
          f"active={a['active_frac']:.4f}>{ACTIVE_MIN}")
    print(f"    A 段用时 {time.time() - tA:.2f} s  ← **单点成本登记** "
          f"(DP-β-2 的判据就挂在这个数上)")

    if _FAILS:
        print(f"\n    **A 段复现失败 => 按纪律先查复现, 不继续** (v16.2 §5.1)")
        print(f"    失败项: {_FAILS}")
        return 3

    # =====================================================================
    print(f"\n{_FREEZE_TAG}")
    print(f"  ***** 格点冻结线 —— 以下开始扫描, 格点自此冻结 (v16.4_plan §3.2) *****")
    print(f"  冻结的 F = {list(F_GRID)}")
    print(f"  冻结的 k = {list(K_GRID)}")
    print(f"  看过 emerged 之后再改格点 = p-hacking。")
    print(f"{_FREEZE_TAG}")

    print(f"\n--- B · (F,k) 扫描 ({len(F_GRID) * len(K_GRID)} 点, 每点 {STEPS} 步) ---")
    rows = []
    for F in F_GRID:
        line = []
        for k in K_GRID:
            r = life(F, k)
            rows.append(dict(F=F, k=k, **{kk: r[kk] for kk in
                                          ('emerged', 'finite', 'contrast',
                                           'active_frac', 'spacing', 'elapsed')}))
            tag = ('? ' if not r['finite'] else ('o ' if r['emerged'] else 'X '))
            line.append(f"{k:.3f}:{tag}")
        print(f"    F={F:.3f} | " + ' '.join(line)
              + f"   ({sum(1 for r in rows if r['F'] == F and r['finite'])}/{len(K_GRID)} 有限)")

    print(f"\n   图例: o = 有限且 emerged=True (涌现);  X = 有限但 emerged=False (**判据判掉**);")
    print(f"         ? = **数值失稳** finite=False (单列, 不计入 N3; 见 N5 口径)")
    print(f"\n   逐点明细 (对比度阈值 {CONTRAST_MIN}, 活化面积阈值 {ACTIVE_MIN}):")
    print(f"    {'F':>6s} {'k':>6s} | {'contrast':>9s} {'active':>8s} | {'emerged':>8s} "
          f"{'finite':>7s} | {'spacing':>9s} | {'用时':>6s}")
    for r in rows:
        print(f"    {r['F']:6.3f} {r['k']:6.3f} | {r['contrast']:9.4f} "
              f"{r['active_frac']:8.4f} | {str(r['emerged']):>8s} "
              f"{str(r['finite']):>7s} | {r['spacing']:9.4f} | {r['elapsed']:5.2f}s")

    print(f"\n    用时: 单点中位 {np.median([r['elapsed'] for r in rows]):.2f} s, "
          f"合计 {sum(r['elapsed'] for r in rows):.1f} s "
          f"(**只报脚本自己的计时, 不报 CPU 时间**)")

    # =====================================================================
    print(f"\n--- C · 落支判定 (N3 / N4 / N5) ---")
    fin = [r for r in rows if r['finite']]
    nonfin = [r for r in rows if not r['finite']]
    judged_false = [r for r in fin if not r['emerged']]   # 被 contrast/active 判掉
    judged_true = [r for r in fin if r['emerged']]

    print(f"    finite=True            : {len(fin)} / {len(rows)}")
    print(f"    其中 emerged=True      : {len(judged_true)}")
    print(f"    其中 emerged=False     : {len(judged_false)}  ← **计入 N3 的**")
    print(f"    finite=False (数值失稳): {len(nonfin)}  ← **单列, 不计入 N3** (N5 口径)")

    if judged_false:
        print(f"\n    被判掉的点 (F, k, contrast, active):")
        for r in judged_false:
            why = []
            if not r['contrast'] > CONTRAST_MIN:
                why.append(f"contrast {r['contrast']:.4f} <= {CONTRAST_MIN}")
            if not r['active_frac'] > ACTIVE_MIN:
                why.append(f"active {r['active_frac']:.4f} <= {ACTIVE_MIN}")
            print(f"      ({r['F']:.3f}, {r['k']:.3f}): contrast={r['contrast']:.4f}, "
                  f"active={r['active_frac']:.4f}  —— {'; '.join(why)}")

    if not fin:
        print(f"\n    => **落支 N5 (仅有数值失稳)**: {len(rows)} 个点全部 `finite=False`。")
        print(f"       判据**未被检验** —— 失败来自 `finite` 那一项, 不是来自物理量。")
        print(f"       **不许归到 N3。** 需先修数值 (dt/分辨率), 再重跑本格点。")
        branch = 'N5'
    elif judged_false and all(not r['finite'] for r in rows if not r['emerged']):
        print(f"\n    => **落支 N5 (仅有数值失稳)**: 全部 `emerged=False` 的点都是 "
              f"`finite=False`,")
        print(f"       **没有一个**是 `finite=True` 却被 `contrast`/`active` 判掉的。")
        print(f"       ⇒ 判据本身**未被检验**。**不许归到 N3。**")
        branch = 'N5'
    elif judged_false:
        near = min(judged_false,
                   key=lambda r: abs(r['F'] - F_PROD) + abs(r['k'] - K_PROD))
        print(f"\n    => **落支 N3 (有检验力)**: 存在 {len(judged_false)} 个 `(F,k)` 使 "
              f"`emerged=False`")
        print(f"       且 `finite=True` ⇒ **V8 的判据能失败**。负对照成立。")
        print(f"       离生产点 ({F_PROD}, {K_PROD}) 最近的失败点 (|ΔF|+|Δk| 意义下): "
              f"({near['F']:.3f}, {near['k']:.3f})")
        print(f"       ⚠️ **边界**: N3 说的是「**判据可证伪**」,**不是**")
        print(f"          「生产点附近不安全」, 更不是「L6 的物理主张被验证」。")
        print(f"          失败点多数落在**远离生产点**的相区上, 这是**预期**的。")
        branch = 'N3'
    else:
        print(f"\n    => **落支 N4 (不敏感)**: 全部 {len(fin)} 个有限点都 `emerged=True`。")
        print(f"       ⇒ 记「**V8 的判据对参数不敏感**」—— 照登, 不包装。")
        branch = 'N4'

    # ---------------- 天花板 (与结果同页, 不许省) ----------------
    print(f"\n--- 天花板 (与结果同页, 不许省) ---")
    print(f"    1. **不改任何登记值**: V8 的登记 (`_runall_v16.3_r1.log:944-945`, "
          f"对比度 {V8_CONTRAST} / 活化 {V8_ACTIVE}) 一个数字不动;")
    print(f"       本冒烟**未加任何守卫 / 指标条目**, 分母 78 / 27 / 13 不变 (DP-β-1)。")
    print(f"    2. **负对照成立 != 升级**: 依 `memory/l1-maturity-held-at-b.md` 逐字")
    print(f"       「负对照缺口补上了 **≠** 该升级」⇒ **L6 成熟度不动**。")
    print(f"    3. **相区声明**: 本冒烟**没有**重新推导 Pearson 相图边界; 格点是**规则格点**")
    print(f"       (无事后挑选余地) + 含生产默认点。主张只到「在这个**声明过的**格点上")
    print(f"       判据会/不会失败」为止。")
    print(f"    4. **`F`/`k` 只是两个自由度**: `N` / `L` / `dt` / `Du` / `Dv` 全部钉在生产值。")
    print(f"       ⇒ 本冒烟**不回答**「换个 N 会怎样」。")
    print(f"    5. **`V8` 是一条 guard (不是负对照)**: `_metric/main.py:146-148` 用 "
          f"`record_guard` 登记,")
    print(f"       故本冒烟补的是「**它能不能失败**」, 而不是「它是不是一条推离」。")

    ok = not _FAILS and branch != 'N5'
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
