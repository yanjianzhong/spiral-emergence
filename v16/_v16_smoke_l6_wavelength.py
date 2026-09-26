# -*- coding: utf-8 -*-
"""
v16 · S1 冒烟 —— L6 斑图波长: 独立估计量互核 + 分辨率/种子 2×2
[_VERSION_TAG = 'v16-smoke-l6-lambda-1']

**这是 v16.2_plan.md §5.1 的 S1**, 不进 runner (`_v16_run_all.py`), 不动任何登记值,
不碰生产路径。真调 `stage6_life` / `derive_L6_grid` / `_char_scale_checked`。

要回答的问题 (背景见 plan 订正②):
  `stage6_life` 返回的 `spacing` (`stage67.py:362`) **至今没有任何守卫读它** ——
  `_v16_run.log:166` 印过 `斑图波长~0.1782`, 但那只是个**被打印的数**。
  本冒烟要把它变成**被两个独立估计量夹住的量**, 并分离一个此前未被识别的混杂因子。

三段:
  A. **生产复现**(自校验, 必须先过): 用与 `_model/main.py` **逐字相同**的调用跑一次,
     `spacing` 必须复现 `0.1782` (`_v16_run.log:166`, 打印精度 4 位)。
     ⚠️ 不过则整个冒烟的结论作废 —— 那说明测的不是生产那个量 (A 段"同方程不同盒子"
     栽过的正是这种坑)。同时**实测** `stage6_life` 的单次用时: 该用时从未被单独登记过。
  B. **两个独立估计量**:
       甲 = `spacing / h`  —— 沿 x 行的零交叉计数 (`stage67.py:360-362`, 域长单位)
       乙 = `_char_scale_checked(v - v̄, box=N, deconv=False)[0]` —— Δ²(k) 谱峰 (格)
     单位口径 (**本次逐行核过, 这是 pre.md 要求"显式定死"的那一条**):
       `_kgrids(shape, box)` 取 `d = box/n` (`registry.py:322`) ⇒ box 传格数时 k 的单位
       是 rad/格; `tier3_desc.py:130` 传的正是 `box = float(max(b.shape))` ⇒ **乙 的单位是格**。
       而 `spacing = 2L/zc` 是**域长**单位 ⇒ 甲 = `spacing/h` (h = L/N) 才可比。
      `deconv=False` 与 `tier3_desc.py:137` 对**连续场**的选择一致 (`registry.py:392-393`:
       "模型侧是连续场, 不能除 —— 对连续场除一个 CIC 窗等于凭空注入一个错误的谱形")。
  C. **分辨率 × 种子 2×2 析因**。`stage67.py:342-346` 的初值是**按格点写的**固定 4 格方块,
     其**物理**边长 `4h = 4*L_domain/N` 随 N 变 ⇒ 换 N 同时换掉了分辨率和种子尺寸,
     这是一个**混杂因子**, 与 A 段当年栽过的"盒子大小 vs 参数相区"同型。
     2×2 = {N=36, 48} × {种子格点固定, 物理固定} ⇒ 把两者分开。
     扫描**只在本冒烟里另起调用**, 不改 `stage6_life` 的初值规则 (那会改生产读数)。

诚实边界 (写在这里, 不许在报告里省):
  * 本冒烟**不判定** L6 的物理主张。`V8 生命斑图涌现` 仍是 L6 七条守卫里**唯一**扛
    物理主张的那一条 (audit `:101`); 这里做的是**测量质量**, 不是物理判定。
  * **不预设通过阈值**。仓库先例 `_v16_run.log:610` 的同类互核结果是 46.6% 的**系统性
    不一致** ⇒ 本项事先不知道会不会对上。先跑出实际偏差, 再由 DP-2 决定它够不够格当判据。
  * `spacing` 的估计量是**沿 x 行的零交叉计数**, 对迷宫/斑点斑图, 逐行散布可能很大;
    故一并报 `zc` 的**行间标准误**, 否则"λ 变了"与"估计量噪声大"分不开。

退出码: 0 = 三段都完成且 A 段复现成功; 3 = **A 段复现失败**(结论作废, 需查);
        2 = 本脚本自身错误。

用法:
    python v16/_v16_smoke_l6_wavelength.py
"""

import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'v16'))

import numpy as np  # noqa: E402
import spiral_model_v16 as G  # noqa: E402
import spiral_metric_v16 as MM  # noqa: E402

_VERSION_TAG = 'v16-smoke-l6-lambda-1'

# A 段的自校验靶: `_v16_run.log:166` 逐字 "斑图波长~0.1782 (域长 0.5)" (打印精度 .4f)
SPACING_LOGGED = 0.1782
SPACING_TOL = 5e-5            # .4f 的半个末位

# C 段: 物理固定种子所锚的参考边长 = 生产配置 (N=36) 下 4 格的物理尺寸
_SEED_REF_CELLS = 4

_FAILS = []


def check(label, cond, detail=''):
    tag = 'PASS' if cond else '**FAIL**'
    print(f"    [{tag}] {label}" + (f"  —— {detail}" if detail else ''))
    if not cond:
        _FAILS.append(label)


# ---------------------------------------------------------------------------
# 与 stage6_life 的**逐字复刻**, 唯一区别是把种子边长参数化。
# 复刻的合法性不靠"看着像", 靠 A 段与真函数**逐位比对**来证 (见 §A2)。
# ---------------------------------------------------------------------------
def rd_run(N, L, seed_cells, F, k, steps, dt, Du, Dv):
    h = L / N
    stab = dt * Du / h ** 2
    if stab > 0.25:
        return {'ok': False, 'stab': stab, 'h': h}

    u = np.ones((N, N))
    v = np.zeros((N, N))
    c = N // 2
    lo = c - seed_cells // 2
    u[lo:lo + seed_cells, lo:lo + seed_cells] = 0.5
    v[lo:lo + seed_cells, lo:lo + seed_cells] = 0.25
    for _ in range(steps):
        lu = (np.roll(u, 1, 0) + np.roll(u, -1, 0) +
              np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4 * u) / h ** 2
        lv = (np.roll(v, 1, 0) + np.roll(v, -1, 0) +
              np.roll(v, 1, 1) + np.roll(v, -1, 1) - 4 * v) / h ** 2
        uv2 = u * v ** 2
        u = u + dt * (Du * lu - uv2 + F * (1 - u))
        v = v + dt * (Dv * lv + uv2 - (F + k) * v)

    finite = bool(np.isfinite(u).all() and np.isfinite(v).all())
    return {'ok': True, 'u': u, 'v': v, 'h': h, 'stab': stab, 'finite': finite,
            'seed_cells': seed_cells, 'seed_phys': seed_cells * h}


def spacing_and_se(v, N, L):
    """与 `stage67.py:360-362` 同款的零交叉估计量, 外加行间标准误。"""
    zcs = np.array([len(np.where(np.diff(np.sign(v[i] - v[i].mean())))[0])
                    for i in range(N)], dtype=float)
    zc = float(zcs.mean())
    se = float(zcs.std(ddof=1) / np.sqrt(N)) if N > 1 else 0.0
    spacing = float(2.0 * L / zc) if zc > 0 else float('inf')
    # λ 与 zc 成反比 ⇒ 相对误差直接传递
    rel_se = float(se / zc) if zc > 0 else float('nan')
    return spacing, zc, se, rel_se


def spectral_lambda(v, N):
    """独立估计量乙: Δ²(k) 谱峰波长, 单位格。"""
    d = v - v.mean()          # 去掉 k=0 直流分量 (Δ²=k²P 在 k=0 处为 0, 故不改判据)
    lam, pk, valid = MM._char_scale_checked(d, float(N), deconv=False)
    i = int(np.argmax(MM._delta2(pk['k'], pk['P'], 2))) if len(pk['k']) else -1
    k_pk = float(pk['k'][i]) if i >= 0 else float('nan')
    k_fund = float(pk['k_fund']) if len(pk['k']) else float('nan')
    return float(lam), bool(valid), k_pk, k_fund


def main():
    t0 = time.time()
    K = G.KNOBS
    print(f"=== {_VERSION_TAG} · L6 斑图波长: 独立估计量互核 + 分辨率/种子 2×2 ===")
    print(f"    口径: 真调 stage6_life / derive_L6_grid / _char_scale_checked; 不进 runner。")

    # --- 生产网格: N 由 derive_L6_grid 派生, 不写死 ---------------------------
    # xi_over_L 不参与 N 的公式 (core.py:485-486 逐字), 它只进台账; 传 1.0 是占位。
    l6 = G.derive_L6_grid(1.0, K)
    N0, L_dom = int(l6['N']), float(l6['L_domain'])
    print(f"\n--- 0 · 生产配置 (由 derive_L6_grid 派生, 非写死) ---")
    print(f"    N={N0} (N_req={l6['N_req']}, N_cap={l6['N_cap']}, 受限于 {l6['bound_by']}), "
          f"L_domain={L_dom}, h={l6['h']:.6f}, dt*Du/h^2={l6['stab']:.4f}")
    print(f"    lambda_target = L_domain/n_lambda = {l6['lambda_target']:.4f}"
          f"  (⚠️ 旋钮靶, 不是预言: n_lambda={K['n_lambda']}, "
          f"pts_per_wavelength={K['pts_per_wavelength']} 都是自由旋钮)")

    # =====================================================================
    print(f"\n--- A · 生产复现 (自校验, 必须先过) ---")
    ta = time.time()
    s6 = G.stage6_life(F=K['F'], k=K['k'], N=N0, L=L_dom,
                       dt=K['dt'], Du=K['Du'], Dv=K['Dv'])
    t_prod = time.time() - ta
    print(f"    → 单次 stage6_life 用时 **{t_prod:.2f} s** "
          f"(该用时此前**从未被单独登记过**, 本行即其首次实测)")

    if not s6.get('ok'):
        print(f"    **FAIL** stage6_life 拒绝运行: {s6.get('reason')}")
        return 2

    sp_prod = float(s6['spacing'])
    print(f"\n    A1 生产读数: spacing={sp_prod:.6f} (域长), h={s6['h']:.6f}, "
          f"contrast={s6['contrast']:.3f}, active={s6['active_frac']:.3f}, "
          f"emerged={s6['emerged']}")
    check(f"spacing 复现登记值 {SPACING_LOGGED} (打印精度 .4f)",
          abs(sp_prod - SPACING_LOGGED) <= SPACING_TOL,
          f"|{sp_prod:.6f} - {SPACING_LOGGED}| = {abs(sp_prod - SPACING_LOGGED):.2e}")

    # A2: 复刻件与真函数逐位比对 —— 这是 C 段用复刻件做扫描的**唯一许可证**
    rep = rd_run(N0, L_dom, _SEED_REF_CELLS, K['F'], K['k'], s6['steps'],
                 K['dt'], K['Du'], K['Dv'])
    dv = float(np.abs(rep['v'] - s6['v']).max())
    du = float(np.abs(rep['u'] - s6['u']).max())
    print(f"\n    A2 复刻件 vs 真函数 (同一 N={N0}, 同一 4 格种子): "
          f"max|dv|={dv:.3e}, max|du|={du:.3e}")
    check("复刻件与 stage6_life 逐位一致 (C 段扫描的许可证)",
          dv == 0.0 and du == 0.0,
          "逐位相同" if (dv == 0.0 and du == 0.0) else "**有差异 => C 段的复刻件不可用**")

    # =====================================================================
    print(f"\n--- B · 两个独立估计量 (N={N0}) ---")
    sp_a, zc, se_zc, rel_se = spacing_and_se(s6['v'], N0, L_dom)
    lam_a = sp_a / s6['h']
    lam_b, valid_b, k_pk, k_fund = spectral_lambda(s6['v'], N0)
    print(f"    甲 (零交叉): spacing={sp_a:.6f} 域长 = {lam_a:.4f} 格"
          f"   [zc={zc:.3f} +- {se_zc:.3f} 行间SE => 相对 {rel_se * 100:.2f}%]")
    print(f"    乙 (D2谱峰): {lam_b:.4f} 格   valid={valid_b}   "
          f"k_peak={k_pk:.6f} rad/格, k_fund={k_fund:.6f} => 峰在第 "
          f"{k_pk / k_fund:.2f} 壳 (壳宽 k_fund, **量化偏差的来源**)")
    dev = abs(lam_a - lam_b) / lam_b if lam_b > 0 else float('nan')
    print(f"    相对偏差 |甲-乙|/乙 = **{dev * 100:.2f}%**")
    print(f"    ⚠️ **不预设阈值**: 仓库先例 (`_v16_run.log:610`) 同类互核是 46.6% 的"
          f"系统性不一致 => 本项能不能对上事先不知道, 此行只登记实测。")
    print(f"    参照: 甲 / (pts_per_wavelength={K['pts_per_wavelength']}) = "
          f"{lam_a / K['pts_per_wavelength']:.4f}"
          f"  (~1 说明 N 就是按 12 格/波长构造的, 不是物理预言)")

    # =====================================================================
    print(f"\n--- C · 分辨率 x 种子 2x2 析因 ---")
    print(f"    混杂因子: 种子是**按格点**写的固定 4 格 => 物理边长 4h 随 N 变。")
    print(f"    物理固定 = 锚住生产 (N=36) 下的物理边长 "
          f"{_SEED_REF_CELLS * L_dom / 36:.6f}, 反解该 N 下的格数。")

    rows = []
    for N in (36, 48):
        h = L_dom / N
        kk_phys = max(2, int(round((_SEED_REF_CELLS * L_dom / 36) / h)))
        for mode, kk in (('格点固定', _SEED_REF_CELLS), ('物理固定', kk_phys)):
            r = rd_run(N, L_dom, kk, K['F'], K['k'], s6['steps'],
                       K['dt'], K['Du'], K['Dv'])
            if not r.get('ok'):
                rows.append((N, mode, kk, None, None, None, None, None, None, r['stab']))
                continue
            sp, zc_, se_, rse = spacing_and_se(r['v'], N, L_dom)
            lam = sp / r['h']          # 格 —— ⚠️ 含一个 N 因子, **不可跨 N 直接比**
            lb, vd, _, _ = spectral_lambda(r['v'], N)
            lb_phys = lb * r['h']
            contrast = float(r['v'].max() - r['v'].min())
            act = float((r['v'] > 0.1).mean())
            rows.append((N, mode, kk, lam, sp, lb, lb_phys, rse, contrast, r['stab']))
            print(f"    N={N} {mode}({kk}格, 物理边长 {kk * r['h']:.5f}): "
                  f"λ甲={sp:.5f} 域长 ({lam:.4f} 格), λ乙={lb_phys:.5f} 域长 ({lb:.4f} 格), "
                  f"zc 相对SE={rse * 100:.2f}%, contrast={contrast:.3f}, "
                  f"dt*Du/h^2={r['stab']:.4f}")

    print(f"\n    {'N':>4s} {'种子':>10s} {'格数':>5s} {'λ甲(域长)':>12s} {'λ乙(域长)':>12s} "
          f"{'λ甲(格)':>10s} {'zc相对SE':>10s} {'contrast':>9s}")
    for N, mode, kk, lam, sp, lb, lbp, rse, ct, stab in rows:
        if sp is None:
            print(f"    {N:4d} {mode:>10s} {kk:5d}   拒绝运行 (dt*Du/h^2={stab:.4f})")
        else:
            print(f"    {N:4d} {mode:>10s} {kk:5d} {sp:12.5f} {lbp:12.5f} "
                  f"{lam:10.4f} {rse * 100:9.2f}% {ct:9.3f}")

    by = {(N, m): (sp, lbp, rse)
          for N, m, kk, lam, sp, lb, lbp, rse, ct, st in rows}
    kk48p = [kk for N, m, kk, *_ in rows if N == 48 and m == '物理固定']
    g36c, g36p = by.get((36, '格点固定')), by.get((36, '物理固定'))
    g48c, g48p = by.get((48, '格点固定')), by.get((48, '物理固定'))
    print(f"\n    读法 (四格齐了才下结论, 缺一格不下):")
    print(f"      ⚠️ 比较**必须在物理单位(域长)上做** —— λ甲_格 = spacing/h 里天然含一个 N")
    print(f"         因子, 拿「格」跨 N 直接比会把单位换算读成物理效应(首版就栽在这里)。")
    print(f"      N=36 两格**必然重合**(物理固定的反解格数就是 4) => 该行不构成证据, "
          f"它只是复刻件的第二次自校验。")
    if None not in (g36c, g36p, g48c, g48p):
        sp36c, _, se36 = g36c
        sp48c, _, se48 = g48c
        sp48p, _, _ = g48p
        d_res = (sp48c - sp36c) / sp36c * 100.0     # 只动分辨率 (种子规则固定)
        d_seed = (sp48p - sp48c) / sp48c * 100.0    # 只动种子 (N 固定)
        se_comb = float(np.hypot(se36, se48)) * 100.0
        print(f"      只动 N (36->48, 种子格点固定): spacing {sp36c:.6f} -> {sp48c:.6f}"
              f"  = {d_res:+.2f}%")
        print(f"      只动种子 (4 格->{kk48p[0] if kk48p else '?'} 格, N=48): "
              f"{sp48c:.6f} -> {sp48p:.6f}  = {d_seed:+.2f}%")
        print(f"      估计量自身噪声底 (两跑合并行间SE): {se_comb:.2f}%"
              f"  => |分辨率效应| / 噪声 = {abs(d_res) / se_comb:.2f}")
        if abs(d_res) > 2.0 * se_comb:
            print(f"      => 超出噪声 2 倍 ⇒ 可读出「**分辨率依赖**」")
        else:
            print(f"      => **在噪声底附近, 不能读作检测到分辨率依赖**;")
            print(f"         但同样**不能**说「分辨率无关」—— 那是把'未检测到'当成'无效应'。")
    else:
        print(f"      ⚠️ 四格未齐, 按计划**不下结论**。")
    print(f"\n      ⚠️ **λ乙 不能用来判分辨率无关性**: 它的壳边界在物理 k 空间里是 "
          f"k = j*2*pi/L, **与 N 无关**")
    print(f"         (FFT 的壳随盒子一起缩放) ⇒ λ乙_物理 必然 N 无关。那是**恒等式**, "
          f"不是证据; 拿它当「分辨率无关」的佐证就是假绿。")

    print(f"\n    诚实边界: 本段只检验『斑图波长由化学参数定, 不由几何定』这句话"
          f"(`_v16_run.log:169` / `main.py:261-262` 一直在印, **至今无守卫**)。"
          f"\n    它**不**判定 L6 的物理主张 —— `V8` 仍是那 7 条里唯一扛物理主张的守卫。")

    ok = not _FAILS
    print(f"\n=== 结果: A 段自校验 {'通过' if ok else '**失败 => 结论作废**'}; "
          f"累计用时 {time.time() - t0:.1f} s"
          f" (其中 stage6_life 单次 {t_prod:.2f} s) ===")
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
