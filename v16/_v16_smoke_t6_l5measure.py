# -*- coding: utf-8 -*-
"""v16-smoke-t6-l5measure-1 -- 阶段 4（接 L5：几何测度）判据的**一次性执行**。

判据来源（**逐字**照 `v16/v16.8_plan.md` §0.1.11，冻结于开跑前）：
  8 个测度全集 / 统一的 [5%,95%] 判据 / 三条「不可用」排除条件 /
  两个负对照（④a 对照自比、④b 保度随机重连）/ 2 个对象 / 落支名 /
  非有限处理取 **A**（`_pctile_rank` 原样丢非有限，同报 `n_eff`，`n_eff<9` 只登记不成结论）。

来源代码（只读，除本文件外不改任何东西）：
  _model/mera.py:53   mera_init(L, chi, seed)      | _model/mera.py:426  build_mera_graph(mera)
  _model/stage5.py:213 curvature_report(G, name)  | _model/stage5.py:260 geometry_controls(G, n_each)
  _model/stage5.py:281-301 对照图构造 recipe（**本文件按 §0.1.11 ⑧ 登记在案的复制**）
  _metric/extras.py:884  _pctile_rank(value, sample)
  _v16_smoke_t6_tiling.py:147 build(dual_rings)    -- 真调，不复制

退出码约定（与阶段 0/1/2 的 EXIT=0 不可直接对比）：
  0 = 已落落支（`T6-L5测度有信号` 或 `T6-L5测度仍零`，都算本阶段完成）；
  3 = **判据未成立**（交叉校验守卫不过 / ④a 判流程不可信）=> **不落任何落支**，先修流程。

边界声明：
  - 不改 78 / 27 / 13；不改 audit；不改 `_v16_run_all.py`；不改 `_model/hyper.py`、T4/T5 及其 log。
  - 不改 `_model/stage5.py` 的构造（复制而非改动），交叉校验守卫是本复制的补丁。
  - 单一对象一次跑完，**不做参数扫描、不事后增删测度**（§0.1.11 ①）。
"""
import os
import sys
import time

_T_PROC = time.time()

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np                       # noqa: E402
import networkx as nx                    # noqa: E402
from _model.mera import mera_init, build_mera_graph            # noqa: E402
from _model.stage5 import curvature_report, geometry_controls  # noqa: E402
from _metric.extras import _pctile_rank                        # noqa: E402

# 真调阶段 1 的生成器；它在 import 时会打印自己的报告，这里静音以免污染本日志
import io                                # noqa: E402
import contextlib                        # noqa: E402
with contextlib.redirect_stdout(io.StringIO()):
    import _v16_smoke_t6_tiling as TILING   # noqa: E402

_T_IMPORT = time.time()

VERSION_TAG = 'v16-smoke-t6-l5measure-1'
STAMP = '2026-10-04'
L_MERA = 32                              # main.py:143/191 用的是 L=32
CHI_MERA, SEED_MERA = 2, 0
DUAL_RINGS = 5                           # 阶段 1 的补丁（§0.1.5）
K_JS = 20                                # deg_js 支撑固定 0..20（>20 并入末格）
DS_TS = np.arange(1, 11, dtype=float)    # d_s 窗口固定 t=1..10（全图同一窗口）
SWAP_SEED = 7                            # ④b 保度重连的固定 seed
GUARD_TOL = 1e-12                        # 交叉校验守卫容差
BAND = (5.0, 95.0)                       # 判据 ②
SELF_MED_LO, SELF_MED_HI = 25.0, 75.0    # ④a：自比百分位中位数须落此区间

_MEAS = ('d_bar', 'n_triangles', 'deg_js', 'girth',
         'diameter', 'd_s', 'forman_mean', 'or_mean')
_LINES = []
_FAIL = []


def say(s):
    print(s)
    sys.stdout.flush()
    _LINES.append(s)


def _rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024.0 / 1024.0
    except Exception:
        return float('nan')


# ---------------------------------------------------------------- 8 个测度

def m_deg_js(G):
    """度分布与**同均值 Poisson** 的 JS 散度；支撑固定 k=0..K_JS（>K_JS 并入末格）。"""
    d = np.array([deg for _, deg in G.degree()], dtype=float)
    if d.size == 0:
        return float('nan')
    h = np.zeros(K_JS + 1)
    for x in np.clip(d.astype(int), 0, K_JS):
        h[x] += 1.0
    p = h / h.sum()
    lam = float(d.mean())
    lp = np.empty(K_JS + 1)
    lp[0] = -lam
    for k in range(1, K_JS + 1):
        lp[k] = lp[k - 1] + np.log(lam) - np.log(k)
    q = np.exp(lp)
    q = q / q.sum()
    m = 0.5 * (p + q)

    def _kl(a, b):
        msk = a > 0
        return float(np.sum(a[msk] * np.log(a[msk] / b[msk])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def m_girth(G):
    """最短环长（**精确**：逐边删边求 u-v 最短路）；无环记 +inf。
    刻意不用 `nx.cycle_basis` -- 后者非唯一（§0.1.11 ③ 的「非唯一」排除条件）。"""
    if G.number_of_edges() == 0 or nx.is_forest(G):
        return float('inf')
    GC = G.copy()
    best = float('inf')
    for u, v in list(G.edges()):
        GC.remove_edge(u, v)
        try:
            dd = nx.shortest_path_length(GC, u, v)
            if dd + 1 < best:
                best = dd + 1
                if best == 3:
                    break
        except nx.NetworkXNoPath:
            pass
        GC.add_edge(u, v)
    return float(best) if np.isfinite(best) else float('inf')


def m_diameter(G):
    """直径；不连通记 +inf（口径自明，不是「不可算」）。"""
    if G.number_of_nodes() == 0:
        return float('nan')
    if not nx.is_connected(G):
        return float('inf')
    return float(nx.diameter(G))


def m_ds(G):
    """谱维数：组合 Laplacian 本征值 -> p(t) = (1/V) sum_k exp(-lam_k t)，
    对 ln p vs ln t 最小二乘，d_s = -2*slope。**窗口固定 t=1..10，全图同一窗口。**"""
    if G.number_of_nodes() < 4:
        return float('nan')
    Lap = nx.laplacian_matrix(G).toarray().astype(float)
    ev = np.linalg.eigvalsh(Lap)
    p = np.array([float(np.mean(np.exp(-ev * t))) for t in DS_TS])
    p = np.clip(p, 1e-300, None)
    x, y = np.log(DS_TS), np.log(p)
    A = np.vstack([x, np.ones_like(x)]).T
    slope = float(np.linalg.lstsq(A, y, rcond=None)[0][0])
    return -2.0 * slope


def values_of(G, rep):
    """8 个测度的一次性取值。A 组与 C 组直接读 `curvature_report` 的字段（同一把尺子）。"""
    d = {
        'd_bar': float(rep['d_bar']),
        'n_triangles': float(rep['n_triangles']),
        'forman_mean': float(rep['forman_mean']),
        'or_mean': float(rep['or_mean']),
    }
    d['deg_js'] = m_deg_js(G)
    d['girth'] = m_girth(G)
    d['diameter'] = m_diameter(G)
    d['d_s'] = m_ds(G)
    return d


# ------------------------------------------------- 对照族：登记在案的复制

def rebuild_controls(G):
    """按 `_model/stage5.py:281-301` 的 recipe 重建对照**图**（该函数只返回报表，不暴露图）。
    返回 [(family, instance, H), ...]，顺序与 `geometry_controls` 的 reports 一一对应。"""
    V, E = G.number_of_nodes(), G.number_of_edges()
    k_ws = max(2, 2 * int(round(E / V))) if V else 2
    h_tree = max(1, int(np.log2(V + 1)) - 1)
    out = []
    for fam in ('gnm', 'ws', 'tree'):
        for s in range(3):
            if fam == 'gnm':
                H = nx.gnm_random_graph(V, E, seed=100 + s)
            elif fam == 'ws':
                H = nx.watts_strogatz_graph(V, k_ws, 0.1, seed=200 + s)
            else:
                H = nx.balanced_tree(2, h_tree)
            out.append((fam, s, H))
    return out


def guard_controls(rebuilt, reports):
    """交叉校验守卫：重建的对照必须逐实例复现报表里的 forman_mean / d_bar / n_triangles。"""
    if len(rebuilt) != len(reports):
        return False, '实例数不符 %d vs %d' % (len(rebuilt), len(reports))
    worst = 0.0
    for (fam, s, H), rep in zip(rebuilt, reports):
        r2, _, _ = curvature_report(H, '守卫 %s #%d' % (fam, s))
        for key in ('forman_mean', 'd_bar'):
            worst = max(worst, abs(float(r2[key]) - float(rep[key])))
        if int(r2['n_triangles']) != int(rep['n_triangles']):
            return False, '三角数不符 %s#%d' % (fam, s)
    return (worst <= GUARD_TOL), 'forman_mean/d_bar 最大偏差 %.3e (容差 %.1e)' % (worst, GUARD_TOL)


# ---------------------------------------------------------------- 流水线

def self_control(ctrl_vals):
    """④a 对照自比：每个对照实例轮流当被测对象，在**其余 8 个**里算百分位。"""
    out = {}
    for m in _MEAS:
        vals = np.array([v[m] for v in ctrl_vals], dtype=float)
        pcts = []
        for i in range(len(vals)):
            others = np.delete(vals, i)
            pcts.append(_pctile_rank(float(vals[i]), list(others)))
        pcts = np.array([p for p in pcts if np.isfinite(p)], dtype=float)
        out[m] = (float(np.median(pcts)) if pcts.size else float('nan'), int(pcts.size))
    return out


def run_object(tag, G, meas_vals_obj, ctrl_vals):
    """一个对象的一套：逐测度算百分位 + n_eff。"""
    row = {}
    for m in _MEAS:
        sample = [v[m] for v in ctrl_vals]
        n_eff = int(np.sum(np.isfinite(np.array(sample, dtype=float))))
        pct = _pctile_rank(meas_vals_obj[m], sample)
        row[m] = {'obj': meas_vals_obj[m], 'pct': pct, 'n_eff': n_eff}
    return row


def _finish(code):
    say('用时 %.2f s   RSS %.1f MB' % (time.time() - _T_PROC, _rss_mb()))
    say('=' * 78)
    with open(os.path.join(_HERE, '_t6_l5measure.log'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(_LINES) + '\n')


def main():
    say('=' * 78)
    say('%s  阶段 4（接 L5 几何测度）  stamp=%s' % (VERSION_TAG, STAMP))
    say('判据 = v16.8_plan.md §0.1.11（冻结于开跑前）；非有限处理 = A')
    say('import 用时 %.1f s' % (_T_IMPORT - _T_PROC))
    say('=' * 78)

    # ---------- 对象 ----------
    mera32, _ = mera_init(L_MERA, CHI_MERA, seed=SEED_MERA)
    G1 = build_mera_graph(mera32)
    coords, faces, tstats = TILING.build(DUAL_RINGS)
    G2 = nx.Graph()
    G2.add_nodes_from(range(len(coords)))
    for f in faces:
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            if a != b:
                G2.add_edge(a, b)
    say('[O1] MERA-L32 : |V|=%d |E|=%d' % (G1.number_of_nodes(), G1.number_of_edges()))
    say('[O2] {7,3}    : |V|=%d |E|=%d (阶段 1 build(5)，1-骨架)'
        % (G2.number_of_nodes(), G2.number_of_edges()))

    objects = [('O1-MERA-L32', G1), ('O2-{7,3}-patch', G2)]
    ctx = {}
    for tag, G in objects:
        say('-' * 78)
        say('对象 %s：生成对照族并交叉校验' % tag)
        reports = geometry_controls(G)
        rebuilt = rebuild_controls(G)
        ok, msg = guard_controls(rebuilt, reports)
        say('  [守卫] 交叉校验 %s : %s' % ('通过' if ok else '**不过**', msg))
        if not ok:
            _FAIL.append('%s 守卫不过' % tag)
            ctx[tag] = None
            continue
        say('  对照族就绪 (9 实例, 3 族)')
        ctx[tag] = {'G': G, 'reports': reports, 'rebuilt': rebuilt}

    if _FAIL:
        say('守卫未过 => 按 §0.1.11 ⑧ **停工上报，不落落支**。')
        _finish(3)
        return 3

    # ---------- ④a 对照自比（两个对象分别做） ----------
    for tag, G in objects:
        c = ctx[tag]
        ctrl_vals = []
        for (fam, s, H), rep in zip(c['rebuilt'], c['reports']):
            ctrl_vals.append(values_of(H, rep))
        c['ctrl_vals'] = ctrl_vals
        c['self'] = self_control(ctrl_vals)

    say('=' * 78)
    say('④a 对照自比（每个测度：9 个自比百分位的中位数，须落 [%.0f, %.0f]）'
        % (SELF_MED_LO, SELF_MED_HI))
    bad_self = []
    for tag, G in objects:
        c = ctx[tag]
        say('  [%s]' % tag)
        for m in _MEAS:
            med, n = c['self'][m]
            flag = '' if (np.isfinite(med) and SELF_MED_LO <= med <= SELF_MED_HI) else '  <== 出界'
            if flag:
                bad_self.append((tag, m))
            say('    %-13s 中位数 %6.2f%%  (n=%d)%s' % (m, med, n, flag))

    # ---------- 逐对象：对象读数 + 百分位 ----------
    say('=' * 78)
    for tag, G in objects:
        c = ctx[tag]
        rep_obj, _, _ = curvature_report(G, '对象 %s' % tag)
        c['obj_vals'] = values_of(G, rep_obj)
        c['row'] = run_object(tag, G, c['obj_vals'], c['ctrl_vals'])
        say('对象 %s 读数（口径：接线图，不依赖态）' % tag)
        say('  %-13s %-16s %8s %8s %6s' % ('测度', '对象值', '百分位', 'n_eff', '状态'))
        for m in _MEAS:
            r = c['row'][m]
            if r['n_eff'] < 9:
                st = '只登记(n_eff<9)'
            elif not np.isfinite(r['pct']):
                st = '不可算'
            elif r['pct'] < BAND[0] or r['pct'] > BAND[1]:
                st = '**端点外**'
            else:
                st = '仍零'
            say('  %-13s %-16.6g %8.1f%% %8d %6s'
                % (m, r['obj'], r['pct'], r['n_eff'], st))

    # ---------- ④b 保度随机重连 ----------
    say('=' * 78)
    say('④b 保度随机重连（对每个对象重连后重跑；重连保 |V|、|E| => **复用同一组对照**）')
    void = {}
    for tag, G in objects:
        c = ctx[tag]
        H = G.copy()
        E = H.number_of_edges()
        try:
            done = nx.double_edge_swap(H, nswap=10 * E, max_tries=100 * E, seed=SWAP_SEED)
        except Exception as e:
            say('  [%s] 重连**不可行** (%s: %s) => 该项登记为「无法做负对照」，不据此作废任何测度'
                % (tag, type(e).__name__, e))
            continue
        rep_r, _, _ = curvature_report(H, '重连 %s' % tag)
        vr = values_of(H, rep_r)
        row_r = run_object(tag + '-rewired', H, vr, c['ctrl_vals'])
        say('  [%s] 完成 %s 次交换；重连图百分位：' % (tag, done))
        for m in _MEAS:
            r = row_r[m]
            out = bool(np.isfinite(r['pct']) and (r['pct'] < BAND[0] or r['pct'] > BAND[1]))
            if out:
                void.setdefault(m, []).append(tag)
            say('    %-13s %8.1f%%  n_eff=%d  %s'
                % (m, r['pct'], r['n_eff'], '<== 端点外 => 该测度作废' if out else '在区间内'))

    # ---------- 落支 ----------
    say('=' * 78)
    signals, registered = [], []
    for tag, G in objects:
        c = ctx[tag]
        for m in _MEAS:
            r = c['row'][m]
            if m in void:
                continue
            if r['n_eff'] < 9:
                registered.append((tag, m, r['pct'], r['n_eff'], 'n_eff<9'))
                continue
            if np.isfinite(r['pct']) and (r['pct'] < BAND[0] or r['pct'] > BAND[1]):
                signals.append((tag, m, r['pct']))
            else:
                registered.append((tag, m, r['pct'], r['n_eff'], '仍零'))

    say('④b 作废的测度: %s' % (void if void else '无'))
    say('端点外的（信号）: %s' % (signals if signals else '无'))
    say('其余登记项: %d 条' % len(registered))
    if bad_self:
        say('④a 出界项(仅登记，不作废): %s' % bad_self)

    self_bad_all = [t for t, _ in objects
                    if all(not (np.isfinite(ctx[t]['self'][m][0])
                                and SELF_MED_LO <= ctx[t]['self'][m][0] <= SELF_MED_HI)
                           for m in _MEAS)]
    if self_bad_all:
        say('④a 判定：对象 %s 的**全部**测度自比中位数出界 => **流程不可信**，不落落支。' % self_bad_all)
        _finish(3)
        return 3

    verdict = 'T6-L5测度有信号' if signals else 'T6-L5测度仍零'
    say('落支 = %s' % verdict)
    say('失败项 = %s' % (_FAIL if _FAIL else '无'))
    _finish(0)
    return 0


if __name__ == '__main__':
    sys.exit(main())
