# -*- coding: utf-8 -*-
"""
_v16_smoke_cmera.py —— v16 · B2 可行性冒烟: 离散化 cMERA 最小原型

本脚本只回答一个问题: **"自己写 cMERA" 这条路走不走得通**
(quimb 1.15.0 已实测无 cMERA / 无 branching, 故必须自写)。四个子问题:

  Q1 能否构造    离散 cMERA 态矢量可建、能量有限、起点合理
  Q2 能否优化    变分优化能把能量压向精确基态
  Q3 能否算 g_uu 径向度量可算、非平凡、有尺度剖面
  Q4 资源        用时与峰值内存在 16G 笔记本上可接受

边界声明 (必读, 防止把冒烟当成物理结果):
  - 本脚本**不产出任何可登记结论**。输出的"g_uu 依赖 h/J" 是**构造使然** ——
    A 参数是被目标态优化出来的, 所以"态依赖"是同义反复。
    完整方案里的真检验是"**复现预期几何形式**"(临界→AdS 型 / 非临界→IR-capped),
    那需要多个 h/J 的优化结果与形式拟合, 不是本冒烟的内容。
  - 新写的 torch 能量函数必须与 v15 的精确对角化**交叉校验**;
    校验不过, 本冒烟直接判失败(不把"能跑"当成"跑对了")。

自包含: 除数值库外无新依赖(numpy/torch 已是 v15 硬依赖)。
v15 只作**外部参照**导入, 用于第 (1) 步的交叉校验, 不改动 v15 任何代码。
"""

import os
import sys
import time
import tracemalloc

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_V15 = os.path.join(_HERE, '..', 'v15')

torch.set_default_dtype(torch.float64)

# ---- 版本标记 -------------------------------------------------------------
_VERSION_TAG = 'v16-smoke-cmera-1'
# cMERA 离散化口径(本实现自己的约定, 不是文献的严格连续极限, 见文末边界):
#   态: |Psi_k> 由 k 层门依次施加于 |0...0>(IR 直积态)
#   每层: 先解纠缠器(2 格点, brickwork 交替) 再标度变换器(1 格点)
#   g_uu(k) = ||G_k|Psi_k>||^2 - |<Psi_k|G_k|Psi_k>|^2   (Fubini-Study 方差形式)
#   其中 G_k = 该层所有反厄米生成元 A 之和, 且 U_gate = expm(A)


# ===========================================================================
# 1. TFI 能量 (可微, 供 torch 自动微分)
#    H = -J Σ_i σz_i σz_{i+1} - h Σ_i σx_i, 周期边界
#    位序与 v15 一致: 张量轴 i <-> 站点 i
# ===========================================================================
def _sign_axis(L, i):
    """σz 在第 i 轴上的对角符号张量, 形状 (1,..,2,..,1)。"""
    shape = [1] * L
    shape[i] = 2
    return torch.tensor([1.0, -1.0], dtype=torch.float64).reshape(shape)


def tfi_energy(psi, L, J=1.0, h=1.0):
    """<psi|H|psi> / <psi|psi>。psi: complex128, 形状 (2,)*L。"""
    prob = (psi.conj() * psi).real
    nrm = prob.sum()
    zz = 0.0
    for i in range(L):
        j = (i + 1) % L
        zz = zz + (prob * _sign_axis(L, i) * _sign_axis(L, j)).sum()
    xf = 0.0
    for i in range(L):
        xf = xf + (psi.conj() * torch.flip(psi, dims=[i])).sum()
    return (-J * zz - h * xf.real) / nrm


# ===========================================================================
# 2. 门的作用 (2 格点 / 1 格点), 及生成元
# ===========================================================================
def _apply_2site(psi, U, i, j, L):
    """把 4x4 酉 U 作用在轴 (i, j) 上。"""
    out = torch.tensordot(U.reshape(2, 2, 2, 2), psi, dims=([2, 3], [i, j]))
    return out.movedim((0, 1), (i, j))


def _apply_1site(psi, U, i, L):
    """把 2x2 酉 U 作用在轴 i 上。"""
    out = torch.tensordot(U, psi, dims=([1], [i]))
    return out.movedim(0, i)


def _gen_and_unitary(params, n):
    """(A, U): A 反厄米生成元, U = expm(A) 酉。params: (2*n*n,) 实数。"""
    re = params[:n * n].reshape(n, n)
    im = params[n * n:].reshape(n, n)
    P = torch.complex(re, im)
    A = P - P.conj().transpose(-1, -2)      # A† = -A
    return A, torch.matrix_exp(A)


# ===========================================================================
# 3. 离散化 cMERA
# ===========================================================================
class CMERA:
    """最小离散 cMERA。L 为偶数; n_layers 为尺度层数。"""

    def __init__(self, L, n_layers, seed=0, init_scale=0.05):
        assert L % 2 == 0, 'L 需为偶数 (brickwork 需要)'
        self.L, self.n_layers = L, n_layers
        g = torch.Generator().manual_seed(seed)
        n_bond = L // 2
        # 2 格点解纠缠器: 每层 n_bond 个, 每个 32 个实参 (4x4 复矩阵 = 2*16)
        self.p2 = (torch.randn(n_layers, n_bond, 32, generator=g,
                               dtype=torch.float64) * init_scale).requires_grad_(True)
        # 1 格点标度变换器: 每层 L 个, 每个 8 个实参 (2x2 复矩阵 = 2*4)
        self.p1 = (torch.randn(n_layers, L, 8, generator=g,
                               dtype=torch.float64) * init_scale).requires_grad_(True)

    def _bonds(self, k):
        """第 k 层的 brickwork 键对 (奇偶层交错)。"""
        L = self.L
        return [(i, (i + 1) % L) for i in range(k % 2, L, 2)]

    def forward(self):
        """返回 (末态 psi, 各层 g_uu)。"""
        L = self.L
        psi = torch.zeros([2] * L, dtype=torch.complex128)
        psi[(0,) * L] = 1.0                      # IR 直积态
        g_uu = []
        for k in range(self.n_layers):
            As2 = [(pair, _gen_and_unitary(self.p2[k, m], 4))
                   for m, pair in enumerate(self._bonds(k))]
            As1 = [(i, _gen_and_unitary(self.p1[k, i], 2)) for i in range(L)]

            # --- g_uu(k): 生成元在 |Psi_k> 上的 Fubini-Study 方差 ---
            Gpsi = 0.0
            for pair, (A, _) in As2:
                Gpsi = Gpsi + _apply_2site(psi, A, pair[0], pair[1], L)
            for i, (A, _) in As1:
                Gpsi = Gpsi + _apply_1site(psi, A, i, L)
            g = ((Gpsi.conj() * Gpsi).sum().real
                 - (psi.conj() * Gpsi).sum().abs() ** 2)
            g_uu.append(g)

            # --- 推进到下一尺度 ---
            for pair, (_, U) in As2:
                psi = _apply_2site(psi, U, pair[0], pair[1], L)
            for i, (_, U) in As1:
                psi = _apply_1site(psi, U, i, L)
        return psi, torch.stack(g_uu)

    def params(self):
        return [self.p2, self.p1]


# ===========================================================================
# 4. 单个 (L, h/J) 点: 构造 -> 优化 -> g_uu 剖面
# ===========================================================================
def run_point(L, h, n_layers, steps, J=1.0, seed=0, verbose=True):
    model = CMERA(L, n_layers, seed=seed)
    opt = torch.optim.Adam(model.params(), lr=0.02)

    t0 = time.time()
    psi, g_uu = model.forward()
    e_init = float(tfi_energy(psi, L, J, h))
    t_first = time.time() - t0

    for _ in range(steps):
        opt.zero_grad()
        psi, _ = model.forward()
        e = tfi_energy(psi, L, J, h)
        e.backward()
        opt.step()

    t_total = time.time() - t0
    with torch.no_grad():
        psi, g_uu = model.forward()
        e_fin = float(tfi_energy(psi, L, J, h))
        nrm = float((psi.conj() * psi).sum().real)
        g_prof = [float(x) for x in g_uu]

    if verbose:
        print(f"  L={L} h/J={h} 层数={n_layers} 步数={steps}")
        print(f"    E_init = {e_init:+.6f}   E_final = {e_fin:+.6f}")
        print(f"    <psi|psi> = {nrm:.9f}   (应为 1)")
        print(f"    g_uu 剖面 = [{', '.join(f'{v:.3e}' for v in g_prof)}]")
        print(f"    首帧 {t_first:.2f}s, 合计 {t_total:.1f}s")
    return {'E_init': e_init, 'E_final': e_fin, 'g_uu': g_prof,
            'norm': nrm, 'seconds': t_total}


# ===========================================================================
# 5. 主流程
# ===========================================================================
def main():
    print(f"=== v16 冒烟 · 离散化 cMERA 可行性  [{_VERSION_TAG}] ===")
    print(f"torch {torch.__version__} / numpy {np.__version__}")

    # ---- (1) 交叉校验: 新写的 torch 能量 vs v15 精确对角化 ----
    print("\n[1] torch 能量函数 与 v15 精确对角化 交叉校验")
    try:
        sys.path.insert(0, os.path.abspath(_V15))
        from spiral_model_v15 import exact_ground_state, jw_ground_energy

        L_chk, h_chk = 8, 1.0
        e_exact, gs_vec = exact_ground_state(L_chk, 1.0, h_chk)   # 返回 (E0, gs)
        e_jw = jw_ground_energy(L_chk, 1.0, h_chk)               # 独立第二条路径
        # 把 ED 基态装成 (2,)*L 张量, 用本脚本的 tfi_energy 复算。
        # 位序: v15 里站点 i <-> 二进制位 (L-1-i), 故 C 序 reshape 后 轴 a <-> 站点 a,
        # 与本脚本 "轴 i <-> 站点 i" 的约定一致 —— 这一步就是把该约定查实。
        vec = np.asarray(gs_vec).reshape([2] * L_chk)
        psi_ref = torch.tensor(vec, dtype=torch.complex128)
        e_torch = float(tfi_energy(psi_ref, L_chk, 1.0, h_chk))
        dev = abs(e_torch - e_exact)
        ref_ok = dev < 1e-9
        print(f"    L={L_chk}, h/J={h_chk}: ED(E0)          = {e_exact:.12f}")
        print(f"    v15 Jordan-Wigner 独立路径 = {e_jw:.12f}"
              f"   (|ED-JW| = {abs(e_exact - e_jw):.2e})")
        print(f"    本脚本 torch 复算 ED 基态  = {e_torch:.12f}   偏差 = {dev:.2e}")
        print(f"    -> {'一致' if ref_ok else '不一致 (冒烟判失败)'}")
        if not ref_ok:
            print("    位序或符号约定有误, 后续结果不可信。")
            return 1
    except Exception as exc:                       # noqa: BLE001
        print(f"    !! 无法导入 v15 做交叉校验: {type(exc).__name__}: {exc}")
        print("    交叉校验未完成 —— 按边界声明, 本冒烟**不得**给出通过结论。")
        return 2

    # ---- (2) 可行性四问 ----
    tracemalloc.start()
    print("\n[2] L=8 原型 (快速)")
    r8 = run_point(L=8, h=1.0, n_layers=4, steps=300)     # noqa: F841
    print("\n[3] L=16 原型 (目标规模)")
    r16 = run_point(L=16, h=1.0, n_layers=5, steps=200)
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("\n[4] 资源")
    print(f"    python 侧峰值内存(tracemalloc) = {peak / 1e6:.1f} MB")
    print(f"    L=16 单点用时 = {r16['seconds']:.1f}s")

    # ---- (3) 结论: 只回答"走不走得通", 不回答物理 ----
    print("\n=== 可行性判定 (只回答通路, 不产出物理结论) ===")
    q1 = r16['norm'] > 0.999
    q2 = r16['E_final'] < r16['E_init']
    q3 = all(v > 0 for v in r16['g_uu']) and len(set(r16['g_uu'])) > 1
    q4 = r16['seconds'] < 600
    print(f"    Q1 构造得出 (归一)      : {'通过' if q1 else '不通过'}")
    print(f"    Q2 优化压得下去         : {'通过' if q2 else '不通过'}"
          f"  (E {r16['E_init']:+.4f} -> {r16['E_final']:+.4f})")
    print(f"    Q3 g_uu 可算且非平凡    : {'通过' if q3 else '不通过'}")
    print(f"    Q4 资源可接受 (<600s)   : {'通过' if q4 else '不通过'}"
          f"  ({r16['seconds']:.1f}s)")
    print("\n    边界: Q2 '压得下去' 只说明优化在动, **不等于**收敛到基态。")
    print("          g_uu 的态依赖是构造使然(同义反复), 不构成物理结论。")
    print("          真检验(复现 AdS 型 / IR-capped)不在本冒烟范围。")
    return 0 if (q1 and q2 and q3 and q4) else 3


if __name__ == '__main__':
    sys.exit(main())
