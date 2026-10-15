> **spiral-emergence: A negative-control ledger for multi-stage numerical pipelines**  
> © 2024-2026 Jianzhong Yan — All commits DCO-signed (`Signed-off-by`).  
> Version: v16.9 | Zenodo concept DOI (all versions): [10.5281/zenodo.22810161](https://doi.org/10.5281/zenodo.22810161) | v16.9 DOI: [10.5281/zenodo.23278413](https://doi.org/10.5281/zenodo.23278413)
>
> **Note on naming.** The project does **not** claim to study spiral geometry or emergence phenomena. The repository name `spiral-emergence` is retained for historical reasons; the scientific contribution is the ledger methodology and its application to tensor-network benchmarks.
>
> **Document scope** This document describes only the current state of the codebase; version history is logged in [`CHANGELOG.md`](CHANGELOG.md). This document is cited by content, not by line number.



<div align="center">

# A negative-control ledger for multi-stage numerical pipelines, demonstrated on tensor-network benchmarks
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22810161.svg)](https://doi.org/10.5281/zenodo.22810161)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0007--0045--7114-green?logo=orcid)](https://orcid.org/0009-0007-0045-7114)
[![DCO](https://img.shields.io/badge/DCO-Signed-brightgreen.svg)](https://developercertificate.org/)
[![Release](https://img.shields.io/github/v/release/yanjianzhong/spiral-emergence)](https://github.com/yanjianzhong/spiral-emergence/releases)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)

</div>

English | [简体中文](./README.zh.md)



## Overview

This repository implements a **negative-control ledger** for multi-stage numerical pipelines, demonstrated on a critical transverse-field Ising chain and tensor-network (MERA) benchmarks.

Such pipelines are ubiquitous in computational physics: ground-state preparation feeds into entanglement extraction, which feeds into scaling-law fitting, which feeds into geometric or dynamical analysis. Each stage inherits the output—and the silent errors—of the previous one. Yet standard verification practice is ad hoc: isolated unit tests on individual stages, with no systematic mechanism to detect errors that propagate across stage boundaries.

This framework addresses that gap by equipping each stage with falsifiable negative controls and honestly logging every outcome—including failures. It provides:

- **Per-stage negative controls** with pre-registered pass/fail criteria, not post-hoc sanity checks
- **A completeness guard** that verifies every stage has at least one registered control
- **An audit table** in which positive results, negative results, zero results, and "no object" (structurally inapplicable) entries are recorded at equal prominence

We demonstrate the framework on two structurally distinct systems:

1. **Critical transverse-field Ising chain** ($h/J = 1$, $c = 1/2$): Seven pipeline stages, each with negative controls, achieving full 7/7 coverage. The central charge is estimated by two independent estimator families (entanglement-spectrum and energy-spectrum), converging to 0.212% agreement. The ledger includes 78 guards, 27 metrics, and 13 registered negative controls.

2. **AKLT chain** ($S = 1$ VBS, Haldane gap, $c = 0$): A non-critical system with no CFT counterpart, added to test whether the framework generalizes beyond criticality. The gapped-system criterion correctly identifies $c = 0$ (open chain: exact zero; ring: 0.0036), and three negative controls (bond-dimension reduction, non-isometric injection, bidirectional isometry pairing) fail as designed—confirming the framework is neither a rubber stamp nor a blanket rejector.

A distinctive feature of this work is its **discipline of honest boundaries**: every limitation—finite-size truncation, source-code-level defects in cited tensor network constructions, zero-result bridges between stages, parameter-dependent analytical claims—is registered in the audit table with the same visibility as positive results. We consider this rigor to be the contribution itself, not supplementary good practice.


## Key results (honest summary)

| Layer | Physical target | Maturity | Key result |
|-------|----------------|----------|------------|
| L1 | Equal-weight vacuum state | B | Overlap with exact ground state monotonic to 0.999024 at h/J=32; `dim_full` consumed |
| L2 | Critical TF-Ising, central charge | **A** | Two independent estimators: entanglement-spectrum family → 0.5; energy-spectrum family compatible to 0.212% (L ≤ 16) |
| L3 | Recursive self-reference | B | Variant B rank-1 in 5/5 N-values; Variant A only at N≥48; transient exponent p median 0.4876 |
| L4 | MERA holographic projection | C | Isometry error ~10⁻¹⁶; curvature in 33rd percentile of null models; upper-layer interface not connected |
| L5 | Discrete geometry from entanglement | C | Controlled recalculation eliminates 252× confounding (null result); 8 geometric measures × 2 objects → no surviving signal |
| L6 | Gray-Scott on emergent geometry | B | Pattern emergence verified; stability condition auto-derives grid; parameters hand-set |
| L7 | Second-order self-reference (consciousness analog) | C | Variant B Jacobian spectral radius 0.9412 ≠ 1 (non-degenerate); analytic closed form holds at 12/12 points but parameter-dependent |

**Ledger status (frozen full-pipeline run):**
- Indicators: 19/19 met (27 registered, 8 diagnostic)
- Guards: 57/57 passed (100.0%; 78 registered, 21 diagnostic)
- Negative controls: 13 (push-away 6 + null-model 1 + two-path 5 + upper-bound 1)
- Coverage: 7/7 layers, gap `[]`

## Requirements

- **Python:** 3.10+ (tested on 3.13.9)
- **Dependencies** (pinned in `v16/requirements.txt`):
  - `matplotlib==3.11.2`
  - `networkx==3.5`
  - `numpy==2.5.3`
  - `quimb==1.15.0`
  - `scipy==1.18.1`
  - `torch==2.9.0`

## Installation
```bash
git clone https://github.com/yanjianzhong/spiral-emergence.git
cd spiral-emergence
pip install -r v16/requirements.txt
```

## Quick start
```bash
#Core seven-stage pipeline (run from v16/ directory)
cd v16
python spiral_model_v16.py > _v16_run.log 2>&1
cat _v16_run.log

#Smoke tests / self-checks (fast: 20, slow: 7, full: 1 = 28 total)
python _v16_run_all.py --list # list all 28 registered entries
python _v16_run_all.py --full # run all 28 entries (~4268 s)

#Freeze check: verify v15 guards unchanged
python _v16_freeze_check.py
```

**Expected output summary (from frozen reference run `_runall_v16.3_r1.log`):**
- Layer 1 (internal self-consistency): 6/6 met (2 diagnostic)
- Layer 2 (microscopic physical benchmark): 13/13 met
- Layer 3 (macroscopic structural benchmark): 0/0 (6 diagnostic)
- Total: 19/19 met
- Guard pass rate = 57/57 = 100.0% (21 diagnostic, 16 designed-to-fail)
- Coverage: 7/7 layers, no-control layers: [], 13 negative controls
- Registry: 27 indicators, 78 guards, 13 negative controls


## Directory structure
```bash
.
├── README.md # This file
├── CHANGELOG.md # Version history
├── LICENSE / NOTICE # Apache-2.0 + mandatory distribution notices
├── CITATION.cff # Citation metadata
├── CONTRIBUTING.md # Contribution guidelines (DCO + audit guard rules)
├── RELEASE_WORKFLOW.md # Release procedure
│
├── v14/ # Frozen historical snapshot
├── v15/ # Frozen (readings/thresholds/denominators unchanged)
└── v16/ # Current version
├── spiral_model_v16.py # Facade → re-exports from _model/
├── spiral_metric_v16.py # Facade → re-exports from _metric/
├── requirements.txt
├── _model/ # Core model code (11 .py files)
│ ├── hyper.py # Hyperinvariant tensor network (not exported to facade)
│ └── tnr.py # TNR (Evenbly–Vidal 2015) building blocks
├── _metric/ # Indicators and guards (9 .py files)
├── _baseline/ # Pre-split originals (frozen)
├── _v16_run_all.py # Single runner (fast/slow/full)
├── _v16_freeze_check.py # v15→v16 freeze verification
├── _v16_cmera_gaussian.py # Gaussian cMERA (B2)
├── _v16_smoke*.py # Individual probes and diagnostic scripts
├── spiral_v16_audit.md # Current audit table
├── spiral_v16_说明.md # Chinese human-readable guide
├── Hyper.tex / TNR.tex / cMERA_Entropy_rev.tex # Source snapshots of cited literature
├── data/ # Cached data (reproducible)
└── result/ # Output figures and JSON
```

## Negative-control methodology

Every stage is equipped with at least one falsifiable negative control. Controls include:
- **Push-away controls:** perturb input, verify output degrades (L1, L4)
- **Null-model controls:** compare against randomized baselines (L4 curvature, L5 geometry)
- **Two-path controls:** independent implementations cross-checked (L6)
- **Upper-bound controls:** verify quantities stay within theoretical limits

All controls, guard assertions, and diagnostic items are registered in a runtime registry (`registry.py`). The ledger enforces: **no silent failures, no retroactive upgrades from diagnostic to indicator.**

## Honest boundaries (key limitations)

1. **L1 downstream:** Overlap-based control measures sensitivity scale, not independent falsification.
2. **L2 energy-spectrum estimator:** Does not converge monotonically with L; compatible to 0.212% only for L ≤ 16. Residuals are not O(1/L) finite-size corrections.
3. **L4 upper-layer interface:** Referenced hyperinvariant tensor network has structural issues at source (matrix Y non-unitary); local dimension incompatibility (χ=16 vs hardcoded 2) prevents bridging to central charge fitter.
4. **L5 bridge:** L4→L5 causal connection remains empty; null results across 8 geometric measures.
5. **L7 parameter dependence:** Analytic closed form for Jacobian spectral radius holds but is parameter-sensitive (g>1 causes instability).
6. **General:** This is a methodological demonstration on benchmark systems, not a claim about fundamental physics of consciousness, life, or spacetime emergence.

## Citation

If you use this software, please cite:
```bibtex
@software{yan2026spiral,
    author = {Yan, JianZhong},
    title = {spiral-emergence: A negative-control ledger for multi-stage numerical pipelines},
    version = {v16.9},
    doi = {10.5281/zenodo.23278413},
    url = {https://github.com/yanjianzhong/spiral-emergence}
}
```

## License

Core numerical code: Apache 2.0 (see [LICENSE](LICENSE))  
Documentation and theory: CC BY-SA 4.0  
Data caches and results: CC0 / per-file declaration

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for DCO sign-off requirements and audit guard conventions.

---

<br>

<p align="center">
  <sub><i>Change manifests as periodic polarization, unified in emptiness;</i></sub><br>
  <sub><i>growth spirals upward, originating from the end.</i></sub>
</p>

<p align="center">— JianZhong Yan</p>
