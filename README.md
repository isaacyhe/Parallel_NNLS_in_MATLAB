# Parallel NNLS for MPI

This repository contains MEX (MATLAB Executable) implementations of high-performance Non-Negative Least Squares (NNLS) solvers accelerated on multicore CPU (OpenMP) and GPU (CUDA).
It accompanies a peer-reviewed conference paper and is mainly used for Magnetic Particle Imaging reconstruction in MATLAB.

## Overview

Solves **min ||Cx − d||² subject to x ≥ 0** with six distinct algorithms across MATLAB, OpenMP-C++, and CUDA — 30 implementations in total.

The collection covers six algorithm families across three platforms:

| Family | Algorithm class | Convergence on PSF-class Tikhonov (κ ≈ 10¹⁰) |
|---|---|---|
| **lsqnonneg** (Lawson-Hanson 1974) | active set | converges in O(100) outer iterations |
| **CAS** — same algorithm, C++/CUDA | active set | same, reimplemented for GPU acceleration |
| **FAST-NNLS** (Bro & de Jong 1997 / Cobb 2025) | active set | converges in O(100) outer iterations |
| **Classic GP** (Goldstein-Levitin-Polyak 1964) | projected gradient | needs ~6.7×10¹⁰ iters for 5% error, ~10¹¹ for 1% |
| **ADMM** (Glowinski 1975 / Boyd 2011) | splitting + Cholesky | converges in ~500 iterations (relErr ~1.9×10⁻³) |
| **CG** (Hestenes-Stiefel 1952) | Krylov | needs ~55,000 iters for 5% error, ~85,000 for 1% |

All MEX solvers expose a MATLAB interface: `x = solver(C, d)` or `x = solver(C, d, num_threads)`.

## All 30 Solvers

The solver collection is **device-grouped** (MATLAB → OMP → CUDA), with the family ordering inside each device group: lsqnonneg/CAS → FAST-NNLS → Classic GP → ADMM → CG.

| #  | Folder | Function | Family | Precision | Platform |
|----|--------|----------|--------|-----------|----------|
| 01 | `01_matlab_lsqnonneg_st`     | `nnls_matlab_lsqnonneg_st`     | lsqnonneg        | FP64 | MATLAB ST |
| 02 | `02_matlab_lsqnonneg_mt`     | `nnls_matlab_lsqnonneg_mt`     | lsqnonneg        | FP64 | MATLAB MT |
| 03 | `03_matlab_fast_nnls_st`     | `nnls_fast_nnls_st`            | FAST-NNLS        | FP64 | MATLAB ST |
| 04 | `04_matlab_fast_nnls_mt`     | `nnls_fast_nnls_mt`            | FAST-NNLS        | FP64 | MATLAB MT |
| 05 | `05_matlab_classic_gp_st`    | `nnls_classic_gp_st`           | Classic GP       | FP64 | MATLAB ST |
| 06 | `06_matlab_classic_gp_mt`    | `nnls_classic_gp_mt`           | Classic GP       | FP64 | MATLAB MT |
| 07 | `07_matlab_admm_st`          | `nnls_admm_st`                 | ADMM             | FP64 | MATLAB ST |
| 08 | `08_matlab_admm_mt`          | `nnls_admm_mt`                 | ADMM             | FP64 | MATLAB MT |
| 09 | `09_matlab_cg_st`            | `nnls_cg_st`                   | CG               | FP64 | MATLAB ST |
| 10 | `10_matlab_cg_mt`            | `nnls_cg_mt`                   | CG               | FP64 | MATLAB MT |
| 11 | `11_classic_as_fp64_omp`     | `nnls_active_set_fp64_omp`     | CAS              | FP64 | C++/OMP |
| 12 | `12_classic_as_fp32_omp`     | `nnls_active_set_fp32_omp`     | CAS              | FP32 | C++/OMP |
| 13 | `13_fast_nnls_fp64_omp`      | `nnls_fast_nnls_fp64_omp`      | FAST-NNLS        | FP64 | C++/OMP |
| 14 | `14_fast_nnls_fp32_omp`      | `nnls_fast_nnls_fp32_omp`      | FAST-NNLS        | FP32 | C++/OMP |
| 15 | `15_classic_gp_fp64_omp`     | `nnls_classic_gp_fp64_omp`     | Classic GP       | FP64 | C++/OMP |
| 16 | `16_classic_gp_fp32_omp`     | `nnls_classic_gp_fp32_omp`     | Classic GP       | FP32 | C++/OMP |
| 17 | `17_admm_fp64_omp`           | `nnls_admm_fp64_omp`           | ADMM             | FP64 | C++/OMP |
| 18 | `18_admm_fp32_omp`           | `nnls_admm_fp32_omp`           | ADMM             | FP32 | C++/OMP |
| 19 | `19_cg_fp64_omp`             | `nnls_cg_fp64_omp`             | CG               | FP64 | C++/OMP |
| 20 | `20_cg_fp32_omp`             | `nnls_cg_fp32_omp`             | CG               | FP32 | C++/OMP |
| 21 | `21_classic_as_fp64_cuda`    | `nnls_active_set_fp64_cuda`    | CAS              | FP64 | CUDA |
| 22 | `22_classic_as_fp32_cuda`    | `nnls_active_set_fp32_cuda`    | CAS              | FP32 | CUDA |
| 23 | `23_fast_nnls_fp64_cuda`     | `nnls_fast_nnls_fp64_cuda`     | FAST-NNLS        | FP64 | CUDA |
| 24 | `24_fast_nnls_fp32_cuda`     | `nnls_fast_nnls_fp32_cuda`     | FAST-NNLS        | FP32 | CUDA |
| 25 | `25_classic_gp_fp64_cuda`    | `nnls_classic_gp_fp64_cuda`    | Classic GP       | FP64 | CUDA |
| 26 | `26_classic_gp_fp32_cuda`    | `nnls_classic_gp_fp32_cuda`    | Classic GP       | FP32 | CUDA |
| 27 | `27_admm_fp64_cuda`          | `nnls_admm_fp64_cuda`          | ADMM             | FP64 | CUDA |
| 28 | `28_admm_fp32_cuda`          | `nnls_admm_fp32_cuda`          | ADMM             | FP32 | CUDA |
| 29 | `29_cg_fp64_cuda`            | `nnls_cg_fp64_cuda`            | CG               | FP64 | CUDA |
| 30 | `30_cg_fp32_cuda`            | `nnls_cg_fp32_cuda`            | CG               | FP32 | CUDA |

**ST** = single-thread, **MT** = multi-thread (MATLAB MKL). CG and Classic GP print a runtime warning at start with the iteration counts needed to reach 5% and 1% error on ill-conditioned Tikhonov problems.

## Quick Start

### Prerequisites

| Component             | Required for             |
|-----------------------|--------------------------|
| MATLAB R2018b+        | All solvers              |
| C++ compiler + OpenMP | OMP MEX (slots 11-20)    |
| CUDA Toolkit 11.0+    | CUDA MEX (slots 21-30)   |
| NVIDIA GPU + cuBLAS + cuSOLVER | CUDA MEX     |

MATLAB-only solvers (slots 01-10) need no compilation.

### Build

```matlab
% From MATLAB, in project root
run('build/build_all.m')
```

This builds all 20 MEX targets (10 OpenMP + 10 CUDA). MATLAB-only solvers are listed but not compiled.

To rebuild only the CUDA targets:
```matlab
addpath build
rebuild_cuda()
```

### Benchmark

```matlab
% Run one solver on one PSF size (auto-generates System_Matrix on first run)
TARGET = 21; PSF_SIZE = 131; run('bench_one.m')   % CAS CUDA FP64 on PSF131

% Quick test on PSF21 (fully self-contained, no generation needed)
TARGET = 21; PSF_SIZE = 21; run('bench_one.m')

% From shell (isolated MATLAB instance per solver)
matlab -batch "TARGET=21; PSF_SIZE=131; run('bench_one.m')"
```

The benchmark script `bench_one.m` takes two parameters:
- `TARGET` (1-30): which solver to run
- `PSF_SIZE` (21, 81, 111, 121, 131, 201): which PSF problem size

On first run for a given PSF size, the System_Matrix_3D.csv is auto-generated from the included PSF CSV via `generate_system_matrix.m` and cached to disk. Subsequent runs load from cache.

Available PSF sizes: 21 (441 vars, instant), 81 (6561 vars), 111 (12321 vars), 121 (14641 vars), 131 (17161 vars), 201 (40401 vars).

## CUDA Options

The GPU build auto-detects your GPU architecture, CUDA toolkit, and host compiler. Override any setting:

```matlab
% From MATLAB
cuda_opts = struct('gpu_arch', '8.0');                           rebuild_cuda(cuda_opts)
cuda_opts = struct('unified_memory', 'managed');                 rebuild_cuda(cuda_opts)
cuda_opts = struct('cuda_path', '/usr/local/cuda-12.6');         rebuild_cuda(cuda_opts)
```

| Option           | Values                          | Default      |
|------------------|---------------------------------|--------------|
| `gpu_arch`       | `7.0`, `8.0`, `8.6`, `9.0`, ... | Auto-detect  |
| `unified_memory` | `off`, `managed`, `prefetch`    | `off`        |
| `cuda_path`      | Path to CUDA toolkit            | Auto-detect  |
| `host_compiler`  | Path to g++ (needs GCC ≤ 12 for CUDA 12.x) | Auto-detect |
| `verbose`        | `true` / `false`                | `false`      |

## Algorithm Notes

### lsqnonneg / CAS — Lawson-Hanson active set (1974)

Both lsqnonneg (MATLAB built-in) and CAS (C++/CUDA reimplementation) implement the same algorithm: maintain a "passive set" of currently-free variables, add the variable with the largest gradient component each outer iteration, solve the unconstrained least-squares subproblem on the passive set via QR (or Cholesky on the normal equations), and kick variables back to the active set with a feasibility line search if any go negative. Convergence is in `O(|passive set|)` outer iterations, **independent of κ**, which is why this family wins on Tikhonov problems.

The CAS-vs-lsqnonneg comparison is the apples-to-apples "what does it cost to write Lawson-Hanson by hand in C++/CUDA" benchmark — the algorithm is identical, only the implementation/language/hardware differs.

### FAST-NNLS — Threshold-based batch active set (Bro & de Jong 1997 / Cobb et al. 2025)

Variant of the Lawson-Hanson lineage that adds and removes multiple variables per iteration using threshold parameters (`theta_add`, `theta_rem`). Pre-computes the Gram matrix `H = C'*C` and `q = C'*d` once, then each inner solve is a small `|P| × |P|` Cholesky on the Gram block. The Gram setup is `O(m·n²)` flops upfront — the dominant cost on CPU; on GPU it runs as one `cublasDsyrk` call in a few seconds.

### ADMM — Boyd splitting with Cholesky x-update (1975 / 2011)

Splits `min ||Cx − d||² + I_+(z)  s.t.  x = z` and alternates:

```
x ← (2 C'C + ρ·I)⁻¹ (2 C'd + ρ(z − u))    # closed-form, via pre-factored Cholesky
z ← max(0, x + u)                            # projection
u ← u + (x − z)                              # dual update
```

The x-update is reduced to two triangular solves after a one-time Cholesky factorization of `2 C'C + ρ·I`. Convergence is essentially independent of cond(C) — exactly the property that gradient methods lack on Tikhonov problems where `κ ≈ L/λ` can reach 10¹⁰. We use `ρ = 10` for FP64 and `ρ = 15` for the FP32 CUDA variant (cuSOLVER `Spotrf` is more sensitive to near-PD matrices than CPU LAPACK).

### CG — Conjugate gradient on the normal equations (Hestenes-Stiefel 1952)

Plain CG on `B'B x = B'd` with a single `max(0, ·)` projection at the end. The bound is **not** enforced during iteration. Convergence rate is `O(√κ)` — on PSF131 (κ ≈ 10¹⁰) this means ~55,000 iterations for `relErr 0.05` (~8 min on CUDA FP32) and ~85,000 for `relErr 0.01` (~12 min on CUDA FP32). At the default `max_iter = 500` the solver prints a warning with these numbers.

### Classic GP — Goldstein-Levitin-Polyak projected gradient (1964)

The textbook plain projected gradient method:
```
x_{k+1} = max(0, x_k − (1/L) · C^T (C x_k − d))
```
with `L` estimated by power iteration. Asymptotic rate `(1 − μ/L)` per iteration, which on Tikhonov PSF problems with `κ ≈ 10¹⁰` is `~4 × 10⁻¹¹` — meaning the method needs `~6.7 × 10¹⁰` iterations for `relErr 0.05` and `~10¹¹` for `relErr 0.01`. At the default `max_iter = 500` the solver prints a warning with these numbers.

## Convergence rate comparison on PSF-class Tikhonov problems (κ ≈ 10¹⁰)

| Method | Rate per iteration | Iters for relErr 0.05 | Iters for relErr 0.01 |
|---|---|---|---|
| Classic GP | `(1 − 1/κ)` | ~6.7 × 10¹⁰ | ~10¹¹ |
| FISTA (Nesterov-accelerated GP) | `(1 − 1/√κ)` | ~1.6 × 10⁵ | ~2.5 × 10⁵ |
| CG (Hestenes-Stiefel) | `(1 − 2/√κ)` | ~5.5 × 10⁴ | ~8.5 × 10⁴ |
| ADMM (500 iters, Cholesky) | — | converged (1.9×10⁻³) | converged |
| Active set (CAS, FAST-NNLS) | — | converged (< 10⁻⁶) | converged |

Active-set and ADMM families converge independently of κ because they solve linear systems directly rather than iterating gradient steps. Gradient-family methods (GP, CG) are bounded by κ and need orders of magnitude more iterations on ill-conditioned problems. All six families are implemented and benchmarked so the convergence behavior can be observed directly.

## Sample Data

The `data/` directory contains PSF data for 6 problem sizes, organized in per-size subfolders:

```
data/
├── psf_21/          PSF21.csv + System_Matrix_3D.csv (1.1 MB, included)
├── psf_81/          PSF81.csv (System_Matrix generated on first run, ~218 MB)
├── psf_111/         PSF111.csv (System_Matrix ~775 MB)
├── psf_121/         PSF121.csv (System_Matrix ~1.1 GB)
├── psf_131/         PSF131.csv (System_Matrix ~1.5 GB)
├── psf_201/         PSF201.csv (System_Matrix ~8.2 GB)
└── generate_system_matrix.m    parameterized generator
```

PSF21 is **fully self-contained** (System_Matrix included at 1.1 MB). For other sizes, `System_Matrix_3D.csv` is generated on first use from the included PSF CSV:

```matlab
addpath('data');
generate_system_matrix(131);   % creates data/psf_131/System_Matrix_3D.csv (~1.5 GB)
```

Or simply run `bench_one.m` — it auto-generates if the System_Matrix is missing.

### Usage with any solver

```matlab
% Load PSF131 Tikhonov problem
A = csvread('data/psf_131/System_Matrix_3D.csv');
v = flipud(csvread('data/psf_131/PSF131.csv', 1, 2));
lambda = 0.001;
[~, n] = size(A);
B = [A; sqrt(lambda) * eye(n)];
d = [v; zeros(n, 1)];

x = nnls_active_set_fp64_cuda(B, d);    % winner: ~2 s on V100, relErr ~7e-8
```

## Directory Structure

```
Parallel_NNLS_for_MPI/
├── README.md                  # this file
├── LICENSE                    # MIT
├── bench_one.m                # benchmark: TARGET (1-30) x PSF_SIZE (21..201)
├── build/
│   ├── build_all.m            # builds all 30 solvers
│   └── rebuild_cuda.m         # CUDA-only rebuild
├── src/
│   ├── 01-10  MATLAB solvers  (lsqnonneg, FAST, GP, ADMM, CG)
│   ├── 11-20  OMP MEX solvers (CAS, FAST, GP, ADMM, CG)
│   ├── 21-30  CUDA MEX solvers(CAS, FAST, GP, ADMM, CG)
│   ├── matlab/                # MATLAB 3D reconstruction script
│   └── python/                # plotting utilities
├── data/
│   ├── psf_21/ ... psf_201/   # per-size PSF data + generated System_Matrix
│   └── generate_system_matrix.m
└── examples/                  # quick_test.m, speed_benchmark.m, basic_usage.m
```

## Troubleshooting

**`mex` fails with OpenMP errors on macOS**
Install libomp: `brew install libomp`, then rebuild.

**All CUDA builds fail**
Check that `nvcc` is on your PATH and supports your GPU:
```bash
nvcc --list-gpu-code
```
If not, specify the CUDA path explicitly:
```matlab
cuda_opts = struct('cuda_path', '/usr/local/cuda-12.6'); rebuild_cuda(cuda_opts)
```

**GCC version too new for CUDA**
CUDA 12.x requires GCC ≤ 12. The build system auto-detects a compatible version, but you can override:
```matlab
cuda_opts = struct('host_compiler', '/usr/bin/g++-12'); rebuild_cuda(cuda_opts)
```

**MATLAB-only solvers work but MEX solvers don't**
Make sure you ran `build_all.m` first. MATLAB-only solvers (slots 01-10) need no compilation.

## References

### Algorithms implemented

- C. L. Lawson and R. J. Hanson, *Solving Least Squares Problems*, Prentice-Hall, 1974. (lsqnonneg / CAS — Lawson-Hanson active set)
- R. Bro and S. De Jong, "A fast non-negativity-constrained least squares algorithm," *J. Chemometrics*, vol. 11, no. 5, pp. 393–401, 1997. (FAST-NNLS lineage)
- J. Cobb et al., "FAST-NNLS: A fast and exact non-negative least squares algorithm," *IEEE BigData*, 2025. (FAST-NNLS variant in this code)
- R. Glowinski and A. Marrocco, "Sur l'approximation, par éléments finis d'ordre un, et la résolution, par pénalisation-dualité d'une classe de problèmes de Dirichlet non linéaires," *RAIRO*, vol. 9, no. 2, pp. 41–76, 1975. (ADMM origin)
- S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein, "Distributed optimization and statistical learning via the alternating direction method of multipliers," *Found. Trends Mach. Learn.*, vol. 3, no. 1, 2011. (ADMM modern reference)
- M. R. Hestenes and E. Stiefel, "Methods of conjugate gradients for solving linear systems," *J. Res. Natl. Bur. Stand.*, vol. 49, pp. 409–436, 1952. (CG)
- A. A. Goldstein, "Convex programming in Hilbert space," *Bull. AMS*, vol. 70, no. 5, pp. 709–710, 1964. (Classic GP)
- E. S. Levitin and B. T. Polyak, "Constrained minimization methods," *USSR Comp. Math. & Math. Physics*, vol. 6, no. 5, pp. 1–50, 1966. (Projected gradient analysis)

### Lower bounds and theory

- Y. Nesterov, "A method for solving the convex programming problem with convergence rate O(1/k²)," *Soviet Math. Dokl.*, vol. 27, no. 2, pp. 372–376, 1983. (First-order lower bound)
- A. Beck and M. Teboulle, "A fast iterative shrinkage-thresholding algorithm for linear inverse problems," *SIAM J. Imag. Sci.*, vol. 2, no. 1, pp. 183–202, 2009. (FISTA)

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{zhu2025accelerating,
  title     = {Accelerating Magnetic Particle Imaging with Data Parallelism: A Comparative Study},
  author    = {Zhu, et al.},
  booktitle = {Proceedings of the IEEE International Symposium on Multi-Core Systems-on-Chip (MCSoC'25)},
  year      = {2025}
}
```

## License

[MIT](LICENSE)
