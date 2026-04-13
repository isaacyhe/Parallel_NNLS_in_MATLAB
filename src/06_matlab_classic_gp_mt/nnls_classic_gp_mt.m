function x = nnls_classic_gp_mt(C, d, num_threads)
% NNLS_CLASSIC_GP_MT  Plain projected gradient for NNLS (multi-thread)
%
%   x = nnls_classic_gp_mt(C, d)
%   x = nnls_classic_gp_mt(C, d, num_threads)
%
%   Iterates  x_{k+1} = max(0, x_k - (1/L) * C^T (C x_k - d))
%   with L estimated by power iteration. MATLAB MKL threads accelerate
%   the per-iteration gemv operations.
%
%   Asymptotic rate (1 - mu/L) per iter; on Tikhonov PSF problems with
%   kappa ~ 1e10 this cannot reach low error in any practical iter budget.
%   See the runtime warning for the iteration counts required.
%
%   Reference:
%     Goldstein, A. A. (1964). Convex programming in Hilbert space.
%     Levitin, E. S. & Polyak, B. T. (1966). Constrained minimization methods.

    if nargin < 2
        error('nnls_classic_gp_mt:input', 'At least two inputs required: C, d');
    end
    [m, n] = size(C);
    if length(d) ~= m
        error('nnls_classic_gp_mt:dim', 'Dimension mismatch: rows(C) ~= length(d)');
    end
    if nargin < 3
        num_threads = maxNumCompThreads;
    end

    orig = maxNumCompThreads(num_threads);
    cleanup = onCleanup(@() maxNumCompThreads(orig));

    C = double(C);
    d = double(d(:));

    max_iter = 500;

    fprintf(['[GP] WARNING: max_iter=%d is insufficient for ill-conditioned NNLS.\n' ...
             '     Plain projected gradient on Tikhonov PSF problems (kappa ~1e10):\n' ...
             '       relErr 0.05  needs ~6.7e10 iterations  (>>years on any hardware)\n' ...
             '       relErr 0.01  needs ~1.0e11 iterations  (>>years on any hardware)\n' ...
             '     This algorithm class cannot reach low error on these problems.\n' ...
             '     For accuracy + speed prefer CAS, FAST-NNLS, or ADMM instead.\n'], ...
             max_iter);

    rng(0);
    v = randn(n, 1);
    v = v / norm(v);
    L = 0;
    for k = 1:30
        Cv = C * v;
        Hv = C' * Cv;
        L_new = norm(Hv);
        if L_new <= 0
            L_new = 1; break;
        end
        v = Hv / L_new;
        if k > 1 && abs(L_new - L) < 1e-4 * L_new
            L = L_new; break;
        end
        L = L_new;
    end
    L = L * 1.01;
    inv_L = 1.0 / L;

    x = zeros(n, 1);
    for k = 1:max_iter
        r = C * x - d;
        g = C' * r;
        x = max(0, x - inv_L * g);
    end
end
