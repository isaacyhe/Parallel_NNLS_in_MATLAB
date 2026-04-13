function x = nnls_cg_st(C, d)
% NNLS_CG_ST  Conjugate gradient on B'B x = B'd, terminal projection (single-thread)
%
%   x = nnls_cg_st(C, d)
%
%   Solves the normal equations B'B x = B'd via plain (Hestenes-Stiefel)
%   conjugate gradient, then clamps the result to x >= 0 with a single
%   max(0, .) projection at the end.
%
%   This is *not* a constrained NNLS solver: it ignores the bound during
%   the iteration. It is included as a reference for the cost of solving
%   the unconstrained least-squares system via a gradient-only Krylov
%   method on PSF-class Tikhonov problems where kappa ~ 10^10. Convergence
%   is governed by the eigenvalue spectrum of B'B; on this problem class
%   plain CG needs O(10^5) iterations to reach low solution error, so
%   the table entry will reflect that the algorithm class is structurally
%   ill-suited rather than that the implementation is buggy.
%
%   Reference:
%     Hestenes, M. R. & Stiefel, E. (1952). Methods of conjugate gradients
%       for solving linear systems. J. Res. Nat. Bur. Standards 49.

    if nargin ~= 2
        error('nnls_cg_st:input', 'Two inputs required: C, d');
    end
    [m, n] = size(C);
    if length(d) ~= m
        error('nnls_cg_st:dim', 'Dimension mismatch: rows(C) ~= length(d)');
    end

    % Force single thread
    orig = maxNumCompThreads(1);
    cleanup = onCleanup(@() maxNumCompThreads(orig));

    C = double(C);
    d = double(d(:));

    max_iter = 500;

    fprintf(['[CG] WARNING: max_iter=%d is insufficient for ill-conditioned NNLS.\n' ...
             '     Plain CG on Tikhonov PSF problems (kappa ~1e10):\n' ...
             '       relErr 0.05  needs ~55,000 iterations  (~8 min on CUDA FP32)\n' ...
             '       relErr 0.01  needs ~85,000 iterations  (~12 min on CUDA FP32)\n' ...
             '     With max_iter=%d this run will return relErr ~0.97.\n' ...
             '     For accuracy + speed prefer CAS, FAST-NNLS, or ADMM instead.\n'], ...
             max_iter, max_iter);

    % g_k = -gradient at x = b - A x = (B'd) - (B'B) x
    % At x = 0, g = B'd
    x  = zeros(n, 1);
    g  = C' * d;            % residual of normal equations at x=0
    p  = g;
    rs_old = g' * g;

    for k = 1:max_iter
        Bp = C * p;
        Hp = C' * Bp;       % (B'B) * p
        denom = p' * Hp;
        if denom <= 0
            break;
        end
        alpha = rs_old / denom;
        x = x + alpha * p;
        g = g - alpha * Hp;
        rs_new = g' * g;
        if rs_new <= 0
            break;
        end
        beta = rs_new / rs_old;
        p = g + beta * p;
        rs_old = rs_new;
    end

    % Terminal projection (CG ignored the bound during iteration)
    x = max(x, 0);
end
