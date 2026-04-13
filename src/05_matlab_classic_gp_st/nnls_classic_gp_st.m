function x = nnls_classic_gp_st(C, d)
% NNLS_CLASSIC_GP_ST  Plain projected gradient for NNLS (single-thread)
%
%   x = nnls_classic_gp_st(C, d)
%
%   Iterates  x_{k+1} = max(0, x_k - (1/L) * C^T (C x_k - d))
%   with L estimated by power iteration on C^T C (no Gram formed).
%
%   Reference baseline. Goldstein-Levitin-Polyak (1964) gradient projection.
%   Asymptotic rate (1 - mu/L) per iter — for Tikhonov PSF problems with
%   kappa ~ 1e10 this is ~4e-11, so the algorithm cannot reach low solution
%   error in any practical iter budget. Included as the textbook reference
%   point against which CG / ADMM / active-set methods are compared.
%
%   Reference:
%     Goldstein, A. A. (1964). Convex programming in Hilbert space. Bull. AMS.
%     Levitin, E. S. & Polyak, B. T. (1966). Constrained minimization methods.

    if nargin ~= 2
        error('nnls_classic_gp_st:input', 'Two inputs required: C, d');
    end
    [m, n] = size(C);
    if length(d) ~= m
        error('nnls_classic_gp_st:dim', 'Dimension mismatch: rows(C) ~= length(d)');
    end

    orig = maxNumCompThreads(1);
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

    % Power iteration on C (no Gram) for L = ||C^T C||_2
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
