function x = nnls_fast_nnls_mt(C, d, num_threads)
% NNLS_FAST_NNLS_MT  FAST-NNLS (Cobb et al. 2025) in pure MATLAB (multi-threaded)
%
%   x = nnls_fast_nnls_mt(C, d)
%   x = nnls_fast_nnls_mt(C, d, num_threads)
%
%   Solves  min ||C*x - d||^2  subject to  x >= 0
%   using the FAST-NNLS algorithm which pre-computes C'*C and C'*d,
%   then uses batch threshold-based addition/removal of variables.
%   MATLAB's MKL threads accelerate matrix operations.
%
%   References:
%     Cobb et al. (2025). FAST-NNLS: A fast and exact non-negative least
%     squares algorithm. IEEE BigData.
%
%   Inputs:
%     C           - System matrix (m x n)
%     d           - Measurement vector (m x 1)
%     num_threads - Number of threads (default: maxNumCompThreads)
%
%   Outputs:
%     x - Solution vector (n x 1) with non-negative entries

    % Input validation
    if nargin < 2
        error('nnls_fast_nnls_mt:input', 'At least two inputs required: C, d');
    end
    [m, n] = size(C);
    if length(d) ~= m
        error('nnls_fast_nnls_mt:dim', 'Dimension mismatch: rows(C) ~= length(d)');
    end
    if nargin < 3
        num_threads = maxNumCompThreads;
    end

    % Set thread count
    orig = maxNumCompThreads(num_threads);
    cleanup = onCleanup(@() maxNumCompThreads(orig));

    C = double(C);
    d = double(d(:));

    tol = 1e-8;
    max_iter = 30 * n;
    theta_add = 0.5;
    theta_rem = 0.5;

    % Pre-compute normal equation components
    ZtZ = C' * C;
    Zty = C' * d;

    % Initialize
    P = false(n, 1);   % Passive set
    x = zeros(n, 1);
    w = Zty - ZtZ * x; % w = -gradient

    iter = 0;

    % Main loop
    while any(~P) && max(w(~P)) > tol
        % Find max w among active variables
        w_active = w;
        w_active(P) = -Inf;
        max_w = max(w_active);

        % BATCH ADD: move all active variables with large gradient to passive
        t_add = max_w * theta_add;
        batch = (~P) & (w > t_add);
        P(batch) = true;

        % Inner loop: solve and fix infeasible
        s = zeros(n, 1);
        while true
            iter = iter + 1;
            if iter > max_iter
                break;
            end

            % Solve sub-problem on passive set
            idx_P = find(P);
            if isempty(idx_P)
                break;
            end
            s(idx_P) = ZtZ(idx_P, idx_P) \ Zty(idx_P);
            s(~P) = 0;

            % Check feasibility
            if all(s(P) > 0)
                break;
            end

            % BATCH REMOVE: find threshold for infeasible variables
            min_s = min(s(P));
            t_rem = min_s * theta_rem;

            infeasible = P & (s < t_rem);
            if ~any(infeasible)
                infeasible = P & (s <= 0);
            end

            % Interpolation step
            alpha = min(x(infeasible) ./ (x(infeasible) - s(infeasible)));
            x = x + alpha * (s - x);

            % Move zero-valued passive variables back to active set
            move_back = P & (abs(x) < tol) & (s <= 0);
            P(move_back) = false;
            x(move_back) = 0;
        end

        x = s;
        w = Zty - ZtZ * x;

        if iter > max_iter
            break;
        end
    end

    % Ensure strict non-negativity
    x = max(0, x);
end
