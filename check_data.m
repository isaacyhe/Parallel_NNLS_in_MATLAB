raw = csvread('data/PSF201.csv');
vals = raw(:,3);
side = floor(sqrt(length(vals)));
C = reshape(vals(1:side*side), side, side);
fprintf('PSF201: C is %d x %d, cond = %.2e, rank = %d\n', size(C,1), size(C,2), cond(C), rank(C));

raw21 = csvread('data/PSF21.csv');
vals21 = raw21(:,3);
side21 = floor(sqrt(length(vals21)));
C21 = reshape(vals21(1:side21*side21), side21, side21);
fprintf('PSF21:  C is %d x %d, cond = %.2e, rank = %d\n', size(C21,1), size(C21,2), cond(C21), rank(C21));

SM = csvread('data/System_Matrix_3D.csv');
fprintf('System_Matrix_3D: %d x %d\n', size(SM,1), size(SM,2));

Re = csvread('data/Re_Coil1.csv');
fprintf('Re_Coil1: %d rows x %d cols (as CSV)\n', size(Re,1), size(Re,2));

% Check how lsqnonneg actually performs - is it really solving or trivially converging?
rng(42);
x_true = abs(randn(side, 1));
d = C * x_true + 0.01 * randn(side, 1);

tic; x1 = lsqnonneg(C, d); t1 = toc;
fprintf('\nlsqnonneg on 201x201: %.4f ms\n', t1*1000);

% Count nonzeros in solution
fprintf('nnz(x) = %d of %d\n', nnz(x1), length(x1));

% Try larger PSFs
psfs = {'data/PSF301.csv', 'data/PSF401.csv', 'data/PSF501.csv'};
for k = 1:length(psfs)
    raw = csvread(psfs{k});
    vals = raw(:,3);
    side = floor(sqrt(length(vals)));
    Ck = reshape(vals(1:side*side), side, side);
    rng(42);
    xk = abs(randn(side, 1));
    dk = Ck * xk + 0.01 * randn(side, 1);
    tic; lsqnonneg(Ck, dk); tk = toc;
    fprintf('lsqnonneg on %dx%d: %.2f ms\n', side, side, tk*1000);
end
exit;
