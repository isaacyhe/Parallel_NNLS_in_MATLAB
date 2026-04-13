% Check all provided data files
fprintf('=== System_Matrix_3D.csv ===\n');
SM = csvread('data/System_Matrix_3D.csv');
fprintf('  Size: %d x %d\n', size(SM,1), size(SM,2));
fprintf('  Range: [%.4f, %.4f]\n', min(SM(:)), max(SM(:)));
fprintf('  Rank: %d, Cond: %.2e\n', rank(SM), cond(SM));
fprintf('  nnz fraction: %.1f%%\n', 100*nnz(SM)/numel(SM));

fprintf('\n=== Re_Coil1.csv ===\n');
Re = csvread('data/Re_Coil1.csv');
fprintf('  Size: %d x %d\n', size(Re,1), size(Re,2));
if size(Re,2) == 3
    fprintf('  Looks like (x,y,value) format\n');
    vals = Re(:,3);
    side = floor(sqrt(length(vals)));
    fprintf('  Values: %d, side=%d => %dx%d matrix\n', length(vals), side, side, side);
    Rm = reshape(vals(1:side*side), side, side);
    fprintf('  Rank: %d, Cond: %.2e\n', rank(Rm), cond(Rm));
end

fprintf('\n=== z_25mm.csv ===\n');
Z = csvread('data/z_25mm.csv');
fprintf('  Size: %d x %d\n', size(Z,1), size(Z,2));
if size(Z,2) == 3
    fprintf('  Looks like (x,y,value) format\n');
    fprintf('  Value range: [%.6f, %.6f]\n', min(Z(:,3)), max(Z(:,3)));
end

fprintf('\n=== PSF files ===\n');
psfs = {'PSF21','PSF111','PSF131','PSF201','PSF301','PSF401','PSF501'};
for k = 1:length(psfs)
    fname = sprintf('data/%s.csv', psfs{k});
    raw = csvread(fname);
    vals = raw(:,3);
    side = floor(sqrt(length(vals)));
    fprintf('  %s: %d values => %dx%d\n', psfs{k}, length(vals), side, side);
end

% Try using System_Matrix_3D as actual system matrix with z_25mm as measurement
fprintf('\n=== Realistic NNLS problem ===\n');
% System_Matrix_3D is 441x441 system matrix
% z_25mm has 441 rows - could be measurement vector
zvals = Z(:,3);
fprintf('System matrix: %dx%d, measurement: %d\n', size(SM,1), size(SM,2), length(zvals));

% Try solving
tic; x_ref = lsqnonneg(SM, zvals); t = toc;
fprintf('lsqnonneg time: %.2f ms, nnz(x)=%d/%d, residual=%.4e\n', ...
    t*1000, nnz(x_ref), length(x_ref), norm(SM*x_ref - zvals));
exit;
