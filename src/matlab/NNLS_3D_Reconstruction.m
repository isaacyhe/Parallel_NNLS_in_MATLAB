function [] = NNLS_3D_Reconstruction(lambda, num_threads)
% NNLS_3D_Reconstruction - Perform 3D MPI reconstruction using NNLS
%
% This function performs 3D Magnetic Particle Imaging reconstruction using
% Non-Negative Least Squares (NNLS) with Tikhonov regularization.
%
% Syntax:
%   NNLS_3D_Reconstruction(lambda, num_threads)
%
% Inputs:
%   lambda - Tikhonov regularization parameter (default: 0.001)
%   num_threads - Number of threads for parallel NNLS solver (default: 8)
%
% Outputs:
%   Generates CSV files with reconstruction results:
%     - Re_Coil*.csv: Reconstructed measurements for each coil
%     - z=*mm.csv: Particle concentration at each z-slice
%
% Example:
%   NNLS_3D_Reconstruction(0.01, 8)
%
% References:
%   Zhu et al. (2025). Accelerating Magnetic Particle Imaging with Data Parallelism: A Comparative Study. Proceedings of IEEE-MCSoC'25.
%
% Date: 2025
% License: MIT

% Set default parameters if not provided
if nargin < 1
    lambda = 0.001;
end
if nargin < 2
    num_threads = 8;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reconstruction Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data space bounds (mm)
Dx_min = 0;      % Data space minimum x
Dx_max = 110;    % Data space maximum x
Dy_min = 0;      % Data space minimum y
Dy_max = 110;    % Data space maximum y

% Reconstruction space bounds (mm)
Cx_min = 0;      % Reconstruction space minimum x
Cx_max = 110;    % Reconstruction space maximum x
Cy_min = 0;      % Reconstruction space minimum y
Cy_max = 110;    % Reconstruction space maximum y
Cz_min = 25.0;   % Reconstruction space minimum z
Cz_max = 25.0;   % Reconstruction space maximum z

% Spatial resolution (mm)
dx = 1;          % x-direction step size
dy = 1;          % y-direction step size
dz = 5.0;        % z-direction step size

% Number of receiver coils
N_coil = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute Grid Dimensions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Nx = (Cx_max - Cx_min) / dx + 1;  % Reconstruction x grid points
Ny = (Cy_max - Cy_min) / dy + 1;  % Reconstruction y grid points
Nz = (Cz_max - Cz_min) / dz + 1;  % Reconstruction z grid points
Mx = (Dx_max - Dx_min) / dx + 1;  % Data x grid points
My = (Dy_max - Dy_min) / dy + 1;  % Data y grid points

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load Measurement Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Loading measurement data...\n');
% Resolve path to data/ directory (works from any working directory)
script_dir = fileparts(mfilename('fullpath'));
data_dir = fullfile(script_dir, '..', '..', 'data');

v = [];
for i = 1:N_coil
    filename = fullfile(data_dir, sprintf('PSF111.csv', i));
    if ~exist(filename, 'file')
        error('Measurement file %s not found', filename);
    end
    vc = flipud(csvread(filename));
    v = vertcat(v, vc);
end

% Load system matrix
fprintf('Loading system matrix...\n');
sm_file = fullfile(data_dir, 'System_Matrix_3D.csv');
if ~exist(sm_file, 'file')
    error('System matrix file not found: %s', sm_file);
end
A = csvread(sm_file);

fprintf('System matrix size: %d x %d\n', size(A, 1), size(A, 2));
fprintf('Measurement vector length: %d\n', length(v));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solve NNLS Problem
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Solving NNLS problem (lambda=%.4f, threads=%d)...\n', lambda, num_threads);
tic;
combined_c = 500 * solve_NNLS(A, v, lambda);
solve_time = toc;
fprintf('NNLS solution completed in %.2f seconds\n', solve_time);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate Reconstructed Measurements
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Computing reconstructed measurements...\n');
re_v = A * combined_c;

% Reshape for visualization
Re_v = reshape(re_v, Mx * My, size(A, 1) / (Mx * My));

% Generate coordinate grids
X1 = linspace(Dx_min, Dx_max, Mx);
X = repmat(X1', My, 1);
Y1 = linspace(Dy_min, Dy_max, My);
Y2 = repmat(Y1, Mx, 1);
Y = reshape(Y2, Mx * My, 1);

% Save reconstructed data for each coil
for i = 1:N_coil
    filename = sprintf('Re_Coil%d.csv', i);
    write_graph_data(filename, flipud(Re_v(:, i)), X, Y);
    fprintf('Saved %s\n', filename);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save Reconstruction Results by Z-slice
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Saving reconstruction results...\n');
c = reshape(combined_c, Nx * Ny, size(A, 2) / (Nx * Ny));

% Generate coordinate grids for reconstruction space
X1 = linspace(Cx_min, Cx_max, Nx);
X = repmat(X1', Ny, 1);
Y1 = linspace(Cy_min, Cy_max, Ny);
Y2 = repmat(Y1, Nx, 1);
Y = reshape(Y2, Nx * Ny, 1);

% Save each z-slice
for i = 1:Nz
    z = Cz_min + (i - 1) * dz;
    filename = sprintf('z=%dmm.csv', z);
    write_graph_data(filename, flipud(c(:, i)), X, Y);
    fprintf('Saved %s\n', filename);
end

fprintf('Reconstruction complete!\n');
end

%% Helper Functions

function [c] = solve_NNLS(A, v, lambda)
% Solve NNLS problem with Tikhonov regularization
%
% Solves: min ||Ax - v||^2 + lambda*||x||^2 subject to x >= 0
%
% Inputs:
%   A - System matrix (m x n)
%   v - Measurement vector (m x 1)
%   lambda - Regularization parameter
%
% Output:
%   c - Solution vector (n x 1)

[~, n] = size(A);

% Augment system with regularization term
B = [A; sqrt(lambda) * eye(n)];
d = [v; zeros(n, 1)];

% Solve using MATLAB's lsqnonneg
c = lsqnonneg(B, d);
end

function [] = write_graph_data(filename, data, X, Y)
% Write data in CSV format for visualization
%
% Inputs:
%   filename - Output CSV filename
%   data - Data values (length N)
%   X - X coordinates (length N)
%   Y - Y coordinates (length N)

fid = fopen(filename, 'w');
if fid == -1
    error('Cannot open file %s for writing', filename);
end

% Write header
fprintf(fid, 'x, y, z\n');

% Write data
for i = 1:length(X)
    fprintf(fid, '%f, %f, %f\n', X(i), Y(i), data(i));
end

fclose(fid);
end
