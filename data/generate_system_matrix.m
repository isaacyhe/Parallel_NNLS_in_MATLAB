function generate_system_matrix(psf_size)
% GENERATE_SYSTEM_MATRIX  Build System_Matrix_3D.csv from a PSF CSV file.
%
%   generate_system_matrix(131)
%
%   Reads data/psf_<N>/PSF<N>.csv, builds the 3D system matrix via
%   shift-invariant PSF convolution, and writes the result to
%   data/psf_<N>/System_Matrix_3D.csv.
%
%   Supported sizes: 21, 81, 111, 121, 131, 201.
%
%   The generated System_Matrix_3D.csv can be large (up to ~8 GB for
%   PSF201) and is .gitignored. This function regenerates it on demand.

    % Grid parameters per PSF size
    params = struct( ...
        'n21',  struct('grid_max', 0.02, 'dx', 0.001), ...
        'n81',  struct('grid_max', 16,   'dx', 0.2),   ...
        'n111', struct('grid_max', 110,  'dx', 1),      ...
        'n121', struct('grid_max', 120,  'dx', 1),      ...
        'n131', struct('grid_max', 130,  'dx', 1),      ...
        'n201', struct('grid_max', 130,  'dx', 0.65)    ...
    );

    key = sprintf('n%d', psf_size);
    if ~isfield(params, key)
        error('generate_system_matrix:size', ...
            'Unsupported PSF size %d. Supported: 21, 81, 111, 121, 131, 201.', psf_size);
    end
    p = params.(key);

    proj = fileparts(fileparts(mfilename('fullpath')));
    if isempty(proj); proj = pwd; end
    data_dir = fullfile(proj, 'data', sprintf('psf_%d', psf_size));
    psf_file = fullfile(data_dir, sprintf('PSF%d.csv', psf_size));
    out_file = fullfile(data_dir, 'System_Matrix_3D.csv');

    if ~exist(psf_file, 'file')
        error('generate_system_matrix:missing', 'PSF file not found: %s', psf_file);
    end

    fprintf('Generating System_Matrix_3D.csv for PSF%d ...\n', psf_size);
    fprintf('  Grid: 0..%.4f, dx=%.4f, n=%d\n', p.grid_max, p.dx, round(p.grid_max/p.dx)+1);

    tic;
    Axy = build_system_matrix_2d(psf_file, 0, p.grid_max, 0, p.grid_max, p.dx);
    t_build = toc;

    % 3D tiling (single z-slice at z=25mm, 1 coil — matches student's System_Matrix_3D.m)
    A = Axy;

    fprintf('  A: %d x %d (%.1f MB)\n', size(A,1), size(A,2), ...
        numel(A)*8/1e6);
    fprintf('  Build time: %.1f s\n', t_build);

    fprintf('  Writing %s ...\n', out_file);
    tic;
    csvwrite(out_file, A);
    fprintf('  Write time: %.1f s\n', toc);
    fprintf('  Done.\n');
end

function A = build_system_matrix_2d(psf_file, PSFx_min, PSFx_max, PSFy_min, PSFy_max, dx)
    % Mirrors the student's System_Matrix_2D.m with parameterized grid.
    dy = dx;
    Dx_min = 0; Dx_max = PSFx_max;
    Dy_min = 0; Dy_max = PSFy_max;
    Cx_min = 0; Cx_max = PSFx_max;
    Cy_min = 0; Cy_max = PSFy_max;

    Mx = round((Dx_max - Dx_min)/dx) + 1;
    My = round((Dy_max - Dy_min)/dy) + 1;
    Nx = round((Cx_max - Cx_min)/dx) + 1;
    Ny = round((Cy_max - Cy_min)/dy) + 1;
    Px = round((PSFx_max - PSFx_min)/dx) + 1;
    Py = round((PSFy_max - PSFy_min)/dy) + 1;

    if abs(Cx_max - Dx_min) > abs(Dx_max - Cx_min)
        m = round(abs(Cx_max - Dx_min)/dx) * 2 + 1;
    else
        m = round(abs(Dx_max - Cx_min)/dx) * 2 + 1;
    end

    if abs(Cy_max - Dy_min) > abs(Dy_max - Cy_min)
        n = round(abs(Cy_max - Dy_min)/dy) * 2 + 1;
    else
        n = round(abs(Dy_max - Cy_min)/dy) * 2 + 1;
    end

    center_x = (m + 1) / 2.0;
    center_y = (n + 1) / 2.0;

    % Read PSF (skip header row, take 3rd column = z-values)
    P = csvread(psf_file, 1, 2);
    P = reshape(P, Px, Py);

    % Embed PSF into padded grid
    Q = zeros(m, n);
    ix = round(center_x - (Px-1)/2.0) : round(center_x + (Px-1)/2.0);
    iy = round(center_y - (Py-1)/2.0) : round(center_y + (Py-1)/2.0);
    Q(ix, iy) = P;

    % Build system matrix via shift-invariant convolution
    A = zeros(Mx * My, Nx * Ny);
    for i = 0:My-1
        for j = 0:Mx-1
            xf = Dx_min + dx * j;
            yf = Dy_min + dy * i;
            for kk = 0:Ny-1
                for l = 0:Nx-1
                    xi = Cx_min + dx * l;
                    yi = Cy_min + dy * kk;
                    x = floor((xi - xf) / dx);
                    y = floor((yi - yf) / dy);
                    A(i*Mx + j + 1, kk*Nx + l + 1) = Q(x + center_x, y + center_y);
                end
            end
        end
    end
end
