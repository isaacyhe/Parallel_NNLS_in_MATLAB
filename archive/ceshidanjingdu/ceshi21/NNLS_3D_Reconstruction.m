function [ ] = NNLS_3D_Reconstruction( lambda, num_threads )
% NNLS偱3D偺擹搙暘晍傪寁嶼乮lambda:惓懃壔僷儔儊乕僞乯, num_threads:线程数

% 参数初始化部分保持不变
Dx_min = 0; Dx_max = 0.02; Dy_min = 0; Dy_max = 0.02;
Cx_min = 0; Cx_max = 0.02; Cy_min = 0; Cy_max = 0.02;
Cz_min = 25.0; Cz_max = 25.0; dx = 0.001; dy = 0.001; dz = 5.0;
N_coil = 1;

Nx = (Cx_max-Cx_min)/dx + 1; Ny = (Cy_max-Cy_min)/dy + 1;
Nz = (Cz_max-Cz_min)/dz + 1; Mx = (Dx_max-Dx_min)/dx + 1;
My = (Dy_max-Dy_min)/dy + 1;

v = [];
for i = 1:N_coil
    filename = sprintf('PSF21.csv', i);
    vc = flipud(csvread(filename, 1, 2));
    temp = v;
    v = vertcat(temp, vc);
end

A = csvread('System_Matrix_3D.csv');
disp(size(A));
disp(size(v));

% 修改部分：调用 C++ MEX 文件，传递 num_threads
[~, n] = size(A);
B = [A; sqrt(lambda)*eye(n)];    % 构造增广矩阵
d = [v; zeros(n, 1)];           % 构造增广向量
combined_c = 500 * nnlsActiveSet21float(B, d, num_threads); % 使用传入的 num_threads

re_v = A * combined_c;

% 后续代码保持不变
[m, ~] = size(A);
Re_v = reshape(re_v, Mx*My, m/(Mx*My));
X1 = linspace(Dx_min, Dx_max, Mx);
X = repmat(X1', My, 1);
Y1 = linspace(Dy_min, Dy_max, My);
Y2 = repmat(Y1, Mx, 1);
Y = reshape(Y2, Mx*My, 1);
for i = 1:N_coil
    filename = sprintf('Re_Coil%d.csv', i);    
    format_GraphR(fopen(filename, 'w'), flipud(Re_v(:, i)), X, Y);
end

[~, n] = size(A);
c = reshape(combined_c, Nx*Ny, n/(Nx*Ny));
X1 = linspace(Cx_min, Cx_max, Nx);
X = repmat(X1', Ny, 1);
Y1 = linspace(Cy_min, Cy_max, Ny);
Y2 = repmat(Y1, Nx, 1);
Y = reshape(Y2, Nx*Ny, 1);
for i = 1:Nz
    z = Cz_min + (i-1)*dz;
    filename = sprintf('z=%dmm.csv', z);    
    format_GraphR(fopen(filename, 'w'), flipud(c(:, i)), X, Y);
end
end

% 保留 format_GraphR 函数定义
function [] = format_GraphR(fid, c, X, Y)
fprintf(fid, '僨乕僞宍幃, 2\n');
fprintf(fid, 'memo1\n');
fprintf(fid, 'x, y, z\n');
data = [X, Y, c];
for i = 1:size(X)
    fprintf(fid, '%f, %f, %f\n', data(i, :));
end
fclose(fid);
end
