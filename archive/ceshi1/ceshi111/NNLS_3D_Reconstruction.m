function [ ] = NNLS_3D_Reconstruction( lambda )
% NNLS偱3D偺擹搙暘晍傪寁嶼乮lamda:惓懃壔僷儔儊乕僞乯

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Dx_min = 0;     % 僨乕僞椞堟x嵗昗偺壓尷抣
Dx_max = 110;      % 僨乕僞椞堟x嵗昗偺忋尷抣                        
Dy_min = 0;     % 僨乕僞椞堟y嵗昗偺壓尷抣
Dy_max = 110;      % 僨乕僞椞堟y嵗昗偺忋尷抣
Cx_min = 0;      % 擹搙椞堟x嵗昗偺壓尷抣 
Cx_max = 110;      % 擹搙椞堟x嵗昗偺忋尷抣
Cy_min = 0;     % 擹搙椞堟y嵗昗偺壓尷抣
Cy_max = 110;      % 擹搙椞堟y嵗昗偺忋尷抣
Cz_min = 25.0;       % 擹搙椞堟z嵗昗偺壓尷抣
Cz_max = 25.0;      % 擹搙椞堟z嵗昗偺忋尷抣
dx = 1;           % x曽岦偺崗傒暆
dy = 1;           % y曽岦偺崗傒暆
dz = 5.0;           % z曽岦偺崗傒暆
N_coil = 1;         % 専弌僐僀儖屄悢
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Nx = (Cx_max-Cx_min)/dx + 1;    % 擹搙椞堟x曽岦偺梫慺悢
Ny = (Cy_max-Cy_min)/dy + 1;    % 擹搙椞堟y曽岦偺梫慺悢
Nz = (Cz_max-Cz_min)/dz + 1;    % 擹搙椞堟z曽岦偺梫慺悢
Mx = (Dx_max-Dx_min)/dx + 1;        % 僨乕僞椞堟x曽岦偺梫慺悢
My = (Dy_max-Dy_min)/dy + 1;        % 僨乕僞椞堟y曽岦偺梫慺悢
v = [];
% 僨乕僞偺撉傒崬傒
for i = 1:N_coil
    filename = sprintf('PSF111.csv', i);
    vc = flipud(csvread(filename, 1, 2));
    %vc = flipud(csvread(filename));
    temp = v;
    v = vertcat(temp, vc);
end

A = csvread('System_Matrix_3D.csv');

disp(size(A));
disp(size(v));

%combined_c = 500 * NNLS(A, v, lambda);

%disp('test');
%tic;
combined_c = 500 * NNLS(A, v, lambda);
%combined_c = 500 * main_mex(A, v);
%combined_c = combined_c' ;
%toc;
%disp(combined_c);
re_v = A * combined_c;

[m, ~] =  size(A);
%Mx = floor(Mx);
%My = floor(My);
%Nx = floor(Nx);
%Ny = floor(Ny);
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

[~, n] =  size(A);
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

function [c] = NNLS(A, v, lambda)
[~, n] = size(A);
B = [A; sqrt(lambda)*eye(n)];
d = [v; zeros(n, 1)];
c = lsqnonneg(B, d);
end

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


