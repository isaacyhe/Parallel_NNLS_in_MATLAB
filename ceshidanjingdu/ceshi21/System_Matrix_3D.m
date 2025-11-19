function [ ] = System_Matrix_3D( )
% PSF僼傽僀儖傪撉傒崬傒丄3D嵞峔惉梡偺僔僗僥儉峴楍傪csv偱弌椡

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Cz_min = 25.0;       % 擹搙椞堟z嵗昗偺壓尷抣
Cz_max = 25.0;      % 擹搙椞堟z嵗昗偺忋尷抣
dz = 5.0;           % z曽岦偺崗傒暆
N_coil = 1;         % 専弌僐僀儖屄悢
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Nz = (Cz_max-Cz_min)/dz + 1;        % 擹搙椞堟z曽岦偺梫慺悢

 A = [];
for i = 1:N_coil
    Ac = [];
    for j = 1:Nz
        z = Cz_min + (j-1)*dz;     
        Axy = System_Matrix_2D('PSF21.csv', 0, 0.02, 0, 0.02);
        temp = Ac;
        Ac = horzcat(temp, Axy);
    end
    temp = A;
    A = vertcat(temp, Ac);
end
disp(size(A));
csvwrite('System_Matrix_3D.csv', A);

end

