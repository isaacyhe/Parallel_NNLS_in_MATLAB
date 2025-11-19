function [A] = System_Matrix_2D(fid, PSFx_min, PSFx_max, PSFy_min, PSFy_max)
% System_Matrix_3D偐傜PSF偺僼傽僀儖ID傪庴偗庢傝丄2師尦偺僔僗僥儉峴楍傪曉偡
% 仸 PSF偼拞怱偑FFP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Dx_min = 0;     % 僨乕僞椞堟x嵗昗偺壓尷抣
Dx_max = 110;      % 僨乕僞椞堟x嵗昗偺忋尷抣
Dy_min = 0;     % 僨乕僞椞堟y嵗昗偺壓尷抣
Dy_max = 110;      % 僨乕僞椞堟y嵗昗偺忋尷抣
Cx_min = 0;     % 擹搙椞堟x嵗昗偺壓尷抣
Cx_max = 110;      % 擹搙椞堟x嵗昗偺忋尷抣
Cy_min = 0;     % 擹搙椞堟y嵗昗偺壓尷抣
Cy_max = 110;      % 擹搙椞堟y嵗昗偺忋尷抣
dx = 1;           % x曽岦偺崗傒暆
dy = 1;           % y曽岦偺崗傒暆
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Mx = (Dx_max-Dx_min)/dx + 1;        % 僨乕僞椞堟x曽岦偺梫慺悢
My = (Dy_max-Dy_min)/dy + 1;        % 僨乕僞椞堟y曽岦偺梫慺悢
Nx = (Cx_max-Cx_min)/dx + 1;        % 擹搙椞堟x曽岦偺梫慺悢
Ny = (Cy_max-Cy_min)/dy + 1;        % 擹搙椞堟y曽岦偺梫慺悢
Px = (PSFx_max-PSFx_min)/dx + 1;    % PSF椞堟x曽岦偺梫慺悢
Py = (PSFy_max-PSFy_min)/dy + 1;    % PSF椞堟y曽岦偺梫慺悢

if abs(Cx_max-Dx_min)>abs(Dx_max-Cx_min)
    m = abs(Cx_max-Dx_min)/dx *2 + 1;
else
    m = abs(Dx_max-Cx_min)/dx *2 + 1;
end

if abs(Cy_max-Dy_min)>abs(Dy_max-Cy_min)
    n = abs(Cy_max-Dy_min)/dy *2 + 1;
else
    n = abs(Dy_max-Cy_min)/dy *2 + 1;
end

%m = floor(m);
%n = floor(n);
center_x = (m+1)/2.0;
center_y = (n+1)/2.0;
%center_x = floor(center_x);
%center_y = floor(center_y);

% PSF偺撉傒崬傒
P = csvread(fid, 1, 2);
%P = csvread(fid);
% PSF傪奼戝偡傞昁梫偑側偄偲偒
%Q = reshape(P, m, n);

% PSF傪奼戝偡傞偲偒`
% 奼戝偟偨椞堟偼0偱杽傔傞
%disp(P);disp(Px);disp(Py);disp(m);
%Px = round(Px);
%Py = round(Py);
P = reshape(P, Px, Py);

Q = zeros(m, n);
Q(center_x-(Px-1)/2.0:center_x+(Px-1)/2.0, ...
    center_y-(Py-1)/2.0:center_y+(Py-1)/2.0) = P;


% 僔僗僥儉峴楍 A 傪峔惉
A = zeros(Mx*My, Nx*Ny);
for i=0:My-1
    for j=0:Mx-1
        xf = Dx_min + dx*j;
        yf = Dy_min + dy*i;
           
        for k=0:Ny-1
            for l=0:Nx-1
                xi = Cx_min + dx*l;
      			yi = Cy_min + dy*k;
				x = (xi-xf)/dx;
         		y = (yi-yf)/dy; 
                %disp(x);
                %disp(y);
                %xs = 31-xf;
                %ys = 31-yf;
                %r2 = (xs-21)^2 + (ys-21)^2;
                %rate = 100/(100+r2);
                x = floor(x);
                y = floor(y);
				A(i*Mx+j+1, k*Nx+l+1) = Q(x+center_x, y+center_y);
                %disp(rate);
            end
        end
        
    end
end

end


