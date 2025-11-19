if error < tolerance
    disp('结果一致');
else
    disp(['结果不一致，误差为: ', num2str(error)]);
end