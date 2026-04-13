S = load('/tmp/bench_131_data.mat');
B = S.B;
fprintf('B size: %dx%d\n', size(B,1), size(B,2));
fprintf('min(B(:)) = %g\n', min(B(:)));
fprintf('max(B(:)) = %g\n', max(B(:)));
fprintf('#negative entries: %d\n', sum(B(:) < 0));
fprintf('#positive entries: %d\n', sum(B(:) > 0));
fprintf('%% negative = %.4f\n', 100*sum(B(:) < 0)/numel(B));
exit
