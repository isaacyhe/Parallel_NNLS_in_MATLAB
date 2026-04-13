%% Build All NNLS Solvers
% Convenience wrapper — builds all 24 solvers (CPU + GPU).
%
% Usage:
%   run('build.m')                                          % Auto-detect
%   cuda_opts = struct('gpu_arch','8.0'); run('build.m')    % Custom GPU arch
%   cuda_opts = struct('unified_memory','managed'); run('build.m')
%
% See also: build/build_all.m, build/rebuild_cuda.m

% Add build/ to path
addpath(fullfile(fileparts(mfilename('fullpath')), 'build'));

% Pass cuda_opts through if defined by caller
if ~exist('cuda_opts', 'var')
    build_all;
else
    build_all;  % build_all reads cuda_opts from workspace
end
