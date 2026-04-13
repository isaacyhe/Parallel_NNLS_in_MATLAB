#!/bin/bash
# Build all 24 NNLS solvers from the terminal.
#
# Usage:
#   ./build.sh                                    # auto-detect everything
#   ./build.sh --gpu-arch 8.0                     # force GPU compute capability
#   ./build.sh --unified-memory managed           # enable CUDA unified memory
#   ./build.sh --gpu-arch 8.0 --unified-memory managed --cuda-path /usr/local/cuda-12.6

set -euo pipefail
cd "$(dirname "$0")"

if ! command -v matlab &>/dev/null; then
    echo "Error: matlab not found on PATH" >&2
    exit 1
fi

# Parse command-line args into a MATLAB struct
cuda_opts_fields=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu-arch)
            cuda_opts_fields="${cuda_opts_fields}'gpu_arch','$2',"
            shift 2 ;;
        --unified-memory)
            cuda_opts_fields="${cuda_opts_fields}'unified_memory','$2',"
            shift 2 ;;
        --cuda-path)
            cuda_opts_fields="${cuda_opts_fields}'cuda_path','$2',"
            shift 2 ;;
        --host-compiler)
            cuda_opts_fields="${cuda_opts_fields}'host_compiler','$2',"
            shift 2 ;;
        --verbose)
            cuda_opts_fields="${cuda_opts_fields}'verbose',true,"
            shift ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--gpu-arch X.Y] [--unified-memory off|managed|prefetch] [--cuda-path PATH] [--host-compiler PATH] [--verbose]" >&2
            exit 1 ;;
    esac
done

if [[ -n "$cuda_opts_fields" ]]; then
    # Strip trailing comma
    cuda_opts_fields="${cuda_opts_fields%,}"
    matlab_cmd="cuda_opts = struct(${cuda_opts_fields}); run('build.m')"
else
    matlab_cmd="run('build.m')"
fi

matlab -batch "$matlab_cmd"
