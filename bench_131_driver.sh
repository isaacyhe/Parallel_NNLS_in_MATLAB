#!/usr/bin/env bash
# Driver: runs each of the 18 new-alg variants in its own MATLAB process
# with an OS-level timeout so a hang on one doesn't block the others.
#
# Usage: bash bench_131_driver.sh [TIMEOUT_SECONDS] [TARGETS...]
#   Default timeout: 300s
#   Default targets: 1..18

set -u
cd /home/matlab/Parallel_NNLS_for_MPI

TIMEOUT="${1:-300}"; shift || true
if [ $# -eq 0 ]; then
    TARGETS=$(seq 1 18)
else
    TARGETS="$@"
fi

MATLAB=/usr/local/MATLAB/R2024b/bin/matlab
SUMMARY=/tmp/bench_131_summary.csv
: > "$SUMMARY"

for T in $TARGETS; do
    echo "=== [$(date +%H:%M:%S)] TARGET=$T  timeout=${TIMEOUT}s ==="
    rm -f /tmp/bench_131_result_$(printf '%02d' $T).txt
    timeout --kill-after=10 "$TIMEOUT" "$MATLAB" -nodisplay -nosplash -batch \
        "TARGET=$T; run('bench_131_one.m')" > /tmp/bench_131_stdout_$(printf '%02d' $T).log 2>&1
    RC=$?
    R=/tmp/bench_131_result_$(printf '%02d' $T).txt
    if [ -f "$R" ]; then
        cat "$R" >> "$SUMMARY"
        tail -1 "$R"
    else
        echo "$T,TARGET_$T,TIMEOUT,TIMEOUT,TIMEOUT,0,rc=$RC" >> "$SUMMARY"
        echo "  -> TIMEOUT or CRASH (rc=$RC)"
    fi
done

echo
echo "=== SUMMARY ==="
cat "$SUMMARY"
