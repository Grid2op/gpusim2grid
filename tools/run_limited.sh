#!/usr/bin/env bash
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Run a command with hard host resource limits, for shared machines (tests, benchmarks, builds):
#
#   tools/run_limited.sh [-m MEM_GB] [-c CPUS] [-t TIMEOUT_S] [-o CORE_OFFSET] -- cmd args...
#
# e.g.  CUDA_VISIBLE_DEVICES=2 tools/run_limited.sh -m 32 -c 8 -- python -m pytest tests/python/topology_actions
#
# The GPU is NOT chosen here: CUDA_VISIBLE_DEVICES is taken from the environment as it is.
#
# Limits applied (cgroup memory limits are often not enforced, e.g. under cgroup v1):
#   * ulimit -d  : caps the data segment (heap / anonymous mmaps) at MEM_GB. CUDA's PROT_NONE
#                  address-space reservations are not counted, so CUDA programs still start.
#   * RSS watchdog: kills the whole process group if its total RSS exceeds MEM_GB (catches
#                  memory the data rlimit does not see, e.g. file-backed mmaps).
#   * taskset + *_NUM_THREADS + the JVM's ActiveProcessorCount (pypowsybl): at most CPUS cores,
#                  the last ones of the machine (shifted by CORE_OFFSET).
#   * nice 10 and a timeout.
set -u
MEM_GB=16
CPUS=8
TIMEOUT=1800
OFFSET=0
while getopts "m:c:t:o:" opt; do
  case $opt in
    m) MEM_GB=$OPTARG ;;
    c) CPUS=$OPTARG ;;
    t) TIMEOUT=$OPTARG ;;
    o) OFFSET=$OPTARG ;;
    *) echo "usage: $0 [-m MEM_GB] [-c CPUS] [-t TIMEOUT_S] [-o CORE_OFFSET] -- cmd args..." >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "usage: $0 [-m MEM_GB] [-c CPUS] [-t TIMEOUT_S] [-o CORE_OFFSET] -- cmd args..." >&2; exit 2; }

NPROC=$(nproc --all)
LAST=$((NPROC - 1 - OFFSET))
FIRST=$((LAST - CPUS + 1))
[ "$FIRST" -ge 0 ] || { echo "[run_limited] $CPUS cores at offset $OFFSET do not fit in $NPROC" >&2; exit 2; }
CORES="${FIRST}-${LAST}"

export OMP_NUM_THREADS=$CPUS OPENBLAS_NUM_THREADS=$CPUS MKL_NUM_THREADS=$CPUS NUMBA_NUM_THREADS=$CPUS
export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-$CPUS}
# numpy's madvise(MADV_HUGEPAGE) with THP defrag=madvise: page faults can stall in memory
# compaction once a process holds tens of GB. Without it, huge pages are used when free.
export NUMPY_MADVISE_HUGEPAGE=0
# pypowsybl's JVM: half the memory cap, at most 24 GB (a larger GraalVM heap breaks)
JVM_GB=$(( MEM_GB / 2 < 24 ? MEM_GB / 2 : 24 ))
[ "$JVM_GB" -ge 1 ] || JVM_GB=1
export GRAALVM_OPTIONS="-Xmx${JVM_GB}g -XX:ActiveProcessorCount=$CPUS"
export JAVA_TOOL_OPTIONS="-Xmx${JVM_GB}g -XX:ActiveProcessorCount=$CPUS"

LIMIT_KB=$((MEM_GB * 1024 * 1024))
echo "[run_limited] ${MEM_GB} GB, cores ${CORES}, timeout ${TIMEOUT} s, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}" >&2

setsid bash -c "ulimit -d $LIMIT_KB; exec timeout --kill-after=30 $TIMEOUT nice -n 10 taskset -c $CORES \"\$@\"" _ "$@" &
CHILD=$!

# RSS watchdog over the process group led by CHILD.
(
  while kill -0 "$CHILD" 2>/dev/null; do
    RSS_KB=$(ps -o rss= -g "$CHILD" 2>/dev/null | awk '{s+=$1} END {print s+0}')
    if [ "${RSS_KB:-0}" -gt "$LIMIT_KB" ]; then
      echo "[run_limited] RSS ${RSS_KB}kB > ${LIMIT_KB}kB: killing process group $CHILD" >&2
      kill -TERM -- -"$CHILD" 2>/dev/null; sleep 5; kill -KILL -- -"$CHILD" 2>/dev/null
      break
    fi
    sleep 1
  done
) &
WATCH=$!

wait "$CHILD"
RC=$?
kill "$WATCH" 2>/dev/null
exit $RC
