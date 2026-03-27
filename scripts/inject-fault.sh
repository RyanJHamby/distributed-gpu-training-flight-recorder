#!/usr/bin/env bash
# Inject synthetic GPU faults for testing. Throttles GPU clocks on a device
# to simulate thermal degradation.
#
# Usage: ./scripts/inject-fault.sh [GPU_INDEX] [DURATION_SECONDS]
# Requires: nvidia-smi with root/sudo

set -euo pipefail

GPU_INDEX="${1:-0}"
DURATION="${2:-5}"

echo "Injecting fault: throttle GPU ${GPU_INDEX} for ${DURATION}s"

# TODO: save current clocks, lock to min freq (nvidia-smi -lgc 210,210),
# sleep $DURATION, restore (nvidia-smi -rgc)
echo "Not yet implemented"
exit 1
