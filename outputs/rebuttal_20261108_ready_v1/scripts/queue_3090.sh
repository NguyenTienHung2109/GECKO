#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
campaign=outputs/rebuttal_20261108_ready_v1
scripts="$campaign/scripts"
mkdir -p "$campaign/logs" "$campaign/tables" outputs/rebuttal_20261107_remaining_v1/artifacts

py=${1:?Pass the compatible Python executable}
export GECKO_RESERVED_GPU_UUID=${2:?Pass the reserved RTX 3090 GPU UUID}
export CUDA_VISIBLE_DEVICES="$GECKO_RESERVED_GPU_UUID"
export PYTHONPATH=outputs/e4_20260929/deps:src:.
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_THREADING_LAYER=GNU

# Refuse to run outside a cgroup with both resource limits enforced.
group=$(sed -n 's/^0:://p' /proc/self/cgroup)
test -n "$group"
read -r quota period < "/sys/fs/cgroup$group/cpu.max"
read -r ram < "/sys/fs/cgroup$group/memory.max"
[[ "$quota" =~ ^[0-9]+$ && "$period" =~ ^[0-9]+$ && "$ram" =~ ^[0-9]+$ ]]
(( quota <= 8 * period && ram <= 68719476736 ))
exec 9>"$campaign/queue_3090.lock"
flock -n 9
exec >> "$campaign/logs/queue_3090.log" 2>&1
echo "START $(date -Is) $(hostname) $GECKO_RESERVED_GPU_UUID cpu.max=$quota/$period memory.max=$ram"

[[ "$GECKO_RESERVED_GPU_UUID" =~ ^GPU-[[:xdigit:]-]+$ ]]
gpu_name=$(nvidia-smi --id "$GECKO_RESERVED_GPU_UUID" --query-gpu=name --format=csv,noheader)
[[ "$gpu_name" == "NVIDIA GeForce RTX 3090" ]]
gpu_processes=$(nvidia-smi --id "$GECKO_RESERVED_GPU_UUID" --query-compute-apps=pid --format=csv,noheader)
[[ -z "$gpu_processes" ]]

"$py" - <<'PY'
import json
import dgl
import torch

with open('outputs/e4_20260929/provenance/environment.json') as f:
    expected = json.load(f)
assert str(torch.__version__) == expected['torch'], torch.__version__
assert dgl.__version__ == expected['dgl'], dgl.__version__
assert torch.cuda.device_count() == 1
assert 'RTX 3090' in torch.cuda.get_device_name(0)
assert torch.ones(1, device='cuda').sum().item() == 1
PY
"$py" "$scripts/p1.py" --prepare

# Short independent diagnostic first. Preserve the exit status while allowing
# the other jobs to finish if compute calibration finds no feasible match.
compute_status=0
if "$py" -u "$scripts/compute_match.py" profile; then
    "$py" -u "$scripts/compute_match.py" compare || compute_status=$?
else
    compute_status=$?
fi
if (( compute_status != 0 )); then
    echo "MATCHED COMPUTE needs investigation (exit $compute_status)"
fi

for seed in 0 1 2; do
    for alpha in 0.1 100; do
        "$py" -u "$scripts/lc_directed.py" --seed "$seed" --alpha "$alpha"
    done
done

mapfile -t jobs < <("$py" -c 'import json,sys; print("\n".join(j["id"] for j in json.load(open(sys.argv[1]))["jobs"]))' "$campaign/provenance/p1_plan.json")
test "${#jobs[@]}" -eq 36
for id in "${jobs[@]}"; do
    echo "START $(date -Is) $id"
    "$py" -u "$scripts/p1.py" --job "$id" >> "$campaign/logs/$id.log" 2>&1
    echo "VERIFIED $(date -Is) $id"
done
"$py" "$scripts/p1_report.py"
echo "QUEUE FINISHED $(date -Is) matched-compute exit=$compute_status"
exit "$compute_status"
