#!/usr/bin/env bash
# Start the 910B resident server (mp.spawn, models stay in HBM).
# Usage:
#   bash run_serve_910b.sh
#   bash run_serve_910b.sh --port 8088
# Do NOT combine with run_910b.sh / torch.distributed.run.

set +euo pipefail
if [[ -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]]; then
  set +euo pipefail
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
  set +euo pipefail
fi
if [[ -f /usr/local/Ascend/driver/bin/setenv.bash ]]; then
  set +euo pipefail
  # shellcheck disable=SC1091
  source /usr/local/Ascend/driver/bin/setenv.bash
  set +euo pipefail
fi
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Prefer sitting next to motion_control.py (910b dir). If this copy is still
# inside 910b_resident_serve/, also try parent / CWD.
cd "${SCRIPT_DIR}"
if [[ ! -f motion_control.py && -f "${SCRIPT_DIR}/../motion_control.py" ]]; then
  cd "${SCRIPT_DIR}/.."
fi
if [[ ! -f motion_control.py && -f /data02/lyh/ics2v-new/src/30011_motion_control2v/910b/motion_control.py ]]; then
  cd /data02/lyh/ics2v-new/src/30011_motion_control2v/910b
fi

export PLATFORM=ascend_npu
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29511}"
export ICS2V_RESIDENT=1
export ICS2V_SKIP_RELAUNCH=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export YOLO_OFFLINE="${YOLO_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

PY="${PYTHON:-/usr/bin/python3}"
SERVE_PY="${SCRIPT_DIR}/serve_resident.py"
CONFIG="${ICS2V_CONFIG_YAML:-motion_control.yaml}"
PORT="${ICS2V_SERVE_PORT:-8088}"
NPROC="${ICS2V_NPROC:-4}"

echo "[run_serve_910b] cwd=$(pwd)"
echo "[run_serve_910b] python=${PY}"
echo "[run_serve_910b] ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "[run_serve_910b] nproc=${NPROC} port=${PORT}"
echo "[run_serve_910b] launcher=mp.spawn (resident, not torch.distributed.run)"

exec "${PY}" "${SERVE_PY}" \
  --config_yaml "${CONFIG}" \
  --nproc "${NPROC}" \
  --port "${PORT}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  "$@"
