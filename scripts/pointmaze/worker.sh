#!/bin/bash
# One GPU job: diagnostic gates, NanoWM baseline, fresh MAE transfer, shared evaluation.
set -euo pipefail
TASK_ROOT="$PWD"
if [[ "${POINTMAZE_LIVE_RESUME:-0}" != 1 ]]; then
  if [[ -f results.tar ]]; then tar -xf results.tar; fi
  if [[ -f checkpoints.tar ]]; then tar -xf checkpoints.tar; fi
fi
mkdir -p reports training goals
archive() {
  status=$?
  trap - EXIT
  if (( status != 0 && status != 85 )); then
    printf '%s\n' "$status" > reports/failure.exit
    echo 'Stopped after failure; debug shell available for one hour.' >&2
    deadline=$((SECONDS+3600))
    while [[ ! -f finish-debug ]] && (( SECONDS < deadline )); do
      if [[ -f debug-resume.request ]]; then
        # Only an explicit operator repair request resumes; never retry a failure automatically.
        mkdir -p reports/debug-repairs
        stamp=$(date -u +%Y%m%dT%H%M%SZ)
        mv debug-resume.request "reports/debug-repairs/$stamp.request"
        mv reports/failure.exit "reports/debug-repairs/$stamp.failure.exit"
        echo 'POINTMAZE_MANUAL_RESUME' >&2
        export POINTMAZE_LIVE_RESUME=1
        exec /bin/bash "$TASK_ROOT/worker.sh"
      fi
      sleep 5
    done
  fi
  shopt -s nullglob
  files=(training/*.pt reports/recovery-reference/*.pt reports/recovery-resumed/*.pt goals/*.pt)
  if (( ${#files[@]} )); then
    tar -cf checkpoints.tar.tmp "${files[@]}" || exit 74
  else
    tar -cf checkpoints.tar.tmp --files-from /dev/null || exit 74
  fi
  mv checkpoints.tar.tmp checkpoints.tar || exit 74
  tar --exclude='*.pt' -cf results.tar.tmp reports training || exit 74
  mv results.tar.tmp results.tar || exit 74
  exit "$status"
}
trap archive EXIT
tar -xf source.tar
if [[ ! -f reports/assets-ready ]]; then
  tar -xf pointmaze-assets-v1.tar.gz
  tar -xf pointmaze-runtime-v1.tar.gz
  tar -xf robojepa-22m-v1.tar
  tar -xf robojepa-source.tar
  tar -xf vjepa-source.tar
  touch reports/assets-ready
fi
# On a migrated allocation only returned reports/checkpoints exist, not unpacked inputs.
if [[ ! -x runtime/.venv/bin/python || ! -f assets/provenance.json || ! -f weights/config.json ]]; then
  tar -xf pointmaze-assets-v1.tar.gz
  tar -xf pointmaze-runtime-v1.tar.gz
  tar -xf robojepa-22m-v1.tar
  tar -xf robojepa-source.tar
  tar -xf vjepa-source.tar
fi
export PATH="$TASK_ROOT/runtime/.venv/bin:$PATH"
export PYTHONPATH="$TASK_ROOT/code/src" ROBOJEPA_REPO="$TASK_ROOT/robo_jepa" PYTHONUNBUFFERED=1
export MUJOCO_PY_MUJOCO_PATH="$TASK_ROOT/native/mujoco210" MUJOCO_PY_FORCE_CPU=1
export LD_LIBRARY_PATH="$TASK_ROOT/native/mujoco210/bin:$TASK_ROOT/native/sysroot/usr/lib/x86_64-linux-gnu:$TASK_ROOT/native/sysroot/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export CPATH="$TASK_ROOT/native/sysroot/usr/include:${CPATH:-}"
export LIBRARY_PATH="$TASK_ROOT/native/sysroot/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
export D4RL_DATASET_DIR="$TASK_ROOT/d4rl-unused" D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=osmesa
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=2
export WANDB_MODE=disabled SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
PY="$TASK_ROOT/runtime/.venv/bin/python"
config=code/configs/pointmaze/robojepa22m_mae_finetune.yaml
protocol=code/configs/pointmaze/shared_benchmark.yaml
goals=goals/pointmaze-h5-seed42.pt
cp code/SOURCE_VERSION.json reports/source.json
cp code/uv.lock reports/uv.lock
"$PY" code/scripts/pointmaze/record_allocation.py --output reports/allocations.jsonl
complete() {
  "$PY" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); sys.exit(0 if p.exists() and json.loads(p.read_text()).get("completed") else 1)' "$1"
}
echo 'POINTMAZE_STAGE=smoke'
if [[ ! -f reports/pytest-passed ]]; then
  "$PY" -m pytest -q code/tests/latent code/tests/test_pusht_controller.py code/tests/test_pusht_loss_comparison.py code/tests/test_pointmaze_allocation.py > reports/pytest.txt 2>&1
  touch reports/pytest-passed
fi
if [[ ! -f reports/hardware-gate.json ]]; then
  "$PY" code/scripts/pusht/verify.py --assets assets --context-frames 3 --loss-type mae --environment point_maze --output reports/hardware-gate.json
fi
if [[ ! -f reports/recovery.complete ]]; then
  "$PY" code/scripts/pusht/recovery_smoke.py --config "$config" --assets assets --output reports
fi
if ! complete reports/nanowm-smoke/summary.json; then
  "$PY" code/scripts/pusht/evaluate.py --config "$protocol" --assets assets --goals "$goals" --model nanowm --output reports/nanowm-smoke --limit 2
fi
if ! complete reports/robojepa-smoke/summary.json; then
  "$PY" code/scripts/pusht/evaluate.py --config "$protocol" --assets assets --goals "$goals" --model robojepa --checkpoint reports/recovery-resumed/latest.pt --output reports/robojepa-smoke --limit 2
fi
touch reports/smoke.complete
# The diagnostics have served their purpose; keep their reports, not training weights.
rm -f reports/recovery-reference/*.pt reports/recovery-resumed/*.pt
echo 'POINTMAZE_STAGE=nanowm_evaluation'
if ! complete reports/nanowm/summary.json; then
  "$PY" code/scripts/pusht/evaluate.py --config "$protocol" --assets assets --goals "$goals" --model nanowm --output reports/nanowm --max-seconds 1700
fi
echo 'POINTMAZE_STAGE=formal_training'
if ! complete training/summary.json; then
  export NETRC="$TASK_ROOT/.netrc" WANDB_MODE=online
  chmod 600 "$NETRC"
  "$PY" code/scripts/pusht/train.py --config "$config" --assets assets --output training --run-id "${POINTMAZE_RUN_ID:?}"
  export WANDB_MODE=disabled
fi
if ! complete training/summary.json; then
  echo 'Training reached its time cap before 3000 updates; no extension or evaluation.' >&2
  touch reports/budget-exhausted
  exit 0
fi
echo 'POINTMAZE_STAGE=robojepa_evaluation'
if ! complete reports/robojepa/summary.json; then
  "$PY" code/scripts/pusht/evaluate.py --config "$protocol" --assets assets --goals "$goals" --model robojepa --checkpoint training/latest.pt --output reports/robojepa --max-seconds 1700
fi
"$PY" code/scripts/pusht/summarize.py --nanowm reports/nanowm --robojepa reports/robojepa --output reports/comparison
"$PY" code/scripts/pointmaze/check_results.py --root .
touch reports/experiment.complete
echo 'POINTMAZE_STAGE=complete'
