#!/bin/bash
# Bounded prefix-history experiment; diagnostics never initialize formal training.
set -euo pipefail
TASK_ROOT="$PWD"
phase="${PUSHT_PHASE:?Set debug, train or evaluate}"
case "$phase" in debug|train|evaluate) ;; *) exit 2 ;; esac
if [[ -f results.tar ]]; then tar -xf results.tar; fi
checkpoint_archive="${PUSHT_CHECKPOINT_ARCHIVE:-checkpoints.tar}"
if [[ -f "$checkpoint_archive" ]]; then tar -xf "$checkpoint_archive"; fi
mkdir -p reports training
archive() {
  status=$?
  trap - EXIT
  if (( status != 0 && status != 85 )); then
    printf '%s\n' "$status" > reports/failure.exit
    echo 'Stopped after failure; debug shell available for one hour.' >&2
    deadline=$((SECONDS+3600))
    while [[ ! -f finish-debug ]] && (( SECONDS < deadline )); do sleep 5; done
  fi
  if [[ "$phase" == train ]]; then
    shopt -s nullglob
    files=(training/*.pt)
    if (( ${#files[@]} == 0 )); then
      tar -cf checkpoints.tar.tmp --files-from /dev/null || exit 74
    else
      tar -cf checkpoints.tar.tmp "${files[@]}" || exit 74
    fi
    mv checkpoints.tar.tmp checkpoints.tar || exit 74
  fi
  tar --exclude='*.pt' -cf results.tar.tmp reports training || exit 74
  mv results.tar.tmp results.tar || exit 74
  exit "$status"
}
trap archive EXIT
tar -xf pusht-runtime-v1.tar
tar -xf pusht-assets-v1.tar --exclude='*/tokens.pth'
tar -xf robojepa-22m-v1.tar
tar -xf source.tar
tar -xf robojepa-source.tar
tar -xf vjepa-source.tar
tar -xf goals.tar
if [[ -f approval.tar ]]; then tar -xf approval.tar; fi
export PYTHONPATH="$PWD/code/src" PYTHONUNBUFFERED=1
export ROBOJEPA_REPO="$PWD/robo_jepa"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=2
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy TQDM_DISABLE=1 WANDB_MODE=disabled
PY="$PWD/runtime/.venv/bin/python"
config=code/configs/pusht/robojepa22m_prefix_finetune.yaml
cp code/SOURCE_VERSION.json reports/source.json
cp code/uv.lock reports/uv.lock
touch ready
if [[ "$phase" == debug ]]; then
  "$PY" -m pytest -q code/tests > reports/pytest.txt 2>&1
fi
# Validate causal history on each actual training/evaluation GPU before proceeding.
if [[ ! -f reports/hardware-gate.json ]]; then
  "$PY" code/scripts/pusht/verify.py --assets assets --context-frames 3 --output reports/hardware-gate.json
fi
if [[ "$phase" == debug ]]; then
  "$PY" code/scripts/pusht/train.py --config "$config" --assets assets --output reports/smoke-training --smoke
  "$PY" code/scripts/pusht/evaluate.py --config code/configs/pusht/shared_benchmark.yaml \
    --assets assets --goals goals/pusht-h3-seed42.pt --output reports/smoke-evaluation \
    --model robojepa --checkpoint reports/smoke-training/best.pt --limit 2
  touch reports/debug.complete
elif [[ "$phase" == train ]]; then
  test -f approval/debug.complete
  export NETRC="$TASK_ROOT/.netrc" WANDB_MODE=online
  chmod 600 "$NETRC"
  "$PY" code/scripts/pusht/train.py --config "$config" --assets assets --output training
else
  test -f training/best.pt
  "$PY" code/scripts/pusht/evaluate.py --config code/configs/pusht/shared_benchmark.yaml \
    --assets assets --goals goals/pusht-h3-seed42.pt --output reports/robojepa-prefix \
    --model robojepa --checkpoint training/best.pt --max-seconds 1700
  "$PY" code/scripts/pusht/summarize.py \
    --nanowm code/reports/pusht/shared-benchmark/nanowm-shared \
    --robojepa reports/robojepa-prefix --output reports/comparison
fi
