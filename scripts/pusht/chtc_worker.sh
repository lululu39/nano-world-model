#!/bin/bash
set -euo pipefail
TASK_ROOT="$PWD"
if [[ -f results.tar ]]; then tar -xf results.tar; fi
if [[ -f checkpoints.tar ]]; then tar -xf checkpoints.tar; fi
mkdir -p training reports
archive() {
  status=$?
  trap - EXIT
  if (( status != 0 && status != 85 )); then
    printf '%s\n' "$status" > reports/failure.exit
    echo 'Training stopped. Debug shell remains available for one hour.' >&2
    deadline=$((SECONDS+3600))
    while [[ ! -f finish-debug ]] && (( SECONDS < deadline )); do sleep 5; done
  fi
  shopt -s nullglob
  files=(training/*.pt)
  if (( ${#files[@]} == 0 )); then
    tar -cf "$TASK_ROOT/checkpoints.tar.tmp" --files-from /dev/null || exit 74
  else
    tar -cf "$TASK_ROOT/checkpoints.tar.tmp" -C "$TASK_ROOT" "${files[@]}" || exit 74
  fi
  if ! mv "$TASK_ROOT/checkpoints.tar.tmp" "$TASK_ROOT/checkpoints.tar" ||
     ! tar --exclude='*.pt' -cf "$TASK_ROOT/results.tar.tmp" -C "$TASK_ROOT" training reports ||
     ! mv "$TASK_ROOT/results.tar.tmp" "$TASK_ROOT/results.tar"; then
    exit 74
  fi
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
export PYTHONPATH="$PWD/code/src" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=2
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy TQDM_DISABLE=1
phase="${PUSHT_PHASE:-all}"
if [[ "$phase" != all && "$phase" != train && "$phase" != evaluate ]]; then exit 2; fi
export WANDB_MODE=disabled
if [[ "$phase" != evaluate ]]; then
  export NETRC="$TASK_ROOT/.netrc" WANDB_MODE=online
  chmod 600 "$NETRC"
fi
cp code/SOURCE_VERSION.json reports/source.json
cp code/uv.lock reports/uv.lock
PY="$PWD/runtime/.venv/bin/python"
# A new GPU family gets a bounded actual-model gate before any formal update.
if [[ "$phase" != evaluate && ! -f reports/hardware-gate.json ]]; then
  "$PY" code/scripts/pusht/verify.py --assets assets --output reports/hardware-gate.json
fi
train_complete=false
if [[ -f training/summary.json ]]; then
  train_complete=$("$PY" -c 'import json; s=json.load(open("training/summary.json")); print(str(s["exit_code"] == 0).lower())')
fi
if [[ "$phase" != evaluate && "$train_complete" != true ]]; then
  "$PY" code/scripts/pusht/train.py --config code/configs/pusht/robojepa22m_finetune.yaml --assets assets --output training
fi
if [[ "$phase" == train ]]; then exit 0; fi
if [[ ! -f training/best.pt ]]; then echo 'Missing trained checkpoint' >&2; exit 2; fi
export WANDB_MODE=disabled
for model in nanowm robojepa; do
  output="reports/$model-shared"
  complete=false
  if [[ -f "$output/summary.json" ]]; then
    complete=$("$PY" -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["completed"]).lower())' "$output/summary.json")
  fi
  if [[ "$complete" == true ]]; then continue; fi
  args=()
  if [[ "$model" == robojepa ]]; then args=(--checkpoint training/best.pt); fi
  "$PY" code/scripts/pusht/evaluate.py --config code/configs/pusht/shared_benchmark.yaml --assets assets --goals goals/pusht-h3-seed42.pt --output "$output" --model "$model" "${args[@]}" --max-seconds 1700
done
"$PY" code/scripts/pusht/summarize.py --nanowm reports/nanowm-shared --robojepa reports/robojepa-shared --output reports/comparison
