#!/bin/bash
# CPU-only preparation; the later GPU job keeps smoke and formal training together.
set -euo pipefail
TASK_ROOT="$PWD"
mkdir -p reports
finish() {
  status=$?
  trap - EXIT
  printf '%s\n' "$status" > reports/prepare.exit
  if (( status != 0 )); then
    echo 'Preparation failed; debug shell retained for one hour.' >&2
    deadline=$((SECONDS+3600))
    while [[ ! -f finish-debug ]] && (( SECONDS < deadline )); do sleep 5; done
  fi
  # Missing successful payloads must not strand the diagnostic report on transfer.
  for archive in assets.tar.gz runtime.tar.gz; do
    if [[ ! -f "$archive" ]]; then tar -czf "$archive" --files-from /dev/null; fi
  done
  tar -cf results.tar reports
  exit "$status"
}
trap finish EXIT
export WANDB_MODE=disabled OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export UV_CACHE_DIR="$TASK_ROOT/uv-cache" HF_HOME="$TASK_ROOT/hf-cache"
tar -xf source.tar
tar -xf pusht-runtime-v1.tar
cp code/pyproject.toml code/uv.lock code/.python-version runtime/
./uv sync --frozen --project runtime --extra pointmaze --python /opt/project/.venv/bin/python
./uv pip check --python runtime/.venv/bin/python > reports/pip-check.txt
./uv pip freeze --python runtime/.venv/bin/python > reports/environment.txt
cp code/SOURCE_VERSION.json reports/source.json
cp code/uv.lock reports/uv.lock
# Reuse only the existing SD encoder; no RGB decoder download or instantiation.
tar -xf pusht-assets-v1.tar assets/vae/config.json assets/vae/encoder.safetensors
tar -xOf pusht-assets-v1.tar assets/provenance.json > pusht-provenance.json
mkdir -p native
curl --fail --location --retry 2 https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -o mujoco210.tar.gz
tar -xf mujoco210.tar.gz -C native
test -f native/mujoco210/bin/libmujoco210.so
# Stage native rendering libraries without installing anything on the host.
mkdir -p native/apt/lists/partial native/apt/cache/archives/partial native/sysroot
apt_options=(-o "Dir::State::lists=$TASK_ROOT/native/apt/lists" -o "Dir::Cache=$TASK_ROOT/native/apt/cache" -o Debug::NoLocking=1)
apt-get "${apt_options[@]}" update > reports/apt-update.txt 2>&1
apt-get "${apt_options[@]}" --yes --download-only --reinstall --no-install-recommends install libosmesa6-dev libglew-dev > reports/apt-download.txt 2>&1
for package in native/apt/cache/archives/*.deb; do
  dpkg-deb -f "$package" Package Version >> reports/native-packages.txt
  dpkg-deb -x "$package" native/sysroot
done
export MUJOCO_PY_MUJOCO_PATH="$TASK_ROOT/native/mujoco210" MUJOCO_PY_FORCE_CPU=1
export LD_LIBRARY_PATH="$TASK_ROOT/native/mujoco210/bin:$TASK_ROOT/native/sysroot/usr/lib/x86_64-linux-gnu:$TASK_ROOT/native/sysroot/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export CPATH="$TASK_ROOT/native/sysroot/usr/include:${CPATH:-}"
export LIBRARY_PATH="$TASK_ROOT/native/sysroot/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
export PYTHONPATH="$TASK_ROOT/code/src" D4RL_DATASET_DIR="$TASK_ROOT/d4rl-unused" D4RL_SUPPRESS_IMPORT_ERROR=1
runtime/.venv/bin/python code/scripts/pointmaze/environment_smoke.py
runtime/.venv/bin/python code/scripts/pointmaze/prepare.py
printf '{"completed":true,"native_engine":"MuJoCo 2.1.0","gpu_smoke_pending":true}\n' > reports/preparation.json
tar -czf assets.tar.gz assets
# Only the staged libraries and engine are needed after successful compilation.
rm -r native/apt
tar -czf runtime.tar.gz runtime native
