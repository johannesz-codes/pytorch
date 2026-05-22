# PyTorch Wheel Build Pipeline

This document describes how Python wheels are built in the PyTorch repository,
covering Linux (manylinux), macOS (ARM64), and Windows platforms.

---

## 1. High-Level Wheel Build Pipeline

```
CI trigger (push to nightly branch / release tag / workflow_dispatch)
  │
  ├── Linux (manywheel)
  │     generated-linux-binary-manywheel-nightly.yml
  │       → _binary-build-linux.yml
  │           → docker run <manylinux2_28-builder image>
  │               → .circleci/scripts/binary_populate_env.sh   (set env vars)
  │               → .ci/manywheel/build.sh                     (dispatch by GPU_ARCH_TYPE)
  │                   → build_cpu.sh | build_cuda.sh | build_rocm.sh | build_xpu.sh
  │                       → source build_common.sh
  │                           → python -m build --wheel --no-isolation
  │
  ├── macOS ARM64
  │     generated-macos-arm64-binary-wheel-nightly.yml
  │       → .circleci/scripts/binary_populate_env.sh            (set env vars)
  │       → .ci/wheel/build_wheel.sh
  │           → python -m build --wheel --no-isolation
  │
  └── Windows
        generated-windows-binary-wheel-nightly.yml
          → .circleci/scripts/binary_populate_env.sh            (set env vars)
          → .circleci/scripts/binary_windows_build.sh
              ├── x86_64: .ci/pytorch/windows/internal/build_wheels.bat
              │     → windows/build_pytorch.bat
              │         → cpu.bat | cuda.bat | xpu.bat
              │             → internal/setup.bat
              │                 → python -m build --wheel --no-isolation
              └── ARM64: .ci/pytorch/windows/arm64/build_pytorch.bat
                    → python -m build --wheel --no-isolation
```

CI workflow files are **generated** from Jinja2 templates in `.github/templates/`
via `.github/scripts/generate_ci_workflows.py`. Do not edit the `generated-*.yml`
files directly.

---

## 2. Files and Scripts Involved

### GitHub Actions Workflows

| File | Purpose |
|------|---------|
| `.github/workflows/generated-linux-binary-manywheel-nightly.yml` | Linux manywheel nightly build (x86_64, CPU/CUDA/ROCm/XPU) |
| `.github/workflows/generated-linux-aarch64-binary-manywheel-nightly.yml` | Linux manywheel nightly build (aarch64) |
| `.github/workflows/generated-linux-s390x-binary-manywheel-nightly.yml` | Linux manywheel nightly build (s390x) |
| `.github/workflows/generated-macos-arm64-binary-wheel-nightly.yml` | macOS ARM64 wheel nightly build |
| `.github/workflows/generated-windows-binary-wheel-nightly.yml` | Windows wheel nightly build (x86_64) |
| `.github/workflows/generated-windows-arm64-binary-wheel-nightly.yml` | Windows wheel nightly build (ARM64) |
| `.github/workflows/_binary-build-linux.yml` | Reusable workflow called by all Linux binary builds |
| `.github/workflows/build-manywheel-images.yml` | Builds the manywheel Docker images themselves |

### Workflow Templates (source of generated workflows)

| File | Purpose |
|------|---------|
| `.github/templates/linux_binary_build_workflow.yml.j2` | Jinja2 template for Linux binary workflows |
| `.github/templates/windows_binary_build_workflow.yml.j2` | Jinja2 template for Windows binary workflows |
| `.github/templates/macos_binary_build_workflow.yml.j2` | Jinja2 template for macOS binary workflows |
| `.github/scripts/generate_ci_workflows.py` | Script that renders the templates into the `generated-*.yml` files |

### Build Scripts — Linux (manywheel)

| File | Purpose |
|------|---------|
| `.ci/manywheel/build.sh` | Entry point: dispatches to arch-specific scripts based on `GPU_ARCH_TYPE` |
| `.ci/manywheel/build_common.sh` | Core build logic: sets up Python, runs `python -m build` |
| `.ci/manywheel/build_cpu.sh` | Sets CPU-specific env vars; sources `build_common.sh` |
| `.ci/manywheel/build_cuda.sh` | Sets CUDA-specific env vars; sources `build_common.sh` |
| `.ci/manywheel/build_rocm.sh` | Sets ROCm-specific env vars; sources `build_common.sh` |
| `.ci/manywheel/build_xpu.sh` | Sets XPU-specific env vars; sources `build_common.sh` |
| `.ci/manywheel/build_libtorch.sh` | Builds the libtorch C++ package (not a Python wheel) |
| `.ci/manywheel/set_desired_python.sh` | Selects the Python interpreter for the build |
| `.ci/manywheel/test_wheel.sh` | Smoke-tests the built wheel |

### Build Scripts — macOS

| File | Purpose |
|------|---------|
| `.ci/wheel/build_wheel.sh` | macOS ARM64 wheel build; runs `python -m build`, then `delocate-wheel` |
| `.ci/wheel/install_libomp.sh` | Installs libomp for OpenMP support on macOS |

### Build Scripts — Windows

| File | Purpose |
|------|---------|
| `.circleci/scripts/binary_windows_build.sh` | Bash entry point; routes to arm64 or x86_64 `.bat` scripts |
| `.ci/pytorch/windows/internal/build_wheels.bat` | x86_64 orchestrator; calls `build_pytorch.bat` after VC/CUDA setup |
| `.ci/pytorch/windows/build_pytorch.bat` | x86_64 wheel build: routes to `cpu.bat`, `cuda.bat`, or `xpu.bat` |
| `.ci/pytorch/windows/cpu.bat` | Sets `USE_CUDA=0`, calls `internal/setup.bat` |
| `.ci/pytorch/windows/cuda.bat` | Sets CUDA env vars, calls `internal/setup.bat` |
| `.ci/pytorch/windows/xpu.bat` | Sets XPU env vars, calls `internal/setup.bat` |
| `.ci/pytorch/windows/internal/setup.bat` | Calls `python -m build --wheel --no-isolation` |
| `.ci/pytorch/windows/arm64/build_pytorch.bat` | ARM64 wheel build; calls `python -m build --wheel --no-isolation` |

### Environment Setup

| File | Purpose |
|------|---------|
| `.circleci/scripts/binary_populate_env.sh` | Sets `PYTORCH_BUILD_VERSION`, `PYTORCH_BUILD_NUMBER`, `DOCKER_IMAGE`, `PIP_UPLOAD_FOLDER`, and writes them to `$BINARY_ENV_FILE` |

### Docker Infrastructure

| File | Purpose |
|------|---------|
| `.ci/docker/manywheel/Dockerfile_2_28` | x86_64 manylinux_2_28 image based on AlmaLinux 8 with devtoolset-13 |
| `.ci/docker/manywheel/Dockerfile_2_28_aarch64` | aarch64 manylinux_2_28 image |
| `.ci/docker/manywheel/Dockerfile_cuda_aarch64` | aarch64 image with CUDA support |
| `.ci/docker/manywheel/Dockerfile_s390x` | s390x image |
| `.ci/docker/manywheel/build.sh` | Builds the manywheel Docker images |
| `.ci/docker/manywheel/build_scripts/` | Scripts installed inside the image (BLAS, SSL, etc.) |
| `.ci/docker/common/` | Common install scripts shared across Docker images |

---

## 3. Exact Commands Used to Build Wheels

### Linux — primary wheel build command

Located in `.ci/manywheel/build_common.sh` (line ~162):

```bash
time CMAKE_ARGS=${CMAKE_ARGS[@]} \
    EXTRA_CAFFE2_CMAKE_FLAGS=${EXTRA_CAFFE2_CMAKE_FLAGS[@]} \
    BUILD_LIBTORCH_CPU_WITH_DEBUG=$BUILD_DEBUG_INFO \
    USE_NCCL=${USE_NCCL} USE_RCCL=${USE_RCCL} USE_KINETO=${USE_KINETO} \
    python -m build --wheel --no-isolation --outdir /tmp/$WHEELHOUSE_DIR
```

The `$WHEELHOUSE_DIR` is set by the arch-specific scripts:
- CPU builds: `wheelhousecpu`
- CUDA builds: `wheelhousecuda` (with version suffix)
- ROCm builds: `wheelhouserocm`

After the wheel is produced it is copied to `$PYTORCH_FINAL_PACKAGE_DIR` (`/artifacts` in CI).
The build directory is then cleaned:

```bash
python setup.py clean
```

### macOS ARM64 — primary wheel build command

Located in `.ci/wheel/build_wheel.sh` (line ~195):

```bash
_PYTHON_HOST_PLATFORM=${mac_version} \
    ARCHFLAGS="-arch arm64" \
    python -m build --wheel --no-isolation \
    --outdir "$whl_tmp_dir" \
    -C--plat-name="${mac_version//[-.]/_}"
```

Followed by dependency relocation:

```bash
delocate-wheel -v <wheel_file>
```

### Windows x86_64 — primary wheel build command

Located in `.ci/pytorch/windows/internal/setup.bat` (line ~89):

```bat
%PYTHON_EXEC% -m build --wheel --no-isolation --outdir "%PYTORCH_FINAL_PACKAGE_DIR%"
```

### Windows ARM64 — primary wheel build command

Located in `.ci/pytorch/windows/arm64/build_pytorch.bat` (line ~51):

```bat
python -m build --wheel --no-isolation --outdir "%PYTORCH_FINAL_PACKAGE_DIR%"
```

### Docker run command (Linux only)

Executed inside `_binary-build-linux.yml`:

```bash
container_name=$(docker run \
  -e BINARY_ENV_FILE -e BUILD_ENVIRONMENT -e DESIRED_CUDA \
  -e DESIRED_PYTHON -e GITHUB_ACTIONS -e GPU_ARCH_TYPE \
  -e GPU_ARCH_VERSION -e LIBTORCH_VARIANT -e PACKAGE_TYPE \
  -e PYTORCH_FINAL_PACKAGE_DIR -e PYTORCH_ROOT -e SKIP_ALL_TESTS \
  -e PYTORCH_EXTRA_INSTALL_REQUIREMENTS \
  --tty --detach \
  -v "${GITHUB_WORKSPACE}:/pytorch" \
  -v "${RUNNER_TEMP}/artifacts:/artifacts" \
  -w / \
  "${DOCKER_IMAGE}"
)
docker exec -t -w "${PYTORCH_ROOT}" "${container_name}" \
  bash -c "bash .circleci/scripts/binary_populate_env.sh"
docker exec -t "${container_name}" \
  bash -c "source ${BINARY_ENV_FILE} && bash /pytorch/.ci/manywheel/build.sh"
```

---

## 4. Docker / Manylinux Environments

### Linux manywheel images

| Architecture | Base Image | Docker Image Name |
|---|---|---|
| x86_64 | `quay.io/pypa/manylinux_2_28_x86_64` (AlmaLinux 8) | `pytorch/manylinux2_28-builder:<tag>` |
| aarch64 | `arm64v8/almalinux:8` | `pytorch/manylinuxaarch64-builder:<tag>` |
| s390x | Custom Ubuntu-based | `pytorch/manylinuxs390x-builder:<tag>` |

The `<tag>` suffix identifies the GPU backend, for example:
- `cpu` — CPU-only wheel
- `cu121` — CUDA 12.1
- `cu124` — CUDA 12.4
- `rocm6.2` — ROCm 6.2

The **manylinux_2_28** standard ensures wheels are compatible with any Linux
distribution using glibc ≥ 2.28 (AlmaLinux 8 / RHEL 8 and later).

### Compiler toolchain inside Linux images

- GCC via `gcc-toolset-13` (devtoolset-13) from the AlmaLinux SCL
- `PATH` and `LD_LIBRARY_PATH` updated to use toolset binaries
- CUDA toolkit installed when building CUDA wheels
- ROCm HIP SDK installed when building ROCm wheels

### macOS

No Docker container is used. The build runs directly on a macOS ARM64 GitHub
Actions runner (`macos-14` or similar). Homebrew is used to install `libomp`.

### Windows

No Docker container is used. The build runs on a Windows GitHub Actions runner
(`windows.12xlarge`). CUDA/XPU toolkits are installed by the `.bat` helper
scripts during CI setup.

---

## Key Environment Variables

| Variable | Description |
|---|---|
| `PYTORCH_BUILD_VERSION` | Wheel version string, e.g. `2.6.0.dev20250305+cpu` |
| `PYTORCH_BUILD_NUMBER` | Build number suffix (default `1`) |
| `GPU_ARCH_TYPE` | Backend: `cpu`, `cuda`, `cuda-aarch64`, `rocm`, `xpu`, `cpu-aarch64`, `cpu-s390x` |
| `GPU_ARCH_VERSION` | Backend version, e.g. `12.4` for CUDA |
| `DESIRED_CUDA` | Legacy variable, e.g. `cpu`, `cu121` |
| `DESIRED_PYTHON` | Python version to build for, e.g. `3.10` |
| `PYTORCH_FINAL_PACKAGE_DIR` | Destination for finished wheels (`/artifacts` in CI) |
| `BINARY_ENV_FILE` | Path to env file written by `binary_populate_env.sh` (`/tmp/env`) |
| `PACKAGE_TYPE` | `manywheel` (Linux), `wheel` (macOS/Windows), or `libtorch` |
