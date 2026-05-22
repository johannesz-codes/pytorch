# Triton Wheel Pipeline in the PyTorch Repository

This document traces every place in the repository where Triton wheels are built,
fetched, pinned, and distributed — from source clone to the PyTorch nightly/release
package index.

<!-- toc -->

- [Section 1 – Files and Workflows Mentioning Triton Packaging](#section-1--files-and-workflows-mentioning-triton-packaging)
- [Section 2 – Workflows That Build, Fetch, or Publish Triton Wheels](#section-2--workflows-that-build-fetch-or-publish-triton-wheels)
- [Section 3 – Scripts Involved and Exact Commands](#section-3--scripts-involved-and-exact-commands)
- [Section 4 – How Triton Reaches the PyTorch Nightly/Release Index](#section-4--how-triton-reaches-the-pytorchnightlyrelease-index)
- [Section 5 – Final Conclusion](#section-5--final-conclusion)

<!-- tocstop -->

---

## Section 1 – Files and Workflows Mentioning Triton Packaging

### Version and commit-pin files

| File | Content | Purpose |
|------|---------|---------|
| `.ci/docker/triton_version.txt` | `3.6.0` | Canonical version string used in wheel names and `install_requires` metadata |
| `.ci/docker/triton_xpu_version.txt` | `3.7.0` | Same for the Intel XPU variant |
| `.ci/docker/ci_commit_pins/triton.txt` | `9844da955a9db14ec69c9aac828ee9803085e288` | Exact upstream commit built for CUDA/ROCm/aarch64 |
| `.ci/docker/ci_commit_pins/triton-xpu.txt` | `307748db7742a0f8259a7ea0336909eb55d2051a` | Exact upstream commit built for XPU |
| `.ci/docker/ci_commit_pins/triton-cpu.txt` | `c7711371cace304afe265c1ffa906415ab82fc66` | Exact upstream commit for the CPU backend (Docker image only) |
| `.github/ci_commit_pins/triton.txt` | Same as above | Symlink-equivalent used by the wheel-build workflow |

### GitHub Actions workflows that mention Triton

| Workflow file | How Triton is referenced |
|---------------|--------------------------|
| `.github/workflows/build-triton-wheel.yml` | **Primary build pipeline** — clones upstream Triton, builds wheels, uploads them to S3/R2 |
| `.github/workflows/nightly.yml` | References `triton-lang` as a monitored repo; pins `triton_version.txt` for nightly Docker images |
| `.github/workflows/docker-release.yml` | Reads `triton_version.txt` and constructs a nightly version string (`3.6.0+git9844da95`) used when building the release Docker image |
| `.github/workflows/inductor-unittest.yml` | Defines a `inductor-triton-cpu` CI variant that installs the Triton CPU backend during image setup |
| `.github/workflows/build-vllm-wheel.yml` | Installs Triton as a runtime dependency when building the vLLM wheel |
| `.github/workflows/docker-builds.yml` | Passes `TRITON=1` build arg; the Docker build step calls `install_triton.sh` |

### Build/packaging scripts

| Script | Role |
|--------|------|
| `.github/scripts/build_triton_wheel.py` | Core build logic: clones upstream Triton at a pinned commit, patches `__version__`, runs `setup.py bdist_wheel` |
| `.github/scripts/windows/build_triton.bat` | Windows wrapper that installs Python tooling then calls `build_triton_wheel.py` |
| `.github/scripts/amd/package_triton_wheel.sh` | Copies ROCm shared libraries (`libamdhip64.so`, `libamd_comgr.so`, etc.) into the wheel tree before it is zipped |
| `.github/scripts/amd/patch_triton_wheel.sh` | Post-packaging patchelf pass: re-writes `RPATH` entries inside the ROCm wheel so all bundled `.so` files find each other |
| `.ci/docker/common/install_triton.sh` | Used inside Docker image builds to clone Triton at the pinned commit and `pip install` the locally built wheel |
| `.ci/pytorch/binary_populate_env.sh` | Sets `PYTORCH_EXTRA_INSTALL_REQUIREMENTS` to declare Triton as a wheel dependency; uses the plain version string for release builds and the `+git<shorthash>` suffix for nightly dev builds |
| `.circleci/scripts/binary_populate_env.sh` | Identical logic for CircleCI-based builds |
| `.circleci/scripts/binary_upload.sh` | Generic S3/R2 upload script invoked by the upload step of `build-triton-wheel.yml` |
| `scripts/install_triton_wheel.sh` | Developer helper: installs the correct pinned Triton wheel from `download.pytorch.org` |
| `scripts/release/apply-release-changes.sh` | Patches `build-triton-wheel.yml` to target a release branch instead of `main` when cutting a release |

### Dependency declarations

`setup.py` (line ~1569) reads `PYTORCH_EXTRA_INSTALL_REQUIREMENTS` and appends its
pipe-delimited contents to `install_requires`.  `binary_populate_env.sh` is the
script that populates that variable with a Triton requirement of the form:

```
triton==3.6.0+git9844da95; platform_system == 'Linux'
```

The `platform_system` marker means Triton is only declared as a dependency on Linux
wheels; macOS and Windows PyTorch wheels omit it.

---

## Section 2 – Workflows That Build, Fetch, or Publish Triton Wheels

### `.github/workflows/build-triton-wheel.yml` — the end-to-end Triton build pipeline

**Triggers**

- Every push to `main`
- Release candidate tags (`v<x>.<y>.<z>-rc<n>`)
- `ciflow/triton_binaries/*` tags
- `workflow_dispatch`
- Pull requests that touch the listed path filters

**Build matrix**

| Axis | Values |
|------|--------|
| Python version | 3.10, 3.11, 3.12, 3.13, 3.13t, 3.14, 3.14t |
| Device backend | `cuda`, `rocm`, `xpu`, `aarch64` |
| Linux builder image | `pytorch/manylinux2_28-builder:cpu` (CUDA/XPU), `pytorch/manylinux2_28-builder:rocm7.2` (ROCm), `pytorch/manylinux2_28_aarch64-builder:cpu-aarch64` (aarch64) |
| Windows | `xpu` device only, on a `windows.4xlarge` runner |

**Jobs**

1. `build-wheel` (Linux) — runs inside the manylinux Docker image:
   - Calls `.github/scripts/build_triton_wheel.py --device=<device> [--release] [--with-clang-ldd]`
   - For CUDA/XPU: runs `auditwheel repair --plat manylinux_2_28_x86_64` to produce a
     self-contained manywheel
   - Artifact name: `triton-wheel-<py_vers>-<device>-manylinux_2_28_x86_64`

2. `build-wheel-win` (Windows) — calls `.github/scripts/windows/build_triton.bat`,
   produces a platform wheel, uploads as `triton-wheel-<py_vers>-<device>`

3. `upload-wheel` — downloads all build artifacts, then calls
   `.circleci/scripts/binary_upload.sh` which:
   - On pushes to `main`: uploads to `s3://pytorch/whl/nightly/` and
     `s3://pytorch-downloads/whl/nightly/` (Cloudflare R2 mirror)
   - On RC tags: uploads to `s3://pytorch/whl/test/`
   - On `workflow_dispatch` / PRs: dry-run only (no actual upload)

---

## Section 3 – Scripts Involved and Exact Commands

### `.github/scripts/build_triton_wheel.py`

```python
triton_repo = "https://github.com/openai/triton"          # CUDA / aarch64 (legacy URL; the repo was
                                                           # transferred to github.com/triton-lang/triton,
                                                           # but GitHub redirects the old URL)
# or
triton_repo = "https://github.com/intel/intel-xpu-backend-for-triton"   # XPU

check_call(["git", "clone", triton_repo, "triton"], cwd=tmpdir)
check_call(["git", "fetch", "origin", commit_hash], cwd=triton_basedir)
check_call(["git", "checkout", commit_hash], cwd=triton_basedir)

# Patch __version__ to the PyTorch-specific version string
patch_init_py(triton_pythondir / "triton" / "__init__.py", version=triton_version)

# Build the wheel
check_call([sys.executable, "setup.py", "bdist_wheel"], cwd=triton_setupdir, env=env)
```

The `TRITON_WHEEL_NAME` environment variable overrides the wheel package name:
- CUDA / aarch64 → `triton`
- ROCm → `triton-rocm`
- XPU → `triton-xpu`

### `.ci/docker/common/install_triton.sh` (Docker image builds)

```bash
git clone --recursive ${TRITON_REPO} triton
git checkout ${TRITON_PINNED_COMMIT}
python -m build --wheel --no-isolation
pip install dist/*.whl
cp dist/*.whl /opt/triton      # for multi-stage builds
```

The pinned commit is read via `get_pinned_commit` from
`.ci/docker/ci_commit_pins/triton.txt` (or `triton-xpu.txt` / `triton-cpu.txt`).

### `.circleci/scripts/binary_upload.sh` (upload step)

```bash
aws s3 cp --no-progress --acl public-read "${pkg}" \
    "s3://pytorch/whl/${UPLOAD_CHANNEL}/" \
    --metadata "checksum-sha256=${sha256}"

# If R2_UPLOAD=true (main branch pushes):
AWS_ACCESS_KEY_ID="${R2_ACCESS_KEY_ID}" ... \
aws s3 cp "${pkg}" "s3://pytorch-downloads/whl/${UPLOAD_CHANNEL}/" \
    --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
```

`UPLOAD_CHANNEL` is `nightly` for `main` pushes and `test` for RC-tagged pushes.
`UPLOAD_SUBFOLDER` is left empty so Triton wheels land at the root of the channel
(e.g., `s3://pytorch/whl/nightly/*.whl`), which is served as
`https://download.pytorch.org/whl/nightly/`.

### `scripts/install_triton_wheel.sh` (developer helper)

```bash
TRITON_VERSION="triton==$(cat .ci/docker/triton_version.txt)"
TRITON_COMMIT_ID="$(head -c 8 .ci/docker/ci_commit_pins/triton.txt)"
pip install --index-url https://download.pytorch.org/whl/nightly/ \
    "${TRITON_VERSION}+git${TRITON_COMMIT_ID}"
```

---

## Section 4 – How Triton Reaches the PyTorch Nightly/Release Index

### Nightly builds

1. Every push to `main` triggers `build-triton-wheel.yml`.
2. The workflow clones the Triton repo at the commit pinned in
   `.ci/docker/ci_commit_pins/triton.txt`, builds wheels for every
   (Python version × device backend) combination, and uploads them to:
   - `https://download.pytorch.org/whl/nightly/` (AWS S3, public-read)
   - `https://pytorch-downloads.r2.cloudflarestorage.com/whl/nightly/` (Cloudflare R2 mirror)
3. The wheel version string is `<version>+git<shorthash>` (e.g., `3.6.0+git9844da95`),
   distinguishing nightly builds from release wheels.
4. `binary_populate_env.sh` injects this exact version string into
   `PYTORCH_EXTRA_INSTALL_REQUIREMENTS`, which `setup.py` then bakes into the
   `install_requires` metadata of the PyTorch wheel that is being built in the
   parallel nightly PyTorch build. Consequently, `pip install torch` from the
   nightly index automatically pulls the matching Triton wheel.

### Release (RC and final)

1. When a release branch is cut, `scripts/release/apply-release-changes.sh` patches
   `build-triton-wheel.yml` to target the `release/<ver>.x` Triton branch instead of
   `main`.
2. RC pushes (`v<x>.<y>.<z>-rc<n>`) upload to `https://download.pytorch.org/whl/test/`.
3. For the final public release:
   - Standard (CUDA) wheels: the `triton` requirement points to PyPI
     (`https://pypi.org/project/triton/`), published by the `triton-lang` organization.
     PyTorch maintainers coordinate the timing with the Triton team (the request
     includes the pinned commit hash from `.ci/docker/ci_commit_pins/triton.txt`) so
     that the matching PyPI release is available before PyTorch's final release.
     There is no automated fallback; if the Triton PyPI release is absent, the
     PyTorch release process stalls.
   - ROCm wheels: depend on `https://download.pytorch.org/whl/triton-rocm/` (no PyPI
     package exists for `triton-rocm`).
   - XPU wheels: depend on `https://download.pytorch.org/whl/triton-xpu/`.

### Docker images

The `docker-builds.yml` workflow passes `TRITON=1` to the Docker build, which invokes
`.ci/docker/common/install_triton.sh`. That script builds Triton from source inside
the image (same pinned commit) and installs the resulting wheel, so CI containers carry
a pre-compiled Triton without needing to download it from the index at runtime.

---

## Section 5 – Final Conclusion

**Is Triton built in the PyTorch repository?**

Yes. The PyTorch repository contains a complete, end-to-end Triton wheel build
pipeline. Triton is **not** a pre-built external artifact for nightly and RC builds;
it is compiled from source inside this repository's CI.

**Where exactly, by which workflow and script?**

| Component | Path |
|-----------|------|
| Triggering workflow | `.github/workflows/build-triton-wheel.yml` |
| Core build script | `.github/scripts/build_triton_wheel.py` |
| Windows build wrapper | `.github/scripts/windows/build_triton.bat` |
| ROCm library packaging | `.github/scripts/amd/package_triton_wheel.sh` |
| ROCm wheel patching | `.github/scripts/amd/patch_triton_wheel.sh` |
| Upload script | `.circleci/scripts/binary_upload.sh` |
| Docker image build | `.ci/docker/common/install_triton.sh` |
| Commit pins | `.ci/docker/ci_commit_pins/triton.txt`, `triton-xpu.txt`, `triton-cpu.txt` |
| Version strings | `.ci/docker/triton_version.txt`, `.ci/docker/triton_xpu_version.txt` |

**Where does PyTorch obtain Triton wheels for releases?**

- For the standard `triton` package on final releases: from PyPI
  (`https://pypi.org/project/triton/`), published by the `triton-lang` organization
  (the repository was originally under the `openai` GitHub org and was transferred to
  `triton-lang`; the PyPI package owner is `triton-lang`).
- For `triton-rocm` and `triton-xpu` on all builds: from the PyTorch package index
  (`https://download.pytorch.org/whl/`), where the wheels were previously uploaded by
  this repository's own `build-triton-wheel.yml` workflow.

**Is there a combined build pipeline that tightly couples Torch and Triton during nightly publication?**

Yes. The version string baked into Triton nightly wheels (`3.6.0+git9844da95`) exactly
matches the `PYTORCH_EXTRA_INSTALL_REQUIREMENTS` value that `binary_populate_env.sh`
injects when building PyTorch wheels. This creates a tight, per-commit coupling between
a given PyTorch nightly wheel and its companion Triton nightly wheel. Installing the
PyTorch nightly wheel will automatically resolve and install the matching Triton nightly
wheel from `https://download.pytorch.org/whl/nightly/`.

**Summary table**

| Question | Answer |
|----------|--------|
| Is Triton built inside this repo? | **Yes** — for nightly and RC wheels on all platforms |
| Are Triton wheels downloaded from elsewhere? | **Yes** — for standard CUDA final releases, from PyPI |
| Are Triton wheels mirrored into the PyTorch index? | **Yes** — ROCm and XPU wheels are uploaded to `download.pytorch.org` and have no separate PyPI presence |
| Is Triton only a loose pip dependency? | **No** — the exact `+git<hash>` suffix ties each PyTorch nightly to a specific Triton commit |
