XPU Backend Technical Overview
==============================

This document provides a comprehensive, code-based overview of the XPU backend
(``torch.xpu``) as implemented in this repository. XPU targets Intel GPU hardware
and is built on the SYCL/oneAPI software stack.

.. contents::
   :local:
   :depth: 2

---

1. Backend Architecture
-----------------------

Integration Model
^^^^^^^^^^^^^^^^^

XPU is integrated as a first-class device backend following the same architecture
pattern as CUDA and MPS. The integration has three distinct layers:

1. **c10/xpu** — Low-level runtime primitives (device management, streams, caching
   allocator, guard implementation). These depend only on SYCL and ``c10`` itself.
2. **aten/src/ATen/xpu** — ATen-level abstractions built on top of ``c10/xpu``
   (context, generator, pinned memory, empty tensor allocation, XPU graphs).
3. **torch/xpu** and **torch/csrc/xpu** — Python-facing API and its CPython
   bindings, mirroring the ``torch.cuda`` surface.

Lazy Initialization
^^^^^^^^^^^^^^^^^^^

XPU state is lazily initialized, controlled by ``torch/xpu/__init__.py:_lazy_init()``.
The first call that requires an XPU device triggers ``torch._C._xpu_init()``, which
sets up the caching allocator and the peer-to-peer access cache. Queued calls
accumulated before initialization are flushed at that point.

Key Runtime Components
^^^^^^^^^^^^^^^^^^^^^^

- **Device management** — ``c10/xpu/XPUFunctions.h`` and ``c10/xpu/XPUFunctions.cpp``
  expose ``device_count()``, ``current_device()``, ``set_device()``,
  ``exchange_device()``, and ``get_raw_device()`` (returning ``sycl::device&``).

- **Streams** — ``c10/xpu/XPUStream.h`` and ``XPUStream.cpp``.  An ``XPUStream``
  wraps a ``sycl::queue``.  Two pools per device are maintained (normal- and
  high-priority), each holding up to 32 queues, allocated round-robin.
  The ``XPUStream`` class provides an implicit conversion to ``sycl::queue&`` and
  exposes ``synchronize()``, ``query()``, and ``priority()``.

- **Events** — ``c10/xpu/XPUEvent.h``.  Events are stored as ``sycl::event*``
  objects.  Timing between events requires SYCL compiler ≥ 2025.0.0
  (``SYCL_COMPILER_VERSION ≥ 20250000``), enforced at runtime.

- **Memory allocator** — ``c10/xpu/XPUCachingAllocator.h`` and
  ``XPUCachingAllocator.cpp``.  Implements a caching block allocator with
  small- and large-pool separation (threshold: 1 MB), memory-pool semantics
  (``MemPool``), per-stream recording, and pluggable allocator support
  (``XPUPluggableAllocator`` in ``torch/csrc/xpu/XPUPluggableAllocator.cpp``).
  Python memory management API lives in ``torch/xpu/memory.py``.

- **Device guard** — ``c10/xpu/impl/XPUGuardImpl.h`` implements
  ``c10::impl::DeviceGuardImplInterface`` for XPU, wiring device set/exchange,
  stream exchange, event operations, and allocator stream recording into the
  standard c10 guard mechanism.

- **Hooks** — ``aten/src/ATen/xpu/detail/XPUHooks.h`` implements
  ``at::XPUHooksInterface`` (registered via ``REGISTER_XPU_HOOKS``).  The hooks
  bridge the ATen-level API (generators, pinned allocator, device-from-pointer,
  ``hasXPU()``) to the underlying runtime.

- **RNG** — ``aten/src/ATen/xpu/XPUGeneratorImpl.h/cpp``.  Uses a Philox
  generator (``PhiloxXpuState``), compatible with XPU graph capture.

- **XPU Graphs** — ``aten/src/ATen/xpu/XPUGraph.h/cpp`` and
  ``torch/csrc/xpu/Graph.cpp``.  Uses
  ``sycl::ext::oneapi::experimental::command_graph`` for command graph capture
  and replay.  Python API exposed in ``torch/xpu/graphs.py``.

- **Level Zero interop** — ``aten/src/ATen/xpu/detail/LazyLevelZero.cpp``
  lazily loads ``libze_loader.so`` (or ``ze_loader.dll`` on Windows) via
  ``at::DynamicLibrary``, giving access to low-level Intel Level Zero API calls
  (kernel inspection, memory query) without a hard link-time dependency.

Execution Path from Python API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A call such as ``torch.mm(a, b)`` on an XPU tensor follows this path:

1. Python tensor method dispatches through the Python dispatcher to C++.
2. The dispatcher consults the dispatch key set; XPU tensors carry
   ``DispatchKey::XPU`` (and ``DispatchKey::AutogradXPU`` when gradients are
   enabled).
3. The ``AutogradXPU`` key is handled first by the autograd engine
   (``aten/src/ATen/core/VariableFallbackKernel.cpp``), which records the
   operation for backward and re-dispatches to ``DispatchKey::XPU``.
4. The XPU kernel registered for ``mm`` (``mm_out_xpu``) is called.  For
   ``mm``, this is implemented in
   ``aten/src/ATen/native/mkldnn/xpu/Blas.cpp``, which delegates to oneDNN
   SYCL primitives.

---

2. Dispatch System Integration
-------------------------------

Dispatch Keys
^^^^^^^^^^^^^

The following dispatch keys are defined for XPU in
``c10/core/DispatchKey.h``/``DispatchKey.cpp``:

- ``DispatchKey::XPU`` — main backend key; kernel implementations are registered
  here.
- ``DispatchKey::AutogradXPU`` — autograd layer key.  A catch-all fallback
  registered in ``aten/src/ATen/core/VariableFallbackKernel.cpp`` handles all
  operators without explicit backward kernels.
- ``DispatchKey::AutocastXPU`` — automatic mixed-precision; referenced in
  ``c10/core/DispatchKey.h`` (line 352) and handled in
  ``torch/amp/autocast_mode.py``.

ATen Operator Registration
^^^^^^^^^^^^^^^^^^^^^^^^^^

Operators are declared in ``aten/src/ATen/native/native_functions.yaml`` with
explicit ``XPU:`` dispatch entries.  As of this writing there are **29 explicit
XPU dispatch entries** covering GEMM variants, scaled matmul, scaled dot-product
attention, int4/int8 packed matmul, and ``addbmm``.  The full list includes:

- ``addmm_out_xpu``, ``_addmm_dtype_xpu``, ``_addmm_dtype_out_xpu``,
  ``addmm_activation_out_xpu``
- ``addmv_out_xpu``
- ``baddbmm_out_xpu``, ``_baddbmm_dtype_xpu``, ``_baddbmm_out_dtype_xpu``
- ``bmm_out_xpu``, ``_bmm_dtype_xpu``, ``_bmm_out_dtype_xpu``
- ``mm_out_xpu``, ``_mm_dtype_xpu``, ``_mm_dtype_out_xpu``
- ``_int_mm_xpu``, ``_int_mm_out_xpu``
- ``_weight_int4pack_mm_xpu``, ``_weight_int8pack_mm_xpu``
- ``_scaled_mm_xpu``, ``_scaled_mm_out_xpu``, ``_scaled_mm_xpu_v2``,
  ``_scaled_mm_xpu_v2_out``
- ``addbmm`` / ``addbmm_`` / ``addbmm_out`` (shared ``CPU, CUDA, XPU`` entry)
- ``_fused_sdp_choice_xpu``, ``_scaled_dot_product_flash_attention_xpu``,
  ``_scaled_dot_product_fused_attention_overrideable_xpu``,
  ``_scaled_dot_product_flash_attention_backward_xpu``

Generated vs. Manually-Written Registration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The code-generation pipeline (``torchgen/gen.py``) generates an XPU registration
file when invoked with the ``--xpu`` flag
(``cmake/Codegen.cmake`` passes ``GEN_XPU_FLAG``). The generated headers and
sources are tracked as ``xpu_generated_headers`` / ``xpu_generated_sources`` and
linked into the build target ``ATEN_XPU_FILES_GEN_TARGET``.

The generated code typically emits:

.. code-block:: c++

   // Example from generated RegisterXPU.cpp (conceptual)
   TORCH_LIBRARY_IMPL(aten, XPU, m) {
     m.impl("mm.out", &at::native::xpu::mm_out);
   }

For XPU, the codegen also injects ``#include <ATen/xpu/EmptyTensor.h>`` (see
``torchgen/dest/register_dispatch_key.py`` line 66–68), which lives in the
in-tree sources rather than the external submodule.

The vast majority of ATen operator coverage for XPU comes from the external
submodule ``third_party/torch-xpu-ops`` (see §3 below).  Operators not present
in ``native_functions.yaml`` under an ``XPU:`` key rely on
``CompositeImplicitAutograd`` or ``CompositeExplicitAutograd`` decompositions,
which transparently decompose into primitives that do have XPU kernels.

Fallback Behavior
^^^^^^^^^^^^^^^^^

There is no explicit CPU fallback registered at the dispatcher level for XPU.
Operations that have no XPU kernel and no composite decomposition will raise a
``NotImplementedError`` at runtime.  Many higher-level operations (element-wise
math, reductions, etc.) are covered by ``CompositeImplicitAutograd`` kernels that
decompose to lower-level primitives, which in turn are provided by
``torch-xpu-ops``.

---

3. Kernel Implementation Locations
------------------------------------

a) In-tree ATen XPU Kernels and Wrappers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 55 45
   :header-rows: 1

   * - Path
     - Role
   * - ``aten/src/ATen/native/mkldnn/xpu/Blas.cpp``
     - GEMM family: ``addmm``, ``addmv``, ``bmm``, ``mm``, ``baddbmm``,
       ``addmm_activation``, scaled matmul.  Calls oneDNN SYCL matmul
       primitive via ``aten/src/ATen/native/mkldnn/xpu/detail/Matmul.cpp``.
   * - ``aten/src/ATen/native/mkldnn/xpu/ScaledBlas.cpp``
     - Scaled BLAS wrappers (``_scaled_mm``, FP8 variants).
   * - ``aten/src/ATen/native/mkldnn/xpu/Conv.cpp``, ``Conv.h``
     - 1D/2D/3D convolution and transposed convolution; delegates to
       ``detail/Conv.cpp`` / ``detail/Deconv.cpp`` via oneDNN convolution
       primitive.
   * - ``aten/src/ATen/native/mkldnn/xpu/Linear.cpp``
     - Dense linear (``mm``-backed); delegates to ``detail/Matmul.cpp``.
   * - ``aten/src/ATen/native/mkldnn/xpu/Attention.cpp``
     - oneDNN-backed scaled dot-product attention.  Uses
       ``detail/Attention.cpp``.
   * - ``aten/src/ATen/native/mkldnn/xpu/qconv.cpp``, ``qlinear.cpp``
     - Quantized convolution and linear.
   * - ``aten/src/ATen/native/mkldnn/xpu/FusionUtils.cpp``
     - Utilities for fused post-ops (e.g., bias-add, activation fusion via
       oneDNN attributes).
   * - ``aten/src/ATen/native/transformers/xpu/attention.cpp``
     - Flash attention forward pass; delegates to ``sycltla::flash_attention_forward``
       declared in ``flash_attn/flash_api.h`` (provided by ``torch-xpu-ops``).
   * - ``aten/src/ATen/native/transformers/xpu/attention_backward.cpp``
     - Flash attention backward pass.
   * - ``aten/src/ATen/native/transformers/xpu/sdp_utils.cpp``, ``sdp_utils.h``
     - SDP backend selection logic for XPU (``_fused_sdp_choice_xpu``).
   * - ``aten/src/ATen/xpu/XPUScaledBlas.cpp``, ``XPUScaledBlas.h``
     - Complex-valued scaled BLAS operations.
   * - ``aten/src/ATen/xpu/EmptyTensor.cpp``, ``EmptyTensor.h``
     - ``empty_xpu`` tensor allocation (storage from ``XPUCachingAllocator``).
   * - ``aten/src/ATen/xpu/CachingHostAllocator.cpp``
     - Pinned (host-accessible, SYCL ``usm::alloc::host``) memory allocator.
   * - ``aten/src/ATen/xpu/PeerToPeerAccess.cpp``
     - Peer device memory access query and enablement.

b) oneDNN / SYCL Primitives Layer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``aten/src/ATen/native/mkldnn/xpu/detail/`` contains the oneDNN integration:

- ``oneDNNContext.h`` / ``oneDNNContext.cpp`` — per-device oneDNN engine and
  stream context backed by the current ``sycl::queue``.
- ``oneDNN.h`` — top-level include for oneDNN-backed XPU operations; provides
  ``at::native::onednn::*`` entry points.
- ``Matmul.cpp``, ``Conv.cpp``, ``Deconv.cpp``, ``Attention.cpp`` — wrappers
  that build oneDNN primitive descriptors, memory descriptors, and submit
  operations to the SYCL queue.
- ``Attr.h`` — oneDNN attribute/post-op builder abstraction.
- ``LRUCache.h`` — LRU cache for oneDNN primitive handles to amortize
  descriptor creation cost.
- ``QConv.cpp``, ``QMatmul.cpp``, ``WoQMatmul.cpp`` — quantized and
  weight-only-quantized variants.

The oneDNN GPU library itself (``libdnnl.a``) is built from source as an
``ExternalProject`` at ``cmake/Modules/FindMKLDNN.cmake``; it is fetched from
``https://github.com/uxlfoundation/oneDNN`` at tag ``v3.10.2``, compiled with
``-DDNNL_GPU_RUNTIME=SYCL``.

c) External Submodule: torch-xpu-ops
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``third_party/torch-xpu-ops`` (hosted at ``https://github.com/intel/torch-xpu-ops``,
pinned commit in ``third_party/xpu.txt``) is the primary provider of XPU kernel
implementations for the ATen operator set.  At build time, if the directory is
not yet present, ``caffe2/CMakeLists.txt`` clones and checks out the pinned
commit automatically.  The submodule is compiled as the CMake target
``torch_xpu_ops`` and linked whole-archive into ``libtorch_xpu.so``.

The submodule provides:

- Element-wise and unary ops, reductions, indexing, RNG, sorting, FFT,
  sparse ops, and more.
- Flash attention kernel (``sycltla::flash_attention_forward`` referenced in
  ``aten/src/ATen/native/transformers/xpu/attention.cpp``).
- Generated per-operator ``ATen/native/xpu/`` headers consumed by the in-tree
  ``Blas.cpp`` (e.g., ``#include <ATen/native/xpu/Blas.h>``).

The generated output of ``torch-xpu-ops`` is installed under
``${TORCH_INSTALL_INCLUDE_DIR}/ATen/xpu/``.

---

4. Operator Coverage (Qualitative)
------------------------------------

The following categories have direct XPU kernel support, either via explicit
``native_functions.yaml`` entries or via ``torch-xpu-ops``:

- **GEMM / linear algebra** — ``mm``, ``bmm``, ``addmm``, ``addmv``,
  ``baddbmm``, ``addbmm``, ``_int_mm``, ``_scaled_mm``,
  ``_weight_int4pack_mm``, ``_weight_int8pack_mm``.
- **Convolution** — forward, backward, transposed, 1D/2D/3D (via oneDNN SYCL).
- **Attention / Transformers** — flash attention (forward + backward), fused
  SDP attention (via oneDNN), ``_fused_sdp_choice`` backend selection.
- **Element-wise ops** — arithmetic, comparison, type promotion, unary
  functions — provided by ``torch-xpu-ops``.
- **Reductions** — sum, mean, max, min, norm — provided by ``torch-xpu-ops``.
- **RNG** — Philox-based random generation in
  ``aten/src/ATen/xpu/XPUGeneratorImpl.cpp``.
- **Memory ops** — tensor allocation (``empty_xpu``), copy (with SYCL DMA),
  fill, zero.
- **Quantized ops** — quantized convolution and linear (``qconv.cpp``,
  ``qlinear.cpp``, ``detail/QConv.cpp``, ``detail/QMatmul.cpp``).
- **Weight-only quantization** — ``detail/WoQMatmul.cpp``.
- **FFT** — via ``torch-xpu-ops`` (references ``_fft_r2c_xpu`` in
  ``torch/_meta_registrations.py``).
- **Sparse ops** — partially covered by ``torch-xpu-ops``; ``SparseCsrTensor.cpp``
  uses ``DispatchKey::kXPU``.
- **Autocast / AMP** — ``DispatchKey::AutocastXPU`` with standard PyTorch AMP
  infrastructure (``torch/amp/autocast_mode.py``).

CPU fallback is not registered at the framework level.  Operations outside the
above categories will raise ``NotImplementedError`` unless covered by a composite
decomposition.

---

5. Autograd Support
--------------------

Autograd for XPU is handled by two mechanisms:

1. **AutogradXPU fallback** — ``aten/src/ATen/core/VariableFallbackKernel.cpp``
   registers a catch-all ``AutogradXPU`` fallback that calls the generic
   ``basic_autograd_not_implemented_fallback``.  This means any operator with
   a composite or explicit XPU kernel automatically gets autograd tracking for
   free, as long as the autograd engine can differentiate it using existing
   backward formulas.

2. **CompositeImplicitAutograd** — operators declared with
   ``CompositeImplicitAutograd`` in ``native_functions.yaml`` are device-agnostic
   and participate in autograd without device-specific backward kernels.  This
   covers the large majority of higher-level ATen operations.

Explicit backward kernels for XPU-specific operations:

- ``_scaled_dot_product_flash_attention_backward_xpu`` — registered in
  ``native_functions.yaml`` (line 15215) and implemented in
  ``aten/src/ATen/native/transformers/xpu/attention_backward.cpp``.

The ``XPUGeneratorImpl`` supports graph-safe state capture
(``graphsafe_set_state`` / ``graphsafe_get_state``) required by XPU graph
capture and replay (``XPUGraph``).

---

6. Build System and Dependencies
----------------------------------

CMake Options
^^^^^^^^^^^^^

- ``USE_XPU`` (default: ``ON``) — master switch; disables XPU if SYCL
  toolkit not found, with a warning.
- ``USE_XCCL`` — enables Intel Collective Communications Library (XCCL) for
  distributed training on XPU.  Automatically disabled if ``USE_XPU=OFF``.
- ``XPU_ENABLE_KINETO`` — enables XPUPTI profiling support (via Kineto).
  Set automatically on Linux; requires SYCL compiler ≥ 2025.0.1 on Windows.
  Can be forced with environment variable ``$XPU_ENABLE_KINETO``.

Environment Variables
^^^^^^^^^^^^^^^^^^^^^

- ``XPU_ENABLE_KINETO`` — force-enable Kineto/XPUPTI support regardless of
  platform defaults.
- ``MAX_JOBS`` — controls the parallel build job count for the oneDNN
  ``ExternalProject``.
- ``LIBTORCH_LIB_PATH`` — path to search for pre-built ``c10_xpu`` library
  (used in ``BUILD_LIBTORCHLESS`` mode).

Dependencies
^^^^^^^^^^^^

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Dependency
     - Role
   * - SYCL / Intel oneAPI DPC++ compiler (``icpx`` / ``icx``)
     - Required host compiler and runtime (``cmake/public/xpu.cmake`` finds
       ``SYCLToolkit``; links ``torch::sycl`` and ``torch::xpurt``).
   * - oneDNN ≥ v3.10.2
     - GPU convolution and matmul primitives; built from source as
       ``xpu_mkldnn_proj`` in ``cmake/Modules/FindMKLDNN.cmake``.
   * - Intel Level Zero (``libze_loader.so``)
     - Low-level device/kernel introspection; loaded lazily at runtime
       (``aten/src/ATen/xpu/detail/LazyLevelZero.cpp``).
   * - OpenCL (``libOpenCL``)
     - Linked when ``USE_XPU`` is enabled
       (``cmake/Dependencies.cmake``, line 717).
   * - XCCL (Intel Collective Communications Library)
     - Optional distributed training backend
       (``cmake/Dependencies.cmake``, ``USE_XCCL`` section).
   * - ``third_party/torch-xpu-ops``
     - Main ATen kernel library; cloned automatically from
       ``https://github.com/intel/torch-xpu-ops`` at the SHA pinned in
       ``third_party/xpu.txt``.
   * - Triton-XPU
     - Inductor/Triton kernel codegen for XPU
       (wheels from ``https://download.pytorch.org/whl/nightly/triton-xpu/``).

Build Artifact
^^^^^^^^^^^^^^

XPU code is compiled into a separate shared library ``libtorch_xpu.so``
(target ``torch_xpu`` in ``caffe2/CMakeLists.txt``).  It links:

- ``torch_xpu_ops`` (whole-archive, from ``third_party/torch-xpu-ops``)
- ``c10_xpu`` (``c10/xpu/``)
- ATen XPU sources (``aten/src/ATen/xpu/``, ``aten/src/ATen/native/mkldnn/xpu/``,
  ``aten/src/ATen/native/transformers/xpu/``)
- ``xpu_mkldnn`` (oneDNN SYCL static library)

---

7. Testing and Maturity Signals
---------------------------------

Test Locations
^^^^^^^^^^^^^^

.. list-table::
   :widths: 45 55
   :header-rows: 1

   * - Path
     - Coverage
   * - ``test/xpu/test_gemm.py``
     - GEMM operations (``mm``, ``bmm``, ``addmm``, scaled matmul, etc.)
   * - ``test/xpu/test_conv.py``
     - Convolution forward, backward, transposed convolution; uses
       ``@onlyXPU`` for XPU-specific assertions.
   * - ``test/xpu/test_fusion.py``
     - Fused operation patterns (convolution + post-ops, etc.)
   * - ``c10/xpu/test/impl/``
     - C++ unit tests: ``XPUStreamTest.cpp``, ``XPUGuardTest.cpp``,
       ``XPUCachingAllocatorTest.cpp``, ``XPUDeviceTest.cpp``
   * - ``aten/src/ATen/test/xpu_generator_test.cpp``
     - XPU generator (Philox RNG) unit tests
   * - ``aten/src/ATen/test/xpu_event_test.cpp``
     - XPU event recording and timing
   * - ``aten/src/ATen/test/xpu_caching_host_allocator_test.cpp``
     - Pinned memory allocator
   * - ``aten/src/ATen/test/xpu_device_test.cpp``
     - Device enumeration
   * - ``aten/src/ATen/test/xpu_reportMemoryUsage_test.cpp``
     - Memory usage reporting
   * - ``test/test_ops.py``
     - Generic operator tests; many ops skip XPU via ``@skipXPU``.
   * - ``test/inductor/test_torchinductor_opinfo.py``
     - Inductor op-info tests; XPU is added via ``allow_xpu=True``.
       ``inductor_expected_failures_single_sample["xpu"]`` lists known
       gaps.
   * - ``test/test_modules.py``
     - Module-level tests; XPU enabled via ``allow_xpu=True``.

Test Infrastructure
^^^^^^^^^^^^^^^^^^^

XPU is integrated into the common device-type testing framework
(``torch/testing/_internal/common_device_type.py``):

- ``XPUTestBase`` — device test base class that enumerates available XPU
  devices.
- ``@onlyXPU`` — marks tests that run exclusively on XPU.
- ``@skipXPU`` / ``@skipXPUIf`` — marks tests that are known to not work on
  XPU.
- ``@expectedFailureXPU`` — marks tests with known failures.
- ``TEST_XPU`` — boolean sentinel (``torch.xpu.is_available()``).
- ``@skipIfNoXPU`` — skips if XPU is not compiled in.

The ``test/run_test.py`` runner exposes a ``--xpu`` flag that selects the
``test_xpu`` test set while excluding the standard GPU test list.

Maturity Signals
^^^^^^^^^^^^^^^^

- XPU was declared **Prototype** in PyTorch 2.5 for Intel Client GPUs and Intel
  Data Center GPU Max Series (per ``docs/source/notes/get_start_xpu.rst``).
- The Inductor opinfo test file maintains an explicit
  ``inductor_expected_failures_single_sample["xpu"]`` dictionary listing known
  operator-level gaps, indicating active but incomplete coverage.
- A significant number of tests in ``test/test_ops.py`` are decorated with
  ``@skipXPU``, reflecting areas where XPU support is still in progress.
- The ``architecture`` field of ``get_device_properties()`` is explicitly
  documented as ``(experimental)`` in ``torch/xpu/__init__.py``.
- Elapsed-time measurement for events (``elapsedTime``) requires SYCL compiler
  ≥ 2025.0.0, with a runtime ``TORCH_CHECK_NOT_IMPLEMENTED`` guard for older
  toolchains.
- Windows XPU support has additional restrictions: ``getGlobalIdxFromDevice``
  and ``getDeviceFromPtr`` are guarded with ``TORCH_CHECK_NOT_IMPLEMENTED`` for
  SYCL compiler versions earlier than 2025.0.0 on Windows.

Distributed / Collective Communication
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

XPU distributed training uses the XCCL backend (``USE_XCCL`` CMake flag;
Intel Collective Communications Library).  The ``torch.distributed`` layer
detects XCCL availability via ``c10d.is_xccl_available()``.  DDP supports
XPU with XCCL for BF16 gradient reduction hooks.

Inductor / Triton Codegen
^^^^^^^^^^^^^^^^^^^^^^^^^

``torch/_inductor/codegen/xpu/device_op_overrides.py`` registers
``XPUDeviceOpOverrides``, which provides device-specific string snippets for
Triton-based kernel codegen: stream acquisition, device guard types
(``AOTIXpuGuard``, ``at::xpu::XPUStreamGuard``), kernel type
(``std::unique_ptr<sycl::kernel>``), and scratch workspace handling.
The Inductor graph compiler recognizes XPU as a GPU-class device
(``torch/_inductor/graph.py`` line 2341).
