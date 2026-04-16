#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/DeviceThreadHandles.h>

namespace at::cuda {
namespace {

void createCusolverDnHandle(cusolverDnHandle_t *handle) {
  TORCH_CUSOLVER_CHECK(cusolverDnCreate(handle));
}

void destroyCusolverDnHandle(cusolverDnHandle_t handle) {
// this is because of something dumb in the ordering of
// destruction. Sometimes atexit, the cuda context (or something)
// would already be destroyed by the time this gets destroyed. It
// happens in fbcode setting. @colesbury and @soumith decided to not destroy
// the handle as a workaround.
//   - Comments of @soumith copied from cuDNN handle pool implementation
#ifdef NO_CUDNN_DESTROY_HANDLE
  (void)handle; // Suppress unused variable warning
#else
    cusolverDnDestroy(handle);
#endif
}

using CuSolverDnPoolType = DeviceThreadHandlePool<cusolverDnHandle_t, createCusolverDnHandle, destroyCusolverDnHandle>;

} // namespace

cusolverDnHandle_t getCurrentCUDASolverDnHandle() {
  c10::DeviceIndex device = 0;
  AT_CUDA_CHECK(c10::cuda::GetDevice(&device));

  // Thread local PoolWindows are lazily-initialized
  // to avoid initialization issues that caused hangs on Windows.
  // See: https://github.com/pytorch/pytorch/pull/22405
  // This thread local unique_ptrs will be destroyed when the thread terminates,
  // releasing its reserved handles back to the pool.
  static auto pool = std::make_shared<CuSolverDnPoolType>();
  thread_local std::unique_ptr<CuSolverDnPoolType::PoolWindow> myPoolWindow(
      pool->newPoolWindow());

  auto handle = myPoolWindow->reserve(device);
  auto stream = c10::cuda::getCurrentCUDAStream();
  TORCH_CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));

#if !defined(USE_ROCM) && defined(CUSOLVER_VERSION) && CUSOLVER_VERSION >= 11300
  {
    auto math_mode = at::globalContext().cusolverDnMathMode();
    cusolverMathMode_t mode = CUSOLVER_DEFAULT_MATH;
    if (math_mode == at::CuSolverDnMathMode::AllowDataTypeConversion) {
      mode = CUSOLVER_FP_EMU_ALLOW_DATA_TYPE_CONVERSION;
    }
    TORCH_CUSOLVER_CHECK(cusolverDnSetMathMode(handle, mode));
  }
#endif

#if !defined(USE_ROCM) && defined(CUSOLVER_VERSION) && CUSOLVER_VERSION >= 11601
  {
    auto strategy = at::globalContext().cusolverDnEmulationStrategy();
    cusolverFpEmulationStrategy_t emu_strategy =
        CUSOLVER_FP_EMU_STRATEGY_DEFAULT;
    if (strategy == at::CuSolverDnEmulationStrategy::DevicePrecision) {
      emu_strategy = CUSOLVER_FP_EMU_STRATEGY_DEVICE_PRECISION;
    } else if (strategy == at::CuSolverDnEmulationStrategy::Precise) {
      emu_strategy = CUSOLVER_FP_EMU_STRATEGY_PRECISE;
    }
    TORCH_CUSOLVER_CHECK(cusolverDnSetEmulationStrategy(handle, emu_strategy));
  }
#endif

  return handle;
}

} // namespace at::cuda
