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

  static const bool use_fp32_bf16x9 =
    c10::utils::check_env("FP32_BF16x9") == true;


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
  if (use_fp32_bf16x9) {
    TORCH_WARN("Setting MathMode to emulated FP32 math");
    TORCH_CUSOLVER_CHECK(cusolverDnSetMathMode(
        handle, CUSOLVER_FP32_EMULATED_BF16X9_MATH));
    TORCH_CUSOLVER_CHECK(cusolverDnSetEmulationStrategy(
        handle, CUDA_EMULATION_STRATEGY_PERFORMANT));
  } else {
    TORCH_WARN("Setting MathMode to default");
    TORCH_CUSOLVER_CHECK(cusolverDnSetMathMode(
        handle, CUSOLVER_DEFAULT_MATH));
    TORCH_CUSOLVER_CHECK(cusolverDnSetEmulationStrategy(
        handle, CUDA_EMULATION_STRATEGY_DEFAULT));
  }
  return handle;
}

} // namespace at::cuda
