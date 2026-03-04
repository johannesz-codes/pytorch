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
#if !defined(USE_ROCM) && defined(CUSOLVER_VERSION) && CUSOLVER_VERSION >= 11101
  // On CUDA >= 11.3 (cuSOLVER >= 11.1.1), cuSOLVER can use TF32 to speed up
  // FP32 computations on Ampere+ architectures, controlled by the linalg
  // float32 precision setting.
  if (!NoTF32Guard::should_disable_tf32() &&
      at::globalContext().float32Precision(
          at::Float32Backend::CUDA, at::Float32Op::LINALG) ==
          at::Float32Precision::TF32) {
    TORCH_CUSOLVER_CHECK(
        cusolverDnSetMathMode(handle, CUSOLVER_TF32_TENSOR_OP_MATH));
  } else {
    TORCH_CUSOLVER_CHECK(
        cusolverDnSetMathMode(handle, CUSOLVER_DEFAULT_MATH));
  }
#endif
  return handle;
}

} // namespace at::cuda
