#include <ATen/hip/HIPContext.h>
#include <c10/core/DeviceGuard.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <torch/all.h>

#include <limits>

namespace {

#define ROCBLAS_CHECK(EXPR)                                                \
  do {                                                                     \
    const rocblas_status status = (EXPR);                                  \
    TORCH_CHECK(status == rocblas_status_success,                          \
                "rocBLAS call failed with status ",                        \
                static_cast<int>(status));                                 \
  } while (false)

#define HIP_CHECK(EXPR)                                                    \
  do {                                                                     \
    const hipError_t status = (EXPR);                                      \
    TORCH_CHECK(status == hipSuccess, "HIP call failed: ",                 \
                hipGetErrorString(status));                                \
  } while (false)

struct RocblasHandleCache {
  rocblas_handle handle = nullptr;
  c10::DeviceIndex device = -1;

  ~RocblasHandleCache() {
    if (handle != nullptr) {
      rocblas_destroy_handle(handle);
    }
  }

  rocblas_handle get(c10::DeviceIndex current_device) {
    if (handle == nullptr || device != current_device) {
      if (handle != nullptr) {
        rocblas_destroy_handle(handle);
        handle = nullptr;
      }
      ROCBLAS_CHECK(rocblas_create_handle(&handle));
      ROCBLAS_CHECK(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
      device = current_device;
    }
    return handle;
  }
};

thread_local RocblasHandleCache handle_cache;

void check_bf16_cuda_contiguous(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA/HIP tensor");
  TORCH_CHECK(tensor.scalar_type() == at::ScalarType::BFloat16, name,
              " must be BF16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

torch::Tensor qwen35_mlp_padded_gemm(const at::Tensor& x,
                                     const at::Tensor& weight,
                                     at::Tensor& scratch,
                                     const int64_t padded_rows) {
  check_bf16_cuda_contiguous(x, "x");
  check_bf16_cuda_contiguous(weight, "weight");
  check_bf16_cuda_contiguous(scratch, "scratch");
  TORCH_CHECK(x.dim() == 2, "x must be 2D");
  TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
  TORCH_CHECK(scratch.dim() == 2, "scratch must be 2D");
  TORCH_CHECK(x.device() == weight.device(),
              "x and weight must be on the same device");
  TORCH_CHECK(x.device() == scratch.device(),
              "x and scratch must be on the same device");

  const int64_t n = x.size(0);
  const int64_t k = x.size(1);
  const int64_t m = weight.size(0);
  TORCH_CHECK(n > 0, "x must have at least one row");
  TORCH_CHECK(padded_rows >= n, "padded_rows must be >= n");
  TORCH_CHECK(weight.size(1) == k, "weight K must match x K");
  TORCH_CHECK(scratch.size(0) == padded_rows,
              "scratch rows must equal padded_rows");
  TORCH_CHECK(scratch.size(1) == k, "scratch K must match x K");
  TORCH_CHECK(m <= std::numeric_limits<int>::max(), "m is too large for rocBLAS");
  TORCH_CHECK(k <= std::numeric_limits<int>::max(), "k is too large for rocBLAS");
  TORCH_CHECK(padded_rows <= std::numeric_limits<int>::max(),
              "padded_rows is too large for rocBLAS");

  const c10::DeviceGuard device_guard(x.device());
  const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

  HIP_CHECK(hipMemcpyAsync(scratch.data_ptr(), x.data_ptr(), x.nbytes(),
                           hipMemcpyDeviceToDevice, stream));

  auto out_padded = torch::empty(
      {padded_rows, m},
      torch::TensorOptions().dtype(x.dtype()).device(x.device()));

  rocblas_handle handle = handle_cache.get(x.get_device());
  ROCBLAS_CHECK(rocblas_set_stream(handle, stream));

  const float alpha = 1.0f;
  const float beta = 0.0f;
  ROCBLAS_CHECK(rocblas_gemm_ex(
      handle, rocblas_operation_transpose, rocblas_operation_none,
      static_cast<rocblas_int>(m), static_cast<rocblas_int>(padded_rows),
      static_cast<rocblas_int>(k), &alpha, weight.data_ptr(),
      rocblas_datatype_bf16_r, static_cast<rocblas_int>(k),
      scratch.data_ptr(), rocblas_datatype_bf16_r, static_cast<rocblas_int>(k),
      &beta, out_padded.data_ptr(), rocblas_datatype_bf16_r,
      static_cast<rocblas_int>(m), out_padded.data_ptr(),
      rocblas_datatype_bf16_r, static_cast<rocblas_int>(m),
      rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));

  return out_padded.narrow(0, 0, n);
}
