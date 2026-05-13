#include <cuda_runtime.h>

#include <cstdint>
#include <numeric>
#include <string>

#include "grid.cuh"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

constexpr int kNq = 6;
constexpr int kNv = 6;
constexpr int kNx = 12;
constexpr int kNu = 6;
constexpr int kQquStride = 18;
constexpr float kDt = 0.02f;
constexpr float kGravity = 9.81f;
constexpr float kJointDamping = 0.1f;

grid::robotModel<float>* d_robot_model = nullptr;

ffi::Error CheckCuda(cudaError_t err, const char* context) {
  if (err == cudaSuccess) {
    return ffi::Error::Success();
  }
  return ffi::Error::Internal(std::string(context) + ": " +
                              cudaGetErrorString(err));
}

ffi::Error EnsureRobotModel() {
  if (d_robot_model != nullptr) {
    return ffi::Error::Success();
  }
  d_robot_model = grid::init_robotModel<float>();
  if (d_robot_model == nullptr) {
    return ffi::Error::Internal("GRiD Z1 robot model initialization failed");
  }
  return ffi::Error::Success();
}

ffi::Error CheckStateBuffer(const ffi::AnyBuffer& x) {
  if (x.element_type() != ffi::DataType::F32) {
    return ffi::Error::InvalidArgument("Z1 GRiD FFI expects float32 buffers");
  }
  auto x_dims = x.dimensions();
  if (x_dims.size() == 0) {
    return ffi::Error::InvalidArgument("Z1 GRiD FFI expects rank >= 1 buffers");
  }
  if (x_dims.back() != kNx) {
    return ffi::Error::InvalidArgument("Z1 state last dimension must be 12");
  }
  return ffi::Error::Success();
}

ffi::Error CheckStateControlBuffers(const ffi::AnyBuffer& x,
                                    const ffi::AnyBuffer& u) {
  if (auto err = CheckStateBuffer(x); err.failure()) {
    return err;
  }
  auto x_dims = x.dimensions();
  if (u.element_type() != ffi::DataType::F32) {
    return ffi::Error::InvalidArgument("Z1 GRiD FFI expects float32 buffers");
  }
  auto u_dims = u.dimensions();
  if (u_dims.size() == 0) {
    return ffi::Error::InvalidArgument("Z1 GRiD FFI expects rank >= 1 buffers");
  }
  if (u_dims.back() != kNu) {
    return ffi::Error::InvalidArgument("Z1 control last dimension must be 6");
  }
  if (x_dims.size() != u_dims.size()) {
    return ffi::Error::InvalidArgument("Z1 state/control ranks must match");
  }
  for (size_t i = 0; i + 1 < x_dims.size(); ++i) {
    if (x_dims[i] != u_dims[i]) {
      return ffi::Error::InvalidArgument(
          "Z1 state/control leading batch dimensions must match");
    }
  }
  return ffi::Error::Success();
}

int64_t LeadingBatchSize(const ffi::AnyBuffer& x) {
  auto dims = x.dimensions();
  int64_t batch = 1;
  for (size_t i = 0; i + 1 < dims.size(); ++i) {
    batch *= dims[i];
  }
  return batch;
}

__device__ __forceinline__ float ReadSymmetricUpper(const float* matrix,
                                                    int row, int col) {
  return row <= col ? matrix[col * kNv + row] : matrix[row * kNv + col];
}

__global__ void PackStateControlKernel(const float* x, const float* u,
                                       float* q_qd_u, int64_t batch) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = static_cast<int>(batch) * kQquStride;
  if (idx >= total) {
    return;
  }

  int b = idx / kQquStride;
  int j = idx % kQquStride;
  if (j < kNq) {
    q_qd_u[idx] = x[b * kNx + j];
  } else if (j < kNq + kNv) {
    q_qd_u[idx] = x[b * kNx + j];
  } else {
    int u_idx = j - kNq - kNv;
    q_qd_u[idx] = u[b * kNu + u_idx] -
                  kJointDamping * x[b * kNx + kNq + u_idx];
  }
}

__global__ void CopyQKernel(const float* x, float* q, int64_t batch) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = static_cast<int>(batch) * kNq;
  if (idx >= total) {
    return;
  }
  int b = idx / kNq;
  int j = idx % kNq;
  q[idx] = x[b * kNx + j];
}

__global__ void CopyEePositionKernel(const float* ee6, float* ee3,
                                     int64_t batch) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = static_cast<int>(batch) * 3;
  if (idx >= total) {
    return;
  }
  int b = idx / 3;
  int row = idx % 3;
  ee3[idx] = ee6[b * 6 + row];
}

__global__ void CopyEeJacobianKernel(const float* dee6x6, float* jac3x12,
                                     int64_t batch) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = static_cast<int>(batch) * 3 * kNx;
  if (idx >= total) {
    return;
  }
  int b = idx / (3 * kNx);
  int local = idx % (3 * kNx);
  int row = local / kNx;
  int col = local % kNx;
  jac3x12[idx] = col < kNq ? dee6x6[b * 36 + col * 6 + row] : 0.0f;
}

__global__ void FinishStepKernel(const float* x, const float* qdd,
                                 float* x_next, int64_t batch) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = static_cast<int>(batch) * kNx;
  if (idx >= total) {
    return;
  }

  int b = idx / kNx;
  int j = idx % kNx;
  if (j < kNq) {
    float v_next = x[b * kNx + kNq + j] + qdd[b * kNv + j] * kDt;
    x_next[idx] = x[b * kNx + j] + v_next * kDt;
  } else {
    int v_idx = j - kNq;
    x_next[idx] = x[b * kNx + j] + qdd[b * kNv + v_idx] * kDt;
  }
}

__global__ void FinishLinearizationKernel(const float* fd_du,
                                          const float* minv, float* a,
                                          float* bmat, int64_t batch) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = static_cast<int>(batch) * (kNx * kNx + kNx * kNu);
  if (idx >= total) {
    return;
  }

  int b = idx / (kNx * kNx + kNx * kNu);
  int local = idx % (kNx * kNx + kNx * kNu);
  const float* fd = fd_du + b * (2 * kNv * kNv);
  const float* minv_b = minv + b * (kNv * kNv);

  if (local < kNx * kNx) {
    int row = local / kNx;
    int col = local % kNx;
    float value = 0.0f;

    if (row < kNq) {
      if (col == row) {
        value += 1.0f;
      }
      if (col >= kNq && col - kNq == row) {
        value += kDt;
      }
    } else if (col == row) {
      value += 1.0f;
    }

    int qdd_row = row < kNq ? row : row - kNq;
    float scale = row < kNq ? kDt * kDt : kDt;
    if (col < kNq) {
      value += scale * fd[qdd_row + kNv * col];
    } else {
      int v_col = col - kNq;
      float df_dv = fd[kNv * kNv + qdd_row + kNv * v_col];
      float df_dtau = ReadSymmetricUpper(minv_b, qdd_row, v_col);
      value += scale * (df_dv - kJointDamping * df_dtau);
    }
    a[b * kNx * kNx + local] = value;
  } else {
    int b_local = local - kNx * kNx;
    int row = b_local / kNu;
    int col = b_local % kNu;
    int qdd_row = row < kNq ? row : row - kNq;
    float scale = row < kNq ? kDt * kDt : kDt;
    bmat[b * kNx * kNu + b_local] =
        scale * ReadSymmetricUpper(minv_b, qdd_row, col);
  }
}

ffi::Error RunForwardDynamics(const float* x, const float* u, float* x_next,
                              int64_t batch, cudaStream_t stream) {
  if (auto err = EnsureRobotModel(); err.failure()) {
    return err;
  }

  float* d_q_qd_u = nullptr;
  float* d_qdd = nullptr;
  if (auto err = CheckCuda(cudaMallocAsync(&d_q_qd_u,
                                           batch * kQquStride * sizeof(float),
                                           stream),
                           "cudaMallocAsync(q_qd_u)");
      err.failure()) {
    return err;
  }
  if (auto err = CheckCuda(
          cudaMallocAsync(&d_qdd, batch * kNv * sizeof(float), stream),
          "cudaMallocAsync(qdd)");
      err.failure()) {
    return err;
  }

  const int threads = 256;
  int pack_blocks =
      (static_cast<int>(batch) * kQquStride + threads - 1) / threads;
  PackStateControlKernel<<<pack_blocks, threads, 0, stream>>>(x, u, d_q_qd_u,
                                                              batch);

  int fd_blocks = static_cast<int>(batch);
  grid::forward_dynamics_kernel<float>
      <<<fd_blocks, threads, grid::FD_DYNAMIC_SHARED_MEM_COUNT * sizeof(float),
         stream>>>(d_qdd, d_q_qd_u, kQquStride, d_robot_model, kGravity, batch);

  int finish_blocks = (static_cast<int>(batch) * kNx + threads - 1) / threads;
  FinishStepKernel<<<finish_blocks, threads, 0, stream>>>(x, d_qdd, x_next,
                                                          batch);

  if (auto err = CheckCuda(cudaGetLastError(), "Z1 GRiD step launch");
      err.failure()) {
    return err;
  }
  CheckCuda(cudaFreeAsync(d_qdd, stream), "cudaFreeAsync(qdd)");
  CheckCuda(cudaFreeAsync(d_q_qd_u, stream), "cudaFreeAsync(q_qd_u)");
  return ffi::Error::Success();
}

ffi::Error StepImpl(ffi::AnyBuffer x, ffi::AnyBuffer u, ffi::AnyBuffer parameter,
                    ffi::Result<ffi::AnyBuffer> x_next,
                    cudaStream_t stream) {
  (void)parameter;
  if (auto err = CheckStateControlBuffers(x, u); err.failure()) {
    return err;
  }
  return RunForwardDynamics(x.typed_data<float>(), u.typed_data<float>(),
                            x_next->typed_data<float>(), LeadingBatchSize(x),
                            stream);
}

ffi::Error StepWithDerivativesImpl(
    ffi::AnyBuffer x, ffi::AnyBuffer u, ffi::AnyBuffer parameter,
    ffi::Result<ffi::AnyBuffer> x_next, ffi::Result<ffi::AnyBuffer> a,
    ffi::Result<ffi::AnyBuffer> bmat, cudaStream_t stream) {
  (void)parameter;
  if (auto err = CheckStateControlBuffers(x, u); err.failure()) {
    return err;
  }
  if (auto err = EnsureRobotModel(); err.failure()) {
    return err;
  }

  int64_t batch = LeadingBatchSize(x);
  const float* x_ptr = x.typed_data<float>();
  const float* u_ptr = u.typed_data<float>();

  float* d_q_qd_u = nullptr;
  float* d_qdd = nullptr;
  float* d_fd_du = nullptr;
  float* d_minv = nullptr;
  float* d_q = nullptr;

  if (auto err = CheckCuda(cudaMallocAsync(&d_q_qd_u,
                                           batch * kQquStride * sizeof(float),
                                           stream),
                           "cudaMallocAsync(q_qd_u)");
      err.failure()) {
    return err;
  }
  if (auto err = CheckCuda(
          cudaMallocAsync(&d_qdd, batch * kNv * sizeof(float), stream),
          "cudaMallocAsync(qdd)");
      err.failure()) {
    return err;
  }
  if (auto err = CheckCuda(
          cudaMallocAsync(&d_fd_du, batch * 2 * kNv * kNv * sizeof(float),
                          stream),
          "cudaMallocAsync(fd_du)");
      err.failure()) {
    return err;
  }
  if (auto err = CheckCuda(
          cudaMallocAsync(&d_minv, batch * kNv * kNv * sizeof(float), stream),
          "cudaMallocAsync(minv)");
      err.failure()) {
    return err;
  }
  if (auto err = CheckCuda(
          cudaMallocAsync(&d_q, batch * kNq * sizeof(float), stream),
          "cudaMallocAsync(q)");
      err.failure()) {
    return err;
  }

  const int threads = 256;
  int pack_blocks =
      (static_cast<int>(batch) * kQquStride + threads - 1) / threads;
  PackStateControlKernel<<<pack_blocks, threads, 0, stream>>>(x_ptr, u_ptr,
                                                              d_q_qd_u, batch);

  int copy_q_blocks =
      (static_cast<int>(batch) * kNq + threads - 1) / threads;
  CopyQKernel<<<copy_q_blocks, threads, 0, stream>>>(x_ptr, d_q, batch);

  int fd_blocks = static_cast<int>(batch);
  grid::forward_dynamics_kernel<float>
      <<<fd_blocks, threads, grid::FD_DYNAMIC_SHARED_MEM_COUNT * sizeof(float),
         stream>>>(d_qdd, d_q_qd_u, kQquStride, d_robot_model, kGravity, batch);

  grid::forward_dynamics_gradient_kernel<float>
      <<<fd_blocks, threads,
         grid::FD_DU_DYNAMIC_SHARED_MEM_COUNT * sizeof(float), stream>>>(
          d_fd_du, d_q_qd_u, kQquStride, d_robot_model, kGravity, batch);

  grid::direct_minv_kernel<float>
      <<<fd_blocks, threads, grid::MINV_DYNAMIC_SHARED_MEM_COUNT * sizeof(float),
         stream>>>(d_minv, d_q, kNq, d_robot_model, batch);

  int finish_blocks = (static_cast<int>(batch) * kNx + threads - 1) / threads;
  FinishStepKernel<<<finish_blocks, threads, 0, stream>>>(
      x_ptr, d_qdd, x_next->typed_data<float>(), batch);

  int lin_total = static_cast<int>(batch) * (kNx * kNx + kNx * kNu);
  int lin_blocks = (lin_total + threads - 1) / threads;
  FinishLinearizationKernel<<<lin_blocks, threads, 0, stream>>>(
      d_fd_du, d_minv, a->typed_data<float>(), bmat->typed_data<float>(),
      batch);

  if (auto err = CheckCuda(cudaGetLastError(),
                           "Z1 GRiD step_with_derivatives launch");
      err.failure()) {
    return err;
  }

  CheckCuda(cudaFreeAsync(d_q, stream), "cudaFreeAsync(q)");
  CheckCuda(cudaFreeAsync(d_minv, stream), "cudaFreeAsync(minv)");
  CheckCuda(cudaFreeAsync(d_fd_du, stream), "cudaFreeAsync(fd_du)");
  CheckCuda(cudaFreeAsync(d_qdd, stream), "cudaFreeAsync(qdd)");
  CheckCuda(cudaFreeAsync(d_q_qd_u, stream), "cudaFreeAsync(q_qd_u)");
  return ffi::Error::Success();
}

ffi::Error EePositionImpl(ffi::AnyBuffer x, ffi::Result<ffi::AnyBuffer> ee,
                          cudaStream_t stream) {
  if (auto err = CheckStateBuffer(x); err.failure()) {
    return err;
  }
  if (auto err = EnsureRobotModel(); err.failure()) {
    return err;
  }

  int64_t batch = LeadingBatchSize(x);
  const float* x_ptr = x.typed_data<float>();
  float* d_q = nullptr;
  float* d_ee6 = nullptr;
  if (auto err = CheckCuda(
          cudaMallocAsync(&d_q, batch * kNq * sizeof(float), stream),
          "cudaMallocAsync(q)");
      err.failure()) {
    return err;
  }
  if (auto err = CheckCuda(
          cudaMallocAsync(&d_ee6, batch * 6 * sizeof(float), stream),
          "cudaMallocAsync(ee6)");
      err.failure()) {
    return err;
  }

  const int threads = 256;
  int q_blocks = (static_cast<int>(batch) * kNq + threads - 1) / threads;
  CopyQKernel<<<q_blocks, threads, 0, stream>>>(x_ptr, d_q, batch);

  int ee_blocks = static_cast<int>(batch);
  grid::end_effector_pose_kernel_end_effector<float>
      <<<ee_blocks, threads,
         grid::EE_POS_DYNAMIC_SHARED_MEM_COUNT * sizeof(float), stream>>>(
          d_ee6, d_q, kNq, d_robot_model, batch);

  int copy_blocks = (static_cast<int>(batch) * 3 + threads - 1) / threads;
  CopyEePositionKernel<<<copy_blocks, threads, 0, stream>>>(
      d_ee6, ee->typed_data<float>(), batch);

  if (auto err = CheckCuda(cudaGetLastError(), "Z1 GRiD ee_position launch");
      err.failure()) {
    return err;
  }

  CheckCuda(cudaFreeAsync(d_ee6, stream), "cudaFreeAsync(ee6)");
  CheckCuda(cudaFreeAsync(d_q, stream), "cudaFreeAsync(q)");
  return ffi::Error::Success();
}

ffi::Error EePositionJacobianImpl(ffi::AnyBuffer x,
                                  ffi::Result<ffi::AnyBuffer> ee,
                                  ffi::Result<ffi::AnyBuffer> jac,
                                  cudaStream_t stream) {
  if (auto err = CheckStateBuffer(x); err.failure()) {
    return err;
  }
  if (auto err = EnsureRobotModel(); err.failure()) {
    return err;
  }

  int64_t batch = LeadingBatchSize(x);
  const float* x_ptr = x.typed_data<float>();
  float* d_q = nullptr;
  float* d_ee6 = nullptr;
  float* d_dee6x6 = nullptr;
  if (auto err = CheckCuda(
          cudaMallocAsync(&d_q, batch * kNq * sizeof(float), stream),
          "cudaMallocAsync(q)");
      err.failure()) {
    return err;
  }
  if (auto err = CheckCuda(
          cudaMallocAsync(&d_ee6, batch * 6 * sizeof(float), stream),
          "cudaMallocAsync(ee6)");
      err.failure()) {
    return err;
  }
  if (auto err = CheckCuda(
          cudaMallocAsync(&d_dee6x6, batch * 36 * sizeof(float), stream),
          "cudaMallocAsync(dee6x6)");
      err.failure()) {
    return err;
  }

  const int threads = 256;
  int q_blocks = (static_cast<int>(batch) * kNq + threads - 1) / threads;
  CopyQKernel<<<q_blocks, threads, 0, stream>>>(x_ptr, d_q, batch);

  int ee_blocks = static_cast<int>(batch);
  grid::end_effector_pose_kernel_end_effector<float>
      <<<ee_blocks, threads,
         grid::EE_POS_DYNAMIC_SHARED_MEM_COUNT * sizeof(float), stream>>>(
          d_ee6, d_q, kNq, d_robot_model, batch);
  grid::end_effector_pose_gradient_kernel_end_effector<float>
      <<<ee_blocks, threads,
         grid::DEE_POS_DYNAMIC_SHARED_MEM_COUNT * sizeof(float), stream>>>(
          d_dee6x6, d_q, kNq, d_robot_model, batch);

  int copy_ee_blocks = (static_cast<int>(batch) * 3 + threads - 1) / threads;
  CopyEePositionKernel<<<copy_ee_blocks, threads, 0, stream>>>(
      d_ee6, ee->typed_data<float>(), batch);

  int copy_j_blocks =
      (static_cast<int>(batch) * 3 * kNx + threads - 1) / threads;
  CopyEeJacobianKernel<<<copy_j_blocks, threads, 0, stream>>>(
      d_dee6x6, jac->typed_data<float>(), batch);

  if (auto err = CheckCuda(cudaGetLastError(),
                           "Z1 GRiD ee_position_jacobian launch");
      err.failure()) {
    return err;
  }

  CheckCuda(cudaFreeAsync(d_dee6x6, stream), "cudaFreeAsync(dee6x6)");
  CheckCuda(cudaFreeAsync(d_ee6, stream), "cudaFreeAsync(ee6)");
  CheckCuda(cudaFreeAsync(d_q, stream), "cudaFreeAsync(q)");
  return ffi::Error::Success();
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpx_grid_step, StepImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpx_grid_step_with_derivatives, StepWithDerivativesImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpx_grid_ee_position, EePositionImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpx_grid_ee_position_jacobian, EePositionJacobianImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>());
