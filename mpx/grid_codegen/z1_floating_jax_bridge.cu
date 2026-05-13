#include <cuda_runtime.h>

#include <cstdint>
#include <string>

#include "grid.cuh"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

constexpr int kNq = 13;
constexpr int kNv = 12;
constexpr int kNx = 19;
constexpr int kNu = 12;
constexpr int kQquStride = 37;
constexpr float kDt = 0.02f;
constexpr float kGravity = 9.81f;
constexpr float kArmature = 0.1f;

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
    return ffi::Error::Internal("GRiD floating Z1 robot model init failed");
  }
  return ffi::Error::Success();
}

ffi::Error CheckStateControlBuffers(const ffi::AnyBuffer& x,
                                    const ffi::AnyBuffer& u) {
  if (x.element_type() != ffi::DataType::F32 ||
      u.element_type() != ffi::DataType::F32) {
    return ffi::Error::InvalidArgument(
        "floating Z1 GRiD FFI expects float32 buffers");
  }
  auto x_dims = x.dimensions();
  auto u_dims = u.dimensions();
  if (x_dims.size() == 0 || u_dims.size() == 0) {
    return ffi::Error::InvalidArgument(
        "floating Z1 GRiD FFI expects rank >= 1 buffers");
  }
  if (x_dims.back() != kNx) {
    return ffi::Error::InvalidArgument(
        "floating Z1 state last dimension must be 19");
  }
  if (u_dims.back() != kNu) {
    return ffi::Error::InvalidArgument(
        "floating Z1 control last dimension must be 12");
  }
  if (x_dims.size() != u_dims.size()) {
    return ffi::Error::InvalidArgument(
        "floating Z1 state/control ranks must match");
  }
  for (size_t i = 0; i + 1 < x_dims.size(); ++i) {
    if (x_dims[i] != u_dims[i]) {
      return ffi::Error::InvalidArgument(
          "floating Z1 leading batch dimensions must match");
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
  const float* xb = x + b * kNx;
  const float* ub = u + b * kNu;

  if (j < kNq) {
    q_qd_u[idx] = xb[j];
  } else if (j < kNq + kNv) {
    int qd = j - kNq;
    q_qd_u[idx] = qd < 6 ? 0.0f : xb[13 + qd - 6];
  } else {
    int tau = j - kNq - kNv;
    q_qd_u[idx] = tau < 6 ? 0.0f : ub[tau - 6];
  }
}

__global__ void PackStateVelocityKernel(const float* x, float* q_qd,
                                        int64_t batch) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = static_cast<int>(batch) * (kNq + kNv);
  if (idx >= total) {
    return;
  }
  int b = idx / (kNq + kNv);
  int j = idx % (kNq + kNv);
  const float* xb = x + b * kNx;
  if (j < kNq) {
    q_qd[idx] = xb[j];
  } else {
    int qd = j - kNq;
    q_qd[idx] = qd < 6 ? 0.0f : xb[13 + qd - 6];
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

__device__ __forceinline__ void IntegrateQuatEuler(const float* q,
                                                   const float* omega,
                                                   float* out) {
  float qw = q[0], qx = q[1], qy = q[2], qz = q[3];
  float wx = omega[0], wy = omega[1], wz = omega[2];
  float half_dt = 0.5f * kDt;
  float raw[4];
  raw[0] = qw + half_dt * (-qx * wx - qy * wy - qz * wz);
  raw[1] = qx + half_dt * (qw * wx + qy * wz - qz * wy);
  raw[2] = qy + half_dt * (qw * wy + qz * wx - qx * wz);
  raw[3] = qz + half_dt * (qw * wz + qx * wy - qy * wx);
  float n = sqrtf(raw[0] * raw[0] + raw[1] * raw[1] +
                  raw[2] * raw[2] + raw[3] * raw[3]);
  n = fmaxf(n, 1e-12f);
  out[0] = raw[0] / n;
  out[1] = raw[1] / n;
  out[2] = raw[2] / n;
  out[3] = raw[3] / n;
}

__global__ void FinishStepKernel(const float* x, const float* u,
                                 const float* qdd, float* x_next,
                                 int64_t batch) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = static_cast<int>(batch) * kNx;
  if (idx >= total) {
    return;
  }

  int b = idx / kNx;
  int j = idx % kNx;
  const float* xb = x + b * kNx;
  const float* ub = u + b * kNu;
  const float* qddb = qdd + b * kNv;
  float* out = x_next + b * kNx;

  if (j < 3) {
    out[j] = xb[j] + ub[6 + j] * kDt;
  } else if (j < 7) {
    float qout[4];
    IntegrateQuatEuler(xb + 3, ub + 9, qout);
    out[j] = qout[j - 3];
  } else if (j < 13) {
    int a = j - 7;
    float dq_next = xb[13 + a] + qddb[6 + a] * kDt;
    out[j] = xb[j] + dq_next * kDt;
  } else {
    int a = j - 13;
    out[j] = xb[j] + qddb[6 + a] * kDt;
  }
}

__device__ bool Invert6x6(const float* src, float* inv) {
  float aug[6][12];
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) {
      aug[r][c] = src[r * 6 + c];
      aug[r][6 + c] = r == c ? 1.0f : 0.0f;
    }
  }
  for (int p = 0; p < 6; ++p) {
    int pivot = p;
    float best = fabsf(aug[p][p]);
    for (int r = p + 1; r < 6; ++r) {
      float v = fabsf(aug[r][p]);
      if (v > best) {
        best = v;
        pivot = r;
      }
    }
    if (best < 1e-8f) {
      return false;
    }
    if (pivot != p) {
      for (int c = 0; c < 12; ++c) {
        float tmp = aug[p][c];
        aug[p][c] = aug[pivot][c];
        aug[pivot][c] = tmp;
      }
    }
    float diag = aug[p][p];
    for (int c = 0; c < 12; ++c) {
      aug[p][c] /= diag;
    }
    for (int r = 0; r < 6; ++r) {
      if (r == p) {
        continue;
      }
      float factor = aug[r][p];
      for (int c = 0; c < 12; ++c) {
        aug[r][c] -= factor * aug[p][c];
      }
    }
  }
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) {
      inv[r * 6 + c] = aug[r][6 + c];
    }
  }
  return true;
}

__global__ void ReducedArmQddKernel(const float* u, const float* bias,
                                    const float* minv, float* qdd,
                                    int64_t batch) {
  int b = blockIdx.x;
  if (b >= batch || threadIdx.x != 0) {
    return;
  }
  const float* ub = u + b * kNu;
  const float* cb = bias + b * kNq;
  const float* hb = minv + b * kNv * kNv;
  float hbb[36], hba[36], hab[36], haa[36], hbb_inv[36], tmp[36], s[36];
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) {
      hbb[r * 6 + c] = ReadSymmetricUpper(hb, r, c);
      hba[r * 6 + c] = ReadSymmetricUpper(hb, r, 6 + c);
      hab[r * 6 + c] = ReadSymmetricUpper(hb, 6 + r, c);
      haa[r * 6 + c] = ReadSymmetricUpper(hb, 6 + r, 6 + c);
    }
  }
  if (!Invert6x6(hbb, hbb_inv)) {
    for (int i = 0; i < kNv; ++i) {
      qdd[b * kNv + i] = 0.0f;
    }
    return;
  }
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) {
      float v = 0.0f;
      for (int k = 0; k < 6; ++k) {
        v += hab[r * 6 + k] * hbb_inv[k * 6 + c];
      }
      tmp[r * 6 + c] = v;
    }
  }
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) {
      float v = haa[r * 6 + c];
      for (int k = 0; k < 6; ++k) {
        v -= tmp[r * 6 + k] * hba[k * 6 + c];
      }
      s[r * 6 + c] = v;
    }
  }

  float g[36], g_inv[36], rhs[6], qdd_arm[6];
  for (int r = 0; r < 6; ++r) {
    rhs[r] = ub[r] - cb[6 + r];
    for (int c = 0; c < 6; ++c) {
      g[r * 6 + c] = (r == c ? 1.0f : 0.0f) + kArmature * s[r * 6 + c];
    }
  }
  if (!Invert6x6(g, g_inv)) {
    for (int i = 0; i < kNv; ++i) {
      qdd[b * kNv + i] = 0.0f;
    }
    return;
  }
  for (int r = 0; r < 6; ++r) {
    float v = 0.0f;
    for (int c = 0; c < 6; ++c) {
      float k_rc = 0.0f;
      for (int m = 0; m < 6; ++m) {
        k_rc += g_inv[r * 6 + m] * s[m * 6 + c];
      }
      v += k_rc * rhs[c];
    }
    qdd_arm[r] = v;
  }
  for (int i = 0; i < 6; ++i) {
    qdd[b * kNv + i] = 0.0f;
    qdd[b * kNv + 6 + i] = qdd_arm[i];
  }
}

__device__ bool ComputeReducedGain(const float* minv_b, float* k_gain) {
  float hbb[36], hba[36], hab[36], haa[36], hbb_inv[36], tmp[36], s[36];
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) {
      hbb[r * 6 + c] = ReadSymmetricUpper(minv_b, r, c);
      hba[r * 6 + c] = ReadSymmetricUpper(minv_b, r, 6 + c);
      hab[r * 6 + c] = ReadSymmetricUpper(minv_b, 6 + r, c);
      haa[r * 6 + c] = ReadSymmetricUpper(minv_b, 6 + r, 6 + c);
    }
  }
  if (!Invert6x6(hbb, hbb_inv)) {
    return false;
  }
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) {
      float v = 0.0f;
      for (int k = 0; k < 6; ++k) {
        v += hab[r * 6 + k] * hbb_inv[k * 6 + c];
      }
      tmp[r * 6 + c] = v;
    }
  }
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) {
      float v = haa[r * 6 + c];
      for (int k = 0; k < 6; ++k) {
        v -= tmp[r * 6 + k] * hba[k * 6 + c];
      }
      s[r * 6 + c] = v;
    }
  }
  float g[36], g_inv[36];
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) {
      g[r * 6 + c] = (r == c ? 1.0f : 0.0f) + kArmature * s[r * 6 + c];
    }
  }
  if (!Invert6x6(g, g_inv)) {
    return false;
  }
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) {
      float v = 0.0f;
      for (int m = 0; m < 6; ++m) {
        v += g_inv[r * 6 + m] * s[m * 6 + c];
      }
      k_gain[r * 6 + c] = v;
    }
  }
  return true;
}

__global__ void FinishLinearizationKernel(const float* x, const float* u,
                                          const float* id_du,
                                          const float* minv, float* a,
                                          float* bmat, int64_t batch) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = static_cast<int>(batch) * (kNx * kNx + kNx * kNu);
  if (idx >= total) {
    return;
  }

  int b = idx / (kNx * kNx + kNx * kNu);
  int local = idx % (kNx * kNx + kNx * kNu);
  const float* xb = x + b * kNx;
  const float* ub = u + b * kNu;
  const float* id = id_du + b * (2 * kNv * kNv);
  const float* minv_b = minv + b * (kNv * kNv);
  float k_gain[36];
  bool has_gain = ComputeReducedGain(minv_b, k_gain);

  if (local < kNx * kNx) {
    int row = local / kNx;
    int col = local % kNx;
    float value = row == col ? 1.0f : 0.0f;

    if (row >= 3 && row < 7 && col >= 3 && col < 7) {
      float wx = ub[9], wy = ub[10], wz = ub[11];
      float h = 0.5f * kDt;
      int r = row - 3;
      int c = col - 3;
      float raw[4] = {
          xb[3] + h * (-xb[4] * wx - xb[5] * wy - xb[6] * wz),
          xb[4] + h * (xb[3] * wx + xb[5] * wz - xb[6] * wy),
          xb[5] + h * (xb[3] * wy + xb[6] * wx - xb[4] * wz),
          xb[6] + h * (xb[3] * wz + xb[4] * wy - xb[5] * wx)};
      float n2 = raw[0] * raw[0] + raw[1] * raw[1] +
                 raw[2] * raw[2] + raw[3] * raw[3];
      float n = sqrtf(fmaxf(n2, 1e-12f));
      float draw_dq[16] = {
          1.0f, -h * wx, -h * wy, -h * wz,
          h * wx, 1.0f, h * wz, -h * wy,
          h * wy, -h * wz, 1.0f, h * wx,
          h * wz, h * wy, -h * wx, 1.0f};
      value = 0.0f;
      for (int k = 0; k < 4; ++k) {
        float p = (r == k ? 1.0f / n : 0.0f) -
                  raw[r] * raw[k] / (n * n * n);
        value += p * draw_dq[k * 4 + c];
      }
    }

    if (row >= 7) {
      int arm_row = row < 13 ? row - 7 : row - 13;
      float scale = row < 13 ? kDt * kDt : kDt;
      if (col >= 7 && col < 13) {
        int q_col = 6 + col - 7;
        float dqdd = 0.0f;
        if (has_gain) {
          for (int k = 0; k < 6; ++k) {
            dqdd -= k_gain[arm_row * 6 + k] * id[(6 + k) + kNv * q_col];
          }
        }
        value += scale * dqdd;
      } else if (col >= 4 && col < 7) {
        int q_col = col - 4;
        float dqdd = 0.0f;
        if (has_gain) {
          for (int k = 0; k < 6; ++k) {
            dqdd -= 2.0f * k_gain[arm_row * 6 + k] *
                    id[(6 + k) + kNv * q_col];
          }
        }
        value += scale * dqdd;
      } else if (col >= 13) {
        int qd_col = 6 + col - 13;
        float dqdd = 0.0f;
        if (has_gain) {
          for (int k = 0; k < 6; ++k) {
            dqdd -= k_gain[arm_row * 6 + k] *
                    id[kNv * kNv + (6 + k) + kNv * qd_col];
          }
        }
        if (row < 13 && arm_row == col - 13) {
          value += kDt;
        }
        value += scale * dqdd;
      }
    }
    a[b * kNx * kNx + local] = value;
  } else {
    int b_local = local - kNx * kNx;
    int row = b_local / kNu;
    int col = b_local % kNu;
    float value = 0.0f;
    if (row < 3 && col >= 6 && col < 9) {
      value = (row == col - 6) ? kDt : 0.0f;
    } else if (row >= 3 && row < 7 && col >= 9) {
      float qw = xb[3], qx = xb[4], qy = xb[5], qz = xb[6];
      float wx = ub[9], wy = ub[10], wz = ub[11];
      float h = 0.5f * kDt;
      int r = row - 3;
      int c = col - 9;
      float raw[4] = {
          qw + h * (-qx * wx - qy * wy - qz * wz),
          qx + h * (qw * wx + qy * wz - qz * wy),
          qy + h * (qw * wy + qz * wx - qx * wz),
          qz + h * (qw * wz + qx * wy - qy * wx)};
      float n2 = raw[0] * raw[0] + raw[1] * raw[1] +
                 raw[2] * raw[2] + raw[3] * raw[3];
      float n = sqrtf(fmaxf(n2, 1e-12f));
      float dq_dw[12] = {
          -h * qx, -h * qy, -h * qz,
          h * qw, -h * qz, h * qy,
          h * qz, h * qw, -h * qx,
          -h * qy, h * qx, h * qw};
      value = 0.0f;
      for (int k = 0; k < 4; ++k) {
        float p = (r == k ? 1.0f / n : 0.0f) -
                  raw[r] * raw[k] / (n * n * n);
        value += p * dq_dw[k * 3 + c];
      }
    } else if (row >= 7) {
      int arm_row = row < 13 ? row - 7 : row - 13;
      float scale = row < 13 ? kDt * kDt : kDt;
      if (col < 6) {
        value = has_gain ? scale * k_gain[arm_row * 6 + col] : 0.0f;
      }
    }
    bmat[b * kNx * kNu + b_local] = value;
  }
}

ffi::Error RunForwardDynamics(const float* x, const float* u, float* x_next,
                              int64_t batch, cudaStream_t stream) {
  if (auto err = EnsureRobotModel(); err.failure()) {
    return err;
  }

  float* d_q_qd_u = nullptr;
  float* d_q_qd = nullptr;
  float* d_qdd = nullptr;
  float* d_bias = nullptr;
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
          cudaMallocAsync(&d_q_qd, batch * (kNq + kNv) * sizeof(float), stream),
          "cudaMallocAsync(q_qd)");
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
          cudaMallocAsync(&d_bias, batch * kNq * sizeof(float), stream),
          "cudaMallocAsync(bias)");
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
  PackStateControlKernel<<<pack_blocks, threads, 0, stream>>>(x, u, d_q_qd_u,
                                                              batch);
  int pack_qd_blocks =
      (static_cast<int>(batch) * (kNq + kNv) + threads - 1) / threads;
  PackStateVelocityKernel<<<pack_qd_blocks, threads, 0, stream>>>(x, d_q_qd,
                                                                  batch);
  int copy_q_blocks =
      (static_cast<int>(batch) * kNq + threads - 1) / threads;
  CopyQKernel<<<copy_q_blocks, threads, 0, stream>>>(x, d_q, batch);

  int fd_blocks = static_cast<int>(batch);
  grid::inverse_dynamics_kernel<float>
      <<<fd_blocks, threads, grid::ID_DYNAMIC_SHARED_MEM_COUNT * sizeof(float),
         stream>>>(d_bias, d_q_qd, kNq + kNv, d_robot_model, kGravity, batch);
  grid::direct_minv_kernel<float>
      <<<fd_blocks, threads, grid::MINV_DYNAMIC_SHARED_MEM_COUNT * sizeof(float),
         stream>>>(d_minv, d_q, kNq, d_robot_model, batch);
  ReducedArmQddKernel<<<fd_blocks, 1, 0, stream>>>(u, d_bias, d_minv, d_qdd,
                                                   batch);

  int finish_blocks = (static_cast<int>(batch) * kNx + threads - 1) / threads;
  FinishStepKernel<<<finish_blocks, threads, 0, stream>>>(x, u, d_qdd, x_next,
                                                          batch);

  if (auto err = CheckCuda(cudaGetLastError(),
                           "floating Z1 GRiD step launch");
      err.failure()) {
    return err;
  }
  CheckCuda(cudaFreeAsync(d_qdd, stream), "cudaFreeAsync(qdd)");
  CheckCuda(cudaFreeAsync(d_q, stream), "cudaFreeAsync(q)");
  CheckCuda(cudaFreeAsync(d_minv, stream), "cudaFreeAsync(minv)");
  CheckCuda(cudaFreeAsync(d_bias, stream), "cudaFreeAsync(bias)");
  CheckCuda(cudaFreeAsync(d_q_qd, stream), "cudaFreeAsync(q_qd)");
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
  float* d_q_qd = nullptr;
  float* d_qdd = nullptr;
  float* d_bias = nullptr;
  float* d_id_du = nullptr;
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
          cudaMallocAsync(&d_q_qd, batch * (kNq + kNv) * sizeof(float), stream),
          "cudaMallocAsync(q_qd)");
      err.failure()) {
    return err;
  }
  if (auto err = CheckCuda(
          cudaMallocAsync(&d_bias, batch * kNq * sizeof(float), stream),
          "cudaMallocAsync(bias)");
      err.failure()) {
    return err;
  }
  if (auto err = CheckCuda(
          cudaMallocAsync(&d_id_du, batch * 2 * kNv * kNv * sizeof(float),
                          stream),
          "cudaMallocAsync(id_du)");
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
  int pack_qd_blocks =
      (static_cast<int>(batch) * (kNq + kNv) + threads - 1) / threads;
  PackStateVelocityKernel<<<pack_qd_blocks, threads, 0, stream>>>(x_ptr, d_q_qd,
                                                                  batch);

  int copy_q_blocks =
      (static_cast<int>(batch) * kNq + threads - 1) / threads;
  CopyQKernel<<<copy_q_blocks, threads, 0, stream>>>(x_ptr, d_q, batch);

  int fd_blocks = static_cast<int>(batch);
  grid::inverse_dynamics_kernel<float>
      <<<fd_blocks, threads, grid::ID_DYNAMIC_SHARED_MEM_COUNT * sizeof(float),
         stream>>>(d_bias, d_q_qd, kNq + kNv, d_robot_model, kGravity, batch);

  grid::direct_minv_kernel<float>
      <<<fd_blocks, threads, grid::MINV_DYNAMIC_SHARED_MEM_COUNT * sizeof(float),
         stream>>>(d_minv, d_q, kNq, d_robot_model, batch);
  ReducedArmQddKernel<<<fd_blocks, 1, 0, stream>>>(u_ptr, d_bias, d_minv, d_qdd,
                                                   batch);
  grid::inverse_dynamics_gradient_kernel<float>
      <<<fd_blocks, threads,
         grid::ID_DU_DYNAMIC_SHARED_MEM_COUNT * sizeof(float), stream>>>(
          d_id_du, d_q_qd, kNq + kNv, d_qdd, d_robot_model, kGravity, batch);

  int finish_blocks = (static_cast<int>(batch) * kNx + threads - 1) / threads;
  FinishStepKernel<<<finish_blocks, threads, 0, stream>>>(
      x_ptr, u_ptr, d_qdd, x_next->typed_data<float>(), batch);

  int lin_total = static_cast<int>(batch) * (kNx * kNx + kNx * kNu);
  int lin_blocks = (lin_total + threads - 1) / threads;
  FinishLinearizationKernel<<<lin_blocks, threads, 0, stream>>>(
      x_ptr, u_ptr, d_id_du, d_minv, a->typed_data<float>(),
      bmat->typed_data<float>(), batch);

  if (auto err = CheckCuda(cudaGetLastError(),
                           "floating Z1 GRiD step_with_derivatives launch");
      err.failure()) {
    return err;
  }

  CheckCuda(cudaFreeAsync(d_q, stream), "cudaFreeAsync(q)");
  CheckCuda(cudaFreeAsync(d_minv, stream), "cudaFreeAsync(minv)");
  CheckCuda(cudaFreeAsync(d_id_du, stream), "cudaFreeAsync(id_du)");
  CheckCuda(cudaFreeAsync(d_bias, stream), "cudaFreeAsync(bias)");
  CheckCuda(cudaFreeAsync(d_qdd, stream), "cudaFreeAsync(qdd)");
  CheckCuda(cudaFreeAsync(d_q_qd, stream), "cudaFreeAsync(q_qd)");
  CheckCuda(cudaFreeAsync(d_q_qd_u, stream), "cudaFreeAsync(q_qd_u)");
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
