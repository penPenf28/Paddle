// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/fusion/gpu/cache_evit.h"
#include <fstream>
#include <iomanip>
#include <type_traits>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/argsort_kernel.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/top_k_kernel.h"

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/kernels/full_kernel.h"

namespace phi {
namespace fusion {

inline cudaError_t GetGridSize(int64_t n,
                               int block_size,
                               int num_waves,
                               int *num_blocks) {
  int dev;
  {
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) {
      return err;
    }
  }
  int sm_count;
  {
    cudaError_t err =
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) {
      return err;
    }
  }
  int tpm;
  {
    cudaError_t err = cudaDeviceGetAttribute(
        &tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    if (err != cudaSuccess) {
      return err;
    }
  }
  *num_blocks =
      std::max<int>(1,
                    std::min<int64_t>((n + block_size - 1) / block_size,
                                      sm_count * tpm / block_size * num_waves));
  return cudaSuccess;
}

// 每个线程处理一个元素
template <typename T, int VecSize>
__global__ void slice_min(const float *x,
                          T *masked_x,
                          int32_t num_head,
                          int32_t seq_k,
                          int32_t sink_tokens,
                          int32_t proxy_tokens,
                          const int32_t elem_cnt) {
  // int32_t tid = blockIdx.x*blockDim.x + threadIdx.x;
  int32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_tid >= elem_cnt) return;
  if (global_tid == 0) {
    printf("---the first element is:%f\n", static_cast<T>(x[global_tid]));
  }

  const int32_t fake_idx = global_tid % seq_k;
  if (fake_idx < sink_tokens || fake_idx >= seq_k - proxy_tokens) {
    masked_x[global_tid] = T(-10000.);
  } else {
    masked_x[global_tid] = static_cast<T>(x[global_tid]);
  }
}

// 此处的topk输入长度还是为seq_k，这里存在优化点，减少数据传输
// 总共有num_head*(topk+random_keeps)个需要处理的元素
template <typename T, int VecSize>
__global__ void token_keep_kernel(int64_t *topk_input,
                                  int32_t *token_keep_out,
                                  int32_t *choice_index,
                                  int32_t topk,
                                  int32_t seq_len,
                                  int32_t random_keeps,
                                  int32_t num_head) {
  // generate random numbers
  int32_t id = blockDim.x * blockIdx.x + threadIdx.x;
  int32_t row_elem_cnt = topk + random_keeps;
  if (id >= num_head * row_elem_cnt) return;
  int32_t seq_len_index = static_cast<int32_t>(id % row_elem_cnt);
  int32_t head_index = static_cast<int32_t>(id / row_elem_cnt);

  // 保留topk
  if (seq_len_index < topk) {
    token_keep_out[id] =
        static_cast<int32_t>(topk_input[head_index * seq_len + seq_len_index]);
  }
  // 随机保留
  else {  // NOLINT
    int32_t topk_input_index = static_cast<int32_t>(
        choice_index[head_index * random_keeps + seq_len_index - topk]);
    token_keep_out[id] = static_cast<int32_t>(
        topk_input[head_index * seq_len + topk_input_index]);
  }
}

template <typename T, int VecSize>
__global__ void random_evit_write_cache(
    const T *in,
    const int32_t *middle_random_perserve_kv_index,
    T *out,
    int32_t seq_len,
    int32_t num_head,
    int32_t topk,
    int32_t sink_tokens,
    int32_t proxy_tokens,
    int32_t random_keeps) {
  constexpr int HalfVecSize = VecSize / 2;
  using T_Vec = phi::AlignedVector<T, VecSize>;
  using Float_Half_Vec = phi::AlignedVector<float, HalfVecSize>;
  T_Vec out_vec;

  int32_t element_num_per_head =
      sink_tokens + topk + random_keeps + proxy_tokens;
  int32_t total_element_num = num_head * element_num_per_head;

  const int64_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t step = blockDim.x * gridDim.x * VecSize;

  for (int64_t element_idx = global_thread_id * VecSize;
       element_idx < total_element_num;
       element_idx += step) {
    int32_t head_idx = element_idx / element_num_per_head;
    int32_t token_idx = (element_idx % element_num_per_head);
    if (token_idx < sink_tokens ||
        token_idx > (element_num_per_head - proxy_tokens)) {
      int64_t load_offset = head_idx * seq_len + token_idx;
      phi::Load(in + load_offset, &out_vec);
    } else {
      for (int vi = 0; vi < VecSize; vi++) {
        int32_t chosen_idx =
            middle_random_perserve_kv_index[head_idx * (topk + random_keeps) +
                                            token_idx - sink_tokens + vi];
        out_vec[vi] = in[head_idx * seq_len + chosen_idx];
      }
    }
    int64_t store_offset = head_idx * element_num_per_head + token_idx;
    phi::Store<T, VecSize>(out_vec, out + store_offset);
  }
}

template <typename T, typename Context>
void CacheEvitKernel(const Context &dev_ctx,
                     const DenseTensor &x,
                     const DenseTensor &choice_index,
                     int32_t num_head,
                     int32_t topk,
                     int32_t sink_tokens,
                     int32_t proxy_tokens,
                     int32_t random_keeps,
                     DenseTensor *out) {
  printf("enter cache_evit kernel\n");
  // 输入的x是经过softmax reduce之后的结果
  auto x_dims = x.dims();
  int32_t seq_k = x_dims[2];
  // step1: 首先给sink_token和proxy_token打上掩码，在后续topk中不被排序
  phi::DenseTensor masked_x;
  masked_x.Resize({{1, num_head, seq_k}});
  T *masked_x_data =
      dev_ctx.template Alloc<T>(&masked_x, masked_x.numel() * sizeof(T));

  constexpr int BLOCK_SIZE = 256;
  dim3 grid_size((num_head * seq_k + BLOCK_SIZE - 1) / BLOCK_SIZE);
  slice_min<T, 1><<<grid_size, BLOCK_SIZE, 0, dev_ctx.stream()>>>(
      x.data<float>(),
      // 这里需要传入data，不是tensor
      masked_x_data,
      num_head,
      seq_k,
      sink_tokens,
      proxy_tokens,
      static_cast<int32_t>(num_head * seq_k));
  printf("finished step1\n");

  // step2: Topk排序，获得topk_idx
  // 这里topk排序保存全部的结果，并不是只保存前topk个idx
  phi::DenseTensor topk_score;
  // topk_score.Resize({{1, num_head, topk}});
  topk_score.Resize({{1, num_head, seq_k}});

  phi::DenseTensor topk_idx;
  // topk_idx.Resize({{1, num_head, topk}});
  topk_idx.Resize({{1, num_head, seq_k}});

  // 这里原实现是全排序，后面取前k个作为topk
  // topk kernel内部会给输出分配空间，这里不需要分配
  phi::TopkKernel<T, phi::GPUContext>(
      dev_ctx, masked_x, seq_k, -1, true, true, &topk_score, &topk_idx);
  printf("finished step2\n");

  // step3: 根据choice_index和topk_idx获取需要保留的token idx
  // 将需要保存的token idx保存到keep_idx中
  phi::DenseTensor keep_idx;
  keep_idx.Resize({{1, num_head, topk + random_keeps}});
  int32_t *keep_idx_data = dev_ctx.template Alloc<int32_t>(
      &keep_idx, keep_idx.numel() * sizeof(int32_t));

  int32_t keep_cnt = num_head * (topk + random_keeps);
  int blocks_token_keep = (keep_cnt + BLOCK_SIZE - 1) / BLOCK_SIZE;
  printf("before kernel3\n");
  token_keep_kernel<T, 1>
      <<<blocks_token_keep, BLOCK_SIZE, 0, dev_ctx.stream()>>>(
          const_cast<int64_t *>(topk_idx.data<int64_t>()),
          const_cast<int32_t *>(keep_idx_data),
          const_cast<int32_t *>(choice_index.data<int32_t>()),
          static_cast<int32_t>(topk),
          static_cast<int32_t>(seq_k),
          static_cast<int32_t>(random_keeps),
          static_cast<int32_t>(num_head));
  printf("finished step3\n");

  // step:4 根据token idx选择token按顺序写入输出
  // keep_idx_data保存的是中间部分topk+random_keeps的token idx
  // 除此之外还有sink和proxy
  auto out_ptr = dev_ctx.template Alloc<T>(out);
  const int VecSize = 16 / sizeof(T);

  int64_t tot_element_num =
      num_head * (sink_tokens + topk + random_keeps + proxy_tokens);
  int64_t tot_pack_num = (tot_element_num + VecSize - 1) / VecSize;
  const int block_size = 128;
  constexpr int32_t kNumWaves = 16;
  int grid_size_x = -1;

  PADDLE_ENFORCE_GPU_SUCCESS(
      GetGridSize(tot_pack_num, block_size, kNumWaves, &grid_size_x));
  dim3 grid_dim = dim3(grid_size_x);
  const cudaStream_t stream = dev_ctx.stream();

  random_evit_write_cache<T, VecSize><<<grid_dim, block_size>>>(x.data<T>(),
                                                                keep_idx_data,
                                                                out_ptr,
                                                                seq_k,
                                                                num_head,
                                                                topk,
                                                                sink_tokens,
                                                                proxy_tokens,
                                                                random_keeps);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaGetLastError());
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(cache_evit,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::CacheEvitKernel,
                   phi::dtype::float16,
                   float){};
