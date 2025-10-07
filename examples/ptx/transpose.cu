// cluster_tma_multicast.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cassert>

namespace cg = cooperative_groups;

/********* Problem Statement ********/
/********* Kernel Design ********/
/* Perform a 4B tiled matrix transpose using:
- TMA for loading tile per CTA
- TMA for storing tile transposed per CTA (use descriptor to transpose)
- Hide TMA latency using large CTAs per SM
- Only 1 thread is needed for TMA load/store, so use 1 warp per CTA
- Our primary goal is to achieve overlapped TMA Loads and Stores
*/

template <size_t TILE_XSIZE, size_t TILE_YSIZE, size_t CTAS_PER_SM>
__global__ __launch_bounds__(1024) void transpose_tma(
    float* src, 
    float* dst,
    CUtensorMap* tma_map_src,
    CUtensorMap* tma_map_dst,
    uint32_t xsize, 
    uint32_t ysize
) {
    uint32_t warp_idx = threadIdx.x / 32;
    uint32_t lane_idx = threadIdx.x % 32;
    uint32_t tile_start_x = (blockIdx.x * blockDim.x + warp_idx) * TILE_XSIZE;
    uint32_t tile_start_y = 0;
    uint32_t tile_step_x = gridDim.x * blockDim.x * TILE_XSIZE;
    uint32_t tile_step_y = TILE_YSIZE;

    __shared__ alignas(8) barrier tma_barriers [CTAS_PER_SM];
    __shared__ alignas(128) float src_smem[TILE_XSIZE * TILE_YSIZE * CTAS_PER_SM];
    barrier::arrival_token tma_token;

    if (lane_idx == 0) {
        init(&tma_barriers[warp_idx], 32);
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    for (uint32_t tile_y = 0; tile_y < ysize; tile_y += tile_step_y) {
        for (uint32_t tile_x = 0; tile_x < xsize; tile_x += tile_step_x) {
            // TMA Load
            if (lane_idx == 0) {
                uint32_t load_x = tile_start_x + tile_x;
                uint32_t load_y = tile_start_y + tile_y;
                cde::cp_async_bulk_tensor_2d_global_to_shared(
                    &smem_src[0]
                    tma_map_src,
                    load_x,
                    load_y,
                    &tma_barriers[warp_idx]
                )
                tma_token= cuda::device::barrier_arrive_tx(&tma_barriers[warp_idx], 1, sizeof(TILE_SIZE_X * TILE_YSIZE * 4));
                tma_barriers[warp_idx].wait(std::move(tma_token));
            }
            else {
                tma_token = tma_barriers[warp_idx].arrive();
                tma_barriers.wait(std::move(tma_token));
            }

            cde::fence_proxy_async_shared_cta();
            // TMA Store Transposed
            if (lane_idx == 0) {
                cde::async_bulk_tensor_2d_shared_to_global(
                    &smem_src[0],
                    tma_map_dst,
                    (tile_start_y + tile_y),
                    (tile_start_x + tile_x),
                    &tma_barriers[warp_idx]
                );

                cde::cp_async_bulk_commit_group();
                cde::cp_async_buld_wait_group_read<0>();
            }
            // No sync threads here to allow overlapping of TMA Loads and Stores
        }
    }
}

int main()
{
    uint32_t xsize = 16384;
    uint32_t ysize = 16384;
    constexpr uint32_ tile_xsize = 32;
    constexpr uint32_t tile_ysize = 32;
    thrust::host_vector<float> h_src(xsize * ysize);
    thrust::host_vector<float> h_dst(xsize * ysize, 0);

    for (uint32_t i = 0; i < xsize * ysize; i++)
        src[i] = i;
       
    thrust::device_vector<float> d_src = h_src;
    thrust::device_vector<float> d_dst = h_dst;
    CUtensorMap *h_tma_map_src, *h_tma_map_dst;
    CUtensorMap *d_tma_map_srcm, *d_tma_map_dst;
    cudaMallocHost(h_tma_map_src, sizeof(CUtensorMap));
    cudaMallocHost(h_tma_map_dst, sizeof(CUtensorMap));
    cudaMallocDevice(d_tma_map_src, sizeof(CUtensorMap));
    cudaMallocDevice(d_tma_map_dst, sizeof(CUtensorMap));

    CUresult res = cuTensorMapEncodeTiled(
        /* ptr to map */ h_tma_map_src,
        /* data type */ CU_TENSOR_MAP_DATA_TYPE_FLOAT,
        /* num dimensions */ 2,
        /* GMEM ptr */ h_src.data(),
        /* GMEM shape */ {xsize, ysize, 1, 1, 1},
        /* GMEM strides */ {sizeof(float), sizeof(float) * xsize, 1, 1, 1},
        /* SMEM Shape */ {tile_xsize, tile_ysize, 1, 1, 1},
        /* SMEM strides */ {1, 1, 1, 1, 1},
        /* Interleaving */ CU_TENSOR_MAP_INTERLEAVE_NONE,
        /* swizzle */ CU_TENSOR_MAP_SWIZZLE_128B,
        /* L2 promotion hint */ CU_TENSOR_MAP_L2_PROMOTION_NONE,
        /* Padding value */ CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    assert (res == CUDA_SUCCESS);

    res = cuTensorMapEncodeTiled(
        /* ptr to map */ h_tma_map_dst,
        /* data type */ CU_TENSOR_MAP_DATA_TYPE_FLOAT,
        /* num dimensions */ 2,
        /* GMEM ptr */ h_dst.data(),
        /* GMEM shape */ {ysize, xsize, 1, 1, 1},
        /* GMEM strides */ {sizeof(float), sizeof(float) * ysize, 1, 1, 1},
        /* SMEM Shape */ {tile_ysize, tile_xsize, 1, 1, 1},
        /* SMEM strides */ {1, 1, 1, 1, 1},
        /* Interleaving */ CU_TENSOR_MAP_INTERLEAVE_NONE,
        /* swizzle */ CU_TENSOR_MAP_SWIZZLE_128B,
        /* L2 promotion hint */ CU_TENSOR_MAP_L2_PROMOTION_NONE,
        /* Padding value */ CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    assert (res == CUDA_SUCCESS);

    res = cudaDeviceSynchronize();
    assert (res == CUDA_SUCCESS);

    res = cudaMemcpy(d_tma_map_src, h_tma_map_src, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    assert (res == CUDA_SUCCESS);
    res = cudaMemcpy(d_tma_map_dst, h_tma_map_dst, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    assert (res == CUDA_SUCCESS);

    constexpr uint32_t NUM_THREADS_PER_CTA = tile_size_x;
    constexpr uint32_t BLOCK_SIZE = 8; // hide TMA latency
    assert (xsize % (tile_size_x * BLOCK_SIZE) == 0);
    uint32_t GRID_SIZE = (xsize / (tile_size_x * BLOCK_SIZE));
    dim3 grid_dims(GRID_SIZE);
    dim3 block_dims(NUM_THREADS_PER_CTA * BLOCK_SIZE);

    transpose_tma<<<grid_dims, block_dims>>>(
        d_src.data(),
        d_dst.data(),
        d_tma_map_src,
        d_tma_map_dst,
        xsize,
        ysize
    );

    res = cudaDeviceSynchronize();
    assert (res == CUDA_SUCCESS);
}