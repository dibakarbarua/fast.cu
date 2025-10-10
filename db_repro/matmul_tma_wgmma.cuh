namespace matmul_tma_wgmma {

/* ------ Descriptors for a shared memory matrix ------ */
// Implementation is derived from PTX guide: https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-descriptor-format
/*

In terms of CuTe layouts the canonical layout can be expressed as follows:

1. MN- major
a.) No-swizzling or Interleaved
((T,1,m),(8,k)):((1,T,SBO),(1T,LBO))
Swizzle<0, 4, 3>

b.) 32B Swizzling
((T,2,m),(8,k)):((1,T,LBO),(2T,SBO))
Swizzle<1, 4, 3>

c.) 64B Swizzling
((T,4,m),(8,k)):((1,T,LBO),(4T,SBO))
Swizzle<2, 4, 3>

d.) 128B Swizzling
((T,8,m),(8,k)):((1,T,LBO),(8T,SBO))
Swizzle<3, 4, 3>

2. K- major
a.) No-swizzling or Interleaved
((8,m),(T,2k)):((1T,SBO),(1,LBO))
Swizzle<0, 4, 3>

b.) 32B Swizzling
((8,m),(T,2k)):((2T,SBO),(1,T))
Swizzle<1, 4, 3>

c.) 64B Swizzling
((8,m),(T,2k)):((4T,SBO),(1,T))
Swizzle<2, 4, 3>

d.) 128B Swizzling
((8,m),(T,2k)):((8T,SBO),(1,T))
Swizzle<3, 4, 3>

where,
T = 128 / sizeof-elements-in-bits T represents scale factor which normalizes matrix element types to 128-bits.
m represents the number of repeating patterns across rows.
k represents the number of repeating patterns across columns.

The leading/stride dimension byte offset is defined differently for transposed and non-transposed matrices. 
The leading/stride byte offset is defined as follows for matrices whose element types are normalized to 128-bits:

LBO: Leading dimension byte offset
1. K-Major
    - No-Swizzling: 
    The offset from the first column to the second columns of the 8x2 tile in the 128-bit element type normalized matrix.
    - Swizzled layouts: not used, assumed to be 1.
2. MN-Major
    - Interleave: offset from the first 8 columns to the next 8 columns.
    - Swizzled layouts: 
    offset from the first (swizzle-byte-size/16) rows to the next (swizzle-byte-size/16) rows.

SBO: Stride dimension byte offset
1. K-Major
    - Swizzled/Interleaved:
    The offset from the first 8 rows to the next 8 rows.
2. MN-Major
    - Interleave: offset from the first row to the next row.
    - Swizzled layout: offset from the first 8 columns to the next 8 columns

Matrix transpose
* When using the normal shared memory layout, there are eight
* 8-way shared memory bank conflict when storing to the transpose.
* When enabling the 128-byte swizzle pattern and using the according access pattern,
* they are eliminated both for load and store.
    for(int sidx_j =threadIdx.x; sidx_j < 8; sidx_j+= blockDim.x){
        for(int sidx_i = 0; sidx_i < 8; ++sidx_i){
            const int swiz_j_idx = (sidx_i % 8) ^ sidx_j;
            const int swiz_i_idx_tr = (sidx_j % 8) ^ sidx_i;
            smem_buffer_tr[sidx_j][swiz_i_idx_tr] = smem_buffer[sidx_i][swiz_j_idx];
        }
    }
*/
__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) { return (((x) & 0x3FFFF) >> 0x4); }

__device__ uint64_t make_smem_desc(bf16* ptr) {
    // Convert shared memory pointer to integer
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = matrix_descriptor_encode(addr);
    desc |= matrix_descriptor_encode((uint64_t)16) << 16; // LBO : Leading dimension byte offset
    desc |= matrix_descriptor_encode((uint64_t)1024) << 32; // SBO: Stride dimension byte offset
    desc |= 1llu << 62; // 128B swizzle
    return desc;
}

/* ------------ WGMMA Synchronization PTX Wrappers ----------------- */
__device__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ void warpgroup_wait() {
    static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}
/*******************************************************************/

/* ------------ WGMMA PTX Wrapper for m64n64k16f32bf16bf16 ---------- */
template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma64(float d_out[4][8], bf16* sA, bf16* sB) {
    uint64_t desc_a = make_smem_desc(&sA[0]);
    uint64_t desc_b = make_smem_desc(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"
        " %32,"
        " %33,"
        " %34, %35, %36, %37, %38;\n"
        "}\n"
        : "+f"(d_out[0][0]), "+f"(d_out[0][1]), "+f"(d_out[0][2]), "+f"(d_out[0][3]), "+f"(d_out[0][4]), "+f"(d_out[0][5]),
          "+f"(d_out[0][6]), "+f"(d_out[0][7]), "+f"(d_out[1][0]), "+f"(d_out[1][1]), "+f"(d_out[1][2]), "+f"(d_out[1][3]),
          "+f"(d_out[1][4]), "+f"(d_out[1][5]), "+f"(d_out[1][6]), "+f"(d_out[1][7]), "+f"(d_out[2][0]), "+f"(d_out[2][1]),
          "+f"(d_out[2][2]), "+f"(d_out[2][3]), "+f"(d_out[2][4]), "+f"(d_out[2][5]), "+f"(d_out[2][6]), "+f"(d_out[2][7]),
          "+f"(d_out[3][0]), "+f"(d_out[3][1]), "+f"(d_out[3][2]), "+f"(d_out[3][3]), "+f"(d_out[3][4]), "+f"(d_out[3][5]),
          "+f"(d_out[3][6]), "+f"(d_out[3][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}
/*******************************************************************/

template <int BM, int BN, int BK, int WGMMA_M, int WGMMA_N, int WGMMA_K, int NUM_THREADS>
__global__ void kernel(int M, int N, int K, float *C, CUtensorMap *tma_map_A, CUtensorMap *tma_map_B, CUtensorMap *tma_map_C) {
    /* 1. Setup SMEM: TMA alignment is 128B, WGMMA alignment is 16B */
    __shared__ alignas(128) bf16 sA[BM * BK]; // ATile -> can also be in Registers
    __shared__ alignas(128) bf16 sB[BN * BK]; // BTile 
    __shared__ alignas(128) float sC[BM * BN]; // CTile -> store to SMEM for TMA Store Reduce

    /* 2. Setup Accumulator: D */
    // Refer to PTX documentation in Sections:
    // 9.7.15.5.1. : k=16 (16-bit A/B)
    // 9.7.15.5.2. : k=8  (32-bit A/B)
    // 9.7.15.5.3. : k=32  (8-bit A/B)
    // 9.7.15.5.4. : k=256 (1-bit A/B)
    // Note: C is 
    // for each WGMMA_M x WGMMA_N tile, we have threads storing registers as
    /*
    PSA: NVIDIA PTX Documentation is hot-garbage.
    < ------- One WGMMA instructions' output --->
    < ------- K = 16 = 16x4 = 64B of C --------->
    Warp0
    <---- 32B ---------->   <---- 32B ---------->  ----- > iterations to WGMMA_N
    | T0   T1   T2   T3  | | T0   T1   T2   T3  | 
    | T4   T5   T6   T7  | | T4   T5   T6   T7  |
    | T8   T9   T10  T11 | | T8   T9   T10  T11 |  
    | T12  T13  T14  T15 | | T12  T13  T14  T15 | 
    | T16  T17  T18  T19 | | T16  T17  T18  T19 |
    | T20  T21  T22  T23 | | T20  T21  T22  T23 | 
    | T24  T25  T26  T27 | | T24  T25  T26  T27 |
    | T28  T29  T30  T31 | | T28  T29  T30  T31 | 
    Warp0
    | T0   T1   T2   T3  | | T0   T1   T2   T3  | 
    | T4   T5   T6   T7  | | T4   T5   T6   T7  |
    | T8   T9   T10  T11 | | T8   T9   T10  T11 |  
    | T12  T13  T14  T15 | | T12  T13  T14  T15 | 
    | T16  T17  T18  T19 | | T16  T17  T18  T19 |
    | T20  T21  T22  T23 | | T20  T21  T22  T23 | 
    | T24  T25  T26  T27 | | T24  T25  T26  T27 |
    | T28  T29  T30  T31 | | T28  T29  T30  T31 | 
    // InnerDim (Per WGMMA instruction) Cvalues per-thread = 4*(8B/dtype_size)
    // OuterDim (Per WGMMA) Cvalues per-thread = (WGMMA_N * dtype_size) / 64B
    Warp1
    Warp1
    Warp2
    Warp2
    Warp3
    Warp3
    */
    float d_out [WGMMA_N/16][4*8/sizeof(float)];
    static_assert(sizeof(d_out) * NUM_THREADS == BM * BN * sizeof(float), "Accumulator size does not match tile size");
    memset(d_out, 0, sizeof(d_out));

    /* 3. Setup Barriers */
    __shared__ barrier barA;
    __shared__ barrier barB;
    __shared__ barrier barC;
    barrier::arrival_token tokenA, tokenB; // phase tokens
    if (threadIdx.x == 0) {
        init(&barA, blockDim.x); // number of threads expected to arrive on tile TMA = CTA Size
        init(&barB, blockDim.x); 
        // only one thread needs to fence and all threads must have arrived
        cde::fence_proxy_async_shared_cta(); // make sure SMEM writes are visible to async proxy
    }
    __syncthreads();

    /* 4. Setup Tile Iterations */
    static_assert(M % BM == 0 && N % BN == 0 && K % BK == 0, "M, N, K must be multiples of BM, BN, BK respectively");
    static_assert(BM % WGMMA_M == 0 && BN % WGMMA_N == 0 && BK % WGMMA_K == 0, "BM, BN, BK must be multiples of WGMMA_M, WGMMA_N, WGMMA_K respectively");
    static constexpr uint32_t kNumWgmmaPerTile = BK / WGMMA_K; // 64/16 = 4
    static constexpr uint32_t kNumWgmmaMTiles = BM / WGMMA_M; // 64/64 = 1
    static constexpr uint32_t kNumWgmmaNTiles = BN / WGMMA_N; // 64/64 = 1
    uint32_t tile_start_m = blockIdx.x;
    uint32_t tile_start_n = blockIdx.y;
    uint32_t tile_step_m = grimDim.x * BM; // Entire CTA is 1 warpgroup or 1 tile
    uint32_t tile_step_n = gridDim.y * BN;
    uint32_t num_iters_K = K / BK;
    
    // Not unrolled, outer-loops
    for(uint32_t tile_idx_m = tile_start_m; tile_idx_m < M; tile_idx_m += tile_step_m) {
        for(uint32_t tile_idx_n = tile_start_n; tile_idx_n < N; tile_idx_n += tile_step_n) {
            for(uint32_t iter_idx_k = 0; iter_idx_k<num_iters_K; iter_idx_k++) {
                // Unrolled inner-loops
                #pragma unroll
                for (uint32_t subtile_idx_m = 0; subtile_idx_m < kNumWgmmaMTiles; subtile_idx_m++) {
                    #pragma unroll
                    for (uint32_t subtile_idx_n = 0; subtile_idx_n < kNumWgmmaNTiles; subtile_idx_n++) {
                        /***** Work on individiual WGMMA_M x WGMMA_N accumulator subtile *****/
                        /* 5. Load A and B Tiles using TMA Async Copy */
                        if (threadIdx.x == 0) {
                            // We are operating in K-Major, minor-dim (rank0) is K
                            cde::cp_async_bulk_tensor_2d_global_to_shared(&sA[0], tma_map_A, iter_idx_k * BK, tile_idx_m * BM, barA);
                            tokenA = cuda::device::barrier_arrive_tx(barA, 1, sizeof(sA)); // signal thread0 arrival on barA
                            cde::cp_async_bulk_tensor_2d_global_to_shared(&sB[0], tma_map_B, iter_idx_k * BK, tile_idx_n * BN, barB);
                            tokenB = cuda::device::barrier_arrive_tx(barB, 1, sizeof(sB)); // signal thread0 arrival on barB
                            cde::fence_proxy_async_shared_cta(); // ensure TMA writes are visible to all threads in CTA
                        }
                        else {
                            tokenA = barA.arrive(); // other threads arrive on barA
                            tokenB = barB.arrive(); // other threads arrive on barB
                        }

                        /* 6. Wait for data to arrive */
                        barA.wait(std::move(tokenA)); // wait for barA to complete
                        barB.wait(std::move(tokenB)); // wait for barB to complete
                        __syncthreads(); // ensure all threads have arrived before using SMEM

                        /* 7. Compute using WGMMA */
                        warpgroup_arrive(); // signal warpgroup arrival for wgmma, only once per tile load
                        #pragma unroll
                        for(int t = 0; t < kNumWgmmaPerTile; t++) {
                            /* Important Note */
                            // We iterate along SMEM multiplicands (sA and sB) in steps of WGMMA_K
                            // because each wgmma call consumes WGMMA_K elements from both sA and sB
                            // The SMEM descriptors constructed in the PTX wrapper informs the ....
                            // ... TensorCore on how to step through the SMEM pointers
                            wgmma64<1, 1, 1, 0, 0>(d_out, &sA[t * WGMMA_K], &sB[t * WGMMA_K]);
                        }
                        warpgroup_commit_batch(); // commit the batch of 4 wgmma calls
                        warpgroup_wait<0>(); // wait for all (<0> pending) wgmma in the batch to complete

                        /* 8. Store C Tile to SMEM: Unswizzled, do not have formula for (r,c) mapping */
                        // We have a WGMMA_M x WGMMA_N tile to store
                        float* block_sC = sC + subtile_idx_m * WGMMA_M * BN + subtile_idx_n * WGMMA_N; // CTile in SMEM
                        int tid = threadIdx.x;
                        int laneIdx = tid % 32;
                        int warpIdx = tid / 32;
                        int rowStart = laneIdx / 4 + warpIdx * 8;
                        int colStart = (laneIdx % 4) * 2;
                        for (int w = 0; w < WGMMA_N/16; w++) {
                            sC[rowStart * WGMMA_N + colStart] = d_out[w/16][0];
                            sC[rowStart * WGMMA_N + colStart + 1] = d_out[w/16][1];
                            sC[(rowStart + 8) * WGMMA_N + colStart] = d_out[w/16][2];
                            sC[(rowStart + 8) * WGMMA_N + colStart + 1] = d_out[w/16][3];
                            sC[rowStart * WGMMA_N + colStart + 8] = d_out[w/16][4];
                            sC[rowStart * WGMMA_N + colStart + 9] = d_out[w/16][5];
                            sC[(rowStart + 8) * WGMMA_N + colStart + 8] = d_out[w/16][6];
                            sC[(rowStart + 8) * WGMMA_N + colStart + 9] = d_out[w/16][7];
                        }

                        /* 9. TMA Store-Reduce C Tile from SMEM to GMEM */
                        cde::fence_proxy_async_shared_cta(); // Make SMEM Writes visible to TMA Engine
                        __syncthreads(); // ensure all threads have arrived before using SMEM
                        if (threadIdx.x == 0) {
                            cde::cp_async_bulk_tensor_2d_shared_to_global_reduce(&sC[0], tma_map_C, tile_idx_m * BM, tile_idx_n * BN, &sC[0]);
                            cde::cp_async_bulk_commit_group(); // commit the bulk async-group
                            cde::cp_async_bulk_wait_group_read<0>(); // wait for the group to complete
                        }
                        // Note that we do not wait for any data to arrive so no TMA Store Barrier is needed
                    __syncthreads();
                    }
                }
            }
        }
    }
}

/* ====== 2D Tensor Tiling General CUDA Nomenclature =======
Rule of thumb for these TMA tensor maps on Hopper
 ===== [ Fastest dimension (minor) first always ] ======
 - Tile (box) size per warp-group = {inner, outer} = {BlockMinorSize, BlockMajorSize} → outer × inner tile.
 - Global shape = {inner_total, outer_total} = {BlockMinorSize*blocks_width, BlockMajorSize*blocks_height}.
 - Global strides (bytes) = {elem_size, elem_size*inner_total}
   [ swap or re-compute if your data is column-major or non-contiguous. ]
*/

template <int BlockMajorSize, int BlockMinorSize, bool Swizzle=true>
void create_tensor_map(CUtensorMap *tma_map, bf16* gmem_ptr, int blocks_height, int blocks_width) {
    void* gmem_address = (void*)gmem_ptr;
    
    /* 
       CUDA TMA APIs always take a 5-D description of the tensor, 
       even if your actual tensor is 1-D, 2-D, 3-D, etc.
    */
    
    uint64_t gmem_prob_shape[5] = {
        (uint64_t)BlockMinorSize*blocks_width, 
        (uint64_t)BlockMajorSize*blocks_height, 
        1, 
        1, 
        1
    };
    uint64_t gmem_prob_stride[5] = {sizeof(bf16), sizeof(bf16) * BlockMinorSize*blocks_width, 0, 0, 0};
    uint32_t smem_box_shape[5] = {uint32_t(BlockMinorSize), uint32_t(BlockMajorSize), 1, 1, 1};
    /* Hopper TMA you should always use smem_box_stride = {1,1,1,1,1} - Dense SMEM + Swizzle */
    uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

    auto swizzle_mode = CU_TENSOR_MAP_SWIZZLE_NONE;
    if constexpr (Swizzle) {
        swizzle_mode = CU_TENSOR_MAP_SWIZZLE_128B;
    }

    CUresult result = cuTensorMapEncodeTiled(
        tma_map, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 
        /* rank */ 2, 
        gmem_address, 
        gmem_prob_shape,
        gmem_prob_stride, 
        smem_box_shape, 
        smem_box_stride, 
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle_mode, 
        CU_TENSOR_MAP_L2_PROMOTION_NONE, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    assert(result == CUDA_SUCCESS);
}

template <int BlockMajorSize, int BlockMinorSize>
__host__ static inline CUtensorMap* allocate_and_create_tensor_map(bf16* src, int blocks_height, int blocks_width) {
    CUtensorMap *tma_map_d;
    cudaMalloc(&tma_map_d, sizeof(CUtensorMap));
    CUtensorMap tma_map_host;
    create_tensor_map<BlockMajorSize, BlockMinorSize>(&tma_map_host, src, blocks_height, blocks_width);
    cudaMemcpy(tma_map_d, &tma_map_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    return tma_map_d;
}

void runTmaWgmmaBF16(int M, int N, int K, bf16 *A, bf16 *B, float *C) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int NUM_THREADS = 128;

    // for repeated kernel runs, don't create descriptor again
    CUtensorMap *d_tma_map_A = 0;
    CUtensorMap *d_tma_map_B = 0;
    int _prev_m=0, _prev_n=0, _prev_k=0;

    // Each WG-tile works on BM, BN, BK
    // Number of TileMs per WG = M/BM
    // Number of TileNs per WG = N/BN
    // Number of TileKs per WG = K/BK
    if (!d_tma_map_A) {
        d_tma_map_A = allocate_and_create_tensor_map<BM, BK>(A, M / BM, K / BK);
        d_tma_map_B = allocate_and_create_tensor_map<BN, BK>(B, N / BN, K / BK);
        d_tma_map_C = allocate_and_create_tensor_map<BM, BN, false>(C, M / BM, N / BN);
        _prev_m = M;
        _prev_n = N;
        _prev_k = K;
    }
    // Assert cached values are of same size
    assert (M == _prev_m && N == _prev_n && K == _prev_k);
    kernel<
    /*BM*/ BM,
    /*BN*/ BN,
    /*BK*/ BK,
    /*WGMMA_M*/ 64,
    /*WGMMA_N*/ 64,
    /*WGMMA_K*/ 16,
    /*NUM_THREADS*/ NUM_THREADS>
    dim3 grid_dims(M/BM, N/BN), dim3 block_dims(NUM_THREADS);
    kernel<<<grid_dims, block_dims>>>(M, N, K, C, d_tma_map_A, d_tma_map_B, d_tma_map_C);
}

} // namespace matmul_tma_wgmma

using matmul_tma_wgmma::runTmaWgmmaBF16;
