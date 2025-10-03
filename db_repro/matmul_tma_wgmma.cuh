namespace matmul_tma_wgmma {

// Descriptor for a shared memory matrix.
// Implementation is derived from PTX guide: https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-descriptor-format
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
__global__ void kernel(int M, int N, int K, bf16 *C, CUtensorMap *tma_map_A, CUtensorMap *tma_map_B) {
    /* 1. Setup SMEM: TMA Engine expects 128B aligned addrs */
    __shared__ alignas(128) bf16 sA[BM * BK]; // ATile -> can also be in Registers
    __shared__ alignas(128) bf16 sB[BN * BK]; // BTile 

    /* 2. Setup Accumulator: D */
    float d_out [WGMMA_N/16][8];
    static_assert(sizeof(d_out) * NUM_THREADS == BM * BN * sizeof(float), "Accumulator size does not match tile size");
    memset(d_out, 0, sizeof(d_out));

    /* 3. Setup Barriers */
    __shared__ barrier barA;
    __shared__ barrier barB;
    barrier::arrival_token tokenA, tokenB; // phase tokens
    if (threadIdx.x == 0) {
        init(&barA, blockDim.x); // number of threads expected to arrive = CTA Size
        init(&barB, blockDim.x);
        // only one thread needs to fence and all threads must have arrived
        cde::fence_proxy_async_shared_cta(); // make sure SMEM writes are visible to async proxy
    }
    __syncthreads();

    /* 4. Setup Tile Iterations */
    static_assert(M % BM == 0 && N % BN == 0 && K % BK == 0, "M, N, K must be multiples of BM, BN, BK respectively");
    static constexpr uint32_t kNumWgmmaPerTile = K / BK;
    uint32_t tile_idx_m = blockIdx.x;
    uint32_t tile_idx_n = blockIdx.y;

    // Not unrolled, inner-loop
    for( uint32_t iter = 0; iter < num_iters_per_wg; ++iter ) {
        /* 5. Load A and B Tiles using TMA Async Copy */
        if (threadIdx.x == 0) {
            // We are operating in K-Major, minor-dim (rank0) is K
            cde::cp_async_bulk_tensor_2d_global_to_shared(&sA[0], tma_map_A, iter * BK, tile_idx_m * BM, barA);
            tokenA = cuda::device::barrier_arrive_tx(barA, 1, sizeof(sA)); // signal thread0 arrival on barA
            cde::cp_async_bulk_tensor_2d_global_to_shared(&sB[0], tma_map_B, iter * BK, tile_idx_n * BN, barB);
            tokenB = cuda::device::barrier_arrive_tx(barB, 1, sizeof(sB)); // signal thread0 arrival on barB
        }
        else {
            tokenA = barA.arrive(); // other threads arrive on barA
            tokenB = barB.arrive(); // other threads arrive on barB
        }

        /* 6. Wait for data to arrive */
        barA.wait(std::move(tokenA)); // wait for barA to complete
        barB.wait(std::move(tokenB)); // wait for barB to complete
        __syncthreads(); // ensure all threads have arrived before using SMEM

        /* 7. Compute using WMMA */
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

template <int BlockMajorSize, int BlockMinorSize>
void create_tensor_map(CUtensorMap *tma_map, bf16* gmem_ptr, int blocks_height, int blocks_width) {
    void* gmem_address = (void*)gmem_ptr;
    
    /* 
       CUDA TMA APIs always take a 5-D description of the tensor, 
       even if your actual tensor is 1-D, 2-D, 3-D, etc.
    */
    
    uint64_t gmem_prob_shape[5] = {(uint64_t)BlockMinorSize*blocks_width, (uint64_t)BlockMajorSize*blocks_height, 1, 1, 1};
    uint64_t gmem_prob_stride[5] = {sizeof(bf16), sizeof(bf16) * BlockMinorSize*blocks_width, 0, 0, 0};
    uint32_t smem_box_shape[5] = {uint32_t(BlockMinorSize), uint32_t(BlockMajorSize), 1, 1, 1};
    /* Hopper TMA you should always use smem_box_stride = {1,1,1,1,1} - Dense SMEM + Swizzle */
    uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

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
        CU_TENSOR_MAP_SWIZZLE_128B, 
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

void runTmaWgmmaBF16(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {
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
    kernel<<<grid_dims, block_dims>>>(M, N, K, C, d_tma_map_A, d_tma_map_B);
}

} // namespace matmul_tma_wgmma

using matmul_tma_wgmma::runTmaWgmmaBF16;