namespace matmul_tma_wgmma {

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
    <<<(M/BM) * (N/BN), NUM_THREADS>>>(M, N, K, C, d_tma_map_A, d_tma_map_B);
}

} // namespace matmul_tma_wgmma

using matmul_tma_wgmma::runTmaWgmmaBF16;