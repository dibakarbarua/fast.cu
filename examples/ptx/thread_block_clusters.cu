// cluster_tma_multicast.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <cstdio>
#include <cassert>

namespace cg = cooperative_groups;

/********* Problem Statement ********/
/*
Below is a compact, end-to-end CUDA C++ (Hopper) example that shows a thread-block cluster kernel where:
- Each CTA in the cluster is assigned a distinct region (“tile”) of the cluster tile in GMEM.
- Each CTA issues one TMA load with multicast so that a single transfer per CTA populates every CTA’s DSMEM copy of that CTA’s tile.
- CTAs arrive at a cluster barrier, then do a simple element-wise atomic add on DSMEM using the CTA’s cluster rank as the operand.
- After another cluster barrier, each CTA TMA-stores its tile back to GMEM.
*/

// --- Tunables ---
constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
using T = float;                  // element type
static_assert(sizeof(T) == 4);

// Simple gmem layout: row-major 2D
struct Tensor2D {
  T* ptr;
  int ld; // leading dimension (stride in elements)
};

// ---------------------------------------------
// Lightweight helpers for DSMEM & cluster
// ---------------------------------------------
__device__ inline T* smem_tile_base()
{
  extern __shared__ __align__(16) unsigned char smem_raw[];
  return reinterpret_cast<T*>(smem_raw); // TILE_M * TILE_N elements reserved
}

// Map my CTA-local DSMEM pointer into peer CTA's DSMEM (rank r)
template <typename U>
__device__ inline U* map_dsmem_to_rank(cg::cluster_group& cluster, U* local_ptr, int peer_rank)
{
  return cluster.map_shared_rank(local_ptr, peer_rank);
}

// ---------------------------------------------
// TMA Multicast helpers (Hopper inline PTX)
// Each helper assumes a 2D tile (TILE_M x TILE_N) of sizeof(T) bytes each.
//
// In practice you’d set up a CUtensorMap once and keep it around;
// for clarity here we address linearly with a base + row offset.
// ---------------------------------------------

// Multicast GMEM->DSMEM for THIS CTA's tile to ALL CTAs' DSMEM.
//   g_src: base of this CTA's tile in GMEM (row-major)
//   s_dst: base of THIS CTA's DSMEM tile (all CTAs have the same layout)
//   ld_g  : leading dim in GMEM (elements)
__device__ inline void tma_multicast_load_2d(const T* g_src, T* s_dst, int ld_g,
                                             int ctas_in_cluster)
{
  // Bytes per row and number of rows
  const uint32_t row_bytes = TILE_N * sizeof(T);
  const uint32_t rows      = TILE_M;

  // Multicast mask: lower N bits for N CTAs in cluster.
  // Hopper expects a 32-bit mask; we assume <= 32 CTAs per cluster.
  const uint32_t mcast_mask = (ctas_in_cluster >= 32) ? 0xFFFF'FFFFu
                                                      : ((1u << ctas_in_cluster) - 1u);

  // NOTE:
  // - We use cp.async.bulk.tensor.2d.[shared::cluster].global ... .multicast
  // - We rely on the compiler/assembler to choose an mbarrier and handle completion.
  // - For brevity, bounds checking omitted.
  asm volatile(
    "{\n\t"
    ".reg .b64 r_g, r_s;\n\t"
    "cvta.to.shared.u64 r_s, %0;\n\t"
    "cvta.to.global.u64 r_g, %1;\n\t"
    // mcast 2D bulk tensor: rows, row_bytes, gmem stride in bytes, smem stride in bytes (=row_bytes)
    "cp.async.bulk.tensor.2d.shared::cluster.global"
    ".mbarrier::complete_tx::bytes"
    ".multicast"
    " [%2], [r_s], [r_g], %3, %4, %5, %4;\n\t"
    "}\n"
    :
    : "l"(s_dst), "l"(g_src),
      "l"(nullptr),                                    // (let assembler choose internal mbarrier)
      "r"(rows), "r"(row_bytes), "r"(ld_g * sizeof(T)), "r"(row_bytes),
      "r"(mcast_mask)
    : "memory"
  );
}

// TMA store DSMEM->GMEM (no multicast, each CTA stores its own tile)
__device__ inline void tma_store_2d(const T* s_src, T* g_dst, int ld_g)
{
  const uint32_t row_bytes = TILE_N * sizeof(T);
  const uint32_t rows      = TILE_M;

  asm volatile(
    "{\n\t"
    ".reg .b64 r_g, r_s;\n\t"
    "cvta.to.shared.u64 r_s, %0;\n\t"
    "cvta.to.global.u64 r_g, %1;\n\t"
    "cp.async.bulk.tensor.2d.global.shared::cluster"
    ".mbarrier::complete_tx::bytes"
    " [r_g], [r_s], %2, %3, %4, %3;\n\t"
    "}\n"
    :
    : "l"(s_src), "l"(g_dst),
      "r"(rows), "r"(row_bytes), "r"(ld_g * sizeof(T)), "r"(row_bytes)
    : "memory"
  );
}

// ---------------------------------------------
// Kernel: cluster CTA tiles with TMA multicast
// ---------------------------------------------
__global__ __cluster_dims__(/*x=*/4, /*y=*/1, /*z=*/1)
void cluster_kernel(Tensor2D in, Tensor2D out,
                    int tiles_per_row, int tiles_per_col)
{
#if __CUDA_ARCH__ < 900
  return; // Hopper only
#endif
  // Cluster group & ranks
  auto cluster = cg::this_cluster();
  const int cluster_size = cluster.dim_blocks().x * cluster.dim_blocks().y * cluster.dim_blocks().z;
  const int cta_rank     = cluster.block_rank();

  // DSMEM tile base (each CTA has the same shape)
  T* s_tile = smem_tile_base();

  // Compute which GMEM tile this CTA owns (simple 1D rank -> 2D mapping)
  // Cluster covers a "cluster tile" of tiles_per_row x tiles_per_col tiles.
  // Each CTA gets one (or more) tiles; here we keep it 1:1 for clarity.
  const int tile_id   = cta_rank;
  const int tile_y    = tile_id / tiles_per_row;
  const int tile_x    = tile_id % tiles_per_row;

  // Guard: if cluster has more CTAs than tiles, let extras idle safely.
  if (tile_y >= tiles_per_col) {
    cluster.sync();
    cluster.sync();
    return;
  }

  // GMEM base for this CTA's tile (row-major)
  const T* g_src_tile = in.ptr  + (tile_y * TILE_M) * in.ld  + (tile_x * TILE_N);
  T*       g_dst_tile = out.ptr + (tile_y * TILE_M) * out.ld + (tile_x * TILE_N);

  // ---------------------------------------------
  // Stage 1: Each CTA issues one TMA multicast load of ITS tile.
  // That single transfer populates *all* CTAs' DSMEM (same address s_tile) with this CTA's tile.
  // After the multicast, every CTA’s DSMEM holds the last multicast written by ANY CTA.
  // To keep tiles distinct concurrently, we store each multicast into a per-CTA DSMEM page.
  // We do that by offsetting s_tile by cta_rank * tile_bytes.
  // ---------------------------------------------
  constexpr size_t tile_elems = size_t(TILE_M) * TILE_N;
  constexpr size_t tile_bytes = tile_elems * sizeof(T);

  // Layout DSMEM as [cluster_size][tile_elems] so every CTA's multicast has its own page.
  T* s_my_page = reinterpret_cast<T*>(
                   reinterpret_cast<char*>(s_tile) + size_t(cta_rank) * tile_bytes);

  // Multicast my GMEM tile into *all CTAs* DSMEM pages at offset for my rank.
  // NOTE: Every CTA calls this; after all these calls complete, each CTA’s DSMEM
  //       contains cluster_size distinct pages (one per CTA’s tile).
  tma_multicast_load_2d(g_src_tile, s_my_page, in.ld, cluster_size);

  // Wait for all multicast transfers to complete and make DSMEM visible cluster-wide.
  cluster.sync();

  // ---------------------------------------------
  // Stage 2: Element-wise atomic op on every tile-page in DSMEM.
  // Operand = (float)cta_rank (of the CTA doing the update).
  // This demonstrates DSMEM atomics + cross-CTA visibility.
  // Each CTA will walk all pages [0..cluster_size)
  // and atomically add its rank into every element of every page.
  // ---------------------------------------------
  const float op = static_cast<float>(cta_rank);
  for (int page = 0; page < cluster_size; ++page) {
    T* page_base = reinterpret_cast<T*>(
                     reinterpret_cast<char*>(s_tile) + size_t(page) * tile_bytes);

    // Stride threads over the tile
    for (int idx = threadIdx.x; idx < (int)tile_elems; idx += blockDim.x) {
      // DSMEM atomic add (supported for 32-bit floats/ints)
      atomicAdd(reinterpret_cast<float*>(&page_base[idx]), op);
    }
    __syncthreads(); // keep per-CTA progress tidy inside a page
  }

  // Make sure all CTAs finished updating all pages
  cluster.sync();

  // ---------------------------------------------
  // Stage 3: TMA store — each CTA writes *its* page back to its GMEM tile.
  // The page we own is at offset [cta_rank].
  // ---------------------------------------------
  tma_store_2d(s_my_page, g_dst_tile, out.ld);
}

// ---------------------------------------------
// Host utilities
// ---------------------------------------------
#define CHECK_CUDA(cmd) do {                               \
  cudaError_t e = (cmd);                                   \
  if (e != cudaSuccess) {                                  \
    fprintf(stderr, "%s failed: %s (%d)\n", #cmd,          \
            cudaGetErrorString(e), int(e));                \
    std::abort();                                          \
  }                                                        \
} while(0)

int main()
{
  // Problem: a cluster processes a tiles_per_row x tiles_per_col group of 64x64 tiles.
  const int tiles_per_row = 2;
  const int tiles_per_col = 2;
  const int cluster_tiles = tiles_per_row * tiles_per_col;   // 4 tiles
  const int cluster_size  = 4;                               // 4 CTAs/cluster

  // Global tensor dimensions
  const int M = tiles_per_col * TILE_M;
  const int N = tiles_per_row * TILE_N;
  const int ld = N;

  // Allocate & initialize input/output
  T *d_in = nullptr, *d_out = nullptr;
  CHECK_CUDA(cudaMalloc(&d_in,  M * N * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_out, M * N * sizeof(T)));
  CHECK_CUDA(cudaMemset(d_out, 0, M * N * sizeof(T)));

  // Fill input with something simple
  {
    std::vector<T> h(M * N);
    for (int r = 0; r < M; ++r)
      for (int c = 0; c < N; ++c)
        h[r * ld + c] = T(r + c);
    CHECK_CUDA(cudaMemcpy(d_in, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  }

  // Pack tensor views
  Tensor2D in{d_in, ld};
  Tensor2D out{d_out, ld};

  // Kernel launch: 1 cluster processes a 2x2 tiling (4 CTAs).
  // You can launch many clusters by increasing gridDim.x by a multiple of cluster_size and
  // offsetting tiles accordingly (left out for clarity).
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(1 * cluster_size, 1, 1);   // one cluster-worth of CTAs
  cfg.blockDim = dim3(256, 1, 1);
  cfg.dynamicSmemBytes = cluster_size * TILE_M * TILE_N * sizeof(T); // DSMEM pages
  cfg.attrs.clear();
  cfg.attrs.push_back(cudaLaunchAttribute{
    .id = cudaLaunchAttributeClusterDimension,
    .val = { .clusterDim = { cluster_size, 1, 1 } }
  });
  cfg.attrs.push_back(cudaLaunchAttribute{
    .id = cudaLaunchAttributeNonPortableClusterSizeAllowed,
    .val = { .nonPortableClusterSizeAllowed = 1 }
  });

  void* args[] = { &in, &out, (void*)&tiles_per_row, (void*)&tiles_per_col };

  CHECK_CUDA(cudaLaunchKernelEx(&cfg,
                                (void*)cluster_kernel,
                                args));

  CHECK_CUDA(cudaDeviceSynchronize());

  // Quick sanity check: after the DSMEM atomics,
  // each output tile = (original tile) + sum_{r=0..3} r  == + (0+1+2+3) == +6
  {
    std::vector<T> h(M * N);
    CHECK_CUDA(cudaMemcpy(h.data(), d_out, h.size() * sizeof(T), cudaMemcpyDeviceToHost));
    // Spot-check a few entries
    printf("out[0,0]=%.1f  out[M-1,N-1]=%.1f  expected offsets +6.0\n",
           double(h[0]), double(h[(M-1)*ld + (N-1)]));
  }

  CHECK_CUDA(cudaFree(d_in));
  CHECK_CUDA(cudaFree(d_out));
  return 0;
}
