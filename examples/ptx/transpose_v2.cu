#include <cuda.h>
#include <stdint.h>

// Choose grid so each CTA handles one 128×128 tile of the source:
// dim3 grid( (N+BN-1)/BN, (M+BM-1)/BM ); dim3 block(32);
// (Block size small: only a couple of threads actively issue TMA.)
// Set SM occupancy: 2–4 CTAs per SM recommended → robust latency hiding via multiple concurrent TMA ops.
// Use cooperative launch only if you add cross-CTA sync (not needed here).

#ifndef BM
#define BM 128
#endif
#ifndef BN
#define BN 128
#endif
// Two sub-tiles per stage to keep TMA engines fed
static constexpr int SUBS = 2;          // 2 stripes: each 128x64
static constexpr int STAGES = 2;        // ping-pong

// Shared memory boxes (double-buffered), and one mbarrier per stage.
extern "C" __global__
void transpose_tma_4B(const CUtensorMap* __restrict__ tma_src,
                      const CUtensorMap* __restrict__ tma_dst,
                      const int M, const int N,     // source dims (M rows, N cols)
                      const int ld_src,             // leading dim (N) in elements
                      const int ld_dst)             // leading dim (M) in elements
{
  using barrier_t = unsigned long long;

  // Shared: [stage][SUBS] 2D landing boxes; we only need the box headers (TMA addresses a box)
  __shared__ uint8_t sbox[STAGES][SUBS][1]; // header symbol only (no manual addressing)
  __shared__ barrier_t s_mbar_load[STAGES];
  __shared__ barrier_t s_mbar_store[STAGES];

  // Init barriers (single thread)
  if (threadIdx.x == 0) {
    #pragma unroll
    for (int s = 0; s < STAGES; ++s) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                   :: "r"(&s_mbar_load[s]), "r"(1));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                   :: "r"(&s_mbar_store[s]), "r"(1));
    }
  }
  __syncthreads();

  // CTA tile coords
  const int tile_m = blockIdx.y * BM;         // row start in src
  const int tile_n = blockIdx.x * BN;         // col start in src

  // Bounds (handle fringes)
  const int valid_m = min(BM, M - tile_m);
  const int valid_n = min(BN, N - tile_n);
  if (valid_m <= 0 || valid_n <= 0) return;

  // Each stage transfers a full 128x128 via two 128x64 sub-tiles:
  // sub 0: cols [0, 64), sub 1: cols [64, 128)
  // For fringes, we clamp cols per sub.
  const int sub_cols[SUBS] = { min(64, valid_n), max(0, valid_n - 64) };

  // Precompute TX sizes (bytes) for load/store of the stage
  auto tx_bytes_sub = [&](int sub)->uint32_t {
    int w = sub_cols[sub];
    if (w <= 0) return 0;
    return uint32_t(valid_m) * uint32_t(w) * 4u;
  };

  // Choose “loader threads”
  const int tLoad0 = 0;   // issues sub 0 load
  const int tLoad1 = 1;   // issues sub 1 load
  const int tStore = 0;   // reuse thread 0 to issue store after loads complete

  // ---------------------------
  // Stage 0: issue LOADs
  // ---------------------------
  // Arrive-expect for both sub-loads
  if (threadIdx.x == tLoad0) {
    uint32_t bytes = tx_bytes_sub(0);
    if (bytes) asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 [%0], %1;" ::
      "r"(&s_mbar_load[0]), "r"(bytes));
  }
  if (threadIdx.x == tLoad1) {
    uint32_t bytes = tx_bytes_sub(1);
    if (bytes) asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 [%0], %1;" ::
      "r"(&s_mbar_load[0]), "r"(bytes));
  }
  __syncthreads();

  // Two independent TMA LOADs (different threads) into stage-0 boxes
  if (threadIdx.x == tLoad0 && sub_cols[0] > 0) {
    // cp.async.bulk.tensor.2d.shared.global.mbarrier::complete_tx::bytes
    asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global"
      ".mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3}], [%4];\n" ::
      // %0 : shared box header (landing for sub 0)
      "r"(&sbox[0][0][0]),
      // %1 : src tensor map
      "l"(tma_src),
      // {%2,%3} : coords in src (row=tile_m, col=tile_n+0)
      "r"(tile_m), "r"(tile_n + 0),
      // %4 : barrier
      "r"(&s_mbar_load[0])
    );
  }
  if (threadIdx.x == tLoad1 && sub_cols[1] > 0) {
    asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global"
      ".mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3}], [%4];\n" ::
      "r"(&sbox[0][1][0]),
      "l"(tma_src),
      "r"(tile_m), "r"(tile_n + 64),
      "r"(&s_mbar_load[0])
    );
  }

  // ---------------------------
  // Main loop: (only one “k” stage here since we do pure copy)
  // But we still demonstrate double-buffered overlap:
  //   stage S store while stage S^1 loads next tile (here: none).
  // ---------------------------

  // Wait for stage-0 LOAD completion (one waiter is enough)
  if (threadIdx.x == tStore) {
    asm volatile("mbarrier.try_wait.parity.shared::cta.b64 %0, [%1];"
                 : "=r"(*(volatile int*)0) : "r"(&s_mbar_load[0]));
  }
  __syncthreads();

  // Issue STORE for stage-0 from SHMEM box to DEST (transposed coords)
  // We split into two sub-stores matching two sub-loads, to overlap with the next loads in a real loop.
  if (threadIdx.x == tStore) {
    // Expect TX = sum of present sub-tiles
    uint32_t bytes0 = tx_bytes_sub(0);
    uint32_t bytes1 = tx_bytes_sub(1);
    uint32_t total   = bytes0 + bytes1;
    if (total) {
      asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 [%0], %1;" ::
        "r"(&s_mbar_store[0]), "r"(total));
    }
  }
  __syncthreads();

  // Sub-store 0: goes to destination **transposed** tile origin (tile_n, tile_m)
  if (threadIdx.x == tStore && sub_cols[0] > 0) {
    // coord in dst: row' = tile_n + 0 .. +63, col' = tile_m .. tile_m+valid_m-1
    // We still give TMA the 2D origin of the box; it knows the shape from the map.
    asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta"
      ".mbarrier::complete_tx::bytes "
      "[%0, {%1, %2}], [%3], [%4];\n" ::
      // %0 : dst tensor map
      "l"(tma_dst),
      // {%1,%2} : coords in dst (row'=tile_n+0, col'=tile_m), i.e. transposed
      "r"(tile_n + 0), "r"(tile_m),
      // %3 : shared source box (stage 0, sub 0)
      "r"(&sbox[0][0][0]),
      // %4 : barrier
      "r"(&s_mbar_store[0])
    );
  }
  // Sub-store 1: rows tile_n+64.., cols tile_m
  if (threadIdx.x == tStore && sub_cols[1] > 0) {
    asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta"
      ".mbarrier::complete_tx::bytes "
      "[%0, {%1, %2}], [%3], [%4];\n" ::
      "l"(tma_dst),
      "r"(tile_n + 64), "r"(tile_m),
      "r"(&sbox[0][1][0]),
      "r"(&s_mbar_store[0])
    );
  }

  // Wait for the STORE to complete before exit (ensures global write visibility)
  if (threadIdx.x == tStore) {
    asm volatile("mbarrier.try_wait.parity.shared::cta.b64 %0, [%1];"
                 : "=r"(*(volatile int*)0) : "r"(&s_mbar_store[0]));
  }
}
