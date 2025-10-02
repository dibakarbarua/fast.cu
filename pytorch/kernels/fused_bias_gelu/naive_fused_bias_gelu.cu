#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename T>
__device__ __forceinline__ T gelu_tanh(T x) {
    // Approx GELU per Hendrycks: 0.5*x*(1+tanh(√(2/π)*(x + 0.044715*x^3)))
    const T kAlpha = T(0.7978845608028654);   // √(2/π)
    const T kCubic = T(0.044715);
    T x3 = x * x * x;
    T t = kAlpha * (x + kCubic * x3);
    return T(0.5) * x * (T(1) + tanh(t));
}

template <typename scalar_t>
__global__ void bias_gelu_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ y,
    int64_t rows, int64_t cols
) {
    // Each block processes one row, threads cover columns
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= rows || col >= cols) return;

    int64_t idx = row * cols + col;
    scalar_t v = x[idx] + bias[col];
    y[idx] = gelu_tanh<scalar_t>(v);
}

torch::Tensor bias_gelu_forward_cuda_v1(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
    TORCH_CHECK(x.scalar_type() == bias.scalar_type(), "dtype mismatch");
    TORCH_CHECK(x.dim() >= 2, "expect at least 2D [N, C, ...]");
    TORCH_CHECK(bias.dim() == 1, "bias must be [C]");

    // Collapse to [N, C] by treating trailing dims as part of C
    auto sizes = x.sizes();
    int64_t rows = 1;
    for (int i = 0; i < (int)x.dim() - 1; ++i) rows *= sizes[i];
    int64_t cols = sizes.back();
    TORCH_CHECK(bias.size(0) == cols, "bias shape mismatch");

    auto y = torch::empty_like(x);

    const int threads = 256;
    dim3 block(threads);
    dim3 grid(rows, (cols + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, x.scalar_type(), "bias_gelu_forward_cuda", [&](){
            bias_gelu_forward_kernel<scalar_t><<<grid, block>>>(
                x.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                y.data_ptr<scalar_t>(),
                rows, cols
            );
        });
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "kernel launch failed");
    return y;
}
