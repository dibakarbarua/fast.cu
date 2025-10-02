#include <torch/extension.h>

// Forward declaration from the .cu translation unit
torch::Tensor bias_gelu_forward_cuda_v1(torch::Tensor x, torch::Tensor bias);

// Updatable stub
torch::Tensor bias_gelu_forward(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.device().is_cuda(), "Only CUDA is implemented in this example");
    return bias_gelu_forward_cuda_v1(x, bias);
}

// ---- Op schema & impl registration ----
//
// torch.ops.kernels.bias_gelu(x, bias) -> Tensor
//
TORCH_LIBRARY(kernels, m) {
    m.def("bias_gelu(Tensor x, Tensor bias) -> Tensor");
}

TORCH_LIBRARY_IMPL(kernels, CUDA, m) {
    m.impl("bias_gelu", torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(bias_gelu_forward)));
}
