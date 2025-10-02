# fused_kernels/__init__.py
import torch as T
from build_ext import build

# Build (or import from cache) at first import:
_ext = build(verbose=False)

def bias_gelu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused bias + GELU forward (CUDA).
    Shapes:
        x:    [*, C]   (* = any shape; last dim is channels)
        bias: [C]
    """
    return T.ops.kernels.bias_gelu(x, bias)

dev = "cuda"
x = T.randn(32, 1024, device=dev, dtype=T.float16)
bias = T.randn(1024, device=dev, dtype=T.float16)

y = bias_gelu(x, bias)
print(y.shape)
