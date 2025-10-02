# fused_kernels/build_ext.py
from torch.utils.cpp_extension import load
import torch, pybind11
from pathlib import Path

ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0

_THIS_DIR = Path(__file__).resolve().parent
SRC = [
    str(_THIS_DIR / "kernels" / "fused_bias_gelu" / "fused_bias_gelu.cpp"),
    str(_THIS_DIR / "kernels" / "fused_bias_gelu" / "naive_fused_bias_gelu.cu"),
]

def build(verbose: bool = False):
    mod = load(
        name="kernels_ext",
        sources=SRC,
        extra_cflags=[
            f"-D_GLIBCXX_USE_CXX11_ABI={ABI}",
            "-O3", 
            "-std=c++17", 
            "-DNDEBUG",
            f"-I{pybind11.get_include()}",
        ],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-lineinfo",
        ],
        verbose=verbose,
    )
    return mod
