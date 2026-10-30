"""Share only backend controls between otherwise independent command schemas."""

from pydantic import BaseModel, ConfigDict


class RuntimeConfig(BaseModel):
    """Select explicit CUDA convolution and matrix multiplication policies."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    cuda_allow_tf32: bool = False
    cudnn_enabled: bool = True
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False
