from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import torch


TensorFactory = Callable[[Callable[..., object], str], Callable[[], None]]


@dataclass(frozen=True)
class BenchmarkCase:
    label: str
    build: TensorFactory


def _make_activation_case(
    tokens: int,
    hidden: int,
    *,
    dtype: torch.dtype = torch.float16,
    threshold: float = 0.0,
    alpha: float = 1.702,
    limit: float = 7.0,
    with_scale: bool = False,
    method: str,
) -> BenchmarkCase:
    def _build(op: Callable[..., object], device: str) -> Callable[[], None]:
      generator = torch.Generator(device="cpu")
      generator.manual_seed(1234)
      x = torch.randn(
          tokens, hidden * 2, dtype=dtype, device=device, generator=generator
      )
      out = torch.empty(tokens, hidden, dtype=dtype, device=device)

      if with_scale:
        scale = torch.full((1,), 0.5, dtype=torch.float32, device=device)
        return lambda: op(out, x, scale)
      if method == "fatrelu_and_mul":
        return lambda: op(out, x, float(threshold))
      if method == "swigluoai_and_mul":
        return lambda: op(out, x, float(alpha), float(limit))
      return lambda: op(out, x)

    suffix = f"{tokens}x{hidden}"
    if with_scale:
      suffix += "_scaled"
    return BenchmarkCase(label=suffix, build=_build)


def _make_rms_norm_case(
    tokens: int,
    hidden: int,
    *,
    dtype: torch.dtype = torch.float16,
    epsilon: float = 1e-5,
    fused: bool = False,
) -> BenchmarkCase:
    def _build(op: Callable[..., object], device: str) -> Callable[[], None]:
      generator = torch.Generator(device="cpu")
      generator.manual_seed(1234)
      x = torch.randn(tokens, hidden, dtype=dtype, device=device, generator=generator)
      w = torch.randn(hidden, dtype=dtype, device=device, generator=generator)

      if fused:
        residual = torch.randn(
            tokens, hidden, dtype=dtype, device=device, generator=generator
        )
        return lambda: op(x, residual, w, float(epsilon))

      out = torch.empty_like(x)
      return lambda: op(out, x, w, float(epsilon))

    label = f"{tokens}x{hidden}"
    return BenchmarkCase(label=label, build=_build)


def _make_rotary_embedding_case(
    tokens: int,
    heads: int,
    head_size: int,
    *,
    dtype: torch.dtype = torch.float16,
    max_position: int = 4096,
    is_neox: bool = True,
) -> BenchmarkCase:
    def _build(op: Callable[..., object], device: str) -> Callable[[], None]:
      generator = torch.Generator(device="cpu")
      generator.manual_seed(1234)
      positions = torch.arange(tokens, device=device, dtype=torch.int64)
      query = torch.randn(
          tokens, heads, head_size, dtype=dtype, device=device, generator=generator
      )
      key = torch.randn(
          tokens, heads, head_size, dtype=dtype, device=device, generator=generator
      )
      cos_sin_cache = torch.randn(
          max_position, head_size, dtype=dtype, device=device, generator=generator
      )
      return lambda: op(positions, query, key, int(head_size), cos_sin_cache, is_neox)

    return BenchmarkCase(label=f"{tokens}x{heads}x{head_size}", build=_build)


def default_registry() -> Dict[str, List[BenchmarkCase]]:
    common_activation_cases = [
        _make_activation_case(4096, 4096, method="silu_and_mul"),
        _make_activation_case(8192, 4096, method="silu_and_mul"),
    ]

    registry: Dict[str, List[BenchmarkCase]] = {
        "silu_and_mul": common_activation_cases,
        "mul_and_silu": [
            _make_activation_case(4096, 4096, method="mul_and_silu"),
            _make_activation_case(8192, 4096, method="mul_and_silu"),
        ],
        "gelu_and_mul": [
            _make_activation_case(4096, 4096, method="gelu_and_mul"),
            _make_activation_case(8192, 4096, method="gelu_and_mul"),
        ],
        "gelu_tanh_and_mul": [
            _make_activation_case(4096, 4096, method="gelu_tanh_and_mul"),
            _make_activation_case(8192, 4096, method="gelu_tanh_and_mul"),
        ],
        "fatrelu_and_mul": [
            _make_activation_case(
                4096, 4096, method="fatrelu_and_mul", threshold=0.1
            )
        ],
        "swigluoai_and_mul": [
            _make_activation_case(
                4096, 4096, method="swigluoai_and_mul", alpha=1.702, limit=7.0
            )
        ],
        "gelu_new": [
            _make_activation_case(4096, 4096, method="gelu_new"),
            _make_activation_case(8192, 4096, method="gelu_new"),
        ],
        "gelu_fast": [
            _make_activation_case(4096, 4096, method="gelu_fast"),
            _make_activation_case(8192, 4096, method="gelu_fast"),
        ],
        "gelu_quick": [
            _make_activation_case(4096, 4096, method="gelu_quick"),
            _make_activation_case(8192, 4096, method="gelu_quick"),
        ],
        "silu_and_mul_quant": [
            _make_activation_case(
                4096, 4096, method="silu_and_mul_quant", with_scale=True
            )
        ],
        "rms_norm": [
            _make_rms_norm_case(4096, 4096),
            _make_rms_norm_case(4096, 8192),
        ],
        "fused_add_rms_norm": [
            _make_rms_norm_case(4096, 4096, fused=True),
            _make_rms_norm_case(4096, 8192, fused=True),
        ],
        "rotary_embedding": [
            _make_rotary_embedding_case(2048, 32, 128),
            _make_rotary_embedding_case(4096, 32, 128),
        ],
    }
    return registry
