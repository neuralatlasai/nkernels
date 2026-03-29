# Legacy Migration Status

Source tree comparison generated on 2026-03-29 09:54:15 UTC.

## Summary
- Legacy `csrc` files: `262`
- `nkernels/csrc` files: `16`
- Relative-path overlap: `1`
- Legacy files still missing in `nkernels/csrc`: `261`
- Unique legacy exported methods: `166`
- Canonical manifest entries retained in `nkernels`: `180`
- Actual binding registrations currently present in `nkernels/csrc`: `0`

## Interpretation
The current `nkernels` tree contains migration infrastructure and API inventorying, but it does not yet contain the legacy binding layer or the full set of legacy kernel implementation units.

## Missing File Samples
- `activation_kernels.cu`
- `attention/attention_dtypes.h`
- `attention/attention_generic.cuh`
- `attention/attention_kernels.cuh`
- `attention/attention_utils.cuh`
- `attention/dtype_bfloat16.cuh`
- `attention/dtype_float16.cuh`
- `attention/dtype_float32.cuh`
- `attention/dtype_fp8.cuh`
- `attention/merge_attn_states.cu`
- `attention/mla/cutlass_sm100_mla/device/sm100_mla.hpp`
- `attention/mla/cutlass_sm100_mla/kernel/sm100_fmha_mla_reduction.hpp`
- `attention/mla/cutlass_sm100_mla/kernel/sm100_fmha_mla_tma_warpspecialized.hpp`
- `attention/mla/cutlass_sm100_mla/kernel/sm100_mla_tile_scheduler.hpp`
- `attention/mla/sm100_cutlass_mla_kernel.cu`
- `attention/paged_attention_v1.cu`
- `attention/paged_attention_v2.cu`
- `attention/vertical_slash_index.cu`
- `cache.h`
- `cache_kernels.cu`
- `cache_kernels_fused.cu`
- `concat_mla_q.cuh`
- `core/batch_invariant.hpp`
- `core/exception.hpp`
- `core/math.hpp`
- `core/registration.h`
- `cpu/activation.cpp`
- `cpu/cpu_arch_macros.h`
- `cpu/cpu_attn.cpp`
- `cpu/cpu_attn_amx.hpp`
- `cpu/cpu_attn_impl.hpp`
- `cpu/cpu_attn_neon.hpp`
- `cpu/cpu_attn_neon_bfmmla.hpp`
- `cpu/cpu_attn_vec.hpp`
- `cpu/cpu_attn_vec16.hpp`
- `cpu/cpu_attn_vxe.hpp`
- `cpu/cpu_fused_moe.cpp`
- `cpu/cpu_types.hpp`
- `cpu/cpu_types_arm.hpp`
- `cpu/cpu_types_riscv.hpp`
- `cpu/cpu_types_scalar.hpp`
- `cpu/cpu_types_vsx.hpp`
- `cpu/cpu_types_vxe.hpp`
- `cpu/cpu_types_x86.hpp`
- `cpu/cpu_wna16.cpp`
- `cpu/dnnl_helper.cpp`
- `cpu/dnnl_helper.h`
- `cpu/dnnl_kernels.cpp`
- `cpu/float_convert.hpp`
- `cpu/generate_cpu_attn_dispatch.py`
- `cpu/layernorm.cpp`
- `cpu/micro_gemm/cpu_micro_gemm_amx.hpp`
- `cpu/micro_gemm/cpu_micro_gemm_impl.hpp`
- `cpu/micro_gemm/cpu_micro_gemm_vec.hpp`
- `cpu/mla_decode.cpp`
- `cpu/pos_encoding.cpp`
- `cpu/sgl-kernels/common.h`
- `cpu/sgl-kernels/gemm.cpp`
- `cpu/sgl-kernels/gemm.h`
- `cpu/sgl-kernels/gemm_fp8.cpp`
- `cpu/sgl-kernels/gemm_int8.cpp`
- `cpu/sgl-kernels/moe.cpp`
- `cpu/sgl-kernels/moe_fp8.cpp`
- `cpu/sgl-kernels/moe_int8.cpp`
- `cpu/sgl-kernels/vec.h`
- `cpu/shm.cpp`
- `cpu/torch_bindings.cpp`
- `cpu/utils.cpp`
- `cpu/utils.hpp`
- `cub_helpers.h`
- `cuda_compat.h`
- `cuda_utils.h`
- `cuda_utils_kernels.cu`
- `cuda_vec_utils.cuh`
- `cuda_view.cu`
- `cumem_allocator.cpp`
- `cumem_allocator_compat.h`
- `custom_all_reduce.cu`
- `custom_all_reduce.cuh`
- `custom_all_reduce_test.cu`
