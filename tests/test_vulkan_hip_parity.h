#pragma once

// Vulkan-vs-HIP parity (brotensor_test_vulkan_hip_parity): the ops the
// siblings run at inference, fed identical inputs on Device::hip(0) and
// Device::vulkan(0) in one process and compared output against output. Where
// the CPU backend runs the op cheaply it is run too (FP32, on the same rounded
// inputs), so a mismatch can be attributed to one backend.
//
// The comparison is range-relative: |vk - hip| <= atol + rtol * max|hip|,
// with per-op tolerances set by the dtype policy (docs/vulkan.md): FP16
// outputs a few FP16 ulps of the output range, BF16 operands are staged as
// FP16 by Vulkan's matrix cores (more mantissa than BF16, so the BF16 tolerance
// covers it), FP32 SIMT paths 1e-5..1e-4, quantised weights the FP16 tile
// rounding of the decoded weight. Built only when HIP and Vulkan are both in
// the build; exits 77 (skipped) when either device is missing.

#include "test_vulkan_common.h"

#include <functional>
#include <string>
#include <vector>

namespace vhp {

using brotensor::Device;
using brotensor::Dtype;
using brotensor::Tensor;

inline constexpr Device kHip = Device::hip(0);
inline constexpr Device kVk = Device::vulkan(0);
inline constexpr Device kCpu = Device::cpu();

// Runs the op on `d` and returns its (main) output as FP32 host values.
using GpuFn = std::function<std::vector<float>(Device d)>;
// Runs the FP32 CPU reference; null when the CPU backend has no cheap twin.
using CpuFn = std::function<std::vector<float>()>;

// Runs gpu(kHip), gpu(kVk) and cpu() (when given), compares Vulkan against
// HIP with atol + rtol * max|hip|, records and prints one table row.
void check(const std::string& op, const std::string& shape, const GpuFn& gpu, const CpuFn& cpu, double atol,
           double rtol);

// Records a row whose values were computed by the caller (multi-output ops).
void check_values(const std::string& op, const std::string& shape, const std::vector<float>& hip,
                  const std::vector<float>& vk, const std::vector<float>* cpu, double atol, double rtol);

// Records an exact-match row (integer outputs): `mismatches` of `n` differ,
// `allowed` tolerated.
void check_exact(const std::string& op, const std::string& shape, int mismatches, int n, int allowed);

// Inputs: random values rounded to `dt`, uploaded at `dt` on `d` (FP32 on CPU).
inline std::vector<float> rnd(std::size_t n, std::uint64_t seed, float lo, float hi, Dtype dt) {
    return vkt::random_values(n, seed, lo, hi, dt);
}
inline Tensor up(const std::vector<float>& v, int rows, int cols, Dtype dt, Device d) {
    return vkt::upload(v, rows, cols, d.type == brotensor::DeviceType::CPU ? Dtype::FP32 : dt, d);
}
inline Tensor i32(const std::vector<int32_t>& v, int rows, int cols, Device d) {
    Tensor t = Tensor::empty_on(d, rows, cols, Dtype::INT32);
    t.copy_from_host_raw(v.data(), v.size() * 4);
    return t;
}
inline std::vector<float> down(const Tensor& t) { return vkt::download(t); }

std::string shp(const char* fmt, ...);

// Groups (test_vulkan_hip_parity.cpp / _ops.cpp).
void run_gemm();
void run_quant();
void run_norm();
void run_attention();
void run_conv();
void run_misc();
void run_audio();

}  // namespace vhp
