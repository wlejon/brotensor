// Metal data movement: contiguous & strided copies, row/channel/batched
// concats, and NCHW <-> sequence transposes via MTLBlitCommandEncoder +
// single-source brass MIR kernels (MirKind::Copy2d, MirKind::Transpose).

#include "mtl_mir_cache.h"

#include <brotensor/runtime.h>

#import "internal.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor::detail::metal {

using metal_impl::new_command_buffer;
using metal_impl::pool_require;

namespace {

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

void need(const char* op, bool ok, const char* what) {
    if (!ok) fail(op, what);
}

std::uint64_t addr(const void* p) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(p));
}

std::size_t elem_bytes(Dtype dt, const char* op) {
    const int e = ::brotensor::dtype_size_bytes(dt);
    if (e <= 0) fail(op, "not defined for block-quantised dtypes");
    return static_cast<std::size_t>(e);
}

void ensure(Tensor& t, long long r, long long c, Dtype dt, const char* op) {
    if (r < 0 || c < 0 || r > 0x7fffffffLL || c > 0x7fffffffLL) fail(op, "output too large");
    if (t.rows != r || t.cols != c || t.dtype != dt) {
        t.resize(static_cast<int>(r), static_cast<int>(c), dt);
    }
}

void copy_bytes_gpu(const void* src, const void* dst, std::size_t bytes, const char* op) {
    if (bytes == 0 || src == dst) return;
    @autoreleasepool {
        NSUInteger os = 0, od = 0;
        id<MTLBuffer> bs = pool_require(src, os, op, "src");
        id<MTLBuffer> bd = pool_require(dst, od, op, "dst");
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        [blit copyFromBuffer:bs sourceOffset:os
                    toBuffer:bd destinationOffset:od
                        size:bytes];
        [blit endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

void transpose_batched(const char* op, const Tensor& X, Tensor& Y,
                       int batch, int rows, int cols) {
    const std::size_t es = elem_bytes(X.dtype, op);
    if (es != 2 && es != 4) fail(op, "FP32 / FP16 / BF16 only on Metal");
    if (batch == 0 || rows == 0 || cols == 0) return;
    if (static_cast<long long>(batch) * rows * cols > X.size()) {
        fail(op, "X is smaller than N*C*H*W");
    }
    const MirKernel mk = mir_kernel({MirKind::Transpose, static_cast<std::uint32_t>(es)});
    const std::uint32_t ub = static_cast<std::uint32_t>(batch);
    const std::uint32_t ur = static_cast<std::uint32_t>(rows);
    const std::uint32_t uc = static_cast<std::uint32_t>(cols);
    const std::uint32_t gx = std::min<std::uint32_t>((uc + 31u) / 32u, 65535u);
    const std::uint32_t gy = std::min<std::uint32_t>((ur + 31u) / 32u, 65535u);
    const std::uint32_t gz = std::min<std::uint32_t>(ub, 65535u);
    mir_dispatch_transpose(mk, gx, gy, gz, op, X.data, Y.data, ub, ur, uc);
}

}  // namespace

void copy_d2d(const Tensor& src, int src_off, Tensor& dst, int dst_off, int n) {
    if (n <= 0) return;
    const std::size_t e = elem_bytes(src.dtype, "copy_d2d");
    const auto* s0 = static_cast<const char*>(src.data) + e * static_cast<std::size_t>(src_off);
    auto* d0 = static_cast<char*>(dst.data) + e * static_cast<std::size_t>(dst_off);
    copy_bytes_gpu(s0, d0, e * static_cast<std::size_t>(n), "copy_d2d");
}

void copy_d2d_strided(const Tensor& src, int src_off, int src_pitch,
                      Tensor& dst, int dst_off, int dst_pitch,
                      int width, int height) {
    if (width <= 0 || height <= 0) return;
    std::size_t e = elem_bytes(src.dtype, "copy_d2d_strided");
    const auto* s0 = static_cast<const char*>(src.data) + e * static_cast<std::size_t>(src_off);
    auto* d0 = static_cast<char*>(dst.data) + e * static_cast<std::size_t>(dst_off);
    if ((src_pitch == width && dst_pitch == width) || height == 1) {
        copy_bytes_gpu(s0, d0, e * static_cast<std::size_t>(width) * static_cast<std::size_t>(height),
                       "copy_d2d_strided");
        return;
    }
    std::uint32_t mult = 1;
    if (e == 8) { mult = 2; e = 4; }
    const std::uint32_t sp = static_cast<std::uint32_t>(src_pitch) * mult;
    const std::uint32_t dp = static_cast<std::uint32_t>(dst_pitch) * mult;
    const std::uint32_t w = static_cast<std::uint32_t>(width) * mult;
    const std::uint32_t h = static_cast<std::uint32_t>(height);
    const int lanes = mir_copy2d_lanes(static_cast<int>(e), w, sp, dp, addr(s0), addr(d0));
    const MirKernel mk = mir_kernel(
        {MirKind::Copy2d, static_cast<std::uint32_t>(e),
         Dtype::FP32, Dtype::FP32, static_cast<std::uint8_t>(lanes)});
    const std::uint32_t nv = w / static_cast<std::uint32_t>(lanes);
    const std::uint32_t gx =
        std::max<std::uint32_t>(1u, std::min<std::uint32_t>((nv + 255u) / 256u, 64u));
    mir_dispatch_copy2d(mk, gx, std::min<std::uint32_t>(h, 65535u),
                        "copy_d2d_strided", s0, d0, sp, dp, w, h);
}

void concat_rows(const std::vector<const Tensor*>& parts, Tensor& out) {
    int total = 0;
    Dtype dt = Dtype::FP32;
    bool seen = false;
    for (const Tensor* p : parts) {
        if (!p) continue;
        total += p->size();
        if (!seen) { dt = p->dtype; seen = true; }
    }
    if (out.rows != total || out.cols != 1 || out.dtype != dt) out.resize(total, 1, dt);
    if (total == 0) return;
    const std::size_t e = elem_bytes(dt, "concat_rows");
    auto* dst = static_cast<char*>(out.data);
    std::size_t off = 0;
    for (const Tensor* p : parts) {
        if (!p || p->size() == 0) continue;
        if (p->dtype != dt) fail("concat_rows", "parts differ in dtype");
        const std::size_t bytes = e * static_cast<std::size_t>(p->size());
        copy_bytes_gpu(p->data, dst + off, bytes, "concat_rows");
        off += bytes;
    }
}

void split_rows(const Tensor& in, const std::vector<Tensor*>& parts) {
    const std::size_t e = elem_bytes(in.dtype, "split_rows");
    const auto* src = static_cast<const char*>(in.data);
    std::size_t off = 0;
    for (Tensor* p : parts) {
        if (!p || p->size() == 0) continue;
        const std::size_t bytes = e * static_cast<std::size_t>(p->size());
        copy_bytes_gpu(src + off, p->data, bytes, "split_rows");
        off += bytes;
    }
}

void concat_batched_rows(const std::vector<const Tensor*>& parts, Tensor& out) {
    const char* op = "concat_batched_rows";
    int B = -1;
    long long total = 0;
    Dtype dt = Dtype::FP32;
    for (const Tensor* p : parts) {
        if (!p) continue;
        if (B < 0) { B = p->rows; dt = p->dtype; }
        need(op, p->rows == B, "parts must have the same number of rows");
        need(op, p->dtype == dt, "parts must share a dtype");
        total += p->cols;
    }
    if (B < 0) { ensure(out, 0, 0, Dtype::FP32, op); return; }
    ensure(out, B, total, dt, op);
    long long off = 0;
    for (const Tensor* p : parts) {
        if (!p || p->cols == 0) continue;
        if (B > 0) {
            metal::copy_d2d_strided(*p, 0, p->cols, out, static_cast<int>(off),
                                    static_cast<int>(total), p->cols, B);
        }
        off += p->cols;
    }
}

void concat_nchw_channels(const std::vector<const Tensor*>& parts, int N, int H, int W,
                          const std::vector<int>& C_per_part, Tensor& out) {
    const char* op = "concat_nchw_channels";
    need(op, parts.size() == C_per_part.size(), "parts.size() != C_per_part.size()");
    need(op, !parts.empty() && parts[0], "no parts");
    const Dtype dt = parts[0]->dtype;
    long long total_C = 0;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        need(op, parts[i] != nullptr, "null part");
        need(op, parts[i]->dtype == dt, "parts must share a dtype");
        need(op, parts[i]->size() == static_cast<long long>(N) * C_per_part[i] * H * W,
             "part size mismatch (expected N*C_i*H*W)");
        total_C += C_per_part[i];
    }
    const long long HW = static_cast<long long>(H) * W;
    ensure(out, N, total_C * HW, dt, op);
    long long c_off = 0;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        const long long chunk = C_per_part[i] * HW;
        if (chunk > 0 && N > 0) {
            metal::copy_d2d_strided(*parts[i], 0, static_cast<int>(chunk),
                                    out, static_cast<int>(c_off * HW),
                                    static_cast<int>(total_C * HW),
                                    static_cast<int>(chunk), N);
        }
        c_off += C_per_part[i];
    }
}

void concat_nchw_channels_backward(const Tensor& dY, int N, int H, int W,
                                   const std::vector<int>& C_per_part,
                                   const std::vector<Tensor*>& parts) {
    const char* op = "concat_nchw_channels_backward";
    need(op, parts.size() == C_per_part.size(), "parts.size() != C_per_part.size()");
    long long total_C = 0;
    for (int c : C_per_part) total_C += c;
    const long long HW = static_cast<long long>(H) * W;
    need(op, dY.size() == N * total_C * HW, "dY size mismatch");
    long long c_off = 0;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        need(op, parts[i] != nullptr, "null part");
        const long long chunk = C_per_part[i] * HW;
        ensure(*parts[i], N, chunk, dY.dtype, op);
        if (chunk > 0 && N > 0) {
            metal::copy_d2d_strided(dY, static_cast<int>(c_off * HW),
                                    static_cast<int>(total_C * HW),
                                    *parts[i], 0, static_cast<int>(chunk),
                                    static_cast<int>(chunk), N);
        }
        c_off += C_per_part[i];
    }
}

void nchw_to_sequence(const Tensor& X, int N, int C, int H, int W, Tensor& Y) {
    if (N < 0 || C < 0 || H < 0 || W < 0) fail("nchw_to_sequence", "negative dimension");
    const int HW = H * W;
    if (Y.rows != N * HW || Y.cols != C || Y.dtype != X.dtype) {
        Y.resize(N * HW, C, X.dtype);
    }
    transpose_batched("nchw_to_sequence", X, Y, N, C, HW);
}

void sequence_to_nchw(const Tensor& X, int N, int C, int H, int W, Tensor& Y) {
    if (N < 0 || C < 0 || H < 0 || W < 0) fail("sequence_to_nchw", "negative dimension");
    const int HW = H * W;
    if (Y.rows != N || Y.cols != C * HW || Y.dtype != X.dtype) {
        Y.resize(N, C * HW, X.dtype);
    }
    transpose_batched("sequence_to_nchw", X, Y, N, HW, C);
}

}  // namespace brotensor::detail::metal
