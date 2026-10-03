// Vulkan data-movement ops: cast, contiguous and strided copies, row
// concat / split, and the NCHW <-> sequence transposes. Contracts follow the
// CUDA backend (src/cuda/concat.cu, elementwise.cu, transpose.cu): offsets
// and pitches are in elements of the source dtype, outputs are resized to the
// input's dtype. Kernels: shaders/cast.comp, copy2d.comp, transpose.comp;
// plain copies are vkCmdCopyBuffer through the stream.

#include "detail/kernels.h"

#include <brotensor/detail/dispatch.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor::detail::vulkan {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

namespace {

// Device-to-device byte copy between two addresses of one device.
void copy_bytes(DeviceCtx& d, std::uint64_t src, std::uint64_t dst, std::size_t bytes) {
    if (bytes == 0 || src == dst) return;
    const Span a = d.allocator().resolve(src, bytes);
    const Span b = d.allocator().resolve(dst, bytes);
    d.stream().copy(a.buf, a.offset, b.buf, b.offset, bytes);
}

std::size_t elem_bytes(Dtype dt, const char* op) {
    const int e = ::brotensor::dtype_size_bytes(dt);
    if (e <= 0) {
        throw std::runtime_error(std::string("brotensor: ") + op +
                                 ": not defined for block-quantised dtypes");
    }
    return static_cast<std::size_t>(e);
}

ShaderId cast_shader(Dtype s, Dtype t) {
    const int a = dt_code(s, "cast"), b = dt_code(t, "cast");
    // shaders.cmake order: f32->f16, f32->bf16, f16->f32, f16->bf16, bf16->f32, bf16->f16
    static const ShaderId table[3][3] = {
        {ShaderId::kCount, ShaderId::cast_f32_to_f16, ShaderId::cast_f32_to_bf16},
        {ShaderId::cast_f16_to_f32, ShaderId::kCount, ShaderId::cast_f16_to_bf16},
        {ShaderId::cast_bf16_to_f32, ShaderId::cast_bf16_to_f16, ShaderId::kCount},
    };
    return table[a][b];
}

struct CastPush { std::uint64_t src, dst; std::uint32_t n; };
struct Copy2dPush { std::uint64_t src, dst; std::uint32_t src_pitch, dst_pitch, width, height; };
struct TransposePush { std::uint64_t src, dst; std::uint32_t batch, rows, cols; };

// out[b][c][r] = in[b][r][c] for in of shape (batch, rows, cols).
void transpose(const char* op, const Tensor& X, Tensor& Y, int batch, int rows, int cols) {
    const std::size_t es = elem_bytes(X.dtype, op);
    if (es != 2 && es != 4) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": FP32 / FP16 / BF16 only on Vulkan");
    }
    if (batch == 0 || rows == 0 || cols == 0) return;
    if (static_cast<long long>(batch) * rows * cols > X.size()) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": X is smaller than N*C*H*W");
    }
    DeviceCtx& d = device_of(X);
    const Kernel& k = d.pipelines().get(es == 4 ? ShaderId::transpose_b4 : ShaderId::transpose_b2);
    const TransposePush pc{addr(X.data), addr(Y.data), static_cast<std::uint32_t>(batch),
                           static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(cols)};
    const std::uint32_t gx = std::min<std::uint32_t>((pc.cols + 31) / 32, 65535);
    const std::uint32_t gy = std::min<std::uint32_t>((pc.rows + 31) / 32, 65535);
    const std::uint32_t gz = std::min<std::uint32_t>(pc.batch, 65535);
    launch(d, k, pc, gx, gy, gz);
}

}  // namespace

void cast(const Tensor& src, Tensor& dst, Dtype out_dtype) {
    if (dst.rows != src.rows || dst.cols != src.cols || dst.dtype != out_dtype) {
        dst.resize(src.rows, src.cols, out_dtype);
    }
    const std::uint32_t n = count32(src, "cast");
    if (n == 0) return;
    DeviceCtx& d = device_of(src);
    if (src.dtype == out_dtype) {
        copy_bytes(d, addr(src.data), addr(dst.data), src.bytes());
        return;
    }
    const ShaderId id = cast_shader(src.dtype, out_dtype);
    const Kernel& k = d.pipelines().get(id);
    const CastPush pc{addr(src.data), addr(dst.data), n};
    launch(d, k, pc, groups_1d(n, k));
}

void copy_d2d(const Tensor& src, int src_off, Tensor& dst, int dst_off, int n) {
    if (n <= 0) return;
    const std::size_t e = elem_bytes(src.dtype, "copy_d2d");
    copy_bytes(device_of(src), addr(src.data) + e * static_cast<std::size_t>(src_off),
               addr(dst.data) + e * static_cast<std::size_t>(dst_off), e * static_cast<std::size_t>(n));
}

void copy_d2d_strided(const Tensor& src, int src_off, int src_pitch, Tensor& dst, int dst_off,
                      int dst_pitch, int width, int height) {
    if (width <= 0 || height <= 0) return;
    std::size_t e = elem_bytes(src.dtype, "copy_d2d_strided");
    DeviceCtx& d = device_of(src);
    const std::uint64_t s0 = addr(src.data) + e * static_cast<std::size_t>(src_off);
    const std::uint64_t d0 = addr(dst.data) + e * static_cast<std::size_t>(dst_off);
    if ((src_pitch == width && dst_pitch == width) || height == 1) {
        copy_bytes(d, s0, d0, e * static_cast<std::size_t>(width) * height);
        return;
    }
    // Every touched range must be inside its tensor's allocation.
    const std::size_t span_s = e * (static_cast<std::size_t>(src_pitch) * (height - 1) + width);
    const std::size_t span_d = e * (static_cast<std::size_t>(dst_pitch) * (height - 1) + width);
    d.allocator().resolve(s0, span_s);
    d.allocator().resolve(d0, span_d);
    std::uint32_t mult = 1;
    if (e == 8) { mult = 2; e = 4; }   // F64 moves as pairs of 32-bit words
    const ShaderId id = e == 4 ? ShaderId::copy2d_b4 : (e == 2 ? ShaderId::copy2d_b2 : ShaderId::copy2d_b1);
    const Kernel& k = d.pipelines().get(id);
    const Copy2dPush pc{s0, d0, static_cast<std::uint32_t>(src_pitch) * mult,
                        static_cast<std::uint32_t>(dst_pitch) * mult,
                        static_cast<std::uint32_t>(width) * mult, static_cast<std::uint32_t>(height)};
    launch(d, k, pc, groups_1d(pc.width, k, 64), std::min<std::uint32_t>(pc.height, 65535));
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
    DeviceCtx& d = device_of(out);
    std::size_t off = 0;
    for (const Tensor* p : parts) {
        if (!p || p->size() == 0) continue;
        if (p->dtype != dt) throw std::runtime_error("brotensor: concat_rows: parts differ in dtype");
        const std::size_t bytes = e * static_cast<std::size_t>(p->size());
        copy_bytes(d, addr(p->data), addr(out.data) + off, bytes);
        off += bytes;
    }
}

void split_rows(const Tensor& in, const std::vector<Tensor*>& parts) {
    const std::size_t e = elem_bytes(in.dtype, "split_rows");
    DeviceCtx& d = device_of(in);
    std::size_t off = 0;
    for (Tensor* p : parts) {
        if (!p || p->size() == 0) continue;
        const std::size_t bytes = e * static_cast<std::size_t>(p->size());
        copy_bytes(d, addr(in.data) + off, addr(p->data), bytes);
        off += bytes;
    }
}

void nchw_to_sequence(const Tensor& X, int N, int C, int H, int W, Tensor& Y) {
    if (N < 0 || C < 0 || H < 0 || W < 0) throw std::runtime_error("nchw_to_sequence: negative dimension");
    const int HW = H * W;
    if (Y.rows != N * HW || Y.cols != C || Y.dtype != X.dtype) Y.resize(N * HW, C, X.dtype);
    transpose("nchw_to_sequence", X, Y, N, C, HW);       // (C, HW) -> (HW, C) per image
}

void sequence_to_nchw(const Tensor& X, int N, int C, int H, int W, Tensor& Y) {
    if (N < 0 || C < 0 || H < 0 || W < 0) throw std::runtime_error("sequence_to_nchw: negative dimension");
    const int HW = H * W;
    if (Y.rows != N || Y.cols != C * HW || Y.dtype != X.dtype) Y.resize(N, C * HW, X.dtype);
    transpose("sequence_to_nchw", X, Y, N, HW, C);       // (HW, C) -> (C, HW) per image
}

void fill_vulkan_vtable_copy(::brotensor::detail::OpsVTable& v) {
    v.cast = &cast;
    v.copy_d2d = &copy_d2d;
    v.copy_d2d_strided = &copy_d2d_strided;
    v.concat_rows = &concat_rows;
    v.split_rows = &split_rows;
    v.nchw_to_sequence = &nchw_to_sequence;
    v.sequence_to_nchw = &sequence_to_nchw;
}

}  // namespace brotensor::detail::vulkan
