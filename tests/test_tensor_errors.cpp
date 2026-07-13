// ─── CPU-only coverage for the error / edge paths of Tensor + the dispatcher ─
//
// tests/test_tensor.cpp is CUDA-gated and tests/test_dispatch.cpp covers the
// happy paths of the runtime policy. This file targets what neither reaches:
// the throw branches and degenerate cases of src/tensor.cpp and src/dispatch.cpp.
//
// Coverage:
//   1. dtype sizing helpers across every Dtype, including the GGUF block-quant
//      carriers, plus their two throw paths.
//   2. Negative dimensions rejected at allocation and at resize().
//   3. resize() on a non-owning view() throws instead of severing the view.
//   4. resize() capacity reuse (shrink / dtype-switch keep the pointer; growth
//      reallocates).
//   5. Empty / zero-size tensors: bytes, clone, zero, host round-trip.
//   6. Copy / move semantics, including self-assignment and self-move.
//   7. Host accessors against the wrong dtype and against a non-CPU tensor —
//      all of which throw std::runtime_error.
//   8. from_raw_bytes_on with a block-quant carrier + its nbytes mismatch throw.
//   9. dispatch: the vtable-lookup throws, the operand-consistency throw, the
//      all-uncommitted degenerate path, the optional-operand path, the operand
//      overflow guard, adopt_output, and a null CPU vtable slot surfacing as
//      "not implemented on CPU".
//
// Everything runs on Device::CPU. Two tests build a *non-owning* Tensor::view
// tagged Device::CUDA over host memory: view() never touches an allocator and
// the code under test (check_host / the dispatch operand check) throws on the
// device tag alone, so no CUDA backend is needed and none is invoked.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

using brotensor::Device;
using brotensor::Dtype;
using brotensor::Tensor;

static int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

// Run `fn`, reporting whether it threw std::runtime_error.
template <typename F>
static bool throws_runtime_error(F&& fn) {
    try {
        fn();
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

// Same, but also requires `needle` to appear in what().
template <typename F>
static bool throws_with(F&& fn, const char* needle) {
    try {
        fn();
    } catch (const std::runtime_error& e) {
        return std::strstr(e.what(), needle) != nullptr;
    }
    return false;
}

// ─── 1. dtype sizing helpers ───────────────────────────────────────────────
static void test_dtype_sizing() {
    std::printf("test_dtype_sizing\n");
    using brotensor::dtype_size_bytes;
    using brotensor::dtype_block_size;
    using brotensor::dtype_block_bytes;
    using brotensor::dtype_storage_bytes;
    using brotensor::dtype_is_quant;

    // Arithmetic / storage dtypes: element-addressable, block size 1.
    struct Plain { Dtype dt; int size; };
    const Plain plain[] = {
        {Dtype::FP32, 4}, {Dtype::FP16, 2}, {Dtype::BF16, 2},
        {Dtype::INT8, 1}, {Dtype::INT32, 4}, {Dtype::F64, 8},
    };
    for (const Plain& p : plain) {
        CHECK(dtype_size_bytes(p.dt) == p.size);
        CHECK(dtype_block_size(p.dt) == 1);
        CHECK(dtype_block_bytes(p.dt) == p.size);
        CHECK(!dtype_is_quant(p.dt));
        CHECK(dtype_storage_bytes(p.dt, 10) ==
              static_cast<std::size_t>(10 * p.size));
        CHECK(dtype_storage_bytes(p.dt, 0) == 0u);
    }

    // GGUF block-quant carriers: no element size, fixed block size + bytes.
    struct Quant { Dtype dt; int block; int bytes; };
    const Quant quants[] = {
        {Dtype::Q4_0, 32, 18},  {Dtype::Q4_1, 32, 20},
        {Dtype::Q5_0, 32, 22},  {Dtype::Q5_1, 32, 24},
        {Dtype::Q8_0, 32, 34},  {Dtype::Q8_1, 32, 36},
        {Dtype::Q2_K, 256, 82}, {Dtype::Q3_K, 256, 110},
        {Dtype::Q4_K, 256, 144},{Dtype::Q5_K, 256, 176},
        {Dtype::Q6_K, 256, 210},{Dtype::Q8_K, 256, 292},
    };
    for (const Quant& q : quants) {
        CHECK(dtype_is_quant(q.dt));
        CHECK(dtype_size_bytes(q.dt) == 0);   // not element-addressable
        CHECK(dtype_block_size(q.dt) == q.block);
        CHECK(dtype_block_bytes(q.dt) == q.bytes);
        // Exactly three blocks.
        CHECK(dtype_storage_bytes(q.dt, 3 * q.block) ==
              static_cast<std::size_t>(3 * q.bytes));
        // Zero elements is zero blocks, not an error.
        CHECK(dtype_storage_bytes(q.dt, 0) == 0u);
        // A partial block is rejected.
        const Dtype dt = q.dt;
        const int block = q.block;
        CHECK(throws_with([&] { dtype_storage_bytes(dt, block + 1); },
                          "block size"));
    }

    // Negative element counts are rejected for every dtype.
    CHECK(throws_with([] { dtype_storage_bytes(Dtype::FP32, -1); },
                      "negative numel"));
    CHECK(throws_runtime_error([] { dtype_storage_bytes(Dtype::Q4_K, -256); }));

    // device_name covers every Device.
    CHECK(std::strcmp(brotensor::device_name(Device::CPU), "CPU") == 0);
    CHECK(std::strcmp(brotensor::device_name(Device::CUDA), "CUDA") == 0);
    CHECK(std::strcmp(brotensor::device_name(Device::Metal), "Metal") == 0);
}

// ─── 2. negative dimensions ────────────────────────────────────────────────
static void test_negative_dims() {
    std::printf("test_negative_dims\n");
    CHECK(throws_with([] { (void)Tensor::empty_on(Device::CPU, -1, 4); },
                      "negative dimension"));
    CHECK(throws_runtime_error([] { (void)Tensor::zeros_on(Device::CPU, 4, -1); }));
    CHECK(throws_runtime_error([] { (void)Tensor::empty(-2, 2); }));
    CHECK(throws_runtime_error([] { (void)Tensor::zeros(2, -2); }));
    CHECK(throws_runtime_error([] { (void)Tensor::zeros_on(Device::CPU, -3, -3); }));

    // resize() rejects the negative dim before touching the tensor.
    Tensor t = Tensor::zeros_on(Device::CPU, 2, 3);
    void* before = t.data;
    CHECK(throws_with([&] { t.resize(-1, 3); }, "negative dimension"));
    CHECK(t.rows == 2 && t.cols == 3);
    CHECK(t.data == before);
    CHECK(throws_runtime_error([&] { t.resize(3, -1); }));
    CHECK(t.rows == 2 && t.cols == 3);
}

// ─── 3. resize() on a non-owning view ──────────────────────────────────────
static void test_resize_view_throws() {
    std::printf("test_resize_view_throws\n");
    Tensor owner = Tensor::zeros_on(Device::CPU, 4, 4);
    owner.host_f32_mut()[5] = 3.5f;
    {
        Tensor v = Tensor::view(Device::CPU, owner.data, 4, 4);
        CHECK(v.data == owner.data);
        CHECK(v.host_f32()[5] == 3.5f);

        // Reshaping would sever the view — rejected.
        CHECK(throws_with([&] { v.resize(2, 8); }, "non-owning view"));
        CHECK(v.data == owner.data);
        CHECK(v.rows == 4 && v.cols == 4);

        // Same shape but a different dtype also reaches the view check.
        CHECK(throws_runtime_error([&] { v.resize(4, 4, Dtype::FP16); }));
        CHECK(v.dtype == Dtype::FP32);

        // Resizing to the shape+dtype it already has short-circuits before the
        // view check — a documented no-op, not an error.
        CHECK(!throws_runtime_error([&] { v.resize(4, 4, Dtype::FP32); }));
        CHECK(v.data == owner.data);

        // A view goes out of scope here without freeing owner's storage.
    }
    CHECK(owner.data != nullptr);
    CHECK(owner.host_f32()[5] == 3.5f);

    // A default-constructed tensor is owns_ == false but data == nullptr — it
    // is NOT a view and resizes normally.
    Tensor fresh;
    CHECK(!throws_runtime_error([&] { fresh.resize(2, 2); }));
    CHECK(fresh.data != nullptr && fresh.rows == 2 && fresh.cols == 2);
}

// ─── 4. resize() capacity reuse ────────────────────────────────────────────
static void test_resize_capacity() {
    std::printf("test_resize_capacity\n");
    Tensor t = Tensor::zeros_on(Device::CPU, 8, 8);       // 256 bytes
    void* p = t.data;
    CHECK(t.bytes() == 256u);

    t.resize(2, 2);                                       // 16 bytes — fits
    CHECK(t.data == p);
    CHECK(t.bytes() == 16u && t.rows == 2 && t.cols == 2);

    t.resize(4, 4, Dtype::FP16);                          // 32 bytes — fits
    CHECK(t.data == p);
    CHECK(t.dtype == Dtype::FP16 && t.bytes() == 32u);

    t.resize(64, 64, Dtype::FP32);                        // grows past capacity
    CHECK(t.data != nullptr);
    CHECK(t.rows == 64 && t.cols == 64 && t.bytes() == 16384u);
    t.zero();
    CHECK(t.host_f32()[0] == 0.0f);

    // Shrinking to nothing leaves a valid, empty tensor.
    t.resize(0, 0);
    CHECK(t.size() == 0 && t.bytes() == 0u);
    CHECK(t.empty());

    // Device is preserved across resize.
    CHECK(t.device == Device::CPU);
}

// ─── 5. empty / zero-size tensors ──────────────────────────────────────────
static void test_empty_tensors() {
    std::printf("test_empty_tensors\n");
    Tensor t;                                    // default-constructed
    CHECK(t.data == nullptr);
    CHECK(t.rows == 0 && t.cols == 0 && t.size() == 0);
    CHECK(t.bytes() == 0u);
    CHECK(t.empty());
    CHECK(t.device == Device::CPU && t.dtype == Dtype::FP32);
    t.zero();                                    // no-op, must not crash

    // A zero-element allocation yields a null pointer, not a 0-byte block.
    Tensor z = Tensor::zeros_on(Device::CPU, 0, 5);
    CHECK(z.data == nullptr);
    CHECK(z.rows == 0 && z.cols == 5);
    CHECK(z.bytes() == 0u);
    CHECK(z.empty());

    // clone() of an empty tensor is an empty tensor.
    Tensor c = z.clone();
    CHECK(c.data == nullptr && c.rows == 0 && c.cols == 5);
    Tensor c2 = t.clone();
    CHECK(c2.data == nullptr && c2.empty());

    // to() on the same device routes through clone().
    Tensor same = z.to(Device::CPU);
    CHECK(same.device == Device::CPU && same.empty());

    // Host round-trip of an empty tensor is an empty vector / a no-op copy.
    CHECK(z.to_host_vector().empty());
    float sink = 42.0f;
    z.copy_to_host(&sink);
    CHECK(sink == 42.0f);                        // nothing was written

    Tensor z16 = Tensor::zeros_on(Device::CPU, 0, 4, Dtype::FP16);
    CHECK(z16.to_host_vector_fp16().empty());
    Tensor zbf = Tensor::zeros_on(Device::CPU, 0, 4, Dtype::BF16);
    CHECK(zbf.to_host_vector_bf16().empty());
    uint16_t sink16 = 7;
    z16.copy_to_host_fp16(&sink16);
    zbf.copy_to_host_bf16(&sink16);
    CHECK(sink16 == 7);
}

// ─── 6. copy / move semantics ──────────────────────────────────────────────
static void test_copy_move() {
    std::printf("test_copy_move\n");
    const float src[4] = {1.0f, -2.0f, 3.0f, -4.0f};
    Tensor a = Tensor::from_host_on(Device::CPU, src, 2, 2);

    // Copy ctor: deep copy, distinct storage.
    Tensor b(a);
    CHECK(b.data != nullptr && b.data != a.data);
    CHECK(b.rows == 2 && b.cols == 2 && b.dtype == Dtype::FP32);
    for (int i = 0; i < 4; ++i) CHECK(b.host_f32()[i] == src[i]);

    // Copy assignment: deep copy over an existing allocation.
    Tensor d = Tensor::zeros_on(Device::CPU, 7, 7);
    d = a;
    CHECK(d.rows == 2 && d.cols == 2);
    CHECK(d.data != a.data);
    for (int i = 0; i < 4; ++i) CHECK(d.host_f32()[i] == src[i]);

    // Self copy-assignment is a no-op (routed through a pointer so the
    // compiler's self-assign diagnostics stay quiet).
    Tensor* pa = &a;
    a = *pa;
    CHECK(a.data != nullptr && a.rows == 2 && a.cols == 2);
    for (int i = 0; i < 4; ++i) CHECK(a.host_f32()[i] == src[i]);

    // Self move-assignment is likewise guarded and leaves the tensor intact.
    a = std::move(*pa);
    CHECK(a.data != nullptr && a.rows == 2 && a.cols == 2);
    for (int i = 0; i < 4; ++i) CHECK(a.host_f32()[i] == src[i]);

    // Move ctor transfers ownership and clears the source.
    void* moved_from = a.data;
    Tensor m(std::move(a));
    CHECK(m.data == moved_from);
    CHECK(m.rows == 2 && m.cols == 2);
    CHECK(a.data == nullptr && a.rows == 0 && a.cols == 0);  // NOLINT

    // Move assignment releases the destination's old storage first.
    Tensor dst = Tensor::zeros_on(Device::CPU, 3, 3);
    dst = std::move(m);
    CHECK(dst.data == moved_from);
    CHECK(dst.rows == 2 && dst.cols == 2);
    CHECK(m.data == nullptr && m.rows == 0);                 // NOLINT
    for (int i = 0; i < 4; ++i) CHECK(dst.host_f32()[i] == src[i]);

    // Copying a non-FP32 tensor preserves the dtype.
    Tensor q = Tensor::zeros_on(Device::CPU, 2, 3, Dtype::BF16);
    q.host_bf16_mut()[4] = brotensor::fp32_to_bf16_bits(1.5f);
    Tensor qc = q;
    CHECK(qc.dtype == Dtype::BF16 && qc.bytes() == 12u);
    CHECK(brotensor::bf16_bits_to_fp32(qc.host_bf16()[4]) == 1.5f);
}

// ─── 7a. host accessors against the wrong dtype ────────────────────────────
static void test_host_accessor_dtype_errors() {
    std::printf("test_host_accessor_dtype_errors\n");
    Tensor f32 = Tensor::zeros_on(Device::CPU, 2, 2, Dtype::FP32);
    Tensor f16 = Tensor::zeros_on(Device::CPU, 2, 2, Dtype::FP16);
    Tensor bf16 = Tensor::zeros_on(Device::CPU, 2, 2, Dtype::BF16);
    Tensor i8 = Tensor::zeros_on(Device::CPU, 2, 2, Dtype::INT8);

    const Tensor& cf16 = f16;
    const Tensor& cf32 = f32;

    // FP32 accessors on a 16-bit tensor.
    CHECK(throws_with([&] { (void)f16.host_f32_mut(); }, "dtype mismatch"));
    CHECK(throws_runtime_error([&] { (void)cf16.host_f32(); }));
    CHECK(throws_runtime_error([&] { (void)f16.at(0, 0); }));
    CHECK(throws_runtime_error([&] { (void)cf16.at(0, 0); }));
    CHECK(throws_with([&] { (void)f16.to_host_vector(); }, "not FP32"));
    CHECK(throws_runtime_error([&] {
        float dst[4];
        f16.copy_to_host(dst);
    }));

    // FP16 accessors on other dtypes.
    CHECK(throws_runtime_error([&] { (void)f32.host_fp16_mut(); }));
    CHECK(throws_runtime_error([&] { (void)cf32.host_fp16(); }));
    CHECK(throws_runtime_error([&] { (void)bf16.host_fp16_mut(); }));
    CHECK(throws_with([&] { (void)f32.to_host_vector_fp16(); }, "not FP16"));
    CHECK(throws_runtime_error([&] {
        uint16_t dst[4];
        f32.copy_to_host_fp16(dst);
    }));

    // BF16 accessors on other dtypes.
    CHECK(throws_runtime_error([&] { (void)f32.host_bf16_mut(); }));
    CHECK(throws_runtime_error([&] { (void)f16.host_bf16_mut(); }));
    CHECK(throws_with([&] { (void)f16.to_host_vector_bf16(); }, "not BF16"));
    CHECK(throws_runtime_error([&] {
        uint16_t dst[4];
        f16.copy_to_host_bf16(dst);
    }));

    // at() bounds checks (both overloads).
    CHECK(throws_with([&] { (void)f32.at(-1, 0); }, "out of range"));
    CHECK(throws_runtime_error([&] { (void)f32.at(0, -1); }));
    CHECK(throws_runtime_error([&] { (void)f32.at(2, 0); }));
    CHECK(throws_runtime_error([&] { (void)f32.at(0, 2); }));
    CHECK(throws_runtime_error([&] { (void)cf32.at(2, 2); }));

    // host_raw is dtype-agnostic — no throw for an INT8 carrier.
    CHECK(!throws_runtime_error([&] { (void)i8.host_raw_mut(); }));
    const Tensor& ci8 = i8;
    CHECK(!throws_runtime_error([&] { (void)ci8.host_raw(); }));
    CHECK(i8.bytes() == 4u);

    // In-range at() works and aliases the buffer.
    f32.at(1, 1) = 9.0f;
    CHECK(cf32.at(1, 1) == 9.0f);
    CHECK(f32[3] == 9.0f);
    CHECK(f32(1, 1) == 9.0f);
    CHECK(f32.ptr()[3] == 9.0f);
}

// ─── 7b. host accessors against a non-CPU tensor ───────────────────────────
//
// A Tensor::view() tagged Device::CUDA over host memory. view() never calls an
// allocator and the destructor does not free a non-owning tensor, so this is
// pure bookkeeping — it exercises check_host()'s throw branch without a GPU.
static void test_host_accessor_device_errors() {
    std::printf("test_host_accessor_device_errors\n");
    std::vector<float> backing(4, 0.0f);
    Tensor fake = Tensor::view(Device::CUDA, backing.data(), 2, 2);
    CHECK(fake.device == Device::CUDA);
    CHECK(!fake.is_host());

    const Tensor& cfake = fake;
    CHECK(throws_with([&] { (void)fake.host_f32_mut(); }, "not CPU"));
    CHECK(throws_runtime_error([&] { (void)cfake.host_f32(); }));
    CHECK(throws_runtime_error([&] { (void)fake.host_fp16_mut(); }));
    CHECK(throws_runtime_error([&] { (void)cfake.host_fp16(); }));
    CHECK(throws_runtime_error([&] { (void)fake.host_bf16_mut(); }));
    CHECK(throws_runtime_error([&] { (void)cfake.host_bf16(); }));
    CHECK(throws_runtime_error([&] { (void)fake.host_raw_mut(); }));
    CHECK(throws_runtime_error([&] { (void)cfake.host_raw(); }));
    CHECK(throws_runtime_error([&] { (void)fake.at(0, 0); }));
    CHECK(throws_runtime_error([&] { (void)cfake.at(0, 0); }));
}

// ─── 8. from_raw_bytes_on + the block-quant carriers ───────────────────────
static void test_raw_bytes_and_quant_storage() {
    std::printf("test_raw_bytes_and_quant_storage\n");
    // (2, 256) Q4_K == 512 elements == 2 blocks == 288 bytes.
    const int rows = 2, cols = 256;
    const std::size_t nbytes = 2u * 144u;
    std::vector<unsigned char> raw(nbytes);
    for (std::size_t i = 0; i < nbytes; ++i) {
        raw[i] = static_cast<unsigned char>(i & 0xFFu);
    }

    Tensor q = Tensor::from_raw_bytes_on(Device::CPU, raw.data(), rows, cols,
                                         Dtype::Q4_K, nbytes);
    CHECK(q.dtype == Dtype::Q4_K);
    CHECK(q.rows == rows && q.cols == cols);
    CHECK(q.bytes() == nbytes);
    CHECK(q.data != nullptr);
    CHECK(std::memcmp(q.host_raw(), raw.data(), nbytes) == 0);

    // clone() of a quant carrier copies bytes(), not elements.
    Tensor qc = q.clone();
    CHECK(qc.dtype == Dtype::Q4_K && qc.bytes() == nbytes);
    CHECK(qc.data != q.data);
    CHECK(std::memcmp(qc.host_raw(), raw.data(), nbytes) == 0);

    // A byte count that disagrees with the shape/dtype is rejected.
    CHECK(throws_with([&] {
        (void)Tensor::from_raw_bytes_on(Device::CPU, raw.data(), rows, cols,
                                        Dtype::Q4_K, nbytes + 1);
    }, "nbytes"));

    // Zero-byte raw upload is a valid no-op.
    Tensor qz = Tensor::from_raw_bytes_on(Device::CPU, raw.data(), 0, cols,
                                          Dtype::Q4_K, 0);
    CHECK(qz.data == nullptr && qz.bytes() == 0u);

    // Allocating a quant tensor whose element count is not a whole number of
    // blocks fails at the bytes() computation.
    CHECK(throws_with([] {
        (void)Tensor::zeros_on(Device::CPU, 1, 100, Dtype::Q4_K);
    }, "block size"));

    // A whole number of blocks allocates fine and zeroes.
    Tensor ok = Tensor::zeros_on(Device::CPU, 1, 256, Dtype::Q8_0);
    CHECK(ok.bytes() == (256u / 32u) * 34u);
    CHECK(ok.data != nullptr);
    const unsigned char* p = static_cast<const unsigned char*>(ok.host_raw());
    for (std::size_t i = 0; i < ok.bytes(); ++i) CHECK(p[i] == 0u);
}

// ─── 9a. dispatch: registration + vtable lookup ────────────────────────────
static void test_dispatch_registration() {
    std::printf("test_dispatch_registration\n");
    namespace d = brotensor::detail;

    CHECK(d::is_registered(Device::CPU));
    CHECK(!throws_runtime_error([] { (void)&d::ops_for(Device::CPU); }));
    CHECK(!throws_runtime_error([] { (void)&d::alloc_for(Device::CPU); }));

    // An unregistered backend throws on lookup. Skipped when the backend is
    // actually present (a CUDA/Metal build).
    if (!d::is_registered(Device::CUDA)) {
        CHECK(throws_with([] { (void)&d::ops_for(Device::CUDA); },
                          "not registered"));
        CHECK(throws_runtime_error([] { (void)&d::alloc_for(Device::CUDA); }));
    } else {
        std::printf("  CUDA registered - skipping unregistered-lookup case\n");
    }
    if (!d::is_registered(Device::Metal)) {
        CHECK(throws_with([] { (void)&d::ops_for(Device::Metal); },
                          "not registered"));
    } else {
        std::printf("  Metal registered - skipping unregistered-lookup case\n");
    }

    // throw_not_implemented builds a "<op>: not implemented on <device>" error.
    CHECK(throws_with([] { d::throw_not_implemented("some_op", Device::CPU); },
                      "not implemented on CPU"));
}

// ─── 9b. dispatch: operand resolution ──────────────────────────────────────
static void test_dispatch_resolution() {
    std::printf("test_dispatch_resolution\n");
    namespace d = brotensor::detail;

    const d::OpsVTable* cpu = &d::ops_for(Device::CPU);

    Tensor a = Tensor::zeros_on(Device::CPU, 1, 1);
    Tensor b = Tensor::zeros_on(Device::CPU, 1, 1);
    Tensor c = Tensor::zeros_on(Device::CPU, 1, 1);
    Tensor e = Tensor::zeros_on(Device::CPU, 1, 1);
    Tensor f = Tensor::zeros_on(Device::CPU, 1, 1);
    Tensor g = Tensor::zeros_on(Device::CPU, 1, 1);
    Tensor h = Tensor::zeros_on(Device::CPU, 1, 1);
    Tensor i = Tensor::zeros_on(Device::CPU, 1, 1);

    // All eight arity overloads resolve to the CPU table.
    CHECK(&d::dispatch(a) == cpu);
    CHECK(&d::dispatch(a, b) == cpu);
    CHECK(&d::dispatch(a, b, c) == cpu);
    CHECK(&d::dispatch(a, b, c, e) == cpu);
    CHECK(&d::dispatch(a, b, c, e, f) == cpu);
    CHECK(&d::dispatch(a, b, c, e, f, g) == cpu);
    CHECK(&d::dispatch(a, b, c, e, f, g, h) == cpu);
    CHECK(&d::dispatch(a, b, c, e, f, g, h, i) == cpu);

    // An uncommitted (data == nullptr) operand is a wildcard: it is skipped by
    // the consistency check and does not pin the device.
    Tensor out;                                   // uncommitted
    CHECK(&d::dispatch(a, out) == cpu);
    CHECK(&d::dispatch(out, a) == cpu);

    // Every operand uncommitted: degenerate but legal — operand 0's tag wins.
    Tensor u1, u2;
    CHECK(&d::dispatch(u1, u2) == cpu);

    // Optional-operand form: null pointers are skipped.
    CHECK(&d::dispatch_with_opts(a, {nullptr}) == cpu);
    CHECK(&d::dispatch_with_opts(a, {nullptr, &b}) == cpu);
    CHECK(&d::dispatch_with_opts(a, b, {nullptr, &c}) == cpu);
    // ... and the no-committed-operand path through the same helper.
    CHECK(&d::dispatch_with_opts(u1, {nullptr, &u2}) == cpu);
    CHECK(&d::dispatch_with_opts(u1, u2, {nullptr}) == cpu);

    // The fixed operand buffer holds 32; 1 + 32 optionals overflows it.
    CHECK(throws_with([&] {
        (void)&d::dispatch_with_opts(a, {
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr});
    }, "exceeds the fixed dispatch buffer"));

    // adopt_output pins an uncommitted output; a committed one keeps its tag.
    Tensor fresh;
    d::adopt_output(fresh, Device::CUDA);
    CHECK(fresh.device == Device::CUDA);          // tag only — still no storage
    CHECK(fresh.data == nullptr);
    d::adopt_output(fresh, Device::CPU);
    CHECK(fresh.device == Device::CPU);
    d::adopt_output(a, Device::CUDA);             // a is committed on CPU
    CHECK(a.device == Device::CPU);
}

// ─── 9c. dispatch: operand device mismatch ─────────────────────────────────
//
// dispatch() only *resolves* a vtable — it never invokes a backend — so a
// non-owning CUDA-tagged view over host memory is enough to drive the
// consistency check, with no GPU present and no GPU code executed. The CPU
// operand comes first, so the op resolves to CPU and the view is the offender.
static void test_dispatch_device_mismatch() {
    std::printf("test_dispatch_device_mismatch\n");
    namespace d = brotensor::detail;

    std::vector<float> backing(4, 0.0f);
    Tensor cpu_t = Tensor::zeros_on(Device::CPU, 2, 2);
    Tensor other = Tensor::view(Device::CUDA, backing.data(), 2, 2);

    CHECK(throws_with([&] { (void)&d::dispatch(cpu_t, other); },
                      "but the op resolved to CPU"));
    CHECK(throws_with([&] { (void)&d::dispatch(cpu_t, cpu_t, other); },
                      "operand 2"));
    CHECK(throws_runtime_error([&] {
        (void)&d::dispatch_with_opts(cpu_t, {&other});
    }));
    CHECK(throws_runtime_error([&] {
        (void)&d::dispatch_with_opts(cpu_t, cpu_t, {nullptr, &other});
    }));
}

// ─── 9d. a null CPU vtable slot surfaces as "not implemented on CPU" ───────
static void test_null_cpu_slot_throws() {
    std::printf("test_null_cpu_slot_throws\n");
    // The CPU backend deliberately leaves the GGUF-quant / W8A16 slots null.
    Tensor w_q4k = Tensor::zeros_on(Device::CPU, 1, 256, Dtype::Q4_K);
    Tensor w_fp16 = Tensor::zeros_on(Device::CPU, 1, 256, Dtype::FP16);
    CHECK(throws_with([&] { brotensor::dequant_q4k_to_fp16(w_q4k, w_fp16); },
                      "dequant_q4k_to_fp16: not implemented on CPU"));

    Tensor w_i8 = Tensor::zeros_on(Device::CPU, 4, 4, Dtype::INT8);
    Tensor scales = Tensor::zeros_on(Device::CPU, 4, 1, Dtype::FP32);
    Tensor x = Tensor::zeros_on(Device::CPU, 4, 2, Dtype::FP16);
    Tensor y = Tensor::zeros_on(Device::CPU, 4, 2, Dtype::FP16);
    CHECK(throws_with([&] {
        brotensor::matmul_int8w_fp16(w_i8, scales, x, y);
    }, "matmul_int8w_fp16: not implemented on CPU"));
}

int main() {
    brotensor::init();
    brotensor::DeviceScope cpu_only(Device::CPU);
    std::printf("test_tensor_errors\n");

    test_dtype_sizing();
    test_negative_dims();
    test_resize_view_throws();
    test_resize_capacity();
    test_empty_tensors();
    test_copy_move();
    test_host_accessor_dtype_errors();
    test_host_accessor_device_errors();
    test_raw_bytes_and_quant_storage();
    test_dispatch_registration();
    test_dispatch_resolution();
    test_dispatch_device_mismatch();
    test_null_cpu_slot_throws();

    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nAll tensor/dispatch error-path checks passed.\n");
    return 0;
}
