#include "kernel_disk_cache.h"

#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <thread>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

#if BROTENSOR_HAS_BRASS_JIT
#include <brass/codegen/kernel_jit.hpp>
#include <brass/core/diagnostics.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/smt_guard_elim.hpp>
#include <brass/mir/superopt.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/target/aot_linker.hpp>
#include <brass/target/target.hpp>
#endif

namespace brotensor::jit {

namespace {

constexpr uint32_t kBtkcMagic   = 0x434B5442u; // "BTKC" in little-endian
constexpr uint32_t kBtkcVersion = 1u;
constexpr std::string_view kBrassAbiStamp = "brass_aot_v1_btkc1";

struct BtkcHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t backend_kind;
    uint32_t reserved;
    uint64_t payload_size;
    uint64_t payload_fnv1a64;
};
static_assert(sizeof(BtkcHeader) == 32, "BtkcHeader must be 32 bytes");

uint64_t fnv1a_64(const void* data, size_t len, uint64_t seed = 14695981039346656037ULL) noexcept {
    const auto* bytes = static_cast<const uint8_t*>(data);
    uint64_t h = seed;
    for (size_t i = 0; i < len; ++i) {
        h ^= static_cast<uint64_t>(bytes[i]);
        h *= 1099511628211ULL;
    }
    return h;
}

std::pair<uint64_t, uint64_t> fnv1a_128(std::string_view s1, std::string_view s2, std::string_view s3) noexcept {
    uint64_t h1 = 14695981039346656037ULL;
    uint64_t h2 = 0x6C62272E07BB0142ULL;
    auto mix = [&](std::string_view sv) {
        for (unsigned char c : sv) {
            h1 ^= static_cast<uint64_t>(c);
            h1 *= 1099511628211ULL;
            h2 ^= static_cast<uint64_t>(c) ^ (h1 >> 32);
            h2 *= 0x100000001B3ULL;
        }
        // Delimiter byte
        h1 ^= 0xFFULL;
        h1 *= 1099511628211ULL;
        h2 ^= 0xFFULL;
        h2 *= 0x100000001B3ULL;
    };
    mix(s1);
    mix(s2);
    mix(s3);
    return {h1, h2};
}

int current_pid() noexcept {
#if defined(_WIN32)
    return _getpid();
#else
    return static_cast<int>(::getpid());
#endif
}

uint64_t next_temp_seq() noexcept {
    static std::atomic<uint64_t> seq{1};
    return seq.fetch_add(1, std::memory_order_relaxed);
}

std::string unique_tmp_suffix() {
    auto tid_hash = std::hash<std::thread::id>{}(std::this_thread::get_id());
    std::ostringstream oss;
    oss << ".tmp." << current_pid() << "." << std::hex << tid_hash << "." << std::dec << next_temp_seq();
    return oss.str();
}

bool write_file_atomic(const std::filesystem::path& target_path, const void* data, size_t size) {
    std::error_code ec;
    std::filesystem::create_directories(target_path.parent_path(), ec);
    if (ec) return false;

    std::filesystem::path tmp_path = target_path;
    tmp_path += unique_tmp_suffix();

    {
        std::ofstream ofs(tmp_path, std::ios::binary | std::ios::trunc);
        if (!ofs.is_open()) return false;
        if (size > 0) {
            ofs.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
        }
        ofs.flush();
        if (!ofs.good()) {
            ofs.close();
            std::filesystem::remove(tmp_path, ec);
            return false;
        }
    }

    std::filesystem::rename(tmp_path, target_path, ec);
    if (ec) {
        std::filesystem::remove(tmp_path, ec);
        return false;
    }
    return true;
}

const char* shared_lib_extension() noexcept {
#if defined(_WIN32)
    return ".dll";
#elif defined(__APPLE__)
    return ".dylib";
#else
    return ".so";
#endif
}

} // namespace

KernelDiskCache& KernelDiskCache::instance() {
    static KernelDiskCache cache;
    return cache;
}

bool KernelDiskCache::is_enabled() const noexcept {
    const char* disable = std::getenv("BRO_ML_CACHE_DISABLE");
    if (disable && (std::strcmp(disable, "1") == 0 || std::strcmp(disable, "true") == 0)) {
        return false;
    }
    return true;
}

std::filesystem::path KernelDiskCache::cache_dir() const {
    {
        std::lock_guard<std::mutex> lk(mutex_);
        if (!override_dir_.empty()) {
            return override_dir_;
        }
    }
    if (const char* env_dir = std::getenv("BRO_ML_CACHE_DIR")) {
        if (env_dir[0] != '\0') {
            return std::filesystem::path(env_dir);
        }
    }
    if (const char* xdg = std::getenv("XDG_CACHE_HOME")) {
        if (xdg[0] != '\0') {
            return std::filesystem::path(xdg) / "brotensor" / "kernels";
        }
    }
    if (const char* home = std::getenv("HOME")) {
        if (home[0] != '\0') {
            return std::filesystem::path(home) / ".cache" / "brotensor" / "kernels";
        }
    }
    std::error_code ec;
    auto tmp = std::filesystem::temp_directory_path(ec);
    if (ec || tmp.empty()) {
        tmp = "/tmp";
    }
    return tmp / "brotensor_kernel_cache";
}

void KernelDiskCache::set_cache_dir(std::filesystem::path dir) {
    std::lock_guard<std::mutex> lk(mutex_);
    override_dir_ = std::move(dir);
}

void KernelDiskCache::clear_cache_dir_override() {
    std::lock_guard<std::mutex> lk(mutex_);
    override_dir_.clear();
}

std::string_view KernelDiskCache::host_cpu_backend_tag() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
    return "cpu_x64";
#elif defined(__aarch64__) || defined(_M_ARM64)
    return "cpu_arm64";
#else
    return "cpu_generic";
#endif
}

std::string KernelDiskCache::compute_key(std::string_view backend_tag, std::string_view signature) const {
    auto [h1, h2] = fnv1a_128(backend_tag, signature, kBrassAbiStamp);
    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(16) << h1 << std::setw(16) << h2;
    return oss.str();
}

std::filesystem::path KernelDiskCache::entry_path_for_key(std::string_view key_hex) const {
    return cache_dir() / (std::string(key_hex) + ".btkc");
}

bool KernelDiskCache::store_raw(BackendKind kind, std::string_view backend_tag, std::string_view signature,
                                const void* data, size_t size_bytes) {
    if (!is_enabled()) return false;

    const std::string key = compute_key(backend_tag, signature);
    const std::filesystem::path path = entry_path_for_key(key);

    BtkcHeader hdr{};
    hdr.magic           = kBtkcMagic;
    hdr.version         = kBtkcVersion;
    hdr.backend_kind    = static_cast<uint32_t>(kind);
    hdr.reserved        = 0;
    hdr.payload_size    = static_cast<uint64_t>(size_bytes);
    hdr.payload_fnv1a64 = fnv1a_64(data, size_bytes);

    std::vector<uint8_t> buf(sizeof(BtkcHeader) + size_bytes);
    std::memcpy(buf.data(), &hdr, sizeof(BtkcHeader));
    if (size_bytes > 0 && data != nullptr) {
        std::memcpy(buf.data() + sizeof(BtkcHeader), data, size_bytes);
    }

    if (!write_file_atomic(path, buf.data(), buf.size())) {
        return false;
    }
    writes_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool KernelDiskCache::load_raw(BackendKind kind, std::string_view backend_tag, std::string_view signature,
                               std::vector<uint8_t>& out_bytes) {
    if (!is_enabled()) {
        misses_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    const std::string key = compute_key(backend_tag, signature);
    const std::filesystem::path path = entry_path_for_key(key);

    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec) {
        misses_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
        misses_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    const std::streamsize total_size = ifs.tellg();
    if (total_size < static_cast<std::streamsize>(sizeof(BtkcHeader))) {
        ifs.close();
        std::filesystem::remove(path, ec);
        corruptions_.fetch_add(1, std::memory_order_relaxed);
        misses_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    ifs.seekg(0, std::ios::beg);
    BtkcHeader hdr{};
    if (!ifs.read(reinterpret_cast<char*>(&hdr), sizeof(BtkcHeader))) {
        ifs.close();
        std::filesystem::remove(path, ec);
        corruptions_.fetch_add(1, std::memory_order_relaxed);
        misses_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    const uint64_t actual_payload_size = static_cast<uint64_t>(total_size) - sizeof(BtkcHeader);
    if (hdr.magic != kBtkcMagic ||
        hdr.version != kBtkcVersion ||
        hdr.backend_kind != static_cast<uint32_t>(kind) ||
        hdr.payload_size != actual_payload_size) {
        ifs.close();
        std::filesystem::remove(path, ec);
        corruptions_.fetch_add(1, std::memory_order_relaxed);
        misses_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    out_bytes.resize(static_cast<size_t>(hdr.payload_size));
    if (hdr.payload_size > 0) {
        if (!ifs.read(reinterpret_cast<char*>(out_bytes.data()), static_cast<std::streamsize>(hdr.payload_size))) {
            ifs.close();
            out_bytes.clear();
            std::filesystem::remove(path, ec);
            corruptions_.fetch_add(1, std::memory_order_relaxed);
            misses_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
    }

    const uint64_t calc_checksum = fnv1a_64(out_bytes.data(), out_bytes.size());
    if (calc_checksum != hdr.payload_fnv1a64) {
        ifs.close();
        out_bytes.clear();
        std::filesystem::remove(path, ec);
        corruptions_.fetch_add(1, std::memory_order_relaxed);
        misses_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    hits_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool KernelDiskCache::store_spirv(std::string_view signature, const std::vector<uint32_t>& words) {
    if (words.empty()) return false;
    return store_raw(BackendKind::VulkanSpirv, "spirv", signature,
                     words.data(), words.size() * sizeof(uint32_t));
}

bool KernelDiskCache::load_spirv(std::string_view signature, std::vector<uint32_t>& out_words) {
    std::vector<uint8_t> bytes;
    if (!load_raw(BackendKind::VulkanSpirv, "spirv", signature, bytes)) {
        return false;
    }
    if (bytes.size() % sizeof(uint32_t) != 0 || bytes.size() < 20) {
        // Invalid SPIR-V word stream
        const std::string key = compute_key("spirv", signature);
        std::error_code ec;
        std::filesystem::remove(entry_path_for_key(key), ec);
        hits_.fetch_sub(1, std::memory_order_relaxed);
        corruptions_.fetch_add(1, std::memory_order_relaxed);
        misses_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    out_words.resize(bytes.size() / sizeof(uint32_t));
    std::memcpy(out_words.data(), bytes.data(), bytes.size());
    // SPIR-V magic word is 0x07230203
    if (out_words[0] != 0x07230203u) {
        out_words.clear();
        const std::string key = compute_key("spirv", signature);
        std::error_code ec;
        std::filesystem::remove(entry_path_for_key(key), ec);
        hits_.fetch_sub(1, std::memory_order_relaxed);
        corruptions_.fetch_add(1, std::memory_order_relaxed);
        misses_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    return true;
}

bool KernelDiskCache::store_ptx(std::string_view signature, std::string_view ptx) {
    if (ptx.empty()) return false;
    return store_raw(BackendKind::CudaPtx, "ptx", signature, ptx.data(), ptx.size());
}

bool KernelDiskCache::load_ptx(std::string_view signature, std::string& out_ptx) {
    std::vector<uint8_t> bytes;
    if (!load_raw(BackendKind::CudaPtx, "ptx", signature, bytes)) {
        return false;
    }
    out_ptx.assign(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    if (out_ptx.find(".version") == std::string::npos && out_ptx.find(".entry") == std::string::npos) {
        out_ptx.clear();
        const std::string key = compute_key("ptx", signature);
        std::error_code ec;
        std::filesystem::remove(entry_path_for_key(key), ec);
        hits_.fetch_sub(1, std::memory_order_relaxed);
        corruptions_.fetch_add(1, std::memory_order_relaxed);
        misses_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    return true;
}

#if BROTENSOR_HAS_BRASS_JIT

namespace {

void write_u32(std::vector<uint8_t>& buf, uint32_t v) {
    const uint8_t b[4] = {
        static_cast<uint8_t>(v & 0xFFu),
        static_cast<uint8_t>((v >> 8) & 0xFFu),
        static_cast<uint8_t>((v >> 16) & 0xFFu),
        static_cast<uint8_t>((v >> 24) & 0xFFu),
    };
    buf.insert(buf.end(), b, b + 4);
}

void write_str(std::vector<uint8_t>& buf, const std::string& s) {
    write_u32(buf, static_cast<uint32_t>(s.size()));
    buf.insert(buf.end(), s.begin(), s.end());
}

bool read_u32(const uint8_t*& cur, const uint8_t* end, uint32_t& out) {
    if (static_cast<size_t>(end - cur) < 4) return false;
    out = static_cast<uint32_t>(cur[0]) |
          (static_cast<uint32_t>(cur[1]) << 8) |
          (static_cast<uint32_t>(cur[2]) << 16) |
          (static_cast<uint32_t>(cur[3]) << 24);
    cur += 4;
    return true;
}

bool read_str(const uint8_t*& cur, const uint8_t* end, std::string& out) {
    uint32_t len = 0;
    if (!read_u32(cur, end, len)) return false;
    if (static_cast<size_t>(end - cur) < len) return false;
    out.assign(reinterpret_cast<const char*>(cur), len);
    cur += len;
    return true;
}

} // namespace

bool KernelDiskCache::store_spirv_kernel(std::string_view signature, const brass::target::SpirvKernel& k) {
    if (k.words.empty()) return false;
    std::vector<uint8_t> buf;
    buf.reserve(128 + k.words.size() * sizeof(uint32_t));
    write_str(buf, k.entry);
    write_u32(buf, k.push_constant_bytes);
    write_u32(buf, k.shared_bytes);
    write_u32(buf, k.local_size[0]);
    write_u32(buf, k.local_size[1]);
    write_u32(buf, k.local_size[2]);
    write_u32(buf, k.local_size_spec_constants ? 1u : 0u);

    write_u32(buf, static_cast<uint32_t>(k.params.size()));
    for (const auto& p : k.params) {
        write_u32(buf, static_cast<uint32_t>(p.type.kind()));
        write_u32(buf, p.offset);
        write_u32(buf, p.size);
    }

    write_u32(buf, static_cast<uint32_t>(k.capabilities.size()));
    for (const auto& c : k.capabilities) write_str(buf, c);

    write_u32(buf, static_cast<uint32_t>(k.extensions.size()));
    for (const auto& e : k.extensions) write_str(buf, e);

    write_u32(buf, static_cast<uint32_t>(k.words.size()));
    for (uint32_t w : k.words) write_u32(buf, w);

    return store_raw(BackendKind::VulkanSpirvMeta, "spirv_meta", signature, buf.data(), buf.size());
}

bool KernelDiskCache::load_spirv_kernel(std::string_view signature, brass::target::SpirvKernel& out) {
    std::vector<uint8_t> buf;
    if (!load_raw(BackendKind::VulkanSpirvMeta, "spirv_meta", signature, buf)) {
        return false;
    }
    const uint8_t* cur = buf.data();
    const uint8_t* end = buf.data() + buf.size();

    brass::target::SpirvKernel k;
    uint32_t spec_flag = 0;
    uint32_t num_params = 0, num_caps = 0, num_exts = 0, num_words = 0;
    bool ok = read_str(cur, end, k.entry) &&
              read_u32(cur, end, k.push_constant_bytes) &&
              read_u32(cur, end, k.shared_bytes) &&
              read_u32(cur, end, k.local_size[0]) &&
              read_u32(cur, end, k.local_size[1]) &&
              read_u32(cur, end, k.local_size[2]) &&
              read_u32(cur, end, spec_flag) &&
              read_u32(cur, end, num_params);
    if (ok) {
        k.local_size_spec_constants = (spec_flag != 0);
        k.params.resize(num_params);
        for (uint32_t i = 0; i < num_params && ok; ++i) {
            uint32_t raw_type = 0;
            ok = read_u32(cur, end, raw_type) &&
                 read_u32(cur, end, k.params[i].offset) &&
                 read_u32(cur, end, k.params[i].size);
            if (ok) k.params[i].type = brass::Type(static_cast<brass::TypeKind>(raw_type));
        }
    }
    if (ok && read_u32(cur, end, num_caps)) {
        k.capabilities.resize(num_caps);
        for (uint32_t i = 0; i < num_caps && ok; ++i) ok = read_str(cur, end, k.capabilities[i]);
    } else {
        ok = false;
    }
    if (ok && read_u32(cur, end, num_exts)) {
        k.extensions.resize(num_exts);
        for (uint32_t i = 0; i < num_exts && ok; ++i) ok = read_str(cur, end, k.extensions[i]);
    } else {
        ok = false;
    }
    if (ok && read_u32(cur, end, num_words) && num_words >= 5) {
        k.words.resize(num_words);
        for (uint32_t i = 0; i < num_words && ok; ++i) ok = read_u32(cur, end, k.words[i]);
    } else {
        ok = false;
    }

    if (!ok || cur != end || k.words.empty() || k.words[0] != 0x07230203u) {
        const std::string key = compute_key("spirv_meta", signature);
        std::error_code ec;
        std::filesystem::remove(entry_path_for_key(key), ec);
        hits_.fetch_sub(1, std::memory_order_relaxed);
        corruptions_.fetch_add(1, std::memory_order_relaxed);
        misses_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    out = std::move(k);
    return true;
}

void* KernelDiskCache::compile_or_load_cpu_kernel(
    std::string_view signature,
    std::string_view entry_name,
    const std::function<void(brass::Module&)>& build_fn,
    const CpuCompileOptions& opts) {

    const std::string_view backend_tag = host_cpu_backend_tag();
    const std::string key = compute_key(backend_tag, signature);

    auto load_shared_lib_bytes = [&](const std::vector<uint8_t>& image_bytes) -> void* {
        if (image_bytes.empty()) return nullptr;
        std::filesystem::path rt_dir = cache_dir() / ".rt";
        std::string fname = key + "_" + std::to_string(current_pid()) + "_" +
                            std::to_string(next_temp_seq()) + shared_lib_extension();
        std::filesystem::path so_path = rt_dir / fname;
        if (!write_file_atomic(so_path, image_bytes.data(), image_bytes.size())) {
            return nullptr;
        }
#if defined(__APPLE__)
        {
            std::string cmd = "codesign -s - -f \"" + so_path.string() + "\" > /dev/null 2>&1";
            int res = std::system(cmd.c_str());
            (void)res;
        }
#endif
        std::string dl_err;
        auto lib = brass::target::DynamicLibrary::open(so_path.string(), &dl_err);
        if (!lib || !lib->is_valid()) {
            std::error_code ec;
            std::filesystem::remove(so_path, ec);
            return nullptr;
        }
        void* sym = lib->get_symbol(std::string(entry_name));
        if (!sym) {
            std::error_code ec;
            std::filesystem::remove(so_path, ec);
            return nullptr;
        }
        // Unlink the temporary runtime .so on POSIX after dlopen so no stale files accumulate
#if !defined(_WIN32)
        {
            std::error_code ec;
            std::filesystem::remove(so_path, ec);
        }
#endif
        std::lock_guard<std::mutex> lk(mutex_);
        loaded_libs_.push_back(std::move(lib));
        return sym;
    };

    // 1. Try persistent on-disk cache hit first
    std::vector<uint8_t> cached_bytes;
    if (load_raw(BackendKind::CpuNative, backend_tag, signature, cached_bytes)) {
        if (void* fn_ptr = load_shared_lib_bytes(cached_bytes)) {
            return fn_ptr;
        }
        // Shared object failed dlopen/dlsym despite header check; treat as corruption
        std::error_code ec;
        std::filesystem::remove(entry_path_for_key(key), ec);
        hits_.fetch_sub(1, std::memory_order_relaxed);
        corruptions_.fetch_add(1, std::memory_order_relaxed);
        misses_.fetch_add(1, std::memory_order_relaxed);
    }

    // 2. Cache miss (or disabled): build MIR module, run SMT guard elim + superopt + loop opts
    brass::Module mod("mod_" + std::string(entry_name));
    mod.set_allow_fp_reassociation(opts.enable_fp_reassociation);
    build_fn(mod);

    for (brass::Function* fn : mod.functions()) {
        if (!fn) continue;
        fn->rebuild_cfg_predecessors();
        if (opts.enable_smt_guard_elim) {
            brass::SmtGuardElimOptions guard_opts;
            guard_opts.max_sat_conflicts = 256;
            guard_opts.max_slice_depth   = 8;
            brass::run_smt_guard_elimination(*fn, mod, guard_opts);
        }
        if (opts.enable_mir_superopt) {
            brass::SuperoptOptions s_opts;
            s_opts.max_sat_conflicts = 256;
            s_opts.max_window_depth  = 3;
            brass::run_mir_superopt(*fn, mod, s_opts);
        }
        fn->rebuild_cfg_predecessors();
    }

    if (opts.enable_loop_opt) {
        brass::LoopOptOptions loop_opts;
        loop_opts.enable_avx2             = opts.enable_avx2;
        loop_opts.enable_fma              = opts.enable_fma;
        loop_opts.enable_fp_reassociation = opts.enable_fp_reassociation;
        loop_opts.enable_vectorize        = opts.enable_vectorize;
        loop_opts.enable_unroll           = opts.enable_unroll;
        loop_opts.unroll_factor           = opts.unroll_factor;
        loop_opts.enable_f64_demote       = false;
        loop_opts.enable_parallel_loops   = false;
        for (brass::Function* fn : mod.functions()) {
            if (fn) {
                brass::optimize_function(*fn, loop_opts);
                fn->rebuild_cfg_predecessors();
            }
        }
    }

    // 3. Link self-contained shared library via AotLinker if module has no external symbols
    bool can_aot_link = mod.external_symbols().empty();
    if (can_aot_link) {
        brass::target::LinkerOptions link_opts;
        link_opts.module_name = key + shared_lib_extension();
        link_opts.soname      = link_opts.module_name;
        link_opts.export_all_functions = true;

        brass::object::ObjectFile obj = brass::object::compile_module_to_object(mod, brass::Target::host());
        std::string link_err;
        std::vector<uint8_t> image_bytes = brass::target::AotLinker::link(obj, link_opts, &link_err);
        if (!image_bytes.empty()) {
            store_raw(BackendKind::CpuNative, backend_tag, signature, image_bytes.data(), image_bytes.size());
            if (void* fn_ptr = load_shared_lib_bytes(image_bytes)) {
                return fn_ptr;
            }
        }
    }

    // 4. Fallback to in-memory KernelJit if AOT linking or filesystem write is unavailable
    brass::codegen::KernelOptions k_opts;
    k_opts.enable_optimizations    = false; // already optimized above
    k_opts.enable_avx2             = opts.enable_avx2;
    k_opts.enable_fma              = opts.enable_fma;
    k_opts.enable_vectorize        = opts.enable_vectorize;
    k_opts.enable_unroll           = opts.enable_unroll;
    k_opts.unroll_factor           = opts.unroll_factor;
    k_opts.enable_fp_reassociation = opts.enable_fp_reassociation;

    brass::codegen::KernelJit jit(k_opts);
    brass::codegen::KernelFunction kfn = jit.compile(mod, entry_name);
    if (!kfn.is_valid()) {
        return nullptr;
    }
    {
        std::lock_guard<std::mutex> lk(mutex_);
        fallback_engines_.push_back(kfn.engine());
    }
    return kfn.entry_point();
}

#endif // BROTENSOR_HAS_BRASS_JIT

KernelDiskCache::Stats KernelDiskCache::stats() const noexcept {
    Stats s;
    s.hits        = hits_.load(std::memory_order_relaxed);
    s.misses      = misses_.load(std::memory_order_relaxed);
    s.writes      = writes_.load(std::memory_order_relaxed);
    s.corruptions = corruptions_.load(std::memory_order_relaxed);
    return s;
}

void KernelDiskCache::reset_stats() noexcept {
    hits_.store(0, std::memory_order_relaxed);
    misses_.store(0, std::memory_order_relaxed);
    writes_.store(0, std::memory_order_relaxed);
    corruptions_.store(0, std::memory_order_relaxed);
}

void KernelDiskCache::clear_loaded_libraries() {
    std::lock_guard<std::mutex> lk(mutex_);
#if BROTENSOR_HAS_BRASS_JIT
    loaded_libs_.clear();
    fallback_engines_.clear();
#endif
}

} // namespace brotensor::jit
