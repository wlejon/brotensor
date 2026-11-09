#include "kernel_disk_cache.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <system_error>
#include <utility>

#if BROTENSOR_HAS_BRASS_JIT
#include <brass/cache/native_module_cache.hpp>
#include <brass/codegen/kernel_jit.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/smt_guard_elim.hpp>
#include <brass/mir/superopt.hpp>
#endif

namespace brotensor::jit {

namespace {

[[maybe_unused]] bool env_flag(const char* name) {
    const char* v = std::getenv(name);
    return v && (std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0);
}

std::filesystem::path env_cache_dir() {
    if (const char* env_dir = std::getenv("BRO_ML_CACHE_DIR")) {
        if (env_dir[0] != '\0') return std::filesystem::path(env_dir);
    }
    if (const char* xdg = std::getenv("XDG_CACHE_HOME")) {
        if (xdg[0] != '\0') return std::filesystem::path(xdg) / "brotensor" / "kernels";
    }
    if (const char* home = std::getenv("HOME")) {
        if (home[0] != '\0') return std::filesystem::path(home) / ".cache" / "brotensor" / "kernels";
    }
    std::error_code ec;
    auto tmp = std::filesystem::temp_directory_path(ec);
    if (ec || tmp.empty()) tmp = "/tmp";
    return tmp / "brotensor_kernel_cache";
}

uint64_t env_max_bytes() {
    if (const char* mb = std::getenv("BRO_ML_CACHE_MAX_MB")) {
        char* end = nullptr;
        const unsigned long long v = std::strtoull(mb, &end, 10);
        if (end != mb && *end == '\0' && v > 0) return static_cast<uint64_t>(v) << 20;
    }
    return KernelDiskCache::kDefaultMaxBytes;
}

#if BROTENSOR_HAS_BRASS_JIT

// The previous cache's files: "<32 hex>.btkc", its temporaries
// "<...>.btkc.tmp.<pid>...", and its load copies under ".rt/". Nothing reads
// them; they only take space.
void remove_old_format(const std::filesystem::path& dir) {
    std::error_code ec;
    std::filesystem::directory_iterator it(dir, ec), end;
    for (; !ec && it != end; it.increment(ec)) {
        const std::string name = it->path().filename().string();
        const bool old = (name.size() > 5 && name.compare(name.size() - 5, 5, ".btkc") == 0) ||
                         name.find(".btkc.tmp.") != std::string::npos;
        std::error_code rec;
        if (old && it->is_regular_file(rec)) std::filesystem::remove(it->path(), rec);
    }
    std::filesystem::remove_all(dir / ".rt", ec);
}

// The pass budgets compile_or_load_cpu_kernel runs with: part of every CPU key.
constexpr uint32_t kGuardMaxSatConflicts = 256;
constexpr uint32_t kGuardMaxSliceDepth   = 8;
constexpr uint32_t kSuperoptMaxSatConflicts = 256;
constexpr uint32_t kSuperoptMaxWindowDepth  = 3;

bool is_damage(brass::cache::Miss why) {
    using brass::cache::Miss;
    return why == Miss::Unreadable || why == Miss::Foreign || why == Miss::KeyMismatch ||
           why == Miss::Truncated || why == Miss::Corrupt;
}

// ── SpirvKernel metadata (everything but the words) ─────────────────────────

void put_u32(std::string& b, uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back(static_cast<char>((v >> (8 * i)) & 0xFFu));
}

void put_str(std::string& b, const std::string& s) {
    put_u32(b, static_cast<uint32_t>(s.size()));
    b.append(s);
}

struct Reader {
    const uint8_t* cur;
    const uint8_t* end;
    bool ok = true;

    uint32_t u32() {
        if (!ok || end - cur < 4) { ok = false; return 0; }
        const uint32_t v = static_cast<uint32_t>(cur[0]) | (static_cast<uint32_t>(cur[1]) << 8) |
                           (static_cast<uint32_t>(cur[2]) << 16) | (static_cast<uint32_t>(cur[3]) << 24);
        cur += 4;
        return v;
    }
    std::string str() {
        const uint32_t n = u32();
        if (!ok || static_cast<size_t>(end - cur) < n) { ok = false; return {}; }
        std::string s(reinterpret_cast<const char*>(cur), n);
        cur += n;
        return s;
    }
    // A count of records at least `min_bytes` each that must fit what is left.
    uint32_t count(size_t min_bytes) {
        const uint32_t n = u32();
        if (ok && static_cast<size_t>(end - cur) / min_bytes < n) ok = false;
        return ok ? n : 0;
    }
};

std::string encode_spirv_meta(const brass::target::SpirvKernel& k) {
    std::string b;
    put_str(b, k.entry);
    put_u32(b, static_cast<uint32_t>(k.stage));
    put_u32(b, k.push_constant_bytes);
    put_u32(b, k.shared_bytes);
    for (uint32_t s : k.local_size) put_u32(b, s);
    put_u32(b, k.local_size_spec_constants ? 1u : 0u);
    put_u32(b, static_cast<uint32_t>(k.params.size()));
    for (const auto& p : k.params) {
        put_u32(b, static_cast<uint32_t>(p.type.kind()));
        put_u32(b, p.offset);
        put_u32(b, p.size);
    }
    put_u32(b, static_cast<uint32_t>(k.descriptors.size()));
    for (const auto& d : k.descriptors) {
        put_u32(b, d.set);
        put_u32(b, d.binding);
        put_u32(b, static_cast<uint32_t>(d.kind));
        put_u32(b, static_cast<uint32_t>(d.dim));
        put_u32(b, d.arrayed ? 1u : 0u);
        put_u32(b, d.depth ? 1u : 0u);
        put_u32(b, static_cast<uint32_t>(d.scalar));
        put_u32(b, d.format);
        put_u32(b, d.bytes);
    }
    put_u32(b, static_cast<uint32_t>(k.capabilities.size()));
    for (const auto& c : k.capabilities) put_str(b, c);
    put_u32(b, static_cast<uint32_t>(k.extensions.size()));
    for (const auto& e : k.extensions) put_str(b, e);
    return b;
}

bool decode_spirv_meta(std::span<const uint8_t> bytes, brass::target::SpirvKernel& k) {
    Reader r{bytes.data(), bytes.data() + bytes.size()};
    k.entry = r.str();
    k.stage = static_cast<brass::target::SpirvStage>(r.u32());
    k.push_constant_bytes = r.u32();
    k.shared_bytes = r.u32();
    for (uint32_t& s : k.local_size) s = r.u32();
    k.local_size_spec_constants = r.u32() != 0;
    k.params.resize(r.count(12));
    for (auto& p : k.params) {
        p.type = brass::Type(static_cast<brass::TypeKind>(r.u32()));
        p.offset = r.u32();
        p.size = r.u32();
    }
    k.descriptors.resize(r.count(36));
    for (auto& d : k.descriptors) {
        d.set = r.u32();
        d.binding = r.u32();
        d.kind = static_cast<brass::target::SpirvDescriptorKind>(r.u32());
        d.dim = static_cast<brass::target::SpirvImageDim>(r.u32());
        d.arrayed = r.u32() != 0;
        d.depth = r.u32() != 0;
        d.scalar = static_cast<brass::target::SpirvImageScalar>(r.u32());
        d.format = r.u32();
        d.bytes = r.u32();
    }
    k.capabilities.resize(r.count(4));
    for (auto& c : k.capabilities) c = r.str();
    k.extensions.resize(r.count(4));
    for (auto& e : k.extensions) e = r.str();
    return r.ok && r.cur == r.end;
}

brass::codegen::KernelFunction jit_compile(brass::Module& mod, std::string_view entry,
                                           const KernelDiskCache::CpuCompileOptions& opts) {
    brass::codegen::KernelOptions k;
    k.enable_optimizations    = false;   // the passes already ran
    k.enable_avx2             = opts.enable_avx2;
    k.enable_fma              = opts.enable_fma;
    k.enable_vectorize        = opts.enable_vectorize;
    k.enable_unroll           = opts.enable_unroll;
    k.unroll_factor           = opts.unroll_factor;
    k.enable_fp_reassociation = opts.enable_fp_reassociation;
    brass::codegen::KernelJit jit(k);
    return jit.compile(mod, entry);
}

// build_fn, then brotensor's passes over what it built.
void build_kernel_module(brass::Module& mod, const std::function<void(brass::Module&)>& build_fn,
                         const KernelDiskCache::CpuCompileOptions& opts) {
    mod.set_allow_fp_reassociation(opts.enable_fp_reassociation);
    build_fn(mod);
    for (brass::Function* fn : mod.functions()) {
        if (!fn) continue;
        fn->rebuild_cfg_predecessors();
        if (opts.enable_smt_guard_elim) {
            brass::SmtGuardElimOptions g;
            g.max_sat_conflicts = kGuardMaxSatConflicts;
            g.max_slice_depth   = kGuardMaxSliceDepth;
            brass::run_smt_guard_elimination(*fn, mod, g);
        }
        if (opts.enable_mir_superopt) {
            brass::SuperoptOptions s;
            s.max_sat_conflicts = kSuperoptMaxSatConflicts;
            s.max_window_depth  = kSuperoptMaxWindowDepth;
            brass::run_mir_superopt(*fn, mod, s);
        }
        fn->rebuild_cfg_predecessors();
    }
    if (opts.enable_loop_opt) {
        brass::LoopOptOptions l;
        l.enable_avx2             = opts.enable_avx2;
        l.enable_fma              = opts.enable_fma;
        l.enable_fp_reassociation = opts.enable_fp_reassociation;
        l.enable_vectorize        = opts.enable_vectorize;
        l.enable_unroll           = opts.enable_unroll;
        l.unroll_factor           = opts.unroll_factor;
        l.enable_f64_demote       = false;
        l.enable_parallel_loops   = false;
        for (brass::Function* fn : mod.functions()) {
            if (!fn) continue;
            brass::optimize_function(*fn, l);
            fn->rebuild_cfg_predecessors();
        }
    }
}

#endif // BROTENSOR_HAS_BRASS_JIT

} // namespace

#if BROTENSOR_HAS_BRASS_JIT
struct KernelDiskCache::Stores {
    Stores(const std::filesystem::path& dir, uint64_t max_bytes) : store(dir, max_bytes), natives(store) {}
    brass::cache::ArtifactCache store;
    brass::cache::NativeModuleCache natives;
};
#else
struct KernelDiskCache::Stores {};
#endif

KernelDiskCache& KernelDiskCache::instance() {
    static KernelDiskCache cache;
    return cache;
}

bool KernelDiskCache::is_enabled() const noexcept {
#if BROTENSOR_HAS_BRASS_JIT
    return !env_flag("BRO_ML_CACHE_DISABLE");
#else
    return false;
#endif
}

std::filesystem::path KernelDiskCache::cache_dir() const {
    {
        std::lock_guard<std::mutex> lk(mutex_);
        if (!override_dir_.empty()) return override_dir_;
    }
    return env_cache_dir();
}

void KernelDiskCache::set_cache_dir(std::filesystem::path dir) {
    std::lock_guard<std::mutex> lk(mutex_);
    override_dir_ = std::move(dir);
}

void KernelDiskCache::clear_cache_dir_override() {
    std::lock_guard<std::mutex> lk(mutex_);
    override_dir_.clear();
}

uint64_t KernelDiskCache::max_bytes() const {
    {
        std::lock_guard<std::mutex> lk(mutex_);
        if (override_max_bytes_ != 0) return override_max_bytes_;
    }
    return env_max_bytes();
}

void KernelDiskCache::set_max_bytes(uint64_t bytes) {
    std::lock_guard<std::mutex> lk(mutex_);
    override_max_bytes_ = bytes;
}

std::shared_ptr<KernelDiskCache::Stores> KernelDiskCache::stores() const {
#if BROTENSOR_HAS_BRASS_JIT
    if (!is_enabled()) return nullptr;
    const std::filesystem::path dir = cache_dir();
    const uint64_t budget = max_bytes();
    std::lock_guard<std::mutex> lk(mutex_);
    if (stores_ && stores_->store.dir() == dir && stores_->store.max_bytes() == budget) return stores_;
    remove_old_format(dir);
    stores_ = std::make_shared<Stores>(dir, budget);
    return stores_;
#else
    return nullptr;
#endif
}

bool KernelDiskCache::store_ptx(std::string_view signature, std::string_view ptx) {
#if BROTENSOR_HAS_BRASS_JIT
    if (ptx.empty()) return false;
    auto st = stores();
    if (!st || !st->store.put(blob_key("ptx", signature), {brass::cache::BlobRef("ptx", ptx)})) return false;
    std::lock_guard<std::mutex> lk(mutex_);
    ++stats_.writes;
    return true;
#else
    (void)signature;
    (void)ptx;
    return false;
#endif
}

bool KernelDiskCache::load_ptx(std::string_view signature, std::string& out_ptx) {
#if BROTENSOR_HAS_BRASS_JIT
    brass::cache::Miss why = brass::cache::Miss::None;
    std::optional<brass::cache::Artifact> art;
    if (auto st = stores()) art = st->store.get(blob_key("ptx", signature), &why);
    bool damaged = is_damage(why);
    if (art) {
        const std::string_view text = art->text("ptx");
        if (text.find(".version") != std::string_view::npos || text.find(".entry") != std::string_view::npos) {
            out_ptx.assign(text);
            std::lock_guard<std::mutex> lk(mutex_);
            ++stats_.hits;
            return true;
        }
        damaged = true;
    }
    std::lock_guard<std::mutex> lk(mutex_);
    ++stats_.misses;
    if (damaged) ++stats_.corruptions;
    return false;
#else
    (void)signature;
    (void)out_ptx;
    return false;
#endif
}

#if BROTENSOR_HAS_BRASS_JIT

brass::cache::CacheKey KernelDiskCache::blob_key(std::string_view kind, std::string_view signature) {
    return brass::cache::KeyBuilder(kKeyDomain)
        .add_brass_build()
        .add("kind", kind)
        .add("signature", signature)
        .finish();
}

brass::cache::CacheKey KernelDiskCache::cpu_key(std::string_view signature, std::string_view entry_name,
                                                const CpuCompileOptions& o) {
    return brass::cache::KeyBuilder(kKeyDomain)
        .add_brass_build()
        .add("kind", "cpu")
        .add("signature", signature)
        .add("entry", entry_name)
        .add_bool("opt.loop_opt", o.enable_loop_opt)
        .add_bool("opt.avx2", o.enable_avx2)
        .add_bool("opt.fma", o.enable_fma)
        .add_bool("opt.vectorize", o.enable_vectorize)
        .add_bool("opt.unroll", o.enable_unroll)
        .add_u64("opt.unroll_factor", o.unroll_factor)
        .add_bool("opt.fp_reassociation", o.enable_fp_reassociation)
        .add_bool("opt.smt_guard_elim", o.enable_smt_guard_elim)
        .add_bool("opt.mir_superopt", o.enable_mir_superopt)
        .add_u64("pass.guard.max_sat_conflicts", kGuardMaxSatConflicts)
        .add_u64("pass.guard.max_slice_depth", kGuardMaxSliceDepth)
        .add_u64("pass.superopt.max_sat_conflicts", kSuperoptMaxSatConflicts)
        .add_u64("pass.superopt.max_window_depth", kSuperoptMaxWindowDepth)
        .finish();
}

std::filesystem::path KernelDiskCache::blob_entry_path(std::string_view kind, std::string_view signature) const {
    auto st = stores();
    return st ? st->store.path_for(blob_key(kind, signature)) : std::filesystem::path();
}

std::filesystem::path KernelDiskCache::cpu_entry_path(std::string_view signature, std::string_view entry_name,
                                                      const CpuCompileOptions& opts) const {
    auto st = stores();
    if (!st) return {};
    return st->store.path_for(st->natives.storage_key(cpu_key(signature, entry_name, opts), {}));
}

uint64_t KernelDiskCache::trim() {
    auto st = stores();
    return st ? st->store.trim() : 0;
}

bool KernelDiskCache::store_spirv_kernel(std::string_view signature, const brass::target::SpirvKernel& k) {
    if (k.words.empty()) return false;
    auto st = stores();
    if (!st) return false;
    const std::string meta = encode_spirv_meta(k);
    const bool ok = st->store.put(blob_key("spirv", signature),
                                  {brass::cache::BlobRef("meta", meta),
                                   brass::cache::BlobRef::of<uint32_t>("spirv", std::span<const uint32_t>(k.words))});
    if (!ok) return false;
    std::lock_guard<std::mutex> lk(mutex_);
    ++stats_.writes;
    return true;
}

bool KernelDiskCache::load_spirv_kernel(std::string_view signature, brass::target::SpirvKernel& out) {
    brass::cache::Miss why = brass::cache::Miss::None;
    std::optional<brass::cache::Artifact> art;
    if (auto st = stores()) art = st->store.get(blob_key("spirv", signature), &why);
    bool damaged = is_damage(why);
    if (art) {
        brass::target::SpirvKernel k;
        const std::span<const uint32_t> words = art->as<uint32_t>("spirv");
        if (words.size() >= 5 && words[0] == 0x07230203u && decode_spirv_meta(art->bytes("meta"), k)) {
            k.words.assign(words.begin(), words.end());
            out = std::move(k);
            std::lock_guard<std::mutex> lk(mutex_);
            ++stats_.hits;
            return true;
        }
        damaged = true;
    }
    std::lock_guard<std::mutex> lk(mutex_);
    ++stats_.misses;
    if (damaged) ++stats_.corruptions;
    return false;
}

void* KernelDiskCache::compile_or_load_cpu_kernel(
    std::string_view signature,
    std::string_view entry_name,
    const std::function<void(brass::Module&)>& build_fn,
    const CpuCompileOptions& opts) {

    brass::codegen::KernelFunction jitted;
    if (auto st = stores()) {
        std::exception_ptr thrown;
        const auto build = [&](brass::Module& mod, std::string& error) {
            try {
                build_kernel_module(mod, build_fn, opts);
            } catch (...) {
                thrown = std::current_exception();
                error = "the kernel's build threw";
                return false;
            }
            // A cached image cannot point into this process: a module that
            // calls functions it does not define is compiled in memory.
            if (!mod.external_symbols().empty()) {
                jitted = jit_compile(mod, entry_name, opts);
                error = "the kernel calls functions it does not define";
                return false;
            }
            return true;
        };
        std::string error;
        std::shared_ptr<const brass::cache::NativeModule> m =
            st->natives.get_or_build(cpu_key(signature, entry_name, opts), build, {}, &error);
        if (thrown) std::rethrow_exception(thrown);
        if (m) {
            if (void* sym = m->symbol(entry_name)) {
                std::lock_guard<std::mutex> lk(mutex_);
                if (m->from_cache()) {
                    ++stats_.hits;
                } else {
                    ++stats_.misses;
                    ++stats_.writes;
                }
                loaded_.push_back(std::move(m));
                return sym;
            }
        }
    }

    // Disabled, uncacheable, or the image would not link or load: in memory.
    if (!jitted.is_valid()) {
        brass::Module mod("brotensor_" + std::string(entry_name));
        build_kernel_module(mod, build_fn, opts);
        jitted = jit_compile(mod, entry_name, opts);
        if (!jitted.is_valid()) return nullptr;
    }
    std::lock_guard<std::mutex> lk(mutex_);
    ++stats_.misses;
    loaded_.push_back(jitted.engine());
    return jitted.entry_point();
}

#endif // BROTENSOR_HAS_BRASS_JIT

KernelDiskCache::Stats KernelDiskCache::stats() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return stats_;
}

void KernelDiskCache::reset_stats() noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    stats_ = Stats{};
}

void KernelDiskCache::clear_loaded_libraries() {
    std::vector<std::shared_ptr<const void>> dropped;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        dropped.swap(loaded_);
    }
}

} // namespace brotensor::jit
