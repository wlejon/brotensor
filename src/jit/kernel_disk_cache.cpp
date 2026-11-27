#include "kernel_disk_cache.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <span>
#include <system_error>
#include <unordered_map>
#include <utility>

#if BROTENSOR_HAS_BRASS_JIT
#include <brass/cache/artifact_cache.hpp>
#include <brass/cache/native_module_cache.hpp>
#include <brass/codegen/kernel_jit.hpp>
#include <brass/mir/function.hpp>
#include <brass/target/msl_target.hpp>
#include <brass/target/spirv_target.hpp>
#endif

namespace brotensor::jit {

namespace {

[[maybe_unused]] bool env_flag(const char* name) {
    const char* v = std::getenv(name);
    return v && (std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0);
}

std::filesystem::path env_cache_dir() {
#if BROTENSOR_HAS_BRASS_JIT
    return brass::cache::default_cache_dir("brotensor", "kernels", "BRO_ML_CACHE_DIR");
#else
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
#endif
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

bool is_damage(brass::cache::Miss why) {
    using brass::cache::Miss;
    return why == Miss::Unreadable || why == Miss::Foreign || why == Miss::KeyMismatch ||
           why == Miss::Truncated || why == Miss::Corrupt;
}

brass::codegen::KernelOptions to_kernel_options(const KernelDiskCache::CpuCompileOptions& opts,
                                                bool enable_optimizations) {
    brass::codegen::KernelOptions k;
    k.opt_level               = 2;
    k.enable_optimizations    = enable_optimizations;
    k.enable_sroa             = true;
    k.enable_bce              = true;
    k.enable_smt_guard_elim   = opts.enable_smt_guard_elim;
    k.enable_superopt         = opts.enable_mir_superopt;
    k.enable_avx2             = opts.enable_avx2;
    k.enable_fma              = opts.enable_fma;
    k.enable_vectorize        = opts.enable_vectorize;
    k.enable_slp              = true;
    k.enable_unroll             = opts.enable_unroll;
    k.unroll_factor             = opts.unroll_factor;
    k.enable_fp_reassociation   = opts.enable_fp_reassociation;
    k.enable_trace_layout       = opts.enable_trace_layout;
    k.enable_post_ra_schedule   = opts.enable_post_ra_schedule;
    k.enable_software_pipelining = opts.enable_software_pipelining;
    return k;
}

brass::codegen::KernelFunction jit_compile(brass::Module& mod, std::string_view entry,
                                           const KernelDiskCache::CpuCompileOptions& opts) {
    brass::codegen::KernelJit jit(to_kernel_options(opts, opts.enable_loop_opt));
    return jit.compile(mod, entry);
}

// build_fn, then Brass's KernelJit optimization & verification pipeline.
void build_kernel_module(brass::Module& mod, const std::function<void(brass::Module&)>& build_fn,
                         const KernelDiskCache::CpuCompileOptions& opts) {
    build_fn(mod);
    brass::codegen::KernelJit::prepare_module(mod, to_kernel_options(opts, opts.enable_loop_opt));
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

bool KernelDiskCache::store_blob(std::string_view kind, std::string_view signature,
                                 const void* data, std::size_t size) {
#if BROTENSOR_HAS_BRASS_JIT
    if (kind.empty() || !data || size == 0) return false;
    auto st = stores();
    if (!st) return false;
    const auto* bytes = static_cast<const uint8_t*>(data);
    if (!st->store.put(blob_key(kind, signature),
                       {brass::cache::BlobRef("blob", std::span<const uint8_t>(bytes, size))})) {
        return false;
    }
    std::lock_guard<std::mutex> lk(mutex_);
    ++stats_.writes;
    return true;
#else
    (void)kind;
    (void)signature;
    (void)data;
    (void)size;
    return false;
#endif
}

bool KernelDiskCache::load_blob(std::string_view kind, std::string_view signature,
                                std::vector<uint8_t>& out) {
#if BROTENSOR_HAS_BRASS_JIT
    if (kind.empty()) return false;
    brass::cache::Miss why = brass::cache::Miss::None;
    std::optional<brass::cache::Artifact> art;
    if (auto st = stores()) art = st->store.get(blob_key(kind, signature), &why);
    const bool damaged = is_damage(why);
    if (art) {
        const auto bytes = art->bytes("blob");
        out.assign(bytes.begin(), bytes.end());
        std::lock_guard<std::mutex> lk(mutex_);
        ++stats_.hits;
        return true;
    }
    std::lock_guard<std::mutex> lk(mutex_);
    ++stats_.misses;
    if (damaged) ++stats_.corruptions;
    return false;
#else
    (void)kind;
    (void)signature;
    (void)out;
    return false;
#endif
}

#if BROTENSOR_HAS_BRASS_JIT

brass::cache::CacheKey KernelDiskCache::blob_key(std::string_view kind, std::string_view signature) {
    return brass::cache::KeyBuilder(kKeyDomain)
        .add_brass_codegen()
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
        .add_bool("opt.trace_layout", o.enable_trace_layout)
        .add_bool("opt.post_ra_schedule", o.enable_post_ra_schedule)
        .add_bool("opt.software_pipelining", o.enable_software_pipelining)
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
    const std::vector<uint8_t> encoded = brass::target::serialize_spirv_kernel(k);
    const bool ok = st->store.put(blob_key("spirv", signature),
                                  {brass::cache::BlobRef("spirv_kernel", encoded)});
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
        if (brass::target::deserialize_spirv_kernel(art->bytes("spirv_kernel"), k) &&
            k.words.size() >= 5 && k.words[0] == 0x07230203u) {
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

namespace {

void append_u32(std::vector<uint8_t>& buf, uint32_t v) {
    for (int i = 0; i < 4; ++i) buf.push_back(static_cast<uint8_t>((v >> (i * 8)) & 0xFFu));
}

void append_str(std::vector<uint8_t>& buf, const std::string& s) {
    append_u32(buf, static_cast<uint32_t>(s.size()));
    buf.insert(buf.end(), reinterpret_cast<const uint8_t*>(s.data()),
               reinterpret_cast<const uint8_t*>(s.data()) + s.size());
}

bool read_u32(std::span<const uint8_t> bytes, size_t& pos, uint32_t& out) {
    if (pos + 4 > bytes.size()) return false;
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= static_cast<uint32_t>(bytes[pos + static_cast<size_t>(i)]) << (i * 8);
    pos += 4;
    out = v;
    return true;
}

bool read_str(std::span<const uint8_t> bytes, size_t& pos, std::string& out) {
    uint32_t n = 0;
    if (!read_u32(bytes, pos, n) || pos + n > bytes.size()) return false;
    out.assign(reinterpret_cast<const char*>(bytes.data() + pos), n);
    pos += n;
    return true;
}

constexpr uint32_t kMslBlobMagic = 0x314C534Du; // "MSL1"

} // namespace

bool KernelDiskCache::store_msl_kernel(std::string_view signature, const brass::target::MslKernel& k) {
    if (k.source.empty()) return false;
    auto st = stores();
    if (!st) return false;
    std::vector<uint8_t> encoded;
    encoded.reserve(64 + k.entry_name.size() + k.source.size() + k.scalar_param_offsets.size() * 4);
    append_u32(encoded, kMslBlobMagic);
    append_str(encoded, k.entry_name);
    append_str(encoded, k.source);
    append_u32(encoded, k.num_buffers);
    append_u32(encoded, k.scalar_params_bytes);
    append_u32(encoded, k.workgroup_x);
    append_u32(encoded, k.workgroup_y);
    append_u32(encoded, k.workgroup_z);
    append_u32(encoded, k.uses_shared_memory ? 1u : 0u);
    append_u32(encoded, k.shared_memory_bytes);
    append_u32(encoded, static_cast<uint32_t>(k.scalar_param_offsets.size()));
    for (uint32_t off : k.scalar_param_offsets) append_u32(encoded, off);

    const bool ok = st->store.put(blob_key("msl", signature),
                                  {brass::cache::BlobRef("msl_kernel", encoded)});
    if (!ok) return false;
    std::lock_guard<std::mutex> lk(mutex_);
    ++stats_.writes;
    return true;
}

bool KernelDiskCache::load_msl_kernel(std::string_view signature, brass::target::MslKernel& out) {
    brass::cache::Miss why = brass::cache::Miss::None;
    std::optional<brass::cache::Artifact> art;
    if (auto st = stores()) art = st->store.get(blob_key("msl", signature), &why);
    bool damaged = is_damage(why);
    if (art) {
        auto raw = art->bytes("msl_kernel");
        size_t pos = 0;
        uint32_t magic = 0, used_smem = 0, noff = 0;
        brass::target::MslKernel k;
        if (read_u32(raw, pos, magic) && magic == kMslBlobMagic &&
            read_str(raw, pos, k.entry_name) &&
            read_str(raw, pos, k.source) && !k.source.empty() &&
            read_u32(raw, pos, k.num_buffers) &&
            read_u32(raw, pos, k.scalar_params_bytes) &&
            read_u32(raw, pos, k.workgroup_x) &&
            read_u32(raw, pos, k.workgroup_y) &&
            read_u32(raw, pos, k.workgroup_z) &&
            read_u32(raw, pos, used_smem) &&
            read_u32(raw, pos, k.shared_memory_bytes) &&
            read_u32(raw, pos, noff) && pos + static_cast<size_t>(noff) * 4 == raw.size()) {
            k.uses_shared_memory = (used_smem != 0);
            k.scalar_param_offsets.resize(noff);
            for (uint32_t i = 0; i < noff; ++i) read_u32(raw, pos, k.scalar_param_offsets[i]);
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
        const auto build_obj = [&](brass::object::ObjectFile& obj, std::string& error) {
            brass::Module mod("brotensor_" + std::string(entry_name));
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
            try {
                brass::codegen::KernelJit jit(to_kernel_options(opts, opts.enable_loop_opt));
                obj = jit.compile_object(mod);
            } catch (const std::exception& e) {
                error = e.what();
                return false;
            } catch (...) {
                error = "KernelJit::compile_object threw";
                return false;
            }
            return true;
        };
        std::string error;
        std::shared_ptr<const brass::cache::NativeModule> m =
            st->natives.get_or_build_object(cpu_key(signature, entry_name, opts), build_obj, {}, &error);
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

std::size_t KernelDiskCache::entry_count() const {
#if BROTENSOR_HAS_BRASS_JIT
    if (auto st = stores()) return st->store.list_keys().size();
#endif
    return 0;
}

bool KernelDiskCache::export_bundle(const std::filesystem::path& out_path,
                                    std::string_view metadata,
                                    std::size_t* entries_written,
                                    std::size_t* bytes_out,
                                    std::string* error) const {
#if BROTENSOR_HAS_BRASS_JIT
    auto st = stores();
    if (!st) {
        if (error) *error = "kernel disk cache is disabled";
        return false;
    }
    brass::cache::BundleStats bs;
    const bool ok = st->store.export_bundle(out_path, metadata, &bs, error);
    if (entries_written) *entries_written = bs.entries_written;
    if (bytes_out) *bytes_out = bs.bytes;
    return ok;
#else
    (void)out_path; (void)metadata; (void)entries_written; (void)bytes_out;
    if (error) *error = "Brass JIT is disabled";
    return false;
#endif
}

bool KernelDiskCache::import_bundle(const std::filesystem::path& in_path,
                                    std::string* metadata_out,
                                    std::size_t* entries_imported,
                                    std::size_t* entries_skipped,
                                    std::size_t* bytes_out,
                                    std::string* error) {
#if BROTENSOR_HAS_BRASS_JIT
    auto st = stores();
    if (!st) {
        if (error) *error = "kernel disk cache is disabled";
        return false;
    }
    brass::cache::BundleStats bs;
    const bool ok = st->store.import_bundle(in_path, metadata_out, &bs, error);
    if (entries_imported) *entries_imported = bs.entries_imported;
    if (entries_skipped) *entries_skipped = bs.entries_skipped;
    if (bytes_out) *bytes_out = bs.bytes;
    return ok;
#else
    (void)in_path; (void)metadata_out; (void)entries_imported; (void)entries_skipped; (void)bytes_out;
    if (error) *error = "Brass JIT is disabled";
    return false;
#endif
}

namespace {
struct FlushHookRegistry {
    std::mutex mu;
    std::unordered_map<int, PipelineCacheFlushFn> hooks;
};
FlushHookRegistry& flush_hooks() {
    static auto* r = new FlushHookRegistry();
    return *r;
}
} // namespace

void register_pipeline_flush_hook(DeviceType type, PipelineCacheFlushFn fn) {
    if (!fn) return;
    auto& r = flush_hooks();
    std::lock_guard<std::mutex> lk(r.mu);
    r.hooks[static_cast<int>(type)] = fn;
}

void flush_pipeline_caches_for(Device dev) {
    PipelineCacheFlushFn fn = nullptr;
    {
        auto& r = flush_hooks();
        std::lock_guard<std::mutex> lk(r.mu);
        auto it = r.hooks.find(static_cast<int>(dev.type));
        if (it != r.hooks.end()) fn = it->second;
    }
    if (fn) fn(dev.index);
}

} // namespace brotensor::jit
