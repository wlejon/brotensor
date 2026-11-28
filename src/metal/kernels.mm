#import "internal.h"

#include "../jit/kernel_disk_cache.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <unistd.h>

#if BROTENSOR_HAS_BRASS_JIT
namespace brass { std::string_view codegen_fingerprint() noexcept; }
#endif

namespace brotensor::metal_impl {

namespace {

std::uint64_t fnv1a_u64(std::uint64_t h, const void* data, std::size_t n) noexcept {
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

class API_AVAILABLE(macos(11.0)) MtlBinaryArchiveStore {
public:
    static MtlBinaryArchiveStore& instance() {
        static MtlBinaryArchiveStore s;
        return s;
    }

    id<MTLBinaryArchive> archive_for(id<MTLDevice> dev) {
        if (!dev || !::brotensor::jit::KernelDiskCache::instance().is_enabled()) return nil;
        if (![dev respondsToSelector:@selector(newBinaryArchiveWithDescriptor:error:)]) return nil;
        std::lock_guard<std::mutex> lk(mu_);
        ensure_loaded_locked(dev);
        return archive_;
    }

    void record_pipeline(id<MTLDevice> dev,
                         MTLComputePipelineDescriptor* pdesc,
                         NSString* msl_source,
                         NSString* fn_name) {
        if (!dev || !pdesc || !msl_source || !fn_name) return;
        const char* s = [msl_source UTF8String];
        const char* f = [fn_name UTF8String];
        std::uint64_t key = 14695981039346656037ULL;
        if (s) key = fnv1a_u64(key, s, std::strlen(s));
        key ^= 0xffu;
        if (f) key = fnv1a_u64(key, f, std::strlen(f));

        std::lock_guard<std::mutex> lk(mu_);
        ensure_loaded_locked(dev);
        if (!archive_ || !seen_.insert(key).second) return;
        NSError* err = nil;
        if ([archive_ addComputePipelineFunctionsWithDescriptor:pdesc error:&err]) {
            if (++dirty_ >= 16u) {
                flush_locked();
            }
        }
    }

    ~MtlBinaryArchiveStore() {
        flush();
    }

    void flush() {
        std::lock_guard<std::mutex> lk(mu_);
        flush_locked();
    }

private:
    MtlBinaryArchiveStore() = default;

    void ensure_loaded_locked(id<MTLDevice> dev) {
        if (init_done_) return;
        init_done_ = true;
        try {
            const std::filesystem::path dir =
                ::brotensor::jit::KernelDiskCache::instance().cache_dir() / "mtl_bin_archives";
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            if (ec) return;

            std::uint64_t env_hash = 14695981039346656037ULL;
            if (NSString* dname = [dev name]) {
                const char* p = [dname UTF8String];
                if (p) env_hash = fnv1a_u64(env_hash, p, std::strlen(p));
            }
            if (NSString* osver = [[NSProcessInfo processInfo] operatingSystemVersionString]) {
                const char* p = [osver UTF8String];
                if (p) env_hash = fnv1a_u64(env_hash, p, std::strlen(p));
            }
#if BROTENSOR_HAS_BRASS_JIT
            const std::string_view fp = brass::codegen_fingerprint();
            env_hash = fnv1a_u64(env_hash, fp.data(), fp.size());
#endif
            char fname[48];
            std::snprintf(fname, sizeof(fname), "archive_%016llx.metallib",
                          static_cast<unsigned long long>(env_hash));
            archive_path_ = dir / fname;

            MTLBinaryArchiveDescriptor* bdesc = [[MTLBinaryArchiveDescriptor alloc] init];
            if (std::filesystem::exists(archive_path_, ec) && !ec) {
                NSString* pstr = [NSString stringWithUTF8String:archive_path_.c_str()];
                bdesc.url = [NSURL fileURLWithPath:pstr];
            }
            NSError* err = nil;
            archive_ = [dev newBinaryArchiveWithDescriptor:bdesc error:&err];
            if (!archive_ && bdesc.url) {
                bdesc.url = nil;
                err = nil;
                archive_ = [dev newBinaryArchiveWithDescriptor:bdesc error:&err];
            }
        } catch (...) {
            archive_ = nil;
        }
    }

    void flush_locked() {
        if (!archive_ || dirty_ == 0 || archive_path_.empty()) return;
        try {
            const std::filesystem::path tmp =
                archive_path_.string() + ".tmp." + std::to_string(static_cast<long>(::getpid()));
            NSString* tstr = [NSString stringWithUTF8String:tmp.c_str()];
            NSURL* turl = [NSURL fileURLWithPath:tstr];
            NSError* err = nil;
            if ([archive_ serializeToURL:turl error:&err]) {
                std::error_code ec;
                std::filesystem::rename(tmp, archive_path_, ec);
                if (!ec) {
                    dirty_ = 0;
                    return;
                }
            }
            std::error_code ec;
            std::filesystem::remove(tmp, ec);
        } catch (...) {
        }
    }

    std::mutex mu_;
    bool init_done_ = false;
    id<MTLBinaryArchive> archive_ = nil;
    std::filesystem::path archive_path_;
    std::unordered_set<std::uint64_t> seen_;
    std::uint32_t dirty_ = 0;
};

} // namespace

void flush_pipeline_archive() {
    if (@available(macOS 11.0, *)) {
        MtlBinaryArchiveStore::instance().flush();
    }
}

id<MTLComputePipelineState> compile_pipeline(NSString* msl_source,
                                             NSString* function_name) {
    @autoreleasepool {
        NSError* err = nil;
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        opts.languageVersion = MTLLanguageVersion3_1;
        id<MTLDevice> dev = device();
        id<MTLLibrary> lib = [dev newLibraryWithSource:msl_source
                                               options:opts
                                                 error:&err];
        if (!lib) {
            std::string m = "Metal: MSL compile failed for ";
            m += [function_name UTF8String];
            if (err) { m += ": "; m += [[err localizedDescription] UTF8String]; }
            throw std::runtime_error(m);
        }
        id<MTLFunction> fn = [lib newFunctionWithName:function_name];
        if (!fn) {
            std::string m = "Metal: function not found: ";
            m += [function_name UTF8String];
            throw std::runtime_error(m);
        }

        id<MTLComputePipelineState> pso = nil;
        NSError* perr = nil;
        if (@available(macOS 11.0, *)) {
            if (id<MTLBinaryArchive> ar = MtlBinaryArchiveStore::instance().archive_for(dev)) {
                MTLComputePipelineDescriptor* pdesc = [[MTLComputePipelineDescriptor alloc] init];
                pdesc.computeFunction = fn;
                pdesc.binaryArchives = @[ar];
                pso = [dev newComputePipelineStateWithDescriptor:pdesc
                                                         options:MTLPipelineOptionNone
                                                      reflection:nil
                                                           error:&perr];
                if (pso) {
                    MtlBinaryArchiveStore::instance().record_pipeline(dev, pdesc, msl_source, function_name);
                    return pso;
                }
                perr = nil;
            }
        }
        pso = [dev newComputePipelineStateWithFunction:fn error:&perr];
        if (!pso) {
            std::string m = "Metal: pipeline build failed for ";
            m += [function_name UTF8String];
            if (perr) { m += ": "; m += [[perr localizedDescription] UTF8String]; }
            throw std::runtime_error(m);
        }
        return pso;
    }
}

}  // namespace brotensor::metal_impl
