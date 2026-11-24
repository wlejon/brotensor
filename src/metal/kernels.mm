#import "internal.h"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace brotensor::metal_impl {

id<MTLComputePipelineState> pipeline(NSString* name) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "Metal: unknown kernel `%s` (precompiled cache removed)",
                  [name UTF8String]);
    throw std::runtime_error(buf);
}

id<MTLComputePipelineState> compile_pipeline(NSString* msl_source,
                                             NSString* function_name) {
    @autoreleasepool {
        NSError* err = nil;
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        opts.languageVersion = MTLLanguageVersion3_1;
        id<MTLLibrary> lib = [device() newLibraryWithSource:msl_source
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
        NSError* perr = nil;
        id<MTLComputePipelineState> pso =
            [device() newComputePipelineStateWithFunction:fn error:&perr];
        if (!pso) {
            std::string m = "Metal: pipeline build failed for ";
            m += [function_name UTF8String];
            if (perr) { m += ": "; m += [[perr localizedDescription] UTF8String]; }
            throw std::runtime_error(m);
        }
        return pso;
    }
}

void dispatch1d_sync(NSString* pipeline_name,
                     NSUInteger n,
                     void (^bind)(id<MTLComputeCommandEncoder>)) {
    if (n == 0) return;
    @autoreleasepool {
        id<MTLComputePipelineState> pso = pipeline(pipeline_name);
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        bind(enc);
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

}  // namespace brotensor::metal_impl

extern "C++" void metal_build_pipelines_();
void metal_build_pipelines_() {}
