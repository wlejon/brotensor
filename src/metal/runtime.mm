#import "internal.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Declared at global scope; implemented in kernels.mm.
extern void metal_build_pipelines_();

namespace brotensor {

namespace {
std::atomic<bool> g_initialized{false};

id<MTLDevice>       g_device = nil;
id<MTLCommandQueue> g_queue  = nil;

void init_runtime_once() {
    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            throw std::runtime_error(
                "brotensor::cuda_init: no Metal device found");
        }
        g_queue = [g_device newCommandQueue];
        if (!g_queue) {
            throw std::runtime_error(
                "brotensor::cuda_init: failed to create MTLCommandQueue");
        }
    }
}
} // namespace

void cuda_init() {
    if (g_initialized.load(std::memory_order_acquire)) return;
    static std::atomic<bool> in_progress{false};
    bool expected = false;
    if (!in_progress.compare_exchange_strong(expected, true)) {
        while (!g_initialized.load(std::memory_order_acquire)) {}
        return;
    }
    init_runtime_once();
    ::metal_build_pipelines_();
    g_initialized.store(true, std::memory_order_release);
    in_progress.store(false, std::memory_order_release);
}

void cuda_sync() {
    // Ops submit command buffers asynchronously (see metal_impl::submit);
    // draining the queue means waiting on the most recent pending buffer.
    metal_impl::flush();
}

// Metal has its own queue model and no per-thread stream concept; these are
// no-ops so source compiles across backends. We keep the same thread_local
// scaffolding so cuda_current_stream() round-trips whatever was set.
namespace {
thread_local void* g_current_stream_metal = nullptr;
} // namespace

void cuda_set_stream(void* stream) {
    g_current_stream_metal = stream;
}

void* cuda_current_stream() {
    return g_current_stream_metal;
}

void cuda_stream_sync(void* /*stream*/) {
    // No-op: ops submit synchronously per command buffer.
}

void cuda_check_throw(int err, const char* expr_text, const char* file, int line) {
    if (err == 0) return;
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
                  "Metal/runtime error %d at %s:%d in `%s`",
                  err, file ? file : "?", line,
                  expr_text ? expr_text : "?");
    throw std::runtime_error(buf);
}

namespace metal_impl {

id<MTLDevice> device() {
    // No reentry into cuda_init() here: device()/queue() are called from
    // metal_build_pipelines_() which itself runs *during* cuda_init, when
    // g_initialized is still false. init_runtime_once() has already set
    // g_device by the time we reach this point, so a direct return is safe.
    if (!g_device) cuda_init();
    return g_device;
}

id<MTLCommandQueue> queue() {
    if (!g_queue) cuda_init();
    return g_queue;
}

// ─── Batched command submission ────────────────────────────────────────────
//
// The op layer used to give every op its own command buffer and block on
// waitUntilCompleted before returning. For an SD U-Net forward — thousands of
// tiny kernels — that per-op CPU<->GPU round-trip dominated wall time.
//
// Deferring the wait but keeping one command buffer per op is NOT correct:
// Metal only guarantees that a kernel observes prior writes when the producer
// and consumer share a command buffer (hazard tracking spans command encoders
// within a buffer) or are separated by an explicit wait. Two command buffers
// committed back-to-back without a wait can execute such that the second
// misses the first's writes.
//
// So we batch: many ops encode into one shared "open" command buffer (each op
// still creates its own command encoder; Metal hazard-tracks between encoders
// in the same buffer). Every kBatch ops the open buffer is committed and a new
// one started. To overlap GPU and CPU, we keep the previously committed batch
// running ("committed") while the CPU encodes the next one, and wait on it
// only just before committing its successor — that wait is also what makes
// cross-batch reads safe. flush() commits and drains everything; brotensor
// calls it before every device->host transfer and inside sync().
//
// Command buffers are held with a manual +1 retain: the queue keeps a
// committed buffer alive only until completion, which can race our wait.
namespace {

thread_local void* g_open      = nullptr;  // +1-retained, uncommitted
thread_local void* g_committed = nullptr;  // +1-retained, committed, not waited
thread_local int   g_op_count  = 0;

// Ops per command buffer. Large enough that the per-batch commit+wait stall
// amortizes to nothing; small enough to bound encoded-command memory and the
// number of resources a single buffer pins until it completes. Overridable
// with BROTENSOR_METAL_BATCH (1 = one command buffer per op, the old
// fully-synchronous behavior — useful for A/B debugging).
int batch_size() {
    static const int n = [] {
        const char* e = getenv("BROTENSOR_METAL_BATCH");
        const int v = e ? atoi(e) : 256;
        return v > 0 ? v : 256;
    }();
    return n;
}

// Wait on (and release) the previously committed batch, if any.
void wait_committed() {
    if (!g_committed) return;
    id<MTLCommandBuffer> c = (__bridge_transfer id<MTLCommandBuffer>)g_committed;
    g_committed = nullptr;
    [c waitUntilCompleted];
}

// Commit the open batch. First wait on the prior committed batch so the one
// we are about to commit cannot start before its predecessor's writes land —
// this is the cross-batch ordering guarantee. The just-committed batch then
// runs on the GPU while the CPU encodes the next one.
void commit_open() {
    if (!g_open) return;
    wait_committed();
    id<MTLCommandBuffer> o = (__bridge id<MTLCommandBuffer>)g_open;
    [o commit];
    g_committed = g_open;   // move the +1 retain
    g_open      = nullptr;
    g_op_count  = 0;
}

} // namespace

id<MTLCommandBuffer> new_command_buffer() {
    if (!g_open) {
        id<MTLCommandBuffer> c = [queue() commandBuffer];
        g_open = (__bridge_retained void*)c;
    }
    return (__bridge id<MTLCommandBuffer>)g_open;
}

void submit(id<MTLCommandBuffer> /*cmd*/) {
    // The op has finished encoding into the open command buffer (returned by
    // new_command_buffer()). Just count it; commit once the batch is full.
    if (++g_op_count >= batch_size()) commit_open();
}

void flush() {
    commit_open();     // commit whatever is still open
    wait_committed();  // drain the GPU
}

void run_graph_sync_off(MPSGraph* g,
                        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds,
                        MPSGraphTensor* result,
                        id<MTLBuffer> result_buffer,
                        NSUInteger result_offset_bytes,
                        NSArray<NSNumber*>* result_shape) {
    @autoreleasepool {
        MPSGraphTensorData* result_td = tensor_data_for_offset(
            result_buffer, result_offset_bytes, result_shape, MPSDataTypeFloat32);
        MPSCommandBuffer* mps_cmd =
            [MPSCommandBuffer commandBufferFromCommandQueue:queue()];
        NSDictionary* results = @{ result : result_td };
        [g encodeToCommandBuffer:mps_cmd
                           feeds:feeds
                targetOperations:nil
               resultsDictionary:results
             executionDescriptor:nil];
        [mps_cmd commit];
        [mps_cmd waitUntilCompleted];
    }
}

void run_graph_sync(MPSGraph* g,
                    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds,
                    MPSGraphTensor* result,
                    id<MTLBuffer> result_buffer,
                    NSArray<NSNumber*>* result_shape) {
    run_graph_sync_off(g, feeds, result, result_buffer, 0, result_shape);
}

MPSGraphTensorData* tensor_data_for(id<MTLBuffer> buf,
                                    NSArray<NSNumber*>* shape,
                                    MPSDataType dt) {
    return [[MPSGraphTensorData alloc] initWithMTLBuffer:buf
                                                   shape:shape
                                                dataType:dt];
}

MPSGraphTensorData* tensor_data_for_offset(id<MTLBuffer> buf,
                                           NSUInteger offset_bytes,
                                           NSArray<NSNumber*>* shape,
                                           MPSDataType dt) {
    if (offset_bytes == 0) {
        return [[MPSGraphTensorData alloc] initWithMTLBuffer:buf
                                                       shape:shape
                                                    dataType:dt];
    }
    MPSShape* mpsShape = shape;
    MPSNDArrayDescriptor* desc =
        [MPSNDArrayDescriptor descriptorWithDataType:dt shape:mpsShape];
    MPSNDArray* nd = [[MPSNDArray alloc] initWithBuffer:buf
                                                 offset:offset_bytes
                                             descriptor:desc];
    return [[MPSGraphTensorData alloc] initWithMPSNDArray:nd];
}

} // namespace metal_impl

} // namespace brotensor
