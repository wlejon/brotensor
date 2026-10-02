// HIP graph capture / replay (public API in include/brotensor/cuda_graph.h).
//
// The HIP counterpart of src/cuda/cuda_graph.cu: CudaGraphCapture brackets a
// fixed-shape op sequence on a dedicated non-blocking capture stream (made the
// current stream, so every op and the allocator enqueue onto it), finish() ends
// capture and instantiates, CudaGraph replays with one hipGraphLaunch. What
// differs is memory: a capture owns a CaptureArena (detail/capture_arena.h)
// that serves every allocation made inside it, so the graph never records the
// MemAlloc / MemFree nodes ROCm replays wrongly, and the instantiated graph
// keeps the arena alive for as long as it can be launched. Launches also go
// through a window that bounds the graph work in flight (see LaunchWindow).

#include <brotensor/cuda_graph.h>

#include "detail/capture_arena.h"
#include "detail/hip_check.h"

#include <hip/hip_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <deque>
#include <string>
#include <map>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace brotensor {
// HIP-internal current-stream hooks (defined in src/hip/runtime.hip).
void* hip_current_stream(int dev);
void hip_set_stream(void* stream, int dev);
}

namespace brotensor::detail::hip {

// ─── CaptureArena ───────────────────────────────────────────────────────────

class CaptureArena {
public:
    explicit CaptureArena(int dev) : dev_(dev) {}
    ~CaptureArena();

    int device() const { return dev_; }

    std::unordered_set<void*> held;              // blocks this graph may touch
    std::multimap<std::size_t, void*> reusable;  // freed inside the capture
    bool sealed = false;

private:
    int dev_;
};

namespace {

// Every block some live arena holds — or that a dead arena handed back while
// the caller still held it. A block can be held by several graphs (one
// captured with it, a later one whose capture freed it); it is released when
// the last of them dies and the caller has freed it.
struct Block {
    std::size_t bytes = 0;  // 0 for memory that predates every capture
    int holders = 0;        // live arenas holding it
    bool live = true;       // the caller still holds it
    bool plain = true;      // a hipMalloc of ours (else: the ordinary free path)
};
std::mutex g_mu;
std::unordered_map<void*, Block> g_blocks;
std::atomic<std::size_t> g_tracked{0};  // g_blocks.size(), read without g_mu

// The calling thread's active arena (non-owning; the capture scope owns it)
// and, per active arena, the one it displaced.
thread_local CaptureArena* t_active = nullptr;
thread_local std::unordered_map<CaptureArena*, CaptureArena*> t_prev;

// Arena blocks are rounded up so in-capture reuse finds a fit more often.
constexpr std::size_t kGranule = 512;

// hipMalloc / hipFree are "unsafe" calls the capturing thread may not make in
// ThreadLocal mode (error 900); relaxing the thread's mode around them is the
// documented escape, and neither touches the capture stream.
struct RelaxedCaptureMode {
    hipStreamCaptureMode mode = hipStreamCaptureModeRelaxed;
    RelaxedCaptureMode() { (void)hipThreadExchangeStreamCaptureMode(&mode); }
    ~RelaxedCaptureMode() { (void)hipThreadExchangeStreamCaptureMode(&mode); }
};

struct DeviceGuard {
    int prior = 0;
    int dev;
    explicit DeviceGuard(int d) : dev(d) {
        (void)hipGetDevice(&prior);
        if (prior != dev) (void)hipSetDevice(dev);
    }
    ~DeviceGuard() {
        if (prior != dev) (void)hipSetDevice(prior);
    }
};

Block& track(void* p, const Block& b) {
    auto [it, inserted] = g_blocks.emplace(p, b);
    if (inserted) g_tracked.fetch_add(1, std::memory_order_relaxed);
    return it->second;
}

// Caller holds g_mu; `p` is held by no arena and freed by the caller.
void release(void* p, int dev) {
    const bool plain = g_blocks[p].plain;
    g_blocks.erase(p);
    g_tracked.fetch_sub(1, std::memory_order_relaxed);
    DeviceGuard dg(dev);
    RelaxedCaptureMode relaxed;
    if (plain) {
        (void)hipFree(p);
        return;
    }
    // The ordinary path frees on the current stream; if this thread is itself
    // capturing, point it at the legacy stream so no free node is recorded.
    void* cur = hip_current_stream(dev);
    hip_set_stream(nullptr, dev);
    hip_free_untracked(p, dev);
    hip_set_stream(cur, dev);
}

}  // namespace

CaptureArena::~CaptureArena() {
    // The graph is gone, but a launch of it may still be running: drain the
    // device before any of its memory is released.
    {
        DeviceGuard dg(dev_);
        RelaxedCaptureMode relaxed;
        (void)hipDeviceSynchronize();
    }
    std::lock_guard<std::mutex> lk(g_mu);
    for (void* p : held) {
        auto it = g_blocks.find(p);
        if (it == g_blocks.end() || --it->second.holders > 0) continue;
        // Still held by the caller: handed back, its free releases it.
        if (!it->second.live) release(p, dev_);
    }
    (void)hipGetLastError();
}

std::shared_ptr<CaptureArena> capture_arena_begin(int dev) {
    auto a = std::make_shared<CaptureArena>(dev);
    t_prev[a.get()] = t_active;
    t_active = a.get();
    return a;
}

void capture_arena_end(const std::shared_ptr<CaptureArena>& arena) {
    if (!arena) return;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        arena->sealed = true;
        arena->reusable.clear();
    }
    auto it = t_prev.find(arena.get());
    if (it != t_prev.end()) {
        if (t_active == arena.get()) t_active = it->second;
        t_prev.erase(it);
    }
}

void* capture_arena_alloc(std::size_t bytes, int dev) {
    CaptureArena* a = t_active;
    if (!a || a->device() != dev) return nullptr;
    bytes = (bytes + kGranule - 1) / kGranule * kGranule;
    std::lock_guard<std::mutex> lk(g_mu);
    auto fit = a->reusable.lower_bound(bytes);
    if (fit != a->reusable.end()) {
        void* p = fit->second;
        a->reusable.erase(fit);
        g_blocks[p].live = true;
        return p;
    }
    void* p = nullptr;
    {
        DeviceGuard dg(dev);
        RelaxedCaptureMode relaxed;
        BROTENSOR_HIP_CHECK(hipMalloc(&p, bytes));
    }
    track(p, Block{bytes, 1, true, true});
    a->held.insert(p);
    return p;
}

bool capture_arena_free(void* ptr, int dev, bool capturing) {
    CaptureArena* active = capturing ? t_active : nullptr;
    if (active && active->device() != dev) active = nullptr;
    if (!active && g_tracked.load(std::memory_order_relaxed) == 0) return false;

    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_blocks.find(ptr);
    if (!active) {
        if (it == g_blocks.end()) return false;
        it->second.live = false;
        if (it->second.holders == 0) release(ptr, dev);  // a handed-back block
        return true;
    }
    if (it == g_blocks.end()) {
        // Predates the capture, so captured kernels may read it on every
        // replay: keep it until the graph dies, then free it the ordinary way.
        track(ptr, Block{0, 1, false, false});
        active->held.insert(ptr);
        return true;
    }
    Block& b = it->second;
    b.live = false;
    if (active->held.insert(ptr).second) {
        // Another graph's block (or a handed-back one) this capture may read.
        ++b.holders;
    } else if (b.holders == 1 && b.bytes != 0) {
        // Allocated and freed inside this capture: later captured work may
        // reuse it, ordered after its last user by the linear capture chain.
        active->reusable.emplace(b.bytes, ptr);
    }
    return true;
}

}  // namespace brotensor::detail::hip

namespace brotensor {

namespace dh = detail::hip;

// ─── Launch window ──────────────────────────────────────────────────────────
//
// ROCm 7.2's hipGraphLaunch can deadlock once the graph work pending on the
// device outgrows the AQL queue (ROC_AQL_QUEUE_SIZE packets, 16384 by
// default): the launch waits for queue space that never frees, the device
// sits idle and the next synchronize never returns. Eager launches past the
// queue size are fine; only graph work hangs. On ROCm 7.2.4 / gfx1151 a
// 1000-kernel graph launched 20 times back to back hung within a few hundred
// rounds, as did one 20000-kernel graph launched alone; with
// ROC_AQL_QUEUE_SIZE=65536 the former ran clean, at 4096 a 500-kernel graph
// x 20 hung, and synchronizing after every launch never hung. (Supertonic's
// ~3400-node flow step replayed 7 times per utterance hit it every few dozen
// utterances.) So launches keep the graph nodes in flight per device under
// half the queue, waiting on the oldest launch when the next would cross it,
// and a graph too big to launch safely even alone is refused at finish().

namespace {

std::size_t aql_queue_packets() {
    static const std::size_t q = [] {
        const char* e = std::getenv("ROC_AQL_QUEUE_SIZE");
        const long v = e ? std::strtol(e, nullptr, 10) : 0;
        return v > 0 ? static_cast<std::size_t>(v) : std::size_t{16384};
    }();
    return q;
}

struct LaunchWindow {
    struct Pending {
        hipEvent_t done;
        std::size_t nodes;
    };
    std::mutex mu;
    std::deque<Pending> pending;
    std::vector<hipEvent_t> spare;
    std::size_t nodes = 0;

    void retire_front() {
        nodes -= pending.front().nodes;
        spare.push_back(pending.front().done);
        pending.pop_front();
    }

    void launch(hipGraphExec_t exec, std::size_t graph_nodes, hipStream_t stream) {
        std::lock_guard<std::mutex> lk(mu);
        while (!pending.empty()) {
            const hipError_t q = hipEventQuery(pending.front().done);
            if (q == hipErrorNotReady) {
                (void)hipGetLastError();  // not an error for the caller's next check
                break;
            }
            retire_front();
        }
        const std::size_t budget = aql_queue_packets() / 2;
        while (!pending.empty() && nodes + graph_nodes > budget) {
            BROTENSOR_HIP_CHECK(hipEventSynchronize(pending.front().done));
            retire_front();
        }
        hipEvent_t ev = nullptr;
        if (!spare.empty()) {
            ev = spare.back();
            spare.pop_back();
        } else {
            BROTENSOR_HIP_CHECK(hipEventCreateWithFlags(&ev, hipEventDisableTiming));
        }
        hipError_t err = hipGraphLaunch(exec, stream);
        if (err == hipSuccess) err = hipEventRecord(ev, stream);
        if (err != hipSuccess) {
            spare.push_back(ev);
            BROTENSOR_HIP_CHECK(err);
        }
        pending.push_back({ev, graph_nodes});
        nodes += graph_nodes;
    }
};

constexpr int kMaxDevices = 16;
LaunchWindow g_windows[kMaxDevices];

}  // namespace

// ─── CudaGraph ──────────────────────────────────────────────────────────────

struct CudaGraph::Impl {
    hipGraphExec_t exec = nullptr;
    int dev = 0;
    std::size_t nodes = 0;
    std::shared_ptr<dh::CaptureArena> arena;  // outlives exec (member order)
    ~Impl() {
        if (exec) (void)hipGraphExecDestroy(exec);
    }
};

CudaGraph::CudaGraph() = default;
CudaGraph::~CudaGraph() = default;
CudaGraph::CudaGraph(CudaGraph&&) noexcept = default;
CudaGraph& CudaGraph::operator=(CudaGraph&&) noexcept = default;

bool CudaGraph::valid() const { return impl_ && impl_->exec != nullptr; }

void CudaGraph::reset() { impl_.reset(); }

void CudaGraph::launch() {
    if (!valid()) {
        throw std::runtime_error("brotensor: CudaGraph::launch: no captured graph");
    }
    auto stream = reinterpret_cast<hipStream_t>(hip_current_stream(impl_->dev));
    g_windows[impl_->dev % kMaxDevices].launch(impl_->exec, impl_->nodes, stream);
}

// ─── CudaGraphCapture ───────────────────────────────────────────────────────

struct CudaGraphCapture::Impl {
    hipStream_t stream = nullptr;
    void* prev_stream = nullptr;
    int dev = 0;
    bool capturing = false;
    std::shared_ptr<dh::CaptureArena> arena;
};

CudaGraphCapture::CudaGraphCapture() : impl_(new Impl) {
    BROTENSOR_HIP_CHECK(hipGetDevice(&impl_->dev));
    // Non-blocking, as on CUDA: a blocking stream may not capture while another
    // thread works on the legacy stream.
    BROTENSOR_HIP_CHECK(hipStreamCreateWithFlags(&impl_->stream, hipStreamNonBlocking));
    impl_->prev_stream = hip_current_stream(impl_->dev);
    // Route subsequent ops (and allocations) onto the capture stream.
    hip_set_stream(impl_->stream, impl_->dev);
    impl_->arena = dh::capture_arena_begin(impl_->dev);
    // ThreadLocal mode: only this thread's work is captured, so unrelated host
    // threads issuing HIP work don't poison the capture.
    hipError_t err = hipStreamBeginCapture(impl_->stream, hipStreamCaptureModeThreadLocal);
    if (err != hipSuccess) {
        dh::capture_arena_end(impl_->arena);
        impl_->arena.reset();
        hip_set_stream(impl_->prev_stream, impl_->dev);
        (void)hipStreamDestroy(impl_->stream);
        impl_->stream = nullptr;
        BROTENSOR_HIP_CHECK(err);
    }
    impl_->capturing = true;
}

CudaGraph CudaGraphCapture::finish() {
    if (!impl_ || !impl_->capturing) {
        throw std::runtime_error("brotensor: CudaGraphCapture::finish: not capturing");
    }
    hipGraph_t graph = nullptr;
    hipError_t end_err = hipStreamEndCapture(impl_->stream, &graph);
    impl_->capturing = false;
    dh::capture_arena_end(impl_->arena);
    hip_set_stream(impl_->prev_stream, impl_->dev);
    BROTENSOR_HIP_CHECK(end_err);

    std::size_t nodes = 0;
    hipError_t err = hipGraphGetNodes(graph, nullptr, &nodes);
    // Even alone on an idle queue a graph near the queue size can hang (see
    // the launch window): refuse it rather than deadlock on its first launch.
    if (err == hipSuccess && nodes > aql_queue_packets() * 3 / 4) {
        (void)hipGraphDestroy(graph);
        throw std::runtime_error(
            "brotensor: CudaGraphCapture::finish: " + std::to_string(nodes) +
            "-node graph exceeds 3/4 of the " + std::to_string(aql_queue_packets()) +
            "-packet AQL queue, which hipGraphLaunch can deadlock on; capture a "
            "smaller region or raise ROC_AQL_QUEUE_SIZE");
    }
    hipGraphExec_t exec = nullptr;
    if (err == hipSuccess) err = hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    (void)hipGraphDestroy(graph);
    BROTENSOR_HIP_CHECK(err);

    CudaGraph g;
    g.impl_.reset(new CudaGraph::Impl);
    g.impl_->exec = exec;
    g.impl_->dev = impl_->dev;
    g.impl_->nodes = nodes;
    g.impl_->arena = std::move(impl_->arena);
    return g;
}

CudaGraphCapture::~CudaGraphCapture() {
    if (!impl_) return;
    if (impl_->capturing) {
        // Abort: drain the in-flight capture and discard the graph.
        hipGraph_t graph = nullptr;
        (void)hipStreamEndCapture(impl_->stream, &graph);
        if (graph) (void)hipGraphDestroy(graph);
        dh::capture_arena_end(impl_->arena);
        hip_set_stream(impl_->prev_stream, impl_->dev);
        (void)hipGetLastError();
    }
    impl_->arena.reset();
    if (impl_->stream) (void)hipStreamDestroy(impl_->stream);
}

}  // namespace brotensor
