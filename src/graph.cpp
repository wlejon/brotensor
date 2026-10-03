// Device-neutral graph capture (public API: include/brotensor/cuda_graph.h).
// CudaGraphCapture picks a device and opens a recorder from the factory its
// backend registered (detail/graph_backend.h); CudaGraph owns what the
// recorder's finish() returned. All backend mechanics stay in the backends.

#include <brotensor/cuda_graph.h>
#include <brotensor/detail/dispatch.h>
#include <brotensor/detail/graph_backend.h>
#include <brotensor/runtime.h>

#include <array>
#include <stdexcept>
#include <string>

namespace brotensor {

namespace detail {

namespace {

// Written by the backends' probes (single-threaded, inside init()), read by
// capture scopes; plain function pointers, zero-initialised before any static
// constructor runs.
std::array<GraphRecorderFactory, 5> g_factories{};

int slot(DeviceType dt) { return static_cast<int>(dt); }

}  // namespace

void register_graph_backend(DeviceType dt, GraphRecorderFactory factory) {
    const int i = slot(dt);
    if (i < 0 || i >= static_cast<int>(g_factories.size())) {
        throw std::runtime_error("brotensor: register_graph_backend: bad device type");
    }
    g_factories[static_cast<std::size_t>(i)] = factory;
}

GraphRecorderFactory graph_backend(DeviceType dt) {
    const int i = slot(dt);
    if (i < 0 || i >= static_cast<int>(g_factories.size())) return nullptr;
    return g_factories[static_cast<std::size_t>(i)];
}

}  // namespace detail

namespace {

bool capturable(Device d) {
    return d.is_gpu() && detail::graph_backend(d.type) != nullptr && detail::is_registered(d);
}

// The device a default-constructed capture records on, as (type, index) with
// index -1 for "the backend's current device". Throws when there is none.
Device pick_capture_device() {
    const Device d = detail::resolve_device_alias(default_device());
    if (capturable(d)) return d;
    // A CPU / Metal default device: the capture used to be CUDA's current
    // stream whatever the default device was; keep that, then Vulkan.
    if (capturable(Device::cuda(0))) return Device(DeviceType::CUDA, -1);
    if (capturable(Device::vulkan(0))) return Device::vulkan(0);
    throw std::runtime_error(
        "brotensor: CudaGraphCapture: no GPU with graph capture (CUDA or Vulkan) is "
        "registered; default device is " + to_string(d));
}

}  // namespace

bool graph_capture_available(Device d) { return capturable(detail::resolve_device_alias(d)); }

bool graph_capture_available() {
    try {
        (void)pick_capture_device();
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

// ─── CudaGraph ──────────────────────────────────────────────────────────────

struct CudaGraph::Impl {
    Device device = Device::cpu();
    std::unique_ptr<detail::GraphExec> exec;
};

CudaGraph::CudaGraph() = default;
CudaGraph::~CudaGraph() = default;
CudaGraph::CudaGraph(CudaGraph&&) noexcept = default;
CudaGraph& CudaGraph::operator=(CudaGraph&&) noexcept = default;

bool CudaGraph::valid() const { return impl_ && impl_->exec != nullptr; }

void CudaGraph::reset() { impl_.reset(); }

Device CudaGraph::device() const { return impl_ ? impl_->device : Device::cpu(); }

void CudaGraph::launch() {
    if (!valid()) {
        throw std::runtime_error("brotensor: CudaGraph::launch: no captured graph");
    }
    impl_->exec->launch();
}

// ─── CudaGraphCapture ───────────────────────────────────────────────────────

struct CudaGraphCapture::Impl {
    Device device = Device::cpu();
    std::unique_ptr<detail::GraphRecorder> recorder;
};

namespace {

std::unique_ptr<detail::GraphRecorder> open_recorder(Device d) {
    detail::GraphRecorderFactory f = detail::graph_backend(d.type);
    if (!f) {
        throw std::runtime_error("brotensor: CudaGraphCapture: " + to_string(d) +
                                 " has no graph capture");
    }
    auto r = f(d.index);
    if (!r) throw std::runtime_error("brotensor: CudaGraphCapture: the backend refused to capture");
    return r;
}

}  // namespace

CudaGraphCapture::CudaGraphCapture() : impl_(new Impl) {
    const Device d = pick_capture_device();
    impl_->recorder = open_recorder(d);
    impl_->device = Device(d.type, impl_->recorder->device_index());
}

CudaGraphCapture::CudaGraphCapture(Device d) : impl_(new Impl) {
    d = detail::resolve_device_alias(d);
    if (!capturable(d)) {
        throw std::runtime_error("brotensor: CudaGraphCapture: " + to_string(d) +
                                 " is not a registered GPU with graph capture");
    }
    impl_->recorder = open_recorder(d);
    impl_->device = Device(d.type, impl_->recorder->device_index());
}

CudaGraphCapture::~CudaGraphCapture() = default;   // recorder's dtor aborts

Device CudaGraphCapture::device() const {
    return impl_ ? impl_->device : Device::cpu();
}

CudaGraph CudaGraphCapture::finish() {
    if (!impl_ || !impl_->recorder) {
        throw std::runtime_error("brotensor: CudaGraphCapture::finish: not capturing");
    }
    auto rec = std::move(impl_->recorder);
    CudaGraph g;
    g.impl_.reset(new CudaGraph::Impl);
    g.impl_->device = impl_->device;
    g.impl_->exec = rec->finish();
    return g;
}

}  // namespace brotensor
