#include <brotensor/detail/cpu/thread_pool.h>

namespace brotensor::detail::cpu {

namespace {
// Set while the calling thread (original caller or a worker) is inside a
// job's fn_ calls. Guards against the reentrancy hazard where fn_ itself
// invokes parallel_for again: without this, every thread currently
// executing fn_ would race to claim the singleton's shared job state for
// the nested call. A nested call instead runs as a plain sequential loop
// on whichever thread hits it.
thread_local bool t_active = false;
}  // namespace

ThreadPool::ThreadPool() {
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    // Reserve one logical core for the calling thread, which also drains
    // work instead of idling while it waits for the pool.
    const unsigned num_workers = hw > 1 ? hw - 1 : 0;
    workers_.reserve(num_workers);
    for (unsigned i = 0; i < num_workers; ++i) {
        workers_.emplace_back([this] { worker_loop(); });
    }
}

ThreadPool::~ThreadPool() {
    shutdown();
}

void ThreadPool::shutdown() {
    shutdown_.store(true, std::memory_order_relaxed);
    generation_.fetch_add(1, std::memory_order_release);
    generation_.notify_all();
    for (auto& t : workers_) {
        if (t.joinable()) t.join();
    }
    // Idempotent: an empty vector makes a second call (e.g. the eventual
    // destructor, after an explicit early shutdown()) a no-op.
    workers_.clear();
}

ThreadPool& ThreadPool::instance() {
    static ThreadPool pool;
    return pool;
}

void ThreadPool::worker_loop() {
    int seen = 0;
    for (;;) {
        generation_.wait(seen);
        seen = generation_.load(std::memory_order_acquire);
        if (shutdown_.load(std::memory_order_relaxed)) return;

        t_active = true;
        for (;;) {
            const std::size_t i = cursor_.fetch_add(1, std::memory_order_relaxed);
            if (i >= n_) break;
            (*fn_)(i);
        }
        t_active = false;
        if (outstanding_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            outstanding_.notify_all();
        }
    }
}

void ThreadPool::run(std::size_t n, const std::function<void(std::size_t)>& fn) {
    if (n == 0) return;
    if (t_active || workers_.empty() || n == 1) {
        // Either nested (this thread is already draining a job's fn_, so
        // the shared job state is in use — see class comment) or not worth
        // dispatching (no spare cores, or a single item).
        for (std::size_t i = 0; i < n; ++i) fn(i);
        return;
    }

    fn_ = &fn;
    n_ = n;
    cursor_.store(0, std::memory_order_relaxed);
    outstanding_.store(static_cast<int>(workers_.size()), std::memory_order_relaxed);

    generation_.fetch_add(1, std::memory_order_release);
    generation_.notify_all();

    // The calling thread drains work too instead of idling.
    t_active = true;
    for (;;) {
        const std::size_t i = cursor_.fetch_add(1, std::memory_order_relaxed);
        if (i >= n_) break;
        fn(i);
    }
    t_active = false;

    for (;;) {
        const int rem = outstanding_.load(std::memory_order_acquire);
        if (rem == 0) break;
        outstanding_.wait(rem);
    }
}

}  // namespace brotensor::detail::cpu
