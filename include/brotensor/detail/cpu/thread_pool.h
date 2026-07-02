#pragma once

// Minimal lock-free thread pool for parallelizing embarrassingly-parallel
// CPU ops (matmul's row axis, a norm's batch*channel axis, attention's
// batch*head axis, ...). Per this project's convention (atomics / single-
// owner / queues, never a lock), work handoff uses std::atomic<T>::wait /
// notify (C++20) instead of a condition_variable + mutex, and work items
// are claimed via an atomic fetch_add cursor rather than a guarded queue.
//
// Usage: brotensor::detail::cpu::parallel_for(n, [&](std::size_t i) { ... });
// blocks the calling thread until fn(i) has run for every i in [0, n) —
// across the pool's workers plus the calling thread itself, which also
// drains work instead of idling. Falls back to a plain sequential loop
// when n <= 1 or the host has no spare cores, so tiny problem sizes don't
// pay thread hand-off cost.
//
// Single job in flight at a time: ThreadPool::instance() is a process-wide
// singleton, and run() assumes it is not re-entered from a second
// concurrent application thread while a call is outstanding (matches this
// library's usage — CPU ops are driven from one calling thread; run()
// parallelizes *within* one op call, not across concurrent callers).
//
// Nested calls (an outer parallel_for's callback invoking parallel_for
// again — e.g. one op calling another that also parallelizes internally)
// are safe but not further parallelized: a per-thread flag detects that the
// calling thread is already draining a pool job and runs the nested call
// as a plain sequential loop instead of touching the shared job state a
// second time, on whichever thread hits it (the original caller or a
// worker).

#include <atomic>
#include <cstddef>
#include <functional>
#include <thread>
#include <vector>

namespace brotensor::detail::cpu {

class ThreadPool {
public:
    static ThreadPool& instance();

    void run(std::size_t n, const std::function<void(std::size_t)>& fn);

    // Workers plus the calling thread, i.e. the effective parallelism a
    // run() call can spread work across.
    int num_threads() const { return static_cast<int>(workers_.size()) + 1; }

    // Explicitly joins every worker thread. Callers should invoke this
    // deterministically during their own shutdown sequence — the pool is
    // a Meyers singleton, so its destructor only runs during the process's
    // static-destruction phase, by which point every *other* thread has
    // already been suspended by RtlExitUserProcess. A worker suspended
    // mid-operation while holding some global lock (e.g. the Debug CRT's
    // iterator-checking mutex, taken by any std::vector destructor) can
    // then deadlock the main thread's own exit-time TLS destructors
    // waiting on that same lock forever. Calling shutdown() early — before
    // any of that starts — avoids the whole class of hazard. Idempotent:
    // safe to call more than once (the eventual destructor call included).
    void shutdown();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

private:
    ThreadPool();
    ~ThreadPool();

    void worker_loop();

    std::vector<std::thread> workers_;

    // Job state, valid only while a run() call is outstanding (single job
    // in flight at a time — see class comment).
    const std::function<void(std::size_t)>* fn_ = nullptr;
    std::size_t n_ = 0;
    std::atomic<std::size_t> cursor_{0};
    std::atomic<int> outstanding_{0};  // workers still draining this job

    // Bumped once per run() call; workers block in generation_.wait(seen)
    // until it changes, then pick up the new job (or exit, on shutdown).
    std::atomic<int> generation_{0};
    std::atomic<bool> shutdown_{false};
};

inline void parallel_for(std::size_t n, const std::function<void(std::size_t)>& fn) {
    ThreadPool::instance().run(n, fn);
}

}  // namespace brotensor::detail::cpu
