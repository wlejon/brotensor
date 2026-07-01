// Coverage for the CPU backend's lock-free thread pool
// (brotensor::detail::cpu::ThreadPool / parallel_for): every index in
// [0, n) is visited exactly once, results are correct across a range of n
// (including 0, 1, and counts on both sides of the worker count), nested
// calls don't deadlock, and repeated calls don't leak or corrupt state.
// Run under a sanitizer-free plain build a few hundred times via a shell
// loop if chasing a suspected race — this single run only catches races
// that reproduce deterministically or by luck.

#include <brotensor/detail/cpu/thread_pool.h>

#include <atomic>
#include <cstdio>
#include <numeric>
#include <vector>

static int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

using brotensor::detail::cpu::parallel_for;
using brotensor::detail::cpu::ThreadPool;

static void test_visits_every_index_once() {
    std::printf("test_visits_every_index_once\n");
    for (std::size_t n : {std::size_t(0), std::size_t(1), std::size_t(2),
                           std::size_t(7), std::size_t(64), std::size_t(997),
                           std::size_t(100000)}) {
        std::vector<std::atomic<int>> hits(n);
        for (auto& h : hits) h.store(0, std::memory_order_relaxed);
        parallel_for(n, [&](std::size_t i) {
            hits[i].fetch_add(1, std::memory_order_relaxed);
        });
        bool all_one = true;
        for (std::size_t i = 0; i < n; ++i) {
            if (hits[i].load(std::memory_order_relaxed) != 1) { all_one = false; break; }
        }
        CHECK(all_one);
    }
}

static void test_accumulation_matches_serial_reference() {
    std::printf("test_accumulation_matches_serial_reference\n");
    const std::size_t n = 50000;
    std::vector<double> data(n);
    for (std::size_t i = 0; i < n; ++i) data[i] = static_cast<double>(i) * 0.5 + 1.0;

    std::vector<double> out_parallel(n, 0.0);
    parallel_for(n, [&](std::size_t i) { out_parallel[i] = data[i] * data[i]; });

    std::vector<double> out_serial(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) out_serial[i] = data[i] * data[i];

    CHECK(out_parallel == out_serial);
}

static void test_repeated_calls_are_consistent() {
    std::printf("test_repeated_calls_are_consistent\n");
    const std::size_t n = 4096;
    for (int round = 0; round < 200; ++round) {
        std::vector<int> out(n, -1);
        parallel_for(n, [&](std::size_t i) { out[i] = static_cast<int>(i); });
        bool ok = true;
        for (std::size_t i = 0; i < n; ++i) {
            if (out[i] != static_cast<int>(i)) { ok = false; break; }
        }
        CHECK(ok);
        if (!ok) break;
    }
}

static void test_nested_calls_do_not_deadlock() {
    std::printf("test_nested_calls_do_not_deadlock\n");
    // The pool documents a single-job-in-flight contract from one calling
    // thread; a call to run() from *inside* the calling thread's own share
    // of outer work (not from a spawned worker) must still complete, since
    // it's sequential with respect to the outer job on that thread.
    const std::size_t outer = 8;
    std::atomic<int> total{0};
    parallel_for(outer, [&](std::size_t /*i*/) {
        parallel_for(16, [&](std::size_t /*j*/) {
            total.fetch_add(1, std::memory_order_relaxed);
        });
    });
    CHECK(total.load() == static_cast<int>(outer * 16));
}

static void test_num_threads_positive() {
    std::printf("test_num_threads_positive\n");
    CHECK(ThreadPool::instance().num_threads() >= 1);
}

int main() {
    test_visits_every_index_once();
    test_accumulation_matches_serial_reference();
    test_repeated_calls_are_consistent();
    test_nested_calls_do_not_deadlock();
    test_num_threads_positive();

    if (g_failures == 0) {
        std::printf("\nall thread_pool tests passed\n");
        return 0;
    }
    std::printf("\n%d failure(s)\n", g_failures);
    return 1;
}
