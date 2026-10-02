#pragma once

// Internal header — the graph-private allocation arena behind CudaGraphCapture
// on HIP (implemented in src/hip/graph.hip, consulted by src/hip/tensor.hip).
//
// Why it exists: ROCm's graph memory nodes are not reliable. A captured
// hipMallocAsync / hipFreeAsync becomes a MemAlloc / MemFree node that the
// runtime maps and unmaps physical pages for on every replay (VM mode, the
// ROCm 7.2 default), and replaying such a graph reads wrong values or faults on
// an unmapped page — on ROCm 7.2.4 / gfx1151 a captured chain of 100 alloc/
// free pairs replayed wrong in most fresh graphs and 700 pairs faulted outright,
// while the same kernels over pre-allocated buffers were always exact. So a
// HIP capture never records memory nodes: every allocation made inside it is a
// plain hipMalloc into an arena the graph owns, and every free inside it is
// deferred to the arena. That is the CUDA graph-memory contract (the memory a
// replay touches stays put for the graph's lifetime) without the nodes.
//
//  - An allocation inside the capture takes a block the same capture already
//    freed (a single captured stream is a linear chain, so the earlier user's
//    last kernel precedes the next user's first one on every replay) or a new
//    hipMalloc made with the thread's capture mode relaxed.
//  - A free inside the capture of an arena block returns it for reuse; a free
//    of memory that predates the capture (captured kernels may still read it)
//    is adopted and released only when the graph dies.
//  - After capture, a free of an arena block is deferred to the graph's death;
//    a block the caller still holds when the graph dies is handed back and
//    hipFree'd by the caller's eventual free.

#include <cstddef>
#include <memory>

namespace brotensor::detail::hip {

class CaptureArena;

// Begin an arena for a capture on `dev` and make it the calling thread's
// active one (the previous active arena, if any, is restored by end).
std::shared_ptr<CaptureArena> capture_arena_begin(int dev);
// Stop routing the calling thread's allocations into `arena`: no further
// in-capture reuse, later frees of its blocks are deferred to its death.
void capture_arena_end(const std::shared_ptr<CaptureArena>& arena);

// hip_alloc while the current stream is capturing: a block from the calling
// thread's active arena on `dev`, or nullptr when there is none (the caller
// then falls back to its own path).
void* capture_arena_alloc(std::size_t bytes, int dev);
// hip_free hook: true when `ptr` was taken care of here (recycled, deferred,
// adopted, or a handed-back block freed), false for the ordinary path.
bool capture_arena_free(void* ptr, int dev, bool capturing);

// The ordinary free path with no arena check (src/hip/tensor.hip): how an
// adopted pre-capture allocation is finally released.
void hip_free_untracked(void* ptr, int dev);

}  // namespace brotensor::detail::hip
