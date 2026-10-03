// Shared by the flash-attention backward kernels (fa_bwd.comp, FMA;
// fa_bwd_cm.comp, cooperative matrix): the push block (BwdPush in
// ops_fa_bwd.cpp) and each query row's key interval. Needs MODE.

#include "op_codes.h"

layout(push_constant) uniform PC {
    uint64_t q, k, v, g;   // Q, K, V, dO (head 0, row 0)
    uint64_t mask;         // FP32 key mask (MASK)
    uint64_t aux, aux2;    // VARLEN: cu_seqlens_q / _k; PACKED: seq_bounds (lq x 2)
    uint64_t stats;        // FP32 (hq * lq) x 4: m (log2 domain), 1 / l, D, -
    uint64_t o0, o1;       // PASS 0: dQ; PASS 1: dK, dV
    uint lq, lk, ldq, ldk, ldg;
    uint ldo;              // row stride of the outputs (dQ, or dK / dV)
    uint window, causal, nseq;
    float scale;           // log2(e) / sqrt(hd)
    float oscale;          // 1 / sqrt(hd)
} pc;

// Keys [lo, hi) of query row qi (empty past lq), as fa_common.glsl's
// fa_interval with query row 0 at position 0.
void bw_interval(uint qi, out int lo, out int hi) {
    lo = 0;
    hi = 0;
    if (qi >= pc.lq) return;
    int a = 0, b = int(pc.lk);
    if (MODE == FA_MODE_ROWS) {
        const int aq = int(qi);
        if (pc.causal != 0u) {
            b = min(b, aq + 1);
            if (pc.window > 0u) a = max(0, aq - int(pc.window) + 1);
        } else if (pc.window > 0u) {
            const int w2 = int(pc.window) / 2;
            a = max(0, aq - w2);
            b = min(b, aq + w2 + 1);
        }
    } else if (MODE == FA_MODE_VARLEN) {
        const int q = int(qi);
        if (pc.nseq == 0u || q >= I32Buf(pc.aux).v[pc.nseq]) return;
        uint s0 = 0u, s1 = pc.nseq;
        while (s1 - s0 > 1u) {
            const uint mid = (s0 + s1) / 2u;
            if (I32Buf(pc.aux).v[mid] <= q) s0 = mid; else s1 = mid;
        }
        a = I32Buf(pc.aux2).v[s0];
        b = I32Buf(pc.aux2).v[s0 + 1u];
        if (pc.causal != 0u) b = min(b, a + q - I32Buf(pc.aux).v[s0] + 1);
    } else if (MODE == FA_MODE_PACKED) {
        a = I32Buf(pc.aux).v[2u * qi];
        b = I32Buf(pc.aux).v[2u * qi + 1u];
        if (pc.window > 0u) {
            const int w2 = int(pc.window) / 2;
            a = max(a, int(qi) - w2);
            b = min(b, int(qi) + w2 + 1);
        }
    }
    a = clamp(a, 0, int(pc.lk));
    b = clamp(b, a, int(pc.lk));
    lo = a;
    hi = b;
}

// Query rows [qlo, qhi) that can attend keys [k0, k1] (k1 inclusive): a
// superset (PACKED: every row; the caller checks each block's intervals).
void bw_query_range(int k0, int k1, out int qlo, out int qhi) {
    qlo = 0;
    qhi = int(pc.lq);
    if (MODE == FA_MODE_ROWS && pc.causal != 0u) {
        qlo = k0;
        if (pc.window > 0u) qhi = min(qhi, k1 + int(pc.window));
    } else if (MODE == FA_MODE_ROWS && pc.window > 0u) {
        const int w2 = int(pc.window) / 2;
        qlo = max(0, k0 - w2);
        qhi = min(qhi, k1 + w2 + 1);
    } else if (MODE == FA_MODE_VARLEN) {
        // The sequences holding k0 and k1: the largest s with cu_k[s] <= key.
        uint sa = 0u, sb = 0u;
        for (uint pass_ = 0u; pass_ < 2u; ++pass_) {
            const int key = pass_ == 0u ? k0 : k1;
            uint s0 = 0u, s1 = pc.nseq;
            while (s1 - s0 > 1u) {
                const uint mid = (s0 + s1) / 2u;
                if (I32Buf(pc.aux2).v[mid] <= key) s0 = mid; else s1 = mid;
            }
            if (pass_ == 0u) sa = s0; else sb = s0;
        }
        qlo = pc.nseq == 0u ? 0 : clamp(I32Buf(pc.aux).v[sa], 0, int(pc.lq));
        qhi = pc.nseq == 0u ? 0 : clamp(I32Buf(pc.aux).v[sb + 1u], qlo, int(pc.lq));
    }
}
