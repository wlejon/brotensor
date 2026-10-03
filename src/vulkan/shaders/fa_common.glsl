// Shared by the attention kernels (fa_cm.comp, fa_rows.comp): the push block
// and each query row's key interval. Needs the MODE specialisation constant.
//
// Q (lq rows, row stride ldq) holds head h at columns [h*HD, (h+1)*HD); K and
// V (row stride ldk, both) hold KV head h / group the same way. The addresses
// already include any column offset (a packed QKV's K and V sections).

#include "op_codes.h"

layout(push_constant) uniform PC {
    uint64_t q, k, v, o;
    uint64_t mask;     // FP32 key-validity vector (> 0.5 = valid), MASK = 1
    uint64_t aux;      // VARLEN: cu_seqlens_q (INT32, nseq + 1); PACKED: seq_bounds (INT32, lq x 2)
    uint64_t aux2;     // VARLEN: cu_seqlens_k
    uint64_t part;     // fa_rows SPLIT: FP32 partials, (rows * hq * nsplit) x (HD + 2)
    uint lq, lk, ldq, ldk, ldo;
    uint group;        // query heads per KV head
    uint window;       // 0 = none
    uint causal;
    uint nseq;         // VARLEN: number of sequences
    uint chunk;        // fa_rows: keys per split
    uint hq;           // query heads
    uint qb0;          // first query block (fa_cm) / row (fa_rows) of this dispatch
    int q_offset;      // ROWS: absolute position of query row 0
    float scale;       // fa_cm: log2(e) / sqrt(HD); fa_rows: 1 / sqrt(HD)
    float softcap;     // fa_rows SOFTCAP: s = softcap * tanh(s / softcap)
} pc;

// Keys [lo, hi) of query row qi, empty past lq. Device-resident sequence
// tables are clamped to [0, lk] so a malformed table cannot read outside K.
void fa_interval(uint qi, out uint lo, out uint hi) {
    lo = 0u;
    hi = 0u;
    if (qi >= pc.lq) return;
    int a = 0, b = int(pc.lk);
    if (MODE == FA_MODE_ROWS) {
        const int aq = int(qi) + pc.q_offset;
        if (pc.causal != 0u) {
            b = min(b, aq + 1);
            if (pc.window > 0u) a = max(0, aq - int(pc.window) + 1);
        } else if (pc.window > 0u) {
            const int w2 = int(pc.window) / 2;
            a = max(0, aq - w2);
            b = min(b, aq + w2 + 1);
        }
    } else if (MODE == FA_MODE_VARLEN) {
        // The sequence holding qi: the largest s < nseq with cu_q[s] <= qi.
        const int q = int(qi);
        if (pc.nseq == 0u || q >= I32Buf(pc.aux).v[pc.nseq]) return;
        uint s0 = 0u, s1 = pc.nseq;   // cu_q[s0] <= q < cu_q[s1]
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
    lo = uint(a);
    hi = uint(b);
}
