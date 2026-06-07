#pragma once

// ─── brotensor ops — umbrella header ──────────────────────────────────────
//
// Every public op, declared once and dispatched at runtime to the backend
// (CPU / CUDA / Metal) where its operand tensors live. Tensors are row-major
// (rows, cols) and carry a Dtype and a Device.
//
// The declarations now live in per-category headers under ops/ — the file
// name is the category, so `ls include/brotensor/ops/` is the table of
// contents and a grep can be scoped to one file. This umbrella includes them
// all, so `#include <brotensor/ops.h>` keeps pulling in the full surface.
//
// Contract conventions used throughout:
//   - Output tensors are resized (and dtype-set, where noted) to the expected
//     shape if mis-shaped — except accumulation outputs (dW, dB, ...), which
//     the caller must pre-size and zero; the op adds into them.
//   - Backward gradient outputs are *overwritten* unless marked *accumulated*.
//   - Synchronisation is the caller's responsibility: call brotensor::sync()
//     before reading GPU results to host. CPU ops are synchronous.
//   - Backends throw std::runtime_error ("brotensor: <op>: <reason>") for
//     contract violations and for unimplemented ops.

#include "ops/activation.h"
#include "ops/attention.h"
#include "ops/codec.h"
#include "ops/concat.h"
#include "ops/conv.h"
#include "ops/conv1d.h"
#include "ops/delta_rule.h"
#include "ops/diffusion.h"
#include "ops/elementwise.h"
#include "ops/embedding.h"
#include "ops/flash_attention.h"
#include "ops/image.h"
#include "ops/linear.h"
#include "ops/loss.h"
#include "ops/lstm.h"
#include "ops/norm.h"
#include "ops/optim.h"
#include "ops/pooling.h"
#include "ops/quant.h"
#include "ops/reduction.h"
#include "ops/resize.h"
#include "ops/rope.h"
#include "ops/sampling.h"
#include "ops/spatial.h"
#include "ops/spectral.h"
#include "ops/stylegan.h"

