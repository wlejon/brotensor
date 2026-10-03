// Operation codes shared by the GLSL kernels and the C++ that launches them.
// Plain #defines so the same file is valid GLSL and C++; a kernel receives
// its code as specialisation constant 0, so every (kernel, code) pair is its
// own pipeline with the switch folded away.
#ifndef BT_VK_OP_CODES_H
#define BT_VK_OP_CODES_H

// unary.comp: y = f(x [, a, b])
#define UOP_COPY        0
#define UOP_RELU        1
#define UOP_TANH        2
#define UOP_SIGMOID     3
#define UOP_SILU        4
#define UOP_GELU_TANH   5
#define UOP_GELU_EXACT  6
#define UOP_QUICK_GELU  7
#define UOP_EXP         8
#define UOP_LOG         9
#define UOP_SIN         10
#define UOP_COS         11
#define UOP_RSQRT       12
#define UOP_ROUND       13
#define UOP_ELU         14   // a = alpha
#define UOP_LEAKY_RELU  15   // a = negative slope
#define UOP_ADD_SCALAR  16   // a = s
#define UOP_SCALE       17   // a = s
#define UOP_CLAMP       18   // a = lo, b = hi

// binary.comp: y = f(y, x [, a, b])
#define BOP_ADD         0
#define BOP_MUL         1
#define BOP_DIV         2
#define BOP_AXPBY       3    // a * y + b * x

// act_bwd.comp: dX = g(p, dY), p is x or y as the op's contract says
#define GOP_RELU        0    // p = x
#define GOP_TANH        1    // p = y
#define GOP_SIGMOID     2    // p = y
#define GOP_SILU        3    // p = x
#define GOP_GELU_TANH   4    // p = x
#define GOP_GELU_EXACT  5    // p = x
#define GOP_QUICK_GELU  6    // p = x
#define GOP_EXP         7    // p = x
#define GOP_LOG         8    // p = x, dX = dY / x
#define GOP_SIN         9    // p = x
#define GOP_COS         10   // p = x
#define GOP_RSQRT       11   // p = y
#define GOP_ELU         12   // p = x, a = alpha
#define GOP_LEAKY_RELU  13   // p = x, a = slope
#define GOP_PASS        14   // dX = dY (round's straight-through estimator)

// bias.comp
#define BIAS_ROW        0    // y[i] += bias[i % D]
#define BIAS_CHANNEL    1    // y[i] += bias[(i / L) % C]

#endif
