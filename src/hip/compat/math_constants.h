#pragma once

#ifndef CUDART_INF_F
#define CUDART_INF_F (__builtin_inff())
#endif

#ifndef CUDART_NAN_F
#define CUDART_NAN_F (__builtin_nanf(""))
#endif

#ifndef CUDART_MIN_DENORM_F
#define CUDART_MIN_DENORM_F 1.401298464e-45f
#endif
