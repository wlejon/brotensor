// Accurate FP32 transcendental helpers for kernels whose results are compared
// against the host's libm (rope.comp, sampler.comp, philox.comp). GLSL's
// built-ins are allowed several ulp (exp, log) or an absolute error bound
// (sin / cos on [-pi, pi] only); these stay within about an ulp of glibc's
// expf / logf / sinf / cosf over the ranges the kernels use. Include after
// common.glsl.

// q ~= a / b, refined to (nearly always) the correctly rounded quotient.
float div_rn(float a, float b) {
    const float r = 1.0 / b;
    const float q = a * r;
    return fma(fma(-q, b, a), r, q);
}

// exp(t) for t <= 0, about 1 ulp: t = n ln2 + r, |r| <= ln2 / 2.
float exp_acc(float t) {
    const float n = round(t * 1.44269504088896341);
    float r = fma(-n, 0.693145751953125, t);         // ln2 high part (exact product)
    r = fma(-n, 1.428606765330187045e-06, r);        // ln2 low part
    float p = 1.0 / 5040.0;
    p = fma(p, r, 1.0 / 720.0);
    p = fma(p, r, 1.0 / 120.0);
    p = fma(p, r, 1.0 / 24.0);
    p = fma(p, r, 1.0 / 6.0);
    p = fma(p, r, 0.5);
    p = fma(p, r, 1.0);
    p = fma(p, r, 1.0);
    return ldexp(p, int(n));
}

// sin and cos of x with a three-part Cody-Waite reduction by pi/2 (fma keeps
// each product exact) and minimax polynomials on [-pi/4, pi/4].
void sincos_acc(float x, out float s, out float c) {
    const float j = round(x * 0.636619772367581343);
    float y = fma(-j, 1.57079637050628662, x);
    y = fma(-j, -4.37113882867379e-08, y);
    y = fma(-j, -1.71512451e-15, y);
    const float z = y * y;
    const float sp = y + y * z * (-1.6666654611e-1 + z * (8.3321608736e-3 + z * -1.9515295891e-4));
    const float cp = 1.0 - 0.5 * z + z * z * (4.166664568298827e-2 + z * (-1.388731625493765e-3 + z * 2.443315711809948e-5));
    const int q = int(j) & 3;
    if (q == 0)      { s = sp;  c = cp;  }
    else if (q == 1) { s = cp;  c = -sp; }
    else if (q == 2) { s = -sp; c = -cp; }
    else             { s = -cp; c = sp;  }
}

// Natural log of a positive normal x, about 1 ulp: x = m 2^e with m in
// [sqrt(1/2), sqrt(2)), log m = 2 atanh(s), s = (m - 1) / (m + 1), the odd
// series to s^11 (|s| <= 0.172), ln 2 split in two for the e term.
float log_acc(float x) {
    int e;
    float m = frexp(x, e);          // m in [0.5, 1)
    if (m < 0.70710678118654752) { m *= 2.0; e -= 1; }
    const float s = div_rn(m - 1.0, m + 1.0);
    const float z = s * s;
    float p = 2.0 / 11.0;
    p = fma(p, z, 2.0 / 9.0);
    p = fma(p, z, 2.0 / 7.0);
    p = fma(p, z, 2.0 / 5.0);
    p = fma(p, z, 2.0 / 3.0);
    const float lm = fma(s * z, p, 2.0 * s);
    const float fe = float(e);
    return fma(fe, 0.693145751953125, fma(fe, 1.428606765330187045e-06, lm));
}
