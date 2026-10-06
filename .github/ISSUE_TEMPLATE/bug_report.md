---
name: Bug report
about: An op returns wrong values, GPU and CPU disagree, a model file fails to load, a crash, or a test fails
labels: bug
---

**The op or loader:** its name, the input shapes and dtypes, and the device
the tensors were on (a few lines that reproduce it are ideal).

**What it should produce** (the CPU reference, PyTorch, or the expected
shape):

**What brotensor produced instead** (the values or max error, the error
message, a crash, or the failing `ctest --output-on-failure` output — paste
it):

```
```

**Environment:**
- OS:
- Backend (CPU / CUDA / Metal / Vulkan) and GPU + driver version:
- Compiler / toolchain (MSVC / GCC / Clang, nvcc version for CUDA):
- brotensor commit:
