#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#define ASSERT_TRUE(cond)                                                      \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "Assertion failed: %s at %s:%d\n", #cond,     \
                         __FILE__, __LINE__);                                  \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

int main() {
    brotensor::init();

    int count = brotensor::cuda_device_count();
    std::cout << "[test_multigpu] CUDA device count: " << count << std::endl;

    if (count < 2) {
        std::cout << "[test_multigpu] Only " << count
                  << " CUDA device(s) found. Skipping multi-GPU P2P tests."
                  << std::endl;
        return 0;
    }

    auto devs = brotensor::available_devices();
    std::cout << "[test_multigpu] Available devices:" << std::endl;
    for (auto d : devs) {
        std::cout << "  - " << brotensor::to_string(d) << " ("
                  << brotensor::device_product_name(d) << ")" << std::endl;
    }

    ASSERT_TRUE(brotensor::is_available(brotensor::Device::cuda(0)));
    ASSERT_TRUE(brotensor::is_available(brotensor::Device::cuda(1)));

    const int R = 128;
    const int C = 256;

    // 1. Create tensor A on GPU 0 and fill with host data
    std::vector<float> host_a(R * C);
    for (int i = 0; i < R * C; ++i) {
        host_a[i] = static_cast<float>(i % 100) * 0.1f;
    }

    brotensor::Tensor t_gpu0 =
        brotensor::Tensor::from_host_on(brotensor::Device::cuda(0), host_a.data(), R, C);
    ASSERT_TRUE(t_gpu0.device == brotensor::Device::cuda(0));

    // 2. Perform P2P peer copy to GPU 1
    brotensor::Tensor t_gpu1 = t_gpu0.to(brotensor::Device::cuda(1));
    ASSERT_TRUE(t_gpu1.device == brotensor::Device::cuda(1));

    // 3. Verify content on GPU 1 matches original
    std::vector<float> readback_gpu1 = t_gpu1.to_host_vector();
    for (int i = 0; i < R * C; ++i) {
        if (std::abs(readback_gpu1[i] - host_a[i]) > 1e-5f) {
            std::fprintf(stderr, "Mismatch at %d: got %f expected %f\n", i,
                         readback_gpu1[i], host_a[i]);
            ASSERT_TRUE(false);
        }
    }

    // 4. Perform an operation on GPU 1 (e.g. ReLU)
    brotensor::Tensor t_relu_gpu1 = brotensor::Tensor::empty_on(
        brotensor::Device::cuda(1), R, C, brotensor::Dtype::FP32);
    brotensor::relu_forward(t_gpu1, t_relu_gpu1);
    ASSERT_TRUE(t_relu_gpu1.device == brotensor::Device::cuda(1));

    // 5. Transfer result back to GPU 0
    brotensor::Tensor t_gpu0_back = t_relu_gpu1.to(brotensor::Device::cuda(0));
    ASSERT_TRUE(t_gpu0_back.device == brotensor::Device::cuda(0));

    std::vector<float> readback_gpu0 = t_gpu0_back.to_host_vector();
    for (int i = 0; i < R * C; ++i) {
        float expected = std::max(0.0f, host_a[i]);
        if (std::abs(readback_gpu0[i] - expected) > 1e-5f) {
            std::fprintf(stderr, "Mismatch after GPU roundtrip at %d: got %f expected %f\n",
                         i, readback_gpu0[i], expected);
            ASSERT_TRUE(false);
        }
    }

    // 6. Test matmul across multiple GPUs in sequence
    // W0 on GPU 0, W1 on GPU 1
    const int K = 64;
    std::vector<float> h_w0(C * K, 0.05f);
    std::vector<float> h_w1(K * 32, 0.02f);

    brotensor::Tensor W0 = brotensor::Tensor::from_host_on(brotensor::Device::cuda(0), h_w0.data(), C, K);
    brotensor::Tensor W1 = brotensor::Tensor::from_host_on(brotensor::Device::cuda(1), h_w1.data(), K, 32);

    // Stage 0 on GPU 0
    brotensor::Tensor h0 = brotensor::Tensor::empty_on(brotensor::Device::cuda(0), R, K, brotensor::Dtype::FP32);
    brotensor::matmul(t_gpu0, W0, h0);

    // Inter-stage transfer: GPU 0 -> GPU 1
    brotensor::Tensor h0_on_gpu1 = h0.to(brotensor::Device::cuda(1));

    // Stage 1 on GPU 1
    brotensor::Tensor h1 = brotensor::Tensor::empty_on(brotensor::Device::cuda(1), R, 32, brotensor::Dtype::FP32);
    brotensor::matmul(h0_on_gpu1, W1, h1);

    std::vector<float> final_out = h1.to_host_vector();
    std::cout << "[test_multigpu] Pipeline execution succeeded! Final tensor size: "
              << h1.rows << "x" << h1.cols << ", sample[0]=" << final_out[0] << std::endl;

    brotensor::sync_all();
    std::cout << "[test_multigpu] ALL TESTS PASSED!" << std::endl;
    return 0;
}
