# Minimal smoke tests for brotensor. Deep CPU↔GPU parity coverage lives in
# brogameagent's tests since those need a CPU reference implementation.

add_executable(brotensor_test test_tensor.cpp)
target_link_libraries(brotensor_test PRIVATE brotensor)

if(BROTENSOR_WITH_CUDA)
    target_include_directories(brotensor_test PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

if(MSVC)
    target_compile_options(brotensor_test PRIVATE /W4)
else()
    target_compile_options(brotensor_test PRIVATE -Wall -Wextra)
endif()

add_test(NAME brotensor_test COMMAND brotensor_test)
