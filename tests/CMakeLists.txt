# Minimal smoke tests for brotensor. Deep CPU↔GPU parity coverage lives in
# brogameagent's tests since those need a CPU reference implementation.

function(brotensor_add_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE brotensor)
    if(BROTENSOR_WITH_CUDA)
        target_include_directories(${name} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    endif()
    if(MSVC)
        target_compile_options(${name} PRIVATE /W4)
    else()
        target_compile_options(${name} PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME ${name} COMMAND ${name})
endfunction()

brotensor_add_test(brotensor_test test_tensor.cpp)
brotensor_add_test(brotensor_test_conv2d test_conv2d.cpp)
brotensor_add_test(brotensor_test_group_norm test_group_norm.cpp)
brotensor_add_test(brotensor_test_activations test_activations.cpp)
brotensor_add_test(brotensor_test_resample test_resample.cpp)
