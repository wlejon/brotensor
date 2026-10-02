# BrotensorHip.cmake — where ROCm is, and which AMD GPUs the HIP backend
# compiles for. Shared by brotensor and by every sibling that compiles its own
# kernels as HIP (brodiffusion, brovisionml) or decides whether to turn the
# HIP backend on (bro), so they all answer these questions the same way.
#
#   brotensor_hip_prepare()        before enable_language(HIP): resolves ROCM_PATH
#                                  and puts it on CMAKE_PREFIX_PATH so
#                                  find_package(hip / hipblas / rocwmma) find it.
#   brotensor_hip_check_archs()    after enable_language(HIP): validates
#                                  CMAKE_HIP_ARCHITECTURES (see below).
#   brotensor_hip_detect_gpus(<var>)  the gfx targets of the AMD GPUs in this
#                                  machine that the backend supports (empty if
#                                  none), for a parent project's auto-detection.
#
# ROCm root. -DROCM_PATH=..., else $ENV{ROCM_PATH}, else /opt/rocm. Nothing in
# the build names /opt/rocm/include directly: include paths come from the
# imported targets (hip::host, roc::hipblas, roc::rocwmma).
#
# Architectures. Leave CMAKE_HIP_ARCHITECTURES unset and CMake initializes it
# from rocm_agent_enumerator, i.e. the GPUs in the build machine (HIP's
# equivalent of CUDA's "native"). Set it — -DCMAKE_HIP_ARCHITECTURES=gfx1100;gfx1151
# — to build for other machines or on one without a GPU. Either way every
# entry must be an RDNA 3 / 3.5 / 4 target (gfx11xx, gfx12xx):
#
#   * Wave32. The native kernels (src/hip/*.hip) and the CUDA kernels compiled
#     through src/hip/compat reduce with 32-lane shuffles and index per-warp
#     scratch by threadIdx / 32 — CUDA's warp model, which is RDNA's default
#     wavefront. GCN / CDNA (gfx8, gfx9: MI-series, Vega) run 64-wide waves
#     and would compute wrong answers, so they are rejected here, as is
#     -mwavefrontsize64 in CMAKE_HIP_FLAGS. The runtime re-checks: a device
#     reporting warpSize != 32 is not registered (src/hip/register.hip).
#   * LDS. Kernels are sized for the 64 KB of LDS one RDNA workgroup can
#     allocate; the attention kernels that stage tiles query the per-block
#     limit at run time and fall back to global reads past it.
#   * WMMA. src/hip/compat/mma.h maps CUDA's nvcuda::wmma onto rocWMMA, and
#     the tensor-core kernels compiled through it (fp16_matmul, the fused
#     flash attention, the quantized WMMA GEMMs) need matrix cores: rocWMMA
#     static-asserts "Unsupported architecture" for gfx10 (RDNA 1/2), so those
#     are rejected too. Checked compiling for gfx1100, gfx1151 and gfx1201.

include_guard(GLOBAL)

set(_BROTENSOR_HIP_SUPPORTED_ARCH_REGEX "^gfx1[12][0-9][0-9a-z]*(:.*)?$")

macro(brotensor_hip_prepare)
    if(NOT ROCM_PATH)
        if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
            set(_brotensor_rocm "$ENV{ROCM_PATH}")
        else()
            set(_brotensor_rocm "/opt/rocm")
        endif()
        set(ROCM_PATH "${_brotensor_rocm}" CACHE PATH "ROCm installation root")
    endif()
    if(NOT EXISTS "${ROCM_PATH}")
        message(FATAL_ERROR
            "HIP backend requested but ROCm was not found at ROCM_PATH='${ROCM_PATH}'. "
            "Install ROCm or pass -DROCM_PATH=<root> (or set the ROCM_PATH environment variable).")
    endif()
    list(FIND CMAKE_PREFIX_PATH "${ROCM_PATH}" _brotensor_rocm_idx)
    if(_brotensor_rocm_idx EQUAL -1)
        list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
    endif()
endmacro()

function(brotensor_hip_check_archs)
    if(NOT CMAKE_HIP_ARCHITECTURES)
        # CMake fills this in from rocm_agent_enumerator when it first
        # identifies the HIP compiler; a build tree whose compiler was
        # identified while the variable was set by hand (as older brotensor
        # did) never got that, so detect here the same way.
        brotensor_hip_detect_gpus(_detected)
        if(_detected)
            set(CMAKE_HIP_ARCHITECTURES "${_detected}" CACHE STRING
                "AMD GPU targets for HIP device code" FORCE)
        endif()
    endif()
    if(NOT CMAKE_HIP_ARCHITECTURES)
        message(FATAL_ERROR
            "HIP backend: no GPU architecture to compile for — rocm_agent_enumerator "
            "found no AMD GPU on this machine. Pass the target explicitly, e.g. "
            "-DCMAKE_HIP_ARCHITECTURES=gfx1151 (several: \"gfx1100;gfx1151\").")
    endif()
    foreach(_arch IN LISTS CMAKE_HIP_ARCHITECTURES)
        if(NOT _arch MATCHES "${_BROTENSOR_HIP_SUPPORTED_ARCH_REGEX}")
            message(FATAL_ERROR
                "HIP backend: CMAKE_HIP_ARCHITECTURES contains '${_arch}', which is not "
                "an RDNA 3+ (gfx11xx / gfx12xx) target. The kernels assume Wave32, "
                "RDNA's 64 KB workgroup LDS and WMMA matrix cores (GCN/CDNA gfx8/gfx9 "
                "run 64-wide waves; rocWMMA does not support gfx10). "
                "See cmake/BrotensorHip.cmake.")
        endif()
    endforeach()
    if(CMAKE_HIP_FLAGS MATCHES "wavefrontsize64")
        message(FATAL_ERROR
            "HIP backend: CMAKE_HIP_FLAGS requests -mwavefrontsize64; the kernels assume Wave32.")
    endif()
    message(STATUS "brotensor: HIP architectures: ${CMAKE_HIP_ARCHITECTURES} (ROCm at ${ROCM_PATH})")
endfunction()

function(brotensor_hip_detect_gpus out_var)
    set(_found "")
    if(NOT ROCM_PATH)
        if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
            set(_root "$ENV{ROCM_PATH}")
        else()
            set(_root "/opt/rocm")
        endif()
    else()
        set(_root "${ROCM_PATH}")
    endif()
    find_program(_brotensor_agent_enum rocm_agent_enumerator
                 HINTS "${_root}/bin" NO_DEFAULT_PATH NO_CACHE)
    if(_brotensor_agent_enum)
        execute_process(COMMAND "${_brotensor_agent_enum}"
                        OUTPUT_VARIABLE _agents ERROR_QUIET
                        RESULT_VARIABLE _rc TIMEOUT 30)
        if(_rc EQUAL 0)
            string(REGEX MATCHALL "gfx[0-9a-z]+" _agents "${_agents}")
            foreach(_a IN LISTS _agents)
                if(_a MATCHES "${_BROTENSOR_HIP_SUPPORTED_ARCH_REGEX}")
                    list(APPEND _found "${_a}")
                endif()
            endforeach()
            list(REMOVE_DUPLICATES _found)
        endif()
    endif()
    set(${out_var} "${_found}" PARENT_SCOPE)
endfunction()
