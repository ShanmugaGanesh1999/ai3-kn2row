set(SYCL_FOUND FALSE)

if(NOT SYCL_ROOT)
    set(SYCL_ROOT $ENV{CMPLR_ROOT})
    if(NOT SYCL_ROOT)
        message(STATUS "Intel oneAPI environment not set")
        return()
    endif()
endif()

if(NOT SYCL_COMPILER)
    if (WIN32)
        set(SYCL_COMPILER icx)
    else()
        find_program(HAS_ICPX "icpx" NO_CACHE)
        if (HAS_ICPX)
            set(SYCL_COMPILER icpx)
        else()
            set(SYCL_COMPILER clang++)
        endif()
    endif()
endif()

set(SYCL_FLAGS "-fsycl")
find_package(CUDA)
if(CUDA_FOUND)
    set(SYCL_FLAGS "${SYCL_FLAGS} -fsycl-targets=nvptx64-nvidia-cuda,spir64")
endif()

set(SYCL_CFLAGS "${SYCL_FLAGS}")
set(SYCL_LFLAGS "${SYCL_CFLAGS} -lsycl")
set(SYCL_INCLUDE_DIR "${SYCL_ROOT}/include/")
find_file(
    SYCL_SYCL_INCLUDE_DIR
    NAMES sycl
    HINTS ${SYCL_ROOT}/include/
    NO_DEFAULT_PATH
)
find_file(
    SYCL_LIBRARY_DIR
    NAMES lib
    HINTS ${SYCL_ROOT}
    NO_DEFAULT_PATH
)
set(SYCL_FOUND TRUE)
