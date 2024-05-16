enable_language("CUDA")

# if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
#   set(CMAKE_CUDA_ARCHITECTURES native)
# endif()

# if (CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
#     set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O1 -g -Xclang -fcuda-allow-variadic-functions")
# else()
#     set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O1 -g --expt-extended-lambda -Xcompiler -Wall,-Wextra,-march=native,-mtune=native")
# endif()

set(CUDA_SRCS
  ${CMAKE_CURRENT_LIST_DIR}/test.cu
)

add_library(expander_cuda ${CUDA_SRCS})