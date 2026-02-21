cmake_minimum_required(VERSION 4.0)
project(VkPrimer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(NvproCore2 REQUIRED)

find_package(Vulkan REQUIRED)

# =====================================================
# Vulkan CUDA Cores
add_executable(VkPrimer app/main.cpp)
target_link_libraries(VkPrimer PRIVATE Vulkan::Vulkan)

add_executable(VkPrimer10k app/main_10k.cpp)
target_link_libraries(VkPrimer10k PRIVATE Vulkan::Vulkan)

# =====================================================
# Vulkan RT Cores
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompileSlang)

set(SPV_OUTPUT_DIR ${CMAKE_BINARY_DIR}/shaders)
set(SLANG_COMMON_FLAGS -profile sm_6_6 -target spirv -fvk-use-scalar-layout)

slang_compile_spirv(
        NAME rt_triangles
        SLANGC ${CMAKE_SOURCE_DIR}/cmake-build-debug/_deps/Slang-linux-x86_64-2026.1.1/bin/slangc
        SOURCE ${CMAKE_SOURCE_DIR}/shader/rt_triangles.slang
        OUT_DIR ${SPV_OUTPUT_DIR}
        FLAGS ${SLANG_COMMON_FLAGS}
        ENTRIES
        raygenMain raygeneration raygen.spv
        missMain miss miss.spv
        chitMain closesthit chit.spv
        aHitMain anyhit ahit.spv
)

add_executable(VkPrimerRT app/main_rt.cpp)
target_link_libraries(VkPrimerRT PRIVATE Vulkan::Vulkan)
target_compile_definitions(VkPrimerRT PRIVATE SHADER_DIR="${SPV_OUTPUT_DIR}")

add_dependencies(VkPrimerRT ${rt_triangles_SPV_TARGET})
target_sources(VkPrimerRT PRIVATE ${rt_triangles_SPV_FILES})