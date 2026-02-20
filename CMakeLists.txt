cmake_minimum_required(VERSION 4.0)
project(VkPrimer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(NvproCore2 REQUIRED)

find_package(Vulkan REQUIRED)

add_executable(VkPrimer main.cpp)
target_link_libraries(VkPrimer PRIVATE Vulkan::Vulkan)

add_executable(VkPrimer10k main_10k.cpp)
target_link_libraries(VkPrimer10k PRIVATE Vulkan::Vulkan)

# =====================================================
# RT
set(SLANGC ${CMAKE_SOURCE_DIR}/cmake-build-debug/_deps/Slang-linux-x86_64-2026.1.1/bin/slangc)
set(SHADER ${CMAKE_SOURCE_DIR}/shader/rt_triangles.slang)

set(SPV_OUTPUT_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SPV_OUTPUT_DIR})

set(COMMON_FLAGS -profile sm_6_6 -target spirv -fvk-use-scalar-layout)

# Output files
set(RAYGEN_SPV ${SPV_OUTPUT_DIR}/raygen.spv)
set(MISS_SPV   ${SPV_OUTPUT_DIR}/miss.spv)
set(CHIT_SPV   ${SPV_OUTPUT_DIR}/chit.spv)
set(AHIT_SPV   ${SPV_OUTPUT_DIR}/ahit.spv)

# One custom command that generates ALL shader outputs
add_custom_command(
    OUTPUT ${RAYGEN_SPV} ${MISS_SPV} ${CHIT_SPV} ${AHIT_SPV}

    COMMAND ${SLANGC} ${SHADER} ${COMMON_FLAGS}
            -entry raygenMain -stage raygeneration -o ${RAYGEN_SPV}

    COMMAND ${SLANGC} ${SHADER} ${COMMON_FLAGS}
            -entry missMain -stage miss -o ${MISS_SPV}

    COMMAND ${SLANGC} ${SHADER} ${COMMON_FLAGS}
            -entry chitMain -stage closesthit -o ${CHIT_SPV}

    COMMAND ${SLANGC} ${SHADER} ${COMMON_FLAGS}
            -entry aHitMain -stage anyhit -o ${AHIT_SPV}

    DEPENDS ${SHADER}
    COMMENT "Compiling Slang shaders to SPIR-V"
    VERBATIM
)

# Custom target that represents the shader build
add_custom_target(compile_shaders
    DEPENDS ${RAYGEN_SPV} ${MISS_SPV} ${CHIT_SPV} ${AHIT_SPV}
)

# Your executable
add_executable(VkPrimerRT main_rt.cpp)

target_link_libraries(VkPrimerRT PRIVATE Vulkan::Vulkan)

target_compile_definitions(VkPrimerRT PRIVATE
    SHADER_DIR="${SPV_OUTPUT_DIR}"
)

# Ensure shaders build before the executable
add_dependencies(VkPrimerRT compile_shaders)

# (Optional) make SPV files visible in IDEs
target_sources(VkPrimerRT PRIVATE
    ${RAYGEN_SPV}
    ${MISS_SPV}
    ${CHIT_SPV}
    ${AHIT_SPV}
)
#add_executable(VkPrimerRT main_rt.cpp)
#target_link_libraries(VkPrimerRT PRIVATE Vulkan::Vulkan)
