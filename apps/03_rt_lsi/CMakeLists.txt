# Put SPIR-V under: <build>/shaders/<this-app-dir>/
get_filename_component(APP_DIR_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
set(SPV_OUTPUT_DIR "${CMAKE_BINARY_DIR}/shaders/${APP_DIR_NAME}")
set(SLANG_COMMON_FLAGS -profile sm_6_6 -target spirv -fvk-use-scalar-layout)

slang_compile_spirv(
    NAME rt_lsi
    SLANGC ${CMAKE_SOURCE_DIR}/cmake-build-debug/_deps/Slang-linux-x86_64-2026.1.1/bin/slangc
    SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/shader/rt_lsi.slang
    OUT_DIR ${SPV_OUTPUT_DIR}
    FLAGS ${SLANG_COMMON_FLAGS}
    ENTRIES
        raygenMain       raygeneration  raygen.spv
        missMain         miss           miss.spv
        isectMain        intersection   isect.spv
        anyhitMain       anyhit         ahit.spv
        closesthitMain   closesthit     chit.spv
)

add_executable(VkPrimeRtLsi main.cpp)
target_link_libraries(VkPrimeRtLsi PRIVATE Vulkan::Vulkan)
target_compile_definitions(VkPrimeRtLsi PRIVATE SHADER_DIR="${SPV_OUTPUT_DIR}")

add_dependencies(VkPrimeRtLsi ${rt_lsi_SPV_TARGET})
target_sources(VkPrimeRtLsi PRIVATE ${rt_lsi_SPV_FILES})
