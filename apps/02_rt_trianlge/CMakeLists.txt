# Put SPIR-V under: <build>/shaders/<this-app-dir>/
get_filename_component(APP_DIR_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
set(SPV_OUTPUT_DIR "${CMAKE_BINARY_DIR}/shaders/${APP_DIR_NAME}")
set(SLANG_COMMON_FLAGS -profile sm_6_6 -target spirv -fvk-use-scalar-layout)

slang_compile_spirv(
        NAME rt_triangles
        SLANGC ${CMAKE_SOURCE_DIR}/cmake-build-debug/_deps/Slang-linux-x86_64-2026.1.1/bin/slangc
        SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/shader/rt_triangles.slang
        OUT_DIR ${SPV_OUTPUT_DIR}
        FLAGS ${SLANG_COMMON_FLAGS}
        ENTRIES
        raygenMain raygeneration raygen.spv
        missMain miss miss.spv
        chitMain closesthit chit.spv
        aHitMain anyhit ahit.spv
)

add_executable(VkPrimerRtTriangle ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(VkPrimerRtTriangle PRIVATE Vulkan::Vulkan)
target_compile_definitions(VkPrimerRtTriangle PRIVATE SHADER_DIR="${SPV_OUTPUT_DIR}")

add_dependencies(VkPrimerRtTriangle ${rt_triangles_SPV_TARGET})
target_sources(VkPrimerRtTriangle PRIVATE ${rt_triangles_SPV_FILES})