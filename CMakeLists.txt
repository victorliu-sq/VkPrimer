cmake_minimum_required(VERSION 4.0)
project(VkPrimer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(NvproCore2 REQUIRED)
find_package(Vulkan REQUIRED)

include(CompileSlang)

# =====================================================
add_subdirectory(apps/01_vec_add)
add_subdirectory(apps/02_rt_trianlge)
add_subdirectory(apps/03_rt_lsi)
