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

add_executable(VkPrimerRT main_rt.cpp)
target_link_libraries(VkPrimerRT PRIVATE Vulkan::Vulkan)
