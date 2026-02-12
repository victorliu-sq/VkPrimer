cmake_minimum_required(VERSION 4.0)
project(VkPrimer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(NvproCore2 REQUIRED)

find_package(Vulkan REQUIRED)

add_executable(VkPrimer main.cpp)
target_link_libraries(VkPrimer PRIVATE Vulkan::Vulkan)
