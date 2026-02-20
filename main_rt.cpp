// file: main.cpp
// Minimal headless Vulkan RT sample:
// - Builds BLAS/TLAS for a couple triangles in XY plane at Z=0
// - Traces orthographic rays along -Z
// - Writes hit world coords to an RGBA32F storage image
// - Copies image back and prints a few pixels
//
// Build deps:
//   - Vulkan SDK (headers + loader)
//   - volk (recommended) OR you can remove volk and use vkGetInstanceProcAddr manually.
// This file uses volk for brevity.
//
// Compile example (Linux-ish):
//   g++ -std=c++20 main.cpp -o rtapp -lvulkan
//
// Runtime requires precompiled SPIR-V (from the Slang file):
//   raygen.spv, miss.spv, chit.spv in working dir.

#define VOLK_IMPLEMENTATION
#include <volk/volk.h>

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#define VK_CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
  std::cerr << "Vulkan error " << _r << " at " << __FILE__ << ":" << __LINE__ << "\n"; std::exit(1); } } while(0)

static std::vector<uint32_t> loadSpv(const char *path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::cerr << "Failed to open " << path << "\n";
    std::exit(1);
  }

  f.seekg(0, std::ios::end);
  size_t sz = (size_t) f.tellg();
  f.seekg(0, std::ios::beg);
  if (sz % 4) {
    std::cerr << "Bad SPV size " << path << "\n";
    std::exit(1);
  }
  std::vector<uint32_t> out(sz / 4);
  f.read((char *) out.data(), (std::streamsize) sz);
  return out;
}

static uint32_t findMemoryType(VkPhysicalDevice phys, uint32_t typeBits, VkMemoryPropertyFlags req) {
  VkPhysicalDeviceMemoryProperties mp{};
  vkGetPhysicalDeviceMemoryProperties(phys, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
    if ((typeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & req) == req)
      return i;
  std::cerr << "No suitable memory type\n";
  std::exit(1);
}

struct Buffer {
  VkBuffer buf{}; // low-level handler
  VkDeviceMemory mem{}; // device memory
  VkDeviceAddress addr{}; // device memory address
  VkDeviceSize size{};
};

static Buffer createBuffer(
  VkDevice dev, VkPhysicalDevice phys,
  VkDeviceSize size, VkBufferUsageFlags usage,
  VkMemoryPropertyFlags memProps,
  bool deviceAddress) {
  Buffer b{};
  b.size = size;

  VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  ci.size = size;
  ci.usage = usage | (deviceAddress ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0);
  ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK(vkCreateBuffer(dev, &ci, nullptr, &b.buf));

  VkMemoryRequirements mr{};
  vkGetBufferMemoryRequirements(dev, b.buf, &mr);

  VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
  flags.flags = deviceAddress ? VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT : 0;

  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = mr.size;
  ai.memoryTypeIndex = findMemoryType(phys, mr.memoryTypeBits, memProps);
  if (deviceAddress) ai.pNext = &flags;

  VK_CHECK(vkAllocateMemory(dev, &ai, nullptr, &b.mem));
  VK_CHECK(vkBindBufferMemory(dev, b.buf, b.mem, 0));

  if (deviceAddress) {
    VkBufferDeviceAddressInfo bai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    bai.buffer = b.buf;
    b.addr = vkGetBufferDeviceAddress(dev, &bai);
  }
  return b;
}

static void *mapBuffer(VkDevice dev, const Buffer &b) {
  void *p = nullptr;
  VK_CHECK(vkMapMemory(dev, b.mem, 0, b.size, 0, &p));
  return p;
}

static void unmapBuffer(VkDevice dev, const Buffer &b) {
  vkUnmapMemory(dev, b.mem);
}

static void destroyBuffer(VkDevice dev, Buffer &b) {
  if (b.buf) vkDestroyBuffer(dev, b.buf, nullptr);
  if (b.mem) vkFreeMemory(dev, b.mem, nullptr);
  b = {};
}

struct Image {
  VkImage img{}; // handler for C++ program to access GPU resource
  VkDeviceMemory mem{}; // the GPU resource / memory location on GPU
  VkImageView view{}; // handler for shader program to access GPU resource
  VkFormat format{}; // pixel format
  uint32_t w{}, h{}; // # of pixels
};

// create an image of certain format
static Image createStorageImageRGBA32F(VkDevice dev, VkPhysicalDevice phys, uint32_t w, uint32_t h) {
  Image im{};
  im.w = w;
  im.h = h;
  im.format = VK_FORMAT_R32G32B32A32_SFLOAT;

  VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ci.imageType = VK_IMAGE_TYPE_2D;
  ci.format = im.format;
  ci.extent = {w, h, 1};
  ci.mipLevels = 1;
  ci.arrayLayers = 1;
  ci.samples = VK_SAMPLE_COUNT_1_BIT;
  ci.tiling = VK_IMAGE_TILING_OPTIMAL;
  ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VK_CHECK(vkCreateImage(dev, &ci, nullptr, &im.img));

  VkMemoryRequirements mr{};
  vkGetImageMemoryRequirements(dev, im.img, &mr);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = mr.size;
  ai.memoryTypeIndex = findMemoryType(phys, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VK_CHECK(vkAllocateMemory(dev, &ai, nullptr, &im.mem));
  VK_CHECK(vkBindImageMemory(dev, im.img, im.mem, 0));

  VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vi.image = im.img;
  vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
  vi.format = im.format;
  vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  vi.subresourceRange.levelCount = 1;
  vi.subresourceRange.layerCount = 1;
  VK_CHECK(vkCreateImageView(dev, &vi, nullptr, &im.view));

  return im;
}

static void destroyImage(VkDevice dev, Image &im) {
  if (im.view) vkDestroyImageView(dev, im.view, nullptr);
  if (im.img) vkDestroyImage(dev, im.img, nullptr);
  if (im.mem) vkFreeMemory(dev, im.mem, nullptr);
  im = {};
}

static VkCommandPool createCmdPool(VkDevice dev, uint32_t qfam) {
  VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  ci.queueFamilyIndex = qfam;
  ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  VkCommandPool pool{};
  VK_CHECK(vkCreateCommandPool(dev, &ci, nullptr, &pool));
  return pool;
}

static VkCommandBuffer createCmdBuffer(VkDevice dev, VkCommandPool pool) {
  VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ai.commandPool = pool;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = 1;
  VkCommandBuffer cmd{};
  VK_CHECK(vkAllocateCommandBuffers(dev, &ai, &cmd));
  return cmd;
}

static void submitAndWait(VkDevice dev, VkQueue q, VkCommandBuffer cmd) {
  VK_CHECK(vkEndCommandBuffer(cmd)); // End Recoding Command
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  VK_CHECK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE)); // Submit to Queue
  VK_CHECK(vkQueueWaitIdle(q)); // Wait for Completion
}

// It changes the image’s layout and synchronizes GPU access so the image can be used safely for a specific purpose (storage, transfer, sampling, etc.).
// In Vulkan, an image must be in the correct layout before you use it, and you must insert pipeline barriers to avoid race conditions.
static void cmdTransitionImage(VkCommandBuffer cmd, VkImage img, VkImageLayout oldL, VkImageLayout newL) {
  VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  b.oldLayout = oldL;
  b.newLayout = newL;
  b.image = img;
  b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  b.subresourceRange.levelCount = 1;
  b.subresourceRange.layerCount = 1;
  b.srcAccessMask = 0;
  b.dstAccessMask = 0;

  VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
  VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

  vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

// ---- Acceleration Structure Helpers ----

// Accel struct represents exactly one acceleration structure object — either a BLAS or a TLAS node — not an entire AS tree.
// Accel = one BLAS or TLAS node
struct Accel {
  VkAccelerationStructureKHR as{}; // handle
  Buffer backing; // buffer, the actual memory that S
  VkDeviceAddress addr{};
};

static VkDeviceAddress getASAddress(VkDevice dev, VkAccelerationStructureKHR as) {
  VkAccelerationStructureDeviceAddressInfoKHR ai{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
  ai.accelerationStructure = as;
  return vkGetAccelerationStructureDeviceAddressKHR(dev, &ai);
}

// Create one BLAS node from Triangles
static Accel createBLAS_Triangles(
  VkDevice dev, VkPhysicalDevice phys, VkCommandBuffer cmd,
  const Buffer &vbo, uint32_t vertexCount, VkDeviceSize vertexStride,
  const Buffer &ibo, uint32_t indexCount) {
  // Geometry
  VkAccelerationStructureGeometryTrianglesDataKHR tri{
    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR
  };
  tri.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
  tri.vertexData.deviceAddress = vbo.addr;
  tri.vertexStride = vertexStride;
  tri.maxVertex = vertexCount;
  tri.indexType = VK_INDEX_TYPE_UINT32;
  tri.indexData.deviceAddress = ibo.addr;

  VkAccelerationStructureGeometryKHR geom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  // geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
  geom.flags = 0; // allow any-hit
  geom.geometry.triangles = tri;

  uint32_t primCount = indexCount / 3;

  VkAccelerationStructureBuildGeometryInfoKHR bgi{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  bgi.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  bgi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  bgi.geometryCount = 1;
  bgi.pGeometries = &geom;

  VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  vkGetAccelerationStructureBuildSizesKHR(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bgi, &primCount,
                                          &sizes);

  Accel out{};
  out.backing = createBuffer(
    dev, phys, sizes.accelerationStructureSize,
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    true);

  VkAccelerationStructureCreateInfoKHR asci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  asci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  asci.size = sizes.accelerationStructureSize;
  asci.buffer = out.backing.buf;
  VK_CHECK(vkCreateAccelerationStructureKHR(dev, &asci, nullptr, &out.as));

  Buffer scratch = createBuffer(
    dev, phys, sizes.buildScratchSize,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    true);

  bgi.dstAccelerationStructure = out.as;
  bgi.scratchData.deviceAddress = scratch.addr;

  VkAccelerationStructureBuildRangeInfoKHR range{};
  range.primitiveCount = primCount;
  const VkAccelerationStructureBuildRangeInfoKHR *pRange = &range;

  vkCmdBuildAccelerationStructuresKHR(cmd, 1, &bgi, &pRange);

  // Barrier so AS build is visible for trace
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
  mb.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
  vkCmdPipelineBarrier(cmd,
                       VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                       VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                       0, 1, &mb, 0, nullptr, 0, nullptr);

  // cleanup scratch after submit (we'll destroy later after queue idle)
  destroyBuffer(dev, scratch);

  out.addr = getASAddress(dev, out.as);
  return out;
}

// Create one TLAS node
// blasAddr: device address of the entire BLAS acceleration structure
static Accel createTLAS_OneInstance(
  VkDevice dev, VkPhysicalDevice phys, VkCommandBuffer cmd,
  VkDeviceAddress blasAddr) {
  VkAccelerationStructureInstanceKHR inst{};
  // identity
  inst.transform.matrix[0][0] = 1.f;
  inst.transform.matrix[1][1] = 1.f;
  inst.transform.matrix[2][2] = 1.f;
  inst.instanceCustomIndex = 0;
  inst.mask = 0xFF;
  inst.instanceShaderBindingTableRecordOffset = 0;
  inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
  inst.accelerationStructureReference = blasAddr;

  Buffer instBuf = createBuffer(
    dev, phys, sizeof(inst),
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    true);

  std::memcpy(mapBuffer(dev, instBuf), &inst, sizeof(inst));
  unmapBuffer(dev, instBuf);

  VkAccelerationStructureGeometryInstancesDataKHR idata{
    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR
  };
  idata.arrayOfPointers = VK_FALSE;
  idata.data.deviceAddress = instBuf.addr;

  VkAccelerationStructureGeometryKHR geom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  geom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  geom.geometry.instances = idata;

  uint32_t primCount = 1;

  VkAccelerationStructureBuildGeometryInfoKHR bgi{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  bgi.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  bgi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  bgi.geometryCount = 1;
  bgi.pGeometries = &geom;

  VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  vkGetAccelerationStructureBuildSizesKHR(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bgi, &primCount,
                                          &sizes);

  Accel out{};
  out.backing = createBuffer(
    dev, phys, sizes.accelerationStructureSize,
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    true);

  VkAccelerationStructureCreateInfoKHR asci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  asci.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  asci.size = sizes.accelerationStructureSize;
  asci.buffer = out.backing.buf;
  VK_CHECK(vkCreateAccelerationStructureKHR(dev, &asci, nullptr, &out.as));

  Buffer scratch = createBuffer(
    dev, phys, sizes.buildScratchSize,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    true);

  bgi.dstAccelerationStructure = out.as;
  bgi.scratchData.deviceAddress = scratch.addr;

  VkAccelerationStructureBuildRangeInfoKHR range{};
  range.primitiveCount = 1;
  const VkAccelerationStructureBuildRangeInfoKHR *pRange = &range;

  vkCmdBuildAccelerationStructuresKHR(cmd, 1, &bgi, &pRange);

  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
  mb.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
  vkCmdPipelineBarrier(cmd,
                       VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                       VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                       0, 1, &mb, 0, nullptr, 0, nullptr);

  // The scratch buffer must stay alive until after vkQueueWaitIdle.
  // destroyBuffer(dev, scratch);
  // destroyBuffer(dev, instBuf);

  out.addr = getASAddress(dev, out.as);
  return out;
}

static void destroyAccel(VkDevice dev, Accel &a) {
  if (a.as) vkDestroyAccelerationStructureKHR(dev, a.as, nullptr);
  destroyBuffer(dev, a.backing);
  a = {};
}

// ---- Main ----

struct Vertex {
  float x, y, z;
};


struct Push {
  // int width;
  // int height;
  // float zOrigin;
  // float zDir;
  int rayCount;
  float originBase[3];
  float dir[3];
};


int main() {
  VK_CHECK(volkInitialize());

  // Instance
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "rt2d";
  app.apiVersion = VK_API_VERSION_1_3;

  const char *instExts[] = {VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME};

  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  ici.enabledExtensionCount = 1;
  ici.ppEnabledExtensionNames = instExts;

  VkInstance instance{};
  VK_CHECK(vkCreateInstance(&ici, nullptr, &instance));
  volkLoadInstance(instance);

  // ---- Pick RT-capable physical device ----
  uint32_t pdCount = 0;
  VK_CHECK(vkEnumeratePhysicalDevices(instance, &pdCount, nullptr));
  if (!pdCount) {
    std::cerr << "No GPU\n";
    return 1;
  }

  std::vector<VkPhysicalDevice> pds(pdCount);
  VK_CHECK(vkEnumeratePhysicalDevices(instance, &pdCount, pds.data()));

  VkPhysicalDevice phys = VK_NULL_HANDLE;

  std::cout << "Available GPUs:\n";

  for (auto pd: pds) {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(pd, &props);
    std::cout << "  " << props.deviceName << "\n";

    // Query RT support
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtFeat{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR
    };

    VkPhysicalDeviceFeatures2 feats{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    feats.pNext = &rtFeat;
    vkGetPhysicalDeviceFeatures2(pd, &feats);

    if (rtFeat.rayTracingPipeline) {
      phys = pd;
      std::cout << "  -> Selected for RT\n";
      break;
    }
  }

  if (phys == VK_NULL_HANDLE) {
    std::cerr << "No RT-capable GPU found\n";
    return 1;
  }

  // ---- Queue family (compute is fine) ----
  // ---- Queue family are just hardware capabilities that already exist on the physical device.
  uint32_t qfCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(phys, &qfCount, nullptr);
  std::vector<VkQueueFamilyProperties> qfs(qfCount);
  vkGetPhysicalDeviceQueueFamilyProperties(phys, &qfCount, qfs.data());

  // Queue families are not created by you. They already exist on the physical device (GPU).
  // You only query them and choose an index.
  uint32_t qfam = UINT32_MAX;
  for (uint32_t i = 0; i < qfCount; i++) {
    // This line just chooses one queue family that supports compute so the program can request a queue from it when creating the logical device.
    if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      qfam = i;
      break;
    }
  }

  if (qfam == UINT32_MAX) {
    std::cerr << "No compute queue found\n";
    return 1;
  }

  // ---- Required RT extensions (minimal) ----
  std::vector<const char *> devExts = {
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
    VK_KHR_SPIRV_1_4_EXTENSION_NAME,
    VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME
  };

  // ---- Query and enable features (correct chain root) ----
  VkPhysicalDeviceBufferDeviceAddressFeatures bda{
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES
  };

  VkPhysicalDeviceAccelerationStructureFeaturesKHR asf{
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR
  };

  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtf{
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR
  };

  // a linked list of features.
  VkPhysicalDeviceFeatures2 feats{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  feats.pNext = &rtf;
  rtf.pNext = &asf;
  asf.pNext = &bda;

  // find all supported features in a physical device and fills in the supported feature booleans
  vkGetPhysicalDeviceFeatures2(phys, &feats);

  if (!rtf.rayTracingPipeline || !asf.accelerationStructure || !bda.bufferDeviceAddress) {
    std::cerr << "RT features not fully supported on selected GPU\n";
    return 1;
  }

  // Enable them
  // rtf.rayTracingPipeline = VK_TRUE;
  // asf.accelerationStructure = VK_TRUE;
  // bda.bufferDeviceAddress = VK_TRUE;

  // ---- Device + queue ----
  float qprio = 1.0f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = qfam; // init qci with index of the selected queue family
  qci.queueCount = 1;
  qci.pQueuePriorities = &qprio;

  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  dci.enabledExtensionCount = (uint32_t) devExts.size();
  dci.ppEnabledExtensionNames = devExts.data();
  dci.pNext = &feats; // ✅ correct chain root

  // Create a logical device from this physical device with these extensions enabled
  VkDevice dev{};
  VK_CHECK(vkCreateDevice(phys, &dci, nullptr, &dev));
  volkLoadDevice(dev);

  // Get an arbitrary queue: the first one.
  VkQueue queue{};
  vkGetDeviceQueue(dev, qfam, 0, &queue);
  // This is where Vulkan/driver allocates 1 queue from family qfam for your new VkDevice.

  // ---- Rest of your original code unchanged from here ----
  // Command setup
  VkCommandPool pool = createCmdPool(dev, qfam);
  VkCommandBuffer cmd = createCmdBuffer(dev, pool);

  // Geometry buffers (3 triangles)
  const float EPSILON = 1e-7f;
  std::vector<Vertex> vertices = {
    // Triangle at z = 0
    {-1.0f - EPSILON, 0, 1.0f * EPSILON},
    {1.0f + EPSILON, EPSILON, 1.0f * EPSILON},
    {1.0f + EPSILON, -EPSILON, 1.0f * EPSILON},

    // Triangle at z = 1
    {-0.5f - EPSILON, 0, 2.0f * EPSILON},
    {0.5f + EPSILON, EPSILON, 2.0f * EPSILON},
    {0.5f + EPSILON, -EPSILON, 2.0f * EPSILON},

    // Triangle at z = 2
    {0.0f - EPSILON, 0, 3.0f * EPSILON},
    {0.0f + EPSILON, EPSILON, 3.0f * EPSILON},
    {0.0f + EPSILON, -EPSILON, 3.0f * EPSILON},
  };

  // In this example, indices are trivial
  // Auto-generate indices: 0,1,2,...,N-1
  std::vector<uint32_t> indices(vertices.size());
  std::iota(indices.begin(), indices.end(), 0u);

  Buffer vbo = createBuffer(dev, phys, sizeof(Vertex) * vertices.size(),
                            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                            true);
  Buffer ibo = createBuffer(dev, phys, sizeof(uint32_t) * indices.size(),
                            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                            true);

  std::memcpy(mapBuffer(dev, vbo), vertices.data(), sizeof(Vertex) * vertices.size());
  unmapBuffer(dev, vbo);
  std::memcpy(mapBuffer(dev, ibo), indices.data(), sizeof(uint32_t) * indices.size());
  unmapBuffer(dev, ibo);

  // Output image
  const uint32_t W = 5, H = 1;
  Image outIm = createStorageImageRGBA32F(dev, phys, W, H);
  // The current layout is undefined: it has no valid contents and it is not usable by any GPU operation yet
  // So your shader cannot work on this image now.

  // Begin Command Buffer
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

  // Build AS
  Accel blas = createBLAS_Triangles(
    dev, phys, cmd,
    vbo, (uint32_t) vertices.size(), sizeof(Vertex),
    ibo, (uint32_t) indices.size());

  Accel tlas = createTLAS_OneInstance(dev, phys, cmd, blas.addr);

  // Transition output image from UNDEFINED to GENERAL for storage writes
  // After that, the image is ready for operations from shader program.
  // TransitionImage is a pipeline barrier command: All commands that depend on this image will wait until the transition is finished.
  cmdTransitionImage(cmd, outIm.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

  // Descriptors: TLAS + storage image
  VkDescriptorSetLayoutBinding b0{};
  b0.binding = 0;
  b0.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
  b0.descriptorCount = 1;
  // b0.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
  b0.stageFlags =
      VK_SHADER_STAGE_RAYGEN_BIT_KHR |
      VK_SHADER_STAGE_MISS_BIT_KHR |
      VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
      VK_SHADER_STAGE_ANY_HIT_BIT_KHR;

  VkDescriptorSetLayoutBinding b1{};
  b1.binding = 1;
  b1.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  b1.descriptorCount = 1;
  b1.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

  VkDescriptorSetLayoutBinding bindings[] = {b0, b1};

  VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dslci.bindingCount = 2;
  dslci.pBindings = bindings;

  VkDescriptorSetLayout dsl{};
  VK_CHECK(vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl));

  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &dsl;

  const uint32_t RAY_COUNT = 5;

  Push push{};
  push.rayCount = RAY_COUNT;

  push.originBase[0] = 0.0f;
  push.originBase[1] = 0.0f;
  push.originBase[2] = 0.0f;

  // dir = +z (must be normalized)
  push.dir[0] = 0.0f;
  push.dir[1] = 0.0f;
  push.dir[2] = 1.0f;

  VkPushConstantRange pcr{};
  pcr.offset = 0;
  pcr.size = sizeof(Push);
  // pcr.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
  pcr.stageFlags =
      VK_SHADER_STAGE_RAYGEN_BIT_KHR |
      VK_SHADER_STAGE_MISS_BIT_KHR |
      VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
      VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pcr;

  VkPipelineLayout pipelineLayout{};
  VK_CHECK(vkCreatePipelineLayout(dev, &plci, nullptr, &pipelineLayout));

  VkDescriptorPoolSize ps[2]{};
  ps[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
  ps[0].descriptorCount = 1;
  ps[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  ps[1].descriptorCount = 1;

  VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpci.maxSets = 1;
  dpci.poolSizeCount = 2;
  dpci.pPoolSizes = ps;
  VkDescriptorPool dpool{};
  VK_CHECK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dpool));

  VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsai.descriptorPool = dpool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &dsl;
  VkDescriptorSet dset{};
  VK_CHECK(vkAllocateDescriptorSets(dev, &dsai, &dset));

  // Assign Descriptors to Bindings in Desc
  VkWriteDescriptorSetAccelerationStructureKHR asWrite{
    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR
  };
  asWrite.accelerationStructureCount = 1;
  asWrite.pAccelerationStructures = &tlas.as;

  VkWriteDescriptorSet w0{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  w0.dstSet = dset;
  w0.dstBinding = 0;
  w0.descriptorCount = 1;
  w0.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
  w0.pNext = &asWrite;

  VkDescriptorImageInfo di{};
  di.imageView = outIm.view;
  di.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

  VkWriteDescriptorSet w1{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  w1.dstSet = dset;
  w1.dstBinding = 1;
  w1.descriptorCount = 1;
  w1.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  w1.pImageInfo = &di;

  VkWriteDescriptorSet writes[] = {w0, w1};
  vkUpdateDescriptorSets(dev, 2, writes, 0, nullptr);

  // ===============================================================
  // Ray tracing pipeline (raygen + miss + chit)
  std::string shaderDir = SHADER_DIR;
  // auto raygenSpv = loadSpv("raygen.spv");
  // auto missSpv = loadSpv("miss.spv");
  // auto chitSpv = loadSpv("chit.spv");
  // auto ahitSpv = loadSpv("ahit.spv");
  auto raygenSpv = loadSpv((shaderDir + "/" + "raygen.spv").c_str());
  auto missSpv = loadSpv((shaderDir + "/" + "miss.spv").c_str());
  auto chitSpv = loadSpv((shaderDir + "/" +"chit.spv").c_str());
  auto ahitSpv = loadSpv((shaderDir + "/"+ "ahit.spv").c_str());

  auto makeModule = [&](const std::vector<uint32_t> &code) {
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = code.size() * 4;
    smci.pCode = code.data();
    VkShaderModule m{};
    VK_CHECK(vkCreateShaderModule(dev, &smci, nullptr, &m));
    return m;
  };

  VkShaderModule mRaygen = makeModule(raygenSpv);
  VkShaderModule mMiss = makeModule(missSpv);
  VkShaderModule mChit = makeModule(chitSpv);
  VkShaderModule mAhit = makeModule(ahitSpv);

  std::vector<VkPipelineShaderStageCreateInfo> stages;
  auto addStage = [&](VkShaderModule m, VkShaderStageFlagBits stage, const char *entry) {
    VkPipelineShaderStageCreateInfo s{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    s.stage = stage;
    s.module = m;
    s.pName = entry;
    stages.push_back(s);
  };

  // addStage(mRaygen, VK_SHADER_STAGE_RAYGEN_BIT_KHR, "raygenMain");
  // addStage(mMiss, VK_SHADER_STAGE_MISS_BIT_KHR, "missMain");
  // addStage(mChit, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, "chitMain");
  addStage(mRaygen, VK_SHADER_STAGE_RAYGEN_BIT_KHR, "main"); // index-0
  addStage(mMiss, VK_SHADER_STAGE_MISS_BIT_KHR, "main"); // index-1
  addStage(mChit, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, "main"); // index-2
  addStage(mAhit, VK_SHADER_STAGE_ANY_HIT_BIT_KHR, "main"); // index-3

  // Shader groups: 0=raygen, 1=miss, 2=hitgroup(chit)
  std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups;

  VkRayTracingShaderGroupCreateInfoKHR g0{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
  g0.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  g0.generalShader = 0;
  g0.closestHitShader = VK_SHADER_UNUSED_KHR;
  g0.anyHitShader = VK_SHADER_UNUSED_KHR;
  g0.intersectionShader = VK_SHADER_UNUSED_KHR;
  groups.push_back(g0);

  VkRayTracingShaderGroupCreateInfoKHR g1{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
  g1.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  g1.generalShader = 1;
  // g1.closestHitShader = VK_SHADER_UNUSED_KHR;
  // g1.anyHitShader = VK_SHADER_UNUSED_KHR;
  // g1.intersectionShader = VK_SHADER_UNUSED_KHR;
  groups.push_back(g1);

  VkRayTracingShaderGroupCreateInfoKHR g2{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
  g2.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
  g2.generalShader = VK_SHADER_UNUSED_KHR;
  g2.closestHitShader = 2;
  // g2.anyHitShader = VK_SHADER_UNUSED_KHR;
  g2.anyHitShader = 3;
  g2.intersectionShader = VK_SHADER_UNUSED_KHR;
  groups.push_back(g2);

  VkRayTracingPipelineCreateInfoKHR rpci{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
  rpci.stageCount = (uint32_t) stages.size();
  rpci.pStages = stages.data();
  rpci.groupCount = (uint32_t) groups.size();
  rpci.pGroups = groups.data();
  rpci.maxPipelineRayRecursionDepth = 1;
  rpci.layout = pipelineLayout;

  std::cout << "raygen size: " << raygenSpv.size() << "\n";
  std::cout << "miss size: " << missSpv.size() << "\n";
  std::cout << "chit size: " << chitSpv.size() << "\n";

  VkPipeline pipeline{};
  VK_CHECK(vkCreateRayTracingPipelinesKHR(dev, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &rpci, nullptr, &pipeline));

  // SBT
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtp{
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR
  };
  VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  p2.pNext = &rtp;
  vkGetPhysicalDeviceProperties2(phys, &p2);

  const uint32_t handleSize = rtp.shaderGroupHandleSize;
  const uint32_t handleAlign = rtp.shaderGroupHandleAlignment;
  auto alignUp = [&](uint32_t v, uint32_t a) { return (v + a - 1) & ~(a - 1); };
  const uint32_t handleSizeAligned = alignUp(handleSize, handleAlign);

  const uint32_t groupCount = (uint32_t) groups.size();
  std::vector<uint8_t> handles(groupCount * handleSize);
  VK_CHECK(vkGetRayTracingShaderGroupHandlesKHR(dev, pipeline, 0, groupCount, handles.size(), handles.data()));

  // One record each
  Buffer sbt = createBuffer(dev, phys, groupCount * (VkDeviceSize) handleSizeAligned,
                            VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                            true);

  // copy the data on the cpu side but gpu can also access it
  uint8_t *sbtMap = (uint8_t *) mapBuffer(dev, sbt);
  for (uint32_t i = 0; i < groupCount; i++)
    std::memcpy(sbtMap + i * handleSizeAligned, handles.data() + i * handleSize, handleSize);
  unmapBuffer(dev, sbt);

  VkStridedDeviceAddressRegionKHR rgenRegion{};
  VkStridedDeviceAddressRegionKHR missRegion{};
  VkStridedDeviceAddressRegionKHR hitRegion{};
  VkStridedDeviceAddressRegionKHR callRegion{};

  rgenRegion.deviceAddress = sbt.addr + 0 * handleSizeAligned;
  rgenRegion.stride = handleSizeAligned;
  rgenRegion.size = handleSizeAligned;

  missRegion.deviceAddress = sbt.addr + 1 * handleSizeAligned;
  missRegion.stride = handleSizeAligned;
  missRegion.size = handleSizeAligned;

  hitRegion.deviceAddress = sbt.addr + 2 * handleSizeAligned;
  hitRegion.stride = handleSizeAligned;
  hitRegion.size = handleSizeAligned;

  // Trace
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipelineLayout, 0, 1, &dset, 0, nullptr);
  vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0, sizeof(Push), &push);

  // ray tracing writes into outIm
  vkCmdTraceRaysKHR(cmd, &rgenRegion, &missRegion, &hitRegion, &callRegion, W, H, 1);

  // Copy image back: outIm -> linear staging buffer via vkCmdCopyImageToBuffer
  // After RT Pipeline Computation
  // Copy into a buffer that cpu can map later (GPU -> GPU)
  VkDeviceSize pixelStride = sizeof(float) * 4;
  VkDeviceSize readbackSize = (VkDeviceSize) W * H * pixelStride;

  Buffer readback = createBuffer(dev, phys, readbackSize,
                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                 false);

  // Transition for transfer
  cmdTransitionImage(cmd, outIm.img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

  VkBufferImageCopy bic{};
  bic.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  bic.imageSubresource.layerCount = 1;
  bic.imageExtent = {W, H, 1};

  vkCmdCopyImageToBuffer(cmd, outIm.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.buf, 1, &bic);

  submitAndWait(dev, queue, cmd);

  // Inspect some pixels
  float *data = (float *) mapBuffer(dev, readback);
  auto at = [&](uint32_t x, uint32_t y)-> float * {
    return data + (y * W + x) * 4;
  };

  // std::cout << "Sample pixels (x,y -> hitPos.xyz,w):\n";
  // for (auto [x,y]: std::vector<std::pair<uint32_t, uint32_t> >{{128, 128}, {64, 64}, {200, 200}, {10, 10}}) {
  //   float *p = at(x, y);
  //   std::cout << "(" << x << "," << y << ") -> "
  //       << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << "\n";
  // }
  //
  std::cout << "Ray results (closest.xyz, hitCount):\n";
  for (uint32_t i = 0; i < RAY_COUNT; i++) {
    float *p = at(i, 0);
    std::cout << "Ray " << i
        << " -> closest=("
        << p[0] << ", " << p[1] << ", " << p[2]
        << "), hits=" << p[3] << "\n";
  }

  unmapBuffer(dev, readback);

  // Cleanup
  destroyBuffer(dev, readback);
  destroyBuffer(dev, sbt);

  vkDestroyPipeline(dev, pipeline, nullptr);
  vkDestroyShaderModule(dev, mRaygen, nullptr);
  vkDestroyShaderModule(dev, mMiss, nullptr);
  vkDestroyShaderModule(dev, mChit, nullptr);

  vkDestroyDescriptorPool(dev, dpool, nullptr);
  vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
  vkDestroyPipelineLayout(dev, pipelineLayout, nullptr);

  destroyAccel(dev, tlas);
  destroyAccel(dev, blas);

  destroyImage(dev, outIm);
  destroyBuffer(dev, vbo);
  destroyBuffer(dev, ibo);

  vkDestroyCommandPool(dev, pool, nullptr);
  vkDestroyDevice(dev, nullptr);
  vkDestroyInstance(instance, nullptr);

  std::cout << "Done.\n";
  return 0;
}
