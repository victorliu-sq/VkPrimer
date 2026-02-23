// main.cpp - Vulkan KHR ray tracing, procedural AABB geometry for LSI
// - BLAS: AABBs (one per base edge)
// - TLAS: one instance
// - Rays: one per query edge (segment in XY, t in [0,1])
// - Intersection shader: segment-segment test
// - Any-hit: append results to SSBO

#define VOLK_IMPLEMENTATION
#include <volk/volk.h>

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
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
  VkBuffer buf{};
  VkDeviceMemory mem{};
  VkDeviceAddress addr{};
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

static void unmapBuffer(VkDevice dev, const Buffer &b) { vkUnmapMemory(dev, b.mem); }

static void destroyBuffer(VkDevice dev, Buffer &b) {
  if (b.buf) vkDestroyBuffer(dev, b.buf, nullptr);
  if (b.mem) vkFreeMemory(dev, b.mem, nullptr);
  b = {};
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
  VK_CHECK(vkEndCommandBuffer(cmd));
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  VK_CHECK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
  VK_CHECK(vkQueueWaitIdle(q));
}

// ---- Accel helpers ----
struct Accel {
  VkAccelerationStructureKHR as{};
  Buffer backing;
  VkDeviceAddress addr{};
};

static VkDeviceAddress getASAddress(VkDevice dev, VkAccelerationStructureKHR as) {
  VkAccelerationStructureDeviceAddressInfoKHR ai{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
  ai.accelerationStructure = as;
  return vkGetAccelerationStructureDeviceAddressKHR(dev, &ai);
}

static void cmdASBuildBarrier(VkCommandBuffer cmd) {
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
  mb.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
  vkCmdPipelineBarrier(cmd,
                       VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                       VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                       0, 1, &mb, 0, nullptr, 0, nullptr);
}

static Accel createBLAS_AABBs(
  VkDevice dev, VkPhysicalDevice phys, VkCommandBuffer cmd,
  const Buffer &aabbBuf, uint32_t aabbCount) {
  VkAccelerationStructureGeometryAabbsDataKHR aabbs{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR};
  aabbs.data.deviceAddress = aabbBuf.addr;
  aabbs.stride = sizeof(VkAabbPositionsKHR);

  VkAccelerationStructureGeometryKHR geom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  geom.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
  geom.flags = 0; // allow any-hit etc.
  geom.geometry.aabbs = aabbs;

  VkAccelerationStructureBuildGeometryInfoKHR bgi{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  bgi.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  bgi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  bgi.geometryCount = 1;
  bgi.pGeometries = &geom;

  uint32_t primCount = aabbCount;

  VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  vkGetAccelerationStructureBuildSizesKHR(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bgi, &primCount,
                                          &sizes);

  Accel out{};
  out.backing = createBuffer(dev, phys, sizes.accelerationStructureSize,
                             VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true);

  VkAccelerationStructureCreateInfoKHR asci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  asci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  asci.size = sizes.accelerationStructureSize;
  asci.buffer = out.backing.buf;
  VK_CHECK(vkCreateAccelerationStructureKHR(dev, &asci, nullptr, &out.as));

  Buffer scratch = createBuffer(dev, phys, sizes.buildScratchSize,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true);

  bgi.dstAccelerationStructure = out.as;
  bgi.scratchData.deviceAddress = scratch.addr;

  VkAccelerationStructureBuildRangeInfoKHR range{};
  range.primitiveCount = primCount;
  const VkAccelerationStructureBuildRangeInfoKHR *pRange = &range;

  vkCmdBuildAccelerationStructuresKHR(cmd, 1, &bgi, &pRange);
  cmdASBuildBarrier(cmd);

  destroyBuffer(dev, scratch);

  out.addr = getASAddress(dev, out.as);
  return out;
}

static Accel createTLAS_OneInstance(
  VkDevice dev, VkPhysicalDevice phys, VkCommandBuffer cmd,
  VkDeviceAddress blasAddr) {
  VkAccelerationStructureInstanceKHR inst{};
  inst.transform.matrix[0][0] = 1.f;
  inst.transform.matrix[1][1] = 1.f;
  inst.transform.matrix[2][2] = 1.f;
  inst.instanceCustomIndex = 0;
  inst.mask = 0xFF;
  inst.instanceShaderBindingTableRecordOffset = 0;
  inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
  inst.accelerationStructureReference = blasAddr;

  Buffer instBuf = createBuffer(dev, phys, sizeof(inst),
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
  out.backing = createBuffer(dev, phys, sizes.accelerationStructureSize,
                             VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true);

  VkAccelerationStructureCreateInfoKHR asci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  asci.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  asci.size = sizes.accelerationStructureSize;
  asci.buffer = out.backing.buf;
  VK_CHECK(vkCreateAccelerationStructureKHR(dev, &asci, nullptr, &out.as));

  Buffer scratch = createBuffer(dev, phys, sizes.buildScratchSize,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true);

  bgi.dstAccelerationStructure = out.as;
  bgi.scratchData.deviceAddress = scratch.addr;

  VkAccelerationStructureBuildRangeInfoKHR range{};
  range.primitiveCount = 1;
  const VkAccelerationStructureBuildRangeInfoKHR *pRange = &range;

  vkCmdBuildAccelerationStructuresKHR(cmd, 1, &bgi, &pRange);
  cmdASBuildBarrier(cmd);

  // NOTE: instBuf + scratch must remain alive until after queue idle in a robust app.
  // This sample keeps them alive by not destroying here (but you should in real code).
  // We'll destroy them after submit.
  out.addr = getASAddress(dev, out.as);
  return out;
}

static void destroyAccel(VkDevice dev, Accel &a) {
  if (a.as) vkDestroyAccelerationStructureKHR(dev, a.as, nullptr);
  destroyBuffer(dev, a.backing);
  a = {};
}

// -------------------
// App data structs
// -------------------
struct Point2 {
  float x, y;
};

struct Edge {
  uint32_t p1_idx, p2_idx, pad0, pad1;
};

struct HitRecord {
  uint32_t queryEid;
  uint32_t baseEid;
  float hitx;
  float hity;
};

struct Push {
  uint32_t queryEdgeCount;
  uint32_t maxOutHits;
  uint32_t pad0;
  uint32_t pad1;
};

int main(int argc, char **argv) {
  VK_CHECK(volkInitialize());

  // Instance
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "lsi_rt_aabb";
  app.apiVersion = VK_API_VERSION_1_3;

  const char *instExts[] = {VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME};
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  ici.enabledExtensionCount = 1;
  ici.ppEnabledExtensionNames = instExts;

  VkInstance instance{};
  VK_CHECK(vkCreateInstance(&ici, nullptr, &instance));
  volkLoadInstance(instance);

  // Pick RT-capable device
  uint32_t pdCount = 0;
  VK_CHECK(vkEnumeratePhysicalDevices(instance, &pdCount, nullptr));
  if (!pdCount) {
    std::cerr << "No GPU\n";
    return 1;
  }

  std::vector<VkPhysicalDevice> pds(pdCount);
  VK_CHECK(vkEnumeratePhysicalDevices(instance, &pdCount, pds.data()));

  VkPhysicalDevice phys = VK_NULL_HANDLE;
  for (auto pd: pds) {
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtFeat{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR
    };
    VkPhysicalDeviceFeatures2 feats{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    feats.pNext = &rtFeat;
    vkGetPhysicalDeviceFeatures2(pd, &feats);
    if (rtFeat.rayTracingPipeline) {
      phys = pd;
      break;
    }
  }
  if (!phys) {
    std::cerr << "No RT-capable GPU found\n";
    return 1;
  }

  // Queue family (compute)
  uint32_t qfCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(phys, &qfCount, nullptr);
  std::vector<VkQueueFamilyProperties> qfs(qfCount);
  vkGetPhysicalDeviceQueueFamilyProperties(phys, &qfCount, qfs.data());

  uint32_t qfam = UINT32_MAX;
  for (uint32_t i = 0; i < qfCount; i++) {
    if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      qfam = i;
      break;
    }
  }
  if (qfam == UINT32_MAX) {
    std::cerr << "No compute queue\n";
    return 1;
  }

  // Device extensions
  std::vector<const char *> devExts = {
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
    VK_KHR_SPIRV_1_4_EXTENSION_NAME,
    VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME
  };

  // Feature chain
  VkPhysicalDeviceBufferDeviceAddressFeatures bda{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
  VkPhysicalDeviceAccelerationStructureFeaturesKHR asf{
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR
  };
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtf{
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR
  };

  VkPhysicalDeviceFeatures2 feats{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  feats.pNext = &rtf;
  rtf.pNext = &asf;
  asf.pNext = &bda;
  vkGetPhysicalDeviceFeatures2(phys, &feats);

  if (!rtf.rayTracingPipeline || !asf.accelerationStructure || !bda.bufferDeviceAddress) {
    std::cerr << "RT features not supported\n";
    return 1;
  }

  float qprio = 1.0f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = qfam;
  qci.queueCount = 1;
  qci.pQueuePriorities = &qprio;

  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  dci.enabledExtensionCount = (uint32_t) devExts.size();
  dci.ppEnabledExtensionNames = devExts.data();
  dci.pNext = &feats;

  VkDevice dev{};
  VK_CHECK(vkCreateDevice(phys, &dci, nullptr, &dev));
  volkLoadDevice(dev);

  VkQueue queue{};
  vkGetDeviceQueue(dev, qfam, 0, &queue);

  // Command buffer
  VkCommandPool pool = createCmdPool(dev, qfam);
  VkCommandBuffer cmd = createCmdBuffer(dev, pool);

  // -------------------------
  // Demo geometry (replace later with your real points/edges)
  // -------------------------
  // Base edges: 3 small segments crossing various x
  std::vector<Point2> basePts = {
    {-0.8f, -0.2f}, {-0.2f, 0.2f},
    {-0.1f, -0.3f}, {0.4f, 0.3f},
    {0.2f, -0.4f}, {0.8f, 0.4f},
  };
  std::vector<Edge> baseEdges = {
    {0, 1, 0, 0},
    {2, 3, 0, 0},
    {4, 5, 0, 0},
  };

  // Query edges: 2 segments
  std::vector<Point2> queryPts = {
    {-1.0f, 0.0f}, {1.0f, 0.0f},
    {-1.0f, 0.2f}, {1.0f, 0.2f},
  };
  std::vector<Edge> queryEdges = {
    {0, 1, 0, 0},
    {2, 3, 0, 0},
  };

  const uint32_t QUERY_COUNT = (uint32_t) queryEdges.size();
  const uint32_t BASE_COUNT = (uint32_t) baseEdges.size();

  // Build AABBs (one per base edge)
  std::vector<VkAabbPositionsKHR> aabbs(BASE_COUNT);
  const float eps = 1e-5f;
  for (uint32_t i = 0; i < BASE_COUNT; i++) {
    auto e = baseEdges[i];
    Point2 p1 = basePts[e.p1_idx];
    Point2 p2 = basePts[e.p2_idx];
    float minx = std::min(p1.x, p2.x) - eps;
    float maxx = std::max(p1.x, p2.x) + eps;
    float miny = std::min(p1.y, p2.y) - eps;
    float maxy = std::max(p1.y, p2.y) + eps;

    aabbs[i].minX = minx;
    aabbs[i].maxX = maxx;
    aabbs[i].minY = miny;
    aabbs[i].maxY = maxy;
    aabbs[i].minZ = -eps;
    aabbs[i].maxZ = +eps;
  }

  // Upload buffers (host-visible for simplicity)
  auto makeHostSSBO = [&](VkDeviceSize sz)-> Buffer {
    return createBuffer(dev, phys, sz,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        true);
  };

  Buffer bQueryPts = makeHostSSBO(sizeof(Point2) * queryPts.size());
  Buffer bQueryEdge = makeHostSSBO(sizeof(Edge) * queryEdges.size());
  Buffer bBasePts = makeHostSSBO(sizeof(Point2) * basePts.size());
  Buffer bBaseEdge = makeHostSSBO(sizeof(Edge) * baseEdges.size());
  Buffer bAABBs = makeHostSSBO(sizeof(VkAabbPositionsKHR) * aabbs.size());

  std::memcpy(mapBuffer(dev, bQueryPts), queryPts.data(), bQueryPts.size);
  unmapBuffer(dev, bQueryPts);
  std::memcpy(mapBuffer(dev, bQueryEdge), queryEdges.data(), bQueryEdge.size);
  unmapBuffer(dev, bQueryEdge);
  std::memcpy(mapBuffer(dev, bBasePts), basePts.data(), bBasePts.size);
  unmapBuffer(dev, bBasePts);
  std::memcpy(mapBuffer(dev, bBaseEdge), baseEdges.data(), bBaseEdge.size);
  unmapBuffer(dev, bBaseEdge);
  std::memcpy(mapBuffer(dev, bAABBs), aabbs.data(), bAABBs.size);
  unmapBuffer(dev, bAABBs);

  // Output buffers
  const uint32_t MAX_HITS = 1024;
  Buffer bOutHits = createBuffer(dev, phys, sizeof(HitRecord) * MAX_HITS,
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                 false);

  Buffer bOutCounter = createBuffer(dev, phys, sizeof(uint32_t),
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                    false);

  // zero counter
  *(uint32_t *) mapBuffer(dev, bOutCounter) = 0;
  unmapBuffer(dev, bOutCounter);

  // Begin cmd
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

  // Build BLAS/TLAS
  Accel blas = createBLAS_AABBs(dev, phys, cmd, bAABBs, BASE_COUNT);
  Accel tlas = createTLAS_OneInstance(dev, phys, cmd, blas.addr);

  // -------------------------
  // Descriptors
  // -------------------------
  // set 0:
  // 0 TLAS
  // 1 query points
  // 2 query edges
  // 3 base points
  // 4 base edges
  // 5 out hits
  // 6 out counter

  VkDescriptorSetLayoutBinding b0{};
  b0.binding = 0;
  b0.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
  b0.descriptorCount = 1;
  b0.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                  VK_SHADER_STAGE_INTERSECTION_BIT_KHR |
                  VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                  VK_SHADER_STAGE_MISS_BIT_KHR |
                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

  auto ssboBinding = [&](uint32_t binding)-> VkDescriptorSetLayoutBinding {
    VkDescriptorSetLayoutBinding b{};
    b.binding = binding;
    b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                   VK_SHADER_STAGE_INTERSECTION_BIT_KHR |
                   VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    return b;
  };

  VkDescriptorSetLayoutBinding bindings[] = {
    b0,
    ssboBinding(1),
    ssboBinding(2),
    ssboBinding(3),
    ssboBinding(4),
    ssboBinding(5),
    ssboBinding(6),
  };

  VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dslci.bindingCount = (uint32_t) (sizeof(bindings) / sizeof(bindings[0]));
  dslci.pBindings = bindings;

  VkDescriptorSetLayout dsl{};
  VK_CHECK(vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl));

  // Push constants
  VkPushConstantRange pcr{};
  pcr.offset = 0;
  pcr.size = sizeof(Push);
  pcr.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                   VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                   VK_SHADER_STAGE_INTERSECTION_BIT_KHR;

  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &dsl;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pcr;

  VkPipelineLayout pipelineLayout{};
  VK_CHECK(vkCreatePipelineLayout(dev, &plci, nullptr, &pipelineLayout));

  // Pool + set
  VkDescriptorPoolSize ps[2]{};
  ps[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
  ps[0].descriptorCount = 1;
  ps[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  ps[1].descriptorCount = 6;

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

  // Writes
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

  auto bufInfo = [&](const Buffer &b)-> VkDescriptorBufferInfo {
    VkDescriptorBufferInfo bi{};
    bi.buffer = b.buf;
    bi.offset = 0;
    bi.range = b.size;
    return bi;
  };

  VkDescriptorBufferInfo i1 = bufInfo(bQueryPts);
  VkDescriptorBufferInfo i2 = bufInfo(bQueryEdge);
  VkDescriptorBufferInfo i3 = bufInfo(bBasePts);
  VkDescriptorBufferInfo i4 = bufInfo(bBaseEdge);
  VkDescriptorBufferInfo i5 = bufInfo(bOutHits);
  VkDescriptorBufferInfo i6 = bufInfo(bOutCounter);

  VkWriteDescriptorSet w[7]{};
  w[0] = w0;

  auto makeSSBOWrite = [&](uint32_t binding, VkDescriptorBufferInfo *info)-> VkWriteDescriptorSet {
    VkWriteDescriptorSet ww{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    ww.dstSet = dset;
    ww.dstBinding = binding;
    ww.descriptorCount = 1;
    ww.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ww.pBufferInfo = info;
    return ww;
  };

  w[1] = makeSSBOWrite(1, &i1);
  w[2] = makeSSBOWrite(2, &i2);
  w[3] = makeSSBOWrite(3, &i3);
  w[4] = makeSSBOWrite(4, &i4);
  w[5] = makeSSBOWrite(5, &i5);
  w[6] = makeSSBOWrite(6, &i6);

  vkUpdateDescriptorSets(dev, 7, w, 0, nullptr);

  // -------------------------
  // RT pipeline (raygen+miss+isect+anyhit+chit)
  // -------------------------
  // auto raygenSpv = loadSpv("raygen.spv");
  // auto missSpv   = loadSpv("miss.spv");
  // auto isectSpv  = loadSpv("isect.spv");
  // auto ahitSpv   = loadSpv("anyhit.spv");
  // auto chitSpv   = loadSpv("chit.spv");
  // Ray tracing pipeline (raygen + miss + isect + ahit + chit)
  std::string shaderDir = SHADER_DIR;

  auto raygenSpv = loadSpv((shaderDir + "/" + "raygen.spv").c_str());
  auto missSpv = loadSpv((shaderDir + "/" + "miss.spv").c_str());
  auto isectSpv = loadSpv((shaderDir + "/" + "isect.spv").c_str());
  auto ahitSpv = loadSpv((shaderDir + "/" + "ahit.spv").c_str());
  auto chitSpv = loadSpv((shaderDir + "/" + "chit.spv").c_str());

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
  VkShaderModule mIsect = makeModule(isectSpv);
  VkShaderModule mAhit = makeModule(ahitSpv);
  VkShaderModule mChit = makeModule(chitSpv);

  std::vector<VkPipelineShaderStageCreateInfo> stages;
  auto addStage = [&](VkShaderModule m, VkShaderStageFlagBits stage, const char *entry) {
    VkPipelineShaderStageCreateInfo s{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    s.stage = stage;
    s.module = m;
    s.pName = entry;
    stages.push_back(s);
  };

  // Entry names must match what you compiled from Slang
  addStage(mRaygen, VK_SHADER_STAGE_RAYGEN_BIT_KHR, "raygenMain"); // stage 0
  addStage(mMiss, VK_SHADER_STAGE_MISS_BIT_KHR, "missMain"); // stage 1
  addStage(mIsect, VK_SHADER_STAGE_INTERSECTION_BIT_KHR, "isectMain"); // stage 2
  addStage(mAhit, VK_SHADER_STAGE_ANY_HIT_BIT_KHR, "anyhitMain"); // stage 3
  addStage(mChit, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, "closesthitMain"); // stage 4

  // Shader groups:
  // 0: raygen general
  // 1: miss general
  // 2: procedural hitgroup (intersection + anyhit + closesthit optional)
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
  groups.push_back(g1);

  VkRayTracingShaderGroupCreateInfoKHR g2{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
  g2.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
  g2.generalShader = VK_SHADER_UNUSED_KHR;
  g2.intersectionShader = 2;
  g2.anyHitShader = 3;
  g2.closestHitShader = 4; // can be unused, but fine
  groups.push_back(g2);

  VkRayTracingPipelineCreateInfoKHR rpci{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
  rpci.stageCount = (uint32_t) stages.size();
  rpci.pStages = stages.data();
  rpci.groupCount = (uint32_t) groups.size();
  rpci.pGroups = groups.data();
  rpci.maxPipelineRayRecursionDepth = 1;
  rpci.layout = pipelineLayout;

  VkPipeline pipeline{};
  VK_CHECK(vkCreateRayTracingPipelinesKHR(dev, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &rpci, nullptr, &pipeline));

  // -------------------------
  // SBT
  // -------------------------
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

  Buffer sbt = createBuffer(dev, phys, groupCount * (VkDeviceSize) handleSizeAligned,
                            VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                            true);

  uint8_t *sbtMap = (uint8_t *) mapBuffer(dev, sbt);
  for (uint32_t i = 0; i < groupCount; i++)
    std::memcpy(sbtMap + i * handleSizeAligned, handles.data() + i * handleSize, handleSize);
  unmapBuffer(dev, sbt);

  VkStridedDeviceAddressRegionKHR rgenRegion{}, missRegion{}, hitRegion{}, callRegion{};
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

  Push push{};
  push.queryEdgeCount = QUERY_COUNT;
  push.maxOutHits = MAX_HITS;

  vkCmdPushConstants(cmd, pipelineLayout,
                     VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                     VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
                     0, sizeof(Push), &push);

  // 1D launch: width=queryEdgeCount, height=1
  vkCmdTraceRaysKHR(cmd, &rgenRegion, &missRegion, &hitRegion, &callRegion, QUERY_COUNT, 1, 1);

  submitAndWait(dev, queue, cmd);

  // Read back hits
  uint32_t hitCount = *(uint32_t *) mapBuffer(dev, bOutCounter);
  unmapBuffer(dev, bOutCounter);

  std::cout << "HitCount = " << hitCount << "\n";
  hitCount = std::min(hitCount, MAX_HITS);

  HitRecord *hits = (HitRecord *) mapBuffer(dev, bOutHits);
  for (uint32_t i = 0; i < hitCount; i++) {
    auto &h = hits[i];
    std::cout << "hit[" << i << "] queryEid=" << h.queryEid
        << " baseEid=" << h.baseEid
        << " P=(" << h.hitx << "," << h.hity << ")\n";
  }
  unmapBuffer(dev, bOutHits);

  // Cleanup (sample-level)
  destroyBuffer(dev, sbt);
  vkDestroyPipeline(dev, pipeline, nullptr);

  vkDestroyShaderModule(dev, mRaygen, nullptr);
  vkDestroyShaderModule(dev, mMiss, nullptr);
  vkDestroyShaderModule(dev, mIsect, nullptr);
  vkDestroyShaderModule(dev, mAhit, nullptr);
  vkDestroyShaderModule(dev, mChit, nullptr);

  vkDestroyDescriptorPool(dev, dpool, nullptr);
  vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
  vkDestroyPipelineLayout(dev, pipelineLayout, nullptr);

  destroyAccel(dev, tlas);
  destroyAccel(dev, blas);

  destroyBuffer(dev, bQueryPts);
  destroyBuffer(dev, bQueryEdge);
  destroyBuffer(dev, bBasePts);
  destroyBuffer(dev, bBaseEdge);
  destroyBuffer(dev, bAABBs);
  destroyBuffer(dev, bOutHits);
  destroyBuffer(dev, bOutCounter);

  vkDestroyCommandPool(dev, pool, nullptr);
  vkDestroyDevice(dev, nullptr);
  vkDestroyInstance(instance, nullptr);

  std::cout << "Done.\n";
  return 0;
}
