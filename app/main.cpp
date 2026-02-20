#include <vulkan/vulkan.h>
#include <cassert>
#include <cstdio>
#include <vector>
#include <iostream>

#define VK_CHECK(x) do { VkResult err = (x); assert(err == VK_SUCCESS); } while(0)

static std::vector<uint32_t> readSpirvU32(const char *filename) {
  FILE *f = std::fopen(filename, "rb");
  assert(f);

  std::fseek(f, 0, SEEK_END);
  size_t sizeBytes = std::ftell(f);
  std::rewind(f);

  assert(sizeBytes % 4 == 0); // SPIR-V is 32-bit words

  std::vector<uint32_t> words(sizeBytes / 4);
  size_t read = std::fread(words.data(), 1, sizeBytes, f);
  assert(read == sizeBytes);

  std::fclose(f);
  return words;
}

static uint32_t findQueueFamily(VkPhysicalDevice phys, VkQueueFlags required) {
  uint32_t count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, nullptr);
  assert(count > 0);

  std::vector<VkQueueFamilyProperties> props(count);
  vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, props.data());

  for (uint32_t i = 0; i < count; i++)
    if ((props[i].queueFlags & required) == required)
      return i;

  assert(false && "No suitable queue family found");
  return 0;
}

static uint32_t findMemoryType(VkPhysicalDevice phys, uint32_t typeBits, VkMemoryPropertyFlags props) {
  VkPhysicalDeviceMemoryProperties mp{};
  vkGetPhysicalDeviceMemoryProperties(phys, &mp);

  for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
    if ((typeBits & (1u << i)) && ((mp.memoryTypes[i].propertyFlags & props) == props))
      return i;
  }
  assert(false && "No suitable memory type found");
  return 0;
}

int main() {
  const uint32_t N = 4;

  // ---- Instance ----
  VkInstance instance = VK_NULL_HANDLE;
  VkInstanceCreateInfo ici{};
  ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  VK_CHECK(vkCreateInstance(&ici, nullptr, &instance));

  // ---- Physical device ----
  uint32_t gpuCount = 0;
  VK_CHECK(vkEnumeratePhysicalDevices(instance, &gpuCount, nullptr));
  assert(gpuCount > 0);

  std::vector<VkPhysicalDevice> gpus(gpuCount);
  VK_CHECK(vkEnumeratePhysicalDevices(instance, &gpuCount, gpus.data()));
  VkPhysicalDevice phys = gpus[0];

  // ---- Compute queue family ----
  uint32_t queueFamilyIndex = findQueueFamily(phys, VK_QUEUE_COMPUTE_BIT);

  // ---- Device + queue ----
  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci{};
  qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  qci.queueFamilyIndex = queueFamilyIndex;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;

  VkDeviceCreateInfo dci{};
  dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;

  VkDevice device = VK_NULL_HANDLE;
  VK_CHECK(vkCreateDevice(phys, &dci, nullptr, &device));

  VkQueue queue = VK_NULL_HANDLE;
  vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

  // ---- Command pool/buffer ----
  VkCommandPool pool = VK_NULL_HANDLE;
  VkCommandPoolCreateInfo pci{};
  pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pci.queueFamilyIndex = queueFamilyIndex;
  VK_CHECK(vkCreateCommandPool(device, &pci, nullptr, &pool));

  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkCommandBufferAllocateInfo cbai{};
  cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbai.commandPool = pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VK_CHECK(vkAllocateCommandBuffers(device, &cbai, &cmd));

  // ---- Buffers (A, B, Out) ----
  VkBuffer buf[3]{};
  VkDeviceMemory mem[3]{};

  auto makeBuffer = [&](int idx) {
    // Create Memory Buffer
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = sizeof(float) * N;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VK_CHECK(vkCreateBuffer(device, &bci, nullptr, &buf[idx]));

    // Allocate Device Memory
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device, buf[idx], &req);

    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = findMemoryType(
      phys, req.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VK_CHECK(vkAllocateMemory(device, &mai, nullptr, &mem[idx]));

    // Bind Device Memory to Memory Buffer
    VK_CHECK(vkBindBufferMemory(device, buf[idx], mem[idx], 0));
  };

  makeBuffer(0);
  makeBuffer(1);
  makeBuffer(2);

  // Write A, B
  float *p = nullptr;
  VK_CHECK(vkMapMemory(device, mem[0], 0, VK_WHOLE_SIZE, 0, (void**)&p));
  for (uint32_t i = 0; i < N; i++) {
    p[i] = float(i);
  }
  vkUnmapMemory(device, mem[0]);

  VK_CHECK(vkMapMemory(device, mem[1], 0, VK_WHOLE_SIZE, 0, (void**)&p));
  for (uint32_t i = 0; i < N; i++) {
    p[i] = float(i) * 10.0f;
  }
  vkUnmapMemory(device, mem[1]);

  // ---- Descriptor set layout ----
  VkDescriptorSetLayoutBinding bindings[3]{};
  for (uint32_t i = 0; i < 3; i++) {
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }

  VkDescriptorSetLayout layout = VK_NULL_HANDLE;
  VkDescriptorSetLayoutCreateInfo dlci{};
  dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dlci.bindingCount = 3;
  dlci.pBindings = bindings;
  VK_CHECK(vkCreateDescriptorSetLayout(device, &dlci, nullptr, &layout));

  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkPipelineLayoutCreateInfo plci{};
  plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &layout;
  VK_CHECK(vkCreatePipelineLayout(device, &plci, nullptr, &pipelineLayout));

  // ---- Descriptor pool + set ----
  VkDescriptorPool poolDesc = VK_NULL_HANDLE;
  VkDescriptorPoolSize ps{};
  ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  ps.descriptorCount = 3;

  VkDescriptorPoolCreateInfo dpci{};
  dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpci.maxSets = 1;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &ps;
  VK_CHECK(vkCreateDescriptorPool(device, &dpci, nullptr, &poolDesc));

  VkDescriptorSet descSet = VK_NULL_HANDLE;
  VkDescriptorSetAllocateInfo dsai{};
  dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsai.descriptorPool = poolDesc;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &layout;
  VK_CHECK(vkAllocateDescriptorSets(device, &dsai, &descSet));

  VkDescriptorBufferInfo dbi[3]{};
  for (int i = 0; i < 3; i++) {
    dbi[i].buffer = buf[i];
    dbi[i].offset = 0;
    dbi[i].range = VK_WHOLE_SIZE;
  }

  VkWriteDescriptorSet writes[3]{};
  for (uint32_t i = 0; i < 3; i++) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = descSet;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &dbi[i];
  }
  vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);

  // ---- Shader module (ALIGNED SPIR-V WORDS!) ----
  auto spirv = readSpirvU32("add.spv"); // must be valid Vulkan SPIR-V
  VkShaderModule shader = VK_NULL_HANDLE;

  VkShaderModuleCreateInfo smci{};
  smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  smci.codeSize = spirv.size() * sizeof(uint32_t);
  smci.pCode = spirv.data();
  VK_CHECK(vkCreateShaderModule(device, &smci, nullptr, &shader));

  // ---- Compute pipeline (SAFE INIT) ----
  VkPipelineShaderStageCreateInfo stage{};
  stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage.module = shader;
  stage.pName = "main";

  VkComputePipelineCreateInfo cpci{};
  cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  cpci.stage = stage;
  cpci.layout = pipelineLayout;

  VkPipeline pipeline = VK_NULL_HANDLE;
  VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline));

  // ---- Record/submit ----
  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout,
                          0, 1, &descSet, 0, nullptr);
  vkCmdDispatch(cmd, N, 1, 1);

  VK_CHECK(vkEndCommandBuffer(cmd));

  VkSubmitInfo si{};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;

  VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
  VK_CHECK(vkQueueWaitIdle(queue));

  // ---- Read Out ----
  VK_CHECK(vkMapMemory(device, mem[2], 0, VK_WHOLE_SIZE, 0, (void**)&p));
  std::cout << "Result:\n";
  for (uint32_t i = 0; i < N; i++) std::cout << p[i] << "\n";
  vkUnmapMemory(device, mem[2]);

  std::cout << "Done\n";
  return 0;
}
