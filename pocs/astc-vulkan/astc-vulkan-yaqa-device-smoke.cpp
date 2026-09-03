#include "astc-vulkan-yaqa.h"

#include <vulkan/vulkan.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

namespace {

std::vector<uint32_t> read_spirv(const char * path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) return {};
    const auto bytes = input.tellg();
    if (bytes <= 0 || bytes % static_cast<std::streamsize>(sizeof(uint32_t)) != 0) return {};
    std::vector<uint32_t> result(static_cast<size_t>(bytes) / sizeof(uint32_t));
    input.seekg(0); input.read(reinterpret_cast<char *>(result.data()), bytes);
    return input ? result : std::vector<uint32_t>{};
}

bool choose_compute(VkInstance instance, VkPhysicalDevice & physical, uint32_t & family) {
    uint32_t count = 0; if (vkEnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS || !count) return false;
    std::vector<VkPhysicalDevice> devices(count); if (vkEnumeratePhysicalDevices(instance, &count, devices.data()) != VK_SUCCESS) return false;
    // Prefer the discrete GPU: this smoke intentionally has no ASTC requirement.
    for (int pass = 0; pass < 2; ++pass) for (auto candidate : devices) {
        VkPhysicalDeviceProperties props{}; vkGetPhysicalDeviceProperties(candidate, &props);
        if ((pass == 0) != (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)) continue;
        uint32_t queues = 0; vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queues, nullptr);
        std::vector<VkQueueFamilyProperties> q(queues); vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queues, q.data());
        for (uint32_t i = 0; i < queues; ++i) if (q[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { physical = candidate; family = i; return true; }
    }
    return false;
}

bool make_buffer(VkPhysicalDevice physical, VkDevice device, VkDeviceSize bytes, VkBuffer & buffer, VkDeviceMemory & memory) {
    const VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
    if (vkCreateBuffer(device, &info, nullptr, &buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements requirements{}; vkGetBufferMemoryRequirements(device, buffer, &requirements);
    VkPhysicalDeviceMemoryProperties properties{}; vkGetPhysicalDeviceMemoryProperties(physical, &properties);
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) if ((requirements.memoryTypeBits & (1u << i)) &&
        (properties.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
        (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) { type = i; break; }
    if (type == UINT32_MAX) return false;
    const VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, requirements.size, type};
    return vkAllocateMemory(device, &allocation, nullptr, &memory) == VK_SUCCESS && vkBindBufferMemory(device, buffer, memory, 0) == VK_SUCCESS;
}

void write_memory(VkDevice device, VkDeviceMemory memory, const void * data, size_t bytes) {
    void * mapped = nullptr; vkMapMemory(device, memory, 0, bytes, 0, &mapped); std::memcpy(mapped, data, bytes); vkUnmapMemory(device, memory);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    const auto spirv = read_spirv(argv[1]); if (spirv.empty()) return 77;
    const std::vector<float> error{1.0f, -2.0f, 0.5f, 3.0f, -1.0f, 2.0f};
    const std::vector<float> input{1.0f, .5f, -1.0f, 2.0f, -1.0f, .25f}; // 2x3
    const std::vector<float> output{.25f, 1.0f, -2.0f, .5f}; // 2x2
    const double expected = astc_vulkan_yaqa_trace_score(error, 2, 3, input, output, 2);
    const VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "astc-yaqa-smoke", 1, "llama.cpp", 1, VK_API_VERSION_1_0};
    const VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0, &app, 0, nullptr, 0, nullptr};
    VkInstance instance = VK_NULL_HANDLE; if (vkCreateInstance(&instance_info, nullptr, &instance) != VK_SUCCESS) return 77;
    VkPhysicalDevice physical = VK_NULL_HANDLE; uint32_t family = UINT32_MAX;
    if (!choose_compute(instance, physical, family)) { vkDestroyInstance(instance, nullptr); return 77; }
    constexpr float priority = 1.0f; const VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, family, 1, &priority};
    const VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, nullptr, 0, 1, &queue_info, 0, nullptr, 0, nullptr, nullptr};
    VkDevice device = VK_NULL_HANDLE; if (vkCreateDevice(physical, &device_info, nullptr, &device) != VK_SUCCESS) { vkDestroyInstance(instance, nullptr); return 77; }
    VkQueue queue = VK_NULL_HANDLE; vkGetDeviceQueue(device, family, 0, &queue);
    VkBuffer buffers[4]{}; VkDeviceMemory memory[4]{}; const size_t bytes[] = {error.size()*4, input.size()*4, output.size()*4, 4};
    bool ok = true; for (uint32_t i=0; i<4; ++i) ok &= make_buffer(physical, device, bytes[i], buffers[i], memory[i]);
    if (ok) { write_memory(device,memory[0],error.data(),bytes[0]); write_memory(device,memory[1],input.data(),bytes[1]); write_memory(device,memory[2],output.data(),bytes[2]); }
    VkDescriptorSetLayout layout=VK_NULL_HANDLE; VkDescriptorPool pool=VK_NULL_HANDLE; VkDescriptorSet set=VK_NULL_HANDLE; VkPipelineLayout pipeline_layout=VK_NULL_HANDLE; VkShaderModule module=VK_NULL_HANDLE; VkPipeline pipeline=VK_NULL_HANDLE; VkCommandPool command_pool=VK_NULL_HANDLE; VkCommandBuffer command=VK_NULL_HANDLE; VkFence fence=VK_NULL_HANDLE;
    const VkDescriptorSetLayoutBinding bindings[4]={{0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr},{1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr},{2,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr},{3,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr}};
    const VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,nullptr,0,4,bindings};
    const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,4}; const VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,nullptr,0,1,1,&pool_size};
    const VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,nullptr,pool,1,&layout};
    if (ok) ok &= vkCreateDescriptorSetLayout(device,&layout_info,nullptr,&layout)==VK_SUCCESS && vkCreateDescriptorPool(device,&pool_info,nullptr,&pool)==VK_SUCCESS;
    VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,nullptr,pool,1,&layout}; if (ok) ok &= vkAllocateDescriptorSets(device,&allocation,&set)==VK_SUCCESS;
    VkDescriptorBufferInfo infos[4]={{buffers[0],0,bytes[0]},{buffers[1],0,bytes[1]},{buffers[2],0,bytes[2]},{buffers[3],0,bytes[3]}}; VkWriteDescriptorSet writes[4]{};
    for(uint32_t i=0;i<4;++i){writes[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;writes[i].dstSet=set;writes[i].dstBinding=i;writes[i].descriptorCount=1;writes[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;writes[i].pBufferInfo=&infos[i];} if(ok) vkUpdateDescriptorSets(device,4,writes,0,nullptr);
    const VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,nullptr,0,spirv.size()*4,spirv.data()}; const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT,0,12}; const VkPipelineLayoutCreateInfo pl_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,nullptr,0,1,&layout,1,&range};
    if(ok) ok &= vkCreateShaderModule(device,&shader_info,nullptr,&module)==VK_SUCCESS && vkCreatePipelineLayout(device,&pl_info,nullptr,&pipeline_layout)==VK_SUCCESS;
    const VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,module,"main",nullptr}; const VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,nullptr,0,stage,pipeline_layout,VK_NULL_HANDLE,-1}; if(ok) ok &= vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&pipeline_info,nullptr,&pipeline)==VK_SUCCESS;
    const VkCommandPoolCreateInfo cp_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,nullptr,0,family}; if(ok) ok &= vkCreateCommandPool(device,&cp_info,nullptr,&command_pool)==VK_SUCCESS;
    const VkCommandBufferAllocateInfo cb_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,nullptr,command_pool,VK_COMMAND_BUFFER_LEVEL_PRIMARY,1}; if(ok) ok &= vkAllocateCommandBuffers(device,&cb_info,&command)==VK_SUCCESS; const VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,nullptr,0}; if(ok) ok &= vkCreateFence(device,&fence_info,nullptr,&fence)==VK_SUCCESS;
    const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,nullptr,VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,nullptr}; const uint32_t push[]={2,3,2}; if(ok) ok &= vkBeginCommandBuffer(command,&begin)==VK_SUCCESS; if(ok){vkCmdBindPipeline(command,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);vkCmdBindDescriptorSets(command,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline_layout,0,1,&set,0,nullptr);vkCmdPushConstants(command,pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,12,push);vkCmdDispatch(command,1,1,1);ok &= vkEndCommandBuffer(command)==VK_SUCCESS;}
    const VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO,nullptr,0,nullptr,nullptr,1,&command,0,nullptr}; if(ok) ok &= vkQueueSubmit(queue,1,&submit,fence)==VK_SUCCESS && vkWaitForFences(device,1,&fence,VK_TRUE,UINT64_MAX)==VK_SUCCESS;
    float actual=NAN; if(ok){void *mapped=nullptr;ok &= vkMapMemory(device,memory[3],0,4,0,&mapped)==VK_SUCCESS;if(ok){std::memcpy(&actual,mapped,4);vkUnmapMemory(device,memory[3]);}}
    if(fence)vkDestroyFence(device,fence,nullptr);if(command_pool)vkDestroyCommandPool(device,command_pool,nullptr);if(pipeline)vkDestroyPipeline(device,pipeline,nullptr);if(module)vkDestroyShaderModule(device,module,nullptr);if(pipeline_layout)vkDestroyPipelineLayout(device,pipeline_layout,nullptr);if(pool)vkDestroyDescriptorPool(device,pool,nullptr);if(layout)vkDestroyDescriptorSetLayout(device,layout,nullptr);for(uint32_t i=0;i<4;++i){if(buffers[i])vkDestroyBuffer(device,buffers[i],nullptr);if(memory[i])vkFreeMemory(device,memory[i],nullptr);}vkDestroyDevice(device,nullptr);vkDestroyInstance(instance,nullptr);
    if(!ok || std::fabs(actual-static_cast<float>(expected))>1e-4f){std::fprintf(stderr,"YAQA GPU smoke failed: %.7f vs %.7f\n",actual,expected);return 1;} std::printf("YAQA GPU trace-score smoke passed: %.7f\n",actual); return 0;
}
