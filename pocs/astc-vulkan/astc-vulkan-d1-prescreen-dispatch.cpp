#include "astc-vulkan-d1-prescreen-dispatch.h"
#include "astc-vulkan-resource.h"

#include <vulkan/vulkan.h>

#include <cstring>
#include <cmath>
#include <fstream>
#include <limits>
#include <utility>

namespace {
std::vector<uint32_t> read_spirv(const std::string & path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    const auto size = file.tellg();
    if (size <= 0 || size % static_cast<std::streamoff>(sizeof(uint32_t)) != 0) return {};
    std::vector<uint32_t> code(static_cast<size_t>(size) / sizeof(uint32_t));
    file.seekg(0); file.read(reinterpret_cast<char *>(code.data()), size);
    return file ? code : std::vector<uint32_t>{};
}
bool choose_compute(VkInstance instance, VkPhysicalDevice & physical, uint32_t & family) {
    uint32_t count = 0; if (vkEnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS || !count) return false;
    std::vector<VkPhysicalDevice> devices(count); if (vkEnumeratePhysicalDevices(instance, &count, devices.data()) != VK_SUCCESS) return false;
    for (int pass = 0; pass < 2; ++pass) for (auto device : devices) {
        VkPhysicalDeviceProperties props{}; vkGetPhysicalDeviceProperties(device, &props);
        if ((pass == 0) != (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)) continue;
        uint32_t queues = 0; vkGetPhysicalDeviceQueueFamilyProperties(device, &queues, nullptr);
        std::vector<VkQueueFamilyProperties> properties(queues); vkGetPhysicalDeviceQueueFamilyProperties(device, &queues, properties.data());
        for (uint32_t index = 0; index < queues; ++index) if (properties[index].queueFlags & VK_QUEUE_COMPUTE_BIT) { physical = device; family = index; return true; }
    }
    return false;
}
bool buffer(VkPhysicalDevice physical, VkDevice device, VkDeviceSize bytes, VkBuffer & handle, VkDeviceMemory & memory) {
    const VkBufferCreateInfo create{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
    if (vkCreateBuffer(device, &create, nullptr, &handle) != VK_SUCCESS) return false;
    VkMemoryRequirements requirements{}; vkGetBufferMemoryRequirements(device, handle, &requirements);
    const uint32_t type = astc_vulkan_find_memory_type(physical, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (type == std::numeric_limits<uint32_t>::max()) return false;
    const VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, requirements.size, type};
    return vkAllocateMemory(device, &allocate, nullptr, &memory) == VK_SUCCESS && vkBindBufferMemory(device, handle, memory, 0) == VK_SUCCESS;
}
bool write(VkDevice device, VkDeviceMemory memory, const void * data, VkDeviceSize bytes) {
    void * mapped = nullptr; if (vkMapMemory(device, memory, 0, bytes, 0, &mapped) != VK_SUCCESS) return false;
    std::memcpy(mapped, data, static_cast<size_t>(bytes)); vkUnmapMemory(device, memory); return true;
}
std::pair<uint32_t, uint32_t> dimensions(astc_vulkan_footprint footprint) {
    switch (footprint) { case astc_vulkan_footprint::k4x4: return {4,4}; case astc_vulkan_footprint::k5x5: return {5,5}; case astc_vulkan_footprint::k6x6: return {6,6}; case astc_vulkan_footprint::k8x6: return {8,6}; case astc_vulkan_footprint::k10x6: return {10,6}; case astc_vulkan_footprint::k8x8: return {8,8}; case astc_vulkan_footprint::k10x8: return {10,8}; default: return {0,0}; }
}
}

bool astc_vulkan_score_d1_prescreen_gpu_default(const std::string & spirv_path,
    const std::vector<float> & weights, uint32_t rows, uint32_t columns,
    const std::vector<float> & energy,
    const std::vector<astc_vulkan_d1_prescreen_candidate> & candidates,
    std::vector<astc_vulkan_d1_prescreen_score> & scores, std::string & error) {
    std::vector<astc_vulkan_d1_prescreen_score> cpu_scores;
    if (!astc_vulkan_score_d1_prescreen_cpu(weights, rows, columns, energy, candidates, cpu_scores)) { error = "invalid D1 pre-screen inputs"; return false; }
    const auto code = read_spirv(spirv_path); if (code.empty()) { error = "invalid D1 pre-screen SPIR-V"; return false; }
    struct gpu_candidate { uint32_t width, height, levels, reserved; };
    std::vector<gpu_candidate> gpu_candidates; for (const auto & candidate : candidates) { const auto d = dimensions(candidate.footprint); if (!d.first) { error = "unsupported D1 pre-screen footprint"; return false; } gpu_candidates.push_back({d.first,d.second,candidate.levels,0}); }
    VkInstance instance = VK_NULL_HANDLE; VkDevice device = VK_NULL_HANDLE; VkPhysicalDevice physical = VK_NULL_HANDLE; uint32_t family = UINT32_MAX; VkQueue queue = VK_NULL_HANDLE;
    VkBuffer buffers[4]{}; VkDeviceMemory memory[4]{}; VkDescriptorSetLayout layout = VK_NULL_HANDLE; VkDescriptorPool pool = VK_NULL_HANDLE; VkDescriptorSet set = VK_NULL_HANDLE; VkPipelineLayout pipeline_layout = VK_NULL_HANDLE; VkShaderModule module = VK_NULL_HANDLE; VkPipeline pipeline = VK_NULL_HANDLE; VkCommandPool command_pool = VK_NULL_HANDLE; VkCommandBuffer command = VK_NULL_HANDLE; VkFence fence = VK_NULL_HANDLE;
    bool ok = true;
    const VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO,nullptr,"astc-d1-prescreen",1,"llama.cpp",1,VK_API_VERSION_1_1}; const VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,nullptr,0,&app,0,nullptr,0,nullptr};
    ok &= vkCreateInstance(&instance_info,nullptr,&instance)==VK_SUCCESS && choose_compute(instance,physical,family);
    constexpr float priority=1.0f; const VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,nullptr,0,family,1,&priority}; const VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,nullptr,0,1,&qi,0,nullptr,0,nullptr,nullptr};
    if (ok) ok &= vkCreateDevice(physical,&di,nullptr,&device)==VK_SUCCESS; if (ok) vkGetDeviceQueue(device,family,0,&queue);
    const VkDeviceSize sizes[]={weights.size()*sizeof(float),energy.size()*sizeof(float),gpu_candidates.size()*sizeof(gpu_candidate),candidates.size()*sizeof(float)};
    for(uint32_t i=0; ok&&i<4; ++i) ok &= buffer(physical,device,sizes[i],buffers[i],memory[i]);
    if(ok) ok &= write(device,memory[0],weights.data(),sizes[0]) && write(device,memory[1],energy.data(),sizes[1]) && write(device,memory[2],gpu_candidates.data(),sizes[2]);
    const VkDescriptorSetLayoutBinding bindings[4]={{0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr},{1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr},{2,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr},{3,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr}}; const VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,nullptr,0,4,bindings}; const VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,4}; const VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,nullptr,0,1,1,&ps};
    if(ok) ok &= vkCreateDescriptorSetLayout(device,&li,nullptr,&layout)==VK_SUCCESS && vkCreateDescriptorPool(device,&pi,nullptr,&pool)==VK_SUCCESS; const VkDescriptorSetAllocateInfo sai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,nullptr,pool,1,&layout}; if(ok) ok &= vkAllocateDescriptorSets(device,&sai,&set)==VK_SUCCESS;
    VkDescriptorBufferInfo infos[4]={{buffers[0],0,sizes[0]},{buffers[1],0,sizes[1]},{buffers[2],0,sizes[2]},{buffers[3],0,sizes[3]}}; VkWriteDescriptorSet writes[4]{}; for(uint32_t i=0;i<4;++i){writes[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;writes[i].dstSet=set;writes[i].dstBinding=i;writes[i].descriptorCount=1;writes[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;writes[i].pBufferInfo=&infos[i];} if(ok) vkUpdateDescriptorSets(device,4,writes,0,nullptr);
    const VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,nullptr,0,code.size()*sizeof(uint32_t),code.data()}; const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT,0,12}; const VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,nullptr,0,1,&layout,1,&range}; if(ok) ok &= vkCreateShaderModule(device,&smi,nullptr,&module)==VK_SUCCESS && vkCreatePipelineLayout(device,&pli,nullptr,&pipeline_layout)==VK_SUCCESS;
    const VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,module,"main",nullptr}; const VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,nullptr,0,stage,pipeline_layout,VK_NULL_HANDLE,-1}; if(ok) ok &= vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&cpi,nullptr,&pipeline)==VK_SUCCESS;
    const VkCommandPoolCreateInfo cpool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,nullptr,0,family}; const VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,nullptr,command_pool,VK_COMMAND_BUFFER_LEVEL_PRIMARY,1}; const VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,nullptr,0}; if(ok) ok &= vkCreateCommandPool(device,&cpool,nullptr,&command_pool)==VK_SUCCESS; const VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,nullptr,command_pool,VK_COMMAND_BUFFER_LEVEL_PRIMARY,1}; if(ok) ok &= vkAllocateCommandBuffers(device,&ai,&command)==VK_SUCCESS && vkCreateFence(device,&fi,nullptr,&fence)==VK_SUCCESS;
    const VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,nullptr,VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,nullptr}; const uint32_t push[]={rows,columns,static_cast<uint32_t>(candidates.size())}; if(ok) ok &= vkBeginCommandBuffer(command,&bi)==VK_SUCCESS; if(ok){vkCmdBindPipeline(command,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);vkCmdBindDescriptorSets(command,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline_layout,0,1,&set,0,nullptr);vkCmdPushConstants(command,pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,12,push);vkCmdDispatch(command,static_cast<uint32_t>(candidates.size()),1,1);ok &= vkEndCommandBuffer(command)==VK_SUCCESS;} const VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO,nullptr,0,nullptr,nullptr,1,&command,0,nullptr}; if(ok) ok &= vkQueueSubmit(queue,1,&submit,fence)==VK_SUCCESS && vkWaitForFences(device,1,&fence,VK_TRUE,UINT64_MAX)==VK_SUCCESS;
    std::vector<float> gpu(candidates.size()); if(ok){void * mapped=nullptr; ok &= vkMapMemory(device,memory[3],0,sizes[3],0,&mapped)==VK_SUCCESS; if(ok){std::memcpy(gpu.data(),mapped,static_cast<size_t>(sizes[3]));vkUnmapMemory(device,memory[3]);}}
    if(fence)vkDestroyFence(device,fence,nullptr);if(command_pool)vkDestroyCommandPool(device,command_pool,nullptr);if(pipeline)vkDestroyPipeline(device,pipeline,nullptr);if(module)vkDestroyShaderModule(device,module,nullptr);if(pipeline_layout)vkDestroyPipelineLayout(device,pipeline_layout,nullptr);if(pool)vkDestroyDescriptorPool(device,pool,nullptr);if(layout)vkDestroyDescriptorSetLayout(device,layout,nullptr);for(uint32_t i=0;i<4;++i){if(buffers[i])vkDestroyBuffer(device,buffers[i],nullptr);if(memory[i])vkFreeMemory(device,memory[i],nullptr);}if(device)vkDestroyDevice(device,nullptr);if(instance)vkDestroyInstance(instance,nullptr);
    if(!ok){error="D1 pre-screen Vulkan dispatch failed";return false;} for(size_t i=0;i<gpu.size();++i) if(std::abs(gpu[i]-cpu_scores[i].weighted_error)>1e-4f){error="D1 pre-screen Vulkan score mismatch";return false;} scores=std::move(cpu_scores); for(size_t i=0;i<scores.size();++i)scores[i].weighted_error=gpu[i]; error.clear(); return true;
}
