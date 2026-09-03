#include "astc-vulkan-yaqa.h"

#include <vulkan/vulkan.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

namespace {

std::vector<uint32_t> read_spirv(const char * path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) return {};
    const auto size = input.tellg();
    if (size <= 0 || size % static_cast<std::streamsize>(sizeof(uint32_t)) != 0) return {};
    std::vector<uint32_t> code(static_cast<size_t>(size) / sizeof(uint32_t));
    input.seekg(0); input.read(reinterpret_cast<char *>(code.data()), size);
    return input ? code : std::vector<uint32_t>{};
}

uint32_t memory_type(VkPhysicalDevice physical, uint32_t bits) {
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physical, &properties);
    for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) != 0 &&
            (properties.memoryTypes[i].propertyFlags &
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) return i;
    }
    return UINT32_MAX;
}

bool buffer(VkPhysicalDevice physical, VkDevice device, VkDeviceSize bytes,
            VkBuffer & result, VkDeviceMemory & memory) {
    const VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
    if (vkCreateBuffer(device, &info, nullptr, &result) != VK_SUCCESS) return false;
    VkMemoryRequirements requirements{}; vkGetBufferMemoryRequirements(device, result, &requirements);
    const uint32_t type = memory_type(physical, requirements.memoryTypeBits);
    if (type == UINT32_MAX) return false;
    const VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, requirements.size, type};
    return vkAllocateMemory(device, &allocation, nullptr, &memory) == VK_SUCCESS &&
           vkBindBufferMemory(device, result, memory, 0) == VK_SUCCESS;
}

bool write(VkDevice device, VkDeviceMemory memory, const void * data, size_t bytes) {
    void * mapped = nullptr; if (vkMapMemory(device, memory, 0, bytes, 0, &mapped) != VK_SUCCESS) return false;
    std::memcpy(mapped, data, bytes); vkUnmapMemory(device, memory); return true;
}

bool device_for(VkInstance instance, VkPhysicalDevice & physical, uint32_t & family) {
    uint32_t count = 0; if (vkEnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS || !count) return false;
    std::vector<VkPhysicalDevice> devices(count); if (vkEnumeratePhysicalDevices(instance, &count, devices.data()) != VK_SUCCESS) return false;
    for (int pass = 0; pass < 2; ++pass) for (auto candidate : devices) {
        VkPhysicalDeviceProperties props{}; vkGetPhysicalDeviceProperties(candidate, &props);
        if ((pass == 0) != (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)) continue;
        uint32_t queues = 0; vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queues, nullptr);
        std::vector<VkQueueFamilyProperties> q(queues); vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queues, q.data());
        for (uint32_t i = 0; i < queues; ++i) if (q[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { physical = candidate; family = i; return true; }
    }
    return false;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: %s <partial.spv> <reduce.spv>\n", argv[0]); return 2; }
    const auto partial_spirv = read_spirv(argv[1]); const auto reduce_spirv = read_spirv(argv[2]);
    if (partial_spirv.empty() || reduce_spirv.empty()) return 77;
    constexpr uint32_t candidates = 7, rows = 4, columns = 6, samples = 8;
    const size_t error_count = static_cast<size_t>(candidates) * rows * columns;
    std::vector<float> errors(error_count), input(static_cast<size_t>(samples) * columns), output(static_cast<size_t>(samples) * rows);
    for (size_t i = 0; i < errors.size(); ++i) errors[i] = std::sin(static_cast<float>(i) * .17f) * .4f;
    for (size_t i = 0; i < input.size(); ++i) input[i] = std::cos(static_cast<float>(i) * .11f) * .7f;
    for (size_t i = 0; i < output.size(); ++i) output[i] = std::sin(static_cast<float>(i) * .07f) * .6f;
    std::vector<float> expected(candidates);
    for (uint32_t c = 0; c < candidates; ++c) {
        std::vector<float> one(errors.begin() + static_cast<size_t>(c) * rows * columns,
                               errors.begin() + static_cast<size_t>(c + 1) * rows * columns);
        expected[c] = static_cast<float>(astc_vulkan_yaqa_trace_score(one, rows, columns, input, output, samples));
    }
    const VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "astc-yaqa-batch-smoke", 1, "llama.cpp", 1, VK_API_VERSION_1_1};
    const VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0, &app, 0, nullptr, 0, nullptr};
    VkInstance instance = VK_NULL_HANDLE; if (vkCreateInstance(&instance_info, nullptr, &instance) != VK_SUCCESS) return 77;
    VkPhysicalDevice physical = VK_NULL_HANDLE; uint32_t family = UINT32_MAX;
    if (!device_for(instance, physical, family)) { vkDestroyInstance(instance, nullptr); return 77; }
    constexpr float priority = 1.0f; const VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, family, 1, &priority};
    const VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, nullptr, 0, 1, &queue_info, 0, nullptr, 0, nullptr, nullptr};
    VkDevice device = VK_NULL_HANDLE; if (vkCreateDevice(physical, &device_info, nullptr, &device) != VK_SUCCESS) { vkDestroyInstance(instance, nullptr); return 77; }
    VkQueue queue = VK_NULL_HANDLE; vkGetDeviceQueue(device, family, 0, &queue);
    const uint32_t pairs = samples * samples, partial_count = candidates * pairs;
    VkBuffer gpu_buffers[5]{}; VkDeviceMemory gpu_memory[5]{};
    const size_t sizes[5] = {errors.size()*4, input.size()*4, output.size()*4, partial_count*4, candidates*4}; bool ok = true;
    for (uint32_t i=0;i<5;++i) ok &= buffer(physical, device, sizes[i], gpu_buffers[i], gpu_memory[i]);
    if (ok) { ok &= write(device,gpu_memory[0],errors.data(),sizes[0]); ok &= write(device,gpu_memory[1],input.data(),sizes[1]); ok &= write(device,gpu_memory[2],output.data(),sizes[2]); }
    VkDescriptorSetLayout partial_layout=0, reduce_layout=0; VkDescriptorPool pool=0; VkDescriptorSet partial_set=0, reduce_set=0; VkPipelineLayout partial_pl=0, reduce_pl=0; VkShaderModule partial_module=0, reduce_module=0; VkPipeline partial_pipeline=0, reduce_pipeline=0; VkCommandPool command_pool=0; VkCommandBuffer command=0; VkFence fence=0;
    const VkDescriptorSetLayoutBinding pb[4]={{0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr},{1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr},{2,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr},{3,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr}};
    const VkDescriptorSetLayoutBinding rb[2]={{0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr},{1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr}};
    const VkDescriptorSetLayoutCreateInfo pli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,nullptr,0,4,pb}; const VkDescriptorSetLayoutCreateInfo rli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,nullptr,0,2,rb};
    if(ok) ok &= vkCreateDescriptorSetLayout(device,&pli,nullptr,&partial_layout)==VK_SUCCESS && vkCreateDescriptorSetLayout(device,&rli,nullptr,&reduce_layout)==VK_SUCCESS;
    const VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,6}; const VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,nullptr,0,2,1,&ps}; if(ok) ok &= vkCreateDescriptorPool(device,&pi,nullptr,&pool)==VK_SUCCESS;
    VkDescriptorSetLayout layouts[2]={partial_layout,reduce_layout}; VkDescriptorSet sets[2]{}; const VkDescriptorSetAllocateInfo si{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,nullptr,pool,2,layouts}; if(ok) ok &= vkAllocateDescriptorSets(device,&si,sets)==VK_SUCCESS; partial_set=sets[0]; reduce_set=sets[1];
    VkDescriptorBufferInfo pinfo[4]={{gpu_buffers[0],0,sizes[0]},{gpu_buffers[1],0,sizes[1]},{gpu_buffers[2],0,sizes[2]},{gpu_buffers[3],0,sizes[3]}}; VkWriteDescriptorSet pw[4]{}; for(uint32_t i=0;i<4;++i){pw[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;pw[i].dstSet=partial_set;pw[i].dstBinding=i;pw[i].descriptorCount=1;pw[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;pw[i].pBufferInfo=&pinfo[i];}
    VkDescriptorBufferInfo rinfo[2]={{gpu_buffers[3],0,sizes[3]},{gpu_buffers[4],0,sizes[4]}}; VkWriteDescriptorSet rw[2]{}; for(uint32_t i=0;i<2;++i){rw[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;rw[i].dstSet=reduce_set;rw[i].dstBinding=i;rw[i].descriptorCount=1;rw[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;rw[i].pBufferInfo=&rinfo[i];} if(ok){vkUpdateDescriptorSets(device,4,pw,0,nullptr);vkUpdateDescriptorSets(device,2,rw,0,nullptr);}
    const VkShaderModuleCreateInfo pmi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,nullptr,0,partial_spirv.size()*4,partial_spirv.data()}; const VkShaderModuleCreateInfo rmi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,nullptr,0,reduce_spirv.size()*4,reduce_spirv.data()}; const VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT,0,20}; const VkPushConstantRange rr{VK_SHADER_STAGE_COMPUTE_BIT,0,8};
    const VkPipelineLayoutCreateInfo ppli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,nullptr,0,1,&partial_layout,1,&pr}; const VkPipelineLayoutCreateInfo rpli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,nullptr,0,1,&reduce_layout,1,&rr}; if(ok) ok &= vkCreateShaderModule(device,&pmi,nullptr,&partial_module)==VK_SUCCESS && vkCreateShaderModule(device,&rmi,nullptr,&reduce_module)==VK_SUCCESS && vkCreatePipelineLayout(device,&ppli,nullptr,&partial_pl)==VK_SUCCESS && vkCreatePipelineLayout(device,&rpli,nullptr,&reduce_pl)==VK_SUCCESS;
    const VkPipelineShaderStageCreateInfo pst{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,partial_module,"main",nullptr}; const VkPipelineShaderStageCreateInfo rst{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,reduce_module,"main",nullptr}; const VkComputePipelineCreateInfo ppc{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,nullptr,0,pst,partial_pl,VK_NULL_HANDLE,-1}; const VkComputePipelineCreateInfo rpc{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,nullptr,0,rst,reduce_pl,VK_NULL_HANDLE,-1}; if(ok) ok &= vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&ppc,nullptr,&partial_pipeline)==VK_SUCCESS && vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&rpc,nullptr,&reduce_pipeline)==VK_SUCCESS;
    const VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,nullptr,VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,family}; const VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,nullptr,0}; if(ok) ok &= vkCreateCommandPool(device,&cpi,nullptr,&command_pool)==VK_SUCCESS; const VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,nullptr,command_pool,VK_COMMAND_BUFFER_LEVEL_PRIMARY,1}; if(ok) ok &= vkAllocateCommandBuffers(device,&cai,&command)==VK_SUCCESS; if(ok) ok &= vkCreateFence(device,&fi,nullptr,&fence)==VK_SUCCESS;
    const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,nullptr,VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,nullptr}; uint32_t pp[5]={candidates,rows,columns,samples,pairs}; uint32_t rp[2]={candidates,pairs}; if(ok) ok &= vkBeginCommandBuffer(command,&begin)==VK_SUCCESS; if(ok){vkCmdBindPipeline(command,VK_PIPELINE_BIND_POINT_COMPUTE,partial_pipeline);vkCmdBindDescriptorSets(command,VK_PIPELINE_BIND_POINT_COMPUTE,partial_pl,0,1,&partial_set,0,nullptr);vkCmdPushConstants(command,partial_pl,VK_SHADER_STAGE_COMPUTE_BIT,0,20,pp);vkCmdDispatch(command,candidates,(pairs+63)/64,1);const VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,VK_NULL_HANDLE,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,VK_QUEUE_FAMILY_IGNORED,VK_QUEUE_FAMILY_IGNORED,gpu_buffers[3],0,sizes[3]};vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,1,&barrier,0,nullptr);vkCmdBindPipeline(command,VK_PIPELINE_BIND_POINT_COMPUTE,reduce_pipeline);vkCmdBindDescriptorSets(command,VK_PIPELINE_BIND_POINT_COMPUTE,reduce_pl,0,1,&reduce_set,0,nullptr);vkCmdPushConstants(command,reduce_pl,VK_SHADER_STAGE_COMPUTE_BIT,0,8,rp);vkCmdDispatch(command,candidates,1,1);ok &= vkEndCommandBuffer(command)==VK_SUCCESS;}
    const VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO,nullptr,0,nullptr,nullptr,1,&command,0,nullptr}; if(ok) ok &= vkQueueSubmit(queue,1,&submit,fence)==VK_SUCCESS && vkWaitForFences(device,1,&fence,VK_TRUE,UINT64_MAX)==VK_SUCCESS; float actual[7]{}; if(ok){void*m=nullptr;ok &= vkMapMemory(device,gpu_memory[4],0,sizes[4],0,&m)==VK_SUCCESS;if(ok){std::memcpy(actual,m,sizes[4]);vkUnmapMemory(device,gpu_memory[4]);}}
    if(fence)vkDestroyFence(device,fence,nullptr);if(command_pool)vkDestroyCommandPool(device,command_pool,nullptr);if(partial_pipeline)vkDestroyPipeline(device,partial_pipeline,nullptr);if(reduce_pipeline)vkDestroyPipeline(device,reduce_pipeline,nullptr);if(partial_module)vkDestroyShaderModule(device,partial_module,nullptr);if(reduce_module)vkDestroyShaderModule(device,reduce_module,nullptr);if(partial_pl)vkDestroyPipelineLayout(device,partial_pl,nullptr);if(reduce_pl)vkDestroyPipelineLayout(device,reduce_pl,nullptr);if(pool)vkDestroyDescriptorPool(device,pool,nullptr);if(partial_layout)vkDestroyDescriptorSetLayout(device,partial_layout,nullptr);if(reduce_layout)vkDestroyDescriptorSetLayout(device,reduce_layout,nullptr);for(uint32_t i=0;i<5;++i){if(gpu_buffers[i])vkDestroyBuffer(device,gpu_buffers[i],nullptr);if(gpu_memory[i])vkFreeMemory(device,gpu_memory[i],nullptr);}vkDestroyDevice(device,nullptr);vkDestroyInstance(instance,nullptr);
    if(!ok){std::fprintf(stderr,"YAQA batch GPU smoke failed during Vulkan execution\n");return 1;} for(uint32_t i=0;i<candidates;++i)if(std::fabs(actual[i]-expected[i])>2e-3f){std::fprintf(stderr,"YAQA batch mismatch candidate %u: %.6f != %.6f\n",i,actual[i],expected[i]);return 1;} std::printf("YAQA batch GPU/CPU equivalence passed for %u candidates (%u trace samples)\n",candidates,samples);return 0;
}
