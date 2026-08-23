# Android NDK compatibility package for ggml-vulkan.
# The NDK contains the headers but does not ship SPIRV-HeadersConfig.cmake.
set(SPIRV-Headers_FOUND TRUE)
if(NOT TARGET SPIRV-Headers::SPIRV-Headers)
    add_library(SPIRV-Headers::SPIRV-Headers INTERFACE IMPORTED)
endif()
