# llama-cli-impl: CLI logic, reusable by app

set(TARGET llama-cli-impl)

add_library(${TARGET} cli.cpp
                      cli-client.cpp
                      cli-context.cpp)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ../server)
target_link_libraries(${TARGET} PUBLIC llama-server-impl llama-common ${CMAKE_THREAD_LIBS_INIT})

# The explicit compiled-model path is optional and isolated from the normal
# GGUF CLI. The ASTC POC is added before tools/, so this target is available
# only when LLAMA_ASTC_VULKAN_POC is enabled.
if(TARGET astc-vulkan-compiled-source)
    target_include_directories(${TARGET} PRIVATE
        ${CMAKE_SOURCE_DIR}/pocs/astc-vulkan)
    target_link_libraries(${TARGET} PRIVATE astc-vulkan-compiled-source)
    target_compile_definitions(${TARGET} PRIVATE
        LLAMA_ASTC_VULKAN_COMPILED_SOURCE_AVAILABLE=1)
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-cli executable

set(TARGET llama-cli)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llama-cli-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
