# llama-perplexity-impl: perplexity logic, reusable by app

set(TARGET llama-perplexity-impl)

add_library(${TARGET} perplexity.cpp)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common llama ${CMAKE_THREAD_LIBS_INIT})
if (TARGET astc-vulkan-llama-provider)
    target_link_libraries(${TARGET} PUBLIC astc-vulkan-llama-provider)
    target_compile_definitions(${TARGET} PRIVATE LLAMA_ASTC_VULKAN_RUNTIME_AVAILABLE=1)
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-perplexity executable

set(TARGET llama-perplexity)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llama-perplexity-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
