set(TARGET llama-memory-poc)
add_executable(${TARGET} memory.cpp)
target_link_libraries(${TARGET} PRIVATE llama-memory llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
