set(TARGET llama-agent-cli)
add_executable(${TARGET}
    agent.cpp
    ../memory/memory-cli-config.cpp
    ../memory/memory-cli-chat.cpp
    ../memory/memory-cli-memory.cpp
    ../memory/memory-cli-run.cpp
    ../memory/memory-cli-agent.cpp
    ../memory/memory-cli-selection.cpp
)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME llama-agent)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
)
target_link_libraries(${TARGET} PRIVATE
    llama-memory
    llama-common
    llama
    llama-agent-poc
    ${CMAKE_THREAD_LIBS_INIT}
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_compile_definitions(${TARGET} PRIVATE LLAMA_MEMORY_POC_USE_AGENT_TOOLS)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
