set(AGENT_RUNTIME_SOURCES
    agent-cli-config.cpp
    agent-cli-generation-utils.cpp
    agent-cli-run-adapter.cpp
    agent-daemon-client.cpp
    agent-tool-provider.cpp
    agent-daemon-adapter.cpp
    agent-daemon-protocol.cpp
    agent-daemon-runtime.cpp
    agent-daemon-dispatcher.cpp
    agent-daemon-service.cpp
    agent-cli-host-adapter.cpp
    agent-cli-inference.cpp
    agent-plan-orchestration.cpp
    agent-runtime-chat-driver.cpp
    agent-runtime-host.cpp
    agent-runtime-resident.cpp
    agent-runtime-session-host.cpp
    agent-runtime-session-manager.cpp
    agent-runtime-execution.cpp
    agent-runtime-session.cpp
    agent-server-context-host.cpp
    agent-server-generation.cpp
    agent-server-inference.cpp
    agent-runtime-assembly.cpp
    agent-cli-runtime.cpp
    ../common/cli-config.cpp
    ../memory/memory-cli-chat.cpp
    ../memory/memory-cli-memory.cpp
    agent-cli-selection.cpp
)

set(TARGET llama-agent-cli)
add_executable(${TARGET}
    agent.cpp
    agent-cli-command.cpp
    ${AGENT_RUNTIME_SOURCES}
    agent-cli-run.cpp
)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME llama-agent)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE
    llama-memory
    llama-common
    llama
    llama-agent-poc
    server-context
    ${CMAKE_THREAD_LIBS_INIT}
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_compile_definitions(${TARGET} PRIVATE LLAMA_MEMORY_POC_USE_AGENT_TOOLS)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

set(TARGET llama-agent-resident-smoke)
add_executable(${TARGET}
    agent-resident-smoke.cpp
    ${AGENT_RUNTIME_SOURCES}
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE
    llama-memory
    llama-common
    llama
    llama-agent-poc
    server-context
    ${CMAKE_THREAD_LIBS_INIT}
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_compile_definitions(${TARGET} PRIVATE LLAMA_MEMORY_POC_USE_AGENT_TOOLS)

set(TARGET llama-agent-tool-provider-smoke)
add_executable(${TARGET}
    agent-tool-provider-smoke.cpp
    ${AGENT_RUNTIME_SOURCES}
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE
    llama-memory
    llama-common
    llama
    llama-agent-poc
    server-context
    ${CMAKE_THREAD_LIBS_INIT}
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_compile_definitions(${TARGET} PRIVATE LLAMA_MEMORY_POC_USE_AGENT_TOOLS)

set(TARGET llama-agent-tool-runtime-smoke)
add_executable(${TARGET}
    agent-tool-runtime-smoke.cpp
    ${AGENT_RUNTIME_SOURCES}
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE
    llama-memory
    llama-common
    llama
    llama-agent-poc
    server-context
    ${CMAKE_THREAD_LIBS_INIT}
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_compile_definitions(${TARGET} PRIVATE LLAMA_MEMORY_POC_USE_AGENT_TOOLS)

set(TARGET llama-agent-mcp-tool-provider-smoke)
add_executable(${TARGET}
    agent-mcp-tool-provider-smoke.cpp
    ${AGENT_RUNTIME_SOURCES}
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE
    llama-memory
    llama-common
    llama
    llama-agent-poc
    server-context
    ${CMAKE_THREAD_LIBS_INIT}
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_compile_definitions(${TARGET} PRIVATE LLAMA_MEMORY_POC_USE_AGENT_TOOLS)

set(TARGET llama-agent-daemon-dispatcher-smoke)
add_executable(${TARGET}
    agent-daemon-dispatcher-smoke.cpp
    ${AGENT_RUNTIME_SOURCES}
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE
    llama-memory
    llama-common
    llama
    llama-agent-poc
    server-context
    ${CMAKE_THREAD_LIBS_INIT}
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_compile_definitions(${TARGET} PRIVATE LLAMA_MEMORY_POC_USE_AGENT_TOOLS)

set(TARGET llama-agent-daemon)
add_executable(${TARGET}
    agent-daemon.cpp
    ${AGENT_RUNTIME_SOURCES}
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE
    llama-memory
    llama-common
    llama
    llama-agent-poc
    server-context
    ${CMAKE_THREAD_LIBS_INIT}
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_compile_definitions(${TARGET} PRIVATE LLAMA_MEMORY_POC_USE_AGENT_TOOLS)
