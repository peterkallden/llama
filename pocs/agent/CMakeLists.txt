set(AGENT_RUNTIME_SOURCES
    agent-cli-config.cpp
    agent-cli-generation-utils.cpp
    agent-cli-run-adapter.cpp
    agent-daemon-client.cpp
    agent-host-config.cpp
    agent-mcp-protocol.cpp
    agent-mcp-stdio-client.cpp
    agent-tool-provider.cpp
    agent-tool-runtime-adapter.cpp
    agent-resource-catalog.cpp
    agent-resource-store.cpp
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

if (LLAMA_MEMORY_COZO)
    list(APPEND AGENT_RUNTIME_SOURCES
        agent-resource-catalog-cozo.cpp
    )
endif()

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

set(TARGET llama-agent-mcp-stdio-client-smoke)
add_executable(${TARGET}
    agent-mcp-stdio-client-smoke.cpp
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

set(TARGET llama-agent-mcp-stdio-server-smoke)
add_executable(${TARGET}
    agent-mcp-stdio-server-smoke.cpp
    ${AGENT_RUNTIME_SOURCES}
)
add_dependencies(${TARGET} llama-agent-mcp-stdio-server)
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

set(TARGET llama-agent-cli-mcp-selection-smoke)
add_executable(${TARGET}
    agent-cli-mcp-selection-smoke.cpp
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

set(TARGET llama-agent-cli-run-mcp-smoke)
add_executable(${TARGET}
    agent-cli-run-mcp-smoke.cpp
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

set(TARGET llama-agent-mcp-stdio-fake-server)
add_executable(${TARGET}
    agent-mcp-stdio-fake-server.cpp
    agent-mcp-server-protocol.cpp
    agent-mcp-server-tool-registry.cpp
    agent-mcp-stdio-server.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE
    llama-common
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-mcp-stdio-server)
add_executable(${TARGET}
    agent-mcp-stdio-server-main.cpp
    ${AGENT_RUNTIME_SOURCES}
    agent-mcp-server-protocol.cpp
    agent-mcp-server-tool-registry.cpp
    agent-mcp-stdio-server.cpp
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

set(TARGET llama-agent-daemon-protocol-smoke)
add_executable(${TARGET}
    agent-daemon-protocol-smoke.cpp
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

set(TARGET llama-agent-daemon-mcp-config-smoke)
add_executable(${TARGET}
    agent-daemon-mcp-config-smoke.cpp
    ${AGENT_RUNTIME_SOURCES}
)
add_dependencies(${TARGET} llama-agent-mcp-stdio-fake-server)
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

set(TARGET llama-agent-runtime-session-host-tooling-smoke)
add_executable(${TARGET}
    agent-runtime-session-host-tooling-smoke.cpp
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

set(TARGET llama-agent-resource-store-smoke)
add_executable(${TARGET}
    agent-resource-store-smoke.cpp
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

set(AGENT_RUNTIME_TARGETS
    llama-agent-cli
    llama-agent-resident-smoke
    llama-agent-tool-provider-smoke
    llama-agent-tool-runtime-smoke
    llama-agent-mcp-tool-provider-smoke
    llama-agent-mcp-stdio-client-smoke
    llama-agent-mcp-stdio-server-smoke
    llama-agent-cli-mcp-selection-smoke
    llama-agent-cli-run-mcp-smoke
    llama-agent-daemon-dispatcher-smoke
    llama-agent-daemon-protocol-smoke
    llama-agent-daemon-mcp-config-smoke
    llama-agent-runtime-session-host-tooling-smoke
    llama-agent-resource-store-smoke
    llama-agent-daemon
)

if (LLAMA_MEMORY_COZO)
    foreach(TARGET ${AGENT_RUNTIME_TARGETS})
        target_include_directories(${TARGET} PRIVATE ${COZO_INCLUDE_DIR})
        target_link_libraries(${TARGET} PRIVATE ${COZO_LIBRARY})
        if (WIN32)
            target_link_libraries(${TARGET} PRIVATE ws2_32 bcrypt ntdll rpcrt4 shlwapi crypt32)
        endif()
    endforeach()
endif()
