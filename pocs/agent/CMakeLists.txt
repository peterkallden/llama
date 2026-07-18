include_directories(${PROJECT_SOURCE_DIR})

set(AGENT_RUNTIME_SOURCES
    ../../tools/agent/cli/agent-cli-config.cpp
    ../../tools/agent/cli/agent-cli-generation-utils.cpp
    ../../tools/agent/cli/agent-cli-run-adapter.cpp
    ../../tools/agent/daemon/agent-daemon-client-admin.cpp
    ../../tools/agent/daemon/agent-daemon-client-status.cpp
    ../../tools/agent/daemon/agent-daemon-client.cpp
    ../../tools/agent/daemon/agent-daemon-jsonl-protocol.cpp
    ../../tools/agent/daemon/agent-daemon-event-collector.cpp
    ../../tools/agent/host/agent-host-config.cpp
    ../../tools/agent/mcp/agent-mcp-protocol.cpp
    ../../tools/agent/mcp/agent-mcp-http-client.cpp
    ../../tools/agent/tooling/agent-selection-contracts.cpp
    ../../tools/agent/mcp/agent-mcp-stdio-client.cpp
    ../../tools/agent/tooling/agent-tool-provider.cpp
    ../../tools/agent/tooling/agent-tool-result-json-contracts.cpp
    ../../tools/agent/tooling/agent-tool-runtime-adapter.cpp
    ../../tools/agent/resource/agent-resource-catalog.cpp
    ../../tools/agent/resource/agent-resource-store.cpp
    ../../tools/agent/daemon/agent-daemon-adapter.cpp
    ../../tools/agent/daemon/agent-daemon-tcp.cpp
    ../../tools/agent/daemon/agent-daemon-unix.cpp
    ../../tools/agent/daemon/agent-daemon-protocol.cpp
    ../../tools/agent/daemon/agent-daemon-runtime.cpp
    ../../tools/agent/daemon/agent-daemon-dispatcher.cpp
    ../../tools/agent/daemon/agent-daemon-service.cpp
    ../../tools/agent/cli/agent-cli-host-adapter.cpp
    ../../tools/agent/cli/agent-cli-inference.cpp
    ../../tools/agent/runtime/agent-plan-orchestration.cpp
    ../../tools/agent/runtime/agent-runtime-chat-driver.cpp
    ../../tools/agent/runtime/agent-runtime-host.cpp
    ../../tools/agent/runtime/agent-runtime-resident.cpp
    ../../tools/agent/runtime/agent-runtime-session-host.cpp
    ../../tools/agent/runtime/agent-runtime-session-manager.cpp
    ../../tools/agent/runtime/agent-runtime-execution.cpp
    ../../tools/agent/runtime/agent-runtime-session.cpp
    ../../tools/agent/runtime/agent-server-context-host.cpp
    ../../tools/agent/runtime/agent-server-generation.cpp
    ../../tools/agent/runtime/agent-server-inference.cpp
    ../../tools/agent/runtime/agent-runtime-assembly.cpp
    ../../tools/agent/cli/agent-cli-runtime.cpp
    ../common/cli-config.cpp
    ../memory/memory-cli-chat.cpp
    ../memory/memory-cli-memory.cpp
    ../../tools/agent/cli/agent-cli-selection.cpp
)

link_libraries(cpp-httplib)

if (LLAMA_MEMORY_COZO)
    list(APPEND AGENT_RUNTIME_SOURCES
        ../../tools/agent/resource/agent-resource-catalog-cozo.cpp
    )
endif()

set(TARGET llama-agent-cli)
add_executable(${TARGET}
    ../../tools/agent/cli/agent-cli-main.cpp
    ../../tools/agent/cli/agent-cli-command.cpp
    ${AGENT_RUNTIME_SOURCES}
    ../../tools/agent/cli/agent-cli-run.cpp
)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME llama-agent)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/cli
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime
    ${PROJECT_SOURCE_DIR}/tools/agent/tooling
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
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
if (WIN32)
    target_link_libraries(${TARGET} PRIVATE ws2_32)
endif()
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_compile_definitions(${TARGET} PRIVATE LLAMA_MEMORY_POC_USE_AGENT_TOOLS)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

set(TARGET llama-agent-resident-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-resident-smoke.cpp
    ${AGENT_RUNTIME_SOURCES}
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/cli
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime
    ${PROJECT_SOURCE_DIR}/tools/agent/tooling
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
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
    smoke/runtime/agent-tool-provider-smoke.cpp
    ${AGENT_RUNTIME_SOURCES}
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/cli
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime
    ${PROJECT_SOURCE_DIR}/tools/agent/tooling
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
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
    smoke/runtime/agent-tool-runtime-smoke.cpp
    ${AGENT_RUNTIME_SOURCES}
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/cli
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime
    ${PROJECT_SOURCE_DIR}/tools/agent/tooling
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
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
    smoke/mcp/agent-mcp-tool-provider-smoke.cpp
    ${AGENT_RUNTIME_SOURCES}
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/cli
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime
    ${PROJECT_SOURCE_DIR}/tools/agent/tooling
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
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
    smoke/mcp/agent-mcp-stdio-client-smoke.cpp
    ${AGENT_RUNTIME_SOURCES}
)
add_dependencies(${TARGET} llama-agent-mcp-stdio-fake-server)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/cli
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime
    ${PROJECT_SOURCE_DIR}/tools/agent/tooling
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
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

set(TARGET llama-agent-mcp-http-client-smoke)
add_executable(${TARGET}
    smoke/mcp/agent-mcp-http-client-smoke.cpp
    ${AGENT_RUNTIME_SOURCES}
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/cli
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime
    ${PROJECT_SOURCE_DIR}/tools/agent/tooling
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
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
    smoke/mcp/agent-mcp-stdio-server-smoke.cpp
    ${AGENT_RUNTIME_SOURCES}
    ${PROJECT_SOURCE_DIR}/tools/agent/mcp/agent-mcp-server-tool-registry.cpp
)
add_dependencies(${TARGET}
    llama-agent-mcp-stdio-server
    llama-agent-mcp-stdio-fake-server
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
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

set(TARGET llama-agent-mcp-http-inbound-dispatcher-smoke)
add_executable(${TARGET}
    smoke/mcp/agent-mcp-http-server-smoke.cpp
    ${AGENT_RUNTIME_SOURCES}
    ../../tools/agent/daemon/agent-daemon-dispatcher.cpp
    ../../tools/agent/daemon/agent-daemon-service.cpp
    ../../tools/agent/daemon/agent-daemon-event-collector.cpp
    ../../tools/agent/mcp/agent-mcp-http-server.cpp
    ../../tools/agent/mcp/agent-mcp-auth.cpp
    ../../tools/agent/mcp/agent-mcp-server-protocol.cpp
    ../../tools/agent/mcp/agent-mcp-server-tool-registry.cpp
    ../../common/agent/schema-contract.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/server
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
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
    smoke/cli/agent-cli-mcp-selection-smoke.cpp
    ${AGENT_RUNTIME_SOURCES}
)
add_dependencies(${TARGET} llama-agent-mcp-stdio-fake-server)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
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
    smoke/cli/agent-cli-run-mcp-smoke.cpp
    ${AGENT_RUNTIME_SOURCES}
)
add_dependencies(${TARGET} llama-agent-mcp-stdio-fake-server)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
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
    ../../tools/agent/mcp/agent-mcp-stdio-fake-server.cpp
    ../../tools/agent/mcp/agent-mcp-server-protocol.cpp
    ../../tools/agent/mcp/agent-mcp-server-tool-registry.cpp
    ../../tools/agent/mcp/agent-mcp-stdio-server.cpp
    ../../common/agent/schema-contract.cpp
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
    ../../tools/agent/mcp/agent-mcp-stdio-server-main.cpp
    ${AGENT_RUNTIME_SOURCES}
    ../../tools/agent/mcp/agent-mcp-server-protocol.cpp
    ../../tools/agent/mcp/agent-mcp-server-tool-registry.cpp
    ../../tools/agent/mcp/agent-mcp-stdio-server.cpp
    ../../tools/agent/mcp/agent-mcp-http-server.cpp
    ../../tools/agent/mcp/agent-mcp-auth.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
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
    smoke/daemon/agent-daemon-dispatcher-smoke.cpp
    ${AGENT_RUNTIME_SOURCES}
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
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
    smoke/daemon/agent-daemon-protocol-smoke.cpp
    ${AGENT_RUNTIME_SOURCES}
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
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

set(TARGET llama-agent-daemon-jsonl-protocol-smoke)
add_executable(${TARGET}
    smoke/daemon/agent-daemon-jsonl-protocol-smoke.cpp
    ../../tools/agent/daemon/agent-daemon-client-status.cpp
    ../../tools/agent/daemon/agent-daemon-jsonl-protocol.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/vendor
)
target_link_libraries(${TARGET} PRIVATE
    ${CMAKE_THREAD_LIBS_INIT}
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-daemon-client-fake-daemon)
add_executable(${TARGET}
    agent-daemon-client-fake-daemon.cpp
    ../../tools/agent/daemon/agent-daemon-jsonl-protocol.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/vendor
)
target_link_libraries(${TARGET} PRIVATE
    ${CMAKE_THREAD_LIBS_INIT}
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-daemon-client-smoke)
add_executable(${TARGET}
    smoke/daemon/agent-daemon-client-smoke.cpp
    ${AGENT_RUNTIME_SOURCES}
)
add_dependencies(${TARGET} llama-agent-daemon-client-fake-daemon)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
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

set(TARGET llama-agent-daemon-wait-events-smoke)
add_executable(${TARGET}
    smoke/daemon/agent-daemon-wait-events-smoke.cpp
    ${AGENT_RUNTIME_SOURCES}
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
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
    smoke/daemon/agent-daemon-mcp-config-smoke.cpp
    ${AGENT_RUNTIME_SOURCES}
    ../../tools/agent/mcp/agent-mcp-auth.cpp
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
    smoke/runtime/agent-runtime-session-host-tooling-smoke.cpp
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

set(TARGET llama-agent-runtime-session-manager-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-runtime-session-manager-smoke.cpp
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

set(TARGET llama-agent-runtime-operation-manager-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-runtime-operation-manager-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-resource-store-smoke)
add_executable(${TARGET}
    smoke/resource/agent-resource-store-smoke.cpp
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
    ../../tools/agent/daemon/agent-daemon-main.cpp
    ${AGENT_RUNTIME_SOURCES}
    ../../tools/agent/mcp/agent-mcp-http-server.cpp
    ../../tools/agent/mcp/agent-mcp-auth.cpp
    ../../tools/agent/mcp/agent-mcp-server-protocol.cpp
    ../../tools/agent/mcp/agent-mcp-server-tool-registry.cpp
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
    llama-agent-mcp-http-client-smoke
    llama-agent-mcp-stdio-server-smoke
    llama-agent-mcp-http-inbound-dispatcher-smoke
    llama-agent-mcp-stdio-server
    llama-agent-cli-mcp-selection-smoke
    llama-agent-cli-run-mcp-smoke
    llama-agent-daemon-dispatcher-smoke
    llama-agent-daemon-protocol-smoke
    llama-agent-daemon-jsonl-protocol-smoke
    llama-agent-daemon-client-smoke
    llama-agent-daemon-wait-events-smoke
    llama-agent-daemon-mcp-config-smoke
    llama-agent-runtime-session-host-tooling-smoke
    llama-agent-runtime-session-manager-smoke
    llama-agent-runtime-operation-manager-smoke
    llama-agent-resource-store-smoke
    llama-agent-daemon
)

set(AGENT_RUNTIME_SMOKE_RUNTIME_TARGETS
    llama-agent-resident-smoke
    llama-agent-tool-provider-smoke
    llama-agent-tool-runtime-smoke
    llama-agent-runtime-session-host-tooling-smoke
    llama-agent-runtime-session-manager-smoke
    llama-agent-runtime-operation-manager-smoke
)

set(AGENT_RUNTIME_SMOKE_MCP_TARGETS
    llama-agent-mcp-tool-provider-smoke
    llama-agent-mcp-stdio-client-smoke
    llama-agent-mcp-http-client-smoke
    llama-agent-mcp-stdio-server-smoke
    llama-agent-mcp-http-inbound-dispatcher-smoke
)

set(AGENT_RUNTIME_SMOKE_CLI_TARGETS
    llama-agent-cli-mcp-selection-smoke
    llama-agent-cli-run-mcp-smoke
)

set(AGENT_RUNTIME_SMOKE_DAEMON_TARGETS
    llama-agent-daemon-dispatcher-smoke
    llama-agent-daemon-protocol-smoke
    llama-agent-daemon-jsonl-protocol-smoke
    llama-agent-daemon-client-smoke
    llama-agent-daemon-wait-events-smoke
    llama-agent-daemon-mcp-config-smoke
)

set(AGENT_RUNTIME_SMOKE_RESOURCE_TARGETS
    llama-agent-resource-store-smoke
)

function(llama_agent_add_smoke_group TARGET_NAME)
    add_custom_target(${TARGET_NAME} DEPENDS ${ARGN})
endfunction()

llama_agent_add_smoke_group(
    llama-agent-smoke-runtime
    ${AGENT_RUNTIME_SMOKE_RUNTIME_TARGETS}
)

llama_agent_add_smoke_group(
    llama-agent-smoke-mcp
    ${AGENT_RUNTIME_SMOKE_MCP_TARGETS}
)

llama_agent_add_smoke_group(
    llama-agent-smoke-cli
    ${AGENT_RUNTIME_SMOKE_CLI_TARGETS}
)

llama_agent_add_smoke_group(
    llama-agent-smoke-daemon
    ${AGENT_RUNTIME_SMOKE_DAEMON_TARGETS}
)

llama_agent_add_smoke_group(
    llama-agent-smoke-resource
    ${AGENT_RUNTIME_SMOKE_RESOURCE_TARGETS}
)

llama_agent_add_smoke_group(
    llama-agent-smoke-all
    ${AGENT_RUNTIME_SMOKE_RUNTIME_TARGETS}
    ${AGENT_RUNTIME_SMOKE_MCP_TARGETS}
    ${AGENT_RUNTIME_SMOKE_CLI_TARGETS}
    ${AGENT_RUNTIME_SMOKE_DAEMON_TARGETS}
    ${AGENT_RUNTIME_SMOKE_RESOURCE_TARGETS}
)

if (LLAMA_MEMORY_COZO)
    foreach(TARGET ${AGENT_RUNTIME_TARGETS})
        llama_link_cozo(${TARGET})
    endforeach()
endif()
