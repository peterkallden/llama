include_directories(${PROJECT_SOURCE_DIR})
if (LLAMA_AGENT_TOOLS_CLANG)
    add_compile_definitions(LLAMA_AGENT_TOOLS_USE_CLANG=1)
endif()

set(AGENT_RUNTIME_SOURCES
    ../../tools/agent/cli/agent-cli-config.cpp
    ../../tools/agent/cli/agent-cli-generation-utils.cpp
    ../../tools/agent/cli/agent-cli-run-adapter.cpp
    ../../tools/agent/daemon/agent-daemon-client-admin.cpp
    ../../tools/agent/daemon/agent-daemon-client-status.cpp
    ../../tools/agent/daemon/agent-daemon-client.cpp
    ../../tools/agent/daemon/agent-daemon-event-collector.cpp
    ../../tools/agent/host/agent-host-config.cpp
    ../../tools/agent/mcp/agent-mcp-protocol.cpp
    ../../tools/agent/mcp/agent-mcp-task.cpp
    ../../tools/agent/mcp/agent-mcp-http-client.cpp
    ../../tools/agent/tooling/agent-selection-contracts.cpp
    ../../tools/agent/mcp/agent-mcp-stdio-client.cpp
    ../../tools/agent/tooling/agent-tool-provider.cpp
    ../../tools/agent/tooling/agent-tool-result-json-contracts.cpp
    ../../tools/agent/tooling/agent-tool-runtime-adapter.cpp
    ../../tools/agent/resource/agent-resource-catalog.cpp
    ../../tools/agent/resource/agent-resource-store.cpp
    ../../tools/agent/data/agent-data-store-factory.cpp
    ../../tools/agent/diagnostics/agent-clangd-protocol.cpp
    ../../tools/agent/diagnostics/agent-clangd-session.cpp
    ../../tools/agent/diagnostics/agent-clangd-provider.cpp
    ../../tools/agent/daemon/agent-daemon-adapter.cpp
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
    ../../tools/agent/runtime/agent-runtime-event-projector.cpp
    ../../tools/agent/runtime/agent-runtime-turn-driver.cpp
    ../../tools/agent/runtime/agent-inference-executor.cpp
    ../../tools/agent/runtime/agent-inference-capacity-gate.cpp
    ../../tools/agent/runtime/agent-runtime-session-manager.cpp
    ../../tools/agent/runtime/agent-runtime-execution.cpp
    ../../tools/agent/runtime/agent-runtime-session.cpp
    ../../tools/agent/runtime/agent-server-context-host.cpp
    ../../tools/agent/runtime/agent-server-generation.cpp
    ../../tools/agent/runtime/agent-server-inference.cpp
    ../../tools/agent/runtime/agent-runtime-assembly.cpp
    ../../tools/agent/cli/agent-cli-runtime.cpp
    ../../tools/agent/cli/agent-cli-options.cpp
    ../../tools/agent/cli/agent-cli-generation.cpp
    ../../tools/agent/cli/agent-cli-memory-tools.cpp
    ../../tools/agent/cli/agent-cli-selection.cpp
)

link_libraries(cpp-httplib)

if (LLAMA_MEMORY_COZO)
    list(APPEND AGENT_RUNTIME_SOURCES
        ../../tools/agent/resource/agent-resource-catalog-cozo.cpp
    )
endif()

# Shared implementation for the agent CLI, daemon, and model-free smoke
# targets.  Keep the source list here so the implementation is compiled once
# per build configuration instead of once per executable target.
add_library(llama-agent-runtime-support ${AGENT_RUNTIME_SOURCES})
target_include_directories(llama-agent-runtime-support
    PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tools/agent/cli
        ${PROJECT_SOURCE_DIR}/tools/agent/daemon
        ${PROJECT_SOURCE_DIR}/tools/agent/host
        ${PROJECT_SOURCE_DIR}/tools/agent/mcp
        ${PROJECT_SOURCE_DIR}/tools/agent/runtime
        ${PROJECT_SOURCE_DIR}/tools/agent/tooling
        ${PROJECT_SOURCE_DIR}/tools/agent/resource
        ${PROJECT_SOURCE_DIR}/tools/agent/data
        ${PROJECT_SOURCE_DIR}/tools/agent/diagnostics
        ${PROJECT_SOURCE_DIR}/tools/server
)
target_compile_features(llama-agent-runtime-support PUBLIC cxx_std_17)
target_compile_definitions(llama-agent-runtime-support
    PUBLIC LLAMA_MEMORY_POC_USE_AGENT_TOOLS
)
target_link_libraries(llama-agent-runtime-support
    PUBLIC
        llama-memory
        llama-common
        llama
        llama-agent-poc
        server-context
        Threads::Threads
    PRIVATE
        cpp-httplib
)
if (WIN32)
    target_link_libraries(llama-agent-runtime-support PUBLIC ws2_32)
endif()
if (LLAMA_MEMORY_COZO)
    target_include_directories(llama-agent-runtime-support PRIVATE ${COZO_INCLUDE_DIR})
    target_link_libraries(llama-agent-runtime-support PUBLIC llama-agent-storage-cozo)
    llama_link_cozo(llama-agent-runtime-support)
endif()
if (BUILD_SHARED_LIBS)
    set_target_properties(llama-agent-runtime-support PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
endif()

# The implementation is now owned by the library.  The variable remains
# defined above as the single source list, but must not be added directly to
# executable targets below.
set(AGENT_RUNTIME_SOURCES)

add_library(llama-agent-daemon-protocol STATIC
    ../../tools/agent/daemon/agent-daemon-jsonl-protocol.cpp
)
target_include_directories(llama-agent-daemon-protocol
    PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tools/agent/daemon
        ${PROJECT_SOURCE_DIR}/vendor
)
target_compile_features(llama-agent-daemon-protocol PUBLIC cxx_std_17)
target_link_libraries(llama-agent-daemon-protocol PUBLIC llama-common Threads::Threads)
if (BUILD_SHARED_LIBS)
    set_target_properties(llama-agent-daemon-protocol PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
endif()

target_link_libraries(llama-agent-runtime-support PUBLIC llama-agent-daemon-protocol)

set(AGENT_MCP_SERVER_SOURCES
    ../../common/agent/schema-contract.cpp
    ../../tools/agent/mcp/agent-mcp-protocol.cpp
    ../../tools/agent/mcp/agent-mcp-task.cpp
    ../../tools/agent/mcp/agent-mcp-server-protocol.cpp
    ../../tools/agent/mcp/agent-mcp-server-tool-registry.cpp
    ../../tools/agent/mcp/agent-mcp-http-server.cpp
    ../../tools/agent/mcp/agent-mcp-agent-tools.cpp
    ../../tools/agent/mcp/agent-mcp-auth.cpp
    ../../tools/agent/mcp/agent-mcp-stdio-server.cpp
)

add_library(llama-agent-mcp-server ${AGENT_MCP_SERVER_SOURCES})
target_include_directories(llama-agent-mcp-server
    PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tools/agent/daemon
        ${PROJECT_SOURCE_DIR}/tools/agent/mcp
        ${PROJECT_SOURCE_DIR}/tools/agent/resource
        ${PROJECT_SOURCE_DIR}/tools/server
)
target_compile_features(llama-agent-mcp-server PUBLIC cxx_std_17)
target_link_libraries(llama-agent-mcp-server
    PUBLIC
        llama-common
        Threads::Threads
    PRIVATE
        cpp-httplib
)
if (BUILD_SHARED_LIBS)
    set_target_properties(llama-agent-mcp-server PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        WINDOWS_EXPORT_ALL_SYMBOLS ON
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

set(TARGET llama-agent-sandbox-contract-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-sandbox-contract-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE
    llama-common
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-sandbox-docker-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-sandbox-docker-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE
    llama-agent-poc
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-sandbox-kubernetes-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-sandbox-kubernetes-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE
    llama-agent-poc
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-workspace-operation-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-workspace-operation-smoke.cpp
    ../../common/agent/research/research-workspace-factory.cpp
    ../../common/agent/research/research-workspace.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE
    llama-common
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

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

set(TARGET llama-agent-mcp-http-vertical-smoke)
add_executable(${TARGET}
    smoke/mcp/agent-mcp-http-vertical-smoke.cpp
    ${AGENT_RUNTIME_SOURCES}
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

set(TARGET llama-agent-mcp-agent-tools-smoke)
add_executable(${TARGET}
    smoke/mcp/agent-mcp-agent-tools-smoke.cpp
    ../../tools/agent/mcp/agent-mcp-protocol.cpp
    ../../tools/agent/mcp/agent-mcp-task.cpp
    ../../common/agent/schema-contract.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE llama-common cpp-httplib ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

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
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/vendor
)
target_link_libraries(${TARGET} PRIVATE
    llama-agent-daemon-protocol
    ${CMAKE_THREAD_LIBS_INIT}
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-daemon-event-stream-contract-smoke)
add_executable(${TARGET}
    smoke/daemon/agent-daemon-event-stream-contract-smoke.cpp
    ../../tools/agent/daemon/agent-daemon-event-collector.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/vendor
)
target_link_libraries(${TARGET} PRIVATE
    llama-agent-daemon-protocol
    ${CMAKE_THREAD_LIBS_INIT}
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-daemon-client-fake-daemon)
add_executable(${TARGET}
    agent-daemon-client-fake-daemon.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/vendor
)
target_link_libraries(${TARGET} PRIVATE
    llama-agent-daemon-protocol
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

set(TARGET llama-agent-deliberation-policy-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-deliberation-policy-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE llama-agent-poc)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-deliberate-runtime-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-deliberate-runtime-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE llama-agent-poc)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-research-runtime-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-research-runtime-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE llama-agent-poc)
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
    ../../tools/agent/daemon/agent-daemon-tcp.cpp
    ../../tools/agent/daemon/agent-daemon-unix.cpp
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

set(AGENT_RUNTIME_LIBRARY_TARGETS
    llama-agent-cli
    llama-agent-resident-smoke
    llama-agent-tool-provider-smoke
    llama-agent-tool-runtime-smoke
    llama-agent-mcp-tool-provider-smoke
    llama-agent-mcp-stdio-client-smoke
    llama-agent-mcp-http-client-smoke
    llama-agent-mcp-stdio-server-smoke
    llama-agent-mcp-http-inbound-dispatcher-smoke
    llama-agent-mcp-http-vertical-smoke
    llama-agent-cli-mcp-selection-smoke
    llama-agent-cli-run-mcp-smoke
    llama-agent-mcp-stdio-server
    llama-agent-daemon-dispatcher-smoke
    llama-agent-daemon-protocol-smoke
    llama-agent-daemon-client-smoke
    llama-agent-daemon-wait-events-smoke
    llama-agent-daemon-mcp-config-smoke
    llama-agent-runtime-session-host-tooling-smoke
    llama-agent-runtime-session-manager-smoke
    llama-agent-resource-store-smoke
    llama-agent-daemon
)

foreach(TARGET ${AGENT_RUNTIME_LIBRARY_TARGETS})
    target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
endforeach()

set(AGENT_MCP_SERVER_LIBRARY_TARGETS
    llama-agent-mcp-stdio-fake-server
    llama-agent-mcp-stdio-server-smoke
    llama-agent-mcp-http-inbound-dispatcher-smoke
    llama-agent-mcp-http-vertical-smoke
    llama-agent-mcp-agent-tools-smoke
    llama-agent-mcp-stdio-server
    llama-agent-daemon-mcp-config-smoke
    llama-agent-daemon
)
foreach(TARGET ${AGENT_MCP_SERVER_LIBRARY_TARGETS})
    target_link_libraries(${TARGET} PRIVATE llama-agent-mcp-server)
endforeach()

set(AGENT_RUNTIME_TARGETS
    llama-agent-cli
    llama-agent-resident-smoke
    llama-agent-tool-provider-smoke
    llama-agent-tool-runtime-smoke
    llama-agent-sandbox-contract-smoke
    llama-agent-sandbox-docker-smoke
    llama-agent-sandbox-kubernetes-smoke
    llama-agent-workspace-operation-smoke
    llama-agent-mcp-tool-provider-smoke
    llama-agent-mcp-stdio-client-smoke
    llama-agent-mcp-http-client-smoke
    llama-agent-mcp-stdio-server-smoke
    llama-agent-mcp-http-inbound-dispatcher-smoke
    llama-agent-mcp-http-vertical-smoke
    llama-agent-mcp-agent-tools-smoke
    llama-agent-mcp-stdio-server
    llama-agent-cli-mcp-selection-smoke
    llama-agent-cli-run-mcp-smoke
    llama-agent-daemon-dispatcher-smoke
    llama-agent-daemon-protocol-smoke
    llama-agent-daemon-jsonl-protocol-smoke
    llama-agent-daemon-event-stream-contract-smoke
    llama-agent-daemon-client-smoke
    llama-agent-daemon-wait-events-smoke
    llama-agent-daemon-mcp-config-smoke
    llama-agent-runtime-session-host-tooling-smoke
    llama-agent-runtime-session-manager-smoke
    llama-agent-runtime-operation-manager-smoke
    llama-agent-resource-store-smoke
    llama-agent-deliberate-runtime-smoke
    llama-agent-research-runtime-smoke
    llama-agent-daemon
)

set(AGENT_RUNTIME_SMOKE_RUNTIME_TARGETS
    llama-agent-resident-smoke
    llama-agent-tool-provider-smoke
    llama-agent-tool-runtime-smoke
    llama-agent-sandbox-contract-smoke
    llama-agent-sandbox-docker-smoke
    llama-agent-sandbox-kubernetes-smoke
    llama-agent-workspace-operation-smoke
    llama-agent-runtime-session-host-tooling-smoke
    llama-agent-runtime-session-manager-smoke
    llama-agent-runtime-operation-manager-smoke
    llama-agent-deliberation-policy-smoke
    llama-agent-deliberate-runtime-smoke
    llama-agent-research-runtime-smoke
)

set(AGENT_RUNTIME_SMOKE_MCP_TARGETS
    llama-agent-mcp-tool-provider-smoke
    llama-agent-mcp-stdio-client-smoke
    llama-agent-mcp-http-client-smoke
    llama-agent-mcp-stdio-server-smoke
    llama-agent-mcp-http-inbound-dispatcher-smoke
    llama-agent-mcp-http-vertical-smoke
    llama-agent-mcp-agent-tools-smoke
)

set(AGENT_RUNTIME_SMOKE_CLI_TARGETS
    llama-agent-cli-mcp-selection-smoke
    llama-agent-cli-run-mcp-smoke
)

set(AGENT_RUNTIME_SMOKE_DAEMON_TARGETS
    llama-agent-daemon-dispatcher-smoke
    llama-agent-daemon-protocol-smoke
    llama-agent-daemon-jsonl-protocol-smoke
    llama-agent-daemon-event-stream-contract-smoke
    llama-agent-daemon-client-smoke
    llama-agent-daemon-wait-events-smoke
    llama-agent-daemon-mcp-config-smoke
)

set(AGENT_RUNTIME_SMOKE_RESOURCE_TARGETS
    llama-agent-resource-store-smoke
)

if (BUILD_TESTING)
    add_test(NAME llama-agent-tool-provider-ctest
        COMMAND $<TARGET_FILE:llama-agent-tool-provider-smoke>)
    set_property(TEST llama-agent-tool-provider-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-tool-runtime-ctest
        COMMAND $<TARGET_FILE:llama-agent-tool-runtime-smoke>)
    set_property(TEST llama-agent-tool-runtime-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-sandbox-contract-ctest
        COMMAND $<TARGET_FILE:llama-agent-sandbox-contract-smoke>)
    set_property(TEST llama-agent-sandbox-contract-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-cli-run-mcp-ctest
        COMMAND $<TARGET_FILE:llama-agent-cli-run-mcp-smoke>)
    set_property(TEST llama-agent-cli-run-mcp-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-mcp-stdio-client-ctest
        COMMAND $<TARGET_FILE:llama-agent-mcp-stdio-client-smoke>)
    set_property(TEST llama-agent-mcp-stdio-client-ctest PROPERTY LABELS agent)
    set_tests_properties(llama-agent-mcp-stdio-client-ctest PROPERTIES TIMEOUT 120)

    add_test(NAME llama-agent-daemon-protocol-ctest
        COMMAND $<TARGET_FILE:llama-agent-daemon-protocol-smoke>)
    set_property(TEST llama-agent-daemon-protocol-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-daemon-client-ctest
        COMMAND $<TARGET_FILE:llama-agent-daemon-client-smoke>)
    set_property(TEST llama-agent-daemon-client-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-sandbox-docker-ctest
        COMMAND $<TARGET_FILE:llama-agent-sandbox-docker-smoke>)
    set_property(TEST llama-agent-sandbox-docker-ctest PROPERTY LABELS sandbox-docker)
    set_tests_properties(llama-agent-sandbox-docker-ctest PROPERTIES SKIP_RETURN_CODE 77)

    add_test(NAME llama-agent-sandbox-kubernetes-ctest
        COMMAND $<TARGET_FILE:llama-agent-sandbox-kubernetes-smoke>)
    set_property(TEST llama-agent-sandbox-kubernetes-ctest PROPERTY LABELS sandbox-kubernetes)
endif()

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

set(AGENT_TEST_PACK_CTEST_TARGETS
    test-agent-contract
    test-memory-store
    test-memory-retrieval
    test-memory-context
    test-memory-policy
    test-memory-tool-service
    test-memory-cozo
    test-plan-store
    test-plan-context
    test-plan-json
    test-plan-goal
    test-plan-memory
    test-plan-blueprint
    test-plan-scheduler
    test-plan-template
    test-plan-cozo
    test-agent-bootstrap
    test-cli-scope
    test-agent-inference
    test-agent-prepared-generation
    test-memory-learning
    test-agent-runtime-json
    test-reflection-json
    test-agent-loop
    test-agent-lifecycle-scenarios
    test-agent-tool-rounds
    test-agent-repository-tools
    test-agent-tool-failure
    test-agent-memory-search-fallback
    test-tool-registry
    test-tool-catalog
    test-tool-adapters
    test-tool-chat-bridge
    test-agent-data-store-cozo
)
set(AGENT_TEST_PACK_AVAILABLE_TARGETS)
foreach(TARGET ${AGENT_TEST_PACK_CTEST_TARGETS})
    if (TARGET ${TARGET})
        list(APPEND AGENT_TEST_PACK_AVAILABLE_TARGETS ${TARGET})
    endif()
endforeach()

add_custom_target(llama-agent-build-pack
    DEPENDS ${AGENT_RUNTIME_TARGETS} ${AGENT_TEST_PACK_AVAILABLE_TARGETS}
)

add_custom_target(llama-agent-ctest-pack
    DEPENDS llama-agent-build-pack
    COMMAND ${CMAKE_CTEST_COMMAND}
            --test-dir ${CMAKE_BINARY_DIR}
            -C $<CONFIG>
            -L agent
            --output-on-failure
    USES_TERMINAL
)

if (WIN32)
    add_custom_target(llama-agent-beta-test-pack
        DEPENDS llama-agent-build-pack
        COMMAND powershell.exe -NoProfile -ExecutionPolicy Bypass
                -File ${PROJECT_SOURCE_DIR}/scripts/test-agent-daemon-beta-smoke.ps1
                -BuildDir ${CMAKE_BINARY_DIR}
                -Configuration $<CONFIG>
                -IncludeCTest
        USES_TERMINAL
    )
elseif (UNIX)
    add_custom_target(llama-agent-beta-test-pack
        DEPENDS llama-agent-build-pack
        COMMAND bash ${PROJECT_SOURCE_DIR}/scripts/test-agent-daemon-beta-smoke.sh
                --build-dir ${CMAKE_BINARY_DIR}
                --configuration $<CONFIG>
                --include-ctest
        USES_TERMINAL
    )
endif()

if (LLAMA_MEMORY_COZO)
    foreach(TARGET ${AGENT_RUNTIME_TARGETS})
        llama_link_cozo(${TARGET})
    endforeach()
endif()
