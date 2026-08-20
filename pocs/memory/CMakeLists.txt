include_directories(${PROJECT_SOURCE_DIR})

set(AGENT_MEMORY_RUNTIME_SOURCES
    ../../tools/agent/cli/agent-cli-config.cpp
    ../../tools/agent/cli/agent-cli-generation-utils.cpp
    ../../tools/agent/cli/agent-cli-run-adapter.cpp
    ../../tools/agent/daemon/agent-daemon-client-admin.cpp
    ../../tools/agent/daemon/agent-daemon-client-status.cpp
    ../../tools/agent/daemon/agent-daemon-client.cpp
    ../../tools/agent/daemon/agent-daemon-event-collector.cpp
    ../../tools/agent/host/agent-host-config.cpp
    ../../tools/agent/host/agent-sandbox-assembly.cpp
    ../../tools/agent/mcp/agent-mcp-protocol.cpp
    ../../tools/agent/mcp/agent-mcp-client-factory.cpp
    ../../tools/agent/tooling/agent-selection-contracts.cpp
    ../../tools/agent/mcp/agent-mcp-stdio-client.cpp
    ../../tools/agent/tooling/agent-tool-provider.cpp
    ../../tools/agent/tooling/agent-tool-result-json-contracts.cpp
    ../../tools/agent/tooling/agent-tool-runtime-adapter.cpp
    ../../tools/agent/resource/agent-resource-catalog.cpp
    ../../tools/agent/resource/agent-resource-store.cpp
    ../../tools/agent/resource/assembly/agent-resource-processor-factory.cpp
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
    ../../tools/agent/runtime/agent-runtime-session-manager.cpp
    ../../tools/agent/runtime/agent-runtime-execution.cpp
    ../../tools/agent/runtime/agent-runtime-session.cpp
    ../../tools/agent/runtime/agent-server-context-host.cpp
    ../../tools/agent/runtime/agent-server-generation.cpp
    ../../tools/agent/runtime/agent-server-inference.cpp
    ../../tools/agent/runtime/agent-runtime-assembly.cpp
    ../../tools/agent/cli/agent-cli-runtime.cpp
    ../../tools/agent/cli/agent-cli-selection.cpp
)

if (LLAMA_MEMORY_COZO)
    list(APPEND AGENT_MEMORY_RUNTIME_SOURCES
        ../../tools/agent/resource/agent-resource-catalog-cozo.cpp
    )
endif()

if (LLAMA_AGENT_RUNTIME)
    # The shared agent implementation is configured before this POC target.
    # Keep the source-list fallback above for memory-only builds, where the
    # agent runtime library is intentionally not available.
    set(AGENT_MEMORY_RUNTIME_SOURCES)
endif()

if (NOT LLAMA_AGENT_RUNTIME)
    list(APPEND AGENT_MEMORY_RUNTIME_SOURCES
        ../../tools/agent/daemon/agent-daemon-jsonl-protocol.cpp
    )
endif()

set(TARGET llama-memory-poc)
add_executable(${TARGET}
    memory.cpp
    ../../tools/agent/cli/agent-cli-options.cpp
    ../../tools/agent/cli/agent-cli-generation.cpp
    ../../tools/agent/cli/agent-cli-memory-tools.cpp
    ../../tools/agent/cli/agent-cli-command.cpp
    ${AGENT_MEMORY_RUNTIME_SOURCES}
    ../../tools/agent/cli/agent-cli-run.cpp
)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME llama-memory)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../agent
    ${PROJECT_SOURCE_DIR}/tools/agent/cli
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime
    ${PROJECT_SOURCE_DIR}/tools/agent/tooling
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-memory llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (LLAMA_AGENT_RUNTIME)
    target_link_libraries(${TARGET} PRIVATE
        llama-agent-core
        llama-agent-runtime-support
        server-context
    )
    target_compile_definitions(${TARGET} PRIVATE LLAMA_MEMORY_POC_USE_AGENT_TOOLS)
endif()

if (LLAMA_MEMORY_COZO)
    llama_link_cozo(${TARGET})
endif()

# The agent package installs its host/runtime binaries separately. The legacy
# memory PoC is still built for agent-backed tests, but is not part of that
# package and therefore must not add an install-time dependency on its binary.
if(LLAMA_TOOLS_INSTALL AND NOT LLAMA_AGENT_RUNTIME)
    install(TARGETS ${TARGET} RUNTIME)
endif()
