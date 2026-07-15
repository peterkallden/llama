set(AGENT_MEMORY_RUNTIME_SOURCES
    ../agent/agent-cli-config.cpp
    ../agent/agent-cli-generation-utils.cpp
    ../agent/agent-cli-run-adapter.cpp
    ../../tools/agent/daemon/agent-daemon-client-admin.cpp
    ../../tools/agent/daemon/agent-daemon-client-status.cpp
    ../../tools/agent/daemon/agent-daemon-client.cpp
    ../agent/agent-daemon-event-collector.cpp
    ../../tools/agent/daemon/agent-daemon-jsonl-protocol.cpp
    ../agent/agent-host-config.cpp
    ../agent/agent-mcp-protocol.cpp
    ../agent/agent-selection-contracts.cpp
    ../agent/agent-mcp-stdio-client.cpp
    ../agent/agent-tool-provider.cpp
    ../agent/agent-tool-result-json-contracts.cpp
    ../agent/agent-tool-runtime-adapter.cpp
    ../../tools/agent/resource/agent-resource-catalog.cpp
    ../../tools/agent/resource/agent-resource-store.cpp
    ../agent/agent-daemon-adapter.cpp
    ../agent/agent-daemon-protocol.cpp
    ../agent/agent-daemon-runtime.cpp
    ../agent/agent-daemon-dispatcher.cpp
    ../agent/agent-daemon-service.cpp
    ../agent/agent-cli-host-adapter.cpp
    ../agent/agent-cli-inference.cpp
    ../agent/agent-plan-orchestration.cpp
    ../agent/agent-runtime-chat-driver.cpp
    ../agent/agent-runtime-host.cpp
    ../agent/agent-runtime-resident.cpp
    ../agent/agent-runtime-session-host.cpp
    ../agent/agent-runtime-session-manager.cpp
    ../agent/agent-runtime-execution.cpp
    ../agent/agent-runtime-session.cpp
    ../agent/agent-server-context-host.cpp
    ../agent/agent-server-generation.cpp
    ../agent/agent-server-inference.cpp
    ../agent/agent-runtime-assembly.cpp
    ../agent/agent-cli-runtime.cpp
    ../agent/agent-cli-selection.cpp
)

if (LLAMA_MEMORY_COZO)
    list(APPEND AGENT_MEMORY_RUNTIME_SOURCES
        ../../tools/agent/resource/agent-resource-catalog-cozo.cpp
    )
endif()

set(TARGET llama-memory-poc)
add_executable(${TARGET}
    memory.cpp
    ../common/cli-config.cpp
    memory-cli-chat.cpp
    memory-cli-memory.cpp
    ../agent/agent-cli-command.cpp
    ${AGENT_MEMORY_RUNTIME_SOURCES}
    ../agent/agent-cli-run.cpp
)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME llama-memory)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../agent
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-memory llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (LLAMA_AGENT_REFLECTION)
    target_link_libraries(${TARGET} PRIVATE llama-agent-poc server-context)
    target_compile_definitions(${TARGET} PRIVATE LLAMA_MEMORY_POC_USE_AGENT_TOOLS)
endif()

if (LLAMA_MEMORY_COZO)
    target_include_directories(${TARGET} PRIVATE ${COZO_INCLUDE_DIR})
    target_link_libraries(${TARGET} PRIVATE ${COZO_LIBRARY})
    if (WIN32)
        target_link_libraries(${TARGET} PRIVATE ws2_32 bcrypt ntdll rpcrt4 shlwapi crypt32)
    endif()
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
