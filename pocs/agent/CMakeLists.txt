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
    ../../tools/agent/host/agent-sandbox-assembly.cpp
    ../../tools/agent/mcp/agent-mcp-http-client.cpp
    ../../tools/agent/mcp/agent-mcp-client-factory.cpp
    ../../tools/agent/tooling/agent-selection-contracts.cpp
    ../../tools/agent/mcp/agent-mcp-stdio-client.cpp
    ../../tools/agent/tooling/agent-tool-provider.cpp
    ../../tools/agent/tooling/agent-tool-result-json-contracts.cpp
    ../../tools/agent/tooling/agent-tool-runtime-adapter.cpp
    ../../tools/agent/resource/agent-resource-catalog.cpp
    ../../tools/agent/resource/agent-resource-store.cpp
    ../../tools/agent/resource/agent-resource-chunker.cpp
    ../../tools/agent/resource/agent-resource-media-type.cpp
    ../../tools/agent/resource/processors/agent-pdf-text-processor.cpp
    ../../tools/agent/resource/processors/agent-pandoc-command.cpp
    ../../tools/agent/resource/processors/agent-docx-text-processor.cpp
    ../../tools/agent/resource/processors/agent-xlsx-workbook-json-processor.cpp
    ../../tools/agent/resource/processors/agent-pdf-page-image-command.cpp
    ../../tools/agent/resource/processors/agent-pdf-page-image-processor.cpp
    ../../tools/agent/resource/processors/agent-tesseract-ocr-command.cpp
    ../../tools/agent/resource/processors/agent-tesseract-ocr-processor.cpp
    ../../tools/agent/resource/agent-resource-processing-service.cpp
    ../../tools/agent/resource/assembly/agent-resource-processor-factory.cpp
    ../../tools/agent/resource/agent-resource-processing-cache.cpp
    ../../common/agent/sandbox/sandbox-local-runtime.cpp
    ../../tools/agent/data/agent-data-store-factory.cpp
    ../../tools/agent/data/agent-dataset-importer.cpp
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

if (LLAMA_MEMORY_COZO)
    list(APPEND AGENT_RUNTIME_SOURCES
        ../../tools/agent/resource/agent-resource-catalog-cozo.cpp
    )
endif()

# Keep the host/runtime assembly independent from the high-churn resource,
# data and diagnostics implementations. These are agent-only libraries; the
# upstream llama.cpp targets remain untouched.
list(REMOVE_ITEM AGENT_RUNTIME_SOURCES
    ../../tools/agent/resource/agent-resource-catalog.cpp
    ../../tools/agent/resource/agent-resource-chunker.cpp
    ../../tools/agent/resource/agent-resource-media-type.cpp
    ../../tools/agent/resource/agent-resource-store.cpp
    ../../tools/agent/resource/agent-resource-processing-service.cpp
    ../../tools/agent/resource/agent-resource-processing-cache.cpp
    ../../tools/agent/resource/assembly/agent-resource-processor-factory.cpp
    ../../tools/agent/resource/processors/agent-pdf-text-processor.cpp
    ../../tools/agent/resource/processors/agent-pandoc-command.cpp
    ../../tools/agent/resource/processors/agent-docx-text-processor.cpp
    ../../tools/agent/resource/processors/agent-xlsx-workbook-json-processor.cpp
    ../../tools/agent/resource/processors/agent-pdf-page-image-command.cpp
    ../../tools/agent/resource/processors/agent-pdf-page-image-processor.cpp
    ../../tools/agent/resource/processors/agent-tesseract-ocr-command.cpp
    ../../tools/agent/resource/processors/agent-tesseract-ocr-processor.cpp
    ../../tools/agent/data/agent-data-store-factory.cpp
    ../../tools/agent/data/agent-dataset-importer.cpp
    ../../tools/agent/diagnostics/agent-clangd-protocol.cpp
    ../../tools/agent/diagnostics/agent-clangd-session.cpp
    ../../tools/agent/diagnostics/agent-clangd-provider.cpp
    ../../tools/agent/resource/agent-resource-catalog-cozo.cpp
)

add_library(llama-agent-resource
    ../../tools/agent/resource/agent-resource-catalog.cpp
    ../../tools/agent/resource/agent-resource-chunker.cpp
    ../../tools/agent/resource/agent-resource-media-type.cpp
    ../../tools/agent/resource/agent-resource-store.cpp
    ../../tools/agent/resource/agent-resource-processing-service.cpp
    ../../tools/agent/resource/agent-resource-processing-cache.cpp
    ../../tools/agent/resource/assembly/agent-resource-processor-factory.cpp
    ../../tools/agent/resource/processors/agent-pdf-text-processor.cpp
    ../../tools/agent/resource/processors/agent-pandoc-command.cpp
    ../../tools/agent/resource/processors/agent-docx-text-processor.cpp
    ../../tools/agent/resource/processors/agent-xlsx-workbook-json-processor.cpp
    ../../tools/agent/resource/processors/agent-pdf-page-image-command.cpp
    ../../tools/agent/resource/processors/agent-pdf-page-image-processor.cpp
    ../../tools/agent/resource/processors/agent-tesseract-ocr-command.cpp
    ../../tools/agent/resource/processors/agent-tesseract-ocr-processor.cpp
)
if (LLAMA_MEMORY_COZO)
    target_sources(llama-agent-resource PRIVATE
        ../../tools/agent/resource/agent-resource-catalog-cozo.cpp
    )
endif()
target_include_directories(llama-agent-resource PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/agent/data
    ${PROJECT_SOURCE_DIR}/tools/agent/diagnostics
)
target_compile_features(llama-agent-resource PUBLIC cxx_std_17)
target_link_libraries(llama-agent-resource PUBLIC
    llama-common
    llama-agent-core
    Threads::Threads
)
if (LLAMA_MEMORY_COZO)
    target_include_directories(llama-agent-resource PRIVATE ${COZO_INCLUDE_DIR})
    llama_link_cozo(llama-agent-resource)
endif()
if (BUILD_SHARED_LIBS)
    set_target_properties(llama-agent-resource PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
endif()

add_library(llama-agent-data
    ../../tools/agent/data/agent-data-store-factory.cpp
    ../../tools/agent/data/agent-dataset-importer.cpp
)
target_include_directories(llama-agent-data PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/data
)
target_compile_features(llama-agent-data PUBLIC cxx_std_17)
target_link_libraries(llama-agent-data PUBLIC llama-agent-core)
if (LLAMA_MEMORY_COZO)
    target_link_libraries(llama-agent-data PUBLIC llama-agent-storage-cozo)
endif()
if (BUILD_SHARED_LIBS)
    set_target_properties(llama-agent-data PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
endif()

add_library(llama-agent-diagnostics
    ../../tools/agent/diagnostics/agent-clangd-protocol.cpp
    ../../tools/agent/diagnostics/agent-clangd-session.cpp
    ../../tools/agent/diagnostics/agent-clangd-provider.cpp
)
target_include_directories(llama-agent-diagnostics PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/diagnostics
)
target_compile_features(llama-agent-diagnostics PUBLIC cxx_std_17)
target_link_libraries(llama-agent-diagnostics PUBLIC
    llama-common
    llama-agent-core
    Threads::Threads
)
if (BUILD_SHARED_LIBS)
    set_target_properties(llama-agent-diagnostics PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        WINDOWS_EXPORT_ALL_SYMBOLS ON
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
        ${PROJECT_SOURCE_DIR}/tools/mtmd
        ${PROJECT_SOURCE_DIR}/tools/server
)
target_compile_features(llama-agent-runtime-support PUBLIC cxx_std_17)
target_compile_definitions(llama-agent-runtime-support
    PUBLIC LLAMA_MEMORY_POC_USE_AGENT_TOOLS
        LLAMA_AGENT_SYSCONFDIR="${LLAMA_AGENT_SYSCONFDIR}"
)
target_link_libraries(llama-agent-runtime-support
    PUBLIC
        llama-agent-memory
        llama-common
        llama
        llama-agent-core
        llama-agent-resource
        llama-agent-data
        llama-agent-diagnostics
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

# Shared MCP contracts and task lifecycle implementation. Keep this separate
# from the runtime and server assemblies so these sources are compiled once
# and reused by both sides.
add_library(llama-agent-mcp-core
    ../../tools/agent/mcp/agent-mcp-protocol.cpp
    ../../tools/agent/mcp/agent-mcp-task.cpp
)
target_include_directories(llama-agent-mcp-core
    PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tools/agent/mcp
        ${PROJECT_SOURCE_DIR}/tools/agent/tooling
)
target_compile_features(llama-agent-mcp-core PUBLIC cxx_std_17)
target_link_libraries(llama-agent-mcp-core PUBLIC llama-common Threads::Threads)
target_link_libraries(llama-agent-runtime-support PUBLIC llama-agent-mcp-core)
if (BUILD_SHARED_LIBS)
    set_target_properties(llama-agent-mcp-core PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
endif()

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
    ../../common/agent/dataset-contracts.cpp
    ../../common/agent/tooling/contracts/schema-contract.cpp
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
        llama-agent-mcp-core
    PRIVATE
        cpp-httplib
)
if (BUILD_SHARED_LIBS)
    set_target_properties(llama-agent-mcp-server PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
endif()

# Install the shared agent runtime closure when producing an agent package.
# The executables below depend on these libraries when BUILD_SHARED_LIBS is
# enabled (the default for the supported Linux and MSVC builds). Keep this
# separate from the smoke targets so package installation does not pull test
# binaries into the runtime package.
if(LLAMA_TOOLS_INSTALL OR LLAMA_AGENT_PACKAGE_INSTALL)
    set(AGENT_PACKAGE_LIBRARY_TARGETS
        llama-agent-memory
        llama-agent-plan
        llama-agent-core
        llama-agent-mcp-core
        llama-agent-mcp-server
        llama-agent-runtime-support
        llama-agent-resource
        llama-agent-data
        llama-agent-diagnostics
    )
    if(TARGET llama-agent-storage-cozo)
        list(APPEND AGENT_PACKAGE_LIBRARY_TARGETS llama-agent-storage-cozo)
    endif()
    install(TARGETS ${AGENT_PACKAGE_LIBRARY_TARGETS}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
endif()

set(TARGET llama-agent-cli)
add_executable(${TARGET}
    ../../tools/agent/cli/agent-cli-main.cpp
    ../../tools/agent/cli/agent-cli-command.cpp
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
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL OR LLAMA_AGENT_PACKAGE_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

set(TARGET llama-agent-resident-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-resident-smoke.cpp
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
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-tool-provider-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-tool-provider-smoke.cpp
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
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-tool-runtime-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-tool-runtime-smoke.cpp
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
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-continuation-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-continuation-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# This legacy runtime-driver collection is kept as a compatibility smoke while
# its cases are moved into focused runtime tests. It must use the runtime
# support closure rather than the core PoC library alone.
set(TARGET llama-agent-inference-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-inference-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-mcp-tool-provider-smoke)
add_executable(${TARGET}
    smoke/mcp/agent-mcp-tool-provider-smoke.cpp
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
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

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
    llama-agent-core
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
    llama-agent-core
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-workspace-operation-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-workspace-operation-smoke.cpp
    ../../common/agent/thinking/research/research-workspace-factory.cpp
    ../../common/agent/thinking/research/research-workspace.cpp
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
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-mcp-http-client-smoke)
add_executable(${TARGET}
    smoke/mcp/agent-mcp-http-client-smoke.cpp
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
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support cpp-httplib)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-mcp-stdio-server-smoke)
add_executable(${TARGET}
    smoke/mcp/agent-mcp-stdio-server-smoke.cpp
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
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-mcp-http-inbound-dispatcher-smoke)
add_executable(${TARGET}
    smoke/mcp/agent-mcp-http-server-smoke.cpp
    ../../common/agent/tooling/contracts/schema-contract.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/server
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support cpp-httplib)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-mcp-http-vertical-smoke)
add_executable(${TARGET}
    smoke/mcp/agent-mcp-http-vertical-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/server
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support cpp-httplib)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-mcp-agent-tools-smoke)
add_executable(${TARGET}
    smoke/mcp/agent-mcp-agent-tools-smoke.cpp
    ../../common/agent/tooling/contracts/schema-contract.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE llama-agent-mcp-core llama-common cpp-httplib ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-cli-mcp-selection-smoke)
add_executable(${TARGET}
    smoke/cli/agent-cli-mcp-selection-smoke.cpp
)
add_dependencies(${TARGET} llama-agent-mcp-stdio-fake-server)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-cli-run-mcp-smoke)
add_executable(${TARGET}
    smoke/cli/agent-cli-run-mcp-smoke.cpp
)
add_dependencies(${TARGET} llama-agent-mcp-stdio-fake-server)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-mcp-stdio-fake-server)
add_executable(${TARGET}
    ../../tools/agent/mcp/agent-mcp-stdio-fake-server.cpp
    ../../common/agent/tooling/contracts/schema-contract.cpp
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
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support cpp-httplib)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-daemon-dispatcher-smoke)
add_executable(${TARGET}
    smoke/daemon/agent-daemon-dispatcher-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-daemon-protocol-smoke)
add_executable(${TARGET}
    smoke/daemon/agent-daemon-protocol-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

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
)
add_dependencies(${TARGET} llama-agent-daemon-client-fake-daemon)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-daemon-wait-events-smoke)
add_executable(${TARGET}
    smoke/daemon/agent-daemon-wait-events-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-daemon-mcp-config-smoke)
add_executable(${TARGET}
    smoke/daemon/agent-daemon-mcp-config-smoke.cpp
)
add_dependencies(${TARGET} llama-agent-mcp-stdio-fake-server)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-config-discovery-smoke)
add_executable(${TARGET}
    smoke/daemon/agent-config-discovery-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/host
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-runtime-session-host-tooling-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-runtime-session-host-tooling-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-runtime-session-manager-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-runtime-session-manager-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

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
target_link_libraries(${TARGET} PRIVATE llama-agent-core)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-deliberate-runtime-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-deliberate-runtime-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE llama-agent-core)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-research-runtime-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-research-runtime-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE llama-agent-core)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-resource-store-smoke)
add_executable(${TARGET}
    smoke/resource/agent-resource-store-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-resource-chunker-smoke)
add_executable(${TARGET}
    smoke/resource/agent-resource-chunker-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-resource-range-chunker-smoke)
add_executable(${TARGET}
    smoke/resource/agent-resource-range-chunker-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-cli-document-table-smoke)
add_executable(${TARGET}
    smoke/cli/agent-cli-document-table-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-sandbox-local-runtime-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-sandbox-local-runtime-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-resource-processing-contract-smoke)
add_executable(${TARGET}
    smoke/resource/agent-resource-processing-contract-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}/common
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-resource-backend-smoke)
add_executable(${TARGET}
    smoke/resource/agent-resource-backend-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-resource-processor-dispatch-smoke)
add_executable(${TARGET}
    smoke/resource/agent-resource-processor-dispatch-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/common
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-resource-media-type-smoke)
add_executable(${TARGET}
    smoke/resource/agent-resource-media-type-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-resource-processing-service-smoke)
add_executable(${TARGET}
    smoke/resource/agent-resource-processing-service-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-pdf-text-processor-smoke)
add_executable(${TARGET}
    smoke/resource/agent-pdf-text-processor-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-docx-text-processor-smoke)
add_executable(${TARGET}
    smoke/resource/agent-docx-text-processor-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-dataset-contract-smoke)
add_executable(${TARGET}
    smoke/data/agent-dataset-contract-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE llama-agent-core)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-research-csv-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-research-csv-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE llama-agent-core)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-dataset-importer-smoke)
add_executable(${TARGET}
    smoke/data/agent-dataset-importer-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/data
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (LLAMA_MEMORY_COZO)
    set(TARGET llama-agent-data-manipulation-smoke)
    add_executable(${TARGET}
        smoke/data/agent-data-manipulation-smoke.cpp
    )
    target_include_directories(${TARGET} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tools/agent/data
    )
    target_link_libraries(${TARGET} PRIVATE llama-agent-storage-cozo)
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
endif()

set(TARGET llama-agent-docx-text-local-e2e-smoke)
add_executable(${TARGET}
    smoke/resource/agent-docx-text-local-e2e-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-pdf-page-image-command-smoke)
add_executable(${TARGET}
    smoke/resource/agent-pdf-page-image-command-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-tesseract-ocr-command-smoke)
add_executable(${TARGET}
    smoke/resource/agent-tesseract-ocr-command-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-resource-processing-host-smoke)
add_executable(${TARGET}
    smoke/resource/agent-resource-processing-host-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-tesseract-ocr-processor-smoke)
add_executable(${TARGET}
    smoke/resource/agent-tesseract-ocr-processor-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-pdf-page-image-processor-smoke)
add_executable(${TARGET}
    smoke/resource/agent-pdf-page-image-processor-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-pdf-page-image-local-e2e-smoke)
add_executable(${TARGET}
    smoke/resource/agent-pdf-page-image-local-e2e-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-daemon)
add_executable(${TARGET}
    ../../tools/agent/daemon/agent-daemon-main.cpp
    ../../tools/agent/daemon/agent-daemon-tcp.cpp
    ../../tools/agent/daemon/agent-daemon-unix.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support cpp-httplib)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL OR LLAMA_AGENT_PACKAGE_INSTALL)
    install(TARGETS llama-agent-mcp-stdio-server llama-agent-daemon RUNTIME)
endif()

if(LLAMA_TOOLS_INSTALL OR LLAMA_AGENT_PACKAGE_INSTALL)
    if(WIN32)
        install(FILES
            ${PROJECT_SOURCE_DIR}/scripts/agent-config-bootstrap.ps1
            DESTINATION bin
            RENAME llama-agent-config-bootstrap.ps1)
    else()
        install(PROGRAMS
            ${PROJECT_SOURCE_DIR}/scripts/agent-config-bootstrap.sh
            DESTINATION bin
            RENAME llama-agent-config-bootstrap)
    endif()

    # Keep this install manifest explicit.  A CONFIGURE_DEPENDS glob here
    # makes Ninja rerun the whole project configuration on CMake versions
    # that do not materialize the unchanged glob-verification output.
    set(AGENT_HOST_CONFIG_EXAMPLES
        ${PROJECT_SOURCE_DIR}/docs/examples/agent-host-config-capabilities.json
        ${PROJECT_SOURCE_DIR}/docs/examples/agent-host-config-inbound-mcp-agent.json
        ${PROJECT_SOURCE_DIR}/docs/examples/agent-host-config-inbound-mcp-auth.json
        ${PROJECT_SOURCE_DIR}/docs/examples/agent-host-config-inbound-mcp-home.json
        ${PROJECT_SOURCE_DIR}/docs/examples/agent-host-config-inbound-mcp-jwt.json
        ${PROJECT_SOURCE_DIR}/docs/examples/agent-host-config-jsonl-tcp.json
        ${PROJECT_SOURCE_DIR}/docs/examples/agent-host-config-jsonl-unix.json
        ${PROJECT_SOURCE_DIR}/docs/examples/agent-host-config-remote-http.json
        ${PROJECT_SOURCE_DIR}/docs/examples/agent-host-config-stdio.json)
    install(FILES
        ${PROJECT_SOURCE_DIR}/docs/examples/agent-config.example.json
        ${AGENT_HOST_CONFIG_EXAMPLES}
        ${PROJECT_SOURCE_DIR}/docs/examples/agent-daemon-requests.jsonl
        ${PROJECT_SOURCE_DIR}/docs/examples/agent-runtime-context-budgets.json
        ${PROJECT_SOURCE_DIR}/docs/examples/llama-agent-daemon.service
        DESTINATION share/llama-agent/examples)
endif()

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
    llama-agent-mcp-stdio-server
    llama-agent-daemon
)

set(AGENT_TEST_BUILD_TARGETS
    llama-agent-cli-document-table-smoke
    llama-agent-resident-smoke
    llama-agent-tool-provider-smoke
    llama-agent-tool-runtime-smoke
    llama-agent-continuation-smoke
    llama-agent-inference-smoke
    llama-agent-sandbox-contract-smoke
    llama-agent-sandbox-local-runtime-smoke
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
    llama-agent-cli-mcp-selection-smoke
    llama-agent-cli-run-mcp-smoke
    llama-agent-daemon-dispatcher-smoke
    llama-agent-daemon-protocol-smoke
    llama-agent-daemon-jsonl-protocol-smoke
    llama-agent-daemon-event-stream-contract-smoke
    llama-agent-daemon-client-smoke
    llama-agent-daemon-wait-events-smoke
    llama-agent-daemon-mcp-config-smoke
    llama-agent-config-discovery-smoke
    llama-agent-runtime-session-host-tooling-smoke
    llama-agent-runtime-session-manager-smoke
    llama-agent-runtime-operation-manager-smoke
    llama-agent-resource-store-smoke
    llama-agent-resource-chunker-smoke
    llama-agent-resource-range-chunker-smoke
    llama-agent-resource-processing-contract-smoke
    llama-agent-resource-backend-smoke
    llama-agent-resource-processor-dispatch-smoke
    llama-agent-resource-media-type-smoke
    llama-agent-resource-processing-service-smoke
    llama-agent-pdf-text-processor-smoke
    llama-agent-docx-text-processor-smoke
    llama-agent-dataset-contract-smoke
    llama-agent-dataset-importer-smoke
    llama-agent-docx-text-local-e2e-smoke
    llama-agent-pdf-page-image-command-smoke
    llama-agent-tesseract-ocr-command-smoke
    llama-agent-tesseract-ocr-processor-smoke
    llama-agent-resource-processing-host-smoke
    llama-agent-pdf-page-image-processor-smoke
    llama-agent-pdf-page-image-local-e2e-smoke
    llama-agent-deliberate-runtime-smoke
    llama-agent-research-runtime-smoke
    llama-agent-research-csv-smoke
)
if (LLAMA_MEMORY_COZO)
    list(APPEND AGENT_TEST_BUILD_TARGETS llama-agent-data-manipulation-smoke)
endif()

set(AGENT_RUNTIME_SMOKE_RUNTIME_TARGETS
    llama-agent-resident-smoke
    llama-agent-tool-provider-smoke
    llama-agent-tool-runtime-smoke
    llama-agent-continuation-smoke
    llama-agent-inference-smoke
    llama-agent-sandbox-contract-smoke
    llama-agent-sandbox-local-runtime-smoke
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
    llama-agent-cli-document-table-smoke
)

set(AGENT_RUNTIME_SMOKE_DAEMON_TARGETS
    llama-agent-daemon-dispatcher-smoke
    llama-agent-daemon-protocol-smoke
    llama-agent-daemon-jsonl-protocol-smoke
    llama-agent-daemon-event-stream-contract-smoke
    llama-agent-daemon-client-smoke
    llama-agent-daemon-wait-events-smoke
    llama-agent-daemon-mcp-config-smoke
    llama-agent-config-discovery-smoke
)

set(AGENT_RUNTIME_SMOKE_RESOURCE_TARGETS
    llama-agent-resource-store-smoke
    llama-agent-resource-chunker-smoke
    llama-agent-resource-range-chunker-smoke
    llama-agent-resource-processing-contract-smoke
    llama-agent-resource-backend-smoke
    llama-agent-resource-processor-dispatch-smoke
    llama-agent-resource-media-type-smoke
    llama-agent-resource-processing-service-smoke
    llama-agent-pdf-text-processor-smoke
    llama-agent-docx-text-processor-smoke
    llama-agent-dataset-contract-smoke
    llama-agent-dataset-importer-smoke
    llama-agent-docx-text-local-e2e-smoke
    llama-agent-pdf-page-image-command-smoke
    llama-agent-resource-processing-host-smoke
    llama-agent-pdf-page-image-processor-smoke
    llama-agent-pdf-page-image-local-e2e-smoke
)

if (BUILD_TESTING)
    add_test(NAME llama-agent-tool-provider-ctest
        COMMAND $<TARGET_FILE:llama-agent-tool-provider-smoke>)
    set_property(TEST llama-agent-tool-provider-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-tool-runtime-ctest
        COMMAND $<TARGET_FILE:llama-agent-tool-runtime-smoke>)
    set_property(TEST llama-agent-tool-runtime-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-continuation-ctest
        COMMAND $<TARGET_FILE:llama-agent-continuation-smoke>)
    set_property(TEST llama-agent-continuation-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-deliberate-runtime-ctest
        COMMAND $<TARGET_FILE:llama-agent-deliberate-runtime-smoke>)
    set_property(TEST llama-agent-deliberate-runtime-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-research-runtime-ctest
        COMMAND $<TARGET_FILE:llama-agent-research-runtime-smoke>)
    set_property(TEST llama-agent-research-runtime-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-research-csv-ctest
        COMMAND $<TARGET_FILE:llama-agent-research-csv-smoke>)
    set_property(TEST llama-agent-research-csv-ctest PROPERTY LABELS agent)
    set_tests_properties(llama-agent-research-csv-ctest PROPERTIES
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

    add_test(NAME llama-agent-inference-ctest
        COMMAND $<TARGET_FILE:llama-agent-inference-smoke>)
    set_property(TEST llama-agent-inference-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-sandbox-contract-ctest
        COMMAND $<TARGET_FILE:llama-agent-sandbox-contract-smoke>)
    set_property(TEST llama-agent-sandbox-contract-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-sandbox-local-runtime-ctest
        COMMAND $<TARGET_FILE:llama-agent-sandbox-local-runtime-smoke>)
    set_property(TEST llama-agent-sandbox-local-runtime-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-workspace-operation-ctest
        COMMAND $<TARGET_FILE:llama-agent-workspace-operation-smoke>)
    set_property(TEST llama-agent-workspace-operation-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-cli-run-mcp-ctest
        COMMAND $<TARGET_FILE:llama-agent-cli-run-mcp-smoke>)
    set_property(TEST llama-agent-cli-run-mcp-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-cli-mcp-selection-ctest
        COMMAND $<TARGET_FILE:llama-agent-cli-mcp-selection-smoke>)
    set_property(TEST llama-agent-cli-mcp-selection-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-cli-document-table-ctest
        COMMAND $<TARGET_FILE:llama-agent-cli-document-table-smoke>)
    set_property(TEST llama-agent-cli-document-table-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-mcp-stdio-client-ctest
        COMMAND $<TARGET_FILE:llama-agent-mcp-stdio-client-smoke>)
    set_property(TEST llama-agent-mcp-stdio-client-ctest PROPERTY LABELS agent)
    set_tests_properties(llama-agent-mcp-stdio-client-ctest PROPERTIES TIMEOUT 120)

    add_test(NAME llama-agent-daemon-protocol-ctest
        COMMAND $<TARGET_FILE:llama-agent-daemon-protocol-smoke>)
    set_property(TEST llama-agent-daemon-protocol-ctest PROPERTY LABELS agent)

    # The protocol smoke contains the executable readiness-gate contract. Keep
    # a separately named CTest entry so assurance reports expose readiness as
    # an explicit check without introducing a second daemon fixture.
    add_test(NAME llama-agent-daemon-readiness-ctest
        COMMAND $<TARGET_FILE:llama-agent-daemon-protocol-smoke>)
    set_property(TEST llama-agent-daemon-readiness-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-daemon-jsonl-protocol-ctest
        COMMAND $<TARGET_FILE:llama-agent-daemon-jsonl-protocol-smoke>)
    set_property(TEST llama-agent-daemon-jsonl-protocol-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-daemon-client-ctest
        COMMAND $<TARGET_FILE:llama-agent-daemon-client-smoke>)
    set_property(TEST llama-agent-daemon-client-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-daemon-mcp-config-ctest
        COMMAND $<TARGET_FILE:llama-agent-daemon-mcp-config-smoke>)
    set_property(TEST llama-agent-daemon-mcp-config-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-config-discovery-ctest
        COMMAND $<TARGET_FILE:llama-agent-config-discovery-smoke>)
    set_property(TEST llama-agent-config-discovery-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-resource-store-ctest
        COMMAND $<TARGET_FILE:llama-agent-resource-store-smoke>)
    set_property(TEST llama-agent-resource-store-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-resource-chunker-ctest
        COMMAND $<TARGET_FILE:llama-agent-resource-chunker-smoke>)
    set_property(TEST llama-agent-resource-chunker-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-resource-range-chunker-ctest
        COMMAND $<TARGET_FILE:llama-agent-resource-range-chunker-smoke>)
    set_property(TEST llama-agent-resource-range-chunker-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-resource-processing-contract-ctest
        COMMAND $<TARGET_FILE:llama-agent-resource-processing-contract-smoke>)
    set_property(TEST llama-agent-resource-processing-contract-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-resource-backend-ctest
        COMMAND $<TARGET_FILE:llama-agent-resource-backend-smoke>)
    set_property(TEST llama-agent-resource-backend-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-resource-processor-dispatch-ctest
        COMMAND $<TARGET_FILE:llama-agent-resource-processor-dispatch-smoke>)
    set_property(TEST llama-agent-resource-processor-dispatch-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-resource-media-type-ctest
        COMMAND $<TARGET_FILE:llama-agent-resource-media-type-smoke>)
    set_property(TEST llama-agent-resource-media-type-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-resource-processing-service-ctest
        COMMAND $<TARGET_FILE:llama-agent-resource-processing-service-smoke>)
    set_property(TEST llama-agent-resource-processing-service-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-pdf-text-processor-ctest
        COMMAND $<TARGET_FILE:llama-agent-pdf-text-processor-smoke>)
    set_property(TEST llama-agent-pdf-text-processor-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-docx-text-processor-ctest
        COMMAND $<TARGET_FILE:llama-agent-docx-text-processor-smoke>)
    set_property(TEST llama-agent-docx-text-processor-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-dataset-contract-ctest
        COMMAND $<TARGET_FILE:llama-agent-dataset-contract-smoke>)
    set_property(TEST llama-agent-dataset-contract-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-dataset-importer-ctest
        COMMAND $<TARGET_FILE:llama-agent-dataset-importer-smoke>)
    set_property(TEST llama-agent-dataset-importer-ctest PROPERTY LABELS agent)

    if (LLAMA_MEMORY_COZO)
        add_test(NAME llama-agent-data-manipulation-ctest
            COMMAND $<TARGET_FILE:llama-agent-data-manipulation-smoke>)
        set_property(TEST llama-agent-data-manipulation-ctest PROPERTY LABELS agent)
    endif()


    add_test(NAME llama-agent-pdf-page-image-command-ctest
        COMMAND $<TARGET_FILE:llama-agent-pdf-page-image-command-smoke>)
    set_property(TEST llama-agent-pdf-page-image-command-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-tesseract-ocr-command-ctest
        COMMAND $<TARGET_FILE:llama-agent-tesseract-ocr-command-smoke>)
    set_property(TEST llama-agent-tesseract-ocr-command-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-tesseract-ocr-processor-ctest
        COMMAND $<TARGET_FILE:llama-agent-tesseract-ocr-processor-smoke>)
    set_property(TEST llama-agent-tesseract-ocr-processor-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-resource-processing-host-ctest
        COMMAND $<TARGET_FILE:llama-agent-resource-processing-host-smoke>)
    set_property(TEST llama-agent-resource-processing-host-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-pdf-page-image-processor-ctest
        COMMAND $<TARGET_FILE:llama-agent-pdf-page-image-processor-smoke>)
    set_property(TEST llama-agent-pdf-page-image-processor-ctest PROPERTY LABELS agent)

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
    test-agent-continuation
    test-agent-research-contract
    test-agent-clangd-protocol
    test-agent-config-bootstrap
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
    test-tool-navigation
    test-agent-data-store-cozo
)
set(AGENT_TEST_PACK_AVAILABLE_TARGETS)
foreach(TARGET ${AGENT_TEST_PACK_CTEST_TARGETS})
    if (TARGET ${TARGET})
        list(APPEND AGENT_TEST_PACK_AVAILABLE_TARGETS ${TARGET})
    endif()
endforeach()

add_custom_target(llama-agent-build-pack
    DEPENDS ${AGENT_RUNTIME_TARGETS}
)

if (BUILD_TESTING)
    add_dependencies(llama-agent-build-pack ${AGENT_TEST_BUILD_TARGETS} ${AGENT_TEST_PACK_AVAILABLE_TARGETS})
endif()

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
