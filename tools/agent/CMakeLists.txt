# Production agent targets. Smoke and CTest definitions remain in pocs/agent.
# This file is buildable with LLAMA_AGENT_BUILD_POCS=OFF.

find_package(Threads REQUIRED)

include_directories(${PROJECT_SOURCE_DIR})
if (LLAMA_AGENT_TOOLS_CLANG)
    add_compile_definitions(LLAMA_AGENT_TOOLS_USE_CLANG=1)
endif()

set(AGENT_RUNTIME_SOURCES
    ${PROJECT_SOURCE_DIR}/tools/agent/cli/agent-cli-config.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/cli/agent-cli-generation-utils.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/cli/agent-cli-run-adapter.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon/agent-daemon-client-admin.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon/agent-daemon-client-status.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon/agent-daemon-client.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon/agent-daemon-event-collector.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/host/agent-host-config.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/openapi/agent-openapi-catalog.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/openapi/agent-openapi-provider.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/openapi/agent-openapi-http.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/openapi/agent-openapi-flydelta-oracle.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/host/agent-sandbox-assembly.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/mcp/agent-mcp-http-client.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/mcp/agent-mcp-http-transport.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/mcp/agent-mcp-client-factory.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/tooling/agent-selection-contracts.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/mcp/agent-mcp-stdio-client.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/tooling/agent-tool-provider.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/tooling/agent-tool-result-json-contracts.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/tooling/agent-tool-runtime-adapter.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/agent-resource-catalog.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/agent-resource-store.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/agent-resource-chunker.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/agent-resource-media-type.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors/agent-pdf-text-processor.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors/agent-pandoc-command.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors/agent-docx-text-processor.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors/agent-xlsx-workbook-json-processor.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors/agent-pdf-page-image-command.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors/agent-pdf-page-image-processor.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors/agent-tesseract-ocr-command.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors/agent-tesseract-ocr-processor.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/agent-resource-processing-service.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/assembly/agent-resource-processor-factory.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/agent-resource-processing-cache.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/data/agent-data-store-factory.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/data/agent-dataset-importer.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/diagnostics/agent-clangd-protocol.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/diagnostics/agent-clangd-session.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/diagnostics/agent-clangd-provider.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/diagnostics/agent-native-crash.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon/agent-daemon-adapter.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon/agent-daemon-protocol.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon/agent-daemon-runtime.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon/agent-daemon-flydelta.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon/flydelta/concept-capture.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon/flydelta/concept-synthesis.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon/flydelta/decision-margin.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon/flydelta/provider.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon/flydelta/binding.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-flydelta-server-host.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/adaptation/flydelta/resource-adapter.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/adaptation/flydelta/resource-scope.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/adaptation/flydelta/resource-json.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/adaptation/flydelta/augmentation-material.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/adaptation/flydelta/teaching-material.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/adaptation/flydelta/capture-material.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon/agent-daemon-dispatcher.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon/agent-daemon-service.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/cli/agent-cli-host-adapter.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/cli/agent-cli-inference.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-plan-orchestration.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-route-compiler.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-runtime-host.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-runtime-resident.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-runtime-session-host.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-runtime-event-projector.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-runtime-turn-driver.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-inference-executor.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-inference-capacity-gate.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-runtime-session-manager.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-runtime-execution.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-runtime-session.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-server-context-host.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-server-generation.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-server-inference.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-runtime-assembly.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/cli/agent-cli-runtime.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/cli/agent-cli-options.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/cli/agent-cli-generation.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/cli/agent-cli-memory-tools.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/cli/agent-cli-selection.cpp
)

if (LLAMA_MEMORY_COZO)
    list(APPEND AGENT_RUNTIME_SOURCES
        ${PROJECT_SOURCE_DIR}/tools/agent/resource/agent-resource-catalog-cozo.cpp
    )
endif()

# Runtime execution is a separate build seam from host assembly. Keep the
# session/turn/inference implementation together so changes there do not
# recompile CLI, daemon and MCP host code unnecessarily.
set(AGENT_RUNTIME_ENGINE_SOURCES
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-runtime-event-projector.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-inference-capacity-gate.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-runtime-session.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-model-residency.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-model-loaders.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-server-context-host.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-server-generation.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-server-inference.cpp
)

list(REMOVE_ITEM AGENT_RUNTIME_SOURCES ${AGENT_RUNTIME_ENGINE_SOURCES})

# The host facade is the seam above the execution engine. Keep runtime
# execution, orchestration and the CLI-facing assembly glue here as one
# dependency direction; these pieces compose the engine and CLI providers and
# must not be split back across support in static builds.
set(AGENT_RUNTIME_HOST_SOURCES
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-runtime-chat-driver.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-runtime-host.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-runtime-resident.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-runtime-session-host.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-runtime-turn-driver.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-inference-executor.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-runtime-session-manager.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-runtime-execution.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-runtime-assembly.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-dataset-workflow-adapter.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-plan-orchestration.cpp
)

list(REMOVE_ITEM AGENT_RUNTIME_SOURCES ${AGENT_RUNTIME_HOST_SOURCES})

# Model-backed implementations of the agent interfaces.  These are runtime
# adapters, not CLI frontend code; resident, daemon and MCP hosts may use them
# without depending on command parsing or process entrypoints.
set(AGENT_RUNTIME_ADAPTER_SOURCES
    ${PROJECT_SOURCE_DIR}/tools/agent/cli/agent-cli-generation-utils.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/cli/agent-cli-generation.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/cli/agent-cli-inference.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/cli/agent-cli-runtime.cpp
)

list(REMOVE_ITEM AGENT_RUNTIME_SOURCES ${AGENT_RUNTIME_ADAPTER_SOURCES})

set(AGENT_RUNTIME_SELECTION_SOURCES
    ${PROJECT_SOURCE_DIR}/tools/agent/cli/agent-cli-selection.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/tooling/agent-selection-contracts.cpp
)

set(AGENT_RUNTIME_PACKAGE_SOURCES
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-runtime-package-io.cpp
)

list(REMOVE_ITEM AGENT_RUNTIME_SOURCES
    ${AGENT_RUNTIME_SELECTION_SOURCES}
    ${AGENT_RUNTIME_PACKAGE_SOURCES}
)

set(AGENT_TOOL_PROVIDER_SOURCES
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-route-compiler.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/tooling/agent-tool-provider.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/tooling/agent-tool-result-json-contracts.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/tooling/agent-tool-runtime-adapter.cpp
)

list(REMOVE_ITEM AGENT_RUNTIME_SOURCES ${AGENT_TOOL_PROVIDER_SOURCES})

set(AGENT_CLI_SOURCES
    ${PROJECT_SOURCE_DIR}/tools/agent/cli/agent-cli-config.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/cli/agent-cli-generation-utils.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/cli/agent-cli-inference.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/cli/agent-cli-memory-tools.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/cli/agent-cli-options.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/cli/agent-cli-generation.cpp
)

list(REMOVE_ITEM AGENT_RUNTIME_SOURCES ${AGENT_CLI_SOURCES})
list(REMOVE_ITEM AGENT_CLI_SOURCES ${AGENT_RUNTIME_ADAPTER_SOURCES})

set(AGENT_DAEMON_SOURCES
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon/agent-daemon-client-admin.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon/agent-daemon-client-status.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon/agent-daemon-client.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon/agent-daemon-event-collector.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon/agent-daemon-adapter.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon/agent-daemon-protocol.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon/agent-daemon-runtime.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon/agent-daemon-flydelta.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon/flydelta/concept-capture.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon/flydelta/concept-synthesis.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon/flydelta/decision-margin.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon/flydelta/provider.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon/flydelta/binding.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime/agent-flydelta-server-host.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/adaptation/flydelta/resource-adapter.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/adaptation/flydelta/resource-scope.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/adaptation/flydelta/resource-json.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/adaptation/flydelta/augmentation-material.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/adaptation/flydelta/teaching-material.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/adaptation/flydelta/capture-material.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon/agent-daemon-dispatcher.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon/agent-daemon-service.cpp
)

list(REMOVE_ITEM AGENT_RUNTIME_SOURCES ${AGENT_DAEMON_SOURCES})

set(AGENT_MCP_CLIENT_SOURCES
    ${PROJECT_SOURCE_DIR}/tools/agent/mcp/agent-mcp-http-client.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/mcp/agent-mcp-http-transport.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/mcp/agent-mcp-stdio-client.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/mcp/agent-mcp-client-factory.cpp
)

list(REMOVE_ITEM AGENT_RUNTIME_SOURCES ${AGENT_MCP_CLIENT_SOURCES})

# Keep the host/runtime assembly independent from the high-churn resource,
# data and diagnostics implementations. These are agent-only libraries; the
# upstream llama.cpp targets remain untouched.
list(REMOVE_ITEM AGENT_RUNTIME_SOURCES
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/agent-resource-catalog.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/agent-resource-chunker.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/agent-resource-media-type.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/agent-resource-store.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/agent-resource-processing-service.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/agent-resource-processing-cache.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/assembly/agent-resource-processor-factory.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors/agent-pdf-text-processor.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors/agent-pandoc-command.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors/agent-docx-text-processor.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors/agent-xlsx-workbook-json-processor.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors/agent-pdf-page-image-command.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors/agent-pdf-page-image-processor.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors/agent-tesseract-ocr-command.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors/agent-tesseract-ocr-processor.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/data/agent-data-store-factory.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/data/agent-dataset-importer.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/diagnostics/agent-clangd-protocol.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/diagnostics/agent-clangd-session.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/diagnostics/agent-clangd-provider.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/agent-resource-catalog-cozo.cpp
)

add_library(llama-agent-resource
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/agent-resource-catalog.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/agent-resource-chunker.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/agent-resource-media-type.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/agent-resource-store.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/agent-resource-processing-service.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/agent-resource-processing-cache.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/assembly/agent-resource-processor-factory.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors/agent-pdf-text-processor.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors/agent-pandoc-command.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors/agent-docx-text-processor.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors/agent-xlsx-workbook-json-processor.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors/agent-pdf-page-image-command.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors/agent-pdf-page-image-processor.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors/agent-tesseract-ocr-command.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors/agent-tesseract-ocr-processor.cpp
)
if (LLAMA_MEMORY_COZO)
    target_sources(llama-agent-resource PRIVATE
        ${PROJECT_SOURCE_DIR}/tools/agent/resource/agent-resource-catalog-cozo.cpp
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
    ${PROJECT_SOURCE_DIR}/tools/agent/data/agent-data-store-factory.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/data/agent-dataset-importer.cpp
)
if (LLAMA_AGENT_STORAGE_SQLITE)
    target_sources(llama-agent-data PRIVATE
        ${PROJECT_SOURCE_DIR}/tools/agent/data/agent-data-store-sqlite.cpp
        ${PROJECT_SOURCE_DIR}/tools/agent/data/agent-data-store-sqlite.h
    )
endif()
target_include_directories(llama-agent-data PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/data
)
target_compile_features(llama-agent-data PUBLIC cxx_std_17)
target_link_libraries(llama-agent-data PUBLIC llama-agent-core)
if (LLAMA_MEMORY_COZO)
    target_link_libraries(llama-agent-data PUBLIC llama-agent-storage-cozo)
endif()
if (LLAMA_AGENT_STORAGE_SQLITE)
    target_link_libraries(llama-agent-data PUBLIC llama-agent-storage-sqlite)
    target_compile_definitions(llama-agent-data PUBLIC LLAMA_AGENT_DATA_USE_SQLITE)
endif()
if (BUILD_SHARED_LIBS)
    set_target_properties(llama-agent-data PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
endif()

# Adaptation persistence keeps the semantic transaction contract in
# common/agent/adaptation while hosting the optional Cozo/SQLite adapters next
# to the other host-side stores. This prevents the common contract library
# from acquiring backend-specific dependencies.
add_library(llama-agent-adaptation-storage
    ${PROJECT_SOURCE_DIR}/tools/agent/adaptation/agent-learning-transaction-store-factory.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/adaptation/agent-learning-lifecycle-store-factory.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/adaptation/flydelta-teaching-material-store.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/adaptation/flydelta-teaching-material-store.h
)
if (LLAMA_MEMORY_COZO)
    target_sources(llama-agent-adaptation-storage PRIVATE
        ${PROJECT_SOURCE_DIR}/tools/agent/adaptation/agent-learning-transaction-store-cozo.cpp
        ${PROJECT_SOURCE_DIR}/tools/agent/adaptation/agent-learning-transaction-store-cozo.h
        ${PROJECT_SOURCE_DIR}/tools/agent/adaptation/agent-learning-lifecycle-store-cozo.cpp
        ${PROJECT_SOURCE_DIR}/tools/agent/adaptation/agent-learning-lifecycle-store-cozo.h
    )
endif()
if (LLAMA_AGENT_STORAGE_SQLITE)
    target_sources(llama-agent-adaptation-storage PRIVATE
        ${PROJECT_SOURCE_DIR}/tools/agent/adaptation/agent-learning-transaction-store-sqlite.cpp
        ${PROJECT_SOURCE_DIR}/tools/agent/adaptation/agent-learning-transaction-store-sqlite.h
        ${PROJECT_SOURCE_DIR}/tools/agent/adaptation/agent-learning-lifecycle-store-sqlite.cpp
        ${PROJECT_SOURCE_DIR}/tools/agent/adaptation/agent-learning-lifecycle-store-sqlite.h
    )
endif()
target_include_directories(llama-agent-adaptation-storage PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/adaptation
)
target_compile_features(llama-agent-adaptation-storage PUBLIC cxx_std_17)
target_link_libraries(llama-agent-adaptation-storage PUBLIC llama-agent-core llama-agent-flydelta)
if (LLAMA_MEMORY_COZO)
    target_compile_definitions(llama-agent-adaptation-storage PRIVATE LLAMA_AGENT_ADAPTATION_USE_COZO)
    target_include_directories(llama-agent-adaptation-storage PRIVATE ${COZO_INCLUDE_DIR})
    target_link_libraries(llama-agent-adaptation-storage PUBLIC llama-agent-storage-cozo)
    llama_link_cozo(llama-agent-adaptation-storage)
endif()
if (LLAMA_AGENT_STORAGE_SQLITE)
    target_compile_definitions(llama-agent-adaptation-storage PRIVATE LLAMA_AGENT_ADAPTATION_USE_SQLITE)
    target_link_libraries(llama-agent-adaptation-storage PUBLIC llama-agent-storage-sqlite)
endif()
if (BUILD_SHARED_LIBS)
    set_target_properties(llama-agent-adaptation-storage PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
endif()

add_library(llama-agent-diagnostics
    ${PROJECT_SOURCE_DIR}/tools/agent/diagnostics/agent-clangd-protocol.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/diagnostics/agent-clangd-session.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/diagnostics/agent-clangd-provider.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/diagnostics/agent-native-crash.cpp
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

# Provider and runtime-adapter implementation is shared by the host, CLI,
# daemon and MCP layers. Keep it below the common tool contracts and above
# runtime assembly so host libraries do not have to depend on one another.
add_library(llama-agent-tool-provider ${AGENT_TOOL_PROVIDER_SOURCES})
target_include_directories(llama-agent-tool-provider
    PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tools/agent/cli
        ${PROJECT_SOURCE_DIR}/tools/agent/daemon
        ${PROJECT_SOURCE_DIR}/tools/agent/mcp
        ${PROJECT_SOURCE_DIR}/tools/agent/runtime
        ${PROJECT_SOURCE_DIR}/tools/agent/tooling
        ${PROJECT_SOURCE_DIR}/tools/agent/resource
        ${PROJECT_SOURCE_DIR}/tools/agent/data
        ${PROJECT_SOURCE_DIR}/tools/server
)
target_compile_features(llama-agent-tool-provider PUBLIC cxx_std_17)
target_link_libraries(llama-agent-tool-provider
    PUBLIC
        llama-agent-core
        llama-agent-tooling
        llama-agent-memory
        llama-agent-plan
        llama-common
        Threads::Threads
)
if (BUILD_SHARED_LIBS)
    set_target_properties(llama-agent-tool-provider PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
endif()

add_library(llama-agent-cli-library ${AGENT_CLI_SOURCES})
set_target_properties(llama-agent-cli-library PROPERTIES OUTPUT_NAME llama-agent-cli)
target_include_directories(llama-agent-cli-library
    PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tools/agent/cli
        ${PROJECT_SOURCE_DIR}/tools/agent/runtime
        ${PROJECT_SOURCE_DIR}/tools/agent/tooling
        ${PROJECT_SOURCE_DIR}/tools/agent/resource
        ${PROJECT_SOURCE_DIR}/tools/agent/data
        ${PROJECT_SOURCE_DIR}/tools/server
)
target_compile_features(llama-agent-cli-library PUBLIC cxx_std_17)
target_compile_definitions(llama-agent-cli-library
    PUBLIC LLAMA_MEMORY_POC_USE_AGENT_TOOLS
)
target_link_libraries(llama-agent-cli-library
    PUBLIC
        llama-agent-core
        llama-agent-flydelta
        llama-agent-memory
        llama-agent-plan
        llama-common
        llama
        Threads::Threads
)
if (BUILD_SHARED_LIBS)
    set_target_properties(llama-agent-cli-library PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
endif()

add_library(llama-agent-runtime-adapters ${AGENT_RUNTIME_ADAPTER_SOURCES})
set_target_properties(llama-agent-runtime-adapters PROPERTIES OUTPUT_NAME llama-agent-runtime-adapters)
target_include_directories(llama-agent-runtime-adapters
    PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tools/agent/cli
        ${PROJECT_SOURCE_DIR}/tools/agent/runtime
        ${PROJECT_SOURCE_DIR}/tools/agent/tooling
        ${PROJECT_SOURCE_DIR}/tools/agent/resource
        ${PROJECT_SOURCE_DIR}/tools/agent/data
        ${PROJECT_SOURCE_DIR}/tools/server
)
target_compile_features(llama-agent-runtime-adapters PUBLIC cxx_std_17)
target_compile_definitions(llama-agent-runtime-adapters
    PUBLIC LLAMA_MEMORY_POC_USE_AGENT_TOOLS
)
target_link_libraries(llama-agent-runtime-adapters
    PUBLIC
        llama-agent-core
        llama-agent-flydelta
        llama-agent-tooling
        llama-agent-memory
        llama-agent-plan
        llama-common
        llama
        Threads::Threads
)
if (BUILD_SHARED_LIBS)
    set_target_properties(llama-agent-runtime-adapters PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
endif()

add_library(llama-agent-runtime-selection ${AGENT_RUNTIME_SELECTION_SOURCES})
set_target_properties(llama-agent-runtime-selection PROPERTIES OUTPUT_NAME llama-agent-runtime-selection)
target_include_directories(llama-agent-runtime-selection
    PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tools/agent/cli
        ${PROJECT_SOURCE_DIR}/tools/agent/runtime
        ${PROJECT_SOURCE_DIR}/tools/agent/tooling
)
target_compile_features(llama-agent-runtime-selection PUBLIC cxx_std_17)
target_link_libraries(llama-agent-runtime-selection
    PUBLIC
        llama-agent-runtime-adapters
        llama-agent-core
        llama-agent-tooling
        llama-agent-plan
        llama-common
        Threads::Threads
)
if (BUILD_SHARED_LIBS)
    set_target_properties(llama-agent-runtime-selection PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
endif()

add_library(llama-agent-runtime-packages ${AGENT_RUNTIME_PACKAGE_SOURCES})
set_target_properties(llama-agent-runtime-packages PROPERTIES OUTPUT_NAME llama-agent-runtime-packages)
target_include_directories(llama-agent-runtime-packages
    PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tools/agent/cli
        ${PROJECT_SOURCE_DIR}/tools/agent/runtime
)
target_compile_features(llama-agent-runtime-packages PUBLIC cxx_std_17)
target_link_libraries(llama-agent-runtime-packages
    PUBLIC
        llama-agent-core
        llama-agent-memory
        llama-agent-plan
        llama-common
        Threads::Threads
)
if (BUILD_SHARED_LIBS)
    set_target_properties(llama-agent-runtime-packages PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
endif()

add_library(llama-agent-runtime-engine ${AGENT_RUNTIME_ENGINE_SOURCES})
target_include_directories(llama-agent-runtime-engine
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
target_compile_features(llama-agent-runtime-engine PUBLIC cxx_std_17)
target_compile_definitions(llama-agent-runtime-engine
    PUBLIC LLAMA_MEMORY_POC_USE_AGENT_TOOLS
)
target_link_libraries(llama-agent-runtime-engine
    PUBLIC
        llama-agent-memory
        llama-common
        llama
        llama-agent-core
        llama-agent-flydelta
        llama-agent-resource
        llama-agent-data
        llama-agent-diagnostics
        llama-agent-runtime-adapters
        server-context
        vendor::hash
        Threads::Threads
    PRIVATE
        cpp-httplib
)
if (WIN32)
    target_link_libraries(llama-agent-runtime-engine PUBLIC ws2_32)
endif()
if (LLAMA_MEMORY_COZO)
    target_include_directories(llama-agent-runtime-engine PRIVATE ${COZO_INCLUDE_DIR})
    target_link_libraries(llama-agent-runtime-engine PUBLIC llama-agent-storage-cozo)
    llama_link_cozo(llama-agent-runtime-engine)
endif()
if (BUILD_SHARED_LIBS)
    set_target_properties(llama-agent-runtime-engine PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
endif()

add_library(llama-agent-runtime-host ${AGENT_RUNTIME_HOST_SOURCES})
target_include_directories(llama-agent-runtime-host
    PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tools/agent/runtime
        ${PROJECT_SOURCE_DIR}/tools/agent/host
        ${PROJECT_SOURCE_DIR}/tools/agent/resource
        ${PROJECT_SOURCE_DIR}/tools/agent/data
)
target_compile_features(llama-agent-runtime-host PUBLIC cxx_std_17)
target_link_libraries(llama-agent-runtime-host
    PUBLIC
        llama-agent-runtime-engine
        llama-agent-adaptation-storage
        llama-agent-flydelta
        llama-agent-memory
        llama-agent-plan
        llama-common
        llama-agent-core
        llama-agent-tool-provider
        llama-agent-cli-library
        llama-agent-runtime-adapters
        llama-agent-runtime-selection
        llama-agent-runtime-packages
        Threads::Threads
)
if (BUILD_SHARED_LIBS)
    set_target_properties(llama-agent-runtime-host PROPERTIES
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
if(LLAMA_AGENT_SANDBOX)
    target_compile_definitions(llama-agent-runtime-support PUBLIC LLAMA_AGENT_SANDBOX=1)
else()
    target_compile_definitions(llama-agent-runtime-support PUBLIC LLAMA_AGENT_SANDBOX=0)
endif()
target_link_libraries(llama-agent-runtime-support
    PUBLIC
        llama-agent-memory
        llama-common
        llama
        llama-agent-core
        llama-agent-flydelta
        llama-agent-resource
        llama-agent-data
        llama-agent-diagnostics
        llama-agent-runtime-engine
        llama-agent-runtime-host
        llama-agent-tool-provider
        llama-agent-cli-library
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

add_library(llama-agent-daemon-library ${AGENT_DAEMON_SOURCES})
set_target_properties(llama-agent-daemon-library PROPERTIES OUTPUT_NAME llama-agent-daemon)
target_include_directories(llama-agent-daemon-library
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
        ${PROJECT_SOURCE_DIR}/tools/server
)
target_compile_features(llama-agent-daemon-library PUBLIC cxx_std_17)
target_compile_definitions(llama-agent-daemon-library
    PUBLIC LLAMA_MEMORY_POC_USE_AGENT_TOOLS
        LLAMA_AGENT_SYSCONFDIR="${LLAMA_AGENT_SYSCONFDIR}"
)
target_link_libraries(llama-agent-daemon-library
    PUBLIC
        llama-agent-runtime-support
        llama-agent-flydelta
        Threads::Threads
)
if (BUILD_SHARED_LIBS)
    set_target_properties(llama-agent-daemon-library PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
endif()

# Shared MCP contracts and task lifecycle implementation. Keep this separate
# from the runtime and server assemblies so these sources are compiled once
# and reused by both sides.
add_library(llama-agent-mcp-core
    ${PROJECT_SOURCE_DIR}/tools/agent/mcp/agent-mcp-protocol.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/mcp/agent-mcp-task.cpp
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

add_library(llama-agent-mcp-client-library ${AGENT_MCP_CLIENT_SOURCES})
set_target_properties(llama-agent-mcp-client-library PROPERTIES OUTPUT_NAME llama-agent-mcp-client)
target_include_directories(llama-agent-mcp-client-library
    PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tools/agent/mcp
        ${PROJECT_SOURCE_DIR}/tools/agent/tooling
        ${PROJECT_SOURCE_DIR}/tools/agent/daemon
        ${PROJECT_SOURCE_DIR}/tools/server
)
target_compile_features(llama-agent-mcp-client-library PUBLIC cxx_std_17)
target_link_libraries(llama-agent-mcp-client-library
    PUBLIC
        llama-agent-mcp-core
        llama-agent-tool-provider
        llama-common
        Threads::Threads
    PRIVATE
        cpp-httplib
)
if (BUILD_SHARED_LIBS)
    set_target_properties(llama-agent-mcp-client-library PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
endif()
target_link_libraries(llama-agent-runtime-support PUBLIC llama-agent-mcp-client-library)

add_library(llama-agent-daemon-protocol STATIC
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon/agent-daemon-jsonl-protocol.cpp
)
target_include_directories(llama-agent-daemon-protocol
    PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tools/agent/daemon
        ${PROJECT_SOURCE_DIR}/vendor
)
target_compile_features(llama-agent-daemon-protocol PUBLIC cxx_std_17)
target_link_libraries(llama-agent-daemon-protocol PUBLIC llama-common Threads::Threads)
target_link_libraries(llama-agent-daemon-protocol PUBLIC llama-agent-core)
if (BUILD_SHARED_LIBS)
    set_target_properties(llama-agent-daemon-protocol PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
endif()

target_link_libraries(llama-agent-runtime-support PUBLIC llama-agent-daemon-protocol)

set(AGENT_MCP_SERVER_SOURCES
    ${PROJECT_SOURCE_DIR}/common/agent/tooling/contracts/schema-contract.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/mcp/agent-mcp-server-protocol.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/mcp/agent-mcp-server-tool-registry.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/mcp/agent-mcp-http-server.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/mcp/agent-mcp-agent-tools.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/mcp/agent-mcp-auth.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/mcp/agent-mcp-stdio-server.cpp
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
        llama-agent-dataset-contracts
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
        llama-agent-dataset-contracts
        llama-agent-memory
        llama-agent-plan
        llama-agent-tooling
        llama-agent-core
        llama-agent-flydelta
        llama-agent-mcp-core
        llama-agent-mcp-server
        llama-agent-tool-provider
        llama-agent-cli-library
        llama-agent-daemon-library
        llama-agent-mcp-client-library
        llama-agent-runtime-engine
        llama-agent-runtime-host
        llama-agent-runtime-adapters
        llama-agent-runtime-selection
        llama-agent-runtime-packages
        llama-agent-runtime-support
        llama-agent-resource
        llama-agent-data
        llama-agent-diagnostics
        llama-agent-adaptation-storage
    )
    if(TARGET llama-agent-sandbox)
        list(APPEND AGENT_PACKAGE_LIBRARY_TARGETS llama-agent-sandbox)
    endif()
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
    ${PROJECT_SOURCE_DIR}/tools/agent/cli/agent-cli-main.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/cli/agent-cli-command.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/cli/agent-cli-run.cpp
)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME llama-agent)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/pocs/memory
    ${PROJECT_SOURCE_DIR}/tools/agent/cli
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime
    ${PROJECT_SOURCE_DIR}/tools/agent/tooling
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support llama-agent-daemon-library)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-mcp-stdio-server)
add_executable(${TARGET}
    ${PROJECT_SOURCE_DIR}/tools/agent/mcp/agent-mcp-stdio-server-main.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/pocs/memory
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support cpp-httplib)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
set(TARGET llama-agent-daemon)
add_executable(${TARGET}
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon/agent-daemon-main.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon/agent-daemon-tcp.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon/agent-daemon-unix.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/pocs/memory
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support llama-agent-daemon-library cpp-httplib)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# The adaptation worker is deliberately a separate process.  It consumes
# immutable job bundles and never shares an inference context or session lane
# with the resident agent.
set(TARGET llama-agent-adaptation-worker)
add_executable(${TARGET}
    ${PROJECT_SOURCE_DIR}/tools/agent/adaptation/agent-learning-worker-main.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/adaptation/agent-learning-worker.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/adaptation
)
target_link_libraries(${TARGET} PRIVATE llama-agent-core vendor::hash)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-adaptation-admin)
add_executable(${TARGET}
    ${PROJECT_SOURCE_DIR}/tools/agent/adaptation/agent-learning-admin-main.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/adaptation
)
target_link_libraries(${TARGET} PRIVATE llama-agent-adaptation-storage)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-web)
add_executable(${TARGET}
    ${PROJECT_SOURCE_DIR}/tools/agent/web/agent-web-main.cpp
    ${PROJECT_SOURCE_DIR}/tools/agent/web/agent-web-server.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE llama-agent-protocol cpp-httplib Threads::Threads)
if(LLAMA_TOOLS_INSTALL OR LLAMA_AGENT_PACKAGE_INSTALL)
    install(TARGETS llama-agent-cli llama-agent-mcp-stdio-server llama-agent-daemon llama-agent-adaptation-worker llama-agent-adaptation-admin llama-agent-web RUNTIME)
    install(PROGRAMS ${PROJECT_SOURCE_DIR}/scripts/agent-adaptation-qlora-trainer.py DESTINATION bin)
    # The external QLoRA helper invokes this converter from the installed
    # package. Keep the helper, converter, and dependency manifests available
    # so a package does not require a source checkout at runtime.
    install(PROGRAMS ${PROJECT_SOURCE_DIR}/convert_lora_to_gguf.py
        DESTINATION share/llama-agent/converter)
    # The converter is a source-tree Python entrypoint. Keep its local
    # modules and model templates beside it so the installed helper does not
    # depend on a checkout of llama.cpp at runtime.
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/gguf-py DESTINATION share/llama-agent/converter
        PATTERN __pycache__ EXCLUDE)
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/conversion DESTINATION share/llama-agent/converter
        PATTERN __pycache__ EXCLUDE)
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/models/templates DESTINATION share/llama-agent/converter/models
        PATTERN __pycache__ EXCLUDE)
    install(FILES
        ${PROJECT_SOURCE_DIR}/requirements/requirements-agent-adaptation-qlora-trainer.txt
        ${PROJECT_SOURCE_DIR}/requirements/requirements-convert_hf_to_gguf.txt
        ${PROJECT_SOURCE_DIR}/requirements/requirements-convert_legacy_llama.txt
        ${PROJECT_SOURCE_DIR}/requirements/requirements-convert_lora_to_gguf.txt
        DESTINATION share/llama-agent/requirements)
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
        install(PROGRAMS
            ${PROJECT_SOURCE_DIR}/scripts/agent-provider-bootstrap.sh
            DESTINATION bin
            RENAME llama-agent-provider-bootstrap)
        install(FILES
            ${PROJECT_SOURCE_DIR}/scripts/agent-provider-bootstrap.py
            DESTINATION bin)
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
        ${PROJECT_SOURCE_DIR}/docs/examples/agent-host-config-stdio.json
        ${PROJECT_SOURCE_DIR}/docs/examples/agent-host-config-model-catalog.json
        ${PROJECT_SOURCE_DIR}/docs/examples/agent-host-config-eurostat.json
        ${PROJECT_SOURCE_DIR}/docs/examples/eurostat-statistics-openapi.json
        ${PROJECT_SOURCE_DIR}/docs/examples/agent-host-config-openalex.json
        ${PROJECT_SOURCE_DIR}/docs/examples/openalex-works-openapi.json
        ${PROJECT_SOURCE_DIR}/docs/examples/openapi-openalex-generated.json
        ${PROJECT_SOURCE_DIR}/docs/examples/agent-flydelta-dataset-question-suite.json
        ${PROJECT_SOURCE_DIR}/docs/examples/agent-flydelta-dataset-question-suite-incremental.json
        ${PROJECT_SOURCE_DIR}/docs/examples/agent-flydelta-dataset-question-suite-incremental-v2.json)
    install(FILES
        ${PROJECT_SOURCE_DIR}/docs/examples/agent-config.example.json
        ${PROJECT_SOURCE_DIR}/docs/examples/agent-bootstrap-workflows-v1.json
        ${AGENT_HOST_CONFIG_EXAMPLES}
        ${PROJECT_SOURCE_DIR}/docs/examples/agent-daemon-requests.jsonl
        ${PROJECT_SOURCE_DIR}/docs/examples/agent-runtime-context-budgets.json
        ${PROJECT_SOURCE_DIR}/docs/examples/llama-agent-daemon.service
        DESTINATION share/llama-agent/examples)

    # The Vue client is an example deployment companion for llama-agent-web.
    # Keep its source, Nginx example and package metadata with the other
    # installed examples, but do not ship generated frontend dependencies or
    # build output in the Debian package.
    install(DIRECTORY
        ${PROJECT_SOURCE_DIR}/examples/llama-agent.web/
        DESTINATION share/llama-agent/examples/llama-agent.web
        PATTERN node_modules EXCLUDE
        PATTERN dist EXCLUDE
        PATTERN .git EXCLUDE)
endif()

# Production targets need the shared MCP server implementation directly.
target_link_libraries(llama-agent-mcp-stdio-server PRIVATE llama-agent-mcp-server)
target_link_libraries(llama-agent-daemon PRIVATE llama-agent-mcp-server)
