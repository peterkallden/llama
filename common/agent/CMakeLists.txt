set(TARGET llama-agent-core)

# Transport-neutral JSONL/SSE framing. Keep this small so daemon, web and
# mobile clients can share the wire contract without linking agent semantics.
add_library(llama-agent-protocol STATIC
    protocol/agent-jsonl.cpp protocol/agent-jsonl.h
    protocol/agent-sse.cpp protocol/agent-sse.h
)
target_include_directories(llama-agent-protocol PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_features(llama-agent-protocol PUBLIC cxx_std_17)
target_link_libraries(llama-agent-protocol PUBLIC llama-common)
set_target_properties(llama-agent-protocol PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (LLAMA_AGENT_SANDBOX)
    add_library(llama-agent-sandbox
        sandbox/sandbox-docker-runtime.cpp
        sandbox/sandbox-docker-runtime.h
        sandbox/sandbox-kubernetes-runtime.cpp
        sandbox/sandbox-kubernetes-runtime.h
        sandbox/sandbox-lxc-runtime.cpp
        sandbox/sandbox-lxc-runtime.h
        sandbox/sandbox-local-runtime.cpp
        sandbox/sandbox-local-runtime.h
    )
    target_include_directories(llama-agent-sandbox PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
    target_compile_features(llama-agent-sandbox PUBLIC cxx_std_17)
    target_link_libraries(llama-agent-sandbox PUBLIC llama-common PRIVATE cpp-httplib)
    if (WIN32)
        target_compile_definitions(llama-agent-sandbox PRIVATE LLAMA_AGENT_WEB_USE_WINHTTP=1)
        target_link_libraries(llama-agent-sandbox PRIVATE winhttp)
    endif()
    if (BUILD_SHARED_LIBS)
        set_target_properties(llama-agent-sandbox PROPERTIES POSITION_INDEPENDENT_CODE ON)
        set_target_properties(llama-agent-sandbox PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()

add_library(llama-agent-dataset-contracts dataset-contracts.cpp dataset-contracts.h)
target_include_directories(llama-agent-dataset-contracts PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_features(llama-agent-dataset-contracts PUBLIC cxx_std_17)
target_link_libraries(llama-agent-dataset-contracts PUBLIC llama-common)
if (BUILD_SHARED_LIBS)
    set_target_properties(llama-agent-dataset-contracts PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
endif()

add_library(${TARGET}
    agent-bootstrap.cpp agent-bootstrap.h
    agent-package-json.cpp agent-package-json.h
    agent-prepared-generation.cpp agent-prepared-generation.h
    credentials/agent-credential-provider.h
    learning/blueprint-selector.cpp learning/blueprint-selector.h
    tooling/catalog/model-projection.cpp tooling/catalog/model-projection.h
    thinking/deliberation-policy.cpp thinking/deliberation-policy.h
    agent-runtime-context.h
    thinking/phases/agent-runtime-reflection-phase.cpp
    thinking/phases/agent-runtime-research-phase.cpp
    agent-runtime.cpp agent-runtime.h
    runtime-json-contracts.cpp runtime-json-contracts.h
    learning/memory-learning.cpp learning/memory-learning.h
    adaptation/learning-observation.cpp adaptation/learning-observation.h
    adaptation/adaptation-observer.h
    adaptation/learning-transaction.cpp adaptation/learning-transaction.h
    adaptation/training-candidate.cpp adaptation/training-candidate.h
    adaptation/corpus-builder.cpp adaptation/corpus-builder.h
    adaptation/trainer-protocol.cpp adaptation/trainer-protocol.h
    thinking/reflection-json.cpp thinking/reflection-json.h
    tooling/contracts/schema-contract.cpp tooling/contracts/schema-contract.h
    tooling/adapters/tool-adapters.cpp tooling/adapters/tool-adapters.h
    tooling/adapters/families/memory-adapters.cpp tooling/adapters/families/memory-adapters.h
    tooling/adapters/families/data-adapters.cpp tooling/adapters/families/data-adapters.h
    tooling/adapters/families/document-adapters.cpp tooling/adapters/families/document-adapters.h
    tooling/adapters/families/repository-adapters.cpp tooling/adapters/families/repository-adapters.h
    tooling/adapters/families/diagnostics-adapters.cpp tooling/adapters/families/diagnostics-adapters.h
    tooling/adapters/families/resource-adapters.cpp tooling/adapters/families/resource-adapters.h
    tooling/adapters/support/adapter-support.h
    tooling/catalog/tool-catalog.cpp tooling/catalog/tool-catalog.h
    tooling/bridge/tool-chat-bridge.cpp tooling/bridge/tool-chat-bridge.h
    tooling/registry/tool-registry.cpp tooling/registry/tool-registry.h
    tooling/contracts/tool-result-contracts.cpp tooling/contracts/tool-result-contracts.h
    tooling/schema/tool-schema-compact.cpp tooling/schema/tool-schema-compact.h
    tooling/routing/tool-navigation.cpp tooling/routing/tool-navigation.h
    thinking/research/research-contract.h
    thinking/research/research-workspace.cpp thinking/research/research-workspace.h
    thinking/research/research-workspace-factory.cpp thinking/research/research-workspace-factory.h
    thinking/research/research-controller.cpp thinking/research/research-controller.h
    thinking/research/research-runner.cpp thinking/research/research-runner.h)

# Tool registration, schema projection and native adapter execution form a
# stable agent-only seam. Keep them out of the core runtime library so changes
# to a tool family do not relink the planning/thinking implementation.
set(AGENT_TOOLING_SOURCES
    tooling/catalog/model-projection.cpp
    tooling/contracts/schema-contract.cpp
    tooling/adapters/tool-adapters.cpp
    tooling/adapters/families/memory-adapters.cpp
    tooling/adapters/families/data-adapters.cpp
    tooling/adapters/families/document-adapters.cpp
    tooling/adapters/families/repository-adapters.cpp
    tooling/adapters/families/diagnostics-adapters.cpp
    tooling/adapters/families/resource-adapters.cpp
    tooling/catalog/tool-catalog.cpp
    tooling/bridge/tool-chat-bridge.cpp
    tooling/registry/tool-registry.cpp
    tooling/contracts/tool-result-contracts.cpp
    tooling/schema/tool-schema-compact.cpp
    tool-family-index.cpp
    tool-workflow-index.cpp
    tooling/routing/tool-navigation.cpp
)
get_target_property(AGENT_CORE_SOURCES ${TARGET} SOURCES)
list(REMOVE_ITEM AGENT_CORE_SOURCES ${AGENT_TOOLING_SOURCES})
set_property(TARGET ${TARGET} PROPERTY SOURCES ${AGENT_CORE_SOURCES})

add_library(llama-agent-tooling ${AGENT_TOOLING_SOURCES})
target_include_directories(llama-agent-tooling PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_features(llama-agent-tooling PUBLIC cxx_std_17)
target_link_libraries(llama-agent-tooling
    PUBLIC llama-common llama-agent-dataset-contracts llama-agent-plan llama-agent-memory
    PRIVATE cpp-httplib
)
if (LLAMA_AGENT_SANDBOX)
    target_link_libraries(llama-agent-tooling PUBLIC llama-agent-sandbox)
endif()
if (LLAMA_AGENT_TOOLS_CLANG)
    target_compile_definitions(llama-agent-tooling PRIVATE LLAMA_AGENT_TOOLS_USE_CLANG=1)
endif()
if (WIN32)
    target_compile_definitions(llama-agent-tooling PRIVATE LLAMA_AGENT_WEB_USE_WINHTTP=1)
    target_link_libraries(llama-agent-tooling PRIVATE winhttp)
endif()
if (BUILD_SHARED_LIBS)
    set_target_properties(llama-agent-tooling PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(llama-agent-tooling PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

add_library(llama-agent-poc ALIAS ${TARGET})
target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_features(${TARGET} PUBLIC cxx_std_17)
target_link_libraries(${TARGET}
    PUBLIC
        llama-agent-dataset-contracts
        llama-agent-plan
        llama-agent-memory
        llama-agent-tooling
        llama-agent-protocol
)
target_link_libraries(${TARGET} PRIVATE cpp-httplib)
if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

if (LLAMA_MEMORY_COZO)
    add_library(llama-agent-storage-cozo
        ${PROJECT_SOURCE_DIR}/tools/agent/data/agent-data-store-cozo.cpp
        ${PROJECT_SOURCE_DIR}/tools/agent/data/agent-data-store-cozo-schema.cpp
        ${PROJECT_SOURCE_DIR}/tools/agent/data/agent-data-store-cozo-query.cpp
        ${PROJECT_SOURCE_DIR}/tools/agent/data/agent-data-store-cozo-aggregate.cpp
        ${PROJECT_SOURCE_DIR}/tools/agent/data/agent-data-store-cozo-join.cpp
        ${PROJECT_SOURCE_DIR}/tools/agent/data/agent-data-store-cozo-reader.cpp
        ${PROJECT_SOURCE_DIR}/tools/agent/data/agent-data-store-cozo-encoding.cpp
    )
    target_include_directories(llama-agent-storage-cozo
        PUBLIC
            ${PROJECT_SOURCE_DIR}/common
            ${PROJECT_SOURCE_DIR}/tools/agent/data
        PRIVATE ${COZO_INCLUDE_DIR}
    )
    target_compile_features(llama-agent-storage-cozo PUBLIC cxx_std_17)
    target_link_libraries(llama-agent-storage-cozo
        PUBLIC llama-agent-core
        PRIVATE Threads::Threads
    )
    llama_link_cozo(llama-agent-storage-cozo)
    if (BUILD_SHARED_LIBS)
        set_target_properties(llama-agent-storage-cozo PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            WINDOWS_EXPORT_ALL_SYMBOLS ON
        )
    endif()
endif()
