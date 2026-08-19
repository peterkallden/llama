set(TARGET llama-agent-core)
add_library(${TARGET} agent-bootstrap.cpp agent-bootstrap.h agent-package-json.cpp agent-package-json.h agent-prepared-generation.cpp agent-prepared-generation.h learning/blueprint-selector.cpp learning/blueprint-selector.h tooling/catalog/model-projection.cpp tooling/catalog/model-projection.h dataset-contracts.cpp dataset-contracts.h thinking/deliberation-policy.cpp thinking/deliberation-policy.h agent-runtime-context.h thinking/phases/agent-runtime-reflection-phase.cpp thinking/phases/agent-runtime-research-phase.cpp agent-runtime.cpp agent-runtime.h runtime-json-contracts.cpp runtime-json-contracts.h learning/memory-learning.cpp learning/memory-learning.h thinking/reflection-json.cpp thinking/reflection-json.h tooling/contracts/schema-contract.cpp tooling/contracts/schema-contract.h sandbox/sandbox-docker-runtime.cpp sandbox/sandbox-docker-runtime.h sandbox/sandbox-kubernetes-runtime.cpp sandbox/sandbox-kubernetes-runtime.h tooling/adapters/tool-adapters.cpp tooling/adapters/tool-adapters.h tooling/adapters/families/memory-adapters.cpp tooling/adapters/families/memory-adapters.h tooling/adapters/families/data-adapters.cpp tooling/adapters/families/data-adapters.h tooling/adapters/families/document-adapters.cpp tooling/adapters/families/document-adapters.h tooling/adapters/families/repository-adapters.cpp tooling/adapters/families/repository-adapters.h tooling/adapters/support/adapter-support.h tooling/catalog/tool-catalog.cpp tooling/catalog/tool-catalog.h tooling/bridge/tool-chat-bridge.cpp tooling/bridge/tool-chat-bridge.h tooling/registry/tool-registry.cpp tooling/registry/tool-registry.h tooling/contracts/tool-result-contracts.cpp tooling/contracts/tool-result-contracts.h tooling/schema/tool-schema-compact.cpp tooling/schema/tool-schema-compact.h tooling/routing/tool-navigation.cpp tooling/routing/tool-navigation.h thinking/research/research-contract.h thinking/research/research-workspace.cpp thinking/research/research-workspace.h thinking/research/research-workspace-factory.cpp thinking/research/research-workspace-factory.h thinking/research/research-controller.cpp thinking/research/research-controller.h thinking/research/research-runner.cpp thinking/research/research-runner.h)
add_library(llama-agent-poc ALIAS ${TARGET})
target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_features(${TARGET} PUBLIC cxx_std_17)
target_link_libraries(${TARGET} PUBLIC llama-agent-plan llama-agent-memory)
target_link_libraries(${TARGET} PRIVATE cpp-httplib)
if (LLAMA_AGENT_TOOLS_CLANG)
    target_compile_definitions(${TARGET} PRIVATE LLAMA_AGENT_TOOLS_USE_CLANG=1)
endif()
if (WIN32)
    target_compile_definitions(${TARGET} PRIVATE LLAMA_AGENT_WEB_USE_WINHTTP=1)
    target_link_libraries(${TARGET} PRIVATE winhttp)
endif()
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
