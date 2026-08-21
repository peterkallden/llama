set(TARGET llama-agent-core)
add_library(${TARGET} agent-bootstrap.cpp agent-bootstrap.h agent-package-json.cpp agent-package-json.h agent-prepared-generation.cpp agent-prepared-generation.h blueprint-selector.cpp blueprint-selector.h dataset-contracts.cpp dataset-contracts.h deliberation-policy.cpp deliberation-policy.h agent-runtime-context.h agent-runtime-reflection-phase.cpp agent-runtime-research-phase.cpp agent-runtime.cpp agent-runtime.h runtime-json-contracts.cpp runtime-json-contracts.h memory-learning.cpp memory-learning.h reflection-json.cpp reflection-json.h schema-contract.cpp schema-contract.h sandbox-docker-runtime.cpp sandbox-docker-runtime.h sandbox-kubernetes-runtime.cpp sandbox-kubernetes-runtime.h tool-adapters.cpp tool-adapters.h tool-catalog.cpp tool-catalog.h tool-chat-bridge.cpp tool-chat-bridge.h tool-family-index.cpp tool-family-index.h tool-workflow-index.cpp tool-workflow-index.h tool-registry.cpp tool-registry.h tool-result-contracts.cpp tool-result-contracts.h tool-schema-compact.cpp tool-schema-compact.h tool-navigation.cpp tool-navigation.h research/research-contract.h research/research-workspace.cpp research/research-workspace.h research/research-workspace-factory.cpp research/research-workspace-factory.h research/research-controller.cpp research/research-controller.h research/research-runner.cpp research/research-runner.h)
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
