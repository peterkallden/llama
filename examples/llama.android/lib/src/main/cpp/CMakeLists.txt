cmake_minimum_required(VERSION 3.31.6)

project("ai-chat" VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}"   CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "" FORCE)

# The Android NDK supplies the Vulkan loader and host shader compiler, but not
# Vulkan-Hpp or a CMake package for SPIR-V-Headers. Use the same compatibility
# setup as the standalone Android native build.
if(ANDROID AND GGML_VULKAN)
    set(SPIRV-Headers_DIR "${CMAKE_CURRENT_LIST_DIR}/cmake/SPIRV-Headers"
        CACHE PATH "Android NDK SPIR-V-Headers compatibility package")
    include("${CMAKE_CURRENT_LIST_DIR}/../../../../../../cmake/android-vulkan.cmake")
    llama_android_prepare_vulkan()
    list(APPEND CMAKE_PREFIX_PATH "${SPIRV-Headers_DIR}")
endif()

# --------------------------------------------------------------------------
# AI Chat library
# --------------------------------------------------------------------------

if(DEFINED ANDROID_ABI)
    message(STATUS "Detected Android ABI: ${ANDROID_ABI}")
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        set(GGML_SYSTEM_ARCH "ARM")
        if(NOT DEFINED GGML_CPU_KLEIDIAI)
            set(GGML_CPU_KLEIDIAI ON)
        endif()
        if(NOT DEFINED GGML_OPENMP)
            set(GGML_OPENMP ON)
        endif()
    elseif(ANDROID_ABI STREQUAL "x86_64")
        set(GGML_SYSTEM_ARCH "x86")
        set(GGML_CPU_KLEIDIAI OFF)
        set(GGML_OPENMP OFF)
    else()
        message(FATAL_ERROR "Unsupported ABI: ${ANDROID_ABI}")
    endif()
endif()

set(LLAMA_SRC ${CMAKE_CURRENT_LIST_DIR}/../../../../../../)
add_subdirectory(${LLAMA_SRC} build-llama)

# Android uses the model-backed agent runtime with an embedded, Service-owned
# daemon profile. The standalone Linux daemon transports, server-context
# loader and external sandbox processes are not part of the APK. Keep this
# target on the same planner/executor/reflection implementation used by the
# full runtime so Android does not grow a second agent contract.
add_library(llama-agent-android-runtime STATIC
        ${LLAMA_SRC}/tools/agent/cli/agent-cli-generation.cpp
        ${LLAMA_SRC}/tools/agent/cli/agent-cli-generation-utils.cpp
        ${LLAMA_SRC}/tools/agent/cli/agent-cli-inference.cpp
        ${LLAMA_SRC}/tools/agent/cli/agent-cli-runtime.cpp
        ${LLAMA_SRC}/tools/agent/cli/agent-cli-selection.cpp
        ${LLAMA_SRC}/tools/agent/runtime/agent-runtime-assembly.cpp
        ${LLAMA_SRC}/tools/agent/runtime/agent-runtime-chat-driver.cpp
        ${LLAMA_SRC}/tools/agent/runtime/agent-runtime-execution.cpp
        ${LLAMA_SRC}/tools/agent/runtime/agent-runtime-host.cpp
        ${LLAMA_SRC}/tools/agent/runtime/agent-plan-orchestration.cpp
        ${LLAMA_SRC}/tools/agent/runtime/agent-runtime-resident.cpp
        ${LLAMA_SRC}/tools/agent/runtime/agent-model-residency.cpp
        ${LLAMA_SRC}/tools/agent/runtime/agent-runtime-session.cpp
        ${LLAMA_SRC}/tools/agent/runtime/agent-runtime-session-host.cpp
        ${LLAMA_SRC}/tools/agent/runtime/agent-runtime-turn-driver.cpp
        ${LLAMA_SRC}/tools/agent/runtime/agent-runtime-package-io.cpp
        ${LLAMA_SRC}/tools/agent/resource/agent-resource-chunker.cpp
        ${LLAMA_SRC}/tools/agent/tooling/agent-tool-provider.cpp
        ${LLAMA_SRC}/tools/agent/tooling/agent-selection-contracts.cpp
        ${LLAMA_SRC}/tools/agent/tooling/agent-tool-result-json-contracts.cpp
        ${LLAMA_SRC}/tools/agent/tooling/agent-tool-runtime-adapter.cpp
        ${LLAMA_SRC}/tools/agent/adaptation/agent-learning-transaction-store-factory.cpp
        ${LLAMA_SRC}/tools/agent/mcp/agent-mcp-protocol.cpp
        ${LLAMA_SRC}/tools/agent/mcp/agent-mcp-http-client.cpp
        ${LLAMA_SRC}/tools/agent/mcp/agent-mcp-http-transport.cpp
        ${CMAKE_CURRENT_LIST_DIR}/agent_android_credentials.cpp
        ${CMAKE_CURRENT_LIST_DIR}/agent_android_mcp_transport.cpp)

target_include_directories(llama-agent-android-runtime PUBLIC
        ${LLAMA_SRC}
        ${LLAMA_SRC}/common
        ${LLAMA_SRC}/tools/agent/cli
        ${LLAMA_SRC}/tools/agent/runtime
        ${LLAMA_SRC}/tools/agent/tooling
        ${LLAMA_SRC}/tools/agent/resource
        ${LLAMA_SRC}/tools/agent/data
        ${LLAMA_SRC}/tools/server)

target_compile_features(llama-agent-android-runtime PUBLIC cxx_std_17)
# This flag excludes only desktop server-context/resident-loader code. It does
# not disable AgentRuntimeService or its native worker queue.
target_compile_definitions(llama-agent-android-runtime PRIVATE
        LLAMA_AGENT_ANDROID_CLI_ONLY=1)
target_link_libraries(llama-agent-android-runtime PUBLIC
        llama-agent-core
        llama-agent-tooling
        llama-agent-memory
        llama-agent-plan
        llama-agent-storage-sqlite
        cpp-httplib
        llama-common
        llama)

add_library(${CMAKE_PROJECT_NAME} SHARED
        ai_chat.cpp
        agent_android_runtime.cpp
        agent_android_storage.cpp)

target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
        GGML_SYSTEM_ARCH=${GGML_SYSTEM_ARCH}
        GGML_CPU_KLEIDIAI=$<BOOL:${GGML_CPU_KLEIDIAI}>
        GGML_OPENMP=$<BOOL:${GGML_OPENMP}>
        LLAMA_AGENT_ANDROID_CPU_CAPABILITY=1
)

if(GGML_VULKAN)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
            LLAMA_AGENT_ANDROID_VULKAN_BUILT=1)
else()
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
            LLAMA_AGENT_ANDROID_VULKAN_BUILT=0)
endif()

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
        ${LLAMA_SRC}
        ${LLAMA_SRC}/common
        ${LLAMA_SRC}/include
        ${LLAMA_SRC}/ggml/include
        ${LLAMA_SRC}/ggml/src)

target_link_libraries(${CMAKE_PROJECT_NAME}
        llama
        llama-common
        llama-agent-memory
        llama-agent-plan
        llama-agent-storage-sqlite
        llama-agent-android-runtime
        android
        log)

if(GGML_VULKAN AND TARGET ggml-vulkan)
    target_include_directories(ggml-vulkan PRIVATE "${_android_vulkan_include}")
    # GGML_BACKEND_DL builds Vulkan as a MODULE. Build it with the Android
    # library so Gradle sees the generated .so; runtime selection remains
    # optional and is handled by ggml's backend loader.
    add_dependencies(${CMAKE_PROJECT_NAME} ggml-vulkan)
endif()
