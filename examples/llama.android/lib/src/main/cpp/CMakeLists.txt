cmake_minimum_required(VERSION 3.31.6)

project("ai-chat" VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}"   CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "" FORCE)

# --------------------------------------------------------------------------
# AI Chat library
# --------------------------------------------------------------------------

if(DEFINED ANDROID_ABI)
    message(STATUS "Detected Android ABI: ${ANDROID_ABI}")
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        set(GGML_SYSTEM_ARCH "ARM")
        set(GGML_CPU_KLEIDIAI ON)
        set(GGML_OPENMP ON)
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

# Android uses the model-backed agent runtime without the Linux host layers
# (daemon, server-context and external sandbox processes). Keep this target
# on the same planner/executor/reflection implementation used by the full
# runtime so Android does not grow a second agent contract.
add_library(llama-agent-android-runtime STATIC
        ${LLAMA_SRC}/tools/agent/cli/agent-cli-generation.cpp
        ${LLAMA_SRC}/tools/agent/cli/agent-cli-generation-utils.cpp
        ${LLAMA_SRC}/tools/agent/cli/agent-cli-inference.cpp
        ${LLAMA_SRC}/tools/agent/cli/agent-cli-runtime.cpp
        ${LLAMA_SRC}/tools/agent/runtime/agent-runtime-assembly.cpp
        ${LLAMA_SRC}/tools/agent/tooling/agent-tool-runtime-adapter.cpp)

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
target_link_libraries(llama-agent-android-runtime PUBLIC
        llama-agent-core
        llama-agent-tooling
        llama-agent-memory
        llama-agent-plan
        llama-agent-storage-sqlite
        llama-common
        llama)

add_library(${CMAKE_PROJECT_NAME} SHARED
        ai_chat.cpp
        agent_android_storage.cpp)

target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
        GGML_SYSTEM_ARCH=${GGML_SYSTEM_ARCH}
        GGML_CPU_KLEIDIAI=$<BOOL:${GGML_CPU_KLEIDIAI}>
        GGML_OPENMP=$<BOOL:${GGML_OPENMP}>
)

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
