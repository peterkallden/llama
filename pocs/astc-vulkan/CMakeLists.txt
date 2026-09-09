# Experimental ASTC Vulkan proof of concept.
# Keep this target independent from ggml-vulkan until the device-backed path
# has correctness and performance evidence.

set(ASTC_VULKAN_POC_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# The optional runtime overlay is linked through llama-server's shared
# library. Keep every ASTC static component PIC-capable, not only the final
# provider target.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Keep the optional neural astcenc ABI visible before the shared GPU finisher
# target is declared; hybrid offline builds must use the same ABI as latent-smoke.
option(GGML_VK_ASTC_EXPERIMENTAL_NEURAL_RANK
    "Enable the side-fork ASTC neural-ranking API in the latent PoC" OFF)
set(GGML_VK_ASTC_NEURAL_RANK_LIBRARY "" CACHE FILEPATH
    "Static library from the isolated ASTC neural-ranking fork")
set(GGML_VK_ASTC_NEURAL_RANK_INCLUDE_DIR "" CACHE PATH
    "Source include directory from the isolated ASTC neural-ranking fork")

# Offline-only primitives shared by artifact metadata and latent experiments.
# They carry no Vulkan or ggml dependency.
add_library(astc-vulkan-algorithms STATIC
    astc-vulkan-artifact-policy.cpp
    astc-vulkan-composition-gate.cpp
    astc-vulkan-block-ldlq.cpp
    astc-vulkan-d1-prescreen.cpp
    astc-vulkan-d2-prescreen.cpp
    astc-vulkan-d2-pair-transform.cpp
    astc-vulkan-d2-row-scale.cpp
    astc-vulkan-format.cpp
    astc-vulkan-gauge.cpp
    astc-vulkan-gpu-d1-ranking.cpp
    astc-vulkan-gpu-ranking.cpp
    astc-vulkan-hash.cpp
    astc-vulkan-astc-mode.cpp
    astc-vulkan-ise.cpp
    astc-vulkan-bit-budget.cpp
    astc-vulkan-objective.cpp
    astc-vulkan-paired.cpp
    astc-vulkan-paired-layout.cpp
    astc-vulkan-paired-selector.cpp
    astc-vulkan-pv.cpp
    astc-vulkan-ranking-plan.cpp
    astc-vulkan-yaqa.cpp)
target_include_directories(astc-vulkan-algorithms PUBLIC
    ${ASTC_VULKAN_POC_INCLUDE_DIR})
target_compile_features(astc-vulkan-algorithms PUBLIC cxx_std_17)

# Offline GPU-encoder experiment.  The cache generator may opt into its
# hybrid backend (GPU proposer + CPU astcenc finisher); CPU astcenc remains the
# legal-block oracle and fallback.  No GPU encoder code participates in the
# inference runtime dispatch.
add_library(astc-vulkan-gpu-encoder STATIC
    gpu-encoder/shared/astc-gpu-encoder.cpp
    gpu-encoder/shared/astc-gpu-encoder-subset.cpp
    gpu-encoder/shared/astc-gpu-selector-adapter.cpp
    gpu-encoder/d1/astc-gpu-d1-exact-subset.cpp
    gpu-encoder/d1/astc-gpu-d1-hybrid.cpp
    gpu-encoder/d1/astc-gpu-d1-candidates.cpp
    gpu-encoder/d1/astc-gpu-d1-neural-rank.cpp
    gpu-encoder/d1/astc-gpu-d1-source.cpp
    gpu-encoder/d2/astc-gpu-d2-candidates.cpp
    gpu-encoder/d2/astc-gpu-d2-exact-subset.cpp
    gpu-encoder/d2/astc-gpu-d2-neural-rank.cpp
    gpu-encoder/d2/astc-gpu-d2-source.cpp)
target_include_directories(astc-vulkan-gpu-encoder PUBLIC
    ${ASTC_VULKAN_POC_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu-encoder/shared
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu-encoder/d1
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu-encoder/d2)
target_link_libraries(astc-vulkan-gpu-encoder PUBLIC astc-vulkan-algorithms)
target_compile_features(astc-vulkan-gpu-encoder PUBLIC cxx_std_17)

add_executable(test-astc-vulkan-gpu-encoder
    tests/test-astc-vulkan-gpu-encoder.cpp)
target_link_libraries(test-astc-vulkan-gpu-encoder PRIVATE astc-vulkan-gpu-encoder)
target_compile_features(test-astc-vulkan-gpu-encoder PRIVATE cxx_std_17)
if (BUILD_TESTING)
    add_test(NAME test-astc-vulkan-gpu-encoder
        COMMAND test-astc-vulkan-gpu-encoder)
    set_property(TEST test-astc-vulkan-gpu-encoder PROPERTY LABELS astc)
endif()

add_executable(test-astc-vulkan-ise
    tests/test-astc-vulkan-ise.cpp)
target_link_libraries(test-astc-vulkan-ise PRIVATE astc-vulkan-algorithms)
target_compile_features(test-astc-vulkan-ise PRIVATE cxx_std_17)
if (BUILD_TESTING)
    add_test(NAME test-astc-vulkan-ise COMMAND test-astc-vulkan-ise)
    set_property(TEST test-astc-vulkan-ise PROPERTY LABELS astc)
endif()

add_executable(test-astc-vulkan-astc-mode
    tests/test-astc-vulkan-astc-mode.cpp)
target_link_libraries(test-astc-vulkan-astc-mode PRIVATE astc-vulkan-algorithms)
target_compile_features(test-astc-vulkan-astc-mode PRIVATE cxx_std_17)
if (BUILD_TESTING)
    add_test(NAME test-astc-vulkan-astc-mode COMMAND test-astc-vulkan-astc-mode)
    set_property(TEST test-astc-vulkan-astc-mode PROPERTY LABELS astc)
endif()

add_executable(test-astc-vulkan-hash
    tests/test-astc-vulkan-hash.cpp)
target_link_libraries(test-astc-vulkan-hash PRIVATE astc-vulkan-algorithms)
target_compile_features(test-astc-vulkan-hash PRIVATE cxx_std_17)

add_executable(test-astc-vulkan-artifact-policy
    tests/test-astc-vulkan-artifact-policy.cpp)
target_link_libraries(test-astc-vulkan-artifact-policy PRIVATE astc-vulkan-algorithms)
target_compile_features(test-astc-vulkan-artifact-policy PRIVATE cxx_std_17)

# Offline-only composition admission: inputs are precomputed, sequential
# full-model replay observations. This target deliberately does not make a
# runtime scheduling or model-execution dependency.
add_executable(test-astc-vulkan-composition-gate
    tests/test-astc-vulkan-composition-gate.cpp)
target_link_libraries(test-astc-vulkan-composition-gate PRIVATE astc-vulkan-algorithms)
target_compile_features(test-astc-vulkan-composition-gate PRIVATE cxx_std_17)

add_executable(astc-vulkan-composition-gate-smoke
    astc-vulkan-composition-gate-smoke.cpp)
target_link_libraries(astc-vulkan-composition-gate-smoke PRIVATE astc-vulkan-algorithms)
target_compile_features(astc-vulkan-composition-gate-smoke PRIVATE cxx_std_17)

add_executable(test-astc-vulkan-gauge
    tests/test-astc-vulkan-gauge.cpp)
target_link_libraries(test-astc-vulkan-gauge PRIVATE astc-vulkan-algorithms)
target_compile_features(test-astc-vulkan-gauge PRIVATE cxx_std_17)

add_executable(test-astc-vulkan-d1-prescreen
    tests/test-astc-vulkan-d1-prescreen.cpp)
target_link_libraries(test-astc-vulkan-d1-prescreen PRIVATE astc-vulkan-algorithms)
target_compile_features(test-astc-vulkan-d1-prescreen PRIVATE cxx_std_17)

add_executable(test-astc-vulkan-d2-prescreen
    tests/test-astc-vulkan-d2-prescreen.cpp)
target_link_libraries(test-astc-vulkan-d2-prescreen PRIVATE astc-vulkan-algorithms)
target_compile_features(test-astc-vulkan-d2-prescreen PRIVATE cxx_std_17)
if (BUILD_TESTING)
    add_test(NAME test-astc-vulkan-d2-prescreen
        COMMAND test-astc-vulkan-d2-prescreen)
    set_property(TEST test-astc-vulkan-d2-prescreen PROPERTY LABELS astc)
endif()

add_executable(test-astc-vulkan-d2-pair-transform
    tests/test-astc-vulkan-d2-pair-transform.cpp)
target_link_libraries(test-astc-vulkan-d2-pair-transform PRIVATE astc-vulkan-algorithms)
target_compile_features(test-astc-vulkan-d2-pair-transform PRIVATE cxx_std_17)
if (BUILD_TESTING)
    add_test(NAME test-astc-vulkan-d2-pair-transform
        COMMAND test-astc-vulkan-d2-pair-transform)
    set_property(TEST test-astc-vulkan-d2-pair-transform PROPERTY LABELS astc)
endif()

add_executable(test-astc-vulkan-d2-row-scale
    tests/test-astc-vulkan-d2-row-scale.cpp)
target_link_libraries(test-astc-vulkan-d2-row-scale PRIVATE astc-vulkan-algorithms)
target_compile_features(test-astc-vulkan-d2-row-scale PRIVATE cxx_std_17)

add_executable(test-astc-vulkan-gpu-ranking
    tests/test-astc-vulkan-gpu-ranking.cpp)
target_link_libraries(test-astc-vulkan-gpu-ranking PRIVATE astc-vulkan-algorithms)
target_compile_features(test-astc-vulkan-gpu-ranking PRIVATE cxx_std_17)

add_executable(test-astc-vulkan-gpu-d1-ranking
    tests/test-astc-vulkan-gpu-d1-ranking.cpp)
target_link_libraries(test-astc-vulkan-gpu-d1-ranking PRIVATE astc-vulkan-algorithms)
target_compile_features(test-astc-vulkan-gpu-d1-ranking PRIVATE cxx_std_17)

add_executable(test-astc-vulkan-ranking-plan
    tests/test-astc-vulkan-ranking-plan.cpp)
target_link_libraries(test-astc-vulkan-ranking-plan PRIVATE astc-vulkan-algorithms)
target_compile_features(test-astc-vulkan-ranking-plan PRIVATE cxx_std_17)

add_executable(test-astc-vulkan-paired
    tests/test-astc-vulkan-paired.cpp)
target_link_libraries(test-astc-vulkan-paired PRIVATE astc-vulkan-algorithms)
target_compile_features(test-astc-vulkan-paired PRIVATE cxx_std_17)

add_executable(test-astc-vulkan-paired-selector
    tests/test-astc-vulkan-paired-selector.cpp)
target_link_libraries(test-astc-vulkan-paired-selector PRIVATE astc-vulkan-algorithms)
target_compile_features(test-astc-vulkan-paired-selector PRIVATE cxx_std_17)

add_executable(test-astc-vulkan-paired-layout
    tests/test-astc-vulkan-paired-layout.cpp)
target_link_libraries(test-astc-vulkan-paired-layout PRIVATE astc-vulkan-driver)
target_compile_features(test-astc-vulkan-paired-layout PRIVATE cxx_std_17)

add_executable(test-astc-vulkan-objective
    tests/test-astc-vulkan-objective.cpp)
target_link_libraries(test-astc-vulkan-objective PRIVATE astc-vulkan-algorithms)
target_compile_features(test-astc-vulkan-objective PRIVATE cxx_std_17)

add_executable(test-astc-vulkan-block-ldlq-config
    tests/test-astc-vulkan-block-ldlq-config.cpp)
target_link_libraries(test-astc-vulkan-block-ldlq-config PRIVATE astc-vulkan-algorithms)
target_compile_features(test-astc-vulkan-block-ldlq-config PRIVATE cxx_std_17)

add_executable(test-astc-vulkan-pv
    tests/test-astc-vulkan-pv.cpp)
target_link_libraries(test-astc-vulkan-pv PRIVATE astc-vulkan-algorithms)
target_compile_features(test-astc-vulkan-pv PRIVATE cxx_std_17)

add_executable(test-astc-vulkan-pv-d2
    tests/test-astc-vulkan-pv-d2.cpp)
target_link_libraries(test-astc-vulkan-pv-d2 PRIVATE astc-vulkan-algorithms)
target_compile_features(test-astc-vulkan-pv-d2 PRIVATE cxx_std_17)

add_executable(test-astc-vulkan-yaqa
    tests/test-astc-vulkan-yaqa.cpp)
target_link_libraries(test-astc-vulkan-yaqa PRIVATE astc-vulkan-algorithms)
target_compile_features(test-astc-vulkan-yaqa PRIVATE cxx_std_17)

add_executable(test-astc-vulkan-contract
    tests/test-astc-vulkan-contract.cpp)
target_include_directories(test-astc-vulkan-contract PRIVATE
    ${ASTC_VULKAN_POC_INCLUDE_DIR})
target_compile_features(test-astc-vulkan-contract PRIVATE cxx_std_17)

add_executable(test-astc-vulkan-smoke
    tests/test-astc-vulkan-smoke.cpp)
target_include_directories(test-astc-vulkan-smoke PRIVATE
    ${ASTC_VULKAN_POC_INCLUDE_DIR})
target_compile_features(test-astc-vulkan-smoke PRIVATE cxx_std_17)

add_executable(test-astc-vulkan-tensor-contract
    tests/test-astc-vulkan-tensor-contract.cpp)
target_include_directories(test-astc-vulkan-tensor-contract PRIVATE
    ${ASTC_VULKAN_POC_INCLUDE_DIR})
target_compile_features(test-astc-vulkan-tensor-contract PRIVATE cxx_std_17)

add_library(astc-vulkan-driver STATIC
    astc-vulkan-driver.cpp
    astc-vulkan-format.cpp
    astc-vulkan-manifest.cpp
    astc-vulkan-tensor-catalog.cpp)
target_include_directories(astc-vulkan-driver PUBLIC
    ${ASTC_VULKAN_POC_INCLUDE_DIR})
target_link_libraries(astc-vulkan-driver PUBLIC astc-vulkan-algorithms)
target_compile_features(astc-vulkan-driver PUBLIC cxx_std_17)

add_library(astc-vulkan-provenance STATIC
    astc-vulkan-provenance.cpp)
target_include_directories(astc-vulkan-provenance PUBLIC
    ${ASTC_VULKAN_POC_INCLUDE_DIR})
target_link_libraries(astc-vulkan-provenance PUBLIC vendor::hash)
target_compile_features(astc-vulkan-provenance PUBLIC cxx_std_17)

add_library(astc-vulkan-cache STATIC
    astc-vulkan-cache.cpp)
target_include_directories(astc-vulkan-cache PUBLIC
    ${ASTC_VULKAN_POC_INCLUDE_DIR})
target_link_libraries(astc-vulkan-cache PUBLIC
    astc-vulkan-driver astc-vulkan-provenance ggml-base)
target_compile_features(astc-vulkan-cache PUBLIC cxx_std_17)

# Shared physical row-band/range contract for D1 and D2. It is deliberately
# independent of Vulkan resources so geometry and cache-range tests stay fast.
add_library(astc-vulkan-stream-loader STATIC
    astc-vulkan-stream-loader.cpp)
target_include_directories(astc-vulkan-stream-loader PUBLIC
    ${ASTC_VULKAN_POC_INCLUDE_DIR})
target_link_libraries(astc-vulkan-stream-loader PUBLIC astc-vulkan-driver)
target_compile_features(astc-vulkan-stream-loader PUBLIC cxx_std_17)

add_executable(test-astc-vulkan-stream-loader
    tests/test-astc-vulkan-stream-loader.cpp)
target_link_libraries(test-astc-vulkan-stream-loader PRIVATE astc-vulkan-stream-loader)
target_compile_features(test-astc-vulkan-stream-loader PRIVATE cxx_std_17)
if (BUILD_TESTING)
    add_test(NAME test-astc-vulkan-stream-loader
        COMMAND test-astc-vulkan-stream-loader)
    set_property(TEST test-astc-vulkan-stream-loader PROPERTY LABELS astc)
endif()

add_executable(test-astc-vulkan-driver
    tests/test-astc-vulkan-driver.cpp)
target_link_libraries(test-astc-vulkan-driver PRIVATE astc-vulkan-driver)
target_include_directories(test-astc-vulkan-driver PRIVATE
    ${ASTC_VULKAN_POC_INCLUDE_DIR})
target_compile_features(test-astc-vulkan-driver PRIVATE cxx_std_17)

add_executable(astc-vulkan-artifact-pack
    astc-vulkan-artifact-pack.cpp)
target_link_libraries(astc-vulkan-artifact-pack PRIVATE
    astc-vulkan-driver astc-vulkan-provenance)
target_include_directories(astc-vulkan-artifact-pack PRIVATE
    ${ASTC_VULKAN_POC_INCLUDE_DIR})
target_compile_features(astc-vulkan-artifact-pack PRIVATE cxx_std_17)

add_executable(astc-vulkan-cache-tool
    astc-vulkan-cache-tool.cpp)
set_target_properties(astc-vulkan-cache-tool PROPERTIES OUTPUT_NAME astc-vulkan-cache)
target_link_libraries(astc-vulkan-cache-tool PRIVATE astc-vulkan-cache)
target_include_directories(astc-vulkan-cache-tool PRIVATE
    ${ASTC_VULKAN_POC_INCLUDE_DIR})
target_compile_features(astc-vulkan-cache-tool PRIVATE cxx_std_17)
if (BUILD_TESTING)
    add_test(NAME test-astc-vulkan-cache-tool-profiles
        COMMAND astc-vulkan-cache-tool profiles)
    set_property(TEST test-astc-vulkan-cache-tool-profiles PROPERTY LABELS astc)
endif()

find_package(Vulkan QUIET)
option(GGML_VK_ASTC_EXPERIMENTAL_SCHEDULER_ADAPTER
    "Build the isolated ASTC scheduler-adapter experiment" OFF)
option(GGML_VK_ASTC_EXPERIMENTAL_GPU_RANKING_DEVICE_SMOKE
    "Register the experimental GPU candidate-ranking device smoke" OFF)
# Some distributions expose the Vulkan loader and headers but make the
# standalone FindVulkan module report Vulkan_FOUND=FALSE (for example when
# optional shader components are absent).  The ggml Vulkan backend already
# relies on the same loader, so provide a local target for this sidecar POC.
if (NOT TARGET Vulkan::Vulkan)
    find_path(ASTC_VULKAN_INCLUDE_DIR NAMES vulkan/vulkan.h)
    find_library(ASTC_VULKAN_LIBRARY NAMES vulkan)
    if (ASTC_VULKAN_INCLUDE_DIR AND ASTC_VULKAN_LIBRARY)
        add_library(astc-vulkan-loader INTERFACE)
        target_include_directories(astc-vulkan-loader INTERFACE
            ${ASTC_VULKAN_INCLUDE_DIR})
        target_link_libraries(astc-vulkan-loader INTERFACE
            ${ASTC_VULKAN_LIBRARY})
        add_library(Vulkan::Vulkan ALIAS astc-vulkan-loader)
    endif()
endif()
if (TARGET Vulkan::Vulkan)
    # These static ASTC components are also linked into llama-server shared
    # libraries when the runtime overlay is enabled.
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    add_library(astc-vulkan-resource STATIC
        astc-vulkan-resource.cpp
        astc-vulkan-budget.cpp)
    target_include_directories(astc-vulkan-resource PUBLIC
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_link_libraries(astc-vulkan-resource PUBLIC Vulkan::Vulkan astc-vulkan-driver)
    target_compile_features(astc-vulkan-resource PUBLIC cxx_std_17)

    add_library(astc-vulkan-shared-device STATIC
        astc-vulkan-shared-device.cpp)
    target_include_directories(astc-vulkan-shared-device PUBLIC
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_link_libraries(astc-vulkan-shared-device PUBLIC
        astc-vulkan-resource Vulkan::Vulkan)
    target_compile_features(astc-vulkan-shared-device PUBLIC cxx_std_17)

    add_library(astc-vulkan-residency STATIC
        astc-vulkan-residency.cpp)
    target_include_directories(astc-vulkan-residency PUBLIC
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_link_libraries(astc-vulkan-residency PUBLIC astc-vulkan-resource)
    target_compile_features(astc-vulkan-residency PUBLIC cxx_std_17)

    # Model-level orchestration deliberately reuses the existing manifest,
    # artifact-policy, cache, and residency contracts. It adds no encoder or
    # Vulkan-resource ownership and is therefore safe to keep offline/runtime
    # selection concerns separate from per-tensor loading.
    add_library(astc-vulkan-model-cache STATIC
        astc-vulkan-model-cache.cpp)
    target_include_directories(astc-vulkan-model-cache PUBLIC
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_link_libraries(astc-vulkan-model-cache PUBLIC
        astc-vulkan-cache astc-vulkan-residency)
    target_compile_features(astc-vulkan-model-cache PUBLIC cxx_std_17)

    # Pre-build discovery is deliberately separate from artifact planning:
    # it consumes a source-model tensor inventory plus usage metrics before
    # any ASTC payload exists, and only emits a shortlist for the builder.
    add_library(astc-vulkan-discovery STATIC
        astc-vulkan-discovery.cpp)
    target_include_directories(astc-vulkan-discovery PUBLIC
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_link_libraries(astc-vulkan-discovery PUBLIC astc-vulkan-model-cache)
    target_compile_features(astc-vulkan-discovery PUBLIC cxx_std_17)

    if (TARGET astc-vulkan-cache-tool)
        target_sources(astc-vulkan-cache-tool PRIVATE astc-vulkan-input.cpp)
        target_link_libraries(astc-vulkan-cache-tool PRIVATE
            astc-vulkan-discovery ggml-base)
    endif()

    add_executable(test-astc-vulkan-discovery
        tests/test-astc-vulkan-discovery.cpp)
    target_include_directories(test-astc-vulkan-discovery PRIVATE
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_link_libraries(test-astc-vulkan-discovery PRIVATE astc-vulkan-discovery)
    target_compile_features(test-astc-vulkan-discovery PRIVATE cxx_std_17)
    if (BUILD_TESTING)
        add_test(NAME test-astc-vulkan-discovery
            COMMAND test-astc-vulkan-discovery)
        set_property(TEST test-astc-vulkan-discovery PROPERTY LABELS astc)
    endif()

    # Runtime-local page ownership consumes planner output but does not own
    # Vulkan images yet. Resource/upload ownership remains in the existing
    # tensor session and sidecar layers.
    add_library(astc-vulkan-page-owner STATIC
        astc-vulkan-page-owner.cpp)
    target_include_directories(astc-vulkan-page-owner PUBLIC
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_link_libraries(astc-vulkan-page-owner PUBLIC astc-vulkan-model-cache)
    target_compile_features(astc-vulkan-page-owner PUBLIC cxx_std_17)

    # Single runtime-facing cache owner. It intentionally stays independent
    # from llama graph objects and Vulkan command buffers; the latter consume
    # this validated, residency-aware contract.
    add_library(astc-vulkan-runtime-overlay STATIC
        astc-vulkan-runtime-overlay.cpp)
    target_include_directories(astc-vulkan-runtime-overlay PUBLIC
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_link_libraries(astc-vulkan-runtime-overlay PUBLIC
        astc-vulkan-page-owner)
    target_compile_features(astc-vulkan-runtime-overlay PUBLIC cxx_std_17)

    add_executable(test-astc-vulkan-page-owner
        tests/test-astc-vulkan-page-owner.cpp)
    target_include_directories(test-astc-vulkan-page-owner PRIVATE
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_link_libraries(test-astc-vulkan-page-owner PRIVATE astc-vulkan-page-owner)
    target_compile_features(test-astc-vulkan-page-owner PRIVATE cxx_std_17)
    if (BUILD_TESTING)
        add_test(NAME test-astc-vulkan-page-owner
            COMMAND test-astc-vulkan-page-owner)
        set_property(TEST test-astc-vulkan-page-owner PROPERTY LABELS astc)
    endif()

    if (TARGET astc-vulkan-cache-tool)
        target_link_libraries(astc-vulkan-cache-tool PRIVATE astc-vulkan-model-cache)
        target_compile_definitions(astc-vulkan-cache-tool PRIVATE
            ASTC_VULKAN_MODEL_CACHE_AVAILABLE=1)
    endif()

    add_library(astc-vulkan-ffn-adapter STATIC
        astc-vulkan-ffn-adapter.cpp)
    target_include_directories(astc-vulkan-ffn-adapter PUBLIC
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_link_libraries(astc-vulkan-ffn-adapter PUBLIC
        astc-vulkan-resource astc-vulkan-driver)
    target_compile_features(astc-vulkan-ffn-adapter PUBLIC cxx_std_17)

    add_library(astc-vulkan-dispatch STATIC
        astc-vulkan-dispatch.cpp)
    target_include_directories(astc-vulkan-dispatch PUBLIC
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_link_libraries(astc-vulkan-dispatch PUBLIC
        astc-vulkan-resource astc-vulkan-driver Vulkan::Vulkan)
    target_compile_features(astc-vulkan-dispatch PUBLIC cxx_std_17)

    # D2 consumes the same sampled-ASTC resource as D1, but needs a distinct
    # logical-height/layout-buffer dispatch contract.
    add_library(astc-vulkan-paired-dispatch STATIC
        astc-vulkan-paired-dispatch.cpp)
    target_include_directories(astc-vulkan-paired-dispatch PUBLIC
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_link_libraries(astc-vulkan-paired-dispatch PUBLIC
        astc-vulkan-resource astc-vulkan-driver astc-vulkan-algorithms Vulkan::Vulkan)
    target_compile_features(astc-vulkan-paired-dispatch PUBLIC cxx_std_17)

    add_executable(test-astc-vulkan-paired-dispatch
        tests/test-astc-vulkan-paired-dispatch.cpp)
    target_include_directories(test-astc-vulkan-paired-dispatch PRIVATE
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_link_libraries(test-astc-vulkan-paired-dispatch PRIVATE astc-vulkan-paired-dispatch)
    target_compile_features(test-astc-vulkan-paired-dispatch PRIVATE cxx_std_17)
    if (BUILD_TESTING)
        add_test(NAME test-astc-vulkan-paired-dispatch
            COMMAND test-astc-vulkan-paired-dispatch)
        set_property(TEST test-astc-vulkan-paired-dispatch PROPERTY LABELS astc)
    endif()

    # Offline-only GPU scoring session. It shares the ordinary ASTC resource
    # uploader with runtime, but it never enters the inference scheduler.
    add_library(astc-vulkan-gpu-ranking-dispatch STATIC
        astc-vulkan-gpu-ranking-dispatch.cpp)
    target_include_directories(astc-vulkan-gpu-ranking-dispatch PUBLIC
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_link_libraries(astc-vulkan-gpu-ranking-dispatch PUBLIC
        astc-vulkan-resource astc-vulkan-algorithms Vulkan::Vulkan)
    target_compile_features(astc-vulkan-gpu-ranking-dispatch PUBLIC cxx_std_17)

    # Reusable two-stage YAQA scorer. Trace buffers and Vulkan pipelines stay
    # resident across candidate batches; this is offline-only and does not
    # change the runtime scheduler or ASTC cache ABI.
    add_library(astc-vulkan-yaqa-dispatch STATIC
        astc-vulkan-yaqa-dispatch.cpp)
    target_include_directories(astc-vulkan-yaqa-dispatch PUBLIC
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_link_libraries(astc-vulkan-yaqa-dispatch PUBLIC Vulkan::Vulkan)
    target_compile_features(astc-vulkan-yaqa-dispatch PUBLIC cxx_std_17)

    add_library(astc-vulkan-d1-prescreen-dispatch STATIC
        astc-vulkan-d1-prescreen-dispatch.cpp)
    target_include_directories(astc-vulkan-d1-prescreen-dispatch PUBLIC
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_link_libraries(astc-vulkan-d1-prescreen-dispatch PUBLIC
        astc-vulkan-resource astc-vulkan-algorithms Vulkan::Vulkan)
    target_compile_features(astc-vulkan-d1-prescreen-dispatch PUBLIC cxx_std_17)

    add_library(astc-vulkan-d2-prescreen-dispatch STATIC
        astc-vulkan-d2-prescreen-dispatch.cpp)
    target_include_directories(astc-vulkan-d2-prescreen-dispatch PUBLIC
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_link_libraries(astc-vulkan-d2-prescreen-dispatch PUBLIC
        astc-vulkan-resource astc-vulkan-algorithms Vulkan::Vulkan)
    target_compile_features(astc-vulkan-d2-prescreen-dispatch PUBLIC cxx_std_17)
    # The cache tool consumes the same bounded D2 profile screen before an
    # optional exact CPU build. Keep this offline-only dependency out of the
    # runtime/cache ABI.
    if (TARGET astc-vulkan-cache-tool)
        target_link_libraries(astc-vulkan-cache-tool PRIVATE astc-vulkan-d2-prescreen-dispatch)
        target_compile_definitions(astc-vulkan-cache-tool PRIVATE
            ASTC_VULKAN_D2_PRESCREEN_AVAILABLE=1)
    endif()
    add_executable(astc-vulkan-d2-prescreen-device-smoke
        astc-vulkan-d2-prescreen-device-smoke.cpp)
    target_include_directories(astc-vulkan-d2-prescreen-device-smoke PRIVATE
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_link_libraries(astc-vulkan-d2-prescreen-device-smoke PRIVATE
        astc-vulkan-d2-prescreen-dispatch)
    target_compile_features(astc-vulkan-d2-prescreen-device-smoke PRIVATE cxx_std_17)

    # Experimental offline physical ASTC proposer. It is linked into the
    # latent/cache producer only through the opt-in hybrid backend; scheduler
    # selection and inference runtime dispatch remain unchanged.
    add_library(astc-vulkan-gpu-encoder-dispatch STATIC
        gpu-encoder/shared/astc-gpu-encoder-dispatch.cpp
        gpu-encoder/shared/astc-gpu-encoder-exact-dispatch.cpp)
    target_include_directories(astc-vulkan-gpu-encoder-dispatch PUBLIC
        ${ASTC_VULKAN_POC_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu-encoder/shared
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu-encoder/d1)
    target_link_libraries(astc-vulkan-gpu-encoder-dispatch PUBLIC
        astc-vulkan-gpu-encoder astc-vulkan-resource Vulkan::Vulkan)
    target_compile_features(astc-vulkan-gpu-encoder-dispatch PUBLIC cxx_std_17)

    add_library(astc-vulkan-sidecar STATIC
        astc-vulkan-sidecar.cpp)
    target_include_directories(astc-vulkan-sidecar PUBLIC
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_link_libraries(astc-vulkan-sidecar PUBLIC
        astc-vulkan-ffn-adapter astc-vulkan-dispatch astc-vulkan-paired-dispatch
        astc-vulkan-stream-loader astc-vulkan-shared-device Vulkan::Vulkan)
    target_compile_features(astc-vulkan-sidecar PUBLIC cxx_std_17)

    add_executable(test-astc-vulkan-page-owner-device
        tests/test-astc-vulkan-page-owner-device.cpp)
    target_include_directories(test-astc-vulkan-page-owner-device PRIVATE
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_link_libraries(test-astc-vulkan-page-owner-device PRIVATE
        astc-vulkan-page-owner astc-vulkan-sidecar)
    target_compile_features(test-astc-vulkan-page-owner-device PRIVATE cxx_std_17)
    if (BUILD_TESTING)
        add_test(NAME test-astc-vulkan-page-owner-device
            COMMAND test-astc-vulkan-page-owner-device)
        set_property(TEST test-astc-vulkan-page-owner-device PROPERTY LABELS astc)
    endif()

    add_executable(test-astc-vulkan-ffn-adapter
        tests/test-astc-vulkan-ffn-adapter.cpp)
    target_link_libraries(test-astc-vulkan-ffn-adapter PRIVATE astc-vulkan-ffn-adapter)
    target_include_directories(test-astc-vulkan-ffn-adapter PRIVATE
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_compile_features(test-astc-vulkan-ffn-adapter PRIVATE cxx_std_17)

    add_executable(test-astc-vulkan-budget
        tests/test-astc-vulkan-budget.cpp)
    target_link_libraries(test-astc-vulkan-budget PRIVATE astc-vulkan-resource)
    target_include_directories(test-astc-vulkan-budget PRIVATE
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_compile_features(test-astc-vulkan-budget PRIVATE cxx_std_17)

    add_executable(test-astc-vulkan-residency
        tests/test-astc-vulkan-residency.cpp)
    target_link_libraries(test-astc-vulkan-residency PRIVATE astc-vulkan-residency)
    target_include_directories(test-astc-vulkan-residency PRIVATE
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_compile_features(test-astc-vulkan-residency PRIVATE cxx_std_17)

    add_executable(test-astc-vulkan-model-cache
        tests/test-astc-vulkan-model-cache.cpp)
    target_link_libraries(test-astc-vulkan-model-cache PRIVATE
        astc-vulkan-model-cache)
    target_include_directories(test-astc-vulkan-model-cache PRIVATE
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_compile_features(test-astc-vulkan-model-cache PRIVATE cxx_std_17)
    if (BUILD_TESTING)
    add_test(NAME test-astc-vulkan-model-cache
        COMMAND test-astc-vulkan-model-cache)
    set_property(TEST test-astc-vulkan-model-cache PROPERTY LABELS astc)
    add_test(NAME test-astc-vulkan-model-cache-cli
        COMMAND test-astc-vulkan-model-cache $<TARGET_FILE:astc-vulkan-cache-tool>)
    set_property(TEST test-astc-vulkan-model-cache-cli PROPERTY LABELS astc)
    endif()

    add_executable(test-astc-vulkan-dispatch
        tests/test-astc-vulkan-dispatch.cpp)
    target_link_libraries(test-astc-vulkan-dispatch PRIVATE astc-vulkan-dispatch)
    target_include_directories(test-astc-vulkan-dispatch PRIVATE
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_compile_features(test-astc-vulkan-dispatch PRIVATE cxx_std_17)

    add_executable(test-astc-vulkan-sidecar
        tests/test-astc-vulkan-sidecar.cpp)
    target_link_libraries(test-astc-vulkan-sidecar PRIVATE astc-vulkan-sidecar)
    target_include_directories(test-astc-vulkan-sidecar PRIVATE
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_compile_features(test-astc-vulkan-sidecar PRIVATE cxx_std_17)

    if (GGML_VK_ASTC_EXPERIMENTAL_SCHEDULER_ADAPTER)
        add_library(astc-vulkan-scheduler-adapter STATIC
            astc-vulkan-scheduler-adapter.cpp)
        target_include_directories(astc-vulkan-scheduler-adapter PUBLIC
            ${ASTC_VULKAN_POC_INCLUDE_DIR})
        target_link_libraries(astc-vulkan-scheduler-adapter PUBLIC
            astc-vulkan-sidecar astc-vulkan-cache)
        target_compile_features(astc-vulkan-scheduler-adapter PUBLIC cxx_std_17)

        # Opt-in runtime bridge: this consumes already-built cache artifacts.
        # It does not participate in encoding or alter the cache format.
        add_library(astc-vulkan-llama-provider STATIC
            astc-vulkan-llama-provider.cpp
            astc-vulkan-ggml-external-op.cpp)
        target_include_directories(astc-vulkan-llama-provider PUBLIC
            ${ASTC_VULKAN_POC_INCLUDE_DIR}
            ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/ggml/include)
        target_link_libraries(astc-vulkan-llama-provider PUBLIC
            astc-vulkan-runtime-overlay astc-vulkan-scheduler-adapter)
        target_compile_features(astc-vulkan-llama-provider PUBLIC cxx_std_17)
        set_target_properties(astc-vulkan-llama-provider PROPERTIES
            POSITION_INDEPENDENT_CODE ON)
        target_compile_definitions(astc-vulkan-llama-provider PRIVATE
            ASTC_VULKAN_D1_SHADER_PATH="${CMAKE_CURRENT_BINARY_DIR}/astc-ffn-matvec.comp.spv"
            ASTC_VULKAN_D2_SHADER_PATH="${CMAKE_CURRENT_BINARY_DIR}/astc-paired-matvec.comp.spv")
        add_dependencies(astc-vulkan-llama-provider
            astc-vulkan-ffn-e2e-shader astc-vulkan-paired-matvec-shader)

        add_executable(test-astc-vulkan-scheduler-adapter
            tests/test-astc-vulkan-scheduler-adapter.cpp)
        target_include_directories(test-astc-vulkan-scheduler-adapter PRIVATE
            ${ASTC_VULKAN_POC_INCLUDE_DIR})
        target_link_libraries(test-astc-vulkan-scheduler-adapter PRIVATE
            astc-vulkan-scheduler-adapter)
        target_compile_features(test-astc-vulkan-scheduler-adapter PRIVATE cxx_std_17)
        if (BUILD_TESTING)
            add_test(NAME test-astc-vulkan-scheduler-adapter
                COMMAND test-astc-vulkan-scheduler-adapter)
            set_property(TEST test-astc-vulkan-scheduler-adapter PROPERTY LABELS astc)
        endif()

        add_executable(astc-vulkan-scheduler-adapter-smoke
            astc-vulkan-scheduler-adapter-smoke.cpp
            astc-vulkan-input.cpp)
        target_include_directories(astc-vulkan-scheduler-adapter-smoke PRIVATE
            ${ASTC_VULKAN_POC_INCLUDE_DIR}
            ${CMAKE_SOURCE_DIR}/ggml/include)
        target_link_libraries(astc-vulkan-scheduler-adapter-smoke PRIVATE
            astc-vulkan-scheduler-adapter astcenc::astcenc-static ggml-base)
        target_compile_features(astc-vulkan-scheduler-adapter-smoke PRIVATE cxx_std_17)
        add_dependencies(astc-vulkan-scheduler-adapter-smoke
            astc-vulkan-ffn-e2e-shader astc-vulkan-paired-matvec-shader)
    endif()

    add_executable(astc-vulkan-capability-probe
        astc-vulkan-capability-probe.cpp)
    target_link_libraries(astc-vulkan-capability-probe PRIVATE Vulkan::Vulkan)
    target_compile_features(astc-vulkan-capability-probe PRIVATE cxx_std_17)

    add_executable(astc-vulkan-device-smoke
        astc-vulkan-device-smoke.cpp)
    target_link_libraries(astc-vulkan-device-smoke PRIVATE
        Vulkan::Vulkan astc-vulkan-resource astc-vulkan-stream-loader
        astc-vulkan-dispatch astc-vulkan-paired-dispatch)
    target_compile_features(astc-vulkan-device-smoke PRIVATE cxx_std_17)
    target_compile_definitions(astc-vulkan-device-smoke PRIVATE
        ASTC_VULKAN_D1_SHADER_PATH="${CMAKE_CURRENT_BINARY_DIR}/astc-ffn-matvec.comp.spv"
        ASTC_VULKAN_PAIRED_SHADER_PATH="${CMAKE_CURRENT_BINARY_DIR}/astc-paired-matvec.comp.spv")
    add_dependencies(astc-vulkan-device-smoke
        astc-vulkan-ffn-e2e-shader astc-vulkan-paired-matvec-shader)

    add_executable(astc-vulkan-paired-timing-smoke
        astc-vulkan-paired-timing-smoke.cpp)
    target_include_directories(astc-vulkan-paired-timing-smoke PRIVATE
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_link_libraries(astc-vulkan-paired-timing-smoke PRIVATE
        Vulkan::Vulkan astc-vulkan-resource astc-vulkan-stream-loader
        astc-vulkan-paired-dispatch astc-vulkan-shared-device)
    target_compile_features(astc-vulkan-paired-timing-smoke PRIVATE cxx_std_17)
    add_dependencies(astc-vulkan-paired-timing-smoke
        astc-vulkan-paired-matvec-shader)

    add_executable(astc-vulkan-gpu-ranking-device-smoke
        astc-vulkan-gpu-ranking-device-smoke.cpp)
    target_link_libraries(astc-vulkan-gpu-ranking-device-smoke PRIVATE
        astc-vulkan-gpu-ranking-dispatch Vulkan::Vulkan)
    target_compile_features(astc-vulkan-gpu-ranking-device-smoke PRIVATE cxx_std_17)

    add_executable(astc-vulkan-yaqa-device-smoke
        astc-vulkan-yaqa-device-smoke.cpp)
    target_link_libraries(astc-vulkan-yaqa-device-smoke PRIVATE
        astc-vulkan-algorithms Vulkan::Vulkan)
    target_compile_features(astc-vulkan-yaqa-device-smoke PRIVATE cxx_std_17)

    add_executable(astc-vulkan-yaqa-batch-device-smoke
        astc-vulkan-yaqa-batch-device-smoke.cpp)
    target_link_libraries(astc-vulkan-yaqa-batch-device-smoke PRIVATE
        astc-vulkan-algorithms Vulkan::Vulkan)
    target_compile_features(astc-vulkan-yaqa-batch-device-smoke PRIVATE cxx_std_17)

    add_executable(astc-vulkan-yaqa-session-device-smoke
        astc-vulkan-yaqa-session-device-smoke.cpp)
    target_include_directories(astc-vulkan-yaqa-session-device-smoke PRIVATE
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_link_libraries(astc-vulkan-yaqa-session-device-smoke PRIVATE
        astc-vulkan-yaqa-dispatch astc-vulkan-algorithms Vulkan::Vulkan)
    target_compile_features(astc-vulkan-yaqa-session-device-smoke PRIVATE cxx_std_17)

    add_executable(astc-vulkan-d1-prescreen-device-smoke
        astc-vulkan-d1-prescreen-device-smoke.cpp)
    target_include_directories(astc-vulkan-d1-prescreen-device-smoke PRIVATE
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_link_libraries(astc-vulkan-d1-prescreen-device-smoke PRIVATE
        astc-vulkan-algorithms astc-vulkan-resource Vulkan::Vulkan)
    target_compile_features(astc-vulkan-d1-prescreen-device-smoke PRIVATE cxx_std_17)

    add_executable(astc-vulkan-gpu-encoder-device-smoke
        astc-vulkan-gpu-encoder-device-smoke.cpp)
    target_include_directories(astc-vulkan-gpu-encoder-device-smoke PRIVATE
        ${ASTC_VULKAN_POC_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu-encoder/shared
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu-encoder/d1)
    target_link_libraries(astc-vulkan-gpu-encoder-device-smoke PRIVATE
        astc-vulkan-gpu-encoder-dispatch)
    target_compile_features(astc-vulkan-gpu-encoder-device-smoke PRIVATE cxx_std_17)

    add_executable(astc-vulkan-gpu-exact-subset-device-smoke
        astc-vulkan-gpu-exact-subset-device-smoke.cpp)
    target_include_directories(astc-vulkan-gpu-exact-subset-device-smoke PRIVATE
        ${ASTC_VULKAN_POC_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu-encoder/shared
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu-encoder/d1)
    target_link_libraries(astc-vulkan-gpu-exact-subset-device-smoke PRIVATE
        astc-vulkan-gpu-encoder-dispatch astc-vulkan-gpu-encoder-finisher)
    target_compile_features(astc-vulkan-gpu-exact-subset-device-smoke PRIVATE cxx_std_17)

    add_executable(astc-vulkan-gpu-exact-d2-8x5-device-smoke
        astc-vulkan-gpu-exact-d2-8x5-device-smoke.cpp)
    target_include_directories(astc-vulkan-gpu-exact-d2-8x5-device-smoke PRIVATE
        ${ASTC_VULKAN_POC_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu-encoder/shared
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu-encoder/d2)
    target_link_libraries(astc-vulkan-gpu-exact-d2-8x5-device-smoke PRIVATE
        astc-vulkan-gpu-encoder-dispatch astc-vulkan-gpu-encoder-finisher)
    target_compile_features(astc-vulkan-gpu-exact-d2-8x5-device-smoke PRIVATE cxx_std_17)

    foreach(ASTC_VULKAN_D2_H5_WIDTH IN ITEMS 6 10)
        add_executable(astc-vulkan-gpu-exact-d2-${ASTC_VULKAN_D2_H5_WIDTH}x5-device-smoke
            astc-vulkan-gpu-exact-d2-8x5-device-smoke.cpp)
        target_include_directories(astc-vulkan-gpu-exact-d2-${ASTC_VULKAN_D2_H5_WIDTH}x5-device-smoke PRIVATE
            ${ASTC_VULKAN_POC_INCLUDE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/gpu-encoder/shared
            ${CMAKE_CURRENT_SOURCE_DIR}/gpu-encoder/d2)
        target_link_libraries(astc-vulkan-gpu-exact-d2-${ASTC_VULKAN_D2_H5_WIDTH}x5-device-smoke PRIVATE
            astc-vulkan-gpu-encoder-dispatch astc-vulkan-gpu-encoder-finisher)
        target_compile_features(astc-vulkan-gpu-exact-d2-${ASTC_VULKAN_D2_H5_WIDTH}x5-device-smoke PRIVATE cxx_std_17)
        target_compile_definitions(astc-vulkan-gpu-exact-d2-${ASTC_VULKAN_D2_H5_WIDTH}x5-device-smoke PRIVATE
            ASTC_VULKAN_D2_EXACT_WIDTH=${ASTC_VULKAN_D2_H5_WIDTH})
    endforeach()

    add_executable(astc-vulkan-gpu-d2-exact-bank-fixture-smoke
        astc-vulkan-gpu-d2-exact-bank-fixture-smoke.cpp)
    target_include_directories(astc-vulkan-gpu-d2-exact-bank-fixture-smoke PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu-encoder/shared
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu-encoder/d2)
    target_link_libraries(astc-vulkan-gpu-d2-exact-bank-fixture-smoke PRIVATE
        astc-vulkan-gpu-encoder-dispatch astc-vulkan-gpu-encoder-finisher)
    target_compile_features(astc-vulkan-gpu-d2-exact-bank-fixture-smoke PRIVATE cxx_std_17)

    add_executable(astc-vulkan-gpu-d1-refined-fixture-smoke
        astc-vulkan-gpu-d1-refined-fixture-smoke.cpp)
    target_include_directories(astc-vulkan-gpu-d1-refined-fixture-smoke PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu-encoder/shared
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu-encoder/d1)
    target_link_libraries(astc-vulkan-gpu-d1-refined-fixture-smoke PRIVATE
        astc-vulkan-gpu-encoder-dispatch astc-vulkan-gpu-encoder-finisher)
    target_compile_features(astc-vulkan-gpu-d1-refined-fixture-smoke PRIVATE cxx_std_17)
endif()

find_package(astcenc CONFIG QUIET)
if (TARGET astcenc::astcenc-static)
    # CPU exact finisher for the isolated GPU-proposer experiment. This is
    # intentionally separate from the generic proposer/dispatch so builds
    # without astcenc can still exercise the physical host contract.
    add_library(astc-vulkan-gpu-encoder-finisher STATIC
        gpu-encoder/shared/astc-gpu-encoder-finisher.cpp)
    target_include_directories(astc-vulkan-gpu-encoder-finisher PUBLIC
        ${ASTC_VULKAN_POC_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu-encoder/shared)
    target_link_libraries(astc-vulkan-gpu-encoder-finisher PUBLIC
        astc-vulkan-gpu-encoder)
    if (GGML_VK_ASTC_EXPERIMENTAL_NEURAL_RANK)
        target_include_directories(astc-vulkan-gpu-encoder-finisher BEFORE PUBLIC
            ${GGML_VK_ASTC_NEURAL_RANK_INCLUDE_DIR})
        target_link_libraries(astc-vulkan-gpu-encoder-finisher PUBLIC
            ${GGML_VK_ASTC_NEURAL_RANK_LIBRARY})
        target_compile_definitions(astc-vulkan-gpu-encoder-finisher PUBLIC
            GGML_VK_ASTC_EXPERIMENTAL_NEURAL_RANK=1)
    else()
        target_link_libraries(astc-vulkan-gpu-encoder-finisher PUBLIC
            astcenc::astcenc-static)
    endif()
    target_compile_features(astc-vulkan-gpu-encoder-finisher PUBLIC cxx_std_17)
    if (TARGET test-astc-vulkan-gpu-encoder)
        target_link_libraries(test-astc-vulkan-gpu-encoder PRIVATE
            astc-vulkan-gpu-encoder-finisher)
    endif()

    # Representation-specific bridge above the shared physical proposer and
    # finisher. Kept separate so the base GPU-encoder contract stays free of
    # D2 paired-row semantics and builds without astcenc remain possible.
    add_library(astc-vulkan-gpu-encoder-d2-hybrid STATIC
        gpu-encoder/d2/astc-gpu-d2-hybrid.cpp)
    target_include_directories(astc-vulkan-gpu-encoder-d2-hybrid PUBLIC
        ${ASTC_VULKAN_POC_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu-encoder/shared
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu-encoder/d2)
    target_link_libraries(astc-vulkan-gpu-encoder-d2-hybrid PUBLIC
        astc-vulkan-gpu-encoder-finisher)
    target_compile_features(astc-vulkan-gpu-encoder-d2-hybrid PUBLIC cxx_std_17)

    if (TARGET test-astc-vulkan-gpu-encoder)
        target_link_libraries(test-astc-vulkan-gpu-encoder PRIVATE
            astc-vulkan-gpu-encoder-d2-hybrid)
    endif()

    if (TARGET astc-vulkan-gpu-encoder-device-smoke)
        target_link_libraries(astc-vulkan-gpu-encoder-device-smoke PRIVATE
            astc-vulkan-gpu-encoder-finisher astc-vulkan-gpu-encoder-d2-hybrid)
    endif()
    if (TARGET astc-vulkan-gpu-encoder-dispatch AND TARGET astc-vulkan-resource)
        add_library(astc-vulkan-gpu-encoder-verify STATIC
            gpu-encoder/shared/astc-gpu-encoder-vulkan-verify.cpp)
        target_include_directories(astc-vulkan-gpu-encoder-verify PUBLIC
            ${ASTC_VULKAN_POC_INCLUDE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/gpu-encoder/shared)
        target_link_libraries(astc-vulkan-gpu-encoder-verify PUBLIC
            astc-vulkan-gpu-encoder-finisher astc-vulkan-resource Vulkan::Vulkan)
        target_compile_features(astc-vulkan-gpu-encoder-verify PUBLIC cxx_std_17)
        if (TARGET astc-vulkan-gpu-encoder-device-smoke)
            target_link_libraries(astc-vulkan-gpu-encoder-device-smoke PRIVATE
                astc-vulkan-gpu-encoder-verify)
        endif()
        if (TARGET astc-vulkan-gpu-exact-subset-device-smoke)
            target_link_libraries(astc-vulkan-gpu-exact-subset-device-smoke PRIVATE
                astc-vulkan-gpu-encoder-verify)
        endif()
        if (TARGET astc-vulkan-gpu-exact-d2-8x5-device-smoke)
            target_link_libraries(astc-vulkan-gpu-exact-d2-8x5-device-smoke PRIVATE
                astc-vulkan-gpu-encoder-verify)
        endif()
        foreach(ASTC_VULKAN_D2_H5_WIDTH IN ITEMS 6 10)
            if (TARGET astc-vulkan-gpu-exact-d2-${ASTC_VULKAN_D2_H5_WIDTH}x5-device-smoke)
                target_link_libraries(astc-vulkan-gpu-exact-d2-${ASTC_VULKAN_D2_H5_WIDTH}x5-device-smoke PRIVATE
                    astc-vulkan-gpu-encoder-verify)
            endif()
        endforeach()
    endif()

    add_executable(astc-vulkan-artifact-decode
        astc-vulkan-artifact-decode.cpp)
    target_link_libraries(astc-vulkan-artifact-decode PRIVATE astcenc::astcenc-static)
    target_compile_features(astc-vulkan-artifact-decode PRIVATE cxx_std_17)

    add_executable(astc-vulkan-astcenc-smoke
        astc-vulkan-astcenc-smoke.cpp)
    target_include_directories(astc-vulkan-astcenc-smoke PRIVATE
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_link_libraries(astc-vulkan-astcenc-smoke PRIVATE astcenc::astcenc-static)
    target_compile_features(astc-vulkan-astcenc-smoke PRIVATE cxx_std_17)

    add_executable(astc-vulkan-encoder-track-smoke
        astc-vulkan-encoder-track-smoke.cpp)
    target_include_directories(astc-vulkan-encoder-track-smoke PRIVATE
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_link_libraries(astc-vulkan-encoder-track-smoke PRIVATE
        astcenc::astcenc-static)
    target_compile_features(astc-vulkan-encoder-track-smoke PRIVATE cxx_std_17)

    add_executable(astc-vulkan-weight-smoke
        astc-vulkan-weight-smoke.cpp)
    target_include_directories(astc-vulkan-weight-smoke PRIVATE
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_link_libraries(astc-vulkan-weight-smoke PRIVATE astcenc::astcenc-static)
    target_compile_features(astc-vulkan-weight-smoke PRIVATE cxx_std_17)

    add_executable(astc-vulkan-level-smoke
        astc-vulkan-level-smoke.cpp)
    target_include_directories(astc-vulkan-level-smoke PRIVATE
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_link_libraries(astc-vulkan-level-smoke PRIVATE astcenc::astcenc-static)
    target_compile_features(astc-vulkan-level-smoke PRIVATE cxx_std_17)

    add_executable(astc-vulkan-channel-semantic-smoke
        astc-vulkan-channel-semantic-smoke.cpp
        astc-vulkan-input.cpp)
    target_include_directories(astc-vulkan-channel-semantic-smoke PRIVATE
        ${ASTC_VULKAN_POC_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/ggml/include)
    target_link_libraries(astc-vulkan-channel-semantic-smoke PRIVATE
        astcenc::astcenc-static ggml-base)
    target_compile_features(astc-vulkan-channel-semantic-smoke PRIVATE cxx_std_17)

    add_executable(astc-vulkan-paired-smoke
        astc-vulkan-paired-smoke.cpp)
    target_include_directories(astc-vulkan-paired-smoke PRIVATE
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_link_libraries(astc-vulkan-paired-smoke PRIVATE
        astcenc::astcenc-static astc-vulkan-algorithms)
    target_compile_features(astc-vulkan-paired-smoke PRIVATE cxx_std_17)

    # Offline-only D2 research smoke. It does not serialize pairing or
    # transform metadata, and therefore cannot affect the runtime ABI.
    add_executable(astc-vulkan-d2-pair-transform-smoke
        astc-vulkan-d2-pair-transform-smoke.cpp
        astc-vulkan-input.cpp)
    target_include_directories(astc-vulkan-d2-pair-transform-smoke PRIVATE
        ${ASTC_VULKAN_POC_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/ggml/include)
    target_link_libraries(astc-vulkan-d2-pair-transform-smoke PRIVATE
        astcenc::astcenc-static astc-vulkan-algorithms ggml-base)
    target_compile_features(astc-vulkan-d2-pair-transform-smoke PRIVATE cxx_std_17)

    if (TARGET ggml-base)
        add_executable(astc-vulkan-paired-model-smoke
            astc-vulkan-paired-model-smoke.cpp
            astc-vulkan-input.cpp)
        target_include_directories(astc-vulkan-paired-model-smoke PRIVATE
            ${ASTC_VULKAN_POC_INCLUDE_DIR}
            ${CMAKE_SOURCE_DIR}/ggml/include)
        target_link_libraries(astc-vulkan-paired-model-smoke PRIVATE
            astcenc::astcenc-static astc-vulkan-algorithms ggml-base)
        target_compile_features(astc-vulkan-paired-model-smoke PRIVATE cxx_std_17)

        add_executable(astc-vulkan-paired-model-smoke-6x5
            astc-vulkan-paired-model-smoke.cpp
            astc-vulkan-input.cpp)
        target_include_directories(astc-vulkan-paired-model-smoke-6x5 PRIVATE
            ${ASTC_VULKAN_POC_INCLUDE_DIR}
            ${CMAKE_SOURCE_DIR}/ggml/include)
        target_link_libraries(astc-vulkan-paired-model-smoke-6x5 PRIVATE
            astcenc::astcenc-static astc-vulkan-algorithms ggml-base)
        target_compile_features(astc-vulkan-paired-model-smoke-6x5 PRIVATE cxx_std_17)
        target_compile_definitions(astc-vulkan-paired-model-smoke-6x5 PRIVATE
            ASTC_VULKAN_PAIRED_D2_6X5=1)

        add_executable(astc-vulkan-paired-model-smoke-10x5
            astc-vulkan-paired-model-smoke.cpp
            astc-vulkan-input.cpp)
        target_include_directories(astc-vulkan-paired-model-smoke-10x5 PRIVATE
            ${ASTC_VULKAN_POC_INCLUDE_DIR}
            ${CMAKE_SOURCE_DIR}/ggml/include)
        target_link_libraries(astc-vulkan-paired-model-smoke-10x5 PRIVATE
            astcenc::astcenc-static astc-vulkan-algorithms ggml-base)
        target_compile_features(astc-vulkan-paired-model-smoke-10x5 PRIVATE cxx_std_17)
        target_compile_definitions(astc-vulkan-paired-model-smoke-10x5 PRIVATE
            ASTC_VULKAN_PAIRED_D2_10X5=1)

        add_executable(astc-vulkan-paired-selection-smoke
            astc-vulkan-paired-selection-smoke.cpp
            astc-vulkan-input.cpp)
        target_include_directories(astc-vulkan-paired-selection-smoke PRIVATE
            ${ASTC_VULKAN_POC_INCLUDE_DIR}
            ${CMAKE_SOURCE_DIR}/ggml/include)
        target_link_libraries(astc-vulkan-paired-selection-smoke PRIVATE
            astcenc::astcenc-static astc-vulkan-algorithms ggml-base)
        target_compile_features(astc-vulkan-paired-selection-smoke PRIVATE cxx_std_17)
        if (TARGET astc-vulkan-gpu-encoder-dispatch AND
            TARGET astc-vulkan-gpu-encoder-finisher)
            target_include_directories(astc-vulkan-paired-selection-smoke PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/gpu-encoder/shared)
            target_link_libraries(astc-vulkan-paired-selection-smoke PRIVATE
                astc-vulkan-gpu-encoder-dispatch astc-vulkan-gpu-encoder-finisher)
            target_compile_definitions(astc-vulkan-paired-selection-smoke PRIVATE
                ASTC_VULKAN_GPU_EXACT_D2_BACKEND=1)
        endif()

        add_executable(astc-vulkan-paired-selection-smoke-6x5
            astc-vulkan-paired-selection-smoke.cpp
            astc-vulkan-input.cpp)
        target_include_directories(astc-vulkan-paired-selection-smoke-6x5 PRIVATE
            ${ASTC_VULKAN_POC_INCLUDE_DIR}
            ${CMAKE_SOURCE_DIR}/ggml/include)
        target_link_libraries(astc-vulkan-paired-selection-smoke-6x5 PRIVATE
            astcenc::astcenc-static astc-vulkan-algorithms ggml-base)
        target_compile_features(astc-vulkan-paired-selection-smoke-6x5 PRIVATE cxx_std_17)
        target_compile_definitions(astc-vulkan-paired-selection-smoke-6x5 PRIVATE
            ASTC_VULKAN_PAIRED_D2_6X5=1)
        if (TARGET astc-vulkan-gpu-encoder-dispatch AND TARGET astc-vulkan-gpu-encoder-finisher)
            target_include_directories(astc-vulkan-paired-selection-smoke-6x5 PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/gpu-encoder/shared)
            target_link_libraries(astc-vulkan-paired-selection-smoke-6x5 PRIVATE
                astc-vulkan-gpu-encoder-dispatch astc-vulkan-gpu-encoder-finisher)
            target_compile_definitions(astc-vulkan-paired-selection-smoke-6x5 PRIVATE
                ASTC_VULKAN_GPU_EXACT_D2_BACKEND=1)
        endif()

        add_executable(astc-vulkan-paired-selection-smoke-10x5
            astc-vulkan-paired-selection-smoke.cpp
            astc-vulkan-input.cpp)
        target_include_directories(astc-vulkan-paired-selection-smoke-10x5 PRIVATE
            ${ASTC_VULKAN_POC_INCLUDE_DIR}
            ${CMAKE_SOURCE_DIR}/ggml/include)
        target_link_libraries(astc-vulkan-paired-selection-smoke-10x5 PRIVATE
            astcenc::astcenc-static astc-vulkan-algorithms ggml-base)
        target_compile_features(astc-vulkan-paired-selection-smoke-10x5 PRIVATE cxx_std_17)
        target_compile_definitions(astc-vulkan-paired-selection-smoke-10x5 PRIVATE
            ASTC_VULKAN_PAIRED_D2_10X5=1)
        if (TARGET astc-vulkan-gpu-encoder-dispatch AND TARGET astc-vulkan-gpu-encoder-finisher)
            target_include_directories(astc-vulkan-paired-selection-smoke-10x5 PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/gpu-encoder/shared)
            target_link_libraries(astc-vulkan-paired-selection-smoke-10x5 PRIVATE
                astc-vulkan-gpu-encoder-dispatch astc-vulkan-gpu-encoder-finisher)
            target_compile_definitions(astc-vulkan-paired-selection-smoke-10x5 PRIVATE
                ASTC_VULKAN_GPU_EXACT_D2_BACKEND=1)
        endif()

        if (GGML_VK_ASTC_EXPERIMENTAL_NEURAL_RANK)
            if (NOT EXISTS "${GGML_VK_ASTC_NEURAL_RANK_LIBRARY}" OR
                NOT EXISTS "${GGML_VK_ASTC_NEURAL_RANK_INCLUDE_DIR}/astcenc.h")
                message(FATAL_ERROR
                    "GGML_VK_ASTC_EXPERIMENTAL_NEURAL_RANK requires an isolated "
                    "ASTC fork library and include directory")
            endif()
            add_executable(astc-vulkan-paired-selection-smoke-neural
                astc-vulkan-paired-selection-smoke.cpp
                astc-vulkan-input.cpp)
            target_include_directories(astc-vulkan-paired-selection-smoke-neural BEFORE PRIVATE
                ${GGML_VK_ASTC_NEURAL_RANK_INCLUDE_DIR}
                ${ASTC_VULKAN_POC_INCLUDE_DIR}
                ${CMAKE_SOURCE_DIR}/ggml/include)
            target_link_libraries(astc-vulkan-paired-selection-smoke-neural PRIVATE
                ${GGML_VK_ASTC_NEURAL_RANK_LIBRARY} astc-vulkan-algorithms ggml-base)
            target_compile_features(astc-vulkan-paired-selection-smoke-neural PRIVATE cxx_std_17)
            target_compile_definitions(astc-vulkan-paired-selection-smoke-neural PRIVATE
                ASTC_VULKAN_PAIRED_NEURAL_ENCODER=1)

            add_executable(astc-vulkan-paired-selection-smoke-neural-6x5
                astc-vulkan-paired-selection-smoke.cpp
                astc-vulkan-input.cpp)
            target_include_directories(astc-vulkan-paired-selection-smoke-neural-6x5 BEFORE PRIVATE
                ${GGML_VK_ASTC_NEURAL_RANK_INCLUDE_DIR}
                ${ASTC_VULKAN_POC_INCLUDE_DIR}
                ${CMAKE_SOURCE_DIR}/ggml/include)
            target_link_libraries(astc-vulkan-paired-selection-smoke-neural-6x5 PRIVATE
                ${GGML_VK_ASTC_NEURAL_RANK_LIBRARY} astc-vulkan-algorithms ggml-base)
            target_compile_features(astc-vulkan-paired-selection-smoke-neural-6x5 PRIVATE cxx_std_17)
            target_compile_definitions(astc-vulkan-paired-selection-smoke-neural-6x5 PRIVATE
                ASTC_VULKAN_PAIRED_NEURAL_ENCODER=1
                ASTC_VULKAN_PAIRED_D2_6X5=1)

            # Isolated research target: physical ASTC remains 8x5, while the
            # offline semantic map swaps the reduction and paired-row axes.
            # It is intentionally not installed or used by cache/runtime code.
            add_executable(astc-vulkan-paired-selection-smoke-neural-8x5-transposed
                astc-vulkan-paired-selection-smoke.cpp
                astc-vulkan-input.cpp)
            target_include_directories(astc-vulkan-paired-selection-smoke-neural-8x5-transposed BEFORE PRIVATE
                ${GGML_VK_ASTC_NEURAL_RANK_INCLUDE_DIR}
                ${ASTC_VULKAN_POC_INCLUDE_DIR}
                ${CMAKE_SOURCE_DIR}/ggml/include)
            target_link_libraries(astc-vulkan-paired-selection-smoke-neural-8x5-transposed PRIVATE
                ${GGML_VK_ASTC_NEURAL_RANK_LIBRARY} astc-vulkan-algorithms ggml-base)
            target_compile_features(astc-vulkan-paired-selection-smoke-neural-8x5-transposed PRIVATE cxx_std_17)
            target_compile_definitions(astc-vulkan-paired-selection-smoke-neural-8x5-transposed PRIVATE
                ASTC_VULKAN_PAIRED_NEURAL_ENCODER=1
                ASTC_VULKAN_PAIRED_D2_TRANSPOSED=1)

            add_executable(astc-vulkan-paired-selection-smoke-neural-10x5
                astc-vulkan-paired-selection-smoke.cpp
                astc-vulkan-input.cpp)
            target_include_directories(astc-vulkan-paired-selection-smoke-neural-10x5 BEFORE PRIVATE
                ${GGML_VK_ASTC_NEURAL_RANK_INCLUDE_DIR}
                ${ASTC_VULKAN_POC_INCLUDE_DIR}
                ${CMAKE_SOURCE_DIR}/ggml/include)
            target_link_libraries(astc-vulkan-paired-selection-smoke-neural-10x5 PRIVATE
                ${GGML_VK_ASTC_NEURAL_RANK_LIBRARY} astc-vulkan-algorithms ggml-base)
            target_compile_features(astc-vulkan-paired-selection-smoke-neural-10x5 PRIVATE cxx_std_17)
            target_compile_definitions(astc-vulkan-paired-selection-smoke-neural-10x5 PRIVATE
                ASTC_VULKAN_PAIRED_NEURAL_ENCODER=1
                ASTC_VULKAN_PAIRED_D2_10X5=1)
        endif()
    endif()

    # One logical weight per texel, stored either as a scalar replicated across
    # RGBA or as a luminance-plus-alpha additive latent. The tool reports the
    # encoder's actual dual-plane decisions; it never assumes one was used.
    add_executable(astc-vulkan-latent-smoke
        astc-vulkan-latent-smoke.cpp
        astc-vulkan-input.cpp)
    target_include_directories(astc-vulkan-latent-smoke PRIVATE
        ${ASTC_VULKAN_POC_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/ggml/include)
    if (GGML_VK_ASTC_EXPERIMENTAL_NEURAL_RANK)
        if (NOT EXISTS "${GGML_VK_ASTC_NEURAL_RANK_LIBRARY}" OR
            NOT EXISTS "${GGML_VK_ASTC_NEURAL_RANK_INCLUDE_DIR}/astcenc.h")
            message(FATAL_ERROR
                "GGML_VK_ASTC_EXPERIMENTAL_NEURAL_RANK requires an isolated "
                "ASTC fork library and include directory")
        endif()
        target_include_directories(astc-vulkan-latent-smoke BEFORE PRIVATE
            ${GGML_VK_ASTC_NEURAL_RANK_INCLUDE_DIR})
    target_link_libraries(astc-vulkan-latent-smoke PRIVATE
            ${GGML_VK_ASTC_NEURAL_RANK_LIBRARY} ggml-base astc-vulkan-algorithms)
    else()
        target_link_libraries(astc-vulkan-latent-smoke PRIVATE
        astcenc::astcenc-static ggml-base astc-vulkan-algorithms)
    endif()
    target_compile_features(astc-vulkan-latent-smoke PRIVATE cxx_std_17)
    if (TARGET astc-vulkan-d1-prescreen-dispatch)
        target_link_libraries(astc-vulkan-latent-smoke PRIVATE astc-vulkan-d1-prescreen-dispatch)
        target_compile_definitions(astc-vulkan-latent-smoke PRIVATE ASTC_VULKAN_D1_PRESCREEN_GPU=1)
    endif()
    if (TARGET astc-vulkan-gpu-encoder-dispatch AND
        TARGET astc-vulkan-gpu-encoder-finisher)
        target_include_directories(astc-vulkan-latent-smoke PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/gpu-encoder/shared)
        target_link_libraries(astc-vulkan-latent-smoke PRIVATE
            astc-vulkan-gpu-encoder-dispatch astc-vulkan-gpu-encoder-finisher)
        target_compile_definitions(astc-vulkan-latent-smoke PRIVATE
            ASTC_VULKAN_GPU_ENCODER_BACKEND=1)
    endif()
    if (GGML_VK_ASTC_EXPERIMENTAL_NEURAL_RANK)
        target_compile_definitions(astc-vulkan-latent-smoke PRIVATE
            GGML_VK_ASTC_EXPERIMENTAL_NEURAL_RANK=1)
    endif()
    if (TARGET ggml-base)
        add_executable(astc-vulkan-quality-smoke
            astc-vulkan-quality-smoke.cpp
            astc-vulkan-input.cpp)
        target_include_directories(astc-vulkan-quality-smoke PRIVATE
            ${ASTC_VULKAN_POC_INCLUDE_DIR}
            ${CMAKE_SOURCE_DIR}/ggml/include)
        target_link_libraries(astc-vulkan-quality-smoke PRIVATE
            astcenc::astcenc-static ggml-base)
        target_compile_features(astc-vulkan-quality-smoke PRIVATE cxx_std_17)

        add_executable(astc-vulkan-native-quant-smoke
            astc-vulkan-native-quant-smoke.cpp
            astc-vulkan-input.cpp)
        target_include_directories(astc-vulkan-native-quant-smoke PRIVATE
            ${ASTC_VULKAN_POC_INCLUDE_DIR}
            ${CMAKE_SOURCE_DIR}/ggml/include)
        target_link_libraries(astc-vulkan-native-quant-smoke PRIVATE
            astcenc::astcenc-static ggml-base)
        target_compile_features(astc-vulkan-native-quant-smoke PRIVATE cxx_std_17)
    endif()
endif()

# The input reader is deliberately independent from Vulkan and astcenc.  It
# uses the same ggml/gguf code as llama.cpp to turn real F16 tensors and
# captured activation traces into the float matrix contract used by the PoC.
if (TARGET ggml-base)
    add_executable(astc-vulkan-input-smoke
        astc-vulkan-input-smoke.cpp
        astc-vulkan-input.cpp)
    target_include_directories(astc-vulkan-input-smoke PRIVATE
        ${ASTC_VULKAN_POC_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/ggml/include)
    target_link_libraries(astc-vulkan-input-smoke PRIVATE ggml-base)
    target_compile_features(astc-vulkan-input-smoke PRIVATE cxx_std_17)

    # Compare existing packed GGUF formats against the same F16 source. This
    # remains an offline research measurement; it does not modify ggml-vulkan.
    add_executable(astc-vulkan-format-smoke
        astc-vulkan-format-smoke.cpp
        astc-vulkan-input.cpp)
    target_include_directories(astc-vulkan-format-smoke PRIVATE
        ${ASTC_VULKAN_POC_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/ggml/include)
    target_link_libraries(astc-vulkan-format-smoke PRIVATE ggml-base)
    target_compile_features(astc-vulkan-format-smoke PRIVATE cxx_std_17)

    add_executable(astc-vulkan-quant-export
        astc-vulkan-quant-export.cpp
        astc-vulkan-input.cpp)
    target_include_directories(astc-vulkan-quant-export PRIVATE
        ${ASTC_VULKAN_POC_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/ggml/include)
    target_link_libraries(astc-vulkan-quant-export PRIVATE ggml-base)
    target_compile_features(astc-vulkan-quant-export PRIVATE cxx_std_17)

    # Reproducibly constructs a multi-prompt activation corpus without changing
    # the versioned trace representation used by selector and replay tools.
    add_executable(astc-vulkan-trace-merge
        astc-vulkan-trace-merge.cpp
        astc-vulkan-input.cpp)
    target_include_directories(astc-vulkan-trace-merge PRIVATE
        ${ASTC_VULKAN_POC_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/ggml/include)
    target_link_libraries(astc-vulkan-trace-merge PRIVATE ggml-base)
    target_compile_features(astc-vulkan-trace-merge PRIVATE cxx_std_17)

    # This tool intentionally uses the current internal layer-input capture
    # extension. It stays in the PoC until an upstream-facing capture API is
    # justified by the measurements it enables.
    if (TARGET llama AND TARGET llama-common)
        add_executable(astc-vulkan-trace-capture
            astc-vulkan-trace-capture.cpp
            astc-vulkan-input.cpp)
        target_include_directories(astc-vulkan-trace-capture PRIVATE
            ${ASTC_VULKAN_POC_INCLUDE_DIR}
            ${CMAKE_SOURCE_DIR}/ggml/include
            ${CMAKE_SOURCE_DIR}/src)
        target_link_libraries(astc-vulkan-trace-capture PRIVATE
            llama-common llama ggml-base ${CMAKE_THREAD_LIBS_INIT})
        target_compile_features(astc-vulkan-trace-capture PRIVATE cxx_std_17)

        add_executable(astc-vulkan-model-replay-smoke
            astc-vulkan-model-replay-smoke.cpp
            astc-vulkan-input.cpp
            astc-vulkan-format.cpp
            astc-vulkan-paired.cpp
            astc-vulkan-paired-layout.cpp)
        target_include_directories(astc-vulkan-model-replay-smoke PRIVATE
            ${ASTC_VULKAN_POC_INCLUDE_DIR}
            ${CMAKE_SOURCE_DIR}/ggml/include
            ${CMAKE_SOURCE_DIR}/src)
        target_link_libraries(astc-vulkan-model-replay-smoke PRIVATE
            llama-common llama ggml-base ${CMAKE_THREAD_LIBS_INIT})
        target_compile_features(astc-vulkan-model-replay-smoke PRIVATE cxx_std_17)
        if (TARGET astc-vulkan-scheduler-adapter)
            target_link_libraries(astc-vulkan-model-replay-smoke PRIVATE
                astc-vulkan-scheduler-adapter astc-vulkan-stream-loader)
            target_compile_definitions(astc-vulkan-model-replay-smoke PRIVATE
                ASTC_VULKAN_MODEL_REPLAY_GPU=1)
            if (TARGET astcenc::astcenc-static)
                target_link_libraries(astc-vulkan-model-replay-smoke PRIVATE
                    astcenc::astcenc-static)
            endif()
        endif()

        # Public, portable replay entry point.  It intentionally shares the
        # proven replay implementation with the smoke target; the latter is
        # retained for regression fixtures and historical scripts.
        add_executable(llama-astc-replay
            astc-vulkan-model-replay-smoke.cpp
            astc-vulkan-input.cpp
            astc-vulkan-format.cpp
            astc-vulkan-paired.cpp
            astc-vulkan-paired-layout.cpp)
        target_include_directories(llama-astc-replay PRIVATE
            ${ASTC_VULKAN_POC_INCLUDE_DIR}
            ${CMAKE_SOURCE_DIR}/ggml/include
            ${CMAKE_SOURCE_DIR}/src)
        target_link_libraries(llama-astc-replay PRIVATE
            llama-common llama ggml-base ${CMAKE_THREAD_LIBS_INIT})
        target_compile_features(llama-astc-replay PRIVATE cxx_std_17)
        if (TARGET astc-vulkan-scheduler-adapter)
            target_link_libraries(llama-astc-replay PRIVATE
                astc-vulkan-scheduler-adapter astc-vulkan-stream-loader)
            target_compile_definitions(llama-astc-replay PRIVATE
                ASTC_VULKAN_MODEL_REPLAY_GPU=1)
            if (TARGET astcenc::astcenc-static)
                target_link_libraries(llama-astc-replay PRIVATE
                    astcenc::astcenc-static)
            endif()
        endif()

        add_executable(astc-vulkan-yaqa-replay-smoke
            astc-vulkan-yaqa-replay-smoke.cpp
            astc-vulkan-input.cpp)
        target_include_directories(astc-vulkan-yaqa-replay-smoke PRIVATE
            ${ASTC_VULKAN_POC_INCLUDE_DIR}
            ${CMAKE_SOURCE_DIR}/ggml/include)
        target_link_libraries(astc-vulkan-yaqa-replay-smoke PRIVATE
            astc-vulkan-algorithms ggml-base)
        target_compile_features(astc-vulkan-yaqa-replay-smoke PRIVATE cxx_std_17)

        add_executable(astc-vulkan-model-baseline-smoke
            astc-vulkan-model-baseline-smoke.cpp)
        target_include_directories(astc-vulkan-model-baseline-smoke PRIVATE
            ${ASTC_VULKAN_POC_INCLUDE_DIR}
            ${CMAKE_SOURCE_DIR}/ggml/include
            ${CMAKE_SOURCE_DIR}/src)
        target_link_libraries(astc-vulkan-model-baseline-smoke PRIVATE
            llama-common llama ggml-base ${CMAKE_THREAD_LIBS_INIT})
        target_compile_features(astc-vulkan-model-baseline-smoke PRIVATE cxx_std_17)
    endif()
endif()

find_program(ASTC_VULKAN_GLSLC glslc)
if (ASTC_VULKAN_GLSLC)
    set(ASTC_VULKAN_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/astc-validate.comp.spv)
    set(ASTC_VULKAN_MATVEC_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/astc-matvec.comp.spv)
    set(ASTC_VULKAN_PAIRED_MATVEC_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/astc-paired-matvec.comp.spv)
    set(ASTC_VULKAN_PAIRED_CANDIDATE_DELTA_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/astc-paired-candidate-delta.comp.spv)
    set(ASTC_VULKAN_PAIRED_PROPOSAL_GAIN_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/astc-paired-proposal-gain.comp.spv)
    set(ASTC_VULKAN_D1_CANDIDATE_DELTA_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/astc-d1-candidate-delta.comp.spv)
    set(ASTC_VULKAN_LOCAL_SELECT_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/astc-local-select.comp.spv)
    set(ASTC_VULKAN_YAQA_TRACE_SCORE_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/yaqa-trace-score.comp.spv)
    set(ASTC_VULKAN_YAQA_TRACE_PARTIAL_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/yaqa-trace-partial.comp.spv)
    set(ASTC_VULKAN_YAQA_TRACE_REDUCE_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/yaqa-trace-reduce.comp.spv)
    set(ASTC_VULKAN_BUFFER_MATVEC_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/buffer-matvec.comp.spv)
    set(ASTC_VULKAN_Q4_MATVEC_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/q4-matvec.comp.spv)
    set(ASTC_VULKAN_TQ2_MATVEC_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/tq2-matvec.comp.spv)
    set(ASTC_VULKAN_Q3_MATVEC_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/q3-k-matvec.comp.spv)
    set(ASTC_VULKAN_TQ1_MATVEC_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/tq1-matvec.comp.spv)
    set(ASTC_VULKAN_SAMPLED_F32_MATVEC_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/sampled-f32-matvec.comp.spv)
    set(ASTC_VULKAN_FFN_E2E_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/astc-ffn-matvec.comp.spv)
    set(ASTC_VULKAN_D1_PRESCREEN_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/d1-prescreen.comp.spv)
    set(ASTC_VULKAN_D2_PRESCREEN_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/d2-prescreen.comp.spv)
    set(ASTC_VULKAN_GPU_D1_PROPOSER_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/astc-gpu-d1-proposer.comp.spv)
    set(ASTC_VULKAN_GPU_D1_PROPOSER_6X6_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/astc-gpu-d1-proposer-6x6.comp.spv)
    set(ASTC_VULKAN_GPU_D1_PROPOSER_8X6_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/astc-gpu-d1-proposer-8x6.comp.spv)
    set(ASTC_VULKAN_GPU_D2_PROPOSER_8X5_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/astc-gpu-d2-proposer-8x5.comp.spv)
    set(ASTC_VULKAN_GPU_D2_PROPOSER_6X5_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/astc-gpu-d2-proposer-6x5.comp.spv)
    set(ASTC_VULKAN_GPU_D2_PROPOSER_10X5_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/astc-gpu-d2-proposer-10x5.comp.spv)
    set(ASTC_VULKAN_GPU_D1_PROPOSER_5X5_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/astc-gpu-d1-proposer-5x5.comp.spv)
    set(ASTC_VULKAN_GPU_EXACT_VOID_EXTENT_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/astc-gpu-exact-void-extent.comp.spv)
    set(ASTC_VULKAN_GPU_EXACT_D1_LUMINANCE_BINARY_6X6_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/astc-gpu-exact-d1-luminance-binary-6x6.comp.spv)
    set(ASTC_VULKAN_GPU_EXACT_D1_LUMINANCE_REFINED_6X6_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/astc-gpu-exact-d1-luminance-refined-6x6.comp.spv)
    set(ASTC_VULKAN_GPU_EXACT_D1_LUMINANCE_MEAN_REFINED_6X6_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/astc-gpu-exact-d1-luminance-mean-refined-6x6.comp.spv)
    set(ASTC_VULKAN_GPU_EXACT_D1_LUMINANCE_QUANTILE_REFINED_6X6_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/astc-gpu-exact-d1-luminance-quantile-refined-6x6.comp.spv)
    set(ASTC_VULKAN_GPU_EXACT_D1_LUMINANCE_BINARY_5X5_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/astc-gpu-exact-d1-luminance-binary-5x5.comp.spv)
    set(ASTC_VULKAN_GPU_EXACT_D1_LUMINANCE_QUANT4_4X4_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/astc-gpu-exact-d1-luminance-quant4-4x4.comp.spv)
    set(ASTC_VULKAN_GPU_EXACT_LA_BINARY_8X5_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/astc-gpu-exact-la-binary-8x5.comp.spv)
    set(ASTC_VULKAN_GPU_EXACT_LA_LUMINANCE_8X5_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/astc-gpu-exact-la-luminance-8x5.comp.spv)
    set(ASTC_VULKAN_GPU_EXACT_LA_ALPHA_8X5_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/astc-gpu-exact-la-alpha-8x5.comp.spv)
    set(ASTC_VULKAN_GPU_EXACT_LA_REFINED_8X5_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/astc-gpu-exact-la-refined-8x5.comp.spv)
    set(ASTC_VULKAN_GPU_EXACT_LA_MEAN_REFINED_8X5_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/astc-gpu-exact-la-mean-refined-8x5.comp.spv)
    set(ASTC_VULKAN_GPU_EXACT_LA_QUANTILE_REFINED_8X5_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/astc-gpu-exact-la-quantile-refined-8x5.comp.spv)
    set(ASTC_VULKAN_GPU_EXACT_LA_DUAL_BINARY_8X5_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/astc-gpu-exact-la-dual-binary-8x5.comp.spv)
    set(ASTC_VULKAN_GPU_EXACT_LA_BINARY_6X5_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/astc-gpu-exact-la-binary-6x5.comp.spv)
    set(ASTC_VULKAN_GPU_EXACT_LA_DUAL_BINARY_6X5_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/astc-gpu-exact-la-dual-binary-6x5.comp.spv)
    set(ASTC_VULKAN_GPU_EXACT_LA_BINARY_10X5_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/astc-gpu-exact-la-binary-10x5.comp.spv)
    set(ASTC_VULKAN_GPU_EXACT_LA_DUAL_BINARY_10X5_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/astc-gpu-exact-la-dual-binary-10x5.comp.spv)
    set(ASTC_VULKAN_GPU_EXACT_PACK_INCLUDE
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-exact-pack.glsl)
    add_custom_command(
        OUTPUT ${ASTC_VULKAN_SHADER_OUTPUT}
        COMMAND ${ASTC_VULKAN_GLSLC}
            --target-env=vulkan1.1
            -fshader-stage=compute
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-validate.comp
            -o ${ASTC_VULKAN_SHADER_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-validate.comp
        COMMENT "Compiling ASTC Vulkan validation shader")
    add_custom_target(astc-vulkan-shader-smoke
        DEPENDS ${ASTC_VULKAN_SHADER_OUTPUT})
    add_custom_command(
        OUTPUT ${ASTC_VULKAN_MATVEC_SHADER_OUTPUT}
        COMMAND ${ASTC_VULKAN_GLSLC}
            --target-env=vulkan1.1
            -fshader-stage=compute
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-matvec.comp
            -o ${ASTC_VULKAN_MATVEC_SHADER_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-matvec.comp
        COMMENT "Compiling ASTC Vulkan matvec shader")
    add_custom_target(astc-vulkan-matvec-shader
        DEPENDS ${ASTC_VULKAN_MATVEC_SHADER_OUTPUT})
    add_custom_command(
        OUTPUT ${ASTC_VULKAN_PAIRED_MATVEC_SHADER_OUTPUT}
        COMMAND ${ASTC_VULKAN_GLSLC}
            --target-env=vulkan1.1
            -fshader-stage=compute
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-paired-matvec.comp
            -o ${ASTC_VULKAN_PAIRED_MATVEC_SHADER_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-paired-matvec.comp
        COMMENT "Compiling experimental paired-D2 ASTC Vulkan matvec shader")
    add_custom_target(astc-vulkan-paired-matvec-shader
        DEPENDS ${ASTC_VULKAN_PAIRED_MATVEC_SHADER_OUTPUT})

    # Both replay entry points load the paired-D2 SPIR-V at runtime. Keep the
    # executable target dependent on it so a source change cannot leave a
    # stale shader beside an otherwise freshly rebuilt replay binary.
    if (TARGET astc-vulkan-model-replay-smoke)
        add_dependencies(astc-vulkan-model-replay-smoke
            astc-vulkan-paired-matvec-shader)
    endif()
    if (TARGET llama-astc-replay)
        add_dependencies(llama-astc-replay
            astc-vulkan-paired-matvec-shader)
    endif()
    add_custom_command(
        OUTPUT ${ASTC_VULKAN_PAIRED_CANDIDATE_DELTA_SHADER_OUTPUT}
        COMMAND ${ASTC_VULKAN_GLSLC}
            --target-env=vulkan1.1
            -fshader-stage=compute
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-paired-candidate-delta.comp
            -o ${ASTC_VULKAN_PAIRED_CANDIDATE_DELTA_SHADER_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-paired-candidate-delta.comp
        COMMENT "Compiling experimental paired-D2 GPU candidate-delta shader")
    add_custom_target(astc-vulkan-paired-candidate-delta-shader
        DEPENDS ${ASTC_VULKAN_PAIRED_CANDIDATE_DELTA_SHADER_OUTPUT})
    add_custom_command(
        OUTPUT ${ASTC_VULKAN_PAIRED_PROPOSAL_GAIN_SHADER_OUTPUT}
        COMMAND ${ASTC_VULKAN_GLSLC}
            --target-env=vulkan1.1
            -fshader-stage=compute
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-paired-proposal-gain.comp
            -o ${ASTC_VULKAN_PAIRED_PROPOSAL_GAIN_SHADER_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-paired-proposal-gain.comp
        COMMENT "Compiling experimental paired-D2 GPU proposal-gain shader")
    add_custom_target(astc-vulkan-paired-proposal-gain-shader
        DEPENDS ${ASTC_VULKAN_PAIRED_PROPOSAL_GAIN_SHADER_OUTPUT})
    add_custom_command(
        OUTPUT ${ASTC_VULKAN_D1_CANDIDATE_DELTA_SHADER_OUTPUT}
        COMMAND ${ASTC_VULKAN_GLSLC}
            --target-env=vulkan1.1
            -fshader-stage=compute
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-d1-candidate-delta.comp
            -o ${ASTC_VULKAN_D1_CANDIDATE_DELTA_SHADER_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-d1-candidate-delta.comp
        COMMENT "Compiling experimental D1 GPU candidate-delta shader")
    add_custom_target(astc-vulkan-d1-candidate-delta-shader
        DEPENDS ${ASTC_VULKAN_D1_CANDIDATE_DELTA_SHADER_OUTPUT})
    add_custom_command(
        OUTPUT ${ASTC_VULKAN_LOCAL_SELECT_SHADER_OUTPUT}
        COMMAND ${ASTC_VULKAN_GLSLC}
            --target-env=vulkan1.1
            -fshader-stage=compute
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-local-select.comp
            -o ${ASTC_VULKAN_LOCAL_SELECT_SHADER_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-local-select.comp
        COMMENT "Compiling experimental GPU local-selection shader")
    add_custom_target(astc-vulkan-local-select-shader
        DEPENDS ${ASTC_VULKAN_LOCAL_SELECT_SHADER_OUTPUT})
    add_custom_command(
        OUTPUT ${ASTC_VULKAN_YAQA_TRACE_SCORE_SHADER_OUTPUT}
        COMMAND ${ASTC_VULKAN_GLSLC}
            --target-env=vulkan1.1
            -fshader-stage=compute
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/yaqa-trace-score.comp
            -o ${ASTC_VULKAN_YAQA_TRACE_SCORE_SHADER_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/yaqa-trace-score.comp
        COMMENT "Compiling YAQA GPU trace-score smoke shader")
    add_custom_target(astc-vulkan-yaqa-trace-score-shader
        DEPENDS ${ASTC_VULKAN_YAQA_TRACE_SCORE_SHADER_OUTPUT})
    add_custom_command(
        OUTPUT ${ASTC_VULKAN_YAQA_TRACE_PARTIAL_SHADER_OUTPUT}
        COMMAND ${ASTC_VULKAN_GLSLC} --target-env=vulkan1.1 -fshader-stage=compute
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/yaqa-trace-partial.comp
            -o ${ASTC_VULKAN_YAQA_TRACE_PARTIAL_SHADER_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/yaqa-trace-partial.comp
        COMMENT "Compiling scalable YAQA partial-score shader")
    add_custom_target(astc-vulkan-yaqa-trace-partial-shader
        DEPENDS ${ASTC_VULKAN_YAQA_TRACE_PARTIAL_SHADER_OUTPUT})
    add_custom_command(
        OUTPUT ${ASTC_VULKAN_YAQA_TRACE_REDUCE_SHADER_OUTPUT}
        COMMAND ${ASTC_VULKAN_GLSLC} --target-env=vulkan1.1 -fshader-stage=compute
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/yaqa-trace-reduce.comp
            -o ${ASTC_VULKAN_YAQA_TRACE_REDUCE_SHADER_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/yaqa-trace-reduce.comp
        COMMENT "Compiling scalable YAQA reduction shader")
    add_custom_target(astc-vulkan-yaqa-trace-reduce-shader
        DEPENDS ${ASTC_VULKAN_YAQA_TRACE_REDUCE_SHADER_OUTPUT})
    add_custom_command(
        OUTPUT ${ASTC_VULKAN_BUFFER_MATVEC_SHADER_OUTPUT}
        COMMAND ${ASTC_VULKAN_GLSLC}
            --target-env=vulkan1.1
            -fshader-stage=compute
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/buffer-matvec.comp
            -o ${ASTC_VULKAN_BUFFER_MATVEC_SHADER_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/buffer-matvec.comp
        COMMENT "Compiling buffer Vulkan matvec shader")
    add_custom_target(astc-vulkan-buffer-matvec-shader
        DEPENDS ${ASTC_VULKAN_BUFFER_MATVEC_SHADER_OUTPUT})
    add_custom_command(
        OUTPUT ${ASTC_VULKAN_Q4_MATVEC_SHADER_OUTPUT}
        COMMAND ${ASTC_VULKAN_GLSLC}
            --target-env=vulkan1.1 -fshader-stage=compute
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/q4-matvec.comp
            -o ${ASTC_VULKAN_Q4_MATVEC_SHADER_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/q4-matvec.comp
        COMMENT "Compiling Q4 Vulkan matvec shader")
    add_custom_target(astc-vulkan-q4-matvec-shader
        DEPENDS ${ASTC_VULKAN_Q4_MATVEC_SHADER_OUTPUT})
    add_custom_command(
        OUTPUT ${ASTC_VULKAN_TQ2_MATVEC_SHADER_OUTPUT}
        COMMAND ${ASTC_VULKAN_GLSLC}
            --target-env=vulkan1.1 -fshader-stage=compute
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/tq2-matvec.comp
            -o ${ASTC_VULKAN_TQ2_MATVEC_SHADER_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/tq2-matvec.comp
        COMMENT "Compiling TQ2 Vulkan matvec shader")
    add_custom_target(astc-vulkan-tq2-matvec-shader
        DEPENDS ${ASTC_VULKAN_TQ2_MATVEC_SHADER_OUTPUT})
    add_custom_command(
        OUTPUT ${ASTC_VULKAN_Q3_MATVEC_SHADER_OUTPUT}
        COMMAND ${ASTC_VULKAN_GLSLC}
            --target-env=vulkan1.1 -fshader-stage=compute
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/q3-k-matvec.comp
            -o ${ASTC_VULKAN_Q3_MATVEC_SHADER_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/q3-k-matvec.comp
        COMMENT "Compiling Q3_K Vulkan matvec shader")
    add_custom_target(astc-vulkan-q3-matvec-shader
        DEPENDS ${ASTC_VULKAN_Q3_MATVEC_SHADER_OUTPUT})
    add_custom_command(
        OUTPUT ${ASTC_VULKAN_TQ1_MATVEC_SHADER_OUTPUT}
        COMMAND ${ASTC_VULKAN_GLSLC}
            --target-env=vulkan1.1 -fshader-stage=compute
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/tq1-matvec.comp
            -o ${ASTC_VULKAN_TQ1_MATVEC_SHADER_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/tq1-matvec.comp
        COMMENT "Compiling TQ1 Vulkan matvec shader")
    add_custom_target(astc-vulkan-tq1-matvec-shader
        DEPENDS ${ASTC_VULKAN_TQ1_MATVEC_SHADER_OUTPUT})
    add_custom_command(
        OUTPUT ${ASTC_VULKAN_SAMPLED_F32_MATVEC_SHADER_OUTPUT}
        COMMAND ${ASTC_VULKAN_GLSLC}
            --target-env=vulkan1.1 -fshader-stage=compute
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sampled-f32-matvec.comp
            -o ${ASTC_VULKAN_SAMPLED_F32_MATVEC_SHADER_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sampled-f32-matvec.comp
        COMMENT "Compiling sampled FP32 Vulkan matvec shader")
    add_custom_target(astc-vulkan-sampled-f32-matvec-shader
        DEPENDS ${ASTC_VULKAN_SAMPLED_F32_MATVEC_SHADER_OUTPUT})
    add_custom_command(
        OUTPUT ${ASTC_VULKAN_FFN_E2E_SHADER_OUTPUT}
        COMMAND ${ASTC_VULKAN_GLSLC}
            --target-env=vulkan1.1 -fshader-stage=compute
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-ffn-matvec.comp
            -o ${ASTC_VULKAN_FFN_E2E_SHADER_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-ffn-matvec.comp
        COMMENT "Compiling ASTC FFN end-to-end shader")
    add_custom_target(astc-vulkan-ffn-e2e-shader
        DEPENDS ${ASTC_VULKAN_FFN_E2E_SHADER_OUTPUT})
    add_custom_command(
        OUTPUT ${ASTC_VULKAN_D1_PRESCREEN_SHADER_OUTPUT}
        COMMAND ${ASTC_VULKAN_GLSLC}
            --target-env=vulkan1.1
            -fshader-stage=compute
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/d1-prescreen.comp
            -o ${ASTC_VULKAN_D1_PRESCREEN_SHADER_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/d1-prescreen.comp
        COMMENT "Compiling D1 GPU pre-screen shader")
    add_custom_target(astc-vulkan-d1-prescreen-shader
        DEPENDS ${ASTC_VULKAN_D1_PRESCREEN_SHADER_OUTPUT})
    add_custom_command(
        OUTPUT ${ASTC_VULKAN_D2_PRESCREEN_SHADER_OUTPUT}
        COMMAND ${ASTC_VULKAN_GLSLC}
            --target-env=vulkan1.1
            -fshader-stage=compute
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/d2-prescreen.comp
            -o ${ASTC_VULKAN_D2_PRESCREEN_SHADER_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/d2-prescreen.comp
        COMMENT "Compiling D2 GPU pre-screen shader")
    add_custom_target(astc-vulkan-d2-prescreen-shader
        DEPENDS ${ASTC_VULKAN_D2_PRESCREEN_SHADER_OUTPUT})
    if (TARGET astc-vulkan-cache-tool)
        add_dependencies(astc-vulkan-cache-tool astc-vulkan-d2-prescreen-shader)
        target_compile_definitions(astc-vulkan-cache-tool PRIVATE
            ASTC_VULKAN_D2_PRESCREEN_SHADER_PATH="${ASTC_VULKAN_D2_PRESCREEN_SHADER_OUTPUT}")
    endif()
    add_custom_command(
        OUTPUT ${ASTC_VULKAN_GPU_D1_PROPOSER_SHADER_OUTPUT}
        COMMAND ${ASTC_VULKAN_GLSLC}
            --target-env=vulkan1.1
            -fshader-stage=compute
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-d1-proposer.comp
            -o ${ASTC_VULKAN_GPU_D1_PROPOSER_SHADER_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-d1-proposer.comp
        COMMENT "Compiling offline D1 GPU ASTC proposer shader")
    add_custom_target(astc-vulkan-gpu-d1-proposer-shader
        DEPENDS ${ASTC_VULKAN_GPU_D1_PROPOSER_SHADER_OUTPUT})
    add_custom_command(
        OUTPUT ${ASTC_VULKAN_GPU_D1_PROPOSER_6X6_SHADER_OUTPUT}
        COMMAND ${ASTC_VULKAN_GLSLC}
            --target-env=vulkan1.1
            -fshader-stage=compute
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-d1-proposer-6x6.comp
            -o ${ASTC_VULKAN_GPU_D1_PROPOSER_6X6_SHADER_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-d1-proposer-6x6.comp
        COMMENT "Compiling offline D1 6x6 GPU ASTC proposer shader")
    add_custom_target(astc-vulkan-gpu-d1-proposer-6x6-shader
        DEPENDS ${ASTC_VULKAN_GPU_D1_PROPOSER_6X6_SHADER_OUTPUT})
    add_custom_command(
        OUTPUT ${ASTC_VULKAN_GPU_EXACT_VOID_EXTENT_SHADER_OUTPUT}
        COMMAND ${ASTC_VULKAN_GLSLC}
            --target-env=vulkan1.1
            -fshader-stage=compute
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-exact-void-extent.comp
            -o ${ASTC_VULKAN_GPU_EXACT_VOID_EXTENT_SHADER_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-exact-void-extent.comp
        COMMENT "Compiling offline exact ASTC void-extent shader")
    add_custom_target(astc-vulkan-gpu-exact-void-extent-shader
        DEPENDS ${ASTC_VULKAN_GPU_EXACT_VOID_EXTENT_SHADER_OUTPUT})
    add_custom_command(
        OUTPUT ${ASTC_VULKAN_GPU_EXACT_D1_LUMINANCE_BINARY_6X6_SHADER_OUTPUT}
        COMMAND ${ASTC_VULKAN_GLSLC}
            --target-env=vulkan1.1
            -fshader-stage=compute
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-exact-d1-luminance-binary-6x6.comp
            -o ${ASTC_VULKAN_GPU_EXACT_D1_LUMINANCE_BINARY_6X6_SHADER_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-exact-d1-luminance-binary-6x6.comp
            ${ASTC_VULKAN_GPU_EXACT_PACK_INCLUDE}
        COMMENT "Compiling exact D1 6x6 luminance ASTC subset shader")
    add_custom_target(astc-vulkan-gpu-exact-d1-luminance-binary-6x6-shader
        DEPENDS ${ASTC_VULKAN_GPU_EXACT_D1_LUMINANCE_BINARY_6X6_SHADER_OUTPUT})
    add_custom_command(
        OUTPUT ${ASTC_VULKAN_GPU_EXACT_D1_LUMINANCE_REFINED_6X6_SHADER_OUTPUT}
        COMMAND ${ASTC_VULKAN_GLSLC}
            --target-env=vulkan1.1
            -fshader-stage=compute
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-exact-d1-luminance-refined-6x6.comp
            -o ${ASTC_VULKAN_GPU_EXACT_D1_LUMINANCE_REFINED_6X6_SHADER_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-exact-d1-luminance-refined-6x6.comp
        COMMENT "Compiling refined exact D1 6x6 luminance ASTC subset shader")
    add_custom_target(astc-vulkan-gpu-exact-d1-luminance-refined-6x6-shader
        DEPENDS ${ASTC_VULKAN_GPU_EXACT_D1_LUMINANCE_REFINED_6X6_SHADER_OUTPUT})
    add_custom_command(
        OUTPUT ${ASTC_VULKAN_GPU_EXACT_D1_LUMINANCE_MEAN_REFINED_6X6_SHADER_OUTPUT}
        COMMAND ${ASTC_VULKAN_GLSLC}
            --target-env=vulkan1.1
            -fshader-stage=compute
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-exact-d1-luminance-mean-refined-6x6.comp
            -o ${ASTC_VULKAN_GPU_EXACT_D1_LUMINANCE_MEAN_REFINED_6X6_SHADER_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-exact-d1-luminance-mean-refined-6x6.comp
        COMMENT "Compiling mean-refined exact D1 6x6 luminance ASTC subset shader")
    add_custom_target(astc-vulkan-gpu-exact-d1-luminance-mean-refined-6x6-shader
        DEPENDS ${ASTC_VULKAN_GPU_EXACT_D1_LUMINANCE_MEAN_REFINED_6X6_SHADER_OUTPUT})
    add_custom_command(
        OUTPUT ${ASTC_VULKAN_GPU_EXACT_D1_LUMINANCE_QUANTILE_REFINED_6X6_SHADER_OUTPUT}
        COMMAND ${ASTC_VULKAN_GLSLC}
            --target-env=vulkan1.1
            -fshader-stage=compute
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-exact-d1-luminance-quantile-refined-6x6.comp
            -o ${ASTC_VULKAN_GPU_EXACT_D1_LUMINANCE_QUANTILE_REFINED_6X6_SHADER_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-exact-d1-luminance-quantile-refined-6x6.comp
        COMMENT "Compiling quantile-refined exact D1 6x6 luminance ASTC subset shader")
    add_custom_target(astc-vulkan-gpu-exact-d1-luminance-quantile-refined-6x6-shader
        DEPENDS ${ASTC_VULKAN_GPU_EXACT_D1_LUMINANCE_QUANTILE_REFINED_6X6_SHADER_OUTPUT})
    add_custom_command(
        OUTPUT ${ASTC_VULKAN_GPU_EXACT_D1_LUMINANCE_BINARY_5X5_SHADER_OUTPUT}
        COMMAND ${ASTC_VULKAN_GLSLC}
            --target-env=vulkan1.1
            -fshader-stage=compute
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-exact-d1-luminance-binary-5x5.comp
            -o ${ASTC_VULKAN_GPU_EXACT_D1_LUMINANCE_BINARY_5X5_SHADER_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-exact-d1-luminance-binary-5x5.comp
            ${ASTC_VULKAN_GPU_EXACT_PACK_INCLUDE}
        COMMENT "Compiling exact D1 5x5 luminance ASTC subset shader")
    add_custom_target(astc-vulkan-gpu-exact-d1-luminance-binary-5x5-shader
        DEPENDS ${ASTC_VULKAN_GPU_EXACT_D1_LUMINANCE_BINARY_5X5_SHADER_OUTPUT})
    add_custom_command(
        OUTPUT ${ASTC_VULKAN_GPU_EXACT_D1_LUMINANCE_QUANT4_4X4_SHADER_OUTPUT}
        COMMAND ${ASTC_VULKAN_GLSLC}
            --target-env=vulkan1.1
            -fshader-stage=compute
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-exact-d1-luminance-quant4-4x4.comp
            -o ${ASTC_VULKAN_GPU_EXACT_D1_LUMINANCE_QUANT4_4X4_SHADER_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-exact-d1-luminance-quant4-4x4.comp
            ${ASTC_VULKAN_GPU_EXACT_PACK_INCLUDE}
        COMMENT "Compiling exact D1 4x4 luminance ASTC subset shader")
    add_custom_target(astc-vulkan-gpu-exact-d1-luminance-quant4-4x4-shader
        DEPENDS ${ASTC_VULKAN_GPU_EXACT_D1_LUMINANCE_QUANT4_4X4_SHADER_OUTPUT})
    add_custom_command(
        OUTPUT ${ASTC_VULKAN_GPU_EXACT_LA_BINARY_8X5_SHADER_OUTPUT}
        COMMAND ${ASTC_VULKAN_GLSLC}
            --target-env=vulkan1.1
            -fshader-stage=compute
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-exact-la-binary-8x5.comp
            -o ${ASTC_VULKAN_GPU_EXACT_LA_BINARY_8X5_SHADER_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-exact-la-binary-8x5.comp
            ${ASTC_VULKAN_GPU_EXACT_PACK_INCLUDE}
        COMMENT "Compiling exact L+A 8x5 ASTC subset shader")
    add_custom_target(astc-vulkan-gpu-exact-la-binary-8x5-shader
        DEPENDS ${ASTC_VULKAN_GPU_EXACT_LA_BINARY_8X5_SHADER_OUTPUT})
    add_custom_command(
        OUTPUT ${ASTC_VULKAN_GPU_EXACT_LA_LUMINANCE_8X5_SHADER_OUTPUT}
        COMMAND ${ASTC_VULKAN_GLSLC}
            --target-env=vulkan1.1
            -fshader-stage=compute
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-exact-la-luminance-8x5.comp
            -o ${ASTC_VULKAN_GPU_EXACT_LA_LUMINANCE_8X5_SHADER_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-exact-la-luminance-8x5.comp
            ${ASTC_VULKAN_GPU_EXACT_PACK_INCLUDE}
        COMMENT "Compiling luminance-guided L+A 8x5 ASTC subset shader")
    add_custom_target(astc-vulkan-gpu-exact-la-luminance-8x5-shader
        DEPENDS ${ASTC_VULKAN_GPU_EXACT_LA_LUMINANCE_8X5_SHADER_OUTPUT})
    add_custom_command(
        OUTPUT ${ASTC_VULKAN_GPU_EXACT_LA_ALPHA_8X5_SHADER_OUTPUT}
        COMMAND ${ASTC_VULKAN_GLSLC}
            --target-env=vulkan1.1
            -fshader-stage=compute
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-exact-la-alpha-8x5.comp
            -o ${ASTC_VULKAN_GPU_EXACT_LA_ALPHA_8X5_SHADER_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-exact-la-alpha-8x5.comp
            ${ASTC_VULKAN_GPU_EXACT_PACK_INCLUDE}
        COMMENT "Compiling Alpha-guided L+A 8x5 ASTC subset shader")
    add_custom_target(astc-vulkan-gpu-exact-la-alpha-8x5-shader
        DEPENDS ${ASTC_VULKAN_GPU_EXACT_LA_ALPHA_8X5_SHADER_OUTPUT})
    foreach(ASTC_VULKAN_D2_REFINED_VARIANT IN ITEMS refined mean-refined quantile-refined)
        string(REPLACE "-" "_" ASTC_VULKAN_D2_REFINED_VARIANT_ID ${ASTC_VULKAN_D2_REFINED_VARIANT})
        string(TOUPPER ${ASTC_VULKAN_D2_REFINED_VARIANT_ID} ASTC_VULKAN_D2_REFINED_VARIANT_UPPER)
        set(ASTC_VULKAN_D2_REFINED_OUTPUT ${ASTC_VULKAN_GPU_EXACT_LA_${ASTC_VULKAN_D2_REFINED_VARIANT_UPPER}_8X5_SHADER_OUTPUT})
        add_custom_command(
            OUTPUT ${ASTC_VULKAN_D2_REFINED_OUTPUT}
            COMMAND ${ASTC_VULKAN_GLSLC}
                --target-env=vulkan1.1
                -fshader-stage=compute
                ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-exact-la-${ASTC_VULKAN_D2_REFINED_VARIANT}-8x5.comp
                -o ${ASTC_VULKAN_D2_REFINED_OUTPUT}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-exact-la-${ASTC_VULKAN_D2_REFINED_VARIANT}-8x5.comp
            COMMENT "Compiling ${ASTC_VULKAN_D2_REFINED_VARIANT} L+A 8x5 ASTC subset shader")
        add_custom_target(astc-vulkan-gpu-exact-la-${ASTC_VULKAN_D2_REFINED_VARIANT}-8x5-shader
            DEPENDS ${ASTC_VULKAN_D2_REFINED_OUTPUT})
    endforeach()
    add_custom_command(
        OUTPUT ${ASTC_VULKAN_GPU_EXACT_LA_DUAL_BINARY_8X5_SHADER_OUTPUT}
        COMMAND ${ASTC_VULKAN_GLSLC}
            --target-env=vulkan1.1
            -fshader-stage=compute
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-exact-la-dual-binary-8x5.comp
            -o ${ASTC_VULKAN_GPU_EXACT_LA_DUAL_BINARY_8X5_SHADER_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-exact-la-dual-binary-8x5.comp
            ${ASTC_VULKAN_GPU_EXACT_PACK_INCLUDE}
        COMMENT "Compiling exact dual-plane L+A 8x5 ASTC subset shader")
    add_custom_target(astc-vulkan-gpu-exact-la-dual-binary-8x5-shader
        DEPENDS ${ASTC_VULKAN_GPU_EXACT_LA_DUAL_BINARY_8X5_SHADER_OUTPUT})
    function(astc_vulkan_add_gpu_exact_d2_h5_shader width mode dual_mode grid_width grid_height)
        set(binary_output ${CMAKE_CURRENT_BINARY_DIR}/astc-gpu-exact-la-binary-${width}x5.comp.spv)
        set(dual_output ${CMAKE_CURRENT_BINARY_DIR}/astc-gpu-exact-la-dual-binary-${width}x5.comp.spv)
        math(EXPR texels "${width} * 5")
        add_custom_command(
            OUTPUT ${binary_output}
            COMMAND ${ASTC_VULKAN_GLSLC}
                --target-env=vulkan1.1 -fshader-stage=compute
                -DASTC_D2_TEXELS=${texels} -DASTC_D2_MODE=${mode}u
                ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-exact-la-binary-h5.comp
                -o ${binary_output}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-exact-la-binary-h5.comp
                ${ASTC_VULKAN_GPU_EXACT_PACK_INCLUDE}
            COMMENT "Compiling exact L+A ${width}x5 ASTC subset shader")
        add_custom_target(astc-vulkan-gpu-exact-la-binary-${width}x5-shader DEPENDS ${binary_output})
        add_custom_command(
            OUTPUT ${dual_output}
            COMMAND ${ASTC_VULKAN_GLSLC}
                --target-env=vulkan1.1 -fshader-stage=compute
                -DASTC_D2_TEXELS=${texels} -DASTC_D2_MODE=${dual_mode}u
                -DASTC_D2_BLOCK_WIDTH=${width} -DASTC_D2_BLOCK_HEIGHT=5
                -DASTC_D2_GRID_WIDTH=${grid_width} -DASTC_D2_GRID_HEIGHT=${grid_height}
                ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-exact-la-dual-binary-h5.comp
                -o ${dual_output}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-exact-la-dual-binary-h5.comp
                ${ASTC_VULKAN_GPU_EXACT_PACK_INCLUDE}
            COMMENT "Compiling exact dual-plane L+A ${width}x5 ASTC subset shader")
        add_custom_target(astc-vulkan-gpu-exact-la-dual-binary-${width}x5-shader DEPENDS ${dual_output})
    endfunction()
    astc_vulkan_add_gpu_exact_d2_h5_shader(6 0x161 0x4a1 5 3)
    astc_vulkan_add_gpu_exact_d2_h5_shader(10 0x165 0x4e1 5 5)
    add_custom_command(
        OUTPUT ${ASTC_VULKAN_GPU_D1_PROPOSER_8X6_SHADER_OUTPUT}
        COMMAND ${ASTC_VULKAN_GLSLC}
            --target-env=vulkan1.1
            -fshader-stage=compute
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-d1-proposer-8x6.comp
            -o ${ASTC_VULKAN_GPU_D1_PROPOSER_8X6_SHADER_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-d1-proposer-8x6.comp
        COMMENT "Compiling offline D1 8x6 GPU ASTC proposer shader")
    add_custom_target(astc-vulkan-gpu-d1-proposer-8x6-shader
        DEPENDS ${ASTC_VULKAN_GPU_D1_PROPOSER_8X6_SHADER_OUTPUT})
    add_custom_command(
        OUTPUT ${ASTC_VULKAN_GPU_D2_PROPOSER_8X5_SHADER_OUTPUT}
        COMMAND ${ASTC_VULKAN_GLSLC}
            --target-env=vulkan1.1
            -fshader-stage=compute
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-d2-proposer-8x5.comp
            -o ${ASTC_VULKAN_GPU_D2_PROPOSER_8X5_SHADER_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-d2-proposer-8x5.comp
        COMMENT "Compiling offline D2 8x5 GPU ASTC proposer shader")
    add_custom_target(astc-vulkan-gpu-d2-proposer-8x5-shader
        DEPENDS ${ASTC_VULKAN_GPU_D2_PROPOSER_8X5_SHADER_OUTPUT})
    add_custom_command(
        OUTPUT ${ASTC_VULKAN_GPU_D2_PROPOSER_6X5_SHADER_OUTPUT}
        COMMAND ${ASTC_VULKAN_GLSLC}
            --target-env=vulkan1.1
            -fshader-stage=compute
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-d2-proposer-6x5.comp
            -o ${ASTC_VULKAN_GPU_D2_PROPOSER_6X5_SHADER_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-d2-proposer-6x5.comp
        COMMENT "Compiling offline D2 6x5 GPU ASTC proposer shader")
    add_custom_target(astc-vulkan-gpu-d2-proposer-6x5-shader
        DEPENDS ${ASTC_VULKAN_GPU_D2_PROPOSER_6X5_SHADER_OUTPUT})
    add_custom_command(
        OUTPUT ${ASTC_VULKAN_GPU_D2_PROPOSER_10X5_SHADER_OUTPUT}
        COMMAND ${ASTC_VULKAN_GLSLC}
            --target-env=vulkan1.1
            -fshader-stage=compute
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-d2-proposer-10x5.comp
            -o ${ASTC_VULKAN_GPU_D2_PROPOSER_10X5_SHADER_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-d2-proposer-10x5.comp
        COMMENT "Compiling offline D2 10x5 GPU ASTC proposer shader")
    add_custom_target(astc-vulkan-gpu-d2-proposer-10x5-shader
        DEPENDS ${ASTC_VULKAN_GPU_D2_PROPOSER_10X5_SHADER_OUTPUT})
    add_custom_command(
        OUTPUT ${ASTC_VULKAN_GPU_D1_PROPOSER_5X5_SHADER_OUTPUT}
        COMMAND ${ASTC_VULKAN_GLSLC}
            --target-env=vulkan1.1
            -fshader-stage=compute
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-d1-proposer-5x5.comp
            -o ${ASTC_VULKAN_GPU_D1_PROPOSER_5X5_SHADER_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/astc-gpu-d1-proposer-5x5.comp
        COMMENT "Compiling offline D1 5x5 GPU ASTC proposer shader")
    add_custom_target(astc-vulkan-gpu-d1-proposer-5x5-shader
        DEPENDS ${ASTC_VULKAN_GPU_D1_PROPOSER_5X5_SHADER_OUTPUT})
    if (TARGET astc-vulkan-device-smoke)
        add_executable(astc-vulkan-shader-device-smoke
            astc-vulkan-shader-device-smoke.cpp)
        target_include_directories(astc-vulkan-shader-device-smoke PRIVATE
            ${ASTC_VULKAN_POC_INCLUDE_DIR})
        target_link_libraries(astc-vulkan-shader-device-smoke PRIVATE Vulkan::Vulkan)
        target_link_libraries(astc-vulkan-shader-device-smoke PRIVATE astc-vulkan-resource)
        target_compile_features(astc-vulkan-shader-device-smoke PRIVATE cxx_std_17)
        add_dependencies(astc-vulkan-shader-device-smoke
            astc-vulkan-shader-smoke
            astc-vulkan-matvec-shader
            astc-vulkan-buffer-matvec-shader
            astc-vulkan-q4-matvec-shader
            astc-vulkan-tq2-matvec-shader
            astc-vulkan-q3-matvec-shader
            astc-vulkan-tq1-matvec-shader
            astc-vulkan-sampled-f32-matvec-shader)
    endif()

    # The executable is always built on a Vulkan system, but its D2 8x5/10x5
    # pipeline test stays opt-in. It passes on the current Intel/Mesa smoke
    # target. The validated policy may prefer GPU delta/gain stages after
    # preflight; selector commit and validation-prefix replay remain on CPU.
    if (ASTC_VULKAN_GLSLC AND GGML_VK_ASTC_EXPERIMENTAL_GPU_RANKING_DEVICE_SMOKE AND
        TARGET astc-vulkan-gpu-ranking-device-smoke)
        add_dependencies(astc-vulkan-gpu-ranking-device-smoke
            astc-vulkan-paired-candidate-delta-shader
            astc-vulkan-paired-proposal-gain-shader
            astc-vulkan-d1-candidate-delta-shader
            astc-vulkan-local-select-shader)
        add_test(NAME astc-vulkan-gpu-ranking-device-smoke
            COMMAND astc-vulkan-gpu-ranking-device-smoke
                ${ASTC_VULKAN_PAIRED_CANDIDATE_DELTA_SHADER_OUTPUT}
                ${ASTC_VULKAN_PAIRED_PROPOSAL_GAIN_SHADER_OUTPUT}
                ${ASTC_VULKAN_D1_CANDIDATE_DELTA_SHADER_OUTPUT}
                ${ASTC_VULKAN_LOCAL_SELECT_SHADER_OUTPUT})
        set_property(TEST astc-vulkan-gpu-ranking-device-smoke PROPERTY LABELS astc)
        set_property(TEST astc-vulkan-gpu-ranking-device-smoke PROPERTY SKIP_RETURN_CODE 77)
    endif()

    if (ASTC_VULKAN_GLSLC AND TARGET astc-vulkan-yaqa-device-smoke)
        add_dependencies(astc-vulkan-yaqa-device-smoke astc-vulkan-yaqa-trace-score-shader)
        add_test(NAME astc-vulkan-yaqa-device-smoke
            COMMAND astc-vulkan-yaqa-device-smoke
                ${ASTC_VULKAN_YAQA_TRACE_SCORE_SHADER_OUTPUT})
        set_property(TEST astc-vulkan-yaqa-device-smoke PROPERTY LABELS astc)
        set_property(TEST astc-vulkan-yaqa-device-smoke PROPERTY SKIP_RETURN_CODE 77)
    endif()

    if (ASTC_VULKAN_GLSLC AND TARGET astc-vulkan-yaqa-batch-device-smoke)
        add_dependencies(astc-vulkan-yaqa-batch-device-smoke
            astc-vulkan-yaqa-trace-partial-shader
            astc-vulkan-yaqa-trace-reduce-shader)
        add_test(NAME astc-vulkan-yaqa-batch-device-smoke
            COMMAND astc-vulkan-yaqa-batch-device-smoke
                ${ASTC_VULKAN_YAQA_TRACE_PARTIAL_SHADER_OUTPUT}
                ${ASTC_VULKAN_YAQA_TRACE_REDUCE_SHADER_OUTPUT})
        set_property(TEST astc-vulkan-yaqa-batch-device-smoke PROPERTY LABELS astc)
        set_property(TEST astc-vulkan-yaqa-batch-device-smoke PROPERTY SKIP_RETURN_CODE 77)
    endif()
    if (ASTC_VULKAN_GLSLC AND TARGET astc-vulkan-yaqa-session-device-smoke)
        add_dependencies(astc-vulkan-yaqa-session-device-smoke
            astc-vulkan-yaqa-trace-partial-shader
            astc-vulkan-yaqa-trace-reduce-shader)
        add_test(NAME astc-vulkan-yaqa-session-device-smoke
            COMMAND astc-vulkan-yaqa-session-device-smoke
                ${ASTC_VULKAN_YAQA_TRACE_PARTIAL_SHADER_OUTPUT}
                ${ASTC_VULKAN_YAQA_TRACE_REDUCE_SHADER_OUTPUT})
        set_property(TEST astc-vulkan-yaqa-session-device-smoke PROPERTY LABELS astc)
        set_property(TEST astc-vulkan-yaqa-session-device-smoke PROPERTY SKIP_RETURN_CODE 77)
    endif()
    if (ASTC_VULKAN_GLSLC AND TARGET astc-vulkan-d1-prescreen-device-smoke)
        add_dependencies(astc-vulkan-d1-prescreen-device-smoke astc-vulkan-d1-prescreen-shader)
        add_test(NAME astc-vulkan-d1-prescreen-device-smoke
            COMMAND astc-vulkan-d1-prescreen-device-smoke
                ${ASTC_VULKAN_D1_PRESCREEN_SHADER_OUTPUT})
        set_property(TEST astc-vulkan-d1-prescreen-device-smoke PROPERTY LABELS astc)
        set_property(TEST astc-vulkan-d1-prescreen-device-smoke PROPERTY SKIP_RETURN_CODE 77)
    endif()
    if (ASTC_VULKAN_GLSLC AND TARGET astc-vulkan-d2-prescreen-device-smoke)
        add_dependencies(astc-vulkan-d2-prescreen-device-smoke astc-vulkan-d2-prescreen-shader)
        add_test(NAME astc-vulkan-d2-prescreen-device-smoke
            COMMAND astc-vulkan-d2-prescreen-device-smoke
                ${ASTC_VULKAN_D2_PRESCREEN_SHADER_OUTPUT})
        set_property(TEST astc-vulkan-d2-prescreen-device-smoke PROPERTY LABELS astc)
        set_property(TEST astc-vulkan-d2-prescreen-device-smoke PROPERTY SKIP_RETURN_CODE 77)
    endif()
    if (ASTC_VULKAN_GLSLC AND TARGET astc-vulkan-gpu-encoder-device-smoke)
        add_dependencies(astc-vulkan-gpu-encoder-device-smoke
            astc-vulkan-gpu-d1-proposer-shader
            astc-vulkan-gpu-d1-proposer-5x5-shader
            astc-vulkan-gpu-d1-proposer-6x6-shader
            astc-vulkan-gpu-d1-proposer-8x6-shader
            astc-vulkan-gpu-d2-proposer-8x5-shader
            astc-vulkan-gpu-d2-proposer-6x5-shader
            astc-vulkan-gpu-d2-proposer-10x5-shader
            astc-vulkan-shader-smoke)
        add_test(NAME astc-vulkan-gpu-encoder-device-smoke
            COMMAND astc-vulkan-gpu-encoder-device-smoke
                ${ASTC_VULKAN_GPU_D1_PROPOSER_SHADER_OUTPUT}
                ${ASTC_VULKAN_SHADER_OUTPUT}
                ${ASTC_VULKAN_GPU_D1_PROPOSER_6X6_SHADER_OUTPUT}
                ${ASTC_VULKAN_GPU_D1_PROPOSER_5X5_SHADER_OUTPUT}
                ${ASTC_VULKAN_GPU_D1_PROPOSER_8X6_SHADER_OUTPUT}
                ${ASTC_VULKAN_GPU_D2_PROPOSER_8X5_SHADER_OUTPUT}
                ${ASTC_VULKAN_GPU_D2_PROPOSER_6X5_SHADER_OUTPUT}
                ${ASTC_VULKAN_GPU_D2_PROPOSER_10X5_SHADER_OUTPUT})
        set_property(TEST astc-vulkan-gpu-encoder-device-smoke PROPERTY LABELS astc)
        set_property(TEST astc-vulkan-gpu-encoder-device-smoke PROPERTY SKIP_RETURN_CODE 77)
    endif()
    if (ASTC_VULKAN_GLSLC AND TARGET astc-vulkan-gpu-exact-subset-device-smoke)
        add_dependencies(astc-vulkan-gpu-exact-subset-device-smoke
            astc-vulkan-gpu-exact-void-extent-shader
            astc-vulkan-gpu-exact-d1-luminance-binary-6x6-shader
            astc-vulkan-gpu-exact-d1-luminance-refined-6x6-shader
            astc-vulkan-gpu-exact-d1-luminance-mean-refined-6x6-shader
            astc-vulkan-gpu-exact-d1-luminance-quantile-refined-6x6-shader
            astc-vulkan-gpu-exact-d1-luminance-binary-5x5-shader
            astc-vulkan-gpu-exact-d1-luminance-quant4-4x4-shader
            astc-vulkan-shader-smoke)
        add_test(NAME astc-vulkan-gpu-exact-subset-device-smoke
            COMMAND astc-vulkan-gpu-exact-subset-device-smoke
                ${ASTC_VULKAN_GPU_EXACT_VOID_EXTENT_SHADER_OUTPUT}
                ${ASTC_VULKAN_GPU_EXACT_D1_LUMINANCE_BINARY_6X6_SHADER_OUTPUT}
                ${ASTC_VULKAN_GPU_EXACT_D1_LUMINANCE_BINARY_5X5_SHADER_OUTPUT}
                ${ASTC_VULKAN_GPU_EXACT_D1_LUMINANCE_QUANT4_4X4_SHADER_OUTPUT}
                ${ASTC_VULKAN_SHADER_OUTPUT}
                ${ASTC_VULKAN_GPU_EXACT_D1_LUMINANCE_REFINED_6X6_SHADER_OUTPUT})
        set_property(TEST astc-vulkan-gpu-exact-subset-device-smoke PROPERTY LABELS astc)
        set_property(TEST astc-vulkan-gpu-exact-subset-device-smoke PROPERTY SKIP_RETURN_CODE 77)
    endif()
    if (ASTC_VULKAN_GLSLC AND TARGET astc-vulkan-gpu-d1-refined-fixture-smoke)
        add_dependencies(astc-vulkan-gpu-d1-refined-fixture-smoke
            astc-vulkan-gpu-exact-d1-luminance-refined-6x6-shader
            astc-vulkan-gpu-exact-d1-luminance-mean-refined-6x6-shader
            astc-vulkan-gpu-exact-d1-luminance-quantile-refined-6x6-shader)
    endif()
    if (ASTC_VULKAN_GLSLC AND TARGET astc-vulkan-gpu-exact-d2-8x5-device-smoke)
        add_dependencies(astc-vulkan-gpu-exact-d2-8x5-device-smoke
            astc-vulkan-gpu-exact-la-binary-8x5-shader
            astc-vulkan-gpu-exact-la-luminance-8x5-shader
            astc-vulkan-gpu-exact-la-alpha-8x5-shader
            astc-vulkan-gpu-exact-la-refined-8x5-shader
            astc-vulkan-gpu-exact-la-mean-refined-8x5-shader
            astc-vulkan-gpu-exact-la-quantile-refined-8x5-shader
            astc-vulkan-gpu-exact-la-dual-binary-8x5-shader
            astc-vulkan-shader-smoke)
        add_test(NAME astc-vulkan-gpu-exact-d2-8x5-device-smoke
            COMMAND astc-vulkan-gpu-exact-d2-8x5-device-smoke
                ${ASTC_VULKAN_GPU_EXACT_LA_BINARY_8X5_SHADER_OUTPUT}
                ${ASTC_VULKAN_GPU_EXACT_LA_LUMINANCE_8X5_SHADER_OUTPUT}
                ${ASTC_VULKAN_GPU_EXACT_LA_ALPHA_8X5_SHADER_OUTPUT}
                ${ASTC_VULKAN_GPU_EXACT_LA_REFINED_8X5_SHADER_OUTPUT}
                ${ASTC_VULKAN_GPU_EXACT_LA_DUAL_BINARY_8X5_SHADER_OUTPUT}
                ${ASTC_VULKAN_SHADER_OUTPUT})
        set_property(TEST astc-vulkan-gpu-exact-d2-8x5-device-smoke PROPERTY LABELS astc)
        set_property(TEST astc-vulkan-gpu-exact-d2-8x5-device-smoke PROPERTY SKIP_RETURN_CODE 77)
    endif()
    foreach(ASTC_VULKAN_D2_H5_WIDTH IN ITEMS 6 10)
        if (ASTC_VULKAN_GLSLC AND TARGET astc-vulkan-gpu-exact-d2-${ASTC_VULKAN_D2_H5_WIDTH}x5-device-smoke)
            add_dependencies(astc-vulkan-gpu-exact-d2-${ASTC_VULKAN_D2_H5_WIDTH}x5-device-smoke
                astc-vulkan-gpu-exact-la-binary-${ASTC_VULKAN_D2_H5_WIDTH}x5-shader
                astc-vulkan-gpu-exact-la-dual-binary-${ASTC_VULKAN_D2_H5_WIDTH}x5-shader
                astc-vulkan-shader-smoke)
            add_test(NAME astc-vulkan-gpu-exact-d2-${ASTC_VULKAN_D2_H5_WIDTH}x5-device-smoke
                COMMAND astc-vulkan-gpu-exact-d2-${ASTC_VULKAN_D2_H5_WIDTH}x5-device-smoke
                    ${ASTC_VULKAN_GPU_EXACT_LA_BINARY_${ASTC_VULKAN_D2_H5_WIDTH}X5_SHADER_OUTPUT}
                    ${ASTC_VULKAN_GPU_EXACT_LA_DUAL_BINARY_${ASTC_VULKAN_D2_H5_WIDTH}X5_SHADER_OUTPUT}
                    ${ASTC_VULKAN_SHADER_OUTPUT})
            set_property(TEST astc-vulkan-gpu-exact-d2-${ASTC_VULKAN_D2_H5_WIDTH}x5-device-smoke PROPERTY LABELS astc)
            set_property(TEST astc-vulkan-gpu-exact-d2-${ASTC_VULKAN_D2_H5_WIDTH}x5-device-smoke PROPERTY SKIP_RETURN_CODE 77)
        endif()
    endforeach()
    if (TARGET ggml-base AND TARGET astc-vulkan-ffn-adapter)
        add_executable(astc-vulkan-ffn-e2e-smoke
            astc-vulkan-ffn-e2e-smoke.cpp
            astc-vulkan-input.cpp)
        target_include_directories(astc-vulkan-ffn-e2e-smoke PRIVATE
            ${ASTC_VULKAN_POC_INCLUDE_DIR}
            ${CMAKE_SOURCE_DIR}/ggml/include)
        target_link_libraries(astc-vulkan-ffn-e2e-smoke PRIVATE
            astc-vulkan-sidecar ggml-base)
        target_compile_features(astc-vulkan-ffn-e2e-smoke PRIVATE cxx_std_17)
        add_dependencies(astc-vulkan-ffn-e2e-smoke astc-vulkan-ffn-e2e-shader)

        add_executable(astc-vulkan-sidecar-compare-smoke
            astc-vulkan-sidecar-compare-smoke.cpp
            astc-vulkan-input.cpp)
        target_include_directories(astc-vulkan-sidecar-compare-smoke PRIVATE
            ${ASTC_VULKAN_POC_INCLUDE_DIR}
            ${CMAKE_SOURCE_DIR}/ggml/include)
        target_link_libraries(astc-vulkan-sidecar-compare-smoke PRIVATE
            astc-vulkan-sidecar ggml-base)
        target_compile_features(astc-vulkan-sidecar-compare-smoke PRIVATE cxx_std_17)
        add_dependencies(astc-vulkan-sidecar-compare-smoke astc-vulkan-ffn-e2e-shader)
    endif()
endif()

if (BUILD_TESTING)
    add_test(NAME test-astc-vulkan-artifact-policy
        COMMAND test-astc-vulkan-artifact-policy)
    set_property(TEST test-astc-vulkan-artifact-policy PROPERTY LABELS astc)
    add_test(NAME test-astc-vulkan-composition-gate
        COMMAND test-astc-vulkan-composition-gate)
    set_property(TEST test-astc-vulkan-composition-gate PROPERTY LABELS astc)
    add_test(NAME test-astc-vulkan-hash
        COMMAND test-astc-vulkan-hash)
    set_property(TEST test-astc-vulkan-hash PROPERTY LABELS astc)

    add_test(NAME test-astc-vulkan-gauge
        COMMAND test-astc-vulkan-gauge)
    set_property(TEST test-astc-vulkan-gauge PROPERTY LABELS astc)
    add_test(NAME test-astc-vulkan-d1-prescreen
        COMMAND test-astc-vulkan-d1-prescreen)
    set_property(TEST test-astc-vulkan-d1-prescreen PROPERTY LABELS astc)
    add_test(NAME test-astc-vulkan-d2-row-scale
        COMMAND test-astc-vulkan-d2-row-scale)
    set_property(TEST test-astc-vulkan-d2-row-scale PROPERTY LABELS astc)

    add_test(NAME test-astc-vulkan-gpu-ranking
        COMMAND test-astc-vulkan-gpu-ranking)
    set_property(TEST test-astc-vulkan-gpu-ranking PROPERTY LABELS astc)
    add_test(NAME test-astc-vulkan-gpu-d1-ranking
        COMMAND test-astc-vulkan-gpu-d1-ranking)
    set_property(TEST test-astc-vulkan-gpu-d1-ranking PROPERTY LABELS astc)

    add_test(NAME test-astc-vulkan-ranking-plan
        COMMAND test-astc-vulkan-ranking-plan)
    set_property(TEST test-astc-vulkan-ranking-plan PROPERTY LABELS astc)

    add_test(NAME test-astc-vulkan-paired
        COMMAND test-astc-vulkan-paired)
    set_property(TEST test-astc-vulkan-paired PROPERTY LABELS astc)

    add_test(NAME test-astc-vulkan-paired-selector
        COMMAND test-astc-vulkan-paired-selector)
    set_property(TEST test-astc-vulkan-paired-selector PROPERTY LABELS astc)

    add_test(NAME test-astc-vulkan-paired-layout
        COMMAND test-astc-vulkan-paired-layout)
    set_property(TEST test-astc-vulkan-paired-layout PROPERTY LABELS astc)

    add_test(NAME test-astc-vulkan-objective
        COMMAND test-astc-vulkan-objective)
    set_property(TEST test-astc-vulkan-objective PROPERTY LABELS astc)

    add_test(NAME test-astc-vulkan-block-ldlq-config
        COMMAND test-astc-vulkan-block-ldlq-config)
    set_property(TEST test-astc-vulkan-block-ldlq-config PROPERTY LABELS astc)

    add_test(NAME test-astc-vulkan-pv
        COMMAND test-astc-vulkan-pv)
    set_property(TEST test-astc-vulkan-pv PROPERTY LABELS astc)
    add_test(NAME test-astc-vulkan-pv-d2
        COMMAND test-astc-vulkan-pv-d2)
    set_property(TEST test-astc-vulkan-pv-d2 PROPERTY LABELS astc)

    add_test(NAME test-astc-vulkan-yaqa
        COMMAND test-astc-vulkan-yaqa)
    set_property(TEST test-astc-vulkan-yaqa PROPERTY LABELS astc)

    add_executable(test-astc-vulkan-provenance
        tests/test-astc-vulkan-provenance.cpp)
    target_link_libraries(test-astc-vulkan-provenance PRIVATE astc-vulkan-provenance)
    target_compile_features(test-astc-vulkan-provenance PRIVATE cxx_std_17)
    add_test(NAME test-astc-vulkan-provenance
        COMMAND test-astc-vulkan-provenance)
    set_property(TEST test-astc-vulkan-provenance PROPERTY LABELS astc)

    add_executable(test-astc-vulkan-cache
        tests/test-astc-vulkan-cache.cpp)
    target_link_libraries(test-astc-vulkan-cache PRIVATE astc-vulkan-cache)
    target_include_directories(test-astc-vulkan-cache PRIVATE
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_compile_features(test-astc-vulkan-cache PRIVATE cxx_std_17)
    add_test(NAME test-astc-vulkan-cache
        COMMAND test-astc-vulkan-cache)
    set_property(TEST test-astc-vulkan-cache PROPERTY LABELS astc)

    add_executable(test-astc-vulkan-error-shaping
        tests/test-astc-vulkan-error-shaping.cpp)
    target_compile_features(test-astc-vulkan-error-shaping PRIVATE cxx_std_17)
    add_test(NAME test-astc-vulkan-error-shaping
        COMMAND test-astc-vulkan-error-shaping)
    set_property(TEST test-astc-vulkan-error-shaping PROPERTY LABELS astc)

    add_executable(test-astc-vulkan-block-ldlq
        tests/test-astc-vulkan-block-ldlq.cpp)
    target_compile_features(test-astc-vulkan-block-ldlq PRIVATE cxx_std_17)
    add_test(NAME test-astc-vulkan-block-ldlq
        COMMAND test-astc-vulkan-block-ldlq)
    set_property(TEST test-astc-vulkan-block-ldlq PROPERTY LABELS astc)

    add_test(NAME test-astc-vulkan-contract
        COMMAND test-astc-vulkan-contract)
    set_property(TEST test-astc-vulkan-contract PROPERTY LABELS astc)

    add_test(NAME test-astc-vulkan-smoke
        COMMAND test-astc-vulkan-smoke)
    set_property(TEST test-astc-vulkan-smoke PROPERTY LABELS astc)

    add_test(NAME test-astc-vulkan-tensor-contract
        COMMAND test-astc-vulkan-tensor-contract)
    set_property(TEST test-astc-vulkan-tensor-contract PROPERTY LABELS astc)

    add_test(NAME test-astc-vulkan-driver
        COMMAND test-astc-vulkan-driver)
    set_property(TEST test-astc-vulkan-driver PROPERTY LABELS astc)

    if (TARGET test-astc-vulkan-ffn-adapter)
        add_test(NAME test-astc-vulkan-ffn-adapter
            COMMAND test-astc-vulkan-ffn-adapter)
        set_property(TEST test-astc-vulkan-ffn-adapter PROPERTY LABELS astc)
    endif()

    if (TARGET test-astc-vulkan-dispatch)
        add_test(NAME test-astc-vulkan-dispatch
            COMMAND test-astc-vulkan-dispatch)
        set_property(TEST test-astc-vulkan-dispatch PROPERTY LABELS astc)
    endif()

    if (TARGET test-astc-vulkan-sidecar)
        add_test(NAME test-astc-vulkan-sidecar
            COMMAND test-astc-vulkan-sidecar)
        set_property(TEST test-astc-vulkan-sidecar PROPERTY LABELS astc)
    endif()

    if (TARGET test-astc-vulkan-budget)
        add_test(NAME test-astc-vulkan-budget
            COMMAND test-astc-vulkan-budget)
        set_property(TEST test-astc-vulkan-budget PROPERTY LABELS astc)
    endif()

    if (TARGET test-astc-vulkan-residency)
        add_test(NAME test-astc-vulkan-residency
            COMMAND test-astc-vulkan-residency)
        set_property(TEST test-astc-vulkan-residency PROPERTY LABELS astc)
    endif()

    if (TARGET astc-vulkan-astcenc-smoke)
        add_test(NAME astc-vulkan-d2-pair-transform-smoke
            COMMAND astc-vulkan-d2-pair-transform-smoke)
        set_property(TEST astc-vulkan-d2-pair-transform-smoke PROPERTY LABELS astc)
        add_test(NAME astc-vulkan-paired-smoke
            COMMAND astc-vulkan-paired-smoke)
        set_property(TEST astc-vulkan-paired-smoke PROPERTY LABELS astc)
        add_test(NAME astc-vulkan-astcenc-smoke
            COMMAND astc-vulkan-astcenc-smoke)
        set_property(TEST astc-vulkan-astcenc-smoke PROPERTY LABELS astc)
        add_test(NAME astc-vulkan-encoder-track-smoke
            COMMAND astc-vulkan-encoder-track-smoke)
        set_property(TEST astc-vulkan-encoder-track-smoke PROPERTY LABELS astc)
        add_test(NAME astc-vulkan-weight-smoke
            COMMAND astc-vulkan-weight-smoke)
        set_property(TEST astc-vulkan-weight-smoke PROPERTY LABELS astc)
        add_test(NAME astc-vulkan-weight-smoke-large
            COMMAND astc-vulkan-weight-smoke --large)
        set_property(TEST astc-vulkan-weight-smoke-large PROPERTY LABELS astc)
        add_test(NAME astc-vulkan-level-smoke
            COMMAND astc-vulkan-level-smoke)
        set_property(TEST astc-vulkan-level-smoke PROPERTY LABELS astc)
        add_test(NAME astc-vulkan-channel-semantic-smoke
            COMMAND astc-vulkan-channel-semantic-smoke)
        set_property(TEST astc-vulkan-channel-semantic-smoke PROPERTY LABELS astc)
        add_test(NAME astc-vulkan-latent-smoke
            COMMAND astc-vulkan-latent-smoke)
        set_property(TEST astc-vulkan-latent-smoke PROPERTY LABELS astc)
        add_test(NAME astc-vulkan-latent-c-delta-smoke
            COMMAND astc-vulkan-latent-smoke --scalar-anchored-c-delta-sweep
                --footprint 6x6 --max-rows 8 --max-columns 32
                --max-samples 8 --max-calibration-samples 8
                --row-strip-chunked --persistent-worker-contexts
                --row-strip-light-diagnostics)
        set_property(TEST astc-vulkan-latent-c-delta-smoke PROPERTY LABELS astc)
        if (GGML_VK_ASTC_EXPERIMENTAL_NEURAL_RANK)
            add_test(NAME astc-vulkan-latent-neural-rank-smoke
                COMMAND astc-vulkan-latent-smoke --neural-rank)
            set_property(TEST astc-vulkan-latent-neural-rank-smoke PROPERTY LABELS astc)
            add_test(NAME astc-vulkan-latent-diverse-coordinate-smoke
                COMMAND astc-vulkan-latent-smoke --neural-rank --coordinate-select
                    --coordinate-diverse)
            set_property(TEST astc-vulkan-latent-diverse-coordinate-smoke PROPERTY LABELS astc)
            add_test(NAME astc-vulkan-latent-regularized-coordinate-smoke
                COMMAND astc-vulkan-latent-smoke --neural-rank --coordinate-select
                    --coordinate-diverse --coordinate-regularized)
            set_property(TEST astc-vulkan-latent-regularized-coordinate-smoke PROPERTY LABELS astc)
            add_test(NAME astc-vulkan-latent-selector-compare-smoke
                COMMAND astc-vulkan-latent-smoke --selector-compare --footprint 4x4
                    --max-rows 8 --max-columns 32)
            set_property(TEST astc-vulkan-latent-selector-compare-smoke PROPERTY LABELS astc)
            add_test(NAME astc-vulkan-latent-candidate-capacity-smoke
                COMMAND astc-vulkan-latent-smoke --candidate-sweep --footprint 6x6
                    --max-rows 8 --max-columns 32)
            set_property(TEST astc-vulkan-latent-candidate-capacity-smoke PROPERTY LABELS astc)
            add_test(NAME astc-vulkan-latent-angular-stability-smoke
                COMMAND astc-vulkan-latent-smoke --candidate-sweep --candidate-angular
                    --stability-shards 4 --footprint 6x6 --max-rows 8 --max-columns 32)
            set_property(TEST astc-vulkan-latent-angular-stability-smoke PROPERTY LABELS astc)
        endif()
        add_test(NAME astc-vulkan-latent-neural-search-smoke
            COMMAND astc-vulkan-latent-smoke --scalar-anchored-gauge-sweep
                --encoder-search neural --neural-candidate-limit 8
                --footprint 6x6 --max-rows 12 --max-columns 12
                --max-samples 4 --max-calibration-samples 2)
        set_property(TEST astc-vulkan-latent-neural-search-smoke PROPERTY LABELS astc)
        add_test(NAME astc-vulkan-latent-weight-grid-gauge-smoke
            COMMAND astc-vulkan-latent-smoke --weight-grid-gauge-sweep
                --footprint 8x8 --max-rows 16 --max-columns 16
                --max-samples 4 --max-calibration-samples 2)
        set_property(TEST astc-vulkan-latent-weight-grid-gauge-smoke PROPERTY LABELS astc)
        add_test(NAME astc-vulkan-latent-validation-artifact-smoke
            COMMAND astc-vulkan-latent-smoke --weight-grid-gauge-sweep
                --row-strip-chunked --footprint 10x6 --max-rows 12 --max-columns 20
                --max-samples 4 --max-calibration-samples 2
                --neutral-payload ${CMAKE_CURRENT_BINARY_DIR}/validation-neutral.astc
                --neutral-reference ${CMAKE_CURRENT_BINARY_DIR}/validation-neutral.rgba.f32
                --neutral-metadata ${CMAKE_CURRENT_BINARY_DIR}/validation-neutral.metadata
                --validation-payload ${CMAKE_CURRENT_BINARY_DIR}/validation-selected.astc
                --validation-reference ${CMAKE_CURRENT_BINARY_DIR}/validation-selected.rgba.f32
                --validation-metadata ${CMAKE_CURRENT_BINARY_DIR}/validation-selected.metadata)
        set_property(TEST astc-vulkan-latent-validation-artifact-smoke PROPERTY LABELS astc)
        add_test(NAME astc-vulkan-latent-few-level-weight-grid-smoke
            COMMAND astc-vulkan-latent-smoke --few-level-weight-grid-gauge-sweep
                --footprint 8x8 --max-rows 16 --max-columns 16
                --max-samples 4 --max-calibration-samples 2)
        set_property(TEST astc-vulkan-latent-few-level-weight-grid-smoke PROPERTY LABELS astc)
        add_test(NAME astc-vulkan-latent-pv-lite-grid-smoke
            COMMAND astc-vulkan-latent-smoke --pv-lite-grid-sweep
                --footprint 8x8 --max-rows 16 --max-columns 16
                --max-samples 4 --max-calibration-samples 2)
        set_property(TEST astc-vulkan-latent-pv-lite-grid-smoke PROPERTY LABELS astc)
        # A partial 10x6 block must retain deterministic clamp padding while
        # excluding those padded texels from the gauge headroom and neural loss.
        add_test(NAME astc-vulkan-latent-pv-lite-edge-smoke
            COMMAND astc-vulkan-latent-smoke --pv-lite-grid-sweep
                --row-strip-chunked --footprint 10x6 --max-rows 11 --max-columns 17
                --max-samples 4 --max-calibration-samples 2
                --decode-loop-log ${CMAKE_CURRENT_BINARY_DIR}/pv-lite-edge-commits.csv
                --validation-payload ${CMAKE_CURRENT_BINARY_DIR}/pv-lite-edge-selected.astc
                --validation-reference ${CMAKE_CURRENT_BINARY_DIR}/pv-lite-edge-selected.rgba.f32
                --validation-metadata ${CMAKE_CURRENT_BINARY_DIR}/pv-lite-edge-selected.metadata)
        set_property(TEST astc-vulkan-latent-pv-lite-edge-smoke PROPERTY LABELS astc)
        add_test(NAME astc-vulkan-latent-pv-lite-coarse-grid-smoke
            COMMAND astc-vulkan-latent-smoke --pv-lite-coarse-grid-sweep
                --row-strip-chunked --footprint 10x6 --max-rows 12 --max-columns 20
                --max-samples 4 --max-calibration-samples 2)
        set_property(TEST astc-vulkan-latent-pv-lite-coarse-grid-smoke PROPERTY LABELS astc)
    endif()

    if (TARGET astc-vulkan-input-smoke)
        add_test(NAME astc-vulkan-input-smoke
            COMMAND astc-vulkan-input-smoke)
        set_property(TEST astc-vulkan-input-smoke PROPERTY LABELS astc)
    endif()

    if (TARGET astc-vulkan-capability-probe)
        add_test(NAME astc-vulkan-capability-probe
            COMMAND astc-vulkan-capability-probe)
        set_property(TEST astc-vulkan-capability-probe PROPERTY LABELS astc)
        set_property(TEST astc-vulkan-capability-probe PROPERTY SKIP_RETURN_CODE 77)
    endif()

    if (TARGET astc-vulkan-device-smoke)
        add_test(NAME astc-vulkan-device-smoke
            COMMAND astc-vulkan-device-smoke)
        set_property(TEST astc-vulkan-device-smoke PROPERTY LABELS astc)
        set_property(TEST astc-vulkan-device-smoke PROPERTY SKIP_RETURN_CODE 77)
    endif()

    if (ASTC_VULKAN_GLSLC)
        add_test(NAME astc-vulkan-shader-compile
            COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
            --target astc-vulkan-shader-smoke)
        set_property(TEST astc-vulkan-shader-compile PROPERTY LABELS astc)
        if (TARGET astc-vulkan-shader-device-smoke)
            add_test(NAME astc-vulkan-shader-device-smoke-4x4
                COMMAND astc-vulkan-shader-device-smoke
                    ${ASTC_VULKAN_SHADER_OUTPUT} 4x4)
            set_property(TEST astc-vulkan-shader-device-smoke-4x4 PROPERTY LABELS astc)
            set_property(TEST astc-vulkan-shader-device-smoke-4x4 PROPERTY SKIP_RETURN_CODE 77)
            add_test(NAME astc-vulkan-shader-device-smoke-5x5
                COMMAND astc-vulkan-shader-device-smoke
                    ${ASTC_VULKAN_SHADER_OUTPUT} 5x5)
            set_property(TEST astc-vulkan-shader-device-smoke-5x5 PROPERTY LABELS astc)
            set_property(TEST astc-vulkan-shader-device-smoke-5x5 PROPERTY SKIP_RETURN_CODE 77)
            add_test(NAME astc-vulkan-shader-device-smoke-6x6
                COMMAND astc-vulkan-shader-device-smoke
                    ${ASTC_VULKAN_SHADER_OUTPUT} 6x6)
            set_property(TEST astc-vulkan-shader-device-smoke-6x6 PROPERTY LABELS astc)
            set_property(TEST astc-vulkan-shader-device-smoke-6x6 PROPERTY SKIP_RETURN_CODE 77)
            add_test(NAME astc-vulkan-shader-device-smoke-8x6-experimental
                COMMAND astc-vulkan-shader-device-smoke
                    ${ASTC_VULKAN_SHADER_OUTPUT} 8x6)
            set_property(TEST astc-vulkan-shader-device-smoke-8x6-experimental PROPERTY LABELS astc)
            set_property(TEST astc-vulkan-shader-device-smoke-8x6-experimental PROPERTY SKIP_RETURN_CODE 77)
            add_test(NAME astc-vulkan-shader-device-smoke-10x6-experimental
                COMMAND astc-vulkan-shader-device-smoke
                    ${ASTC_VULKAN_SHADER_OUTPUT} 10x6)
            set_property(TEST astc-vulkan-shader-device-smoke-10x6-experimental PROPERTY LABELS astc)
            set_property(TEST astc-vulkan-shader-device-smoke-10x6-experimental PROPERTY SKIP_RETURN_CODE 77)
            add_test(NAME astc-vulkan-shader-device-smoke-8x8-experimental
                COMMAND astc-vulkan-shader-device-smoke
                    ${ASTC_VULKAN_SHADER_OUTPUT} 8x8)
            set_property(TEST astc-vulkan-shader-device-smoke-8x8-experimental PROPERTY LABELS astc)
            set_property(TEST astc-vulkan-shader-device-smoke-8x8-experimental PROPERTY SKIP_RETURN_CODE 77)
            add_test(NAME astc-vulkan-shader-device-smoke-10x8-experimental
                COMMAND astc-vulkan-shader-device-smoke
                    ${ASTC_VULKAN_SHADER_OUTPUT} 10x8)
            set_property(TEST astc-vulkan-shader-device-smoke-10x8-experimental PROPERTY LABELS astc)
            set_property(TEST astc-vulkan-shader-device-smoke-10x8-experimental PROPERTY SKIP_RETURN_CODE 77)
            add_test(NAME astc-vulkan-shader-device-smoke-4x4-nonlocal
                COMMAND astc-vulkan-shader-device-smoke
                    ${ASTC_VULKAN_SHADER_OUTPUT} 4x4 nonlocal)
            set_property(TEST astc-vulkan-shader-device-smoke-4x4-nonlocal PROPERTY LABELS astc)
            set_property(TEST astc-vulkan-shader-device-smoke-4x4-nonlocal PROPERTY SKIP_RETURN_CODE 77)
            add_test(NAME astc-vulkan-shader-device-smoke-5x5-nonlocal
                COMMAND astc-vulkan-shader-device-smoke
                    ${ASTC_VULKAN_SHADER_OUTPUT} 5x5 nonlocal)
            set_property(TEST astc-vulkan-shader-device-smoke-5x5-nonlocal PROPERTY LABELS astc)
            set_property(TEST astc-vulkan-shader-device-smoke-5x5-nonlocal PROPERTY SKIP_RETURN_CODE 77)
            add_test(NAME astc-vulkan-shader-device-smoke-6x6-nonlocal
                COMMAND astc-vulkan-shader-device-smoke
                    ${ASTC_VULKAN_SHADER_OUTPUT} 6x6 nonlocal)
            set_property(TEST astc-vulkan-shader-device-smoke-6x6-nonlocal PROPERTY LABELS astc)
            set_property(TEST astc-vulkan-shader-device-smoke-6x6-nonlocal PROPERTY SKIP_RETURN_CODE 77)
        endif()
    endif()
endif()
