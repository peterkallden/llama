# Experimental ASTC Vulkan proof of concept.
# Keep this target independent from ggml-vulkan until the device-backed path
# has correctness and performance evidence.

set(ASTC_VULKAN_POC_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

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
    astc-vulkan-manifest.cpp)
target_include_directories(astc-vulkan-driver PUBLIC
    ${ASTC_VULKAN_POC_INCLUDE_DIR})
target_compile_features(astc-vulkan-driver PUBLIC cxx_std_17)

add_library(astc-vulkan-provenance STATIC
    astc-vulkan-provenance.cpp)
target_include_directories(astc-vulkan-provenance PUBLIC
    ${ASTC_VULKAN_POC_INCLUDE_DIR})
target_link_libraries(astc-vulkan-provenance PUBLIC vendor::hash)
target_compile_features(astc-vulkan-provenance PUBLIC cxx_std_17)

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

find_package(Vulkan QUIET)
option(GGML_VK_ASTC_EXPERIMENTAL_SCHEDULER_ADAPTER
    "Build the isolated ASTC scheduler-adapter experiment" OFF)
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
    add_library(astc-vulkan-resource STATIC
        astc-vulkan-resource.cpp)
    target_include_directories(astc-vulkan-resource PUBLIC
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_link_libraries(astc-vulkan-resource PUBLIC Vulkan::Vulkan astc-vulkan-driver)
    target_compile_features(astc-vulkan-resource PUBLIC cxx_std_17)

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

    add_library(astc-vulkan-sidecar STATIC
        astc-vulkan-sidecar.cpp)
    target_include_directories(astc-vulkan-sidecar PUBLIC
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_link_libraries(astc-vulkan-sidecar PUBLIC
        astc-vulkan-ffn-adapter astc-vulkan-dispatch Vulkan::Vulkan)
    target_compile_features(astc-vulkan-sidecar PUBLIC cxx_std_17)

    add_executable(test-astc-vulkan-ffn-adapter
        tests/test-astc-vulkan-ffn-adapter.cpp)
    target_link_libraries(test-astc-vulkan-ffn-adapter PRIVATE astc-vulkan-ffn-adapter)
    target_include_directories(test-astc-vulkan-ffn-adapter PRIVATE
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_compile_features(test-astc-vulkan-ffn-adapter PRIVATE cxx_std_17)

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
            astc-vulkan-sidecar)
        target_compile_features(astc-vulkan-scheduler-adapter PUBLIC cxx_std_17)

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
            astc-vulkan-scheduler-adapter ggml-base)
        target_compile_features(astc-vulkan-scheduler-adapter-smoke PRIVATE cxx_std_17)
        add_dependencies(astc-vulkan-scheduler-adapter-smoke astc-vulkan-ffn-e2e-shader)
    endif()

    add_executable(astc-vulkan-capability-probe
        astc-vulkan-capability-probe.cpp)
    target_link_libraries(astc-vulkan-capability-probe PRIVATE Vulkan::Vulkan)
    target_compile_features(astc-vulkan-capability-probe PRIVATE cxx_std_17)

    add_executable(astc-vulkan-device-smoke
        astc-vulkan-device-smoke.cpp)
    target_link_libraries(astc-vulkan-device-smoke PRIVATE Vulkan::Vulkan astc-vulkan-resource)
    target_compile_features(astc-vulkan-device-smoke PRIVATE cxx_std_17)
endif()

find_package(astcenc CONFIG QUIET)
if (TARGET astcenc::astcenc-static)
    add_executable(astc-vulkan-artifact-decode
        astc-vulkan-artifact-decode.cpp)
    target_link_libraries(astc-vulkan-artifact-decode PRIVATE astcenc::astcenc-static)
    target_compile_features(astc-vulkan-artifact-decode PRIVATE cxx_std_17)

    option(GGML_VK_ASTC_EXPERIMENTAL_NEURAL_RANK
        "Enable the side-fork ASTC neural-ranking API in the latent PoC" OFF)
    set(GGML_VK_ASTC_NEURAL_RANK_LIBRARY "" CACHE FILEPATH
        "Static library from the isolated ASTC neural-ranking fork")
    set(GGML_VK_ASTC_NEURAL_RANK_INCLUDE_DIR "" CACHE PATH
        "Source include directory from the isolated ASTC neural-ranking fork")
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
            ${GGML_VK_ASTC_NEURAL_RANK_LIBRARY} ggml-base)
    else()
        target_link_libraries(astc-vulkan-latent-smoke PRIVATE
            astcenc::astcenc-static ggml-base)
    endif()
    target_compile_features(astc-vulkan-latent-smoke PRIVATE cxx_std_17)
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
            astc-vulkan-input.cpp)
        target_include_directories(astc-vulkan-model-replay-smoke PRIVATE
            ${ASTC_VULKAN_POC_INCLUDE_DIR}
            ${CMAKE_SOURCE_DIR}/ggml/include
            ${CMAKE_SOURCE_DIR}/src)
        target_link_libraries(astc-vulkan-model-replay-smoke PRIVATE
            llama-common llama ggml-base ${CMAKE_THREAD_LIBS_INIT})
        target_compile_features(astc-vulkan-model-replay-smoke PRIVATE cxx_std_17)

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
    add_executable(test-astc-vulkan-provenance
        tests/test-astc-vulkan-provenance.cpp)
    target_link_libraries(test-astc-vulkan-provenance PRIVATE astc-vulkan-provenance)
    target_compile_features(test-astc-vulkan-provenance PRIVATE cxx_std_17)
    add_test(NAME test-astc-vulkan-provenance
        COMMAND test-astc-vulkan-provenance)
    set_property(TEST test-astc-vulkan-provenance PROPERTY LABELS astc)

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

    if (TARGET astc-vulkan-astcenc-smoke)
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
