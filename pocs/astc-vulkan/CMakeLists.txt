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

find_package(Vulkan QUIET)
if (Vulkan_FOUND)
    add_executable(astc-vulkan-capability-probe
        astc-vulkan-capability-probe.cpp)
    target_link_libraries(astc-vulkan-capability-probe PRIVATE Vulkan::Vulkan)
    target_compile_features(astc-vulkan-capability-probe PRIVATE cxx_std_17)

    add_executable(astc-vulkan-device-smoke
        astc-vulkan-device-smoke.cpp)
    target_link_libraries(astc-vulkan-device-smoke PRIVATE Vulkan::Vulkan)
    target_compile_features(astc-vulkan-device-smoke PRIVATE cxx_std_17)
endif()

find_package(astcenc CONFIG QUIET)
if (TARGET astcenc::astcenc-static)
    add_executable(astc-vulkan-astcenc-smoke
        astc-vulkan-astcenc-smoke.cpp)
    target_include_directories(astc-vulkan-astcenc-smoke PRIVATE
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_link_libraries(astc-vulkan-astcenc-smoke PRIVATE astcenc::astcenc-static)
    target_compile_features(astc-vulkan-astcenc-smoke PRIVATE cxx_std_17)

    add_executable(astc-vulkan-weight-smoke
        astc-vulkan-weight-smoke.cpp)
    target_include_directories(astc-vulkan-weight-smoke PRIVATE
        ${ASTC_VULKAN_POC_INCLUDE_DIR})
    target_link_libraries(astc-vulkan-weight-smoke PRIVATE astcenc::astcenc-static)
    target_compile_features(astc-vulkan-weight-smoke PRIVATE cxx_std_17)
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
endif()

find_program(ASTC_VULKAN_GLSLC glslc)
if (ASTC_VULKAN_GLSLC)
    set(ASTC_VULKAN_SHADER_OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/astc-validate.comp.spv)
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
    if (TARGET astc-vulkan-device-smoke)
        add_executable(astc-vulkan-shader-device-smoke
            astc-vulkan-shader-device-smoke.cpp)
        target_include_directories(astc-vulkan-shader-device-smoke PRIVATE
            ${ASTC_VULKAN_POC_INCLUDE_DIR})
        target_link_libraries(astc-vulkan-shader-device-smoke PRIVATE Vulkan::Vulkan)
        target_compile_features(astc-vulkan-shader-device-smoke PRIVATE cxx_std_17)
        add_dependencies(astc-vulkan-shader-device-smoke astc-vulkan-shader-smoke)
    endif()
endif()

if (BUILD_TESTING)
    add_test(NAME test-astc-vulkan-contract
        COMMAND test-astc-vulkan-contract)
    set_property(TEST test-astc-vulkan-contract PROPERTY LABELS astc)

    add_test(NAME test-astc-vulkan-smoke
        COMMAND test-astc-vulkan-smoke)
    set_property(TEST test-astc-vulkan-smoke PROPERTY LABELS astc)

    add_test(NAME test-astc-vulkan-tensor-contract
        COMMAND test-astc-vulkan-tensor-contract)
    set_property(TEST test-astc-vulkan-tensor-contract PROPERTY LABELS astc)

    if (TARGET astc-vulkan-astcenc-smoke)
        add_test(NAME astc-vulkan-astcenc-smoke
            COMMAND astc-vulkan-astcenc-smoke)
        set_property(TEST astc-vulkan-astcenc-smoke PROPERTY LABELS astc)
        add_test(NAME astc-vulkan-weight-smoke
            COMMAND astc-vulkan-weight-smoke)
        set_property(TEST astc-vulkan-weight-smoke PROPERTY LABELS astc)
        add_test(NAME astc-vulkan-weight-smoke-large
            COMMAND astc-vulkan-weight-smoke --large)
        set_property(TEST astc-vulkan-weight-smoke-large PROPERTY LABELS astc)
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
            add_test(NAME astc-vulkan-shader-device-smoke-6x6
                COMMAND astc-vulkan-shader-device-smoke
                    ${ASTC_VULKAN_SHADER_OUTPUT} 6x6)
            set_property(TEST astc-vulkan-shader-device-smoke-6x6 PROPERTY LABELS astc)
            set_property(TEST astc-vulkan-shader-device-smoke-6x6 PROPERTY SKIP_RETURN_CODE 77)
            add_test(NAME astc-vulkan-shader-device-smoke-4x4-nonlocal
                COMMAND astc-vulkan-shader-device-smoke
                    ${ASTC_VULKAN_SHADER_OUTPUT} 4x4 nonlocal)
            set_property(TEST astc-vulkan-shader-device-smoke-4x4-nonlocal PROPERTY LABELS astc)
            set_property(TEST astc-vulkan-shader-device-smoke-4x4-nonlocal PROPERTY SKIP_RETURN_CODE 77)
            add_test(NAME astc-vulkan-shader-device-smoke-6x6-nonlocal
                COMMAND astc-vulkan-shader-device-smoke
                    ${ASTC_VULKAN_SHADER_OUTPUT} 6x6 nonlocal)
            set_property(TEST astc-vulkan-shader-device-smoke-6x6-nonlocal PROPERTY LABELS astc)
            set_property(TEST astc-vulkan-shader-device-smoke-6x6-nonlocal PROPERTY SKIP_RETURN_CODE 77)
        endif()
    endif()
endif()
