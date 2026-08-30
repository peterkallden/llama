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
endif()

if (BUILD_TESTING)
    add_test(NAME test-astc-vulkan-contract
        COMMAND test-astc-vulkan-contract)
    set_property(TEST test-astc-vulkan-contract PROPERTY LABELS astc)

    add_test(NAME test-astc-vulkan-smoke
        COMMAND test-astc-vulkan-smoke)
    set_property(TEST test-astc-vulkan-smoke PROPERTY LABELS astc)

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
    endif()
endif()
