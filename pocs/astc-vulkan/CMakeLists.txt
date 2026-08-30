# Experimental ASTC Vulkan proof of concept.
# Keep this target independent from ggml-vulkan until the device-backed path
# has correctness and performance evidence.

set(ASTC_VULKAN_POC_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/ggml/src/ggml-vulkan)

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

if (BUILD_TESTING)
    add_test(NAME test-astc-vulkan-contract
        COMMAND test-astc-vulkan-contract)
    set_property(TEST test-astc-vulkan-contract PROPERTY LABELS astc)

    add_test(NAME test-astc-vulkan-smoke
        COMMAND test-astc-vulkan-smoke)
    set_property(TEST test-astc-vulkan-smoke PROPERTY LABELS astc)
endif()
