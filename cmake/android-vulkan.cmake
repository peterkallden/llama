# Shared Android Vulkan compatibility setup.
#
# The Android NDK provides the Vulkan C headers and loader, but it does not
# provide Vulkan-Hpp (vulkan/vulkan.hpp).  Vulkan-Hpp is a host-side header
# dependency used while compiling the Android target.  Stage only that header
# tree into the build directory; never add the host /usr/include directory to
# the Android include path.

function(llama_android_prepare_vulkan)
    if(NOT ANDROID OR NOT GGML_VULKAN)
        return()
    endif()

    set(_android_spirv_headers
        "${CMAKE_ANDROID_NDK}/sources/third_party/shaderc/third_party/spirv-tools/external/spirv-headers/include")
    if(NOT EXISTS "${_android_spirv_headers}/spirv/unified1/spirv.hpp")
        message(FATAL_ERROR "Android Vulkan build requires SPIR-V headers from the NDK")
    endif()
    include_directories(SYSTEM "${_android_spirv_headers}")

    find_path(VULKAN_HPP_INCLUDE_DIR
        NAMES vulkan/vulkan.hpp
        PATHS
            "$ENV{VULKAN_SDK}/Include"
            "/usr/include"
            "/usr/local/include"
        NO_CMAKE_FIND_ROOT_PATH)
    if(NOT VULKAN_HPP_INCLUDE_DIR)
        message(FATAL_ERROR
            "Android Vulkan build requires the Vulkan-Hpp header (vulkan/vulkan.hpp); "
            "install the Vulkan development headers on the host")
    endif()

    set(_android_vulkan_include
        "${CMAKE_CURRENT_BINARY_DIR}/android-vulkan-host-include")
    file(COPY "${VULKAN_HPP_INCLUDE_DIR}/vulkan"
         DESTINATION "${_android_vulkan_include}")
    if(EXISTS "${VULKAN_HPP_INCLUDE_DIR}/vk_video")
        file(COPY "${VULKAN_HPP_INCLUDE_DIR}/vk_video"
             DESTINATION "${_android_vulkan_include}")
    endif()
    include_directories(SYSTEM "${_android_vulkan_include}")

    set(Vulkan_GLSLC_EXECUTABLE
        "${CMAKE_ANDROID_NDK}/shader-tools/linux-x86_64/glslc"
        CACHE FILEPATH "Android host glslc used to compile Vulkan shaders")
    set(Vulkan_GLSLANG_VALIDATOR_EXECUTABLE
        "${Vulkan_GLSLC_EXECUTABLE}"
        CACHE FILEPATH "Android host Vulkan shader compiler")
    if(NOT EXISTS "${Vulkan_GLSLC_EXECUTABLE}")
        message(FATAL_ERROR "Android Vulkan build requires the NDK glslc shader compiler")
    endif()

    # FindSPIRV-Headers is not supplied by the NDK.  Callers may provide their
    # own package, while the Android app uses the checked-in compatibility one.
    if(NOT SPIRV-Headers_DIR)
        set(SPIRV-Headers_DIR
            "${CMAKE_SOURCE_DIR}/examples/llama.android/lib/src/main/cpp/cmake/SPIRV-Headers"
            CACHE PATH "Android NDK SPIR-V-Headers compatibility package")
    endif()
    list(APPEND CMAKE_PREFIX_PATH "${SPIRV-Headers_DIR}")
    set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" PARENT_SCOPE)
endfunction()
