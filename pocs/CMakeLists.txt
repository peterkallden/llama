# dependencies

find_package(Threads REQUIRED)

# third-party

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (EMSCRIPTEN)
else()
    if (NOT GGML_BACKEND_DL)
        add_subdirectory(vdot)
    endif()
    if (LLAMA_AGENT_RUNTIME)
        add_subdirectory(agent)
    endif()
    if (LLAMA_MEMORY)
        add_subdirectory(memory)
    endif()
endif()
