set(TARGET llama-agent-memory)

add_library(${TARGET}
    ../agent/agent-contract.h
    memory-context.cpp
    memory-context.h
    memory-candidate.h
    memory-in-memory.cpp
    memory-in-memory.h
    memory-policy-pack.h
    memory-policy.cpp
    memory-policy.h
    memory-retrieval.cpp
    memory-retrieval.h
    memory-tool-service.cpp
    memory-tool-service.h
    memory-store.h
    memory-types.h
)
add_library(llama-memory ALIAS ${TARGET})

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_features(${TARGET} PUBLIC cxx_std_17)
target_link_libraries(${TARGET} PUBLIC llama-common)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

if (LLAMA_MEMORY_COZO)
    if (NOT COZO_INCLUDE_DIR)
        find_path(COZO_INCLUDE_DIR cozo_c.h)
    endif()
    if (NOT COZO_LIBRARY)
        find_library(COZO_LIBRARY NAMES cozo_c cozo)
    endif()

    if (NOT COZO_INCLUDE_DIR)
        message(FATAL_ERROR "LLAMA_MEMORY_COZO=ON requires Cozo C API headers. Set COZO_INCLUDE_DIR to the directory containing cozo_c.h.")
    endif()
    if (NOT COZO_LIBRARY)
        message(FATAL_ERROR "LLAMA_MEMORY_COZO=ON requires a Cozo C API library. Set COZO_LIBRARY to the Cozo library path.")
    endif()

    target_sources(${TARGET} PRIVATE
        cozo/cozo-schema.cpp
        cozo/cozo-schema.h
        cozo/memory-cozo.cpp
        cozo/memory-cozo.h
    )
    target_compile_definitions(${TARGET} PUBLIC LLAMA_MEMORY_USE_COZO)
    llama_link_cozo(${TARGET})
endif()
