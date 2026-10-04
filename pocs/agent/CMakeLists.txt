# Agent POC, smoke and CTest definitions. Production targets are owned by
# tools/agent and are consumed here only as dependencies.

set(TARGET llama-agent-openapi-provider-smoke)
add_executable(${TARGET}
    smoke/openapi/agent-openapi-provider-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/openapi
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-openapi-http-smoke)
add_executable(${TARGET}
    smoke/openapi/agent-openapi-http-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/openapi
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support cpp-httplib)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# This target intentionally is not registered as a default CTest: it calls
# the public Eurostat service and is therefore opt-in for live integration.
set(TARGET llama-agent-openapi-eurostat-live-smoke)
add_executable(${TARGET}
    smoke/openapi/agent-openapi-eurostat-live-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/openapi
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# This target intentionally is not registered as a default CTest: it calls
# the public OpenAlex service and is therefore opt-in for live integration.
set(TARGET llama-agent-openapi-openalex-live-smoke)
add_executable(${TARGET}
    smoke/openapi/agent-openapi-openalex-live-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/openapi
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Smoke-only model-format probe. It uses the resident server-context path and
# renders the existing OpenAPI-derived model-facing contract, but deliberately
# leaves the native production tool parser disabled. This target never changes
# production prompts, parsers, or execution wiring.
set(TARGET llama-agent-openalex-compact-dsl-model-smoke)
add_executable(${TARGET}
    smoke/cli/agent-cli-openalex-compact-dsl-model-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/openapi
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# This target intentionally remains opt-in and is not part of the installed
# package.  It calls the live Eurostat service and is neither a default CTest
# nor part of the agent test build pack.

set(TARGET llama-agent-resident-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-resident-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/cli
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime
    ${PROJECT_SOURCE_DIR}/tools/agent/tooling
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-model-residency-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-model-residency-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-engine)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Optional model-backed loader smoke. It intentionally is not registered with
# CTest: operators must provide two explicit GGUF paths before it may load
# model files.
set(TARGET llama-agent-two-model-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-two-model-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-engine)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Optional model-backed smoke. It exercises one real resident inference turn
# with adaptation capture enabled; the full offline promotion lifecycle is
# covered by llama-agent-adaptation-e2e-smoke above.
set(TARGET llama-agent-model-adaptation-smoke)
add_executable(${TARGET}
    smoke/adaptation/agent-learning-model-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/adaptation
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Model-free host smoke for the complete FlyDelta tool-repair loop. It uses a
# deterministic wrong/correct tool fixture and never claims to exercise a
# real model's hidden states.
set(TARGET llama-agent-flydelta-tool-repair-smoke)
add_executable(${TARGET}
    smoke/adaptation/repair/tool.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE llama-agent-core)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Model-free smoke for donor qualification, surface residualization and the
# bounded four-arm augmentation control set. Real-model work remains opt-in.
set(TARGET llama-agent-flydelta-representation-augmentation-smoke)
add_executable(${TARGET}
    smoke/adaptation/representation/augmentation.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE llama-agent-core)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Host-taught concept smoke. Offline mode exercises all concept families with
# deterministic matched trajectories; --model runs the same builder from
# real model-host captures and reports unresolved concepts without granting
# learning credit.
set(TARGET llama-agent-flydelta-concept-smoke)
add_executable(${TARGET}
    smoke/adaptation/concept/smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/cli
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Model-free host-supported Oracle smoke. It uses the native Cozo dataset
# registry and the shared Oracle suite; no model output is treated as learning
# credit. Sanitizer builds intentionally omit Cozo, so they do not build this
# backend-specific smoke.
if (LLAMA_MEMORY_COZO)
    set(TARGET llama-agent-flydelta-oracle-smoke)
    add_executable(${TARGET}
        smoke/adaptation/concept/oracle.cpp
    )
    target_include_directories(${TARGET} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}
    )
    target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
endif()
# Model-free host contract smoke for the procedure/blueprint teaching seam.
# It validates host admission and explicit contrast provenance; model-backed
# concept extraction remains covered by the existing concept smoke.
set(TARGET llama-agent-flydelta-procedure-teaching-smoke)
add_executable(${TARGET}
    smoke/adaptation/procedure/teaching.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE llama-agent-core)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Model-free functional smoke for the complete teaching-material capture path:
# relation readiness, bounded capture worker slice, persisted trajectory refs,
# and the synthesis follow-up job.
set(TARGET llama-agent-flydelta-concept-capture-smoke)
add_executable(${TARGET}
    smoke/adaptation/concept/capture.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE llama-agent-core)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Optional real-model concept capture/synthesis smoke. It uses the same
# resident daemon/server-context binding as production: a localized search
# slice creates the anchor, concept capture creates matched trajectories, and
# synthesis grafts its experimental direction back into ordinary search.
set(TARGET llama-agent-flydelta-concept-synthesis-model-smoke)
add_executable(${TARGET}
    smoke/adaptation/concept/synthesis-model.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/cli
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support llama-agent-daemon-library)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Optional real-model FlyDelta A/B smoke. It is CTest-compatible but skips
# when LLAMA_AGENT_MODEL is not supplied, like the existing model smokes.
set(TARGET llama-agent-flydelta-model-ab-smoke)
add_executable(${TARGET}
    smoke/adaptation/execution/model-ab.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/cli
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Optional real-model cvec batch smoke. It uses two resident server slots and
# two concurrent generation requests with distinct FlyDelta overlays. The
# smoke is skipped unless both a model and the explicit server opt-in exist.
set(TARGET llama-agent-flydelta-cvec-batch-model-smoke)
add_executable(${TARGET}
    smoke/adaptation/execution/cvec-batch-model.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Optional real-model FlyDelta runtime smoke. It exercises the public
# session-host turn path with exactly three counterfactual arms: no-op, small
# overlay, and larger overlay. It is skipped unless a model is supplied.
set(TARGET llama-agent-flydelta-runtime-smoke)
add_executable(${TARGET}
    smoke/adaptation/execution/runtime.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Optional model-backed FlyDelta repair smoke. It exercises the complete
# host-certified bridge: real failed/repaired generations, hidden-state pair,
# repair delta, basis/DeltaMemory training, and exactly three counterfactual
# inference arms. Its deterministic host fixture uses the native Cozo dataset
# registry, so Cozo-free sanitizer builds omit the binary entirely.
if (LLAMA_MEMORY_COZO)
    set(TARGET llama-agent-flydelta-repair-model-smoke)
    add_executable(${TARGET}
        smoke/adaptation/repair/model.cpp
        smoke/adaptation/repair/support.cpp
        smoke/adaptation/repair/report.cpp
    )
    target_include_directories(${TARGET} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tools/agent/cli
        ${PROJECT_SOURCE_DIR}/tools/agent/runtime
    )
    target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
endif()

# Dedicated causal residual-patch smoke. It uses the resident server-context
# path and compares exact replacement, scaled repair delta, existing direction
# material and a norm-matched random control. It never grants learning credit.
set(TARGET llama-agent-flydelta-causal-patch-model-smoke)
add_executable(${TARGET}
    smoke/adaptation/causal/patch-model.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/cli
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Optional natural-language dataset question smoke. It uses the same compact
# model-facing contract as the host, captures bounded hidden states, and
# reports mismatches as repair observations. It never promotes a mismatch to
# FlyDelta evidence without a separate host verifier.
set(TARGET llama-agent-flydelta-dataset-question-model-smoke)
add_executable(${TARGET}
    smoke/adaptation/dataset/question-model.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/cli
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Always-runnable contract smoke for the checked-in suite and its plans.
set(TARGET llama-agent-flydelta-dataset-question-contract-smoke)
add_executable(${TARGET}
    smoke/adaptation/dataset/question-contract.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE llama-agent-core)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Always-runnable contract smoke for teacher-forced contrast validity. It
# prevents identical positive/negative continuations from becoming a silent
# zero-margin diagnostic.
set(TARGET llama-agent-flydelta-teacher-forced-contrast-contract-smoke)
add_executable(${TARGET}
    smoke/adaptation/contracts/teacher-forced-contrast.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE llama-agent-core)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Deterministic replay of model-facing repair failures. This exercises the
# durable learning observation seam without loading a model or a data backend.
set(TARGET llama-agent-flydelta-repair-echo-replay-smoke)
add_executable(${TARGET}
    smoke/adaptation/repair/echo-replay.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE llama-agent-core)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# The repair contract uses the same native data adapters and Cozo-backed
# fixture as a host turn. Keep it conditional because Android intentionally
# omits Cozo from its reduced agent build.
if (LLAMA_MEMORY_COZO)
    set(TARGET llama-agent-flydelta-dataset-question-repair-contract-smoke)
    add_executable(${TARGET}
        smoke/adaptation/repair/dataset-question-contract.cpp
    )
    target_include_directories(${TARGET} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tools/agent/data
    )
    target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
    target_compile_features(${TARGET} PRIVATE cxx_std_17)

    # Optional real-model companion to the contract smoke. It captures a
    # natural failed generation and a host-guided repaired generation, then
    # feeds only host-certified pairs into the existing direction search.
    set(TARGET llama-agent-flydelta-dataset-question-repair-model-smoke)
    add_executable(${TARGET}
        smoke/adaptation/repair/dataset-question-model.cpp
    )
    target_include_directories(${TARGET} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tools/agent/cli
        ${PROJECT_SOURCE_DIR}/tools/agent/runtime
        ${PROJECT_SOURCE_DIR}/tools/agent/data
    )
    target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
endif()

set(TARGET llama-agent-tool-provider-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-tool-provider-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/cli
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime
    ${PROJECT_SOURCE_DIR}/tools/agent/tooling
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-tool-runtime-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-tool-runtime-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/cli
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime
    ${PROJECT_SOURCE_DIR}/tools/agent/tooling
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-continuation-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-continuation-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# This legacy runtime-driver collection is kept as a compatibility smoke while
# its cases are moved into focused runtime tests. It must use the runtime
# support closure rather than the core PoC library alone.
set(TARGET llama-agent-inference-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-inference-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-mcp-tool-provider-smoke)
add_executable(${TARGET}
    smoke/mcp/agent-mcp-tool-provider-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/cli
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime
    ${PROJECT_SOURCE_DIR}/tools/agent/tooling
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-sandbox-contract-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-sandbox-contract-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE
    llama-agent-sandbox
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-sandbox-docker-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-sandbox-docker-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE
    llama-agent-sandbox
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-sandbox-kubernetes-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-sandbox-kubernetes-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE
    llama-agent-sandbox
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-sandbox-lxc-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-sandbox-lxc-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE
    llama-agent-sandbox
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-workspace-operation-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-workspace-operation-smoke.cpp
    ../../common/agent/thinking/research/research-workspace-factory.cpp
    ../../common/agent/thinking/research/research-workspace.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE
    llama-common
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-mcp-stdio-client-smoke)
add_executable(${TARGET}
    smoke/mcp/agent-mcp-stdio-client-smoke.cpp
)
add_dependencies(${TARGET} llama-agent-mcp-stdio-fake-server)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/cli
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime
    ${PROJECT_SOURCE_DIR}/tools/agent/tooling
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-mcp-http-client-smoke)
add_executable(${TARGET}
    smoke/mcp/agent-mcp-http-client-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/cli
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/tools/agent/runtime
    ${PROJECT_SOURCE_DIR}/tools/agent/tooling
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support cpp-httplib)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-mcp-stdio-server-smoke)
add_executable(${TARGET}
    smoke/mcp/agent-mcp-stdio-server-smoke.cpp
)
add_dependencies(${TARGET}
    llama-agent-mcp-stdio-server
    llama-agent-mcp-stdio-fake-server
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-mcp-http-inbound-dispatcher-smoke)
add_executable(${TARGET}
    smoke/mcp/agent-mcp-http-server-smoke.cpp
    ../../common/agent/tooling/contracts/schema-contract.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/server
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support llama-agent-daemon-library cpp-httplib)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-mcp-http-vertical-smoke)
add_executable(${TARGET}
    smoke/mcp/agent-mcp-http-vertical-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/server
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support llama-agent-daemon-library cpp-httplib)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-mcp-agent-tools-smoke)
add_executable(${TARGET}
    smoke/mcp/agent-mcp-agent-tools-smoke.cpp
    ../../common/agent/tooling/contracts/schema-contract.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE llama-agent-mcp-core llama-common cpp-httplib ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-cli-mcp-selection-smoke)
add_executable(${TARGET}
    smoke/cli/agent-cli-mcp-selection-smoke.cpp
)
add_dependencies(${TARGET} llama-agent-mcp-stdio-fake-server)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-cli-run-mcp-smoke)
add_executable(${TARGET}
    smoke/cli/agent-cli-run-mcp-smoke.cpp
)
add_dependencies(${TARGET} llama-agent-mcp-stdio-fake-server)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-mcp-stdio-fake-server)
add_executable(${TARGET}
    ../../tools/agent/mcp/agent-mcp-stdio-fake-server.cpp
    ../../common/agent/tooling/contracts/schema-contract.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE
    llama-common
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-daemon-dispatcher-smoke)
add_executable(${TARGET}
    smoke/daemon/agent-daemon-dispatcher-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support llama-agent-daemon-library)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-daemon-protocol-smoke)
add_executable(${TARGET}
    smoke/daemon/agent-daemon-protocol-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support llama-agent-daemon-library)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Functional FlyDelta admin path. This is intentionally a smoke executable,
# not a contract CTest: it exercises queue/worker/lifecycle/review wiring and
# emits the trace that proves the end-to-end path.
set(TARGET llama-agent-daemon-flydelta-admin-smoke)
add_executable(${TARGET}
    smoke/daemon/flydelta/admin.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support llama-agent-daemon-library)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Functional FlyDelta admin path with the resident model host. This remains a
# smoke executable, not a contract CTest: it proves model execution,
# reference-backed fixtures, worker persistence, review and canary staging.
set(TARGET llama-agent-daemon-flydelta-model-smoke)
add_executable(${TARGET}
    smoke/daemon/flydelta/model.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support llama-agent-daemon-library)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-daemon-jsonl-protocol-smoke)
add_executable(${TARGET}
    smoke/daemon/agent-daemon-jsonl-protocol-smoke.cpp
    ../../tools/agent/daemon/agent-daemon-client-status.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/vendor
)
target_link_libraries(${TARGET} PRIVATE
    llama-agent-daemon-protocol
    ${CMAKE_THREAD_LIBS_INIT}
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-jsonl-wire-contract-smoke)
add_executable(${TARGET}
    smoke/daemon/agent-jsonl-wire-contract-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE
    llama-agent-core
    ${CMAKE_THREAD_LIBS_INIT}
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-web-sse-contract-smoke)
add_executable(${TARGET}
    smoke/daemon/agent-web-sse-contract-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE llama-agent-protocol)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-daemon-event-stream-contract-smoke)
add_executable(${TARGET}
    smoke/daemon/agent-daemon-event-stream-contract-smoke.cpp
    ../../tools/agent/daemon/agent-daemon-event-collector.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/vendor
)
target_link_libraries(${TARGET} PRIVATE
    llama-agent-daemon-protocol
    ${CMAKE_THREAD_LIBS_INIT}
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-daemon-client-fake-daemon)
add_executable(${TARGET}
    agent-daemon-client-fake-daemon.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/vendor
)
target_link_libraries(${TARGET} PRIVATE
    llama-agent-daemon-protocol
    ${CMAKE_THREAD_LIBS_INIT}
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-daemon-client-smoke)
add_executable(${TARGET}
    smoke/daemon/agent-daemon-client-smoke.cpp
)
add_dependencies(${TARGET} llama-agent-daemon-client-fake-daemon)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support llama-agent-daemon-library)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-daemon-wait-events-smoke)
add_executable(${TARGET}
    smoke/daemon/agent-daemon-wait-events-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support llama-agent-daemon-library)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-daemon-mcp-config-smoke)
add_executable(${TARGET}
    smoke/daemon/agent-daemon-mcp-config-smoke.cpp
)
add_dependencies(${TARGET} llama-agent-mcp-stdio-fake-server)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support llama-agent-daemon-library)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-openapi-catalog-smoke)
add_executable(${TARGET}
    smoke/openapi/agent-openapi-catalog-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/host
    ${PROJECT_SOURCE_DIR}/tools/agent/openapi
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-config-discovery-smoke)
add_executable(${TARGET}
    smoke/daemon/agent-config-discovery-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/host
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-runtime-session-host-tooling-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-runtime-session-host-tooling-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-runtime-session-manager-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-runtime-session-manager-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-runtime-operation-manager-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-runtime-operation-manager-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-deliberation-policy-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-deliberation-policy-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE llama-agent-core)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-deliberate-runtime-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-deliberate-runtime-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE llama-agent-core)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-research-runtime-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-research-runtime-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE llama-agent-core)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-resource-store-smoke)
add_executable(${TARGET}
    smoke/resource/agent-resource-store-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-resource-chunker-smoke)
add_executable(${TARGET}
    smoke/resource/agent-resource-chunker-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-resource-range-chunker-smoke)
add_executable(${TARGET}
    smoke/resource/agent-resource-range-chunker-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-cli-document-table-smoke)
add_executable(${TARGET}
    smoke/cli/agent-cli-document-table-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-cli-openapi-host-smoke)
add_executable(${TARGET}
    smoke/cli/agent-cli-openapi-host-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/daemon
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support cpp-httplib)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-sandbox-local-runtime-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-sandbox-local-runtime-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-resource-processing-contract-smoke)
add_executable(${TARGET}
    smoke/resource/agent-resource-processing-contract-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}/common
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-resource-backend-smoke)
add_executable(${TARGET}
    smoke/resource/agent-resource-backend-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-resource-processor-dispatch-smoke)
add_executable(${TARGET}
    smoke/resource/agent-resource-processor-dispatch-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/common
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-resource-media-type-smoke)
add_executable(${TARGET}
    smoke/resource/agent-resource-media-type-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-resource-processing-service-smoke)
add_executable(${TARGET}
    smoke/resource/agent-resource-processing-service-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../memory
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-pdf-text-processor-smoke)
add_executable(${TARGET}
    smoke/resource/agent-pdf-text-processor-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/server
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-docx-text-processor-smoke)
add_executable(${TARGET}
    smoke/resource/agent-docx-text-processor-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-dataset-contract-smoke)
add_executable(${TARGET}
    smoke/data/agent-dataset-contract-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE llama-agent-core)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (LLAMA_AGENT_STORAGE_SQLITE)
    set(TARGET llama-agent-storage-sqlite-smoke)
    add_executable(${TARGET}
        smoke/storage/agent-sqlite-memory-smoke.cpp
    )
    target_include_directories(${TARGET} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}
    )
    target_link_libraries(${TARGET} PRIVATE llama-agent-memory)
    target_compile_features(${TARGET} PRIVATE cxx_std_17)

    set(TARGET llama-agent-plan-storage-sqlite-smoke)
    add_executable(${TARGET}
        smoke/storage/agent-sqlite-plan-smoke.cpp
    )
    target_include_directories(${TARGET} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}
    )
    target_link_libraries(${TARGET} PRIVATE llama-agent-plan)
    target_compile_features(${TARGET} PRIVATE cxx_std_17)

    set(TARGET llama-agent-data-storage-sqlite-smoke)
    add_executable(${TARGET}
        smoke/storage/agent-sqlite-data-smoke.cpp
    )
    target_include_directories(${TARGET} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tools/agent/data
    )
    target_link_libraries(${TARGET} PRIVATE llama-agent-data)
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
endif()

set(TARGET llama-agent-research-csv-smoke)
add_executable(${TARGET}
    smoke/runtime/agent-research-csv-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE llama-agent-core)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-dataset-importer-smoke)
add_executable(${TARGET}
    smoke/data/agent-dataset-importer-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/data
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (LLAMA_MEMORY_COZO)
    set(TARGET llama-agent-data-manipulation-smoke)
    add_executable(${TARGET}
        smoke/data/agent-data-manipulation-smoke.cpp
    )
    target_include_directories(${TARGET} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tools/agent/data
    )
    target_link_libraries(${TARGET} PRIVATE llama-agent-storage-cozo)
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
endif()

set(TARGET llama-agent-docx-text-local-e2e-smoke)
add_executable(${TARGET}
    smoke/resource/agent-docx-text-local-e2e-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-pdf-page-image-command-smoke)
add_executable(${TARGET}
    smoke/resource/agent-pdf-page-image-command-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-tesseract-ocr-command-smoke)
add_executable(${TARGET}
    smoke/resource/agent-tesseract-ocr-command-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-resource-processing-host-smoke)
add_executable(${TARGET}
    smoke/resource/agent-resource-processing-host-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-tesseract-ocr-processor-smoke)
add_executable(${TARGET}
    smoke/resource/agent-tesseract-ocr-processor-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-pdf-page-image-processor-smoke)
add_executable(${TARGET}
    smoke/resource/agent-pdf-page-image-processor-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-pdf-page-image-local-e2e-smoke)
add_executable(${TARGET}
    smoke/resource/agent-pdf-page-image-local-e2e-smoke.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/resource
    ${PROJECT_SOURCE_DIR}/tools/agent/resource/processors
)
target_link_libraries(${TARGET} PRIVATE llama-agent-runtime-support)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-agent-adaptation-e2e-smoke)
add_executable(${TARGET}
    smoke/adaptation/agent-learning-adaptation-e2e-smoke.cpp
    ../../tools/agent/adaptation/agent-learning-adaptation-orchestrator.cpp
    ../../tools/agent/adaptation/agent-learning-worker.cpp
    ../../tools/agent/adaptation/agent-learning-adapter-import.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/agent/adaptation
)
target_link_libraries(${TARGET} PRIVATE
    llama-agent-adaptation-storage
    vendor::hash
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
set(AGENT_MCP_SERVER_LIBRARY_TARGETS
    llama-agent-mcp-stdio-fake-server
    llama-agent-mcp-stdio-server-smoke
    llama-agent-mcp-http-inbound-dispatcher-smoke
    llama-agent-mcp-http-vertical-smoke
    llama-agent-mcp-agent-tools-smoke
    llama-agent-mcp-stdio-server
    llama-agent-daemon-mcp-config-smoke
    llama-agent-openapi-catalog-smoke
    llama-agent-openapi-provider-smoke
    llama-agent-openapi-http-smoke
    llama-agent-daemon
)
foreach(TARGET ${AGENT_MCP_SERVER_LIBRARY_TARGETS})
    target_link_libraries(${TARGET} PRIVATE llama-agent-mcp-server)
endforeach()

set(AGENT_RUNTIME_TARGETS
    llama-agent-cli
    llama-agent-mcp-stdio-server
    llama-agent-daemon
    llama-agent-adaptation-worker
    llama-agent-adaptation-admin
    llama-agent-web
)

set(AGENT_TEST_BUILD_TARGETS
    llama-agent-cli-document-table-smoke
    llama-agent-adaptation-e2e-smoke
    llama-agent-model-adaptation-smoke
    llama-agent-flydelta-tool-repair-smoke
    llama-agent-flydelta-representation-augmentation-smoke
    llama-agent-flydelta-procedure-teaching-smoke
    llama-agent-flydelta-concept-capture-smoke
    llama-agent-flydelta-concept-smoke
    llama-agent-flydelta-concept-synthesis-model-smoke
    llama-agent-flydelta-model-ab-smoke
    llama-agent-flydelta-cvec-batch-model-smoke
    llama-agent-flydelta-runtime-smoke
    llama-agent-flydelta-causal-patch-model-smoke
    llama-agent-flydelta-dataset-question-model-smoke
    llama-agent-flydelta-dataset-question-contract-smoke
    llama-agent-flydelta-teacher-forced-contrast-contract-smoke
    llama-agent-flydelta-repair-echo-replay-smoke
    llama-agent-resident-smoke
    llama-agent-model-residency-smoke
    llama-agent-two-model-smoke
    llama-agent-tool-provider-smoke
    llama-agent-tool-runtime-smoke
    llama-agent-continuation-smoke
    llama-agent-inference-smoke
    llama-agent-sandbox-contract-smoke
    llama-agent-sandbox-local-runtime-smoke
    llama-agent-sandbox-docker-smoke
    llama-agent-sandbox-kubernetes-smoke
    llama-agent-sandbox-lxc-smoke
    llama-agent-workspace-operation-smoke
    llama-agent-mcp-tool-provider-smoke
    llama-agent-mcp-stdio-client-smoke
    llama-agent-mcp-http-client-smoke
    llama-agent-mcp-stdio-server-smoke
    llama-agent-mcp-http-inbound-dispatcher-smoke
    llama-agent-mcp-http-vertical-smoke
    llama-agent-mcp-agent-tools-smoke
    llama-agent-cli-mcp-selection-smoke
    llama-agent-cli-run-mcp-smoke
    llama-agent-daemon-dispatcher-smoke
    llama-agent-daemon-protocol-smoke
    llama-agent-daemon-jsonl-protocol-smoke
    llama-agent-jsonl-wire-contract-smoke
    llama-agent-web-sse-contract-smoke
    llama-agent-daemon-event-stream-contract-smoke
    llama-agent-daemon-client-smoke
    llama-agent-daemon-wait-events-smoke
    llama-agent-daemon-mcp-config-smoke
    llama-agent-config-discovery-smoke
    llama-agent-runtime-session-host-tooling-smoke
    llama-agent-runtime-session-manager-smoke
    llama-agent-runtime-operation-manager-smoke
    llama-agent-deliberation-policy-smoke
    llama-agent-resource-store-smoke
    llama-agent-resource-chunker-smoke
    llama-agent-resource-range-chunker-smoke
    llama-agent-resource-processing-contract-smoke
    llama-agent-resource-backend-smoke
    llama-agent-resource-processor-dispatch-smoke
    llama-agent-resource-media-type-smoke
    llama-agent-resource-processing-service-smoke
    llama-agent-pdf-text-processor-smoke
    llama-agent-docx-text-processor-smoke
    llama-agent-dataset-contract-smoke
    llama-agent-dataset-importer-smoke
    llama-agent-docx-text-local-e2e-smoke
    llama-agent-pdf-page-image-command-smoke
    llama-agent-tesseract-ocr-command-smoke
    llama-agent-tesseract-ocr-processor-smoke
    llama-agent-resource-processing-host-smoke
    llama-agent-pdf-page-image-processor-smoke
    llama-agent-pdf-page-image-local-e2e-smoke
    llama-agent-deliberate-runtime-smoke
    llama-agent-research-runtime-smoke
    llama-agent-research-csv-smoke
)
if (LLAMA_MEMORY_COZO)
    list(APPEND AGENT_TEST_BUILD_TARGETS
        llama-agent-data-manipulation-smoke
        llama-agent-flydelta-oracle-smoke
        llama-agent-flydelta-repair-model-smoke
        llama-agent-flydelta-dataset-question-repair-contract-smoke
        llama-agent-flydelta-dataset-question-repair-model-smoke)
endif()

set(AGENT_RUNTIME_SMOKE_RUNTIME_TARGETS
    llama-agent-adaptation-e2e-smoke
    llama-agent-model-adaptation-smoke
    llama-agent-flydelta-tool-repair-smoke
    llama-agent-flydelta-procedure-teaching-smoke
    llama-agent-flydelta-concept-synthesis-model-smoke
    llama-agent-flydelta-model-ab-smoke
    llama-agent-flydelta-cvec-batch-model-smoke
    llama-agent-flydelta-runtime-smoke
    llama-agent-flydelta-dataset-question-model-smoke
    llama-agent-flydelta-dataset-question-repair-model-smoke
    llama-agent-resident-smoke
    llama-agent-model-residency-smoke
    llama-agent-tool-provider-smoke
    llama-agent-tool-runtime-smoke
    llama-agent-continuation-smoke
    llama-agent-inference-smoke
    llama-agent-sandbox-contract-smoke
    llama-agent-sandbox-local-runtime-smoke
    llama-agent-sandbox-docker-smoke
    llama-agent-sandbox-kubernetes-smoke
    llama-agent-sandbox-lxc-smoke
    llama-agent-workspace-operation-smoke
    llama-agent-runtime-session-host-tooling-smoke
    llama-agent-runtime-session-manager-smoke
    llama-agent-runtime-operation-manager-smoke
    llama-agent-deliberation-policy-smoke
    llama-agent-deliberate-runtime-smoke
    llama-agent-research-runtime-smoke
)
if (LLAMA_MEMORY_COZO)
    list(APPEND AGENT_RUNTIME_SMOKE_RUNTIME_TARGETS
        llama-agent-flydelta-repair-model-smoke)
endif()

set(AGENT_RUNTIME_SMOKE_MCP_TARGETS
    llama-agent-mcp-tool-provider-smoke
    llama-agent-mcp-stdio-client-smoke
    llama-agent-mcp-http-client-smoke
    llama-agent-mcp-stdio-server-smoke
    llama-agent-mcp-http-inbound-dispatcher-smoke
    llama-agent-mcp-http-vertical-smoke
    llama-agent-mcp-agent-tools-smoke
)

set(AGENT_RUNTIME_SMOKE_CLI_TARGETS
    llama-agent-cli-mcp-selection-smoke
    llama-agent-cli-run-mcp-smoke
    llama-agent-cli-document-table-smoke
)

set(AGENT_RUNTIME_SMOKE_DAEMON_TARGETS
    llama-agent-daemon-dispatcher-smoke
    llama-agent-daemon-protocol-smoke
    llama-agent-daemon-jsonl-protocol-smoke
    llama-agent-daemon-event-stream-contract-smoke
    llama-agent-daemon-client-smoke
    llama-agent-daemon-wait-events-smoke
    llama-agent-daemon-mcp-config-smoke
    llama-agent-config-discovery-smoke
)

set(AGENT_RUNTIME_SMOKE_RESOURCE_TARGETS
    llama-agent-resource-store-smoke
    llama-agent-resource-chunker-smoke
    llama-agent-resource-range-chunker-smoke
    llama-agent-resource-processing-contract-smoke
    llama-agent-resource-backend-smoke
    llama-agent-resource-processor-dispatch-smoke
    llama-agent-resource-media-type-smoke
    llama-agent-resource-processing-service-smoke
    llama-agent-pdf-text-processor-smoke
    llama-agent-docx-text-processor-smoke
    llama-agent-dataset-contract-smoke
    llama-agent-dataset-importer-smoke
    llama-agent-docx-text-local-e2e-smoke
    llama-agent-pdf-page-image-command-smoke
    llama-agent-resource-processing-host-smoke
    llama-agent-pdf-page-image-processor-smoke
    llama-agent-pdf-page-image-local-e2e-smoke
)

if (BUILD_TESTING)
    if (LLAMA_AGENT_STORAGE_SQLITE)
        add_test(NAME llama-agent-storage-sqlite-ctest
            COMMAND $<TARGET_FILE:llama-agent-storage-sqlite-smoke>)
        set_property(TEST llama-agent-storage-sqlite-ctest PROPERTY LABELS agent-storage)
        add_test(NAME llama-agent-plan-storage-sqlite-ctest
            COMMAND $<TARGET_FILE:llama-agent-plan-storage-sqlite-smoke>)
        set_property(TEST llama-agent-plan-storage-sqlite-ctest PROPERTY LABELS agent-storage)
        add_test(NAME llama-agent-data-storage-sqlite-ctest
            COMMAND $<TARGET_FILE:llama-agent-data-storage-sqlite-smoke>)
        set_property(TEST llama-agent-data-storage-sqlite-ctest PROPERTY LABELS agent-storage)
    endif()

    add_test(NAME llama-agent-tool-provider-ctest
        COMMAND $<TARGET_FILE:llama-agent-tool-provider-smoke>)
    set_property(TEST llama-agent-tool-provider-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-model-residency-ctest
        COMMAND $<TARGET_FILE:llama-agent-model-residency-smoke>)
    set_property(TEST llama-agent-model-residency-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-adaptation-e2e-ctest
        COMMAND $<TARGET_FILE:llama-agent-adaptation-e2e-smoke>)
    set_property(TEST llama-agent-adaptation-e2e-ctest PROPERTY LABELS agent-adaptation)

    add_test(NAME llama-agent-model-adaptation-ctest
        COMMAND $<TARGET_FILE:llama-agent-model-adaptation-smoke>)
    set_property(TEST llama-agent-model-adaptation-ctest PROPERTY LABELS agent-adaptation)
    set_tests_properties(llama-agent-model-adaptation-ctest PROPERTIES
        SKIP_RETURN_CODE 77 TIMEOUT 180)

    add_test(NAME llama-agent-flydelta-tool-repair-ctest
        COMMAND $<TARGET_FILE:llama-agent-flydelta-tool-repair-smoke>)
    set_property(TEST llama-agent-flydelta-tool-repair-ctest PROPERTY LABELS agent-adaptation)

    add_test(NAME llama-agent-flydelta-representation-augmentation-ctest
        COMMAND $<TARGET_FILE:llama-agent-flydelta-representation-augmentation-smoke>)
    set_property(TEST llama-agent-flydelta-representation-augmentation-ctest PROPERTY LABELS agent-adaptation)

    add_test(NAME llama-agent-flydelta-concept-ctest
        COMMAND $<TARGET_FILE:llama-agent-flydelta-concept-smoke>)
    set_property(TEST llama-agent-flydelta-concept-ctest PROPERTY LABELS agent-adaptation)
    set_tests_properties(llama-agent-flydelta-concept-ctest PROPERTIES TIMEOUT 60)

    if (TARGET llama-agent-flydelta-oracle-smoke)
        add_test(NAME llama-agent-flydelta-oracle-ctest
            COMMAND $<TARGET_FILE:llama-agent-flydelta-oracle-smoke>)
        set_property(TEST llama-agent-flydelta-oracle-ctest PROPERTY LABELS agent-adaptation)
    endif()

    add_test(NAME llama-agent-flydelta-procedure-teaching-ctest
        COMMAND $<TARGET_FILE:llama-agent-flydelta-procedure-teaching-smoke>)
    set_property(TEST llama-agent-flydelta-procedure-teaching-ctest PROPERTY LABELS agent-adaptation)

    add_test(NAME llama-agent-flydelta-concept-capture-ctest
        COMMAND $<TARGET_FILE:llama-agent-flydelta-concept-capture-smoke>)
    set_property(TEST llama-agent-flydelta-concept-capture-ctest PROPERTY LABELS agent-adaptation)
    set_tests_properties(llama-agent-flydelta-concept-capture-ctest PROPERTIES TIMEOUT 60)

    add_test(NAME llama-agent-flydelta-concept-synthesis-model-ctest
        COMMAND $<TARGET_FILE:llama-agent-flydelta-concept-synthesis-model-smoke>)
    set_property(TEST llama-agent-flydelta-concept-synthesis-model-ctest PROPERTY LABELS agent-adaptation)
    set_tests_properties(llama-agent-flydelta-concept-synthesis-model-ctest PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 1200
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

    add_test(NAME llama-agent-flydelta-model-ab-ctest
        COMMAND $<TARGET_FILE:llama-agent-flydelta-model-ab-smoke>)
    set_property(TEST llama-agent-flydelta-model-ab-ctest PROPERTY LABELS agent-adaptation)
    set_tests_properties(llama-agent-flydelta-model-ab-ctest PROPERTIES
        SKIP_RETURN_CODE 77 TIMEOUT 180)

    add_test(NAME llama-agent-flydelta-cvec-batch-model-ctest
        COMMAND $<TARGET_FILE:llama-agent-flydelta-cvec-batch-model-smoke>)
    set_property(TEST llama-agent-flydelta-cvec-batch-model-ctest PROPERTY LABELS agent-adaptation)
    set_tests_properties(llama-agent-flydelta-cvec-batch-model-ctest PROPERTIES
        SKIP_RETURN_CODE 77 TIMEOUT 600)

    add_test(NAME llama-agent-flydelta-runtime-ctest
        COMMAND $<TARGET_FILE:llama-agent-flydelta-runtime-smoke>)
    set_property(TEST llama-agent-flydelta-runtime-ctest PROPERTY LABELS agent-adaptation)
    set_tests_properties(llama-agent-flydelta-runtime-ctest PROPERTIES
        SKIP_RETURN_CODE 77 TIMEOUT 300)

    if (TARGET llama-agent-flydelta-repair-model-smoke)
        add_test(NAME llama-agent-flydelta-repair-model-ctest
            COMMAND $<TARGET_FILE:llama-agent-flydelta-repair-model-smoke>
                --n-gpu-layers 99)
        set_property(TEST llama-agent-flydelta-repair-model-ctest PROPERTY LABELS agent-adaptation)
        set_tests_properties(llama-agent-flydelta-repair-model-ctest PROPERTIES
            SKIP_RETURN_CODE 77 TIMEOUT 900)
    endif()

    add_test(NAME llama-agent-flydelta-causal-patch-model-ctest
        COMMAND $<TARGET_FILE:llama-agent-flydelta-causal-patch-model-smoke>)
    set_property(TEST llama-agent-flydelta-causal-patch-model-ctest PROPERTY LABELS agent-adaptation)
    set_tests_properties(llama-agent-flydelta-causal-patch-model-ctest PROPERTIES
        SKIP_RETURN_CODE 77 TIMEOUT 900)

    add_test(NAME llama-agent-flydelta-dataset-question-contract-ctest
        COMMAND $<TARGET_FILE:llama-agent-flydelta-dataset-question-contract-smoke>
            ${PROJECT_SOURCE_DIR}/docs/examples/agent-flydelta-dataset-question-suite.json)
    set_property(TEST llama-agent-flydelta-dataset-question-contract-ctest PROPERTY LABELS agent-adaptation)

    add_test(NAME llama-agent-flydelta-teacher-forced-contrast-contract-ctest
        COMMAND $<TARGET_FILE:llama-agent-flydelta-teacher-forced-contrast-contract-smoke>)
    set_property(TEST llama-agent-flydelta-teacher-forced-contrast-contract-ctest PROPERTY LABELS agent-adaptation)

    add_test(NAME llama-agent-flydelta-repair-echo-replay-ctest
        COMMAND $<TARGET_FILE:llama-agent-flydelta-repair-echo-replay-smoke>
            ${PROJECT_SOURCE_DIR}/docs/examples/agent-flydelta-dataset-repair-echo-replay.jsonl)
    set_property(TEST llama-agent-flydelta-repair-echo-replay-ctest PROPERTY LABELS agent-adaptation)

    if (LLAMA_MEMORY_COZO)
        add_test(NAME llama-agent-flydelta-dataset-question-repair-contract-ctest
            COMMAND $<TARGET_FILE:llama-agent-flydelta-dataset-question-repair-contract-smoke>
                ${PROJECT_SOURCE_DIR}/docs/examples/agent-flydelta-dataset-question-suite.json)
        set_property(TEST llama-agent-flydelta-dataset-question-repair-contract-ctest PROPERTY LABELS agent-adaptation)

        add_test(NAME llama-agent-flydelta-dataset-question-repair-model-ctest
            COMMAND $<TARGET_FILE:llama-agent-flydelta-dataset-question-repair-model-smoke>
                --suite ${PROJECT_SOURCE_DIR}/docs/examples/agent-flydelta-dataset-question-suite.json
                --n-gpu-layers 99)
        set_property(TEST llama-agent-flydelta-dataset-question-repair-model-ctest PROPERTY LABELS agent-adaptation)
        set_tests_properties(llama-agent-flydelta-dataset-question-repair-model-ctest PROPERTIES
            SKIP_RETURN_CODE 77 TIMEOUT 1800)
    endif()

    add_test(NAME llama-agent-flydelta-dataset-question-model-ctest
        COMMAND $<TARGET_FILE:llama-agent-flydelta-dataset-question-model-smoke>
            --suite ${PROJECT_SOURCE_DIR}/docs/examples/agent-flydelta-dataset-question-suite.json
            --n-gpu-layers 99)
    set_property(TEST llama-agent-flydelta-dataset-question-model-ctest PROPERTY LABELS agent-adaptation)
    set_tests_properties(llama-agent-flydelta-dataset-question-model-ctest PROPERTIES
        SKIP_RETURN_CODE 77 TIMEOUT 1800)

    add_test(NAME llama-agent-tool-runtime-ctest
        COMMAND $<TARGET_FILE:llama-agent-tool-runtime-smoke>)
    set_property(TEST llama-agent-tool-runtime-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-continuation-ctest
        COMMAND $<TARGET_FILE:llama-agent-continuation-smoke>)
    set_property(TEST llama-agent-continuation-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-deliberate-runtime-ctest
        COMMAND $<TARGET_FILE:llama-agent-deliberate-runtime-smoke>)
    set_property(TEST llama-agent-deliberate-runtime-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-research-runtime-ctest
        COMMAND $<TARGET_FILE:llama-agent-research-runtime-smoke>)
    set_property(TEST llama-agent-research-runtime-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-research-csv-ctest
        COMMAND $<TARGET_FILE:llama-agent-research-csv-smoke>)
    set_property(TEST llama-agent-research-csv-ctest PROPERTY LABELS agent)
    set_tests_properties(llama-agent-research-csv-ctest PROPERTIES
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

    add_test(NAME llama-agent-inference-ctest
        COMMAND $<TARGET_FILE:llama-agent-inference-smoke>)
    set_property(TEST llama-agent-inference-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-sandbox-contract-ctest
        COMMAND $<TARGET_FILE:llama-agent-sandbox-contract-smoke>)
    set_property(TEST llama-agent-sandbox-contract-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-sandbox-local-runtime-ctest
        COMMAND $<TARGET_FILE:llama-agent-sandbox-local-runtime-smoke>)
    set_property(TEST llama-agent-sandbox-local-runtime-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-workspace-operation-ctest
        COMMAND $<TARGET_FILE:llama-agent-workspace-operation-smoke>)
    set_property(TEST llama-agent-workspace-operation-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-cli-run-mcp-ctest
        COMMAND $<TARGET_FILE:llama-agent-cli-run-mcp-smoke>)
    set_property(TEST llama-agent-cli-run-mcp-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-cli-mcp-selection-ctest
        COMMAND $<TARGET_FILE:llama-agent-cli-mcp-selection-smoke>)
    set_property(TEST llama-agent-cli-mcp-selection-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-cli-document-table-ctest
        COMMAND $<TARGET_FILE:llama-agent-cli-document-table-smoke>)
    set_property(TEST llama-agent-cli-document-table-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-cli-openapi-host-ctest
        COMMAND $<TARGET_FILE:llama-agent-cli-openapi-host-smoke>)
    set_property(TEST llama-agent-cli-openapi-host-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-deliberation-policy-ctest
        COMMAND $<TARGET_FILE:llama-agent-deliberation-policy-smoke>)
    set_property(TEST llama-agent-deliberation-policy-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-mcp-stdio-client-ctest
        COMMAND $<TARGET_FILE:llama-agent-mcp-stdio-client-smoke>)
    set_property(TEST llama-agent-mcp-stdio-client-ctest PROPERTY LABELS agent)
    # The fake stdio server deliberately exercises process shutdown and pipe
    # diagnostics. Run it serially so concurrent MCP smoke processes cannot
    # turn the child-side close into a nondeterministic SIGPIPE.
    set_tests_properties(llama-agent-mcp-stdio-client-ctest PROPERTIES TIMEOUT 120 RUN_SERIAL TRUE)

    add_test(NAME llama-agent-mcp-http-client-ctest
        COMMAND $<TARGET_FILE:llama-agent-mcp-http-client-smoke>)
    set_property(TEST llama-agent-mcp-http-client-ctest PROPERTY LABELS agent)
    set_tests_properties(llama-agent-mcp-http-client-ctest PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE)

    add_test(NAME llama-agent-daemon-protocol-ctest
        COMMAND $<TARGET_FILE:llama-agent-daemon-protocol-smoke>)
    set_property(TEST llama-agent-daemon-protocol-ctest PROPERTY LABELS agent)

    # The protocol smoke contains the executable readiness-gate contract. Keep
    # a separately named CTest entry so assurance reports expose readiness as
    # an explicit check without introducing a second daemon fixture.
    add_test(NAME llama-agent-daemon-readiness-ctest
        COMMAND $<TARGET_FILE:llama-agent-daemon-protocol-smoke>)
    set_property(TEST llama-agent-daemon-readiness-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-daemon-jsonl-protocol-ctest
        COMMAND $<TARGET_FILE:llama-agent-daemon-jsonl-protocol-smoke>)
    set_property(TEST llama-agent-daemon-jsonl-protocol-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-jsonl-wire-contract-ctest
        COMMAND $<TARGET_FILE:llama-agent-jsonl-wire-contract-smoke>)
    set_property(TEST llama-agent-jsonl-wire-contract-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-web-sse-contract-ctest
        COMMAND $<TARGET_FILE:llama-agent-web-sse-contract-smoke>)
    set_property(TEST llama-agent-web-sse-contract-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-daemon-client-ctest
        COMMAND $<TARGET_FILE:llama-agent-daemon-client-smoke>)
    set_property(TEST llama-agent-daemon-client-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-daemon-mcp-config-ctest
        COMMAND $<TARGET_FILE:llama-agent-daemon-mcp-config-smoke>)
    set_property(TEST llama-agent-daemon-mcp-config-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-openapi-catalog-ctest
        COMMAND $<TARGET_FILE:llama-agent-openapi-catalog-smoke>)
    set_property(TEST llama-agent-openapi-catalog-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-openapi-provider-ctest
        COMMAND $<TARGET_FILE:llama-agent-openapi-provider-smoke>)
    set_property(TEST llama-agent-openapi-provider-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-openapi-http-ctest
        COMMAND $<TARGET_FILE:llama-agent-openapi-http-smoke>)
    set_property(TEST llama-agent-openapi-http-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-config-discovery-ctest
        COMMAND $<TARGET_FILE:llama-agent-config-discovery-smoke>)
    set_property(TEST llama-agent-config-discovery-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-resource-store-ctest
        COMMAND $<TARGET_FILE:llama-agent-resource-store-smoke>)
    set_property(TEST llama-agent-resource-store-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-resource-chunker-ctest
        COMMAND $<TARGET_FILE:llama-agent-resource-chunker-smoke>)
    set_property(TEST llama-agent-resource-chunker-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-resource-range-chunker-ctest
        COMMAND $<TARGET_FILE:llama-agent-resource-range-chunker-smoke>)
    set_property(TEST llama-agent-resource-range-chunker-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-resource-processing-contract-ctest
        COMMAND $<TARGET_FILE:llama-agent-resource-processing-contract-smoke>)
    set_property(TEST llama-agent-resource-processing-contract-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-resource-backend-ctest
        COMMAND $<TARGET_FILE:llama-agent-resource-backend-smoke>)
    set_property(TEST llama-agent-resource-backend-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-resource-processor-dispatch-ctest
        COMMAND $<TARGET_FILE:llama-agent-resource-processor-dispatch-smoke>)
    set_property(TEST llama-agent-resource-processor-dispatch-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-resource-media-type-ctest
        COMMAND $<TARGET_FILE:llama-agent-resource-media-type-smoke>)
    set_property(TEST llama-agent-resource-media-type-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-resource-processing-service-ctest
        COMMAND $<TARGET_FILE:llama-agent-resource-processing-service-smoke>)
    set_property(TEST llama-agent-resource-processing-service-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-pdf-text-processor-ctest
        COMMAND $<TARGET_FILE:llama-agent-pdf-text-processor-smoke>)
    set_property(TEST llama-agent-pdf-text-processor-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-docx-text-processor-ctest
        COMMAND $<TARGET_FILE:llama-agent-docx-text-processor-smoke>)
    set_property(TEST llama-agent-docx-text-processor-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-dataset-contract-ctest
        COMMAND $<TARGET_FILE:llama-agent-dataset-contract-smoke>)
    set_property(TEST llama-agent-dataset-contract-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-dataset-importer-ctest
        COMMAND $<TARGET_FILE:llama-agent-dataset-importer-smoke>)
    set_property(TEST llama-agent-dataset-importer-ctest PROPERTY LABELS agent)

    if (LLAMA_MEMORY_COZO)
        add_test(NAME llama-agent-data-manipulation-ctest
            COMMAND $<TARGET_FILE:llama-agent-data-manipulation-smoke>)
        set_property(TEST llama-agent-data-manipulation-ctest PROPERTY LABELS agent)
    endif()


    add_test(NAME llama-agent-pdf-page-image-command-ctest
        COMMAND $<TARGET_FILE:llama-agent-pdf-page-image-command-smoke>)
    set_property(TEST llama-agent-pdf-page-image-command-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-tesseract-ocr-command-ctest
        COMMAND $<TARGET_FILE:llama-agent-tesseract-ocr-command-smoke>)
    set_property(TEST llama-agent-tesseract-ocr-command-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-tesseract-ocr-processor-ctest
        COMMAND $<TARGET_FILE:llama-agent-tesseract-ocr-processor-smoke>)
    set_property(TEST llama-agent-tesseract-ocr-processor-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-resource-processing-host-ctest
        COMMAND $<TARGET_FILE:llama-agent-resource-processing-host-smoke>)
    set_property(TEST llama-agent-resource-processing-host-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-pdf-page-image-processor-ctest
        COMMAND $<TARGET_FILE:llama-agent-pdf-page-image-processor-smoke>)
    set_property(TEST llama-agent-pdf-page-image-processor-ctest PROPERTY LABELS agent)

    add_test(NAME llama-agent-sandbox-docker-ctest
        COMMAND $<TARGET_FILE:llama-agent-sandbox-docker-smoke>)
    set_property(TEST llama-agent-sandbox-docker-ctest PROPERTY LABELS sandbox-docker)
    set_tests_properties(llama-agent-sandbox-docker-ctest PROPERTIES SKIP_RETURN_CODE 77)

    add_test(NAME llama-agent-sandbox-kubernetes-ctest
        COMMAND $<TARGET_FILE:llama-agent-sandbox-kubernetes-smoke>)
    set_property(TEST llama-agent-sandbox-kubernetes-ctest PROPERTY LABELS sandbox-kubernetes)

    add_test(NAME llama-agent-sandbox-lxc-ctest
        COMMAND $<TARGET_FILE:llama-agent-sandbox-lxc-smoke>)
    set_property(TEST llama-agent-sandbox-lxc-ctest PROPERTY LABELS sandbox-lxc)
    set_tests_properties(llama-agent-sandbox-lxc-ctest PROPERTIES SKIP_RETURN_CODE 77)
endif()

function(llama_agent_add_smoke_group TARGET_NAME)
    add_custom_target(${TARGET_NAME} DEPENDS ${ARGN})
endfunction()

llama_agent_add_smoke_group(
    llama-agent-smoke-runtime
    ${AGENT_RUNTIME_SMOKE_RUNTIME_TARGETS}
)

llama_agent_add_smoke_group(
    llama-agent-smoke-mcp
    ${AGENT_RUNTIME_SMOKE_MCP_TARGETS}
)

llama_agent_add_smoke_group(
    llama-agent-smoke-cli
    ${AGENT_RUNTIME_SMOKE_CLI_TARGETS}
)

llama_agent_add_smoke_group(
    llama-agent-smoke-daemon
    ${AGENT_RUNTIME_SMOKE_DAEMON_TARGETS}
)

llama_agent_add_smoke_group(
    llama-agent-smoke-resource
    ${AGENT_RUNTIME_SMOKE_RESOURCE_TARGETS}
)

llama_agent_add_smoke_group(
    llama-agent-smoke-all
    ${AGENT_RUNTIME_SMOKE_RUNTIME_TARGETS}
    ${AGENT_RUNTIME_SMOKE_MCP_TARGETS}
    ${AGENT_RUNTIME_SMOKE_CLI_TARGETS}
    ${AGENT_RUNTIME_SMOKE_DAEMON_TARGETS}
    ${AGENT_RUNTIME_SMOKE_RESOURCE_TARGETS}
)

set(AGENT_TEST_PACK_CTEST_TARGETS
    test-agent-contract
    test-agent-prepared-generation
    test-agent-model-profile
    test-agent-model-catalog
    test-agent-model-profile-cache
    test-agent-continuation
    test-agent-research-contract
    test-agent-runtime-json
    test-agent-loop
    test-agent-tool-failure
    test-agent-tool-rounds
    test-memory-learning
    test-agent-learning-observation
    test-agent-learning-domain-policy
    test-agent-learning-cause
    test-agent-adaptation-evidence
    test-agent-adaptation-evidence-routing
    test-agent-synthetic-tool-cases
    test-agent-auto-training-trigger
    test-agent-learning-case
    test-agent-learning-transaction
    test-agent-concept-candidate-index
    test-agent-concept-input-routing
    test-agent-training-candidate
    test-agent-learning-corpus
    test-agent-learning-trainer
    test-agent-learning-worker
    test-agent-learning-lifecycle
    test-agent-learning-adapter
    test-agent-learning-adapter-import
    test-agent-learning-evaluation
    test-agent-flydelta
    test-agent-flydelta-contracts
    test-agent-flydelta-experiment
    test-agent-flydelta-evidence
    test-agent-flydelta-basis
    test-agent-flydelta-concept
    test-agent-flydelta-oracles
    test-agent-flydelta-teaching-material
    test-agent-flydelta-training
    test-agent-flydelta-training-corpus
    test-agent-flydelta-job
    test-agent-flydelta-queue
    test-agent-flydelta-collection
    test-agent-flydelta-worker
    test-agent-flydelta-worker-budget
    test-agent-flydelta-pipeline
    test-agent-flydelta-gate
    test-agent-flydelta-promotion
    test-agent-flydelta-overlay
    test-agent-flydelta-activation
    test-agent-flydelta-deployment
    test-agent-flydelta-sideband
    test-agent-flydelta-sideband-review-store
    test-agent-flydelta-capture-manifest
    test-agent-flydelta-repair-deltas
    test-agent-flydelta-sparse-cvec
    test-agent-flydelta-representation-diagnostics
    test-agent-flydelta-batch-benchmark
    test-agent-flydelta-dose-controller
    test-agent-flydelta-evidence-depth
    test-agent-flydelta-aggregation
    test-agent-flydelta-direction-search
    test-agent-flydelta-candidate-lifecycle
    test-agent-flydelta-runtime-observer
    test-agent-flydelta-server-binding
    test-agent-flydelta-layer-search
    test-agent-flydelta-scale-search
    test-agent-flydelta-alpha-response-search
    test-agent-flydelta-intervention-region-search
    test-agent-flydelta-whirlpool-search
    test-agent-flydelta-whirlpool-ab
    test-agent-flydelta-search-pipeline
    test-agent-flydelta-coefficient-search
    test-agent-flydelta-deep-search
    test-agent-flydelta-artifact-lifecycle
    test-agent-flydelta-hidden-state-hook
    test-agent-flydelta-layer-discovery
    test-agent-flydelta-two-pass
    test-agent-flydelta-artifact-store
    test-agent-flydelta-evaluator
    test-agent-flydelta-experiment-orchestration
    test-agent-flydelta-sideband-controller
    test-agent-flydelta-representation-augmentation
    test-agent-model-facing-tool-chain
    test-agent-learning-transaction-backends
    test-agent-flydelta-teaching-material-cozo
    test-agent-clangd-protocol
    test-agent-native-crash
    test-agent-native-crash-smoke
    test-agent-native-crash-cdb-smoke
    test-agent-config-bootstrap
    test-memory-store
    test-memory-retrieval
    test-memory-context
    test-memory-policy
    test-memory-tool-service
    test-memory-cozo
    test-plan-store
    test-plan-context
    test-plan-json
    test-plan-goal
    test-plan-memory
    test-plan-blueprint
    test-plan-scheduler
    test-plan-template
    test-plan-cozo
    test-agent-bootstrap
    test-cli-scope
    test-agent-prepared-generation
    test-agent-daemon-scope
    test-memory-learning
    test-agent-runtime-json
    test-reflection-json
    test-agent-loop
    test-agent-lifecycle-scenarios
    test-agent-tool-rounds
    test-agent-repository-tools
    test-agent-tool-failure
    test-agent-memory-search-fallback
    test-tool-registry
    test-tool-catalog
    test-tool-family-index
    test-tool-adapters
    test-tool-chat-bridge
    test-agent-tool-output-codec
    test-tool-navigation
    test-agent-data-store-cozo
    test-agent-runtime-contract
    test-agent-event-queue
    llama-agent-cli-openapi-host-smoke
    llama-agent-openapi-catalog-smoke
    llama-agent-openapi-provider-smoke
    llama-agent-openapi-http-smoke
)
set(AGENT_TEST_PACK_AVAILABLE_TARGETS)
foreach(TARGET ${AGENT_TEST_PACK_CTEST_TARGETS})
    if (TARGET ${TARGET})
        list(APPEND AGENT_TEST_PACK_AVAILABLE_TARGETS ${TARGET})
    endif()
endforeach()

add_custom_target(llama-agent-build-pack
    DEPENDS ${AGENT_RUNTIME_TARGETS}
)

if (BUILD_TESTING)
    add_dependencies(llama-agent-build-pack ${AGENT_TEST_BUILD_TARGETS} ${AGENT_TEST_PACK_AVAILABLE_TARGETS})
endif()

add_custom_target(llama-agent-ctest-pack
    DEPENDS llama-agent-build-pack
    COMMAND ${CMAKE_CTEST_COMMAND}
            --test-dir ${CMAKE_BINARY_DIR}
            -C $<CONFIG>
            -L agent
            --output-on-failure
    USES_TERMINAL
)

if (WIN32)
    add_custom_target(llama-agent-beta-test-pack
        DEPENDS llama-agent-build-pack
        COMMAND powershell.exe -NoProfile -ExecutionPolicy Bypass
                -File ${PROJECT_SOURCE_DIR}/scripts/test-agent-daemon-beta-smoke.ps1
                -BuildDir ${CMAKE_BINARY_DIR}
                -Configuration $<CONFIG>
                -IncludeCTest
        USES_TERMINAL
    )
elseif (UNIX)
    add_custom_target(llama-agent-beta-test-pack
        DEPENDS llama-agent-build-pack
        COMMAND bash ${PROJECT_SOURCE_DIR}/scripts/test-agent-daemon-beta-smoke.sh
                --build-dir ${CMAKE_BINARY_DIR}
                --configuration $<CONFIG>
                --include-ctest
        USES_TERMINAL
    )
endif()
