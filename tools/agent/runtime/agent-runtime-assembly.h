#pragma once

#include "../tooling/agent-tool-provider.h"
#include "agent/agent-inference.h"
#include "agent/agent-context-budgets.h"
#include "agent/agent-runtime.h"
#include "agent/learning/memory-learning.h"
#include "agent/adaptation/learning-transaction.h"
#include "agent/adaptation/flydelta/flydelta-capture.h"
#include "agent/adaptation/flydelta/flydelta-teaching-material.h"
#include "agent/adaptation/flydelta/flydelta-teaching-relation.h"
#include "agent/adaptation/flydelta/flydelta-runtime-observer.h"
#include "../adaptation/agent-learning-transaction-store.h"
#include "agent/runtime/agent-inference-contracts.h"

#include "chat.h"
#include "llama.h"
#include "memory/memory-store.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

struct common_agent_inference_capabilities {
    bool text = true;
    bool image = false;
    bool audio = false;
};

bool parse_agent_inference_backend(const std::string & value, agent_inference_backend & backend);

// Host-owned observer for resolved, verified teaching material. The runtime
// assembly only forwards the relation; persistence/readiness remain outside
// the generic runtime and may use the existing host artifact store.
using common_agent_flydelta_teaching_material_observer = std::function<bool(
        const common_flydelta_teaching_relation & relation,
        std::string & error)>;

struct common_agent_generation_config {
    int n_predict = 0;
    int n_threads = 2;
    bool generation_trace = false;
    size_t context_size_tokens = 0;
    common_agent_context_budget_config context_budgets;
    // Optional host-side preflight that narrows the model-facing tool view by
    // generated family before the planner sees individual tool contracts.
    bool enable_tool_family_routing = false;
};

struct common_agent_runtime_config {
    common_agent_generation_config generation_config;
    common_agent_context_budget_config context_budgets;
    bool enable_memory_learning = false;
    common_memory_learning_config memory_learning_config;
    std::function<bool(const std::string & text, std::vector<float> & embedding, std::string & error)> embed_memory;
    // Bounded internal inference slices after a generation limit. Zero
    // disables automatic continuation for compatibility.
    size_t max_continuations = 2;
    common_agent_context_token_estimator context_token_estimator;
    // Adaptation capture is opt-in. An empty transaction path uses an
    // assembly-local store, which is useful for embedding hosts and tests.
    bool enable_adaptation_capture = false;
    common_learning_transaction_observer_config adaptation_config;
    std::string adaptation_transaction_backend = "auto";
    std::string adaptation_transaction_path;
    bool enable_flydelta_candidate_lifecycle = false;
    std::string flydelta_lifecycle_backend = "auto";
    std::string flydelta_lifecycle_path;
    bool enable_flydelta_capture_candidates = false;
    std::string flydelta_model_profile_fingerprint;
    std::string flydelta_capture_layout_revision;
    size_t flydelta_max_capture_candidates = 64;
    // Host-owned bridge that enqueues a reference-only donor_capture job.
    // It is best-effort and must not make a user turn fail on queue pressure.
    std::function<bool(
            const common_flydelta_capture_candidate &,
            std::string &)> flydelta_capture_job_enqueue;
    // Host-owned semantic resolver. It supplies immutable procedure/blueprint
    // refs; the assembly turns a resolved relation into the existing
    // reference-only FlyDelta capture path.
    common_agent_procedure_teaching_request_provider procedure_teaching_request_provider;
    // Host-owned correction resolver. It must resolve the observed execution,
    // semantic repair and contrast before returning a request; free correction
    // text is never passed to FlyDelta as training material.
    common_agent_user_correction_teaching_request_provider user_correction_teaching_request_provider;
    common_agent_user_taught_concept_relation_provider user_taught_concept_relation_provider;
    common_agent_flydelta_teaching_material_observer flydelta_teaching_material_observer;
    // Optional shared host-owned material index. When present and no explicit
    // observer is supplied, the runtime assembly observes resolved relations
    // in this index so the model-host readiness callback can inspect the same
    // state.
    std::shared_ptr<common_flydelta_teaching_material_runtime> flydelta_teaching_material_runtime;
};

struct common_agent_runtime_build_config {
    common_agent_generation_config generation_config;
    common_agent_context_budget_config context_budgets;
    bool enable_memory_learning = false;
    common_memory_learning_config memory_learning_config;
    std::function<bool(const std::string & text, std::vector<float> & embedding, std::string & error)> embed_memory;
    size_t max_continuations = 2;
    common_agent_context_token_estimator context_token_estimator;
    bool enable_adaptation_capture = false;
    common_learning_transaction_observer_config adaptation_config;
    std::string adaptation_transaction_backend = "auto";
    std::string adaptation_transaction_path;
    bool enable_flydelta_candidate_lifecycle = false;
    std::string flydelta_lifecycle_backend = "auto";
    std::string flydelta_lifecycle_path;
    bool enable_flydelta_capture_candidates = false;
    std::string flydelta_model_profile_fingerprint;
    std::string flydelta_capture_layout_revision;
    size_t flydelta_max_capture_candidates = 64;
    std::function<bool(
            const common_flydelta_capture_candidate &,
            std::string &)> flydelta_capture_job_enqueue;
    common_agent_procedure_teaching_request_provider procedure_teaching_request_provider;
    common_agent_user_correction_teaching_request_provider user_correction_teaching_request_provider;
    common_agent_user_taught_concept_relation_provider user_taught_concept_relation_provider;
    common_agent_flydelta_teaching_material_observer flydelta_teaching_material_observer;
    std::shared_ptr<common_flydelta_teaching_material_runtime> flydelta_teaching_material_runtime;
};

common_agent_inference_options make_agent_inference_options(
    common_agent_inference_options config);

common_agent_runtime_config make_agent_runtime_config(
    common_agent_runtime_build_config config);

struct common_agent_inference_session {
    agent_inference_backend backend = agent_inference_backend::cli;
    common_agent_inference_capabilities capabilities;
    std::shared_ptr<void> keepalive;
    llama_model * model = nullptr;
    const common_chat_templates * templates = nullptr;
    // Host-visible identity for readiness/tracing.  This is deliberately
    // metadata only; the model never selects or mutates its own profile.
    std::string profile_id;
    std::string profile_cache_key;
    std::unique_ptr<common_agent_inference> inference;
};

struct common_agent_runtime_assembly {
    std::unique_ptr<common_planner> planner;
    std::unique_ptr<common_action_executor> executor;
    std::unique_ptr<common_reflection_engine> reflector;
    std::unique_ptr<common_memory_candidate_extractor> candidate_extractor;
    std::unique_ptr<common_memory_post_turn_learner> memory_learner;
    std::unique_ptr<common_learning_transaction_store> adaptation_store;
    std::unique_ptr<common_learning_lifecycle_store> flydelta_lifecycle_store;
    std::unique_ptr<common_learning_transaction_observer> adaptation_observer;
    std::unique_ptr<common_flydelta_capture_candidate_collector> flydelta_capture_collector;
    std::unique_ptr<common_flydelta_runtime_candidate_observer> flydelta_runtime_candidate_observer;
    std::string adaptation_error;
    std::unique_ptr<common_agent_tool_runtime> tool_runtime;
    std::unique_ptr<common_agent_runtime> runtime;
};

common_agent_runtime_assembly make_agent_runtime_assembly(
    common_memory_store & memory_store,
    common_plan_store & plan_store,
    common_agent_inference & inference,
    const common_agent_runtime_config & runtime_config,
    const std::vector<common_chat_tool> & tools,
    agent_tool_view * tool_view);
