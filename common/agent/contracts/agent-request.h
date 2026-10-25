#pragma once

#include "agent/agent-continuation.h"
#include "agent/agent-generation.h"
#include "agent/agent-scope.h"
#include "agent/investigation-contracts.h"
#include "agent/agent-working-state.h"
#include "agent/dataset-contracts.h"
#include "agent/contracts/agent-events.h"
#include "agent/contracts/agent-learning.h"
#include "agent/thinking/deliberation-policy.h"
#include "agent/thinking/cognitive-mode.h"
#include "agent/thinking/research/research-contract.h"
#include "agent/tooling/contracts/tool-runtime-contract.h"
#include "agent/tooling/schema/tool-output-codec.h"
#include "memory/memory-candidate.h"
#include "memory/memory-policy-pack.h"
#include "memory/memory-types.h"
#include "plan/plan-types.h"
#include "chat.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

struct common_flydelta_activation_result;
struct common_flydelta_hidden_state_capture_request;

struct common_agent_objective {
    std::string purpose;
    std::string desired_outcome;
    std::vector<std::string> success_criteria;
    std::vector<std::string> constraints;
};

struct common_agent_input_resource {
    common_runtime_resource_ref resource;
    std::string role = "reference";
    bool required = false;
};

// Typed host binding for turns whose task depends on reading supplied
// resources. Resource identities and completeness requirements are host-owned;
// neither a model-selected URI nor an unrelated successful tool call can
// satisfy this contract.
enum class common_agent_resource_read_intent {
    orientation,
    question,
    document_review,
};

struct common_agent_resource_read_target {
    std::string uri;
    std::string content_fingerprint;
    std::string representation;
    std::string parent_uri;
    uint64_t size_bytes = 0;
    bool required = true;
};

inline const char * common_agent_resource_read_intent_name(common_agent_resource_read_intent intent) {
    switch (intent) {
        case common_agent_resource_read_intent::orientation: return "orientation";
        case common_agent_resource_read_intent::document_review: return "document_review";
        default: return "question";
    }
}

struct common_agent_resource_read_binding {
    common_agent_resource_read_intent intent = common_agent_resource_read_intent::question;
    std::vector<std::string> resource_uris;
    std::vector<common_agent_resource_read_target> targets;
    // Host-verified certificates imported from an earlier full orientation of
    // the same source fingerprint. This is runtime evidence, not model input.
    std::vector<common_agent_resource_read_target> reusable_orientation_targets;
    std::string task_fingerprint;
    bool require_content_read = true;
    bool require_full_coverage = false;
    std::string document_uri;
    uint64_t document_revision = 0;
    std::vector<std::string> selected_block_ids;
};

// Host-owned arguments that are already resolved before planning. The model
// may omit these fields, but it may not replace them with a conflicting
// value. The JSON remains a model-facing tool argument object so the same
// contract works for native, MCP and OpenAPI tools.
struct common_agent_tool_argument_binding {
    std::string tool_name;
    std::string arguments_json = "{}";
    std::string source_ref;
    bool model_visible = true;
};

using common_agent_workflow_continuation_provider = std::function<bool(
    const common_plan_state & plan,
    const std::string & completed_step_id,
    const common_plan_observation & observation,
    std::vector<common_plan_step> & continuation_steps,
    std::string & error)>;

struct common_agent_request {
    std::vector<common_chat_msg> messages;
    common_memory_scope memory_scope = common_memory_scope::session;
    common_plan_scope plan_scope = common_plan_scope::turn;
    std::string namespace_id = "local";
    std::string session_id = "default";
    std::string project_id;
    std::string turn_id;
    std::optional<std::string> plan_id;
    // Host-owned research binding. When present, the workspace is a draft
    // based on a published revision; model input cannot select its identity.
    std::optional<common_agent_research_draft_ref> research_draft;
    // Host-loaded immutable base for the current chat's persistent draft.
    // This is never accepted from model or client tool arguments.
    std::optional<common_agent_research_workspace> research_workspace_seed;
    // Provenance for a host-selected route when ordinary planning is used
    // instead of a host-materialized blueprint/workflow instance.
    std::optional<common_plan_route_binding> route_binding;
    std::optional<std::string> blueprint_plan_id;
    std::optional<common_plan_workflow_binding> selected_workflow;
    std::optional<common_plan_workflow_definition> workflow_definition;
    bool enable_memory = false;
    bool enable_planning = true;
    bool enable_reflection = true;
    // CLI/tests may require at least one successfully completed tool call
    // before the turn is allowed to fall back to a plain answer.
    bool require_tool_execution = false;
    // JSON is available by explicit selection; compact DSL is the default.
    common_agent_tool_output_format tool_output_format = common_agent_tool_output_format::compact_dsl;
    common_agent_deliberation_policy deliberation_policy;
    std::string prompt;
    std::vector<common_agent_input_resource> input_resources;
    std::optional<common_agent_resource_read_binding> resource_read_binding;
    // Reference-only case context; the host resolves it under current scope.
    std::optional<common_investigation_context_binding> investigation_context;
    // Host-resolved kinds for explicitly selected and authorized case sources.
    // Absent means unknown; inventory alone must not imply a source selection.
    std::optional<std::vector<common_investigation_source_kind>> investigation_source_kinds;
    // Host-owned candidates discovered from the scoped resource store. These
    // are available for selection but are not current-turn inputs.
    std::vector<common_agent_input_resource> available_resources;
    // Structured arguments supplied by the host for a selected tool. These
    // are merged before planner/tool validation and execution.
    std::vector<common_agent_tool_argument_binding> tool_argument_bindings;
    // Optional host-owned A* continuation for a selected workflow. It may
    // append only steps within the already selected route contract.
    common_agent_workflow_continuation_provider workflow_continuation;
    // Host-owned, scoped dataset inventory captured for this turn. These are
    // lookup candidates, not model-created plan aliases.
    std::vector<common_agent_dataset_descriptor> available_datasets;
    std::optional<common_agent_working_state> working_state;
    std::optional<common_agent_objective> objective;
    // Host-prepared, immutable per-turn FlyDelta activation. Runtime code
    // propagates this snapshot but never invents or mutates it.
    std::shared_ptr<const common_flydelta_activation_result> flydelta_activation;
    // Optional host-owned capture request for bounded diagnostic/counterfactual
    // experiments. It is not serialized into the model-facing prompt
    // contract.
    std::shared_ptr<const common_flydelta_hidden_state_capture_request> flydelta_capture;
    std::optional<common_memory_policy_pack> policy_pack;
    std::vector<common_memory_hit> memories;
    std::optional<common_agent_user_correction> user_correction;
    std::optional<common_agent_user_taught_concept> user_taught_concept;
    std::optional<common_memory_candidate> explicit_memory_candidate;
    bool explicit_memory_confirmed = false;
    std::optional<common_agent_tool_call> tool_call;
    common_agent_event_sink event_sink;
    size_t max_iterations = 2;
    // Actual review evaluations per turn, including review after a rewrite.
    // Exhaustion never grants acceptance to a rejected/unreviewed draft.
    size_t max_reflection_rounds = 1;
    // Shared by initial execution and all repair slices in this turn.
    size_t max_tool_batches = 1;
    bool allow_policy_gated_tool_proposals = false;
    std::function<bool()> research_should_stop;
    std::function<common_agent_research_stop_reason()> research_stop_reason;
    // Per-turn activity label supplied and transitioned by the host/runtime.
    // It is prompt/trace context, not durable plan state or tool authority.
    common_agent_cognitive_state cognitive_state;
    // Host-owned bounded reader. No model tool call, new authority or store.
    std::function<bool(const std::string &, nlohmann::ordered_json &, std::string &)> grounding_dataset_reader;
    // Immutable host-resolved method for ordinary planning when no executable
    // workflow is selected. Never accepted from client/model arguments.
    std::optional<common_plan_state> blueprint_definition;
};

inline common_agent_scope common_agent_scope_from_request(const common_agent_request & request) {
    common_agent_scope scope;
    scope.memory_scope = request.memory_scope;
    scope.plan_scope = request.plan_scope;
    scope.namespace_id = request.namespace_id;
    scope.session_id = request.session_id;
    scope.project_id = request.project_id;
    scope.turn_id = request.turn_id;
    return scope;
}

inline void common_agent_scope_apply(const common_agent_scope & scope, common_agent_request & request) {
    request.memory_scope = scope.memory_scope;
    request.plan_scope = scope.plan_scope;
    request.namespace_id = scope.namespace_id;
    request.session_id = scope.session_id;
    request.project_id = scope.project_id;
    request.turn_id = scope.turn_id;
}
