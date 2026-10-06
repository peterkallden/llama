#pragma once

#include "agent/contracts/agent-request.h"
#include "agent/agent-generation.h"
#include "plan/plan-blueprint.h"
#include "plan/plan-store.h"

#include <optional>
#include <functional>
#include <string>
#include <vector>

// A candidate is deliberately a narrow view of an installed blueprint.  A
// selector may choose an id, but never supplies storage or runtime identity.
struct common_blueprint_candidate {
    std::string logical_id;
    std::string persisted_id;
    std::string description;
    // Bounded projection of the persisted plan's applicability contract. The
    // selector receives existing plan data, never a second blueprint model.
    std::string purpose;
    std::string goal;
    std::string success_criteria;
    std::vector<common_plan_constraint> constraints;
    std::vector<common_plan_assumption> assumptions;
    std::vector<std::string> contributions;
    std::vector<std::string> required_capabilities;
    common_plan_workflow_policy workflow_policy = common_plan_workflow_policy::preferred;
    std::vector<std::string> procedure_refs;
    std::vector<common_plan_workflow_binding> workflow_bindings;
    std::string source_revision;
};

enum class common_blueprint_selection_decision { none, instantiate, failed };

struct common_blueprint_selection {
    common_blueprint_selection_decision decision = common_blueprint_selection_decision::none;
    std::optional<std::string> logical_id;
    float confidence = 0.0f;
    std::string reason;
    std::optional<common_agent_generated_text_result> generation;
};

class common_blueprint_selector {
public:
    virtual ~common_blueprint_selector() = default;
    virtual common_blueprint_selection select(
        const common_agent_request & request,
        const std::vector<common_blueprint_candidate> & candidates,
        std::string & error) = 0;
    virtual common_blueprint_selection select_result(
            const common_agent_request & request,
            const std::vector<common_blueprint_candidate> & candidates,
            std::string & error) {
        return select(request, candidates, error);
    }
};

// Adapts an explicit caller choice to the same bounded orchestration used by
// model-driven selection. Candidate validation remains owned by the caller.
class common_explicit_blueprint_selector final : public common_blueprint_selector {
public:
    explicit common_explicit_blueprint_selector(std::string logical_id);
    common_blueprint_selection select(
        const common_agent_request & request,
        const std::vector<common_blueprint_candidate> & candidates,
        std::string & error) override;
private:
    std::string logical_id;
};

enum class common_blueprint_materialization_outcome {
    not_applicable,
    applied,
    failed_safely,
};

using common_blueprint_instance_materializer = std::function<
    common_blueprint_materialization_outcome(
        const common_agent_request &, const common_plan_state &, common_plan_state &, std::string &)>;

struct common_blueprint_selection_config {
    std::string task_plan_id;
    std::string session_id;
    common_plan_scope scope = common_plan_scope::turn;
    int64_t now = 0;
    float minimum_confidence = 0.75f;
    size_t maximum_candidates = 16;
    size_t maximum_candidate_text_bytes = 4096;
    size_t minimum_keyword_fallback_score = 2;
    bool allow_keyword_fallback = true;
    // Optional host snapshot identity. A non-matching blueprint is stale and
    // cannot be selected or resumed.
    std::string expected_source_revision;
    // Populated only after the host has resolved the active tool profile.
    // Unknown capabilities remain unknown during earlier bootstrap setup.
    std::vector<std::string> available_capabilities;
    bool capabilities_resolved = false;
    // Host-resolved hard constraint identifiers that cannot be honored in
    // this turn. Textual constraints without a host decision remain subject
    // to normal plan validation rather than heuristic native filtering.
    std::vector<std::string> blocked_constraint_ids;
    // Route compiler output. The selector may instantiate only the already
    // host-approved workflow binding; it never resolves a new identity.
    std::optional<common_plan_workflow_binding> selected_workflow;
    // Host-computed provenance for the selected route. The selector copies it
    // into the concrete task instance; model output cannot create or replace
    // this binding.
    std::optional<common_plan_route_binding> route_binding;
    // Optional host-owned planner backend. It may replace the freshly
    // instantiated task with a verified, ordinary task plan before the plan
    // store sees it. A declined callback leaves the blueprint instance
    // unchanged; it must never invent a fallback plan itself.
    common_blueprint_instance_materializer materialize_instance;
};

enum class common_blueprint_selection_outcome { resumed, declined, failed_safely, instantiated };

struct common_blueprint_selection_rejection {
    std::string logical_id;
    std::string reason;
};

struct common_blueprint_selection_result {
    common_blueprint_selection_outcome outcome = common_blueprint_selection_outcome::declined;
    std::optional<std::string> logical_id;
    float confidence = 0.0f;
    std::string reason;
    size_t candidate_count = 0;
    size_t eligible_count = 0;
    std::vector<common_blueprint_selection_rejection> rejections;
};

// This owns the trust boundary around automatic blueprint choice. Failures in
// a selector are reported as a safe fallback; store failures remain errors.
bool common_agent_select_and_instantiate_blueprint(
    common_plan_store & plan_store,
    const common_agent_request & request,
    common_blueprint_selector & selector,
    const std::vector<common_blueprint_candidate> & candidates,
    const common_blueprint_selection_config & config,
    common_blueprint_selection_result & result,
    std::string & error);
