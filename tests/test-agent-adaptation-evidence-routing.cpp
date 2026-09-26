#include "agent/adaptation/adaptation-evidence-routing.h"

#include <cassert>

#define CHECK(condition) do { if (!(condition)) return 1; } while (false)

static bool has_source(
        const std::vector<common_adaptation_evidence_source_match> & matches,
        common_adaptation_evidence_source source,
        bool * ready = nullptr) {
    for (const auto & match : matches) {
        if (match.source == source) {
            if (ready) *ready = match.candidate_ready;
            return true;
        }
    }
    return false;
}

int main() {
    std::string error;
    common_agent_request request;
    common_plan_state plan;
    common_agent_result result;
    result.learning_signals.push_back({common_learning_signal_type::tool_failure,
        "plan", "step", "dataset.inspect", "evidence:failed", "failed"});
    result.learning_signals.push_back({common_learning_signal_type::successful_recovery,
        "plan", "step", "dataset.inspect", "evidence:repaired", "repaired"});
    result.learning_signals.push_back({common_learning_signal_type::reflection_hint,
        "plan", "", "", "evidence:reflection", "alternative"});
    result.reflected = true;
    const auto matches = common_adaptation_evidence_sources_for_turn(request, plan, result);

    bool ready = false;
    CHECK(has_source(matches, common_adaptation_evidence_source::tool_repair, &ready));
    if (!ready) return 1;
    CHECK(has_source(matches, common_adaptation_evidence_source::reflection_alternative, &ready));
    if (ready) return 1;
    CHECK(matches.size() == 2);

    common_plan_observation observation;
    observation.id = "resource:1";
    observation.source = "dataset.inspect";
    observation.resource_refs.push_back({});
    plan.observations.push_back(observation);
    const auto with_resource = common_adaptation_evidence_sources_for_turn(request, plan, result);
    CHECK(has_source(with_resource, common_adaptation_evidence_source::dataset_resource));

    common_agent_result other_sources;
    other_sources.learning_signals.push_back({common_learning_signal_type::planning_revision,
        "plan", "step", {}, "evidence:plan", "revised plan"});
    other_sources.learning_signals.push_back({common_learning_signal_type::research_verification,
        "plan", "step", {}, "evidence:research", "research verified"});
    other_sources.learning_signals.push_back({common_learning_signal_type::procedure_verification,
        "plan", "step", {}, "evidence:procedure", "procedure verified"});
    other_sources.learning_signals.push_back({common_learning_signal_type::blueprint_verification,
        "plan", "step", {}, "evidence:blueprint", "blueprint verified"});
    const auto generic = common_adaptation_evidence_sources_for_turn(request, plan, other_sources);
    CHECK(has_source(generic, common_adaptation_evidence_source::planning_revision));
    CHECK(has_source(generic, common_adaptation_evidence_source::research_alternative));
    CHECK(has_source(generic, common_adaptation_evidence_source::procedure_blueprint));

    common_adaptation_evidence generic_evidence;
    common_adaptation_evidence_relation generic_relation;
    generic_relation.id = "adaptation://evidence/planning-1";
    generic_relation.source = common_adaptation_evidence_source::planning_revision;
    generic_relation.behavior_key = "planning/plan-revision";
    generic_relation.task_fingerprint = "sha256:planning-task";
    generic_relation.baseline_ref = "execution:old-plan";
    generic_relation.candidate_ref = "execution:new-plan";
    generic_relation.verifier_ref = "verifier:planning";
    generic_relation.host_verified = true;
    CHECK(common_adaptation_evidence_from_turn(request, plan, other_sources,
        generic_relation, generic_evidence, error));
    CHECK(generic_evidence.source == common_adaptation_evidence_source::planning_revision);

    common_adaptation_evidence_relation relation;
    relation.id = "adaptation://evidence/tool-repair-1";
    relation.source = common_adaptation_evidence_source::tool_repair;
    relation.behavior_key = "tool_use/diagnostics/missing-argument";
    relation.task_fingerprint = "sha256:task";
    relation.baseline_ref = "execution:failed";
    relation.candidate_ref = "execution:repaired";
    relation.verifier_ref = "verifier:tests";
    relation.transaction_ids = {"learning://failed", "learning://repaired"};
    relation.cause = common_learning_cause::model_behavior;
    relation.host_verified = true;
    common_adaptation_evidence evidence;
    CHECK(common_adaptation_evidence_from_turn(request, plan, result, relation, evidence, error));
    CHECK(evidence.source == common_adaptation_evidence_source::tool_repair);
    CHECK(evidence.host_verified);

    auto failure_only = result;
    failure_only.learning_signals.resize(1);
    CHECK(!common_adaptation_evidence_from_turn(request, {}, failure_only, relation, evidence, error));

    result.learning_signals.clear();
    result.reflected = false;
    const auto empty = common_adaptation_evidence_sources_for_turn(request, {}, result);
    CHECK(empty.empty());
    return 0;
}
