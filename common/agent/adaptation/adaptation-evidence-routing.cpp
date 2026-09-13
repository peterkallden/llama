#include "agent/adaptation/adaptation-evidence-routing.h"

#include <algorithm>

namespace {

bool has_signal(const common_agent_result & result, common_learning_signal_type type) {
    return std::any_of(result.learning_signals.begin(), result.learning_signals.end(),
        [type](const auto & signal) { return signal.type == type; });
}

std::vector<std::string> signal_evidence(
        const common_agent_result & result,
        common_learning_signal_type type) {
    std::vector<std::string> refs;
    for (const auto & signal : result.learning_signals) {
        if (signal.type == type && !signal.evidence_id.empty()) refs.push_back(signal.evidence_id);
    }
    std::sort(refs.begin(), refs.end());
    refs.erase(std::unique(refs.begin(), refs.end()), refs.end());
    return refs;
}

bool has_materialized_data(const common_plan_state & plan) {
    return std::any_of(plan.observations.begin(), plan.observations.end(),
        [](const auto & observation) {
            return !observation.dataset_refs.empty() || !observation.resource_refs.empty();
        });
}

} // namespace

std::vector<common_adaptation_evidence_source_match>
common_adaptation_evidence_sources_for_turn(
        const common_agent_request & request,
        const common_plan_state & plan,
        const common_agent_result & result) {
    (void) request;
    std::vector<common_adaptation_evidence_source_match> matches;

    const bool has_failure = has_signal(result, common_learning_signal_type::tool_failure);
    const bool has_recovery = has_signal(result, common_learning_signal_type::successful_recovery);
    if (has_failure || has_recovery) {
        common_adaptation_evidence_source_match match;
        match.source = common_adaptation_evidence_source::tool_repair;
        match.evidence_refs = signal_evidence(result, common_learning_signal_type::tool_failure);
        const auto recovered = signal_evidence(result, common_learning_signal_type::successful_recovery);
        match.evidence_refs.insert(match.evidence_refs.end(), recovered.begin(), recovered.end());
        std::sort(match.evidence_refs.begin(), match.evidence_refs.end());
        match.evidence_refs.erase(std::unique(match.evidence_refs.begin(), match.evidence_refs.end()), match.evidence_refs.end());
        match.candidate_ready = has_failure && has_recovery && !match.evidence_refs.empty();
        matches.push_back(std::move(match));
    }

    if (result.reflected || has_signal(result, common_learning_signal_type::reflection_hint)) {
        common_adaptation_evidence_source_match match;
        match.source = common_adaptation_evidence_source::reflection_alternative;
        match.evidence_refs = signal_evidence(result, common_learning_signal_type::reflection_hint);
        // Reflection creates a candidate, but the existing result contract
        // does not certify that alternative. A host verifier must complete
        // the relation before candidate_ready can become true.
        match.candidate_ready = false;
        matches.push_back(std::move(match));
    }

    if (result.research_result || result.research_workspace_checkpoint || result.research_verification) {
        common_adaptation_evidence_source_match match;
        match.source = common_adaptation_evidence_source::research_alternative;
        match.candidate_ready = false;
        matches.push_back(std::move(match));
    }

    if (has_signal(result, common_learning_signal_type::user_correction)) {
        common_adaptation_evidence_source_match match;
        match.source = common_adaptation_evidence_source::user_correction;
        match.evidence_refs = signal_evidence(result, common_learning_signal_type::user_correction);
        match.candidate_ready = false;
        matches.push_back(std::move(match));
    }

    if (has_materialized_data(plan)) {
        common_adaptation_evidence_source_match match;
        match.source = common_adaptation_evidence_source::dataset_resource;
        match.candidate_ready = false;
        matches.push_back(std::move(match));
    }
    return matches;
}

