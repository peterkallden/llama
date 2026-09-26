#include "agent/adaptation/flydelta/oracles/flydelta-oracle-contracts.h"

#include <array>
#include <utility>

const char * common_flydelta_oracle_strength_name(common_flydelta_oracle_strength value) {
    switch (value) {
        case common_flydelta_oracle_strength::deterministic: return "deterministic";
        case common_flydelta_oracle_strength::host_supported: return "host_supported";
        case common_flydelta_oracle_strength::model_supported: return "model_supported";
    }
    return "unknown";
}

const char * common_flydelta_oracle_verdict_name(common_flydelta_oracle_verdict value) {
    switch (value) {
        case common_flydelta_oracle_verdict::satisfied: return "satisfied";
        case common_flydelta_oracle_verdict::violated: return "violated";
        case common_flydelta_oracle_verdict::not_applicable: return "not_applicable";
        case common_flydelta_oracle_verdict::unknown: return "unknown";
    }
    return "unknown";
}

const char * common_flydelta_oracle_phase_name(common_flydelta_oracle_phase value) {
    switch (value) {
        case common_flydelta_oracle_phase::concept: return "concept";
        case common_flydelta_oracle_phase::synthesis: return "synthesis";
    }
    return "unknown";
}

const char * common_flydelta_oracle_probe_kind_name(common_flydelta_oracle_probe_kind value) {
    switch (value) {
        case common_flydelta_oracle_probe_kind::target: return "target";
        case common_flydelta_oracle_probe_kind::paraphrase: return "paraphrase";
        case common_flydelta_oracle_probe_kind::transfer: return "transfer";
        case common_flydelta_oracle_probe_kind::control: return "control";
        case common_flydelta_oracle_probe_kind::competing: return "competing";
    }
    return "unknown";
}

namespace {

bool run_one(
        common_flydelta_oracle_strength strength,
        const common_flydelta_oracle_evaluator & evaluator,
        const common_flydelta_oracle_request & request,
        const std::string & observed,
        common_flydelta_oracle_result & result,
        std::string & error) {
    if (!evaluator) return false;
    common_flydelta_oracle_result candidate;
    if (!evaluator(request, observed, candidate, error)) return false;
    candidate.strength = strength;
    if (candidate.oracle_ref.empty()) candidate.oracle_ref = request.oracle_ref;
    if (candidate.oracle_revision.empty()) candidate.oracle_revision = request.oracle_revision;
    if (candidate.policy_revision.empty()) candidate.policy_revision = request.policy_revision;
    result = std::move(candidate);
    return true;
}

} // namespace

bool common_flydelta_oracle_evaluate(
        const common_flydelta_oracle_evaluator_chain & evaluators,
        const common_flydelta_oracle_request & request,
        const std::string & observed,
        common_flydelta_oracle_result & result,
        std::string & error) {
    error.clear();
    result = {};
    common_flydelta_oracle_result unresolved;
    std::string evaluator_error;

    const std::array<std::pair<common_flydelta_oracle_strength,
            const common_flydelta_oracle_evaluator *>, 3> ordered = {{
        {common_flydelta_oracle_strength::deterministic, &evaluators.deterministic},
        {common_flydelta_oracle_strength::host_supported, &evaluators.host_supported},
        {common_flydelta_oracle_strength::model_supported, &evaluators.model_supported},
    }};
    for (const auto & entry : ordered) {
        evaluator_error.clear();
        common_flydelta_oracle_result candidate;
        if (!run_one(entry.first, *entry.second, request, observed, candidate, evaluator_error)) {
            if (!evaluator_error.empty()) error = evaluator_error;
            continue;
        }
        if (candidate.known) {
            result = std::move(candidate);
            return true;
        }
        unresolved = std::move(candidate);
    }
    if (!unresolved.oracle_ref.empty() || !unresolved.reason.empty()) {
        result = std::move(unresolved);
        return true;
    }
    if (error.empty()) error = "no oracle evaluator owns this semantic kind";
    return false;
}
