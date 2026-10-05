#include "agent/adaptation/flydelta/oracles/contracts.h"
#include "agent/adaptation/flydelta/oracles/dataset-operation.h"

#include <array>
#include <algorithm>
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

const char * common_flydelta_oracle_violation_kind_name(
        common_flydelta_oracle_violation_kind value) {
    switch (value) {
        case common_flydelta_oracle_violation_kind::none: return "none";
        case common_flydelta_oracle_violation_kind::contract_violation:
            return "contract_violation";
        case common_flydelta_oracle_violation_kind::wrong_tool: return "wrong_tool";
        case common_flydelta_oracle_violation_kind::missing_required_grouping:
            return "missing_required_grouping";
        case common_flydelta_oracle_violation_kind::wrong_grouping_field:
            return "wrong_grouping_field";
        case common_flydelta_oracle_violation_kind::missing_measure: return "missing_measure";
        case common_flydelta_oracle_violation_kind::wrong_measure_column:
            return "wrong_measure_column";
        case common_flydelta_oracle_violation_kind::wrong_measure_function:
            return "wrong_measure_function";
        case common_flydelta_oracle_violation_kind::unattributable: return "unattributable";
    }
    return "unattributable";
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
        std::string & error,
        const std::string & evaluator_ref = {},
        const std::string & evaluator_revision = {}) {
    if (!evaluator) return false;
    common_flydelta_oracle_result candidate;
    if (!evaluator(request, observed, candidate, error)) return false;
    candidate.strength = strength;
    if (candidate.oracle_ref.empty()) candidate.oracle_ref = request.oracle_ref;
    if (candidate.oracle_revision.empty()) candidate.oracle_revision = request.oracle_revision;
    if (candidate.policy_revision.empty()) candidate.policy_revision = request.policy_revision;
    if (!evaluator_ref.empty()) candidate.evaluator_ref = evaluator_ref;
    if (!evaluator_revision.empty()) candidate.evaluator_revision = evaluator_revision;
    result = std::move(candidate);
    return true;
}

bool registration_matches(
        const common_flydelta_oracle_evaluator_registration & registration,
        const common_flydelta_oracle_request & request) {
    return (registration.semantic_kind.empty() ||
            registration.semantic_kind == request.semantic_kind) &&
        (registration.expected_contract_kind.empty() ||
            registration.expected_contract_kind == request.expected_contract_kind) &&
        (registration.expected_contract_ref.empty() ||
            registration.expected_contract_ref == request.expected_contract_ref) &&
        (registration.expected_contract_revision.empty() ||
            registration.expected_contract_revision == request.expected_contract_revision) &&
        (registration.evaluator_revision.empty() || request.oracle_revision.empty() ||
            registration.evaluator_revision == request.oracle_revision);
}

} // namespace

bool common_flydelta_register_oracle_evaluator(
        common_flydelta_oracle_registry & registry,
        common_flydelta_oracle_evaluator_registration registration,
        std::string & error) {
    error.clear();
    if (!registration.evaluator ||
            (registration.semantic_kind.empty() && registration.expected_contract_kind.empty())) {
        error = "Oracle evaluator registration requires an evaluator and semantic ownership";
        return false;
    }
    const auto duplicate = std::find_if(
            registry.evaluators.begin(), registry.evaluators.end(),
            [&](const auto & existing) {
                return existing.strength == registration.strength &&
                    existing.semantic_kind == registration.semantic_kind &&
                    existing.expected_contract_kind == registration.expected_contract_kind &&
                    existing.expected_contract_ref == registration.expected_contract_ref &&
                    existing.expected_contract_revision == registration.expected_contract_revision &&
                    existing.evaluator_revision == registration.evaluator_revision;
            });
    if (duplicate != registry.evaluators.end()) {
        error = "Oracle evaluator ownership is already registered for this identity";
        return false;
    }
    registry.evaluators.push_back(std::move(registration));
    return true;
}

common_flydelta_oracle_registry common_flydelta_make_default_oracle_registry() {
    common_flydelta_oracle_registry registry;
    std::string error;
    (void) common_flydelta_register_oracle_evaluator(registry, {
        common_flydelta_oracle_strength::deterministic,
        "flydelta://evaluator/dataset-operation",
        "v1",
        "dataset_operation",
        "semantic_decision",
        "flydelta://contract/dataset-operation",
        "v1",
        common_flydelta_dataset_operation_oracle,
    }, error);
    return registry;
}

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

bool common_flydelta_oracle_evaluate(
        const common_flydelta_oracle_registry & registry,
        const common_flydelta_oracle_request & request,
        const std::string & observed,
        common_flydelta_oracle_result & result,
        std::string & error) {
    error.clear();
    result = {};
    common_flydelta_oracle_result unresolved;

    const common_flydelta_oracle_strength strengths[] = {
        common_flydelta_oracle_strength::deterministic,
        common_flydelta_oracle_strength::host_supported,
        common_flydelta_oracle_strength::model_supported,
    };
    for (const auto strength : strengths) {
        std::vector<const common_flydelta_oracle_evaluator_registration *> matches;
        for (const auto & registration : registry.evaluators) {
            if (registration.strength == strength && registration_matches(registration, request)) {
                matches.push_back(&registration);
            }
        }
        if (matches.empty()) continue;
        if (matches.size() > 1) {
            error = "Oracle evaluator ownership is ambiguous for the request";
            return false;
        }

        const auto & registration = *matches.front();
        common_flydelta_oracle_result candidate;
        std::string evaluator_error;
        if (!run_one(registration.strength, registration.evaluator, request, observed,
                candidate, evaluator_error, registration.evaluator_ref,
                registration.evaluator_revision)) {
            error = evaluator_error.empty()
                ? "registered Oracle evaluator declined its owned request" : evaluator_error;
            return false;
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
    error = "no registered Oracle evaluator owns this semantic contract";
    return false;
}
