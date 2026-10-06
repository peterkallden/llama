#include "agent/learning/blueprint-selector.h"

#include <algorithm>
#include <cctype>
#include <set>

namespace {

bool has_all_capabilities(
        const std::vector<std::string> & required,
        const std::vector<std::string> & available) {
    return std::all_of(required.begin(), required.end(), [&](const auto & value) {
        return std::find(available.begin(), available.end(), value) != available.end();
    });
}

std::set<std::string> keyword_set(const std::string & text) {
    static const std::set<std::string> ignored = {
        "about", "after", "agent", "and", "answer", "before", "code", "for", "from", "into", "issue", "that", "the", "this", "with"
    };
    std::set<std::string> words;
    std::string word;
    for (const unsigned char ch : text) {
        if (std::isalnum(ch)) {
            word.push_back((char) std::tolower(ch));
        } else if (!word.empty()) {
            if (word.size() >= 4 && !ignored.count(word)) words.insert(std::move(word));
            word.clear();
        }
    }
    if (word.size() >= 4 && !ignored.count(word)) words.insert(std::move(word));
    return words;
}

const common_blueprint_candidate * keyword_fallback(
        const common_agent_request & request,
        const std::vector<common_blueprint_candidate> & candidates,
        size_t minimum_score) {
    constexpr size_t kMinimumDistinctKeywordMatches = 2;
    std::string request_text = request.prompt;
    if (request.policy_pack) {
        request_text += " " + request.policy_pack->purpose + " " + request.policy_pack->goal +
            " " + request.policy_pack->success_criteria;
        for (const auto & constraint : request.policy_pack->constraints) request_text += " " + constraint;
        for (const auto & procedure : request.policy_pack->preferred_procedures) request_text += " " + procedure;
    }
    const auto request_words = keyword_set(request_text);
    const common_blueprint_candidate * best = nullptr;
    size_t best_score = 0;
    bool tied = false;
    for (const auto & candidate : candidates) {
        std::set<std::string> matched_words;
        const auto weighted_overlap = [&](const std::string & text, size_t weight) {
            size_t value = 0;
            for (const auto & word : keyword_set(text)) {
                if (request_words.count(word)) {
                    ++value;
                    matched_words.insert(word);
                }
            }
            return value * weight;
        };
        size_t score = 0;
        score += weighted_overlap(candidate.purpose, 6);
        score += weighted_overlap(candidate.goal, 5);
        score += weighted_overlap(candidate.success_criteria, 4);
        score += weighted_overlap(candidate.description, 2);
        for (const auto & contribution : candidate.contributions) score += weighted_overlap(contribution, 3);
        for (const auto & constraint : candidate.constraints) score += weighted_overlap(constraint.description, 1);
        if (matched_words.size() < kMinimumDistinctKeywordMatches) continue;
        if (score > best_score) { best = &candidate; best_score = score; tied = false; }
        else if (score != 0 && score == best_score) tied = true;
    }
    return best_score >= minimum_score && !tied ? best : nullptr;
}

bool validate_candidate_catalog(
        const std::vector<common_blueprint_candidate> & candidates,
        const common_blueprint_selection_config & config,
        std::string & error) {
    if (config.maximum_candidates == 0 || config.maximum_candidate_text_bytes == 0) {
        error = "blueprint selection bounds are invalid";
        return false;
    }
    std::set<std::string> logical_ids;
    std::set<std::string> persisted_ids;
    const auto valid_text = [&](const std::string & value) {
        return value.size() <= config.maximum_candidate_text_bytes;
    };
    for (const auto & candidate : candidates) {
        if (candidate.logical_id.empty() || candidate.persisted_id.empty() ||
                !logical_ids.insert(candidate.logical_id).second ||
                !persisted_ids.insert(candidate.persisted_id).second ||
                !valid_text(candidate.logical_id) || !valid_text(candidate.persisted_id) ||
                !valid_text(candidate.source_revision) ||
                !valid_text(candidate.description) || !valid_text(candidate.purpose) ||
                !valid_text(candidate.goal) || !valid_text(candidate.success_criteria) ||
                candidate.required_capabilities.size() > 32 || candidate.constraints.size() > 32 ||
                candidate.assumptions.size() > 32 || candidate.contributions.size() > 32) {
            error = "blueprint candidate catalog contains duplicate or oversized entries";
            return false;
        }
        for (const auto & value : candidate.required_capabilities) if (!valid_text(value)) {
            error = "blueprint candidate capability exceeds bounds";
            return false;
        }
        for (const auto & constraint : candidate.constraints) if (
                constraint.id.empty() || !valid_text(constraint.id) ||
                !valid_text(constraint.description)) {
            error = "blueprint candidate constraint exceeds bounds";
            return false;
        }
        for (const auto & assumption : candidate.assumptions) if (
                assumption.id.empty() || !valid_text(assumption.id) ||
                !valid_text(assumption.statement)) {
            error = "blueprint candidate assumption exceeds bounds";
            return false;
        }
        for (const auto & contribution : candidate.contributions) if (!valid_text(contribution)) {
            error = "blueprint candidate contribution exceeds bounds";
            return false;
        }
    }
    error.clear();
    return true;
}

} // namespace

common_explicit_blueprint_selector::common_explicit_blueprint_selector(std::string logical_id) : logical_id(std::move(logical_id)) {}

common_blueprint_selection common_explicit_blueprint_selector::select(
        const common_agent_request &,
        const std::vector<common_blueprint_candidate> &,
        std::string & error) {
    error.clear();
    return {common_blueprint_selection_decision::instantiate, logical_id, 1.0f, "explicitly selected"};
}

bool common_agent_select_and_instantiate_blueprint(
        common_plan_store & plan_store,
        const common_agent_request & request,
        common_blueprint_selector & selector,
        const std::vector<common_blueprint_candidate> & candidates,
        const common_blueprint_selection_config & config,
        common_blueprint_selection_result & result,
        std::string & error) {
    result = {};
    error.clear();
    result.candidate_count = candidates.size();
    if (config.task_plan_id.empty() || config.session_id.empty()) {
        error = "blueprint selection requires task plan and session ids";
        return false;
    }
    const auto existing = plan_store.get(config.task_plan_id, error);
    if (!error.empty()) return false;
    if (existing) {
        if (existing->kind != common_plan_kind::task ||
                !common_plan_scope_matches(*existing, config.scope, request.namespace_id,
                    request.session_id, request.project_id, request.turn_id)) {
            result.outcome = common_blueprint_selection_outcome::failed_safely;
            result.reason = "existing plan is not a compatible task plan";
            return true;
        }
        result.outcome = common_blueprint_selection_outcome::resumed;
        result.reason = "existing task plan takes precedence";
        return true;
    }

    if (!validate_candidate_catalog(candidates, config, error)) return false;
    if (candidates.empty() || candidates.size() > config.maximum_candidates) {
        result.outcome = common_blueprint_selection_outcome::failed_safely;
        result.reason = candidates.empty() ? "no installed blueprint candidates" : "too many blueprint candidates";
        return true;
    }

    // Native eligibility is resolved before model ranking. This uses the
    // persisted plan as the source of truth, so a selector cannot choose a
    // missing, non-blueprint, or out-of-scope record by logical id.
    std::vector<common_blueprint_candidate> eligible;
    eligible.reserve(candidates.size());
    for (const auto & candidate : candidates) {
        const auto blueprint = plan_store.get(candidate.persisted_id, error);
        if (!error.empty()) return false;
        const bool has_known_false_assumption = blueprint && std::any_of(
            blueprint->assumptions.begin(), blueprint->assumptions.end(), [](const auto & assumption) {
                return !assumption.valid;
            });
        const bool missing_required_capability = blueprint && config.capabilities_resolved &&
            !has_all_capabilities(blueprint->required_capabilities, config.available_capabilities);
        const bool blocked_hard_constraint = blueprint && std::any_of(
            blueprint->constraints.begin(), blueprint->constraints.end(), [&](const common_plan_constraint & constraint) {
                return constraint.hard && std::find(config.blocked_constraint_ids.begin(),
                    config.blocked_constraint_ids.end(), constraint.id) != config.blocked_constraint_ids.end();
            });
        std::string rejection;
        if (!blueprint) rejection = "persisted blueprint is unavailable";
        else if (blueprint->kind != common_plan_kind::blueprint) rejection = "persisted plan is not a blueprint";
        else if (has_known_false_assumption) rejection = "blueprint has a known-false assumption";
        else if (!candidate.source_revision.empty() && candidate.source_revision != blueprint->source_revision) rejection = "candidate source revision does not match persisted blueprint";
        else if (!config.expected_source_revision.empty() && blueprint->source_revision != config.expected_source_revision) rejection = "blueprint source revision is stale for the current host";
        else if (missing_required_capability) rejection = "required host capability is unavailable";
        else if (blocked_hard_constraint) rejection = "hard constraint conflicts with host policy";
        else {
            std::string validation_error;
            if (!common_plan_validate_blueprint(*blueprint, {}, validation_error)) {
                rejection = "blueprint failed structural validation: " + validation_error;
            }
        }
        if (rejection.empty() &&
                !common_plan_scope_matches(*blueprint, config.scope, request.namespace_id,
                    request.session_id, request.project_id, request.turn_id)) rejection = "blueprint is outside the current scope";
        if (!rejection.empty()) {
            result.rejections.push_back({candidate.logical_id, std::move(rejection)});
            continue;
        }
        eligible.push_back(candidate);
    }
    result.eligible_count = eligible.size();
    if (eligible.empty()) {
        result.outcome = common_blueprint_selection_outcome::declined;
        result.reason = "no eligible blueprint candidates in the current scope";
        return true;
    }

    std::string selection_error;
    const auto choice = selector.select(request, eligible, selection_error);
    result.confidence = choice.confidence;
    result.reason = choice.reason.empty() ? selection_error : choice.reason;
    if (!selection_error.empty() || choice.decision == common_blueprint_selection_decision::failed) {
        result.outcome = common_blueprint_selection_outcome::failed_safely;
        return true;
    }
    const common_blueprint_candidate * candidate = nullptr;
    if (choice.decision == common_blueprint_selection_decision::instantiate && choice.logical_id && choice.confidence >= config.minimum_confidence) {
        const auto found = std::find_if(eligible.begin(), eligible.end(), [&](const auto & value) {
            return value.logical_id == *choice.logical_id;
        });
        if (found == eligible.end()) {
            result.outcome = common_blueprint_selection_outcome::failed_safely;
            result.reason = "selector returned an unavailable blueprint";
            return true;
        }
        candidate = &*found;
    } else if (choice.decision == common_blueprint_selection_decision::instantiate) {
        candidate = config.allow_keyword_fallback
            ? keyword_fallback(request, eligible, config.minimum_keyword_fallback_score)
            : nullptr;
        if (candidate) {
            result.confidence = 0.0f;
            result.reason = "native keyword fallback after model reported low confidence";
        } else {
            result.outcome = common_blueprint_selection_outcome::declined;
            return true;
        }
    } else {
        // A model that explicitly says that no blueprint applies must leave
        // the turn to the normal agent workflow.  Falling back here can turn
        // a single generic word overlap into an unrelated reasoning-only
        // task plan and thereby suppress available tool/research steps.
        result.outcome = common_blueprint_selection_outcome::declined;
        return true;
    }
    if (!candidate) {
        result.outcome = common_blueprint_selection_outcome::failed_safely;
        return true;
    }
    const auto blueprint = plan_store.get(candidate->persisted_id, error);
    if (!error.empty()) return false;
    if (!blueprint || blueprint->kind != common_plan_kind::blueprint) {
        result.outcome = common_blueprint_selection_outcome::failed_safely;
        result.reason = "installed candidate is not a blueprint";
        return true;
    }
    common_plan_state instance;
    if (!common_plan_instantiate_blueprint(*blueprint, config.task_plan_id, config.session_id, instance, error, config.scope, config.now)) return false;
    // Blueprint templates deliberately contain no caller identity.  The
    // instantiated task must inherit it before persistence, otherwise a
    // turn- or project-scoped runtime cannot resume the plan it just made.
    instance.namespace_id = request.namespace_id;
    instance.project_id = request.project_id;
    instance.turn_id = request.turn_id;
    if (config.selected_workflow) {
        const auto found = std::find_if(instance.workflow_bindings.begin(), instance.workflow_bindings.end(),
            [&](const auto & binding) { return binding.workflow_ref == config.selected_workflow->workflow_ref &&
                binding.workflow_revision == config.selected_workflow->workflow_revision; });
        if (found == instance.workflow_bindings.end()) {
            result.outcome = common_blueprint_selection_outcome::failed_safely;
            result.reason = "route selected a workflow not bound by the blueprint";
            return true;
        }
        instance.selected_workflow = *config.selected_workflow;
        const auto workflows = plan_store.list(error);
        if (!error.empty()) return false;
        const auto workflow = std::find_if(workflows.begin(), workflows.end(), [&](const auto & plan) {
            return plan.kind == common_plan_kind::workflow && plan.workflow_definition &&
                common_plan_scope_matches(plan, config.scope, request.namespace_id,
                    request.session_id, request.project_id, request.turn_id) &&
                plan.workflow_definition->workflow_ref == config.selected_workflow->workflow_ref &&
                plan.workflow_definition->workflow_revision == config.selected_workflow->workflow_revision;
        });
        if (workflow == workflows.end()) {
            result.outcome = common_blueprint_selection_outcome::failed_safely;
            result.reason = "selected workflow definition is unavailable in the blueprint scope";
            return true;
        }
        instance.workflow_definition = workflow->workflow_definition;
        const auto append_requirement = [&](const std::string & requirement) {
            if (std::find(instance.required_capabilities.begin(), instance.required_capabilities.end(),
                    requirement) == instance.required_capabilities.end()) {
                instance.required_capabilities.push_back(requirement);
            }
        };
        for (const auto & capability : instance.workflow_definition->required_capabilities) {
            append_requirement(capability);
        }
    }
    if (config.route_binding) instance.route_binding = *config.route_binding;
    if (config.materialize_instance) {
        common_plan_state materialized;
        std::string materialization_error;
        const auto materialization = config.materialize_instance(
            request, instance, materialized, materialization_error);
        if (materialization == common_blueprint_materialization_outcome::failed_safely) {
            result.outcome = common_blueprint_selection_outcome::failed_safely;
            result.reason = materialization_error.empty()
                ? "host planner backend declined safely"
                : materialization_error;
            return true;
        }
        if (materialization == common_blueprint_materialization_outcome::applied) {
            if (materialized.kind != common_plan_kind::task ||
                    materialized.id != instance.id ||
                    materialized.session_id != instance.session_id ||
                    materialized.namespace_id != instance.namespace_id ||
                    materialized.project_id != instance.project_id ||
                    materialized.turn_id != instance.turn_id ||
                    materialized.steps.empty()) {
                result.outcome = common_blueprint_selection_outcome::failed_safely;
                result.reason = "host planner backend returned an invalid task instance";
                return true;
            }
            instance = std::move(materialized);
            if (config.route_binding) instance.route_binding = *config.route_binding;
            result.reason = result.reason.empty()
                ? "blueprint instantiated by host planner backend"
                : result.reason + "; host planner backend materialized a verified workflow";
        }
    }
    if (!plan_store.create(instance, error)) return false;
    result.outcome = common_blueprint_selection_outcome::instantiated;
    result.logical_id = candidate->logical_id;
    return true;
}
