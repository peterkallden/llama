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
        const auto overlap = [&](const std::string & text) {
            const auto words = keyword_set(text);
            size_t value = 0;
            for (const auto & word : request_words) value += words.count(word);
            return value;
        };
        size_t score = 0;
        score += overlap(candidate.purpose) * 6;
        score += overlap(candidate.goal) * 5;
        score += overlap(candidate.success_criteria) * 4;
        score += overlap(candidate.description) * 2;
        for (const auto & contribution : candidate.contributions) score += overlap(contribution) * 3;
        for (const auto & constraint : candidate.constraints) score += overlap(constraint.description);
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
    } else {
        candidate = config.allow_keyword_fallback
            ? keyword_fallback(request, eligible, config.minimum_keyword_fallback_score)
            : nullptr;
        if (candidate) {
            result.confidence = 0.0f;
            result.reason = "native keyword fallback after model declined or reported low confidence";
        } else {
            result.outcome = common_blueprint_selection_outcome::declined;
            return true;
        }
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
    if (!plan_store.create(instance, error)) return false;
    result.outcome = common_blueprint_selection_outcome::instantiated;
    result.logical_id = candidate->logical_id;
    return true;
}
