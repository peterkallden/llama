#include "agent/memory-learning.h"

#include <algorithm>
#include <chrono>

static bool valid_score(float value) {
    return value >= 0.0f && value <= 1.0f;
}

const char * common_memory_learning_decision_name(common_memory_learning_decision decision) {
    switch (decision) {
        case common_memory_learning_decision::no_candidate: return "no_candidate";
        case common_memory_learning_decision::rejected:     return "rejected";
        case common_memory_learning_decision::accepted:     return "accepted";
        case common_memory_learning_decision::duplicate:    return "duplicate";
        case common_memory_learning_decision::conflict:     return "conflict";
        case common_memory_learning_decision::failed:       return "failed";
    }
    return "failed";
}

common_memory_post_turn_learner::common_memory_post_turn_learner(
        common_memory_store & store,
        common_memory_candidate_extractor & extractor,
        embedder embed,
        common_memory_learning_config config)
    : store(store), extractor(extractor), embed(std::move(embed)), config(config) {}

common_memory_learning_result common_memory_post_turn_learner::learn(
        const common_agent_request & request,
        const common_plan_state & plan,
        const common_agent_result & result) {
    common_memory_learning_result outcome;
    std::string error;
    const auto extracted = extractor.extract(request, plan, result, error);
    if (!error.empty()) {
        outcome.decision = common_memory_learning_decision::failed;
        outcome.reason = "candidate extraction failed safely: " + error;
        return outcome;
    }
    if (!extracted.candidate) {
        outcome.decision = common_memory_learning_decision::no_candidate;
        outcome.reason = extracted.reason.empty() ? "no durable candidate proposed" : extracted.reason;
        return outcome;
    }

    const auto & candidate = *extracted.candidate;
    outcome.candidate = candidate;
    if ((candidate.kind != common_memory_kind::procedure && candidate.kind != common_memory_kind::preference && candidate.kind != common_memory_kind::fact) ||
            candidate.content.empty() || candidate.content.size() > 512 || candidate.rationale.size() > 240 ||
            !valid_score(candidate.importance) || !valid_score(candidate.confidence) || !valid_score(candidate.expected_reuse)) {
        outcome.decision = common_memory_learning_decision::rejected;
        outcome.reason = "candidate failed native shape validation";
        return outcome;
    }
    if (candidate.confidence < config.min_confidence) {
        outcome.decision = common_memory_learning_decision::rejected;
        outcome.reason = "candidate confidence is below threshold";
        return outcome;
    }
    if (candidate.expected_reuse < config.min_expected_reuse) {
        outcome.decision = common_memory_learning_decision::rejected;
        outcome.reason = "candidate expected reuse is below threshold";
        return outcome;
    }
    if (candidate.kind == common_memory_kind::procedure && !candidate.explicit_user_provenance &&
            candidate.evidence_ids.empty() && candidate.source_plan_step_ids.empty()) {
        outcome.decision = common_memory_learning_decision::rejected;
        outcome.reason = "procedure candidate lacks user provenance or evidence";
        return outcome;
    }

    std::vector<float> embedding;
    if (!embed(candidate.content, embedding, error) || embedding.empty()) {
        outcome.decision = common_memory_learning_decision::failed;
        outcome.reason = "candidate embedding failed safely" + (error.empty() ? std::string{} : ": " + error);
        return outcome;
    }

    // Scope and identity are runtime-owned. Learning is deliberately never global.
    common_memory_remember_request remember;
    remember.kind = candidate.kind;
    remember.content = candidate.content;
    remember.rationale = candidate.rationale;
    remember.importance = candidate.importance;
    remember.confidence = candidate.confidence;
    remember.source_role = "assistant";
    remember.source_turn_id = request.turn_id;
    remember.namespace_id = request.namespace_id;
    remember.session_id = request.session_id;
    remember.project_id = request.project_id;
    remember.turn_id = request.turn_id;
    remember.scope = request.project_id.empty() ? common_memory_scope::session : common_memory_scope::project;

    const auto now = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    auto policy = common_memory_evaluate_remember_request(store, remember, embedding, now, error);
    outcome.related_count = policy.related_hits.size();
    if (!error.empty()) {
        outcome.decision = common_memory_learning_decision::failed;
        outcome.reason = "memory policy failed safely: " + error;
        return outcome;
    }
    if (policy.decision == common_memory_remember_decision::duplicate) {
        outcome.decision = common_memory_learning_decision::duplicate;
        outcome.reason = policy.reason;
        return outcome;
    }
    if (policy.decision == common_memory_remember_decision::conflict) {
        outcome.decision = common_memory_learning_decision::conflict;
        outcome.reason = policy.reason;
        return outcome;
    }
    if (policy.decision != common_memory_remember_decision::accept || !policy.record) {
        outcome.decision = common_memory_learning_decision::rejected;
        outcome.reason = policy.reason;
        return outcome;
    }

    auto record = *policy.record;
    record.metadata["learning_stage"] = "post_turn";
    record.metadata["expected_reuse"] = std::to_string(candidate.expected_reuse);
    record.metadata["source_plan_id"] = plan.id;
    record.metadata["source_plan_step_ids"] = candidate.source_plan_step_ids.empty() ? "" : candidate.source_plan_step_ids.front();
    if (!store.put(record, error)) {
        outcome.decision = common_memory_learning_decision::failed;
        outcome.reason = "memory persistence failed safely: " + error;
        return outcome;
    }
    outcome.decision = common_memory_learning_decision::accepted;
    outcome.reason = policy.reason;
    outcome.stored_memory_id = record.id;
    return outcome;
}
