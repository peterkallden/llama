#include "agent/learning/memory-learning.h"

#include <algorithm>
#include <chrono>
#include <set>
#include <sstream>

static bool valid_score(float value) {
    return value >= 0.0f && value <= 1.0f;
}
static bool has_verified_step(const common_plan_state & plan, const std::string & id) {
    for (const auto & step : plan.steps) {
        if (step.id != id || step.status != common_plan_step_status::completed) continue;
        if (step.result_summary) return true;
        for (const auto & observation : plan.observations) {
            if (observation.id == id || observation.id.rfind("tool:" + id + ":", 0) == 0) return true;
        }
    }
    return false;
}
static std::string join_ids(const std::vector<std::string> & ids) {
    std::ostringstream out;
    for (size_t i = 0; i < ids.size(); ++i) { if (i) out << ','; out << ids[i]; }
    return out.str();
}
static std::string join_learning_signal_types(const std::vector<common_learning_signal> & signals) {
    std::set<std::string> types;
    for (const auto & signal : signals) types.insert(common_learning_signal_type_name(signal.type));
    std::ostringstream out;
    for (auto it = types.begin(); it != types.end(); ++it) {
        if (it != types.begin()) out << ',';
        out << *it;
    }
    return out.str();
}
static std::string join_learning_tools(const std::vector<common_learning_signal> & signals) {
    std::set<std::string> tools;
    for (const auto & signal : signals) if (!signal.tool_name.empty()) tools.insert(signal.tool_name);
    std::ostringstream out;
    for (auto it = tools.begin(); it != tools.end(); ++it) {
        if (it != tools.begin()) out << ',';
        out << *it;
    }
    return out.str();
}
static bool comma_list_contains(const std::string & values, const std::string & value) {
    if (value.empty()) return false;
    size_t start = 0;
    while (start <= values.size()) {
        const size_t end = values.find(',', start);
        if (values.substr(start, end == std::string::npos ? std::string::npos : end - start) == value) return true;
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return false;
}
static size_t metadata_count(const common_memory_record & record, const char * key) {
    const auto value = record.metadata.find(key);
    if (value == record.metadata.end() || value->second.empty()) return 0;
    try { return std::stoull(value->second); } catch (const std::exception &) { return 0; }
}
static common_plan_scope blueprint_scope_for(const common_memory_record & procedure) {
    return procedure.scope == common_memory_scope::project ? common_plan_scope::project : common_plan_scope::session;
}
static bool same_procedure_scope(const common_memory_record & procedure, const common_agent_request & request) {
    const auto expected_scope = request.project_id.empty() ? common_memory_scope::session : common_memory_scope::project;
    return procedure.scope == expected_scope && procedure.namespace_id == request.namespace_id &&
        procedure.session_id == request.session_id && procedure.project_id == request.project_id;
}

const char * common_memory_learning_decision_name(common_memory_learning_decision decision) {
    switch (decision) {
        case common_memory_learning_decision::no_candidate: return "no_candidate";
        case common_memory_learning_decision::awaiting_confirmation: return "awaiting_confirmation";
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
    const bool explicit_capture = request.explicit_memory_candidate.has_value();
    common_memory_candidate candidate;
    if (explicit_capture) {
        candidate = *request.explicit_memory_candidate;
        // The explicit input is accepted as a user-owned acquisition signal,
        // but still passes through the same native memory policy and storage.
        candidate.explicit_user_provenance = true;
    } else {
        const auto extracted = extractor.extract_result(request, plan, result, error);
        outcome.generation = extracted.generation;
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
        candidate = *extracted.candidate;
        // Model output is only a claim. It cannot self-attest user provenance.
        candidate.explicit_user_provenance = false;
    }
    candidate.source_plan_step_ids.erase(std::remove_if(candidate.source_plan_step_ids.begin(), candidate.source_plan_step_ids.end(),
        [&](const std::string & id) { return !has_verified_step(plan, id); }), candidate.source_plan_step_ids.end());
    candidate.evidence_ids.erase(std::remove_if(candidate.evidence_ids.begin(), candidate.evidence_ids.end(),
        [&](const std::string & id) {
            if (id.rfind("memory:", 0) == 0) {
                const auto memory_id = id.substr(7);
                return std::none_of(request.memories.begin(), request.memories.end(), [&](const common_memory_hit & hit) { return hit.memory.id == memory_id; });
            }
            return std::none_of(plan.observations.begin(), plan.observations.end(), [&](const common_plan_observation & observation) { return observation.id == id; });
        }), candidate.evidence_ids.end());
    outcome.candidate = candidate;
    if (explicit_capture && !request.explicit_memory_confirmed) {
        outcome.decision = common_memory_learning_decision::awaiting_confirmation;
        outcome.reason = "explicit memory candidate requires host confirmation before persistence";
        return outcome;
    }
    const bool allowed_kind = candidate.kind == common_memory_kind::procedure ||
        candidate.kind == common_memory_kind::preference || candidate.kind == common_memory_kind::fact ||
        (explicit_capture && (candidate.kind == common_memory_kind::decision || candidate.kind == common_memory_kind::constraint));
    if (!allowed_kind ||
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
    if (!explicit_capture && candidate.expected_reuse < config.min_expected_reuse) {
        outcome.decision = common_memory_learning_decision::rejected;
        outcome.reason = "candidate expected reuse is below threshold";
        return outcome;
    }
    if (!explicit_capture && candidate.kind == common_memory_kind::procedure && candidate.evidence_ids.empty() && candidate.source_plan_step_ids.empty()) {
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
    remember.source_role = explicit_capture ? "user" : "assistant";
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
        const auto existing = std::find_if(policy.related_hits.begin(), policy.related_hits.end(), [&](const auto & hit) {
            return hit.memory.kind == candidate.kind;
        });
        if (existing != policy.related_hits.end()) outcome.stored_memory_id = existing->memory.id;
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
    record.metadata["learning_stage"] = explicit_capture ? "explicit_capture" : "post_turn";
    std::string acquisition_source = "successful_execution_or_reflection";
    if (std::any_of(result.learning_signals.begin(), result.learning_signals.end(), [](const auto & signal) {
            return signal.type == common_learning_signal_type::user_correction;
        })) {
        acquisition_source = "explicit_user_correction";
    } else if (std::any_of(result.learning_signals.begin(), result.learning_signals.end(), [](const auto & signal) {
            return signal.type == common_learning_signal_type::successful_recovery;
        })) {
        acquisition_source = "successful_recovery";
    }
    record.metadata["acquisition_source"] = explicit_capture ? "explicit_user_statement" : acquisition_source;
    record.metadata["expected_reuse"] = std::to_string(candidate.expected_reuse);
    record.metadata["source_plan_id"] = plan.id;
    record.metadata["explicit_user_provenance"] = candidate.explicit_user_provenance ? "true" : "false";
    record.metadata["source_plan_step_ids"] = join_ids(candidate.source_plan_step_ids);
    record.metadata["evidence_ids"] = join_ids(candidate.evidence_ids);
    if (candidate.kind == common_memory_kind::procedure && !explicit_capture) record.metadata["procedure_lifecycle"] = "candidate";
    if (!result.learning_signals.empty()) {
        record.metadata["learning_signal_types"] = join_learning_signal_types(result.learning_signals);
        const auto tools = join_learning_tools(result.learning_signals);
        if (!tools.empty()) record.metadata["learning_tools"] = tools;
    }
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

std::vector<common_memory_hit> common_memory_select_procedure_memories(
        const std::vector<common_memory_hit> & hits,
        const common_plan_state & plan,
        const common_plan_step & step,
        size_t limit) {
    struct ranked_hit { common_memory_hit hit; int context_score = 0; size_t order = 0; };
    std::set<std::string> context_tools;
    if (step.selected_tool) context_tools.insert(*step.selected_tool);
    if (step.tool_call) context_tools.insert(step.tool_call->name);
    bool has_tool_failure = false;
    for (const auto & observation : plan.observations) {
        if (observation.confidence > 0.0f || observation.id.rfind("tool:", 0) != 0) continue;
        has_tool_failure = true;
        if (!observation.source.empty()) context_tools.insert(observation.source);
    }

    std::vector<ranked_hit> ranked;
    for (size_t i = 0; i < hits.size(); ++i) {
        if (hits[i].memory.kind != common_memory_kind::procedure) continue;
        int context_score = 0;
        const auto tools = hits[i].memory.metadata.find("learning_tools");
        if (tools != hits[i].memory.metadata.end()) {
            for (const auto & tool : context_tools) if (comma_list_contains(tools->second, tool)) { context_score += 2; break; }
        }
        const auto signals = hits[i].memory.metadata.find("learning_signal_types");
        if (has_tool_failure && signals != hits[i].memory.metadata.end() && comma_list_contains(signals->second, "tool_failure")) ++context_score;
        ranked.push_back({hits[i], context_score, i});
    }
    std::stable_sort(ranked.begin(), ranked.end(), [](const ranked_hit & a, const ranked_hit & b) {
        return a.context_score > b.context_score;
    });
    std::vector<common_memory_hit> selected;
    for (const auto & hit : ranked) {
        selected.push_back(hit.hit);
        if (selected.size() == limit) break;
    }
    return selected;
}

common_procedure_blueprint_promotion_result common_memory_post_turn_learner::promote_completed_procedure(
        const common_agent_request & request,
        const common_plan_state & plan,
        common_plan_store & plan_store,
        const std::string & procedure_memory_id) {
    common_procedure_blueprint_promotion_result outcome;
    std::string error;
    if (plan.status != common_plan_status::completed || plan.id.empty() || plan.steps.empty()) {
        outcome.reason = "plan did not complete successfully";
        return outcome;
    }
    const auto loaded = store.get(procedure_memory_id, error);
    if (!error.empty() || !loaded || loaded->kind != common_memory_kind::procedure || !same_procedure_scope(*loaded, request)) {
        outcome.reason = error.empty() ? "procedure is unavailable in the completed plan scope" : error;
        return outcome;
    }
    auto procedure = *loaded;
    if (procedure.metadata["procedure_last_success_plan_id"] == plan.id) {
        outcome.verified_uses = metadata_count(procedure, "procedure_verified_uses");
        outcome.reason = "procedure was already counted for this plan";
        return outcome;
    }
    outcome.verified_uses = metadata_count(procedure, "procedure_verified_uses") + 1;
    procedure.metadata["procedure_verified_uses"] = std::to_string(outcome.verified_uses);
    procedure.metadata["procedure_last_success_plan_id"] = plan.id;
    procedure.metadata["procedure_lifecycle"] = "verified";

    const std::string blueprint_id = "learned-blueprint:" + procedure.id;
    if (outcome.verified_uses >= config.procedure_blueprint_min_verified_uses) {
        const auto existing = plan_store.get(blueprint_id, error);
        if (!error.empty()) { outcome.reason = error; return outcome; }
        if (existing && existing->kind != common_plan_kind::blueprint) {
            outcome.reason = "promotion blueprint id is already occupied";
            return outcome;
        }
        if (!existing) {
            common_plan_state blueprint;
            blueprint.id = blueprint_id;
            blueprint.namespace_id = procedure.namespace_id;
            blueprint.session_id = procedure.session_id;
            blueprint.project_id = procedure.project_id;
            blueprint.kind = common_plan_kind::blueprint;
            blueprint.scope = blueprint_scope_for(procedure);
            // Preserve the plan's applicability contract.  A promoted
            // blueprint must carry the same stable purpose and execution
            // boundaries that were verified with the procedure; otherwise
            // later selection cannot distinguish a compatible objective
            // from a merely topical one.
            blueprint.purpose = plan.purpose.empty() ? procedure.summary : plan.purpose;
            blueprint.goal = plan.goal.empty() ? procedure.summary : plan.goal;
            blueprint.success_criteria = plan.success_criteria.empty() ? "Complete the reusable procedure safely." : plan.success_criteria;
            blueprint.required_capabilities = plan.required_capabilities;
            blueprint.constraints = plan.constraints;
            blueprint.assumptions = plan.assumptions;
            blueprint.created_at = plan.updated_at;
            blueprint.updated_at = plan.updated_at;
            bool has_final = false;
            for (const auto & source : plan.steps) {
                common_plan_step step = source;
                step.status = common_plan_step_status::pending;
                step.blocked_by.clear();
                step.selected_tool.reset();
                step.tool_call.reset();
                step.required_evidence.clear();
                step.source_memory_ids.clear();
                step.result_summary.reset();
                step.generated_from_memory = false;
                step.created_at = blueprint.created_at;
                step.updated_at = blueprint.updated_at;
                if (step.mode == common_plan_step_mode::tool) step.mode = common_plan_step_mode::reasoning;
                has_final = has_final || step.mode == common_plan_step_mode::final_response;
                blueprint.steps.push_back(std::move(step));
            }
            if (!has_final) {
                common_plan_step final_step;
                final_step.id = "answer";
                final_step.title = "Answer";
                final_step.objective = "Answer the user using the completed procedure.";
                final_step.mode = common_plan_step_mode::final_response;
                for (const auto & step : blueprint.steps) final_step.depends_on.push_back(step.id);
                blueprint.steps.push_back(std::move(final_step));
            }
            if (!plan_store.create(blueprint, error)) { outcome.reason = error; return outcome; }
        }
        procedure.metadata["promoted_blueprint_id"] = blueprint_id;
        procedure.metadata["procedure_lifecycle"] = "promoted";
        outcome.blueprint_id = blueprint_id;
    }
    if (!store.put(procedure, error)) { outcome.reason = error; return outcome; }
    outcome.reason = outcome.blueprint_id ? "procedure promoted to blueprint" : "procedure success recorded";
    return outcome;
}
