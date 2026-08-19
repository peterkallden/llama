#include "agent/learning/memory-learning.h"
#include "memory/memory-in-memory.h"
#include "plan/plan-in-memory.h"

#include <cassert>

class extractor final : public common_memory_candidate_extractor {
public:
    common_memory_candidate_result next;
    std::string failure;

    common_memory_candidate_result extract(const common_agent_request &, const common_plan_state &, const common_agent_result &, std::string & error) override {
        error = failure;
        return next;
    }
};

static common_memory_candidate procedure(const std::string & content) {
    common_memory_candidate candidate;
    candidate.kind = common_memory_kind::procedure;
    candidate.content = content;
    candidate.rationale = "Verified reusable method.";
    candidate.importance = 0.8f;
    candidate.confidence = 0.9f;
    candidate.expected_reuse = 0.8f;
    candidate.source_plan_step_ids = {"verify"};
    return candidate;
}

int main() {
    std::string error;
    common_memory_in_memory_store store;
    assert(store.open("", error));
    extractor candidate_extractor;
    common_memory_post_turn_learner learner(store, candidate_extractor,
        [](const std::string & content, std::vector<float> & embedding, std::string & embedding_error) {
            embedding = content.find("tool authority") != std::string::npos
                ? std::vector<float>{0.0f, 1.0f}
                : std::vector<float>{1.0f, 0.0f};
            embedding_error.clear();
            return true;
        });

    common_agent_request request;
    request.namespace_id = "local";
    request.session_id = "session-a";
    request.project_id = "project-a";
    request.memory_scope = common_memory_scope::global; // Must not influence automatic learning scope.
    common_plan_state plan;
    plan.id = "plan-a";
    plan.purpose = "Safely verify durable agent memory.";
    plan.goal = "Verify persisted memory";
    plan.success_criteria = "The record can be read back.";
    plan.constraints.push_back({"bounded", "Use only bounded repository reads.", true});
    plan.assumptions.push_back({"repository", "The repository is available locally.", 0.9f, true, {"read-back"}});
    plan.status = common_plan_status::completed;
    common_plan_step verify{"verify", "Verify", "Verify persistence"};
    verify.mode = common_plan_step_mode::tool;
    verify.selected_tool = "repository.read";
    verify.tool_call = common_plan_tool_call{"repository.read", R"({"path":"memory.db"})"};
    verify.status = common_plan_step_status::completed;
    plan.steps.push_back(verify);
    plan.observations.push_back({"tool:verify:read-back", "read-back", "record persisted", 1.0f, {}, {}, 0});
    common_agent_result result;
    result.response = "done";
    result.learning_signals.push_back({common_learning_signal_type::tool_failure, "plan-a", "verify", "repository.read", "tool:verify:repository.read", "repository read failed"});

    candidate_extractor.next = {};
    candidate_extractor.next.reason = "ordinary one-off task";
    assert(learner.learn(request, plan, result).decision == common_memory_learning_decision::no_candidate);

    candidate_extractor.next = {};
    candidate_extractor.next.candidate = procedure("Verify persistence by reopening the database and reading the record back.");
    const auto accepted = learner.learn(request, plan, result);
    assert(accepted.decision == common_memory_learning_decision::accepted && accepted.stored_memory_id);
    const auto stored = store.get(*accepted.stored_memory_id, error);
    assert(stored && stored->scope == common_memory_scope::project && stored->project_id == "project-a");
    assert(stored->metadata.at("learning_stage") == "post_turn");
    assert(stored->metadata.at("procedure_lifecycle") == "candidate");
    assert(stored->metadata.at("source_plan_step_ids") == "verify");
    assert(stored->metadata.at("learning_signal_types") == "tool_failure");
    assert(stored->metadata.at("learning_tools") == "repository.read");

    common_memory_candidate explicit_decision;
    explicit_decision.kind = common_memory_kind::decision;
    explicit_decision.content = "The runtime keeps tool authority host-owned.";
    explicit_decision.rationale = "Explicit project decision.";
    explicit_decision.confidence = 0.9f;
    request.explicit_memory_candidate = explicit_decision;
    const auto pending_explicit = learner.learn(request, plan, result);
    assert(pending_explicit.decision == common_memory_learning_decision::awaiting_confirmation);
    assert(pending_explicit.candidate && !pending_explicit.stored_memory_id);
    request.explicit_memory_confirmed = true;
    const auto explicit_result = learner.learn(request, plan, result);
    assert(explicit_result.decision == common_memory_learning_decision::accepted && explicit_result.stored_memory_id);
    const auto explicit_stored = store.get(*explicit_result.stored_memory_id, error);
    assert(explicit_stored && explicit_stored->kind == common_memory_kind::decision);
    assert(explicit_stored->metadata.at("learning_stage") == "explicit_capture");
    assert(explicit_stored->metadata.at("acquisition_source") == "explicit_user_statement");
    assert(explicit_stored->metadata.at("explicit_user_provenance") == "true");
    request.explicit_memory_candidate.reset();
    request.explicit_memory_confirmed = false;

    candidate_extractor.next.candidate = explicit_decision;
    assert(learner.learn(request, plan, result).decision == common_memory_learning_decision::rejected);
    candidate_extractor.next.candidate.reset();
    candidate_extractor.next.candidate = procedure("Verify persistence by reopening the database and reading the record back.");

    common_memory_hit generic_procedure;
    generic_procedure.memory.id = "generic";
    generic_procedure.memory.kind = common_memory_kind::procedure;
    generic_procedure.final_score = 0.95f;
    common_memory_hit tool_recovery_procedure;
    tool_recovery_procedure.memory.id = "tool-recovery";
    tool_recovery_procedure.memory.kind = common_memory_kind::procedure;
    tool_recovery_procedure.memory.metadata["learning_tools"] = "repository.read";
    tool_recovery_procedure.memory.metadata["learning_signal_types"] = "tool_failure,successful_recovery";
    tool_recovery_procedure.final_score = 0.75f;
    common_plan_step recovery_step{"recover", "Recover", "Recover from the failed read"};
    recovery_step.mode = common_plan_step_mode::reasoning;
    plan.observations.push_back({"tool:verify:repository.read", "repository.read", "read failed", 0.0f, {}, {}, 0});
    const auto procedures = common_memory_select_procedure_memories({generic_procedure, tool_recovery_procedure}, plan, recovery_step);
    assert(procedures.size() == 2 && procedures.front().memory.id == "tool-recovery");

    common_plan_in_memory_store plans;
    assert(plans.open("", error));
    for (int i = 1; i <= 3; ++i) {
        plan.id = "verified-plan-" + std::to_string(i);
        const auto promotion = learner.promote_completed_procedure(request, plan, plans, *accepted.stored_memory_id);
        assert(promotion.verified_uses == (size_t) i);
        if (i < 3) assert(!promotion.blueprint_id);
        else {
            assert(promotion.blueprint_id);
            const auto blueprint = plans.get(*promotion.blueprint_id, error);
            assert(blueprint && blueprint->kind == common_plan_kind::blueprint);
            assert(blueprint->scope == common_plan_scope::project && blueprint->project_id == "project-a");
            assert(blueprint->purpose == plan.purpose);
            assert(blueprint->constraints.size() == 1 && blueprint->constraints.front().id == "bounded");
            assert(blueprint->assumptions.size() == 1 && blueprint->assumptions.front().id == "repository");
            assert(!blueprint->steps.front().tool_call && blueprint->steps.front().mode == common_plan_step_mode::reasoning);
            assert(blueprint->steps.back().mode == common_plan_step_mode::final_response);
        }
    }
    const auto promoted_procedure = store.get(*accepted.stored_memory_id, error);
    assert(promoted_procedure && promoted_procedure->metadata.at("procedure_verified_uses") == "3");
    assert(promoted_procedure->metadata.at("procedure_lifecycle") == "promoted");
    assert(promoted_procedure->metadata.count("promoted_blueprint_id") == 1);

    const auto duplicate = learner.learn(request, plan, result);
    assert(duplicate.decision == common_memory_learning_decision::duplicate);

    candidate_extractor.next.candidate->confidence = 0.5f;
    assert(learner.learn(request, plan, result).decision == common_memory_learning_decision::rejected);
    candidate_extractor.next.candidate->confidence = 0.9f;
    candidate_extractor.next.candidate->expected_reuse = 0.4f;
    assert(learner.learn(request, plan, result).decision == common_memory_learning_decision::rejected);
    candidate_extractor.next.candidate->expected_reuse = 0.8f;
    candidate_extractor.next.candidate->source_plan_step_ids.clear();
    candidate_extractor.next.candidate->evidence_ids.clear();
    candidate_extractor.next.candidate->explicit_user_provenance = false;
    assert(learner.learn(request, plan, result).decision == common_memory_learning_decision::rejected);

    candidate_extractor.next.candidate->source_plan_step_ids = {"invented"};
    candidate_extractor.next.candidate->explicit_user_provenance = true;
    assert(learner.learn(request, plan, result).decision == common_memory_learning_decision::rejected);

    candidate_extractor.failure = "malformed model JSON";
    assert(learner.learn(request, plan, result).decision == common_memory_learning_decision::failed);
    return 0;
}
