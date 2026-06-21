#include "agent/memory-learning.h"
#include "memory/memory-in-memory.h"

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
        [](const std::string &, std::vector<float> & embedding, std::string & embedding_error) {
            embedding = {1.0f, 0.0f};
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
    common_plan_step verify{"verify", "Verify", "Verify persistence"};
    verify.status = common_plan_step_status::completed;
    plan.steps.push_back(verify);
    plan.observations.push_back({"tool:verify:read-back", "read-back", "record persisted", 1.0f, {}, 0});
    common_agent_result result;
    result.response = "done";

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
    assert(stored->metadata.at("source_plan_step_ids") == "verify");

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
