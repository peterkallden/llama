// Optional post-turn durable-memory learning, owned by the native runtime.
#pragma once

#include "agent/contracts/agent-request.h"
#include "agent/contracts/agent-result.h"
#include "memory/memory-candidate.h"
#include "memory/memory-policy.h"
#include "plan/plan-store.h"

#include <functional>

class common_memory_candidate_extractor {
public:
    virtual ~common_memory_candidate_extractor() = default;
    virtual common_memory_candidate_result extract(
        const common_agent_request & request,
        const common_plan_state & plan,
        const common_agent_result & result,
        std::string & error) = 0;
    virtual common_memory_candidate_result extract_result(
            const common_agent_request & request,
            const common_plan_state & plan,
            const common_agent_result & result,
            std::string & error) {
        return extract(request, plan, result, error);
    }
};

enum class common_memory_learning_decision {
    no_candidate,
    awaiting_confirmation,
    rejected,
    accepted,
    duplicate,
    conflict,
    failed,
};

struct common_memory_learning_config {
    float min_confidence = 0.75f;
    float min_expected_reuse = 0.65f;
    size_t procedure_blueprint_min_verified_uses = 3;
};

struct common_memory_learning_result {
    common_memory_learning_decision decision = common_memory_learning_decision::no_candidate;
    std::optional<common_memory_candidate> candidate;
    std::string reason;
    size_t related_count = 0;
    std::optional<std::string> stored_memory_id;
    std::optional<common_agent_generated_text_result> generation;
};

struct common_procedure_blueprint_promotion_result {
    std::string reason;
    size_t verified_uses = 0;
    std::optional<std::string> blueprint_id;
};

const char * common_memory_learning_decision_name(common_memory_learning_decision decision);

// Selects a bounded procedure-only context slice from already retrieved
// memories. Native tool metadata and a recorded tool failure may reorder that
// slice, but never cause a new memory lookup or invent additional context.
std::vector<common_memory_hit> common_memory_select_procedure_memories(
    const std::vector<common_memory_hit> & hits,
    const common_plan_state & plan,
    const common_plan_step & step,
    size_t limit = 3);

class common_memory_post_turn_learner {
public:
    using embedder = std::function<bool(const std::string &, std::vector<float> &, std::string &)>;

    common_memory_post_turn_learner(
        common_memory_store & store,
        common_memory_candidate_extractor & extractor,
        embedder embed,
        common_memory_learning_config config = {});

    common_memory_learning_result learn(
        const common_agent_request & request,
        const common_plan_state & plan,
        const common_agent_result & result);

    // Advances a procedure's verified-use counter after a completed plan and,
    // once the bounded threshold is reached, persists a sanitized blueprint.
    // The caller supplies the procedure ID returned by learn() for this turn.
    common_procedure_blueprint_promotion_result promote_completed_procedure(
        const common_agent_request & request,
        const common_plan_state & plan,
        common_plan_store & plan_store,
        const std::string & procedure_memory_id);

private:
    common_memory_store & store;
    common_memory_candidate_extractor & extractor;
    embedder embed;
    common_memory_learning_config config;
};
