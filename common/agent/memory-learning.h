// Optional post-turn durable-memory learning, owned by the native runtime.
#pragma once

#include "agent/agent-contract.h"
#include "memory/memory-candidate.h"
#include "memory/memory-policy.h"

#include <functional>

class common_memory_candidate_extractor {
public:
    virtual ~common_memory_candidate_extractor() = default;
    virtual common_memory_candidate_result extract(
        const common_agent_request & request,
        const common_plan_state & plan,
        const common_agent_result & result,
        std::string & error) = 0;
};

enum class common_memory_learning_decision {
    no_candidate,
    rejected,
    accepted,
    duplicate,
    conflict,
    failed,
};

struct common_memory_learning_config {
    float min_confidence = 0.75f;
    float min_expected_reuse = 0.65f;
};

struct common_memory_learning_result {
    common_memory_learning_decision decision = common_memory_learning_decision::no_candidate;
    std::optional<common_memory_candidate> candidate;
    std::string reason;
    size_t related_count = 0;
    std::optional<std::string> stored_memory_id;
};

const char * common_memory_learning_decision_name(common_memory_learning_decision decision);

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

private:
    common_memory_store & store;
    common_memory_candidate_extractor & extractor;
    embedder embed;
    common_memory_learning_config config;
};
