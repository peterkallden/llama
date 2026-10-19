#pragma once

#include "agent/adaptation/semi-oracle.h"

#include <functional>
#include <string>
#include <vector>

// A domain-neutral state-transition contract for bounded deterministic
// planning. Domain adapters own state validation, transition verification,
// goal semantics and heuristic admissibility; the search engine owns only
// ordering, duplicate suppression and resource bounds.
struct common_semi_oracle_search_transition {
    std::string state;
    std::string action;
    size_t cost = 1;
};

struct common_semi_oracle_search_problem {
    std::string initial_state;
    std::function<bool(const std::string &, bool &, std::string &)> is_goal;
    std::function<bool(const std::string &,
            std::vector<common_semi_oracle_search_transition> &,
            std::string &)> expand;
    std::function<size_t(const std::string &)> heuristic;
    // Guidance is an ordering hint only. It is never consulted by the goal
    // predicate, transition validator, or heuristic.
    std::function<size_t(const common_semi_oracle_search_transition &)> guidance_rank;
    bool heuristic_admissible = false;
    common_semi_oracle_transition_adviser transition_adviser;
    bool record_history = false;
};

struct common_semi_oracle_search_step {
    std::string state;
    std::string action;
    size_t cost = 0;
};

struct common_semi_oracle_search_result {
    bool found = false;
    bool exhausted = false;
    bool optimality_proven = false;
    size_t total_cost = 0;
    size_t expansions = 0;
    size_t depth_reached = 0;
    std::vector<common_semi_oracle_search_step> path;
    std::string error;
    struct observation {
        std::string action;
        // goal_reached means a verified suffix exists; dead_end requires an
        // expanded state with no legal outgoing edges. All else is unknown.
        std::string outcome;
    };
    std::vector<observation> observations;
    size_t guidance_fallbacks = 0;
    std::string guidance_error;
};

bool common_semi_oracle_search_astar(
        const common_semi_oracle_search_problem & problem,
        const common_semi_oracle_budget & budget,
        common_semi_oracle_search_result & result);
