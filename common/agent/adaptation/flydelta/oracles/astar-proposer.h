#pragma once

#include <functional>
#include <string>
#include <vector>

// Generic, bounded A* support for future oracle probe/candidate construction.
// It knows only state fingerprints and transition costs; semantic truth stays
// in the selected oracle and the host. This keeps the proposer cheap and
// reusable across deterministic, host-supported and model-supported oracles.
struct common_flydelta_astar_successor {
    std::string state;
    std::string action;
    float cost = 1.0f;
    float heuristic = 0.0f;
};

struct common_flydelta_astar_request {
    std::string start_state;
    std::string goal_state;
    size_t max_expansions = 64;
    size_t max_path_length = 8;
    std::function<bool(const std::string & state)> is_goal;
    std::function<bool(
            const std::string & state,
            std::vector<common_flydelta_astar_successor> & successors,
            std::string & error)> expand;
};

struct common_flydelta_astar_result {
    bool found = false;
    bool exhausted = false;
    size_t expanded = 0;
    float total_cost = 0.0f;
    std::vector<std::string> states;
    std::vector<std::string> actions;
};

bool common_flydelta_astar_propose(
        const common_flydelta_astar_request & request,
        common_flydelta_astar_result & result,
        std::string & error);
