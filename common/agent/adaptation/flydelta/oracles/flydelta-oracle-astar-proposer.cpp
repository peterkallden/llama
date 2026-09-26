#include "agent/adaptation/flydelta/oracles/flydelta-oracle-astar-proposer.h"

#include <cmath>
#include <limits>
#include <queue>
#include <unordered_map>

namespace {

struct node {
    std::string state;
    std::vector<std::string> states;
    std::vector<std::string> actions;
    float cost = 0.0f;
    float estimate = 0.0f;
};

struct node_compare {
    bool operator()(const node & left, const node & right) const {
        return left.estimate > right.estimate;
    }
};

} // namespace

bool common_flydelta_astar_propose(
        const common_flydelta_astar_request & request,
        common_flydelta_astar_result & result,
        std::string & error) {
    error.clear();
    result = {};
    if (request.start_state.empty() || !request.expand ||
            (!request.is_goal && request.goal_state.empty())) {
        error = "A* proposer requires a start, expansion callback and goal";
        return false;
    }
    if (request.max_expansions == 0 || request.max_path_length == 0) {
        error = "A* proposer bounds must be positive";
        return false;
    }

    const auto is_goal = [&](const std::string & state) {
        return request.is_goal ? request.is_goal(state) : state == request.goal_state;
    };
    std::priority_queue<node, std::vector<node>, node_compare> frontier;
    node start;
    start.state = request.start_state;
    start.states.push_back(start.state);
    frontier.push(std::move(start));
    std::unordered_map<std::string, float> best_cost;
    best_cost[request.start_state] = 0.0f;

    while (!frontier.empty() && result.expanded < request.max_expansions) {
        node current = frontier.top();
        frontier.pop();
        const auto known = best_cost.find(current.state);
        if (known != best_cost.end() && current.cost > known->second + 1e-6f) continue;
        if (is_goal(current.state)) {
            result.found = true;
            result.total_cost = current.cost;
            result.states = std::move(current.states);
            result.actions = std::move(current.actions);
            return true;
        }
        if (current.states.size() >= request.max_path_length) continue;
        ++result.expanded;
        std::vector<common_flydelta_astar_successor> successors;
        if (!request.expand(current.state, successors, error)) return false;
        for (const auto & successor : successors) {
            if (successor.state.empty() || successor.cost < 0.0f ||
                    !std::isfinite(successor.cost) || !std::isfinite(successor.heuristic)) {
                error = "A* proposer received an invalid successor";
                return false;
            }
            const float next_cost = current.cost + successor.cost;
            const auto prior = best_cost.find(successor.state);
            if (prior != best_cost.end() && next_cost >= prior->second - 1e-6f) continue;
            best_cost[successor.state] = next_cost;
            node next;
            next.state = successor.state;
            next.states = current.states;
            next.states.push_back(successor.state);
            next.actions = current.actions;
            next.actions.push_back(successor.action);
            next.cost = next_cost;
            next.estimate = next_cost + successor.heuristic;
            frontier.push(std::move(next));
        }
    }
    result.exhausted = result.expanded >= request.max_expansions;
    return true;
}
