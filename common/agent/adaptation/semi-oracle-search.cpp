#include "agent/adaptation/semi-oracle-search.h"

#include <algorithm>
#include <limits>
#include <queue>
#include <unordered_map>

namespace {

struct node {
    std::string state;
    size_t g = 0;
    size_t h = 0;
    size_t parent = std::numeric_limits<size_t>::max();
    std::string action;
    size_t edge_cost = 0;
    size_t depth = 0;
    size_t ordinal = 0;
    bool dead_end = false;
};

struct queue_item { size_t f = 0, h = 0, ordinal = 0, node_index = 0; };
struct later_first {
    bool operator()(const queue_item & lhs, const queue_item & rhs) const {
        if (lhs.f != rhs.f) return lhs.f > rhs.f;
        if (lhs.h != rhs.h) return lhs.h > rhs.h;
        return lhs.ordinal > rhs.ordinal;
    }
};

bool add_saturated(size_t a, size_t b, size_t & value) {
    if (b > std::numeric_limits<size_t>::max() - a) return false;
    value = a + b;
    return true;
}

} // namespace

bool common_semi_oracle_search_astar(
        const common_semi_oracle_search_problem & problem,
        const common_semi_oracle_budget & budget,
        common_semi_oracle_search_result & result) {
    result = {};
    if (problem.initial_state.empty() || !problem.is_goal || !problem.expand ||
            budget.max_expansions == 0 || budget.max_depth == 0) {
        result.error = "A* requires an initial state, callbacks and a non-zero bounded budget";
        return false;
    }

    std::vector<node> nodes;
    std::priority_queue<queue_item, std::vector<queue_item>, later_first> frontier;
    std::unordered_map<std::string, size_t> best_cost;
    size_t ordinal = 0;
    bool depth_limited = false;
    const auto finish_history = [&](size_t goal_index) {
        if (!problem.record_history) return;
        std::vector<bool> on_path(nodes.size(), false);
        for (size_t at = goal_index; at != std::numeric_limits<size_t>::max(); at = nodes[at].parent)
            on_path[at] = true;
        for (size_t i = 1; i < nodes.size(); ++i) {
            // Superseded paths must not supply extra evidence.
            const auto best = best_cost.find(nodes[i].state);
            if (!on_path[i] && (best == best_cost.end() || best->second != nodes[i].g)) continue;
            result.observations.push_back({nodes[i].action,
                on_path[i] ? "goal_reached" : nodes[i].dead_end ? "dead_end" : "unknown",
                nodes[nodes[i].parent].state, nodes[i].state});
        }
    };
    node start;
    start.state = problem.initial_state;
    start.h = problem.heuristic ? problem.heuristic(start.state) : 0;
    start.ordinal = ordinal++;
    nodes.push_back(start);
    frontier.push({start.h, start.h, start.ordinal, 0});
    best_cost.emplace(start.state, 0);

    while (!frontier.empty()) {
        const auto item = frontier.top();
        frontier.pop();
        const node current = nodes[item.node_index];
        const auto best = best_cost.find(current.state);
        if (best == best_cost.end() || best->second != current.g) continue;

        bool goal = false;
        if (!problem.is_goal(current.state, goal, result.error)) return false;
        if (goal) {
            result.stop_reason = "goal_reached";
            result.found = true;
            result.total_cost = current.g;
            result.depth_reached = current.depth;
            result.optimality_proven = problem.heuristic_admissible;
            for (size_t at = item.node_index;
                    at != std::numeric_limits<size_t>::max(); at = nodes[at].parent) {
                const auto & step = nodes[at];
                result.path.push_back({step.state, step.action, step.edge_cost});
            }
            std::reverse(result.path.begin(), result.path.end());
            finish_history(item.node_index);
            return true;
        }
        if (current.depth >= budget.max_depth) {
            depth_limited = true;
            continue;
        }
        if (result.expansions >= budget.max_expansions) {
            result.stop_reason = "max_expansions";
            finish_history(std::numeric_limits<size_t>::max());
            return true;
        }

        std::vector<common_semi_oracle_search_transition> transitions;
        if (!problem.expand(current.state, transitions, result.error)) return false;
        ++result.expansions;
        nodes[item.node_index].dead_end = transitions.empty();
        result.depth_reached = std::max(result.depth_reached, current.depth + 1);
        if (problem.guidance_rank) {
            std::stable_sort(transitions.begin(), transitions.end(), [&](const auto & lhs, const auto & rhs) {
                return problem.guidance_rank(lhs) < problem.guidance_rank(rhs);
            });
        }
        if (problem.transition_adviser && transitions.size() > 1) {
            std::vector<std::string> actions;
            for (const auto & transition : transitions) actions.push_back(transition.action);
            std::vector<size_t> order;
            std::string guidance_error;
            bool valid = false;
            try {
                valid = problem.transition_adviser(current.state, actions, order, guidance_error) &&
                    order.size() == transitions.size();
                std::vector<bool> seen(transitions.size(), false);
                for (size_t index : order) {
                    if (index >= seen.size() || seen[index]) { valid = false; break; }
                    seen[index] = true;
                }
            } catch (...) {
                valid = false;
                guidance_error = "transition adviser threw an exception";
            }
            if (valid) {
                auto original = transitions;
                for (size_t i = 0; i < order.size(); ++i) transitions[i] = original[order[i]];
            } else {
                ++result.guidance_fallbacks;
                result.guidance_error = guidance_error.empty() ? "invalid transition permutation" : guidance_error;
            }
        }
        for (const auto & transition : transitions) {
            if (transition.state.empty() || transition.cost == 0) continue;
            size_t next_g = 0;
            if (!add_saturated(current.g, transition.cost, next_g)) continue;
            const auto previous = best_cost.find(transition.state);
            if (previous != best_cost.end() && previous->second <= next_g) continue;
            const size_t h = problem.heuristic ? problem.heuristic(transition.state) : 0;
            size_t f = 0;
            if (!add_saturated(next_g, h, f)) continue;
            node next;
            next.state = transition.state;
            next.g = next_g;
            next.h = h;
            next.parent = item.node_index;
            next.action = transition.action;
            next.edge_cost = transition.cost;
            next.depth = current.depth + 1;
            next.ordinal = ordinal++;
            const size_t next_index = nodes.size();
            nodes.push_back(std::move(next));
            best_cost[transition.state] = next_g;
            frontier.push({f, h, nodes.back().ordinal, next_index});
        }
    }
    result.exhausted = !depth_limited;
    result.stop_reason = depth_limited ? "max_depth" : "frontier_exhausted";
    finish_history(std::numeric_limits<size_t>::max());
    return true;
}
