#include "agent/adaptation/semi-oracle-search.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
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
    double historical_score = 0.0;
    bool dead_end = false;
};

struct queue_item { size_t f = 0, h = 0, ordinal = 0, node_index = 0; double score = 0.0; };
struct queue_order {
    bool operator()(const queue_item & lhs, const queue_item & rhs) const {
        if (lhs.f != rhs.f) return lhs.f < rhs.f;
        if (lhs.h != rhs.h) return lhs.h < rhs.h;
        return lhs.ordinal < rhs.ordinal;
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
    return common_semi_oracle_search_run(problem, budget, {}, false, result);
}

bool common_semi_oracle_search_run(
        const common_semi_oracle_search_problem & problem,
        const common_semi_oracle_budget & budget,
        const common_semi_oracle_search_policy & policy,
        const bool use_historical_order,
        common_semi_oracle_search_result & result) {
    result = {};
    const bool focal = policy.algorithm == common_semi_oracle_search_algorithm::focal;
    const bool exact = policy.algorithm == common_semi_oracle_search_algorithm::exact;
    if (problem.initial_state.empty() || !problem.is_goal || !problem.expand ||
            budget.max_expansions == 0 || budget.max_depth == 0 ||
            (!exact && !focal) ||
            !std::isfinite(policy.suboptimality_bound) || policy.suboptimality_bound < 1.0 ||
            policy.suboptimality_bound > 2.0 ||
            (!focal && policy.suboptimality_bound != 1.0)) {
        result.error = "search requires callbacks, a non-zero bounded budget, and a supported exact/focal bound";
        return false;
    }
    result.algorithm = policy.algorithm;
    result.suboptimality_bound = policy.suboptimality_bound;

    std::vector<node> nodes;
    std::set<queue_item, queue_order> frontier;
    std::unordered_map<std::string, size_t> best_cost;
    size_t ordinal = 0;
    bool depth_limited = false;
    const auto finish_history = [&](size_t goal_index) {
        if (!problem.record_history) return;
        std::vector<bool> on_path(nodes.size(), false);
        for (size_t at = goal_index; at != std::numeric_limits<size_t>::max(); at = nodes[at].parent)
            on_path[at] = true;
        for (size_t i = 1; i < nodes.size(); ++i) {
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
    frontier.insert({start.h, start.h, start.ordinal, 0, 0.0});
    best_cost.emplace(start.state, 0);

    while (!frontier.empty()) {
        for(auto it=frontier.begin();it!=frontier.end();) {
            const auto best=best_cost.find(nodes[it->node_index].state);
            if(best==best_cost.end()||best->second!=nodes[it->node_index].g)it=frontier.erase(it);
            else ++it;
        }
        if(frontier.empty())break;
        auto selected = frontier.begin();
        if (focal) {
            const size_t f_min = frontier.begin()->f;
            const long double limit = static_cast<long double>(policy.suboptimality_bound) * f_min;
            for (auto it = frontier.begin(); it != frontier.end() &&
                    static_cast<long double>(it->f) <= limit; ++it) {
                const bool better = use_historical_order
                    ? (it->score < selected->score || (it->score == selected->score &&
                        (it->h < selected->h || (it->h == selected->h && it->ordinal < selected->ordinal))))
                    : (it->h < selected->h || (it->h == selected->h && it->ordinal < selected->ordinal));
                if (better) selected = it;
            }
        } else if (use_historical_order) {
            const size_t f_min = frontier.begin()->f;
            for (auto it = frontier.begin(); it != frontier.end() && it->f == f_min; ++it) {
                if (it->score < selected->score || (it->score == selected->score &&
                        (it->h < selected->h || (it->h == selected->h && it->ordinal < selected->ordinal))))
                    selected = it;
            }
        }
        const queue_item item = *selected;
        frontier.erase(selected);
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
            result.optimality_proven = problem.heuristic_admissible &&
                (!focal || policy.suboptimality_bound == 1.0);
            result.suboptimality_bound_proven = problem.heuristic_admissible && focal;
            for (size_t at = item.node_index; at != std::numeric_limits<size_t>::max(); at = nodes[at].parent)
                result.path.push_back({nodes[at].state, nodes[at].action, nodes[at].edge_cost});
            std::reverse(result.path.begin(), result.path.end());
            finish_history(item.node_index);
            return true;
        }
        if (current.depth >= budget.max_depth) { depth_limited = true; continue; }
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
        if (problem.guidance_rank) std::stable_sort(transitions.begin(), transitions.end(),
            [&](const auto & lhs, const auto & rhs) { return problem.guidance_rank(lhs) < problem.guidance_rank(rhs); });

        if (problem.transition_adviser && transitions.size() > 1) {
            std::vector<std::string> actions;
            for (const auto & transition : transitions) actions.push_back(transition.action);
            std::vector<size_t> order; std::string guidance_error; bool valid = false;
            try {
                valid = problem.transition_adviser(current.state, actions, order, guidance_error) && order.size() == transitions.size();
                std::vector<bool> seen(transitions.size(), false);
                for (size_t index : order) { if (index >= seen.size() || seen[index]) { valid = false; break; } seen[index] = true; }
            } catch (...) { valid = false; guidance_error = "transition adviser threw an exception"; }
            if (valid) {
                auto original = transitions;
                for (size_t i = 0; i < order.size(); ++i) transitions[i] = original[order[i]];
            } else {
                ++result.guidance_fallbacks;
                result.guidance_error = guidance_error.empty() ? "invalid transition permutation" : guidance_error;
            }
        }
        for (size_t transition_index = 0; transition_index < transitions.size(); ++transition_index) {
            const auto & transition = transitions[transition_index];
            if (transition.state.empty() || transition.cost == 0) continue;
            size_t next_g = 0;
            if (!add_saturated(current.g, transition.cost, next_g)) continue;
            const auto previous = best_cost.find(transition.state);
            if (previous != best_cost.end() && previous->second <= next_g) continue;
            const size_t h = problem.heuristic ? problem.heuristic(transition.state) : 0;
            size_t f = 0;
            if (!add_saturated(next_g, h, f)) continue;
            double score = 0.0;
            if(problem.historical_score) {
                try { score=problem.historical_score(transition.state,transition.action); }
                catch(...) { ++result.guidance_fallbacks; result.guidance_error="historical score callback threw an exception"; }
            }
            if (!std::isfinite(score)) score = 0.0;
            // When the adviser reordered the legal siblings, retain that stable
            // rank as the historical score if no domain score callback exists.
            if (!problem.historical_score && use_historical_order) score = static_cast<double>(transition_index);
            node next;
            next.state = transition.state; next.g = next_g; next.h = h; next.parent = item.node_index;
            next.action = transition.action; next.edge_cost = transition.cost; next.depth = current.depth + 1;
            next.ordinal = ordinal++; next.historical_score = score;
            const size_t next_index = nodes.size(); nodes.push_back(std::move(next));
            best_cost[transition.state] = next_g;
            frontier.insert({f, h, nodes.back().ordinal, next_index, score});
        }
    }
    result.exhausted = !depth_limited;
    result.stop_reason = depth_limited ? "max_depth" : "frontier_exhausted";
    finish_history(std::numeric_limits<size_t>::max());
    return true;
}
