#include "agent/adaptation/semi-oracle-search.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
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
    std::string posterior_identity;
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
    size_t decisions = 0;
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
    const auto current_entry = [&](const queue_item & item) {
        const auto best=best_cost.find(nodes[item.node_index].state);
        return best!=best_cost.end()&&best->second==nodes[item.node_index].g;
    };

    while (!frontier.empty()) {
        size_t stale_removed = 0;
        // Only a stale minimum can affect f_min. Avoid an O(frontier) sweep
        // before every expansion of ordinary exact A*.
        while(!frontier.empty()&&!current_entry(*frontier.begin())){frontier.erase(frontier.begin());++stale_removed;}
        if(frontier.empty())break;
        auto selected = frontier.begin();
        if (focal) {
            const size_t f_min = frontier.begin()->f;
            const long double limit = static_cast<long double>(policy.suboptimality_bound) * f_min;
            for (auto it = frontier.begin(); it != frontier.end() &&
                    static_cast<long double>(it->f) <= limit; ++it) {
                if(!current_entry(*it))continue;
                const bool better = use_historical_order
                    ? (it->score < selected->score || (it->score == selected->score &&
                        (it->h < selected->h || (it->h == selected->h && it->ordinal < selected->ordinal))))
                    : (it->h < selected->h || (it->h == selected->h && it->ordinal < selected->ordinal));
                if (better) selected = it;
            }
        } else if (use_historical_order) {
            const size_t f_min = frontier.begin()->f;
            for (auto it = frontier.begin(); it != frontier.end() && it->f == f_min; ++it) {
                if(!current_entry(*it))continue;
                const bool after_h = policy.exact_history_order ==
                    common_semi_oracle_history_order::after_heuristic;
                const bool better = after_h
                    ? (it->h < selected->h || (it->h == selected->h &&
                        (it->score < selected->score || (it->score == selected->score && it->ordinal < selected->ordinal))))
                    : (it->score < selected->score || (it->score == selected->score &&
                        (it->h < selected->h || (it->h == selected->h && it->ordinal < selected->ordinal))));
                if (better)
                    selected = it;
            }
        }
        const queue_item item = *selected;
        common_semi_oracle_frontier_event event;
        const bool observe = static_cast<bool>(problem.frontier_observer);
        if (observe) {
            event.decision = decisions;
            event.expansions_used = result.expansions;
            event.expansions_remaining = budget.max_expansions > result.expansions
                ? budget.max_expansions - result.expansions : 0;
            event.f_min = frontier.begin()->f;
            event.focal_limit = focal ? static_cast<double>(policy.suboptimality_bound) * event.f_min
                                      : static_cast<double>(event.f_min);
            std::set<size_t> focal_classes;
            for (const auto & candidate : frontier) {
                if (!current_entry(candidate)) continue;
                ++event.frontier_size;
                if (candidate.f == event.f_min) ++event.equal_f_candidates;
                if (focal && static_cast<long double>(candidate.f) <=
                        static_cast<long double>(policy.suboptimality_bound) * event.f_min) {
                    ++event.focal_candidates;
                    focal_classes.insert(candidate.f);
                }
            }
            event.focal_f_classes = focal ? focal_classes.size() : 1;
            event.selected_g = nodes[item.node_index].g;
            event.selected_h = item.h;
            event.selected_f = item.f;
            event.selected_score = item.score;
            event.selected_state = nodes[item.node_index].state;
            event.selected_action = nodes[item.node_index].action;
            event.posterior_identity = nodes[item.node_index].posterior_identity;
            event.stale_entries_removed = stale_removed;
        }
        ++decisions;
        frontier.erase(selected);
        const node current = nodes[item.node_index];
        const auto best = best_cost.find(current.state);
        if (best == best_cost.end() || best->second != current.g) continue;

        bool goal = false;
        if (!problem.is_goal(current.state, goal, result.error)) return false;
        if (goal) {
            if (observe) { try { problem.frontier_observer(event); } catch (...) {} }
            result.stop_reason = "goal_reached";
            result.found = true;
            result.total_cost = current.g;
            result.depth_reached = current.depth;
            // Depth pruning can discard a cheaper route before this goal is
            // selected. Admissibility alone then proves no global cost bound.
            result.optimality_proven = problem.heuristic_admissible && !depth_limited &&
                (!focal || policy.suboptimality_bound == 1.0);
            result.suboptimality_bound_proven = problem.heuristic_admissible && !depth_limited && focal;
            for (size_t at = item.node_index; at != std::numeric_limits<size_t>::max(); at = nodes[at].parent)
                result.path.push_back({nodes[at].state, nodes[at].action, nodes[at].edge_cost});
            std::reverse(result.path.begin(), result.path.end());
            finish_history(item.node_index);
            return true;
        }
        if (current.depth >= budget.max_depth) {
            depth_limited = true;
            if (observe) { try { problem.frontier_observer(event); } catch (...) {} }
            continue;
        }
        if (result.expansions >= budget.max_expansions) {
            result.stop_reason = "max_expansions";
            if (observe) { try { problem.frontier_observer(event); } catch (...) {} }
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

        // Singleton successors also need their absolute posterior score when
        // compared with nodes generated by other parents in the frontier.
        if (problem.transition_adviser && (transitions.size() > 1 ||
                (!transitions.empty() && problem.transition_score))) {
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
            if (previous != best_cost.end() && previous->second <= next_g) {
                if (observe) ++event.duplicate_successors_suppressed;
                continue;
            }
            if (previous != best_cost.end() && observe) ++event.states_reopened;
            const size_t h = problem.heuristic ? problem.heuristic(transition.state) : 0;
            size_t f = 0;
            if (!add_saturated(next_g, h, f)) continue;
            double score = 0.0;
            if(use_historical_order && (problem.transition_score || problem.historical_score)) {
                try {
                    score = problem.transition_score
                        ? problem.transition_score(current.state, transition.state, transition.action)
                        : problem.historical_score(transition.state,transition.action);
                    if (!std::isfinite(score)) throw std::invalid_argument("non-finite historical score");
                } catch(...) {
                    ++result.guidance_fallbacks;
                    result.guidance_error="historical score callback failed";
                    score=0.0;
                }
            }
            // A sibling ordinal is not a comparable score across parents.
            // Permutation-only advisers still affect the stable insertion order.
            node next;
            next.state = transition.state; next.g = next_g; next.h = h; next.parent = item.node_index;
            next.action = transition.action; next.edge_cost = transition.cost; next.depth = current.depth + 1;
            next.ordinal = ordinal++; next.historical_score = score;
            if (problem.historical_identity) {
                try { next.posterior_identity = problem.historical_identity(current.state, transition.state, transition.action); }
                catch (...) { next.posterior_identity.clear(); }
            }
            const size_t next_index = nodes.size(); nodes.push_back(std::move(next));
            best_cost[transition.state] = next_g;
            frontier.insert({f, h, nodes.back().ordinal, next_index, score});
        }
        if (observe) { try { problem.frontier_observer(event); } catch (...) {} }
    }
    result.exhausted = !depth_limited;
    result.stop_reason = depth_limited ? "max_depth" : "frontier_exhausted";
    finish_history(std::numeric_limits<size_t>::max());
    return true;
}
