#include "agent/adaptation/flydelta/oracles/workflow-proposal.h"

#include <cstdint>
#include <iomanip>
#include <sstream>

namespace {

bool valid_identity(const common_flydelta_workflow_proposal_request & request) {
    return request.schema_version == 1 && !request.proposal_id.empty() &&
        !request.blueprint_ref.empty() && !request.blueprint_revision.empty() &&
        !request.workflow_ref.empty() && !request.workflow_revision.empty() &&
        !request.start_state.empty() && request.expand && request.materialize;
}

std::string path_fingerprint(
        const std::string & graph_revision,
        const std::string & start_state,
        const std::string & goal_state,
        const std::vector<std::string> & states,
        const std::vector<std::string> & actions) {
    std::string canonical = graph_revision + "\n" + start_state + "\n" + goal_state;
    for (const auto & state : states) canonical += "\nstate:" + state;
    for (const auto & action : actions) canonical += "\naction:" + action;
    uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char byte : canonical) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    std::ostringstream output;
    output << "fnv1a64:" << std::hex << std::setw(16) << std::setfill('0') << hash;
    return output.str();
}

} // namespace

bool common_flydelta_propose_workflow_path(
        const common_flydelta_workflow_proposal_request & request,
        common_flydelta_workflow_proposal & result,
        std::string & error) {
    error.clear();
    result = {};
    result.proposal_id = request.proposal_id;
    result.blueprint_ref = request.blueprint_ref;
    result.blueprint_revision = request.blueprint_revision;
    result.workflow_ref = request.workflow_ref;
    result.workflow_revision = request.workflow_revision;
    result.graph_revision = request.graph_revision;
    result.start_state = request.start_state;
    result.goal_state = request.goal_state;
    if (!valid_identity(request)) {
        result.status = common_flydelta_astar_status::invalid_request;
        error = "workflow proposal requires bounded blueprint/workflow identity and callbacks";
        return false;
    }

    common_flydelta_astar_request astar;
    astar.start_state = request.start_state;
    astar.goal_state = request.goal_state;
    astar.max_expansions = request.max_expansions;
    astar.max_path_length = request.max_path_length;
    astar.is_goal = request.is_goal;
    astar.expand = request.expand;
    if (!common_flydelta_astar_propose(astar, result.search, error)) {
        result.status = result.search.status;
        return false;
    }
    result.status = result.search.status;
    if (!result.search.found) return true;

    result.state_refs = result.search.states;
    result.transition_refs = result.search.actions;
    result.path_fingerprint = path_fingerprint(
        request.graph_revision, request.start_state, request.goal_state,
        result.state_refs, result.transition_refs);

    if (!request.materialize(result.search.states, result.search.actions,
            result.canonical_steps, error)) {
        result.status = common_flydelta_astar_status::expansion_failed;
        if (error.empty()) error = "workflow proposal materialization failed";
        return false;
    }
    if (result.canonical_steps.empty() ||
            result.canonical_steps.size() > request.max_path_length) {
        result.status = common_flydelta_astar_status::expansion_failed;
        error = "workflow proposal materialization violated path bounds";
        return false;
    }
    result.proposed = true;
    return true;
}
