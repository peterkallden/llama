#pragma once

#include "agent/adaptation/flydelta/oracles/astar-proposer.h"
#include "agent/tool-workflow-index.h"

#include <functional>
#include <string>
#include <vector>

// Host-owned input for proposing a bounded workflow path. The expansion and
// materialization callbacks are the only domain-specific part; A* sees only
// opaque states and action refs.
struct common_flydelta_workflow_proposal_request {
    int schema_version = 1;
    std::string proposal_id;
    std::string blueprint_ref;
    std::string blueprint_revision;
    std::string workflow_ref;
    std::string workflow_revision;
    std::string start_state;
    std::string goal_state;
    size_t max_expansions = 64;
    size_t max_path_length = 8;
    std::function<bool(const std::string & state)> is_goal;
    std::function<bool(
            const std::string & state,
            std::vector<common_flydelta_astar_successor> & successors,
            std::string & error)> expand;
    std::function<bool(
            const std::vector<std::string> & states,
            const std::vector<std::string> & actions,
            std::vector<common_tool_workflow_step_view> & steps,
            std::string & error)> materialize;
};

// This is a proposal, not a TeachingRelation or an Oracle result. It carries
// only bounded state/action refs and a host-canonical step view.
struct common_flydelta_workflow_proposal {
    int schema_version = 1;
    std::string proposal_id;
    std::string blueprint_ref;
    std::string blueprint_revision;
    std::string workflow_ref;
    std::string workflow_revision;
    common_flydelta_astar_status status = common_flydelta_astar_status::frontier_exhausted;
    bool proposed = false;
    common_flydelta_astar_result search;
    std::vector<common_tool_workflow_step_view> canonical_steps;
};

bool common_flydelta_propose_workflow_path(
        const common_flydelta_workflow_proposal_request & request,
        common_flydelta_workflow_proposal & result,
        std::string & error);

