#pragma once

#include "agent/agent-recirculation.h"
#include "agent/agent-scope.h"
#include "agent/adaptation/lifecycle-store.h"

#include <string>

// Experimental memory in the existing FlyDelta lifecycle journal. A retained
// profile is only a confidence-proxy observation. It cannot enter adapter,
// sideband, canary, or promotion state.
bool common_flydelta_recirculation_remember(
        common_learning_lifecycle_store & journal,
        const common_agent_scope & scope,
        const std::string & model_profile_fingerprint,
        const std::string & context_key,
        const common_agent_recirculation_request & request,
        const common_agent_recirculation_observation & observation,
        std::string & error);

bool common_flydelta_recirculation_recall(
        const common_learning_lifecycle_store & journal,
        const common_agent_scope & scope,
        const std::string & model_profile_fingerprint,
        const std::string & context_key,
        common_agent_recirculation_request & request,
        bool & found,
        std::string & error);

// A recalled profile that failed the same locked proxy rule is disabled for
// this scope/context. A later, different validated profile may supersede it.
bool common_flydelta_recirculation_suppress(
        common_learning_lifecycle_store & journal,
        const common_agent_scope & scope,
        const std::string & model_profile_fingerprint,
        const std::string & context_key,
        const common_agent_recirculation_request & request,
        std::string & error);
