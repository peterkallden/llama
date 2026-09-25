#pragma once

#include "agent-daemon-adapter.h"

namespace agent_daemon_flydelta_internal {

bool configure_daemon_flydelta_model_residency(
        const daemon_options & options,
        common_agent_model_catalog & catalog,
        bool & native_batch,
        bool & allow_scalar_fallback,
        std::string & error);

bool configure_daemon_flydelta_runtime(
        const daemon_options & options,
        common_agent_daemon_runtime & runtime,
        std::string & error);

void configure_daemon_flydelta_runtime_config(
        const daemon_options & options,
        const common_agent_daemon_runtime & runtime,
        common_agent_runtime_config & config);

} // namespace agent_daemon_flydelta_internal
