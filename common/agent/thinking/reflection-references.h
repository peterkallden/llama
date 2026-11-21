#pragma once

#include "agent/runtime/agent-reflection-engine.h"
#include <map>

// Request-local model view. Durable plan IDs and observations remain untouched.
class common_reflection_evidence_references {
public:
    explicit common_reflection_evidence_references(const common_plan_state &);
    std::string project(const std::string &) const;
    // Only host-owned reference fields are projected; source content remains verbatim.
    std::string project_context(const std::string &) const;
    std::string project_method(const std::string &) const;
    bool restore(common_reflection_result &, std::string & error) const;
private:
    std::map<std::string,std::string> handles;
    std::map<std::string,std::string> phase_handles;
    std::map<std::string,std::string> prior_phase_aliases;
};
