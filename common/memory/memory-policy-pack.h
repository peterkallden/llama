#pragma once

#include <string>
#include <vector>

// Host-owned, declarative policy context that can be rendered into prompts
// without becoming an executable plan or a separate memory store.
struct common_memory_policy_pack {
    std::string id;
    std::string purpose;
    std::string goal;
    std::string success_criteria;
    std::vector<std::string> constraints;
    std::vector<std::string> decisions;
    std::vector<std::string> preferred_procedures;
};
