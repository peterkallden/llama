#include "research-identity.h"

#include "hash/hash.h"
#include <nlohmann/json.hpp>

namespace {
using json = nlohmann::ordered_json;

std::string digest_id(const std::string & prefix, const json & material) {
    const auto value = material.dump();
    return prefix + hash_sha256_hex(value.data(), value.size()).substr(0, 40);
}
}

std::string common_agent_research_identity(const common_agent_scope & scope) {
    if (scope.namespace_id.empty()) return {};
    if (scope.project_id.empty() && scope.session_id.empty()) return {};
    const json owner = scope.project_id.empty()
        ? json::array({scope.namespace_id, "session", scope.session_id})
        : json::array({scope.namespace_id, "project", scope.project_id});
    return digest_id("research:", owner);
}

std::string common_agent_research_draft_identity(
        const std::string & research_id, const std::string & session_id) {
    if (research_id.empty() || session_id.empty()) return {};
    return digest_id("draft:", json::array({research_id, session_id}));
}
