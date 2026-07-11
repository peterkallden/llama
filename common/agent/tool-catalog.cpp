#include "agent/tool-catalog.h"

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

namespace {

common_tool_definition tool(
        const char * name, const char * description, const char * input, const char * result,
        const char * executor, common_tool_risk_class risk, bool confirmation = false,
        uint32_t timeout_ms = 1000, size_t max_result_bytes = 16384, const char * policy = "{}") {
    common_tool_definition definition;
    definition.name = name;
    definition.description = description;
    definition.input_schema_json = input;
    definition.result_schema_json = result;
    definition.executor_id = executor;
    definition.risk_class = risk;
    definition.requires_confirmation = confirmation;
    definition.timeout_ms = timeout_ms;
    definition.max_result_bytes = max_result_bytes;
    definition.policy_json = policy;
    return definition;
}

std::vector<common_tool_definition> builtin_definitions() {
    const char * empty = R"({"type":"object","additionalProperties":false})";
    const char * object = R"({"type":"object"})";
    return {
        tool("calculator", "Evaluate a bounded arithmetic expression.", R"({"type":"object","additionalProperties":false,"required":["expression"],"properties":{"expression":{"type":"string","minLength":1,"maxLength":256}}})", object, "builtin.calculator", common_tool_risk_class::local_read),
        tool("time_now", "Return the current UTC time.", R"({"type":"object","additionalProperties":false,"properties":{"timezone":{"type":"string","enum":["UTC"]}}})", object, "builtin.time_now", common_tool_risk_class::local_read),
        tool("memory_search", "Search memories available to the current runtime scope.", R"({"type":"object","additionalProperties":false,"required":["query"],"properties":{"query":{"type":"string","minLength":1,"maxLength":1024},"limit":{"type":"integer","minimum":1,"maximum":8}}})", object, "builtin.memory_search", common_tool_risk_class::local_read),
        tool("memory_get", "Get a previously retrieved memory by its opaque id.", R"({"type":"object","additionalProperties":false,"required":["id"],"properties":{"id":{"type":"string","minLength":1,"maxLength":256}}})", object, "builtin.memory_get", common_tool_risk_class::local_read),
        tool("memory_inspect", "Inspect bounded memory statistics for the current scope.", empty, object, "builtin.memory_inspect", common_tool_risk_class::local_read),
        tool("memory_conflict_check", "Find potentially conflicting memories in the current scope.", R"({"type":"object","additionalProperties":false,"required":["content"],"properties":{"content":{"type":"string","minLength":1,"maxLength":2048}}})", object, "builtin.memory_conflict_check", common_tool_risk_class::local_read),
        tool("memory_remember", "Propose a policy-gated persistent memory.", R"({"type":"object","additionalProperties":false,"required":["kind","content"],"properties":{"kind":{"type":"string","enum":["fact","preference","procedure","constraint","decision","goal","observation","reflection","episode"]},"content":{"type":"string","minLength":1,"maxLength":512},"importance":{"type":"number","minimum":0,"maximum":1},"confidence":{"type":"number","minimum":0,"maximum":1},"rationale":{"type":"string","minLength":1,"maxLength":240}}})", object, "builtin.memory_remember", common_tool_risk_class::memory_proposal, true),
        tool("memory_propose_update", "Propose a version-checked memory correction or merge.", R"({"type":"object","additionalProperties":false,"required":["id","expected_version","content","rationale"],"properties":{"id":{"type":"string"},"expected_version":{"type":"integer","minimum":0},"content":{"type":"string","minLength":1,"maxLength":2048},"rationale":{"type":"string","minLength":1,"maxLength":512}}})", object, "builtin.memory_propose_update", common_tool_risk_class::memory_proposal, true),
        tool("memory_propose_forget", "Propose archival of a memory; never hard-delete it.", R"({"type":"object","additionalProperties":false,"required":["id","rationale"],"properties":{"id":{"type":"string"},"rationale":{"type":"string","minLength":1,"maxLength":512}}})", object, "builtin.memory_propose_forget", common_tool_risk_class::memory_proposal, true),
        tool("memory_link", "Propose a typed relationship between two memories or a memory and a plan.", R"({"type":"object","additionalProperties":false,"required":["from","relation","to","rationale"],"properties":{"from":{"type":"string"},"relation":{"type":"string","maxLength":64},"to":{"type":"string"},"rationale":{"type":"string","minLength":1,"maxLength":512}}})", object, "builtin.memory_link", common_tool_risk_class::memory_proposal, true),
        tool("memory_compact_propose", "Propose consolidation of supplied source memories into one summary.", R"({"type":"object","additionalProperties":false,"required":["source_ids","content","rationale"],"properties":{"source_ids":{"type":"array","minItems":2,"maxItems":8,"items":{"type":"string"}},"content":{"type":"string","minLength":1,"maxLength":2048},"rationale":{"type":"string","minLength":1,"maxLength":512}}})", object, "builtin.memory_compact_propose", common_tool_risk_class::memory_proposal, true),
        tool("plan_get", "Return the plan bound to the current runtime turn.", R"({"type":"object","additionalProperties":false,"properties":{"include_completed":{"type":"boolean"},"include_history":{"type":"boolean"}}})", object, "builtin.plan_get", common_tool_risk_class::local_read),
        tool("plan_propose", "Propose version-checked plan operations for native policy evaluation.", R"({"type":"object","additionalProperties":false,"required":["expected_version","operations"],"properties":{"expected_version":{"type":"integer","minimum":0},"operations":{"type":"array","minItems":1,"maxItems":8,"items":{"type":"object"}}}})", object, "builtin.plan_propose", common_tool_risk_class::plan_proposal, true),
        tool("repository_list", "List a bounded directory tree inside the runtime repository root.", R"({"type":"object","additionalProperties":false,"properties":{"path":{"type":"string","maxLength":512},"depth":{"type":"integer","minimum":0,"maximum":3}}})", object, "builtin.repository_list", common_tool_risk_class::local_read),
        tool("repository_search", "Search bounded text files inside the runtime repository root.", R"({"type":"object","additionalProperties":false,"required":["query"],"properties":{"query":{"type":"string","minLength":1,"maxLength":256},"path":{"type":"string","maxLength":512},"max_results":{"type":"integer","minimum":1,"maximum":32}}})", object, "builtin.repository_search", common_tool_risk_class::local_read),
        tool("repository_read", "Read a bounded line range from a text file inside the runtime repository root.", R"({"type":"object","additionalProperties":false,"required":["path"],"properties":{"path":{"type":"string","minLength":1,"maxLength":512},"start_line":{"type":"integer","minimum":1,"maximum":1000000},"end_line":{"type":"integer","minimum":1,"maximum":1000000}}})", object, "builtin.repository_read", common_tool_risk_class::local_read),
        tool("repository_diff", "Return a bounded read-only Git working-tree diff summary.", R"({"type":"object","additionalProperties":false})", object, "builtin.repository_diff", common_tool_risk_class::local_read),
        tool("repository_log", "Return a bounded read-only Git commit log.", R"({"type":"object","additionalProperties":false,"properties":{"limit":{"type":"integer","minimum":1,"maximum":20}}})", object, "builtin.repository_log", common_tool_risk_class::local_read),
        tool("resource_read", "Read a bounded host-owned resource payload by its opaque URI.", R"({"type":"object","additionalProperties":false,"required":["uri"],"properties":{"uri":{"type":"string","minLength":1,"maxLength":512},"max_bytes":{"type":"integer","minimum":1,"maximum":32768}}})", object, "builtin.resource_read", common_tool_risk_class::local_read),
        tool("web_search", "Search the public web through a bounded HTTPS provider and return result candidates.", R"({"type":"object","additionalProperties":false,"required":["query"],"properties":{"query":{"type":"string","minLength":1,"maxLength":512},"limit":{"type":"integer","minimum":1,"maximum":8},"site":{"type":"string","minLength":1,"maxLength":256}}})", object, "builtin.web_search", common_tool_risk_class::network_read, false, 10000, 65536, R"({"https_only":true,"provider":"duckduckgo-lite","block_private_networks":true})"),
        tool("web_fetch", "Fetch a public HTTPS URL through the native safe HTTP client.", R"({"type":"object","additionalProperties":false,"required":["url"],"properties":{"url":{"type":"string","minLength":9,"maxLength":2048},"max_bytes":{"type":"integer","minimum":1,"maximum":500000},"extract":{"type":"string","enum":["text"]}}})", object, "builtin.web_fetch", common_tool_risk_class::network_read, false, 10000, 65536, R"({"https_only":true,"max_redirects":3,"block_private_networks":true})"),
    };
}

common_tool_profile profile(const char * id, const char * description, std::initializer_list<const char *> tools) {
    common_tool_profile value;
    value.id = id;
    value.description = description;
    for (const auto * name : tools) value.members.push_back({name, 1, true, "{}"});
    return value;
}

std::vector<common_tool_profile> builtin_profiles() {
    return {
        profile("minimal", "Local deterministic utility tools.", {"calculator", "time_now"}),
        profile("memory-read", "Read-only scoped memory and plan inspection.", {"calculator", "time_now", "memory_search", "memory_get", "memory_inspect", "memory_conflict_check", "plan_get", "resource_read"}),
        profile("memory", "Memory inspection plus policy-gated memory and plan proposals.", {"calculator", "time_now", "memory_search", "memory_get", "memory_inspect", "memory_conflict_check", "memory_remember", "memory_propose_update", "memory_propose_forget", "memory_link", "memory_compact_propose", "plan_get", "plan_propose", "resource_read"}),
        profile("research", "Memory profile with repository inspection and safe network-read declarations.", {"calculator", "time_now", "memory_search", "memory_get", "memory_inspect", "memory_conflict_check", "memory_remember", "plan_get", "plan_propose", "repository_list", "repository_search", "repository_read", "repository_diff", "repository_log", "resource_read", "web_search", "web_fetch"}),
    };
}

bool validate(const common_tool_definition & definition, std::string & error) {
    if (definition.name.empty() || definition.executor_id.empty()) { error = "tool definition requires a name and native executor id"; return false; }
    const auto input = json::parse(definition.input_schema_json, nullptr, false);
    const auto result = json::parse(definition.result_schema_json, nullptr, false);
    const auto policy = json::parse(definition.policy_json, nullptr, false);
    if (!input.is_object() || input.value("type", std::string()) != "object" || !result.is_object() || !policy.is_object()) { error = "tool definition has invalid JSON schema or policy"; return false; }
    return true;
}

} // namespace

const char * common_tool_risk_class_name(common_tool_risk_class value) {
    switch (value) {
        case common_tool_risk_class::local_read: return "local_read";
        case common_tool_risk_class::memory_proposal: return "memory_proposal";
        case common_tool_risk_class::plan_proposal: return "plan_proposal";
        case common_tool_risk_class::network_read: return "network_read";
    }
    return "unknown";
}

bool common_tool_catalog::bootstrap(const std::string & profile_id, common_tool_bootstrap_result & result, std::string & error) {
    result = {};
    const auto all_definitions = builtin_definitions();
    const auto all_profiles = builtin_profiles();
    const auto selected = profile_id.empty() ? "minimal" : profile_id;
    bool known_profile = false;
    for (const auto & profile : all_profiles) if (profile.id == selected) known_profile = true;
    if (!known_profile) { error = "unknown built-in tool profile: " + selected; return false; }
    for (const auto & definition : all_definitions) {
        if (!validate(definition, error)) return false;
        const auto key = definition.name + "@" + std::to_string(definition.version);
        if (definitions.count(key)) result.definitions_unchanged.push_back(key);
        else { definitions.emplace(key, definition); result.definitions_created.push_back(key); }
    }
    for (const auto & profile : all_profiles) {
        if (profiles.count(profile.id)) result.profiles_unchanged.push_back(profile.id);
        else { profiles.emplace(profile.id, profile); result.profiles_created.push_back(profile.id); }
    }
    error.clear();
    return true;
}

const common_tool_definition * common_tool_catalog::find_definition(const std::string & name, uint32_t version) const {
    const auto it = definitions.find(name + "@" + std::to_string(version));
    return it == definitions.end() ? nullptr : &it->second;
}

const common_tool_profile * common_tool_catalog::find_profile(const std::string & id) const {
    const auto it = profiles.find(id);
    return it == profiles.end() ? nullptr : &it->second;
}

std::vector<common_tool_definition> common_tool_catalog::load_profile(const std::string & id, std::string & error) const {
    const auto * profile = find_profile(id);
    if (!profile || !profile->enabled) { error = "tool profile is unavailable"; return {}; }
    std::vector<common_tool_definition> loaded;
    for (const auto & member : profile->members) {
        if (!member.enabled) continue;
        const auto * definition = find_definition(member.tool_name, member.tool_version);
        if (!definition || !definition->enabled) { error = "tool profile references an unavailable tool"; return {}; }
        loaded.push_back(*definition);
    }
    error.clear();
    return loaded;
}
