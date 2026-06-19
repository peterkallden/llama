#include "agent/reflection-json.h"
#include <nlohmann/json.hpp>
using json = nlohmann::ordered_json;
bool common_reflection_parse_json(const std::string & text, common_reflection_result & result, std::string & error, size_t max_operations) {
    try { auto j = json::parse(text); if (!j.is_object() || !j.contains("decision") || !j["decision"].is_string()) { error = "reflection must be a JSON object with decision"; return false; }
        const auto d = j["decision"].get<std::string>(); if (d == "accept") result.decision = common_reflection_decision::accept; else if (d == "revise") result.decision = common_reflection_decision::revise; else if (d == "replan") result.decision = common_reflection_decision::replan; else if (d == "request_action") result.decision = common_reflection_decision::request_action; else if (d == "abort") result.decision = common_reflection_decision::abort; else { error = "unsupported reflection decision"; return false; }
        result.ready_to_answer = j.value("ready_to_answer", false); result.confidence = j.value("confidence", 0.5f); if (j.contains("revision_guidance")) for (const auto & v : j["revision_guidance"]) { if (!v.is_string()) { error = "revision guidance must be strings"; return false; } result.revision_guidance.push_back(v.get<std::string>()); }
        if (j.contains("operations") && j["operations"].size() > max_operations) { error = "too many reflection operations"; return false; } error.clear(); return true;
    } catch (const json::exception &) { error = "malformed reflection JSON"; return false; }
}
