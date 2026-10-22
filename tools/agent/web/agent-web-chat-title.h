#pragma once

#include <nlohmann/json.hpp>
#include <string>

// Presentation only: never modify the plan's goal or the caller-owned purpose.
inline std::string agent_web_chat_title(const std::string & text) {
    const auto start = text.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return {};
    auto title = text.substr(start, text.find_first_of("\r\n", start) - start);
    const auto end = title.find_last_not_of(" \t");
    title.resize(end + 1);
    if (title.size() > 80) {
        size_t boundary = 77;
        while (boundary && (static_cast<unsigned char>(title[boundary]) & 0xc0) == 0x80) --boundary;
        title.resize(boundary);
        title += "...";
    }
    return title;
}

inline bool agent_web_apply_goal_title(nlohmann::ordered_json & chat, const std::string & turn_id,
        const std::string & plan_id, const nlohmann::ordered_json & plans) {
    if (plan_id.empty() || !plans.is_array() || chat.value("latest_turn_id", std::string()) != turn_id) return false;
    const nlohmann::ordered_json * latest = nullptr;
    for (const auto & plan : plans) {
        if (!plan.is_object() || !plan.contains("plan_id") || plan["plan_id"] != plan_id ||
                !plan.contains("goal") || !plan["goal"].is_string() ||
                !plan.contains("version") || (!plan["version"].is_number_integer() || plan["version"] < 0)) continue;
        if (!latest || plan["version"].get<uint64_t>() > (*latest)["version"].get<uint64_t>()) latest = &plan;
    }
    if (!latest) return false;
    const auto title = agent_web_chat_title((*latest)["goal"].get<std::string>());
    if (title.empty()) return false;
    chat["title"] = title;
    chat["title_source"] = "plan_goal";
    chat["title_plan_id"] = plan_id;
    chat["title_plan_version"] = (*latest)["version"];
    return true;
}
