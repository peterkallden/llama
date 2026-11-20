#pragma once

#include <nlohmann/json.hpp>
#include <cstdint>
#include <string>
#include <vector>

struct common_source_reference {
    std::string type = {};
    std::string id = {};
    std::string location = {};
    std::string revision = {};
    int64_t observed_at = 0;
    std::string branch = {};
    bool local_changes = false;
};

inline bool operator==(const common_source_reference & a, const common_source_reference & b) {
    return a.type == b.type && a.id == b.id && a.location == b.location && a.revision == b.revision &&
        a.observed_at == b.observed_at && a.branch == b.branch && a.local_changes == b.local_changes;
}
inline void to_json(nlohmann::ordered_json & j, const common_source_reference & r) {
    j = {{"type", r.type}, {"id", r.id}, {"location", r.location}, {"revision", r.revision},
        {"observed_at", r.observed_at}, {"branch", r.branch}, {"local_changes", r.local_changes}};
}
inline void from_json(const nlohmann::ordered_json & j, common_source_reference & r) {
    r.type = j.at("type").get<std::string>(); r.id = j.at("id").get<std::string>();
    r.location = j.value("location", std::string{}); r.revision = j.value("revision", std::string{});
    r.observed_at = j.value("observed_at", int64_t(0)); r.branch = j.value("branch", std::string{});
    r.local_changes = j.value("local_changes", false);
}
inline bool common_source_references_valid(const std::vector<common_source_reference> & refs) {
    if (refs.size() > 16) return false;
    for (const auto & r : refs) {
        if (r.type.empty() || r.type.size() > 32 || r.id.empty() || r.id.size() > 512 ||
                r.location.size() > 512 || r.revision.size() > 256 || r.branch.size() > 256 || r.observed_at < 0) return false;
        for (const auto * value : {&r.type, &r.id, &r.location, &r.revision, &r.branch})
            if (value->find('\0') != std::string::npos) return false;
    }
    return true;
}
