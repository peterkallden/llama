#pragma once

#include "memory/memory-types.h"

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

class common_memory_store {
public:
    virtual ~common_memory_store() = default;

    virtual bool open(const std::string & path, std::string & error) = 0;
    virtual void close() = 0;

    virtual bool put(const common_memory_record & record, std::string & error) = 0;
    virtual std::optional<common_memory_record> get(const std::string & id, std::string & error) = 0;
    virtual std::vector<common_memory_record> list(const common_memory_query & query, std::string & error) = 0;
    virtual std::vector<common_memory_hit> search(const common_memory_query & query, std::string & error) = 0;
    virtual bool relate(const std::string & from, const std::string & relation, const std::string & to, float weight, std::string & error) = 0;
    virtual std::vector<common_memory_relation> list_relations(
        const std::string & id, const std::string & relation, bool outgoing,
        const common_memory_query & scope, std::string & error) = 0;
    virtual bool erase(const std::string & id, std::string & error) = 0;
};

inline bool common_memory_same_scope(const common_memory_record & a, const common_memory_record & b) {
    return a.scope == b.scope && a.namespace_id == b.namespace_id &&
        a.session_id == b.session_id && a.project_id == b.project_id && a.turn_id == b.turn_id;
}

inline bool common_memory_validate_relation(
        common_memory_store & store, const std::string & from,
        const std::string & relation, const std::string & to,
        std::string & error) {
    if (relation != "supersedes") { error.clear(); return true; }
    auto source = store.get(from, error);
    if (!error.empty() || !source) { if (error.empty()) error = "supersedes source memory does not exist"; return false; }
    auto target = store.get(to, error);
    if (!error.empty() || !target) { if (error.empty()) error = "supersedes target memory does not exist"; return false; }
    if (!common_memory_same_scope(*source, *target)) { error = "supersedes relation must stay within the same memory scope"; return false; }
    common_memory_query scope;
    scope.scope = source->scope;
    scope.namespace_id = source->namespace_id;
    scope.session_id = source->session_id;
    scope.project_id = source->project_id;
    scope.turn_id = source->turn_id;
    scope.global_opt_in = source->scope == common_memory_scope::global;
    std::vector<std::string> pending{to};
    std::vector<std::string> seen;
    while (!pending.empty()) {
        auto current = std::move(pending.back()); pending.pop_back();
        if (current == from) { error = "supersedes relation would create a cycle"; return false; }
        if (std::find(seen.begin(), seen.end(), current) != seen.end()) continue;
        seen.push_back(current);
        for (const auto & edge : store.list_relations(current, "supersedes", true, scope, error)) {
            if (!error.empty()) return false;
            pending.push_back(edge.to);
        }
    }
    error.clear();
    return true;
}
