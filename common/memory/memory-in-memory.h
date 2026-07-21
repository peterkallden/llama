#pragma once

#include "memory/memory-store.h"

#include <map>
#include <tuple>

class common_memory_in_memory_store : public common_memory_store {
public:
    bool open(const std::string & path, std::string & error) override;
    void close() override;

    bool put(const common_memory_record & record, std::string & error) override;
    std::optional<common_memory_record> get(const std::string & id, std::string & error) override;
    std::vector<common_memory_record> list(const common_memory_query & query, std::string & error) override;
    std::vector<common_memory_hit> search(const common_memory_query & query, std::string & error) override;
    bool relate(const std::string & from, const std::string & relation, const std::string & to, float weight, std::string & error) override;
    bool erase(const std::string & id, std::string & error) override;

private:
    struct edge {
        std::string from;
        std::string relation;
        std::string to;
        float weight = 0.0f;
        int64_t created_at = 0;
    };

    bool opened = false;
    std::map<std::string, common_memory_record> records;
    std::vector<edge> edges;
};
