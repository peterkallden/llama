#pragma once

#include "memory/memory-store.h"

class common_memory_cozo_store : public common_memory_store {
public:
    common_memory_cozo_store();
    ~common_memory_cozo_store() override;

    bool open(const std::string & path, std::string & error) override;
    void close() override;

    bool put(const common_memory_record & record, std::string & error) override;
    std::optional<common_memory_record> get(const std::string & id, std::string & error) override;
    std::vector<common_memory_record> list(const common_memory_query & query, std::string & error) override;
    std::vector<common_memory_hit> search(const common_memory_query & query, std::string & error) override;
    bool relate(const std::string & from, const std::string & relation, const std::string & to, float weight, std::string & error) override;
    bool erase(const std::string & id, std::string & error) override;

private:
    int32_t db_id = -1;

    bool run(const std::string & script, const std::string & params_json, std::string & result_json, std::string & error) const;
};
