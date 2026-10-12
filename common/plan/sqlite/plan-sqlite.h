#pragma once

#include "plan/plan-in-memory.h"
#include "agent/storage/sqlite/sqlite-database.h"

class common_plan_sqlite_store final : public common_plan_store {
public:
    common_plan_sqlite_store() = default;
    ~common_plan_sqlite_store() override;

    bool open(const std::string & path, std::string & error) override;
    void close() override;
    bool create(const common_plan_state & plan, std::string & error) override;
    std::optional<common_plan_state> get(const std::string & plan_id, std::string & error) override;
    std::vector<common_plan_state> list(std::string & error) override;
    bool apply(const common_plan_operation & operation, common_plan_state & updated_plan, std::string & error) override;
    std::vector<common_plan_event> history(const std::string & plan_id, std::string & error) override;
    bool erase(const std::string & plan_id, std::string & error) override;

private:
    bool ensure_schema(std::string & error);
    bool load_cache(std::string & error);
    bool persist_plan(const common_plan_state & plan, std::string & error);
    bool persist_event(const std::string & plan_id, const common_plan_event & event, const common_plan_operation & operation, std::string & error);

    common_sqlite_database database_;
    common_plan_in_memory_store cache_;
};
