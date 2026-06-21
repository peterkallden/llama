#pragma once
#include "plan/plan-in-memory.h"
class common_plan_cozo_store final : public common_plan_store {
public:
    common_plan_cozo_store(); ~common_plan_cozo_store() override;
    bool open(const std::string & path, std::string & error) override; void close() override;
    bool create(const common_plan_state & plan, std::string & error) override;
    std::optional<common_plan_state> get(const std::string & plan_id, std::string & error) override;
    std::vector<common_plan_state> list(std::string & error) override;
    bool apply(const common_plan_operation & operation, common_plan_state & updated_plan, std::string & error) override;
    std::vector<common_plan_event> history(const std::string & plan_id, std::string & error) override;
    bool erase(const std::string & plan_id, std::string & error) override;
private:
    int32_t db_id = -1; common_plan_in_memory_store cache;
    bool run(const std::string & script, const std::string & params_json, std::string & result_json, std::string & error) const;
    bool persist_plan(const common_plan_state & plan, std::string & error);
    bool reload_cache(std::string & error);
};
