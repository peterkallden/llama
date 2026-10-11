#pragma once

#include "agent/adaptation/semi-oracle-store.h"

#include <cstdint>

class agent_semi_oracle_cozo_store final : public common_semi_oracle_store {
public:
    ~agent_semi_oracle_cozo_store() override;
    bool open(const std::string & path, std::string & error);
    void close();
    bool put_revision(const common_semi_oracle_ruleset &, std::string &) override;
    bool get_revision(const std::string &, uint64_t, common_semi_oracle_ruleset &, bool &, std::string &) const override;
    bool promote_to_host(const std::string &, uint64_t, std::string &) override;
    bool bind_session(const std::string &, const std::string &, uint64_t, std::string &) override;
    bool get_session_binding(const std::string &, std::string &, uint64_t &, bool &, std::string &) const override;
    bool append_evaluation(const std::string &, const common_semi_oracle_result &, std::string &) override;
    std::vector<std::string> list_host_rulesets(std::string &) const override;
private:
    bool run(const std::string &, const std::string &, std::string &, std::string &) const;
    int32_t db_id_ = -1;
};
