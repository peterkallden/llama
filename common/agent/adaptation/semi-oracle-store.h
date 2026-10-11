#pragma once

#include "agent/adaptation/semi-oracle.h"

#include <string>
#include <vector>

class common_semi_oracle_store {
public:
    virtual ~common_semi_oracle_store() = default;
    virtual bool put_revision(const common_semi_oracle_ruleset & ruleset, std::string & error) = 0;
    virtual bool get_revision(const std::string & id, uint64_t revision,
            common_semi_oracle_ruleset & ruleset, bool & found, std::string & error) const = 0;
    virtual bool promote_to_host(const std::string & id, uint64_t revision, std::string & error) = 0;
    virtual bool bind_session(const std::string & session_id, const std::string & id,
            uint64_t revision, std::string & error) = 0;
    virtual bool get_session_binding(const std::string & session_id, std::string & id,
            uint64_t & revision, bool & found, std::string & error) const = 0;
    virtual bool append_evaluation(const std::string & evaluation_id,
            const common_semi_oracle_result & result, std::string & error) = 0;
    virtual std::vector<std::string> list_host_rulesets(std::string & error) const = 0;
};
