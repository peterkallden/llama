#pragma once
#include "agent/adaptation/semi-oracle.h"

bool common_semi_oracle_target_rules(const common_semi_oracle_ruleset &, std::string & error);
bool common_semi_oracle_target_state(const common_semi_oracle_ruleset &, const std::string &, std::string & error);
bool common_semi_oracle_target_problem(const common_semi_oracle_ruleset &, const std::string & dsl,
        std::string & state, std::string & error);
bool common_semi_oracle_target_evaluate(const common_semi_oracle_ruleset &,
        const common_semi_oracle_query &, common_semi_oracle_result &, std::string & error);
// These projections use only declared geometry, never a solved distance map.
std::string common_semi_oracle_target_group(const std::string & state);
std::string common_semi_oracle_target_category(const std::string & state, const std::string & action);
