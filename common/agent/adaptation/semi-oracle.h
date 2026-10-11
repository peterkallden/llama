#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

enum class common_semi_oracle_domain { placement_game, linear_algebra };
enum class common_semi_oracle_scope { session, host };
enum class common_semi_oracle_rule_authority { declared, verified, inferred };
enum class common_semi_oracle_level { local, bounded, guided };
enum class common_semi_oracle_verdict { satisfied, violated, unknown };

const char * common_semi_oracle_domain_name(common_semi_oracle_domain value);
const char * common_semi_oracle_scope_name(common_semi_oracle_scope value);
const char * common_semi_oracle_rule_authority_name(common_semi_oracle_rule_authority value);
const char * common_semi_oracle_level_name(common_semi_oracle_level value);
const char * common_semi_oracle_verdict_name(common_semi_oracle_verdict value);

struct common_semi_oracle_rule_assertion {
    std::string key;
    std::string value;
    common_semi_oracle_rule_authority authority = common_semi_oracle_rule_authority::declared;
    std::string source_ref;
};

struct common_semi_oracle_ruleset {
    int schema_version = 1;
    std::string id;                 // host-generated opaque identity
    std::string name;
    std::string description;
    common_semi_oracle_domain domain = common_semi_oracle_domain::placement_game;
    common_semi_oracle_scope scope = common_semi_oracle_scope::session;
    std::string scope_id;
    uint64_t revision = 1;
    uint64_t parent_revision = 0;
    std::string semantic_fingerprint;
    std::string action_contract_fingerprint;
    std::string spec_json;
    std::vector<common_semi_oracle_rule_assertion> assertions;
    bool active = false;
};

struct common_semi_oracle_budget {
    size_t max_expansions = 128;
    size_t max_depth = 8;
};

struct common_semi_oracle_query {
    std::string ruleset_id;
    uint64_t ruleset_revision = 0;
    common_semi_oracle_level level = common_semi_oracle_level::local;
    std::string state_json;
    std::string action_dsl;
    std::string goal;
    common_semi_oracle_budget budget;
    // Optional host-selected action ordering. It can affect exploration order only.
    std::vector<std::string> guidance;
};

struct common_semi_oracle_result {
    common_semi_oracle_verdict verdict = common_semi_oracle_verdict::unknown;
    std::string ruleset_id;
    uint64_t ruleset_revision = 0;
    std::string evaluator_ref;
    std::string evaluator_revision;
    common_semi_oracle_level level = common_semi_oracle_level::local;
    size_t expansions = 0;
    size_t depth_reached = 0;
    std::string proof_json;
    std::string counterexample_json;
    std::string reason;
    std::string guidance_ref;
};

// A deliberately small, non-executable RuleSpec DSL. Only registered domain
// families and host-owned primitives can be represented.
bool common_semi_oracle_parse_ruleset_dsl(
        const std::string & text,
        const std::string & host_id,
        const std::string & scope_id,
        common_semi_oracle_ruleset & ruleset,
        std::string & error,
        common_semi_oracle_rule_authority assertion_authority =
            common_semi_oracle_rule_authority::inferred);
bool common_semi_oracle_revise_ruleset(
        const common_semi_oracle_ruleset & current,
        const std::string & revised_dsl,
        common_semi_oracle_rule_authority assertion_authority,
        const std::string & new_session_scope_id,
        common_semi_oracle_ruleset & revised,
        std::string & error);
bool common_semi_oracle_validate_ruleset(
        const common_semi_oracle_ruleset & ruleset,
        std::string & error);
std::string common_semi_oracle_ruleset_to_json(const common_semi_oracle_ruleset & ruleset);
bool common_semi_oracle_ruleset_from_json(
        const std::string & text,
        common_semi_oracle_ruleset & ruleset,
        std::string & error);

// The dynamic action contract is rendered from the active host-validated revision.
std::string common_semi_oracle_action_contract(const common_semi_oracle_ruleset & ruleset);

bool common_semi_oracle_evaluate(
        const common_semi_oracle_ruleset & ruleset,
        const common_semi_oracle_query & query,
        common_semi_oracle_result & result,
        std::string & error);

// The shared teaching seam may consume only a proven contrast under rules whose
// applicable assertions are declared or verified.
bool common_semi_oracle_learning_authorized(
        const common_semi_oracle_ruleset & ruleset,
        const common_semi_oracle_result & result);
