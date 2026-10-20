#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

enum class common_semi_oracle_domain { placement_game, linear_algebra, target_hunt };
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

enum class common_semi_oracle_search_algorithm { exact, focal };
enum class common_semi_oracle_history_order { before_heuristic, after_heuristic };
struct common_semi_oracle_search_policy {
    common_semi_oracle_search_algorithm algorithm = common_semi_oracle_search_algorithm::exact;
    double suboptimality_bound = 1.0;
    common_semi_oracle_history_order exact_history_order = common_semi_oracle_history_order::before_heuristic;
};

struct common_semi_oracle_frontier_event {
    size_t decision = 0;
    size_t expansions_used = 0;
    size_t expansions_remaining = 0;
    size_t frontier_size = 0;
    size_t equal_f_candidates = 0;
    size_t focal_candidates = 0;
    size_t focal_f_classes = 0;
    size_t f_min = 0;
    double focal_limit = 0.0;
    size_t selected_g = 0, selected_h = 0, selected_f = 0;
    double selected_score = 0.0;
    std::string selected_state, selected_action, posterior_identity;
    size_t stale_entries_removed = 0;
    size_t duplicate_successors_suppressed = 0;
    size_t states_reopened = 0;
};
using common_semi_oracle_frontier_observer = std::function<void(const common_semi_oracle_frontier_event &)>;

// Receives only host-generated legal successor actions. Returns a permutation
// of their indices; it cannot introduce a state, action, cost or heuristic.
using common_semi_oracle_transition_adviser = std::function<bool(
        const std::string &, const std::vector<std::string> &,
        std::vector<size_t> &, std::string &)>;

// Host-owned score for one legal transition: state before, state after, action.
// Lower scores are preferred. This changes frontier ordering only.
using common_semi_oracle_transition_score = std::function<double(
        const std::string &, const std::string &, const std::string &)>;

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
    common_semi_oracle_transition_adviser transition_adviser;
    common_semi_oracle_transition_score transition_score;
    std::function<std::string(const std::string &, const std::string &, const std::string &)> historical_identity;
    bool record_search_history = false;
    bool use_historical_order = false;
    common_semi_oracle_search_policy search_policy;
    common_semi_oracle_frontier_observer frontier_observer;
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
    // Host-owned search diagnostics, separate from proof and training credit.
    std::string search_guidance_json;
    // Versioned, host-verified Target Hunt branch labels. These are distinct
    // from censored search-suffix observations and never enter the proof.
    std::string search_branch_quality_json;
    common_semi_oracle_search_policy applied_search_policy;
};

enum class common_semi_oracle_placement_value { loss = -1, draw = 0, win = 1, unknown = 2 };

struct common_semi_oracle_placement_action_value {
    std::string ruleset_id;
    uint64_t ruleset_revision = 0;
    std::string evaluator_ref;
    std::string evaluator_revision;
    std::string state_fingerprint;
    std::string persisted_evaluation_ref;
    bool ranking_complete = false;
    std::string player;
    int row = -1;
    int column = -1;
    // Outcome is always from the perspective of the player to move in the
    // supplied state, assuming optimal play by both players.
    common_semi_oracle_placement_value value = common_semi_oracle_placement_value::unknown;
    size_t expansions = 0;
    size_t depth_reached = 0;
    std::string proof_json;
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

// Validates host-owned domain state before it is bound to an active RuleSet.
// It accepts no model-supplied state on the runtime action path.
bool common_semi_oracle_validate_state(
        const common_semi_oracle_ruleset & ruleset,
        const std::string & state_json,
        std::string & error);

bool common_semi_oracle_evaluate(
        const common_semi_oracle_ruleset & ruleset,
        const common_semi_oracle_query & query,
        common_semi_oracle_result & result,
        std::string & error);

// Computes a proven minimax value for every legal placement in one state.
// If the shared search budget is exhausted, unresolved actions are UNKNOWN;
// guidance cannot change values and is intentionally not accepted here.
bool common_semi_oracle_rank_placements(
        const common_semi_oracle_ruleset & ruleset,
        const std::string & state_json,
        common_semi_oracle_budget budget,
        std::vector<common_semi_oracle_placement_action_value> & values,
        std::string & error);

// Stable leakage group for a placement-game position. Square boards include
// all dihedral symmetries; rectangular boards include dimension-preserving
// reflections. RuleSet revision and side-to-move are part of the identity.
bool common_semi_oracle_placement_state_group(
        const common_semi_oracle_ruleset & ruleset,
        const std::string & state_json,
        std::string & group_key,
        std::string & error);

// The shared teaching seam may consume only a proven contrast under rules whose
// applicable assertions are declared or verified.
bool common_semi_oracle_learning_authorized(
        const common_semi_oracle_ruleset & ruleset,
        const common_semi_oracle_result & result);
