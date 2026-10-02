#pragma once

#include <string>
#include <vector>

// A small host-facing intermediate representation for concept smoke and
// extraction supervision. It describes the behavior, not the model's JSON
// spelling or a particular tool registry schema.
struct common_flydelta_semantic_predicate {
    std::string field;
    std::string operation;
    std::string value;
};

struct common_flydelta_semantic_decision {
    int schema_version = 1;
    std::string operation;
    std::string dataset;
    std::vector<std::string> group_by;
    std::string aggregate_function;
    std::string aggregate_field;
    std::vector<common_flydelta_semantic_predicate> predicates;
    std::string order_by_field;
    std::string order_by_direction;
    int limit = 0;
};

enum class common_flydelta_semantic_decision_status {
    valid,
    parse_failure,
    unsupported_operation,
    missing_field,
    invalid_value,
};

const char * common_flydelta_semantic_decision_status_name(
        common_flydelta_semantic_decision_status status);

bool common_flydelta_semantic_decision_validate(
        const common_flydelta_semantic_decision & decision,
        std::string & error);

// Accepts either the small SemanticDecision IR or a model-shaped tool call:
// {"name":"data.filter","arguments":{...}}. The result is deterministic
// and independent of whitespace, argument ordering and accepted aliases.
bool common_flydelta_parse_semantic_decision(
        const std::string & text,
        common_flydelta_semantic_decision & decision,
        common_flydelta_semantic_decision_status & status,
        std::string & error);

bool common_flydelta_semantic_decision_equal(
        const common_flydelta_semantic_decision & left,
        const common_flydelta_semantic_decision & right);

// Lenient, host-owned observation used only to measure iterative semantic
// progress. It never makes an invalid model-facing call executable and never
// substitutes for the strict Oracle result.
enum class common_flydelta_semantic_progress_field_state {
    unknown,
    missing,
    mismatch,
    partial,
    matched,
};

const char * common_flydelta_semantic_progress_field_state_name(
        common_flydelta_semantic_progress_field_state state);

struct common_flydelta_semantic_progress_observation {
    int schema_version = 1;
    bool available = false;
    bool contract_valid = false;
    std::string operation;
    std::string dataset;
    std::string grouping;
    std::string measure;
    common_flydelta_semantic_progress_field_state operation_state =
        common_flydelta_semantic_progress_field_state::unknown;
    common_flydelta_semantic_progress_field_state dataset_state =
        common_flydelta_semantic_progress_field_state::unknown;
    common_flydelta_semantic_progress_field_state grouping_state =
        common_flydelta_semantic_progress_field_state::unknown;
    common_flydelta_semantic_progress_field_state measure_state =
        common_flydelta_semantic_progress_field_state::unknown;
    common_flydelta_semantic_progress_field_state contract_state =
        common_flydelta_semantic_progress_field_state::unknown;
    int score = 0;
};

enum class common_flydelta_semantic_progress_outcome {
    unknown,
    unchanged,
    improved,
    solved,
    regressed,
};

const char * common_flydelta_semantic_progress_outcome_name(
        common_flydelta_semantic_progress_outcome outcome);

struct common_flydelta_semantic_progress {
    int schema_version = 1;
    common_flydelta_semantic_progress_outcome outcome =
        common_flydelta_semantic_progress_outcome::unknown;
    bool comparable = false;
    int baseline_score = 0;
    int candidate_score = 0;
    std::vector<std::string> improved_dimensions;
    std::vector<std::string> regressed_dimensions;
    std::vector<std::string> residual_dimensions;
};

// Extracts bounded, best-effort semantic fields from model-facing output.
// Strict parsing remains authoritative for execution and evidence.
bool common_flydelta_observe_semantic_progress(
        const std::string & generated,
        const common_flydelta_semantic_decision & expected,
        common_flydelta_semantic_progress_observation & observation,
        std::string & error);

common_flydelta_semantic_progress common_flydelta_compare_semantic_progress(
        const common_flydelta_semantic_progress_observation & baseline,
        const common_flydelta_semantic_progress_observation & candidate,
        bool baseline_strict_passed,
        bool candidate_strict_passed);
