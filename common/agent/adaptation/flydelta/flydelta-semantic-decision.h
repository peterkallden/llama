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

