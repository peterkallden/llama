#pragma once

#include "agent/tooling/catalog/tool-catalog.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

// Host-generated contract mutations are safe, deterministic fixtures. They
// exercise the same model-facing contract that a real model would receive,
// but they are not evidence of model behaviour and must not be promoted to a
// FlyDelta learning candidate on their own.
enum class common_synthetic_tool_mutation {
    missing_required,
    wrong_type,
    invalid_enum,
    unexpected_property,
    malformed_json,
};

const char * common_synthetic_tool_mutation_name(common_synthetic_tool_mutation mutation);

struct common_synthetic_tool_case {
    int schema_version = 1;
    std::string id;
    uint64_t seed = 0;
    std::string tool_name;
    std::string tool_family;
    std::string provider_kind;
    common_synthetic_tool_mutation mutation = common_synthetic_tool_mutation::missing_required;

    // Host-facing material used for deterministic validation and repair.
    std::string host_input_schema_json;
    std::string baseline_arguments_json;
    std::string mutated_arguments_json;
    std::string expected_arguments_json;

    // The exact model-facing projection and compact contract supplied to a
    // model in a later model-backed experiment.
    std::string model_input_schema_json;
    std::string model_facing_contract;
    std::string model_facing_prompt;

    std::string schema_fingerprint;
    std::string verifier_revision = "synthetic-tool-validator-v1";
    bool mutated_valid = false;
    bool repair_valid = false;
    bool host_verified = false;
};

bool common_synthetic_tool_case_validate(
        const common_synthetic_tool_case & value,
        size_t max_text_size,
        std::string & error);

// Generates bounded mutations from a host-owned tool definition. The output
// order and values are stable for the same definition and seed.
bool common_synthetic_tool_cases_generate(
        const common_tool_definition & definition,
        const std::string & tool_family,
        const std::string & provider_kind,
        uint64_t seed,
        size_t max_cases,
        std::vector<common_synthetic_tool_case> & cases,
        std::string & error);

// Re-runs the host validator against a generated case. This is deliberately
// separate from generation so tests and future model-backed runners can use
// the same verifier without trusting fixture metadata.
bool common_synthetic_tool_case_verify(
        common_synthetic_tool_case & value,
        std::string & error);

std::string common_synthetic_tool_case_to_json(
        const common_synthetic_tool_case & value);

bool common_synthetic_tool_case_from_json(
        const std::string & text,
        size_t max_text_size,
        common_synthetic_tool_case & value,
        std::string & error);

bool common_synthetic_tool_cases_export_jsonl(
        const std::vector<common_synthetic_tool_case> & cases,
        const std::filesystem::path & path,
        size_t max_bytes,
        std::string & error);
