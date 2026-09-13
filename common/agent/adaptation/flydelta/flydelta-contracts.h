#pragma once

#include "agent/adaptation/flydelta/flydelta.h"
#include "agent/adaptation/learning-transaction.h"

#include <cstddef>
#include <string>
#include <vector>

enum class common_flydelta_candidate_status { observed, eligible, approved, rejected, revoked };

const char * common_flydelta_candidate_status_name(common_flydelta_candidate_status status);

// A capture manifest references bounded, already verified evidence. It does
// not contain raw prompts, tool output, credentials or activation tensors.
struct common_flydelta_capture_manifest {
    int schema_version = 1;
    std::string id;
    std::string observation_id;
    std::string model_profile_fingerprint;
    std::string template_fingerprint;
    std::string positive_execution_ref;
    std::string negative_execution_ref;
    std::string capture_layout_revision;
    std::string evidence_hash;
    bool redaction_attested = false;
    size_t captured_bytes = 0;
};

bool common_flydelta_capture_manifest_validate(
        const common_flydelta_capture_manifest & manifest,
        size_t max_captured_bytes,
        std::string & error);
std::string common_flydelta_capture_manifest_to_json(
        const common_flydelta_capture_manifest & manifest);
bool common_flydelta_capture_manifest_from_json(
        const std::string & text,
        size_t max_captured_bytes,
        common_flydelta_capture_manifest & manifest,
        std::string & error);

struct common_flydelta_candidate_policy {
    size_t min_observations = 3;
    size_t min_verified_recoveries = 2;
    size_t max_observations = 64;
    size_t max_capture_manifests = 64;
    float min_confidence = 0.80f;
};

// Runtime V0 input. The artifact must already have been resolved and
// verified by the host. The generation path only applies this data to a fresh
// context; it never loads an artifact or mutates a resident model.
struct common_flydelta_static_overlay {
    bool enabled = false;
    std::string artifact_id;
    int32_t n_embd = 0;
    int32_t il_start = 1;
    int32_t il_end = 0;
    float scale = 1.0f;
    std::vector<float> data;
};

bool common_flydelta_static_overlay_validate(
        const common_flydelta_static_overlay & overlay,
        size_t model_n_embd,
        size_t model_n_layers,
        size_t max_bytes,
        std::string & error);

// This record contains references and qualification evidence only. It is a
// sideband candidate, not an activation request and not a training corpus.
struct common_flydelta_candidate {
    int schema_version = 1;
    std::string id;
    common_agent_scope scope;
    std::vector<std::string> transaction_ids;
    std::vector<std::string> capture_manifest_ids;
    common_learning_cause cause = common_learning_cause::unknown;
    common_learning_verification verification = common_learning_verification::unverified;
    std::string learning_domain;
    std::string tool_family;
    std::string provider_kind;
    size_t observed_occurrences = 0;
    size_t verified_recoveries = 0;
    size_t contradictions = 0;
    float confidence = 0.0f;
    common_flydelta_candidate_status status = common_flydelta_candidate_status::observed;
};

bool common_flydelta_candidate_validate(
        const common_flydelta_candidate & candidate,
        const common_flydelta_candidate_policy & policy,
        std::string & error);

bool common_flydelta_candidate_from_transactions(
        const std::vector<common_learning_transaction> & transactions,
        const common_flydelta_candidate_policy & policy,
        common_flydelta_candidate & candidate,
        std::string & error);

struct common_flydelta_evaluation_report {
    int schema_version = 1;
    std::string revision_id;
    std::string candidate_id;
    std::string baseline_profile_id;
    std::string candidate_profile_id;
    std::string test_suite_revision;
    bool intended_behavior_passed = false;
    bool retention_passed = false;
    bool agent_regression_passed = false;
    size_t evaluated_turns = 0;
    size_t baseline_successes = 0;
    size_t candidate_successes = 0;
    size_t candidate_interventions = 0;
    size_t false_interventions = 0;
    std::string status = "failed";
};

bool common_flydelta_evaluation_report_validate(
        const common_flydelta_evaluation_report & report,
        std::string & error);
std::string common_flydelta_evaluation_report_to_json(
        const common_flydelta_evaluation_report & report);
bool common_flydelta_evaluation_report_from_json(
        const std::string & text,
        common_flydelta_evaluation_report & report,
        std::string & error);
