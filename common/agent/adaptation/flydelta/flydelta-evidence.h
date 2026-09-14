#pragma once

#include "agent/adaptation/flydelta/flydelta-experiment.h"
#include "agent/adaptation/flydelta/flydelta-training.h"
#include "agent/adaptation/learning-transaction.h"

#include <cstddef>
#include <string>
#include <vector>

// A bounded relation between one host-verified baseline attempt and one
// host-verified alternative. Raw prompts and outputs remain in the normal
// evidence store; FlyDelta keeps references and fingerprints only.
struct common_flydelta_behavior_transition {
    int schema_version = 1;
    std::string id;
    common_adaptation_evidence_source source = common_adaptation_evidence_source::tool_repair;
    common_agent_scope scope;
    std::string task_fingerprint;
    std::string baseline_transaction_id;
    std::string candidate_transaction_id;
    std::string baseline_execution_ref;
    std::string candidate_execution_ref;
    std::string host_verifier_ref;
};

bool common_flydelta_behavior_transition_validate(
        const common_flydelta_behavior_transition & transition,
        std::string & error);
// Convenience adapter for the existing tool failure/recovery signals.
bool common_flydelta_tool_repair_transition_from_transactions(
        const common_learning_transaction & failed,
        const common_learning_transaction & repaired,
        const std::string & task_fingerprint,
        const std::string & failed_execution_ref,
        const std::string & repaired_execution_ref,
        const std::string & host_verifier_ref,
        common_flydelta_behavior_transition & transition,
        std::string & error);
// Generic adapter for any host-verified baseline/candidate relation. Both
// transaction IDs must be present in the evidence relation; the host decides
// what the candidate means and which verifier produced the evidence.
bool common_flydelta_behavior_transition_from_evidence(
        const common_adaptation_evidence & evidence,
        const std::string & baseline_transaction_id,
        const std::string & candidate_transaction_id,
        common_flydelta_behavior_transition & transition,
        std::string & error);

struct common_flydelta_contrast_set {
    int schema_version = 1;
    std::string id;
    std::string behavior_key;
    common_agent_scope scope;
    std::vector<std::string> transition_ids;
    std::vector<std::string> positive_transaction_ids;
    std::vector<std::string> negative_transaction_ids;
};

bool common_flydelta_contrast_set_validate(
        const common_flydelta_contrast_set & contrast_set,
        size_t max_transitions,
        std::string & error);
bool common_flydelta_contrast_set_from_transitions(
        const std::string & id,
        const std::string & behavior_key,
        const std::vector<common_flydelta_behavior_transition> & transitions,
        size_t max_transitions,
        common_flydelta_contrast_set & contrast_set,
        std::string & error);

// Credit is derived from a baseline/candidate counterfactual report. It is
// negative evidence as well as positive evidence; UNKNOWN is never promoted.
struct common_flydelta_intervention_credit {
    int schema_version = 1;
    std::string experiment_id;
    std::string candidate_id;
    std::string fixture_id;
    common_flydelta_counterfactual_outcome outcome = common_flydelta_counterfactual_outcome::unknown;
    float quality_delta = 0.0f;
    bool eligible_for_learning = false;
};

bool common_flydelta_intervention_credit_validate(
        const common_flydelta_intervention_credit & credit,
        std::string & error);
bool common_flydelta_intervention_credit_from_report(
        const common_flydelta_counterfactual_report & report,
        common_flydelta_intervention_credit & credit,
        std::string & error);

// Host bridge from the shared evidence ledger into a FlyDelta experiment.
// This is reference-only: it does not load captures, infer a repair, or
// assert that a reflection suggestion is correct.
struct common_flydelta_experiment_seed {
    int schema_version = 1;
    std::string id;
    std::string behavior_key;
    common_adaptation_evidence_source source = common_adaptation_evidence_source::tool_repair;
    common_agent_scope scope;
    common_flydelta_training_split split = common_flydelta_training_split::train;
    std::string task_fingerprint;
    std::string model_profile_fingerprint;
    std::string tokenizer_fingerprint;
    std::string template_fingerprint;
    std::string execution_context_fingerprint;
    std::string baseline_ref;
    std::string candidate_ref;
    std::string verifier_ref;
    std::string evidence_ref;
    std::vector<std::string> transaction_ids;
};

bool common_flydelta_experiment_seed_validate(
        const common_flydelta_experiment_seed & seed,
        std::string & error);
bool common_flydelta_experiment_seed_from_evidence(
        const common_adaptation_evidence & evidence,
        const std::string & behavior_key,
        const std::string & model_profile_fingerprint,
        const std::string & tokenizer_fingerprint,
        const std::string & template_fingerprint,
        const std::string & execution_context_fingerprint,
        common_flydelta_training_split split,
        common_flydelta_experiment_seed & seed,
        std::string & error);
bool common_flydelta_experiment_fixture_from_seed(
        const common_flydelta_experiment_seed & seed,
        common_flydelta_experiment_fixture & fixture,
        std::string & error);
