#pragma once

#include "agent/adaptation/flydelta/experiment/flydelta-outcome-guidance.h"
#include "agent/agent-scope.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

enum class common_flydelta_guidance_decision_point {
    deep_full_generation_selection,
};

enum class common_flydelta_guidance_capability {
    observe_only,
    reorder,
    select_budgeted_subset,
    prune,
};

enum class common_flydelta_guidance_qualification_status {
    insufficient_evidence,
    qualified,
    paused,
};

const char * common_flydelta_guidance_decision_point_name(
        common_flydelta_guidance_decision_point point);
const char * common_flydelta_guidance_capability_name(
        common_flydelta_guidance_capability capability);
const char * common_flydelta_guidance_qualification_status_name(
        common_flydelta_guidance_qualification_status status);
std::string common_flydelta_outcome_guidance_scope_fingerprint(
        const common_agent_scope & scope);
std::string common_flydelta_guidance_qualification_policy_revision(
        const common_flydelta_outcome_guidance_config & config);
common_flydelta_outcome_guidance_mode common_flydelta_guidance_effective_mode(
        common_flydelta_outcome_guidance_mode operator_mode,
        common_flydelta_guidance_decision_point point,
        bool qualified);

struct common_flydelta_guidance_qualification_context {
    int schema_version = 1;
    common_flydelta_guidance_decision_point decision_point =
        common_flydelta_guidance_decision_point::deep_full_generation_selection;
    common_flydelta_guidance_capability capability =
        common_flydelta_guidance_capability::select_budgeted_subset;
    // The turn-insensitive identity is retained explicitly so the lifecycle
    // envelope can be checked against the authorized namespace/project/session.
    common_agent_scope authorized_scope;
    common_flydelta_outcome_guidance_context outcome_context;
    std::string policy_revision;
    size_t selection_limit = 0;
    double progress_reward = 0.25;
    double harm_penalty = 5.0;
};

struct common_flydelta_guidance_arm_measurement {
    std::string arm_id;
    common_flydelta_counterfactual_outcome outcome =
        common_flydelta_counterfactual_outcome::unknown;
    // Equal-share attribution of measured batch model time. This supports
    // cost diagnostics, not a claim of measured standalone policy latency.
    double inference_ms = 0.0;
    bool inference_cost_available = false;
    std::string evidence_ref;
    std::optional<common_flydelta_progress_evidence> progress;
};

struct common_flydelta_guidance_fixture_evaluation {
    std::string fixture_group_id;
    std::vector<std::string> baseline_selection;
    std::vector<std::string> guided_selection;
    std::vector<std::string> random_ablation_selection;
    std::vector<common_flydelta_guidance_arm_measurement> union_arm_results;
    std::string evaluation_id;
    std::string history_fingerprint;
    std::string candidate_pool_fingerprint;
    double baseline_inference_ms = 0.0;
    bool baseline_cost_available = false;
    std::string baseline_evidence_ref;
};

using common_flydelta_guidance_union_runner = std::function<bool(
        const std::string & fixture_group_id,
        const std::vector<std::string> & union_arm_ids,
        std::vector<common_flydelta_guidance_arm_measurement> & results,
        std::string & error)>;

struct common_flydelta_guidance_qualification_result {
    int schema_version = 1;
    common_flydelta_guidance_qualification_context context;
    common_flydelta_guidance_qualification_status status =
        common_flydelta_guidance_qualification_status::insufficient_evidence;
    size_t fixture_groups = 0;
    size_t minimum_fixture_groups = 20;
    size_t baseline_helped = 0;
    size_t guided_helped = 0;
    size_t baseline_harmed = 0;
    size_t guided_harmed = 0;
    size_t random_helped = 0;
    size_t random_harmed = 0;
    size_t baseline_unknown = 0;
    size_t guided_unknown = 0;
    size_t random_unknown = 0;
    bool inference_costs_available = false;
    size_t baseline_selected_arms = 0;
    size_t guided_selected_arms = 0;
    size_t random_selected_arms = 0;
    double baseline_utility_per_selected_arm = 0.0;
    double guided_utility_per_selected_arm = 0.0;
    double random_utility_per_selected_arm = 0.0;
    double guided_minus_baseline_utility = 0.0;
    double guided_minus_baseline_lower_95_approx = 0.0;
    double baseline_inference_ms = 0.0;
    double guided_inference_ms = 0.0;
    double random_inference_ms = 0.0;
    uint64_t evidence_watermark = 0;
    uint64_t revalidate_after_watermark = 0;
    std::string evaluation_fingerprint;
    std::string evaluation_payload_json;
    size_t baseline_improved = 0, guided_improved = 0, random_improved = 0;
    size_t baseline_progress_regressed = 0, guided_progress_regressed = 0, random_progress_regressed = 0;
    size_t baseline_progress_unknown = 0, guided_progress_unknown = 0, random_progress_unknown = 0;
};

// Returns a stable, duplicate-free union in baseline/guided/random order.
bool common_flydelta_guidance_union_selection(
        const std::vector<std::string> & baseline,
        const std::vector<std::string> & guided,
        const std::vector<std::string> & random_ablation,
        std::vector<std::string> & union_ids,
        std::string & error);

// Executes the policy union exactly once for a fixture group. The runner must
// return one measured, host-verified result for every union arm, and no others.
bool common_flydelta_guidance_run_matched_union(
        common_flydelta_guidance_fixture_evaluation & fixture,
        const common_flydelta_guidance_union_runner & runner,
        std::string & error);

// Scores a paired cohort where each fixture group is independent and each
// policy is evaluated from the same actually executed union-arm outcomes.
// Unknown outcomes have zero utility and remain separately unclassified.
bool common_flydelta_guidance_qualify_deep_selection(
        const common_flydelta_guidance_qualification_context & context,
        const std::vector<common_flydelta_guidance_fixture_evaluation> & cohort,
        uint64_t evidence_watermark,
        common_flydelta_guidance_qualification_result & result,
        std::string & error);

bool common_flydelta_guidance_append_qualification(
        common_learning_lifecycle_store & store,
        const common_flydelta_lifecycle_event_context & event_context,
        const common_flydelta_guidance_qualification_result & result,
        std::string & error);

// Persist a real Deep union evaluation and recompute the paired qualification
// from earlier compatible, independent fixture groups. Duplicate executions
// cannot add group weight; failed persistence never changes search results.
bool common_flydelta_guidance_record_matched_evaluation(
        common_learning_lifecycle_store & store,
        const common_flydelta_lifecycle_event_context & event_context,
        const common_flydelta_guidance_qualification_context & context,
        const common_flydelta_guidance_fixture_evaluation & fixture,
        std::string & error,
        common_flydelta_guidance_qualification_result * qualification = nullptr);

// Rebuild from already registered union outcomes, including after an
// evaluation append succeeded but qualification publication was interrupted.
bool common_flydelta_guidance_refresh_qualification(
        common_learning_lifecycle_store & store,
        const common_flydelta_lifecycle_event_context & event_context,
        const common_flydelta_guidance_qualification_context & context,
        std::string & error,
        common_flydelta_guidance_qualification_result * qualification = nullptr);

// Finds the latest compatible qualification. It is usable only before the
// explicit revalidation watermark; expired/paused records never authorize
// selection. The caller must still apply the operator's mode as an upper bound.
bool common_flydelta_guidance_find_qualification(
        const std::vector<common_learning_lifecycle_record> & records,
        const common_flydelta_guidance_qualification_context & context,
        uint64_t current_watermark,
        common_flydelta_guidance_qualification_result & result,
        bool & usable,
        std::string & reason,
        std::string & error);
