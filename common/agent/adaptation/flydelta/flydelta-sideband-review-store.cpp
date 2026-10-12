#include "agent/adaptation/flydelta/flydelta-sideband-review-store.h"

#include <algorithm>
#include <chrono>
#include <map>
#include <nlohmann/json.hpp>
#include <utility>

using json = nlohmann::ordered_json;

namespace {

bool bounded(const std::string & value, size_t max = 512) {
    return !value.empty() && value.size() <= max;
}

common_flydelta_counterfactual_outcome parse_counterfactual_outcome(const std::string & value) {
    if (value == "helped") return common_flydelta_counterfactual_outcome::helped;
    if (value == "neutral") return common_flydelta_counterfactual_outcome::neutral;
    if (value == "harmed") return common_flydelta_counterfactual_outcome::harmed;
    return common_flydelta_counterfactual_outcome::unknown;
}

std::string now_id() {
    return std::to_string(static_cast<unsigned long long>(
        std::chrono::system_clock::now().time_since_epoch().count()));
}

common_learning_lifecycle_status lifecycle_status(common_flydelta_review_action action) {
    switch (action) {
        case common_flydelta_review_action::admit_experimental: return common_learning_lifecycle_status::observed;
        case common_flydelta_review_action::promote_to_candidate: return common_learning_lifecycle_status::eligible;
        case common_flydelta_review_action::approve_canary: return common_learning_lifecycle_status::eligible;
        case common_flydelta_review_action::reject: return common_learning_lifecycle_status::rejected;
        case common_flydelta_review_action::stage_canary: return common_learning_lifecycle_status::canary;
        case common_flydelta_review_action::activate: return common_learning_lifecycle_status::active;
        case common_flydelta_review_action::rollback: return common_learning_lifecycle_status::active;
        case common_flydelta_review_action::close_canary: return common_learning_lifecycle_status::retired;
        case common_flydelta_review_action::reserve_canary_observation: return common_learning_lifecycle_status::canary;
        case common_flydelta_review_action::complete_canary_observation: return common_learning_lifecycle_status::canary;
        case common_flydelta_review_action::attach_canary_evaluation: return common_learning_lifecycle_status::canary;
        case common_flydelta_review_action::retire: return common_learning_lifecycle_status::retired;
        case common_flydelta_review_action::revoke: return common_learning_lifecycle_status::revoked;
    }
    return common_learning_lifecycle_status::failed;
}

json policy_json(const common_flydelta_promotion_policy & policy) {
    return json{
        {"min_trials", policy.min_trials},
        {"min_known_trials", policy.min_known_trials},
        {"min_helped_trials", policy.min_helped_trials},
        {"max_harmed_trials", policy.max_harmed_trials},
        {"min_help_confidence", policy.min_help_confidence},
        {"max_unknown_ratio", policy.max_unknown_ratio},
        {"max_trials", policy.max_trials},
    };
}

void policy_from_json(const json & value, common_flydelta_promotion_policy & policy) {
    policy.min_trials = value.value("min_trials", policy.min_trials);
    policy.min_known_trials = value.value("min_known_trials", policy.min_known_trials);
    policy.min_helped_trials = value.value("min_helped_trials", policy.min_helped_trials);
    policy.max_harmed_trials = value.value("max_harmed_trials", policy.max_harmed_trials);
    policy.min_help_confidence = value.value("min_help_confidence", policy.min_help_confidence);
    policy.max_unknown_ratio = value.value("max_unknown_ratio", policy.max_unknown_ratio);
    policy.max_trials = value.value("max_trials", policy.max_trials);
}

int replay_order(common_flydelta_review_action action) {
    switch (action) {
        case common_flydelta_review_action::admit_experimental: return 0;
        case common_flydelta_review_action::promote_to_candidate: return 1;
        case common_flydelta_review_action::approve_canary: return 2;
        case common_flydelta_review_action::stage_canary: return 3;
        case common_flydelta_review_action::activate: return 4;
        case common_flydelta_review_action::rollback: return 5;
        case common_flydelta_review_action::close_canary: return 6;
        case common_flydelta_review_action::reserve_canary_observation: return 7;
        case common_flydelta_review_action::complete_canary_observation: return 8;
        case common_flydelta_review_action::attach_canary_evaluation: return 9;
        case common_flydelta_review_action::retire: return 10;
        case common_flydelta_review_action::revoke: return 11;
        case common_flydelta_review_action::reject: return 12;
    }
    return 8;
}

json summary_json(const common_flydelta_promotion_summary & summary) {
    return json{
        {"schema_version", summary.schema_version},
        {"id", summary.id},
        {"candidate_id", summary.candidate_id},
        {"baseline_profile_id", summary.baseline_profile_id},
        {"candidate_profile_id", summary.candidate_profile_id},
        {"total_trials", summary.total_trials},
        {"known_trials", summary.known_trials},
        {"helped_trials", summary.helped_trials},
        {"neutral_trials", summary.neutral_trials},
        {"harmed_trials", summary.harmed_trials},
        {"unknown_trials", summary.unknown_trials},
        {"help_confidence", summary.help_confidence},
        {"mean_quality_delta", summary.mean_quality_delta},
        {"status", common_flydelta_candidate_status_name(summary.status)},
    };
}

json canary_progress_json(const common_flydelta_semantic_progress & progress) {
    return {{"outcome", common_flydelta_semantic_progress_outcome_name(progress.outcome)},
        {"comparable", progress.comparable}, {"baseline_score", progress.baseline_score},
        {"candidate_score", progress.candidate_score},
        {"improved_dimensions", progress.improved_dimensions},
        {"regressed_dimensions", progress.regressed_dimensions},
        {"residual_dimensions", progress.residual_dimensions}};
}

void canary_progress_from_json(const json & value, common_flydelta_semantic_progress & progress) {
    progress = {};
    const auto outcome = value.value("outcome", "unknown");
    if (outcome == "improved") progress.outcome = common_flydelta_semantic_progress_outcome::improved;
    else if (outcome == "solved") progress.outcome = common_flydelta_semantic_progress_outcome::solved;
    else if (outcome == "unchanged") progress.outcome = common_flydelta_semantic_progress_outcome::unchanged;
    else if (outcome == "regressed") progress.outcome = common_flydelta_semantic_progress_outcome::regressed;
    progress.comparable = value.value("comparable", false);
    progress.baseline_score = value.value("baseline_score", 0);
    progress.candidate_score = value.value("candidate_score", 0);
    progress.improved_dimensions = value.value("improved_dimensions", std::vector<std::string>{});
    progress.regressed_dimensions = value.value("regressed_dimensions", std::vector<std::string>{});
    progress.residual_dimensions = value.value("residual_dimensions", std::vector<std::string>{});
}

bool summary_from_json(const json & value,
        common_flydelta_promotion_summary & summary, std::string & error) {
    summary = {};
    summary.schema_version = value.value("schema_version", 0);
    summary.id = value.value("id", "");
    summary.candidate_id = value.value("candidate_id", "");
    summary.baseline_profile_id = value.value("baseline_profile_id", "");
    summary.candidate_profile_id = value.value("candidate_profile_id", "");
    summary.total_trials = value.value("total_trials", 0U);
    summary.known_trials = value.value("known_trials", 0U);
    summary.helped_trials = value.value("helped_trials", 0U);
    summary.neutral_trials = value.value("neutral_trials", 0U);
    summary.harmed_trials = value.value("harmed_trials", 0U);
    summary.unknown_trials = value.value("unknown_trials", 0U);
    summary.help_confidence = value.value("help_confidence", 0.0f);
    summary.mean_quality_delta = value.value("mean_quality_delta", 0.0f);
    const auto status = value.value("status", "observed");
    if (status == "observed") summary.status = common_flydelta_candidate_status::observed;
    else if (status == "eligible") summary.status = common_flydelta_candidate_status::eligible;
    else if (status == "approved") summary.status = common_flydelta_candidate_status::approved;
    else if (status == "rejected") summary.status = common_flydelta_candidate_status::rejected;
    else if (status == "revoked") summary.status = common_flydelta_candidate_status::revoked;
    else { error = "unknown FlyDelta promotion summary status"; return false; }
    return true;
}

} // namespace

const char * common_flydelta_review_source_name(common_flydelta_review_source source) {
    switch (source) {
        case common_flydelta_review_source::operator_action: return "operator";
        case common_flydelta_review_source::host_automation: return "host_automation";
    }
    return "operator";
}

const char * common_flydelta_review_action_name(common_flydelta_review_action action) {
    switch (action) {
        case common_flydelta_review_action::admit_experimental: return "admit_experimental";
        case common_flydelta_review_action::promote_to_candidate: return "promote_to_candidate";
        case common_flydelta_review_action::approve_canary: return "approve_canary";
        case common_flydelta_review_action::reject: return "reject";
        case common_flydelta_review_action::stage_canary: return "stage_canary";
        case common_flydelta_review_action::activate: return "activate";
        case common_flydelta_review_action::rollback: return "rollback";
        case common_flydelta_review_action::close_canary: return "close_canary";
        case common_flydelta_review_action::reserve_canary_observation: return "reserve_canary_observation";
        case common_flydelta_review_action::complete_canary_observation: return "complete_canary_observation";
        case common_flydelta_review_action::attach_canary_evaluation: return "attach_canary_evaluation";
        case common_flydelta_review_action::retire: return "retire";
        case common_flydelta_review_action::revoke: return "revoke";
    }
    return "admit_experimental";
}

bool parse_common_flydelta_review_source(
        const std::string & value, common_flydelta_review_source & source, std::string & error) {
    if (value == "operator") source = common_flydelta_review_source::operator_action;
    else if (value == "host_automation") source = common_flydelta_review_source::host_automation;
    else { error = "unknown FlyDelta review source"; return false; }
    return true;
}

bool parse_common_flydelta_review_action(
        const std::string & value, common_flydelta_review_action & action, std::string & error) {
    if (value == "admit_experimental") action = common_flydelta_review_action::admit_experimental;
    else if (value == "promote_to_candidate") action = common_flydelta_review_action::promote_to_candidate;
    else if (value == "approve_canary") action = common_flydelta_review_action::approve_canary;
    else if (value == "reject") action = common_flydelta_review_action::reject;
    else if (value == "stage_canary") action = common_flydelta_review_action::stage_canary;
    else if (value == "activate") action = common_flydelta_review_action::activate;
    else if (value == "rollback") action = common_flydelta_review_action::rollback;
    else if (value == "close_canary") action = common_flydelta_review_action::close_canary;
    else if (value == "reserve_canary_observation") action = common_flydelta_review_action::reserve_canary_observation;
    else if (value == "complete_canary_observation") action = common_flydelta_review_action::complete_canary_observation;
    else if (value == "attach_canary_evaluation") action = common_flydelta_review_action::attach_canary_evaluation;
    else if (value == "retire") action = common_flydelta_review_action::retire;
    else if (value == "revoke") action = common_flydelta_review_action::revoke;
    else { error = "unknown FlyDelta review action"; return false; }
    return true;
}

bool common_flydelta_sideband_review_validate(
        const common_flydelta_sideband_review & review, std::string & error) {
    error.clear();
    if (review.schema_version != 1 || !bounded(review.event_id) || !bounded(review.actor_id) ||
            !bounded(review.manifest.id) || !common_flydelta_sideband_manifest_validate(review.manifest, error)) {
        if (error.empty()) error = "FlyDelta sideband review identity is invalid";
        return false;
    }
    if ((review.action == common_flydelta_review_action::stage_canary ||
            review.action == common_flydelta_review_action::approve_canary ||
            review.action == common_flydelta_review_action::promote_to_candidate) &&
            !bounded(review.evaluation_revision)) {
        error = "FlyDelta sideband review requires an evaluation revision";
        return false;
    }
    if ((review.action == common_flydelta_review_action::approve_canary ||
            review.action == common_flydelta_review_action::reject) &&
            !bounded(review.reason)) {
        error = "FlyDelta review decision requires a reason";
        return false;
    }
    if (review.action == common_flydelta_review_action::approve_canary &&
            !review.has_canary_admission &&
            (!bounded(review.promotion_summary_ref) ||
             !bounded(review.evaluation_report_ref))) {
        error = "FlyDelta canary approval requires durable report references";
        return false;
    }
    if (review.action == common_flydelta_review_action::revoke && !bounded(review.reason)) {
        error = "FlyDelta sideband revoke review requires a reason";
        return false;
    }
    if (review.action == common_flydelta_review_action::activate ||
            review.action == common_flydelta_review_action::rollback) {
        if (!bounded(review.manifest.id) ||
                (!review.binding_key.empty() && !bounded(review.binding_key)) ||
                (review.action == common_flydelta_review_action::rollback &&
                 (review.manifest.status != common_flydelta_sideband_status::active ||
                  !bounded(review.binding_key)))) {
            error = "FlyDelta activation binding review is incomplete";
            return false;
        }
        if (!review.manifest.binding_key.empty() &&
                review.manifest.binding_key != review.binding_key) {
            error = "FlyDelta activation binding does not match revision provenance";
            return false;
        }
        if (!review.expected_current_revision_id.empty() &&
                review.binding_key.empty()) {
            error = "FlyDelta activation binding expected revision lacks a binding key";
            return false;
        }
        if (!review.expected_current_revision_id.empty() &&
                !bounded(review.expected_current_revision_id)) {
            error = "FlyDelta activation binding expected revision is invalid";
            return false;
        }
    }
    if (review.action == common_flydelta_review_action::stage_canary ||
            review.action == common_flydelta_review_action::approve_canary) {
        const bool stage_manifest = review.action == common_flydelta_review_action::stage_canary;
        const bool promotion_admitted = review.has_promotion_evidence &&
            common_flydelta_promotion_summary_validate(
                review.promotion_summary, review.promotion_policy, error) &&
            review.promotion_summary.status == common_flydelta_candidate_status::eligible &&
            common_flydelta_evaluation_report_validate(review.evaluation, error) &&
            review.evaluation.status == "passed" &&
            review.promotion_summary.candidate_id == review.manifest.id &&
            review.evaluation.candidate_id == review.manifest.id &&
            review.evaluation.revision_id == review.evaluation_revision;
        error.clear();
        const bool progress_admitted = review.has_canary_admission &&
            review.manifest.generalization.level == common_flydelta_generalization_level::model &&
            review.canary_progress.comparable &&
            (review.canary_progress.outcome == common_flydelta_semantic_progress_outcome::improved ||
             review.canary_progress.outcome == common_flydelta_semantic_progress_outcome::solved) &&
            review.canary_progress.regressed_dimensions.empty();
        if ((!stage_manifest && review.manifest.status != common_flydelta_sideband_status::candidate) ||
                (stage_manifest && review.manifest.status != common_flydelta_sideband_status::candidate &&
                 review.manifest.status != common_flydelta_sideband_status::canary) ||
                (!promotion_admitted && !progress_admitted)) {
            if (error.empty()) error = "FlyDelta canary review lacks valid promotion evidence";
            return false;
        }
    }
    if (review.action == common_flydelta_review_action::stage_canary) {
        if (!review.has_canary_envelope || review.canary_envelope_event_id != review.event_id ||
                !common_flydelta_canary_envelope_validate(review.canary_envelope, error) ||
                review.canary_envelope.candidate_revision_id != review.manifest.id ||
                review.canary_envelope.binding_key != review.manifest.binding_key ||
                review.canary_envelope.behavior_key != review.manifest.applicability.behavior_key ||
                review.canary_envelope.scope_fingerprint != review.manifest.applicability.scope_fingerprint) {
            if (error.empty()) error = "FlyDelta canary review envelope is invalid";
            return false;
        }
    }
    if (review.action == common_flydelta_review_action::close_canary &&
            (!bounded(review.canary_envelope_event_id) || !bounded(review.reason))) {
        error = "FlyDelta canary close requires an envelope id and reason";
        return false;
    }
    if (review.action == common_flydelta_review_action::reserve_canary_observation &&
            (!bounded(review.canary_envelope_event_id) || !bounded(review.observation_id))) {
        error = "FlyDelta canary observation reservation requires an envelope id";
        return false;
    }
    if (review.action == common_flydelta_review_action::complete_canary_observation &&
            (!bounded(review.canary_envelope_event_id) || !bounded(review.observation_id) ||
             !bounded(review.allocation_id) || !bounded(review.scope_step_id))) {
        error = "FlyDelta canary observation completion is incomplete";
        return false;
    }
    if (review.action == common_flydelta_review_action::attach_canary_evaluation &&
            (!bounded(review.canary_envelope_event_id) || !bounded(review.observation_id) ||
             !bounded(review.evaluation_report_ref) || !bounded(review.counterfactual_report_ref) ||
             review.observation_status != common_flydelta_canary_observation_status::evaluated)) {
        error = "FlyDelta canary evaluation attachment is incomplete";
        return false;
    }
    if (review.action == common_flydelta_review_action::complete_canary_observation &&
            review.observation_status == common_flydelta_canary_observation_status::evaluated &&
            !bounded(review.evaluation_report_ref)) {
        error = "FlyDelta evaluated observation requires an evaluation report";
        return false;
    }
    return true;
}

std::string common_flydelta_sideband_review_to_json(
        const common_flydelta_sideband_review & review) {
    return json{
        {"schema_version", review.schema_version},
        {"kind", "flydelta-sideband-review"},
        {"event_id", review.event_id},
        {"actor_id", review.actor_id},
        {"source", common_flydelta_review_source_name(review.source)},
        {"action", common_flydelta_review_action_name(review.action)},
        {"policy_revision", review.policy_revision},
        {"evaluation_revision", review.evaluation_revision},
        {"promotion_summary_ref", review.promotion_summary_ref},
        {"evaluation_report_ref", review.evaluation_report_ref},
        {"binding_key", review.binding_key},
        {"expected_current_revision_id", review.expected_current_revision_id},
        {"has_canary_envelope", review.has_canary_envelope},
        {"has_canary_admission", review.has_canary_admission},
        {"canary_progress", canary_progress_json(review.canary_progress)},
        {"canary_envelope_event_id", review.canary_envelope_event_id},
        {"observation_id", review.observation_id},
        {"allocation_id", review.allocation_id},
        {"scope_step_id", review.scope_step_id},
        {"observation_status", common_flydelta_canary_observation_status_name(review.observation_status)},
        {"canary_outcome", common_flydelta_counterfactual_outcome_name(review.canary_outcome)},
        {"counterfactual_report_ref", review.counterfactual_report_ref},
        {"target_gain", review.target_gain},
        {"control_regression", review.control_regression},
        {"competitor_regression", review.competitor_regression},
        {"canary_envelope", {
            {"schema_version", review.canary_envelope.schema_version},
            {"binding_key", review.canary_envelope.binding_key},
            {"candidate_revision_id", review.canary_envelope.candidate_revision_id},
            {"behavior_key", review.canary_envelope.behavior_key},
            {"scope_fingerprint", review.canary_envelope.scope_fingerprint},
            {"traffic_basis_points", review.canary_envelope.traffic_basis_points},
            {"expires_at_epoch_ms", review.canary_envelope.expires_at_epoch_ms},
            {"max_observations", review.canary_envelope.max_observations},
            {"max_scale", review.canary_envelope.max_scale},
            {"compatibility", {
                {"base_model_id", review.canary_envelope.compatibility.base_model_id},
                {"base_model_fingerprint", review.canary_envelope.compatibility.base_model_fingerprint},
                {"tokenizer_fingerprint", review.canary_envelope.compatibility.tokenizer_fingerprint},
                {"template_fingerprint", review.canary_envelope.compatibility.template_fingerprint},
                {"architecture", review.canary_envelope.compatibility.architecture},
                {"inference_layout_revision", review.canary_envelope.compatibility.inference_layout_revision},
            }},
            {"oracle_revision", review.canary_envelope.oracle_revision},
            {"policy_revision", review.canary_envelope.policy_revision},
            {"baseline_deployment_fingerprint", review.canary_envelope.baseline_deployment_fingerprint},
            {"rollback_revision_id", review.canary_envelope.rollback_revision_id},
        }},
        {"reason", review.reason},
        {"manifest", json::parse(common_flydelta_sideband_manifest_to_json(review.manifest))},
        {"has_promotion_evidence", review.has_promotion_evidence},
        {"promotion_policy", policy_json(review.promotion_policy)},
        {"promotion_summary", summary_json(review.promotion_summary)},
        {"evaluation", json::parse(common_flydelta_evaluation_report_to_json(review.evaluation))},
    }.dump();
}

bool common_flydelta_sideband_review_from_json(
        const std::string & text, common_flydelta_sideband_review & review, std::string & error) {
    error.clear();
    try {
        const auto value = json::parse(text);
        if (!value.is_object() || value.value("kind", "") != "flydelta-sideband-review" ||
                !value.contains("manifest") || !value["manifest"].is_object()) {
            error = "FlyDelta sideband review JSON is invalid";
            return false;
        }
        review = {};
        review.schema_version = value.value("schema_version", 0);
        review.event_id = value.value("event_id", "");
        review.actor_id = value.value("actor_id", "");
        review.policy_revision = value.value("policy_revision", "");
        review.evaluation_revision = value.value("evaluation_revision", "");
        review.promotion_summary_ref = value.value("promotion_summary_ref", "");
        review.evaluation_report_ref = value.value("evaluation_report_ref", "");
        review.binding_key = value.value("binding_key", "");
        review.expected_current_revision_id = value.value("expected_current_revision_id", "");
        review.has_canary_envelope = value.value("has_canary_envelope", false);
        review.has_canary_admission = value.value("has_canary_admission", false);
        canary_progress_from_json(value.value("canary_progress", json::object()), review.canary_progress);
        review.canary_envelope_event_id = value.value("canary_envelope_event_id", "");
        review.observation_id = value.value("observation_id", "");
        review.allocation_id = value.value("allocation_id", "");
        review.scope_step_id = value.value("scope_step_id", "");
        if (!parse_common_flydelta_canary_observation_status(
                value.value("observation_status", "generation_failed"),
                review.observation_status, error)) return false;
        review.canary_outcome = parse_counterfactual_outcome(value.value("canary_outcome", "unknown"));
        review.counterfactual_report_ref = value.value("counterfactual_report_ref", "");
        review.target_gain = value.value("target_gain", 0.0f);
        review.control_regression = value.value("control_regression", 0.0f);
        review.competitor_regression = value.value("competitor_regression", 0.0f);
        const auto envelope = value.value("canary_envelope", json::object());
        review.canary_envelope.schema_version = envelope.value("schema_version", 1);
        review.canary_envelope.binding_key = envelope.value("binding_key", "");
        review.canary_envelope.candidate_revision_id = envelope.value("candidate_revision_id", "");
        review.canary_envelope.behavior_key = envelope.value("behavior_key", "");
        review.canary_envelope.scope_fingerprint = envelope.value("scope_fingerprint", "");
        review.canary_envelope.traffic_basis_points = envelope.value("traffic_basis_points", 0U);
        review.canary_envelope.expires_at_epoch_ms = envelope.value("expires_at_epoch_ms", 0ULL);
        review.canary_envelope.max_observations = envelope.value("max_observations", 0U);
        review.canary_envelope.max_scale = envelope.value("max_scale", 0.0f);
        const auto envelope_compatibility = envelope.value("compatibility", json::object());
        review.canary_envelope.compatibility.base_model_id = envelope_compatibility.value("base_model_id", "");
        review.canary_envelope.compatibility.base_model_fingerprint = envelope_compatibility.value("base_model_fingerprint", "");
        review.canary_envelope.compatibility.tokenizer_fingerprint = envelope_compatibility.value("tokenizer_fingerprint", "");
        review.canary_envelope.compatibility.template_fingerprint = envelope_compatibility.value("template_fingerprint", "");
        review.canary_envelope.compatibility.architecture = envelope_compatibility.value("architecture", "");
        review.canary_envelope.compatibility.inference_layout_revision = envelope_compatibility.value("inference_layout_revision", "");
        review.canary_envelope.oracle_revision = envelope.value("oracle_revision", "");
        review.canary_envelope.policy_revision = envelope.value("policy_revision", "");
        review.canary_envelope.baseline_deployment_fingerprint = envelope.value("baseline_deployment_fingerprint", "");
        review.canary_envelope.rollback_revision_id = envelope.value("rollback_revision_id", "");
        review.reason = value.value("reason", "");
        if (!parse_common_flydelta_review_source(value.value("source", "operator"), review.source, error) ||
                !parse_common_flydelta_review_action(value.value("action", ""), review.action, error) ||
                !common_flydelta_sideband_manifest_from_json(value["manifest"].dump(), review.manifest, error)) return false;
        review.has_promotion_evidence = value.value("has_promotion_evidence", false);
        if (review.has_promotion_evidence) {
            policy_from_json(value.value("promotion_policy", json::object()), review.promotion_policy);
            if (!summary_from_json(value.value("promotion_summary", json::object()),
                    review.promotion_summary, error) ||
                    !common_flydelta_evaluation_report_from_json(
                        value.value("evaluation", json::object()).dump(), review.evaluation, error)) return false;
        }
    } catch (const std::exception & exception) {
        error = std::string("invalid FlyDelta sideband review JSON: ") + exception.what();
        return false;
    }
    return common_flydelta_sideband_review_validate(review, error);
}

bool common_flydelta_sideband_review_store::append(
        const common_flydelta_sideband_review & review, std::string & error) {
    if (!common_flydelta_sideband_review_validate(review, error)) return false;
    if (review.action == common_flydelta_review_action::complete_canary_observation ||
            review.action == common_flydelta_review_action::attach_canary_evaluation) {
        std::string existing_error;
        const auto existing_reviews = list(existing_error);
        if (!existing_error.empty()) {
            error = existing_error;
            return false;
        }
        bool has_reservation = false;
        bool has_evaluated_completion = false;
        for (const auto & existing : existing_reviews) {
            if (existing.observation_id != review.observation_id ||
                    existing.canary_envelope_event_id != review.canary_envelope_event_id) continue;
            if (existing.action == common_flydelta_review_action::reserve_canary_observation) {
                has_reservation = true;
            }
            if (existing.action == common_flydelta_review_action::complete_canary_observation &&
                    existing.observation_status == common_flydelta_canary_observation_status::evaluated) {
                has_evaluated_completion = true;
            }
            const bool same_terminal = existing.action == review.action;
            if (same_terminal && existing.event_id != review.event_id) {
                error = "FlyDelta canary observation terminal state already exists";
                return false;
            }
        }
        if (!has_reservation) {
            error = "FlyDelta canary observation has no reservation";
            return false;
        }
        if (review.action == common_flydelta_review_action::attach_canary_evaluation &&
                !has_evaluated_completion) {
            error = "FlyDelta canary evaluation has no evaluated terminal observation";
            return false;
        }
    }
    common_learning_lifecycle_record record;
    record.event_id = review.event_id;
    record.subject_id = review.manifest.id;
    record.kind = common_learning_lifecycle_kind::flydelta_result;
    record.status = lifecycle_status(review.action);
    record.idempotency_key = "flydelta-review:" + review.event_id;
    record.source_id = review.actor_id;
    record.namespace_id = review.manifest.namespace_id;
    record.project_id = review.manifest.project_id;
    record.content_hash = review.manifest.artifact_hash;
    record.created_at = now_id();
    record.payload_json = common_flydelta_sideband_review_to_json(review);
    return journal_.append(record, error);
}

bool common_flydelta_sideband_review_store::apply(
        common_flydelta_sideband_registry & registry,
        const common_flydelta_sideband_review & review,
        bool explicit_host_approval,
        std::string & error) const {
    switch (review.action) {
        case common_flydelta_review_action::admit_experimental:
            return registry.admit_experimental(review.manifest, error);
        case common_flydelta_review_action::promote_to_candidate:
            return registry.promote_experimental(review.manifest.id, review.evaluation_revision,
                error, explicit_host_approval);
        case common_flydelta_review_action::approve_canary:
            // Approval is a durable decision record. It intentionally does
            // not mutate the registry; stage_canary applies the existing
            // controller only after this record exists.
            error.clear();
            return true;
        case common_flydelta_review_action::reject:
            error.clear();
            return true;
        case common_flydelta_review_action::stage_canary:
            if (!explicit_host_approval) {
                error = "FlyDelta canary staging requires explicit host approval";
                return false;
            }
            if (const auto it = registry.list().find(review.manifest.id);
                    it != registry.list().end() &&
                    it->second.status == common_flydelta_sideband_status::canary &&
                    it->second.evaluation_revision == review.evaluation_revision) {
                error.clear();
                return true;
            }
            return registry.stage_canary(review.manifest.id, review.evaluation_revision, error,
                review.has_canary_admission && !review.has_promotion_evidence);
        case common_flydelta_review_action::activate:
            if (!explicit_host_approval) {
                error = "FlyDelta activation requires explicit host approval";
                return false;
            }
            if (!registry.activate(review.manifest.id, error)) return false;
            return review.binding_key.empty() || registry.bind_revision(
                review.binding_key, review.manifest.id,
                review.expected_current_revision_id, error);
        case common_flydelta_review_action::rollback:
            if (!explicit_host_approval) {
                error = "FlyDelta rollback requires explicit host approval";
                return false;
            }
            return registry.bind_revision(
                review.binding_key, review.manifest.id,
                review.expected_current_revision_id, error);
        case common_flydelta_review_action::close_canary:
            // Closing is journal-only: it removes a temporary request override
            // without changing an immutable candidate or active binding.
            error.clear();
            return true;
        case common_flydelta_review_action::reserve_canary_observation:
            // Exposure reservations are durable journal facts only. They do
            // not mutate registry lifecycle or open-canary state during
            // replay.
            error.clear();
            return true;
        case common_flydelta_review_action::complete_canary_observation:
        case common_flydelta_review_action::attach_canary_evaluation:
            // Completion and report attachment are journal projections only;
            // registry lifecycle remains unchanged until policy disposition.
            error.clear();
            return true;
        case common_flydelta_review_action::retire:
            return registry.retire(review.manifest.id, error);
        case common_flydelta_review_action::revoke:
            return registry.revoke(review.manifest.id, review.reason, error);
    }
    error = "unsupported FlyDelta review action";
    return false;
}

bool common_flydelta_sideband_review_store::apply_and_append(
        common_flydelta_sideband_registry & registry,
        const common_flydelta_sideband_review & review,
        bool explicit_host_approval,
        std::string & error) {
    if (!common_flydelta_sideband_review_validate(review, error)) return false;
    const auto existing = list(error);
    if (!error.empty()) return false;
    for (const auto & item : existing) {
        if (item.event_id == review.event_id) {
            if (common_flydelta_sideband_review_to_json(item) ==
                    common_flydelta_sideband_review_to_json(review)) return true;
            error = "FlyDelta review event id conflicts with existing record";
            return false;
        }
    }
    if (review.action == common_flydelta_review_action::stage_canary) {
        for (const auto & item : open_canaries(error)) {
            if (!error.empty()) return false;
            if (!item.has_canary_envelope) continue;
            const auto & left = item.canary_envelope;
            const auto & right = review.canary_envelope;
            if (left.binding_key == right.binding_key &&
                    left.behavior_key == right.behavior_key &&
                    left.scope_fingerprint == right.scope_fingerprint &&
                    item.event_id != review.event_id) {
                error = "FlyDelta canary conflicts with an open replacement for the same binding, behavior and scope";
                return false;
            }
        }
    }
    // Apply to a copy first. If persistence fails, the live registry must not
    // advance without a durable journal event.
    auto next = registry;
    if (!apply(next, review, explicit_host_approval, error) || !append(review, error)) return false;
    registry = std::move(next);
    return true;
}

std::vector<common_flydelta_sideband_review> common_flydelta_sideband_review_store::list(
        std::string & error) const {
    error.clear();
    std::vector<common_flydelta_sideband_review> result;
    for (const auto & record : journal_.list(error)) {
        if (!error.empty()) return {};
        if (record.kind != common_learning_lifecycle_kind::flydelta_result) continue;
        const auto payload = json::parse(record.payload_json, nullptr, false);
        if (!payload.is_object() || payload.value("kind", "") != "flydelta-sideband-review") continue;
        common_flydelta_sideband_review review;
        if (!common_flydelta_sideband_review_from_json(record.payload_json, review, error)) return {};
        result.push_back(std::move(review));
    }
    return result;
}

std::vector<common_flydelta_sideband_review> common_flydelta_sideband_review_store::open_canaries(
        std::string & error) const {
    auto reviews = list(error);
    if (!error.empty()) return {};
    std::map<std::string, common_flydelta_sideband_review> open;
    for (const auto & review : reviews) {
        if (review.action == common_flydelta_review_action::stage_canary && review.has_canary_envelope) {
            open[review.event_id] = review;
        } else if (review.action == common_flydelta_review_action::close_canary) {
            open.erase(review.canary_envelope_event_id);
        }
    }
    std::vector<common_flydelta_sideband_review> result;
    for (auto & item : open) result.push_back(std::move(item.second));
    return result;
}

bool common_flydelta_sideband_review_store::replay(
        common_flydelta_sideband_registry & registry, std::string & error) const {
    error.clear();
    const auto reviews = list(error);
    if (!error.empty()) return false;
    auto is_binding_event = [](const common_flydelta_sideband_review & review) {
        return review.action == common_flydelta_review_action::rollback ||
            (review.action == common_flydelta_review_action::activate &&
             !review.binding_key.empty());
    };
    std::vector<common_flydelta_sideband_review> preparation;
    std::vector<common_flydelta_sideband_review> bindings;
    std::vector<common_flydelta_sideband_review> retirement;
    for (const auto & review : reviews) {
        if (is_binding_event(review)) {
            bindings.push_back(review);
        } else if (review.action == common_flydelta_review_action::retire ||
                review.action == common_flydelta_review_action::revoke ||
                review.action == common_flydelta_review_action::reject) {
            retirement.push_back(review);
        } else {
            preparation.push_back(review);
        }
    }
    auto order_reviews = [](auto & ordered) {
        std::stable_sort(ordered.begin(), ordered.end(), [](const auto & left, const auto & right) {
        const int left_order = replay_order(left.action);
        const int right_order = replay_order(right.action);
        return left_order != right_order
            ? left_order < right_order
            : left.event_id < right.event_id;
        });
    };
    order_reviews(preparation);
    order_reviews(retirement);
    for (const auto & review : preparation) {
        if (!apply(registry, review, true, error)) return false;
    }
    // Binding events retain append order. A binding is a selection history,
    // so sorting these records by event id could replay A -> B as B -> A.
    for (const auto & review : bindings) {
        if (!apply(registry, review, true, error)) return false;
    }
    for (const auto & review : retirement) {
        if (!error.empty() || !apply(registry, review, true, error)) return false;
    }
    return true;
}
