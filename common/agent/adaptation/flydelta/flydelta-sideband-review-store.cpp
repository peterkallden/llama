#include "agent/adaptation/flydelta/flydelta-sideband-review-store.h"

#include <algorithm>
#include <chrono>
#include <nlohmann/json.hpp>
#include <utility>

using json = nlohmann::ordered_json;

namespace {

bool bounded(const std::string & value, size_t max = 512) {
    return !value.empty() && value.size() <= max;
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
        case common_flydelta_review_action::retire: return 5;
        case common_flydelta_review_action::revoke: return 6;
        case common_flydelta_review_action::reject: return 7;
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
            (!bounded(review.promotion_summary_ref) ||
             !bounded(review.evaluation_report_ref))) {
        error = "FlyDelta canary approval requires durable report references";
        return false;
    }
    if (review.action == common_flydelta_review_action::revoke && !bounded(review.reason)) {
        error = "FlyDelta sideband revoke review requires a reason";
        return false;
    }
    if (review.action == common_flydelta_review_action::stage_canary ||
            review.action == common_flydelta_review_action::approve_canary) {
        const bool stage_manifest = review.action == common_flydelta_review_action::stage_canary;
        if ((!stage_manifest && review.manifest.status != common_flydelta_sideband_status::candidate) ||
                (stage_manifest && review.manifest.status != common_flydelta_sideband_status::candidate &&
                 review.manifest.status != common_flydelta_sideband_status::canary) ||
                !review.has_promotion_evidence ||
                !common_flydelta_promotion_summary_validate(
                    review.promotion_summary, review.promotion_policy, error) ||
                review.promotion_summary.status != common_flydelta_candidate_status::eligible ||
                !common_flydelta_evaluation_report_validate(review.evaluation, error) ||
                review.evaluation.status != "passed" ||
                review.promotion_summary.candidate_id != review.manifest.id ||
                review.evaluation.candidate_id != review.manifest.id ||
                review.evaluation.revision_id != review.evaluation_revision) {
            if (error.empty()) error = "FlyDelta canary review lacks valid promotion evidence";
            return false;
        }
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
            return registry.stage_canary(review.manifest.id, review.evaluation_revision, error);
        case common_flydelta_review_action::activate:
            if (!explicit_host_approval) {
                error = "FlyDelta activation requires explicit host approval";
                return false;
            }
            return registry.activate(review.manifest.id, error);
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

bool common_flydelta_sideband_review_store::replay(
        common_flydelta_sideband_registry & registry, std::string & error) const {
    error.clear();
    auto reviews = list(error);
    std::stable_sort(reviews.begin(), reviews.end(), [](const auto & left, const auto & right) {
        const int left_order = replay_order(left.action);
        const int right_order = replay_order(right.action);
        return left_order != right_order
            ? left_order < right_order
            : left.event_id < right.event_id;
    });
    for (const auto & review : reviews) {
        if (!error.empty() || !apply(registry, review, true, error)) return false;
    }
    return true;
}
