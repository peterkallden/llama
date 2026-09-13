#include "agent/adaptation/flydelta/flydelta-promotion.h"

#include <cmath>

namespace {

bool nonempty_bounded(const std::string & value, size_t max_size = 512) {
    return !value.empty() && value.size() <= max_size;
}

bool unit(float value) {
    return std::isfinite(value) && value >= 0.0f && value <= 1.0f;
}

} // namespace

bool common_flydelta_promotion_policy_validate(
        const common_flydelta_promotion_policy & policy,
        std::string & error) {
    error.clear();
    if (policy.min_trials == 0 || policy.max_trials < policy.min_trials ||
            policy.min_known_trials > policy.max_trials || policy.min_helped_trials > policy.max_trials ||
            policy.max_harmed_trials > policy.max_trials || !unit(policy.min_help_confidence) ||
            !unit(policy.max_unknown_ratio)) {
        error = "FlyDelta promotion policy is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_promotion_summary_validate(
        const common_flydelta_promotion_summary & summary,
        const common_flydelta_promotion_policy & policy,
        std::string & error) {
    error.clear();
    if (!common_flydelta_promotion_policy_validate(policy, error)) return false;
    if (summary.schema_version != 1 || !nonempty_bounded(summary.id) ||
            !nonempty_bounded(summary.candidate_id) || !nonempty_bounded(summary.baseline_profile_id) ||
            !nonempty_bounded(summary.candidate_profile_id) ||
            summary.baseline_profile_id == summary.candidate_profile_id ||
            summary.total_trials > policy.max_trials ||
            summary.known_trials + summary.unknown_trials != summary.total_trials ||
            summary.helped_trials + summary.neutral_trials + summary.harmed_trials != summary.known_trials ||
            !unit(summary.help_confidence) || !std::isfinite(summary.mean_quality_delta) ||
            summary.mean_quality_delta < -1.0f || summary.mean_quality_delta > 1.0f ||
            summary.harmed_trials > summary.known_trials) {
        error = "FlyDelta promotion summary is inconsistent";
        return false;
    }
    if (summary.status == common_flydelta_candidate_status::approved &&
            summary.status != common_flydelta_candidate_status::eligible) {
        // Keep the approval state explicit, but do not make it possible to
        // construct an approved summary from a failed qualification pass.
        if (summary.total_trials < policy.min_trials || summary.known_trials < policy.min_known_trials ||
                summary.helped_trials < policy.min_helped_trials ||
                summary.harmed_trials > policy.max_harmed_trials ||
                summary.help_confidence < policy.min_help_confidence || summary.total_trials == 0 ||
                static_cast<float>(summary.unknown_trials) / static_cast<float>(summary.total_trials) > policy.max_unknown_ratio) {
            error = "FlyDelta approved summary does not meet promotion policy";
            return false;
        }
    }
    if (summary.status == common_flydelta_candidate_status::eligible ||
            summary.status == common_flydelta_candidate_status::approved) {
        if (summary.total_trials < policy.min_trials || summary.known_trials < policy.min_known_trials ||
                summary.helped_trials < policy.min_helped_trials ||
                summary.harmed_trials > policy.max_harmed_trials ||
                summary.help_confidence < policy.min_help_confidence || summary.total_trials == 0 ||
                static_cast<float>(summary.unknown_trials) / static_cast<float>(summary.total_trials) > policy.max_unknown_ratio) {
            error = "FlyDelta eligible summary does not meet promotion policy";
            return false;
        }
    }
    return true;
}

bool common_flydelta_promotion_summary_from_reports(
        const std::string & summary_id,
        const std::vector<common_flydelta_counterfactual_report> & reports,
        const common_flydelta_promotion_policy & policy,
        common_flydelta_promotion_summary & summary,
        std::string & error) {
    error.clear();
    if (!common_flydelta_promotion_policy_validate(policy, error) ||
            !nonempty_bounded(summary_id) || reports.empty() || reports.size() > policy.max_trials) {
        if (error.empty()) error = "FlyDelta promotion input bound is invalid";
        return false;
    }
    const auto & first = reports.front();
    if (!common_flydelta_counterfactual_report_validate(first, error)) return false;
    summary = {};
    summary.id = summary_id;
    summary.candidate_id = first.candidate_id;
    summary.baseline_profile_id = first.baseline_profile_id;
    summary.candidate_profile_id = first.candidate_profile_id;
    float quality_sum = 0.0f;
    for (const auto & report : reports) {
        if (!common_flydelta_counterfactual_report_validate(report, error) ||
                report.candidate_id != summary.candidate_id ||
                report.baseline_profile_id != summary.baseline_profile_id ||
                report.candidate_profile_id != summary.candidate_profile_id) {
            error = "FlyDelta promotion reports are incompatible";
            return false;
        }
        ++summary.total_trials;
        quality_sum += report.quality_delta;
        switch (report.outcome) {
            case common_flydelta_counterfactual_outcome::helped: ++summary.helped_trials; break;
            case common_flydelta_counterfactual_outcome::neutral: ++summary.neutral_trials; break;
            case common_flydelta_counterfactual_outcome::harmed: ++summary.harmed_trials; break;
            case common_flydelta_counterfactual_outcome::unknown: ++summary.unknown_trials; break;
        }
    }
    summary.known_trials = summary.helped_trials + summary.neutral_trials + summary.harmed_trials;
    summary.help_confidence = summary.known_trials == 0 ? 0.0f :
        static_cast<float>(summary.helped_trials) / static_cast<float>(summary.known_trials);
    summary.mean_quality_delta = quality_sum / static_cast<float>(summary.total_trials);
    const float unknown_ratio = static_cast<float>(summary.unknown_trials) /
        static_cast<float>(summary.total_trials);
    if (summary.total_trials >= policy.min_trials && summary.known_trials >= policy.min_known_trials &&
            summary.helped_trials >= policy.min_helped_trials &&
            summary.harmed_trials <= policy.max_harmed_trials &&
            summary.help_confidence >= policy.min_help_confidence &&
            unknown_ratio <= policy.max_unknown_ratio) {
        summary.status = common_flydelta_candidate_status::eligible;
    }
    return common_flydelta_promotion_summary_validate(summary, policy, error);
}
