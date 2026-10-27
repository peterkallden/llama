#include "agent/adaptation/flydelta/experiment/flydelta-guidance-qualification.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>
#include <sstream>
#include "hash/hash.h"

namespace {

using json = nlohmann::ordered_json;
constexpr size_t kMinimumFixtureGroups = 20;
constexpr double kHarmPenalty = 5.0;
constexpr uint64_t kRevalidationWatermarkWindow = 20;
constexpr size_t kMaxQualificationGroups = 64;

bool bounded(const std::string & value, size_t limit = 512) {
    return !value.empty() && value.size() <= limit;
}

json context_json(const common_flydelta_outcome_guidance_context & c) {
    return {
        {"scope_fingerprint", c.scope_fingerprint},
        {"model_profile_fingerprint", c.model_profile_fingerprint},
        {"tokenizer_fingerprint", c.tokenizer_fingerprint},
        {"template_fingerprint", c.template_fingerprint},
        {"baseline_deployment_fingerprint", c.baseline_deployment_fingerprint},
        {"verifier_revision", c.verifier_revision},
        {"ruleset_revision", c.ruleset_revision},
        {"action_contract_fingerprint", c.action_contract_fingerprint},
        {"semantic_domain", c.semantic_domain},
        {"strategy", c.strategy}, {"estimator", c.estimator},
    };
}

json qualification_context_json(const common_flydelta_guidance_qualification_context & c) {
    return {
        {"schema_version", c.schema_version},
        {"decision_point", common_flydelta_guidance_decision_point_name(c.decision_point)},
        {"capability", common_flydelta_guidance_capability_name(c.capability)},
        {"policy_revision", c.policy_revision},
        {"selection_limit", c.selection_limit},
        {"progress_reward", c.progress_reward}, {"harm_penalty", c.harm_penalty},
        {"authorized_scope", {
            {"memory_scope", static_cast<int>(c.authorized_scope.memory_scope)},
            {"plan_scope", static_cast<int>(c.authorized_scope.plan_scope)},
            {"namespace_id", c.authorized_scope.namespace_id},
            {"project_id", c.authorized_scope.project_id},
            {"session_id", c.authorized_scope.session_id},
            {"memory_global_opt_in", c.authorized_scope.memory_global_opt_in},
        }},
        {"outcome_context", context_json(c.outcome_context)},
    };
}

bool qualification_context_from_json(
        const json & j, common_flydelta_guidance_qualification_context & c) {
    if (!j.is_object() || j.value("schema_version", 0) != 1 ||
            j.value("decision_point", std::string()) != "deep_full_generation_selection" ||
            j.value("capability", std::string()) != "select_budgeted_subset" ||
            !j.contains("outcome_context") || !j["outcome_context"].is_object() ||
            !j.contains("authorized_scope") || !j["authorized_scope"].is_object()) return false;
    const auto & x = j["outcome_context"];
    const auto & scope = j["authorized_scope"];
    c = {};
    c.policy_revision = j.value("policy_revision", std::string());
    c.selection_limit = j.value("selection_limit", size_t{0});
    c.progress_reward = j.value("progress_reward", 0.25);
    c.harm_penalty = j.value("harm_penalty", kHarmPenalty);
    const int memory_scope = scope.value("memory_scope", -1);
    const int plan_scope = scope.value("plan_scope", -1);
    if (memory_scope < 0 || memory_scope > 3 || plan_scope < 0 || plan_scope > 3) return false;
    c.authorized_scope.memory_scope = static_cast<common_memory_scope>(memory_scope);
    c.authorized_scope.plan_scope = static_cast<common_plan_scope>(plan_scope);
    c.authorized_scope.namespace_id = scope.value("namespace_id", std::string());
    c.authorized_scope.project_id = scope.value("project_id", std::string());
    c.authorized_scope.session_id = scope.value("session_id", std::string());
    c.authorized_scope.memory_global_opt_in = scope.value("memory_global_opt_in", false);
    c.outcome_context.scope_fingerprint = x.value("scope_fingerprint", std::string());
    c.outcome_context.model_profile_fingerprint = x.value("model_profile_fingerprint", std::string());
    c.outcome_context.tokenizer_fingerprint = x.value("tokenizer_fingerprint", std::string());
    c.outcome_context.template_fingerprint = x.value("template_fingerprint", std::string());
    c.outcome_context.baseline_deployment_fingerprint = x.value("baseline_deployment_fingerprint", std::string());
    c.outcome_context.verifier_revision = x.value("verifier_revision", std::string());
    c.outcome_context.ruleset_revision = x.value("ruleset_revision", std::string());
    c.outcome_context.action_contract_fingerprint = x.value("action_contract_fingerprint", std::string());
    c.outcome_context.semantic_domain = x.value("semantic_domain", std::string());
    c.outcome_context.strategy = x.value("strategy", std::string());
    c.outcome_context.estimator = x.value("estimator", std::string());
    return true;
}

bool context_valid(const common_flydelta_guidance_qualification_context & c) {
    const auto & x = c.outcome_context;
    return c.schema_version == 1 && std::isfinite(c.progress_reward) && c.progress_reward >= 0 && c.progress_reward <= 1 &&
        std::isfinite(c.harm_penalty) && c.harm_penalty >= 0 &&
        c.decision_point == common_flydelta_guidance_decision_point::deep_full_generation_selection &&
        c.capability == common_flydelta_guidance_capability::select_budgeted_subset &&
        c.selection_limit > 0 && c.selection_limit <= 16 &&
        bounded(c.policy_revision) && bounded(x.scope_fingerprint) &&
        bounded(c.authorized_scope.namespace_id) && bounded(c.authorized_scope.session_id) &&
        x.scope_fingerprint == common_flydelta_outcome_guidance_scope_fingerprint(c.authorized_scope) &&
        bounded(x.model_profile_fingerprint) && bounded(x.tokenizer_fingerprint) &&
        bounded(x.template_fingerprint) && bounded(x.baseline_deployment_fingerprint) &&
        bounded(x.verifier_revision) && bounded(x.semantic_domain) &&
        bounded(x.strategy) && bounded(x.estimator);
}

json result_json(const common_flydelta_guidance_qualification_result & r);

json fixture_json(const common_flydelta_guidance_fixture_evaluation & f) {
    const auto fingerprint = [](const std::vector<std::string> & ids) {
        const auto bytes = json(ids).dump();
        return "sha256:" + hash_sha256_hex(bytes.data(), bytes.size());
    };
    std::vector<std::string> union_ids;
    std::string error;
    common_flydelta_guidance_union_selection(f.baseline_selection, f.guided_selection,
        f.random_ablation_selection, union_ids, error);
    json arms = json::array();
    auto ordered = f.union_arm_results;
    std::sort(ordered.begin(), ordered.end(), [](const auto & a, const auto & b) { return a.arm_id < b.arm_id; });
    for (const auto & arm : ordered) {
        json value = {{"arm_id", arm.arm_id}, {"outcome", common_flydelta_counterfactual_outcome_name(arm.outcome)},
            {"inference_ms", arm.inference_ms}, {"inference_cost_available", arm.inference_cost_available},
            {"evidence_ref", arm.evidence_ref}};
        if (arm.progress) value["progress"] = json::parse(common_flydelta_progress_evidence_to_json(*arm.progress));
        arms.push_back(std::move(value));
    }
    return {{"fixture_group_id", f.fixture_group_id}, {"evaluation_id", f.evaluation_id},
        {"history_fingerprint", f.history_fingerprint}, {"candidate_pool_fingerprint", f.candidate_pool_fingerprint},
        {"baseline_selection", f.baseline_selection}, {"guided_selection", f.guided_selection},
        {"baseline_selection_fingerprint", fingerprint(f.baseline_selection)},
        {"guided_selection_fingerprint", fingerprint(f.guided_selection)},
        {"random_selection_fingerprint", fingerprint(f.random_ablation_selection)},
        {"union_selection", union_ids}, {"union_selection_fingerprint", fingerprint(union_ids)},
        {"random_ablation_selection", f.random_ablation_selection}, {"union_arm_results", arms},
        {"baseline_inference_ms", f.baseline_inference_ms}, {"baseline_cost_available", f.baseline_cost_available},
        {"baseline_evidence_ref", f.baseline_evidence_ref},
        {"inference_cost_basis", "equal_share_batch_model_ms"}};
}

bool fixture_from_json(const json & j, common_flydelta_guidance_fixture_evaluation & f) {
    if (!j.is_object() || !j.contains("union_arm_results") || !j["union_arm_results"].is_array()) return false;
    f = {};
    f.fixture_group_id = j.at("fixture_group_id").get<std::string>();
    f.evaluation_id = j.value("evaluation_id", std::string());
    f.history_fingerprint = j.value("history_fingerprint", std::string());
    f.candidate_pool_fingerprint = j.value("candidate_pool_fingerprint", std::string());
    f.baseline_selection = j.at("baseline_selection").get<std::vector<std::string>>();
    f.guided_selection = j.at("guided_selection").get<std::vector<std::string>>();
    f.random_ablation_selection = j.at("random_ablation_selection").get<std::vector<std::string>>();
    f.baseline_inference_ms = j.value("baseline_inference_ms", 0.0);
    f.baseline_cost_available = j.value("baseline_cost_available", false);
    f.baseline_evidence_ref = j.value("baseline_evidence_ref", std::string());
    for (const auto & a : j["union_arm_results"]) {
        common_flydelta_guidance_arm_measurement arm;
        arm.arm_id = a.at("arm_id").get<std::string>();
        const auto outcome = a.at("outcome").get<std::string>();
        if (outcome == "helped") arm.outcome = common_flydelta_counterfactual_outcome::helped;
        else if (outcome == "neutral") arm.outcome = common_flydelta_counterfactual_outcome::neutral;
        else if (outcome == "harmed") arm.outcome = common_flydelta_counterfactual_outcome::harmed;
        else if (outcome != "unknown") return false;
        arm.inference_ms = a.at("inference_ms").get<double>();
        arm.inference_cost_available = a.value("inference_cost_available", false);
        arm.evidence_ref = a.value("evidence_ref", std::string());
        if (a.contains("progress")) {
            arm.progress.emplace();
            std::string error;
            if (!common_flydelta_progress_evidence_from_json(a.at("progress").dump(), *arm.progress, error)) return false;
        }
        f.union_arm_results.push_back(std::move(arm));
    }
    return true;
}

bool parse_result(const json & j, common_flydelta_guidance_qualification_result & r) {
    try {
    if (!j.is_object() || j.value("record_type", std::string()) != "flydelta_guidance_qualification" ||
            j.value("schema_version", 0) != 1 || !j.contains("context")) return false;
    r = {};
    if (!qualification_context_from_json(j["context"], r.context)) return false;
    const auto status = j.value("status", std::string());
    if (status == "insufficient_evidence") r.status = common_flydelta_guidance_qualification_status::insufficient_evidence;
    else if (status == "qualified") r.status = common_flydelta_guidance_qualification_status::qualified;
    else if (status == "paused") r.status = common_flydelta_guidance_qualification_status::paused;
    else return false;
    r.fixture_groups = j.value("fixture_groups", size_t{0});
    r.minimum_fixture_groups = j.value("minimum_fixture_groups", kMinimumFixtureGroups);
    r.baseline_helped = j.value("baseline_helped", size_t{0});
    r.guided_helped = j.value("guided_helped", size_t{0});
    r.baseline_harmed = j.value("baseline_harmed", size_t{0});
    r.guided_harmed = j.value("guided_harmed", size_t{0});
    r.random_helped = j.value("random_helped", size_t{0});
    r.random_harmed = j.value("random_harmed", size_t{0});
    r.baseline_unknown = j.value("baseline_unknown", size_t{0});
    r.guided_unknown = j.value("guided_unknown", size_t{0});
    r.random_unknown = j.value("random_unknown", size_t{0});
    r.baseline_improved = j.value("baseline_improved", size_t{0});
    r.guided_improved = j.value("guided_improved", size_t{0});
    r.random_improved = j.value("random_improved", size_t{0});
    r.baseline_progress_regressed = j.value("baseline_progress_regressed", size_t{0});
    r.guided_progress_regressed = j.value("guided_progress_regressed", size_t{0});
    r.random_progress_regressed = j.value("random_progress_regressed", size_t{0});
    r.baseline_progress_unknown = j.value("baseline_progress_unknown", size_t{0});
    r.guided_progress_unknown = j.value("guided_progress_unknown", size_t{0});
    r.random_progress_unknown = j.value("random_progress_unknown", size_t{0});
    r.inference_costs_available = j.value("inference_costs_available", false);
    r.baseline_selected_arms = j.value("baseline_selected_arms", size_t{0});
    r.guided_selected_arms = j.value("guided_selected_arms", size_t{0});
    r.random_selected_arms = j.value("random_selected_arms", size_t{0});
    r.baseline_utility_per_selected_arm = j.value("baseline_utility_per_selected_arm", 0.0);
    r.guided_utility_per_selected_arm = j.value("guided_utility_per_selected_arm", 0.0);
    r.random_utility_per_selected_arm = j.value("random_utility_per_selected_arm", 0.0);
    r.guided_minus_baseline_utility = j.value("guided_minus_baseline_utility", 0.0);
    r.guided_minus_baseline_lower_95_approx = j.value("guided_minus_baseline_lower_95_approx", 0.0);
    r.baseline_inference_ms = j.value("baseline_inference_ms", 0.0);
    r.guided_inference_ms = j.value("guided_inference_ms", 0.0);
    r.random_inference_ms = j.value("random_inference_ms", 0.0);
    r.evidence_watermark = j.value("evidence_watermark", uint64_t{0});
    r.revalidate_after_watermark = j.value("revalidate_after_watermark", uint64_t{0});
    r.evaluation_fingerprint = j.value("evaluation_fingerprint", std::string());
    r.evaluation_payload_json = j.value("evaluation_payload_json", std::string());
    const std::string computed = "sha256:" + hash_sha256_hex(
        r.evaluation_payload_json.data(), r.evaluation_payload_json.size());
    const bool finite_metrics = std::isfinite(r.baseline_utility_per_selected_arm) &&
        std::isfinite(r.guided_utility_per_selected_arm) &&
        std::isfinite(r.random_utility_per_selected_arm) &&
        std::isfinite(r.guided_minus_baseline_utility) &&
        std::isfinite(r.guided_minus_baseline_lower_95_approx) &&
        std::isfinite(r.baseline_inference_ms) && std::isfinite(r.guided_inference_ms) &&
        std::isfinite(r.random_inference_ms) && r.baseline_inference_ms >= 0.0 &&
        r.guided_inference_ms >= 0.0 && r.random_inference_ms >= 0.0;
    const bool status_valid = r.fixture_groups < r.minimum_fixture_groups
        ? r.status == common_flydelta_guidance_qualification_status::insufficient_evidence
        : r.status == common_flydelta_guidance_qualification_status::paused ||
            (r.status == common_flydelta_guidance_qualification_status::qualified &&
             r.guided_minus_baseline_lower_95_approx > 0.0 &&
             r.baseline_selected_arms > 0 && r.guided_selected_arms > 0 &&
             static_cast<double>(r.guided_harmed) / r.guided_selected_arms <=
                static_cast<double>(r.baseline_harmed) / r.baseline_selected_arms);
    json evaluation = json::parse(r.evaluation_payload_json, nullptr, false);
    const bool valid = context_valid(r.context) && bounded(r.evaluation_fingerprint) &&
        !r.evaluation_payload_json.empty() && computed == r.evaluation_fingerprint &&
        r.fixture_groups <= 1000000 && r.minimum_fixture_groups == kMinimumFixtureGroups &&
        finite_metrics && status_valid && evaluation.is_object() &&
        evaluation.value("evidence_watermark", uint64_t{0}) == r.evidence_watermark &&
        evaluation.contains("context") && evaluation["context"] == qualification_context_json(r.context) &&
        evaluation.contains("cohort") && evaluation["cohort"].is_array() &&
        evaluation["cohort"].size() == r.fixture_groups;
    if (!valid) return false;
    // A self-consistent hash alone does not prove that a summary matches its
    // arm evidence. Recompute every metric and the status on read/append.
    std::vector<common_flydelta_guidance_fixture_evaluation> cohort;
    for (const auto & row : evaluation["cohort"]) {
        common_flydelta_guidance_fixture_evaluation fixture;
        if (!fixture_from_json(row, fixture)) return false;
        cohort.push_back(std::move(fixture));
    }
    common_flydelta_guidance_qualification_result recomputed;
    std::string error;
    return common_flydelta_guidance_qualify_deep_selection(r.context, cohort, r.evidence_watermark,
        recomputed, error) && result_json(recomputed) == result_json(r);
    } catch (const std::exception &) { return false; }
}

double utility(common_flydelta_counterfactual_outcome outcome) {
    switch (outcome) {
        case common_flydelta_counterfactual_outcome::helped: return 1.0;
        case common_flydelta_counterfactual_outcome::harmed: return -kHarmPenalty;
        case common_flydelta_counterfactual_outcome::neutral:
        case common_flydelta_counterfactual_outcome::unknown: return 0.0;
    }
    return 0.0;
}

struct policy_summary {
    double utility_sum = 0.0;
    double cost_sum = 0.0;
    size_t arms = 0;
    size_t helped = 0;
    size_t harmed = 0;
    size_t unknown = 0;
    size_t improved = 0, progress_regressed = 0, progress_unknown = 0;
    bool costs_available = true;
};

bool evaluate_selection(const std::vector<std::string> & selection,
        const std::map<std::string, common_flydelta_guidance_arm_measurement> & arms,
        policy_summary & summary, std::string & error,
        const common_flydelta_guidance_qualification_context & context) {
    summary = {};
    if (selection.empty()) { error = "qualification selection must not be empty"; return false; }
    std::set<std::string> unique;
    for (const auto & id : selection) {
        if (!bounded(id) || !unique.insert(id).second) {
            error = "qualification selection contains an empty or duplicate arm id"; return false;
        }
        const auto found = arms.find(id);
        if (found == arms.end()) { error = "qualification selection is missing a union outcome"; return false; }
        const auto & arm = found->second;
        if (!std::isfinite(arm.inference_ms) || arm.inference_ms < 0.0) {
            error = "qualification arm inference cost is invalid"; return false;
        }
        summary.utility_sum += arm.outcome == common_flydelta_counterfactual_outcome::harmed
            ? -context.harm_penalty : utility(arm.outcome);
        auto progress = common_flydelta_progress_transition::unknown;
        if (arm.progress) {
            if (!common_flydelta_progress_evidence_validate(*arm.progress, error) ||
                    arm.progress->candidate_evidence_ref != arm.evidence_ref) {
                if (error.empty()) error = "Qualification progress references a different candidate observation";
                return false;
            }
            progress = common_flydelta_progress_evidence_classify(*arm.progress);
        }
        const bool improved = progress == common_flydelta_progress_transition::improved &&
            arm.outcome != common_flydelta_counterfactual_outcome::helped &&
            arm.outcome != common_flydelta_counterfactual_outcome::harmed;
        summary.improved += improved;
        summary.progress_regressed += progress == common_flydelta_progress_transition::regressed;
        summary.progress_unknown += progress == common_flydelta_progress_transition::unknown;
        if (improved) summary.utility_sum += context.progress_reward;
        if (progress == common_flydelta_progress_transition::regressed &&
                arm.outcome != common_flydelta_counterfactual_outcome::harmed)
            summary.utility_sum -= context.harm_penalty;
        summary.cost_sum += arm.inference_ms;
        ++summary.arms;
        summary.helped += arm.outcome == common_flydelta_counterfactual_outcome::helped;
        summary.harmed += arm.outcome == common_flydelta_counterfactual_outcome::harmed;
        summary.unknown += arm.outcome == common_flydelta_counterfactual_outcome::unknown;
        summary.costs_available = summary.costs_available && arm.inference_cost_available;
    }
    return true;
}

json result_json(const common_flydelta_guidance_qualification_result & r) {
    return {
        {"record_type", "flydelta_guidance_qualification"}, {"schema_version", 1},
        {"context", qualification_context_json(r.context)},
        {"status", common_flydelta_guidance_qualification_status_name(r.status)},
        {"fixture_groups", r.fixture_groups}, {"minimum_fixture_groups", r.minimum_fixture_groups},
        {"baseline_improved", r.baseline_improved}, {"guided_improved", r.guided_improved}, {"random_improved", r.random_improved},
        {"baseline_progress_regressed", r.baseline_progress_regressed}, {"guided_progress_regressed", r.guided_progress_regressed},
        {"random_progress_regressed", r.random_progress_regressed}, {"baseline_progress_unknown", r.baseline_progress_unknown},
        {"guided_progress_unknown", r.guided_progress_unknown}, {"random_progress_unknown", r.random_progress_unknown},
        {"baseline_helped", r.baseline_helped}, {"guided_helped", r.guided_helped},
        {"baseline_harmed", r.baseline_harmed}, {"guided_harmed", r.guided_harmed},
        {"random_helped", r.random_helped}, {"random_harmed", r.random_harmed},
        {"baseline_unknown", r.baseline_unknown}, {"guided_unknown", r.guided_unknown},
        {"random_unknown", r.random_unknown}, {"inference_costs_available", r.inference_costs_available},
        {"inference_cost_basis", "equal_share_batch_model_ms"},
        {"baseline_selected_arms", r.baseline_selected_arms},
        {"guided_selected_arms", r.guided_selected_arms},
        {"random_selected_arms", r.random_selected_arms},
        {"baseline_utility_per_selected_arm", r.baseline_utility_per_selected_arm},
        {"guided_utility_per_selected_arm", r.guided_utility_per_selected_arm},
        {"random_utility_per_selected_arm", r.random_utility_per_selected_arm},
        {"guided_minus_baseline_utility", r.guided_minus_baseline_utility},
        {"guided_minus_baseline_lower_95_approx", r.guided_minus_baseline_lower_95_approx},
        {"baseline_inference_ms", r.baseline_inference_ms},
        {"guided_inference_ms", r.guided_inference_ms}, {"random_inference_ms", r.random_inference_ms},
        {"evidence_watermark", r.evidence_watermark},
        {"revalidate_after_watermark", r.revalidate_after_watermark},
        {"evaluation_fingerprint", r.evaluation_fingerprint},
        {"evaluation_payload_json", r.evaluation_payload_json},
        {"qualification_method", "paired_fixture_progress_normal_95_approx_v2"},
        {"harm_penalty", r.context.harm_penalty}, {"progress_reward", r.context.progress_reward},
    };
}

} // namespace

const char * common_flydelta_guidance_decision_point_name(
        common_flydelta_guidance_decision_point p) {
    return p == common_flydelta_guidance_decision_point::deep_full_generation_selection
        ? "deep_full_generation_selection" : "unknown";
}

const char * common_flydelta_guidance_capability_name(
        common_flydelta_guidance_capability c) {
    switch (c) {
        case common_flydelta_guidance_capability::observe_only: return "observe_only";
        case common_flydelta_guidance_capability::reorder: return "reorder";
        case common_flydelta_guidance_capability::select_budgeted_subset: return "select_budgeted_subset";
        case common_flydelta_guidance_capability::prune: return "prune";
    }
    return "unknown";
}

const char * common_flydelta_guidance_qualification_status_name(
        common_flydelta_guidance_qualification_status s) {
    switch (s) {
        case common_flydelta_guidance_qualification_status::insufficient_evidence: return "insufficient_evidence";
        case common_flydelta_guidance_qualification_status::qualified: return "qualified";
        case common_flydelta_guidance_qualification_status::paused: return "paused";
    }
    return "insufficient_evidence";
}

std::string common_flydelta_outcome_guidance_scope_fingerprint(
        const common_agent_scope & scope) {
    const std::string identity =
        std::to_string(static_cast<int>(scope.memory_scope)) + "\n" +
        std::to_string(static_cast<int>(scope.plan_scope)) + "\n" +
        scope.namespace_id + "\n" + scope.session_id + "\n" + scope.project_id + "\n" +
        (scope.memory_global_opt_in ? "1" : "0");
    return "sha256:" + hash_sha256_hex(identity.data(), identity.size());
}

std::string common_flydelta_guidance_qualification_policy_revision(
        const common_flydelta_outcome_guidance_config & config) {
    const json policy = {
        {"algorithm", "deep-full-generation-mixed-progress-selection-v3"},
        {"projection_builder_revision", "flydelta-outcome-projection-v5"},
        {"seed", config.seed}, {"harm_penalty", config.harm_penalty},
        {"progress_reward", config.progress_reward}, {"progress_contract", "paired-goal-transitions-v1"},
        {"min_backoff_groups", config.min_backoff_groups},
        {"max_prior_strength", config.max_prior_strength},
        {"selection_schedule", "two-ordinary-one-posterior-one-uniform-v1"},
    };
    const std::string bytes = policy.dump();
    return "sha256:" + hash_sha256_hex(bytes.data(), bytes.size());
}

common_flydelta_outcome_guidance_mode common_flydelta_guidance_effective_mode(
        const common_flydelta_outcome_guidance_mode operator_mode,
        const common_flydelta_guidance_decision_point point,
        const bool qualified) {
    if (operator_mode == common_flydelta_outcome_guidance_mode::experimental &&
            point == common_flydelta_guidance_decision_point::deep_full_generation_selection &&
            !qualified) return common_flydelta_outcome_guidance_mode::shadow;
    return operator_mode;
}

bool common_flydelta_guidance_union_selection(
        const std::vector<std::string> & baseline,
        const std::vector<std::string> & guided,
        const std::vector<std::string> & random_ablation,
        std::vector<std::string> & union_ids, std::string & error) {
    error.clear(); union_ids.clear();
    std::set<std::string> seen;
    if (baseline.size() != guided.size() || baseline.size() != random_ablation.size() || baseline.size() > 16) {
        error = "matched policies must use the same bounded Top-K budget"; return false;
    }
    for (const auto * selection : {&baseline, &guided, &random_ablation}) {
        if (selection->empty()) { error = "matched qualification selection must not be empty"; return false; }
        std::set<std::string> within;
        for (const auto & id : *selection) {
            if (!bounded(id) || !within.insert(id).second) {
                error = "matched qualification selection contains an empty or duplicate arm"; return false;
            }
            if (seen.insert(id).second) union_ids.push_back(id);
        }
    }
    return true;
}

bool common_flydelta_guidance_run_matched_union(
        common_flydelta_guidance_fixture_evaluation & fixture,
        const common_flydelta_guidance_union_runner & runner,
        std::string & error) {
    error.clear();
    if (!bounded(fixture.fixture_group_id) || !runner) {
        error = "matched qualification fixture/runner is invalid";
        return false;
    }
    std::vector<std::string> union_ids;
    if (!common_flydelta_guidance_union_selection(fixture.baseline_selection,
            fixture.guided_selection, fixture.random_ablation_selection, union_ids, error)) return false;
    std::vector<common_flydelta_guidance_arm_measurement> results;
    if (!runner(fixture.fixture_group_id, union_ids, results, error)) {
        if (error.empty()) error = "matched qualification union runner failed";
        return false;
    }
    std::set<std::string> expected(union_ids.begin(), union_ids.end());
    std::set<std::string> actual;
    for (const auto & result : results) {
        if (!bounded(result.arm_id) || !std::isfinite(result.inference_ms) ||
                result.inference_ms < 0.0 ||
                result.outcome < common_flydelta_counterfactual_outcome::unknown ||
                result.outcome > common_flydelta_counterfactual_outcome::harmed ||
                !actual.insert(result.arm_id).second) {
            error = "matched qualification union runner returned invalid/duplicate evidence";
            return false;
        }
    }
    if (actual != expected) {
        error = "matched qualification union runner omitted or added an arm";
        return false;
    }
    fixture.union_arm_results = std::move(results);
    return true;
}

bool common_flydelta_guidance_qualify_deep_selection(
        const common_flydelta_guidance_qualification_context & context,
        const std::vector<common_flydelta_guidance_fixture_evaluation> & cohort,
        const uint64_t evidence_watermark,
        common_flydelta_guidance_qualification_result & result,
        std::string & error) {
    using namespace std;
    error.clear(); result = {};
    if (!context_valid(context) || cohort.size() > 100000) {
        error = "Deep guidance qualification context/cohort is invalid"; return false;
    }
    result.context = context;
    result.inference_costs_available = !cohort.empty();
    result.fixture_groups = cohort.size();
    result.evidence_watermark = evidence_watermark;
    result.revalidate_after_watermark = evidence_watermark >
            std::numeric_limits<uint64_t>::max() - kRevalidationWatermarkWindow
        ? std::numeric_limits<uint64_t>::max()
        : evidence_watermark + kRevalidationWatermarkWindow;
    vector<double> paired_differences;
    json canonical = json::array();
    set<string> fixtures;
    double baseline_total_utility = 0.0, guided_total_utility = 0.0, random_total_utility = 0.0;
    size_t baseline_total_arms = 0, guided_total_arms = 0, random_total_arms = 0;
    for (const auto & fixture : cohort) {
        if (!bounded(fixture.fixture_group_id) || !fixtures.insert(fixture.fixture_group_id).second) {
            error = "qualification cohort has an empty or repeated fixture group"; return false;
        }
        if (fixture.baseline_selection.size() != context.selection_limit ||
                !std::isfinite(fixture.baseline_inference_ms) || fixture.baseline_inference_ms < 0.0) {
            error = "qualification fixture has incompatible Top-K or baseline cost"; return false;
        }
        map<string, common_flydelta_guidance_arm_measurement> arms;
        for (const auto & arm : fixture.union_arm_results) {
            if (!bounded(arm.arm_id) || !isfinite(arm.inference_ms) || arm.inference_ms < 0.0 ||
                    arm.outcome < common_flydelta_counterfactual_outcome::unknown ||
                    arm.outcome > common_flydelta_counterfactual_outcome::harmed ||
                    !arms.emplace(arm.arm_id, arm).second) {
                error = "qualification union has an invalid or duplicate arm result"; return false;
            }
        }
        vector<string> expected_union;
        if (!common_flydelta_guidance_union_selection(fixture.baseline_selection,
                fixture.guided_selection, fixture.random_ablation_selection, expected_union, error)) return false;
        set<string> expected_set(expected_union.begin(), expected_union.end());
        set<string> actual_set;
        for (const auto & entry : arms) actual_set.insert(entry.first);
        if (expected_set != actual_set) {
            error = "qualification union result set differs from the three selected policies";
            return false;
        }
        policy_summary baseline, guided, random;
        if (!evaluate_selection(fixture.baseline_selection, arms, baseline, error, context) ||
                !evaluate_selection(fixture.guided_selection, arms, guided, error, context) ||
                !evaluate_selection(fixture.random_ablation_selection, arms, random, error, context)) return false;
        const double b = baseline.utility_sum / baseline.arms;
        const double g = guided.utility_sum / guided.arms;
        paired_differences.push_back(g - b);
        baseline_total_utility += baseline.utility_sum;
        guided_total_utility += guided.utility_sum;
        random_total_utility += random.utility_sum;
        baseline_total_arms += baseline.arms;
        guided_total_arms += guided.arms;
        random_total_arms += random.arms;
        result.baseline_improved += baseline.improved;
        result.guided_improved += guided.improved;
        result.random_improved += random.improved;
        result.baseline_progress_regressed += baseline.progress_regressed;
        result.guided_progress_regressed += guided.progress_regressed;
        result.random_progress_regressed += random.progress_regressed;
        result.baseline_progress_unknown += baseline.progress_unknown;
        result.guided_progress_unknown += guided.progress_unknown;
        result.random_progress_unknown += random.progress_unknown;
        result.baseline_helped += baseline.helped;
        result.guided_helped += guided.helped;
        result.baseline_harmed += baseline.harmed;
        result.guided_harmed += guided.harmed;
        result.random_helped += random.helped;
        result.random_harmed += random.harmed;
        result.baseline_unknown += baseline.unknown;
        result.guided_unknown += guided.unknown;
        result.random_unknown += random.unknown;
        result.inference_costs_available = result.inference_costs_available &&
            baseline.costs_available && guided.costs_available && random.costs_available && fixture.baseline_cost_available;
        result.baseline_inference_ms += baseline.cost_sum + fixture.baseline_inference_ms;
        result.guided_inference_ms += guided.cost_sum + fixture.baseline_inference_ms;
        result.random_inference_ms += random.cost_sum + fixture.baseline_inference_ms;
        result.baseline_selected_arms += baseline.arms;
        result.guided_selected_arms += guided.arms;
        result.random_selected_arms += random.arms;
        canonical.push_back(fixture_json(fixture));
    }
    if (baseline_total_arms) result.baseline_utility_per_selected_arm = baseline_total_utility / baseline_total_arms;
    if (guided_total_arms) result.guided_utility_per_selected_arm = guided_total_utility / guided_total_arms;
    if (random_total_arms) result.random_utility_per_selected_arm = random_total_utility / random_total_arms;
    result.guided_minus_baseline_utility = paired_differences.empty() ? 0.0 :
        accumulate(paired_differences.begin(), paired_differences.end(), 0.0) / paired_differences.size();
    if (paired_differences.size() > 1) {
        double square_sum = 0.0;
        for (const double d : paired_differences) {
            const double centered = d - result.guided_minus_baseline_utility;
            square_sum += centered * centered;
        }
        const double sample_sd = sqrt(square_sum / (paired_differences.size() - 1));
        result.guided_minus_baseline_lower_95_approx = result.guided_minus_baseline_utility -
            1.96 * sample_sd / sqrt(static_cast<double>(paired_differences.size()));
    } else {
        result.guided_minus_baseline_lower_95_approx = 0.0;
    }
    result.minimum_fixture_groups = kMinimumFixtureGroups;
    if (cohort.size() < kMinimumFixtureGroups) {
        result.status = common_flydelta_guidance_qualification_status::insufficient_evidence;
    } else {
        const double baseline_harm_rate = static_cast<double>(result.baseline_harmed) / result.baseline_selected_arms;
        const double guided_harm_rate = static_cast<double>(result.guided_harmed) / result.guided_selected_arms;
        const double baseline_regression_rate = static_cast<double>(result.baseline_progress_regressed) / result.baseline_selected_arms;
        const double guided_regression_rate = static_cast<double>(result.guided_progress_regressed) / result.guided_selected_arms;
        result.status = result.guided_minus_baseline_lower_95_approx > 0.0 &&
                guided_harm_rate <= baseline_harm_rate && guided_regression_rate <= baseline_regression_rate
            ? common_flydelta_guidance_qualification_status::qualified
            : common_flydelta_guidance_qualification_status::paused;
    }
    canonical = {{"context", qualification_context_json(context)},
        {"evidence_watermark", evidence_watermark}, {"cohort", canonical}};
    const string bytes = canonical.dump();
    result.evaluation_fingerprint = "sha256:" + hash_sha256_hex(bytes.data(), bytes.size());
    result.evaluation_payload_json = bytes;
    return true;
}

bool common_flydelta_guidance_append_qualification(
        common_learning_lifecycle_store & store,
        const common_flydelta_lifecycle_event_context & event_context,
        const common_flydelta_guidance_qualification_result & result,
        std::string & error) {
    error.clear();
    const bool metrics_finite = std::isfinite(result.baseline_utility_per_selected_arm) &&
        std::isfinite(result.guided_utility_per_selected_arm) &&
        std::isfinite(result.random_utility_per_selected_arm) &&
        std::isfinite(result.guided_minus_baseline_utility) &&
        std::isfinite(result.guided_minus_baseline_lower_95_approx) &&
        std::isfinite(result.baseline_inference_ms) && std::isfinite(result.guided_inference_ms) &&
        std::isfinite(result.random_inference_ms) && result.baseline_inference_ms >= 0.0 &&
        result.guided_inference_ms >= 0.0 && result.random_inference_ms >= 0.0;
    if (event_context.event_id.empty() || event_context.idempotency_key.empty() ||
            event_context.source_id.empty() || !context_valid(result.context) ||
            event_context.scope.namespace_id != result.context.authorized_scope.namespace_id ||
            event_context.scope.project_id != result.context.authorized_scope.project_id ||
            event_context.scope.session_id != result.context.authorized_scope.session_id ||
            common_flydelta_outcome_guidance_scope_fingerprint(event_context.scope) !=
                result.context.outcome_context.scope_fingerprint ||
            !bounded(result.evaluation_fingerprint) || result.fixture_groups > 100000 ||
            !metrics_finite ||
            result.evaluation_payload_json.empty() ||
            ("sha256:" + hash_sha256_hex(result.evaluation_payload_json.data(),
                result.evaluation_payload_json.size())) != result.evaluation_fingerprint ||
            !std::isfinite(result.guided_minus_baseline_utility) ||
            (result.status == common_flydelta_guidance_qualification_status::qualified &&
             result.fixture_groups < result.minimum_fixture_groups)) {
        error = "FlyDelta qualification lifecycle event is invalid"; return false;
    }
    common_learning_lifecycle_record record;
    record.event_id = event_context.event_id;
    record.subject_id = "deep_full_generation_selection";
    record.kind = common_learning_lifecycle_kind::flydelta_experiment;
    record.status = common_learning_lifecycle_status::observed;
    record.idempotency_key = event_context.idempotency_key;
    record.source_id = event_context.source_id;
    record.namespace_id = event_context.scope.namespace_id;
    record.project_id = event_context.scope.project_id;
    record.session_id = event_context.scope.session_id;
    const json payload = result_json(result);
    common_flydelta_guidance_qualification_result validated;
    if (!parse_result(payload, validated)) {
        error = "FlyDelta qualification result failed its stored-contract validation";
        return false;
    }
    record.payload_json = payload.dump();
    record.content_hash = "sha256:" + hash_sha256_hex(
        record.payload_json.data(), record.payload_json.size());
    record.created_at = event_context.created_at;
    return store.append(record, error);
}

bool common_flydelta_guidance_record_matched_evaluation(
        common_learning_lifecycle_store & store,
        const common_flydelta_lifecycle_event_context & event,
        const common_flydelta_guidance_qualification_context & context,
        const common_flydelta_guidance_fixture_evaluation & fixture,
        std::string & error,
        common_flydelta_guidance_qualification_result * qualification) {
    error.clear();
    if (qualification) *qualification = {};
    try {
        if (!context_valid(context) || event.event_id.empty() || event.idempotency_key.empty() ||
                event.source_id.empty() || event.created_at.empty() ||
                common_flydelta_outcome_guidance_scope_fingerprint(event.scope) != context.outcome_context.scope_fingerprint ||
                !bounded(fixture.evaluation_id) || !bounded(fixture.history_fingerprint) ||
                !bounded(fixture.candidate_pool_fingerprint) || !bounded(fixture.baseline_evidence_ref)) {
            error = "matched evaluation identity, scope or baseline evidence is invalid"; return false;
        }
        for (const auto & arm : fixture.union_arm_results) {
            if (!bounded(arm.evidence_ref)) { error = "matched arm has no host evidence reference"; return false; }
        }
        common_flydelta_guidance_qualification_result checked;
        if (!common_flydelta_guidance_qualify_deep_selection(context, {fixture}, 0, checked, error)) return false;
        const json payload = {{"record_type", "flydelta_guidance_matched_evaluation"},
            {"schema_version", 1}, {"context", qualification_context_json(context)}, {"fixture", fixture_json(fixture)}};
        common_learning_lifecycle_record record;
        record.event_id = event.event_id;
        record.idempotency_key = event.idempotency_key;
        record.subject_id = "deep_full_generation_selection";
        record.kind = common_learning_lifecycle_kind::flydelta_experiment;
        record.status = common_learning_lifecycle_status::observed;
        record.source_id = event.source_id;
        record.namespace_id = event.scope.namespace_id;
        record.project_id = event.scope.project_id;
        record.session_id = event.scope.session_id;
        record.created_at = event.created_at;
        record.payload_json = payload.dump();
        record.content_hash = "sha256:" + hash_sha256_hex(record.payload_json.data(), record.payload_json.size());
        if (!store.append(record, error)) return false;
        return common_flydelta_guidance_refresh_qualification(store, event, context, error, qualification);
    } catch (const std::exception & ex) {
        error = std::string("matched evaluation persistence failed: ") + ex.what(); return false;
    } catch (...) {
        error = "matched evaluation persistence failed"; return false;
    }
}

bool common_flydelta_guidance_refresh_qualification(
        common_learning_lifecycle_store & store,
        const common_flydelta_lifecycle_event_context & event,
        const common_flydelta_guidance_qualification_context & context,
        std::string & error,
        common_flydelta_guidance_qualification_result * qualification) {
    error.clear();
    if (qualification) *qualification = {};
    try {
        if (!context_valid(context) ||
                common_flydelta_outcome_guidance_scope_fingerprint(event.scope) != context.outcome_context.scope_fingerprint) {
            error = "qualification refresh context or scope is invalid"; return false;
        }
        std::vector<common_learning_lifecycle_record> records;
        uint64_t watermark = 0;
        if (!store.list_scope_with_watermark(event.scope.namespace_id, event.scope.project_id, event.scope.session_id,
                records, watermark, error)) return false;
        std::sort(records.begin(), records.end(), [](const auto & a, const auto & b) {
            return a.registration_order < b.registration_order;
        });
        std::map<std::string, std::string> attempts;
        std::set<std::string> fixture_groups;
        std::vector<common_flydelta_guidance_fixture_evaluation> cohort;
        uint64_t evidence_watermark = 0;
        for (const auto & row : records) {
            if (row.kind != common_learning_lifecycle_kind::flydelta_experiment ||
                    row.registration_order == 0 || row.registration_order > watermark ||
                    row.namespace_id != event.scope.namespace_id || row.project_id != event.scope.project_id ||
                    row.session_id != event.scope.session_id) continue;
            const auto j = json::parse(row.payload_json, nullptr, false);
            if (!j.is_object() || j.value("record_type", std::string()) != "flydelta_guidance_matched_evaluation" ||
                    !j.contains("context") || j["context"] != qualification_context_json(context)) continue;
            if (j.value("schema_version", 0) != 1 || row.content_hash !=
                    "sha256:" + hash_sha256_hex(row.payload_json.data(), row.payload_json.size())) {
                error = "matched evaluation record failed integrity validation"; return false;
            }
            const auto inserted = attempts.emplace(row.idempotency_key, row.content_hash);
            if (!inserted.second) {
                if (inserted.first->second != row.content_hash) {
                    error = "conflicting matched evaluation retry"; return false;
                }
                continue;
            }
            common_flydelta_guidance_fixture_evaluation restored;
            if (!fixture_from_json(j.at("fixture"), restored) || fixture_json(restored) != j["fixture"]) {
                error = "invalid matched evaluation fixture"; return false;
            }
            // Keep the first registered experiment for each fixture group.
            // Retries/repeated decisions cannot become extra independent data.
            if (fixture_groups.insert(restored.fixture_group_id).second) {
                cohort.push_back(std::move(restored));
                evidence_watermark = row.registration_order;
            }
        }
        common_flydelta_guidance_qualification_result result;
        if (cohort.empty()) { error = "no compatible matched evaluation to refresh"; return false; }
        if (cohort.size() > kMaxQualificationGroups) {
            cohort.erase(cohort.begin(), cohort.end() - kMaxQualificationGroups);
        }
        if (!common_flydelta_guidance_qualify_deep_selection(context, cohort, evidence_watermark, result, error)) return false;
        auto qualification_event = event;
        qualification_event.event_id = "flydelta:qualification:" + result.evaluation_fingerprint;
        qualification_event.idempotency_key = qualification_event.event_id;
        if (!common_flydelta_guidance_append_qualification(store, qualification_event, result, error)) return false;
        if (qualification) *qualification = std::move(result);
        return true;
    } catch (const std::exception & ex) {
        error = std::string("matched evaluation persistence failed: ") + ex.what();
        return false;
    } catch (...) {
        error = "matched evaluation persistence failed"; return false;
    }
}

bool common_flydelta_guidance_find_qualification(
        const std::vector<common_learning_lifecycle_record> & records,
        const common_flydelta_guidance_qualification_context & context,
        const uint64_t current_watermark,
        common_flydelta_guidance_qualification_result & result,
        bool & usable,
        std::string & reason,
        std::string & error) {
    error.clear(); reason.clear(); usable = false; result = {};
    if (!context_valid(context)) { error = "requested qualification context is invalid"; return false; }
    bool found = false;
    std::vector<const common_learning_lifecycle_record *> ordered;
    for (const auto & record : records) {
        if (record.kind != common_learning_lifecycle_kind::flydelta_experiment ||
                record.namespace_id != context.authorized_scope.namespace_id ||
                record.project_id != context.authorized_scope.project_id ||
                record.session_id != context.authorized_scope.session_id ||
                record.registration_order == 0 || record.registration_order > current_watermark) continue;
        ordered.push_back(&record);
    }
    std::sort(ordered.begin(), ordered.end(), [](const auto * a, const auto * b) {
        return a->registration_order > b->registration_order;
    });
    for (const auto * candidate : ordered) {
        const auto & record = *candidate;
        try {
            const json payload = json::parse(record.payload_json, nullptr, false);
            if (payload.is_object() && payload.value("record_type", std::string()) ==
                    "flydelta_guidance_qualification") {
                // Ignore other selector budgets/policies before validating
                // their detailed result. They cannot authorize this request.
                if (!payload.contains("context") || payload["context"] != qualification_context_json(context)) continue;
                common_flydelta_guidance_qualification_result parsed;
                if (!parse_result(payload, parsed)) {
                    error = "qualification record schema/fingerprint is invalid";
                    return false;
                }
                if (qualification_context_json(parsed.context) != qualification_context_json(context)) continue;
                const std::string record_fingerprint = "sha256:" + hash_sha256_hex(
                    record.payload_json.data(), record.payload_json.size());
                if (record.content_hash != record_fingerprint) {
                    error = "qualification content fingerprint mismatch"; return false;
                }
                result = std::move(parsed);
                found = true;
                break;
            }
        } catch (const std::exception &) {
            error = "malformed qualification record"; return false;
        }
    }
    if (!found) { reason = "qualification_not_found"; return true; }
    if (result.status != common_flydelta_guidance_qualification_status::qualified) {
        reason = common_flydelta_guidance_qualification_status_name(result.status); return true;
    }
    if (result.evidence_watermark > current_watermark || current_watermark >= result.revalidate_after_watermark) {
        reason = "qualification_revalidation_due"; return true;
    }
    usable = true;
    reason = "qualified_for_deep_budgeted_selection";
    return true;
}
