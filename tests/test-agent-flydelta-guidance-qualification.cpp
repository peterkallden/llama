#include "agent/adaptation/flydelta/experiment/flydelta-guidance-qualification.h"

#include <iostream>

#define CHECK(condition) do { if (!(condition)) { std::cerr << "failed line " << __LINE__ << '\n'; return __LINE__; } } while (false)

struct interrupted_qualification_store : common_learning_lifecycle_store {
    common_learning_in_memory_lifecycle_store backing;
    bool fail_publication = true;
    bool append(const common_learning_lifecycle_record & record, std::string & error) override {
        if (fail_publication && record.event_id.find("flydelta:qualification:") == 0) {
            fail_publication = false;
            error = "simulated qualification publication interruption";
            return false;
        }
        return backing.append(record, error);
    }
    bool contains_idempotency(const std::string & key, bool & contains, std::string & error) const override {
        return backing.contains_idempotency(key, contains, error);
    }
    std::vector<common_learning_lifecycle_record> list(std::string & error) const override { return backing.list(error); }
};

static common_flydelta_guidance_qualification_context context() {
    common_flydelta_guidance_qualification_context c;
    c.authorized_scope.namespace_id = "ns";
    c.authorized_scope.project_id = "project";
    c.authorized_scope.session_id = "session";
    c.outcome_context.scope_fingerprint =
        common_flydelta_outcome_guidance_scope_fingerprint(c.authorized_scope);
    c.outcome_context.model_profile_fingerprint = "model";
    c.outcome_context.tokenizer_fingerprint = "tokenizer";
    c.outcome_context.template_fingerprint = "template";
    c.outcome_context.baseline_deployment_fingerprint = "baseline";
    c.outcome_context.verifier_revision = "verifier:v1";
    c.outcome_context.semantic_domain = "domain";
    c.outcome_context.strategy = "coordinate";
    c.outcome_context.estimator = "coefficient-margin-geometry-v1";
    c.policy_revision = common_flydelta_guidance_qualification_policy_revision(
        common_flydelta_outcome_guidance_config{});
    c.selection_limit = 1;
    return c;
}

static std::vector<common_flydelta_guidance_fixture_evaluation> cohort(bool guided_wins) {
    std::vector<common_flydelta_guidance_fixture_evaluation> values;
    for (size_t i = 0; i < 24; ++i) {
        common_flydelta_guidance_fixture_evaluation fixture;
        fixture.fixture_group_id = "fixture:" + std::to_string(i);
        fixture.evaluation_id = "evaluation:" + std::to_string(i);
        fixture.history_fingerprint = "history:fixture:" + std::to_string(i);
        fixture.candidate_pool_fingerprint = "pool:fixture:" + std::to_string(i);
        fixture.baseline_evidence_ref = "evidence:baseline:" + std::to_string(i);
        fixture.baseline_selection = {"base"};
        fixture.guided_selection = {"guide"};
        fixture.random_ablation_selection = {"random"};
        fixture.union_arm_results = {
            {"base", guided_wins ? common_flydelta_counterfactual_outcome::neutral
                                 : common_flydelta_counterfactual_outcome::helped, 100.0},
            {"guide", guided_wins ? common_flydelta_counterfactual_outcome::helped
                                  : common_flydelta_counterfactual_outcome::neutral, 80.0},
            {"random", common_flydelta_counterfactual_outcome::neutral, 90.0},
        };
        for (auto & arm : fixture.union_arm_results) arm.evidence_ref = "evidence:" + arm.arm_id;
        values.push_back(std::move(fixture));
    }
    return values;
}

int main() {
    std::string error;
    common_agent_scope first_scope;
    first_scope.namespace_id = "workspace";
    first_scope.project_id = "project-a";
    first_scope.session_id = "session-a";
    first_scope.turn_id = "turn-1";
    auto next_turn_scope = first_scope;
    next_turn_scope.turn_id = "turn-2";
    CHECK(common_flydelta_outcome_guidance_scope_fingerprint(first_scope) ==
        common_flydelta_outcome_guidance_scope_fingerprint(next_turn_scope));
    next_turn_scope.session_id = "session-b";
    CHECK(common_flydelta_outcome_guidance_scope_fingerprint(first_scope) !=
        common_flydelta_outcome_guidance_scope_fingerprint(next_turn_scope));
    common_flydelta_outcome_guidance_config guidance_config;
    CHECK(common_flydelta_guidance_effective_mode(
        common_flydelta_outcome_guidance_mode::off,
        common_flydelta_guidance_decision_point::deep_full_generation_selection, true) ==
        common_flydelta_outcome_guidance_mode::off);
    CHECK(common_flydelta_guidance_effective_mode(
        common_flydelta_outcome_guidance_mode::experimental,
        common_flydelta_guidance_decision_point::deep_full_generation_selection, false) ==
        common_flydelta_outcome_guidance_mode::shadow);
    CHECK(common_flydelta_guidance_effective_mode(
        common_flydelta_outcome_guidance_mode::experimental,
        common_flydelta_guidance_decision_point::deep_full_generation_selection, true) ==
        common_flydelta_outcome_guidance_mode::experimental);
    CHECK(common_flydelta_guidance_effective_mode(
        common_flydelta_outcome_guidance_mode::shadow,
        common_flydelta_guidance_decision_point::deep_full_generation_selection, true) ==
        common_flydelta_outcome_guidance_mode::shadow);
    const auto policy_revision = common_flydelta_guidance_qualification_policy_revision(guidance_config);
    guidance_config.harm_penalty += 1.0;
    CHECK(policy_revision != common_flydelta_guidance_qualification_policy_revision(guidance_config));
    std::vector<std::string> union_ids;
    CHECK(common_flydelta_guidance_union_selection(
        {"a", "b"}, {"b", "c"}, {"c", "d"}, union_ids, error));
    CHECK((union_ids == std::vector<std::string>{"a", "b", "c", "d"}));
    CHECK(!common_flydelta_guidance_union_selection({"a", "a"}, {"b"}, {"c"}, union_ids, error));
    CHECK(common_flydelta_guidance_union_selection(
        {"a", "b"}, {"b", "c"}, {"c", "d"}, union_ids, error));
    common_flydelta_guidance_fixture_evaluation matched_fixture;
    matched_fixture.fixture_group_id = "fixture:union";
    matched_fixture.baseline_selection = {"a", "b"};
    matched_fixture.guided_selection = {"b", "c"};
    matched_fixture.random_ablation_selection = {"c", "d"};
    size_t union_runs = 0;
    CHECK(common_flydelta_guidance_run_matched_union(matched_fixture,
        [&](const std::string & fixture_id, const std::vector<std::string> & ids,
                std::vector<common_flydelta_guidance_arm_measurement> & results,
                std::string &) {
            ++union_runs;
            if (fixture_id != "fixture:union" || ids != union_ids) return false;
            for (const auto & id : ids) results.push_back({id,
                common_flydelta_counterfactual_outcome::neutral, 1.0});
            return true;
        }, error));
    CHECK(union_runs == 1 && matched_fixture.union_arm_results.size() == union_ids.size());

    common_flydelta_guidance_qualification_result result;
    CHECK(common_flydelta_guidance_qualify_deep_selection(
        context(), cohort(true), 0, result, error));
    CHECK(result.status == common_flydelta_guidance_qualification_status::qualified);
    CHECK(result.fixture_groups == 24);
    CHECK(result.guided_helped == 24 && result.baseline_helped == 0);
    CHECK(result.guided_inference_ms == 24.0 * 80.0);
    CHECK(result.baseline_inference_ms == 24.0 * 100.0);
    CHECK(result.guided_minus_baseline_lower_95_approx > 0.0);

    common_learning_in_memory_lifecycle_store store;
    common_flydelta_lifecycle_event_context event;
    event.event_id = "qualification:1";
    event.idempotency_key = event.event_id;
    event.source_id = "matched-deep-evaluation";
    event.scope = context().authorized_scope;
    event.created_at = "2026-10-20T00:00:00Z";
    auto wrong_scope_event = event;
    wrong_scope_event.event_id = "qualification:wrong-scope";
    wrong_scope_event.idempotency_key = wrong_scope_event.event_id;
    wrong_scope_event.scope.session_id = "another-session";
    CHECK(!common_flydelta_guidance_append_qualification(
        store, wrong_scope_event, result, error));
    auto changed_summary = result;
    ++changed_summary.guided_helped;
    CHECK(!common_flydelta_guidance_append_qualification(store, event, changed_summary, error));
    CHECK(common_flydelta_guidance_append_qualification(store, event, result, error));
    std::vector<common_learning_lifecycle_record> records;
    uint64_t watermark = 0;
    CHECK(store.list_scope_with_watermark("ns", "project", "session", records, watermark, error));
    bool usable = false;
    std::string reason;
    common_flydelta_guidance_qualification_result restored;
    CHECK(common_flydelta_guidance_find_qualification(
        records, context(), watermark, restored, usable, reason, error));
    CHECK(usable && restored.evaluation_fingerprint == result.evaluation_fingerprint);
    CHECK(common_flydelta_guidance_find_qualification(
        records, context(), result.revalidate_after_watermark, restored, usable, reason, error));
    CHECK(!usable && reason == "qualification_revalidation_due");

    auto incompatible = context();
    incompatible.outcome_context.verifier_revision = "verifier:v2";
    CHECK(common_flydelta_guidance_find_qualification(
        records, incompatible, watermark, restored, usable, reason, error));
    CHECK(!usable && reason == "qualification_not_found");
    incompatible = context();
    incompatible.selection_limit = 2;
    CHECK(common_flydelta_guidance_find_qualification(
        records, incompatible, watermark, restored, usable, reason, error));
    CHECK(!usable && reason == "qualification_not_found");
    CHECK(common_flydelta_guidance_find_qualification(
        records, context(), 0, restored, usable, reason, error));
    CHECK(!usable && reason == "qualification_not_found");
    auto tampered = records;
    const auto status_pos = tampered.front().payload_json.find("qualified");
    CHECK(status_pos != std::string::npos);
    tampered.front().payload_json.replace(status_pos, 9, "paused");
    CHECK(!common_flydelta_guidance_find_qualification(
        tampered, context(), watermark, restored, usable, reason, error));

    CHECK(common_flydelta_guidance_qualify_deep_selection(
        context(), cohort(false), 200, result, error));
    CHECK(result.status == common_flydelta_guidance_qualification_status::paused);
    auto too_small = cohort(true);
    too_small.resize(5);
    CHECK(common_flydelta_guidance_qualify_deep_selection(
        context(), too_small, 300, result, error));
    CHECK(result.status == common_flydelta_guidance_qualification_status::insufficient_evidence);

    CHECK(!common_flydelta_guidance_union_selection({"a"}, {"b", "c"}, {"d"}, union_ids, error));
    common_learning_in_memory_lifecycle_store matched_store;
    const auto matched_cohort = cohort(true);
    for (size_t i = 0; i < matched_cohort.size(); ++i) {
        auto matched_event = event;
        matched_event.event_id = "matched:" + std::to_string(i);
        matched_event.idempotency_key = matched_event.event_id;
        CHECK(common_flydelta_guidance_record_matched_evaluation(
            matched_store, matched_event, context(), matched_cohort[i], error));
    }
    CHECK(matched_store.list_scope_with_watermark("ns", "project", "session", records, watermark, error));
    CHECK(common_flydelta_guidance_find_qualification(records, context(), watermark, restored, usable, reason, error));
    CHECK(usable && restored.fixture_groups == matched_cohort.size());
    CHECK(!restored.inference_costs_available); // costs were not independently supplied
    const auto count = records.size();
    auto repeated_event = event;
    repeated_event.event_id = "matched:23";
    repeated_event.idempotency_key = repeated_event.event_id;
    CHECK(common_flydelta_guidance_record_matched_evaluation(
        matched_store, repeated_event, context(), matched_cohort.back(), error));
    CHECK(matched_store.list_scope_with_watermark("ns", "project", "session", records, watermark, error));
    CHECK(common_flydelta_guidance_find_qualification(records, context(), watermark, restored, usable, reason, error));
    CHECK(restored.fixture_groups == matched_cohort.size());
    // An idempotent retry must not add a new evaluation OR qualification.
    CHECK(records.size() == count);
    interrupted_qualification_store interrupted;
    auto interrupted_event = event;
    interrupted_event.event_id = "interrupted:matched";
    interrupted_event.idempotency_key = interrupted_event.event_id;
    CHECK(!common_flydelta_guidance_record_matched_evaluation(
        interrupted, interrupted_event, context(), matched_cohort.front(), error));
    CHECK(interrupted.list_scope_with_watermark("ns", "project", "session", records, watermark, error));
    CHECK(records.size() == 1); // union evidence is durable; no qualification yet
    CHECK(common_flydelta_guidance_refresh_qualification(
        interrupted, interrupted_event, context(), error, &restored));
    CHECK(restored.fixture_groups == 1 && restored.status ==
        common_flydelta_guidance_qualification_status::insufficient_evidence);
    CHECK(interrupted.list_scope_with_watermark("ns", "project", "session", records, watermark, error));
    CHECK(records.size() == 2);
    // UNKNOWN full outcomes can still supply paired, declared-goal progress.
    auto partial_cohort = cohort(true);
    for (auto & fixture : partial_cohort) {
        for (auto & arm : fixture.union_arm_results) {
            arm.outcome = common_flydelta_counterfactual_outcome::unknown;
            if (arm.arm_id != "guide") continue;
            common_flydelta_progress_evidence e;
            e.baseline_ref = fixture.fixture_group_id + ":baseline";
            e.candidate_ref = fixture.fixture_group_id + ":candidate";
            e.baseline_evidence_ref = fixture.baseline_evidence_ref;
            e.candidate_evidence_ref = arm.evidence_ref;
            e.baseline.objective_id = fixture.fixture_group_id;
            e.baseline.objective_revision = "goals:v1"; e.baseline.verifier_ref = "verifier:v1";
            e.baseline.assessment_known = e.baseline.regression_known = true;
            e.baseline.satisfied_goals = {"A"}; e.baseline.remaining_goals = {"B", "C"};
            e.candidate = e.baseline;
            e.candidate.satisfied_goals = {"A", "B"}; e.candidate.remaining_goals = {"C"};
            arm.progress = std::move(e);
        }
    }
    CHECK(common_flydelta_guidance_qualify_deep_selection(context(), partial_cohort, 0, result, error));
    CHECK(result.status == common_flydelta_guidance_qualification_status::qualified);
    CHECK(result.guided_improved == 24 && result.guided_helped == 0 && result.guided_unknown == 24);
    auto helped_cohort = partial_cohort;
    for (auto & fixture : helped_cohort) fixture.union_arm_results[1].outcome = common_flydelta_counterfactual_outcome::helped;
    CHECK(common_flydelta_guidance_qualify_deep_selection(context(), helped_cohort, 0, result, error));
    CHECK(result.guided_helped == 24 && result.guided_improved == 0 && result.guided_utility_per_selected_arm == 1.0);
    auto wrong_reference = partial_cohort;
    wrong_reference.front().union_arm_results[1].progress->candidate_evidence_ref = "unrelated:evidence";
    CHECK(!common_flydelta_guidance_qualify_deep_selection(context(), wrong_reference, 0, result, error));
    auto missing_coverage = partial_cohort;
    for (auto & fixture : missing_coverage) fixture.union_arm_results[1].progress->candidate.regression_known = false;
    CHECK(common_flydelta_guidance_qualify_deep_selection(context(), missing_coverage, 0, result, error));
    CHECK(result.status != common_flydelta_guidance_qualification_status::qualified && result.guided_improved == 0);
    auto regressed = partial_cohort;
    regressed.front().union_arm_results[1].progress->candidate.has_regression = true;
    CHECK(common_flydelta_guidance_qualify_deep_selection(context(), regressed, 0, result, error));
    CHECK(result.status != common_flydelta_guidance_qualification_status::qualified && result.guided_progress_regressed == 1);
    auto partial_context = context(); partial_context.progress_reward = 0;
    CHECK(common_flydelta_guidance_qualify_deep_selection(partial_context, partial_cohort, 0, result, error));
    CHECK(result.status != common_flydelta_guidance_qualification_status::qualified);
    auto reward_config = common_flydelta_outcome_guidance_config{};
    const auto previous_policy = common_flydelta_guidance_qualification_policy_revision(reward_config);
    reward_config.progress_reward = 0;
    CHECK(previous_policy != common_flydelta_guidance_qualification_policy_revision(reward_config));
    std::cout << "Deep selection qualification contracts passed\n";
    return 0;
}
