#include "agent/adaptation/flydelta/experiment/flydelta-guidance-qualification.h"

#include <iostream>

#define CHECK(condition) do { if (!(condition)) { std::cerr << "failed line " << __LINE__ << '\n'; return __LINE__; } } while (false)

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
    return c;
}

static std::vector<common_flydelta_guidance_fixture_evaluation> cohort(bool guided_wins) {
    std::vector<common_flydelta_guidance_fixture_evaluation> values;
    for (size_t i = 0; i < 24; ++i) {
        common_flydelta_guidance_fixture_evaluation fixture;
        fixture.fixture_group_id = "fixture:" + std::to_string(i);
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
        context(), cohort(true), 100, result, error));
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
    std::cout << "Deep selection qualification contracts passed\n";
    return 0;
}
