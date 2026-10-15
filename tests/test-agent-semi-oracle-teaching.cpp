#include "agent/adaptation/semi-oracle-teaching.h"
#include "agent/adaptation/semi-oracle-sft-candidate.h"
#include "agent/adaptation/flydelta/flydelta-capture.h"
#include "agent/adaptation/trainer-protocol.h"

#include <algorithm>
#include <nlohmann/json.hpp>
#define CHECK(x) do { if (!(x)) return __LINE__; } while (false)

int main() {
    std::string error;
    common_semi_oracle_ruleset ruleset;
    CHECK(common_semi_oracle_parse_ruleset_dsl(
        "name: Three in a row\ndescription: test\ndomain: placement_game\nrows: 3\ncolumns: 3\nwin_length: 3\nplayers: X, O\n",
        "ruleset://host/test", "session-test", ruleset, error,
        common_semi_oracle_rule_authority::declared));
    common_semi_oracle_result rejected, accepted;
    for (auto * result : {&rejected, &accepted}) {
        result->ruleset_id = ruleset.id;
        result->ruleset_revision = ruleset.revision;
        result->evaluator_ref = "semi-oracle://evaluator/placement_game";
        result->evaluator_revision = "v1";
    }
    rejected.verdict = common_semi_oracle_verdict::violated;
    const auto game_state = nlohmann::json::parse(
        R"json({"cells":["X","X",null,"O","O",null,null,null,null],"turn":"X"})json");
    rejected.counterexample_json = nlohmann::json{
        {"state", game_state}, {"action", {{"row", 2}, {"column", 2}}},
        {"violation", "chosen move misses immediate win"},
    }.dump();
    accepted.verdict = common_semi_oracle_verdict::satisfied;
    accepted.proof_json = nlohmann::json{
        {"state", game_state}, {"action", {{"row", 0}, {"column", 2}}},
        {"next_state", nlohmann::json::object()},
    }.dump();

    common_flydelta_teaching_relation relation;
    common_agent_scope scope;
    scope.namespace_id = "host";
    scope.session_id = "session-test";
    CHECK(common_semi_oracle_make_teaching_relation(ruleset, rejected, accepted,
        scope, "relation-1", "task://position-1", "run://bad", "run://good", "evidence://pair",
        relation, error));
    CHECK(relation.source == common_adaptation_evidence_source::semi_oracle_preference);
    CHECK(relation.host_approved && relation.oracle_revision.find(ruleset.semantic_fingerprint) != std::string::npos);
    CHECK(relation.task_fingerprint.rfind("fnv1a64:", 0) == 0);
    auto mismatched_state = accepted;
    mismatched_state.proof_json = nlohmann::json{
        {"state", {{"cells", {"X", "X", nullptr, "O", "O", nullptr, nullptr, nullptr, nullptr}},
            {"turn", "O"}}},
        {"action", {{"row", 0}, {"column", 2}}},
    }.dump();
    CHECK(!common_semi_oracle_make_teaching_relation(ruleset, rejected, mismatched_state,
        scope, "relation-mismatched-state", "task://position-1", "run://bad", "run://good",
        "evidence://pair", relation, error));
    CHECK(!common_semi_oracle_make_teaching_relation(ruleset, rejected, accepted,
        scope, "relation-missing-capture", "task://position-1", "", "run://good",
        "evidence://pair", relation, error));
    auto other_session_scope = scope;
    other_session_scope.session_id = "session-other";
    CHECK(!common_semi_oracle_make_teaching_relation(ruleset, rejected, accepted,
        other_session_scope, "relation-wrong-session", "task://position-1", "run://bad",
        "run://good", "evidence://pair", relation, error));
    accepted.ruleset_revision++;
    CHECK(!common_semi_oracle_make_teaching_relation(ruleset, rejected, accepted,
        scope, "relation-2", "task://position-1", "run://bad", "run://good", "evidence://pair",
        relation, error));

    common_semi_oracle_ruleset algebra_ruleset;
    CHECK(common_semi_oracle_parse_ruleset_dsl(
        "name: Linear equations\ndescription: exact linear equality\ndomain: linear_algebra\nvariable: x\n",
        "ruleset://host/algebra", "session-algebra", algebra_ruleset, error,
        common_semi_oracle_rule_authority::declared));
    common_agent_scope algebra_scope = scope;
    algebra_scope.session_id = "session-algebra";
    common_semi_oracle_result algebra_rejected, algebra_accepted;
    for (auto * result : {&algebra_rejected, &algebra_accepted}) {
        result->ruleset_id = algebra_ruleset.id;
        result->ruleset_revision = algebra_ruleset.revision;
        result->evaluator_ref = "semi-oracle://evaluator/linear_algebra";
        result->evaluator_revision = "v1";
    }
    algebra_rejected.verdict = common_semi_oracle_verdict::violated;
    algebra_rejected.counterexample_json = nlohmann::json{
        {"before", {{"a", "2"}, {"b", "6"}}},
        {"after", {{"a", "2"}, {"b", "3"}}},
    }.dump();
    algebra_accepted.verdict = common_semi_oracle_verdict::satisfied;
    algebra_accepted.proof_json = nlohmann::json{
        {"before", {{"a", "2"}, {"b", "6"}}},
        {"after", {{"a", "2"}, {"b", "6"}}},
        {"next_state", {{"lhs", "2x+6"}, {"rhs", "10"}}},
    }.dump();
    CHECK(common_semi_oracle_make_teaching_relation(algebra_ruleset,
        algebra_rejected, algebra_accepted, algebra_scope, "relation-algebra",
        "ignored-caller-task-fingerprint", "run://bad-rewrite", "run://valid-rewrite",
        "evidence://algebra-pair", relation, error));
    CHECK(relation.source == common_adaptation_evidence_source::semi_oracle_preference);
    CHECK(relation.verifier_ref == algebra_accepted.evaluator_ref);
    CHECK(common_semi_oracle_make_controlled_teaching_relation(algebra_ruleset,
        algebra_rejected, algebra_accepted, algebra_scope, "relation-algebra-controlled",
        "ignored-caller-task-fingerprint", "run://bad-rewrite", "run://valid-rewrite",
        "run://control-rewrite", "evidence://algebra-pair", relation, error));
    CHECK(relation.control_ref == "run://control-rewrite" &&
        relation.control_origin == common_flydelta_teaching_origin::host_counterfactual);
    CHECK(!common_semi_oracle_make_controlled_teaching_relation(algebra_ruleset,
        algebra_rejected, algebra_accepted, algebra_scope, "relation-algebra-no-control",
        "ignored-caller-task-fingerprint", "run://bad-rewrite", "run://valid-rewrite",
        "", "evidence://algebra-pair", relation, error));
    common_learning_transaction transaction;
    transaction.id = "learning://semi-oracle/algebra";
    transaction.observation.id = transaction.id;
    transaction.observation.source_turn_id = "turn://algebra";
    transaction.observation.source_plan_id = "plan://algebra";
    transaction.observation.signals.push_back({common_learning_signal_type::semi_oracle_preference,
        "plan://algebra", "rewrite", "oracle.action.verify", "evidence://algebra-pair",
        "verified algebra rewrite contrast", "semi-oracle", "host"});
    common_adaptation_evidence_relation evidence_relation;
    CHECK(common_agent_teaching_relation_to_evidence_relation(
        relation, transaction, evidence_relation, error));
    common_agent_request request;
    request.namespace_id = algebra_scope.namespace_id;
    request.session_id = algebra_scope.session_id;
    request.turn_id = "turn://algebra";
    common_agent_result observed_result;
    observed_result.learning_signals = transaction.observation.signals;
    common_adaptation_evidence evidence;
    CHECK(common_adaptation_evidence_from_turn(request, {}, observed_result,
        evidence_relation, evidence, error));
    common_flydelta_capture_candidate_collector collector(
        "model-profile:test", "capture-layout:test", 4);
    CHECK(collector.observe_verified_relation(evidence_relation, evidence, transaction, error));
    CHECK(collector.candidates().size() == 1);
    CHECK(collector.candidates().front().source == common_adaptation_evidence_source::semi_oracle_preference);
    algebra_accepted.proof_json = nlohmann::json{
        {"before", {{"a", "7"}, {"b", "6"}}},
        {"after", {{"a", "7"}, {"b", "6"}}},
    }.dump();
    CHECK(!common_semi_oracle_make_teaching_relation(algebra_ruleset,
        algebra_rejected, algebra_accepted, algebra_scope, "relation-algebra-mismatch",
        "task://other", "run://bad-rewrite", "run://valid-rewrite",
        "evidence://algebra-pair", relation, error));
    auto unknown_rewrite = algebra_rejected;
    unknown_rewrite.verdict = common_semi_oracle_verdict::unknown;
    unknown_rewrite.counterexample_json.clear();
    CHECK(!common_semi_oracle_make_teaching_relation(algebra_ruleset,
        unknown_rewrite, algebra_accepted, algebra_scope, "relation-algebra-unknown",
        "task://unknown", "run://unknown", "run://valid-rewrite",
        "evidence://algebra-pair", relation, error));
    auto inferred_algebra = algebra_ruleset;
    inferred_algebra.assertions.front().authority = common_semi_oracle_rule_authority::inferred;
    CHECK(!common_semi_oracle_make_teaching_relation(inferred_algebra,
        algebra_rejected, algebra_accepted, algebra_scope, "relation-algebra-inferred",
        "task://inferred", "run://bad-rewrite", "run://valid-rewrite",
        "evidence://algebra-pair", relation, error));

    std::vector<common_semi_oracle_placement_action_value> values;
    const std::string state = R"json({"cells":["X","X",null,"O","O",null,null,null,null],"turn":"X"})json";
    CHECK(common_semi_oracle_rank_placements(ruleset, state, {2000, 9}, values, error));
    const auto preferred = std::find_if(values.begin(), values.end(),
        [](const auto & value) { return value.row == 0 && value.column == 2; });
    const auto rejected_move = std::find_if(values.begin(), values.end(),
        [](const auto & value) { return value.row == 2 && value.column == 2; });
    CHECK(preferred != values.end() && rejected_move != values.end());
    auto persisted_preferred = *preferred;
    auto persisted_rejected = *rejected_move;
    persisted_preferred.persisted_evaluation_ref = "semi-oracle://evaluation/preference";
    persisted_rejected.persisted_evaluation_ref = persisted_preferred.persisted_evaluation_ref;
    persisted_preferred.ranking_complete = true;
    persisted_rejected.ranking_complete = true;
    CHECK(common_semi_oracle_make_preference_relation(ruleset, persisted_rejected, persisted_preferred,
        scope, "relation-preference", "run://suboptimal", "run://optimal",
        "run://matched-control", "semi-oracle://evaluation/preference", relation, error));
    CHECK(relation.task_fingerprint == preferred->state_fingerprint);
    CHECK(relation.control_ref == "run://matched-control");
    CHECK(relation.contrast_ref.find("<") != std::string::npos);
    CHECK(relation.source == common_adaptation_evidence_source::semi_oracle_preference);
    CHECK(relation.oracle_revision.find(ruleset.action_contract_fingerprint) != std::string::npos);
    CHECK(relation.teaching_key.find(ruleset.action_contract_fingerprint) != std::string::npos);
    common_flydelta_teaching_relation relation_without_control;
    CHECK(!common_semi_oracle_make_preference_relation(ruleset, persisted_rejected, persisted_preferred,
        scope, "relation-preference-no-control", "run://suboptimal", "run://optimal",
        "", "semi-oracle://evaluation/preference", relation_without_control, error));
    nlohmann::json ranking_payload = {
        {"algorithm", "memoized_minimax_v1"},
        {"state_fingerprint", persisted_preferred.state_fingerprint},
        {"action_contract_fingerprint", ruleset.action_contract_fingerprint},
        {"actions", nlohmann::json::array()},
    };
    for (const auto & value : values) {
        ranking_payload["actions"].push_back({{"player", value.player}, {"row", value.row},
            {"column", value.column}, {"value", static_cast<int>(value.value)},
            {"proof_json", value.proof_json}});
    }
    const auto persisted_payload = nlohmann::json{
        {"ruleset_id", ruleset.id}, {"ruleset_revision", ruleset.revision},
        {"verdict", "SATISFIED"}, {"evaluator_ref", "semi-oracle://evaluator/placement_game"},
        {"evaluator_revision", "minimax-v1"}, {"proof_json", ranking_payload.dump()},
    }.dump();
    auto forged_preferred = persisted_preferred;
    forged_preferred.proof_json = "{}";
    CHECK(!common_semi_oracle_make_preference_relation(ruleset, persisted_rejected, forged_preferred,
        scope, "relation-forged", "run://suboptimal", "run://optimal",
        "run://matched-control", "semi-oracle://evaluation/preference", relation, error));
    auto equal_value = persisted_preferred;
    equal_value.row = 2;
    equal_value.column = 1;
    equal_value.value = common_semi_oracle_placement_value::win;
    CHECK(!common_semi_oracle_make_preference_relation(ruleset, persisted_preferred, equal_value,
        scope, "relation-tie", "run://one", "run://two", "run://control",
        "semi-oracle://evaluation/preference", relation, error));
    persisted_preferred.persisted_evaluation_ref = "semi-oracle://evaluation/test";
    persisted_rejected.persisted_evaluation_ref = persisted_preferred.persisted_evaluation_ref;
    persisted_preferred.ranking_complete = false;
    common_training_candidate sft_candidate;
    CHECK(!common_semi_oracle_make_sft_candidate(ruleset, persisted_preferred, persisted_payload, scope,
        "Play X on the current board and choose the move with best outcome.",
        "stub:test-fixture", "test fixture has no user content", true, sft_candidate, error));
    persisted_preferred.ranking_complete = true;
    CHECK(common_semi_oracle_make_sft_candidate(ruleset, persisted_preferred, persisted_payload, scope,
        "Play X on the current board and choose the move with best outcome.",
        "stub:test-fixture", "test fixture has no user content", false, sft_candidate, error));
    CHECK(sft_candidate.provenance == common_training_candidate_provenance::host_verified_demonstration);
    CHECK(sft_candidate.status == common_training_candidate_status::eligible);
    CHECK(sft_candidate.observed_occurrences == 0 && sft_candidate.verified_recoveries == 0);
    auto inferred_ruleset = ruleset;
    inferred_ruleset.assertions.front().authority = common_semi_oracle_rule_authority::inferred;
    common_training_candidate inferred_candidate;
    CHECK(!common_semi_oracle_make_sft_candidate(inferred_ruleset, persisted_preferred,
        persisted_payload, scope, "Play X on the current board and choose the move with best outcome.",
        "stub:test-fixture", "test fixture has no user content", true, inferred_candidate, error));
    common_training_candidate_policy candidate_policy;
    CHECK(!common_training_candidate_qualifies(sft_candidate, candidate_policy, error));
    CHECK(common_semi_oracle_make_sft_candidate(ruleset, persisted_preferred, persisted_payload, scope,
        "Play X on the current board and choose the move with best outcome.",
        "stub:test-fixture", "test fixture has no user content", true, sft_candidate, error));
    candidate_policy.allow_host_verified_demonstrations = true;
    CHECK(common_training_candidate_qualifies(sft_candidate, candidate_policy, error));
    CHECK(sft_candidate.split_group_key.rfind("semi-oracle://position-group/", 0) == 0);
    common_learning_corpus_policy sft_policy;
    sft_policy.redaction_policy_id = sft_candidate.redaction_policy_id;
    common_learning_corpus_revision sft_corpus;
    CHECK(common_learning_build_corpus({sft_candidate}, sft_policy, sft_corpus, error));
    const auto sft_row = nlohmann::json::parse(sft_corpus.jsonl);
    CHECK(sft_row.at("input") == sft_candidate.approved_prompt);
    CHECK(sft_row.at("target") == sft_candidate.approved_target);
    CHECK(sft_row.at("split") == "train");

    common_training_candidate dpo_candidate;
    const bool dpo_built = common_semi_oracle_make_dpo_candidate(ruleset, persisted_preferred,
        persisted_rejected, persisted_payload, scope,
        "Choose a move with the best proven outcome from this board.",
        "stub:test-fixture", "test fixture has no user content", true,
        dpo_candidate, error);
    CHECK(dpo_built);
    CHECK(!dpo_candidate.rejected_target.empty());
    CHECK(dpo_candidate.approved_target != dpo_candidate.rejected_target);
    CHECK(dpo_candidate.contrast_ref.find("minimax-v1:") == 0);
    CHECK(dpo_candidate.transaction_ids.size() == 2);
    auto equal_value_rejected = persisted_preferred;
    equal_value_rejected.row = 2;
    equal_value_rejected.column = 1;
    equal_value_rejected.value = common_semi_oracle_placement_value::win;
    const auto equal_value_action = std::find_if(values.begin(), values.end(),
        [](const auto & value) { return value.row == 2 && value.column == 1; });
    CHECK(equal_value_action != values.end());
    equal_value_rejected.proof_json = equal_value_action->proof_json;
    common_training_candidate invalid_dpo_candidate;
    CHECK(!common_semi_oracle_make_dpo_candidate(ruleset, persisted_preferred,
        equal_value_rejected, persisted_payload, scope,
        "Choose a move with the best proven outcome from this board.",
        "stub:test-fixture", "test fixture has no user content", true,
        invalid_dpo_candidate, error));
    CHECK(invalid_dpo_candidate.id.empty());
    auto dpo_policy = sft_policy;
    dpo_policy.objective = common_learning_objective::dpo;
    common_learning_corpus_revision dpo_corpus;
    CHECK(!common_learning_build_corpus({dpo_candidate}, dpo_policy, dpo_corpus, error));
    dpo_policy.allow_host_verified_dpo_pairs = true;
    const bool dpo_corpus_built = common_learning_build_corpus({dpo_candidate}, dpo_policy, dpo_corpus, error);
    CHECK(dpo_corpus_built);
    const auto dpo_row = nlohmann::json::parse(dpo_corpus.jsonl);
    CHECK(dpo_row.at("prompt") == dpo_candidate.approved_prompt);
    CHECK(dpo_row.at("chosen") == dpo_candidate.approved_target);
    CHECK(dpo_row.at("rejected") == dpo_candidate.rejected_target);

    common_learning_training_job sft_job;
    CHECK(common_learning_make_training_job(sft_corpus, "base:qwen-instruct:training-v1",
        "semi-oracle-sft-v1", "fake-sft", "corpus-contract-test", sft_job, error));
    CHECK(sft_job.objective == common_learning_objective::sft);
    common_learning_training_job dpo_job;
    CHECK(common_learning_make_training_job(dpo_corpus, "base:qwen-instruct:training-v1",
        "semi-oracle-dpo-v1", "fake-dpo", "corpus-contract-test", dpo_job, error));
    CHECK(dpo_job.objective == common_learning_objective::dpo);
    CHECK(dpo_job.corpus_revision_id == dpo_corpus.id);
    common_learning_corpus_revision invalid_dpo_corpus;
    CHECK(!common_learning_build_corpus({sft_candidate}, dpo_policy, invalid_dpo_corpus, error));
    return 0;
}
