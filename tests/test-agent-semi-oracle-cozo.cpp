#include "semi-oracle-store-cozo.h"
#include "agent/adaptation/semi-oracle-tool.h"
#include "agent/adaptation/flydelta/flydelta-teaching-relation.h"
#include "agent/adaptation/training-candidate.h"

#include <filesystem>
#include <string>
#include <chrono>
#include <cstdio>

#define CHECK(x) do { if (!(x)) return __LINE__; } while (false)

int main() {
    const auto nonce=std::chrono::steady_clock::now().time_since_epoch().count();
    const auto dir=std::filesystem::temp_directory_path()/("llama-agent-semi-oracle-cozo-test-"+std::to_string(nonce));
    std::error_code ec;std::filesystem::create_directories(dir,ec);
    CHECK(!ec);
    const auto db=(dir/"semi-oracle.db").string();std::string error;
    agent_semi_oracle_cozo_store store;CHECK(store.open(db,error));
    common_semi_oracle_ruleset r;
    CHECK(common_semi_oracle_parse_ruleset_dsl("name: TTT\ndescription: test\ndomain: placement_game\nrows: 3\ncolumns: 3\nwin_length: 3\nplayers: X, O\n","ruleset://test/1","session-1",r,error, common_semi_oracle_rule_authority::declared));
    CHECK(store.put_revision(r,error));
    common_semi_oracle_ruleset loaded;bool found=false;
    CHECK(store.get_revision(r.id,1,loaded,found,error)&&found&&loaded.name=="TTT");
    auto changed=r;changed.name="mutated";
    CHECK(!store.put_revision(changed,error));
    CHECK(store.bind_session("session-1",r.id,1,error));
    std::string id;uint64_t revision=0;
    CHECK(store.get_session_binding("session-1",id,revision,found,error)&&found&&id==r.id&&revision==1);
    common_semi_oracle_tool_service play_service(store,"host","session-play");
    common_semi_oracle_ruleset active_game;
    CHECK(play_service.define_ruleset(
        "name: Session game\ndescription: persisted board test\ndomain: placement_game\nrows: 3\ncolumns: 3\nwin_length: 3\nplayers: X, O\n",
        common_semi_oracle_rule_authority::declared, active_game, error));
    common_semi_oracle_result turn_result;
    std::string before, after;
    CHECK(play_service.verify_active_placement_action(
        "open! game.place_mark\nplayer: X\nrow: 0\ncolumn: 0\n",
        turn_result, before, after, error));
    CHECK(turn_result.verdict == common_semi_oracle_verdict::satisfied &&
        before.find("\"turn\":\"X\"") != std::string::npos &&
        after.find("\"turn\":\"O\"") != std::string::npos);
    CHECK(play_service.verify_active_placement_action(
        "open! game.place_mark\nplayer: O\nrow: 0\ncolumn: 0\n",
        turn_result, before, after, error));
    CHECK(turn_result.verdict == common_semi_oracle_verdict::violated && after.empty());
    common_semi_oracle_tool_service algebra_service(store,"host","session-algebra");
    common_semi_oracle_ruleset active_algebra;
    CHECK(algebra_service.define_ruleset(
        "name: Session algebra\ndescription: persisted equation test\ndomain: linear_algebra\nvariable: x\n",
        common_semi_oracle_rule_authority::declared, active_algebra, error));
    CHECK(algebra_service.initialize_active_state(
        R"json({"lhs":"2(x+3)","rhs":"10"})json", false, error));
    CHECK(algebra_service.verify_active_action(
        "open! algebra.rewrite_equation\nlhs: 2x+6\n",
        turn_result, before, after, error));
    CHECK(turn_result.verdict == common_semi_oracle_verdict::satisfied &&
        after.find("2x+6") != std::string::npos);
    common_semi_oracle_tool_service service(store,"host","session-1");
    std::vector<common_semi_oracle_placement_action_value> ranked;
    CHECK(service.rank_placements(
        R"json({"cells":["X","X",null,"O","O",null,null,null,null],"turn":"X"})json",
        {10000,9},ranked,error));
    CHECK(!ranked.empty()&&!ranked.front().persisted_evaluation_ref.empty());
    const auto preferred = std::max_element(ranked.begin(), ranked.end(), [](const auto & left, const auto & right) {
        return static_cast<int>(left.value) < static_cast<int>(right.value);
    });
    const auto rejected = std::min_element(ranked.begin(), ranked.end(), [](const auto & left, const auto & right) {
        return static_cast<int>(left.value) < static_cast<int>(right.value);
    });
    CHECK(preferred != ranked.end() && rejected != ranked.end() &&
        static_cast<int>(preferred->value) > static_cast<int>(rejected->value));
    common_flydelta_teaching_relation preference;
    const bool relation_built = service.make_preference_relation(*rejected, *preferred,
        "relation://cozo/preference", "run://model/suboptimal", "run://solver/preferred",
        "run://control/matched", preferred->persisted_evaluation_ref + "/contrast",
        preference, error);
    CHECK(relation_built);
    CHECK(preference.host_approved && preference.baseline_origin == common_flydelta_teaching_origin::observed &&
        preference.conditioned_origin == common_flydelta_teaching_origin::host_derived &&
        preference.control_origin == common_flydelta_teaching_origin::host_counterfactual);
    auto forged = *rejected;
    forged.proof_json = "{}";
    CHECK(!service.make_preference_relation(forged, *preferred,
        "relation://cozo/forged", "run://model/suboptimal", "run://solver/preferred",
        "run://control/matched", preferred->persisted_evaluation_ref + "/contrast",
        preference, error));
    common_training_candidate candidate;
    CHECK(service.make_sft_candidate(ranked.front(), "Play the strongest move.",
        "stub:redaction", "fixture-with-no-user-data", false, candidate, error));
    CHECK(candidate.status==common_training_candidate_status::eligible);
    const bool dpo_candidate_built = service.make_dpo_candidate(*preferred, *rejected,
        "Play X on this board; choose the move with the best proven outcome.",
        "stub:redaction", "fixture-with-no-user-data", true, candidate, error);
    CHECK(dpo_candidate_built && !candidate.rejected_target.empty() &&
        candidate.contrast_ref.find("minimax-v1:") == 0 &&
        candidate.transaction_ids.size() == 2);
    auto unpersisted=ranked.front();
    unpersisted.persisted_evaluation_ref="semi-oracle://evaluation/not-present";
    CHECK(!service.make_sft_candidate(unpersisted, "Play the strongest move.",
        "stub:redaction", "fixture-with-no-user-data", true, candidate, error));
    std::string stored_payload;
    CHECK(store.get_evaluation_payload(ranked.front().persisted_evaluation_ref,stored_payload,found,error)&&found);
    CHECK(stored_payload.find("memoized_minimax_v1")!=std::string::npos);
    common_semi_oracle_interaction interaction;
    interaction.id = "semi-oracle://interaction/test-1";
    interaction.session_id = "session-1";
    interaction.ruleset_id = r.id;
    interaction.ruleset_revision = r.revision;
    interaction.action_contract_fingerprint = r.action_contract_fingerprint;
    interaction.domain = "placement_game";
    interaction.contrast_kind = "strategic_preference";
    interaction.evidence_ref = ranked.front().persisted_evaluation_ref;
    interaction.verifier_ref = "semi-oracle://evaluator/placement_game";
    interaction.verifier_revision = "minimax-v1";
    interaction.task_fingerprint = ranked.front().state_fingerprint;
    interaction.state_json = R"json({"cells":["X","X",null,"O","O",null,null,null,null],"turn":"X"})json";
    interaction.control_state_json = interaction.state_json;
    interaction.baseline_action_dsl = "open! game.place_mark\nplayer: X\nrow: 2\ncolumn: 0";
    interaction.preferred_action_dsl = "open! game.place_mark\nplayer: X\nrow: 0\ncolumn: 2";
    interaction.control_action_dsl = "open! game.place_mark\nplayer: X\nrow: 2\ncolumn: 1";
    interaction.baseline_result_json = R"json({"verdict":"SATISFIED"})json";
    interaction.preferred_result_json = R"json({"verdict":"SATISFIED"})json";
    interaction.contrast_ref = "minimax:test";
    if (!store.put_interaction(interaction, error)) {
        std::fprintf(stderr, "interaction persistence failed: %s\n", error.c_str());
        return __LINE__;
    }
    common_semi_oracle_interaction loaded_interaction;
    CHECK(store.get_interaction(interaction.id, loaded_interaction, found, error) && found);
    CHECK(loaded_interaction.preferred_action_dsl == interaction.preferred_action_dsl);
    auto mutated_interaction = interaction;
    mutated_interaction.control_action_dsl += "\n# changed";
    CHECK(!store.put_interaction(mutated_interaction, error));
    CHECK(store.promote_to_host(r.id,1,error));
    auto hosts=store.list_host_rulesets(error);CHECK(hosts.size()==1&&hosts.front()==r.id);
    CHECK(store.get_revision(r.id,1,loaded,found,error)&&found&&loaded.scope==common_semi_oracle_scope::host&&loaded.semantic_fingerprint==r.semantic_fingerprint);
    common_semi_oracle_result verdict;verdict.ruleset_id=r.id;verdict.ruleset_revision=1;verdict.verdict=common_semi_oracle_verdict::satisfied;verdict.proof_json="{}";
    CHECK(store.append_evaluation("eval-1",verdict,error));
    common_semi_oracle_result goal_result;
    CHECK(algebra_service.search_active_algebra_goal(goal_result, error));
    CHECK(!goal_result.search_guidance_json.empty());
    std::vector<common_semi_oracle_search_history_record> search_history;
    CHECK(store.list_search_history(active_algebra.id, active_algebra.revision, 128, search_history, error));
    CHECK(search_history.size() == 1 && search_history[0].guidance_json == goal_result.search_guidance_json);
    store.close();
    CHECK(store.open(db,error));
    CHECK(store.list_search_history(active_algebra.id, active_algebra.revision, 128, search_history, error));
    CHECK(search_history.size() == 1 && search_history[0].guidance_json == goal_result.search_guidance_json);
    CHECK(store.list_search_history(active_algebra.id, active_algebra.revision + 1, 128, search_history, error));
    CHECK(search_history.empty());
    CHECK(store.get_interaction(interaction.id, loaded_interaction, found, error) && found);
    std::string persisted_state;
    CHECK(store.get_session_state("session-play", active_game.id,
        active_game.revision, persisted_state, found, error) && found);
    CHECK(persisted_state.find("\"turn\":\"O\"") != std::string::npos);
    common_semi_oracle_tool_service resumed(store,"host","session-play");
    CHECK(resumed.verify_active_placement_action(
        "open! game.place_mark\nplayer: O\nrow: 1\ncolumn: 1\n",
        turn_result, before, after, error));
    CHECK(turn_result.verdict == common_semi_oracle_verdict::satisfied);
    common_semi_oracle_tool_service resumed_algebra(store,"host","session-algebra");
    CHECK(resumed_algebra.verify_active_action(
        "open! algebra.rewrite_equation\nlhs: 2x\nrhs: 4\n",
        turn_result, before, after, error));
    CHECK(turn_result.verdict == common_semi_oracle_verdict::satisfied &&
        after.find("\"rhs\":\"4\"") != std::string::npos);
    store.close();
    std::filesystem::remove_all(dir,ec);
    return 0;
}
