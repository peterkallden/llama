#include "semi-oracle-store-cozo.h"
#include "agent/adaptation/semi-oracle-tool.h"
#include "agent/adaptation/flydelta/flydelta-teaching-relation.h"
#include "agent/adaptation/training-candidate.h"

#include <filesystem>
#include <string>
#include <chrono>

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
    CHECK(store.promote_to_host(r.id,1,error));
    auto hosts=store.list_host_rulesets(error);CHECK(hosts.size()==1&&hosts.front()==r.id);
    CHECK(store.get_revision(r.id,1,loaded,found,error)&&found&&loaded.scope==common_semi_oracle_scope::host&&loaded.semantic_fingerprint==r.semantic_fingerprint);
    common_semi_oracle_result verdict;verdict.ruleset_id=r.id;verdict.ruleset_revision=1;verdict.verdict=common_semi_oracle_verdict::satisfied;verdict.proof_json="{}";
    CHECK(store.append_evaluation("eval-1",verdict,error));
    store.close();
    CHECK(store.open(db,error));
    std::string persisted_state;
    CHECK(store.get_session_game_state("session-play", active_game.id,
        active_game.revision, persisted_state, found, error) && found);
    CHECK(persisted_state.find("\"turn\":\"O\"") != std::string::npos);
    common_semi_oracle_tool_service resumed(store,"host","session-play");
    CHECK(resumed.verify_active_placement_action(
        "open! game.place_mark\nplayer: O\nrow: 1\ncolumn: 1\n",
        turn_result, before, after, error));
    CHECK(turn_result.verdict == common_semi_oracle_verdict::satisfied);
    store.close();
    std::filesystem::remove_all(dir,ec);
    return 0;
}
