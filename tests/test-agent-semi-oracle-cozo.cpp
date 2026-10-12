#include "semi-oracle-store-cozo.h"
#include "agent/adaptation/semi-oracle-tool.h"
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
    CHECK(common_semi_oracle_parse_ruleset_dsl("name: TTT\ndescription: test\ndomain: placement_game\nrows: 3\ncolumns: 3\nwin_length: 3\nplayers: X, O\n","ruleset://test/1","session-1",r,error));
    CHECK(store.put_revision(r,error));
    common_semi_oracle_ruleset loaded;bool found=false;
    CHECK(store.get_revision(r.id,1,loaded,found,error)&&found&&loaded.name=="TTT");
    auto changed=r;changed.name="mutated";
    CHECK(!store.put_revision(changed,error));
    CHECK(store.bind_session("session-1",r.id,1,error));
    std::string id;uint64_t revision=0;
    CHECK(store.get_session_binding("session-1",id,revision,found,error)&&found&&id==r.id&&revision==1);
    common_semi_oracle_tool_service service(store,"host","session-1");
    std::vector<common_semi_oracle_placement_action_value> ranked;
    CHECK(service.rank_placements(
        R"json({"cells":["X","X",null,"O","O",null,null,null,null],"turn":"X"})json",
        {10000,9},ranked,error));
    CHECK(!ranked.empty()&&!ranked.front().persisted_evaluation_ref.empty());
    common_training_candidate candidate;
    CHECK(service.make_sft_candidate(ranked.front(), "Play the strongest move.",
        "stub:redaction", "fixture-with-no-user-data", false, candidate, error));
    CHECK(candidate.status==common_training_candidate_status::eligible);
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
    std::filesystem::remove_all(dir,ec);
    return 0;
}
