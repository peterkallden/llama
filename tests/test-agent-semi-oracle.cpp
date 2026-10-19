#include "agent/adaptation/semi-oracle.h"
#include "agent/adaptation/semi-oracle-search.h"
#include "agent/adaptation/semi-oracle-tool.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <map>
#include <string>
#include <tuple>
#include <stdexcept>
#include <cstdio>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); return 1; } } while (false)

class memory_store final : public common_semi_oracle_store {
public:
    bool put_revision(const common_semi_oracle_ruleset & r,std::string & error) override {
        const auto key=std::make_pair(r.id,r.revision);auto it=rules.find(key);
        if(it!=rules.end()){if(common_semi_oracle_ruleset_to_json(it->second)==common_semi_oracle_ruleset_to_json(r))return true;error="immutable";return false;}
        rules[key]=r;return true;
    }
    bool get_revision(const std::string & id,uint64_t rev,common_semi_oracle_ruleset & r,bool & found,std::string &) const override {
        const auto it=rules.find({id,rev});found=it!=rules.end();if(found)r=it->second;return true;
    }
    bool promote_to_host(const std::string & id,uint64_t rev,std::string &) override {
        auto it=rules.find({id,rev});if(it==rules.end())return false;it->second.scope=common_semi_oracle_scope::host;it->second.scope_id="host";host.push_back(id);return true;
    }
    bool bind_session(const std::string & session,const std::string & id,uint64_t rev,std::string &) override {bindings[session]={id,rev};return true;}
    bool get_session_binding(const std::string & session,std::string & id,uint64_t & rev,bool & found,std::string &) const override {
        const auto it=bindings.find(session);found=it!=bindings.end();if(found){id=it->second.first;rev=it->second.second;}return true;
    }
    bool get_session_game_state(const std::string & session,const std::string & ruleset,uint64_t revision,
            std::string & state,bool & found,std::string &) const override {
        const auto it=game_states.find({session,ruleset,revision});found=it!=game_states.end();
        if(found)state=it->second;else state.clear();return true;
    }
    bool put_session_game_state(const std::string & session,const std::string & ruleset,uint64_t revision,
            const std::string & state,std::string &) override {
        game_states[{session,ruleset,revision}]=state;return true;
    }
    bool append_evaluation(const std::string &,const common_semi_oracle_result & r,std::string &) override {evaluations.push_back(r);return true;}
    bool list_search_history(const std::string & id, uint64_t revision, size_t limit,
            std::vector<common_semi_oracle_search_history_record> & rows, std::string & error) const override {
        ++history_reads; rows.clear(); error.clear();
        if (fail_history) { error = "injected history error"; return false; }
        if (override_history) {
            rows = history_rows;
            if (rows.size() > limit) rows.resize(limit);
            return true;
        }
        for (size_t i = 0; i < evaluations.size() && rows.size() < limit; ++i)
            if (evaluations[i].ruleset_id == id && evaluations[i].ruleset_revision == revision &&
                    !evaluations[i].search_guidance_json.empty())
                rows.push_back({"evaluation-" + std::to_string(i), evaluations[i].search_guidance_json});
        return true;
    }
    mutable size_t history_reads = 0;
    bool fail_history = false;
    bool override_history = false;
    std::vector<common_semi_oracle_search_history_record> history_rows;
    std::vector<std::string> list_host_rulesets(std::string &) const override {return host;}
    std::map<std::pair<std::string,uint64_t>,common_semi_oracle_ruleset> rules;
    std::map<std::string,std::pair<std::string,uint64_t>> bindings;
    std::map<std::tuple<std::string,std::string,uint64_t>,std::string> game_states;
    std::vector<std::string> host;
    std::vector<common_semi_oracle_result> evaluations;
};

int main() {
    std::string error;
    common_semi_oracle_ruleset game;
    CHECK(common_semi_oracle_parse_ruleset_dsl(
        "name: Three in a row\ndescription: User supplied 3 by 3 game\ndomain: placement_game\nrows: 3\ncolumns: 3\nwin_length: 3\nplayers: X, O\n",
        "ruleset://host/a1", "session-1", game, error,
        common_semi_oracle_rule_authority::declared));
    CHECK(game.active && game.scope == common_semi_oracle_scope::session && game.revision == 1);
    CHECK(game.semantic_fingerprint.rfind("fnv1a64:",0)==0);
    CHECK(common_semi_oracle_action_contract(game).find("choose exactly one player value from: X or O")!=std::string::npos);

    common_semi_oracle_query q;
    q.ruleset_id=game.id;q.ruleset_revision=game.revision;q.state_json=R"({"cells":[null,null,null,null,null,null,null,null,null],"turn":"X"})";
    q.action_dsl="open! game.place_mark\nplayer: X\nrow: 0\ncolumn: 1\n";
    common_semi_oracle_result result;
    CHECK(common_semi_oracle_evaluate(game,q,result,error));
    CHECK(result.verdict==common_semi_oracle_verdict::satisfied&&!result.proof_json.empty());
    q.action_dsl="open! game.place_mark\nplayer: O\nrow: 0\ncolumn: 1\n";
    CHECK(common_semi_oracle_evaluate(game,q,result,error));
    CHECK(result.verdict==common_semi_oracle_verdict::violated&&!result.counterexample_json.empty());
    q.action_dsl="open! game.place_mark\nplayer: X\nrow: 8\ncolumn: 1\n";
    CHECK(common_semi_oracle_evaluate(game,q,result,error));
    CHECK(result.verdict==common_semi_oracle_verdict::violated);

    q.state_json=R"json({"cells":["X","X",null,"O","O",null,null,null,null],"turn":"X"})json";
    q.action_dsl="open! game.place_mark\nplayer: X\nrow: 0\ncolumn: 2\n";
    CHECK(common_semi_oracle_evaluate(game,q,result,error));
    CHECK(result.verdict==common_semi_oracle_verdict::satisfied&&result.proof_json.find("terminal_win")!=std::string::npos);
    q.level=common_semi_oracle_level::bounded;q.goal="win:X";
    q.state_json=R"json({"cells":["X","O","X","X","O","O","O","X","X"],"turn":"O"})json";
    CHECK(common_semi_oracle_evaluate(game,q,result,error));
    CHECK(result.verdict==common_semi_oracle_verdict::violated&&!result.counterexample_json.empty());
    q.state_json=R"json({"cells":["X",null,null,null,null,null,null,null,null],"turn":"X"})json";
    CHECK(common_semi_oracle_evaluate(game,q,result,error));
    CHECK(result.verdict==common_semi_oracle_verdict::unknown);

    std::vector<common_semi_oracle_placement_action_value> move_values;
    CHECK(common_semi_oracle_rank_placements(game,
        R"json({"cells":["X","X",null,"O","O",null,null,null,null],"turn":"X"})json",
        {2000, 9}, move_values, error));
    CHECK(move_values.size() == 5);
    const auto preferred_move = std::find_if(move_values.begin(), move_values.end(),
        [](const auto & move) { return move.row == 0 && move.column == 2; });
    const auto rejected_move = std::find_if(move_values.begin(), move_values.end(),
        [](const auto & move) { return move.row == 2 && move.column == 2; });
    CHECK(preferred_move != move_values.end() && rejected_move != move_values.end());
    CHECK(preferred_move->value == common_semi_oracle_placement_value::win);
    CHECK(rejected_move->value == common_semi_oracle_placement_value::loss);
    CHECK(preferred_move->proof_json.find("memoized_minimax_v1") != std::string::npos);
    CHECK(preferred_move->state_fingerprint == rejected_move->state_fingerprint);
    std::vector<common_semi_oracle_placement_action_value> draw_values;
    CHECK(common_semi_oracle_rank_placements(game,
        R"json({"cells":[null,null,null,null,null,null,null,null,null],"turn":"X"})json",
        {100000, 9}, draw_values, error));
    CHECK(draw_values.size() == 9);
    CHECK(std::all_of(draw_values.begin(), draw_values.end(), [](const auto & move) {
        return move.value == common_semi_oracle_placement_value::draw;
    }));
    memory_store store;
    std::string position_group, rotated_group;
    CHECK(common_semi_oracle_placement_state_group(game,
        R"json({"cells":["X",null,null,null,"O",null,null,null,null],"turn":"X"})json",
        position_group, error));
    CHECK(common_semi_oracle_placement_state_group(game,
        R"json({"cells":[null,null,"X",null,"O",null,null,null,null],"turn":"X"})json",
        rotated_group, error));
    CHECK(position_group == rotated_group);
    CHECK(store.put_revision(game, error));
    CHECK(store.bind_session("session-1", game.id, game.revision, error));
    common_semi_oracle_tool_service session_oracle(store, "host", "session-1");
    std::vector<common_semi_oracle_placement_action_value> persisted_values;
    CHECK(session_oracle.rank_placements(
        R"json({"cells":[null,null,null,null,null,null,null,null,null],"turn":"X"})json",
        {100000, 9}, persisted_values, error));
    CHECK(!persisted_values.front().persisted_evaluation_ref.empty());
    CHECK(persisted_values.front().ranking_complete);
    CHECK(store.evaluations.size() == 1 &&
        store.evaluations.front().proof_json.find("memoized_minimax_v1") != std::string::npos);
    common_semi_oracle_ruleset short_line_game;
    CHECK(common_semi_oracle_parse_ruleset_dsl(
        "name: Two in a row\ndescription: alternate small placement rule set\ndomain: placement_game\nrows: 3\ncolumns: 3\nwin_length: 2\nplayers: X, O\n",
        "ruleset://host/short-line", "session-short-line", short_line_game, error,
        common_semi_oracle_rule_authority::declared));
    std::vector<common_semi_oracle_placement_action_value> short_line_values;
    CHECK(common_semi_oracle_rank_placements(short_line_game,
        R"json({"cells":["X",null,null,null,"O",null,null,null,null],"turn":"X"})json",
        {1000, 8}, short_line_values, error));
    CHECK(std::any_of(short_line_values.begin(), short_line_values.end(), [](const auto & move) {
        return move.value == common_semi_oracle_placement_value::win;
    }));
    common_semi_oracle_ruleset rectangular_game;
    CHECK(common_semi_oracle_parse_ruleset_dsl(
        "name: Three across on a short board\ndescription: 2 by 3 placement game\ndomain: placement_game\nrows: 2\ncolumns: 3\nwin_length: 3\nplayers: X, O\n",
        "ruleset://host/rectangular", "session-rectangular", rectangular_game, error,
        common_semi_oracle_rule_authority::declared));
    std::vector<common_semi_oracle_placement_action_value> rectangular_values;
    CHECK(common_semi_oracle_rank_placements(rectangular_game,
        R"json({"cells":[null,null,null,null,null,null],"turn":"X"})json",
        {10000, 6}, rectangular_values, error));
    CHECK(rectangular_values.size() == 6);
    std::vector<common_semi_oracle_placement_action_value> incomplete_values;
    CHECK(common_semi_oracle_rank_placements(game,
        R"json({"cells":[null,null,null,null,null,null,null,null,null],"turn":"X"})json",
        {1, 1}, incomplete_values, error));
    CHECK(std::any_of(incomplete_values.begin(), incomplete_values.end(), [](const auto & move) {
        return move.value == common_semi_oracle_placement_value::unknown;
    }));
    CHECK(!common_semi_oracle_rank_placements(game,
        R"json({"cells":["X","X","X","O","O",null,null,null,null],"turn":"O"})json",
        {2000, 9}, incomplete_values, error));

    q.level=common_semi_oracle_level::bounded;
    q.state_json=R"({"cells":["X","X",null,"O","O",null,null,null,null],"turn":"X"})";
    q.goal="win:X";q.budget.max_depth=4;q.budget.max_expansions=100;
    CHECK(common_semi_oracle_evaluate(game,q,result,error));
    CHECK(result.verdict==common_semi_oracle_verdict::satisfied&&!result.proof_json.empty());
    const auto bounded=result;
    q.level=common_semi_oracle_level::guided;q.guidance={"0,2"};
    CHECK(common_semi_oracle_evaluate(game,q,result,error));
    CHECK(result.verdict==bounded.verdict&&!result.proof_json.empty()&&result.guidance_ref=="host-guidance:provided");
    CHECK(common_semi_oracle_learning_authorized(game,result));
    auto inferred=game;inferred.assertions.front().authority=common_semi_oracle_rule_authority::inferred;
    CHECK(!common_semi_oracle_learning_authorized(inferred,result));
    auto wrong_revision=q;wrong_revision.ruleset_revision++;
    CHECK(!common_semi_oracle_evaluate(game,wrong_revision,result,error));
    common_semi_oracle_ruleset algebra;
    CHECK(common_semi_oracle_parse_ruleset_dsl(
        "name: Linear equations\ndescription: Exact one variable linear equation rules\ndomain: linear_algebra\nvariable: x\n",
        "ruleset://host/a2", "session-2", algebra, error,
        common_semi_oracle_rule_authority::declared));
    common_semi_oracle_query aq;aq.ruleset_id=algebra.id;aq.ruleset_revision=1;
    aq.state_json=R"json({"lhs":"2(x+3)","rhs":"10"})json";
    aq.action_dsl="open! algebra.rewrite_equation\nlhs: 2(x + 3)\nrhs: 10\n";
    CHECK(common_semi_oracle_evaluate(algebra,aq,result,error));
    CHECK(result.verdict==common_semi_oracle_verdict::violated&&result.reason.find("no textual change")!=std::string::npos);
    aq.action_dsl="open! algebra.rewrite_equation\nlhs: 2x+6\nrhs: 10\n";
    CHECK(common_semi_oracle_evaluate(algebra,aq,result,error));
    CHECK(result.verdict==common_semi_oracle_verdict::satisfied);
    aq.action_dsl="lhs: 2x+6\n";
    CHECK(common_semi_oracle_evaluate(algebra,aq,result,error));
    CHECK(result.verdict==common_semi_oracle_verdict::satisfied&&result.proof_json.find("after")!=std::string::npos);
    aq.action_dsl="open! algebra.rewrite_equation\nlhs: 2x+3\nrhs: 10\n";
    CHECK(common_semi_oracle_evaluate(algebra,aq,result,error));
    CHECK(result.verdict==common_semi_oracle_verdict::violated);
    aq.state_json=R"json({"lhs":"2x+6","rhs":"10"})json";
    aq.action_dsl="open! algebra.rewrite_equation\nlhs: 2x\nrhs: 4\n";
    CHECK(common_semi_oracle_evaluate(algebra,aq,result,error));
    CHECK(result.verdict==common_semi_oracle_verdict::satisfied);
    aq.state_json=R"json({"lhs":"2(x+3)","rhs":"10"})json";
    aq.action_dsl="open! algebra.rewrite_equation\nlhs: x/0\nrhs: 1\n";
    CHECK(common_semi_oracle_evaluate(algebra,aq,result,error));
    CHECK(result.verdict==common_semi_oracle_verdict::unknown);
    aq.level=common_semi_oracle_level::bounded;aq.goal="solve:x";aq.action_dsl="open! algebra.rewrite_equation\nlhs: 2x+6\nrhs: 10\n";
    CHECK(common_semi_oracle_evaluate(algebra,aq,result,error));
    CHECK(result.verdict==common_semi_oracle_verdict::satisfied&&result.proof_json.find("solution")!=std::string::npos);
    const auto algebra_proof=nlohmann::ordered_json::parse(result.proof_json);
    CHECK(algebra_proof.value("algorithm",std::string())=="bounded_astar_v1");
    CHECK(algebra_proof.value("optimality_proven",false));
    CHECK(algebra_proof.contains("path")&&algebra_proof["path"].size()==4);
    CHECK(algebra_proof["path"][1].value("action",std::string())=="simplify lhs");
    CHECK(algebra_proof["path"][2].value("action",std::string())=="move constants to the right");
    CHECK(algebra_proof["path"][3].value("action",std::string())=="divide both sides by the non-zero coefficient");
    CHECK(result.evaluator_revision=="linear_astar_v1");
    auto bounded_algebra=aq;bounded_algebra.budget.max_depth=1;
    CHECK(common_semi_oracle_evaluate(algebra,bounded_algebra,result,error));
    CHECK(result.verdict==common_semi_oracle_verdict::unknown);
    auto guided_algebra=aq;guided_algebra.level=common_semi_oracle_level::guided;
    guided_algebra.guidance={"move constants to the right","simplify lhs"};
    CHECK(common_semi_oracle_evaluate(algebra,guided_algebra,result,error));
    CHECK(result.verdict==common_semi_oracle_verdict::satisfied);
    CHECK(common_semi_oracle_learning_authorized(algebra,result));
    CHECK(common_semi_oracle_validate_state(algebra,
        R"json({"lhs":"2(x+3)","rhs":"10"})json", error));
    CHECK(!common_semi_oracle_validate_state(algebra,
        R"json({"lhs":"x*x","rhs":"10"})json", error));

    const auto encoded=common_semi_oracle_ruleset_to_json(game);
    common_semi_oracle_ruleset roundtrip;
    CHECK(common_semi_oracle_ruleset_from_json(encoded,roundtrip,error));
    CHECK(roundtrip.id==game.id&&roundtrip.semantic_fingerprint==game.semantic_fingerprint);

    common_semi_oracle_tool_service service(store,"host-test","session-tool");
    common_semi_oracle_ruleset managed;
    CHECK(service.define_ruleset("name: Tool game\ndescription: session game\ndomain: placement_game\nrows: 3\ncolumns: 3\nwin_length: 3\nplayers: X, O\n",common_semi_oracle_rule_authority::declared,managed,error));
    CHECK(managed.id!=""&&managed.scope==common_semi_oracle_scope::session&&managed.active);
    std::string action_contract;CHECK(service.action_contract(action_contract,error)&&action_contract.find("row: integer, 0-based")!=std::string::npos);
    CHECK(service.verify_action(R"json({"cells":[null,null,null,null,null,null,null,null,null],"turn":"X"})json",
        "open! game.place_mark\nplayer: X\nrow: 0\ncolumn: 0\n",common_semi_oracle_level::local,"",
        common_semi_oracle_budget{}, {},result,error));
    CHECK(result.verdict==common_semi_oracle_verdict::satisfied&&store.evaluations.size()==2);
    common_semi_oracle_tool_service algebra_service(store,"host-test","session-algebra");
    common_semi_oracle_ruleset managed_algebra;
    CHECK(algebra_service.define_ruleset("name: Tool algebra\ndescription: session equation\ndomain: linear_algebra\nvariable: x\n",common_semi_oracle_rule_authority::declared,managed_algebra,error));
    CHECK(algebra_service.initialize_active_state(R"json({"lhs":"2(x+3)","rhs":"10"})json",false,error));
    common_semi_oracle_result searched_algebra;
    CHECK(algebra_service.search_active_algebra_goal(searched_algebra,error));
    CHECK(searched_algebra.verdict==common_semi_oracle_verdict::satisfied&&
        searched_algebra.evaluator_revision=="linear_astar_v1"&&
        searched_algebra.proof_json.find("bounded_astar_v1")!=std::string::npos);
    auto focal_algebra_config=common_semi_oracle_outcome_guidance_config{};
    focal_algebra_config.search_algorithm="focal";focal_algebra_config.suboptimality_bound=1.1;
    common_semi_oracle_tool_service focal_algebra_service(store,"host-test","session-algebra",
        focal_algebra_config,"project-test");
    CHECK(focal_algebra_service.search_active_goal(searched_algebra,error));
    CHECK(searched_algebra.applied_search_policy.algorithm==common_semi_oracle_search_algorithm::exact&&
        searched_algebra.applied_search_policy.suboptimality_bound==1.0);
    const auto first_guidance = nlohmann::ordered_json::parse(searched_algebra.search_guidance_json);
    CHECK(first_guidance["diagnostics"]["mode"] == "shadow");
    CHECK(first_guidance["diagnostics"]["fixture_groups"] == 0);
    CHECK(first_guidance["diagnostics"]["fallback_reason"] == "");
    common_semi_oracle_query history_query;
    history_query.ruleset_id = managed_algebra.id;
    history_query.ruleset_revision = managed_algebra.revision;
    history_query.level = common_semi_oracle_level::bounded;
    history_query.goal = "solve:x";
    history_query.budget = {2048, 16};
    history_query.state_json = R"({"lhs":"3x+6","rhs":"x+12"})";
    history_query.action_dsl = "lhs: 3x+6\n";
    common_semi_oracle_outcome_guidance_config guidance_config;
    CHECK(common_semi_oracle_outcome_guidance_config_validate(guidance_config,error));
    auto focal_config=guidance_config;focal_config.search_algorithm="focal";focal_config.suboptimality_bound=1.1;
    CHECK(common_semi_oracle_outcome_guidance_config_validate(focal_config,error));
    auto invalid_policy=guidance_config;invalid_policy.suboptimality_bound=1.1;
    CHECK(!common_semi_oracle_outcome_guidance_config_validate(invalid_policy,error));
    invalid_policy=guidance_config;invalid_policy.sampling_policy="sample-anything";
    CHECK(!common_semi_oracle_outcome_guidance_config_validate(invalid_policy,error));
    common_semi_oracle_result shadow_result, off_result, experimental_result;
    CHECK(common_semi_oracle_evaluate_with_history(store, guidance_config, "host-test", "session-algebra",
        managed_algebra, history_query, shadow_result, error));
    const auto first_projection_trace = nlohmann::ordered_json::parse(shadow_result.search_guidance_json);
    CHECK(first_projection_trace["diagnostics"].contains("projection_cache_hit"));
    CHECK(common_semi_oracle_evaluate_with_history(store, guidance_config, "host-test", "session-algebra",
        managed_algebra, history_query, shadow_result, error));
    CHECK(nlohmann::ordered_json::parse(shadow_result.search_guidance_json)["diagnostics"]["projection_cache_hit"] == true);
    const auto shadow_history = nlohmann::ordered_json::parse(shadow_result.search_guidance_json);
    CHECK(shadow_history["diagnostics"]["fixture_groups"] == 1);
    CHECK(!shadow_history["diagnostics"]["decisions"].empty());
    for (const auto & decision : shadow_history["diagnostics"]["decisions"])
        for (size_t i = 0; i < decision["executed_order"].size(); ++i)
            CHECK(decision["executed_order"][i] == i);
    const auto reads_before_off = store.history_reads;
    guidance_config.mode = "off";
    CHECK(common_semi_oracle_evaluate_with_history(store, guidance_config, "host-test", "session-algebra",
        managed_algebra, history_query, off_result, error));
    CHECK(store.history_reads == reads_before_off && off_result.search_guidance_json.empty());
    CHECK(off_result.proof_json == shadow_result.proof_json && off_result.verdict == shadow_result.verdict);
    guidance_config.mode = "experimental";
    CHECK(common_semi_oracle_evaluate_with_history(store, guidance_config, "host-test", "session-algebra",
        managed_algebra, history_query, experimental_result, error));
    CHECK(experimental_result.verdict == off_result.verdict);
    CHECK(nlohmann::ordered_json::parse(experimental_result.proof_json)["solution"] ==
        nlohmann::ordered_json::parse(off_result.proof_json)["solution"]);
    // Same fixture group cannot teach itself; other sessions cannot use it.
    auto own_fixture = history_query;
    own_fixture.state_json = R"json({"lhs":"2(x+3)","rhs":"10"})json";
    own_fixture.action_dsl = "lhs: 2(x+3)\n";
    CHECK(common_semi_oracle_evaluate_with_history(store, guidance_config, "host-test", "session-algebra",
        managed_algebra, own_fixture, experimental_result, error));
    const auto own_fixture_trace = nlohmann::ordered_json::parse(experimental_result.search_guidance_json);
    CHECK(own_fixture_trace["diagnostics"]["fixture_groups"] == 0);
    CHECK(own_fixture_trace["diagnostics"]["projection_cache_hit"] == true);
    CHECK(common_semi_oracle_evaluate_with_history(store, guidance_config, "host-test", "another-session",
        managed_algebra, history_query, experimental_result, error));
    CHECK(nlohmann::ordered_json::parse(experimental_result.search_guidance_json)["diagnostics"]["fixture_groups"] == 0);
    CHECK(common_semi_oracle_evaluate_with_history(store, guidance_config, "host-test", "session-algebra",
        managed_algebra, history_query, experimental_result, error, "different-project"));
    CHECK(nlohmann::ordered_json::parse(experimental_result.search_guidance_json)["diagnostics"]["fixture_groups"] == 0);
    // Repeating one fixture cannot turn it into multiple independent groups.
    store.evaluations.push_back(searched_algebra);
    CHECK(common_semi_oracle_evaluate_with_history(store, guidance_config, "host-test", "session-algebra",
        managed_algebra, history_query, experimental_result, error));
    const auto repeated_history = nlohmann::ordered_json::parse(experimental_result.search_guidance_json);
    CHECK(repeated_history["diagnostics"]["fixture_groups"] == 1);
    for (const auto & decision : repeated_history["diagnostics"]["decisions"])
        for (const auto & prediction : decision["predictions"])
            CHECK(prediction["effective_group_weight"].get<double>() <= 1.0 + 1e-12);
    auto shallow_query = history_query;
    shallow_query.budget.max_depth = 1;
    CHECK(common_semi_oracle_evaluate_with_history(store, guidance_config, "host-test", "session-algebra",
        managed_algebra, shallow_query, experimental_result, error));
    CHECK(experimental_result.verdict == common_semi_oracle_verdict::unknown);
    const auto bounded_history = nlohmann::ordered_json::parse(experimental_result.search_guidance_json);
    for (const auto & weights : bounded_history["observations"]) {
        CHECK(weights[0] == 0 && weights[1] == 0);
        CHECK(weights[2].get<double>() > 0.0);
    }
    auto inferred_rules = managed_algebra;
    inferred_rules.assertions.front().authority = common_semi_oracle_rule_authority::inferred;
    const auto reads_before_inferred = store.history_reads;
    CHECK(common_semi_oracle_evaluate_with_history(store, guidance_config, "host-test", "session-algebra",
        inferred_rules, history_query, experimental_result, error));
    CHECK(store.history_reads == reads_before_inferred && experimental_result.search_guidance_json.empty());
    store.override_history = true;
    store.history_rows = {{"retry", searched_algebra.search_guidance_json}, {"retry", searched_algebra.search_guidance_json}};
    CHECK(common_semi_oracle_evaluate_with_history(store, guidance_config, "host-test", "session-algebra",
        managed_algebra, history_query, experimental_result, error));
    const auto deduplicated = nlohmann::ordered_json::parse(experimental_result.search_guidance_json);
    CHECK(deduplicated["diagnostics"]["fixture_groups"] == 0);
    CHECK(deduplicated["diagnostics"]["duplicate_records"] == 1);
    auto conflict = nlohmann::ordered_json::parse(searched_algebra.search_guidance_json);
    conflict["observations"]["simplify lhs"] = {0.0, 1.0, 0.0};
    store.history_rows[1].guidance_json = conflict.dump();
    CHECK(common_semi_oracle_evaluate_with_history(store, guidance_config, "host-test", "session-algebra",
        managed_algebra, history_query, experimental_result, error));
    const auto conflicted = nlohmann::ordered_json::parse(experimental_result.search_guidance_json);
    CHECK(conflicted["diagnostics"]["fixture_groups"] == 0);
    CHECK(conflicted["diagnostics"]["conflicting_records"] == 1);
    store.history_rows = {{"malformed", "{}"}};
    CHECK(common_semi_oracle_evaluate_with_history(store, guidance_config, "host-test", "session-algebra",
        managed_algebra, history_query, experimental_result, error));
    CHECK(nlohmann::ordered_json::parse(experimental_result.search_guidance_json)["diagnostics"]["rejected_records"] == 1);
    store.override_history = false;
    store.fail_history = true;
    CHECK(common_semi_oracle_evaluate_with_history(store, guidance_config, "host-test", "session-algebra",
        managed_algebra, history_query, experimental_result, error));
    CHECK(experimental_result.proof_json == off_result.proof_json);
    CHECK(nlohmann::ordered_json::parse(experimental_result.search_guidance_json)["diagnostics"]["fallback_reason"] ==
        "search_history_unavailable");
    store.fail_history = false;
    std::string state_before,state_after;
    CHECK(algebra_service.verify_active_action("open! algebra.rewrite_equation\nlhs: 2x+6\n",result,state_before,state_after,error));
    CHECK(result.verdict==common_semi_oracle_verdict::satisfied&&
        state_before.find("2(x+3)")!=std::string::npos&&state_after.find("2x+6")!=std::string::npos);
    CHECK(algebra_service.verify_active_action("lhs: 2x+3\n",result,state_before,state_after,error));
    CHECK(result.verdict==common_semi_oracle_verdict::violated&&state_after.empty());
    common_semi_oracle_ruleset revised;
    CHECK(service.define_ruleset("name: Tool game corrected\ndescription: session game\ndomain: placement_game\nrows: 3\ncolumns: 3\nwin_length: 3\nplayers: X, O\n",common_semi_oracle_rule_authority::declared,revised,error));
    CHECK(revised.id==managed.id&&revised.revision==2&&revised.parent_revision==1);
    CHECK(service.promote_active_ruleset_to_host(error));
    CHECK(store.list_host_rulesets(error).size()==1);
    // A* advice changes only equal-priority successor ordering. Unexplored
    // siblings stay unknown; a leaf is a dead end only after expansion.
    common_semi_oracle_search_problem order_problem;
    order_problem.initial_state = "root";
    order_problem.heuristic_admissible = true;
    order_problem.record_history = true;
    order_problem.is_goal = [](const std::string & state, bool & goal, std::string &) {
        goal = state == "goal"; return true;
    };
    order_problem.expand = [](const std::string & state,
            std::vector<common_semi_oracle_search_transition> & transitions, std::string &) {
        transitions.clear();
        if (state == "root") transitions = {{"dead", "first", 1}, {"goal", "second", 1}};
        return true;
    };
    common_semi_oracle_search_result normal_order, advised_order;
    CHECK(common_semi_oracle_search_astar(order_problem, {8, 4}, normal_order));
    CHECK(normal_order.found && normal_order.observations[0].outcome == "dead_end");
    order_problem.transition_adviser = [](const std::string &, const std::vector<std::string> &,
            std::vector<size_t> & order, std::string &) { order = {1, 0}; return true; };
    CHECK(common_semi_oracle_search_astar(order_problem, {8, 4}, advised_order));
    CHECK(advised_order.found && advised_order.total_cost == normal_order.total_cost &&
        advised_order.expansions < normal_order.expansions);
    CHECK(std::any_of(advised_order.observations.begin(), advised_order.observations.end(),
        [](const auto & observation) { return observation.outcome == "unknown"; }));
    order_problem.transition_adviser = [](const std::string &, const std::vector<std::string> &,
            std::vector<size_t> & order, std::string &) { order = {0, 0}; return true; };
    CHECK(common_semi_oracle_search_astar(order_problem, {8, 4}, advised_order));
    CHECK(advised_order.found && advised_order.guidance_fallbacks == 1 &&
        advised_order.expansions == normal_order.expansions);
    order_problem.transition_adviser = [](const std::string &, const std::vector<std::string> &,
            std::vector<size_t> &, std::string &) -> bool { throw std::runtime_error("test adviser"); };
    CHECK(common_semi_oracle_search_astar(order_problem, {8, 4}, advised_order));
    CHECK(advised_order.found && advised_order.guidance_fallbacks == 1);

    // Focal A* may choose a more expensive goal only inside its host-selected
    // f-bound; the result must distinguish the bound proof from optimality.
    common_semi_oracle_search_problem focal_problem;
    focal_problem.initial_state = "root";
    focal_problem.heuristic_admissible = true;
    focal_problem.is_goal = [](const std::string & state, bool & goal, std::string &) {
        goal = state == "goal"; return true;
    };
    focal_problem.expand = [](const std::string & state,
            std::vector<common_semi_oracle_search_transition> & transitions, std::string &) {
        transitions.clear();
        if (state == "root") transitions = {{"optimal", "short", 1}, {"goal", "long", 3}};
        else if (state == "optimal") transitions = {{"goal", "finish", 1}};
        return true;
    };
    focal_problem.heuristic = [](const std::string & state) { return state == "optimal" ? size_t{1} : size_t{0}; };
    focal_problem.historical_score = [](const std::string & state, const std::string &) {
        return state == "goal" ? 0.0 : 1.0;
    };
    common_semi_oracle_search_policy exact_policy;
    common_semi_oracle_search_result exact_search, focal_search;
    CHECK(common_semi_oracle_search_run(focal_problem, {16, 8}, exact_policy, false, exact_search));
    CHECK(exact_search.found && exact_search.total_cost == 2 && exact_search.optimality_proven);
    common_semi_oracle_search_policy focal_policy;
    focal_policy.algorithm = common_semi_oracle_search_algorithm::focal;
    focal_policy.suboptimality_bound = 1.5;
    CHECK(common_semi_oracle_search_run(focal_problem, {16, 8}, focal_policy, true, focal_search));
    CHECK(focal_search.found && focal_search.total_cost == 3 &&
        !focal_search.optimality_proven && focal_search.suboptimality_bound_proven &&
        focal_search.suboptimality_bound == 1.5);

    // A cheap path was pruned by depth, so selecting an expensive goal later
    // proves reachability but cannot establish a global cost guarantee.
    auto depth_pruned=focal_problem;
    depth_pruned.heuristic=[](const std::string &){return size_t{0};};
    depth_pruned.historical_score={};
    depth_pruned.expand=[](const std::string & state,
            std::vector<common_semi_oracle_search_transition> & next,std::string &){
        next.clear();
        if(state=="root")next={{"cheap","enter",1},{"goal","expensive",10}};
        else if(state=="cheap")next={{"pruned","continue",1}};
        else if(state=="pruned")next={{"goal","finish",1}};
        return true;
    };
    CHECK(common_semi_oracle_search_run(depth_pruned,{16,2},exact_policy,false,exact_search));
    CHECK(exact_search.found&&exact_search.total_cost==10&&!exact_search.optimality_proven);
    CHECK(common_semi_oracle_search_run(depth_pruned,{16,2},focal_policy,false,focal_search));
    CHECK(focal_search.found&&!focal_search.optimality_proven&&!focal_search.suboptimality_bound_proven);

    // Absolute incoming scores compare singleton branches from different
    // parents; sibling ordinal zero for both would choose the wrong goal.
    common_semi_oracle_search_problem absolute_score;
    absolute_score.initial_state="root";
    absolute_score.heuristic_admissible=true;
    absolute_score.is_goal=[](const std::string & state,bool & goal,std::string &){
        goal=state=="goal-left"||state=="goal-right";return true;
    };
    absolute_score.expand=[](const std::string & state,
            std::vector<common_semi_oracle_search_transition> & next,std::string &){
        next.clear();
        if(state=="root")next={{"left","left",1},{"right","right",1}};
        else if(state=="left")next={{"goal-left","finish",1}};
        else if(state=="right")next={{"goal-right","finish",1}};
        return true;
    };
    absolute_score.transition_score=[](const std::string & before,const std::string &,const std::string &){
        return before=="left"?.9:before=="right"?.1:0.0;
    };
    CHECK(common_semi_oracle_search_run(absolute_score,{16,8},exact_policy,true,exact_search));
    CHECK(exact_search.found&&exact_search.path.back().state=="goal-right"&&exact_search.optimality_proven);
    CHECK(common_semi_oracle_search_run(absolute_score,{16,8},exact_policy,false,exact_search));
    CHECK(exact_search.path.back().state=="goal-left");
    return 0;
}
