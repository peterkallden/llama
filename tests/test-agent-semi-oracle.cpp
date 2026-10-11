#include "agent/adaptation/semi-oracle.h"
#include "agent/adaptation/semi-oracle-tool.h"

#include <map>
#include <string>

#define CHECK(x) do { if (!(x)) return __LINE__; } while (false)

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
    bool append_evaluation(const std::string &,const common_semi_oracle_result & r,std::string &) override {evaluations.push_back(r);return true;}
    std::vector<std::string> list_host_rulesets(std::string &) const override {return host;}
    std::map<std::pair<std::string,uint64_t>,common_semi_oracle_ruleset> rules;
    std::map<std::string,std::pair<std::string,uint64_t>> bindings;
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
    CHECK(common_semi_oracle_action_contract(game).find("player: X, O")!=std::string::npos);

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

    const auto encoded=common_semi_oracle_ruleset_to_json(game);
    common_semi_oracle_ruleset roundtrip;
    CHECK(common_semi_oracle_ruleset_from_json(encoded,roundtrip,error));
    CHECK(roundtrip.id==game.id&&roundtrip.semantic_fingerprint==game.semantic_fingerprint);

    memory_store store;common_semi_oracle_tool_service service(store,"host-test","session-tool");
    common_semi_oracle_ruleset managed;
    CHECK(service.define_ruleset("name: Tool game\ndescription: session game\ndomain: placement_game\nrows: 3\ncolumns: 3\nwin_length: 3\nplayers: X, O\n",common_semi_oracle_rule_authority::declared,managed,error));
    CHECK(managed.id!=""&&managed.scope==common_semi_oracle_scope::session&&managed.active);
    std::string action_contract;CHECK(service.action_contract(action_contract,error)&&action_contract.find("row: integer, 0-based")!=std::string::npos);
    CHECK(service.verify_action(R"json({"cells":[null,null,null,null,null,null,null,null,null],"turn":"X"})json",
        "open! game.place_mark\nplayer: X\nrow: 0\ncolumn: 0\n",common_semi_oracle_level::local,"",
        common_semi_oracle_budget{}, {},result,error));
    CHECK(result.verdict==common_semi_oracle_verdict::satisfied&&store.evaluations.size()==1);
    common_semi_oracle_ruleset revised;
    CHECK(service.define_ruleset("name: Tool game corrected\ndescription: session game\ndomain: placement_game\nrows: 3\ncolumns: 3\nwin_length: 3\nplayers: X, O\n",common_semi_oracle_rule_authority::declared,revised,error));
    CHECK(revised.id==managed.id&&revised.revision==2&&revised.parent_revision==1);
    CHECK(service.promote_active_ruleset_to_host(error));
    CHECK(store.list_host_rulesets(error).size()==1);
    return 0;
}
