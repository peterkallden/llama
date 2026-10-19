#include "agent/adaptation/semi-oracle-target.h"
#include "agent/adaptation/semi-oracle-tool.h"
#include "tools/agent/adaptation/semi-oracle-store-factory.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <stdexcept>

namespace {
using json=nlohmann::ordered_json;
void require(bool ok,const std::string & message){if(!ok)throw std::runtime_error(message);}
const std::string rules="name: Target Hunt\ndescription: Orthogonal Maze smoke\ndomain: target_hunt\nmovement: orthogonal\nmax_rows: 32\nmax_columns: 32\n";
// Read-only history snapshot over the production store; writes remain ordinary evaluations.
class frozen_store final:public common_semi_oracle_store {
    common_semi_oracle_store & base;
public:
    std::vector<common_semi_oracle_search_history_record> history;
    bool fail_reads=false;
    explicit frozen_store(common_semi_oracle_store & store):base(store){}
    bool put_revision(const common_semi_oracle_ruleset & r,std::string & e)override{return base.put_revision(r,e);}
    bool get_revision(const std::string & id,uint64_t rev,common_semi_oracle_ruleset & r,bool & f,std::string & e)const override{return base.get_revision(id,rev,r,f,e);}
    bool promote_to_host(const std::string & id,uint64_t rev,std::string & e)override{return base.promote_to_host(id,rev,e);}
    bool bind_session(const std::string & s,const std::string & id,uint64_t rev,std::string & e)override{return base.bind_session(s,id,rev,e);}
    bool get_session_binding(const std::string & s,std::string & id,uint64_t & rev,bool & f,std::string & e)const override{return base.get_session_binding(s,id,rev,f,e);}
    bool get_session_state(const std::string & s,const std::string & id,uint64_t rev,std::string & v,bool & f,std::string & e)const override{return base.get_session_state(s,id,rev,v,f,e);}
    bool put_session_state(const std::string & s,const std::string & id,uint64_t rev,const std::string & v,std::string & e)override{return base.put_session_state(s,id,rev,v,e);}
    bool append_evaluation(const std::string & id,const common_semi_oracle_result & r,std::string & e)override{return base.append_evaluation(id,r,e);}
    bool list_search_history(const std::string &,uint64_t,size_t limit,std::vector<common_semi_oracle_search_history_record> & rows,std::string & e)const override{
        if(fail_reads){e="injected history read failure";return false;}
        e.clear();rows=history;if(rows.size()>limit)rows.resize(limit);return true;}
    std::vector<std::string> list_host_rulesets(std::string & e)const override{return base.list_host_rulesets(e);}
};
struct fixture {std::vector<std::string> grid;int sr=0,sc=0,tr=0,tc=0;};
std::string dsl(const fixture & f){std::string grid;for(const auto & row:f.grid){if(!grid.empty())grid+='/';grid+=row;}
    return "grid: "+grid+"\nstart: "+std::to_string(f.sr)+","+std::to_string(f.sc)+"\ntarget: "+std::to_string(f.tr)+","+std::to_string(f.tc)+"\n";}
// Independent reference: no calls to the production successor builder, heuristic or adviser.
int bfs(const fixture & f){const int h=int(f.grid.size()),w=int(f.grid[0].size());std::vector<int> dist(h*w,-1);
    std::queue<int> q;q.push(f.sr*w+f.sc);dist[q.front()]=0;
    while(!q.empty()){int p=q.front();q.pop();int r=p/w,c=p%w;if(r==f.tr&&c==f.tc)return dist[p];
        for(const auto delta:std::vector<std::pair<int,int>>{{-1,0},{1,0},{0,-1},{0,1}}){int nr=r+delta.first,nc=c+delta.second;
            if(nr<0||nc<0||nr>=h||nc>=w||f.grid[nr][nc]=='#'||dist[nr*w+nc]>=0)continue;
            dist[nr*w+nc]=dist[p]+1;q.push(nr*w+nc);}}
    return -1;}
fixture generate(size_t id,std::mt19937_64 & rng){int n=id%2?13:9;fixture f;f.grid.assign(n,std::string(n,'.'));
    if(id%4<2){for(int r=1;r<n-1;r+=2){int gap=int(rng()%n);for(int c=0;c<n;++c)if(c!=gap)f.grid[r][c]='#';}}
    else for(auto & row:f.grid)for(auto & c:row)if(rng()%4==0)c='#';
    f.sr=int(rng()%n);f.sc=int(rng()%n);f.tr=int(rng()%n);f.tc=int(rng()%n);f.grid[f.sr][f.sc]='.';f.grid[f.tr][f.tc]='.';return f;}
void check_result(const fixture & f,const common_semi_oracle_result & r){int distance=bfs(f);
    if(r.verdict==common_semi_oracle_verdict::satisfied){auto proof=json::parse(r.proof_json);require(distance>=0&&proof["total_cost"]==distance,"A* path differs from independent BFS");
        const auto & path=proof["path"];require(!path.empty(),"empty proven path");
        int pr=f.sr,pc=f.sc;for(size_t i=0;i<path.size();++i){const auto & p=path[i]["state"]["position"];int nr=p[0],nc=p[1];
            require(nr>=0&&nc>=0&&nr<int(f.grid.size())&&nc<int(f.grid[0].size())&&f.grid[nr][nc]=='.',"illegal proof coordinate");
            require(i?std::abs(nr-pr)+std::abs(nc-pc)==1:nr==pr&&nc==pc,"non-adjacent proof path");pr=nr;pc=nc;}
        require(pr==f.tr&&pc==f.tc,"proof does not end at target");}
    if(r.verdict==common_semi_oracle_verdict::violated)require(distance<0,"false unreachable verdict");}
}

int main(int argc,char ** argv){try{
    std::ofstream trace;if(argc==3&&std::string(argv[1])=="--trace"){trace.open(argv[2]);require(bool(trace),"cannot open trace");}
    else require(argc==1,"usage: target-hunt-smoke [--trace FILE]");
    std::string error;auto store=make_agent_semi_oracle_store("in_memory","",error);require(bool(store),error);
    common_semi_oracle_tool_service training(*store,"target-smoke","target-session",{"shadow",0,1024},"target-project");
    common_semi_oracle_ruleset rule;require(training.define_ruleset(rules,common_semi_oracle_rule_authority::declared,rule,error),error);
    std::string state;require(training.define_problem("grid: .../.#./...\nstart: 0,0\ntarget: 2,2",state,error),error);
    const auto original=state;
    std::string reflected,rectangle,rotated;
    require(common_semi_oracle_target_problem(rule,"grid: .../.#./...\nstart: 0,2\ntarget: 2,0",reflected,error),error);
    require(common_semi_oracle_target_group(original)==common_semi_oracle_target_group(reflected),"reflection leaked between fixture groups");
    require(common_semi_oracle_target_problem(rule,"grid: ..#/...\nstart: 0,0\ntarget: 1,2",rectangle,error),error);
    require(common_semi_oracle_target_problem(rule,"grid: ../../.#\nstart: 0,1\ntarget: 2,0",rotated,error),error);
    require(common_semi_oracle_target_group(rectangle)==common_semi_oracle_target_group(rotated),"rectangular rotation leaked between fixture groups");
    require(!training.define_problem("grid: ../.\nstart: 0,0\ntarget: 1,0",state,error),"ragged grid accepted");
    require(!training.define_problem("grid: .#\nstart: 0,0\ntarget: 0,1",state,error),"blocked target accepted");
    require(!training.define_problem("grid: ..\nstart: -1,0\ntarget: 0,1",state,error),"negative coordinate accepted");
    require(!training.define_problem("grid: "+std::string(33,'.')+"\nstart: 0,0\ntarget: 0,1",state,error),"oversized map accepted");
    bool found=false;std::string retained;
    require(store->get_session_state("target-session",rule.id,rule.revision,retained,found,error)&&found&&retained==original,"invalid problem replaced active state");
    common_semi_oracle_result move;std::string before,after;
    require(training.verify_active_action("direction: north",move,before,after,error)&&move.verdict==common_semi_oracle_verdict::violated,"illegal move accepted");
    require(training.verify_active_action("open! target_hunt.move\ndirection: east",move,before,after,error)&&move.verdict==common_semi_oracle_verdict::satisfied,"legal move rejected");
    require(before==original,"rejected move changed position");
    require(common_semi_oracle_target_group(original)==common_semi_oracle_target_group(after),"movement split one map into independent fixture groups");
    require(!common_semi_oracle_learning_authorized(rule,move),"grid acquired unsupported learning credit");
    require(training.initialize_active_state(original,true,error),error);
    common_semi_oracle_result bounded;
    require(training.search_active_goal(bounded,error,{1,256})&&bounded.verdict==common_semi_oracle_verdict::unknown&&bounded.reason.find("max_expansions")!=std::string::npos,"expansion stop was not explicit UNKNOWN");
    require(training.search_active_goal(bounded,error,{2048,1})&&bounded.verdict==common_semi_oracle_verdict::unknown&&bounded.reason.find("max_depth")!=std::string::npos,"depth stop was not explicit UNKNOWN");
    for(const auto & f:std::vector<fixture>{{{"."},0,0,0,0},{{".#."},0,0,0,2},{{"...",".#.","..."},0,0,2,2}}){
        require(training.define_problem(dsl(f),state,error),error);common_semi_oracle_result r;require(training.search_active_goal(r,error),error);check_result(f,r);}
    require(training.define_problem("grid: .../.../...\nstart: 0,0\ntarget: 2,2",state,error),error);
    common_semi_oracle_result south_first,east_first;
    require(training.search_active_goal(south_first,error,{2048,1024},{"south","east","north","west"}),error);
    require(training.search_active_goal(east_first,error,{2048,1024},{"east","south","north","west"}),error);
    const auto south_path=json::parse(south_first.proof_json),east_path=json::parse(east_first.proof_json);
    require(south_path["total_cost"]==4&&east_path["total_cost"]==4&&south_path["path"]!=east_path["path"],"equal-priority successor ordering did not reach the executed path");
    require(training.define_problem("grid: .../...\nstart: 0,0\ntarget: 0,2",state,error),error);
    require(training.search_active_goal(south_first,error,{2048,1024},{"south","east","north","west"}),error);
    require(json::parse(south_first.proof_json)["path"][1]["state"]["position"]==json::array({0,1}),"ordering overrode f/h priority");
    std::mt19937_64 rng(20261019);std::set<std::string> groups;std::vector<fixture> fixtures;
    for(size_t attempt=0;fixtures.size()<96&&attempt<10000;++attempt){auto f=generate(attempt,rng);
        if(f.sr==f.tr&&f.sc==f.tc)continue;
        require(training.define_problem(dsl(f),state,error),error);
        if(groups.insert(common_semi_oracle_target_group(state)).second)fixtures.push_back(std::move(f));}
    require(fixtures.size()==96,"insufficient independent maps");
    for(size_t i=0;i<64;++i){require(training.define_problem(dsl(fixtures[i]),state,error),error);
        std::vector<std::string> order={"north","south","east","west"};std::shuffle(order.begin(),order.end(),rng);
        for(size_t budget:{16,64,256}){common_semi_oracle_result r;require(training.search_active_goal(r,error,{budget,256},order),error);check_result(fixtures[i],r);
            if(trace)trace<<json{{"phase","train"},{"fixture",i},{"budget",budget},{"verdict",common_semi_oracle_verdict_name(r.verdict)},
                {"expansions",r.expansions},{"guidance",json::parse(r.search_guidance_json)}}.dump()<<'\n';}}
    frozen_store frozen(*store);require(store->list_search_history(rule.id,rule.revision,1024,frozen.history,error),error);
    frozen.history.erase(std::remove_if(frozen.history.begin(),frozen.history.end(),[](const auto & record){
        const auto context=json::parse(record.guidance_json).at("context");
        const auto budget=context.at("max_expansions").template get<size_t>();
        return context.at("max_depth")!=256||(budget!=16&&budget!=64&&budget!=256);
    }),frozen.history.end());
    require(frozen.history.size()==192,"training snapshot must contain exactly three searches for each of 64 maps");
    // Keep every prediction input and source ID, while avoiding repeated parsing
    // of verbose diagnostics. The original evaluations remain in the production store/trace.
    for(auto & record:frozen.history){auto data=json::parse(record.guidance_json);
        record.guidance_json=json{{"schema_version",data["schema_version"]},{"context",data["context"]},
            {"fixture_group",data["fixture_group"]},{"observations",data["observations"]}}.dump();}
    if(trace){
        json rows=json::array();
        for(const auto & record:frozen.history) {
            rows.push_back({{"evaluation_id",record.evaluation_id},{"payload",json::parse(record.guidance_json)}});
        }
        trace<<json{{"phase","frozen_history"},{"ruleset",json::parse(common_semi_oracle_ruleset_to_json(rule))},{"rows",rows}}.dump()<<'\n';
    }
    // The exact same read projection must isolate scope/revision and deduplicate retries.
    require(training.define_problem(dsl(fixtures[64]),state,error),error);
    common_semi_oracle_query probe;probe.ruleset_id=rule.id;probe.ruleset_revision=rule.revision;
    probe.state_json=state;probe.goal="reach:target";probe.level=common_semi_oracle_level::bounded;probe.budget={256,256};
    common_semi_oracle_result checked;
    auto evaluate=[&](const std::string & host,const common_semi_oracle_ruleset & r){
        require(common_semi_oracle_evaluate_with_history(frozen,{"shadow",0,1024},host,"target-session",r,probe,checked,error,"target-project"),error);
        return json::parse(checked.search_guidance_json)["diagnostics"];};
    require(evaluate("target-smoke",rule)["fixture_groups"]>0,"matching training history was not used");
    frozen.fail_reads=true;
    require(evaluate("target-smoke",rule)["fallback_reason"]=="search_history_unavailable","history error did not fall back");
    frozen.fail_reads=false;
    require(evaluate("another-host",rule)["fixture_groups"]==0,"history crossed host scope");
    auto next_revision=rule;++next_revision.revision;probe.ruleset_revision=next_revision.revision;
    require(evaluate("target-smoke",next_revision)["fixture_groups"]==0,"history crossed rule revision");probe.ruleset_revision=rule.revision;
    auto copy=frozen.history.front();frozen.history.push_back(copy);
    require(evaluate("target-smoke",rule)["duplicate_records"]==1,"retry not deduplicated");
    auto conflicting=json::parse(frozen.history.back().guidance_json);
    conflicting["context"]["goal"]="conflicting-goal";
    frozen.history.back().guidance_json=conflicting.dump();
    require(evaluate("target-smoke",rule)["conflicting_records"]==1,"conflicting retry not excluded");frozen.history.pop_back();
    json summaries=json::array();size_t changed=0,changed_paths=0;
    for(size_t budget:{16,64,256})for(uint64_t seed=0;seed<8;++seed){
        json row={{"budget",budget},{"seed",seed}};
        for(const auto * mode:{"off","shadow","experimental"}){size_t hits=0,unknown=0,expansions=0;double elapsed=0,projection_ms=0,adviser_ms=0,search_ms=0;
            common_semi_oracle_tool_service service(frozen,"target-smoke","target-session",{mode,seed,1024},"target-project");
            for(size_t i=64;i<96;++i){require(service.define_problem(dsl(fixtures[i]),state,error),error);
                // Stable problem identity across variants: canonical instance bytes, not a fresh host nonce.
                auto stable=json::parse(state);stable["problem_id"]="heldout-"+std::to_string(i);state=stable.dump();
                require(service.initialize_active_state(state,true,error),error);
                auto start=std::chrono::steady_clock::now();common_semi_oracle_result r;require(service.search_active_goal(r,error,{budget,256}),error);
                elapsed+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();check_result(fixtures[i],r);
                hits+=r.verdict==common_semi_oracle_verdict::satisfied;unknown+=r.verdict==common_semi_oracle_verdict::unknown;expansions+=r.expansions;
                auto guidance=json::parse(r.search_guidance_json.empty()?"{}":r.search_guidance_json);
                if(guidance.contains("diagnostics")){
                    const auto & diagnostics=guidance["diagnostics"];
                    projection_ms+=diagnostics.value("projection_ms",0.0);adviser_ms+=diagnostics.value("adviser_ms",0.0);search_ms+=diagnostics.value("search_ms",0.0);
                }
                if(std::string(mode)=="experimental")for(const auto & decision:guidance["diagnostics"]["decisions"]){
                    std::vector<size_t> original_order(decision["executed_order"].size());std::iota(original_order.begin(),original_order.end(),size_t{0});
                    if(decision["executed_order"]!=json(original_order))++changed;}
                if(trace)trace<<json{{"phase","heldout"},{"fixture",i},{"fixture_group",common_semi_oracle_target_group(state)},
                    {"problem_state",json::parse(state)},
                    {"mode",mode},{"seed",seed},{"budget",budget},{"verdict",common_semi_oracle_verdict_name(r.verdict)},
                    {"stop_reason",r.reason},
                    {"expansions",r.expansions},{"bfs_distance",bfs(fixtures[i])},{"proof",json::parse(r.proof_json.empty()?"null":r.proof_json)},
                    {"guidance",guidance}}.dump()<<'\n';
                // For every task, shadow must preserve the complete returned proof and expansion count.
                static std::map<std::tuple<size_t,uint64_t,size_t>,std::pair<std::string,size_t>> reference;
                auto key=std::make_tuple(budget,seed,i);
                if(std::string(mode)=="off")reference[key]={r.proof_json,r.expansions};
                if(std::string(mode)=="shadow")require(reference.at(key)==std::make_pair(r.proof_json,r.expansions),"shadow changed executed search");
                if(std::string(mode)=="experimental"&&reference.at(key)!=std::make_pair(r.proof_json,r.expansions))++changed_paths;
            }
            row[mode]={{"found",hits},{"unknown",unknown},{"unreachable",32-hits-unknown},{"expansions",expansions},{"total_ms",elapsed},
                {"projection_ms",std::string(mode)=="off"?json(nullptr):json(projection_ms)},
                {"adviser_ms",std::string(mode)=="off"?json(nullptr):json(adviser_ms)},
                {"search_evaluation_ms",std::string(mode)=="off"?json(nullptr):json(search_ms)}};
        }summaries.push_back(row);
    }
    require(changed>0,"historical adviser never changed any successor order");
    require(changed_paths>0,"guidance changed orders but never changed an executed path or expansion count");
    std::cout<<json{{"train_maps",64},{"heldout_maps",32},{"history_source","compact_frozen_projection"},{"frozen_records",frozen.history.size()},
        {"changed_orders",changed},{"changed_searches",changed_paths},{"comparisons",summaries}}.dump(2)<<'\n';
    return 0;
}catch(const std::exception & e){std::cerr<<"Target Hunt smoke failed: "<<e.what()<<'\n';return 1;}}
