#include "agent/adaptation/semi-oracle-target.h"
#include "agent/adaptation/semi-oracle-search.h"
#include "hash/hash.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <map>
#include <sstream>
#include <stdexcept>

namespace {
using json = nlohmann::ordered_json;
const std::array<std::string, 4> names{{"north", "south", "east", "west"}};
const int dr[] = {-1, 1, 0, 0}, dc[] = {0, 0, 1, -1};
std::string digest(const std::string & s) { return "sha256:" + hash_sha256_hex(s.data(), s.size()); }
std::string trim(const std::string & s) {
    const auto a = s.find_first_not_of(" \r\t"), b = s.find_last_not_of(" \r\t");
    return a == std::string::npos ? "" : s.substr(a, b-a+1);
}
bool fields(const std::string & text, const std::string & header,
        std::map<std::string, std::string> & out, std::string & error) {
    std::istringstream input(text); std::string line;
    while (std::getline(input, line)) {
        line = trim(line); if (line.empty()) continue;
        if (line == header && out.empty()) continue;
        const auto colon = line.find(':');
        if (colon == std::string::npos) { error = "expected field: value"; return false; }
        auto value = trim(line.substr(colon+1));
        if (!value.empty() && value.front() == '"') {
            const auto parsed = json::parse(value, nullptr, false);
            if (!parsed.is_string()) { error = "invalid quoted value"; return false; }
            value = parsed.get<std::string>();
        }
        if (!out.emplace(trim(line.substr(0, colon)), value).second) {
            error = "duplicate Target Hunt field"; return false;
        }
    }
    return true;
}
bool coordinate(const std::string & text, json & point) {
    std::istringstream in(text); int r, c; char comma;
    if (!(in >> r >> comma >> c) || comma != ',' || r < 0 || c < 0) return false;
    in >> std::ws; if (!in.eof()) return false;
    point = json::array({r,c}); return true;
}
struct board {
    json data;
    std::vector<std::string> grid;
    int row = 0, col = 0, tr = 0, tc = 0;
    bool free(int r, int c) const {
        return r >= 0 && c >= 0 && r < int(grid.size()) && c < int(grid[0].size()) && grid[r][c] == '.';
    }
};
bool parse(const std::string & text, board & b, std::string & error) {
    try {
        b.data = json::parse(text);
        if (!b.data.is_object() || !b.data.at("grid").is_array()) throw std::runtime_error("shape");
        b.grid = b.data.at("grid").get<std::vector<std::string>>();
        if (b.grid.empty() || b.grid.size() > 32 || b.grid[0].empty() || b.grid[0].size() > 32)
            throw std::runtime_error("size");
        for (const auto & row : b.grid) if (row.size() != b.grid[0].size() || row.find_first_not_of(".#") != std::string::npos)
            throw std::runtime_error("grid");
        for (const auto * key : {"start", "target", "position"}) {
            const auto & p = b.data.at(key);
            if (!p.is_array() || p.size() != 2 || !p[0].is_number_integer() || !p[1].is_number_integer() ||
                    p[0]<0 || p[1]<0 || p[0]>31 || p[1]>31 ||
                    !b.free(p[0].get<int>(), p[1].get<int>())) throw std::runtime_error("coordinate");
        }
        b.row = b.data["position"][0]; b.col = b.data["position"][1];
        b.tr = b.data["target"][0]; b.tc = b.data["target"][1];
        return true;
    } catch (...) { error = "Target Hunt requires a rectangular .# grid and free start, target and position coordinates within 32x32"; return false; }
}
int action_index(const std::string & action) {
    const auto it = std::find(names.begin(), names.end(), action);
    return it == names.end() ? -1 : int(it-names.begin());
}
bool move(const board & b, const std::string & action, std::string & next) {
    const int i = action_index(action);
    if (i < 0 || !b.free(b.row+dr[i], b.col+dc[i])) return false;
    auto data = b.data; data["position"] = {b.row+dr[i], b.col+dc[i]}; next = data.dump(); return true;
}
}

bool common_semi_oracle_target_rules(const common_semi_oracle_ruleset & r, std::string & error) {
    if(r.domain!=common_semi_oracle_domain::target_hunt){error="Target Hunt requires a target_hunt RuleSet";return false;}
    try {
        const auto s = json::parse(r.spec_json);
        if (!s.is_object() || s.size() != 3 || s.at("movement") != "orthogonal" ||
                !s.at("max_rows").is_number_integer() || !s.at("max_columns").is_number_integer()) throw std::runtime_error("shape");
        for (const auto * key : {"max_rows", "max_columns"})
            if (s[key] < 1 || s[key] > 32) throw std::runtime_error("bound");
        return true;
    } catch (...) { error = "target_hunt requires movement orthogonal and max_rows/max_columns in 1..32"; return false; }
}
bool common_semi_oracle_target_state(const common_semi_oracle_ruleset & r, const std::string & text, std::string & error) {
    board b; if (!common_semi_oracle_target_rules(r,error) || !parse(text,b,error)) return false;
    const auto s = json::parse(r.spec_json);
    if (b.grid.size() > s["max_rows"].get<size_t>() || b.grid[0].size() > s["max_columns"].get<size_t>()) {
        error = "Target Hunt map exceeds active RuleSet dimensions"; return false;
    }
    return true;
}
bool common_semi_oracle_target_problem(const common_semi_oracle_ruleset & r, const std::string & dsl,
        std::string & state, std::string & error) {
    std::map<std::string,std::string> f;
    if (dsl.empty() || dsl.size() > 4096 || !fields(dsl,"open! oracle.problem.define",f,error)) {
        if (error.empty()) error = "Target Hunt problem must contain 1..4096 bytes";
        return false;
    }
    if (f.size() != 3 || !f.count("grid") || !f.count("start") || !f.count("target")) {
        error = "Target Hunt problem accepts grid, start and target only"; return false;
    }
    json data; data["grid"] = json::array(); std::string row; std::istringstream rows(f["grid"]);
    while (std::getline(rows,row,'/')) data["grid"].push_back(row);
    if (f["grid"].empty() || f["grid"].back() == '/') { error = "empty map row"; return false; }
    if (!coordinate(f["start"],data["start"]) || !coordinate(f["target"],data["target"])) {
        error = "coordinates must be zero-based row,column integers"; return false;
    }
    data["position"] = data["start"];
    if (!common_semi_oracle_target_state(r,data.dump(),error)) return false;
    data["problem_fingerprint"] = digest(data.dump()); state = data.dump(); return true;
}
std::string common_semi_oracle_target_category(const std::string & state, const std::string & action) {
    board b; std::string error, next;
    if (!parse(state,b,error) || !move(b,action,next)) return {};
    const int ai = action_index(action), nr = b.row+dr[ai], nc = b.col+dc[ai];
    const int gx=(b.tc-nc)>0?1:(b.tc-nc)<0?-1:0;
    const int gy=(b.tr-nr)>0?1:(b.tr-nr)<0?-1:0;
    const bool can_continue=(gx!=0&&b.free(nr,nc+gx))||(gy!=0&&b.free(nr+gy,nc));
    const int continuation_blocked=(gx!=0||gy!=0)&&!can_continue;
    int box_cells=0,box_walls=0;
    for(int rr=std::min(nr,b.tr);rr<=std::max(nr,b.tr);++rr)
        for(int cc=std::min(nc,b.tc);cc<=std::max(nc,b.tc);++cc){
            ++box_cells;if(!b.free(rr,cc))++box_walls;
        }
    const int box_density_bucket=box_cells>0?std::min(3,(4*box_walls)/box_cells):0;
    auto transform=[](int x,int y,int rotation,bool reflect){
        if(reflect)x=-x;
        for(int i=0;i<rotation;++i){const int previous=x;x=-y;y=previous;}
        return std::pair<int,int>{x,y};
    };
    std::string canonical;
    for(int rotation=0;rotation<4;++rotation)for(bool reflect:{false,true}){
        std::string mask(9,'0');
        for(int y=-1;y<=1;++y)for(int x=-1;x<=1;++x){
            const auto p=transform(x,y,rotation,reflect);
            // Transform each observed cell into the same coordinate frame
            // used for the goal and action vectors below.
            mask[size_t((p.second+1)*3+p.first+1)]=b.free(nr+y,nc+x)?'1':'0';
        }
        const auto goal=transform(gx,gy,rotation,reflect);
        const auto move_vector=transform(dc[ai],dr[ai],rotation,reflect);
        // Whether the most direct continuation is blocked is intentionally a
        // local/topological feature, not an exact-distance or path oracle.
        const std::string candidate=mask+":"+std::to_string(goal.first)+","+std::to_string(goal.second)+":"+
            std::to_string(move_vector.first)+","+std::to_string(move_vector.second)+":"+
            std::to_string(continuation_blocked)+":"+std::to_string(box_density_bucket);
        if(canonical.empty()||candidate<canonical)canonical=candidate;
    }
    return "target_hunt_features_v3:fine:"+digest(canonical);
}
std::string common_semi_oracle_target_coarse_category(const std::string & state, const std::string & action) {
    board b; std::string error, next;
    if(!parse(state,b,error)||!move(b,action,next))return {};
    const int ai=action_index(action),nr=b.row+dr[ai],nc=b.col+dc[ai];
    int exits=0;
    for(int j=0;j<4;++j)if(b.free(nr+dr[j],nc+dc[j])&&!(nr+dr[j]==b.row&&nc+dc[j]==b.col))++exits;
    const int before=std::abs(b.row-b.tr)+std::abs(b.col-b.tc);
    const int after=std::abs(nr-b.tr)+std::abs(nc-b.tc);
    const int progress=after<before?1:after==before?0:-1;
    // Keep exact exit degree as well as terminal / straight / corner / junction.
    const int shape=exits==0?0:exits==1?(b.free(nr+dr[ai],nc+dc[ai])?1:2):3;
    int box_cells=0,box_walls=0;
    for(int r=std::min(nr,b.tr);r<=std::max(nr,b.tr);++r)
        for(int c=std::min(nc,b.tc);c<=std::max(nc,b.tc);++c){++box_cells;if(!b.free(r,c))++box_walls;}
    const int density=box_cells>0?std::min(3,(4*box_walls)/box_cells):0;
    const int gx=(b.tc-nc)>0?1:(b.tc-nc)<0?-1:0;
    const int gy=(b.tr-nr)>0?1:(b.tr-nr)<0?-1:0;
    const bool can_continue=(gx!=0&&b.free(nr,nc+gx))||(gy!=0&&b.free(nr+gy,nc));
    const int blocked=(gx!=0||gy!=0)&&!can_continue;
    return "target_hunt_features_v3:coarse:"+std::to_string(progress)+":"+
        std::to_string(exits)+":"+std::to_string(shape)+":"+
        std::to_string(blocked)+":"+std::to_string(density);
}
std::string common_semi_oracle_target_group(const std::string & state) {
    board b; std::string error; if (!parse(state,b,error)) return {};
    std::vector<std::string> variants;
    for (int rotation=0;rotation<4;++rotation) for (bool reflect : {false,true}) {
        int rows=int(b.grid.size()), cols=int(b.grid[0].size());
        auto transform = [&](int r,int c) {
            int h=rows,w=cols; if(reflect)c=w-1-c;
            for(int k=0;k<rotation;++k){const int nr=c;c=h-1-r;r=nr;std::swap(h,w);}
            return std::array<int,2>{{r,c}};
        };
        int h=rotation%2?cols:rows,w=rotation%2?rows:cols;
        std::vector<std::string> grid(h,std::string(w,'.'));
        for(int r=0;r<rows;++r)for(int c=0;c<cols;++c){auto p=transform(r,c);grid[p[0]][p[1]]=b.grid[r][c];}
        auto start=transform(b.data["start"][0],b.data["start"][1]); auto target=transform(b.tr,b.tc);
        // Every position of one declared map/start/target belongs to the same leakage group.
        variants.push_back(json{{"grid",grid},{"start",start},{"target",target}}.dump());
    }
    return digest(*std::min_element(variants.begin(),variants.end()));
}
bool common_semi_oracle_target_evaluate(const common_semi_oracle_ruleset & r,
        const common_semi_oracle_query & q, common_semi_oracle_result & out, std::string & error) {
    if (!common_semi_oracle_target_state(r,q.state_json,error)) return false;
    board b; if (!parse(q.state_json,b,error)) return false;
    out.evaluator_revision = "target_hunt_astar_v3";
    if (q.level == common_semi_oracle_level::local) {
        std::map<std::string,std::string> f;
        if (!fields(q.action_dsl,"open! target_hunt.move",f,error)) return false;
        if (f.size()!=1 || !f.count("direction") || action_index(f["direction"])<0) {
            error = "target_hunt.move requires direction north/south/east/west"; return false;
        }
        std::string next;
        if (!move(b,f["direction"],next)) {
            out.verdict=common_semi_oracle_verdict::violated;
            out.reason="move crosses an obstacle or map boundary";
            out.counterexample_json=json{{"state",b.data},{"direction",f["direction"]}}.dump(); return true;
        }
        out.verdict=common_semi_oracle_verdict::satisfied;out.reason="legal unit-cost orthogonal move";
        out.proof_json=json{{"next_state",json::parse(next)},{"cost",1},{"state_fingerprint",digest(q.state_json)},
            {"ruleset_fingerprint",r.semantic_fingerprint},{"action_contract_fingerprint",r.action_contract_fingerprint}}.dump();return true;
    }
    if(q.goal!="reach:target"){error="Target Hunt search requires host goal reach:target";return false;}
    common_semi_oracle_search_problem problem; problem.initial_state=b.data.dump();
    problem.is_goal=[](const std::string & s,bool & goal,std::string & e){board x;if(!parse(s,x,e))return false;goal=x.row==x.tr&&x.col==x.tc;return true;};
    problem.expand=[&](const std::string & s,std::vector<common_semi_oracle_search_transition> & transitions,std::string & e){
        board x;if(!parse(s,x,e))return false;
        auto order=std::vector<std::string>(names.begin(),names.end());
        if(!q.guidance.empty())std::stable_sort(order.begin(),order.end(),[&](const auto & a,const auto & z){
            return std::find(q.guidance.begin(),q.guidance.end(),a)<std::find(q.guidance.begin(),q.guidance.end(),z);});
        for(const auto & action:order){std::string next;if(move(x,action,next))transitions.push_back({next,action,1});}return true;};
    problem.heuristic=[](const std::string & s){board x;std::string e;parse(s,x,e);return size_t(std::abs(x.row-x.tr)+std::abs(x.col-x.tc));};
    problem.heuristic_admissible=true;problem.transition_adviser=q.transition_adviser;problem.record_history=q.record_search_history;
    problem.transition_score=q.transition_score;
    common_semi_oracle_search_result result;
    if(!common_semi_oracle_search_run(problem,q.budget,q.search_policy,q.use_historical_order,result)){error=result.error;return false;}
    out.applied_search_policy = q.search_policy;
    out.expansions=result.expansions;out.depth_reached=result.depth_reached;
    json path=json::array();std::string previous=q.state_json;
    for(size_t i=0;i<result.path.size();++i){const auto & step=result.path[i];
        if(i){board x;std::string next;if(!parse(previous,x,error)||!move(x,step.action,next)||next!=step.state){error="Target Hunt path verification failed";return false;}}
        path.push_back({{"state",{{"position",json::parse(step.state)["position"]}}},{"action",step.action},{"cost",step.cost}});previous=step.state;}
    out.verdict=result.found?common_semi_oracle_verdict::satisfied:result.exhausted?common_semi_oracle_verdict::violated:common_semi_oracle_verdict::unknown;
    out.reason=result.found
        ? (result.optimality_proven ? "verified shortest path" : result.suboptimality_bound_proven
            ? "verified path within host-selected Focal A* cost bound"
            : "verified path; global cost guarantee unavailable after depth pruning")
        : result.exhausted?"reachable state space exhausted without target":"search budget exhausted: "+result.stop_reason;
    if(result.found)out.proof_json=json{{"algorithm",result.algorithm==common_semi_oracle_search_algorithm::focal?"focal_astar_v1":"exact_astar_v1"},
        {"suboptimality_bound",result.suboptimality_bound},{"suboptimality_bound_proven",result.suboptimality_bound_proven},
        {"initial_state",b.data},{"path",path},{"total_cost",result.total_cost},{"optimality_proven",result.optimality_proven},
        {"ruleset_fingerprint",r.semantic_fingerprint},{"action_contract_fingerprint",r.action_contract_fingerprint}}.dump();
    if(result.exhausted)out.counterexample_json=json{{"exhausted",true}}.dump();
    if(q.record_search_history){
        json observations=json::array();
        for(const auto & o:result.observations) {
            observations.push_back({{"action",o.action},{"outcome",o.outcome},{"state_before",o.state_before},{"state_after",o.state_after}});
        }
        out.search_guidance_json=json{{"observations",observations},{"callback_fallbacks",result.guidance_fallbacks},{"callback_error",result.guidance_error}}.dump();
    }
    return true;
}
