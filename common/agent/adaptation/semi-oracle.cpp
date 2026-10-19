#include "agent/adaptation/semi-oracle.h"
#include "agent/adaptation/semi-oracle-search.h"
#include "agent/adaptation/semi-oracle-target.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <utility>

using json = nlohmann::ordered_json;

const char * common_semi_oracle_domain_name(common_semi_oracle_domain v) {
    switch(v) {
        case common_semi_oracle_domain::placement_game: return "placement_game";
        case common_semi_oracle_domain::linear_algebra: return "linear_algebra";
        case common_semi_oracle_domain::target_hunt: return "target_hunt";
    }
    return "unsupported";
}
const char * common_semi_oracle_scope_name(common_semi_oracle_scope v) {
    return v == common_semi_oracle_scope::session ? "session" : "host";
}
const char * common_semi_oracle_rule_authority_name(common_semi_oracle_rule_authority v) {
    switch (v) {
        case common_semi_oracle_rule_authority::declared: return "declared";
        case common_semi_oracle_rule_authority::verified: return "verified";
        case common_semi_oracle_rule_authority::inferred: return "inferred";
    }
    return "inferred";
}
const char * common_semi_oracle_level_name(common_semi_oracle_level v) {
    switch (v) {
        case common_semi_oracle_level::local: return "local";
        case common_semi_oracle_level::bounded: return "bounded";
        case common_semi_oracle_level::guided: return "guided";
    }
    return "local";
}
const char * common_semi_oracle_verdict_name(common_semi_oracle_verdict v) {
    switch (v) {
        case common_semi_oracle_verdict::satisfied: return "SATISFIED";
        case common_semi_oracle_verdict::violated: return "VIOLATED";
        case common_semi_oracle_verdict::unknown: return "UNKNOWN";
    }
    return "UNKNOWN";
}

namespace {

std::string trim(std::string value) {
    const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char c) { return std::isspace(c); });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) { return std::isspace(c); }).base();
    return first < last ? std::string(first, last) : std::string();
}

std::string compact_expression(const std::string & value) {
    std::string result;
    result.reserve(value.size());
    for (const unsigned char c : value) if (!std::isspace(c)) result.push_back(static_cast<char>(c));
    return result;
}

uint64_t fingerprint(const std::string & text) {
    uint64_t value = 1469598103934665603ULL;
    for (const unsigned char c : text) { value ^= c; value *= 1099511628211ULL; }
    return value;
}

std::string semantic_fingerprint_for(const std::string & spec_json, common_semi_oracle_domain domain) {
    std::ostringstream fp;fp<<"fnv1a64:"<<std::hex<<fingerprint(spec_json+"\n"+common_semi_oracle_domain_name(domain));return fp.str();
}

std::string action_fingerprint_for(const std::string & semantic_fingerprint, common_semi_oracle_domain domain) {
    return std::string("action-v1:")+common_semi_oracle_domain_name(domain)+":"+semantic_fingerprint;
}

bool number(const std::string & text, int & value) {
    if (text.empty()) return false;
    try {
        size_t used = 0;
        const auto parsed = std::stol(text, &used);
        if (used != text.size() || parsed < std::numeric_limits<int>::min() || parsed > std::numeric_limits<int>::max()) return false;
        value = static_cast<int>(parsed);
        return true;
    } catch (...) { return false; }
}

struct rational {
    int64_t n = 0, d = 1;
    static bool add_checked(int64_t a,int64_t b,int64_t & out) {
        if ((b>0&&a>std::numeric_limits<int64_t>::max()-b)||(b<0&&a<std::numeric_limits<int64_t>::min()-b)) return false;
        out=a+b;return true;
    }
    static bool mul_checked(int64_t a,int64_t b,int64_t & out) {
        if(a==0||b==0){out=0;return true;}
        if((a==-1&&b==std::numeric_limits<int64_t>::min())||(b==-1&&a==std::numeric_limits<int64_t>::min()))return false;
        if(a>0 ? (b>0 ? a>std::numeric_limits<int64_t>::max()/b : b<std::numeric_limits<int64_t>::min()/a)
               : (b>0 ? a<std::numeric_limits<int64_t>::min()/b : a<std::numeric_limits<int64_t>::max()/b)) return false;
        out=a*b;return true;
    }
    static bool make(int64_t n, int64_t d, rational & out) {
        if (d == 0) return false;
        if (n==std::numeric_limits<int64_t>::min()||d==std::numeric_limits<int64_t>::min())return false;
        if (d < 0) { n = -n; d = -d; }
        const auto absn=n<0?-n:n;const auto g=std::gcd(absn,d);
        out.n=n/g;out.d=d/g;return true;
    }
    static bool negate(rational value,rational & out) {return value.n!=std::numeric_limits<int64_t>::min()&&make(-value.n,value.d,out);}
    bool add(rational b, rational & o) const { int64_t l,r,dn,numer;return mul_checked(n,b.d,l)&&mul_checked(b.n,d,r)&&add_checked(l,r,numer)&&mul_checked(d,b.d,dn)&&make(numer,dn,o); }
    bool sub(rational b, rational & o) const { int64_t l,r,dn,numer,neg;if(b.n==std::numeric_limits<int64_t>::min())return false;neg=-b.n;return mul_checked(n,b.d,l)&&mul_checked(neg,d,r)&&add_checked(l,r,numer)&&mul_checked(d,b.d,dn)&&make(numer,dn,o); }
    bool mul(rational b, rational & o) const { int64_t numer,denom;return mul_checked(n,b.n,numer)&&mul_checked(d,b.d,denom)&&make(numer,denom,o); }
    bool div(rational b, rational & o) const { int64_t numer,denom;return b.n&&mul_checked(n,b.d,numer)&&mul_checked(d,b.n,denom)&&make(numer,denom,o); }
    bool zero() const { return n == 0; }
    std::string str() const { return d == 1 ? std::to_string(n) : std::to_string(n) + "/" + std::to_string(d); }
};

struct linear { rational a, b; };

class expression_parser {
public:
    explicit expression_parser(std::string input) : text_(std::move(input)) {}
    bool parse(linear & out) {
        skip(); if (!sum(out)) return false; skip(); return pos_ == text_.size();
    }
private:
    void skip() { while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) ++pos_; }
    bool sum(linear & out) {
        if (!product(out)) return false;
        for (;;) {
            skip(); if (pos_ >= text_.size() || (text_[pos_] != '+' && text_[pos_] != '-')) return true;
            const char op = text_[pos_++]; linear rhs; if (!product(rhs)) return false;
            if (op == '-') { rational neg; if (!rational::negate(rhs.a,neg)) return false; rhs.a=neg; if (!rational::negate(rhs.b,neg)) return false; rhs.b=neg; }
            if (!out.a.add(rhs.a, out.a) || !out.b.add(rhs.b, out.b)) return false;
        }
    }
    bool product(linear & out) {
        if (!atom(out)) return false;
        for (;;) {
            skip(); if (pos_ >= text_.size()) return true;
            char op = text_[pos_];
            if (op == '*' || op == '/') ++pos_;
            else if (op == 'x' || op == 'X' || op == '(') op='*';
            else return true;
            linear rhs; if (!atom(rhs)) return false;
            if (op == '*') {
                if (!out.a.zero() && !rhs.a.zero()) return false;
                rational aa, bb, t;
                if (!out.a.mul(rhs.b, aa) || !rhs.a.mul(out.b, t) || !aa.add(t, aa) || !out.b.mul(rhs.b, bb)) return false;
                out = {aa, bb};
            } else {
                if (!rhs.a.zero() || rhs.b.zero()) return false;
                rational aa, bb;
                if (!out.a.div(rhs.b, aa) || !out.b.div(rhs.b, bb)) return false;
                out = {aa, bb};
            }
        }
    }
    bool atom(linear & out) {
        skip(); if (pos_ >= text_.size()) return false;
        if (text_[pos_] == '+') { ++pos_; return atom(out); }
        if (text_[pos_] == '-') { ++pos_; if (!atom(out)) return false; rational n; if (!rational::negate(out.a,n)) return false; out.a=n; if (!rational::negate(out.b,n)) return false; out.b=n; return true; }
        if (text_[pos_] == '(') { ++pos_; if (!sum(out)) return false; skip(); if (pos_ >= text_.size() || text_[pos_++] != ')') return false; return true; }
        if (text_[pos_] == 'x' || text_[pos_] == 'X') { ++pos_; out = {{1,1},{0,1}}; return true; }
        const size_t begin = pos_; while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
        if (begin == pos_) return false;
        int64_t integer = 0; try { integer = std::stoll(text_.substr(begin, pos_-begin)); } catch (...) { return false; }
        if(integer>1000000000LL)return false;
        out = {{0,1},{integer,1}}; return true;
    }
    std::string text_; size_t pos_ = 0;
};

bool parse_equation(const std::string & text, linear & diff) {
    const auto eq = text.find('=');
    if (eq == std::string::npos || text.find('=', eq+1) != std::string::npos) return false;
    linear lhs, rhs;
    if (!expression_parser(text.substr(0, eq)).parse(lhs) || !expression_parser(text.substr(eq+1)).parse(rhs)) return false;
    rational a, b;
    if (!lhs.a.sub(rhs.a, a) || !lhs.b.sub(rhs.b, b)) return false;
    diff = {a,b}; return true;
}

bool proportional_nonzero(const linear & a, const linear & b) {
    if (a.a.zero() && a.b.zero()) return b.a.zero() && b.b.zero();
    if (b.a.zero() && b.b.zero()) return false;
    rational x, y;
    if (!a.a.mul(b.b, x) || !a.b.mul(b.a, y) || x.n != y.n || x.d != y.d) return false;
    return true;
}

std::string format_linear(const linear & value) {
    std::string result;
    if (!value.a.zero()) {
        if (value.a.n == -1 && value.a.d == 1) result = "-x";
        else if (value.a.n == 1 && value.a.d == 1) result = "x";
        else result = value.a.str() + "x";
    }
    if (!value.b.zero()) {
        if (result.empty()) result = value.b.str();
        else if (value.b.n < 0) result += value.b.str();
        else result += "+" + value.b.str();
    }
    return result.empty() ? "0" : result;
}

std::string canonical_equation_state(const linear & lhs, const linear & rhs) {
    return json{{"lhs", format_linear(lhs)}, {"rhs", format_linear(rhs)}}.dump();
}

bool equation_state(const std::string & raw, linear & lhs, linear & rhs) {
    const auto state = json::parse(raw, nullptr, false);
    return state.is_object() && state.contains("lhs") && state.contains("rhs") &&
        state["lhs"].is_string() && state["rhs"].is_string() &&
        expression_parser(state["lhs"].get<std::string>()).parse(lhs) &&
        expression_parser(state["rhs"].get<std::string>()).parse(rhs);
}

bool append_equation_transition(
        const std::string & old_state,
        const linear & old_lhs,
        const linear & old_rhs,
        const linear & new_lhs,
        const linear & new_rhs,
        const std::string & action,
        std::vector<common_semi_oracle_search_transition> & transitions) {
    // Every generated edge is checked using the same exact rational
    // equivalence rule as model-proposed rewrites.
    linear old_diff, new_diff;
    if (!old_lhs.a.sub(old_rhs.a, old_diff.a) || !old_lhs.b.sub(old_rhs.b, old_diff.b) ||
            !new_lhs.a.sub(new_rhs.a, new_diff.a) || !new_lhs.b.sub(new_rhs.b, new_diff.b) ||
            !proportional_nonzero(old_diff, new_diff)) return false;
    const auto next = canonical_equation_state(new_lhs, new_rhs);
    if (next == old_state) return true;
    if (std::none_of(transitions.begin(), transitions.end(), [&](const auto & item) {
            return item.state == next;
        })) transitions.push_back({next, action, 1});
    return true;
}

bool build_algebra_search_problem(
        const std::string & initial,
        const std::vector<std::string> & guidance,
        common_semi_oracle_search_problem & problem) {
    problem = {};
    linear initial_lhs, initial_rhs;
    if (!equation_state(initial, initial_lhs, initial_rhs)) return false;
    const auto initial_json = json::parse(initial, nullptr, false);
    problem.initial_state = json{{"lhs", initial_json["lhs"]}, {"rhs", initial_json["rhs"]}}.dump();
    problem.heuristic = [](const std::string &) { return size_t(0); };
    problem.heuristic_admissible = true;
    problem.is_goal = [](const std::string & raw, bool & goal, std::string & error) {
        error.clear(); goal = false;
        linear lhs, rhs;
        if (!equation_state(raw, lhs, rhs)) {
            error = "A* encountered an invalid equation state";
            return false;
        }
        goal = lhs.a.n == 1 && lhs.a.d == 1 && lhs.b.zero() &&
            rhs.a.zero();
        return true;
    };
    problem.expand = [](const std::string & raw,
            std::vector<common_semi_oracle_search_transition> & transitions,
            std::string & error) {
        error.clear(); transitions.clear();
        linear lhs, rhs;
        if (!equation_state(raw, lhs, rhs)) {
            error = "A* encountered an invalid equation state";
            return false;
        }

        linear simplified;
        if (expression_parser(format_linear(lhs)).parse(simplified)) {
            append_equation_transition(raw, lhs, rhs, simplified, rhs,
                "simplify lhs", transitions);
        }
        if (expression_parser(format_linear(rhs)).parse(simplified)) {
            append_equation_transition(raw, lhs, rhs, lhs, simplified,
                "simplify rhs", transitions);
        }

        const auto state = json::parse(raw, nullptr, false);
        const bool lhs_simplified = state.is_object() && state.contains("lhs") &&
            state["lhs"].is_string() && compact_expression(state["lhs"].get<std::string>()) == format_linear(lhs);
        const bool rhs_simplified = state.is_object() && state.contains("rhs") &&
            state["rhs"].is_string() && compact_expression(state["rhs"].get<std::string>()) == format_linear(rhs);

        // Subtract the variable term on the right from both sides.
        rational coefficient;
        if (lhs_simplified && rhs_simplified && lhs.a.sub(rhs.a, coefficient)) {
            const linear moved_variable{coefficient, lhs.b};
            append_equation_transition(raw, lhs, rhs, moved_variable, {{0, 1}, rhs.b},
                "move variable terms to the left", transitions);
        }

        // Subtract the left constant from both sides.
        rational constant;
        if (lhs_simplified && rhs_simplified && lhs.a.sub(rhs.a, coefficient) && rhs.b.sub(lhs.b, constant)) {
            const linear moved_constant{coefficient, {0, 1}};
            append_equation_transition(raw, lhs, rhs, moved_constant, {{0, 1}, constant},
                "move constants to the right", transitions);
        }

        // Divide both sides by the non-zero coefficient.
        if (lhs_simplified && rhs_simplified && !lhs.a.zero() && lhs.b.zero() && rhs.a.zero()) {
            rational solution;
            if (rhs.b.div(lhs.a, solution)) {
                append_equation_transition(raw, lhs, rhs, {{1, 1}, {0, 1}}, {{0, 1}, solution},
                    "divide both sides by the non-zero coefficient", transitions);
            }
        }
        return true;
    };
    if (!guidance.empty()) {
        problem.guidance_rank = [guidance](const common_semi_oracle_search_transition & transition) {
            const auto found = std::find(guidance.begin(), guidance.end(), transition.action);
            return found == guidance.end()
                ? guidance.size() : static_cast<size_t>(std::distance(guidance.begin(), found));
        };
    }
    return true;
}

bool parse_lines(const std::string & text, std::map<std::string,std::string> & fields, std::string & error) {
    std::istringstream in(text); std::string line;
    while (std::getline(in,line)) {
        line = trim(line); if (line.empty() || line[0] == '#') continue;
        if (fields.empty() && line.rfind("open! ",0)==0) continue;
        const auto colon=line.find(':');
        if (colon==std::string::npos) { error="RuleSpec/action line must be `field: value`"; return false; }
        const auto key=trim(line.substr(0,colon)), value=trim(line.substr(colon+1));
        if (key.empty() || value.empty()) { error="RuleSpec/action field cannot be empty"; return false; }
        if (!fields.emplace(key,value).second) { error="RuleSpec field repeated: "+key; return false; }
    }
    return true;
}

bool optional_header_matches(const std::string & text,const std::string & expected,std::string & error) {
    const auto first=text.find_first_not_of(" \t\r\n");if(first==std::string::npos)return true;
    const auto end=text.find_first_of("\r\n",first);const auto line=trim(text.substr(first,end==std::string::npos?std::string::npos:end-first));
    if(line.rfind("open!",0)==0&&line!=expected){error="DSL selected a tool outside the exact host contract: "+line;return false;}
    return true;
}

json spec(const common_semi_oracle_ruleset & r) { return json::parse(r.spec_json, nullptr, false); }

bool game_won(const json & cells, int rows, int cols, int run, const std::string & player);

bool game_state(const common_semi_oracle_ruleset & r, const std::string & raw,
        json & state, int & rows, int & cols, int & win, std::string & error) {
    const auto s=spec(r);
    rows=s.value("rows",0); cols=s.value("columns",0); win=s.value("win_length",0);
    state=json::parse(raw,nullptr,false);
    if (!state.is_object() || !state.contains("cells") || !state["cells"].is_array() ||
            state["cells"].size()!=static_cast<size_t>(rows*cols) || !state.contains("turn") ||
            !state["turn"].is_string()) { error="placement_game state must contain a flat cells array and turn"; return false; }
    const auto players=s.value("players",std::vector<std::string>{});
    if (players.size()!=2 || (state["turn"]!=players[0] && state["turn"]!=players[1])) { error="placement_game state turn is not a declared player"; return false; }
    size_t first_count=0,second_count=0;
    for (const auto & cell:state["cells"]) {
        if (!cell.is_null() && (!cell.is_string() || (cell!=players[0] && cell!=players[1]))) { error="placement_game state contains an undeclared mark"; return false; }
        if(cell==players[0])++first_count;else if(cell==players[1])++second_count;
    }
    if(first_count<second_count||first_count>second_count+1||
            state["turn"]!=(first_count==second_count?players[0]:players[1])) {
        error="placement_game state violates alternating-turn mark counts";return false;
    }
    if(game_won(state["cells"],rows,cols,win,players[0])&&game_won(state["cells"],rows,cols,win,players[1])) {
        error="placement_game state has winners for both players";return false;
    }
    return true;
}

bool game_won(const json & cells, int rows, int cols, int run, const std::string & player) {
    const int dirs[4][2]={{1,0},{0,1},{1,1},{1,-1}};
    for(int y=0;y<rows;++y) for(int x=0;x<cols;++x) for(const auto & d:dirs) {
        int endx=x+(run-1)*d[0], endy=y+(run-1)*d[1];
        if(endx<0||endx>=cols||endy<0||endy>=rows) continue;
        bool yes=true; for(int k=0;k<run;++k) if(cells[static_cast<size_t>((y+k*d[1])*cols+x+k*d[0])]!=player) { yes=false; break; }
        if(yes) return true;
    }
    return false;
}

} // namespace

bool common_semi_oracle_parse_ruleset_dsl(const std::string & text, const std::string & host_id,
        const std::string & scope_id, common_semi_oracle_ruleset & r, std::string & error,
        common_semi_oracle_rule_authority assertion_authority) {
    error.clear(); std::map<std::string,std::string> f;
    if(!optional_header_matches(text,"open! oracle.ruleset.define",error)) return false;
    if (!parse_lines(text,f,error)) return false;
    r={}; r.id=host_id; r.scope_id=scope_id; r.scope=common_semi_oracle_scope::session;
    r.active=false;
    json s=json::object();
    for(const auto & kv:f) {
        const auto & k=kv.first; const auto & v=kv.second;
        if(k=="name") r.name=v;
        else if(k=="description") r.description=v;
        else if(k=="domain") {
            if(v=="placement_game") r.domain=common_semi_oracle_domain::placement_game;
            else if(v=="linear_algebra") r.domain=common_semi_oracle_domain::linear_algebra;
            else if(v=="target_hunt") r.domain=common_semi_oracle_domain::target_hunt;
            else { error="unsupported semi-oracle domain: "+v; return false; }
        } else if(k=="rows"||k=="columns"||k=="win_length"||k=="max_rows"||k=="max_columns") {
            int n=0; if(!number(v,n)) { error="RuleSpec integer field is invalid: "+k; return false; }
            s[k]=n;
        } else if(k=="players") {
            auto val=v; std::replace(val.begin(),val.end(),',',' '); std::istringstream p(val); std::string token; std::vector<std::string> players;
            while(p>>token) players.push_back(token);
            s[k]=players;
        } else if(k=="variable") { s[k]=v; }
        else if(k=="movement") { s[k]=v; }
        else { error="unsupported RuleSpec field: "+k; return false; }
    }
    r.spec_json=s.dump();
    r.assertions.push_back({"rules",text,assertion_authority,assertion_authority==common_semi_oracle_rule_authority::declared?"user://rules":"model://ruleset-hypothesis"});
    if (!common_semi_oracle_validate_ruleset(r,error)) return false;
    r.semantic_fingerprint=semantic_fingerprint_for(r.spec_json,r.domain);
    r.action_contract_fingerprint=action_fingerprint_for(r.semantic_fingerprint,r.domain);
    r.active=true;
    return true;
}

bool common_semi_oracle_revise_ruleset(const common_semi_oracle_ruleset & current,
        const std::string & revised_dsl,common_semi_oracle_rule_authority authority,
        const std::string & new_session_scope_id,
        common_semi_oracle_ruleset & revised,std::string & error) {
    if(!current.active||!common_semi_oracle_validate_ruleset(current,error))return false;
    if(new_session_scope_id.empty()){error="RuleSet revision requires a session scope";return false;}
    if(current.revision==std::numeric_limits<uint64_t>::max()){error="RuleSet revision counter is exhausted";return false;}
    if(!common_semi_oracle_parse_ruleset_dsl(revised_dsl,current.id,new_session_scope_id,revised,error,authority))return false;
    revised.scope=common_semi_oracle_scope::session;revised.scope_id=new_session_scope_id;revised.revision=current.revision+1;
    revised.parent_revision=current.revision;revised.active=true;
    return true;
}

bool common_semi_oracle_validate_ruleset(const common_semi_oracle_ruleset & r, std::string & error) {
    error.clear();
    if(r.schema_version!=1||r.id.empty()||r.name.empty()||r.description.empty()||r.scope_id.empty()||r.revision==0) { error="RuleSet requires host id, name, description, positive revision and scope id"; return false; }
    const auto s=spec(r);
    if(!s.is_object()) { error="RuleSet spec is invalid JSON"; return false; }
    if(r.assertions.empty()){error="RuleSet requires provenance-bearing rule assertions";return false;}
    for(const auto & assertion:r.assertions)if(assertion.key.empty()||assertion.value.empty()||assertion.source_ref.empty()) {error="RuleSet assertion provenance is incomplete";return false;}
    if(r.domain==common_semi_oracle_domain::placement_game) {
        for(const auto * key:{"rows","columns","win_length"})if(!s.contains(key)||!s[key].is_number_integer()) {error=std::string("placement_game requires integer field ")+key;return false;}
        if(!s.contains("players")||!s["players"].is_array()||std::any_of(s["players"].begin(),s["players"].end(),[](const auto & p){return !p.is_string();})) {error="placement_game requires a string array of players";return false;}
        for(const auto & entry:s.items())if(entry.key()!="rows"&&entry.key()!="columns"&&entry.key()!="win_length"&&entry.key()!="players") {error="placement_game RuleSpec contains unsupported field: "+entry.key();return false;}
        int rows=0,cols=0,win=0;try{rows=s["rows"].get<int>();cols=s["columns"].get<int>();win=s["win_length"].get<int>();}catch(...){error="placement_game integer field is out of range";return false;}
        const auto players=s["players"].get<std::vector<std::string>>();
        if(rows<1||cols<1||rows>8||cols>8||win<2||win>std::max(rows,cols)||players.size()!=2||players[0].empty()||players[1].empty()||players[0]==players[1]) {
            error="placement_game requires rows/columns in 1..8, a valid win_length, and two distinct players"; return false;
        }
    } else if(r.domain==common_semi_oracle_domain::target_hunt) {
        if(!common_semi_oracle_target_rules(r,error)) return false;
    } else {
        // Algebra retains its original RuleSpec.
        for(const auto & entry:s.items())if(entry.key()!="variable") {error="linear_algebra RuleSpec contains unsupported field: "+entry.key();return false;}
        if(!s.contains("variable")||!s["variable"].is_string()){error="linear_algebra requires a string variable field";return false;}
        const auto variable=s["variable"].get<std::string>();
        if(variable!="x") { error="linear_algebra V1 supports only variable x"; return false; }
    }
    if(!r.semantic_fingerprint.empty()&&r.semantic_fingerprint!=semantic_fingerprint_for(r.spec_json,r.domain)) {error="RuleSet semantic fingerprint does not match its domain specification";return false;}
    if(!r.action_contract_fingerprint.empty()&&r.action_contract_fingerprint!=action_fingerprint_for(r.semantic_fingerprint,r.domain)) {error="RuleSet action contract fingerprint is stale";return false;}
    return true;
}

std::string common_semi_oracle_ruleset_to_json(const common_semi_oracle_ruleset & r) {
    json assertions=json::array(); for(const auto & a:r.assertions) assertions.push_back({{"key",a.key},{"value",a.value},{"authority",common_semi_oracle_rule_authority_name(a.authority)},{"source_ref",a.source_ref}});
    return json{{"schema_version",r.schema_version},{"id",r.id},{"name",r.name},{"description",r.description},{"domain",common_semi_oracle_domain_name(r.domain)},{"scope",common_semi_oracle_scope_name(r.scope)},{"scope_id",r.scope_id},{"revision",r.revision},{"parent_revision",r.parent_revision},{"semantic_fingerprint",r.semantic_fingerprint},{"action_contract_fingerprint",r.action_contract_fingerprint},{"spec_json",r.spec_json},{"assertions",assertions},{"active",r.active}}.dump();
}

bool common_semi_oracle_ruleset_from_json(const std::string & text, common_semi_oracle_ruleset & r, std::string & error) {
    error.clear(); const auto j=json::parse(text,nullptr,false); if(!j.is_object()) { error="RuleSet JSON is invalid"; return false; }
    try {
        r={}; r.schema_version=j.value("schema_version",0); r.id=j.value("id",""); r.name=j.value("name",""); r.description=j.value("description","");
        const auto d=j.value("domain",""); if(d=="placement_game") r.domain=common_semi_oracle_domain::placement_game; else if(d=="linear_algebra") r.domain=common_semi_oracle_domain::linear_algebra; else if(d=="target_hunt") r.domain=common_semi_oracle_domain::target_hunt; else {error="RuleSet domain is unknown";return false;}
        const auto sc=j.value("scope",""); if(sc=="session") r.scope=common_semi_oracle_scope::session; else if(sc=="host") r.scope=common_semi_oracle_scope::host; else {error="RuleSet scope is unknown";return false;}
        r.scope_id=j.value("scope_id",""); r.revision=j.value("revision",uint64_t(0)); r.parent_revision=j.value("parent_revision",uint64_t(0)); r.semantic_fingerprint=j.value("semantic_fingerprint",""); r.action_contract_fingerprint=j.value("action_contract_fingerprint",""); r.spec_json=j.value("spec_json",""); r.active=j.value("active",false);
        for(const auto & a:j.value("assertions",json::array())) { common_semi_oracle_rule_authority au; const auto value=a.value("authority",""); if(value=="declared") au=common_semi_oracle_rule_authority::declared; else if(value=="verified") au=common_semi_oracle_rule_authority::verified; else if(value=="inferred") au=common_semi_oracle_rule_authority::inferred; else {error="RuleSet assertion authority is invalid";return false;} r.assertions.push_back({a.value("key", ""),a.value("value", ""),au,a.value("source_ref", "")}); }
    } catch(const std::exception & e) {error=std::string("RuleSet JSON is malformed: ")+e.what();return false;}
    return common_semi_oracle_validate_ruleset(r,error);
}

std::string common_semi_oracle_action_contract(const common_semi_oracle_ruleset & r) {
    if(r.domain==common_semi_oracle_domain::target_hunt)
        return "open! target_hunt.move\ndirection: north | south | east | west\nThe host supplies the current position; only legal orthogonal unit moves advance it.";
    if(r.domain==common_semi_oracle_domain::placement_game) {
        const auto s=spec(r);const auto players=s.value("players",std::vector<std::string>{});
        std::string p;for(size_t i=0;i<players.size();++i){if(i)p+=" or ";p+=players[i];}
        return "Rule set: "+r.name+"@r"+std::to_string(r.revision)+"\nReturn only these fields, one per line; choose exactly one player value from: "+p+".\nplayer: current player's single symbol\nrow: integer, 0-based\ncolumn: integer, 0-based\nConstraint: target cell must be empty and player must match turn. The host selects the placement transition.";
    }
    return "Rule set: "+r.name+"@r"+std::to_string(r.revision)+"\nReturn changed fields only, one per line:\nlhs: linear expression in x (omit to keep current lhs)\nrhs: linear expression in x (omit to keep current rhs)\nConstraint: resulting equation must be equivalent over exact rational numbers. The host selects the rewrite evaluator and fills omitted fields from current state.";
}

bool common_semi_oracle_validate_state(const common_semi_oracle_ruleset & r,
        const std::string & state_json, std::string & error) {
    error.clear();
    if (!r.active || !common_semi_oracle_validate_ruleset(r, error)) {
        if (error.empty()) error = "RuleSet is not active";
        return false;
    }
    if (r.domain == common_semi_oracle_domain::placement_game) {
        json state; int rows = 0, columns = 0, win_length = 0;
        return game_state(r, state_json, state, rows, columns, win_length, error);
    }
    if (r.domain == common_semi_oracle_domain::target_hunt)
        return common_semi_oracle_target_state(r, state_json, error);
    const auto state = json::parse(state_json, nullptr, false);
    if (!state.is_object() || !state.contains("lhs") || !state.contains("rhs") ||
            !state["lhs"].is_string() || !state["rhs"].is_string()) {
        error = "linear_algebra state requires lhs and rhs strings";
        return false;
    }
    linear normalized;
    if (!parse_equation(state["lhs"].get<std::string>() + "=" +
            state["rhs"].get<std::string>(), normalized)) {
        error = "linear_algebra state is outside the supported exact linear fragment";
        return false;
    }
    return true;
}

bool common_semi_oracle_evaluate(const common_semi_oracle_ruleset & r, const common_semi_oracle_query & q,
        common_semi_oracle_result & out, std::string & error) {
    error.clear(); out={}; out.ruleset_id=r.id; out.ruleset_revision=r.revision; out.level=q.level;
    out.evaluator_ref=std::string("semi-oracle://evaluator/")+common_semi_oracle_domain_name(r.domain);
    out.evaluator_revision=r.domain==common_semi_oracle_domain::linear_algebra &&
            q.level!=common_semi_oracle_level::local && q.goal=="solve:x"
        ? "linear_astar_v1" : "v1";
    if(!r.active || !common_semi_oracle_validate_ruleset(r,error)) { if(error.empty()) error="RuleSet is not active"; return false; }
    if(q.ruleset_id!=r.id||q.ruleset_revision!=r.revision) {error="query RuleSet identity or revision does not match active RuleSet";return false;}
    if(q.budget.max_expansions==0||q.budget.max_expansions>100000||q.budget.max_depth==0||q.budget.max_depth>(r.domain==common_semi_oracle_domain::target_hunt?1024:64)) {error="semi-oracle search budget is outside the supported bound";return false;}
    if(q.level==common_semi_oracle_level::guided&&!q.guidance.empty()) out.guidance_ref="host-guidance:provided";
    if(r.domain==common_semi_oracle_domain::target_hunt)
        return common_semi_oracle_target_evaluate(r,q,out,error);
    if(r.domain==common_semi_oracle_domain::linear_algebra) {
        std::map<std::string,std::string> f;
        if(!optional_header_matches(q.action_dsl,"open! algebra.rewrite_equation",error)) return false;
        if(!parse_lines(q.action_dsl,f,error)) return false;
        if(f.empty()||f.size()>2||f.size()!=f.count("lhs")+f.count("rhs")) {error="rewrite_equation accepts lhs and/or rhs only; omitted fields retain their current values";return false;}
        const auto current=json::parse(q.state_json,nullptr,false);
        if(!current.is_object()||!current.contains("lhs")||!current.contains("rhs")||!current["lhs"].is_string()||!current["rhs"].is_string()) {error="linear_algebra state requires lhs and rhs strings";return false;}
        const auto old_lhs=current["lhs"].get<std::string>();
        const auto old_rhs=current["rhs"].get<std::string>();
        const auto new_lhs=f.count("lhs")?f["lhs"]:old_lhs;
        const auto new_rhs=f.count("rhs")?f["rhs"]:old_rhs;
        linear before,after; const std::string old_eq=old_lhs+"="+old_rhs; const std::string new_eq=new_lhs+"="+new_rhs;
        if(!parse_equation(old_eq,before)||!parse_equation(new_eq,after)) {out.reason="expression is outside the supported exact linear fragment";out.verdict=common_semi_oracle_verdict::unknown;return true;}
        if(q.level!=common_semi_oracle_level::local && q.goal=="solve:x") {
            if(before.a.zero()) {out.verdict=before.b.zero()?common_semi_oracle_verdict::satisfied:common_semi_oracle_verdict::violated;out.reason=before.b.zero()?"every value of x satisfies the equation":"equation has no solution";if(out.verdict==common_semi_oracle_verdict::satisfied)out.proof_json=json{{"solution","all_real_x"},{"normalized_a",before.a.str()},{"normalized_b",before.b.str()}}.dump();else out.counterexample_json=json{{"normalized_a",before.a.str()},{"normalized_b",before.b.str()}}.dump();out.expansions=1;return true;}
            common_semi_oracle_search_problem problem;
            const std::string initial=json{{"lhs",old_lhs},{"rhs",old_rhs}}.dump();
            if(!build_algebra_search_problem(initial,
                    q.level==common_semi_oracle_level::guided?q.guidance:std::vector<std::string>{},
                    problem)) {
                out.verdict=common_semi_oracle_verdict::unknown;
                out.reason="could not construct a supported algebra rewrite search problem";
                return true;
            }
            common_semi_oracle_search_result searched;
            problem.transition_adviser = q.transition_adviser;
            problem.record_history = q.record_search_history;
            if(!common_semi_oracle_search_astar(problem,q.budget,searched)) {
                out.verdict=common_semi_oracle_verdict::unknown;
                out.reason=searched.error.empty()?"bounded algebra A* failed":searched.error;
                return true;
            }
            out.expansions=searched.expansions;out.depth_reached=searched.depth_reached;
            if(searched.found) {
                json path=json::array();
                for(const auto & step:searched.path) {
                    path.push_back({{"state",json::parse(step.state,nullptr,false)},
                        {"action",step.action},{"cost",step.cost}});
                }
                const auto final_state=json::parse(searched.path.back().state,nullptr,false);
                linear final_lhs,final_rhs;
                if(!final_state.is_object()||!equation_state(searched.path.back().state,final_lhs,final_rhs)||
                        final_lhs.a.n!=1||final_lhs.a.d!=1||!final_lhs.b.zero()||
                        !final_rhs.a.zero()) {
                    out.verdict=common_semi_oracle_verdict::unknown;
                    out.reason="A* returned a path whose final equation could not be validated";
                    return true;
                }
                out.verdict=common_semi_oracle_verdict::satisfied;
                out.proof_json=json{{"algorithm","bounded_astar_v1"},{"heuristic","zero_admissible"},
                    {"optimality_proven",searched.optimality_proven},{"goal",q.goal},
                    {"before",{{"lhs",old_lhs},{"rhs",old_rhs}}},
                    {"ruleset_id",r.id},{"ruleset_revision",r.revision},
                    {"rule_fingerprint",r.semantic_fingerprint},
                    {"action_contract_fingerprint",r.action_contract_fingerprint},
                    {"solution",final_rhs.b.str()},{"total_cost",searched.total_cost},
                    {"expansions",searched.expansions},{"depth_reached",searched.depth_reached},
                    {"path",path}}.dump();
                out.reason="bounded A* found a path of exact-equivalence-preserving rewrites to x = rational";
            } else if(searched.exhausted) {
                out.verdict=common_semi_oracle_verdict::violated;
                out.counterexample_json=json{{"algorithm","bounded_astar_v1"},{"goal",q.goal},
                    {"ruleset_id",r.id},{"ruleset_revision",r.revision},
                    {"expansions",searched.expansions},{"result","goal_unreachable_in_complete_transition_graph"}}.dump();
                out.reason="A* exhausted the complete supported rewrite graph without reaching a unique rational solution";
            } else {
                out.verdict=common_semi_oracle_verdict::unknown;
                out.reason="algebra A* reached its expansion or depth bound before proving a path";
            }
            if (q.record_search_history) {
                json observations = json::array();
                for (const auto & item : searched.observations)
                    observations.push_back({{"action", item.action}, {"outcome", item.outcome}});
                out.search_guidance_json = json{{"observations", observations},
                    {"callback_fallbacks", searched.guidance_fallbacks},
                    {"callback_error", searched.guidance_error}}.dump();
            }
            return true;
        }
        const bool no_change = compact_expression(old_lhs) == compact_expression(new_lhs) &&
                compact_expression(old_rhs) == compact_expression(new_rhs);
        const bool valid=proportional_nonzero(before,after); out.verdict=no_change?common_semi_oracle_verdict::violated:(valid?common_semi_oracle_verdict::satisfied:common_semi_oracle_verdict::violated);
        json evidence={{"before",{{"a",before.a.str()},{"b",before.b.str()}}},{"after",{{"a",after.a.str()},{"b",after.b.str()}}}};
        if(valid&&!no_change) {
            evidence["next_state"]={{"lhs",new_lhs},{"rhs",new_rhs}};
            out.proof_json=evidence.dump();
        } else out.counterexample_json=evidence.dump();
        out.reason=no_change?"proposed rewrite makes no textual change to the equation":
                (valid?"equation equivalence proven by exact rational normalization":"normalized equation changed under the proposed rewrite");
        out.expansions=1; return true;
    }
    json state; int rows=0,cols=0,win=0;
    if(!game_state(r,q.state_json,state,rows,cols,win,error)) {out.reason=error;out.verdict=common_semi_oracle_verdict::unknown;error.clear();return true;}
    const auto s=spec(r); const auto players=s.value("players",std::vector<std::string>{});
    if(q.level==common_semi_oracle_level::local) {
        if(!optional_header_matches(q.action_dsl,"open! game.place_mark",error)) return false;
        std::map<std::string,std::string> f; if(!parse_lines(q.action_dsl,f,error)) return false;
        if(f.size()!=3||!f.count("player")||!f.count("row")||!f.count("column")) {error="place_mark requires player, row and column";return false;}
        int y=0,x=0; if(!number(f["row"],y)||!number(f["column"],x)) {error="row and column must be integers";return false;}
        std::string why;
        if(y<0||y>=rows||x<0||x>=cols) why="placement is outside the board";
        else if(f["player"]!=state["turn"]) why="move is not for the current player";
        else if(f["player"]!=players[0]&&f["player"]!=players[1]) why="move uses an undeclared player";
        else if(!state["cells"][static_cast<size_t>(y*cols+x)].is_null()) why="target cell is occupied";
        else if(game_won(state["cells"],rows,cols,win,players[0])||game_won(state["cells"],rows,cols,win,players[1])) why="game is already terminal";
        if(!why.empty()) {out.verdict=common_semi_oracle_verdict::violated;out.reason=why;out.counterexample_json=json{{"state",state},{"action",f},{"violation",why}}.dump();return true;}
        auto next=state; next["cells"][static_cast<size_t>(y*cols+x)]=f["player"];
        next["turn"]=(f["player"]==players[0]?players[1]:players[0]);
        const bool won=game_won(next["cells"],rows,cols,win,f["player"]);
        out.verdict=common_semi_oracle_verdict::satisfied; out.proof_json=json{{"state",state},{"action",f},{"next_state",next},{"terminal_win",won}}.dump(); out.reason="legal placement transition verified";out.expansions=1;return true;
    }
    if(q.goal.rfind("win:",0)!=0) {out.verdict=common_semi_oracle_verdict::unknown;out.reason="bounded placement_game query requires goal `win:PLAYER`";return true;}
    const auto target=q.goal.substr(4); if(target!=players[0]&&target!=players[1]) {out.verdict=common_semi_oracle_verdict::unknown;out.reason="goal names an undeclared player";return true;}
    // Bounded minimax. Guidance only reorders legal placements.
    struct search_result { int value=0; json line=json::array(); }; // +1 forced win, -1 disproven within complete tree, 0 budget/depth unknown
    const auto start_cells=state["cells"];
    std::function<search_result(json,std::string,size_t)> visit;
    visit=[&](json cells,std::string turn,size_t depth)->search_result {
        if(game_won(cells,rows,cols,win,target)) return {1,json::array()};
        const auto opponent=target==players[0]?players[1]:players[0];
        if(game_won(cells,rows,cols,win,opponent)) return {-1,json::array()};
        bool full=true; std::vector<std::string> moves;
        for(size_t i=0;i<cells.size();++i) if(cells[i].is_null()) {full=false; moves.push_back(std::to_string(i/cols)+","+std::to_string(i%cols));}
        if(full) return {-1,json::array()};
        if(depth>=q.budget.max_depth||out.expansions>=q.budget.max_expansions) return {0,json::array()};
        if(q.level==common_semi_oracle_level::guided) std::stable_sort(moves.begin(),moves.end(),[&](const auto & a,const auto & b){auto ia=std::find(q.guidance.begin(),q.guidance.end(),a),ib=std::find(q.guidance.begin(),q.guidance.end(),b);return ia<ib;});
        bool unknown=false;
        for(const auto & m:moves) {
            ++out.expansions; const auto comma=m.find(','); int y=std::stoi(m.substr(0,comma)),x=std::stoi(m.substr(comma+1)); auto next=cells;next[static_cast<size_t>(y*cols+x)]=turn;
            const auto next_turn=turn==players[0]?players[1]:players[0]; auto child=visit(std::move(next),next_turn,depth+1);
            if(turn==target&&child.value==1) {child.line.insert(child.line.begin(),m);out.depth_reached=std::max(out.depth_reached,depth+1);return child;}
            if(turn!=target&&child.value==-1) {child.line.insert(child.line.begin(),m);out.depth_reached=std::max(out.depth_reached,depth+1);return child;}
            if(child.value==0) unknown=true;
        }
        if(unknown) return {0,json::array()};
        return {turn==target?-1:1,json::array()};
    };
    auto searched=visit(start_cells,state["turn"].get<std::string>(),0);
    out.verdict=searched.value==1?common_semi_oracle_verdict::satisfied:searched.value==-1?common_semi_oracle_verdict::violated:common_semi_oracle_verdict::unknown;
    if(searched.value==1) out.proof_json=json{{"goal",q.goal},{"principal_variation",searched.line},{"expansions",out.expansions}}.dump();
    else if(searched.value==-1) out.counterexample_json=json{{"goal",q.goal},{"result","goal_not_forced_within_bounded_search"},{"expansions",out.expansions}}.dump();
    else out.reason="search budget or depth ended before a proof or counterexample";
    if(out.guidance_ref.empty()) out.reason=searched.value==0?out.reason:"bounded deterministic game-tree search completed";
    else if(searched.value!=0) out.reason="guided bounded search found a verified result; guidance affected ordering only";
    return true;
}

bool common_semi_oracle_learning_authorized(const common_semi_oracle_ruleset & r, const common_semi_oracle_result & result) {
    if(r.domain==common_semi_oracle_domain::target_hunt) return false;
    if(!r.active||result.ruleset_id!=r.id||result.ruleset_revision!=r.revision||result.verdict==common_semi_oracle_verdict::unknown||
            result.evaluator_ref!=std::string("semi-oracle://evaluator/")+common_semi_oracle_domain_name(r.domain)||
            (result.evaluator_revision!="v1"&&result.evaluator_revision!="linear_astar_v1")) return false;
    const auto proof=json::parse(result.proof_json,nullptr,false);
    const auto counterexample=json::parse(result.counterexample_json,nullptr,false);
    if(result.verdict==common_semi_oracle_verdict::satisfied&&!proof.is_object()) return false;
    if(result.verdict==common_semi_oracle_verdict::violated&&!counterexample.is_object()) return false;
    return std::all_of(r.assertions.begin(),r.assertions.end(),[](const auto & a){return a.authority!=common_semi_oracle_rule_authority::inferred;});
}

bool common_semi_oracle_rank_placements(
        const common_semi_oracle_ruleset & r,
        const std::string & state_json,
        common_semi_oracle_budget budget,
        std::vector<common_semi_oracle_placement_action_value> & values,
        std::string & error) {
    values.clear();
    error.clear();
    if (!r.active || r.domain != common_semi_oracle_domain::placement_game ||
            !common_semi_oracle_validate_ruleset(r, error)) {
        if (error.empty()) error = "placement action ranking requires an active placement_game RuleSet";
        return false;
    }
    if (budget.max_expansions == 0 || budget.max_expansions > 100000 ||
            budget.max_depth == 0 || budget.max_depth > 64) {
        error = "semi-oracle search budget is outside the supported bound";
        return false;
    }

    json state;
    int rows = 0, columns = 0, win_length = 0;
    if (!game_state(r, state_json, state, rows, columns, win_length, error)) return false;
    const auto players = spec(r).value("players", std::vector<std::string>{});
    const std::string root_player = state["turn"].get<std::string>();
    if (game_won(state["cells"], rows, columns, win_length, players[0]) ||
            game_won(state["cells"], rows, columns, win_length, players[1])) {
        error = "cannot rank actions from a terminal placement_game state";
        return false;
    }

    const std::string canonical_state = state.dump();
    std::ostringstream state_identity;
    state_identity << "fnv1a64:" << std::hex << fingerprint(canonical_state);
    std::map<std::string, int> memo;
    size_t expansions = 0;
    size_t max_depth_reached = 0;

    std::function<int(const json &, const std::string &, size_t)> solve;
    solve = [&](const json & cells, const std::string & turn, size_t depth) -> int {
        if (game_won(cells, rows, columns, win_length, root_player)) return 1;
        const std::string other = root_player == players[0] ? players[1] : players[0];
        if (game_won(cells, rows, columns, win_length, other)) return -1;
        bool full = true;
        for (const auto & cell : cells) if (cell.is_null()) { full = false; break; }
        if (full) return 0;

        const std::string key = cells.dump() + "|" + turn;
        const auto known = memo.find(key);
        if (known != memo.end()) return known->second;
        if (depth >= budget.max_depth || expansions >= budget.max_expansions) {
            return 2;
        }
        ++expansions;
        max_depth_reached = std::max(max_depth_reached, depth);

        const bool maximizing = turn == root_player;
        int best = maximizing ? -1 : 1;
        bool unknown_child = false;
        for (size_t i = 0; i < cells.size(); ++i) {
            if (!cells[i].is_null()) continue;
            auto next = cells;
            next[i] = turn;
            const std::string next_turn = turn == players[0] ? players[1] : players[0];
            const int child = solve(next, next_turn, depth + 1);
            if (child == 2) {
                unknown_child = true;
                continue;
            }
            if (maximizing) {
                best = std::max(best, child);
                if (best == 1) { memo.emplace(key, best); return best; }
            } else {
                best = std::min(best, child);
                if (best == -1) { memo.emplace(key, best); return best; }
            }
        }
        if (unknown_child) return 2;
        memo.emplace(key, best);
        return best;
    };

    for (size_t i = 0; i < state["cells"].size(); ++i) {
        if (!state["cells"][i].is_null()) continue;
        const size_t expansions_before_action = expansions;
        const int row = static_cast<int>(i / static_cast<size_t>(columns));
        const int column = static_cast<int>(i % static_cast<size_t>(columns));
        auto next = state["cells"];
        next[i] = root_player;
        int value = 0;
        if (game_won(next, rows, columns, win_length, root_player)) {
            value = 1;
        } else {
            bool full = true;
            for (const auto & cell : next) if (cell.is_null()) { full = false; break; }
            if (!full) {
                const std::string next_turn = root_player == players[0] ? players[1] : players[0];
                value = solve(next, next_turn, 1);
            }
        }

        common_semi_oracle_placement_action_value action;
        action.ruleset_id = r.id;
        action.ruleset_revision = r.revision;
        action.evaluator_ref = "semi-oracle://evaluator/placement_game";
        action.evaluator_revision = "minimax-v1";
        action.state_fingerprint = state_identity.str();
        action.player = root_player;
        action.row = row;
        action.column = column;
        action.expansions = expansions - expansions_before_action;
        action.depth_reached = max_depth_reached;
        if (value == 2) {
            action.value = common_semi_oracle_placement_value::unknown;
        } else {
            action.value = static_cast<common_semi_oracle_placement_value>(value);
            const char * label = value > 0 ? "win" : value < 0 ? "loss" : "draw";
            action.proof_json = json{
                {"algorithm", "memoized_minimax_v1"},
                {"expansions", action.expansions},
                {"ruleset_id", r.id},
                {"ruleset_revision", r.revision},
                {"evaluator_ref", action.evaluator_ref},
                {"evaluator_revision", action.evaluator_revision},
                {"state_fingerprint", action.state_fingerprint},
                {"state", state},
                {"player", root_player},
                {"action", {{"row", row}, {"column", column}}},
                {"outcome", label},
                {"rule_fingerprint", r.semantic_fingerprint},
                {"action_contract_fingerprint", r.action_contract_fingerprint},
            }.dump();
        }
        values.push_back(std::move(action));
    }
    if (values.empty()) {
        error = "placement_game state has no legal actions";
        return false;
    }
    return true;
}

bool common_semi_oracle_placement_state_group(
        const common_semi_oracle_ruleset & r,
        const std::string & state_json,
        std::string & group_key,
        std::string & error) {
    group_key.clear();
    error.clear();
    if (!r.active || r.domain != common_semi_oracle_domain::placement_game ||
            !common_semi_oracle_validate_ruleset(r, error)) {
        if (error.empty()) error = "position grouping requires an active placement_game RuleSet";
        return false;
    }
    json state;
    int rows = 0, columns = 0, win_length = 0;
    if (!game_state(r, state_json, state, rows, columns, win_length, error)) return false;
    (void) win_length;

    std::vector<std::string> variants;
    const auto & cells = state["cells"];
    const int reflection_count = rows == columns ? 2 : 4;
    const int rotations = rows == columns ? 4 : 1;
    for (int reflected = 0; reflected < reflection_count; ++reflected) {
        for (int rotation = 0; rotation < rotations; ++rotation) {
            std::vector<std::string> transformed(static_cast<size_t>(rows * columns));
            for (int row = 0; row < rows; ++row) {
                for (int column = 0; column < columns; ++column) {
                    int r0 = (rows != columns && (reflected & 1)) ? rows - 1 - row : row;
                    int c0 = (rows == columns ? (reflected ? columns - 1 - column : column) :
                        ((reflected & 2) ? columns - 1 - column : column));
                    for (int i = 0; i < rotation; ++i) {
                        const int next_r = c0;
                        const int next_c = rows - 1 - r0;
                        r0 = next_r;
                        c0 = next_c;
                    }
                    if (r0 < 0 || r0 >= rows || c0 < 0 || c0 >= columns) continue;
                    const auto & cell = cells[static_cast<size_t>(row * columns + column)];
                    transformed[static_cast<size_t>(r0 * columns + c0)] = cell.is_null()
                        ? "_" : cell.get<std::string>();
                }
            }
            std::ostringstream canonical;
            canonical << r.semantic_fingerprint << '|' << r.revision << '|'
                      << state["turn"].get<std::string>() << '|';
            for (const auto & cell : transformed) canonical << cell << ';';
            variants.push_back(canonical.str());
        }
    }
    const auto & canonical = *std::min_element(variants.begin(), variants.end());
    std::ostringstream key;
    key << "semi-oracle://position-group/fnv1a64:" << std::hex << fingerprint(canonical);
    group_key = key.str();
    return true;
}
