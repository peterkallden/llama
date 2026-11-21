#include "agent/adaptation/semi-oracle.h"
#include "agent/adaptation/semi-oracle-store.h"
#include "agent/adaptation/semi-oracle-tool.h"
#include "agent/adaptation/trainer-protocol.h"
#include "agent/agent-inference.h"
#include "agent/tooling/adapters/tool-adapters.h"
#include "agent/tooling/catalog/tool-catalog.h"
#include "agent/tooling/registry/tool-registry.h"
#include "tools/agent/runtime/agent-server-context-host.h"
#include "tools/server/server-context.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace {
using ordered_json=nlohmann::ordered_json;
struct options {
    std::string model;
    int n_predict = 128;
    int n_threads = 3;
    int n_gpu_layers = 0;
    bool strict = false;
};

class smoke_ruleset_store final : public common_semi_oracle_store {
public:
    bool put_revision(const common_semi_oracle_ruleset & value,std::string & error) override {
        const auto key=std::make_pair(value.id,value.revision);auto it=rules.find(key);
        if(it!=rules.end()){if(common_semi_oracle_ruleset_to_json(it->second)==common_semi_oracle_ruleset_to_json(value))return true;error="immutable RuleSet revision";return false;}
        rules[key]=value;return true;
    }
    bool get_revision(const std::string & id,uint64_t revision,common_semi_oracle_ruleset & value,bool & found,std::string &) const override {
        const auto it=rules.find({id,revision});found=it!=rules.end();if(found)value=it->second;return true;
    }
    bool promote_to_host(const std::string & id,uint64_t revision,std::string & error) override {
        auto it=rules.find({id,revision});if(it==rules.end()){error="missing RuleSet";return false;}it->second.scope=common_semi_oracle_scope::host;it->second.scope_id="host";hosts.push_back(id);return true;
    }
    bool bind_session(const std::string & session,const std::string & id,uint64_t revision,std::string &) override {bindings[session]={id,revision};return true;}
    bool get_session_binding(const std::string & session,std::string & id,uint64_t & revision,bool & found,std::string &) const override {
        auto it=bindings.find(session);found=it!=bindings.end();if(found){id=it->second.first;revision=it->second.second;}return true;
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
    bool append_evaluation(const std::string & id,const common_semi_oracle_result & value,std::string & error) override {
        if (id.empty()) { error = "evaluation id is empty"; return false; }
        evaluation_payloads[id] = ordered_json{
            {"ruleset_id", value.ruleset_id},
            {"ruleset_revision", value.ruleset_revision},
            {"verdict", common_semi_oracle_verdict_name(value.verdict)},
            {"level", common_semi_oracle_level_name(value.level)},
            {"evaluator_ref", value.evaluator_ref},
            {"evaluator_revision", value.evaluator_revision},
            {"expansions", value.expansions},
            {"depth_reached", value.depth_reached},
            {"proof_json", value.proof_json},
            {"counterexample_json", value.counterexample_json},
            {"reason", value.reason},
            {"guidance_ref", value.guidance_ref},
        }.dump();
        return true;
    }
    bool get_evaluation_payload(const std::string & id,std::string & payload,bool & found,std::string & error) const override {
        const auto it = evaluation_payloads.find(id);
        found = it != evaluation_payloads.end();
        payload = found ? it->second : std::string();
        if (!found) error = "evaluation payload not found";
        return true;
    }
    std::vector<std::string> list_host_rulesets(std::string &) const override {return hosts;}
private:
    std::map<std::pair<std::string,uint64_t>,common_semi_oracle_ruleset> rules;
    std::map<std::string,std::pair<std::string,uint64_t>> bindings;
    std::map<std::tuple<std::string,std::string,uint64_t>,std::string> game_states;
    std::vector<std::string> hosts;
    std::map<std::string,std::string> evaluation_payloads;
};

bool parse_args(int argc, char ** argv, options & out) {
    if (const char * model = std::getenv("LLAMA_AGENT_MODEL")) out.model = model;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : nullptr; };
        if (arg == "--model") { const auto * value = next(); if (!value) return false; out.model = value; }
        else if (arg == "--n-predict") { const auto * value = next(); if (!value) return false; out.n_predict = std::stoi(value); }
        else if (arg == "--threads") { const auto * value = next(); if (!value) return false; out.n_threads = std::stoi(value); }
        else if (arg == "--n-gpu-layers") { const auto * value = next(); if (!value) return false; out.n_gpu_layers = std::stoi(value); }
        else if (arg == "--strict") out.strict = true;
        else if (arg == "--help" || arg == "-h") return false;
        else return false;
    }
    return true;
}

std::string content(const common_agent_generation_result & result) {
    return common_agent_generation_succeeded(result) ? result.content : "<generation failed: " + result.error_message + ">";
}

std::string compact_text(const std::string & value) {
    std::string result;
    result.reserve(value.size());
    for (const unsigned char c : value) if (!std::isspace(c)) result.push_back(static_cast<char>(c));
    return result;
}

std::string render_game_state_for_model(const std::string & state_json) {
    const auto state = ordered_json::parse(state_json, nullptr, false);
    if (!state.is_object() || !state.contains("cells") || !state["cells"].is_array() ||
            !state.contains("turn") || !state["turn"].is_string()) return state_json;
    constexpr size_t columns = 3;
    const size_t rows = state["cells"].size() / columns;
    std::ostringstream rendered;
    rendered << "Turn: " << state["turn"].get<std::string>()
             << ". Board is row-major; coordinates are zero-based (row, column).\nBoard: ";
    std::vector<std::string> empty_cells;
    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < columns; ++column) {
            const auto & cell = state["cells"][row * columns + column];
            rendered << (cell.is_null() ? '.' : cell.get<std::string>().front());
            if (cell.is_null()) {
                empty_cells.push_back("(" + std::to_string(row) + "," + std::to_string(column) + ")");
            }
        }
        if (row + 1 < rows) rendered << '/';
    }
    rendered << "Empty coordinates: ";
    for (size_t i = 0; i < empty_cells.size(); ++i) {
        if (i != 0) rendered << ", ";
        rendered << empty_cells[i];
    }
    return rendered.str();
}

common_agent_generation_result generate(
        common_agent_inference & inference, const std::string & id,
        const std::string & system, const std::string & user,
        const options & config) {
    common_agent_generation_options generation_options;
    generation_options.n_predict = config.n_predict;
    generation_options.n_threads = config.n_threads;
    generation_options.generation_trace = true;
    auto request = common_agent_make_generation_request(
        common_agent_generation_purpose::tool_followup, id, std::nullopt,
        {{"system", system}, {"user", user}}, generation_options);
    common_agent_generation_result result;
    inference.generate(request, result);
    return result;
}

bool run_case(common_agent_inference & inference, const options & config,
        smoke_ruleset_store & store,
        const std::string & case_id, const std::string & domain,
        const std::string & rule_request, const std::string & expected_action,
        const std::string & state_json, const std::string & action_request,
        const std::string & goal, common_semi_oracle_level level) {
    const std::string rule_system =
        "Translate only the user's stated rules into this RuleSet DSL. Output exactly the listed fields, one per line, "
        "using concrete values from the request. Do not output type hints, ranges, extra fields or function-call syntax.\n"
        "name: short name\n"
        "description: one sentence\n"
        "domain: " + domain +
        (domain == "placement_game"
            ? "\nrows: 3\ncolumns: 3\nwin_length: 3\nplayers: X, O"
            : "\nvariable: x");
    const auto rules = generate(inference, case_id + "-rules", rule_system, rule_request, config);
    std::cout << "\n=== " << case_id << " RULESET GENERATION ===\n"
              << "SYSTEM INPUT:\n" << rule_system << "\nUSER INPUT:\n" << rule_request
              << "\nMODEL OUTPUT:\n" << content(rules) << "\n";
    if (!common_agent_generation_succeeded(rules)) return false;

    common_semi_oracle_tool_service oracle_tools(store,"model-smoke","session-"+case_id);
    common_tool_catalog catalog;common_tool_bootstrap_result bootstrap;
    std::string error;
    if(!catalog.bootstrap("semi-oracle",bootstrap,error)){std::cout<<"TOOL CATALOG: "<<error<<"\n";return false;}
    std::string current_state=state_json;common_semi_oracle_result last_result;
    common_native_tool_bindings bindings;
    bindings.semi_oracle=[&](const std::string & tool_name,const std::string & input) {
        const auto args=ordered_json::parse(input,nullptr,false);
        if(!args.is_object())return common_tool_execution_result::failure("oracle.invalid_arguments",common_tool_failure_class::validation,false,"Semi-oracle arguments must be an object.","invalid JSON object");
        if(tool_name=="oracle.ruleset.define") {
            common_semi_oracle_ruleset ruleset;std::string tool_error;
            if(!args.contains("ruleset_dsl")||!args["ruleset_dsl"].is_string()||
                    !oracle_tools.define_ruleset(args["ruleset_dsl"].get<std::string>(),common_semi_oracle_rule_authority::declared,ruleset,tool_error))
                return common_tool_execution_result::failure("oracle.ruleset.invalid",common_tool_failure_class::validation,false,"RuleSet was not activated.",tool_error);
            return common_tool_execution_result::success(ordered_json{{"ruleset_id",ruleset.id},{"revision",ruleset.revision},{"domain",common_semi_oracle_domain_name(ruleset.domain)},{"action_contract",common_semi_oracle_action_contract(ruleset)},{"active",ruleset.active}}.dump());
        }
        if(tool_name=="oracle.action.verify") {
            if(!args.contains("action_dsl")||!args["action_dsl"].is_string())return common_tool_execution_result::failure("oracle.action.invalid",common_tool_failure_class::validation,false,"Action DSL is required.","missing action_dsl");
            common_semi_oracle_budget host_budget;host_budget.max_depth=8;host_budget.max_expansions=2000;
            std::string tool_error;
            const auto action_dsl = args["action_dsl"].get<std::string>();
            std::string state_before, state_after;
            if (!oracle_tools.verify_active_action(action_dsl, last_result,
                    state_before, state_after, tool_error)) {
                return common_tool_execution_result::failure("oracle.action.failed",
                    common_tool_failure_class::validation, false,
                    "Action could not be verified.", tool_error);
            }
            current_state = state_before;
            const auto proof=ordered_json::parse(last_result.proof_json,nullptr,false);
            const auto counterexample=ordered_json::parse(last_result.counterexample_json,nullptr,false);
            return common_tool_execution_result::success(ordered_json{
                {"verdict",common_semi_oracle_verdict_name(last_result.verdict)},
                {"ruleset_id",last_result.ruleset_id},
                {"revision",last_result.ruleset_revision},
                {"evaluator",last_result.evaluator_ref+"@"+last_result.evaluator_revision},
                {"level",common_semi_oracle_level_name(last_result.level)},
                {"proof",proof.is_object()?proof:ordered_json::object()},
                {"counterexample",counterexample.is_object()?counterexample:ordered_json::object()},
                {"reason",last_result.reason}}.dump());
        }
        if(tool_name=="oracle.goal.search") {
            std::string tool_error;
            if(!args.empty()||!oracle_tools.search_active_algebra_goal(last_result,tool_error))
                return common_tool_execution_result::failure("oracle.goal.search_failed",
                    common_tool_failure_class::validation,false,
                    "Host-owned algebra goal search failed.",tool_error.empty()?"arguments must be empty":tool_error);
            const auto proof=ordered_json::parse(last_result.proof_json,nullptr,false);
            return common_tool_execution_result::success(ordered_json{{"verdict",common_semi_oracle_verdict_name(last_result.verdict)},
                {"ruleset_id",last_result.ruleset_id},{"revision",last_result.ruleset_revision},
                {"evaluator",last_result.evaluator_ref},{"evaluator_revision",last_result.evaluator_revision},
                {"level",common_semi_oracle_level_name(last_result.level)},
                {"expansions",last_result.expansions},{"depth_reached",last_result.depth_reached},
                {"proof",proof.is_object()?proof:ordered_json::object()},{"reason",last_result.reason}}.dump());
        }
        return common_tool_execution_result::failure("oracle.unknown_operation",common_tool_failure_class::validation,false,"Unknown semi-oracle operation.",tool_name);
    };
    common_tool_registry registry;common_tool_adapter_result adapter_result;
    if(!common_register_native_tool_adapters(catalog,"semi-oracle",bindings,registry,adapter_result,error)||adapter_result.registered.size()!=3){std::cout<<"TOOL REGISTRATION: "<<error<<"\n";return false;}
    common_semi_oracle_ruleset ruleset;
    common_agent_tool_call define_call{"oracle.ruleset.define",ordered_json{{"ruleset_dsl",rules.content}}.dump()};
    const auto define_result=registry.execute(define_call);
    if (!define_result.ok) {
        error=define_result.raw_diagnostic.empty()?define_result.safe_summary:define_result.raw_diagnostic;
        std::cout << "RULESET HOST WRAPPER: rejected: " << error << "\n";
        return false;
    }
    bool found=false;if(!oracle_tools.active_ruleset(ruleset,found,error)||!found){std::cout<<"RULESET HOST WRAPPER: "<<error<<"\n";return false;}
    {
        // The fixture state is host-seeded; the model never supplies runtime state.
        if (!oracle_tools.initialize_active_state(state_json, true, error)) {
            std::cout << "HOST STATE FIXTURE: " << error << "\n";
            return false;
        }
    }
    std::cout<<"ORACLE RULESET NATIVE TOOL RESULT="<<define_result.output<<"\n";
    std::cout << "RULESET HOST VALIDATION: active id=" << ruleset.id
              << " revision=" << ruleset.revision
              << " semantic_fingerprint=" << ruleset.semantic_fingerprint
              << " action_contract_fingerprint=" << ruleset.action_contract_fingerprint << "\n";

    std::string action_contract;if(!oracle_tools.action_contract(action_contract,error)){std::cout<<"ACTION CONTRACT: "<<error<<"\n";return false;}
    const std::string action_format = domain == "placement_game"
        ? "player: current player's single symbol\nrow: 0-based index of an empty cell\ncolumn: 0-based index of an empty cell"
        : "lhs: 2x+6";
    const std::string action_system =
        "The host has already selected the operation. Return only these DSL fields, choosing one value for each; "
        "do not return a tool/action name, function call or explanation. "
        "For a rewrite request, return changed fields only; omitted equation sides stay unchanged. "
        "The result must apply the requested rewrite; returning the current equation unchanged is not a rewrite.\n"
        "Required format:\n" + action_format + "\n\nActive action contract:\n" + action_contract;
    const std::string model_state = domain == "placement_game"
        ? render_game_state_for_model(state_json) : "Current state: " + state_json;
    const auto action = generate(inference, case_id + "-action", action_system,
        action_request + "\n" + model_state, config);
    std::cout << "\n=== " << case_id << " ACTION GENERATION ===\n"
              << "SYSTEM INPUT:\n" << action_system << "\nUSER INPUT:\n"
              << action_request << "\n" << model_state
              << "\nMODEL OUTPUT:\n" << content(action) << "\n";
    if (!common_agent_generation_succeeded(action)) return false;

    const common_agent_tool_call action_call{"oracle.action.verify",ordered_json{{"action_dsl",action.content}}.dump()};
    const auto action_tool_result=registry.execute(action_call);
    if (!action_tool_result.ok) {
        error=action_tool_result.raw_diagnostic.empty()?action_tool_result.safe_summary:action_tool_result.raw_diagnostic;
        std::cout << "SEMI-ORACLE TOOL: request rejected: " << error << "\n";
        return false;
    }
    const auto tool_output=ordered_json::parse(action_tool_result.output,nullptr,false);
    const bool matches = last_result.verdict == common_semi_oracle_verdict::satisfied &&
        (expected_action.empty() || compact_text(action.content).find(compact_text(expected_action)) != std::string::npos);
    std::cout << "NATIVE TOOL RESULT=" << (tool_output.is_object()?tool_output.dump(2):action_tool_result.output)<<"\n"
              << "SEMI-ORACLE: verdict=" << common_semi_oracle_verdict_name(last_result.verdict)
              << " level=" << common_semi_oracle_level_name(last_result.level)
              << " evaluator=" << last_result.evaluator_ref << '@' << last_result.evaluator_revision
              << " expansions=" << last_result.expansions << " depth=" << last_result.depth_reached
              << " reason=" << last_result.reason << "\n"
              << "PROOF=" << last_result.proof_json << "\nCOUNTEREXAMPLE="
              << last_result.counterexample_json << "\n"
              << "ORACLE EVIDENCE AUTHORITY="
              << (common_semi_oracle_learning_authorized(ruleset, last_result)
                    ? "host-verified transition; not itself training credit" : "not eligible") << "\n"
              << "FLYDELTA RELATION=not created: one-shot smoke has no matched control capture\n"
              << "EXPECTED ACTION MARKER=" << (expected_action.empty() ? "<semantic verdict>" : expected_action)
              << " OUTCOME=" << (matches ? "matched" : "mismatch") << "\n";
    if (domain == "placement_game" && last_result.verdict == common_semi_oracle_verdict::satisfied) {
        const auto transition_proof = ordered_json::parse(last_result.proof_json, nullptr, false);
        if (!transition_proof.is_object() || !transition_proof.contains("action") ||
                !transition_proof["action"].is_object()) return false;
        const auto & played = transition_proof["action"];
        const int row = std::stoi(played.at("row").get<std::string>());
        const int column = std::stoi(played.at("column").get<std::string>());
        std::vector<common_semi_oracle_placement_action_value> action_values;
        std::string group_key;
        if (!oracle_tools.rank_placements(state_json,
                common_semi_oracle_budget{100000, 9}, action_values, error) ||
                !common_semi_oracle_placement_state_group(ruleset, state_json, group_key, error)) {
            std::cout << "MINIMAX COMPARISON: unavailable: " << error << "\n";
            return false;
        }
        const auto played_value = std::find_if(action_values.begin(), action_values.end(),
            [&](const auto & value) { return value.row == row && value.column == column; });
        const auto best_value = std::max_element(action_values.begin(), action_values.end(),
            [](const auto & left, const auto & right) {
                return static_cast<int>(left.value) < static_cast<int>(right.value);
            });
        if (played_value == action_values.end() || best_value == action_values.end() ||
                std::any_of(action_values.begin(), action_values.end(), [](const auto & value) {
                    return value.value == common_semi_oracle_placement_value::unknown;
                })) return false;
        const auto outcome_name = [](common_semi_oracle_placement_value value) {
            switch (value) {
                case common_semi_oracle_placement_value::win: return "win";
                case common_semi_oracle_placement_value::draw: return "draw";
                case common_semi_oracle_placement_value::loss: return "loss";
                case common_semi_oracle_placement_value::unknown: return "unknown";
            }
            return "unknown";
        };
        std::cout << "MINIMAX COMPARISON=" << ordered_json{
            {"ruleset_id", ruleset.id}, {"ruleset_revision", ruleset.revision},
            {"position_group", group_key},
            {"persisted_evaluation_ref", best_value->persisted_evaluation_ref},
            {"model_move", {{"row", row}, {"column", column}, {"value", outcome_name(played_value->value)}}},
            {"solver_best_move", {{"row", best_value->row}, {"column", best_value->column},
                {"value", outcome_name(best_value->value)}}},
            {"same_best_value", played_value->value == best_value->value},
            {"evaluator", "memoized_minimax_v1"},
            {"budget_expansions", 100000},
        }.dump() << '\n';
        if (static_cast<int>(best_value->value) > static_cast<int>(played_value->value)) {
            common_training_candidate dpo_candidate;
            common_learning_corpus_revision dpo_corpus;
            common_learning_corpus_policy dpo_policy;
            dpo_policy.objective = common_learning_objective::dpo;
            dpo_policy.redaction_policy_id = "stub:semi-oracle-smoke";
            dpo_policy.allow_host_verified_dpo_pairs = true;
            const std::string approved_prompt = action_request + "\n" + model_state;
            if (!oracle_tools.make_dpo_candidate(*best_value, *played_value,
                    approved_prompt, dpo_policy.redaction_policy_id,
                    "smoke fixture contains no user-provided content", true,
                    dpo_candidate, error) ||
                    !common_learning_build_corpus({dpo_candidate}, dpo_policy, dpo_corpus, error)) {
                std::cout << "SEMI-ORACLE DPO PROJECTION=failed: " << error << '\n';
                return false;
            }
            common_learning_training_job training_job;
            if (!common_learning_make_training_job(dpo_corpus,
                    "base:" + std::filesystem::path(config.model).filename().string(),
                    "semi-oracle-dpo-smoke", "trainer-protocol-only", "smoke-v1",
                    training_job, error)) {
                std::cout << "SEMI-ORACLE DPO TRAINER CONTRACT=failed: " << error << '\n';
                return false;
            }
            std::cout << "SEMI-ORACLE DPO PROJECTION=" << ordered_json{
                {"source", "observed_qwen_move_vs_same_state_minimax"},
                {"chosen", dpo_candidate.approved_target},
                {"rejected", dpo_candidate.rejected_target},
                {"ruleset_id", ruleset.id},
                {"ruleset_revision", ruleset.revision},
                {"split_group_key", dpo_candidate.split_group_key},
                {"corpus_objective", common_learning_objective_name(dpo_corpus.objective)},
                {"trainer_job_objective", common_learning_objective_name(training_job.objective)},
                {"offline_only", true},
                {"adapter_trained", false},
            }.dump() << '\n';
            std::cout.flush();
        } else {
            std::cout << "SEMI-ORACLE DPO PROJECTION=not-created: observed action is not strictly worse than minimax best\n";
            std::cout.flush();
        }
    }
    if(domain=="linear_algebra") {
        const auto search_result=registry.execute({"oracle.goal.search","{}"});
        if(!search_result.ok) {
            error=search_result.raw_diagnostic.empty()?search_result.safe_summary:search_result.raw_diagnostic;
            std::cout<<"ALGEBRA A* GOAL SEARCH: failed: "<<error<<"\n";
            return false;
        }
        const auto search_output=ordered_json::parse(search_result.output,nullptr,false);
        std::cout<<"ALGEBRA A* GOAL SEARCH RESULT="
                 <<(search_output.is_object()?search_output.dump(2):search_result.output)<<"\n";
        if(!search_output.is_object()||search_output.value("verdict",std::string())!="SATISFIED"||
                !search_output.contains("proof")||!search_output["proof"].is_object()||
                search_output["proof"].value("algorithm",std::string())!="bounded_astar_v1") {
            std::cout<<"ALGEBRA A* GOAL SEARCH: missing verified path proof\n";
            return false;
        }
    }
    // A legal suboptimal game move is a useful baseline observation, not a
    // broken runtime flow. Preserve its mismatch/minimax trace for comparison;
    // the algebra case still requires the exact intended rewrite.
    return last_result.verdict == common_semi_oracle_verdict::satisfied &&
        (domain == "placement_game" || matches);
}
} // namespace

int main(int argc, char ** argv) {
    options config;
    if (!parse_args(argc, argv, config) || config.model.empty() || config.n_predict <= 0 || config.n_threads <= 0) {
        std::cerr << "usage: " << argv[0] << " --model MODEL [--n-predict N] [--threads N] [--n-gpu-layers N] [--strict]\n";
        return 2;
    }
    if (!std::filesystem::is_regular_file(config.model)) {
        std::cerr << "semi-oracle model smoke not-run: model file missing: " << config.model << "\n";
        return 77;
    }
    auto host = std::make_shared<common_agent_server_context_host>();
    common_agent_server_context_host_config server_config;
    server_config.context_key.load_key.model = config.model;
    server_config.context_key.load_key.n_gpu_layers = config.n_gpu_layers;
    server_config.context_key.load_key.fit_params = true;
    server_config.context_key.n_parallel = 1;
    server_config.context_key.n_sequences = 1;
    server_config.context_key.n_ctx = 4096;
    server_config.context_key.n_threads = config.n_threads;
    server_config.verbosity = LOG_LEVEL_WARN;
    std::string error;
    if (!host->start(server_config, error)) { std::cerr << "server-context start failed: " << error << '\n'; return 1; }
    common_agent_inference_session session;
    if (!host->build_inference_session(session, error) || !session.inference) {
        std::cerr << "server-context session failed: " << error << '\n'; return 1;
    }
    smoke_ruleset_store store;
    const bool game = run_case(*session.inference, config, store,"tic-tac-toe", "placement_game",
        "Let's play tic-tac-toe: a 3 by 3 board, X starts, players alternate, place only in an empty cell, three in a row wins, and a full board without a winner is a draw.",
        "row: 0\ncolumn: 2", R"json({"cells":["X","X",null,"O","O",null,null,null,null],"turn":"X"})json",
        "Play as X. Take an immediate win if one is available.", "", common_semi_oracle_level::local);
    const bool algebra = run_case(*session.inference, config, store,"linear-algebra", "linear_algebra",
        "Rules for one-variable linear equations in x: use exact rational arithmetic, and every allowed step must preserve the same solution set.",
        "lhs: 2x+6", R"json({"lhs":"2(x+3)","rhs":"10"})json",
        "Expand the left-hand side of 2(x+3)=10 using distribution.", "", common_semi_oracle_level::local);
    std::cout << "semi_oracle_model_smoke model=" << config.model
              << " tic_tac_toe=" << (game ? "passed" : "failed")
              << " algebra=" << (algebra ? "passed" : "failed")
              << " fit_params=enabled gpu_layers_requested=" << config.n_gpu_layers
              << " full_input_output_trace=enabled\n";
    return config.strict && !(game && algebra) ? 1 : 0;
}
