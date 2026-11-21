#include "agent/agent-runtime.h"
#include "agent/grounding/workflow-grounding.h"
#include "agent/tooling/catalog/tool-catalog.h"
#include "plan/cozo/plan-cozo.h"
#include "tools/agent/data/agent-data-store-cozo.h"
#include "tools/agent/resource/agent-resource-store.h"
#include "tools/agent/tooling/agent-tool-provider.h"
#include "tools/agent/tooling/agent-tool-runtime-adapter.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>

using json = nlohmann::ordered_json;
namespace {
void require(bool ok, const std::string & error) {
    if (!ok) { std::cerr << error << '\n'; std::exit(1); }
}
struct planner final : common_planner {
    std::string scenario;
    common_plan_proposal create_plan(const common_agent_request & request, std::string & error) override {
        common_plan_proposal proposal;
        auto & plan = proposal.plan;
        plan.id = "analysis-" + scenario; plan.namespace_id = request.namespace_id;
        plan.session_id = request.session_id; plan.turn_id = request.turn_id;
        plan.scope = request.plan_scope; plan.status = common_plan_status::active;
        plan.goal = request.prompt; plan.purpose = request.prompt;
        plan.workflow_definition = request.workflow_definition;
        common_plan_step measure;
        measure.id = "measure"; measure.title = "Describe amount"; measure.objective = request.prompt;
        measure.mode = common_plan_step_mode::tool; measure.status = common_plan_step_status::active;
        measure.selected_tool = "statistics.describe";
        json arguments = {{"dataset","dataset://smoke/orders"},{"columns",json::array({"amount"})}};
        if (scenario == "argument-repair" || scenario == "grouped") arguments["group_by"] = json::array({"customer_id"});
        if (scenario == "coverage-repair") arguments["max_scan_rows"] = 1;
        measure.tool_call = common_plan_tool_call{"statistics.describe",arguments.dump()};
        common_plan_step answer;
        answer.id = "answer"; answer.title = "Answer"; answer.objective = request.prompt;
        answer.mode = common_plan_step_mode::final_response; answer.status = common_plan_step_status::pending;
        answer.depends_on = {measure.id};
        plan.steps = {measure,answer};
        if (scenario == "tool-only") plan.steps.resize(1);
        plan.active_step_id = measure.id;
        error.clear(); return proposal;
    }
};
struct executor final : common_action_executor {
    bool omit_first_measure = false;
    size_t drafts = 0;
    std::string generate_draft(const common_agent_request & request, const common_plan_state & plan,
            const std::vector<std::string> &, std::string & error) override {
        const auto handles = common_agent_workflow_claim_handles(request, plan, &error);
        if (!error.empty()) return {};
        ++drafts;
        // Controlled fixture projection exercises the real host checks; it is
        // intentionally not evidence about a language model's understanding.
        std::string text = "answer: Executed amount statistics, as declared by the source-bound claims.";
        for (size_t index = 0; index < handles.size(); ++index) {
            if (omit_first_measure && drafts == 1 && index == 1) continue;
            text += "\nclaim: " + handles[index]["handle"].get<std::string>() + "\nvalue: " + handles[index]["value"].dump();
        }
        return text;
    }
};
struct reflector final : common_reflection_engine {
    common_reflection_result evaluate(const common_agent_request &, const common_plan_state & plan,
            const std::string &, std::string & error) override {
        error.clear(); common_reflection_result result;
        for (const auto & step : plan.steps) if (step.status == common_plan_step_status::failed && step.tool_call) {
            result.decision = common_reflection_decision::revise;
            result.next_action = common_agent_reflection_next_action::revise_plan;
            common_plan_operation repair;
            repair.kind = common_plan_operation_kind::replace_step; repair.plan_id = plan.id;
            repair.expected_version = plan.version; repair.step_id = step.id; repair.step = step;
            repair.step->status = common_plan_step_status::pending;
            const auto selected = json::parse(step.tool_call->result_requirement_json)["value"];
            repair.step->tool_call->arguments_json = json{{"dataset",selected["dataset"]},
                {"columns",selected["columns"]},{"group_by",selected["group_by"]},{"max_scan_rows",100000}}.dump();
            repair.reason_summary = "fixture argument repair from the retained task requirement";
            result.proposed_plan_operations = {repair}; return result;
        }
        result.decision = common_reflection_decision::accept;
        result.ready_to_answer = true; return result;
    }
};
} // namespace
int main() {
    const auto root = std::filesystem::temp_directory_path() / "llama-agent-analysis-result-smoke";
    std::filesystem::create_directories(root);
    const auto data_path = root / "data.cozo";
    std::filesystem::remove(data_path);
    common_agent_cozo_data_store data;
    std::string error;
    require(data.open(data_path.string(), error), error);
    agent_catalogued_resource_store resources(std::make_shared<agent_in_memory_blob_store>(),
        std::make_unique<agent_in_memory_resource_catalog>());
    agent_resource_runtime source_runtime;
    source_runtime.store = &resources; source_runtime.namespace_id = "analysis-smoke";
    source_runtime.session_id = "analysis-smoke"; source_runtime.turn_id = "fixture";
    agent_resource_put_request original;
    original.name = "orders.csv"; original.mime_type = "text/csv";
    original.text = "customer_id,amount\n10,12\n10,8\n11,20\n";
    original.scope = common_runtime_resource_scope::session;
    apply_agent_resource_runtime(source_runtime, original);
    agent_resource_descriptor source;
    require(resources.put_text(original, source, error), error);
    common_agent_dataset_descriptor descriptor;
    descriptor.ref = {"dataset://smoke/orders","orders",3,2,source.uri,"csv"};
    descriptor.columns = {{"customer_id",common_agent_dataset_column_type::integer,false},
        {"amount",common_agent_dataset_column_type::integer,false}};
    descriptor.import_processor_id = "smoke-csv";
    require(data.put_dataset_descriptor(descriptor, error), error);
    require(data.put_row(descriptor.ref.uri,"1",R"({"customer_id":10,"amount":12})",error), error);
    require(data.put_row(descriptor.ref.uri,"2",R"({"customer_id":10,"amount":8})",error), error);
    require(data.put_row(descriptor.ref.uri,"3",R"({"customer_id":11,"amount":20})",error), error);
    common_tool_catalog catalog; common_tool_bootstrap_result bootstrap;
    require(catalog.bootstrap("analysis",bootstrap,error),error);
    native_agent_tool_provider provider(catalog,[&](const agent_tool_context & context,
            common_native_tool_bindings & bindings, std::string &) {
        bindings.data_store = &data; bindings.resource_runtime = source_runtime;
        bindings.resource_runtime.turn_id = context.scope.turn_id;
        return true;
    });
    for (const auto * scenario : {"overall","grouped","tool-only","argument-repair","coverage-repair","answer-repair"}) {
        const auto path = root / (std::string(scenario) + ".cozo");
        std::filesystem::remove(path);
        common_plan_cozo_store plans; require(plans.open(path.string(),error),error);
        agent_tool_context context; context.profile_id = "analysis"; context.max_calls = 8;
        context.scope.namespace_id = "analysis-smoke"; context.scope.session_id = "analysis-smoke";
        context.scope.turn_id = scenario; context.request_id = scenario;
        auto view = provider.resolve_tools(context,error); require(bool(view),error);
        auto tools = make_provider_agent_tool_runtime(*view);
        planner planning; planning.scenario = scenario;
        executor drafting; drafting.omit_first_measure = std::string(scenario) == "answer-repair";
        reflector reflection;
        common_agent_runtime runtime(plans,planning,drafting,reflection,tools.get(),nullptr);
        common_agent_request request;
        request.prompt = "Report the executed amount count and mean.";
        request.namespace_id = "analysis-smoke"; request.session_id = "analysis-smoke";
        request.turn_id = scenario; request.available_datasets = {descriptor};
        request.enable_memory = false; request.enable_reflection = true;
        request.deliberation_policy.max_plan_revisions = 2;
        request.max_iterations = 4; request.max_reflection_rounds = 3; request.max_tool_batches = 3;
        request.workflow_definition = common_plan_workflow_definition{};
        request.workflow_definition->workflow_ref = "workflow://dataset/analysis";
        request.workflow_definition->workflow_revision = "v4";
        request.workflow_definition->allowed_tools = {"statistics.describe"};
        request.workflow_definition->declared_claim_tools = {"statistics.describe"};
        request.workflow_definition->result_contracts = {{"statistics.describe","statistics.describe.v1"}};
        request.result_requirement_selector = [scenario](const common_agent_request &, const common_plan_state & plan,
                const common_plan_step &, std::string & selected, common_agent_generated_text_result & generated, std::string &) {
            generated.status = common_agent_generation_status::completed;
            generated.stop_reason = common_agent_generation_stop_reason::eos;
            selected = json{{"schema","analysis-result.v1"},{"profile","statistics.describe.v1"},
                {"workflow",plan.workflow_definition->workflow_ref},{"workflow_revision",plan.workflow_definition->workflow_revision},
                {"value",{{"dataset","dataset://smoke/orders"},{"columns",json::array({"amount"})},
                    {"measures",json::array({"count","mean"})},{"coverage","complete"},
                    {"group_by",std::string(scenario) == "grouped" ? json::array({"customer_id"}) : json::array()}}}}.dump();
            return true;
        };
        const auto result = runtime.run(request);
        for (const auto & trace : result.trace) std::cout << "scenario=" << scenario << " step=" << trace.step_id
            << " operation=" << trace.tool_name << " detail=" << trace.detail << '\n';
        require(result.error.empty() && !result.response.empty(),std::string(scenario) + ": " + result.error);
        if (std::string(scenario) == "answer-repair") require(drafting.drafts == 2,"missing measure did not request response repair");
        const auto id = result.plan_id.value_or("");
        plans.close(); require(plans.open(path.string(),error),error);
        const auto saved = plans.get(id,error);
        require(saved && saved->steps[0].tool_call && !saved->steps[0].tool_call->result_requirement_json.empty(),"step requirement lost on restart");
        const auto handles = common_agent_workflow_claim_handles(request,*saved);
        require(handles.size() == (std::string(scenario) == "grouped" ? 4u : 2u),"required evidence missing after restart");
        std::cout << "scenario=" << scenario << " persisted_claims=" << handles.size() << " status=passed\n";
    }
    data.close(); std::filesystem::remove_all(root);
}
