#include "agent-dataset-workflow-adapter.h"

#include "agent-route-compiler.h"
#include "agent/adaptation/flydelta/oracles/blueprint-workflow.h"
#include "agent/adaptation/flydelta/oracles/workflow-proposal.h"

#include <algorithm>
#include <nlohmann/json.hpp>

namespace {

bool is_dataset_operation(const std::string & name) {
    return name == "data.query" || name == "data.filter" ||
        name == "data.aggregate" || name == "data.join" ||
        name == "data.transform" || name == "statistics.describe" ||
        name == "statistics.outliers" || name == "statistics.value_counts";
}

const common_agent_dataset_descriptor * find_dataset(
        const std::vector<common_agent_dataset_descriptor> & datasets,
        const std::string & reference) {
    const common_agent_dataset_descriptor * named = nullptr;
    for (const auto & dataset : datasets) {
        if (dataset.ref.uri == reference) return &dataset;
        if (dataset.ref.name != reference) continue;
        if (named != nullptr) return nullptr;
        named = &dataset;
    }
    return named;
}

} // namespace

common_blueprint_instance_materializer make_agent_dataset_workflow_materializer(
        common_plan_store & plan_store,
        const common_agent_runtime_tooling * tooling) {
    return common_flydelta_make_dataset_blueprint_plan_materializer(
        [&plan_store, tooling](const common_agent_request & request,
                const common_plan_state & instance,
                common_flydelta_dataset_blueprint_request & out,
                std::string & provider_error) {
            out = {};
            provider_error.clear();
            if (tooling == nullptr || tooling->available_datasets.empty() ||
                    !instance.selected_workflow) return false;

            std::string store_error;
            const auto plans = plan_store.list(store_error);
            if (!store_error.empty()) {
                provider_error = store_error;
                return false;
            }
            const common_plan_workflow_definition * selected = nullptr;
            std::vector<std::string> allowed_tools;
            for (const auto & plan : plans) {
                if (plan.kind != common_plan_kind::workflow || !plan.workflow_definition) continue;
                const auto & definition = *plan.workflow_definition;
                if (definition.family != "dataset" ||
                        definition.workflow_ref != instance.selected_workflow->workflow_ref ||
                        definition.workflow_revision != instance.selected_workflow->workflow_revision) {
                    continue;
                }
                std::string reason;
                if (!common_agent_resolve_workflow_tools(
                        definition, *tooling, allowed_tools, reason)) {
                    provider_error = "selected dataset workflow is no longer resolvable: " + reason;
                    return false;
                }
                selected = &definition;
                break;
            }
            if (selected == nullptr) return false;

            std::string terminal;
            std::string terminal_arguments = "{}";
            for (const auto & binding : request.tool_argument_bindings) {
                if (is_dataset_operation(binding.tool_name) &&
                        std::find(allowed_tools.begin(), allowed_tools.end(), binding.tool_name) !=
                            allowed_tools.end()) {
                    terminal = binding.tool_name;
                    terminal_arguments = binding.arguments_json;
                    break;
                }
            }
            // Operation parameters are semantic inputs, not safe defaults.
            // Without an explicit host binding, leave operation selection to
            // the ordinary planner, which sees every tool in the route envelope.
            if (terminal.empty()) {
                return false;
            }
            auto arguments = nlohmann::ordered_json::parse(terminal_arguments, nullptr, false);
            if (!arguments.is_object()) {
                provider_error = "host operation binding has invalid JSON arguments";
                return false;
            }

            const common_agent_dataset_descriptor * dataset = nullptr;
            if (terminal == "data.join") {
                if (!arguments.contains("left") || !arguments["left"].is_string() ||
                        !arguments.contains("right") || !arguments["right"].is_string() ||
                        !arguments.contains("on") || !arguments["on"].is_array() ||
                        arguments["on"].empty()) {
                    provider_error = "data.join requires host-bound left, right and on arguments";
                    return false;
                }
                const auto * left = find_dataset(tooling->available_datasets,
                    arguments["left"].get<std::string>());
                const auto * right = find_dataset(tooling->available_datasets,
                    arguments["right"].get<std::string>());
                if (left == nullptr || right == nullptr || left->ref.uri == right->ref.uri) {
                    provider_error = "data.join inputs must resolve to two distinct in-scope datasets";
                    return false;
                }
                dataset = left;
                out.second_dataset_ref = right->ref.uri;
                out.second_dataset_name = right->ref.name;
                out.second_schema_known = !right->columns.empty();
                arguments["left"] = left->ref.uri;
                arguments["right"] = right->ref.uri;
            } else if (arguments.contains("dataset")) {
                if (!arguments["dataset"].is_string()) {
                    provider_error = "host-bound dataset argument must be a string";
                    return false;
                }
                const auto * bound_dataset = find_dataset(tooling->available_datasets,
                    arguments["dataset"].get<std::string>());
                if (bound_dataset == nullptr) {
                    provider_error = "host-bound dataset is not in the current scoped inventory";
                    return false;
                }
                dataset = bound_dataset;
            } else {
                // A scoped inventory is not a source selection. Leave an
                // unbound operation to normal model planning, never the first
                // catalogue item or a substring in the user's request.
                return false;
            }
            if (dataset->ref.uri.empty() || dataset->ref.name.empty()) {
                provider_error = "selected dataset is missing a host-resolved identity";
                return false;
            }

            out.schema_version = 1;
            out.proposal_id = "proposal://dataset-blueprint";
            std::string blueprint_name = instance.id.substr(instance.id.rfind(':') + 1);
            if (instance.derived_from_plan_id) {
                const auto marker = instance.derived_from_plan_id->rfind("blueprint:");
                if (marker != std::string::npos) {
                    blueprint_name = instance.derived_from_plan_id->substr(
                        marker + std::string("blueprint:").size());
                }
            }
            out.blueprint_ref = "blueprint://" + blueprint_name;
            out.blueprint_revision = instance.source_revision;
            out.graph_revision = selected->graph_revision;
            out.workflow_ref = selected->workflow_ref;
            out.workflow_revision = selected->workflow_revision;
            out.max_path_length = terminal == "data.join" ? 8 : 4;
            out.dataset_ref = dataset->ref.uri;
            out.dataset_name = dataset->ref.name;
            out.schema_known = !dataset->columns.empty();
            out.terminal_tool = terminal;
            if (terminal != "data.join") {
                arguments["dataset"] = dataset->ref.uri;
            }
            out.terminal_arguments_json = arguments.dump();
            return true;
        },
        [&plan_store, tooling](const common_flydelta_workflow_proposal & proposal,
                std::string & verifier_error) {
            verifier_error.clear();
            std::string store_error;
            const auto plans = plan_store.list(store_error);
            if (!store_error.empty()) {
                verifier_error = store_error;
                return false;
            }
            for (const auto & plan : plans) {
                if (plan.kind != common_plan_kind::workflow || !plan.workflow_definition) continue;
                const auto & definition = *plan.workflow_definition;
                if (definition.workflow_ref != proposal.workflow_ref ||
                        definition.workflow_revision != proposal.workflow_revision) continue;
                std::vector<std::string> allowed_tools;
                std::string resolution_error;
                if (tooling == nullptr || !common_agent_resolve_workflow_tools(
                        definition, *tooling, allowed_tools, resolution_error)) {
                    verifier_error = resolution_error.empty()
                        ? "workflow tools are unavailable in the current host view"
                        : resolution_error;
                    return false;
                }
                for (const auto & reference : proposal.binding_refs) {
                    if (find_dataset(tooling->available_datasets, reference) == nullptr) {
                        verifier_error = "workflow proposal references a dataset outside the current scoped inventory";
                        return false;
                    }
                }
                for (const auto & step : proposal.canonical_steps) {
                    if (std::find(allowed_tools.begin(), allowed_tools.end(), step.tool_name) ==
                            allowed_tools.end()) {
                        verifier_error = "workflow proposal contains a tool outside the persisted workflow contract";
                        return false;
                    }
                }
                return true;
            }
            verifier_error = "workflow proposal does not match a persisted workflow definition";
            return false;
        });
}
