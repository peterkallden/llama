#include "agent-dataset-workflow-adapter.h"

#include "agent-route-compiler.h"
#include "agent/adaptation/flydelta/oracles/blueprint-workflow.h"
#include "agent/adaptation/flydelta/oracles/workflow-proposal.h"

#include <algorithm>
#include <cctype>
#include <nlohmann/json.hpp>

namespace {

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

bool is_dataset_operation(const std::string & name) {
    return name == "data.query" || name == "data.filter" ||
        name == "data.aggregate" || name == "data.join" ||
        name == "data.transform" || name == "statistics.describe" ||
        name == "statistics.outliers" || name == "statistics.value_counts";
}

const common_agent_dataset_descriptor * find_dataset(
        const std::vector<common_agent_dataset_descriptor> & datasets,
        const std::string & reference) {
    const auto found = std::find_if(datasets.begin(), datasets.end(), [&](const auto & dataset) {
        return dataset.ref.uri == reference || dataset.ref.name == reference;
    });
    return found == datasets.end() ? nullptr : &*found;
}

} // namespace

common_agent_workflow_continuation_provider
make_agent_resource_document_workflow_continuation(
        common_agent_route_candidate route,
        std::vector<common_agent_tool_argument_binding> bindings) {
    return [route = std::move(route), bindings = std::move(bindings)](
            const common_plan_state & plan,
            const std::string & completed_step_id,
            const common_plan_observation & observation,
            std::vector<common_plan_step> & continuation_steps,
            std::string & error) {
        continuation_steps.clear();
        error.clear();
        if (route.kind != common_agent_route_kind::blueprint_workflow ||
                !route.workflow || route.workflow->workflow_ref !=
                    "workflow://resource/document-analysis" ||
                !plan.route_binding ||
                plan.route_binding->route_id != route.id ||
                !plan.workflow_definition ||
                plan.workflow_definition->workflow_ref != route.workflow->workflow_ref ||
                plan.workflow_definition->workflow_revision != route.workflow->workflow_revision) {
            return true;
        }

        const auto table_step = std::find_if(plan.steps.begin(), plan.steps.end(),
            [](const auto & step) {
                return step.status == common_plan_step_status::completed && step.tool_call &&
                    step.tool_call->name == "document.table";
            });
        if (table_step == plan.steps.end() || observation.id.empty()) return true;

        const auto binding = std::find_if(bindings.begin(), bindings.end(), [&](const auto & value) {
            return is_dataset_operation(value.tool_name) && value.tool_name != "data.join" &&
                std::find(route.resolved_tools.begin(), route.resolved_tools.end(), value.tool_name) !=
                    route.resolved_tools.end();
        });
        if (binding == bindings.end() || binding->tool_name == observation.source) return true;
        const bool already_planned = std::any_of(plan.steps.begin(), plan.steps.end(), [&](const auto & step) {
            return step.tool_call && step.tool_call->name == binding->tool_name &&
                step.status != common_plan_step_status::failed &&
                step.status != common_plan_step_status::skipped;
        });
        if (already_planned) return true;

        auto arguments = nlohmann::ordered_json::parse(binding->arguments_json, nullptr, false);
        if (!arguments.is_object()) {
            error = "host-bound document analysis arguments are invalid JSON";
            return false;
        }
        arguments["dataset"] = nlohmann::ordered_json{
            {"$from_step", table_step->id}, {"$json_pointer", "/dataset"}};
        common_flydelta_workflow_proposal_request proposal_request;
        proposal_request.proposal_id = "proposal://document-continuation/" +
            plan.id + "/" + std::to_string(plan.version);
        proposal_request.blueprint_ref = "blueprint://" + route.blueprint_logical_id;
        proposal_request.blueprint_revision = route.blueprint_revision;
        proposal_request.workflow_ref = route.workflow->workflow_ref;
        proposal_request.workflow_revision = route.workflow->workflow_revision;
        proposal_request.graph_revision = route.graph_revision;
        proposal_request.start_state = "document-table-ready";
        proposal_request.goal_state = "requested-operation-ready";
        proposal_request.max_expansions = 4;
        proposal_request.max_path_length = 2;
        proposal_request.is_goal = [](const std::string & state) {
            return state == "requested-operation-ready";
        };
        proposal_request.expand = [](const std::string & state,
                std::vector<common_flydelta_astar_successor> & successors,
                std::string & expand_error) {
            successors.clear();
            expand_error.clear();
            if (state == "document-table-ready") {
                successors.push_back({"requested-operation-ready", "run-bound-operation", 1.0f, 0.0f});
            }
            return true;
        };
        const std::string operation_tool_name = binding->tool_name;
        proposal_request.materialize = [operation_tool_name, arguments](
                const std::vector<std::string> &,
                const std::vector<std::string> & actions,
                std::vector<common_tool_workflow_step_view> & steps,
                std::string & materialize_error) {
            steps.clear();
            materialize_error.clear();
            if (actions.size() != 1 || actions.front() != "run-bound-operation") {
                materialize_error = "document continuation path is outside its bounded contract";
                return false;
            }
            steps.push_back({operation_tool_name, arguments.dump()});
            return true;
        };
        common_flydelta_workflow_proposal proposal;
        if (!common_flydelta_propose_workflow_path(proposal_request, proposal, error) ||
                !proposal.proposed || proposal.canonical_steps.size() != 1) {
            if (error.empty()) error = "A* could not resolve the requested document continuation";
            return false;
        }
        const auto & proposed = proposal.canonical_steps.front();
        if (std::find(route.resolved_tools.begin(), route.resolved_tools.end(), proposed.tool_name) ==
                route.resolved_tools.end()) {
            error = "A* document continuation proposed a tool outside the selected route";
            return false;
        }
        common_plan_step step;
        step.id = plan.id + ":astar-cont:" + std::to_string(plan.version);
        step.title = proposed.tool_name;
        step.objective = "Continue document analysis with the host-bound operation.";
        step.intended_contribution = step.objective;
        step.status = common_plan_step_status::pending;
        step.depends_on = {completed_step_id};
        step.mode = common_plan_step_mode::tool;
        step.selected_tool = proposed.tool_name;
        step.tool_call = common_plan_tool_call{proposed.tool_name, proposed.arguments_json};
        step.semantic_alias = "astar:" + proposal.path_fingerprint;
        continuation_steps.push_back(std::move(step));
        return true;
    };
}

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

            const common_agent_dataset_descriptor * dataset = &tooling->available_datasets.front();
            const auto prompt = lowercase(request.prompt);
            for (const auto & candidate : tooling->available_datasets) {
                const auto name = lowercase(candidate.ref.name);
                if (!name.empty() && prompt.find(name) != std::string::npos) {
                    dataset = &candidate;
                    break;
                }
            }
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
