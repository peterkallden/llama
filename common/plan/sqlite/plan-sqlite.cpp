#include "plan-sqlite.h"
#include "plan/plan-workflow-verification.h"
#include "plan/plan-resource-provenance-json.h"
#include <stdexcept>

#include <nlohmann/json.hpp>

#include <map>

using json = nlohmann::ordered_json;

namespace {

const char * scope_name(common_plan_scope value) { return value == common_plan_scope::session ? "session" : value == common_plan_scope::project ? "project" : value == common_plan_scope::global ? "global" : "turn"; }
common_plan_scope parse_scope(const std::string & value) { return value == "session" ? common_plan_scope::session : value == "project" ? common_plan_scope::project : value == "global" ? common_plan_scope::global : common_plan_scope::turn; }
const char * kind_name(common_plan_kind value) { return value == common_plan_kind::blueprint ? "blueprint" : value == common_plan_kind::workflow ? "workflow" : "task"; }
common_plan_kind parse_kind(const std::string & value) { return value == "blueprint" ? common_plan_kind::blueprint : value == "workflow" ? common_plan_kind::workflow : common_plan_kind::task; }
const char * workflow_policy_name(common_plan_workflow_policy value) { return value == common_plan_workflow_policy::required ? "required" : "preferred"; }
common_plan_workflow_policy parse_workflow_policy(const std::string & value) { return value == "required" ? common_plan_workflow_policy::required : common_plan_workflow_policy::preferred; }
const char * status_name(common_plan_status value) { switch (value) { case common_plan_status::proposed: return "proposed"; case common_plan_status::active: return "active"; case common_plan_status::completed: return "completed"; case common_plan_status::blocked: return "blocked"; case common_plan_status::failed: return "failed"; case common_plan_status::cancelled: return "cancelled"; } return "proposed"; }
common_plan_status parse_status(const std::string & value) { if (value == "active") return common_plan_status::active; if (value == "completed") return common_plan_status::completed; if (value == "blocked") return common_plan_status::blocked; if (value == "failed") return common_plan_status::failed; if (value == "cancelled") return common_plan_status::cancelled; return common_plan_status::proposed; }
const char * step_status_name(common_plan_step_status value) { switch (value) { case common_plan_step_status::pending: return "pending"; case common_plan_step_status::active: return "active"; case common_plan_step_status::completed: return "completed"; case common_plan_step_status::blocked: return "blocked"; case common_plan_step_status::skipped: return "skipped"; case common_plan_step_status::failed: return "failed"; } return "pending"; }
common_plan_step_status parse_step_status(const std::string & value) { if (value == "active") return common_plan_step_status::active; if (value == "completed") return common_plan_step_status::completed; if (value == "blocked") return common_plan_step_status::blocked; if (value == "skipped") return common_plan_step_status::skipped; if (value == "failed") return common_plan_step_status::failed; return common_plan_step_status::pending; }
const char * step_mode_name(common_plan_step_mode value) { return value == common_plan_step_mode::tool ? "tool" : value == common_plan_step_mode::reasoning ? "reasoning" : value == common_plan_step_mode::artifact_compose ? "artifact_compose" : "final_response"; }
common_plan_step_mode parse_step_mode(const std::string & value) { return value == "tool" ? common_plan_step_mode::tool : value == "reasoning" ? common_plan_step_mode::reasoning : value == "artifact_compose" ? common_plan_step_mode::artifact_compose : common_plan_step_mode::final_response; }
common_plan_operation_kind parse_operation_kind(const std::string & value) {
    const common_plan_operation_kind values[] = {
        common_plan_operation_kind::create_plan, common_plan_operation_kind::revise_goal,
        common_plan_operation_kind::add_step, common_plan_operation_kind::revise_step,
        common_plan_operation_kind::replace_step, common_plan_operation_kind::remove_step,
        common_plan_operation_kind::activate_step, common_plan_operation_kind::reset_step,
        common_plan_operation_kind::complete_step, common_plan_operation_kind::block_step,
        common_plan_operation_kind::unblock_step, common_plan_operation_kind::fail_step,
        common_plan_operation_kind::skip_step, common_plan_operation_kind::add_dependency,
        common_plan_operation_kind::remove_dependency, common_plan_operation_kind::add_constraint,
        common_plan_operation_kind::add_assumption, common_plan_operation_kind::invalidate_assumption,
        common_plan_operation_kind::record_observation, common_plan_operation_kind::set_next_action,
        common_plan_operation_kind::request_replan, common_plan_operation_kind::complete_plan,
        common_plan_operation_kind::request_route_transition,
        common_plan_operation_kind::fail_plan,
    };
    for (const auto candidate : values) if (value == common_plan_operation_kind_name(candidate)) return candidate;
    return common_plan_operation_kind::add_step;
}

json serialize_plan_resource_ref(const common_runtime_resource_ref & resource) {
    json value={{"uri", resource.uri}, {"name", resource.name},
        {"description", resource.description}, {"mime_type", resource.mime_type},
        {"size_bytes", resource.size_bytes},
        {"scope", common_runtime_resource_scope_name(resource.scope)},
        {"metadata", {{"purpose", resource.metadata.purpose},
            {"content_summary", resource.metadata.content_summary},
            {"usage_hint", resource.metadata.usage_hint}, {"limitations", resource.metadata.limitations},
            {"keywords", resource.metadata.keywords}, {"entities", resource.metadata.entities}}}};
    common_plan_resource_provenance_to_json(resource,value);
    return value;
}

common_runtime_resource_ref parse_plan_resource_ref(const json & value) {
    common_runtime_resource_ref resource;
    resource.uri = value.value("uri", std::string{});
    resource.name = value.value("name", std::string{});
    resource.description = value.value("description", std::string{});
    resource.mime_type = value.value("mime_type", std::string{});
    resource.size_bytes = value.value("size_bytes", size_t(0));
    const auto scope = value.value("scope", std::string("turn"));
    resource.scope = scope == "session" ? common_runtime_resource_scope::session :
        scope == "project" ? common_runtime_resource_scope::project : common_runtime_resource_scope::turn;
    if (value.contains("metadata") && value["metadata"].is_object()) {
        const auto & metadata = value["metadata"];
        resource.metadata.purpose = metadata.value("purpose", std::string{});
        resource.metadata.content_summary = metadata.value("content_summary", std::string{});
        resource.metadata.usage_hint = metadata.value("usage_hint", std::string{});
        resource.metadata.limitations = metadata.value("limitations", std::string{});
        resource.metadata.keywords = metadata.value("keywords", std::vector<std::string>{});
        resource.metadata.entities = metadata.value("entities", std::vector<std::string>{});
    }
    common_plan_resource_provenance_from_json(value,resource);
    return resource;
}

json serialize_workflow_definition(const common_plan_workflow_definition & definition) {
    json value = {{"workflow_ref", definition.workflow_ref}, {"workflow_revision", definition.workflow_revision},
        {"family", definition.family}, {"graph_revision", definition.graph_revision},
        {"allowed_tools", definition.allowed_tools}, {"required_capabilities", definition.required_capabilities},
        {"optional_capabilities", definition.optional_capabilities}, {"required_context", definition.required_context},
        {"continuation_workflow_targets", definition.continuation_workflow_targets}};
    if (!definition.continuation_links.empty()) {
        value["continuation_links"] = json::array();
        for (const auto & link : definition.continuation_links) {
            json item = {{"workflow_ref", link.target_workflow_ref}, {"workflow_revision", link.target_workflow_revision}, {"bindings", json::array()}};
            if (!link.source_transition_id.empty()) item["from_transition"] = link.source_transition_id;
            if (!link.source_terminal_state.empty()) item["from_terminal_state"] = link.source_terminal_state;
            for (const auto & binding : link.argument_bindings) {
                json binding_json = {{"transition", binding.target_transition_id},
                    {"tool", binding.target_tool_name}, {"target", binding.target_json_pointer},
                    {"source", binding.source_json_pointer}, {"required", binding.required}};
                if (!binding.source_transition_id.empty()) binding_json["from_transition"] = binding.source_transition_id;
                item["bindings"].push_back(std::move(binding_json));
            }
            value["continuation_links"].push_back(std::move(item));
        }
    }
    common_plan_workflow_grounding_to_json(definition, value);
    if (definition.resource_read_policy == common_plan_workflow_resource_read_policy::chunked) {
        value["resource_read_policy"] = "chunked";
    }
    if (!definition.transitions.empty()) {
        value["start_state"] = definition.start_state;
        value["terminal_states"] = definition.terminal_states;
        value["transitions"] = json::array();
        for (const auto & transition : definition.transitions) {
            json item = {
                {"id", transition.id}, {"from", transition.from_state}, {"to", transition.to_state},
                {"kind", transition.kind == common_plan_workflow_transition_kind::model_choice ? "model_choice" : transition.kind == common_plan_workflow_transition_kind::artifact_compose ? "artifact_compose" : "tool"},
                {"tool", transition.tool_name}, {"arguments", transition.arguments_template_json},
                {"candidates", transition.candidate_json_pointer},
                {"candidate_id", transition.candidate_id_json_pointer}, {"cost", transition.cost},
                {"model_supplies_arguments", transition.model_supplies_arguments},
                {"model_instruction", transition.model_instruction},
                {"artifact_profile_ref", transition.artifact_profile_ref},
                {"artifact_profile_revision", transition.artifact_profile_revision},
                {"artifact_section_id", transition.artifact_section_id},
                {"artifact_profile_json", transition.artifact_profile_json},
                {"optional", transition.optional}, {"max_pages", transition.max_pages}};
            if (!transition.choices.empty()) {
                item["choices"] = json::array();
                for (const auto & choice : transition.choices) item["choices"].push_back({
                    {"id", choice.id}, {"description", choice.description}, {"to", choice.to_state},
                    {"requires", choice.required_observation_json_pointers}});
            }
            if (!transition.argument_bindings.empty()) {
                item["bindings"] = json::array();
                for (const auto & binding : transition.argument_bindings) item["bindings"].push_back({
                    {"target", binding.target_json_pointer}, {"from", binding.source_transition_id},
                    {"source", binding.source_json_pointer}, {"required", binding.required}});
            }
            value["transitions"].push_back(std::move(item));
        }
    }
    return value;
}

common_plan_workflow_definition parse_workflow_definition(const json & definition) {
    common_plan_workflow_definition value;
    std::string grounding_error;
    if (!common_plan_workflow_grounding_from_json(definition, value, grounding_error)) throw std::runtime_error(grounding_error);
    value.workflow_ref = definition.value("workflow_ref", std::string{});
    value.workflow_revision = definition.value("workflow_revision", std::string{});
    value.family = definition.value("family", std::string{});
    value.graph_revision = definition.value("graph_revision", std::string{});
    value.allowed_tools = definition.value("allowed_tools", std::vector<std::string>{});
    value.required_capabilities = definition.value("required_capabilities", std::vector<std::string>{});
    value.optional_capabilities = definition.value("optional_capabilities", std::vector<std::string>{});
    value.required_context = definition.value("required_context", std::vector<std::string>{});
    value.continuation_workflow_targets = definition.value("continuation_workflow_targets", std::vector<std::string>{});
    for (const auto & item : definition.value("continuation_links", json::array())) {
        common_plan_workflow_continuation_link link;
        link.target_workflow_ref = item.value("workflow_ref", std::string{});
        link.source_transition_id = item.value("from_transition", std::string{});
        link.source_terminal_state = item.value("from_terminal_state", std::string{});
        link.target_workflow_revision = item.value("workflow_revision", std::string{});
        for (const auto & binding : item.value("bindings", json::array())) link.argument_bindings.push_back({
            binding.value("transition", std::string{}),
            binding.value("tool", std::string{}), binding.value("target", std::string{}),
            binding.value("source", std::string{}), binding.value("required", true),
            binding.value("from_transition", std::string{})});
        value.continuation_links.push_back(std::move(link));
    }
    if (definition.value("resource_read_policy", std::string("single_range")) == "chunked") {
        value.resource_read_policy = common_plan_workflow_resource_read_policy::chunked;
    }
    value.start_state = definition.value("start_state", std::string{});
    value.terminal_states = definition.value("terminal_states", std::vector<std::string>{});
    for (const auto & item : definition.value("transitions", json::array())) {
        common_plan_workflow_transition transition;
        transition.id = item.value("id", std::string{});
        transition.from_state = item.value("from", std::string{});
        transition.to_state = item.value("to", std::string{});
        const auto transition_kind = item.value("kind", std::string("tool"));
        transition.kind = transition_kind == "model_choice" ? common_plan_workflow_transition_kind::model_choice :
            transition_kind == "artifact_compose" ? common_plan_workflow_transition_kind::artifact_compose : common_plan_workflow_transition_kind::tool;
        transition.tool_name = item.value("tool", std::string{});
        transition.arguments_template_json = item.value("arguments", std::string("{}"));
        transition.candidate_json_pointer = item.value("candidates", std::string{});
        transition.cost = item.value("cost", 1.0f);
        transition.candidate_id_json_pointer = item.value("candidate_id", std::string("/id"));
        transition.model_supplies_arguments = item.value("model_supplies_arguments", false);
        transition.model_instruction = item.value("model_instruction", std::string{});
        transition.artifact_profile_ref = item.value("artifact_profile_ref", std::string{});
        transition.artifact_profile_revision = item.value("artifact_profile_revision", std::string{});
        transition.artifact_section_id = item.value("artifact_section_id", std::string{});
        transition.artifact_profile_json = item.value("artifact_profile_json", std::string{});
        transition.optional = item.value("optional", false);
        transition.max_pages = item.value("max_pages", uint8_t{0});
        for (const auto & choice_json : item.value("choices", json::array())) {
            common_plan_workflow_transition::choice choice;
            choice.id = choice_json.value("id", std::string{});
            choice.description = choice_json.value("description", std::string{});
            choice.to_state = choice_json.value("to", std::string{});
            choice.required_observation_json_pointers = choice_json.value("requires", std::vector<std::string>{});
            transition.choices.push_back(std::move(choice));
        }
        for (const auto & binding_json : item.value("bindings", json::array())) transition.argument_bindings.push_back({
            binding_json.value("target", std::string{}), binding_json.value("from", std::string{}),
            binding_json.value("source", std::string{}), binding_json.value("required", true)});
        value.transitions.push_back(std::move(transition));
    }
    return value;
}

json serialize_plan(const common_plan_state & plan) {
    json steps = json::array();
    for (const auto & step : plan.steps) {
        json value = {
            {"id", step.id}, {"title", step.title}, {"objective", step.objective},
            {"intended_contribution", step.intended_contribution},
            {"mode", step_mode_name(common_plan_step_effective_mode(step))},
            {"status", step_status_name(step.status)}, {"depends_on", step.depends_on},
            {"blocked_by", step.blocked_by}, {"required_evidence", step.required_evidence},
            {"source_memory_ids", step.source_memory_ids}, {"optional", step.optional},
            {"generated_from_memory", step.generated_from_memory},
            {"created_at", step.created_at}, {"updated_at", step.updated_at}
        };
        if (step.selected_tool) value["selected_tool"] = *step.selected_tool;
        if (step.result_summary) value["result_summary"] = *step.result_summary;
        if (step.semantic_alias) value["semantic_alias"] = *step.semantic_alias;
        if(step.artifact_binding)value["artifact_binding"]={{"transition_id",step.artifact_binding->transition_id},{"profile_ref",step.artifact_binding->profile_ref},{"profile_revision",step.artifact_binding->profile_revision},{"section_id",step.artifact_binding->section_id},{"snapshot_id",step.artifact_binding->snapshot_id}};
        if (step.tool_call) value["tool_call"] = {{"name", step.tool_call->name}, {"arguments_json", step.tool_call->arguments_json}};
        steps.push_back(std::move(value));
    }
    json constraints = json::array();
    for (const auto & value : plan.constraints) constraints.push_back({{"id", value.id}, {"description", value.description}, {"hard", value.hard}});
    json assumptions = json::array();
    for (const auto & value : plan.assumptions) assumptions.push_back({{"id", value.id}, {"statement", value.statement}, {"confidence", value.confidence}, {"valid", value.valid}, {"evidence_ids", value.evidence_ids}});
    json observations = json::array();
    for (const auto & observation : plan.observations) {
        json resource_refs = json::array();
        for (const auto & resource : observation.resource_refs) resource_refs.push_back(serialize_plan_resource_ref(resource));
        json dataset_refs = json::array();
        for (const auto & dataset : observation.dataset_refs) dataset_refs.push_back({
            {"uri", dataset.uri}, {"name", dataset.name}, {"row_count", dataset.row_count},
            {"column_count", dataset.column_count}, {"source_resource_uri", dataset.source_resource_uri},
            {"source_representation", dataset.source_representation}, {"source_provider", dataset.source_provider},
            {"source_operation", dataset.source_operation}, {"source_request_json", dataset.source_request_json},
            {"retrieved_at", dataset.retrieved_at}, {"content_hash", dataset.content_hash}});
        observations.push_back({{"id", observation.id}, {"source", observation.source},
            {"summary", observation.summary}, {"confidence", observation.confidence},
            {"evidence_ids", observation.evidence_ids}, {"created_at", observation.created_at},
            {"resource_refs", resource_refs}, {"dataset_refs", dataset_refs}});
        if (observation.resource_read_provenance) {
            observations.back()["resource_read_provenance"] = {
                {"task_fingerprint", observation.resource_read_provenance->task_fingerprint},
                {"intent", observation.resource_read_provenance->intent},
                {"source_uri", observation.resource_read_provenance->source_uri},
                {"source_fingerprint", observation.resource_read_provenance->source_fingerprint},
                {"representation", observation.resource_read_provenance->representation},
                {"source_parent_uri", observation.resource_read_provenance->source_parent_uri}};
        }
    }
    json bindings = json::array();
    for (const auto & binding : plan.workflow_bindings) bindings.push_back({{"workflow_ref", binding.workflow_ref}, {"workflow_revision", binding.workflow_revision}});
    json continuation_history = json::array();
    for (const auto & item : plan.workflow_continuation_history) continuation_history.push_back({
        {"link_id", item.link_id}, {"source_fingerprint", item.source_fingerprint},
        {"source_workflow_ref", item.source_workflow_ref},
        {"source_workflow_revision", item.source_workflow_revision},
        {"source_terminal_state", item.source_terminal_state},
        {"target_workflow_ref", item.target_workflow_ref},
        {"target_workflow_revision", item.target_workflow_revision}});
    json procedures = plan.procedure_refs;
    json value = {{"id", plan.id}, {"namespace_id", plan.namespace_id}, {"session_id", plan.session_id}, {"project_id", plan.project_id}, {"turn_id", plan.turn_id}, {"source_revision", plan.source_revision}, {"selection_description", plan.selection_description}, {"kind", kind_name(plan.kind)}, {"derived_from_plan_id", plan.derived_from_plan_id}, {"scope", scope_name(plan.scope)}, {"status", status_name(plan.status)}, {"purpose", plan.purpose}, {"goal", plan.goal}, {"success_criteria", plan.success_criteria}, {"required_capabilities", plan.required_capabilities}, {"workflow_policy", workflow_policy_name(plan.workflow_policy)}, {"procedure_refs", procedures}, {"steps", steps}, {"constraints", constraints}, {"assumptions", assumptions}, {"observations", observations}, {"workflow_bindings", bindings}, {"workflow_continuation_history", continuation_history}, {"active_step_id", plan.active_step_id}, {"next_action", plan.next_action}, {"version", plan.version}, {"created_at", plan.created_at}, {"updated_at", plan.updated_at}};
    if (plan.selected_workflow) value["selected_workflow"] = {
        {"workflow_ref", plan.selected_workflow->workflow_ref},
        {"workflow_revision", plan.selected_workflow->workflow_revision},
    };
    if (plan.workflow_definition) value["workflow_definition"] = serialize_workflow_definition(*plan.workflow_definition);
    if (plan.route_binding) value["route_binding"] = {
        {"route_id", plan.route_binding->route_id},
        {"blueprint_ref", plan.route_binding->blueprint_ref},
        {"blueprint_revision", plan.route_binding->blueprint_revision},
        {"workflow_ref", plan.route_binding->workflow_ref},
        {"workflow_revision", plan.route_binding->workflow_revision},
        {"graph_revision", plan.route_binding->graph_revision},
        {"execution_envelope_fingerprint", plan.route_binding->execution_envelope_fingerprint},
        {"policy_revision", plan.route_binding->policy_revision},
        {"continuation_link_id", plan.route_binding->continuation_link_id},
        {"continuation_source_fingerprint", plan.route_binding->continuation_source_fingerprint},
        {"workflow_step_start_index", plan.route_binding->workflow_step_start_index},
    };
    return value;
}

bool deserialize_plan(const std::string & text, common_plan_state & plan) {
    const auto value = json::parse(text, nullptr, false);
    if (!value.is_object()) return false;
    plan = {};
    plan.id = value.value("id", std::string{});
    plan.namespace_id = value.value("namespace_id", std::string("local"));
    plan.session_id = value.value("session_id", std::string{});
    plan.project_id = value.value("project_id", std::string{});
    plan.turn_id = value.value("turn_id", std::string{});
    plan.source_revision = value.value("source_revision", std::string{});
    plan.selection_description = value.value("selection_description", std::string{});
    plan.kind = parse_kind(value.value("kind", std::string("task")));
    plan.scope = parse_scope(value.value("scope", std::string("turn")));
    plan.status = parse_status(value.value("status", std::string("proposed")));
    plan.purpose = value.value("purpose", std::string{});
    plan.goal = value.value("goal", std::string{});
    plan.success_criteria = value.value("success_criteria", std::string{});
    plan.required_capabilities = value.value("required_capabilities", std::vector<std::string>{});
    plan.workflow_policy = parse_workflow_policy(value.value("workflow_policy", std::string("preferred")));
    plan.procedure_refs = value.value("procedure_refs", std::vector<std::string>{});
    for (const auto & binding : value.value("workflow_bindings", json::array())) plan.workflow_bindings.push_back({binding.value("workflow_ref", std::string{}), binding.value("workflow_revision", std::string{})});
    for (const auto & item : value.value("workflow_continuation_history", json::array())) plan.workflow_continuation_history.push_back({
        item.value("link_id", std::string{}), item.value("source_fingerprint", std::string{}),
        item.value("source_workflow_ref", std::string{}), item.value("source_workflow_revision", std::string{}),
        item.value("source_terminal_state", std::string{}), item.value("target_workflow_ref", std::string{}),
        item.value("target_workflow_revision", std::string{})});
    if (value.contains("selected_workflow") && value["selected_workflow"].is_object()) plan.selected_workflow = common_plan_workflow_binding{value["selected_workflow"].value("workflow_ref", std::string{}), value["selected_workflow"].value("workflow_revision", std::string{})};
    if (value.contains("workflow_definition") && value["workflow_definition"].is_object()) plan.workflow_definition = parse_workflow_definition(value["workflow_definition"]);
    if (value.contains("route_binding") && value["route_binding"].is_object()) { const auto & binding = value["route_binding"]; plan.route_binding = common_plan_route_binding{binding.value("route_id", std::string{}), binding.value("blueprint_ref", std::string{}), binding.value("blueprint_revision", std::string{}), binding.value("workflow_ref", std::string{}), binding.value("workflow_revision", std::string{}), binding.value("graph_revision", std::string{}), binding.value("execution_envelope_fingerprint", std::string{}), binding.value("policy_revision", std::string{}), binding.value("continuation_link_id", std::string{}), binding.value("continuation_source_fingerprint", std::string{}), binding.value("workflow_step_start_index", size_t(0))}; }
    plan.version = value.value("version", uint64_t(0));
    plan.created_at = value.value("created_at", int64_t(0));
    plan.updated_at = value.value("updated_at", int64_t(0));
    if (value.contains("derived_from_plan_id") && value["derived_from_plan_id"].is_string()) plan.derived_from_plan_id = value["derived_from_plan_id"].get<std::string>();
    if (value.contains("active_step_id") && value["active_step_id"].is_string()) plan.active_step_id = value["active_step_id"].get<std::string>();
    if (value.contains("next_action") && value["next_action"].is_string()) plan.next_action = value["next_action"].get<std::string>();
    for (const auto & item : value.value("steps", json::array())) {
        common_plan_step step;
        step.id = item.value("id", std::string{}); step.title = item.value("title", std::string{}); step.objective = item.value("objective", std::string{}); step.intended_contribution = item.value("intended_contribution", std::string{});
        step.mode = parse_step_mode(item.value("mode", std::string("final_response"))); step.status = parse_step_status(item.value("status", std::string("pending"))); step.depends_on = item.value("depends_on", std::vector<std::string>{}); step.blocked_by = item.value("blocked_by", std::vector<std::string>{}); step.required_evidence = item.value("required_evidence", std::vector<std::string>{}); step.source_memory_ids = item.value("source_memory_ids", std::vector<std::string>{}); step.optional = item.value("optional", false); step.generated_from_memory = item.value("generated_from_memory", false); step.created_at = item.value("created_at", int64_t(0)); step.updated_at = item.value("updated_at", int64_t(0));
        if (item.contains("selected_tool") && item["selected_tool"].is_string()) step.selected_tool = item["selected_tool"].get<std::string>();
        if (item.contains("result_summary") && item["result_summary"].is_string()) step.result_summary = item["result_summary"].get<std::string>();
            if (item.contains("semantic_alias") && item["semantic_alias"].is_string()) step.semantic_alias = item["semantic_alias"].get<std::string>();
            if(item.contains("artifact_binding")&&item["artifact_binding"].is_object()){const auto & b=item["artifact_binding"];step.artifact_binding=common_plan_artifact_binding{b.value("transition_id",std::string{}),b.value("profile_ref",std::string{}),b.value("profile_revision",std::string{}),b.value("section_id",std::string{}),b.value("snapshot_id",std::string{})};}
            else step.artifact_binding=common_plan_artifact_binding_effective(step);
        if (item.contains("tool_call") && item["tool_call"].is_object()) step.tool_call = common_plan_tool_call{item["tool_call"].value("name", std::string{}), item["tool_call"].value("arguments_json", std::string("{}"))};
        plan.steps.push_back(std::move(step));
    }
    for (const auto & item : value.value("constraints", json::array())) plan.constraints.push_back({item.value("id", std::string{}), item.value("description", std::string{}), item.value("hard", true)});
    for (const auto & item : value.value("assumptions", json::array())) { common_plan_assumption assumption; assumption.id = item.value("id", std::string{}); assumption.statement = item.value("statement", std::string{}); assumption.confidence = item.value("confidence", 0.5f); assumption.valid = item.value("valid", true); assumption.evidence_ids = item.value("evidence_ids", std::vector<std::string>{}); plan.assumptions.push_back(std::move(assumption)); }
    for (const auto & item : value.value("observations", json::array())) {
        common_plan_observation observation;
        observation.id = item.value("id", std::string{}); observation.source = item.value("source", std::string{});
        observation.summary = item.value("summary", std::string{}); observation.confidence = item.value("confidence", 0.5f);
        observation.evidence_ids = item.value("evidence_ids", std::vector<std::string>{});
        observation.created_at = item.value("created_at", int64_t(0));
        if (item.contains("resource_read_provenance") && item["resource_read_provenance"].is_object()) {
            const auto & provenance = item["resource_read_provenance"];
            observation.resource_read_provenance = common_plan_resource_read_provenance{
                provenance.value("task_fingerprint", std::string{}),
                provenance.value("intent", std::string{}),
                provenance.value("source_uri", std::string{}),
                provenance.value("source_fingerprint", std::string{}),
                provenance.value("representation", std::string{}),
                provenance.value("source_parent_uri", std::string{})};
        }
        for (const auto & resource : item.value("resource_refs", json::array())) observation.resource_refs.push_back(parse_plan_resource_ref(resource));
        for (const auto & dataset : item.value("dataset_refs", json::array())) observation.dataset_refs.push_back({
            dataset.value("uri", std::string{}), dataset.value("name", std::string{}),
            dataset.value("row_count", size_t(0)), dataset.value("column_count", size_t(0)),
            dataset.value("source_resource_uri", std::string{}), dataset.value("source_representation", std::string{}),
            dataset.value("source_provider", std::string{}), dataset.value("source_operation", std::string{}),
            dataset.value("source_request_json", std::string{}), dataset.value("retrieved_at", int64_t(0)),
            dataset.value("content_hash", std::string{})});
        plan.observations.push_back(std::move(observation));
    }
    return !plan.id.empty();
}

json serialize_event(const common_plan_event & event, const common_plan_operation & operation) {
    json result = {{"sequence", event.sequence}, {"prior_version", event.prior_version}, {"new_version", event.new_version}, {"accepted", event.accepted}, {"reason_summary", event.reason_summary}, {"created_at", event.created_at}, {"operation", common_plan_operation_kind_name(operation.kind)}, {"plan_id", operation.plan_id}, {"expected_version", operation.expected_version}, {"step_id", operation.step_id}, {"target_id", operation.target_id}, {"value", operation.value}, {"evidence_ids", operation.evidence_ids}, {"host_authorized", operation.host_authorized}};
    if (operation.route_binding) result["route_binding"] = {{"route_id", operation.route_binding->route_id}, {"blueprint_ref", operation.route_binding->blueprint_ref}, {"blueprint_revision", operation.route_binding->blueprint_revision}, {"workflow_ref", operation.route_binding->workflow_ref}, {"workflow_revision", operation.route_binding->workflow_revision}, {"graph_revision", operation.route_binding->graph_revision}, {"execution_envelope_fingerprint", operation.route_binding->execution_envelope_fingerprint}, {"policy_revision", operation.route_binding->policy_revision}, {"continuation_link_id", operation.route_binding->continuation_link_id}, {"continuation_source_fingerprint", operation.route_binding->continuation_source_fingerprint}, {"workflow_step_start_index", operation.route_binding->workflow_step_start_index}};
    if (operation.workflow_definition) result["workflow_definition"] = serialize_workflow_definition(*operation.workflow_definition);
    if (!operation.route_required_capabilities.empty()) result["route_required_capabilities"] = operation.route_required_capabilities;
    if (!operation.route_procedure_refs.empty()) result["route_procedure_refs"] = operation.route_procedure_refs;
    return result;
}

}

common_plan_sqlite_store::~common_plan_sqlite_store() { close(); }

bool common_plan_sqlite_store::open(const std::string & path, std::string & error) {
    close();
    if (!database_.open(path, error) || !database_.execute("PRAGMA journal_mode = WAL;", error) || !ensure_schema(error) || !load_cache(error)) { close(); return false; }
    return true;
}

void common_plan_sqlite_store::close() { cache_.close(); database_.close(); }

bool common_plan_sqlite_store::ensure_schema(std::string & error) {
    return database_.execute("CREATE TABLE IF NOT EXISTS agent_plan (id TEXT PRIMARY KEY, state_json TEXT NOT NULL); CREATE TABLE IF NOT EXISTS agent_plan_event (plan_id TEXT NOT NULL, sequence INTEGER NOT NULL, event_json TEXT NOT NULL, PRIMARY KEY(plan_id, sequence));", error);
}

bool common_plan_sqlite_store::load_cache(std::string & error) {
    if (!cache_.open("", error)) return false;
    common_sqlite_statement statement;
    if (!database_.prepare("SELECT state_json FROM agent_plan ORDER BY id;", statement, error)) return false;
    bool row = false;
    while (statement.step(row, error) && row) { common_plan_state plan; if (!deserialize_plan(reinterpret_cast<const char *>(statement.column_text(0)), plan) || !cache_.create(plan, error)) return false; }
    if (!error.empty()) return false;

    std::map<std::string, std::vector<common_plan_event>> histories;
    if (!database_.prepare("SELECT plan_id,event_json FROM agent_plan_event ORDER BY plan_id,sequence;", statement, error)) return false;
    while (statement.step(row, error) && row) {
        const std::string plan_id = reinterpret_cast<const char *>(statement.column_text(0));
        const auto value = json::parse(reinterpret_cast<const char *>(statement.column_text(1)), nullptr, false);
        if (!value.is_object()) { error = "sqlite plan event contains invalid JSON"; return false; }
        common_plan_event event;
        event.sequence = value.value("sequence", uint64_t(0));
        event.prior_version = value.value("prior_version", uint64_t(0));
        event.new_version = value.value("new_version", uint64_t(0));
        event.accepted = value.value("accepted", false);
        event.reason_summary = value.value("reason_summary", std::string{});
        event.created_at = value.value("created_at", int64_t(0));
        event.operation.plan_id = value.value("plan_id", plan_id);
        event.operation.kind = parse_operation_kind(value.value("operation", std::string{}));
        event.operation.expected_version = value.value("expected_version", uint64_t(0));
        if (value.contains("step_id") && value["step_id"].is_string()) event.operation.step_id = value["step_id"].get<std::string>();
        if (value.contains("target_id") && value["target_id"].is_string()) event.operation.target_id = value["target_id"].get<std::string>();
        if (value.contains("value") && value["value"].is_string()) event.operation.value = value["value"].get<std::string>();
        event.operation.reason_summary = event.reason_summary;
        event.operation.evidence_ids = value.value("evidence_ids", std::vector<std::string>{});
        event.operation.host_authorized = value.value("host_authorized", false);
        if (value.contains("route_binding") && value["route_binding"].is_object()) { const auto & binding = value["route_binding"]; event.operation.route_binding = common_plan_route_binding{binding.value("route_id", std::string{}), binding.value("blueprint_ref", std::string{}), binding.value("blueprint_revision", std::string{}), binding.value("workflow_ref", std::string{}), binding.value("workflow_revision", std::string{}), binding.value("graph_revision", std::string{}), binding.value("execution_envelope_fingerprint", std::string{}), binding.value("policy_revision", std::string{}), binding.value("continuation_link_id", std::string{}), binding.value("continuation_source_fingerprint", std::string{}), binding.value("workflow_step_start_index", size_t(0))}; }
        if (value.contains("workflow_definition") && value["workflow_definition"].is_object()) event.operation.workflow_definition = parse_workflow_definition(value["workflow_definition"]);
        event.operation.route_required_capabilities = value.value("route_required_capabilities", std::vector<std::string>{});
        event.operation.route_procedure_refs = value.value("route_procedure_refs", std::vector<std::string>{});
        histories[plan_id].push_back(std::move(event));
    }
    if (!error.empty()) return false;
    for (auto & history : histories) if (!cache_.restore_history(history.first, std::move(history.second), error)) return false;
    return error.empty();
}

bool common_plan_sqlite_store::persist_plan(const common_plan_state & plan, std::string & error) {
    common_sqlite_statement statement;
    if (!database_.prepare("INSERT INTO agent_plan(id,state_json) VALUES(?,?) ON CONFLICT(id) DO UPDATE SET state_json=excluded.state_json;", statement, error) || !statement.bind_text(1, plan.id, error) || !statement.bind_text(2, serialize_plan(plan).dump(), error)) return false;
    bool row = false;
    return statement.step(row, error);
}

bool common_plan_sqlite_store::persist_event(const std::string & plan_id, const common_plan_event & event, const common_plan_operation & operation, std::string & error) {
    common_sqlite_statement statement;
    if (!database_.prepare("INSERT OR REPLACE INTO agent_plan_event(plan_id,sequence,event_json) VALUES(?,?,?);", statement, error) || !statement.bind_text(1, plan_id, error) || !statement.bind_int64(2, static_cast<int64_t>(event.sequence), error) || !statement.bind_text(3, serialize_event(event, operation).dump(), error)) return false;
    bool row = false;
    return statement.step(row, error);
}

bool common_plan_sqlite_store::create(const common_plan_state & plan, std::string & error) {
    if (!database_.is_open()) { error = "plan store is not open"; return false; }
    common_sqlite_transaction transaction(database_);
    if (!transaction.begin(error) || !persist_plan(plan, error) || !transaction.commit(error)) return false;
    return cache_.create(plan, error);
}

std::optional<common_plan_state> common_plan_sqlite_store::get(const std::string & plan_id, std::string & error) { return cache_.get(plan_id, error); }
std::vector<common_plan_state> common_plan_sqlite_store::list(std::string & error) { return cache_.list(error); }
std::vector<common_plan_event> common_plan_sqlite_store::history(const std::string & plan_id, std::string & error) { return cache_.history(plan_id, error); }

bool common_plan_sqlite_store::apply(const common_plan_operation & operation, common_plan_state & updated_plan, std::string & error) {
    if (!database_.is_open()) { error = "plan store is not open"; return false; }
    if (!cache_.apply(operation, updated_plan, error)) return false;
    const auto events = cache_.history(operation.plan_id, error);
    if (events.empty()) return false;
    common_sqlite_transaction transaction(database_);
    if (!transaction.begin(error) || !persist_plan(updated_plan, error) || !persist_event(operation.plan_id, events.back(), operation, error) || !transaction.commit(error)) return false;
    return true;
}

bool common_plan_sqlite_store::erase(const std::string & plan_id, std::string & error) {
    if (!database_.is_open()) { error = "plan store is not open"; return false; }
    common_sqlite_transaction transaction(database_);
    common_sqlite_statement statement;
    if (!transaction.begin(error) || !database_.prepare("DELETE FROM agent_plan_event WHERE plan_id = ?;", statement, error) || !statement.bind_text(1, plan_id, error)) return false;
    bool row = false;
    if (!statement.step(row, error) || !database_.prepare("DELETE FROM agent_plan WHERE id = ?;", statement, error) || !statement.bind_text(1, plan_id, error) || !statement.step(row, error) || !transaction.commit(error)) return false;
    return cache_.erase(plan_id, error);
}
