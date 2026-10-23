#include "agent/agent-package-json.h"
#include "plan/plan-workflow-verification.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <functional>
#include <map>
#include <set>

using json = nlohmann::ordered_json;

namespace {
bool strings(const json & value, std::vector<std::string> & out) {
    if (!value.is_array()) return false;
    out.clear();
    for (const auto & item : value) { if (!item.is_string()) return false; out.push_back(item.get<std::string>()); }
    return true;
}

bool valid_id(const std::string & id) {
    if (id.empty() || id.size() > 128 || id.find("bootstrap:") != std::string::npos) return false;
    const auto alnum = [](char c) { return std::isalnum(static_cast<unsigned char>(c)) != 0; };
    if (!alnum(id.front())) return false;
    return std::all_of(id.begin() + 1, id.end(), [&](char c) { return alnum(c) || c == '.' || c == '_' || c == '-'; });
}

bool constraints(const json & value, std::vector<common_plan_constraint> & out) {
    if (!value.is_array()) return false;
    out.clear();
    for (const auto & item : value) {
        if (!item.is_object() || !item.contains("id") || !item["id"].is_string() || !item.contains("description") || !item["description"].is_string()) return false;
        out.push_back({item["id"].get<std::string>(), item["description"].get<std::string>(), item.value("hard", true)});
    }
    return true;
}

bool assumptions(const json & value, std::vector<common_plan_assumption> & out) {
    if (!value.is_array()) return false;
    out.clear();
    for (const auto & item : value) {
        if (!item.is_object() || !item.contains("id") || !item["id"].is_string() || !item.contains("statement") || !item["statement"].is_string()) return false;
        common_plan_assumption assumption;
        assumption.id = item["id"].get<std::string>();
        assumption.statement = item["statement"].get<std::string>();
        assumption.confidence = item.value("confidence", 0.5f);
        assumption.valid = true;
        out.push_back(std::move(assumption));
    }
    return true;
}

bool workflow_policy(const json & value, common_plan_workflow_policy & out) {
    if (!value.is_string()) return false;
    const auto text = value.get<std::string>();
    if (text == "preferred") {
        out = common_plan_workflow_policy::preferred;
        return true;
    }
    if (text == "required") {
        out = common_plan_workflow_policy::required;
        return true;
    }
    return false;
}

const char * workflow_policy_name(common_plan_workflow_policy policy) {
    return policy == common_plan_workflow_policy::required ? "required" : "preferred";
}

const char * workflow_transition_kind_name(common_plan_workflow_transition_kind kind) {
    return kind == common_plan_workflow_transition_kind::model_choice ? "model_choice" :
        kind == common_plan_workflow_transition_kind::artifact_compose ? "artifact_compose" : "tool";
}

bool workflow_transition_kind(const json & value, common_plan_workflow_transition_kind & out) {
    if (!value.is_string()) return false;
    if (value.get<std::string>() == "tool") { out = common_plan_workflow_transition_kind::tool; return true; }
    if (value.get<std::string>() == "model_choice") { out = common_plan_workflow_transition_kind::model_choice; return true; }
    if (value.get<std::string>() == "artifact_compose") { out = common_plan_workflow_transition_kind::artifact_compose; return true; }
    return false;
}

bool workflow_graph(const json & item, common_plan_workflow_definition & definition, std::string & error) {
    if (!common_plan_workflow_grounding_from_json(item, definition, error)) return false;
    const auto read_policy = item.value("resource_read_policy", std::string("single_range"));
    if (read_policy == "chunked") {
        definition.resource_read_policy = common_plan_workflow_resource_read_policy::chunked;
    } else if (read_policy != "single_range") {
        error = "workflow resource_read_policy must be single_range or chunked";
        return false;
    }
    if (!item.contains("transitions")) return true;
    if (!item["transitions"].is_array() || !item.contains("start_state") || !item["start_state"].is_string() ||
            !item.contains("terminal_states") || !strings(item["terminal_states"], definition.terminal_states)) {
        error = "workflow graph requires start_state, terminal_states and transitions";
        return false;
    }
    definition.start_state = item["start_state"].get<std::string>();
    for (const auto & source : item["transitions"]) {
        if (!source.is_object()) { error = "workflow transition must be an object"; return false; }
        common_plan_workflow_transition transition;
        transition.id = source.value("id", std::string{});
        transition.from_state = source.value("from", std::string{});
        transition.to_state = source.value("to", std::string{});
        transition.tool_name = source.value("tool", std::string{});
        transition.arguments_template_json = source.value("arguments", std::string("{}"));
        transition.candidate_json_pointer = source.value("candidates", std::string{});
        transition.candidate_id_json_pointer = source.value("candidate_id", std::string("/id"));
        transition.model_supplies_arguments = source.value("model_supplies_arguments", false);
        transition.model_instruction = source.value("model_instruction", std::string{});
        transition.artifact_profile_ref = source.value("artifact_profile_ref", std::string{});
        transition.artifact_profile_revision = source.value("artifact_profile_revision", std::string{});
        transition.artifact_section_id = source.value("artifact_section_id", std::string{});
        transition.cost = source.value("cost", 1.0f);
        transition.optional = source.value("optional", false);
        if (source.contains("max_pages")) {
            if (!source["max_pages"].is_number_integer()) {
                error = "workflow transition max_pages must be an integer";
                return false;
            }
            if (source["max_pages"] < 2 || source["max_pages"] > 5) {
                error = "workflow transition max_pages must be between 2 and 5";
                return false;
            }
            transition.max_pages = source["max_pages"].get<uint8_t>();
        }
        if (source.contains("choices")) {
            if (!source["choices"].is_array()) { error = "workflow transition choices must be an array"; return false; }
            for (const auto & choice_source : source["choices"]) {
                if (!choice_source.is_object()) { error = "workflow choice must be an object"; return false; }
                common_plan_workflow_transition::choice choice;
                choice.id = choice_source.value("id", std::string{});
                choice.description = choice_source.value("description", std::string{});
                choice.to_state = choice_source.value("to", std::string{});
                if (choice_source.contains("requires") &&
                        !strings(choice_source["requires"], choice.required_observation_json_pointers)) {
                    error = "workflow choice requires must be JSON pointer strings"; return false;
                }
                transition.choices.push_back(std::move(choice));
            }
        }
        if (source.contains("bindings")) {
            if (!source["bindings"].is_array()) { error = "workflow transition bindings must be an array"; return false; }
            for (const auto & binding_source : source["bindings"]) {
                if (!binding_source.is_object()) { error = "workflow argument binding must be an object"; return false; }
                transition.argument_bindings.push_back({
                    binding_source.value("target", std::string{}),
                    binding_source.value("from", std::string("$previous")),
                    binding_source.value("source", std::string{}),
                    binding_source.value("required", true)});
            }
        }
        if (source.contains("kind") && !workflow_transition_kind(source["kind"], transition.kind)) {
            error = "workflow transition kind must be tool, model_choice or artifact_compose";
            return false;
        }
        definition.transitions.push_back(std::move(transition));
    }
    return true;
}

bool validate(const common_agent_bootstrap_package & package, std::string & error) {
    if (package.name.empty() || package.name.size() > 128 || package.version.empty() || package.version.size() > 64) { error = "package name and version are required"; return false; }
    std::set<std::string> procedures, blueprints, workflows, artifact_profiles;
    for (const auto & profile : package.artifact_profiles) {
        if (!valid_id(profile.id) || profile.revision.empty() || profile.revision.size() > 128 ||
                profile.title.empty() || profile.title.size() > 256 || profile.sections.empty() ||
                profile.sections.size() > 32 || !artifact_profiles.insert(profile.id + "@" + profile.revision).second) {
            error = "invalid or duplicate artifact profile"; return false;
        }
        std::set<std::string> sections;
        for (const auto & section : profile.sections) if (!valid_id(section.id) || section.title.empty() ||
                section.title.size() > 256 || section.purpose.size() > 2048 || !sections.insert(section.id).second) {
            error = "artifact profile has invalid or duplicate section"; return false;
        }
    }
    for (const auto & procedure : package.procedures) {
        if (!valid_id(procedure.id) || procedure.content.empty() || procedure.content.size() > 8192 || procedure.summary.size() > 1024 || procedure.importance < 0 || procedure.importance > 1 || procedure.confidence < 0 || procedure.confidence > 1 || !procedures.insert(procedure.id).second) { error = "invalid or duplicate procedure definition"; return false; }
    }
    for (const auto & blueprint : package.blueprints) {
        if (!valid_id(blueprint.id) || blueprint.source_revision.size() > 128 || blueprint.goal.empty() || blueprint.goal.size() > 4096 || blueprint.success_criteria.empty() || blueprint.success_criteria.size() > 4096 || blueprint.steps.empty() || blueprint.steps.size() > 64 || !blueprints.insert(blueprint.id).second) { error = "invalid or duplicate blueprint definition"; return false; }
        std::set<std::string> steps;
        for (const auto & step : blueprint.steps) {
            if (!valid_id(step.id) || step.title.empty() || step.objective.empty() || step.tool_call || step.selected_tool || !steps.insert(step.id).second) { error = "invalid blueprint step or tool binding"; return false; }
        }
        for (const auto & step : blueprint.steps) for (const auto & dependency : step.depends_on) if (!steps.count(dependency) || dependency == step.id) { error = "blueprint dependency references an unknown step"; return false; }
        std::set<std::string> constraints, assumptions;
        for (const auto & procedure_ref : blueprint.procedure_refs) if (procedure_ref.empty() || procedure_ref.size() > 256) { error = "invalid blueprint procedure reference"; return false; }
        for (const auto & capability : blueprint.required_capabilities) if (capability.empty() || capability.size() > 128) { error = "invalid blueprint required capability"; return false; }
        for (const auto & constraint : blueprint.constraints) if (!valid_id(constraint.id) || constraint.description.empty() || constraint.description.size() > 4096 || !constraints.insert(constraint.id).second) { error = "invalid or duplicate blueprint constraint"; return false; }
        for (const auto & assumption : blueprint.assumptions) if (!valid_id(assumption.id) || assumption.statement.empty() || assumption.statement.size() > 4096 || assumption.confidence < 0 || assumption.confidence > 1 || !assumptions.insert(assumption.id).second) { error = "invalid or duplicate blueprint assumption"; return false; }
        std::set<std::string> bindings;
        for (const auto & binding : blueprint.workflow_bindings) if (binding.workflow_ref.empty() || binding.workflow_ref.size() > 256 || binding.workflow_revision.empty() || binding.workflow_revision.size() > 128 || !bindings.insert(binding.workflow_ref + "@" + binding.workflow_revision).second) { error = "invalid or duplicate blueprint workflow binding"; return false; }
        // Every dependency points to an earlier-free DAG; bounded DFS detects cycles.
        std::set<std::string> visiting, visited;
        std::function<bool(const std::string &)> visit = [&](const std::string & id) {
            if (visiting.count(id)) return false;
            if (visited.count(id)) return true;
            visiting.insert(id);
            const auto it = std::find_if(blueprint.steps.begin(), blueprint.steps.end(), [&](const auto & step) { return step.id == id; });
            for (const auto & dep : it->depends_on) if (!visit(dep)) return false;
            visiting.erase(id); visited.insert(id); return true;
        };
        for (const auto & step : blueprint.steps) if (!visit(step.id)) { error = "blueprint dependencies contain a cycle"; return false; }
    }
    for (const auto & workflow : package.workflows) {
        const auto & definition = workflow.definition;
        if (!valid_id(workflow.id) || workflow.source_revision.size() > 128 ||
                definition.workflow_ref.empty() || definition.workflow_ref.size() > 256 ||
                definition.workflow_revision.empty() || definition.workflow_revision.size() > 128 ||
                definition.family.empty() || definition.family.size() > 64 ||
                definition.graph_revision.empty() || definition.graph_revision.size() > 128 ||
                (definition.allowed_tools.empty() && definition.required_capabilities.empty() &&
                 definition.optional_capabilities.empty()) ||
                !workflows.insert(workflow.id).second) {
            error = "invalid or duplicate workflow definition"; return false;
        }
        for (const auto & tool : definition.allowed_tools) if (tool.empty() || tool.size() > 128) { error = "invalid workflow allowed tool"; return false; }
        std::set<std::string> required_capabilities;
        for (const auto & capability : definition.required_capabilities) if (capability.empty() || capability.size() > 128 || !required_capabilities.insert(capability).second) { error = "invalid or duplicate workflow required capability"; return false; }
        std::set<std::string> optional_capabilities;
        for (const auto & capability : definition.optional_capabilities) if (capability.empty() || capability.size() > 128 || required_capabilities.count(capability) || !optional_capabilities.insert(capability).second) { error = "invalid, duplicate, or required workflow optional capability"; return false; }
        std::set<std::string> required_context;
        for (const auto & requirement : definition.required_context) if (requirement.empty() || requirement.size() > 128 || !required_context.insert(requirement).second) { error = "invalid or duplicate workflow context requirement"; return false; }
        std::set<std::string> continuation_targets;
        for (const auto & target : definition.continuation_workflow_targets) if (target.empty() || target.size() > 256 || !continuation_targets.insert(target).second) { error = "invalid or duplicate workflow continuation target"; return false; }
        std::set<std::string> continuation_link_targets;
        for (const auto & link : definition.continuation_links) {
            if (link.target_workflow_ref.empty() || link.target_workflow_ref.size() > 256 ||
                    link.source_transition_id.size() > 64 ||
                    link.source_terminal_state.size() > 64 ||
                    link.target_workflow_revision.size() > 128 ||
                    !continuation_link_targets.insert(link.source_transition_id + "@" +
                        link.source_terminal_state + "->" + link.target_workflow_ref + "@" +
                        link.target_workflow_revision).second) {
                error = "invalid or duplicate workflow continuation link"; return false;
            }
            if (link.source_terminal_state.empty() ||
                    std::find(definition.terminal_states.begin(), definition.terminal_states.end(),
                        link.source_terminal_state) == definition.terminal_states.end()) {
                error = "workflow continuation link must name a declared source terminal state";
                return false;
            }
            std::set<std::pair<std::string, std::string>> link_fields;
            for (const auto & binding : link.argument_bindings) {
                if ((binding.target_transition_id.empty() && binding.target_tool_name.empty()) ||
                        binding.target_transition_id.size() > 64 || binding.target_tool_name.size() > 128 ||
                        binding.source_transition_id.size() > 64 ||
                        binding.target_json_pointer.empty() || binding.target_json_pointer.front() != '/' ||
                        binding.source_json_pointer.empty() || binding.source_json_pointer.front() != '/' ||
                        !link_fields.emplace(binding.target_transition_id + ":" + binding.target_tool_name,
                            binding.target_json_pointer).second) {
                    error = "workflow continuation binding requires tool and unique source/target pointers"; return false;
                }
            }
        }
        for (const auto & check : definition.verification_checks) {
            const auto known_transition = [&](const std::string & id) {
                return std::any_of(definition.transitions.begin(), definition.transitions.end(),
                    [&](const auto & transition) { return transition.id == id &&
                        transition.kind == common_plan_workflow_transition_kind::tool; });
            };
            if (check.phase != common_plan_verification_phase::before_answer && !known_transition(check.transition_id)) {
                error = "verification check names an unavailable tool transition"; return false;
            }
            for (const auto * operand : {&check.left, &check.right}) {
                if ((operand->source == "observation" || operand->source == "transition_arguments") &&
                        !known_transition(operand->transition_id)) {
                    error = "verification operand names an unavailable tool transition"; return false;
                }
                if (check.phase == common_plan_verification_phase::before_tool && operand->source == "result") {
                    error = "before_tool verification cannot read a future result"; return false;
                }
            }
        }
        if (!definition.transitions.empty()) {
            if (!valid_id(definition.start_state) || definition.terminal_states.empty()) { error = "workflow graph has invalid start or terminal states"; return false; }
            std::set<std::string> transition_ids, states{definition.start_state};
            for (const auto & state : definition.terminal_states) if (!valid_id(state) || !states.insert(state).second) { error = "workflow graph has invalid or duplicate terminal state"; return false; }
            std::map<std::string, std::vector<std::string>> predecessors;
            for (const auto & transition : definition.transitions) {
                const auto arguments = json::parse(transition.arguments_template_json, nullptr, false);
                const bool invalid_model_choice =
                    transition.kind == common_plan_workflow_transition_kind::model_choice &&
                    (!transition.tool_name.empty() ||
                     (transition.choices.empty() &&
                        (transition.candidate_json_pointer.empty() ||
                         transition.candidate_json_pointer.front() != '/' ||
                         transition.candidate_id_json_pointer.empty() ||
                         transition.candidate_id_json_pointer.front() != '/')) ||
                     (!transition.choices.empty() && !transition.candidate_json_pointer.empty()));
                if (!valid_id(transition.id) || !transition_ids.insert(transition.id).second ||
                        !valid_id(transition.from_state) || !valid_id(transition.to_state) ||
                        !arguments.is_object() || transition.cost <= 0.0f || transition.cost > 1000.0f ||
                        (transition.kind == common_plan_workflow_transition_kind::tool && transition.tool_name.empty()) ||
                        invalid_model_choice) {
                    error = "workflow graph has an invalid transition"; return false;
                }
                if (transition.model_instruction.size() > 2048 ||
                        transition.model_instruction.find('\0') != std::string::npos) {
                    error = "workflow transition model_instruction exceeds its bounded text contract";
                    return false;
                }
                if (transition.model_supplies_arguments &&
                        (transition.kind != common_plan_workflow_transition_kind::tool ||
                         transition.from_state != definition.start_state)) {
                    error = "model-supplied workflow arguments are supported only on a graph entry tool transition";
                    return false;
                }
                if (transition.kind == common_plan_workflow_transition_kind::tool &&
                        (!transition.choices.empty() || !transition.candidate_json_pointer.empty())) {
                    error = "tool transitions cannot declare model choices"; return false;
                }
                if (transition.max_pages > 0 &&
                        (transition.kind != common_plan_workflow_transition_kind::tool ||
                         transition.max_pages < 2 || transition.max_pages > 5)) {
                    error = "workflow pagination requires a tool transition and a page limit from 2 to 5";
                    return false;
                }
                if (transition.kind == common_plan_workflow_transition_kind::artifact_compose) {
                    const std::string key = transition.artifact_profile_ref + "@" + transition.artifact_profile_revision;
                    const auto profile = std::find_if(package.artifact_profiles.begin(), package.artifact_profiles.end(), [&](const auto & p) {
                        return p.id == transition.artifact_profile_ref && p.revision == transition.artifact_profile_revision;
                    });
                    if (transition.artifact_profile_ref.empty() || transition.artifact_profile_revision.empty() ||
                            !artifact_profiles.count(key) || profile == package.artifact_profiles.end() ||
                            std::none_of(profile->sections.begin(), profile->sections.end(), [&](const auto & s) { return s.id == transition.artifact_section_id; }) ||
                            !transition.tool_name.empty() || !transition.argument_bindings.empty()) {
                        error = "artifact_compose transition must bind a declared profile section and no tool"; return false;
                    }
                }
                std::set<std::string> choice_ids;
                for (const auto & choice : transition.choices) {
                    if (!valid_id(choice.id) || !choice_ids.insert(choice.id).second ||
                            choice.description.empty() || !valid_id(choice.to_state)) {
                        error = "workflow transition has an invalid or duplicate fixed choice"; return false;
                    }
                    states.insert(choice.to_state);
                    predecessors[choice.to_state].push_back(transition.from_state);
                    for (const auto & pointer : choice.required_observation_json_pointers) {
                        if (pointer.empty() || pointer.front() != '/') {
                            error = "workflow choice requirement must be an RFC-6901 JSON pointer"; return false;
                        }
                    }
                }
                std::set<std::string> targets;
                for (const auto & binding : transition.argument_bindings) {
                    if (transition.kind != common_plan_workflow_transition_kind::tool ||
                            binding.target_json_pointer.empty() || binding.target_json_pointer.front() != '/' ||
                            binding.source_json_pointer.empty() || binding.source_json_pointer.front() != '/' ||
                            (!binding.source_transition_id.empty() && binding.source_transition_id != "$previous" &&
                                !valid_id(binding.source_transition_id)) ||
                            !targets.insert(binding.target_json_pointer).second) {
                        error = "workflow argument binding requires unique target and source JSON pointers"; return false;
                    }
                }
                states.insert(transition.from_state); states.insert(transition.to_state);
                predecessors[transition.to_state].push_back(transition.from_state);
            }
            for (const auto & link : definition.continuation_links) {
                for (const auto & binding : link.argument_bindings) {
                    if (!binding.source_transition_id.empty() &&
                            !transition_ids.count(binding.source_transition_id)) {
                        error = "workflow continuation binding references an unknown source transition";
                        return false;
                    }
                }
                if (link.source_terminal_state.empty()) {
                    error = "workflow continuation link requires a declared source terminal state";
                    return false;
                }
                if (!link.source_transition_id.empty() &&
                        !transition_ids.count(link.source_transition_id)) {
                    error = "workflow continuation link references an unknown source transition";
                    return false;
                }
                if (!link.source_terminal_state.empty() &&
                        std::find(definition.terminal_states.begin(), definition.terminal_states.end(),
                            link.source_terminal_state) == definition.terminal_states.end()) {
                    error = "workflow continuation link references a non-terminal source state";
                    return false;
                }
                if (!link.source_transition_id.empty() && !link.source_terminal_state.empty()) {
                    const auto source = std::find_if(definition.transitions.begin(), definition.transitions.end(),
                        [&](const auto & candidate) { return candidate.id == link.source_transition_id; });
                    const bool reaches_terminal = source != definition.transitions.end() &&
                        (source->to_state == link.source_terminal_state ||
                         std::any_of(source->choices.begin(), source->choices.end(), [&](const auto & choice) {
                             return choice.to_state == link.source_terminal_state;
                         }));
                    if (!reaches_terminal) {
                        error = "workflow continuation source transition does not reach its declared terminal state";
                        return false;
                    }
                }
            }
            for (const auto & terminal : definition.terminal_states) {
                if (std::any_of(definition.transitions.begin(), definition.transitions.end(), [&](const auto & transition) { return transition.from_state == terminal; })) {
                    error = "workflow terminal state has outgoing transition"; return false;
                }
            }
            // A bounded workflow must have a possible route to a terminal
            // state from every declared state.  This rejects dead branches
            // and closed cycles before a package becomes executable.
            std::set<std::string> can_finish(definition.terminal_states.begin(), definition.terminal_states.end());
            std::vector<std::string> pending(definition.terminal_states.begin(), definition.terminal_states.end());
            while (!pending.empty()) {
                const auto state = pending.back(); pending.pop_back();
                for (const auto & predecessor : predecessors[state]) {
                    if (can_finish.insert(predecessor).second) pending.push_back(predecessor);
                }
            }
            for (const auto & state : states) if (!can_finish.count(state)) {
                error = "workflow graph has a state with no terminal path"; return false;
            }
        }
    }
    for (const auto & workflow : package.workflows) {
        for (const auto & link : workflow.definition.continuation_links) {
            const auto target = std::find_if(package.workflows.begin(), package.workflows.end(), [&](const auto & candidate) {
                return candidate.definition.workflow_ref == link.target_workflow_ref &&
                    (link.target_workflow_revision.empty() ||
                     candidate.definition.workflow_revision == link.target_workflow_revision);
            });
            if (target == package.workflows.end()) continue; // May be supplied by an already-installed package.
            for (const auto & binding : link.argument_bindings) {
                if (binding.target_transition_id.empty()) continue;
                const auto transition = std::find_if(target->definition.transitions.begin(), target->definition.transitions.end(),
                    [&](const auto & candidate) { return candidate.id == binding.target_transition_id; });
                if (transition == target->definition.transitions.end() ||
                        transition->kind != common_plan_workflow_transition_kind::tool ||
                        (!binding.target_tool_name.empty() && transition->tool_name != binding.target_tool_name)) {
                    error = "workflow continuation binding references an invalid target transition";
                    return false;
                }
            }
        }
    }
    return true;
}
}

bool common_agent_package_parse_json(const std::string & text, common_agent_bootstrap_package & package, std::string & error) {
    error.clear();
    try {
        // Reject ambiguous contracts before the parser can replace an earlier key.
        // Track each object separately so sibling objects may use the same fields.
        std::vector<std::set<std::string>> object_keys;
        bool duplicate_key = false;
        const auto root = json::parse(text, [&](int, json::parse_event_t event, json & value) {
            if (event == json::parse_event_t::object_start) object_keys.emplace_back();
            else if (event == json::parse_event_t::object_end) object_keys.pop_back();
            else if (event == json::parse_event_t::key &&
                    !object_keys.back().insert(value.get<std::string>()).second) duplicate_key = true;
            return true;
        }, false);
        if (duplicate_key) { error = "package contains a duplicate JSON object key"; return false; }
        if (!root.is_object() || (root.value("schema_version", 0) != 1 && root.value("schema_version", 0) != 2)) {
            error = "package requires supported schema_version (1 or 2)"; return false;
        }
        if (root.value("schema_version", 0) == 1 && root.contains("workflows")) {
            for (const auto & workflow : root["workflows"]) {
                if (workflow.value("require_declared_claims", false) ||
                        !workflow.value("verification_checks", json::array()).empty()) {
                    error = "workflow verification contracts require package schema_version 2"; return false;
                }
            }
        }
        package = {};
        package.name = root.value("name", std::string{});
        package.version = root.value("version", std::string{});
        if (root.contains("artifact_profiles")) {
            if (!root["artifact_profiles"].is_array()) { error = "artifact_profiles must be an array"; return false; }
            for (const auto & item : root["artifact_profiles"]) {
                if (!item.is_object() || !item.contains("sections") || !item["sections"].is_array()) { error = "artifact profile requires sections"; return false; }
                common_plan_artifact_profile profile;
                profile.id = item.value("id", std::string{}); profile.revision = item.value("revision", std::string{});
                profile.title = item.value("title", std::string{}); profile.purpose = item.value("purpose", std::string{});
                for (const auto & sj : item["sections"]) {
                    if (!sj.is_object()) { error = "artifact profile section must be an object"; return false; }
                    common_plan_artifact_section_profile section;
                    section.id=sj.value("id",std::string{}); section.title=sj.value("title",std::string{}); section.purpose=sj.value("purpose",std::string{});
                    if (sj.contains("required_observation_sources") && !strings(sj["required_observation_sources"],section.required_observation_sources)) { error="artifact section sources must be strings";return false; }
                    if (sj.contains("allowed_block_kinds") && !strings(sj["allowed_block_kinds"],section.allowed_block_kinds)) { error="artifact allowed_block_kinds must be strings";return false; }
                    section.required=sj.value("required",true);
                    if (sj.contains("evidence_relations") && !strings(sj["evidence_relations"],section.evidence_relations)) { error="artifact evidence_relations must be strings";return false; }
                    profile.sections.push_back(std::move(section));
                }
                package.artifact_profiles.push_back(std::move(profile));
            }
        }
        if (root.contains("procedures")) {
            if (!root["procedures"].is_array()) { error = "package procedures must be an array"; return false; }
            for (const auto & item : root["procedures"]) {
                if (!item.is_object()) { error = "package procedure must be an object"; return false; }
                common_agent_bootstrap_procedure procedure;
                procedure.id = item.value("id", std::string{}); procedure.content = item.value("content", std::string{}); procedure.summary = item.value("summary", procedure.id);
                procedure.importance = item.value("importance", 0.8f); procedure.confidence = item.value("confidence", 1.0f);
                package.procedures.push_back(std::move(procedure));
            }
        }
        if (root.contains("blueprints")) {
            if (!root["blueprints"].is_array()) { error = "package blueprints must be an array"; return false; }
            for (const auto & item : root["blueprints"]) {
                if (!item.is_object() || !item.contains("steps") || !item["steps"].is_array()) { error = "package blueprint requires steps"; return false; }
                common_agent_bootstrap_blueprint blueprint;
                blueprint.id = item.value("id", std::string{}); blueprint.source_revision = item.value("source_revision", std::string{}); blueprint.selection_description = item.value("selection_description", std::string{});
                blueprint.purpose = item.value("purpose", std::string{});
        blueprint.goal = item.value("goal", std::string{}); blueprint.success_criteria = item.value("success_criteria", std::string{});
        if (item.contains("required_capabilities") && !strings(item["required_capabilities"], blueprint.required_capabilities)) { error = "blueprint required_capabilities must be strings"; return false; }
                if (item.contains("workflow_policy") && !workflow_policy(item["workflow_policy"], blueprint.workflow_policy)) { error = "blueprint workflow_policy must be preferred or required"; return false; }
                if (item.contains("procedure_refs") && !strings(item["procedure_refs"], blueprint.procedure_refs)) { error = "blueprint procedure_refs must be strings"; return false; }
                if (item.contains("next_action")) { if (!item["next_action"].is_string()) { error = "next_action must be a string"; return false; } blueprint.next_action = item["next_action"].get<std::string>(); }
                if (item.contains("constraints") && !constraints(item["constraints"], blueprint.constraints)) { error = "blueprint constraints are invalid"; return false; }
                if (item.contains("assumptions") && !assumptions(item["assumptions"], blueprint.assumptions)) { error = "blueprint assumptions are invalid"; return false; }
                if (item.contains("workflow_bindings")) {
                    if (!item["workflow_bindings"].is_array()) { error = "blueprint workflow_bindings must be an array"; return false; }
                    for (const auto & source : item["workflow_bindings"]) {
                        if (!source.is_object() || !source["workflow_ref"].is_string() || !source["workflow_revision"].is_string()) { error = "blueprint workflow binding is invalid"; return false; }
                        blueprint.workflow_bindings.push_back({source["workflow_ref"].get<std::string>(), source["workflow_revision"].get<std::string>()});
                    }
                }
                for (const auto & source : item["steps"]) {
                    if (!source.is_object()) { error = "package step must be an object"; return false; }
                    common_plan_step step; step.id = source.value("id", std::string{}); step.title = source.value("title", std::string{}); step.objective = source.value("objective", std::string{}); step.intended_contribution = source.value("contribution", step.objective); step.optional = source.value("optional", false);
                    if (source.contains("depends_on") && !strings(source["depends_on"], step.depends_on)) { error = "step dependencies must be strings"; return false; }
                    if (source.contains("required_evidence") && !strings(source["required_evidence"], step.required_evidence)) { error = "step required_evidence must be strings"; return false; }
                    blueprint.steps.push_back(std::move(step));
                }
                package.blueprints.push_back(std::move(blueprint));
            }
        }
        if (root.contains("workflows")) {
            if (!root["workflows"].is_array()) { error = "package workflows must be an array"; return false; }
            for (const auto & item : root["workflows"]) {
                if (!item.is_object()) { error = "package workflow must be an object"; return false; }
                common_agent_bootstrap_workflow workflow;
                workflow.id = item.value("id", std::string{});
                workflow.source_revision = item.value("source_revision", std::string{});
                workflow.definition.workflow_ref = item.value("workflow_ref", std::string{});
                workflow.definition.workflow_revision = item.value("workflow_revision", std::string{});
                workflow.definition.family = item.value("family", std::string{});
                workflow.definition.graph_revision = item.value("graph_revision", std::string{});
                if (item.contains("allowed_tools") && !strings(item["allowed_tools"], workflow.definition.allowed_tools)) { error = "workflow allowed_tools must be strings"; return false; }
                if (item.contains("required_capabilities") && !strings(item["required_capabilities"], workflow.definition.required_capabilities)) { error = "workflow required_capabilities must be strings"; return false; }
                if (item.contains("optional_capabilities") && !strings(item["optional_capabilities"], workflow.definition.optional_capabilities)) { error = "workflow optional_capabilities must be strings"; return false; }
                if (item.contains("required_context") && !strings(item["required_context"], workflow.definition.required_context)) { error = "workflow required_context must be strings"; return false; }
                if (item.contains("continuation_workflow_targets") && !strings(item["continuation_workflow_targets"], workflow.definition.continuation_workflow_targets)) { error = "workflow continuation_workflow_targets must be strings"; return false; }
                if (item.contains("continuation_links")) {
                    if (!item["continuation_links"].is_array()) { error = "workflow continuation_links must be an array"; return false; }
                    for (const auto & link_json : item["continuation_links"]) {
                        if (!link_json.is_object()) { error = "workflow continuation link must be an object"; return false; }
                        common_plan_workflow_continuation_link link;
                        link.target_workflow_ref = link_json.value("workflow_ref", std::string{});
                        link.source_transition_id = link_json.value("from_transition", std::string{});
                        link.source_terminal_state = link_json.value("from_terminal_state", std::string{});
                        link.target_workflow_revision = link_json.value("workflow_revision", std::string{});
                        if (link_json.contains("bindings")) {
                            if (!link_json["bindings"].is_array()) { error = "workflow continuation bindings must be an array"; return false; }
                            for (const auto & binding_json : link_json["bindings"]) {
                                if (!binding_json.is_object()) { error = "workflow continuation binding must be an object"; return false; }
                                link.argument_bindings.push_back({
                                    binding_json.value("transition", std::string{}),
                                    binding_json.value("tool", std::string{}),
                                    binding_json.value("target", std::string{}),
                                    binding_json.value("source", std::string{}),
                                    binding_json.value("required", true),
                                    binding_json.value("from_transition", std::string{})});
                            }
                        }
                        workflow.definition.continuation_links.push_back(std::move(link));
                    }
                }
                if (!workflow_graph(item, workflow.definition, error)) return false;
                package.workflows.push_back(std::move(workflow));
            }
        }
        return validate(package, error);
    } catch (const std::exception & e) { error = std::string("invalid package JSON: ") + e.what(); return false; }
}

bool common_agent_package_to_json(const common_agent_bootstrap_package & package, std::string & text, std::string & error, bool pretty) {
    if (!validate(package, error)) return false;
    json root = {{"schema_version", 1}, {"name", package.name}, {"version", package.version}, {"procedures", json::array()}, {"workflows", json::array()}, {"blueprints", json::array()}, {"artifact_profiles", json::array()}};
    if (std::any_of(package.workflows.begin(), package.workflows.end(), [](const auto & workflow) {
            return workflow.definition.require_declared_claims || !workflow.definition.verification_checks.empty();
        })) root["schema_version"] = 2;
    for (const auto & procedure : package.procedures) root["procedures"].push_back({{"id", procedure.id}, {"summary", procedure.summary}, {"content", procedure.content}, {"importance", procedure.importance}, {"confidence", procedure.confidence}});
    for (const auto & profile : package.artifact_profiles) {
        json value={{"id",profile.id},{"revision",profile.revision},{"title",profile.title},{"purpose",profile.purpose},{"sections",json::array()}};
        for(const auto & section:profile.sections) value["sections"].push_back({{"id",section.id},{"title",section.title},{"purpose",section.purpose},{"required_observation_sources",section.required_observation_sources},{"allowed_block_kinds",section.allowed_block_kinds},{"required",section.required},{"evidence_relations",section.evidence_relations}});
        root["artifact_profiles"].push_back(std::move(value));
    }
    for (const auto & blueprint : package.blueprints) {
        json value = {{"id", blueprint.id}, {"goal", blueprint.goal}, {"success_criteria", blueprint.success_criteria}, {"steps", json::array()}};
        if (!blueprint.source_revision.empty()) value["source_revision"] = blueprint.source_revision;
        if (!blueprint.selection_description.empty()) value["selection_description"] = blueprint.selection_description;
        if (!blueprint.purpose.empty()) value["purpose"] = blueprint.purpose;
        if (blueprint.workflow_policy != common_plan_workflow_policy::preferred) value["workflow_policy"] = workflow_policy_name(blueprint.workflow_policy);
        if (!blueprint.procedure_refs.empty()) value["procedure_refs"] = blueprint.procedure_refs;
        if (!blueprint.required_capabilities.empty()) value["required_capabilities"] = blueprint.required_capabilities;
        if (blueprint.next_action) value["next_action"] = *blueprint.next_action;
        if (!blueprint.constraints.empty()) { value["constraints"] = json::array(); for (const auto & constraint : blueprint.constraints) value["constraints"].push_back({{"id", constraint.id}, {"description", constraint.description}, {"hard", constraint.hard}}); }
        if (!blueprint.assumptions.empty()) { value["assumptions"] = json::array(); for (const auto & assumption : blueprint.assumptions) value["assumptions"].push_back({{"id", assumption.id}, {"statement", assumption.statement}, {"confidence", assumption.confidence}}); }
        if (!blueprint.workflow_bindings.empty()) { value["workflow_bindings"] = json::array(); for (const auto & binding : blueprint.workflow_bindings) value["workflow_bindings"].push_back({{"workflow_ref", binding.workflow_ref}, {"workflow_revision", binding.workflow_revision}}); }
        for (const auto & step : blueprint.steps) { json item = {{"id", step.id}, {"title", step.title}, {"objective", step.objective}}; if (!step.intended_contribution.empty() && step.intended_contribution != step.objective) item["contribution"] = step.intended_contribution; if (step.optional) item["optional"] = true; if (!step.depends_on.empty()) item["depends_on"] = step.depends_on; if (!step.required_evidence.empty()) item["required_evidence"] = step.required_evidence; value["steps"].push_back(std::move(item)); }
        root["blueprints"].push_back(std::move(value));
    }
    for (const auto & workflow : package.workflows) {
        json value = {{"id", workflow.id}, {"workflow_ref", workflow.definition.workflow_ref}, {"workflow_revision", workflow.definition.workflow_revision}, {"family", workflow.definition.family}, {"graph_revision", workflow.definition.graph_revision}};
        if (!workflow.source_revision.empty()) value["source_revision"] = workflow.source_revision;
        if (!workflow.definition.allowed_tools.empty()) value["allowed_tools"] = workflow.definition.allowed_tools;
        if (!workflow.definition.required_capabilities.empty()) value["required_capabilities"] = workflow.definition.required_capabilities;
        if (!workflow.definition.optional_capabilities.empty()) value["optional_capabilities"] = workflow.definition.optional_capabilities;
        if (!workflow.definition.required_context.empty()) value["required_context"] = workflow.definition.required_context;
        if (!workflow.definition.continuation_workflow_targets.empty()) value["continuation_workflow_targets"] = workflow.definition.continuation_workflow_targets;
        if (!workflow.definition.continuation_links.empty()) {
            value["continuation_links"] = json::array();
            for (const auto & link : workflow.definition.continuation_links) {
                json item = {{"workflow_ref", link.target_workflow_ref}};
                if (!link.source_transition_id.empty()) item["from_transition"] = link.source_transition_id;
                if (!link.source_terminal_state.empty()) item["from_terminal_state"] = link.source_terminal_state;
                if (!link.target_workflow_revision.empty()) item["workflow_revision"] = link.target_workflow_revision;
                if (!link.argument_bindings.empty()) {
                    item["bindings"] = json::array();
                    for (const auto & binding : link.argument_bindings) {
                        json binding_json = {
                            {"transition", binding.target_transition_id},
                            {"tool", binding.target_tool_name},
                            {"target", binding.target_json_pointer},
                            {"source", binding.source_json_pointer},
                            {"required", binding.required}};
                        if (!binding.source_transition_id.empty()) binding_json["from_transition"] = binding.source_transition_id;
                        item["bindings"].push_back(std::move(binding_json));
                    }
                }
                value["continuation_links"].push_back(std::move(item));
            }
        }
        common_plan_workflow_grounding_to_json(workflow.definition, value);
        if (workflow.definition.resource_read_policy == common_plan_workflow_resource_read_policy::chunked) {
            value["resource_read_policy"] = "chunked";
        }
        if (!workflow.definition.transitions.empty()) {
            value["start_state"] = workflow.definition.start_state;
            value["terminal_states"] = workflow.definition.terminal_states;
            value["transitions"] = json::array();
            for (const auto & transition : workflow.definition.transitions) {
                json item = {{"id", transition.id}, {"from", transition.from_state}, {"to", transition.to_state},
                    {"kind", workflow_transition_kind_name(transition.kind)}, {"arguments", transition.arguments_template_json}, {"cost", transition.cost}};
                if (!transition.tool_name.empty()) item["tool"] = transition.tool_name;
                if (transition.model_supplies_arguments) item["model_supplies_arguments"] = true;
                if (!transition.model_instruction.empty()) item["model_instruction"] = transition.model_instruction;
                if (transition.kind == common_plan_workflow_transition_kind::artifact_compose) {
                    item["artifact_profile_ref"] = transition.artifact_profile_ref;
                    item["artifact_profile_revision"] = transition.artifact_profile_revision;
                    item["artifact_section_id"] = transition.artifact_section_id;
                }
                if (!transition.candidate_json_pointer.empty()) item["candidates"] = transition.candidate_json_pointer;
                if (transition.kind == common_plan_workflow_transition_kind::model_choice &&
                        transition.choices.empty() && transition.candidate_id_json_pointer != "/id") {
                    item["candidate_id"] = transition.candidate_id_json_pointer;
                }
                if (!transition.choices.empty()) {
                    item["choices"] = json::array();
                    for (const auto & choice : transition.choices) {
                        json choice_json = {{"id", choice.id}, {"description", choice.description}, {"to", choice.to_state}};
                        if (!choice.required_observation_json_pointers.empty()) {
                            choice_json["requires"] = choice.required_observation_json_pointers;
                        }
                        item["choices"].push_back(std::move(choice_json));
                    }
                }
                if (!transition.argument_bindings.empty()) {
                    item["bindings"] = json::array();
                    for (const auto & binding : transition.argument_bindings) {
                        item["bindings"].push_back({{"target", binding.target_json_pointer},
                            {"from", binding.source_transition_id},
                            {"source", binding.source_json_pointer}, {"required", binding.required}});
                    }
                }
                if (transition.optional) item["optional"] = true;
                if (transition.max_pages > 0) item["max_pages"] = transition.max_pages;
                value["transitions"].push_back(std::move(item));
            }
        }
        root["workflows"].push_back(std::move(value));
    }
    text = pretty ? root.dump(2) + "\n" : root.dump(); error.clear(); return true;
}
