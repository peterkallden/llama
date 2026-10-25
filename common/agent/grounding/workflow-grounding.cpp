#include "workflow-grounding.h"
#include "plan/plan-workflow-state.h"
#include "agent/tooling/contracts/schema-contract.h"
#include <algorithm>
#include <cstdint>
#include <map>
#include <iomanip>
#include <set>
#include <sstream>

using json = nlohmann::ordered_json;
namespace {
std::string fingerprint(const std::string & text) {
    uint64_t hash = 14695981039346656037ULL;
    for (unsigned char byte : text) { hash ^= byte; hash *= 1099511628211ULL; }
    return std::to_string(hash);
}
bool present(const json & value) {
    return !value.is_null() && (!value.is_string() || !value.get<std::string>().empty()) &&
        ((!value.is_array() && !value.is_object()) || !value.empty());
}
bool resolve(const common_plan_verification_operand & operand, const common_agent_request & request,
        const common_plan_state & plan, const json & arguments, const json & result,
        json & value, json & evidence, std::string & error) {
    json root;
    if (operand.source == "literal") root = json::parse(operand.literal_json);
    else if (operand.source == "arguments") root = arguments;
    else if (operand.source == "result") root = result;
    else if (operand.source == "observation" || operand.source == "transition_arguments") {
        const auto * observed = common_plan_workflow_observation(plan, operand.transition_id, false, operand.tool_name);
        if (!observed) { error = "completed source observation unavailable"; return false; }
        root = json::parse(observed->summary, nullptr, false);
        evidence.push_back(observed->id);
        if (operand.source == "transition_arguments") {
            bool found = false;
            const size_t begin = plan.route_binding ? plan.route_binding->workflow_step_start_index : 0;
            for (size_t index = begin; index < plan.steps.size(); ++index) {
                const auto & step = plan.steps[index];
                if (observed->id.rfind("tool:" + step.id + ":", 0) != 0) continue;
                for (auto item = plan.observations.rbegin(); item != plan.observations.rend(); ++item) if (item->id == "workflow_arguments:" + step.id && item->source == "workflow_arguments") {
                    root = json::parse(item->summary, nullptr, false); evidence.push_back(item->id); found = true; break;
                }
            }
            if (!found) { error = "verified transition arguments unavailable"; return false; }
        }
    } else if (operand.source == "handoff" && plan.route_binding && plan.workflow_definition) {
        const auto id = "workflow_handoff:" + plan.route_binding->continuation_source_fingerprint +
            ":" + plan.workflow_definition->workflow_ref;
        bool found = false;
        for (auto observed = plan.observations.rbegin(); observed != plan.observations.rend(); ++observed) {
            if (observed->id != id || observed->source != "workflow_handoff" ||
                    !common_plan_workflow_observation_is_usable(*observed)) continue;
            root = json::parse(observed->summary);
            if (root.value("target_revision", std::string{}) != plan.workflow_definition->workflow_revision ||
                    root.value("source_fingerprint", std::string{}) != plan.route_binding->continuation_source_fingerprint) continue;
            evidence.push_back(observed->id); found = true; break;
        }
        if (!found) { error = "declared handoff evidence unavailable"; return false; }
    } else { error = "verification source unavailable"; return false; }
    try {
        const json::json_pointer pointer(operand.pointer);
        if (root.is_discarded() || !root.contains(pointer)) { error = "verification field missing: " + operand.pointer; return false; }
        value = root.at(pointer);
        if (operand.dataset_rows) {
            if (!value.is_string() || !request.grounding_dataset_reader) { error = "bounded dataset reader unavailable"; return false; }
            const auto uri = value.get<std::string>();
            if (!request.grounding_dataset_reader(uri, value, error)) return false;
            evidence.push_back(uri);
        }
        return true;
    } catch (const std::exception &) { error = "verification pointer or dataset is invalid"; return false; }
}
json values(const json & value) { return value.is_array() ? value : json::array({value}); }

// Evaluate whole tuples against normalized CubeRegions. Never merge regions
// into independent per-dimension allow-lists. Unsupported constructs abstain.
std::string dimension_membership(const json & arguments, const json & rows, const json & mapping) {
    if (!arguments.is_object() || !rows.is_array() || mapping.empty()) return "unknown";
    struct region { bool include = true; std::map<std::string, std::set<std::string>> dimensions; };
    std::map<size_t, region> regions;
    for (const auto & row : rows) {
        if (!row.is_object()) return "unknown";
        if (row.value("artifact_kind", std::string{}) != "Value") continue;
        if (!row.contains("constraint_region") || !row["constraint_region"].is_number_integer() || row["constraint_region"] < 0) return "unknown";
        if (row.value("key_include", std::string("true")) != "true") return "unknown";
        const auto include = row.value("region_include", std::string{});
        if (include != "true" && include != "false") return "unknown";
        auto & group = regions[row["constraint_region"].get<size_t>()];
        group.include = include == "true";
        group.dimensions[row.value("dimension_id", std::string{})].insert(row.value("code", std::string{}));
    }
    if (regions.empty()) return "unknown";
    std::vector<std::map<std::string, std::string>> tuples(1);
    for (const auto & field : mapping.items()) {
        if (!arguments.contains(field.key()) || !present(arguments[field.key()])) return "unknown";
        std::vector<std::map<std::string, std::string>> expanded;
        for (const auto & value : values(arguments[field.key()])) {
            if (!value.is_string() || value.get<std::string>().empty()) return "unknown";
            for (auto tuple : tuples) {
                tuple[field.value().get<std::string>()] = value.get<std::string>();
                expanded.push_back(std::move(tuple));
                if (expanded.size() > 256) return "unknown";
            }
        }
        tuples = std::move(expanded);
    }
    if (tuples.empty()) return "unknown";
    for (const auto & tuple : tuples) {
        bool included = false;
        for (const auto & entry : regions) {
            bool matches = true;
            for (const auto & dimension : entry.second.dimensions) {
                const auto selected = tuple.find(dimension.first);
                if (selected == tuple.end()) {
                    // A varying unbound dimension prevents proving the tuple.
                    if (dimension.second.size() != 1) return "unknown";
                } else matches = matches && dimension.second.count(selected->second) != 0;
            }
            for (const auto & selected : tuple) if (!entry.second.dimensions.count(selected.first)) return "unknown";
            if (!matches) continue;
            if (!entry.second.include) return "failed";
            included = true;
        }
        if (!included) return "failed";
    }
    return "passed";
}
std::string evaluate(const common_plan_verification_check & check, const json & left, const json & right) {
    switch (check.operation) {
        case common_plan_verification_operation::required: return present(left) ? "passed" : "unknown";
        case common_plan_verification_operation::equal: return left == right ? "passed" : "failed";
        case common_plan_verification_operation::member:
            if (!right.is_array() || right.empty() || !present(left)) return "unknown";
            for (const auto & value : values(left)) if (std::find(right.begin(), right.end(), value) == right.end()) return "failed";
            return "passed";
        case common_plan_verification_operation::status:
            if (!right.is_array() || right.empty() || (left.is_array() && left.empty())) return "unknown";
            for (const auto & item : values(left)) {
                const auto status = item.is_object() ? item.value("status", json{}) : item;
                if (std::find(right.begin(), right.end(), status) == right.end()) return "unknown";
            }
            return "passed";
        case common_plan_verification_operation::count:
            if ((!left.is_array() && !left.is_object()) || !right.is_number_integer() || right < 0) return "unknown";
            return left.size() == right.get<size_t>() ? "passed" : "failed";
        case common_plan_verification_operation::dimension_membership:
            if (left.is_array()) {
                if (left.empty()) return "unknown";
                for (const auto & row : left) {
                    const auto status = dimension_membership(row, right, json::parse(check.dimension_mapping_json));
                    if (status != "passed") return status;
                }
                return "passed";
            }
            return dimension_membership(left, right, json::parse(check.dimension_mapping_json));
        case common_plan_verification_operation::coverage: {
            if (!left.is_array() || left.empty() || !right.is_object()) return "unknown";
            const auto mapping = json::parse(check.dimension_mapping_json);
            if (mapping.empty()) return "unknown";
            std::vector<json> tuples{json::object()};
            for (const auto & field : mapping.items()) {
                if (!right.contains(field.key())) return "unknown";
                const auto expected = values(right[field.key()]);
                if (expected.empty()) return "unknown";
                std::vector<json> expanded;
                for (const auto & tuple : tuples) for (const auto & code : expected) {
                    auto next = tuple; next[field.value().get<std::string>()] = code;
                    expanded.push_back(std::move(next));
                    if (expanded.size() > 256) return "unknown";
                }
                tuples = std::move(expanded);
                for (const auto & row : left) {
                    if (!row.is_object() || !row.contains(field.value().get<std::string>()) ||
                            !row.contains("value") || !present(row["value"])) return "unknown";
                    if (std::find(expected.begin(), expected.end(), row[field.value().get<std::string>()]) == expected.end()) return "failed";
                }
                for (const auto & value : expected) {
                    if (std::none_of(left.begin(), left.end(), [&](const auto & row) {
                        return row.contains(field.value().get<std::string>()) && row[field.value().get<std::string>()] == value;
                    })) return "failed";
                }
            }
            for (const auto & tuple : tuples) if (std::none_of(left.begin(), left.end(), [&](const auto & row) {
                for (const auto & field : tuple.items()) if (!row.contains(field.key()) || row[field.key()] != field.value()) return false;
                return true;
            })) return "failed";
            return "passed";
        }
    }
    return "unknown";
}
}
static const common_plan_observation * declared_operation_observation(const common_plan_state & plan) {
    if (!plan.workflow_definition) return nullptr;
    const size_t start = plan.route_binding
        ? std::min(plan.route_binding->workflow_step_start_index, plan.steps.size()) : 0;
    for (size_t index = plan.steps.size(); index > start; --index) {
        const auto & step = plan.steps[index - 1];
        if (!step.tool_call || step.status != common_plan_step_status::completed) continue;
        if (std::find(plan.workflow_definition->declared_claim_tools.begin(),
                plan.workflow_definition->declared_claim_tools.end(), step.tool_call->name) ==
                plan.workflow_definition->declared_claim_tools.end() &&
                !plan.workflow_definition->result_contracts.count(step.tool_call->name)) continue;
        if (const auto * observation = common_plan_workflow_observation(plan, {}, false, step.tool_call->name)) return observation;
    }
    return nullptr;
}

bool common_agent_workflow_requires_claims(const common_plan_state & plan) {
    if (plan.workflow_definition) {
        if (plan.workflow_definition->require_declared_claims) return true;
        if (declared_operation_observation(plan)) return true;
    }
    return std::any_of(plan.observations.begin(), plan.observations.end(), [](const auto & observation) {
        if (observation.source != "workflow_grounding_policy" || observation.confidence <= 0) return false;
        const auto value = json::parse(observation.summary, nullptr, false);
        return value.is_object() && value.contains("require_declared_claims") && value["require_declared_claims"] == true;
    });
}

// Profiles describe typed operation results, not caller intent. Intent remains
// a model interpretation, checked separately against the original task.
bool common_agent_result_contract_available(const std::string & operation, const std::string & profile) {
    return operation == "statistics.describe" && profile == "statistics.describe.v1";
}

std::string common_agent_result_requirement_schema(const std::string & profile) {
    if (profile != "statistics.describe.v1") return {};
    return R"({"type":"object","additionalProperties":false,"required":["dataset","columns","measures","group_by","coverage"],"properties":{"dataset":{"type":"string","minLength":1,"maxLength":256,"x-agent-type":"dataset_ref"},"columns":{"type":"array","minItems":1,"maxItems":16,"x-agent-column-of":"dataset","items":{"type":"string","minLength":1,"maxLength":128}},"measures":{"type":"array","minItems":1,"maxItems":6,"items":{"type":"string","enum":["count","null_count","min","max","mean","stddev"]}},"group_by":{"type":"array","maxItems":16,"x-agent-column-of":"dataset","items":{"type":"string","minLength":1,"maxLength":128}},"coverage":{"type":"string","enum":["complete","bounded"]}}})";
}

bool common_agent_validate_result_requirement(const std::string & profile,
        const std::string & text, std::string & normalized, std::string & error) {
    const auto schema = common_agent_result_requirement_schema(profile);
    if (schema.empty()) { error = "unsupported result contract profile: " + profile; return false; }
    if (text.size() > 8192 || !common_schema_normalize_and_validate_object(text, schema, normalized, error)) return false;
    const auto value = json::parse(normalized);
    for (const auto * field : {"columns", "measures", "group_by"}) {
        std::set<std::string> unique;
        for (const auto & item : value[field]) if (!unique.insert(item.get<std::string>()).second) {
            error = "duplicate result requirement " + std::string(field); return false;
        }
    }
    if (value["columns"].size() * value["measures"].size() > 16) {
        error = "result requirement exceeds claim budget"; return false;
    }
    return true;
}

namespace {
bool requirement_value(const common_plan_tool_call & call, const std::string & profile,
        json & value, std::string & error) {
    const auto envelope = json::parse(call.result_requirement_json, nullptr, false);
    if (!envelope.is_object() || envelope.value("schema", json{}) != "analysis-result.v1" ||
            envelope.value("profile", json{}) != profile || !envelope.contains("value")) {
        error = "missing or incompatible step-bound result requirement"; return false;
    }
    std::string normalized;
    if (!common_agent_validate_result_requirement(profile, envelope["value"].dump(), normalized, error)) return false;
    value = json::parse(normalized);
    return true;
}

// Return exact source positions, including null measures and explicit partial
// coverage. Nothing is estimated or synthesized into a numeric observation.
bool required_result_cells(const json & requirement, const json & payload,
        json & cells, std::string & error) {
    cells = json::array();
    if (payload.dump().size() > 65536) { error = "analysis result exceeds the retained evidence byte budget"; return false; }
    if (!payload.is_object() || payload.value("ok", json(true)) != true ||
            !payload.contains("result") || !payload["result"].is_object()) {
        error = "typed analysis result unavailable"; return false;
    }
    const auto & result = payload["result"];
    if (result.value("dataset", json{}) != requirement["dataset"] ||
            result.value("selected_columns", json{}) != requirement["columns"] ||
            result.value("group_by", json::array()) != requirement["group_by"]) {
        error = "result dataset, columns or grouping differs from the selected requirement"; return false;
    }
    if (!result.contains("scan_truncated") || !result["scan_truncated"].is_boolean() ||
            !result.contains("scanned_rows") || (!result["scanned_rows"].is_number_integer() || result["scanned_rows"] < 0)) {
        error = "analysis coverage metadata unavailable"; return false;
    }
    if ((payload.contains("result_truncated") && !payload["result_truncated"].is_boolean()) ||
            (result.contains("result_truncated") && !result["result_truncated"].is_boolean())) {
        error = "analysis truncation metadata is invalid"; return false;
    }
    const bool output_truncated = payload.value("result_truncated", false) || result.value("result_truncated", false);
    if (output_truncated || (requirement["coverage"] == "complete" && result["scan_truncated"] == true)) {
        error = "analysis coverage does not meet the selected requirement"; return false;
    }
    const auto append = [&](const json & columns, const std::string & prefix) {
        if (!columns.is_array() || columns.size() != requirement["columns"].size()) return false;
        std::set<std::string> names;
        for (size_t index = 0; index < columns.size(); ++index) {
            const auto & column = columns[index];
            if (!column.is_object() || !column.contains("name") || !column["name"].is_string() ||
                    !names.insert(column["name"].get<std::string>()).second ||
                    std::find(requirement["columns"].begin(), requirement["columns"].end(), column["name"]) == requirement["columns"].end()) return false;
            for (const auto & measure : requirement["measures"]) {
                const auto name = measure.get<std::string>();
                if (!column.contains(name) || (!column[name].is_number() && !column[name].is_null())) return false;
                // Null is valid only for undefined measures in an empty column.
                if ((name == "count" || name == "null_count") &&
                        (!column[name].is_number_integer() || column[name] < 0)) return false;
                if (column[name].is_null() && column.value("count", json{}) != 0) return false;
                cells.push_back({{"pointer",prefix + "/" + std::to_string(index) + "/" + name},
                    {"value",column[name]}, {"column",column["name"]}, {"measure",name}});
            }
        }
        return true;
    };
    if (requirement["group_by"].empty()) {
        if (!append(result.value("columns", json{}), "/result/columns")) {
            error = "selected column measures are missing or invalid"; return false;
        }
    } else {
        const auto groups = result.value("groups", json{});
        if (!groups.is_array() || groups.size() > 16) { error = "group projection is unavailable or exceeds the claim budget"; return false; }
        std::set<std::string> keys;
        for (size_t index = 0; index < groups.size(); ++index) {
            const auto & group = groups[index];
            if (!group.is_object()) { error = "invalid group projection"; return false; }
            json key = json::array(), group_identity = json::object();
            const auto first_cell = cells.size();
            for (const auto & field : requirement["group_by"]) {
                const auto name = field.get<std::string>();
                if (!group.contains(name) || (!group[name].is_primitive())) { error = "group key missing or invalid"; return false; }
                key.push_back(group[name]);
                group_identity[name] = group[name];
            }
            if (!keys.insert(key.dump()).second ||
                    !append(group.value("columns", json{}), "/result/groups/" + std::to_string(index) + "/columns")) {
                error = "grouped measures are missing, duplicated or invalid"; return false;
            }
            for (size_t cell = first_cell; cell < cells.size(); ++cell) cells[cell]["group"] = group_identity;
        }
        if (groups.empty()) {
            if (result["scanned_rows"] != 0) { error = "empty groups do not establish empty input"; return false; }
            cells.push_back({{"pointer","/result/scanned_rows"},{"value",0},{"limitation","empty_group_set"}});
        }
    }
    if (result["scan_truncated"] == true)
        cells.push_back({{"pointer","/result/scan_truncated"},{"value",true},{"limitation","partial_scan"}});
    if (cells.size() > 16) { error = "required result cells exceed the claim budget"; return false; }
    return true;
}

const common_plan_observation * step_result(const common_plan_state & plan, const common_plan_step & step) {
    for (auto item = plan.observations.rbegin(); item != plan.observations.rend(); ++item) {
        const auto * owner = common_plan_find_tool_observation_step(plan, *item);
        if (owner && owner->id == step.id && step.tool_call && item->source == step.tool_call->name &&
                common_plan_workflow_observation_is_usable(*item)) return &*item;
    }
    return nullptr;
}

common_agent_grounding_result verify_result_contract(const common_agent_request & request, const common_plan_state & plan,
        common_plan_verification_phase phase, const std::string & step_id,
        const json & arguments, const json & payload) {
    common_agent_grounding_result checked;
    if (!plan.workflow_definition || plan.workflow_definition->result_contracts.empty()) return checked;
    const size_t first = plan.route_binding ? std::min(plan.route_binding->workflow_step_start_index, plan.steps.size()) : 0;
    for (size_t index = first; index < plan.steps.size(); ++index) {
        const auto & step = plan.steps[index];
        if (!step.tool_call || (phase != common_plan_verification_phase::before_answer && step.id != step_id)) continue;
        const auto contract = plan.workflow_definition->result_contracts.find(step.tool_call->name);
        if (contract == plan.workflow_definition->result_contracts.end()) continue;
        if (phase == common_plan_verification_phase::before_answer && (step.status != common_plan_step_status::completed || step.optional)) continue;
        std::string error;
        json requirement, cells;
        bool passed = requirement_value(*step.tool_call, contract->second, requirement, error);
        if (passed) {
            const auto envelope = json::parse(step.tool_call->result_requirement_json);
            if (envelope.value("workflow", json{}) != plan.workflow_definition->workflow_ref ||
                    envelope.value("workflow_revision", json{}) != plan.workflow_definition->workflow_revision) {
                passed = false; error = "result requirement belongs to a different workflow revision";
            }
        }
        std::string category = "argument_mismatch";
        if (passed && phase == common_plan_verification_phase::before_tool && step.status != common_plan_step_status::completed) {
            const auto envelope = json::parse(step.tool_call->result_requirement_json);
            const auto version = envelope.value("source_version", json::object());
            if (!version.is_object()) { passed = false; error = "invalid bound source version"; }
            else if (version.value("content_hash", json("")) != "" || version.value("read_context_id", json("")) != "") {
                const auto dataset = std::find_if(request.available_datasets.begin(), request.available_datasets.end(),
                    [&](const auto & item) { return item.ref.uri == requirement["dataset"]; });
                if (dataset == request.available_datasets.end() ||
                        version.value("content_hash", json("")) != dataset->ref.content_hash ||
                        version.value("read_context_id", json("")) != dataset->read_context_id) {
                    passed = false; error = "bound dataset revision or read context is stale";
                }
            }
        }
        if (passed && phase == common_plan_verification_phase::before_tool) {
            for (const auto * field : {"dataset", "columns", "group_by"}) {
                const auto actual = arguments.is_object() ? arguments.value(field, json::array()) : json{};
                if (actual != requirement[field]) {
                    passed = false; error = std::string("argument ") + field + " differs from selected result requirement"; break;
                }
            }
        } else if (passed) {
            category = "insufficient_analysis";
            const auto * observed = phase == common_plan_verification_phase::before_answer ? step_result(plan, step) : nullptr;
            const auto result = observed ? json::parse(observed->summary, nullptr, false) : payload;
            passed = required_result_cells(requirement, result, cells, error);
            if (passed && phase == common_plan_verification_phase::before_answer) {
                category = "answer_coverage";
                const auto claims = payload.value("verified_claims", json::array());
                for (const auto & cell : cells) {
                    if (!observed || !claims.is_array() || std::none_of(claims.begin(), claims.end(), [&](const auto & claim) {
                        return claim.value("status", json{}) == "passed" &&
                            claim.value("evidence_ids", json::array()) == json::array({observed->id}) &&
                            claim.value("source_location", json{}) == cell["pointer"] && claim.value("value", json{}) == cell["value"];
                    })) { passed = false; error = "answer omits a required verified measure or limitation"; break; }
                }
            }
        }
        checked.records.push_back({{"check","analysis-result-contract"},{"step_id",step.id},{"analysis_step_id",step.id},
            {"phase", phase == common_plan_verification_phase::before_tool ? "before_tool" :
                phase == common_plan_verification_phase::after_tool ? "after_tool" : "before_answer"},
            {"status",passed ? "passed" : "failed"},{"required",true},{"workflow_revision",plan.workflow_definition->workflow_revision},{"category",category},
            {"repair",category == "answer_coverage" ? "response" : "analysis"},
            {"reason",passed ? "bound result requirement satisfied" : error},
            {"input_fingerprint",fingerprint(step.tool_call->result_requirement_json + arguments.dump() + payload.dump())},
            {"requirement_fingerprint",fingerprint(step.tool_call->result_requirement_json)}});
        if (const auto * observed = step_result(plan, step)) checked.records.back()["evidence_ids"] = json::array({observed->id});
        if (!passed) { checked.passed = false; checked.summary += step.id + ": " + error + "; "; }
    }
    return checked;
}
} // namespace

common_agent_grounding_result common_agent_verify_workflow(const common_agent_request & request,
        const common_plan_state & plan, common_plan_verification_phase phase,
        const std::string & step_id, const std::string & arguments_text, const std::string & result_text,
        const std::string & canonical_tool_name) {
    auto outcome = verify_result_contract(request, plan, phase, step_id,
        json::parse(arguments_text, nullptr, false), json::parse(result_text, nullptr, false));
    if (!plan.workflow_definition) return outcome;
    std::string transition;
    std::string current_tool = canonical_tool_name;
    for (const auto & step : plan.steps) if (step.id == step_id && step.semantic_alias) {
        const auto comma = step.semantic_alias->rfind(',');
        const auto prefix = std::string("workflow-transition:");
        if (step.semantic_alias->rfind(prefix, 0) == 0) transition = step.semantic_alias->substr(comma == std::string::npos ? prefix.size() : comma + 1);
    }
    for (const auto & step : plan.steps) if (step.id == step_id && step.tool_call) current_tool = step.tool_call->name;
    if (phase != common_plan_verification_phase::before_answer) {
        for (const auto & check : plan.workflow_definition->verification_checks) {
            if (!check.required || check.phase == common_plan_verification_phase::before_answer) continue;
            const auto target = std::find_if(plan.workflow_definition->transitions.begin(), plan.workflow_definition->transitions.end(),
                [&](const auto & candidate) { return candidate.id == check.transition_id && candidate.tool_name == current_tool; });
            if (target != plan.workflow_definition->transitions.end() && transition != target->id) {
                const auto actual = std::find_if(plan.workflow_definition->transitions.begin(), plan.workflow_definition->transitions.end(),
                    [&](const auto & candidate) { return candidate.id == transition && candidate.tool_name == current_tool; });
                if (actual != plan.workflow_definition->transitions.end()) continue;
                outcome.passed = false;
                outcome.summary = "guarded tool lacks its stable workflow transition identity";
                outcome.records.push_back({{"check",check.id},{"status","unknown"},{"required",true},{"reason",outcome.summary}});
                return outcome;
            }
        }
    }
    const auto arguments = json::parse(arguments_text, nullptr, false);
    const auto result = json::parse(result_text, nullptr, false);
    for (const auto & check : plan.workflow_definition->verification_checks) {
        if (check.phase != phase || (phase != common_plan_verification_phase::before_answer &&
                ((!check.tool_name.empty() && check.tool_name != current_tool) ||
                 (!check.transition_id.empty() && check.transition_id != transition)))) continue;
        json left, right, evidence = json::array();
        std::string error, status = "unknown";
        try {
            if (resolve(check.left, request, plan, arguments, result, left, evidence, error) &&
                    (check.operation == common_plan_verification_operation::required ||
                     resolve(check.right, request, plan, arguments, result, right, evidence, error))) status = evaluate(check, left, right);
        } catch (const std::exception &) { error = "unsupported or malformed verification data"; }
        const auto identity = plan.workflow_definition->workflow_ref + "@" + plan.workflow_definition->workflow_revision +
            check.id + std::to_string(static_cast<int>(check.phase)) + std::to_string(static_cast<int>(check.operation)) +
            check.dimension_mapping_json + check.tool_name + arguments_text + result_text + left.dump() + right.dump();
        outcome.records.push_back({{"check", check.id}, {"status", status}, {"required", check.required},
            {"phase", phase == common_plan_verification_phase::before_tool ? "before_tool" :
                phase == common_plan_verification_phase::after_tool ? "after_tool" : "before_answer"},
            {"transition", transition},
            {"reason", error.empty() ? "declared condition " + status : error}, {"evidence_ids", evidence},
            {"input_fingerprint", fingerprint(identity)}, {"workflow_revision", plan.workflow_definition->workflow_revision}});
        if (check.operation == common_plan_verification_operation::coverage && status == "passed") {
            json expected = json::object();
            const auto mapping = json::parse(check.dimension_mapping_json);
            for (const auto & field : mapping.items()) expected[field.value().get<std::string>()] = values(right[field.key()]);
            outcome.records.back()["expected_dimensions"] = std::move(expected);
        }
        if (check.required && status != "passed") {
            outcome.passed = false;
            outcome.summary += check.id + ": " + status + (error.empty() ? "" : " (" + error + ")") + "; ";
            if (status == "unknown" && check.operation == common_plan_verification_operation::required &&
                    check.left.source == "arguments" && outcome.clarification_question.empty())
                outcome.clarification_question = check.clarification_question;
        }
    }
    return outcome;
}

json common_agent_workflow_claim_handles(const common_agent_request & request, const common_plan_state & plan,
        std::string * diagnostic) {
    if (diagnostic) diagnostic->clear();
    json handles = json::array();
    // A contracted analysis exposes its required cells first, across all
    // completed operations in this route segment. Never silently drop a
    // requirement because a later read or the handle limit hid its evidence.
    if (plan.workflow_definition && !plan.workflow_definition->result_contracts.empty()) {
        bool contracted = false;
        const size_t first = plan.route_binding ? std::min(plan.route_binding->workflow_step_start_index, plan.steps.size()) : 0;
        for (size_t index = first; index < plan.steps.size(); ++index) {
            const auto & step = plan.steps[index];
            if (!step.tool_call || step.status != common_plan_step_status::completed || step.optional) continue;
            const auto profile = plan.workflow_definition->result_contracts.find(step.tool_call->name);
            if (profile == plan.workflow_definition->result_contracts.end()) continue;
            contracted = true;
            const auto * observed = step_result(plan, step);
            json requirement, cells;
            std::string reason;
            if (!observed || !requirement_value(*step.tool_call, profile->second, requirement, reason) ||
                    !required_result_cells(requirement, json::parse(observed->summary, nullptr, false), cells, reason)) {
                if (diagnostic) *diagnostic = reason.empty() ? "required analysis evidence unavailable" : reason;
                return json::array();
            }
            const auto payload = json::parse(observed->summary);
            for (const auto & cell : cells) {
                const auto pointer = cell["pointer"].get<std::string>();
                json handle = {{"handle","E" + std::to_string(handles.size() + 1)},
                    {"kind","result_field"},{"value",cell["value"]},
                    {"status",cell["value"].is_null() ? json("undefined") : json(nullptr)},
                    {"evidence_id",observed->id},{"operation",observed->source},
                    {"source_location",pointer},{"source_resource",""},{"content_hash",""},
                    {"dataset",requirement["dataset"]},
                    {"coverage",payload["result"]["scan_truncated"] == true ? "partial" : "complete"},
                    {"evidence_fingerprint",fingerprint(observed->id + observed->summary + pointer)}};
                for (const auto * field : {"column","measure","limitation","group"}) if (cell.contains(field)) handle[field] = cell[field];
                if (handles.size() >= 16 || handles.dump().size() + handle.dump().size() > 8192) {
                    if (diagnostic) *diagnostic = "required analysis evidence exceeds the bounded claim context";
                    return json::array();
                }
                handles.push_back(std::move(handle));
            }
        }
        if (contracted) return handles;
    }

    // Only active-route successful results produce claim handles. Older
    // source data remains reachable through the declared handoff snapshot.
    const auto * declared = declared_operation_observation(plan);
    const bool inline_claims = declared != nullptr;
    const auto * observed = declared ? declared : common_plan_workflow_observation(plan, {}, true);
    if (!observed) observed = common_plan_workflow_observation(plan, {});
    if (!observed) return handles;
    for (const auto & dataset : observed->dataset_refs) {
        if (inline_claims || !request.grounding_dataset_reader) break;
        json rows;
        std::string error;
        if (!request.grounding_dataset_reader(dataset.uri, rows, error) || !rows.is_array()) continue;
        for (size_t index = 0; index < rows.size() && handles.size() < 16; ++index) {
            const auto & row = rows[index];
            if (!row.is_object() || !row.contains("value") || !row["value"].is_number()) continue;
            bool matches_expectations = true;
            for (const auto & item : plan.observations) {
                if (item.source != "workflow_verification") continue;
                const auto proof = json::parse(item.summary, nullptr, false);
                if (!proof.is_object() || !proof.contains("expected_dimensions") ||
                        proof.value("status", std::string{}) != "passed") continue;
                for (const auto & field : proof["expected_dimensions"].items()) {
                    if (!row.contains(field.key()) || !field.value().is_array() ||
                            std::find(field.value().begin(), field.value().end(), row[field.key()]) == field.value().end()) matches_expectations = false;
                }
            }
            if (!matches_expectations) continue;
            const auto unit = row.contains("unit_label") ? row["unit_label"] : row.value("unit", json{});
            if (!unit.is_string() || unit.get<std::string>().empty() || unit.get<std::string>().size() > 256) continue;
            auto dimensions = row;
            dimensions.erase("value"); dimensions.erase("unit"); dimensions.erase("unit_label"); dimensions.erase("status");
            for (auto field = dimensions.begin(); field != dimensions.end();) {
                if (field.key().size() >= 6 && field.key().compare(field.key().size() - 6, 6, "_label") == 0) field = dimensions.erase(field);
                else ++field;
            }
            const auto location = dataset.uri + "#/rows/" + std::to_string(index) + "/value";
            json handle = {{"handle", "E" + std::to_string(handles.size() + 1)},
                {"evidence_fingerprint", fingerprint(observed->id + location + row.dump())},
                {"value", row["value"]}, {"unit", unit}, {"dimensions", dimensions},
                {"status", row.value("status", json{})}, {"evidence_id", observed->id},
                {"dataset", dataset.uri}, {"source_resource", dataset.source_resource_uri},
                {"source_location", location}, {"content_hash", dataset.content_hash}};
            if (handles.dump().size() + handle.dump().size() > 8192) return handles;
            handles.push_back(std::move(handle));
        }
    }
    if (!handles.empty()) return handles;
    // Inline typed results are evidence too. Reference the original observation
    // and JSON position instead of inventing a dataset, unit or source URI.
    const auto payload = json::parse(observed->summary, nullptr, false);
    if (!payload.is_object()) return handles;
    const bool wrapped = payload.contains("result");
    const auto & result = wrapped ? payload["result"] : payload;
    const auto append = [&](auto && self, const json & value, const std::string & pointer, size_t depth) -> void {
        if (depth > 8 || handles.size() >= 16) return;
        if (value.is_number()) {
            json handle = {{"handle", "E" + std::to_string(handles.size() + 1)},
                {"kind", "result_field"}, {"value", value}, {"status", nullptr},
                {"evidence_id", observed->id}, {"operation", observed->source},
                {"source_location", pointer},
                {"evidence_fingerprint", fingerprint(observed->id + observed->summary + pointer)},
                {"coverage", payload.value("result_truncated", json(false)) == true ||
                    (result.is_object() && (result.value("scan_truncated", json(false)) == true ||
                        result.value("result_truncated", json(false)) == true)) ? "partial" : "unspecified"},
                {"dataset", result.is_object() && result.contains("dataset") && result["dataset"].is_string() ? result["dataset"] : json("")},
                {"source_resource", ""}, {"content_hash", ""}};
            if (handles.dump().size() + handle.dump().size() <= 8192) handles.push_back(std::move(handle));
        } else if (value.is_object()) {
            for (const auto & field : value.items()) {
                std::string token;
                for (char ch : field.key()) token += ch == '~' ? "~0" : ch == '/' ? "~1" : std::string(1, ch);
                self(self, field.value(), pointer + "/" + token, depth + 1);
            }
        } else if (value.is_array()) {
            for (size_t index = 0; index < value.size() && handles.size() < 16; ++index)
                self(self, value[index], pointer + "/" + std::to_string(index), depth + 1);
        }
    };
    append(append, result, wrapped ? "/result" : "", 0);
    return handles;
}

std::string common_agent_grounded_draft_schema() {
    return R"({"type":"object","additionalProperties":false,"required":["answer","claims"],"properties":{"answer":{"type":"string","minLength":1,"maxLength":16384},"claims":{"type":"array","minItems":1,"maxItems":16,"items":{"type":"object","additionalProperties":false,"required":["handle","value"],"properties":{"handle":{"type":"string","maxLength":32},"value":{"anyOf":[{"type":"number"},{"type":"null"},{"type":"boolean"}]},"unit":{"type":"string","maxLength":256},"dimensions":{"type":"object","additionalProperties":{"type":"string"}}}}}}})";
}

std::string common_agent_grounded_draft_grammar(const json & handles) {
    if (!handles.is_array() || handles.empty()) return {};
    std::string grammar = "root ::= answer (\"\\n\" answer){0,7} claim{1,16}\n"
        "answer ::= \"answer: \" [^\\n\\r]{1,16384}\nclaim ::= ";
    bool first = true;
    for (const auto & handle : handles) {
        std::string literal = "\nclaim: " + handle.at("handle").get<std::string>() +
            "\nvalue: " + handle.at("value").dump();
        if (handle.value("kind", std::string{}) != "result_field") {
            literal += "\nunit: " + handle.at("unit").dump();
            for (const auto & field : handle.at("dimensions").items()) {
                const auto code = field.value().is_string() ? field.value().get<std::string>() : field.value().dump();
                literal += "\ndimension: " + json(field.key()).dump() + " " + json(code).dump();
            }
        }
        if (!first) grammar += " | ";
        first = false;
        grammar += json(literal).dump();
    }
    return grammar + "\n";
}

bool common_agent_parse_grounded_draft(const std::string & text, bool json_format,
        std::string & answer, json & claims, std::string & error) {
    answer.clear(); claims = json::array(); error.clear();
    if (text.size() > 65536) { error = "grounded draft exceeds byte bound"; return false; }
    if (json_format) {
        std::string normalized;
        if (!common_schema_normalize_and_validate_object(text, common_agent_grounded_draft_schema(), normalized, error)) return false;
        const auto value = json::parse(normalized, nullptr, false);
        if (!value.is_object() || !value.contains("answer") || !value["answer"].is_string() ||
                !value.contains("claims") || !value["claims"].is_array()) { error = "grounded draft requires answer and claims"; return false; }
        answer = value["answer"].get<std::string>(); claims = value["claims"];
    } else {
        std::istringstream lines(text);
        std::string line;
        json * active = nullptr;
        while (std::getline(lines, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            const auto colon = line.find(':');
            if (colon == std::string::npos) { error = "grounded draft field must have a colon"; return false; }
            const auto key = line.substr(0, colon);
            auto value = line.substr(colon + 1);
            const auto first = value.find_first_not_of(" \t"); value = first == std::string::npos ? "" : value.substr(first);
            if (key == "answer") { if (!answer.empty()) answer += "\n"; answer += value; }
            else if (key == "claim") { claims.push_back({{"handle", value}}); active = &claims.back(); }
            else if (active && key == "dimension") {
                std::istringstream fields(value);
                std::string name, code;
                fields >> std::quoted(name);
                if (fields.fail() || name.empty()) { error = "dimension requires a name"; return false; }
                std::getline(fields, code);
                const auto code_begin = code.find_first_not_of(" \t");
                if (code_begin == std::string::npos) { error = "claim dimension code is empty"; return false; }
                code = code.substr(code_begin);
                if (code.front() == '"') {
                    const auto quoted = json::parse(code, nullptr, false);
                    if (!quoted.is_string()) { error = "invalid quoted dimension code"; return false; }
                    code = quoted.get<std::string>();
                }
                if (!active->contains("dimensions")) (*active)["dimensions"] = json::object();
                if ((*active)["dimensions"].contains(name)) { error = "duplicate claim dimension"; return false; }
                (*active)["dimensions"][name] = code;
            }
            else if (active && (key == "value" || key == "unit" || key == "dimensions")) {
                if (active->contains(key)) { error = "duplicate grounded claim field"; return false; }
                auto parsed = json::parse(value, nullptr, false);
                if (key == "unit" && parsed.is_discarded()) (*active)[key] = value;
                else if (!parsed.is_discarded()) (*active)[key] = parsed;
                else { error = "invalid grounded claim value"; return false; }
            } else { error = "unknown grounded draft field"; return false; }
        }
    }
    if (answer.empty() || answer.size() > 16384 || claims.empty() || claims.size() > 16) {
        error = "grounded draft requires a bounded answer and 1..16 declared claims"; return false;
    }
    std::string normalized;
    if (!common_schema_normalize_and_validate_object(json{{"answer",answer},{"claims",claims}}.dump(),
            common_agent_grounded_draft_schema(), normalized, error)) return false;
    return true;
}
common_agent_grounding_result common_agent_verify_declared_claims(const json & claims, const json & handles,
        std::vector<common_agent_research_verification_claim> & verified) {
    common_agent_grounding_result outcome;
    verified.clear();
    if (!claims.is_array() || claims.empty() || claims.size() > 16 || !handles.is_array()) {
        outcome.passed = false; outcome.summary = "declared numeric claims are missing or invalid"; return outcome;
    }
    std::set<std::string> seen;
    for (const auto & claim : claims) {
        const auto found = std::find_if(handles.begin(), handles.end(), [&](const auto & handle) {
            return claim.is_object() && claim.contains("handle") && claim["handle"] == handle["handle"];
        });
        const bool result_field = found != handles.end() && found->value("kind", std::string{}) == "result_field";
        bool matching_dimensions = found != handles.end();
        if (!result_field && matching_dimensions) {
            matching_dimensions = claim.contains("dimensions") && claim["dimensions"].is_object() &&
                claim["dimensions"].size() == (*found)["dimensions"].size();
            if (matching_dimensions) for (const auto & field : (*found)["dimensions"].items()) {
                if (!claim["dimensions"].contains(field.key())) { matching_dimensions = false; break; }
                const auto & proposed = claim["dimensions"][field.key()];
                if (proposed != field.value() && !(proposed.is_string() &&
                        (field.value().is_number() || field.value().is_boolean()) && proposed.get<std::string>() == field.value().dump()))
                    matching_dimensions = false;
            }
        } else if (result_field) {
            matching_dimensions = (!claim.contains("dimensions") || claim["dimensions"] == json::object()) &&
                (!claim.contains("unit") || claim["unit"] == "");
        }
        const bool matched = found != handles.end() && claim.contains("value") &&
            (claim["value"].is_number() || (result_field && (claim["value"].is_null() || claim["value"].is_boolean()))) &&
            claim["value"] == (*found)["value"] &&
            (result_field || (claim.contains("unit") && claim["unit"] == (*found)["unit"])) &&
            matching_dimensions &&
            (!present((*found)["status"]) || (result_field && claim["value"].is_null() && (*found)["status"] == "undefined")) && seen.insert((*found)["handle"].get<std::string>()).second;
        outcome.records.push_back({{"status", matched ? "passed" : "unknown"},
            {"handle", claim.is_object() ? claim.value("handle", json{}) : json{}},
            {"reason", matched ? "declared claim matches a host result position" : "claim lacks matching cell, unit, dimensions or usable status"}});
        if (matched) {
            outcome.records.back()["evidence_ids"] = json::array({(*found)["evidence_id"]});
            outcome.records.back()["source_location"] = (*found)["source_location"];
            outcome.records.back()["value"] = claim["value"];
            for (const auto * field : {"column","measure","group","limitation"})
                if (found->contains(field)) outcome.records.back()[field] = (*found)[field];
            outcome.records.back()["content_hash"] = (*found)["content_hash"];
            outcome.records.back()["input_fingerprint"] = (*found)["evidence_fingerprint"];
        }
        if (!matched) { outcome.passed = false; outcome.summary += "unsupported declared numeric claim; "; continue; }
        // Undefined statistics and explicit coverage flags are verifiable
        // limitations, not numerical research claims.
        if (!claim["value"].is_number()) continue;
        common_agent_research_verification_claim accepted;
        accepted.claim_id = "numeric:" + (*found)["evidence_fingerprint"].get<std::string>();
        accepted.statement = claim.dump();
        accepted.evidence_ids = {(*found)["evidence_id"].get<std::string>()};
        const auto source = (*found)["source_resource"].get<std::string>();
        if (!source.empty()) accepted.resource_uris = {source};
        accepted.confidence = 1.0;
        accepted.numeric_value_json = claim["value"].dump();
        accepted.unit = result_field ? std::string{} : claim["unit"].get<std::string>();
        accepted.dimensions_json = result_field ? "{}" : (*found)["dimensions"].dump();
        accepted.dataset_uri = (*found)["dataset"].get<std::string>();
        accepted.source_location = (*found)["source_location"].get<std::string>();
        verified.push_back(std::move(accepted));
    }
    return outcome;
}

std::string common_agent_render_result_requirements(const common_plan_state & plan) {
    std::string text;
    const size_t first = plan.route_binding ? std::min(plan.route_binding->workflow_step_start_index, plan.steps.size()) : 0;
    for (size_t index = first; index < plan.steps.size(); ++index) {
        const auto & step = plan.steps[index];
        if (!step.tool_call || step.tool_call->result_requirement_json.empty()) continue;
        const auto envelope = json::parse(step.tool_call->result_requirement_json, nullptr, false);
        if (!envelope.is_object() || !envelope.contains("value") || !envelope["value"].is_object()) continue;
        if (text.empty()) text = "\n[Step-bound result requirements: model interpretations, not evidence]\n";
        text += "step: " + json(step.id).dump() + "\noperation: " + json(step.tool_call->name).dump() + "\n";
        for (const auto & field : envelope["value"].items()) text += field.key() + ": " + field.value().dump() + "\n";
    }
    if (!text.empty()) text += "Check these choices against the original user goal. Consistency alone does not establish task correctness. "
        "If the analysis is insufficient or misdirected, request bounded analysis repair rather than merely rewriting the answer.\n";
    return text;
}
