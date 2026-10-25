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
                plan.workflow_definition->declared_claim_tools.end()) continue;
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

common_agent_grounding_result common_agent_verify_workflow(const common_agent_request & request,
        const common_plan_state & plan, common_plan_verification_phase phase,
        const std::string & step_id, const std::string & arguments_text, const std::string & result_text,
        const std::string & canonical_tool_name) {
    common_agent_grounding_result outcome;
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

json common_agent_workflow_claim_handles(const common_agent_request & request, const common_plan_state & plan) {
    json handles = json::array();
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
    return R"({"type":"object","additionalProperties":false,"required":["answer","claims"],"properties":{"answer":{"type":"string","minLength":1,"maxLength":16384},"claims":{"type":"array","minItems":1,"maxItems":16,"items":{"type":"object","additionalProperties":false,"required":["handle","value"],"properties":{"handle":{"type":"string","maxLength":32},"value":{"type":"number"},"unit":{"type":"string","maxLength":256},"dimensions":{"type":"object","additionalProperties":{"type":"string"}}}}}}})";
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
        const bool matched = found != handles.end() && claim.contains("value") && claim["value"].is_number() &&
            claim["value"] == (*found)["value"] &&
            (result_field || (claim.contains("unit") && claim["unit"] == (*found)["unit"])) &&
            matching_dimensions &&
            (!present((*found)["status"])) && seen.insert((*found)["handle"].get<std::string>()).second;
        outcome.records.push_back({{"status", matched ? "passed" : "unknown"},
            {"handle", claim.is_object() ? claim.value("handle", json{}) : json{}},
            {"reason", matched ? "declared numeric claim matches a host cell" : "claim lacks matching cell, unit, dimensions or usable status"}});
        if (matched) {
            outcome.records.back()["evidence_ids"] = json::array({(*found)["evidence_id"]});
            outcome.records.back()["source_location"] = (*found)["source_location"];
            outcome.records.back()["value"] = claim["value"];
            outcome.records.back()["content_hash"] = (*found)["content_hash"];
            outcome.records.back()["input_fingerprint"] = (*found)["evidence_fingerprint"];
        }
        if (!matched) { outcome.passed = false; outcome.summary += "unsupported declared numeric claim; "; continue; }
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
