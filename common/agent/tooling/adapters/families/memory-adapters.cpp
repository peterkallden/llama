#include "agent/tooling/adapters/families/memory-adapters.h"

#include "agent/tooling/contracts/tool-result-contracts.h"
#include "memory/memory-retrieval.h"
#include "memory/memory-tool-service.h"

#include <nlohmann/json.hpp>
#include <ctime>
#include <functional>

using json = nlohmann::ordered_json;

namespace {

common_tool_execution_result failure(
        std::string code,
        common_tool_failure_class kind,
        std::string safe_summary,
        std::string diagnostic = {}) {
    return common_tool_execution_result::failure(
            std::move(code), kind, false, std::move(safe_summary), std::move(diagnostic));
}

common_tool_execution_result validation_failure(
        std::string code, std::string diagnostic, std::string summary = "Tool arguments are invalid.") {
    return failure(std::move(code), common_tool_failure_class::validation,
            std::move(summary), std::move(diagnostic));
}

common_tool_execution_result execution_failure(
        std::string code, std::string diagnostic, std::string summary) {
    return failure(std::move(code), common_tool_failure_class::execution,
            std::move(summary), std::move(diagnostic));
}

common_tool_execution_result not_found_failure(
        std::string code, std::string diagnostic, std::string summary) {
    return failure(std::move(code), common_tool_failure_class::not_found,
            std::move(summary), std::move(diagnostic));
}

bool parse_object(const std::string & input, json & output, std::string & error) {
    output = json::parse(input, nullptr, false);
    if (!output.is_object()) {
        error = "tool arguments must be a JSON object";
        return false;
    }
    return true;
}

bool register_definition(const common_tool_definition & definition,
        common_tool_registry & registry,
        std::function<common_tool_execution_result(const std::string &)> handler,
        std::string & error,
        bool read_only = true,
        bool policy_gated = false) {
    common_registered_tool tool;
    tool.name = definition.name;
    tool.version = definition.version;
    tool.executor_id = definition.executor_id;
    tool.arguments_schema = definition.input_schema_json;
    tool.read_only = read_only;
    tool.policy_gated = policy_gated;
    tool.handler = std::move(handler);
    return registry.register_tool(std::move(tool), error);
}

} // namespace

bool common_try_register_memory_tool_adapter(
        const common_tool_definition & definition,
        const common_native_tool_bindings & bindings,
        common_tool_registry & registry,
        bool & installed,
        std::string & error) {
    installed = false;
    const bool is_memory_tool = definition.executor_id == "builtin.memory_search" ||
            definition.executor_id == "builtin.memory_get" ||
            definition.executor_id == "builtin.memory_inspect" ||
            definition.executor_id == "builtin.memory_conflict_check" ||
            definition.executor_id == "builtin.memory_remember";
    if (!is_memory_tool) return false;

    if (definition.executor_id == "builtin.memory_search" && bindings.memory_store) {
        installed = register_definition(definition, registry, [bindings](const std::string & input) {
            common_memory_tool_context context;
            context.query_defaults = bindings.memory_query;
            context.embed = bindings.embed_memory_query;
            common_memory_tool_search_result search_result;
            common_memory_tool_service service(*bindings.memory_store);
            std::string err;
            if (!service.search(context, input, search_result, err)) {
                return execution_failure("tool.memory_search.retrieve_failed", std::move(err), "Memory search failed.");
            }
            return common_tool_execution_result::success(common_tool_memory_search_result_to_json({
                search_result.hits,
            }).dump());
        }, error);
    } else if (definition.executor_id == "builtin.memory_get" && bindings.memory_store) {
        installed = register_definition(definition, registry, [bindings](const std::string & input) {
            std::string err;
            json arguments;
            if (!parse_object(input, arguments, err) || !arguments.contains("id") || !arguments["id"].is_string()) {
                if (err.empty()) err = "memory_get requires an id";
                return validation_failure("tool.memory_get.invalid_id", std::move(err), "Memory get requires a valid id.");
            }
            const auto id = arguments["id"].get<std::string>();
            if (id.empty() || id.size() > 256) {
                return validation_failure("tool.memory_get.out_of_bounds", "memory id is out of bounds", "Memory id is out of bounds.");
            }
            const auto memory = bindings.memory_store->get(id, err);
            if (!err.empty()) return execution_failure("tool.memory_get.load_failed", std::move(err), "Memory could not be loaded.");
            if (!memory || !common_memory_scope_matches(*memory, bindings.memory_query)) {
                return not_found_failure("tool.memory_get.unavailable", "memory is unavailable in the current scope", "Memory is unavailable in the current scope.");
            }
            return common_tool_execution_result::success(common_tool_memory_get_result_to_json(*memory).dump());
        }, error);
    } else if (definition.executor_id == "builtin.memory_inspect" && bindings.memory_store) {
        installed = register_definition(definition, registry, [bindings](const std::string & input) {
            json arguments;
            std::string err;
            if (!parse_object(input, arguments, err)) {
                return validation_failure("tool.memory_inspect.invalid_arguments", std::move(err));
            }
            const auto records = bindings.memory_store->list(bindings.memory_query, err);
            if (!err.empty()) return execution_failure("tool.memory_inspect.load_failed", std::move(err), "Memory inspection failed.");
            json by_kind = json::object();
            for (const auto & record : records) {
                const auto kind = common_memory_kind_name(record.kind);
                by_kind[kind] = by_kind.value(kind, 0) + 1;
            }
            return common_tool_execution_result::success(json({
                {"count", records.size()}, {"by_kind", std::move(by_kind)},
            }).dump());
        }, error);
    } else if (definition.executor_id == "builtin.memory_conflict_check" && bindings.memory_store) {
        installed = register_definition(definition, registry, [bindings](const std::string & input) {
            json arguments;
            std::string err;
            if (!parse_object(input, arguments, err) || !arguments.contains("content") || !arguments["content"].is_string() || arguments["content"].get<std::string>().empty()) {
                return validation_failure("tool.memory_conflict_check.invalid_content", "memory_conflict_check requires content");
            }
            common_memory_query query = bindings.memory_query;
            query.text = arguments["content"].get<std::string>();
            query.limit = 3;
            query.minimum_score = 0.5f;
            common_memory_retrieval retrieval(*bindings.memory_store);
            const auto hits = retrieval.retrieve(query, err);
            if (!err.empty()) return execution_failure("tool.memory_conflict_check.search_failed", std::move(err), "Memory conflict check failed.");
            json matches = json::array();
            for (const auto & hit : hits) matches.push_back({
                {"id", hit.memory.id}, {"kind", common_memory_kind_name(hit.memory.kind)},
                {"content", hit.memory.content}, {"score", hit.final_score},
            });
            return common_tool_execution_result::success(json({
                {"conflict", !matches.empty()}, {"matches", std::move(matches)},
            }).dump());
        }, error);
    } else if (definition.executor_id == "builtin.memory_remember" && bindings.memory_store) {
        installed = register_definition(definition, registry, [bindings](const std::string & input) {
            common_memory_tool_context context;
            context.query_defaults = bindings.memory_query;
            context.allow_write_proposals = true;
            context.now = std::time(nullptr);
            context.embed = bindings.embed_memory_query;

            common_memory_tool_remember_result remember_result;
            common_memory_tool_service service(*bindings.memory_store);
            std::string err;
            if (!service.remember_proposal(context, input, remember_result, err)) {
                return execution_failure("tool.memory_remember.policy_failed", std::move(err), "Memory remember proposal failed.");
            }

            const auto & proposal = remember_result.proposal;
            const auto & decision = remember_result.decision;
            common_tool_memory_remember_payload response{
                true,
                common_memory_remember_decision_name(decision.decision),
                decision.reason,
                proposal.kind,
                proposal.scope,
                proposal.content,
                decision.related_hits.size(),
            };
            for (const auto & hit : decision.related_hits) {
                response.related.push_back({hit.memory.id, hit.memory.kind, hit.final_score, hit.memory.content});
            }
            if (decision.record.has_value()) {
                if (!bindings.memory_store->put(*decision.record, err)) {
                    response.ok = false;
                    response.decision = "reject";
                    response.error = "failed to persist accepted memory: " + err;
                } else {
                    response.id = decision.record->id;
                }
            }
            return common_tool_execution_result::success(common_tool_memory_remember_result_to_json(response).dump());
        }, error, false, true);
    } else if (definition.executor_id == "builtin.memory_remember" && bindings.memory_remember_proposal) {
        installed = register_definition(definition, registry, bindings.memory_remember_proposal, error, false, true);
    }
    return true;
}
