#include "agent/tooling/adapters/families/document-adapters.h"

#include "agent/tooling/adapters/support/adapter-support.h"

using json = common_adapter_json;

bool common_try_register_document_tool_adapter(
        const common_tool_definition & definition,
        const common_native_tool_bindings & bindings,
        common_tool_registry & registry,
        bool & installed,
        std::string & error) {
    installed = false;
    const bool is_document_tool = definition.executor_id == "builtin.document.tables" ||
            definition.executor_id == "builtin.document.table";
    if (!is_document_tool) return false;

    if (definition.executor_id == "builtin.document.tables" && bindings.document_tables) {
        installed = common_adapter_register_definition(definition, registry, [bindings](const std::string & input) {
            std::string parse_error;
            json arguments;
            if (!common_adapter_parse_object(input, arguments, parse_error) ||
                    !arguments.contains("resource") || !arguments["resource"].is_string()) {
                return common_adapter_validation_failure(
                    "tool.document.tables.invalid_arguments",
                    parse_error.empty() ? "document.tables requires resource" : std::move(parse_error));
            }
            const int max_results = arguments.value("max_results", 32);
            if (max_results < 1 || max_results > 64) {
                return common_adapter_validation_failure(
                    "tool.document.tables.invalid_limit",
                    "document.tables max_results is out of bounds");
            }
            return bindings.document_tables(arguments.dump());
        }, error);
    } else if (definition.executor_id == "builtin.document.table" && bindings.document_table) {
        installed = common_adapter_register_definition(definition, registry, [bindings](const std::string & input) {
            std::string parse_error;
            json arguments;
            if (!common_adapter_parse_object(input, arguments, parse_error) ||
                    !arguments.contains("resource") || !arguments["resource"].is_string()) {
                return common_adapter_validation_failure(
                    "tool.document.table.invalid_arguments",
                    parse_error.empty() ? "document.table requires resource" : std::move(parse_error));
            }
            size_t locators = 0;
            for (const auto * key : {"table", "table_index", "node_id"}) {
                if (arguments.contains(key)) ++locators;
            }
            if (locators != 1) {
                return common_adapter_validation_failure(
                    "tool.document.table.invalid_locator",
                    "document.table requires exactly one of table, table_index or node_id");
            }
            return bindings.document_table(arguments.dump());
        }, error);
    }
    return true;
}
