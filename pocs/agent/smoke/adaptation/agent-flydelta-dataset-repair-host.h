#pragma once

#include "agent-data-store-cozo.h"
#include "agent/runtime-json-contracts.h"
#include "agent/tooling/adapters/tool-adapters.h"
#include "agent/tooling/catalog/tool-catalog.h"
#include "agent/tooling/registry/tool-registry.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <string>

// Small host fixture shared by the dataset repair contract and model smokes.
// It deliberately uses the real native data adapters over a Cozo store: the
// model-facing expected plan is therefore verified by the same registry and
// schema path as a host turn, rather than by a hard-coded tool-name check.
class agent_flydelta_dataset_repair_host {
public:
    bool open(const std::string & fixture_id, std::string & error) {
        namespace fs = std::filesystem;
        root_ = fs::temp_directory_path() / ("llama-agent-flydelta-dataset-repair-" + fixture_id);
        std::error_code fs_error;
        fs::remove_all(root_, fs_error);
        fs::create_directories(root_, fs_error);
        if (fs_error) {
            error = "could not create dataset repair fixture directory";
            return false;
        }
        if (!store_.open((root_ / "data.cozo").string(), error)) return false;
        if (!seed_sales(error)) return false;

        common_tool_bootstrap_result bootstrap;
        if (!catalog_.bootstrap("analysis", bootstrap, error)) return false;
        common_native_tool_bindings bindings;
        bindings.data_store = &store_;
        common_tool_adapter_result adapters;
        return common_register_native_tool_adapters(
            catalog_, "analysis", bindings, registry_, adapters, error);
    }

    void close() {
        store_.close();
        std::error_code error;
        std::filesystem::remove_all(root_, error);
    }

    bool execute_step(const nlohmann::ordered_json & step,
            common_tool_execution_result & result, std::string & error) const {
        if (!step.is_object() || !step.contains("tool") || !step["tool"].is_string() ||
                !step.contains("args") || !step["args"].is_object()) {
            error = "dataset repair plan step must contain tool and object args";
            return false;
        }
        const common_agent_tool_call call{
            step["tool"].get<std::string>(), step["args"].dump()};
        const auto arguments = nlohmann::ordered_json::parse(
            call.arguments_json, nullptr, false);
        if (arguments.is_discarded() || !arguments.is_object()) {
            error = "dataset repair plan step arguments are not a JSON object";
            return false;
        }
        return execute_call(call.name, arguments, result, error);
    }

    bool execute_call(const std::string & name, const nlohmann::ordered_json & arguments,
            common_tool_execution_result & result, std::string & error) const {
        if (name.empty() || !arguments.is_object()) {
            error = "model tool call must contain a name and object arguments";
            return false;
        }
        nlohmann::ordered_json normalized;
        if (!normalize_call(name, arguments, normalized, error)) return false;
        return execute_normalized_call(name, normalized, result, error);
    }

    bool execute_normalized_call(const std::string & name,
            const nlohmann::ordered_json & normalized,
            common_tool_execution_result & result, std::string & error) const {
        if (name.empty() || !normalized.is_object()) {
            error = "normalized host tool call must contain a name and object arguments";
            return false;
        }
        result = registry_.execute({name, normalized.dump()});
        if (!result.ok) {
            error = result.failure_code + ": " + result.safe_summary;
            return false;
        }
        return true;
    }

    bool normalize_call(const std::string & name, const nlohmann::ordered_json & arguments,
            nlohmann::ordered_json & normalized, std::string & error) const {
        const common_agent_request & runtime_request = runtime_request_;
        bool defaults_applied = false;
        nlohmann::ordered_json runtime_normalized;
        if (!common_agent_runtime_apply_safe_tool_defaults_to_json(
                runtime_request, name, arguments, runtime_normalized,
                defaults_applied, error)) return false;
        std::string normalized_text;
        if (!registry_.normalize_and_validate(
                {name, runtime_normalized.dump()}, normalized_text, error)) {
            return false;
        }
        normalized = nlohmann::ordered_json::parse(normalized_text, nullptr, false);
        if (normalized.is_discarded() || !normalized.is_object()) {
            error = "host returned invalid normalized tool arguments";
            return false;
        }
        return true;
    }

    bool has_tool(const std::string & name) const { return registry_.contains(name); }

private:
    bool seed_sales(std::string & error) {
        const std::string dataset = "dataset://local/sales";
        if (!store_.put_row(dataset, "1", R"({"order_id":1,"customer_id":10,"region":"north","amount":12,"units":2})", error) ||
                !store_.put_row(dataset, "2", R"({"order_id":2,"customer_id":11,"region":"south","amount":8,"units":1})", error) ||
                !store_.put_row(dataset, "3", R"({"order_id":3,"customer_id":10,"region":"north","amount":4,"units":5})", error) ||
                !store_.put_row(dataset, "4", R"({"order_id":4,"customer_id":12,"region":"east","amount":21,"units":3})", error)) {
            return false;
        }
        common_agent_dataset_descriptor sales;
        sales.ref = {dataset, "Sales", 4, 5, "resource://fixture/sales", "fixture:rows"};
        sales.columns = {
            {"order_id", common_agent_dataset_column_type::integer, false},
            {"customer_id", common_agent_dataset_column_type::integer, false},
            {"region", common_agent_dataset_column_type::string, false},
            {"amount", common_agent_dataset_column_type::integer, false},
            {"units", common_agent_dataset_column_type::integer, false},
        };
        sales.source_workbook_name = "sales.xlsx";
        sales.source_sheet_name = "Sales";
        sales.source_sheet_index = 0;
        sales.import_processor_id = "flydelta-dataset-repair-fixture";
        if (!store_.put_dataset_descriptor(sales, error)) return false;
        runtime_request_.available_datasets = {sales};
        return true;
    }

    std::filesystem::path root_;
    common_agent_cozo_data_store store_;
    common_tool_catalog catalog_;
    common_tool_registry registry_;
    common_agent_request runtime_request_;
};
