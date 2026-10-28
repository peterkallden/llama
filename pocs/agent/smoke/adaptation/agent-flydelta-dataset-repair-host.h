#pragma once

#include "agent-data-store-cozo.h"
#include "agent/agent-inference.h"
#include "agent/runtime-json-contracts.h"
#include "agent/tooling/adapters/tool-adapters.h"
#include "agent/tooling/catalog/tool-catalog.h"
#include "agent/tooling/catalog/model-projection.h"
#include "agent/tooling/contracts/schema-contract.h"
#include "agent/tooling/registry/tool-registry.h"
#include "agent/tooling/schema/tool-output-codec.h"
#include "agent/tooling/schema/tool-schema-compact.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <string>
#include <vector>

inline constexpr const char * agent_flydelta_dataset_dsl_prefix = "open!";

// Same forced-context representation used by runtime teaching capture:
// preserve the original task prefix and evaluate a host-owned continuation
// without asking the model to generate or echo it again.
inline common_agent_generation_request agent_flydelta_dataset_capture_context(
        const common_agent_generation_request & prefix, const std::string & continuation) {
    auto request = prefix;
    request.options.n_predict = 0;
    request.flydelta_activation.reset();
    request.messages.push_back({"assistant", continuation});
    return request;
}

inline bool agent_flydelta_dataset_dsl_contrast(
        common_agent_teacher_forced_choice_request & request,
        const std::string & positive, const std::string & negative,
        std::string & error) {
    const std::string prefix = "open! ";
    if (positive.compare(0, prefix.size(), prefix) != 0 ||
            negative.compare(0, prefix.size(), prefix) != 0 ||
            positive.find_first_not_of(" \t\r\n", prefix.size()) == std::string::npos ||
            negative.find_first_not_of(" \t\r\n", prefix.size()) == std::string::npos) {
        error = "dataset repair scoring requires two complete compact DSL calls";
        return false;
    }
    // A trailing space merges with the next word in BPE tokenizers. Keep
    // it in the targets so the forced prefix ends at a stable token boundary.
    request.choice_prefix = agent_flydelta_dataset_dsl_prefix;
    request.positive_continuation = positive.substr(request.choice_prefix.size());
    request.negative_continuation = negative.substr(request.choice_prefix.size());
    return common_agent_teacher_forced_contrast_is_valid(
        request.choice_prefix, request.positive_continuation,
        request.negative_continuation, error);
}

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
        if (!common_register_native_tool_adapters(
                catalog_, "analysis", bindings, registry_, adapters, error)) return false;
        std::vector<std::string> names;
        for (const auto & definition : catalog_.load_profile("analysis", error)) {
            if (registry_.contains(definition.name)) names.push_back(definition.name);
        }
        return error.empty() && select_model_tools(names, error);
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
        const auto * definition = catalog_.find_definition(name);
        if (!definition || !registry_.contains(name)) {
            error = "dataset repair tool is not registered: " + name;
            return false;
        }
        auto resolved_arguments = arguments;
        const auto schema = nlohmann::ordered_json::parse(
            common_tool_model_input_schema(*definition), nullptr, false);
        if (!common_agent_resolve_tool_dataset_handles(
                runtime_request.available_datasets, schema, resolved_arguments, error)) return false;
        if (!common_agent_runtime_apply_safe_tool_defaults_to_json(
                runtime_request, name, resolved_arguments, runtime_normalized,
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

    bool select_model_tools(const std::vector<std::string> & names, std::string & error) {
        model_tools_.clear();
        for (const auto & name : names) {
            const auto * definition = catalog_.find_definition(name);
            if (!definition || !registry_.contains(name)) {
                error = "dataset repair model tool is not registered: " + name;
                return false;
            }
            model_tools_.push_back({definition->name, definition->description,
                common_tool_model_input_schema(*definition), common_tool_model_result_schema(*definition)});
        }
        error.clear();
        return true;
    }

    std::string render_projected_model_contract(const std::string & name, std::string & error) const {
        const auto * definition = catalog_.find_definition(name);
        if (!definition) { error = "unknown model contract: " + name; return {}; }
        auto schema = nlohmann::ordered_json::parse(common_tool_model_input_schema(*definition));
        if (!common_agent_project_dataset_model_schema(schema, runtime_request_.available_datasets,
                nlohmann::ordered_json::object(), error)) return {};
        // Match the production selector: examples must satisfy the final
        // host-projected schema rather than advertise unbound aliases.
        auto examples = nlohmann::ordered_json::array();
        for (const auto & example : schema.value("x-agent-examples", nlohmann::ordered_json::array())) {
            std::string normalized, example_error;
            if (common_schema_normalize_and_validate_object(example.dump(), schema.dump(), normalized, example_error))
                examples.push_back(example);
        }
        schema["x-agent-examples"] = std::move(examples);
        return common_render_compact_tool_description(definition->name, definition->description,
            schema.dump(), common_tool_model_result_schema(*definition), error);
    }

    bool parse_model_call(const std::string & text, common_agent_tool_call & call,
            std::string & error) const {
        return common_parse_model_tool_call(common_agent_tool_output_format::compact_dsl,
            text, model_tools_, call, error);
    }

    std::string render_model_call(const std::string & name,
            const nlohmann::ordered_json & arguments, std::string & error) const {
        const auto * definition = catalog_.find_definition(name);
        // An explicit rejected fixture may name an unknown tool. It still
        // needs a flat textual target, but never gains execution authority.
        const auto inferred_schema = [](const auto & self, const nlohmann::ordered_json & value)
                -> nlohmann::ordered_json {
            if (value.is_object()) {
                nlohmann::ordered_json properties = nlohmann::ordered_json::object();
                for (const auto & field : value.items()) properties[field.key()] = self(self, field.value());
                return {{"type", "object"}, {"properties", properties}};
            }
            if (value.is_array()) {
                nlohmann::ordered_json items = nlohmann::ordered_json::object();
                for (const auto & item : value) {
                    const auto next = self(self, item);
                    if (items.empty()) items = next;
                    else if (items.value("type", std::string{}) == "object" &&
                            next.value("type", std::string{}) == "object") {
                        items["properties"].update(next["properties"]);
                    }
                }
                return {{"type", "array"}, {"items", items}};
            }
            if (value.is_string()) return {{"type", "string"}};
            if (value.is_boolean()) return {{"type", "boolean"}};
            if (value.is_number()) return {{"type", "number"}};
            return nlohmann::ordered_json::object();
        };
        return common_render_flat_tool_call(name, definition
                ? common_tool_model_input_schema(*definition)
                : inferred_schema(inferred_schema, arguments).dump(),
            arguments.dump(), error);
    }

    const std::vector<common_agent_dataset_descriptor> & available_datasets() const {
        return runtime_request_.available_datasets;
    }

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
        sales.ref = {dataset, "Sales", 4, 5, "resource://fixture/sales.csv", "csv:dataset"};
        sales.columns = {
            {"order_id", common_agent_dataset_column_type::integer, false},
            {"customer_id", common_agent_dataset_column_type::integer, false},
            {"region", common_agent_dataset_column_type::string, false},
            {"amount", common_agent_dataset_column_type::integer, false},
            {"units", common_agent_dataset_column_type::integer, false},
        };
        sales.source_workbook_name = "sales.csv";
        sales.source_sheet_name = "Sales";
        sales.source_sheet_index = 0;
        sales.import_processor_id = "csv-resource-import-v1";
        sales.import_processor_version = "1";
        if (!store_.put_dataset_descriptor(sales, error)) return false;
        runtime_request_.available_datasets = {sales};
        return true;
    }

    std::filesystem::path root_;
    common_agent_cozo_data_store store_;
    common_tool_catalog catalog_;
    common_tool_registry registry_;
    std::vector<common_chat_tool> model_tools_;
    common_agent_request runtime_request_;
};
