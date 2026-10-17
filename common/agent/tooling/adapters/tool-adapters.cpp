#include "agent/tooling/adapters/tool-adapters.h"
#include "agent/tooling/adapters/families/data-adapters.h"
#include "agent/tooling/adapters/families/diagnostics-adapters.h"
#include "agent/tooling/adapters/families/document-adapters.h"
#include "agent/tooling/adapters/families/memory-adapters.h"
#include "agent/tooling/adapters/families/oracle-adapters.h"
#include "agent/tooling/adapters/families/repository-adapters.h"
#include "agent/tooling/adapters/families/resource-adapters.h"
#include "agent/artifact/artifact-contract.h"

#include "agent/tooling/contracts/tool-result-contracts.h"
#include "base64.hpp"
#include "http.h"
#include "memory/memory-retrieval.h"
#include "memory/memory-tool-service.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <regex>
#include <set>
#include <sstream>

#ifdef LLAMA_AGENT_WEB_USE_WINHTTP
#include <windows.h>
#include <winhttp.h>
#endif

using json = nlohmann::ordered_json;

namespace {

class calculator_parser {
public:
    explicit calculator_parser(const std::string & text) : text(text) {}

    bool parse(double & value, std::string & error) {
        value = expression(error);
        skip_space();
        if (!error.empty()) return false;
        if (pos != text.size() || !std::isfinite(value)) { error = "invalid bounded arithmetic expression"; return false; }
        return true;
    }

private:
    double expression(std::string & error) {
        double value = term(error);
        while (error.empty()) {
            skip_space();
            if (take('+')) value += term(error);
            else if (take('-')) value -= term(error);
            else break;
        }
        return value;
    }
    double term(std::string & error) {
        double value = factor(error);
        while (error.empty()) {
            skip_space();
            if (take('*')) value *= factor(error);
            else if (take('/')) {
                const double divisor = factor(error);
                if (error.empty() && divisor == 0.0) error = "division by zero";
                else value /= divisor;
            } else break;
        }
        return value;
    }
    double factor(std::string & error) {
        skip_space();
        if (take('+')) return factor(error);
        if (take('-')) return -factor(error);
        if (take('(')) {
            const double value = expression(error);
            skip_space();
            if (error.empty() && !take(')')) error = "missing closing parenthesis";
            return value;
        }
        const size_t begin = pos;
        bool digit = false;
        while (pos < text.size() && (std::isdigit((unsigned char) text[pos]) || text[pos] == '.')) { digit = digit || std::isdigit((unsigned char) text[pos]); ++pos; }
        if (!digit || pos - begin > 64) { error = "expected a number"; return 0.0; }
        try { return std::stod(text.substr(begin, pos - begin)); }
        catch (...) { error = "invalid number"; return 0.0; }
    }
    void skip_space() { while (pos < text.size() && std::isspace((unsigned char) text[pos])) ++pos; }
    bool take(char c) { if (pos < text.size() && text[pos] == c) { ++pos; return true; } return false; }
    const std::string & text;
    size_t pos = 0;
};

bool parse_object(const std::string & arguments_json, json & arguments, std::string & error) {
    arguments = json::parse(arguments_json, nullptr, false);
    if (!arguments.is_object()) { error = "tool arguments must be a JSON object"; return false; }
    return true;
}

common_tool_execution_result tool_success_json(const json & value) {
    return common_tool_execution_result::success(value.dump());
}

common_tool_execution_result tool_success_text(std::string value) {
    return common_tool_execution_result::success(std::move(value));
}

common_tool_execution_result tool_execution_failure(
    std::string code,
    std::string raw_diagnostic,
    std::string safe_summary);

common_tool_execution_result execute_data_backend(
        const common_native_tool_bindings & bindings,
        const std::string & operation,
        const std::string & input) {
    if (bindings.data_store == nullptr) {
        return tool_execution_failure(
            "tool.data.backend_unavailable",
            "structured data backend is unavailable",
            "The configured data backend is unavailable.");
    }
    std::string result;
    std::string error;
    if (!bindings.data_store->execute(operation, input, result, error)) {
        return tool_execution_failure(
            "tool.data.backend_failed",
            error.empty() ? "structured data backend failed" : std::move(error),
            "The structured data operation failed.");
    }
    const auto parsed = json::parse(result, nullptr, false);
    if (!parsed.is_object() && !parsed.is_array()) {
        return tool_execution_failure(
            "tool.data.invalid_backend_result",
            "structured data backend returned invalid JSON",
            "The structured data backend returned an invalid result.");
    }
    return tool_success_json(parsed);
}

common_runtime_resource_metadata make_tool_resource_metadata(
        std::string purpose,
        std::string content_summary,
        std::string usage_hint,
        std::string limitations,
        std::vector<std::string> keywords = {},
        std::vector<std::string> entities = {}) {
    common_runtime_resource_metadata metadata;
    metadata.purpose = std::move(purpose);
    metadata.content_summary = std::move(content_summary);
    metadata.usage_hint = std::move(usage_hint);
    metadata.limitations = std::move(limitations);
    metadata.keywords = std::move(keywords);
    metadata.entities = std::move(entities);
    return metadata;
}

bool persist_tool_resource(
        const common_native_tool_bindings & bindings,
        const std::string & name,
        const std::string & description,
        const std::string & mime_type,
        const std::string & text,
        const std::string & source_tool,
        const common_runtime_resource_metadata & metadata,
        common_runtime_resource_ref & resource,
        std::string & error,
        common_runtime_resource_lineage lineage = {}) {
    if (bindings.resource_runtime.store == nullptr) {
        error = "resource store is unavailable";
        return false;
    }

    agent_resource_put_request request;
    request.name = name;
    request.description = description;
    request.mime_type = mime_type;
    request.text = text;
    request.scope = common_runtime_resource_scope::turn;
    request.source_provider = "native";
    request.source_tool = source_tool;
    request.metadata = metadata;
    request.lineage = std::move(lineage);
    apply_agent_resource_runtime(bindings.resource_runtime, request);

    agent_resource_descriptor descriptor;
    if (!bindings.resource_runtime.store->put_text(request, descriptor, error)) {
        return false;
    }

    resource = descriptor;
    error.clear();
    return true;
}

bool persist_tool_bytes(
        const common_native_tool_bindings & bindings,
        const std::string & name,
        const std::string & description,
        const std::string & mime_type,
        const std::string & bytes,
        const std::string & source_tool,
        const common_runtime_resource_metadata & metadata,
        common_runtime_resource_ref & resource,
        std::string & error,
        common_runtime_resource_lineage lineage = {}) {
    if (bindings.resource_runtime.store == nullptr) {
        error = "resource store is unavailable";
        return false;
    }

    agent_resource_put_request request;
    request.name = name;
    request.description = description;
    request.mime_type = mime_type;
    request.bytes = bytes;
    request.bytes_are_authoritative = true;
    request.scope = common_runtime_resource_scope::turn;
    request.source_provider = "native";
    request.source_tool = source_tool;
    request.metadata = metadata;
    request.lineage = std::move(lineage);
    apply_agent_resource_runtime(bindings.resource_runtime, request);

    agent_resource_descriptor descriptor;
    if (!bindings.resource_runtime.store->put_bytes(request, descriptor, error)) {
        return false;
    }

    resource = descriptor;
    error.clear();
    return true;
}

agent_resource_read_authority make_resource_read_authority(
        const common_native_tool_bindings & bindings) {
    return make_agent_resource_read_authority(bindings.resource_runtime, std::time(nullptr));
}

common_tool_execution_result tool_failure(std::string code, common_tool_failure_class failure_class, bool retryable,
        std::string safe_summary, std::string raw_diagnostic) {
    return common_tool_execution_result::failure(std::move(code), failure_class, retryable, std::move(safe_summary), std::move(raw_diagnostic));
}

common_tool_execution_result tool_validation_failure(std::string code, std::string raw_diagnostic, std::string safe_summary = "Tool arguments are invalid.") {
    return tool_failure(std::move(code), common_tool_failure_class::validation, false, std::move(safe_summary), std::move(raw_diagnostic));
}

common_tool_execution_result tool_execution_failure(std::string code, std::string raw_diagnostic, std::string safe_summary) {
    return tool_failure(std::move(code), common_tool_failure_class::execution, false, std::move(safe_summary), std::move(raw_diagnostic));
}

common_tool_execution_result tool_not_found_failure(std::string code, std::string raw_diagnostic, std::string safe_summary) {
    return tool_failure(std::move(code), common_tool_failure_class::not_found, false, std::move(safe_summary), std::move(raw_diagnostic));
}

common_tool_execution_result tool_network_failure(std::string code, std::string raw_diagnostic, std::string safe_summary, bool retryable = true) {
    return tool_failure(std::move(code), common_tool_failure_class::network, retryable, std::move(safe_summary), std::move(raw_diagnostic));
}

common_tool_execution_result tool_limit_failure(std::string code, std::string raw_diagnostic, std::string safe_summary) {
    return tool_failure(std::move(code), common_tool_failure_class::limit, false, std::move(safe_summary), std::move(raw_diagnostic));
}

std::string utc_now() {
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &time);
#else
    gmtime_r(&time, &utc);
#endif
    std::ostringstream out;
    out << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

bool register_definition(const common_tool_definition & definition, common_tool_registry & registry,
        std::function<common_tool_execution_result(const std::string &)> handler, std::string & error, bool read_only = true, bool policy_gated = false) {
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

bool make_sandbox_request(
        const std::string & tool_name,
        const std::string & input,
        common_agent_sandbox_request & request,
        std::string & error) {
    json arguments;
    if (!parse_object(input, arguments, error) || !arguments.contains("target") || !arguments["target"].is_string()) {
        if (error.empty()) error = tool_name + " requires a target";
        return false;
    }
    const auto target = arguments["target"].get<std::string>();
    if (target.empty() || target.size() > 256) {
        error = tool_name + " target is out of bounds";
        return false;
    }
    request = {};
    const bool is_test = tool_name == "development.test";
    request.operation_id = tool_name + "/" + target;
    request.execution_class = "developer-build";
    request.command.program = is_test ? "agent.development.test" : "agent.development.build";
    request.command.arguments.push_back(target);
    if (arguments.contains("configuration")) {
        if (!arguments["configuration"].is_string()) { error = tool_name + " configuration must be a string"; return false; }
        request.command.arguments.push_back(arguments["configuration"].get<std::string>());
    }
    if (is_test && arguments.contains("filter")) {
        if (!arguments["filter"].is_string()) { error = "development.test filter must be a string"; return false; }
        request.command.arguments.push_back(arguments["filter"].get<std::string>());
    }
    request.limits.timeout_ms = arguments.value("timeout_ms", 120000u);
    request.limits.cpu_count = 1;
    request.limits.max_output_bytes = 65536;
    request.network = common_agent_sandbox_network_scope::none;
    request.filesystem = common_agent_sandbox_filesystem_scope::workspace_write;
    request.artifacts.collect = true;
    if (arguments.contains("resource_refs")) {
        if (!arguments["resource_refs"].is_array() || arguments["resource_refs"].size() > 32) {
            error = tool_name + " resource_refs must be an array with at most 32 entries";
            return false;
        }
        for (const auto & value : arguments["resource_refs"]) {
            if (!value.is_string() || value.get<std::string>().empty()) {
                error = tool_name + " resource_refs entries must be non-empty strings";
                return false;
            }
            common_runtime_resource_ref resource;
            resource.uri = value.get<std::string>();
            request.workspace.input_resources.push_back(std::move(resource));
        }
    }
    error.clear();
    return true;
}

std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return value;
}

std::string trim_copy(const std::string & value) {
    size_t begin = 0, end = value.size();
    while (begin < end && std::isspace((unsigned char) value[begin])) ++begin;
    while (end > begin && std::isspace((unsigned char) value[end - 1])) --end;
    return value.substr(begin, end - begin);
}

bool valid_accept_header(const std::string & value) {
    if (value.empty() || value.size() > 512) return false;
    return std::all_of(value.begin(), value.end(), [](unsigned char c) {
        // Accept is the only caller-controlled request header. Keep it printable
        // ASCII so it cannot inject another HTTP header or a request line.
        return c >= 0x20 && c <= 0x7e;
    });
}

std::string csv_escape(const json & value) {
    if (value.is_null()) return {};
    std::string text = value.is_string() ? value.get<std::string>() : value.dump();
    const bool quote = text.find_first_of(",\"\r\n") != std::string::npos;
    if (!quote) return text;
    std::string escaped = "\"";
    for (const char c : text) {
        if (c == '\"') escaped += "\"\"";
        else escaped += c;
    }
    escaped += '"';
    return escaped;
}

common_tool_execution_result export_dataset_csv(
        const common_native_tool_bindings & bindings,
        const json & arguments) {
    if (!bindings.data_store) {
        return tool_execution_failure(
            "tool.artifact.export.backend_unavailable",
            "dataset export requires a structured data backend",
            "The dataset backend is unavailable.");
    }
    if (!arguments.contains("source_dataset") || !arguments["source_dataset"].is_string() ||
            arguments["source_dataset"].get<std::string>().empty()) {
        return tool_validation_failure(
            "tool.artifact.export.invalid_dataset",
            "source_dataset must be a non-empty dataset URI");
    }
    const auto format = lower_copy(arguments.value("format", std::string("csv")));
    if (format != "csv") {
        return tool_validation_failure(
            "tool.artifact.export.unsupported_format",
            "dataset artifact export currently supports only csv");
    }
    const auto name = arguments.value("name", std::string("dataset-export.csv"));
    const int max_rows = arguments.value("max_rows", 10000);
    if (name.empty() || name.size() > 256 || max_rows < 1 || max_rows > 10000) {
        return tool_validation_failure(
            "tool.artifact.export.out_of_bounds",
            "dataset CSV export name or max_rows is out of bounds");
    }

    const auto dataset_uri = arguments["source_dataset"].get<std::string>();
    common_agent_dataset_descriptor descriptor;
    std::string error;
    if (!bindings.data_store->get_dataset_descriptor(dataset_uri, descriptor, error)) {
        return tool_not_found_failure(
            "tool.artifact.export.dataset_unavailable", std::move(error),
            "The source dataset is unavailable.");
    }
    if (descriptor.ref.row_count > static_cast<size_t>(max_rows)) {
        return tool_limit_failure(
            "tool.artifact.export.row_limit",
            "dataset contains more rows than the bounded CSV export limit",
            "The dataset is too large for this bounded export.");
    }

    json query = {
        {"dataset", dataset_uri}, {"limit", max_rows},
        {"max_scan_rows", max_rows}, {"max_result_rows", max_rows}};
    const auto queried = execute_data_backend(bindings, "data.query", query.dump());
    if (!queried.ok) return queried;
    const auto result = json::parse(queried.output, nullptr, false);
    if (!result.is_object() || result.value("scan_truncated", false) ||
            result.value("result_truncated", false) || !result.contains("rows") ||
            !result["rows"].is_array()) {
        return tool_limit_failure(
            "tool.artifact.export.incomplete_dataset",
            "dataset query was incomplete or did not return rows",
            "The dataset could not be exported completely.");
    }

    std::vector<std::string> columns;
    for (const auto & column : descriptor.columns) columns.push_back(column.name);
    if (columns.empty() && result.contains("columns") && result["columns"].is_array()) {
        for (const auto & column : result["columns"]) if (column.is_string()) columns.push_back(column.get<std::string>());
    }
    if (columns.size() > 512) {
        return tool_limit_failure(
            "tool.artifact.export.column_limit",
            "dataset contains more than 512 columns",
            "The dataset has too many columns for this export.");
    }
    std::string csv;
    for (size_t index = 0; index < columns.size(); ++index) {
        if (index) csv += ',';
        csv += csv_escape(columns[index]);
    }
    csv += '\n';
    for (const auto & row : result["rows"]) {
        for (size_t index = 0; index < columns.size(); ++index) {
            if (index) csv += ',';
            csv += row.is_object() ? csv_escape(row.value(columns[index], json())) : std::string();
        }
        csv += '\n';
        if (csv.size() > 65536) {
            return tool_limit_failure(
                "tool.artifact.export.byte_limit",
                "dataset CSV export exceeded the bounded artifact size",
                "The dataset is too large for this bounded export.");
        }
    }

    common_runtime_resource_ref resource;
    common_runtime_resource_lineage lineage;
    lineage.parent_uri = dataset_uri;
    lineage.chunk_count = 1;
    lineage.byte_length = csv.size();
    lineage.derivation = "artifact.export:dataset-csv";
    if (!persist_tool_resource(
            bindings, name, "CSV artifact exported from a dataset", "text/csv", csv,
            "artifact.export",
            make_tool_resource_metadata(
                "Provide a bounded CSV artifact derived from a structured dataset.",
                "CSV export of dataset " + dataset_uri + ".",
                "Use the artifact resource for download or bounded resource reads.",
                "The export is bounded by row, column and byte limits; the source dataset remains authoritative."),
            resource, error, std::move(lineage))) {
        return tool_execution_failure("tool.artifact.export.store_failed", std::move(error), "The CSV artifact could not be stored.");
    }
    return common_tool_execution_result::success(
        json({{"resource", resource.uri}, {"name", resource.name}, {"mime_type", resource.mime_type},
              {"source_dataset", dataset_uri}, {"rows", result["rows"].size()}, {"columns", columns.size()}}).dump(),
        "Exported a bounded CSV artifact from the dataset.", {std::move(resource)});
}

common_tool_execution_result export_saved_document(
        const common_native_tool_bindings & bindings, const json & arguments) {
    const auto uri=arguments.value("source_document",std::string{});
    const auto format=arguments.value("format",std::string("markdown"));
    if(uri.empty()||(format!="markdown"&&format!="docx"))
        return tool_validation_failure("tool.artifact.export.invalid_document_export","source_document requires a saved document resource and format markdown or docx");
    if(bindings.resource_runtime.store==nullptr)
        return tool_execution_failure("tool.artifact.export.store_unavailable","resource store is unavailable","Document export is unavailable.");
    const auto authority=make_agent_resource_read_authority(bindings.resource_runtime,static_cast<int64_t>(std::time(nullptr)));
    agent_resource_descriptor report_descriptor;std::string error;
    if(!bindings.resource_runtime.store->stat(uri,authority,report_descriptor,error)||
            (report_descriptor.mime_type!="application/vnd.llama-agent.report+json"&&report_descriptor.mime_type!="application/vnd.llama-agent.document+json"))
        return tool_not_found_failure("tool.artifact.export.document_unavailable",std::move(error),"The saved document revision is not available in this scope.");
    std::string serialized;
    if(!bindings.resource_runtime.store->read_text(uri,authority,4*1024*1024,serialized,error))
        return tool_execution_failure("tool.artifact.export.document_read_failed",std::move(error),"The saved report could not be read.");
    common_agent_artifact_document document;
    if(!common_agent_artifact_document_from_json(serialized,document,error))
        return tool_validation_failure("tool.artifact.export.document_invalid",std::move(error));
    std::string markdown;
    if(!common_agent_artifact_render_markdown(document,markdown,error))
        return tool_execution_failure("tool.artifact.export.document_render_failed",std::move(error),"The saved report could not be rendered.");
    std::string editable;
    if(!common_agent_artifact_render_editable_markdown(document,editable,error))
        return tool_execution_failure("tool.artifact.export.document_render_failed",std::move(error),"The saved report could not be rendered.");
    const auto markdown_key="document-export:"+document.id+":r"+std::to_string(document.revision)+":markdown";
    common_runtime_resource_ref markdown_resource;
    bool have_markdown=false;
    if(!document.editor_markdown_uri.empty()){
        agent_resource_descriptor descriptor;
        if(bindings.resource_runtime.store->stat(document.editor_markdown_uri,authority,descriptor,error)&&descriptor.mime_type=="text/markdown"){
            std::string stored;if(bindings.resource_runtime.store->read_text(descriptor.uri,authority,4*1024*1024,stored,error)&&stored==markdown){
                markdown_resource.uri=descriptor.uri;markdown_resource.name=descriptor.name;markdown_resource.mime_type=descriptor.mime_type;
                markdown_resource.size_bytes=descriptor.size_bytes;markdown_resource.scope=descriptor.scope;markdown_resource.metadata=descriptor.metadata;markdown_resource.lineage=descriptor.lineage;have_markdown=true;
            }
        }
        error.clear();
    }
    if(!have_markdown){
        std::vector<agent_resource_descriptor> resources;
        if(!bindings.resource_runtime.store->list(authority,resources,error))return tool_execution_failure("tool.artifact.export.document_index_failed",std::move(error),"The document export index could not be read.");
        for(const auto & resource:resources)if(resource.metadata.processing_cache_key==markdown_key){
            std::string stored;if(resource.mime_type!="text/markdown"||!bindings.resource_runtime.store->read_text(resource.uri,authority,4*1024*1024,stored,error)||stored!=markdown)
                return tool_execution_failure("tool.artifact.export.document_cache_invalid","cached Markdown differs from the immutable report revision","The document export cache could not be verified.");
            markdown_resource.uri=resource.uri;markdown_resource.name=resource.name;markdown_resource.mime_type=resource.mime_type;markdown_resource.size_bytes=resource.size_bytes;markdown_resource.scope=resource.scope;markdown_resource.metadata=resource.metadata;markdown_resource.lineage=resource.lineage;have_markdown=true;break;
        }
        if(!have_markdown){
            agent_resource_put_request put;put.name=document.id+"-r"+std::to_string(document.revision)+".md";put.description=document.title;put.mime_type="text/markdown";put.text=markdown;
            put.scope=bindings.resource_runtime.project_id.empty()?common_runtime_resource_scope::session:common_runtime_resource_scope::project;
            put.source_provider="native";put.source_tool="artifact.export";put.metadata=make_tool_resource_metadata("Editable Markdown projection of a saved report revision.",document.title,"Use this resource as the Markdown export.","The structured report revision remains authoritative.");put.metadata.processing_cache_key=markdown_key;
            put.lineage.parent_uri=uri;put.lineage.chunk_count=1;put.lineage.byte_length=markdown.size();put.lineage.derivation="artifact.export:document-markdown";
            apply_agent_resource_runtime(bindings.resource_runtime,put);agent_resource_descriptor descriptor;
            if(!bindings.resource_runtime.store->put_text(put,descriptor,error))return tool_execution_failure("tool.artifact.export.markdown_store_failed",std::move(error),"The Markdown export could not be stored.");
            markdown_resource.uri=descriptor.uri;markdown_resource.name=descriptor.name;markdown_resource.mime_type=descriptor.mime_type;markdown_resource.size_bytes=descriptor.size_bytes;markdown_resource.scope=descriptor.scope;markdown_resource.metadata=descriptor.metadata;markdown_resource.lineage=descriptor.lineage;
        }
    }
    common_runtime_resource_ref final_resource=markdown_resource;
    if(format=="docx"){
        const auto cache_key="document-export:"+document.id+":r"+std::to_string(document.revision)+":docx";
        std::vector<agent_resource_descriptor> resources;
        if(!bindings.resource_runtime.store->list(authority,resources,error))return tool_execution_failure("tool.artifact.export.document_index_failed",std::move(error),"The document export index could not be read.");
        bool cached=false;
        for(const auto & resource:resources)if(resource.metadata.processing_cache_key==cache_key){
            if(resource.mime_type!="application/vnd.openxmlformats-officedocument.wordprocessingml.document"||resource.size_bytes==0)return tool_execution_failure("tool.artifact.export.docx_cache_invalid","cached DOCX has an invalid media type or empty content","The DOCX export cache could not be verified.");
            final_resource.uri=resource.uri;final_resource.name=resource.name;final_resource.mime_type=resource.mime_type;final_resource.size_bytes=resource.size_bytes;final_resource.scope=resource.scope;final_resource.metadata=resource.metadata;final_resource.lineage=resource.lineage;cached=true;break;
        }
        if(!cached){
            if(bindings.resource_processing_service==nullptr&&!bindings.resource_processing_provider_factory)return tool_execution_failure("tool.artifact.export.docx_unavailable","resource processing is unavailable","DOCX export is unavailable.");
            agent_resource_processing_binding_request request;request.source_uri=markdown_resource.uri;request.operation_id=cache_key;request.authority=authority;request.media_type.declared_type="text/markdown";request.media_type.resolved_type="text/markdown";request.target_representation="docx";request.target_media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document";request.purpose=agent_resource_processing_purpose::artifact_generation;request.limits.max_source_bytes=4*1024*1024;request.limits.max_output_bytes=8*1024*1024;request.limits.max_duration_ms=180000;request.limits.max_generated_resources=1;
            std::shared_ptr<agent_resource_processing_provider> owned;auto * processor=bindings.resource_processing_service;
            if(bindings.resource_processing_provider_factory){owned=bindings.resource_processing_provider_factory(request);processor=owned.get();}
            if(processor==nullptr)return tool_execution_failure("tool.artifact.export.docx_unavailable","resource processor provider could not be created","DOCX export is unavailable.");
            const auto processed=processor->process(request);
            if(!processed.success||processed.outputs.size()!=1||processed.outputs.front().bytes.empty()||common_normalize_resource_media_type(processed.outputs.front().mime_type)!=common_normalize_resource_media_type(request.target_media_type))
                return tool_execution_failure("tool.artifact.export.docx_failed",processed.failure_code+": "+processed.safe_summary,"The report could not be converted to DOCX.");
            agent_resource_put_request put;put.name=document.id+"-r"+std::to_string(document.revision)+".docx";put.description=document.title;put.mime_type=request.target_media_type;put.bytes=processed.outputs.front().bytes;put.bytes_are_authoritative=true;
            put.scope=bindings.resource_runtime.project_id.empty()?common_runtime_resource_scope::session:common_runtime_resource_scope::project;put.source_provider="native";put.source_tool="artifact.export";put.metadata=processed.outputs.front().metadata;put.metadata.processing_cache_key=cache_key;put.lineage.parent_uri=markdown_resource.uri;put.lineage.chunk_count=1;put.lineage.byte_length=put.bytes.size();put.lineage.derivation="artifact.export:document-docx";
            apply_agent_resource_runtime(bindings.resource_runtime,put);agent_resource_descriptor descriptor;
            if(!bindings.resource_runtime.store->put_bytes(put,descriptor,error))return tool_execution_failure("tool.artifact.export.docx_store_failed",std::move(error),"The DOCX export could not be stored.");
            final_resource.uri=descriptor.uri;final_resource.name=descriptor.name;final_resource.mime_type=descriptor.mime_type;final_resource.size_bytes=descriptor.size_bytes;final_resource.scope=descriptor.scope;final_resource.metadata=descriptor.metadata;final_resource.lineage=descriptor.lineage;
        }
    }
    json result_json{{"resource",final_resource.uri},{"document_resource",uri},{"markdown_resource",markdown_resource.uri},
        {"document_id",document.id},{"document_kind",document.kind==common_agent_artifact_document_kind::document?"document":"report"},{"revision",document.revision},{"content_fingerprint",document.content_fingerprint},
        {"previous_revision_uri",document.previous_revision_uri},{"format",format}};
    if(document.kind==common_agent_artifact_document_kind::report)result_json["report_resource"]=uri;
    const auto result=result_json.dump();
    std::vector<common_runtime_resource_ref> refs{final_resource};if(final_resource.uri!=markdown_resource.uri)refs.push_back(markdown_resource);refs.push_back(report_descriptor);
    return common_tool_execution_result::success(result,"Exported the selected immutable report revision.",std::move(refs));
}

common_tool_execution_result export_plan_report(
        const common_native_tool_bindings & bindings, const json & arguments) {
    if (bindings.plan_store == nullptr || bindings.plan_id == nullptr || bindings.plan_id->empty())
        return tool_validation_failure("tool.artifact.export.no_active_plan", "report export requires a host-bound active plan");
    const std::string requested_plan = arguments.value("source_plan", *bindings.plan_id);
    if (requested_plan != *bindings.plan_id)
        return tool_validation_failure("tool.artifact.export.plan_scope", "source_plan must match the active host-bound plan");
    const auto format=arguments.value("format",std::string("markdown"));
    if(format!="markdown"&&format!="docx")return tool_validation_failure("tool.artifact.export.unsupported_format","report export supports markdown or docx");
    std::string error;
    auto plan = bindings.plan_store->get(requested_plan, error);
    if (!plan) return tool_not_found_failure("tool.artifact.export.plan_unavailable", std::move(error), "The source plan is unavailable.");
    if (plan->namespace_id != bindings.resource_runtime.namespace_id ||
            (!plan->project_id.empty() && plan->project_id != bindings.resource_runtime.project_id) ||
            (plan->project_id.empty() && plan->session_id != bindings.resource_runtime.session_id))
        return tool_execution_failure("tool.artifact.export.plan_scope", "source plan is outside the active resource scope", "The source plan is not available in this scope.");
    if (!plan->workflow_definition)
        return tool_validation_failure("tool.artifact.export.no_report_profile", "the active plan has no declared artifact profile");
    const common_plan_workflow_transition * profile_transition = nullptr;
    for (const auto & transition : plan->workflow_definition->transitions) {
        if (transition.kind != common_plan_workflow_transition_kind::artifact_compose) continue;
        if (profile_transition != nullptr && (profile_transition->artifact_profile_ref != transition.artifact_profile_ref ||
                profile_transition->artifact_profile_revision != transition.artifact_profile_revision ||
                profile_transition->artifact_profile_json != transition.artifact_profile_json))
            return tool_validation_failure("tool.artifact.export.multiple_profiles", "one report export cannot combine different artifact profiles");
        profile_transition = &transition;
    }
    if (profile_transition == nullptr)
        return tool_validation_failure("tool.artifact.export.no_report_profile", "the active plan has no declared artifact profile");
    common_agent_artifact_profile profile;
    if (!common_agent_artifact_profile_from_json(profile_transition->artifact_profile_json, profile, error))
        return tool_execution_failure("tool.artifact.export.invalid_profile", std::move(error), "The report profile could not be validated.");

    common_agent_artifact_snapshot snapshot;
    if (!common_agent_artifact_snapshot_from_plan(*plan, profile, snapshot, error))
        return tool_execution_failure("tool.artifact.export.snapshot_failed", std::move(error), "The report source snapshot could not be created.");
    std::string snapshot_json;
    if (!common_agent_artifact_snapshot_to_json(snapshot, snapshot_json, error))
        return tool_execution_failure("tool.artifact.export.snapshot_failed", std::move(error), "The report source snapshot could not be serialized.");
    const auto base_name=arguments.value("name",profile.id+"-report");
    const auto report_id=arguments.value("report_id",std::string("report:")+plan->id);
    if(base_name.empty()||base_name.size()>256||report_id.empty()||report_id.size()>256)
        return tool_validation_failure("tool.artifact.export.invalid_name","report name or identity is outside bounds");
    // Validate the complete document and deterministic Markdown before the
    // first resource write. The final pass substitutes the stored snapshot URI.
    common_agent_artifact_document document;
    if(!common_agent_artifact_document_from_plan(*plan,profile,snapshot.id,document,error,report_id))
        return tool_execution_failure("tool.artifact.export.document_failed",std::move(error),"The report could not be assembled from validated sections.");
    // Manifests are the durable revision index. This also works when the next
    // explicit composition runs in a new plan but continues the same report ID.
    if(bindings.resource_runtime.store==nullptr)return tool_execution_failure("tool.artifact.export.store_unavailable","resource store is unavailable","Report export is unavailable.");
    const auto authority=make_agent_resource_read_authority(bindings.resource_runtime,static_cast<int64_t>(std::time(nullptr)));
    std::vector<agent_resource_descriptor> existing_resources;
    if(!bindings.resource_runtime.store->list(authority,existing_resources,error))
        return tool_execution_failure("tool.artifact.export.revision_index_failed",std::move(error),"The existing report revision index could not be read safely.");
    const auto manifest_prefix="artifact-export:"+report_id+":r";
    uint64_t latest_revision=0;std::string latest_report_uri;bool same_snapshot=false;
    std::optional<nlohmann::json> verified_same_snapshot_manifest;
    std::optional<std::string> verified_document_json;
    std::string verified_same_snapshot_manifest_uri;
    for(const auto & candidate:existing_resources){const auto & key=candidate.metadata.processing_cache_key;
        if(key.rfind(manifest_prefix,0)!=0||key.size()<9||key.substr(key.size()-9)!=":manifest")continue;
        std::string manifest_text;
        if(!bindings.resource_runtime.store->read_text(candidate.uri,authority,1024*1024,manifest_text,error))
            return tool_execution_failure("tool.artifact.export.manifest_read_failed",std::move(error),"An existing report manifest could not be verified.");
        const auto manifest=nlohmann::json::parse(manifest_text,nullptr,false);
        if(!manifest.is_object()||manifest.value("schema_version",0)!=1||manifest.value("report_id",std::string{})!=report_id)continue;
        const auto prior_revision=manifest.value("revision",uint64_t(0));
        const auto prior_export_id=manifest.value("export_id",std::string{});
        const auto prior_format=manifest.value("format",std::string{});
        const auto prior_snapshot_uri=manifest.value("snapshot_resource",std::string{});
        const auto prior_report_uri=manifest.value("report_resource",std::string{});
        const auto prior_markdown_uri=manifest.value("markdown_resource",std::string{});
        const auto prior_final_uri=manifest.value("resource",std::string{});
        if(prior_revision==0||prior_export_id.empty()||(prior_format!="markdown"&&prior_format!="docx")||
                prior_snapshot_uri.empty()||prior_report_uri.empty()||prior_markdown_uri.empty()||prior_final_uri.empty()||
                key!="artifact-export:"+prior_export_id+":manifest")continue;
        agent_resource_descriptor prior_snapshot_descriptor,prior_report_descriptor,prior_markdown_descriptor,prior_final_descriptor;
        if(!bindings.resource_runtime.store->stat(prior_snapshot_uri,authority,prior_snapshot_descriptor,error)||
                !bindings.resource_runtime.store->stat(prior_report_uri,authority,prior_report_descriptor,error)||
                !bindings.resource_runtime.store->stat(prior_markdown_uri,authority,prior_markdown_descriptor,error)||
                !bindings.resource_runtime.store->stat(prior_final_uri,authority,prior_final_descriptor,error)){
            error.clear();continue;
        }
        std::string prior_snapshot_text,prior_report_text,prior_markdown_text;
        if(!bindings.resource_runtime.store->read_text(prior_snapshot_uri,authority,4*1024*1024,prior_snapshot_text,error)||
                !bindings.resource_runtime.store->read_text(prior_report_uri,authority,4*1024*1024,prior_report_text,error)||
                !bindings.resource_runtime.store->read_text(prior_markdown_uri,authority,4*1024*1024,prior_markdown_text,error)){
            error.clear();continue;
        }
        common_agent_artifact_snapshot prior_snapshot;common_agent_artifact_document prior_document;std::string validation_error;
        if(!common_agent_artifact_snapshot_from_json(prior_snapshot_text,prior_snapshot,validation_error)||
                !common_agent_artifact_document_from_json(prior_report_text,prior_document,validation_error)||
                prior_snapshot.fingerprint!=manifest.value("snapshot_fingerprint",std::string{})||
                prior_document.id!=report_id||prior_document.revision!=prior_revision||
                prior_document.snapshot_fingerprint!=prior_snapshot.fingerprint||prior_document.snapshot_uri!=prior_snapshot_uri||
                prior_document.profile_id!=manifest.value("profile_id",std::string{})||
                prior_document.profile_revision!=manifest.value("profile_revision",std::string{}))continue;
        std::string expected_prior_markdown;
        if(!common_agent_artifact_render_markdown(prior_document,expected_prior_markdown,validation_error)||
                expected_prior_markdown!=prior_markdown_text||
                prior_markdown_descriptor.mime_type!="text/markdown"||
                (prior_format=="markdown"&&prior_final_uri!=prior_markdown_uri)||
                (prior_format=="docx"&&prior_final_descriptor.mime_type!="application/vnd.openxmlformats-officedocument.wordprocessingml.document"))continue;
        if(prior_format=="docx"){
            std::string prior_docx;
            if(!bindings.resource_runtime.store->read_bytes(prior_final_uri,authority,8*1024*1024,prior_docx,error)||prior_docx.empty()){
                error.clear();continue;
            }
        }
        if(prior_revision>latest_revision){latest_revision=prior_revision;latest_report_uri=prior_report_uri;}
        if(manifest.value("snapshot_fingerprint",std::string{})==snapshot.fingerprint){
            // A format change exports the published immutable document,
            // including its renderer version and original bibliography.
            same_snapshot=true;document=prior_document;verified_document_json=prior_report_text;
            if(prior_format==format){verified_same_snapshot_manifest=manifest;verified_same_snapshot_manifest_uri=candidate.uri;}
        }
    }
    uint64_t current_revision=0;std::string current_document_uri;
    if(!common_adapter_document_current_revision(bindings,report_id,current_document_uri,current_revision,error))
        return tool_execution_failure("tool.artifact.export.revision_index_failed",std::move(error),"The current edited report revision could not be verified.");
    if(current_revision>latest_revision){latest_revision=current_revision;latest_report_uri=current_document_uri;}
    if(!same_snapshot&&latest_revision>0){document.revision=latest_revision+1;document.previous_revision_uri=latest_report_uri;}
    std::string document_json,markdown;
    if(!common_agent_artifact_document_to_json(document,document_json,error)||!common_agent_artifact_render_markdown(document,markdown,error))
        return tool_execution_failure("tool.artifact.export.render_failed",std::move(error),"The report could not be rendered.");
    if(verified_document_json)document_json=*verified_document_json;
    const auto content_export_id=report_id+":r"+std::to_string(document.revision)+":"+snapshot.fingerprint;
    const auto export_id=content_export_id+":"+format;
    auto cache_key=[&](const std::string & part){
        return part=="docx"||part=="manifest" ? "artifact-export:"+export_id+":"+part : "artifact-content:"+content_export_id+":"+part;
    };
    auto find_cached=[&](const std::string & key,const std::string & mime,const std::string & expected,
            common_runtime_resource_ref & out,bool & found)->bool{
        found=false;std::vector<agent_resource_descriptor> descriptors;
        if(bindings.resource_runtime.store==nullptr){error="resource store is unavailable";return false;}
        if(!bindings.resource_runtime.store->list(authority,descriptors,error))return false;
        for(const auto & descriptor:descriptors){if(descriptor.metadata.processing_cache_key!=key)continue;
            if(descriptor.mime_type!=mime){error="cached artifact key has a conflicting media type";return false;}
            std::string existing;
            if(!bindings.resource_runtime.store->read_bytes(descriptor.uri,authority,expected.size()+1,existing,error))return false;
            if(existing!=expected){error="cached artifact key content does not match the frozen report";return false;}
            out=descriptor;found=true;error.clear();return true;
        }
        error.clear();return true;
    };
    if(verified_same_snapshot_manifest&&
            verified_same_snapshot_manifest->value("export_id",std::string{})==export_id){
        std::vector<common_runtime_resource_ref> refs;
        agent_resource_descriptor final_descriptor,report_descriptor,snapshot_descriptor,markdown_descriptor,manifest_descriptor;
        const auto & manifest=*verified_same_snapshot_manifest;
        if(!bindings.resource_runtime.store->stat(manifest.value("resource",std::string{}),authority,final_descriptor,error)||
                !bindings.resource_runtime.store->stat(manifest.value("report_resource",std::string{}),authority,report_descriptor,error)||
                !bindings.resource_runtime.store->stat(manifest.value("snapshot_resource",std::string{}),authority,snapshot_descriptor,error)||
                !bindings.resource_runtime.store->stat(manifest.value("markdown_resource",std::string{}),authority,markdown_descriptor,error)||
                !bindings.resource_runtime.store->stat(verified_same_snapshot_manifest_uri,authority,manifest_descriptor,error))
            return tool_execution_failure("tool.artifact.export.manifest_read_failed",std::move(error),"The completed report publication could not be restored safely.");
        refs={final_descriptor,report_descriptor,snapshot_descriptor,markdown_descriptor,manifest_descriptor};
        const auto output=json{{"resource",final_descriptor.uri},{"name",final_descriptor.name},{"mime_type",final_descriptor.mime_type},
            {"report_resource",report_descriptor.uri},{"snapshot_resource",snapshot_descriptor.uri},{"markdown_resource",markdown_descriptor.uri},
            {"manifest_resource",manifest_descriptor.uri},{"report_id",document.id},{"snapshot_fingerprint",snapshot.fingerprint},
            {"previous_revision_uri",document.previous_revision_uri},{"export_id",export_id},{"revision",document.revision},
            {"profile_id",document.profile_id},{"profile_revision",document.profile_revision},
            {"source_plan_id",document.source_plan_id},{"source_plan_version",document.source_plan_version},
            {"coverage_complete",document.coverage_complete},{"format",format}}.dump();
        return common_tool_execution_result::success(output,"Reused a verified completed report publication.",std::move(refs));
    }
    auto put_scoped = [&](const std::string & name, const std::string & summary, const std::string & mime,
            const std::string & text, const std::string & parent, const std::string & derivation,
            const std::string & key, common_runtime_resource_ref & out) -> bool {
        if (bindings.resource_runtime.store == nullptr) { error="resource store is unavailable"; return false; }
        bool cached=false;
        if(!find_cached(key,mime,text,out,cached))return false;
        if(cached)return true;
        agent_resource_put_request put; put.name=name; put.description=summary; put.mime_type=mime; put.text=text;
        put.scope=bindings.resource_runtime.project_id.empty()?common_runtime_resource_scope::session:common_runtime_resource_scope::project;
        put.source_provider="native";put.source_tool="artifact.export";
        put.metadata=make_tool_resource_metadata("A revisioned report artifact with host-owned provenance.",summary,"Read or download this report resource.","Source observations and evidence remain authoritative.");
        put.metadata.processing_cache_key=key;
        put.lineage.parent_uri=parent;put.lineage.derivation=derivation;
        if(!parent.empty()){put.lineage.chunk_count=1;put.lineage.byte_length=text.size();}
        apply_agent_resource_runtime(bindings.resource_runtime,put);
        agent_resource_descriptor descriptor;
        if(!bindings.resource_runtime.store->put_text(put,descriptor,error))return false;
        out.uri=descriptor.uri;out.name=descriptor.name;out.description=descriptor.description;
        out.mime_type=descriptor.mime_type;out.size_bytes=descriptor.size_bytes;out.scope=descriptor.scope;
        out.metadata=descriptor.metadata;out.lineage=descriptor.lineage;
        return true;
    };
    common_runtime_resource_ref snapshot_resource;
    if(!put_scoped(base_name+"-sources.json","Immutable report source snapshot","application/json",snapshot_json,{},"artifact.snapshot:v2",cache_key("snapshot"),snapshot_resource))
        return tool_execution_failure("tool.artifact.export.snapshot_store_failed",std::move(error),"The report source snapshot could not be stored.");
    // The revision/previous-revision decision above is authoritative for this
    // export attempt. Changing the snapshot URI must not rebuild the document
    // from plan observations and overwrite that decision.
    document.snapshot_uri=snapshot_resource.uri;
    if(!common_agent_artifact_document_to_json(document,document_json,error)||!common_agent_artifact_render_markdown(document,markdown,error))
        return tool_execution_failure("tool.artifact.export.render_failed",std::move(error),"The report could not be rendered.");
    if(verified_document_json)document_json=*verified_document_json;
    common_runtime_resource_ref report_resource;
    if(!put_scoped(base_name+"-r"+std::to_string(document.revision)+"-report.json","Structured report document", "application/vnd.llama-agent.report+json",document_json,snapshot_resource.uri,"artifact.report:v2",cache_key("report"),report_resource))
        return tool_execution_failure("tool.artifact.export.report_store_failed",std::move(error),"The structured report could not be stored.");
    common_runtime_resource_ref markdown_resource;
    if(!put_scoped(base_name+"-r"+std::to_string(document.revision)+".md",profile.title,"text/markdown",markdown,report_resource.uri,"artifact.export:markdown:v2",cache_key("markdown"),markdown_resource))
        return tool_execution_failure("tool.artifact.export.markdown_store_failed",std::move(error),"The Markdown report could not be stored.");

    common_runtime_resource_ref final_resource=markdown_resource;
    if(format=="docx"){
        if(bindings.resource_processing_service==nullptr&&!bindings.resource_processing_provider_factory)return tool_execution_failure("tool.artifact.export.docx_unavailable","resource processing is unavailable","DOCX export is unavailable.");
        agent_resource_processing_binding_request conversion;conversion.source_uri=markdown_resource.uri;conversion.operation_id="artifact-export:"+plan->id;
        conversion.authority=make_agent_resource_read_authority(bindings.resource_runtime,static_cast<int64_t>(std::time(nullptr)));
        conversion.media_type.declared_type="text/markdown";conversion.media_type.resolved_type="text/markdown";conversion.target_representation="docx";
        conversion.target_media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document";
        conversion.purpose=agent_resource_processing_purpose::artifact_generation;conversion.limits.max_source_bytes=4*1024*1024;
        conversion.limits.max_output_bytes=8*1024*1024;conversion.limits.max_duration_ms=180000;conversion.limits.max_generated_resources=1;
        std::shared_ptr<agent_resource_processing_provider> owned_processor;
        auto * processor = bindings.resource_processing_service;
        if (bindings.resource_processing_provider_factory) {
            owned_processor = bindings.resource_processing_provider_factory(conversion);
            processor = owned_processor.get();
        }
        if (processor == nullptr) return tool_execution_failure("tool.artifact.export.docx_unavailable","resource processor provider could not be created","DOCX export is unavailable.");
        const auto processed=processor->process(conversion);
        if(!processed.success||(processed.outputs.empty()&&processed.resources.empty()))return tool_execution_failure(
            "tool.artifact.export.docx_failed",
            processed.failure_code + (processed.safe_summary.empty() ? std::string{} : ": " + processed.safe_summary),
            "The report could not be converted to DOCX.");
        std::string docx_bytes;
        std::string docx_mime=conversion.target_media_type;
        common_runtime_resource_metadata docx_metadata;
        if(!processed.outputs.empty()){
            const auto & output=processed.outputs.front();
            if(processed.outputs.size()!=1)return tool_execution_failure("tool.artifact.export.docx_failed","processor returned multiple DOCX outputs","The report could not be converted to DOCX.");
            docx_bytes=output.bytes;
            if(!output.mime_type.empty())docx_mime=output.mime_type;
            docx_metadata=output.metadata;
        }else{
            if(processed.resources.size()!=1||bindings.resource_runtime.store==nullptr)return tool_execution_failure("tool.artifact.export.docx_failed","processor returned an invalid derived DOCX resource","The report could not be converted to DOCX.");
            agent_resource_descriptor processed_resource;
            if(!bindings.resource_runtime.store->stat(processed.resources.front().uri,conversion.authority,processed_resource,error)||
                    !bindings.resource_runtime.store->read_bytes(processed_resource.uri,conversion.authority,conversion.limits.max_output_bytes,docx_bytes,error))
                return tool_execution_failure("tool.artifact.export.docx_failed",std::move(error),"The generated DOCX resource could not be verified.");
            if(!processed_resource.mime_type.empty())docx_mime=processed_resource.mime_type;
            docx_metadata=processed_resource.metadata;
        }
        if(docx_bytes.empty()||common_normalize_resource_media_type(docx_mime)!=
                common_normalize_resource_media_type(conversion.target_media_type))
            return tool_execution_failure("tool.artifact.export.docx_failed","processor returned empty bytes or an unexpected DOCX media type","The report could not be converted to DOCX.");
        bool cached_docx=false;
        if(!find_cached(cache_key("docx"),docx_mime,docx_bytes,final_resource,cached_docx))return tool_execution_failure("tool.artifact.export.docx_cache_failed",std::move(error),"The prior DOCX export could not be safely reused.");
        if(!cached_docx){
            common_runtime_resource_lineage docx_lineage;docx_lineage.parent_uri=markdown_resource.uri;docx_lineage.chunk_count=1;docx_lineage.byte_length=docx_bytes.size();docx_lineage.derivation="artifact.export:docx";
            agent_resource_put_request put;put.name=base_name+".docx";put.description=profile.title;
            put.mime_type=docx_mime;
            put.bytes=docx_bytes;put.bytes_are_authoritative=true;
            put.scope=bindings.resource_runtime.project_id.empty()?common_runtime_resource_scope::session:common_runtime_resource_scope::project;
            put.source_provider="native";put.source_tool="artifact.export";put.metadata=std::move(docx_metadata);put.metadata.processing_cache_key=cache_key("docx");put.lineage=std::move(docx_lineage);
            apply_agent_resource_runtime(bindings.resource_runtime,put);
            agent_resource_descriptor descriptor;
            if(bindings.resource_runtime.store==nullptr||!bindings.resource_runtime.store->put_bytes(put,descriptor,error))
                return tool_execution_failure("tool.artifact.export.docx_store_failed",std::move(error),"The DOCX report could not be stored.");
            final_resource=descriptor;
        }
    }
    const auto manifest_json=json{{"schema_version",1},{"export_id",export_id},{"report_id",document.id},
        {"revision",document.revision},{"previous_revision_uri",document.previous_revision_uri},
        {"snapshot_fingerprint",snapshot.fingerprint},{"profile_id",document.profile_id},
        {"profile_revision",document.profile_revision},{"format",format},{"coverage_complete",document.coverage_complete},
        {"snapshot_resource",snapshot_resource.uri},{"report_resource",report_resource.uri},
        {"markdown_resource",markdown_resource.uri},{"resource",final_resource.uri}}.dump();
    common_runtime_resource_ref manifest_resource;
    if(!put_scoped(base_name+"-r"+std::to_string(document.revision)+"-"+format+"-manifest.json",
            "Complete report publication manifest","application/json",manifest_json,report_resource.uri,
            "artifact.export:manifest:v1",cache_key("manifest"),manifest_resource))
        return tool_execution_failure("tool.artifact.export.manifest_failed",std::move(error),"The report resources were stored but the publication manifest could not be verified.");
    const auto output=json{{"resource",final_resource.uri},{"name",final_resource.name},{"mime_type",final_resource.mime_type},
        {"report_resource",report_resource.uri},{"snapshot_resource",snapshot_resource.uri},{"markdown_resource",markdown_resource.uri},
        {"manifest_resource",manifest_resource.uri},{"report_id",document.id},{"snapshot_fingerprint",snapshot.fingerprint},
        {"previous_revision_uri",document.previous_revision_uri},{"export_id",export_id},
        {"revision",document.revision},{"profile_id",document.profile_id},{"profile_revision",document.profile_revision},
        {"source_plan_id",document.source_plan_id},{"source_plan_version",document.source_plan_version},
        {"coverage_complete",document.coverage_complete},{"format",format}}.dump();
    std::vector<common_runtime_resource_ref> exported_resources;
    exported_resources.push_back(std::move(final_resource));
    exported_resources.push_back(std::move(report_resource));
    exported_resources.push_back(std::move(snapshot_resource));
    exported_resources.push_back(std::move(markdown_resource));
    exported_resources.push_back(std::move(manifest_resource));
    return common_tool_execution_result::success(output,
        "Exported a report with source snapshot and revision provenance.",std::move(exported_resources));
}

bool agent_resource_has_text_representation(const agent_resource_descriptor & descriptor) {
    return common_resource_media_type_is_text_like(descriptor.mime_type);
}

std::vector<std::string> agent_resource_available_representations(
        const agent_resource_descriptor & descriptor) {
    std::vector<std::string> representations{"bytes"};
    if (agent_resource_has_text_representation(descriptor)) {
        representations.push_back("text");
    }
    return representations;
}

std::string collapse_ws(const std::string & value) {
    std::string out;
    bool spaced = false;
    for (unsigned char c : value) {
        if (std::isspace(c)) {
            if (!spaced && !out.empty()) out.push_back(' ');
            spaced = true;
        } else {
            out.push_back((char) c);
            spaced = false;
        }
    }
    return trim_copy(out);
}

void replace_all(std::string & value, const std::string & from, const std::string & to) {
    if (from.empty()) return;
    size_t pos = 0;
    while ((pos = value.find(from, pos)) != std::string::npos) {
        value.replace(pos, from.size(), to);
        pos += to.size();
    }
}

std::string html_decode(std::string value) {
    replace_all(value, "&amp;", "&");
    replace_all(value, "&lt;", "<");
    replace_all(value, "&gt;", ">");
    replace_all(value, "&quot;", "\"");
    replace_all(value, "&#39;", "'");
    replace_all(value, "&nbsp;", " ");
    return value;
}

std::string html_to_text(std::string html) {
    html = std::regex_replace(html, std::regex(R"(<script[\s\S]*?</script>)", std::regex::icase), " ");
    html = std::regex_replace(html, std::regex(R"(<style[\s\S]*?</style>)", std::regex::icase), " ");
    html = std::regex_replace(html, std::regex(R"(<[^>]+>)"), " ");
    return collapse_ws(html_decode(html));
}

std::string html_title(const std::string & html) {
    std::smatch match;
    if (!std::regex_search(html, match, std::regex(R"(<title[^>]*>([\s\S]*?)</title>)", std::regex::icase))) return {};
    return collapse_ws(html_to_text(match[1].str()));
}

json trim_search_results_for_inline(
        const json & results,
        size_t inline_limit) {
    if (!results.is_array() || results.size() <= inline_limit) {
        return results;
    }

    json trimmed = json::array();
    for (size_t i = 0; i < inline_limit; ++i) {
        trimmed.push_back(results.at(i));
    }
    return trimmed;
}

std::string url_encode(const std::string & value) {
    std::ostringstream out;
    out << std::hex << std::uppercase;
    for (unsigned char c : value) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') out << c;
        else if (c == ' ') out << '+';
        else out << '%' << std::setw(2) << std::setfill('0') << (int) c;
    }
    return out.str();
}

std::string url_decode(const std::string & value) {
    std::string out;
    out.reserve(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        if (value[i] == '+' ) out.push_back(' ');
        else if (value[i] == '%' && i + 2 < value.size()) {
            const auto hex = value.substr(i + 1, 2);
            char * end = nullptr;
            const long code = std::strtol(hex.c_str(), &end, 16);
            if (end == hex.c_str() + 2) {
                out.push_back((char) code);
                i += 2;
            } else out.push_back(value[i]);
        } else out.push_back(value[i]);
    }
    return out;
}

bool private_ipv4(const std::string & host) {
    int a = -1, b = -1, c = -1, d = -1;
    char tail = 0;
    if (std::sscanf(host.c_str(), "%d.%d.%d.%d%c", &a, &b, &c, &d, &tail) != 4) return false;
    if (a == 10 || a == 127) return true;
    if (a == 192 && b == 168) return true;
    if (a == 172 && b >= 16 && b <= 31) return true;
    if (a == 169 && b == 254) return true;
    if (a == 0) return true;
    return false;
}

bool public_https_url(const common_http_url & parts, std::string & error) {
    if (parts.scheme != "https") { error = "only HTTPS URLs are allowed"; return false; }
    if (!parts.user.empty() || !parts.password.empty()) { error = "credentialed URLs are not allowed"; return false; }
    const auto host = lower_copy(parts.host);
    if (host.empty()) { error = "URL host is empty"; return false; }
    if (host == "localhost" || host == "::1" || host.find(".local") != std::string::npos) { error = "private or loopback hosts are not allowed"; return false; }
    if (private_ipv4(host)) { error = "private or loopback hosts are not allowed"; return false; }
    if (host.find(':') != std::string::npos) {
        if (host.rfind("fc", 0) == 0 || host.rfind("fd", 0) == 0 || host.rfind("fe80", 0) == 0 || host == "::1") {
            error = "private or loopback hosts are not allowed"; return false;
        }
    }
    return true;
}

#ifdef LLAMA_AGENT_WEB_USE_WINHTTP

std::wstring utf8_to_wide(const std::string & value) {
    if (value.empty()) return {};
    const int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), (int) value.size(), nullptr, 0);
    if (count <= 0) return {};
    std::wstring wide((size_t) count, L'\0');
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), (int) value.size(), wide.data(), count) != count) return {};
    return wide;
}

std::string winhttp_error(DWORD code) {
    LPSTR text = nullptr;
    const DWORD length = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR) &text, 0, nullptr);
    std::string result = length && text ? trim_copy(text) : "WinHTTP error " + std::to_string(code);
    if (text) LocalFree(text);
    return result;
}

bool winhttp_query_string(HINTERNET request, DWORD query, std::string & value, std::string & error) {
    DWORD bytes = 0;
    WinHttpQueryHeaders(request, query, WINHTTP_HEADER_NAME_BY_INDEX, nullptr, &bytes, WINHTTP_NO_HEADER_INDEX);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || bytes < sizeof(wchar_t)) {
        error = "WinHTTP header query failed: " + winhttp_error(GetLastError());
        return false;
    }
    std::wstring wide(bytes / sizeof(wchar_t), L'\0');
    if (!WinHttpQueryHeaders(request, query, WINHTTP_HEADER_NAME_BY_INDEX, wide.data(), &bytes, WINHTTP_NO_HEADER_INDEX)) {
        error = "WinHTTP header query failed: " + winhttp_error(GetLastError());
        return false;
    }
    const int output_size = WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (output_size <= 0) { error = "WinHTTP header conversion failed"; return false; }
    value.resize((size_t) output_size);
    WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, value.data(), output_size, nullptr, nullptr);
    value.pop_back();
    return true;
}

bool winhttp_fetch_text(const std::string & url, const common_http_url & parts, size_t max_bytes,
        const std::string & accept, std::string & body, int & status, std::string & content_type,
        std::string & error, bool & truncated) {
    const auto host = utf8_to_wide(parts.host);
    const auto path = utf8_to_wide(parts.path);
    if (host.empty() || path.empty()) { error = "URL is not valid UTF-8"; return false; }

    HINTERNET session = WinHttpOpen(L"llama-agent/1", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
        WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!session) { error = "WinHTTP session failed: " + winhttp_error(GetLastError()); return false; }
    const auto close_session = [&] { WinHttpCloseHandle(session); };
    WinHttpSetTimeouts(session, 10000, 10000, 10000, 10000);

    HINTERNET connection = WinHttpConnect(session, host.c_str(), (INTERNET_PORT) parts.port, 0);
    if (!connection) { error = "WinHTTP connection failed: " + winhttp_error(GetLastError()); close_session(); return false; }
    const auto close_connection = [&] { WinHttpCloseHandle(connection); close_session(); };

    HINTERNET request = WinHttpOpenRequest(connection, L"GET", path.c_str(), nullptr, WINHTTP_NO_REFERER,
        WINHTTP_DEFAULT_ACCEPT_TYPES, WINHTTP_FLAG_SECURE);
    if (!request) { error = "WinHTTP request creation failed: " + winhttp_error(GetLastError()); close_connection(); return false; }
    const auto close_request = [&] { WinHttpCloseHandle(request); close_connection(); };
    const std::wstring headers = L"Accept: " + utf8_to_wide(accept) + L"\r\n";
    if (!WinHttpSendRequest(request, headers.c_str(), (DWORD) -1L, WINHTTP_NO_REQUEST_DATA, 0, 0, 0) || !WinHttpReceiveResponse(request, nullptr)) {
        error = "WinHTTP request failed: " + winhttp_error(GetLastError()); close_request(); return false;
    }

    DWORD status_code = 0;
    DWORD status_size = sizeof(status_code);
    if (!WinHttpQueryHeaders(request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX,
            &status_code, &status_size, WINHTTP_NO_HEADER_INDEX)) {
        error = "WinHTTP status query failed: " + winhttp_error(GetLastError()); close_request(); return false;
    }
    status = (int) status_code;
    if (status < 200 || status >= 300) { error = "HTTP status " + std::to_string(status); close_request(); return false; }
    std::string ignored;
    if (!winhttp_query_string(request, WINHTTP_QUERY_CONTENT_TYPE, content_type, ignored)) content_type.clear();

    body.clear();
    body.reserve(std::min<size_t>(max_bytes, 65536));
    for (;;) {
        DWORD available = 0;
        if (!WinHttpQueryDataAvailable(request, &available)) { error = "WinHTTP read failed: " + winhttp_error(GetLastError()); close_request(); return false; }
        if (available == 0) break;
        const size_t remaining = max_bytes > body.size() ? max_bytes - body.size() : 0;
        if (remaining == 0) { truncated = true; break; }
        const DWORD to_read = (DWORD) std::min<size_t>(available, std::min<size_t>(remaining, 8192));
        std::string chunk(to_read, '\0');
        DWORD received = 0;
        if (!WinHttpReadData(request, chunk.data(), to_read, &received)) { error = "WinHTTP read failed: " + winhttp_error(GetLastError()); close_request(); return false; }
        body.append(chunk.data(), received);
        if (received < available || body.size() >= max_bytes) { truncated = true; break; }
    }
    close_request();
    return true;
}

#endif

bool http_fetch_text(const std::string & url, size_t max_bytes, json & response, std::string & error,
        std::string * raw_body = nullptr,
        const std::string & accept = "text/html, text/plain;q=0.9, application/xhtml+xml;q=0.8") {
    const auto request_started = std::chrono::steady_clock::now();
    std::string final_url = url;
    const auto record_http = [&](int status, const std::string & content_type, size_t bytes) {
        response["http"] = {
            {"provider", "web.fetch"}, {"operation", "fetch"}, {"method", "GET"},
            {"requested_url", url}, {"final_url", url}, {"status", status},
            {"content_type", content_type}, {"response_bytes", bytes},
            {"duration_ms", std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - request_started).count()},
        };
    };
    try {
#ifdef LLAMA_AGENT_WEB_USE_WINHTTP
        const auto parts = common_http_parse_url(url);
        if (!public_https_url(parts, error)) return false;
        std::string body;
        bool truncated = false;
        int status = 0;
        std::string content_type;
        if (!winhttp_fetch_text(url, parts, max_bytes, accept, body, status, content_type, error, truncated)) {
            record_http(status, content_type, body.size());
            return false;
        }
#else
        auto [cli, parts] = common_http_client(url);
        if (!public_https_url(parts, error)) return false;
        cli.set_connection_timeout(10);
        cli.set_read_timeout(10);
        cli.set_write_timeout(10);
        cli.set_follow_location(true);
        cli.set_default_headers({{"User-Agent", "llama-agent/1"}, {"Accept", accept}});

        std::string body;
        body.reserve(std::min<size_t>(max_bytes, 65536));
        bool truncated = false;
        int status = 0;
        std::string content_type;
        auto res = cli.Get(parts.path,
            [&](const httplib::Response & r) {
                status = r.status;
                if (r.has_header("Content-Type")) content_type = r.get_header_value("Content-Type");
                if (r.status >= 300 && r.status < 400 && r.has_header("Location")) {
                    const auto location = r.get_header_value("Location");
                    if (location.find("://") != std::string::npos) {
                        final_url = location;
                    } else if (location.rfind("//", 0) == 0) {
                        final_url = parts.scheme + ":" + location;
                    } else {
                        const auto query = parts.path.find_first_of("?#");
                        auto current_path = parts.path.substr(0, query);
                        const auto origin = parts.scheme + "://" + common_http_format_host(parts.host) +
                            ((parts.scheme == "https" && parts.port == 443) ||
                             (parts.scheme == "http" && parts.port == 80)
                                ? "" : ":" + std::to_string(parts.port));
                        if (!location.empty() && location.front() == '/') {
                            final_url = origin + location;
                        } else {
                            const auto slash = current_path.rfind('/');
                            final_url = origin + current_path.substr(0, slash == std::string::npos ? 0 : slash + 1) + location;
                        }
                    }
                }
                return r.status >= 200 && r.status < 300;
            },
            [&](const char * data, size_t len) {
                const size_t remaining = max_bytes > body.size() ? max_bytes - body.size() : 0;
                const size_t take = std::min(len, remaining);
                body.append(data, take);
                if (take < len) {
                    truncated = true;
                    return false;
                }
                return true;
            },
            nullptr);
        if (!res) {
            record_http(status, content_type, body.size());
            response["http"]["final_url"] = final_url;
            if (truncated) {
                response["truncated"] = true;
                error = "response exceeds max_bytes";
                return false;
            }
            error = status > 0 ? "HTTP status " + std::to_string(status) :
                "HTTP request failed: " + std::string(httplib::to_string(res.error()));
            return false;
        }
#endif
        record_http(status, content_type, body.size());
        if (response["http"].contains("final_url")) {
            response["http"]["final_url"] = final_url;
        }
        if (raw_body) *raw_body = body;
        auto http_metadata = response["http"];
        http_metadata["final_url"] = http_metadata.value("final_url", url);
        response = {
            {"url", url},
            {"final_url", http_metadata.value("final_url", url)},
            {"status", status},
            {"content_type", content_type},
            {"title", html_title(body)},
            {"text", html_to_text(body)},
            {"truncated", truncated},
            {"http", std::move(http_metadata)},
        };
        return true;
    } catch (const std::exception & e) {
        error = e.what();
        return false;
    }
}

bool parse_search_results(const std::string & html, int limit, json & results) {
    results = json::array();
    std::regex anchor(R"ddg(<a[^>]*href="([^"]+)"[^>]*>([\s\S]*?)</a>)ddg", std::regex::icase);
    std::set<std::string> seen;
    for (std::sregex_iterator it(html.begin(), html.end(), anchor), end; it != end && results.size() < (size_t) limit; ++it) {
        std::string href = html_decode((*it)[1].str());
        std::string title = collapse_ws(html_to_text((*it)[2].str()));
        if (title.empty()) continue;
        const auto lower_href = lower_copy(href);
        if (lower_href.find("duckduckgo.com/l/?") != std::string::npos) {
            const auto pos = href.find("uddg=");
            if (pos != std::string::npos) {
                auto encoded = href.substr(pos + 5);
                const auto amp = encoded.find('&');
                if (amp != std::string::npos) encoded = encoded.substr(0, amp);
                href = url_decode(encoded);
            }
        }
        if (lower_copy(href).rfind("https://", 0) != 0) continue;
        if (!seen.insert(href).second) continue;
        results.push_back({{"title", title}, {"url", href}, {"snippet", ""}, {"source", "duckduckgo-lite"}});
    }
    return !results.empty();
}

} // namespace

bool common_register_native_tool_adapters(const common_tool_catalog & catalog, const std::string & profile_id,
        const common_native_tool_bindings & bindings, common_tool_registry & registry,
        common_tool_adapter_result & result, std::string & error) {
    result = {};
    const auto definitions = catalog.load_profile(profile_id, error);
    if (!error.empty()) return false;
    for (const auto & definition : definitions) {
        bool installed = false;
        const bool handled_by_memory_family = common_try_register_memory_tool_adapter(
                definition, bindings, registry, installed, error);
        if (handled_by_memory_family) {
            // The memory family owns its bindings and policy details.
        } else if (common_try_register_oracle_tool_adapter(
                definition,bindings,registry,installed,error)) {
            // The Oracle facade owns RuleSet/session binding and evaluation provenance.
        } else if (common_try_register_data_tool_adapter(
                definition, bindings, registry, installed, error)) {
            // The data family owns dataset, data and statistics adapters.
        } else if (common_try_register_document_tool_adapter(
                definition, bindings, registry, installed, error)) {
            // The document family owns document table projection adapters.
        } else if (common_try_register_repository_tool_adapter(
                definition, bindings, registry, installed, error)) {
            // The repository family owns repository/workspace filesystem and Git adapters.
        } else if (common_try_register_diagnostics_tool_adapter(
                definition, bindings, registry, installed, error)) {
            // The diagnostics family owns compiler, symbol and test-analysis adapters.
        } else if (common_try_register_resource_tool_adapter(
                definition, bindings, registry, installed, error)) {
            // The resource family owns resource inspection, reads and representation materialization.
        } else if (definition.executor_id == "builtin.calculator") {
            installed = register_definition(definition, registry, [](const std::string & input) {
                std::string err;
                json arguments;
                if (!parse_object(input, arguments, err) || !arguments.contains("expression") || !arguments["expression"].is_string()) {
                    if (err.empty()) err = "calculator requires an expression";
                    return tool_validation_failure("tool.calculator.invalid_expression", std::move(err), "Calculator requires a valid expression.");
                }
                const auto expression = arguments["expression"].get<std::string>();
                if (expression.size() > 256) return tool_limit_failure("tool.calculator.expression_too_large", "calculator expression exceeds limit", "Calculator expression exceeds the allowed size.");
                double value = 0.0;
                calculator_parser parser(expression);
                if (!parser.parse(value, err)) return tool_validation_failure("tool.calculator.invalid_expression", std::move(err), "Calculator expression could not be parsed.");
                return tool_success_json({{"value", value}});
            }, error);
        } else if (definition.executor_id == "builtin.time_now") {
            installed = register_definition(definition, registry, [](const std::string & input) {
                std::string err;
                json arguments;
                if (!parse_object(input, arguments, err)) return tool_validation_failure("tool.time_now.invalid_arguments", std::move(err));
                const auto timezone = arguments.value("timezone", std::string("UTC"));
                if (timezone != "UTC") return tool_validation_failure("tool.time_now.unsupported_timezone", "time_now currently supports only UTC", "Only UTC is currently supported.");
                return tool_success_json({{"timezone", "UTC"}, {"time", utc_now()}});
            }, error);
        } else if (definition.executor_id == "builtin.artifact.export" && bindings.resource_runtime.store != nullptr) {
            installed = register_definition(definition, registry, [bindings](const std::string & input) {
                std::string err; json arguments;
                if (!parse_object(input, arguments, err)) return tool_validation_failure("tool.artifact.export.invalid_arguments", std::move(err));
                if (arguments.contains("source_document")) return export_saved_document(bindings, arguments);
                if (arguments.contains("source_dataset")) return export_dataset_csv(bindings, arguments);
                if (arguments.contains("source_plan") || arguments.value("format", std::string{}) == "markdown" || arguments.value("format", std::string{}) == "docx") return export_plan_report(bindings, arguments);
                if (!arguments.contains("name") || !arguments["name"].is_string() || !arguments.contains("content") || !arguments["content"].is_string()) return tool_validation_failure("tool.artifact.export.invalid_arguments", "artifact.export requires name and content, or source_dataset");
                common_runtime_resource_ref resource;
                const auto mime = arguments.value("mime_type", std::string("text/plain"));
                if (!persist_tool_resource(bindings, arguments["name"].get<std::string>(), "Exported tool artifact", mime, arguments["content"].get<std::string>(), "artifact.export", make_tool_resource_metadata("artifact", "Exported tool result", "Read as an artifact resource", "Bounded host-owned export"), resource, err)) return tool_execution_failure("tool.artifact.export.failed", std::move(err), "Artifact export failed.");
                return tool_success_json({{"resource", resource.uri}, {"name", resource.name}, {"mime_type", resource.mime_type}});
            }, error, false, true);
        } else if ((definition.executor_id == "sandbox.development.build" || definition.executor_id == "sandbox.development.test") && bindings.sandbox_execute) {
            installed = register_definition(definition, registry, [bindings, tool_name = definition.name](const std::string & input) {
                common_agent_sandbox_request request;
                std::string err;
                if (!make_sandbox_request(tool_name, input, request, err)) {
                    return tool_validation_failure("tool." + tool_name + ".invalid_arguments", std::move(err), "Sandbox execution arguments are invalid.");
                }
                return bindings.sandbox_execute(std::move(request));
            }, error, false, true);
        } else if (definition.executor_id == "builtin.web_search") {
            if (bindings.web_search) {
                installed = register_definition(definition, registry, bindings.web_search, error);
            } else {
                installed = register_definition(definition, registry, [bindings](const std::string & input) {
                    std::string err;
                    json arguments;
                    if (!parse_object(input, arguments, err) || !arguments.contains("query") || !arguments["query"].is_string()) {
                        if (err.empty()) err = "web_search requires a query";
                        return tool_validation_failure("tool.web_search.invalid_query", std::move(err), "Web search requires a valid query.");
                    }
                    const auto query = trim_copy(arguments["query"].get<std::string>());
                    const int limit = arguments.value("limit", 5);
                    const auto site = trim_copy(arguments.value("site", std::string{}));
                    if (query.empty() || query.size() > 512 || limit < 1 || limit > 8 || site.size() > 256) {
                        return tool_validation_failure("tool.web_search.out_of_bounds", "web_search arguments are out of bounds", "Web search arguments are out of bounds.");
                    }
                    const auto search_query = site.empty() ? query : query + " site:" + site;
                    json fetched = json::object();
                    std::string raw_html;
                    if (!http_fetch_text("https://lite.duckduckgo.com/lite/?q=" + url_encode(search_query), 128 * 1024, fetched, err, &raw_html)) {
                        return tool_network_failure("tool.web_search.fetch_failed", std::move(err), "Web search request failed.");
                    }
                    json results;
                    if (!parse_search_results(raw_html, limit, results)) {
                        return tool_success_json(common_tool_web_search_result_to_json({
                            json::array(),
                            "duckduckgo-lite",
                            std::nullopt,
                            std::nullopt,
                        }));
                    }

                    common_tool_web_search_result payload{
                        results,
                        "duckduckgo-lite",
                        std::nullopt,
                        std::nullopt,
                    };

                    const std::string full_payload = common_tool_web_search_result_to_json(payload).dump();
                    const bool should_externalize =
                        bindings.resource_runtime.store != nullptr &&
                        (full_payload.size() > 2048 || results.size() > 3);
                    if (!should_externalize) {
                        return common_tool_execution_result::success(
                            common_tool_web_search_result_to_json(payload).dump(),
                            "Web search returned " + std::to_string(results.size()) + " candidate(s).");
                    }

                    std::vector<std::string> keywords;
                    keywords.push_back(query);
                    if (!site.empty()) {
                        keywords.push_back(site);
                    }

                    common_runtime_resource_ref resource_ref;
                    if (!persist_tool_resource(
                            bindings,
                            "web-search-results.json",
                            "Full web search result set for the current turn.",
                            "application/json",
                            full_payload,
                            "web_search",
                            make_tool_resource_metadata(
                                "Preserve the full bounded web search candidate set outside the inline model context.",
                                "DuckDuckGo Lite search candidates for query \"" + query + "\".",
                                "Use this resource when the planner or a later tool step needs the full candidate set rather than the inline top hits.",
                                "Search results are unverified provider candidates with short snippets; they are not fetched page contents.",
                                std::move(keywords)),
                            resource_ref,
                            err)) {
                        return tool_execution_failure("tool.web_search.resource_store_failed", std::move(err), "Web search results could not be materialized as a host resource.");
                    }

                    payload.results = trim_search_results_for_inline(results, 3);
                    payload.truncated = results.size() > 3;
                    payload.total_results = results.size();

                    return common_tool_execution_result::success(
                        common_tool_web_search_result_to_json(payload).dump(),
                        "Web search returned " + std::to_string(results.size()) + " candidate(s); the full result set was stored as a turn resource.",
                        {std::move(resource_ref)});
                }, error);
            }
        } else if (definition.executor_id == "builtin.web_fetch") {
            if (bindings.web_fetch) {
                installed = register_definition(definition, registry, bindings.web_fetch, error);
            } else {
                installed = register_definition(definition, registry, [bindings](const std::string & input) {
                    std::string err;
                    json arguments;
                    if (!parse_object(input, arguments, err) || !arguments.contains("url") || !arguments["url"].is_string()) {
                        if (err.empty()) err = "web_fetch requires a url";
                        return tool_validation_failure("tool.web_fetch.invalid_url", std::move(err), "Web fetch requires a valid URL.");
                    }
                    const auto url = trim_copy(arguments["url"].get<std::string>());
                    const auto extract = arguments.value("extract", std::string("text"));
                    const int max_bytes = arguments.value("max_bytes", 10 * 1024 * 1024);
                    std::string accept = "*/*";
                    if (arguments.contains("accept")) {
                        if (!arguments["accept"].is_string()) {
                            return tool_validation_failure("tool.web_fetch.out_of_bounds", "web_fetch accept must be a string", "Web fetch arguments are out of bounds.");
                        }
                        accept = trim_copy(arguments["accept"].get<std::string>());
                    }
                    if (url.size() < 9 || url.size() > 2048 || extract != "text" || max_bytes < 1 ||
                            max_bytes > 10 * 1024 * 1024 || !valid_accept_header(accept)) {
                        return tool_validation_failure("tool.web_fetch.out_of_bounds", "web_fetch arguments are out of bounds", "Web fetch arguments are out of bounds.");
                    }
                    json fetched;
                    std::string raw_body;
                    if (!http_fetch_text(url, (size_t) max_bytes, fetched, err, &raw_body, accept)) {
                        auto failure = fetched.is_object() && fetched.value("truncated", false)
                            ? tool_limit_failure("tool.web_fetch.byte_limit", std::move(err),
                                "Resource exceeds max_bytes; request a larger bound within the tool's limit.")
                            : tool_network_failure("tool.web_fetch.request_failed", std::move(err), "Web fetch request failed.");
                        if (fetched.contains("http") && fetched["http"].is_object()) {
                            const auto & http = fetched["http"];
                            failure.http.provider = http.value("provider", std::string{});
                            failure.http.operation = http.value("operation", std::string{});
                            failure.http.method = http.value("method", std::string{});
                            failure.http.requested_url = http.value("requested_url", std::string{});
                            failure.http.final_url = http.value("final_url", std::string{});
                            failure.http.status = http.value("status", 0);
                            failure.http.content_type = http.value("content_type", std::string{});
                            failure.http.response_bytes = http.value("response_bytes", size_t(0));
                            failure.http.duration_ms = http.value("duration_ms", int64_t(0));
                        }
                        return failure;
                    }

                    const std::string mime_type = common_normalize_resource_media_type(
                        fetched.value("content_type", std::string{}));
                    const bool text_like = common_resource_media_type_is_text_like(mime_type);
                    common_tool_web_fetch_result fetch_result{
                        fetched.value("url", url),
                        fetched.value("final_url", url),
                        fetched.value("status", 0),
                        mime_type.empty() ? "application/octet-stream" : mime_type,
                        text_like ? fetched.value("title", std::string{}) : std::string{},
                        text_like ? fetched.value("text", std::string{}) : std::string{},
                        fetched.value("truncated", false),
                    };
                    const std::string title = trim_copy(fetch_result.title);
                    const bool truncated = fetch_result.truncated;
                    common_tool_http_metadata http_metadata;
                    if (fetched.contains("http") && fetched["http"].is_object()) {
                        const auto & http = fetched["http"];
                        http_metadata.provider = http.value("provider", std::string{});
                        http_metadata.operation = http.value("operation", std::string{});
                        http_metadata.method = http.value("method", std::string{});
                        http_metadata.requested_url = http.value("requested_url", std::string{});
                        http_metadata.final_url = http.value("final_url", std::string{});
                        http_metadata.status = http.value("status", 0);
                        http_metadata.content_type = http.value("content_type", std::string{});
                        http_metadata.response_bytes = http.value("response_bytes", size_t(0));
                        http_metadata.duration_ms = http.value("duration_ms", int64_t(0));
                    }
                    // Always materialize the authoritative fetched body when
                    // the host resource store is available. This preserves a
                    // PDF, XML or JSON representation for later document
                    // processing instead of storing only an extracted-text
                    // JSON wrapper.
                    const bool should_externalize = bindings.resource_runtime.store != nullptr;
                    auto payload = common_tool_web_fetch_make_inline_result(fetch_result);
                    if (!should_externalize) {
                        auto success = common_tool_execution_result::success(
                            common_tool_web_fetch_inline_result_to_json(payload).dump(),
                            title.empty()
                                ? "Fetched bounded page text from " + url + "; only a bounded preview is available."
                                : "Fetched bounded page text for \"" + title + "\"; only a bounded preview is available.");
                        success.http = std::move(http_metadata);
                        return success;
                    }

                    std::vector<std::string> keywords;
                    keywords.push_back(url);
                    if (!title.empty()) {
                        keywords.push_back(title);
                    }

                    auto fetch_metadata=make_tool_resource_metadata(
                        "Preserve the authoritative bounded fetched body outside the inline model context.",
                        title.empty()?"Fetched bounded content from "+url+".":"Fetched bounded content for \""+title+"\".",
                        "Use this resource for bounded reads or host-provided document text processing; the inline result is only a preview.",
                        truncated?"The fetched body was truncated by the configured byte limit; do not treat it as a complete source.":
                            "The resource is the fetched response body; derived text remains subject to the selected host processor.",
                        std::move(keywords));
                    fetch_metadata.source_provider="native";
                    fetch_metadata.source_operation="web.fetch";
                    fetch_metadata.source_request_json=arguments.dump();

                    common_runtime_resource_ref resource_ref;
                    if (!persist_tool_bytes(
                            bindings,
                            "web-fetch-content",
                            "Bounded fetched web content for the current turn.",
                            fetch_result.content_type,
                            raw_body,
                            "web_fetch",
                            std::move(fetch_metadata),
                            resource_ref,
                            err)) {
                        return tool_execution_failure("tool.web_fetch.resource_store_failed", std::move(err), "Web fetch result could not be materialized as a host resource.");
                    }
                    http_metadata.resource_ref = resource_ref.uri;

                    payload.resource_ref = resource_ref.uri;

                    auto success = common_tool_execution_result::success(
                        common_tool_web_fetch_inline_result_to_json(payload).dump(),
                        title.empty()
                            ? "Fetched bounded page text from " + url + "; the full payload was stored as a turn resource."
                            : "Fetched bounded page text for \"" + title + "\"; the full payload was stored as a turn resource.",
                        {std::move(resource_ref)});
                    success.http = std::move(http_metadata);
                    return success;
                }, error);
            }
        } else if (definition.executor_id == "builtin.openapi_connect") {
            if (bindings.openapi_connect) {
                installed = register_definition(definition, registry, bindings.openapi_connect, error);
            }
        } else if (definition.executor_id == "builtin.plan_get" && bindings.plan_store && bindings.plan_id != nullptr && !bindings.plan_id->empty()) {
            installed = register_definition(definition, registry, [bindings](const std::string & input) {
                std::string err;
                json arguments;
                if (!parse_object(input, arguments, err)) return tool_validation_failure("tool.plan_get.invalid_arguments", std::move(err));
                const auto plan = bindings.plan_store->get(*bindings.plan_id, err);
                if (!err.empty()) return tool_execution_failure("tool.plan_get.load_failed", std::move(err), "Plan could not be loaded.");
                if (!plan) return tool_not_found_failure("tool.plan_get.unavailable", "bound plan is unavailable", "Bound plan is unavailable.");
                common_tool_plan_get_payload response;
                response.plan_id = plan->id;
                response.version = plan->version;
                response.goal = plan->goal;
                response.active_step = plan->active_step_id;
                response.next_action = plan->next_action;
                for (const auto & step : plan->steps) {
                    response.steps.push_back({
                        step.id,
                        step.title,
                        step.objective,
                        (int) step.status,
                        step.selected_tool,
                    });
                }
                if (arguments.value("include_history", false)) {
                    const auto history = bindings.plan_store->history(*bindings.plan_id, err);
                    if (!err.empty()) return tool_execution_failure("tool.plan_get.history_failed", std::move(err), "Plan history could not be loaded.");
                    response.history_count = history.size();
                }
                return tool_success_text(common_tool_plan_get_result_to_json(response).dump());
            }, error);
        }
        if (!error.empty()) return false;
        if (installed) result.registered.push_back(definition.name);
        else result.unavailable.push_back(definition.name);
    }
    error.clear();
    return true;
}
