#include "agent-xlsx-workbook-json-processor.h"

#include <utility>

namespace {
constexpr const char * xlsx_mime =
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";

agent_resource_processing_result failure(
        const std::string & id, std::string code, std::string summary) {
    agent_resource_processing_result result;
    result.processor_id = id;
    result.failure_code = std::move(code);
    result.safe_summary = std::move(summary);
    return result;
}

bool is_zip(const std::string & bytes) {
    return bytes.size() >= 4 && bytes.compare(0, 4, "PK\x03\x04") == 0;
}
}

bool agent_xlsx_workbook_json_processor::supports(
        const std::string & mime_type, const std::string & target_representation) const {
    return common_normalize_resource_media_type(mime_type) == xlsx_mime &&
        target_representation == "workbook-json";
}

agent_resource_processor::support agent_xlsx_workbook_json_processor::supports(
        const agent_resource_processing_request & request) const {
    const auto type = request.media_type.resolved_type.empty()
        ? request.source.mime_type : request.media_type.resolved_type;
    const bool target = request.target_media_type.empty() ||
        common_normalize_resource_media_type(request.target_media_type) == "application/json";
    return {supports(type, request.target_representation) && target, 70, false, false};
}

agent_resource_processing_result agent_xlsx_workbook_json_processor::process(
        const agent_resource_processing_request & request) const {
    if (!supports(request).supported) {
        return failure(id(), "resource.unsupported_media_type",
            "The XLSX processor requires an XLSX source and workbook-json representation.");
    }
    if (!is_zip(request.source_bytes)) {
        return failure(id(), "resource.invalid_document", "The source is not a recognizable XLSX package.");
    }
    if (executable_.empty() || script_.empty()) {
        return failure(id(), "resource.processor_unavailable",
            "The XLSX processor executable and script must be configured.");
    }
    common_agent_sandbox_request command;
    command.operation_id = context_.operation_id;
    command.project_id = context_.workspace.project_id;
    command.workspace_id = context_.workspace.workspace_id;
    command.execution_class = "resource.processor.xlsx";
    command.workspace.input_resources.push_back(request.source);
    command.limits.timeout_ms = 120000;
    command.limits.max_output_bytes = request.limits.max_output_bytes > 0
        ? request.limits.max_output_bytes : 4 * 1024 * 1024;
    command.network = common_agent_sandbox_network_scope::none;
    command.filesystem = common_agent_sandbox_filesystem_scope::artifact_write;
    command.artifacts.collect = true;
    command.artifacts.max_bytes = command.limits.max_output_bytes;
    const std::string output_name = "xlsx-workbook.json";
    command.artifacts.paths.push_back(output_name);
    command.command.program = executable_;
    command.command.working_directory = "/workspace/source";
    command.command.arguments = {
        script_, "/workspace/source/" + request.source.name,
        "/workspace/artifacts/" + output_name,
    };

    common_agent_sandbox_result sandbox_result;
    std::vector<agent_resource_processing_host_artifact> artifacts;
    std::string error;
    if (!host_.execute(context_, std::move(command), sandbox_result, artifacts, error)) {
        return failure(id(), "resource.processor_execution_failed", std::move(error));
    }
    if (sandbox_result.status != common_agent_sandbox_status::completed) {
        return failure(id(), sandbox_result.status == common_agent_sandbox_status::timed_out
            ? "resource.processing_timeout" : "resource.processor_failed",
            sandbox_result.error.empty() ? "XLSX processing did not complete." : sandbox_result.error);
    }
    if (artifacts.size() != 1 || artifacts.front().bytes.empty()) {
        return failure(id(), "resource.output_invalid", "XLSX processor did not return one bounded workbook envelope.");
    }
    agent_resource_processing_result result;
    result.success = true;
    result.processor_id = id();
    result.safe_summary = "XLSX workbook normalized into a bounded worksheet envelope.";
    agent_resource_processing_output output;
    output.name = artifacts.front().name;
    output.description = "Structured workbook representation derived from the authoritative XLSX resource.";
    output.mime_type = "application/json";
    output.bytes = artifacts.front().bytes;
    output.metadata.purpose = "structured XLSX workbook representation";
    output.metadata.content_summary = "Bounded worksheet metadata and rows for host dataset import.";
    output.metadata.usage_hint = "Pass to the host-owned worksheet dataset importer; it is not itself a dataset.";
    output.metadata.limitations = "Formula evaluation and workbook visual fidelity are not provided by this first slice.";
    output.lineage.parent_uri = request.source.uri;
    output.lineage.chunk_index = 0;
    output.lineage.chunk_count = 1;
    output.lineage.byte_length = request.source.size_bytes;
    output.lineage.derivation = "resource.process:" + id();
    result.outputs.push_back(std::move(output));
    return result;
}
