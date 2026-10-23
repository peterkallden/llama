#pragma once

#include "agent/sandbox/sandbox-contract.h"
#include "agent/sandbox/sandbox-policy.h"
#include "agent/sandbox/sandbox-runtime.h"
#include "agent/workspace-manager.h"
#include "agent/tooling/contracts/tool-runtime-contract.h"
#include "hash/hash.h"

#include <nlohmann/json.hpp>

#include <utility>
#include <filesystem>
#include <fstream>
#include <set>
#include <ctime>

class common_agent_sandbox_tool_helper {
public:
    common_agent_sandbox_tool_helper(
            common_agent_sandbox_runtime & runtime,
            common_agent_sandbox_policy policy)
        : runtime(runtime), policy(std::move(policy)) {}

    common_tool_execution_result run(
            const common_agent_sandbox_request & request) {
        std::string error;
        if (!policy.validate(request, error)) {
            return common_tool_execution_result::failure(
                "sandbox.policy_rejected",
                common_tool_failure_class::policy,
                false,
                "The sandbox request was rejected by host policy.",
                error);
        }

        auto resolved_request = request;
        if (resolved_request.image.empty()) resolved_request.image = policy.image;
        common_agent_sandbox_result result;
        if (!runtime.execute(resolved_request, result, error)) {
            return common_tool_execution_result::failure(
                "sandbox.execution_failed",
                common_tool_failure_class::execution,
                false,
                "The sandbox backend failed to execute the request.",
                error);
        }
        if (result.status != common_agent_sandbox_status::completed) {
            if (result.status == common_agent_sandbox_status::backend_unavailable) {
                return common_tool_execution_result::failure(
                    "sandbox.backend_unavailable",
                    common_tool_failure_class::execution,
                    false,
                    "No sandbox execution backend is configured for this host.",
                    result.error);
            }
            const auto failure_class = result.status == common_agent_sandbox_status::timed_out
                ? common_tool_failure_class::timeout
                : common_tool_failure_class::execution;
            return common_tool_execution_result::failure(
                result.status == common_agent_sandbox_status::timed_out
                    ? "sandbox.timed_out"
                    : "sandbox.execution_failed",
                failure_class,
                false,
                "The sandbox did not complete successfully.",
                result.error);
        }

        nlohmann::ordered_json output = {
            {"status", common_agent_sandbox_status_name(result.status)},
            {"exit_code", result.exit_code},
            {"stdout", result.stdout_excerpt},
            {"stderr", result.stderr_excerpt},
            {"backend_execution_id", result.backend_execution_id},
        };
        if (!request.workspace.source_snapshot_uri.empty()) {
            output["source_snapshot"] = {
                {"resource_uri", request.workspace.source_snapshot_uri},
                {"manifest_sha256", request.workspace.source_snapshot_sha256},
                {"materialized", true},
            };
        }
        return common_tool_execution_result::success(
            output.dump(),
            "Sandbox execution completed.",
            result.artifacts);
    }

    common_tool_execution_result run_for_workspace(
            const common_agent_workspace_context & context,
            const std::string & operation_id,
            common_agent_sandbox_request request,
            common_agent_workspace_operation * operation_out = nullptr) {
        if (workspace_manager == nullptr) {
            return common_tool_execution_result::failure(
                "sandbox.workspace_manager_unavailable",
                common_tool_failure_class::execution,
                false,
                "The host has not configured a workspace manager.",
                "sandbox workspace manager is required for workspace execution");
        }
        common_agent_workspace_operation operation;
        std::string error;
        if (!prepare_workspace_request(
                context, operation_id, request, operation, operation_out, error)) {
            const bool materialization_failed =
                error.rfind("sandbox resource materialization failed:", 0) == 0;
            return common_tool_execution_result::failure(
                materialization_failed
                    ? "sandbox.resource_materialization_failed"
                    : "sandbox.workspace_setup_failed",
                common_tool_failure_class::execution,
                false,
                materialization_failed
                    ? "A sandbox input resource could not be materialized."
                    : "The sandbox workspace could not be prepared.",
                error);
        }
        const auto started_at = static_cast<int64_t>(std::time(nullptr));
        common_agent_sandbox_result sandbox_result;
        const bool executed = run_raw(request, sandbox_result, error);
        const auto completed_at = static_cast<int64_t>(std::time(nullptr));
        std::vector<common_runtime_resource_ref> retained_outputs;
        if (executed && resource_store != nullptr) {
            if (!sandbox_result.stdout_excerpt.empty()) {
                common_runtime_resource_ref output_ref;
                std::string output_error;
                if (persist_output(context, operation_id, "stdout", sandbox_result.stdout_excerpt, output_ref, output_error))
                    retained_outputs.push_back(std::move(output_ref));
            }
            if (!sandbox_result.stderr_excerpt.empty()) {
                common_runtime_resource_ref output_ref;
                std::string output_error;
                if (persist_output(context, operation_id, "stderr", sandbox_result.stderr_excerpt, output_ref, output_error))
                    retained_outputs.push_back(std::move(output_ref));
            }
        }
        nlohmann::ordered_json receipt = {
            {"schema", "development-execution-receipt.v1"},
            {"operation_id", request.operation_id},
            {"workspace_id", context.workspace_id},
            {"project_id", context.project_id},
            {"execution_class", request.execution_class},
            {"logical_target", request.logical_target},
            {"configuration", request.requested_configuration},
            {"test_filter", request.requested_test_filter},
            {"program", request.command.program},
            {"started_at", started_at},
            {"completed_at", completed_at},
            {"status", executed ? common_agent_sandbox_status_name(sandbox_result.status) : "failed"},
            {"exit_code", executed ? nlohmann::ordered_json(sandbox_result.exit_code) : nlohmann::ordered_json(nullptr)},
            {"backend_execution_id", executed ? sandbox_result.backend_execution_id : std::string{}},
            {"source_snapshot", request.workspace.source_snapshot_uri.empty()
                ? nlohmann::ordered_json(nullptr)
                : nlohmann::ordered_json{{"resource_uri", request.workspace.source_snapshot_uri},
                    {"manifest_sha256", request.workspace.source_snapshot_sha256}, {"materialized", true}}},
            {"stdout_sha256", executed ? "sha256:" + hash_sha256_hex(sandbox_result.stdout_excerpt.data(), sandbox_result.stdout_excerpt.size()) : std::string{}},
            {"stderr_sha256", executed ? "sha256:" + hash_sha256_hex(sandbox_result.stderr_excerpt.data(), sandbox_result.stderr_excerpt.size()) : std::string{}},
            {"output_resources", nlohmann::ordered_json::array()},
            {"usage", executed ? nlohmann::ordered_json{{"wall_time_ms", sandbox_result.usage.wall_time_ms},
                {"cpu_time_ms", sandbox_result.usage.cpu_time_ms}, {"peak_memory_bytes", sandbox_result.usage.peak_memory_bytes},
                {"output_bytes", sandbox_result.usage.output_bytes}} : nlohmann::ordered_json::object()},
            {"artifacts", nlohmann::ordered_json::array()},
        };
        for (const auto & output_ref : retained_outputs) {
            receipt["output_resources"].push_back({{"uri", output_ref.uri}, {"name", output_ref.name},
                {"size_bytes", output_ref.size_bytes}, {"content_hash", output_ref.metadata.content_hash}});
        }
        for (const auto & artifact : sandbox_result.artifacts) {
            receipt["artifacts"].push_back({{"uri", artifact.uri}, {"name", artifact.name},
                {"mime_type", artifact.mime_type}, {"size_bytes", artifact.size_bytes}, {"sha256", artifact.metadata.content_hash}});
        }
        common_runtime_resource_ref receipt_ref;
        std::string receipt_error;
        const bool retained = persist_receipt(context, operation_id, receipt.dump(), receipt_ref, receipt_error);
        if (!executed || sandbox_result.status != common_agent_sandbox_status::completed) {
            const auto code = !executed ? "sandbox.execution_failed" :
                sandbox_result.status == common_agent_sandbox_status::timed_out ? "sandbox.timed_out" :
                sandbox_result.status == common_agent_sandbox_status::cancelled ? "sandbox.cancelled" : "sandbox.execution_failed";
            const auto failure_class = !executed ? common_tool_failure_class::execution :
                sandbox_result.status == common_agent_sandbox_status::timed_out ? common_tool_failure_class::timeout : common_tool_failure_class::execution;
            auto failure = common_tool_execution_result::failure(code, failure_class, false,
                "The sandbox did not complete successfully.", executed ? sandbox_result.error : error);
            if (retained) {
                failure.resource_refs = retained_outputs;
                failure.resource_refs.push_back(receipt_ref);
            }
            else if (receipt_error.size()) failure.raw_diagnostic += (failure.raw_diagnostic.empty() ? "" : "; ") + std::string("receipt retention failed: ") + receipt_error;
            return failure;
        }
        if (!retained) return common_tool_execution_result::failure(
            "sandbox.receipt_retention_failed", common_tool_failure_class::execution, false,
            "The verification completed, but its durable execution receipt could not be saved.", receipt_error);
        receipt["summary"] = "Configured development operation completed; inspect the retained receipt and output resources for source and result provenance.";
        receipt["receipt_uri"] = receipt_ref.uri;
        receipt["stdout"] = sandbox_result.stdout_excerpt;
        receipt["stderr"] = sandbox_result.stderr_excerpt;
        retained_outputs.push_back(receipt_ref);
        return common_tool_execution_result::success(receipt.dump(), "Sandbox execution completed with a retained receipt.", std::move(retained_outputs));
    }

    bool run_raw_for_workspace(
            const common_agent_workspace_context & context,
            const std::string & operation_id,
            common_agent_sandbox_request request,
            common_agent_sandbox_result & result,
            std::string & error,
            common_agent_workspace_operation * operation_out = nullptr) {
        common_agent_workspace_operation operation;
        if (!prepare_workspace_request(
                context, operation_id, request, operation, operation_out, error)) {
            return false;
        }
        return run_raw(request, result, error);
    }

    bool run_raw(
            const common_agent_sandbox_request & request,
            common_agent_sandbox_result & result,
            std::string & error) {
        if (!policy.validate(request, error)) return false;
        auto resolved_request = request;
        if (resolved_request.image.empty()) resolved_request.image = policy.image;
        if (!runtime.execute(resolved_request, result, error)) return false;
        return true;
    }

    void set_workspace_manager(common_agent_workspace_manager * manager) {
        workspace_manager = manager;
    }

    void set_resource_store(
            agent_resource_store * store,
            agent_resource_read_authority authority = {}) {
        resource_store = store;
        resource_authority = std::move(authority);
    }

private:
    bool persist_receipt(
            const common_agent_workspace_context & context,
            const std::string & operation_id,
            const std::string & contents,
            common_runtime_resource_ref & output,
            std::string & error) const {
        if (resource_store == nullptr) {
            error = "resource store is unavailable";
            return false;
        }
        agent_resource_put_request request;
        request.name = "development-receipt-" + hash_sha256_hex(operation_id.data(), operation_id.size()).substr(0, 20) + ".json";
        request.description = "Immutable receipt for one configured development build or test execution.";
        request.mime_type = "application/vnd.llama-agent.development-receipt+json";
        request.text = contents;
        request.scope = context.project_id.empty()
            ? common_runtime_resource_scope::session : common_runtime_resource_scope::project;
        request.namespace_id = context.namespace_id;
        request.session_id = context.session_id;
        request.project_id = context.project_id;
        request.turn_id = context.turn_id;
        request.source_provider = "native";
        request.source_tool = "development.build-test";
        request.created_at = static_cast<int64_t>(std::time(nullptr));
        request.metadata.purpose = "development_execution_receipt";
        request.metadata.content_summary = "Configured build or test status bound to an immutable source snapshot when one was supplied.";
        request.metadata.content_hash = "sha256:" + hash_sha256_hex(contents.data(), contents.size());
        request.lineage.derivation = "development-execution-receipt:v1";
        agent_resource_descriptor descriptor;
        if (!resource_store->put_text(request, descriptor, error) || descriptor.uri.empty()) return false;
        output = descriptor;
        error.clear();
        return true;
    }

    bool persist_output(
            const common_agent_workspace_context & context,
            const std::string & operation_id,
            const std::string & stream_name,
            const std::string & contents,
            common_runtime_resource_ref & output,
            std::string & error) const {
        agent_resource_put_request request;
        request.name = "development-" + stream_name + "-" +
            hash_sha256_hex(operation_id.data(), operation_id.size()).substr(0, 20) + ".log";
        request.description = "Bounded output from one configured development build or test execution.";
        request.mime_type = "text/plain";
        request.text = contents;
        request.scope = context.project_id.empty()
            ? common_runtime_resource_scope::session : common_runtime_resource_scope::project;
        request.namespace_id = context.namespace_id;
        request.session_id = context.session_id;
        request.project_id = context.project_id;
        request.turn_id = context.turn_id;
        request.source_provider = "native";
        request.source_tool = "development.build-test";
        request.created_at = static_cast<int64_t>(std::time(nullptr));
        request.metadata.purpose = "development_execution_output";
        request.metadata.content_summary = "Bounded stdout or stderr excerpt from a host-approved developer execution.";
        request.metadata.content_hash = "sha256:" + hash_sha256_hex(contents.data(), contents.size());
        request.lineage.derivation = "development-execution-output:v1";
        agent_resource_descriptor descriptor;
        if (!resource_store->put_text(request, descriptor, error) || descriptor.uri.empty()) return false;
        output = descriptor;
        error.clear();
        return true;
    }

    bool prepare_workspace_request(
            const common_agent_workspace_context & context,
            const std::string & operation_id,
            common_agent_sandbox_request & request,
            common_agent_workspace_operation & operation,
            common_agent_workspace_operation * operation_out,
            std::string & error) {
        if (workspace_manager == nullptr) {
            error = "sandbox workspace manager is required for workspace execution";
            return false;
        }
        if (!workspace_manager->create_operation(context, operation_id, operation, error)) {
            return false;
        }
        request.operation_id = operation.operation_id;
        request.project_id = context.project_id;
        request.workspace_id = context.workspace_id;
        request.workspace.source_path = operation.source_path;
        request.workspace.writable_path = operation.writable_path;
        request.workspace.artifact_path = operation.artifact_path;
        if (resource_store != nullptr) {
            bool saw_source_snapshot = false;
            for (size_t index = 0; index < request.workspace.input_resources.size(); ++index) {
                auto resource = request.workspace.input_resources[index];
                agent_resource_descriptor descriptor;
                if (!resource_store->stat(resource.uri, resource_authority, descriptor, error)) return false;
                resource = descriptor;
                if (resource.mime_type == "application/vnd.llama-agent.code-source-snapshot+json") {
                    if (saw_source_snapshot) {
                        error = "only one source snapshot may be supplied to a development operation";
                        return false;
                    }
                    saw_source_snapshot = true;
                    if (!materialize_code_source_snapshot(operation, resource, request, error)) {
                        error = "sandbox resource materialization failed: " + error;
                        return false;
                    }
                    continue;
                }
                const auto file_name = resource.name.empty()
                    ? "input-" + std::to_string(index + 1) + ".txt"
                    : resource.name;
                std::string materialized_path;
                if (!workspace_manager->materialize_resource(
                        operation,
                        resource,
                        *resource_store,
                        resource_authority,
                        file_name,
                        request.limits.max_input_resource_bytes,
                        materialized_path,
                        error)) {
                    error = "sandbox resource materialization failed: " + error;
                    return false;
                }
            }
            if (request.require_source_snapshot && !saw_source_snapshot) {
                error = "the required source_snapshot_ref does not resolve to a code source snapshot";
                return false;
            }
        } else if (request.require_source_snapshot) {
            error = "a resource store is required to resolve the requested source snapshot";
            return false;
        }
        if (operation_out != nullptr) *operation_out = operation;
        return true;
    }

    bool materialize_code_source_snapshot(
            const common_agent_workspace_operation & operation,
            const common_runtime_resource_ref & resource,
            common_agent_sandbox_request & request,
            std::string & error) const {
        constexpr size_t max_snapshot_bytes = 256 * 1024 * 1024;
        constexpr size_t max_snapshot_files = 8192;
        std::string serialized;
        if (!resource_store->read_text(resource.uri, resource_authority,
                max_snapshot_bytes, serialized, error)) return false;
        const auto snapshot = nlohmann::json::parse(serialized, nullptr, false);
        if (!snapshot.is_object() || snapshot.value("schema", std::string{}) != "code-source-snapshot.v1" ||
                !snapshot.value("complete", false) || !snapshot.contains("files") || !snapshot["files"].is_array() ||
                snapshot["files"].size() > max_snapshot_files) {
            error = "source snapshot is malformed, incomplete or exceeds the supported file count";
            return false;
        }
        size_t total_bytes = 0;
        std::set<std::string> seen_paths;
        const std::filesystem::path source_root(operation.source_path);
        for (const auto & file : snapshot["files"]) {
            if (!file.is_object() || !file.contains("path") || !file["path"].is_string() ||
                    !file.contains("resource_uri") || !file["resource_uri"].is_string() ||
                    !file.contains("bytes") || !file["bytes"].is_number_unsigned() ||
                    !file.contains("sha256") || !file["sha256"].is_string() ||
                    !file.contains("permissions") || !file["permissions"].is_number_unsigned()) {
                error = "source snapshot contains an invalid file entry";
                return false;
            }
            const std::string relative_text = file["path"].get<std::string>();
            const std::filesystem::path relative(relative_text);
            if (relative_text.empty() || relative.is_absolute() || relative.has_root_name() ||
                    relative_text.find('\\') != std::string::npos || seen_paths.count(relative.generic_string()) != 0) {
                error = "source snapshot contains an unsafe or duplicate path";
                return false;
            }
            for (const auto & component : relative) {
                if (component == "." || component == "..") {
                    error = "source snapshot path escapes the staged source root";
                    return false;
                }
            }
            seen_paths.insert(relative.generic_string());
            const auto declared_size = file["bytes"].get<uint64_t>();
            if (declared_size > request.limits.max_input_resource_bytes ||
                    declared_size > 64 * 1024 * 1024 || total_bytes > max_snapshot_bytes - declared_size) {
                error = "source snapshot exceeds its byte limits";
                return false;
            }
            std::string bytes;
            const auto uri = file["resource_uri"].get<std::string>();
            if (uri.rfind("resource://", 0) != 0 ||
                    !resource_store->read_bytes(uri, resource_authority,
                        static_cast<size_t>(declared_size), bytes, error)) return false;
            if (bytes.size() != declared_size || file["sha256"].get<std::string>() !=
                    "sha256:" + hash_sha256_hex(bytes.data(), bytes.size())) {
                error = "source snapshot file bytes do not match their declared size and hash";
                return false;
            }
            total_bytes += bytes.size();
            const auto output_path = source_root / relative;
            std::error_code fs_error;
            std::filesystem::create_directories(output_path.parent_path(), fs_error);
            if (fs_error || !std::filesystem::is_directory(output_path.parent_path(), fs_error) || fs_error) {
                error = "source snapshot directory could not be created";
                return false;
            }
            const auto canonical_root = std::filesystem::weakly_canonical(source_root, fs_error);
            if (fs_error) { error = "staged source root could not be resolved"; return false; }
            const auto canonical_parent = std::filesystem::weakly_canonical(output_path.parent_path(), fs_error);
            if (fs_error || (canonical_parent != canonical_root &&
                    canonical_parent.generic_string().rfind(canonical_root.generic_string() + "/", 0) != 0)) {
                error = "source snapshot path escapes the staged source root";
                return false;
            }
            if (std::filesystem::exists(output_path, fs_error) || fs_error) {
                error = "source snapshot path collides with an existing staged file";
                return false;
            }
            std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
            if (!output) { error = "source snapshot file could not be opened"; return false; }
            output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
            output.close();
            if (!output) { error = "source snapshot file could not be written"; return false; }
            const auto mode = file["permissions"].get<uint32_t>() & 0777u;
            std::filesystem::permissions(output_path,
                static_cast<std::filesystem::perms>(mode), std::filesystem::perm_options::replace, fs_error);
            if (fs_error) { error = "source snapshot file mode could not be applied"; return false; }
        }
        request.workspace.source_snapshot_uri = resource.uri;
        request.workspace.source_snapshot_sha256 = "sha256:" + hash_sha256_hex(serialized.data(), serialized.size());
        error.clear();
        return true;
    }

    common_agent_sandbox_runtime & runtime;
    common_agent_sandbox_policy policy;
    common_agent_workspace_manager * workspace_manager = nullptr;
    agent_resource_store * resource_store = nullptr;
    agent_resource_read_authority resource_authority;
};

class common_agent_sandbox_local_test_runtime final : public common_agent_sandbox_runtime {
public:
    bool execute(
            const common_agent_sandbox_request & request,
            common_agent_sandbox_result & result,
            std::string & error) override {
        if (request.command.program.empty()) {
            error = "local test sandbox requires a declared program";
            return false;
        }
        last_request = request;
        result = {};
        result.status = common_agent_sandbox_status::completed;
        result.exit_code = 0;
        result.stdout_excerpt = "local sandbox test backend";
        result.backend_execution_id = "local-test/" + request.operation_id;
        error.clear();
        return true;
    }

    common_agent_sandbox_request last_request;
};
