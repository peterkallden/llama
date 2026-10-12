#include "sandbox-kubernetes-runtime.h"

#include <sheredom/subprocess.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace fs = std::filesystem;
using json = nlohmann::ordered_json;

void append_limited(std::string & output, const char * data, size_t size, size_t limit) {
    if (output.size() >= limit) return;
    output.append(data, std::min(size, limit - output.size()));
}

bool valid_working_directory(const std::string & path) {
    return path.empty() || path == "/workspace/source" ||
        path == "/workspace/writable" || path == "/workspace/artifacts";
}

std::string safe_job_name(const std::string & operation_id) {
    std::string name = "llama-agent-";
    for (const char value : operation_id) {
        if ((value >= 'a' && value <= 'z') || (value >= '0' && value <= '9') || value == '-') {
            name.push_back(value);
        } else if (value >= 'A' && value <= 'Z') {
            name.push_back(static_cast<char>(value - 'A' + 'a'));
        } else {
            name.push_back('-');
        }
    }
    while (!name.empty() && name.back() == '-') name.pop_back();
    if (name.size() > 63) name.resize(63);
    while (!name.empty() && name.back() == '-') name.pop_back();
    return name.empty() ? "llama-agent-operation" : name;
}

bool add_host_path_volume(
        json & volumes,
        json & mounts,
        const std::string & path,
        const char * name,
        const char * target,
        bool readonly,
        std::string & error) {
    if (path.empty()) return true;
    std::error_code fs_error;
    if (!fs::is_directory(path, fs_error) || fs_error) {
        error = "Kubernetes sandbox mount source is not an accessible directory: " + path;
        return false;
    }
    volumes.push_back({
        {"name", name},
        {"hostPath", {{"path", fs::absolute(path).string()}, {"type", "Directory"}}},
    });
    mounts.push_back({
        {"name", name},
        {"mountPath", target},
        {"readOnly", readonly},
    });
    return true;
}

std::string safe_volume_name(const std::string & value) {
    std::string name;
    for (const char character : value) {
        const bool safe =
            (character >= 'a' && character <= 'z') ||
            (character >= '0' && character <= '9') || character == '-';
        name.push_back(safe ? character : '-');
    }
    while (!name.empty() && name.front() == '-') name.erase(name.begin());
    while (!name.empty() && name.back() == '-') name.pop_back();
    if (name.empty()) name = "workspace";
    if (name.size() > 40) name.resize(40);
    while (!name.empty() && name.back() == '-') name.pop_back();
    return name;
}

json pvc_manifest(
        const std::string & name,
        const std::string & namespace_name,
        const std::string & storage_class,
        const std::string & size) {
    json spec = {
        {"accessModes", json::array({"ReadWriteOnce"})},
        {"resources", {{"requests", {{"storage", size}}}}},
    };
    if (!storage_class.empty()) spec["storageClassName"] = storage_class;
    return {
        {"apiVersion", "v1"},
        {"kind", "PersistentVolumeClaim"},
        {"metadata", {{"name", name}, {"namespace", namespace_name}}},
        {"spec", spec},
    };
}

bool run_kubectl(
        const common_agent_kubernetes_sandbox_config & config,
        const std::vector<std::string> & arguments,
        uint32_t timeout_ms,
        size_t output_limit,
        std::string & output,
        int & exit_code,
        bool & timed_out,
        std::string & error) {
    std::vector<std::string> args;
    args.push_back(config.executable.empty() ? "kubectl" : config.executable);
    if (!config.kubeconfig.empty()) {
        args.push_back("--kubeconfig");
        args.push_back(config.kubeconfig);
    }
    if (!config.context.empty()) {
        args.push_back("--context");
        args.push_back(config.context);
    }
    if (config.insecure_skip_tls_verify) args.push_back("--insecure-skip-tls-verify=true");
    args.insert(args.end(), arguments.begin(), arguments.end());
    std::vector<char *> argv;
    argv.reserve(args.size() + 1);
    for (auto & arg : args) argv.push_back(arg.data());
    argv.push_back(nullptr);

    subprocess_s process{};
    const int options = subprocess_option_combined_stdout_stderr |
        subprocess_option_enable_async | subprocess_option_no_window |
        subprocess_option_inherit_environment | subprocess_option_search_user_path;
    if (subprocess_create(argv.data(), options, &process) != 0) {
        error = "unable to start kubectl process";
        return false;
    }

    const auto started = std::chrono::steady_clock::now();
    char buffer[4096];
    timed_out = false;
    output.clear();
    while (subprocess_alive(&process)) {
        const unsigned count = subprocess_read_stdout(&process, buffer, sizeof(buffer));
        append_limited(output, buffer, count, output_limit);
        if (timeout_ms != 0 && std::chrono::steady_clock::now() - started >=
                std::chrono::milliseconds(timeout_ms)) {
            timed_out = true;
            subprocess_terminate(&process);
            break;
        }
        if (count == 0) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    while (const unsigned count = subprocess_read_stdout(&process, buffer, sizeof(buffer))) {
        append_limited(output, buffer, count, output_limit);
    }
    subprocess_join(&process, &exit_code);
    subprocess_destroy(&process);
    error.clear();
    return true;
}

void collect_artifacts(
        const common_agent_sandbox_request & request,
        common_agent_sandbox_result & result) {
    if (!request.artifacts.collect || request.workspace.artifact_path.empty()) return;
    std::error_code fs_error;
    const fs::path root(request.workspace.artifact_path);
    if (!fs::is_directory(root, fs_error) || fs_error) return;
    size_t collected = 0;
    for (const auto & entry : fs::recursive_directory_iterator(root, fs_error)) {
        if (fs_error || collected >= request.artifacts.max_bytes) break;
        if (!entry.is_regular_file(fs_error) || fs_error) continue;
        const auto size = static_cast<size_t>(entry.file_size(fs_error));
        if (fs_error || collected + size > request.artifacts.max_bytes) continue;
        const auto relative = fs::relative(entry.path(), root, fs_error);
        if (fs_error) continue;
        common_runtime_resource_ref artifact;
        artifact.uri = "artifact://" + request.operation_id + "/" + relative.generic_string();
        artifact.name = relative.generic_string();
        artifact.size_bytes = size;
        artifact.scope = common_runtime_resource_scope::turn;
        result.artifacts.push_back(std::move(artifact));
        collected += size;
    }
}

} // namespace

bool common_agent_sandbox_kubernetes_runtime::execute(
        const common_agent_sandbox_request & request,
        common_agent_sandbox_result & result,
        std::string & error) {
    result = {};
    const std::string job = safe_job_name(request.operation_id);
    result.backend_execution_id = "kubernetes/job/" + job;
    if (request.operation_id.empty() || request.command.program.empty()) {
        error = "Kubernetes sandbox requires an operation id and program";
        result.error = error;
        return false;
    }
    if (!common_agent_sandbox_validate_capabilities(request, capabilities(), error)) {
        result.error = error;
        return false;
    }
    if (request.network != common_agent_sandbox_network_scope::none) {
        error = "Kubernetes sandbox currently supports only network=none";
        result.error = error;
        return false;
    }
    if (!valid_working_directory(request.command.working_directory)) {
        error = "Kubernetes sandbox working directory must use a sandbox workspace path";
        result.error = error;
        return false;
    }
    if (request.image.empty()) {
        error = "Kubernetes sandbox requires an image";
        result.error = error;
        return false;
    }

    if (request.workspace.source_path.empty() || request.workspace.writable_path.empty() ||
            request.workspace.artifact_path.empty() ||
            !fs::is_directory(request.workspace.source_path) ||
            !fs::is_directory(request.workspace.writable_path) ||
            !fs::is_directory(request.workspace.artifact_path)) {
        error = "Kubernetes sandbox requires host workspace directories for PVC materialization";
        result.error = error;
        return false;
    }
    const std::string identity = safe_volume_name(
        request.project_id.empty() ? request.workspace_id : request.project_id);
    const std::string workspace_claim = "llama-agent-workspace-" + identity;
    const std::string artifact_claim = "llama-agent-artifacts-" + identity;
    const std::string operation_path = identity + "/" + safe_volume_name(request.operation_id);
    const std::string staging_pod = safe_job_name(request.operation_id + "-stage");

    const json claims = {
        {"apiVersion", "v1"},
        {"kind", "List"},
        {"items", json::array({
            pvc_manifest(workspace_claim, config.namespace_name, config.storage_class, config.workspace_storage_size),
            pvc_manifest(artifact_claim, config.namespace_name, config.storage_class, config.artifact_storage_size),
        })},
    };
    const auto manifest_path = fs::temp_directory_path() / (job + ".json");
    {
        std::ofstream file(manifest_path);
        if (!file) {
            error = "unable to create Kubernetes PVC manifest";
            result.error = error;
            return false;
        }
        file << claims.dump(2);
    }
    const auto cleanup_manifest = [&]() { std::error_code ignored; fs::remove(manifest_path, ignored); };
    const auto namespace_args = [&]() {
        return std::vector<std::string>{"-n", config.namespace_name};
    };
    std::string output;
    int exit_code = -1;
    bool timed_out = false;
    auto apply_claims = namespace_args();
    apply_claims.insert(apply_claims.end(), {"apply", "-f", manifest_path.string()});
    if (config.insecure_skip_tls_verify) apply_claims.push_back("--validate=false");
    if (!run_kubectl(config, apply_claims, 10000, request.limits.max_output_bytes, output, exit_code, timed_out, error) || exit_code != 0) {
        result.error = error.empty() ? output : error;
        cleanup_manifest();
        return false;
    }

    const std::string staging_command =
        "mkdir -p /mnt/work/" + operation_path + "/source /mnt/work/" + operation_path +
        "/writable /mnt/artifacts/" + operation_path +
        " && chown -R 65532:65532 /mnt/work/" + operation_path + "/writable /mnt/artifacts/" + operation_path +
        " && sleep 3600";
    json staging_container = json::object();
    staging_container["name"] = "staging";
    staging_container["image"] = config.staging_image;
    staging_container["command"] = json::array({"sh", "-c"});
    staging_container["args"] = json::array({staging_command});
    staging_container["volumeMounts"] = json::array({
        {{"name", "workspace"}, {"mountPath", "/mnt/work"}},
        {{"name", "artifacts"}, {"mountPath", "/mnt/artifacts"}},
    });
    json staging_spec = json::object();
    staging_spec["restartPolicy"] = "Never";
    staging_spec["containers"] = json::array({staging_container});
    staging_spec["volumes"] = json::array({
        {{"name", "workspace"}, {"persistentVolumeClaim", {{"claimName", workspace_claim}}}},
        {{"name", "artifacts"}, {"persistentVolumeClaim", {{"claimName", artifact_claim}}}},
    });
    const json staging_manifest = {
        {"apiVersion", "v1"},
        {"kind", "Pod"},
        {"metadata", {{"name", staging_pod}, {"namespace", config.namespace_name}}},
        {"spec", staging_spec},
    };
    {
        std::ofstream file(manifest_path);
        if (!file) {
            error = "unable to create Kubernetes staging manifest";
            result.error = error;
            cleanup_manifest();
            return false;
        }
        file << staging_manifest.dump(2);
    }
    auto apply_staging = namespace_args();
    apply_staging.insert(apply_staging.end(), {"apply", "-f", manifest_path.string()});
    if (config.insecure_skip_tls_verify) apply_staging.push_back("--validate=false");
    if (!run_kubectl(config, apply_staging, 10000, request.limits.max_output_bytes, output, exit_code, timed_out, error) || exit_code != 0) {
        result.error = error.empty() ? output : error;
        cleanup_manifest();
        return false;
    }
    auto wait_staging = namespace_args();
    wait_staging.insert(wait_staging.end(), {"wait", "--for=condition=Ready", "pod/" + staging_pod, "--timeout=60s"});
    if (!run_kubectl(config, wait_staging, config.staging_timeout_ms, request.limits.max_output_bytes, output, exit_code, timed_out, error) || exit_code != 0) {
        result.error = error.empty() ? output : error;
        cleanup_manifest();
        return false;
    }
    auto copy_to_pvc = [&](const std::string & local, const std::string & remote) {
        auto args = namespace_args();
        const auto local_path = fs::relative(fs::absolute(local), fs::current_path()).generic_string();
        args.insert(args.end(), {"cp", local_path + "/.", staging_pod + ":" + remote});
        return run_kubectl(config, args, 120000, request.limits.max_output_bytes, output, exit_code, timed_out, error) && exit_code == 0;
    };
    if (!copy_to_pvc(request.workspace.source_path, "/mnt/work/" + operation_path + "/source")) {
        result.error = error.empty() ? output : error;
        cleanup_manifest();
        return false;
    }
    auto copy_from_pvc = [&](const std::string & remote, const std::string & local) {
        auto args = namespace_args();
        const auto local_path = fs::relative(fs::absolute(local), fs::current_path()).generic_string();
        args.insert(args.end(), {"cp", staging_pod + ":" + remote + "/.", local_path});
        return run_kubectl(config, args, 120000, request.limits.max_output_bytes, output, exit_code, timed_out, error) && exit_code == 0;
    };
    cleanup_manifest();

    json volumes = json::array({
        {{"name", "workspace"}, {"persistentVolumeClaim", {{"claimName", workspace_claim}}}},
        {{"name", "artifacts"}, {"persistentVolumeClaim", {{"claimName", artifact_claim}}}},
    });
    json mounts = json::array({
        {{"name", "workspace"}, {"mountPath", "/workspace/source"}, {"subPath", operation_path + "/source"}, {"readOnly", true}},
        {{"name", "workspace"}, {"mountPath", "/workspace/writable"}, {"subPath", operation_path + "/writable"},
            {"readOnly", request.filesystem != common_agent_sandbox_filesystem_scope::workspace_write}},
        {{"name", "artifacts"}, {"mountPath", "/workspace/artifacts"}, {"subPath", operation_path + "/artifacts"},
            {"readOnly", request.filesystem != common_agent_sandbox_filesystem_scope::artifact_write &&
                request.filesystem != common_agent_sandbox_filesystem_scope::workspace_write}},
    });

    json container = {
        {"name", "sandbox"},
        {"image", request.image},
        {"imagePullPolicy", "IfNotPresent"},
        {"command", json::array({request.command.program})},
        {"args", request.command.arguments},
        {"workingDir", request.command.working_directory.empty() ? "/workspace/source" : request.command.working_directory},
        {"volumeMounts", mounts},
        {"securityContext", {
            {"allowPrivilegeEscalation", false},
            {"readOnlyRootFilesystem", true},
            {"runAsNonRoot", true},
            {"runAsUser", 65532},
            {"runAsGroup", 65532},
            {"capabilities", {{"drop", json::array({"ALL"})}}},
        }},
    };
    if (request.limits.memory_bytes != 0 || request.limits.cpu_count != 0) {
        json resource_limits = json::object();
        if (request.limits.cpu_count != 0) resource_limits["cpu"] = std::to_string(request.limits.cpu_count);
        if (request.limits.memory_bytes != 0) resource_limits["memory"] = std::to_string(request.limits.memory_bytes);
        container["resources"] = {{"limits", resource_limits}};
    }
    json pod_spec = {
        {"restartPolicy", "Never"},
        {"automountServiceAccountToken", false},
        {"securityContext", {{"fsGroup", 65532}}},
        {"containers", json::array({container})},
        {"volumes", volumes},
    };
    if (!config.service_account.empty()) pod_spec["serviceAccountName"] = config.service_account;
    if (!config.runtime_class.empty()) pod_spec["runtimeClassName"] = config.runtime_class;
    json pod_template = json::object();
    pod_template["metadata"] = {
        {"labels", {{"app", "llama-agent-sandbox"}, {"job", job}}},
    };
    pod_template["spec"] = pod_spec;
    json job_spec = json::object();
    job_spec["backoffLimit"] = 0;
    job_spec["ttlSecondsAfterFinished"] = 300;
    job_spec["template"] = pod_template;
    const json job_manifest = {
        {"apiVersion", "batch/v1"},
        {"kind", "Job"},
        {"metadata", {{"name", job}, {"namespace", config.namespace_name}}},
        {"spec", job_spec},
    };
    const json network_policy = {
        {"apiVersion", "networking.k8s.io/v1"},
        {"kind", "NetworkPolicy"},
        {"metadata", {{"name", job + "-deny-egress"}, {"namespace", config.namespace_name}}},
        {"spec", {
            {"podSelector", {{"matchLabels", {{"job", job}}}}},
            {"policyTypes", json::array({"Egress"})},
            {"egress", json::array()},
        }},
    };
    const json manifest = {
        {"apiVersion", "v1"},
        {"kind", "List"},
        {"items", json::array({job_manifest, network_policy})},
    };

    const auto started = std::chrono::steady_clock::now();
    {
        std::ofstream file(manifest_path);
        if (!file) {
            error = "unable to create Kubernetes Job manifest";
            result.error = error;
            cleanup_manifest();
            return false;
        }
        file << manifest.dump(2);
    }
    auto apply_args = namespace_args();
    apply_args.insert(apply_args.end(), {"apply", "-f", manifest_path.string()});
    if (config.insecure_skip_tls_verify) apply_args.push_back("--validate=false");
    if (!run_kubectl(config, apply_args, 10000, request.limits.max_output_bytes, output, exit_code, timed_out, error) || exit_code != 0) {
        result.error = error.empty() ? output : error;
        cleanup_manifest();
        return false;
    }
    auto wait_args = namespace_args();
    wait_args.insert(wait_args.end(), {"wait", "--for=condition=complete", "job/" + job,
        "--timeout=" + std::to_string(std::max<uint32_t>(1, request.limits.timeout_ms / 1000)) + "s"});
    const bool waited = run_kubectl(config, wait_args, request.limits.timeout_ms,
        request.limits.max_output_bytes, output, exit_code, timed_out, error);
    const int wait_exit_code = exit_code;
    result.usage.wall_time_ms = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started).count());
    if (!waited || timed_out) {
        result.status = common_agent_sandbox_status::timed_out;
        result.error = "Kubernetes sandbox timed out";
    } else {
        auto logs_args = namespace_args();
        logs_args.insert(logs_args.end(), {"logs", "job/" + job});
        std::string logs;
        bool logs_timed_out = false;
        run_kubectl(config, logs_args, 10000, request.limits.max_output_bytes, logs, exit_code, logs_timed_out, error);
        result.stdout_excerpt = logs;
        result.usage.output_bytes = logs.size();
        result.exit_code = exit_code;
        if (wait_exit_code == 0) {
            result.status = common_agent_sandbox_status::completed;
            if (!copy_from_pvc("/mnt/artifacts/" + operation_path + "/artifacts", request.workspace.artifact_path)) {
                result.status = common_agent_sandbox_status::failed;
                result.error = error.empty() ? output : error;
            }
            collect_artifacts(request, result);
        } else {
            result.status = common_agent_sandbox_status::failed;
            result.error = logs.empty() ? output : logs;
        }
    }
    if (config.cleanup) {
        auto delete_args = namespace_args();
        delete_args.insert(delete_args.end(), {"delete", "job/" + job, "pod/" + staging_pod,
            "networkpolicy/" + job + "-deny-egress", "--ignore-not-found=true", "--wait=false"});
        std::string ignored_output;
        bool ignored_timeout = false;
        run_kubectl(config, delete_args, 10000, 4096, ignored_output, exit_code, ignored_timeout, error);
    }
    const bool project_scope = !request.project_id.empty();
    const bool delete_claims = config.pvc_retention == "operation" ||
        (config.pvc_retention == "session" && !project_scope);
    if (delete_claims) {
        auto delete_claim_args = namespace_args();
        delete_claim_args.insert(delete_claim_args.end(), {"delete", "pvc", workspace_claim, artifact_claim,
            "--ignore-not-found=true", "--wait=false"});
        std::string ignored_output;
        bool ignored_timeout = false;
        run_kubectl(config, delete_claim_args, 10000, 4096, ignored_output, exit_code, ignored_timeout, error);
    }
    cleanup_manifest();
    error.clear();
    return true;
}
