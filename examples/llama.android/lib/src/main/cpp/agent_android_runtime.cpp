#include "agent_android_runtime.h"

#include <memory>
#include <dlfcn.h>
#include <algorithm>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <filesystem>
#include <fstream>

#include "agent/runtime/agent-event-queue.h"
#include "agent/contracts/agent-request.h"
#include "common/agent/protocol/agent-jsonl.h"
#include "memory/sqlite/memory-sqlite.h"
#include "plan/sqlite/plan-sqlite.h"
#include "tools/agent/runtime/agent-runtime-control.h"
#include "tools/agent/runtime/agent-runtime-session-host.h"
#include "tools/agent/runtime/agent-session-context-store.h"
#include "tools/agent/runtime/agent-research-workspace-host.h"
#include "tools/agent/runtime/agent-http-inference.h"
#include "tools/agent/mcp/agent-mcp-resource-materializer.h"
#include "tools/agent/tooling/agent-tool-provider.h"
#include "tools/agent/resource/agent-resource-store.h"
#include "tools/agent/data/agent-data-store-sqlite.h"
#include "agent/tooling/adapters/adapter-bindings.h"
#include "agent_android_mcp_transport.h"
#include "agent_android_credentials.h"
#include "agent_android_resource_catalog.h"

namespace {
// Keep the local resource namespace unambiguous while preserving access to
// resources returned by an attached MCP provider under distinct names.
class android_mcp_client_view final : public agent_mcp_tool_client {
public:
    android_mcp_client_view(std::shared_ptr<agent_mcp_http_client> client, agent_resource_store * store,
            std::shared_ptr<void> store_owner)
        : client_(std::move(client)), store_(store), store_owner_(std::move(store_owner)) {}
    bool list_tools(const agent_tool_context & context, std::vector<mcp_agent_tool_definition> & tools,
            std::string & error) override {
        if (!client_->list_tools(context, tools, error)) return false;
        for (auto & tool : tools) if (local_family(tool.name)) tool.name = "remote_" + tool.name;
        return true;
    }
    bool call_tool(const agent_tool_context & context, const mcp_agent_tool_definition & tool,
            const std::string & arguments, mcp_agent_tool_call_result & result, std::string & error) override {
        auto original = tool;
        if (original.name.rfind("remote_", 0) == 0 && local_family(original.name.substr(7)))
            original.name.erase(0, 7);
        if (!client_->call_tool(context, original, arguments, result, error)) return false;
        std::string materialization_error;
        if (!agent_materialize_mcp_resources(*client_, *store_, context, "android-mcp", original.name, result, materialization_error)) {
            result = {};
            result.failure_code = "mcp.resource_materialization_failed";
            result.safe_summary = "The remote resource could not be saved completely for this session.";
            result.raw_diagnostic = materialization_error;
        }
        error.clear();
        return true;
    }
    bool read_resource(const std::string & uri, mcp_agent_resource_read_result & result, std::string & error) override {
        return client_->read_resource(uri, result, error);
    }
private:
    static bool local_family(const std::string & name) {
        return name.rfind("resource.", 0) == 0 || name.rfind("dataset.", 0) == 0 ||
            name.rfind("data.", 0) == 0 || name.rfind("statistics.", 0) == 0;
    }
    std::shared_ptr<agent_mcp_http_client> client_;
    agent_resource_store * store_;
    std::shared_ptr<void> store_owner_;
};

class android_http_resident final : public common_agent_runtime_resident_model {
public:
    common_agent_model_selection selection;
    std::shared_ptr<agent_mcp_http_transport> transport;
    std::shared_ptr<common_agent_credential_provider> credentials;
    std::string credential_ref;

    bool supports_json_schema() const override { return selection.supports_json_schema; }
    bool supports_native_tool_calls() const override { return selection.supports_native_tool_calls; }
    std::unique_ptr<common_agent_inference> create_inference() const override {
        auto channel = transport;
        auto provider = credentials;
        auto reference = credential_ref;
        return make_agent_http_inference(selection, [channel, provider, reference](
                const std::string & url, const std::string & body, uint32_t connect_ms,
                uint32_t request_ms, int & status, std::string & output, std::string & error) {
            if (!channel) { error = "Android HTTPS transport is unavailable"; return false; }
            agent_mcp_http_request request;
            request.url = url;
            request.body = body;
            request.connect_timeout_ms = connect_ms;
            request.request_timeout_ms = request_ms;
            request.max_result_bytes = 16U * 1024U * 1024U;
            request.headers["Content-Type"] = "application/json";
            if (!reference.empty()) {
                std::string secret;
                if (!provider || !provider->resolve(reference, secret, error) || secret.empty()) {
                    error = "HTTP model credential is unavailable";
                    return false;
                }
                request.headers["Authorization"] = "Bearer " + secret;
            }
            agent_mcp_http_response response;
            if (!channel->post(request, response, error)) return false;
            status = response.status;
            output = std::move(response.body);
            return true;
        });
    }
};

class android_http_loader final : public common_agent_runtime_model_loader {
public:
    std::shared_ptr<agent_mcp_http_transport> transport;
    std::shared_ptr<common_agent_credential_provider> credentials;
    std::string credential_ref;
    bool load(const common_agent_model_selection & selection,
            std::shared_ptr<common_agent_runtime_resident_model> & model,
            std::string & error) override {
        if (!transport || selection.backend != "http" || selection.endpoint.rfind("https://", 0) != 0 ||
                selection.model_id.empty() || selection.context_size_tokens == 0) {
            error = "Android HTTP model requires HTTPS, model ID and context size";
            return false;
        }
        if (!credential_ref.empty()) {
            std::string secret;
            if (!credentials || !credentials->resolve(credential_ref, secret, error) || secret.empty()) {
                error = "HTTP model credential is unavailable";
                return false;
            }
        }
        auto loaded = std::make_shared<android_http_resident>();
        loaded->selection = selection;
        loaded->transport = transport;
        loaded->credentials = credentials;
        loaded->credential_ref = credential_ref;
        model = std::move(loaded);
        error.clear();
        return true;
    }
};

struct android_agent_runtime_handle {
    android_agent_runtime_handle(std::string directory, std::string model)
        : storage_directory(std::move(directory)),
          model_path(std::move(model)),
          cancellation(std::make_shared<common_agent_runtime_cancellation_state>()) {}
    std::string storage_directory;
    std::string model_path;
    bool http_model = false;
    int max_output_tokens = 384;
    std::shared_ptr<common_agent_runtime_cancellation_state> cancellation;
    common_agent_event_queue events;
    common_memory_sqlite_store memory_store;
    common_plan_sqlite_store plan_store;
    common_agent_sqlite_data_store data_store;
    common_tool_catalog catalog;
    std::unique_ptr<agent_resource_store> resources;
    std::unique_ptr<common_agent_runtime_session_host> session_host;
    bool storage_ready = false;
    std::shared_ptr<agent_mcp_http_client> mcp_client;
    std::shared_ptr<agent_mcp_http_transport> android_https_transport;
    bool android_https_supported = false;
    std::shared_ptr<common_agent_credential_provider> android_credential_provider;
    bool android_credentials_supported = false;
    std::mutex mcp_mutex;
    std::mutex turn_mutex;
    std::thread turn_worker;
    bool turn_active = false;
    uint64_t next_request_id = 1;
    struct completed_turn {
        uint64_t request_id = 0;
        common_agent_runtime_session_host_turn_result result;
    };
    std::deque<completed_turn> completed_turns;
};

std::mutex handles_mutex;
std::unordered_map<jlong, std::shared_ptr<android_agent_runtime_handle>> handles;
jlong next_handle = 1;

std::string to_string(JNIEnv * env, jstring value) {
    if (!value) return {};
    const char * chars = env->GetStringUTFChars(value, nullptr);
    const std::string result = chars ? chars : "";
    env->ReleaseStringUTFChars(value, chars);
    return result;
}

std::string json_escape(const std::string & value) {
    std::string result;
    for (const char character : value) {
        if (character == '"') result += "\\\"";
        else if (character == '\\') result += "\\\\";
        else if (character == '\n') result += "\\n";
        else if (character == '\r') result += "\\r";
        else if (character == '\t') result += "\\t";
        else result += character;
    }
    return result;
}

std::shared_ptr<android_agent_runtime_handle> find_handle(jlong handle) {
    const auto it = handles.find(handle);
    return it == handles.end() ? nullptr : it->second;
}

std::string event_json(const common_agent_event & event) {
    common_agent_jsonl_event_entry wire_event;
    wire_event.type = common_agent_event_type_name(event.type);
    wire_event.event_type = wire_event.type;
    wire_event.detail = event.detail;
    wire_event.memory_id = event.memory_id;
    if (event.plan_id.has_value()) wire_event.plan_id = *event.plan_id;
    wire_event.step_id = event.step_id;
    wire_event.observation_id = event.observation_id;
    wire_event.tool_name = event.tool_name;
    wire_event.resource_uri = event.resource_uri;
    return common_agent_jsonl_make_event_message(wire_event).dump();
}

std::string result_json(uint64_t request_id, const common_agent_runtime_session_host_turn_result & result) {
    common_agent_jsonl_turn_result wire_result;
    wire_result.request_id = std::to_string(request_id);
    wire_result.ok = result.ok;
    wire_result.cancelled = result.cancelled;
    wire_result.response = result.response;
    wire_result.plan_id = result.plan_id;
    wire_result.error = result.error;
    wire_result.failure_class = common_agent_failure_class_name(result.failure_class);
    wire_result.event_count = result.event_count;
    return common_agent_jsonl_make_turn_result(wire_result).dump();
}

std::string mcp_tools_json(const std::vector<mcp_agent_tool_definition> & tools) {
    std::string result = "[";
    for (size_t i = 0; i < tools.size(); ++i) {
        if (i != 0) result += ",";
        const auto & tool = tools[i];
        result += "{\"name\":\"" + json_escape(tool.name) +
            "\",\"description\":\"" + json_escape(tool.description) +
            "\",\"input_schema\":" + tool.input_schema_json + "}";
    }
    return result + "]";
}
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_arm_aichat_agent_AgentRuntime_nativeCreate(JNIEnv * env, jclass, jstring directory, jstring model, jstring http_config) {
    const std::string storage_directory = to_string(env, directory);
    if (storage_directory.empty()) return 0;
    const std::string model_path = to_string(env, model);
    auto runtime = std::make_shared<android_agent_runtime_handle>(storage_directory, model_path);
    const std::string http_json = to_string(env, http_config);
    const auto profile = http_json.empty() ? nlohmann::json() : nlohmann::json::parse(http_json, nullptr, false);
    if (!http_json.empty() && (!profile.is_object() || !profile.contains("endpoint") ||
            !profile.contains("model_id") || !profile.contains("context_size"))) return 0;
    runtime->http_model = !http_json.empty();
    JavaVM * java_vm = nullptr;
    if (env->GetJavaVM(&java_vm) == JNI_OK) {
        jclass transport_class = env->FindClass("com/arm/aichat/agent/AndroidMcpTransport");
        if (transport_class) {
            const jmethodID post_method = env->GetStaticMethodID(
                transport_class, "post",
                "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;III)Ljava/lang/String;");
            if (post_method) {
                runtime->android_https_transport = make_agent_mcp_android_https_transport(
                    java_vm, transport_class, post_method);
                runtime->android_https_supported = runtime->android_https_transport != nullptr;
            }
            env->DeleteLocalRef(transport_class);
        }
        if (env->ExceptionCheck()) env->ExceptionClear();
        jclass credential_store_class = env->FindClass("com/arm/aichat/agent/AndroidCredentialStore");
        if (credential_store_class) {
            const jmethodID resolve_method = env->GetStaticMethodID(
                credential_store_class, "resolve",
                "(Ljava/lang/String;)Ljava/lang/String;");
            if (resolve_method) {
                runtime->android_credential_provider = make_agent_android_credential_provider(
                    java_vm, credential_store_class, resolve_method);
                runtime->android_credentials_supported = runtime->android_credential_provider != nullptr;
            }
            env->DeleteLocalRef(credential_store_class);
        }
        if (env->ExceptionCheck()) env->ExceptionClear();
    }
    std::string error;
    if (!runtime->memory_store.open(storage_directory + "/memory.sqlite", error) ||
            !runtime->plan_store.open(storage_directory + "/plan.sqlite", error) ||
            !runtime->data_store.open(storage_directory + "/datasets.sqlite", error)) {
        runtime->plan_store.close();
        runtime->memory_store.close();
        return 0;
    }
    common_tool_bootstrap_result bootstrap;
    if (!runtime->catalog.bootstrap("research", bootstrap, error)) return 0;
    auto catalog = std::make_unique<android_resource_catalog>();
    if (!catalog->open(storage_directory + "/resources.sqlite", error)) return 0;
    auto blobs = std::make_shared<agent_filesystem_blob_store>(storage_directory + "/resource-blobs");
    runtime->resources = std::make_unique<agent_catalogued_resource_store>(std::move(blobs), std::move(catalog));

    common_agent_runtime_session_host_build_config host_build_config{
        runtime->memory_store,
        runtime->plan_store,
        {},
        {},
        {},
        {},
        common_memory_scope::session,
        false,
        {},
        {},
        {},
    };
    host_build_config.resident_request.model = model_path;
    host_build_config.resident_request.inference_backend = "cli";
    host_build_config.resident_request.n_threads = 4;
    host_build_config.resident_request.context_size_tokens = 3072;
    host_build_config.resident_request.n_predict = 384;
    host_build_config.context_resource_store = runtime->resources.get();
    host_build_config.context_resource_store_persistent = true;
    if (runtime->http_model) {
      try {
        if (!profile["endpoint"].is_string() || !profile["model_id"].is_string() ||
                !profile["context_size"].is_number_unsigned()) return 0;
        common_agent_model_catalog catalog;
        common_agent_model_base_spec base;
        base.backend = "http";
        base.endpoint = profile["endpoint"].get<std::string>();
        base.model_id = profile["model_id"].get<std::string>();
        base.supports_json_schema = profile.value("supports_json_schema", true);
        base.supports_native_tool_calls = profile.value("supports_native_tool_calls", true);
        base.map_tool_names = profile.value("map_tool_names", false);
        base.token_limit_parameter = profile.value("token_limit_parameter", std::string("max_tokens"));
        base.reasoning_effort = profile.value("reasoning_effort", std::string());
        base.strict_json_schema = profile.value("strict_json_schema", true);
        base.connect_timeout_ms = profile.value("connect_timeout_ms", 5000U);
        base.request_timeout_ms = profile.value("request_timeout_ms", 120000U);
        common_agent_model_profile_spec model_profile;
        model_profile.base_model_id = "android-http";
        model_profile.context_size_tokens = profile["context_size"].get<size_t>();
        const auto output_format = profile.value("tool_output_format", std::string("json"));
        if (output_format == "json") model_profile.tool_output_format = common_agent_tool_output_format::native;
        else if (output_format == "jsonl") model_profile.tool_output_format = common_agent_tool_output_format::jsonl;
        else if (output_format == "dsl") model_profile.tool_output_format = common_agent_tool_output_format::compact_dsl;
        else return 0;
        model_profile.tool_output_format_configured = true;
        catalog.bases["android-http"] = base;
        catalog.profiles[catalog.default_profile] = model_profile;
        auto loader = std::make_shared<android_http_loader>();
        loader->transport = runtime->android_https_transport;
        loader->credentials = runtime->android_credential_provider;
        loader->credential_ref = profile.value("credential_ref", std::string());
        host_build_config.model_residency = std::make_shared<common_agent_runtime_model_residency>(
            std::move(catalog), std::unordered_map<std::string, std::shared_ptr<common_agent_runtime_model_loader>>{{"http", loader}});
        host_build_config.resident_request.inference_backend = "http";
        host_build_config.resident_request.context_size_tokens = model_profile.context_size_tokens;
        runtime->max_output_tokens = profile.value("max_output_tokens", 1024);
        if (runtime->max_output_tokens < 1 || runtime->max_output_tokens > 32768) return 0;
        host_build_config.resident_request.n_predict = runtime->max_output_tokens;
      } catch (const std::exception &) {
        // Corrupt host configuration must not cross JNI as a C++ exception.
        return 0;
      }
    }
    const std::weak_ptr<android_agent_runtime_handle> weak_runtime = runtime;
    host_build_config.tooling_resolver = [weak_runtime](
            const common_agent_runtime_resident_runtime *,
            const common_agent_runtime_session_host_turn_request & request,
            common_agent_runtime_tooling & tooling,
            std::string & error) {
        const auto runtime = weak_runtime.lock();
        if (!runtime) {
            error = "Android agent runtime is no longer available";
            return false;
        }

        std::shared_ptr<agent_mcp_http_client> client;
        {
            std::lock_guard<std::mutex> lock(runtime->mcp_mutex);
            client = runtime->mcp_client;
        }
        agent_tool_context context;
        context.request_id = request.request_id.empty() ? request.turn_id : request.request_id;
        context.turn_id = request.turn_id;
        context.scope.namespace_id = request.namespace_id;
        context.scope.session_id = request.session_id;
        context.scope.project_id = request.project_id;
        context.scope.turn_id = request.turn_id;
        context.memory_scope = request.memory_scope;
        context.plan_scope = request.plan_scope;
        context.profile_id = "research";
        context.allowed_exposed_tool_names = request.allowed_exposed_tool_names;
        context.allow_network = true;
        context.max_calls = 4;
        context.execution_control = request.execution_control;

        auto mcp_view = client ? std::make_shared<android_mcp_client_view>(client, runtime->resources.get(), runtime) : nullptr;
        if (context.allowed_exposed_tool_names.empty()) {
            context.allowed_exposed_tool_names = {"resource.inspect", "resource.read", "dataset.list",
                "dataset.select", "dataset.inspect", "dataset.schema", "dataset.sample", "dataset.validate",
                "data.query", "data.filter", "data.aggregate", "data.join", "data.transform",
                "statistics.describe", "statistics.outliers", "statistics.value_counts"};
            if (client) {
                std::vector<mcp_agent_tool_definition> remote;
                if (!mcp_view->list_tools(context, remote, error)) return false;
                for (const auto & tool : remote) {
                    context.allowed_exposed_tool_names.push_back(tool.name.find('.') == std::string::npos
                        ? "android-mcp_" + tool.name : tool.name);
                }
            }
        }
        agent_resource_runtime resource_runtime;
        resource_runtime.store = runtime->resources.get();
        resource_runtime.namespace_id = request.namespace_id;
        resource_runtime.session_id = request.session_id;
        resource_runtime.project_id = request.project_id;
        resource_runtime.turn_id = request.turn_id;
        native_agent_tool_provider native(runtime->catalog,
            [resource_runtime, runtime](const agent_tool_context & tool_context, common_native_tool_bindings & bindings, std::string & binding_error) {
                bindings.resource_runtime = resource_runtime;
                bindings.data_store = &runtime->data_store;
                bindings.tool_scope = tool_context.scope;
                bindings.allow_network = tool_context.allow_network;
                bindings.memory_store = &runtime->memory_store;
                bindings.plan_store = &runtime->plan_store;
                binding_error.clear();
                return true;
            });
        composite_agent_tool_provider providers;
        providers.add_provider(native);
        std::unique_ptr<mcp_agent_tool_provider> remote_provider;
        if (client) {
            remote_provider = std::make_unique<mcp_agent_tool_provider>("android-mcp", *mcp_view, "android-mcp");
            providers.add_provider(*remote_provider);
        }
        auto view = providers.resolve_tools(context, error);
        if (!view) return false;
        auto shared_view = std::shared_ptr<agent_tool_view>(std::move(view));
        tooling.tools = shared_view->chat_tools();
        tooling.profile_tools_active = true;
        tooling.tool_view = shared_view.get();
        if (mcp_view) tooling.owned_resources.push_back(std::static_pointer_cast<void>(mcp_view));
        tooling.owned_resources.push_back(std::static_pointer_cast<void>(shared_view));
        tooling.resource_runtime = resource_runtime;
        error.clear();
        return true;
    };
    runtime->session_host = std::make_unique<common_agent_runtime_session_host>(
        make_agent_runtime_session_host_config(std::move(host_build_config)));
    runtime->storage_ready = true;

    std::lock_guard<std::mutex> lock(handles_mutex);
    const jlong handle = next_handle++;
    handles.emplace(handle, runtime);
    return handle;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_arm_aichat_agent_AgentRuntime_nativeCancel(JNIEnv * env, jclass, jlong handle, jstring reason) {
    std::lock_guard<std::mutex> lock(handles_mutex);
    auto runtime = find_handle(handle);
    return runtime && runtime->cancellation->request_cancel(to_string(env, reason)) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_arm_aichat_agent_AgentRuntime_nativeIsCancelled(JNIEnv *, jclass, jlong handle) {
    std::lock_guard<std::mutex> lock(handles_mutex);
    auto runtime = find_handle(handle);
    return runtime && runtime->cancellation->is_cancelled() ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_arm_aichat_agent_AgentRuntime_nativeResetCancellation(JNIEnv *, jclass, jlong handle) {
    std::lock_guard<std::mutex> lock(handles_mutex);
    auto runtime = find_handle(handle);
    if (!runtime) return JNI_FALSE;
    {
        std::lock_guard<std::mutex> turn_lock(runtime->turn_mutex);
        if (runtime->turn_active) return JNI_FALSE;
    }
    runtime->cancellation->reset();
    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_arm_aichat_agent_AgentRuntime_nativePrepareModel(JNIEnv *, jclass, jlong handle) {
    std::shared_ptr<android_agent_runtime_handle> runtime;
    {
        std::lock_guard<std::mutex> lock(handles_mutex);
        runtime = find_handle(handle);
    }
    if (!runtime || !runtime->session_host || (!runtime->http_model && runtime->model_path.empty())) return JNI_FALSE;

    common_agent_runtime_session_host_turn_request request;
    request.mode = common_agent_runtime_host_mode::chat;
    request.prompt = "model preparation";
    request.session_id = "android";
    request.namespace_id = "default";
    request.turn_id = "model-preparation";
    request.n_predict = 1;
    request.execution_control.cancellation = runtime->cancellation;
    common_agent_event loading;
    loading.type = common_agent_event_type::model_loading;
    loading.detail = runtime->http_model ? "preparing HTTP model" : "loading configured GGUF model";
    runtime->events.push(loading);
    std::string error;
    const bool prepared = runtime->session_host->prepare_model(request, error);
    common_agent_event result;
    result.type = prepared ? common_agent_event_type::runtime_ready : common_agent_event_type::runtime_failed;
    result.detail = prepared ? "model loaded and runtime ready" : error;
    runtime->events.push(result);
    return prepared ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_arm_aichat_agent_AgentRuntime_nativeCapabilities(JNIEnv * env, jclass, jlong handle) {
    std::shared_ptr<android_agent_runtime_handle> runtime;
    {
        std::lock_guard<std::mutex> lock(handles_mutex);
        runtime = find_handle(handle);
    }
    if (!runtime) return nullptr;

#if defined(__aarch64__)
    constexpr const char * abi = "arm64-v8a";
    constexpr bool neon = true;
#elif defined(__x86_64__)
    constexpr const char * abi = "x86_64";
    constexpr bool neon = false;
#else
    constexpr const char * abi = "unknown";
    constexpr bool neon = false;
#endif
#if LLAMA_AGENT_ANDROID_VULKAN_BUILT
    constexpr bool vulkan_built = true;
#else
    constexpr bool vulkan_built = false;
#endif
    void * vulkan_loader = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    const bool vulkan_loader_available = vulkan_loader != nullptr;
    if (vulkan_loader) dlclose(vulkan_loader);

    bool mcp_configured = false;
    {
        std::lock_guard<std::mutex> lock(runtime->mcp_mutex);
        mcp_configured = runtime->mcp_client != nullptr;
    }
    const std::string json = "{\"abi\":\"" + std::string(abi) +
        "\",\"cpu_neon\":" + (neon ? "true" : "false") +
        "\",\"cpu_backend_built\":true" +
        ",\"vulkan_backend_built\":" + (vulkan_built ? "true" : "false") +
        ",\"vulkan_loader_available\":" + (vulkan_loader_available ? "true" : "false") +
        ",\"sqlite_storage\":" + (runtime->storage_ready ? "true" : "false") +
        ",\"mcp_remote_configured\":" + (mcp_configured ? "true" : "false") +
        ",\"mcp_https_supported\":" + (runtime->android_https_supported ? "true" : "false") +
        ",\"credential_provider_supported\":" + (runtime->android_credentials_supported ? "true" : "false") +
        ",\"inference_backend\":\"" + (runtime->http_model ? "http" : "cli") + "\"}";
    return env->NewStringUTF(json.c_str());
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_arm_aichat_agent_AgentRuntime_nativeConfigureMcp(
        JNIEnv * env, jclass, jlong handle, jstring server_name, jstring url, jstring bearer_token,
        jstring credential_ref) {
    std::shared_ptr<android_agent_runtime_handle> runtime;
    {
        std::lock_guard<std::mutex> lock(handles_mutex);
        runtime = find_handle(handle);
    }
    if (!runtime) return JNI_FALSE;
    const std::string name = to_string(env, server_name);
    const std::string endpoint = to_string(env, url);
    const std::string token = to_string(env, bearer_token);
    const std::string credential = to_string(env, credential_ref);
    if (name.empty() || endpoint.empty() ||
            (endpoint.rfind("http://", 0) != 0 && endpoint.rfind("https://", 0) != 0)) {
        return JNI_FALSE;
    }
    const bool https = endpoint.rfind("https://", 0) == 0;
    if (https && !runtime->android_https_transport) return JNI_FALSE;
    agent_mcp_http_client_config config{
        name, endpoint, token, {}, 5000, 30000, 2000, 256 * 1024};
    if (https) config.transport = runtime->android_https_transport;
    config.credential_ref = credential;
    config.credential_provider = runtime->android_credential_provider;
    std::lock_guard<std::mutex> lock(runtime->mcp_mutex);
    runtime->mcp_client = std::make_shared<agent_mcp_http_client>(std::move(config));
    return JNI_TRUE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_arm_aichat_agent_AgentRuntime_nativeMcpTools(JNIEnv * env, jclass, jlong handle) {
    std::shared_ptr<android_agent_runtime_handle> runtime;
    {
        std::lock_guard<std::mutex> lock(handles_mutex);
        runtime = find_handle(handle);
    }
    if (!runtime) return nullptr;
    std::lock_guard<std::mutex> lock(runtime->mcp_mutex);
    if (!runtime->mcp_client) return nullptr;
    agent_tool_context context;
    std::vector<mcp_agent_tool_definition> tools;
    std::string error;
    if (!runtime->mcp_client->list_tools(context, tools, error)) return nullptr;
    const std::string json = mcp_tools_json(tools);
    return env->NewStringUTF(json.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_arm_aichat_agent_AgentRuntime_nativeMcpCall(
        JNIEnv * env, jclass, jlong handle, jstring tool_name, jstring arguments_json) {
    std::shared_ptr<android_agent_runtime_handle> runtime;
    {
        std::lock_guard<std::mutex> lock(handles_mutex);
        runtime = find_handle(handle);
    }
    if (!runtime) return nullptr;
    std::lock_guard<std::mutex> lock(runtime->mcp_mutex);
    if (!runtime->mcp_client) return nullptr;
    agent_tool_context context;
    std::vector<mcp_agent_tool_definition> tools;
    std::string error;
    if (!runtime->mcp_client->list_tools(context, tools, error)) return nullptr;
    const std::string requested = to_string(env, tool_name);
    const auto it = std::find_if(tools.begin(), tools.end(), [&requested](const auto & tool) {
        return tool.name == requested;
    });
    if (it == tools.end()) return nullptr;
    mcp_agent_tool_call_result result;
    if (!runtime->mcp_client->call_tool(context, *it, to_string(env, arguments_json), result, error)) {
        return nullptr;
    }
    const std::string json = "{\"ok\":" + std::string(result.ok ? "true" : "false") +
        ",\"structured_content\":" + (result.structured_content_json.empty() ? "null" : result.structured_content_json) +
        ",\"text\":\"" + json_escape(result.text_content) + "\"}";
    return env->NewStringUTF(json.c_str());
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_arm_aichat_agent_AgentRuntime_nativeSubmitTurn(
        JNIEnv * env, jclass, jlong handle, jstring prompt, jstring mode,
        jstring session_id, jstring namespace_id, jstring project_id, jstring turn_id,
        jstring resource_refs_json) {
    std::shared_ptr<android_agent_runtime_handle> runtime;
    {
        std::lock_guard<std::mutex> lock(handles_mutex);
        runtime = find_handle(handle);
    }
    if (!runtime || !runtime->session_host || (!runtime->http_model && runtime->model_path.empty())) return JNI_FALSE;

    common_agent_jsonl_turn_request wire_request;
    wire_request.prompt = to_string(env, prompt);
    wire_request.mode = to_string(env, mode);
    wire_request.session_id = to_string(env, session_id);
    wire_request.namespace_id = to_string(env, namespace_id);
    wire_request.project_id = to_string(env, project_id);
    wire_request.turn_id = to_string(env, turn_id);
    wire_request.n_predict = runtime->max_output_tokens;

    const auto resource_refs = nlohmann::json::parse(
        to_string(env, resource_refs_json), nullptr, false);
    if (!resource_refs.is_array() || resource_refs.size() > 32) return JNI_FALSE;
    std::vector<std::string> resource_types;
    for (const auto & value : resource_refs) {
        if (value.is_string()) {
            if (value.get<std::string>().empty()) return JNI_FALSE;
            wire_request.resource_refs.push_back(value.get<std::string>());
            resource_types.emplace_back();
        } else if (value.is_object() && value.contains("path") && value["path"].is_string() &&
                !value["path"].get<std::string>().empty() &&
                (!value.contains("mime_type") || value["mime_type"].is_null() || value["mime_type"].is_string())) {
            wire_request.resource_refs.push_back(value["path"].get<std::string>());
            resource_types.push_back(value.contains("mime_type") && value["mime_type"].is_string()
                ? value["mime_type"].get<std::string>() : "");
        } else return JNI_FALSE;
    }

    std::thread previous_worker;
    uint64_t request_id = 0;
    {
        std::lock_guard<std::mutex> lock(runtime->turn_mutex);
        if (runtime->turn_active) return JNI_FALSE;
        previous_worker = std::move(runtime->turn_worker);
        runtime->turn_active = true;
        request_id = runtime->next_request_id++;
    }
    if (previous_worker.joinable()) previous_worker.join();

    common_agent_runtime_session_host_turn_request request;
    request.mode = wire_request.mode == "agent" ?
        common_agent_runtime_host_mode::agent : common_agent_runtime_host_mode::chat;
    request.prompt = wire_request.prompt;
    request.session_id = wire_request.session_id;
    request.namespace_id = wire_request.namespace_id;
    request.project_id = wire_request.project_id;
    request.turn_id = wire_request.turn_id;
    request.n_predict = wire_request.n_predict;
    for (size_t resource_index = 0; resource_index < wire_request.resource_refs.size(); ++resource_index) {
        const auto & uri = wire_request.resource_refs[resource_index];
        std::error_code ec;
        const auto root = std::filesystem::weakly_canonical(
            std::filesystem::path(runtime->storage_directory) / "resources", ec);
        if (ec) { std::lock_guard<std::mutex> lock(runtime->turn_mutex); runtime->turn_active = false; return JNI_FALSE; }
        const auto file = std::filesystem::canonical(uri, ec);
        if (ec || file.parent_path() != root || !std::filesystem::is_regular_file(file, ec) || ec ||
                std::filesystem::file_size(file, ec) > 10U * 1024U * 1024U || ec) {
            std::lock_guard<std::mutex> lock(runtime->turn_mutex); runtime->turn_active = false; return JNI_FALSE;
        }
        std::ifstream stream(file, std::ios::binary);
        if (!stream) { std::lock_guard<std::mutex> lock(runtime->turn_mutex); runtime->turn_active = false; return JNI_FALSE; }
        std::string bytes((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
        if (stream.bad() || bytes.size() > 10U * 1024U * 1024U) {
            std::lock_guard<std::mutex> lock(runtime->turn_mutex); runtime->turn_active = false; return JNI_FALSE;
        }
        agent_resource_put_request put;
        put.name = file.filename().string();
        put.mime_type = resource_types[resource_index].empty() ? "application/octet-stream" : resource_types[resource_index];
        put.bytes = std::move(bytes);
        put.bytes_are_authoritative = true;
        put.scope = common_runtime_resource_scope::session;
        put.namespace_id = request.namespace_id;
        put.session_id = request.session_id;
        put.project_id = request.project_id;
        put.source_provider = "android";
        put.source_tool = "attachment";
        agent_resource_descriptor descriptor;
        std::string resource_error;
        if (!runtime->resources->put_bytes(put, descriptor, resource_error)) {
            std::lock_guard<std::mutex> lock(runtime->turn_mutex); runtime->turn_active = false; return JNI_FALSE;
        }
        request.input_resources.push_back({descriptor, "attachment_question", true});
    }
    request.execution_control.cancellation = runtime->cancellation;
    request.event_sink = [runtime](const common_agent_event & event) {
        runtime->events.push(event);
    };

    runtime->turn_worker = std::thread([runtime, request_id, request = std::move(request)]() mutable {
        common_agent_runtime_session_host_turn_result result;
        std::string error;
        agent_research_host_resources research{&runtime->data_store, runtime->resources.get()};
        if (agent_prepare_research_workspace(research, request, error)) {
            std::map<std::string, agent_resource_descriptor> retained;
            if (request.research_draft) {
                request.research_progress_sink = [&](const common_agent_research_workspace & workspace, std::string & sink_error) {
                    common_agent_runtime_session_host_turn_result checkpoint_result;
                    auto recoverable = workspace;
                    for (auto & task : recoverable.tasks)
                        if (task.status == common_agent_research_task_status::active) task.status = common_agent_research_task_status::failed;
                    for (auto & gap : recoverable.gaps)
                        if (gap.status == common_agent_research_gap_status::investigating) gap.status = common_agent_research_gap_status::open;
                    checkpoint_result.research_workspace_checkpoint = make_common_agent_research_workspace_checkpoint(
                        recoverable, static_cast<size_t>(workspace.iterations_completed) + 1, sink_error);
                    if (!sink_error.empty()) return false;
                    common_agent_research_draft_ref advanced;
                    if (!agent_persist_research_workspace(research, request, checkpoint_result, sink_error, &advanced, &retained)) return false;
                    request.research_draft = std::move(advanced);
                    return true;
                };
            }
            runtime->session_host->run_turn(request, result, error);
            std::string persistence_error;
            if (!agent_persist_research_workspace(research, request, result, persistence_error, nullptr, &retained)) {
                result.ok = false;
                if (!result.error.empty()) result.error += "; ";
                result.error += "research draft persistence failed: " + persistence_error;
            }
        }
        if (!error.empty() && result.error.empty()) result.error = error;
        {
            std::lock_guard<std::mutex> lock(runtime->turn_mutex);
            runtime->completed_turns.push_back({request_id, std::move(result)});
            runtime->turn_active = false;
        }
    });
    return JNI_TRUE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_arm_aichat_agent_AgentRuntime_nativeHistory(
        JNIEnv * env, jclass, jlong handle, jstring namespace_id, jstring session_id) {
    std::shared_ptr<android_agent_runtime_handle> runtime;
    { std::lock_guard<std::mutex> lock(handles_mutex); runtime = find_handle(handle); }
    if (!runtime || !runtime->resources) return nullptr;
    const auto ns = to_string(env, namespace_id);
    const auto session = to_string(env, session_id);
    if (ns.empty() || session.empty()) return nullptr;
    agent_session_context_store store(runtime->resources.get(), ns, session, true);
    std::vector<common_agent_conversation_segment> segments;
    std::optional<common_agent_context_checkpoint> checkpoint;
    uint64_t epoch = 0;
    std::string error;
    if (!store.load(segments, checkpoint, epoch, error)) return nullptr;
    nlohmann::json history = nlohmann::json::array();
    for (const auto & segment : segments) {
        history.push_back({{"id", segment.request_id},
            {"role", common_agent_conversation_role_name(segment.role)}, {"content", segment.content}});
    }
    return env->NewStringUTF(history.dump().c_str());
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_arm_aichat_agent_AgentRuntime_nativeReadResource(
        JNIEnv * env, jclass, jlong handle, jstring uri, jstring namespace_id,
        jstring session_id, jstring project_id, jstring turn_id) {
    std::shared_ptr<android_agent_runtime_handle> runtime;
    { std::lock_guard<std::mutex> lock(handles_mutex); runtime = find_handle(handle); }
    if (!runtime || !runtime->resources) return nullptr;
    agent_resource_read_authority authority;
    authority.namespace_id = to_string(env, namespace_id);
    authority.session_id = to_string(env, session_id);
    authority.project_id = to_string(env, project_id);
    authority.turn_id = to_string(env, turn_id);
    std::string bytes, error;
    if (!runtime->resources->read_bytes(to_string(env, uri), authority, 10U * 1024U * 1024U, bytes, error)) return nullptr;
    auto result = env->NewByteArray(static_cast<jsize>(bytes.size()));
    if (result && !bytes.empty()) env->SetByteArrayRegion(result, 0, static_cast<jsize>(bytes.size()),
        reinterpret_cast<const jbyte *>(bytes.data()));
    return result;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_arm_aichat_agent_AgentRuntime_nativePollEvent(JNIEnv * env, jclass, jlong handle) {
    std::shared_ptr<android_agent_runtime_handle> runtime;
    {
        std::lock_guard<std::mutex> lock(handles_mutex);
        runtime = find_handle(handle);
    }
    if (!runtime) return nullptr;
    common_agent_event event;
    if (!runtime->events.try_pop(event)) return nullptr;
    const std::string json = event_json(event);
    return env->NewStringUTF(json.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_arm_aichat_agent_AgentRuntime_nativePollResult(JNIEnv * env, jclass, jlong handle) {
    std::shared_ptr<android_agent_runtime_handle> runtime;
    {
        std::lock_guard<std::mutex> lock(handles_mutex);
        runtime = find_handle(handle);
    }
    if (!runtime) return nullptr;
    std::lock_guard<std::mutex> lock(runtime->turn_mutex);
    if (runtime->completed_turns.empty()) return nullptr;
    auto completed = std::move(runtime->completed_turns.front());
    runtime->completed_turns.pop_front();
    const std::string json = result_json(completed.request_id, completed.result);
    return env->NewStringUTF(json.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_arm_aichat_agent_AgentRuntime_nativeState(JNIEnv * env, jclass, jlong handle) {
    std::shared_ptr<android_agent_runtime_handle> runtime;
    {
        std::lock_guard<std::mutex> lock(handles_mutex);
        runtime = find_handle(handle);
    }
    if (!runtime) return nullptr;
    const std::string state = "{\"storage_directory\":\"" + json_escape(runtime->storage_directory) +
        "\",\"model_configured\":" + (runtime->model_path.empty() && !runtime->http_model ? "false" : "true") +
        ",\"storage_ready\":" + (runtime->storage_ready ? "true" : "false") +
        ",\"cancelled\":" + (runtime->cancellation->is_cancelled() ? "true" : "false") +
        ",\"queued_events\":" + std::to_string(runtime->events.size()) +
        ",\"turn_active\":" + ([&runtime] {
            std::lock_guard<std::mutex> lock(runtime->turn_mutex);
            return runtime->turn_active;
        }() ? "true" : "false") + "}";
    return env->NewStringUTF(state.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_arm_aichat_agent_AgentRuntime_nativeClose(JNIEnv *, jclass, jlong handle) {
    std::shared_ptr<android_agent_runtime_handle> runtime;
    {
        std::lock_guard<std::mutex> lock(handles_mutex);
        const auto it = handles.find(handle);
        if (it == handles.end()) return;
        runtime = std::move(it->second);
        handles.erase(it);
    }
    runtime->events.close();
    if (runtime->turn_worker.joinable()) runtime->turn_worker.join();
}
