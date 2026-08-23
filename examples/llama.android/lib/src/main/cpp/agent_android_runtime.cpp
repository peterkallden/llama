#include "agent_android_runtime.h"

#include <memory>
#include <dlfcn.h>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#include "agent/runtime/agent-event-queue.h"
#include "memory/sqlite/memory-sqlite.h"
#include "plan/sqlite/plan-sqlite.h"
#include "tools/agent/runtime/agent-runtime-control.h"
#include "tools/agent/runtime/agent-runtime-session-host.h"

namespace {
struct android_agent_runtime_handle {
    android_agent_runtime_handle(std::string directory, std::string model)
        : storage_directory(std::move(directory)),
          model_path(std::move(model)),
          cancellation(std::make_shared<common_agent_runtime_cancellation_state>()) {}
    std::string storage_directory;
    std::string model_path;
    std::shared_ptr<common_agent_runtime_cancellation_state> cancellation;
    common_agent_event_queue events;
    common_memory_sqlite_store memory_store;
    common_plan_sqlite_store plan_store;
    std::unique_ptr<common_agent_runtime_session_host> session_host;
    bool storage_ready = false;
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
    std::string result = "{\"type\":\"" +
        json_escape(common_agent_event_type_name(event.type)) +
        "\",\"detail\":\"" + json_escape(event.detail) + "\"";
    if (!event.memory_id.empty()) result += ",\"memory_id\":\"" + json_escape(event.memory_id) + "\"";
    if (event.plan_id.has_value()) result += ",\"plan_id\":\"" + json_escape(*event.plan_id) + "\"";
    if (!event.step_id.empty()) result += ",\"step_id\":\"" + json_escape(event.step_id) + "\"";
    if (!event.observation_id.empty()) result += ",\"observation_id\":\"" + json_escape(event.observation_id) + "\"";
    if (!event.tool_name.empty()) result += ",\"tool_name\":\"" + json_escape(event.tool_name) + "\"";
    if (!event.resource_uri.empty()) result += ",\"resource_uri\":\"" + json_escape(event.resource_uri) + "\"";
    return result + "}";
}

std::string result_json(uint64_t request_id, const common_agent_runtime_session_host_turn_result & result) {
    return "{\"request_id\":" + std::to_string(request_id) +
        ",\"ok\":" + (result.ok ? "true" : "false") +
        ",\"cancelled\":" + (result.cancelled ? "true" : "false") +
        ",\"response\":\"" + json_escape(result.response) + "\"" +
        ",\"plan_id\":\"" + json_escape(result.plan_id) + "\"" +
        ",\"error\":\"" + json_escape(result.error) + "\"" +
        ",\"failure_class\":\"" + common_agent_failure_class_name(result.failure_class) + "\"" +
        ",\"event_count\":" + std::to_string(result.event_count) + "}";
}
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_arm_aichat_agent_AgentRuntime_nativeCreate(JNIEnv * env, jclass, jstring directory, jstring model) {
    const std::string storage_directory = to_string(env, directory);
    if (storage_directory.empty()) return 0;
    const std::string model_path = to_string(env, model);
    auto runtime = std::make_shared<android_agent_runtime_handle>(storage_directory, model_path);
    std::string error;
    if (!runtime->memory_store.open(storage_directory + "/memory.sqlite", error) ||
            !runtime->plan_store.open(storage_directory + "/plan.sqlite", error)) {
        runtime->plan_store.close();
        runtime->memory_store.close();
        return 0;
    }

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

    const std::string json = "{\"abi\":\"" + std::string(abi) +
        "\",\"cpu_neon\":" + (neon ? "true" : "false") +
        "\",\"cpu_backend_built\":true" +
        ",\"vulkan_backend_built\":" + (vulkan_built ? "true" : "false") +
        ",\"vulkan_loader_available\":" + (vulkan_loader_available ? "true" : "false") +
        ",\"sqlite_storage\":" + (runtime->storage_ready ? "true" : "false") +
        ",\"inference_backend\":\"cli\"}";
    return env->NewStringUTF(json.c_str());
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_arm_aichat_agent_AgentRuntime_nativeSubmitTurn(
        JNIEnv * env, jclass, jlong handle, jstring prompt, jstring mode,
        jstring session_id, jstring namespace_id, jstring project_id, jstring turn_id) {
    std::shared_ptr<android_agent_runtime_handle> runtime;
    {
        std::lock_guard<std::mutex> lock(handles_mutex);
        runtime = find_handle(handle);
    }
    if (!runtime || !runtime->session_host || runtime->model_path.empty()) return JNI_FALSE;

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
    request.mode = to_string(env, mode) == "agent" ?
        common_agent_runtime_host_mode::agent : common_agent_runtime_host_mode::chat;
    request.prompt = to_string(env, prompt);
    request.session_id = to_string(env, session_id);
    request.namespace_id = to_string(env, namespace_id);
    request.project_id = to_string(env, project_id);
    request.turn_id = to_string(env, turn_id);
    request.n_predict = 384;
    request.execution_control.cancellation = runtime->cancellation;
    request.event_sink = [runtime](const common_agent_event & event) {
        runtime->events.push(event);
    };

    runtime->turn_worker = std::thread([runtime, request_id, request = std::move(request)]() mutable {
        common_agent_runtime_session_host_turn_result result;
        std::string error;
        runtime->session_host->run_turn(request, result, error);
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
        "\",\"model_configured\":" + (runtime->model_path.empty() ? "false" : "true") +
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
