#include "agent_android_runtime.h"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "agent/runtime/agent-event-queue.h"
#include "tools/agent/runtime/agent-runtime-control.h"

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
};

std::mutex handles_mutex;
std::unordered_map<jlong, std::unique_ptr<android_agent_runtime_handle>> handles;
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

android_agent_runtime_handle * find_handle(jlong handle) {
    const auto it = handles.find(handle);
    return it == handles.end() ? nullptr : it->second.get();
}
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_arm_aichat_agent_AgentRuntime_nativeCreate(JNIEnv * env, jclass, jstring directory, jstring model) {
    const std::string storage_directory = to_string(env, directory);
    if (storage_directory.empty()) return 0;
    std::lock_guard<std::mutex> lock(handles_mutex);
    const jlong handle = next_handle++;
    handles.emplace(handle, std::make_unique<android_agent_runtime_handle>(
        storage_directory, to_string(env, model)));
    return handle;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_arm_aichat_agent_AgentRuntime_nativeCancel(JNIEnv * env, jclass, jlong handle, jstring reason) {
    std::lock_guard<std::mutex> lock(handles_mutex);
    auto * runtime = find_handle(handle);
    return runtime && runtime->cancellation->request_cancel(to_string(env, reason)) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_arm_aichat_agent_AgentRuntime_nativeIsCancelled(JNIEnv *, jclass, jlong handle) {
    std::lock_guard<std::mutex> lock(handles_mutex);
    auto * runtime = find_handle(handle);
    return runtime && runtime->cancellation->is_cancelled() ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_arm_aichat_agent_AgentRuntime_nativeResetCancellation(JNIEnv *, jclass, jlong handle) {
    std::lock_guard<std::mutex> lock(handles_mutex);
    auto * runtime = find_handle(handle);
    if (!runtime) return JNI_FALSE;
    runtime->cancellation->reset();
    return JNI_TRUE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_arm_aichat_agent_AgentRuntime_nativePollEvent(JNIEnv * env, jclass, jlong handle) {
    std::lock_guard<std::mutex> lock(handles_mutex);
    auto * runtime = find_handle(handle);
    if (!runtime) return nullptr;
    // The queue is filled by the future common-runtime turn bridge. Do not
    // manufacture completion events while that bridge is not connected.
    return nullptr;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_arm_aichat_agent_AgentRuntime_nativeState(JNIEnv * env, jclass, jlong handle) {
    std::lock_guard<std::mutex> lock(handles_mutex);
    auto * runtime = find_handle(handle);
    if (!runtime) return nullptr;
    const std::string state = "{\"storage_directory\":\"" + json_escape(runtime->storage_directory) +
        "\",\"model_configured\":" + (runtime->model_path.empty() ? "false" : "true") +
        "\",\"cancelled\":" + (runtime->cancellation->is_cancelled() ? "true" : "false") +
        ",\"queued_events\":" + std::to_string(runtime->events.size()) + "}";
    return env->NewStringUTF(state.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_arm_aichat_agent_AgentRuntime_nativeClose(JNIEnv *, jclass, jlong handle) {
    std::lock_guard<std::mutex> lock(handles_mutex);
    const auto it = handles.find(handle);
    if (it != handles.end()) {
        it->second->events.close();
        handles.erase(it);
    }
}
