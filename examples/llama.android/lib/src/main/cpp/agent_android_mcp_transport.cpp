#include "agent_android_mcp_transport.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <string>

namespace {
using json = nlohmann::ordered_json;

class android_https_transport final : public agent_mcp_http_transport {
public:
    android_https_transport(JavaVM * java_vm, jclass transport_class, jmethodID post_method)
        : java_vm(java_vm), post_method(post_method) {
        JNIEnv * env = nullptr;
        if (this->java_vm && this->java_vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) == JNI_OK) {
            this->transport_class = static_cast<jclass>(env->NewGlobalRef(transport_class));
        }
    }

    ~android_https_transport() override {
        if (!java_vm || !transport_class) return;
        JNIEnv * env = nullptr;
        bool detach = false;
        if (java_vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK) {
            if (java_vm->AttachCurrentThread(&env, nullptr) != JNI_OK) return;
            detach = true;
        }
        env->DeleteGlobalRef(transport_class);
        if (detach) java_vm->DetachCurrentThread();
    }

    bool post(const agent_mcp_http_request & request,
            agent_mcp_http_response & response, std::string & error) override {
        if (request.url.rfind("https://", 0) != 0) {
            error = "Android MCP transport only supports https:// endpoints";
            return false;
        }
        if (!transport_class || !post_method) {
            error = "Android HTTPS MCP transport is unavailable";
            return false;
        }

        JNIEnv * env = nullptr;
        bool detach = false;
        if (java_vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK) {
            if (java_vm->AttachCurrentThread(&env, nullptr) != JNI_OK) {
                error = "could not attach MCP transport thread to the Android VM";
                return false;
            }
            detach = true;
        }

        json headers = json::object();
        for (const auto & header : request.headers) headers[header.first] = header.second;
        const std::string headers_json = headers.dump();
        const jint connect_timeout = static_cast<jint>(std::min<uint32_t>(request.connect_timeout_ms, 0x7fffffff));
        const jint request_timeout = static_cast<jint>(std::min<uint32_t>(request.request_timeout_ms, 0x7fffffff));
        const jint max_result = static_cast<jint>(std::min<size_t>(request.max_result_bytes, 0x7fffffff));
        jstring url = env->NewStringUTF(request.url.c_str());
        jstring headers_value = env->NewStringUTF(headers_json.c_str());
        jstring body = env->NewStringUTF(request.body.c_str());
        jstring result_value = static_cast<jstring>(env->CallStaticObjectMethod(
            transport_class, post_method, url, headers_value, body,
            connect_timeout, request_timeout, max_result));
        env->DeleteLocalRef(url);
        env->DeleteLocalRef(headers_value);
        env->DeleteLocalRef(body);

        if (env->ExceptionCheck()) {
            env->ExceptionClear();
            if (result_value) env->DeleteLocalRef(result_value);
            if (detach) java_vm->DetachCurrentThread();
            error = "Android HTTPS MCP request failed";
            return false;
        }
        if (!result_value) {
            if (detach) java_vm->DetachCurrentThread();
            error = "Android HTTPS MCP transport returned no response";
            return false;
        }
        const char * chars = env->GetStringUTFChars(result_value, nullptr);
        const std::string result_json = chars ? chars : "";
        if (chars) env->ReleaseStringUTFChars(result_value, chars);
        env->DeleteLocalRef(result_value);
        if (detach) java_vm->DetachCurrentThread();

        const json parsed = json::parse(result_json, nullptr, false);
        if (parsed.is_discarded() || !parsed.is_object()) {
            error = "Android HTTPS MCP transport returned invalid response metadata";
            return false;
        }
        response.status = parsed.value("status", 0);
        response.body = parsed.value("body", "");
        if (response.body.size() > request.max_result_bytes) {
            error = "MCP HTTP response exceeded max_result_bytes";
            return false;
        }
        if (parsed.contains("headers") && parsed["headers"].is_object()) {
            for (const auto & item : parsed["headers"].items()) {
                if (item.value().is_string()) response.headers[item.key()] = item.value().get<std::string>();
            }
        }
        return true;
    }

private:
    JavaVM * java_vm = nullptr;
    jclass transport_class = nullptr;
    jmethodID post_method = nullptr;
};
}

std::shared_ptr<agent_mcp_http_transport> make_agent_mcp_android_https_transport(
        JavaVM * java_vm, jclass transport_class, jmethodID post_method) {
    auto result = std::make_shared<android_https_transport>(java_vm, transport_class, post_method);
    return result;
}
