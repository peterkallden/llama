#include "agent_android_credentials.h"

namespace {
class android_credential_provider final : public common_agent_credential_provider {
public:
    android_credential_provider(JavaVM * java_vm, jclass store_class, jmethodID resolve_method)
        : java_vm(java_vm), resolve_method(resolve_method) {
        JNIEnv * env = nullptr;
        if (this->java_vm && this->java_vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) == JNI_OK) {
            this->store_class = static_cast<jclass>(env->NewGlobalRef(store_class));
        }
    }

    ~android_credential_provider() override {
        if (!java_vm || !store_class) return;
        JNIEnv * env = nullptr;
        bool detach = false;
        if (java_vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK) {
            if (java_vm->AttachCurrentThread(&env, nullptr) != JNI_OK) return;
            detach = true;
        }
        env->DeleteGlobalRef(store_class);
        if (detach) java_vm->DetachCurrentThread();
    }

    bool resolve(const std::string & credential_ref, std::string & secret, std::string & error) override {
        secret.clear();
        if (credential_ref.empty() || !store_class || !resolve_method) {
            error = "Android credential provider is unavailable";
            return false;
        }
        JNIEnv * env = nullptr;
        bool detach = false;
        if (java_vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK) {
            if (java_vm->AttachCurrentThread(&env, nullptr) != JNI_OK) {
                error = "could not attach credential provider thread to the Android VM";
                return false;
            }
            detach = true;
        }
        jstring ref = env->NewStringUTF(credential_ref.c_str());
        jstring value = static_cast<jstring>(env->CallStaticObjectMethod(store_class, resolve_method, ref));
        env->DeleteLocalRef(ref);
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
            if (value) env->DeleteLocalRef(value);
            if (detach) java_vm->DetachCurrentThread();
            error = "Android credential provider failed to resolve credential_ref";
            return false;
        }
        if (!value) {
            if (detach) java_vm->DetachCurrentThread();
            error = "Android credential_ref was not found";
            return false;
        }
        const char * chars = env->GetStringUTFChars(value, nullptr);
        secret = chars ? chars : "";
        if (chars) env->ReleaseStringUTFChars(value, chars);
        env->DeleteLocalRef(value);
        if (detach) java_vm->DetachCurrentThread();
        if (secret.empty()) {
            error = "Android credential_ref resolved to an empty secret";
            return false;
        }
        return true;
    }

private:
    JavaVM * java_vm = nullptr;
    jclass store_class = nullptr;
    jmethodID resolve_method = nullptr;
};
}

std::shared_ptr<common_agent_credential_provider> make_agent_android_credential_provider(
        JavaVM * java_vm, jclass credential_store_class, jmethodID resolve_method) {
    return std::make_shared<android_credential_provider>(java_vm, credential_store_class, resolve_method);
}
