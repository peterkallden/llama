#include <jni.h>

#include <string>

#include "memory/memory-types.h"
#include "memory/sqlite/memory-sqlite.h"
#include "plan/plan-types.h"
#include "plan/sqlite/plan-sqlite.h"

static std::string android_string(JNIEnv * env, jstring value) {
    if (!value) return {};
    const char * chars = env->GetStringUTFChars(value, nullptr);
    std::string result = chars ? chars : "";
    env->ReleaseStringUTFChars(value, chars);
    return result;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_arm_aichat_AgentStorage_selfTest(JNIEnv * env, jclass, jstring j_directory) {
    const std::string directory = android_string(env, j_directory);
    if (directory.empty()) return JNI_FALSE;

    std::string error;
    common_memory_sqlite_store memory;
    if (!memory.open(directory + "/memory.sqlite", error)) return JNI_FALSE;

    common_memory_record record;
    record.id = "android-storage-self-test";
    record.kind = common_memory_kind::observation;
    record.content = "SQLite storage is available on Android";
    record.summary = "Android SQLite smoke";
    record.scope = common_memory_scope::global;
    record.namespace_id = "android-self-test";
    if (!memory.put(record, error) || !memory.get(record.id, error).has_value()) {
        return JNI_FALSE;
    }
    memory.close();

    common_plan_sqlite_store plan;
    if (!plan.open(directory + "/plan.sqlite", error)) return JNI_FALSE;

    common_plan_state state;
    state.id = "android-storage-self-test";
    state.session_id = "android-self-test";
    state.namespace_id = "android-self-test";
    state.scope = common_plan_scope::global;
    state.purpose = "Verify native SQLite plan persistence";
    state.goal = "Open, create and reload a plan on Android";
    plan.erase(state.id, error);
    if (!plan.create(state, error) || !plan.get(state.id, error).has_value()) {
        return JNI_FALSE;
    }
    plan.close();
    return JNI_TRUE;
}
