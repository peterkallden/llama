#pragma once

#include <jni.h>

extern "C" {
JNIEXPORT jlong JNICALL Java_com_arm_aichat_agent_AgentRuntime_nativeCreate(JNIEnv *, jclass, jstring, jstring);
JNIEXPORT jboolean JNICALL Java_com_arm_aichat_agent_AgentRuntime_nativeCancel(JNIEnv *, jclass, jlong, jstring);
JNIEXPORT jboolean JNICALL Java_com_arm_aichat_agent_AgentRuntime_nativeIsCancelled(JNIEnv *, jclass, jlong);
JNIEXPORT jboolean JNICALL Java_com_arm_aichat_agent_AgentRuntime_nativeResetCancellation(JNIEnv *, jclass, jlong);
JNIEXPORT jstring JNICALL Java_com_arm_aichat_agent_AgentRuntime_nativeCapabilities(JNIEnv *, jclass, jlong);
JNIEXPORT jboolean JNICALL Java_com_arm_aichat_agent_AgentRuntime_nativeConfigureMcp(JNIEnv *, jclass, jlong, jstring, jstring, jstring, jstring);
JNIEXPORT jstring JNICALL Java_com_arm_aichat_agent_AgentRuntime_nativeMcpTools(JNIEnv *, jclass, jlong);
JNIEXPORT jstring JNICALL Java_com_arm_aichat_agent_AgentRuntime_nativeMcpCall(JNIEnv *, jclass, jlong, jstring, jstring);
JNIEXPORT jboolean JNICALL Java_com_arm_aichat_agent_AgentRuntime_nativeSubmitTurn(JNIEnv *, jclass, jlong, jstring, jstring, jstring, jstring, jstring, jstring);
JNIEXPORT jstring JNICALL Java_com_arm_aichat_agent_AgentRuntime_nativePollEvent(JNIEnv *, jclass, jlong);
JNIEXPORT jstring JNICALL Java_com_arm_aichat_agent_AgentRuntime_nativePollResult(JNIEnv *, jclass, jlong);
JNIEXPORT jstring JNICALL Java_com_arm_aichat_agent_AgentRuntime_nativeState(JNIEnv *, jclass, jlong);
JNIEXPORT void JNICALL Java_com_arm_aichat_agent_AgentRuntime_nativeClose(JNIEnv *, jclass, jlong);
}
