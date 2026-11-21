#pragma once

#include <jni.h>

#include <memory>

#include "common/agent/credentials/agent-credential-provider.h"

std::shared_ptr<common_agent_credential_provider> make_agent_android_credential_provider(
        JavaVM * java_vm,
        jclass credential_store_class,
        jmethodID resolve_method);
