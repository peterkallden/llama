package com.arm.aichat.agent

import java.io.Closeable
import org.json.JSONArray

/** Host-owned lifecycle facade; turn submission remains on the common runtime seam. */
class AgentRuntime private constructor(private var nativeHandle: Long) : Closeable {
    companion object {
        init { System.loadLibrary("ai-chat") }

        @JvmStatic
        fun create(storageDirectory: String, modelPath: String? = null): AgentRuntime? {
            val handle = nativeCreate(storageDirectory, modelPath)
            return if (handle == 0L) null else AgentRuntime(handle)
        }

        @JvmStatic private external fun nativeCreate(storageDirectory: String, modelPath: String?): Long
        @JvmStatic private external fun nativeCancel(handle: Long, reason: String?): Boolean
        @JvmStatic private external fun nativeIsCancelled(handle: Long): Boolean
        @JvmStatic private external fun nativeResetCancellation(handle: Long): Boolean
        @JvmStatic private external fun nativePrepareModel(handle: Long): Boolean
        @JvmStatic private external fun nativeCapabilities(handle: Long): String?
        @JvmStatic private external fun nativeConfigureMcp(handle: Long, serverName: String, url: String, bearerToken: String?, credentialRef: String?): Boolean
        @JvmStatic private external fun nativeMcpTools(handle: Long): String?
        @JvmStatic private external fun nativeMcpCall(handle: Long, toolName: String, argumentsJson: String): String?
        @JvmStatic private external fun nativeSubmitTurn(
            handle: Long,
            prompt: String,
            mode: String,
            sessionId: String,
            namespaceId: String,
            projectId: String?,
            turnId: String,
            resourceRefsJson: String,
        ): Boolean
        @JvmStatic private external fun nativePollEvent(handle: Long): String?
        @JvmStatic private external fun nativePollResult(handle: Long): String?
        @JvmStatic private external fun nativeState(handle: Long): String?
        @JvmStatic private external fun nativeClose(handle: Long)
    }

    fun cancel(reason: String = "cancelled"): Boolean =
        nativeHandle != 0L && nativeCancel(nativeHandle, reason)

    fun isCancelled(): Boolean = nativeHandle != 0L && nativeIsCancelled(nativeHandle)

    /** Prepare the host-owned cancellation state for a subsequent turn. */
    fun resetCancellation(): Boolean =
        nativeHandle != 0L && nativeResetCancellation(nativeHandle)

    /** Load the configured model without consuming a user turn. */
    fun prepareModel(): Boolean = nativeHandle != 0L && nativePrepareModel(nativeHandle)

    /** Returns host capability facts; build capability and device availability are separate. */
    fun capabilities(): String? = if (nativeHandle == 0L) null else nativeCapabilities(nativeHandle)

    /** Configures an HTTP MCP endpoint using the shared native MCP transport. */
    fun configureMcp(serverName: String, url: String, bearerToken: String? = null): Boolean =
        configureMcp(serverName, url, bearerToken, null)

    /** Configure MCP using a host-owned credential reference instead of an inline secret. */
    fun configureMcp(
        serverName: String,
        url: String,
        bearerToken: String? = null,
        credentialRef: String? = null,
    ): Boolean = nativeHandle != 0L && nativeConfigureMcp(
        nativeHandle, serverName, url, bearerToken, credentialRef)

    /** Returns the remote MCP tool catalog as bounded JSON, or null when unavailable. */
    fun mcpTools(): String? = if (nativeHandle == 0L) null else nativeMcpTools(nativeHandle)

    /** Calls one tool from the configured remote MCP endpoint. */
    fun mcpCall(toolName: String, argumentsJson: String = "{}"): String? =
        if (nativeHandle == 0L) null else nativeMcpCall(nativeHandle, toolName, argumentsJson)

    /** Submit a non-blocking chat/agent turn owned by the native runtime worker. */
    fun submitTurn(
        prompt: String,
        mode: String = "chat",
        sessionId: String = "android",
        namespaceId: String = "default",
        projectId: String? = null,
        turnId: String = "turn-${System.currentTimeMillis()}",
        resourceRefs: List<String> = emptyList(),
    ): Boolean = nativeHandle != 0L && nativeSubmitTurn(
        nativeHandle, prompt, mode, sessionId, namespaceId, projectId, turnId,
        JSONArray(resourceRefs).toString())

    /** Returns one queued structured event, or null when no event is available. */
    fun pollEvent(): String? = if (nativeHandle == 0L) null else nativePollEvent(nativeHandle)

    /** Returns one completed turn result, or null while the worker is still running. */
    fun pollResult(): String? = if (nativeHandle == 0L) null else nativePollResult(nativeHandle)

    fun state(): String? = if (nativeHandle == 0L) null else nativeState(nativeHandle)

    override fun close() {
        if (nativeHandle != 0L) {
            nativeClose(nativeHandle)
            nativeHandle = 0L
        }
    }
}
