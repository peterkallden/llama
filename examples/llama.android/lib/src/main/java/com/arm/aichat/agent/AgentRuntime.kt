package com.arm.aichat.agent

import java.io.Closeable

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
        @JvmStatic private external fun nativeSubmitTurn(
            handle: Long,
            prompt: String,
            mode: String,
            sessionId: String,
            namespaceId: String,
            projectId: String?,
            turnId: String,
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

    /** Submit a non-blocking chat/agent turn owned by the native runtime worker. */
    fun submitTurn(
        prompt: String,
        mode: String = "chat",
        sessionId: String = "android",
        namespaceId: String = "default",
        projectId: String? = null,
        turnId: String = "turn-${System.currentTimeMillis()}",
    ): Boolean = nativeHandle != 0L && nativeSubmitTurn(
        nativeHandle, prompt, mode, sessionId, namespaceId, projectId, turnId)

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
