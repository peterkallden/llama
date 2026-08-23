package com.arm.aichat.agent

import java.io.Closeable

/** Host-owned lifecycle facade; turn submission remains on the common runtime seam. */
class AgentRuntime private constructor(private var nativeHandle: Long) : Closeable {
    companion object {
        init { System.loadLibrary("ai-chat") }

        @JvmStatic
        fun create(storageDirectory: String): AgentRuntime? {
            val handle = nativeCreate(storageDirectory)
            return if (handle == 0L) null else AgentRuntime(handle)
        }

        @JvmStatic private external fun nativeCreate(storageDirectory: String): Long
        @JvmStatic private external fun nativeCancel(handle: Long, reason: String?): Boolean
        @JvmStatic private external fun nativeIsCancelled(handle: Long): Boolean
        @JvmStatic private external fun nativePollEvent(handle: Long): String?
        @JvmStatic private external fun nativeState(handle: Long): String?
        @JvmStatic private external fun nativeClose(handle: Long)
    }

    fun cancel(reason: String = "cancelled"): Boolean =
        nativeHandle != 0L && nativeCancel(nativeHandle, reason)

    fun isCancelled(): Boolean = nativeHandle != 0L && nativeIsCancelled(nativeHandle)

    /** Returns one queued structured event, or null when no event is available. */
    fun pollEvent(): String? = if (nativeHandle == 0L) null else nativePollEvent(nativeHandle)

    fun state(): String? = if (nativeHandle == 0L) null else nativeState(nativeHandle)

    override fun close() {
        if (nativeHandle != 0L) {
            nativeClose(nativeHandle)
            nativeHandle = 0L
        }
    }
}
