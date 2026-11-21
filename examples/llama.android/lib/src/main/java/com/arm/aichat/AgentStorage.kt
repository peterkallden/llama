package com.arm.aichat

/** Minimal native storage probe for the optional Android SQLite agent profile. */
object AgentStorage {
    init {
        System.loadLibrary("ai-chat")
    }

    /** Opens, writes and reloads the native SQLite memory and plan stores. */
    @JvmStatic
    external fun selfTest(directory: String): Boolean
}
