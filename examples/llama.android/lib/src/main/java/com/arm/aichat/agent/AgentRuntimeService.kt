package com.arm.aichat.agent

import android.app.Service
import android.content.Intent
import android.os.Binder
import android.os.IBinder

/**
 * Optional Android lifecycle owner for the common agent runtime facade.
 * The Activity/UI is a client; planning and inference remain native/common.
 */
class AgentRuntimeService : Service() {
    companion object {
        const val EXTRA_STORAGE_DIRECTORY = "com.arm.aichat.agent.STORAGE_DIRECTORY"
        const val EXTRA_MODEL_PATH = "com.arm.aichat.agent.MODEL_PATH"
    }

    inner class LocalBinder : Binder() {
        fun runtime(): AgentRuntimeService = this@AgentRuntimeService
    }

    private val binder = LocalBinder()
    private var runtime: AgentRuntime? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val directory = intent?.getStringExtra(EXTRA_STORAGE_DIRECTORY)
        val modelPath = intent?.getStringExtra(EXTRA_MODEL_PATH)
        if (!directory.isNullOrBlank() && runtime == null) {
            runtime = AgentRuntime.create(directory, modelPath)
        }
        return START_NOT_STICKY
    }

    override fun onBind(intent: Intent?): IBinder = binder

    fun cancel(reason: String = "cancelled"): Boolean = runtime?.cancel(reason) ?: false

    fun resetCancellation(): Boolean = runtime?.resetCancellation() ?: false

    fun submitTurn(
        prompt: String,
        mode: String = "chat",
        sessionId: String = "android",
        namespaceId: String = "default",
        projectId: String? = null,
        turnId: String = "turn-${System.currentTimeMillis()}",
    ): Boolean = runtime?.submitTurn(
        prompt, mode, sessionId, namespaceId, projectId, turnId) ?: false

    fun pollEvent(): String? = runtime?.pollEvent()

    fun pollResult(): String? = runtime?.pollResult()

    fun state(): String? = runtime?.state()

    override fun onDestroy() {
        runtime?.close()
        runtime = null
        super.onDestroy()
    }
}
