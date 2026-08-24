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
    private var storageDirectory: String? = null
    private var modelPath: String? = null

    override fun onCreate() {
        super.onCreate()
        AndroidCredentialStore.initialize(this)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val directory = intent?.getStringExtra(EXTRA_STORAGE_DIRECTORY)
        val modelPath = intent?.getStringExtra(EXTRA_MODEL_PATH)
        if (!directory.isNullOrBlank()) {
            configureModel(directory, modelPath)
        }
        return START_NOT_STICKY
    }

    override fun onBind(intent: Intent?): IBinder = binder

    /**
     * Configure the common runtime for the selected app-private model.
     *
     * The Service owns replacement of the native runtime. The Activity/UI
     * never creates a second native handle and never needs to know whether a
     * model change requires rebuilding the session host.
     */
    fun configureModel(directory: String, selectedModelPath: String?): Boolean {
        if (directory.isBlank()) return false
        if (runtime != null && storageDirectory == directory && modelPath == selectedModelPath) {
            return true
        }

        runtime?.cancel("model_reconfigured")
        runtime?.close()
        runtime = AgentRuntime.create(directory, selectedModelPath)
        storageDirectory = directory
        modelPath = selectedModelPath
        return runtime != null
    }

    fun hasRuntime(): Boolean = runtime != null

    /** Release model memory while retaining storage/session identity. */
    fun pauseModel(): Boolean {
        val current = runtime ?: return modelPath != null
        current.cancel("model_paused")
        current.close()
        runtime = null
        return true
    }

    /** Recreate the native runtime from the retained app-private configuration. */
    fun resumeModel(): Boolean {
        if (runtime != null) return true
        val directory = storageDirectory ?: return false
        runtime = AgentRuntime.create(directory, modelPath)
        return runtime != null
    }

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

    fun capabilities(): String? = runtime?.capabilities()

    fun configureMcp(
        serverName: String,
        url: String,
        bearerToken: String? = null,
        credentialRef: String? = null,
    ): Boolean = runtime?.configureMcp(serverName, url, bearerToken, credentialRef) ?: false

    fun putCredential(reference: String, secret: String) =
        AndroidCredentialStore.put(reference, secret)

    fun removeCredential(reference: String) = AndroidCredentialStore.remove(reference)

    fun mcpTools(): String? = runtime?.mcpTools()

    fun mcpCall(toolName: String, argumentsJson: String = "{}"): String? =
        runtime?.mcpCall(toolName, argumentsJson)

    override fun onDestroy() {
        runtime?.close()
        runtime = null
        storageDirectory = null
        modelPath = null
        super.onDestroy()
    }
}
