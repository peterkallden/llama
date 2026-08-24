package com.example.llama

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.IBinder
import com.arm.aichat.agent.AgentRuntimeService
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.Closeable

/**
 * Activity-facing client for the Android agent Service.
 *
 * This class deliberately deals in serialized event/result messages. Their
 * meaning remains the common JSONL wire contract; the app does not recreate
 * native runtime or planner objects in Kotlin.
 */
class AgentClientSession(context: Context) : Closeable {
    private val appContext = context.applicationContext
    private var service: AgentRuntimeService? = null
    private var connection: ServiceConnection? = null
    private var pollingJob: Job? = null

    val isConnected: Boolean
        get() = service != null

    fun bind(onConnected: () -> Unit = {}): Boolean {
        if (connection != null) return true

        val intent = Intent(appContext, AgentRuntimeService::class.java)
        val newConnection = object : ServiceConnection {
            override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
                service = (binder as? AgentRuntimeService.LocalBinder)?.runtime()
                onConnected()
            }

            override fun onServiceDisconnected(name: ComponentName?) {
                service = null
            }
        }
        connection = newConnection
        return appContext.bindService(intent, newConnection, Context.BIND_AUTO_CREATE)
    }

    fun configureModel(modelPath: String?): Boolean {
        val directory = appContext.filesDir.resolve("agent").absolutePath
        return service?.configureModel(directory, modelPath) ?: false
    }

    fun configureAndPrepareModel(modelPath: String?): Boolean {
        val directory = appContext.filesDir.resolve("agent").absolutePath
        return service?.configureAndPrepareModel(directory, modelPath) ?: false
    }

    fun pauseModel(): Boolean = service?.pauseModel() ?: false

    fun resumeModel(): Boolean = service?.resumeModel() ?: false

    fun resumeAndPrepareModel(): Boolean = service?.resumeAndPrepareModel() ?: false

    fun hasRuntime(): Boolean = service?.hasRuntime() == true

    fun capabilities(): String? = service?.capabilities()

    fun state(): String? = service?.state()

    fun lifecycleState(): String = service?.lifecycleState() ?: "disconnected"

    fun mcpTools(): String? = service?.mcpTools()

    fun configureMcp(
        serverName: String,
        url: String,
        bearerToken: String? = null,
        credentialRef: String? = null,
    ): Boolean = service?.configureMcp(serverName, url, bearerToken, credentialRef) ?: false

    fun mcpCall(toolName: String, argumentsJson: String): String? =
        service?.mcpCall(toolName, argumentsJson)

    fun submitTurn(
        prompt: String,
        mode: String = "agent",
        sessionId: String = "android",
        namespaceId: String = "default",
        projectId: String? = null,
        turnId: String = "turn-${System.currentTimeMillis()}",
        resourceRefs: List<String> = emptyList(),
    ): Boolean = service?.submitTurn(
        prompt, mode, sessionId, namespaceId, projectId, turnId, resourceRefs) ?: false

    fun cancel(reason: String = "cancelled"): Boolean = service?.cancel(reason) ?: false

    fun startPolling(
        scope: CoroutineScope,
        onEvent: (String) -> Unit,
        onResult: (String) -> Unit,
    ) {
        if (pollingJob != null) return
        pollingJob = scope.launch(Dispatchers.IO) {
            while (isActive) {
                val current = service
                if (current != null) {
                    val event = current.pollEvent()
                    val result = current.pollResult()
                    if (event != null) withContext(Dispatchers.Main) { onEvent(event) }
                    if (result != null) withContext(Dispatchers.Main) { onResult(result) }
                }
                delay(POLL_INTERVAL_MS)
            }
        }
    }

    fun stopPolling() {
        pollingJob?.cancel()
        pollingJob = null
    }

    override fun close() {
        stopPolling()
        val currentConnection = connection
        if (currentConnection != null) {
            appContext.unbindService(currentConnection)
        }
        connection = null
        service = null
    }

    private companion object {
        const val POLL_INTERVAL_MS = 40L
    }
}
