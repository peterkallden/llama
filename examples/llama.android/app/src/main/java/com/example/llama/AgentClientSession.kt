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

    fun submitTurn(
        prompt: String,
        mode: String = "agent",
        sessionId: String = "android",
        namespaceId: String = "default",
        projectId: String? = null,
        turnId: String = "turn-${System.currentTimeMillis()}",
    ): Boolean = service?.submitTurn(
        prompt, mode, sessionId, namespaceId, projectId, turnId) ?: false

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
