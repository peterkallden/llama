package com.example.llama

import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.app.AlertDialog
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.addCallback
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.arm.aichat.agent.AndroidModelManager
import com.arm.aichat.agent.AndroidImportedResource
import com.arm.aichat.agent.AndroidResourceStore
import com.arm.aichat.gguf.GgufMetadata
import com.arm.aichat.gguf.GgufMetadataReader
import com.google.android.material.floatingactionbutton.FloatingActionButton
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.util.UUID

/** Minimal client for the Service-owned llama-agent runtime. */
class MainActivity : AppCompatActivity() {
    private lateinit var ggufTv: TextView
    private lateinit var messagesRv: RecyclerView
    private lateinit var eventsRv: RecyclerView
    private lateinit var userInputEt: EditText
    private lateinit var userActionFab: FloatingActionButton
    private lateinit var runtimeStatusTv: TextView

    private lateinit var agentSession: AgentClientSession
    private lateinit var modelManager: AndroidModelManager
    private lateinit var resourceStore: AndroidResourceStore
    private var isModelReady = false
    private var turnActive = false
    private var pendingModelPath: String? = null
    private val attachments = mutableListOf<AndroidImportedResource>()

    private val messages = mutableListOf<Message>()
    private val events = mutableListOf<AgentEventRow>()
    private val messageAdapter = MessageAdapter(messages)
    private val eventAdapter = AgentEventAdapter(events)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        onBackPressedDispatcher.addCallback {
            if (turnActive) {
                agentSession.cancel("activity_back")
                Toast.makeText(this, "Cancelling agent turn", Toast.LENGTH_SHORT).show()
            } else {
                Log.w(TAG, "Back press does not stop the Service-owned runtime")
            }
        }

        ggufTv = findViewById(R.id.gguf)
        messagesRv = findViewById(R.id.messages)
        messagesRv.layoutManager = LinearLayoutManager(this).apply { stackFromEnd = true }
        messagesRv.adapter = messageAdapter
        eventsRv = findViewById(R.id.events)
        eventsRv.layoutManager = LinearLayoutManager(this).apply { stackFromEnd = true }
        eventsRv.adapter = eventAdapter
        userInputEt = findViewById(R.id.user_input)
        userActionFab = findViewById(R.id.fab)
        runtimeStatusTv = findViewById(R.id.runtime_status)
        findViewById<Button>(R.id.settings_button).setOnClickListener {
            showRuntimeStatus()
        }
        findViewById<Button>(R.id.mcp_button).setOnClickListener { showMcpTools() }
        findViewById<Button>(R.id.attachment_button).setOnClickListener {
            getResources.launch(arrayOf("*/*"))
        }
        modelManager = AndroidModelManager(this)
        resourceStore = AndroidResourceStore(this)
        agentSession = AgentClientSession(this)

        agentSession.bind {
            pendingModelPath?.let { path ->
                isModelReady = agentSession.configureModel(path)
                if (isModelReady) updateModelReadyUi()
            }
            refreshRuntimeStatus()
            agentSession.startPolling(lifecycleScope, ::handleAgentEvent, ::handleAgentResult)
        }

        userActionFab.setOnClickListener {
            if (isModelReady) handleUserInput() else getContent.launch(arrayOf("*/*"))
        }
    }

    private val getContent = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri -> uri?.let { handleSelectedModel(it) } }

    private val getResources = registerForActivityResult(
        ActivityResultContracts.OpenMultipleDocuments()
    ) { uris ->
        if (uris.isNotEmpty()) importResources(uris)
    }

    private fun importResources(uris: List<Uri>) {
        lifecycleScope.launch(Dispatchers.IO) {
            val imported = uris.map { uri ->
                resourceStore.import(uri, displayName(uri))
            }
            withContext(Dispatchers.Main) {
                attachments.addAll(imported)
                findViewById<Button>(R.id.attachment_button).text =
                    "Attach (${attachments.size})"
                Toast.makeText(this@MainActivity,
                    "Attached ${imported.size} resource(s)", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun displayName(uri: Uri): String =
        uri.lastPathSegment?.substringAfterLast('/')?.ifBlank { null }
            ?: "resource-${System.currentTimeMillis()}"

    private fun handleSelectedModel(uri: Uri) {
        userActionFab.isEnabled = false
        userInputEt.hint = "Parsing GGUF..."
        ggufTv.text = "Parsing metadata from selected file\n$uri"

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val metadata = contentResolver.openInputStream(uri)?.use {
                    GgufMetadataReader.create().readStructuredMetadata(it)
                } ?: error("Unable to read selected model")

                withContext(Dispatchers.Main) {
                    ggufTv.text = metadata.toString()
                    userInputEt.hint = "Importing model..."
                }

                val modelName = metadata.filename() + FILE_EXTENSION_GGUF
                val imported = modelManager.importGguf(uri, modelName)
                pendingModelPath = imported.path
                val configured = agentSession.configureModel(imported.path)

                withContext(Dispatchers.Main) {
                    isModelReady = configured
                    if (configured) updateModelReadyUi() else {
                        userInputEt.hint = "Agent service is not ready"
                        userActionFab.isEnabled = true
                    }
                    refreshRuntimeStatus()
                }
            } catch (error: Exception) {
                Log.e(TAG, "Unable to prepare model", error)
                withContext(Dispatchers.Main) {
                    userInputEt.hint = "Select a GGUF model to retry"
                    userActionFab.isEnabled = true
                    Toast.makeText(this@MainActivity,
                        error.message ?: "Unable to prepare model", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun handleUserInput() {
        val userMsg = userInputEt.text.toString().trim()
        if (userMsg.isEmpty()) {
            Toast.makeText(this, "Input message is empty!", Toast.LENGTH_SHORT).show()
            return
        }

        userInputEt.text = null
        userInputEt.isEnabled = false
        userActionFab.isEnabled = false
        turnActive = true
        messages.add(Message(UUID.randomUUID().toString(), userMsg, true))
        messages.add(Message(UUID.randomUUID().toString(), "", false))
        messageAdapter.notifyItemRangeInserted(messages.size - 2, 2)
        messagesRv.scrollToPosition(messages.lastIndex)

        val resourceRefs = attachments.map { it.path }
        if (!agentSession.submitTurn(userMsg, resourceRefs = resourceRefs)) {
            messages.removeAt(messages.lastIndex)
            messages.removeAt(messages.lastIndex)
            messageAdapter.notifyDataSetChanged()
            turnActive = false
            userInputEt.isEnabled = true
            userActionFab.isEnabled = true
            Toast.makeText(this, "Unable to start agent turn", Toast.LENGTH_SHORT).show()
        }
    }

    private fun handleAgentEvent(message: String) {
        val event = AgentClientMessages.parseEvent(message) ?: return
        events.add(AgentEventRow(UUID.randomUUID().toString(), event))
        eventAdapter.notifyItemInserted(events.lastIndex)
        eventsRv.scrollToPosition(events.lastIndex)
        val detail = event.detail
        if (detail.isNotBlank()) Log.d(TAG, "Agent event: $detail")
    }

    private fun handleAgentResult(message: String) {
        val result = AgentClientMessages.parseResult(message) ?: return
        val response = if (result.ok) result.response else result.error

        if (messages.isNotEmpty() && !messages.last().isUser) {
            messages[messages.lastIndex] = messages.last().copy(content = response)
            messageAdapter.notifyItemChanged(messages.lastIndex)
            messagesRv.scrollToPosition(messages.lastIndex)
        }
        turnActive = false
        userInputEt.isEnabled = isModelReady
        userActionFab.isEnabled = true
        refreshRuntimeStatus()
    }

    private fun refreshRuntimeStatus() {
        val state = agentSession.state()
        runtimeStatusTv.text = if (state.isNullOrBlank()) {
            "Agent runtime: unavailable"
        } else {
            "Agent runtime: ${if (isModelReady) "ready" else "idle"}"
        }
    }

    private fun showRuntimeStatus() {
        val state = agentSession.state().orEmpty()
        val capabilities = agentSession.capabilities().orEmpty()
        val content = buildString {
            append("State\n")
            append(prettyJson(state))
            append("\n\nCapabilities\n")
            append(prettyJson(capabilities))
            append("\n\nAttachments: ")
            append(attachments.size)
        }
        AlertDialog.Builder(this)
            .setTitle("Agent runtime")
            .setMessage(content)
            .setPositiveButton("OK", null)
            .show()
    }

    private fun prettyJson(value: String): String = runCatching {
        org.json.JSONObject(value).toString(2)
    }.getOrDefault(if (value.isBlank()) "unavailable" else value)

    private fun showMcpTools() {
        lifecycleScope.launch(Dispatchers.IO) {
            val raw = agentSession.mcpTools()
            withContext(Dispatchers.Main) {
                if (raw.isNullOrBlank()) {
                    AlertDialog.Builder(this@MainActivity)
                        .setTitle("MCP tools")
                        .setMessage("No MCP endpoint is configured or reachable.")
                        .setPositiveButton("OK", null)
                        .show()
                    return@withContext
                }
                showMcpToolPicker(raw)
            }
        }
    }

    private fun showMcpToolPicker(raw: String) {
        val tools = runCatching { JSONArray(raw) }.getOrNull()
        if (tools == null || tools.length() == 0) {
            Toast.makeText(this, "MCP returned no tools", Toast.LENGTH_SHORT).show()
            return
        }
        val names = Array(tools.length()) { index ->
            tools.optJSONObject(index)?.optString("name", "tool-$index") ?: "tool-$index"
        }
        AlertDialog.Builder(this)
            .setTitle("MCP tools")
            .setItems(names) { _, index ->
                tools.optJSONObject(index)?.let { showMcpToolForm(it) }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun showMcpToolForm(tool: JSONObject) {
        val properties = tool.optJSONObject("input_schema")?.optJSONObject("properties")
        val fields = linkedMapOf<String, EditText>()
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 8, 48, 0)
        }

        if (properties == null || properties.length() == 0) {
            container.addView(TextView(this).apply { text = "This tool has no arguments." })
        } else {
            for (name in properties.keys()) {
                val property = properties.optJSONObject(name)
                val input = EditText(this).apply {
                    hint = property?.optString("description")?.ifBlank { name } ?: name
                }
                fields[name] = input
                container.addView(input)
            }
        }

        AlertDialog.Builder(this)
            .setTitle(tool.optString("name", "MCP tool"))
            .setView(container)
            .setNegativeButton("Cancel", null)
            .setPositiveButton("Call") { _, _ ->
                val arguments = JSONObject()
                fields.forEach { (name, input) ->
                    if (input.text.isNotBlank()) arguments.put(name, input.text.toString())
                }
                callMcpTool(tool.optString("name"), arguments.toString())
            }
            .show()
    }

    private fun callMcpTool(toolName: String, argumentsJson: String) {
        lifecycleScope.launch(Dispatchers.IO) {
            val result = agentSession.mcpCall(toolName, argumentsJson)
            withContext(Dispatchers.Main) {
                AlertDialog.Builder(this@MainActivity)
                    .setTitle(toolName)
                    .setMessage(prettyJson(result.orEmpty()))
                    .setPositiveButton("OK", null)
                    .show()
            }
        }
    }

    private fun updateModelReadyUi() {
        userInputEt.hint = "Type and send a message!"
        userInputEt.isEnabled = true
        userActionFab.setImageResource(R.drawable.outline_send_24)
        userActionFab.isEnabled = true
    }

    override fun onStart() {
        super.onStart()
        if (::agentSession.isInitialized && pendingModelPath != null && !agentSession.hasRuntime()) {
            isModelReady = agentSession.resumeModel()
            if (isModelReady) updateModelReadyUi()
            refreshRuntimeStatus()
        }
    }

    override fun onStop() {
        if (::agentSession.isInitialized && agentSession.hasRuntime()) {
            if (turnActive) {
                agentSession.cancel("activity_stopped")
                turnActive = false
            }
            agentSession.pauseModel()
            isModelReady = false
            userInputEt.isEnabled = false
            refreshRuntimeStatus()
        }
        super.onStop()
    }

    override fun onDestroy() {
        if (::agentSession.isInitialized && turnActive) {
            agentSession.cancel("activity_destroyed")
        }
        if (::agentSession.isInitialized) agentSession.close()
        super.onDestroy()
    }

    companion object {
        private val TAG = MainActivity::class.java.simpleName
        private const val FILE_EXTENSION_GGUF = ".gguf"
    }
}

fun GgufMetadata.filename() = when {
    basic.name != null -> basic.name?.let { name ->
        basic.sizeLabel?.let { size -> "$name-$size" } ?: name
    }
    architecture?.architecture != null -> architecture?.architecture?.let { arch ->
        basic.uuid?.let { uuid -> "$arch-$uuid" } ?: "$arch-${System.currentTimeMillis()}"
    }
    else -> "model-${System.currentTimeMillis().toHexString()}"
}
