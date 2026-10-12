package com.example.llama

import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.app.AlertDialog
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ProgressBar
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

/** Test/example client for the Service-owned llama-agent runtime. */
class MainActivity : AppCompatActivity() {
    private lateinit var ggufTv: TextView
    private lateinit var messagesRv: RecyclerView
    private lateinit var eventsRv: RecyclerView
    private lateinit var userInputEt: EditText
    private lateinit var userActionFab: FloatingActionButton
    private lateinit var runtimeStatusTv: TextView
    private lateinit var modelProgress: ProgressBar

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
    private val eventAdapter = AgentEventAdapter(events, ::downloadArtifact)
    private var pendingArtifact: AgentClientEvent? = null

    private val createArtifactDocument = registerForActivityResult(
        ActivityResultContracts.CreateDocument("*/*")
    ) { destination ->
        val artifact = pendingArtifact
        pendingArtifact = null
        if (destination != null && artifact != null) exportArtifact(artifact, destination)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        onBackPressedDispatcher.addCallback {
            if (turnActive) {
                agentSession.cancel("activity_back")
                Toast.makeText(this@MainActivity, "Cancelling agent turn", Toast.LENGTH_SHORT).show()
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
        modelProgress = findViewById(R.id.model_progress)
        findViewById<Button>(R.id.settings_button).setOnClickListener {
            showRuntimeStatus()
        }
        findViewById<Button>(R.id.mcp_button).setOnClickListener { showMcpTools() }
        findViewById<Button>(R.id.attachment_button).setOnClickListener {
            showAttachmentManager()
        }
        modelManager = AndroidModelManager(this)
        resourceStore = AndroidResourceStore(this)
        agentSession = AgentClientSession(this)
        restoreClientState(savedInstanceState)

        agentSession.bind {
            pendingModelPath?.let { path -> prepareModelAsync(path) }
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
                updateAttachmentButton()
                Toast.makeText(this@MainActivity,
                    "Attached ${imported.size} resource(s)", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun displayName(uri: Uri): String =
        uri.lastPathSegment?.substringAfterLast('/')?.ifBlank { null }
            ?: "resource-${System.currentTimeMillis()}"

    private fun updateAttachmentButton() {
        findViewById<Button>(R.id.attachment_button).text =
            if (attachments.isEmpty()) "Attach" else "Attach (${attachments.size})"
    }

    private fun showAttachmentManager() {
        if (attachments.isEmpty()) {
            getResources.launch(arrayOf("*/*"))
            return
        }
        val selected = BooleanArray(attachments.size)
        val names = attachments.map { it.path.substringAfterLast('/') }.toTypedArray()
        AlertDialog.Builder(this)
            .setTitle("Attachments")
            .setMultiChoiceItems(names, selected) { _, index, checked -> selected[index] = checked }
            .setNegativeButton("Cancel", null)
            .setNeutralButton("Add") { _, _ -> getResources.launch(arrayOf("*/*")) }
            .setPositiveButton("Remove selected") { _, _ ->
                val remaining = attachments.filterIndexed { index, _ -> !selected[index] }
                attachments.filterIndexed { index, _ -> selected[index] }
                    .forEach { resourceStore.delete(it) }
                attachments.clear()
                attachments.addAll(remaining)
                updateAttachmentButton()
            }
            .show()
    }

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
                withContext(Dispatchers.Main) {
                    prepareModelAsync(imported.path)
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

    private fun prepareModelAsync(path: String, resume: Boolean = false) {
        isModelReady = false
        modelProgress.visibility = android.view.View.VISIBLE
        runtimeStatusTv.text = "Agent runtime: loading model..."
        userInputEt.isEnabled = false
        userInputEt.hint = "Loading model..."
        userActionFab.isEnabled = false
        lifecycleScope.launch(Dispatchers.IO) {
            val ready = if (resume) {
                agentSession.resumeAndPrepareModel()
            } else {
                agentSession.configureAndPrepareModel(path)
            }
            withContext(Dispatchers.Main) {
                isModelReady = ready
                if (ready) {
                    updateModelReadyUi()
                } else {
                    modelProgress.visibility = android.view.View.GONE
                    runtimeStatusTv.text = "Agent runtime: failed"
                    userInputEt.hint = "Unable to load model; select or retry"
                    userActionFab.isEnabled = true
                }
                refreshRuntimeStatus()
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

    private fun downloadArtifact(event: AgentClientEvent) {
        if (event.resourceUri.isBlank()) return
        pendingArtifact = event
        createArtifactDocument.launch(suggestedArtifactName(event.resourceUri))
    }

    private fun suggestedArtifactName(resourceUri: String): String =
        Uri.parse(resourceUri).lastPathSegment?.substringAfterLast('/')
            ?.ifBlank { null } ?: "agent-artifact-${System.currentTimeMillis()}"

    private fun exportArtifact(event: AgentClientEvent, destination: Uri) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                resourceStore.copyTo(event.resourceUri, destination)
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "Artifact saved", Toast.LENGTH_SHORT).show()
                }
            } catch (error: Exception) {
                Log.e(TAG, "Unable to export artifact", error)
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity,
                        error.message ?: "Unable to save artifact", Toast.LENGTH_LONG).show()
                }
            }
        }
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
        val lifecycle = agentSession.lifecycleState()
        val state = agentSession.state()
        runtimeStatusTv.text = if (lifecycle == "disconnected" || state.isNullOrBlank()) {
            "Agent runtime: unavailable"
        } else {
            "Agent runtime: $lifecycle"
        }
        modelProgress.visibility = if (lifecycle == "loading")
            android.view.View.VISIBLE else android.view.View.GONE
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
            .setNeutralButton("Configure MCP") { _, _ -> showMcpConfiguration() }
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

    private fun showMcpConfiguration() {
        val fields = listOf(
            "Server name" to "android",
            "HTTPS URL" to "",
            "Bearer token (dev only)" to "",
            "Credential reference" to "",
        ).map { (hint, value) ->
            EditText(this).apply {
                this.hint = hint
                setText(value)
                if (hint.contains("token", ignoreCase = true)) {
                    inputType = android.text.InputType.TYPE_CLASS_TEXT or
                        android.text.InputType.TYPE_TEXT_VARIATION_PASSWORD
                }
            }
        }
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 8, 48, 0)
            fields.forEach { addView(it) }
        }
        AlertDialog.Builder(this)
            .setTitle("Configure MCP")
            .setView(container)
            .setNegativeButton("Cancel", null)
            .setPositiveButton("Save") { _, _ ->
                val configured = agentSession.configureMcp(
                    fields[0].text.toString().trim(), fields[1].text.toString().trim(),
                    fields[2].text.toString().trim().ifBlank { null },
                    fields[3].text.toString().trim().ifBlank { null },
                )
                Toast.makeText(this,
                    if (configured) "MCP configured" else "Unable to configure MCP",
                    Toast.LENGTH_SHORT).show()
                refreshRuntimeStatus()
            }
            .show()
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
        modelProgress.visibility = android.view.View.GONE
        runtimeStatusTv.text = "Agent runtime: ready"
        userInputEt.hint = "Type and send a message!"
        userInputEt.isEnabled = true
        userActionFab.setImageResource(R.drawable.outline_send_24)
        userActionFab.isEnabled = true
    }

    override fun onStart() {
        super.onStart()
        if (::agentSession.isInitialized && agentSession.isConnected &&
                pendingModelPath != null && !agentSession.hasRuntime()) {
            prepareModelAsync(pendingModelPath!!, resume = true)
        }
    }

    private fun restoreClientState(state: Bundle?) {
        pendingModelPath = state?.getString(KEY_MODEL_PATH)
        val paths = state?.getStringArrayList(KEY_ATTACHMENT_PATHS).orEmpty()
        val uris = state?.getStringArrayList(KEY_ATTACHMENT_URIS).orEmpty()
        paths.forEachIndexed { index, path ->
            if (java.io.File(path).isFile) attachments.add(
                AndroidImportedResource(uris.getOrNull(index).orEmpty(), path, null,
                    java.io.File(path).length())
            )
        }
        updateAttachmentButton()
    }

    override fun onSaveInstanceState(outState: Bundle) {
        outState.putString(KEY_MODEL_PATH, pendingModelPath)
        outState.putStringArrayList(KEY_ATTACHMENT_PATHS,
            ArrayList(attachments.map { it.path }))
        outState.putStringArrayList(KEY_ATTACHMENT_URIS,
            ArrayList(attachments.map { it.uri }))
        super.onSaveInstanceState(outState)
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
        private const val KEY_MODEL_PATH = "android_client_model_path"
        private const val KEY_ATTACHMENT_PATHS = "android_client_attachment_paths"
        private const val KEY_ATTACHMENT_URIS = "android_client_attachment_uris"
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
