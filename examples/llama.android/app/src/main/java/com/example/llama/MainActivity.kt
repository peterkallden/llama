package com.example.llama

import android.net.Uri
import android.content.Intent
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
    private var currentTurnId: String? = null
    private var lastReplyAvailable = false
    private var pendingModelPath: String? = null
    private val profilePrefs by lazy { getSharedPreferences("agent-http-profile", MODE_PRIVATE) }
    private val sessionPrefs by lazy { getSharedPreferences("agent-session", MODE_PRIVATE) }
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
        findViewById<Button>(R.id.server_button).setOnClickListener {
            startActivity(Intent(this, AgentServerActivity::class.java))
        }
        findViewById<Button>(R.id.attachment_button).setOnClickListener {
            showAttachmentManager()
        }
        findViewById<Button>(R.id.edit_button).setOnClickListener { chooseEditableResource() }
        findViewById<Button>(R.id.save_reply_button).setOnClickListener { saveLastReply() }
        modelManager = AndroidModelManager(this)
        resourceStore = AndroidResourceStore(this)
        agentSession = AgentClientSession(this)
        restoreClientState(savedInstanceState)

        agentSession.bind {
            val profile = profilePrefs.getString("active", null)
            if (profile != null) prepareHttpAsync(profile)
            else pendingModelPath?.let { path -> prepareModelAsync(path) }
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
                val unsupported = imported.any { item ->
                    val type = item.mimeType.orEmpty()
                    !type.startsWith("text/") && type != "application/json"
                }
                Toast.makeText(this@MainActivity,
                    if (unsupported) "Attached files; some formats have no local text processor. Use an agent server for document conversion."
                    else "Attached ${imported.size} resource(s)", Toast.LENGTH_LONG).show()
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

    private fun chooseEditableResource() {
        val editable = attachments.filter { it.mimeType?.startsWith("text/") == true || it.mimeType == "application/json" }
        if (editable.isEmpty()) {
            Toast.makeText(this, "Attach a text document first", Toast.LENGTH_SHORT).show()
            return
        }
        AlertDialog.Builder(this).setTitle("Edit a copy")
            .setItems(editable.map { it.path.substringAfterLast('/') }.toTypedArray()) { _, index ->
                lifecycleScope.launch(Dispatchers.IO) {
                    val text = runCatching {
                        val file = java.io.File(editable[index].path)
                        require(file.length() <= 1024 * 1024) { "Document is too large for the text editor" }
                        file.readText(Charsets.UTF_8)
                    }
                    withContext(Dispatchers.Main) {
                        text.onSuccess { showTextEditor(editable[index], it) }
                            .onFailure { Toast.makeText(this@MainActivity, it.message ?: "Unable to open document", Toast.LENGTH_LONG).show() }
                    }
                }
            }.setNegativeButton("Cancel", null).show()
    }

    private fun saveLastReply() {
        val reply = if (lastReplyAvailable) messages.lastOrNull { !it.isUser && it.content.isNotBlank() }?.content else null
        if (reply == null) {
            Toast.makeText(this, "There is no reply to save", Toast.LENGTH_SHORT).show()
            return
        }
        lifecycleScope.launch(Dispatchers.IO) {
            val result = runCatching { resourceStore.saveText("agent-reply.txt", reply) }
            withContext(Dispatchers.Main) {
                result.onSuccess { attachments.add(it); updateAttachmentButton() }
                    .onFailure { Toast.makeText(this@MainActivity, it.message ?: "Unable to save reply", Toast.LENGTH_LONG).show() }
            }
        }
    }

    private fun showTextEditor(source: AndroidImportedResource, text: String) {
        val editor = EditText(this).apply {
            setText(text)
            gravity = android.view.Gravity.TOP
            inputType = android.text.InputType.TYPE_CLASS_TEXT or android.text.InputType.TYPE_TEXT_FLAG_MULTI_LINE
            minLines = 12
        }
        AlertDialog.Builder(this).setTitle("Edit text copy")
            .setView(editor).setNegativeButton("Cancel", null)
            .setPositiveButton("Save and attach") { _, _ ->
                val revision = editor.text.toString()
                lifecycleScope.launch(Dispatchers.IO) {
                    val result = runCatching { resourceStore.saveText("edited-${source.path.substringAfterLast('/')}.txt", revision) }
                    withContext(Dispatchers.Main) {
                        result.onSuccess { attachments.add(it); updateAttachmentButton() }
                            .onFailure { Toast.makeText(this@MainActivity, it.message ?: "Unable to save document", Toast.LENGTH_LONG).show() }
                    }
                }
            }.show()
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
                profilePrefs.edit().remove("active").apply()
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
                    restoreHistory()
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
        val turnId = "turn-${UUID.randomUUID()}"
        currentTurnId = turnId
        if (!agentSession.submitTurn(userMsg,
                sessionId = sessionPrefs.getString("session_id", "android") ?: "android",
                projectId = sessionPrefs.getString("project_id", "")?.ifBlank { null },
                turnId = turnId,
                resourceRefs = resourceRefs, resourceMimeTypes = attachments.map { it.mimeType })) {
            messages.removeAt(messages.lastIndex)
            messages.removeAt(messages.lastIndex)
            messageAdapter.notifyDataSetChanged()
            turnActive = false
            currentTurnId = null
            userInputEt.isEnabled = true
            userActionFab.isEnabled = true
            Toast.makeText(this, "Unable to start agent turn", Toast.LENGTH_SHORT).show()
        }
    }

    private fun handleAgentEvent(message: String) {
        val parsed = AgentClientMessages.parseEvent(message) ?: return
        val event = if (currentTurnId != null) parsed.copy(json = runCatching {
            JSONObject(parsed.json).put("turn_id", currentTurnId).toString()
        }.getOrDefault(parsed.json)) else parsed
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
                if (event.resourceUri.startsWith("agent-resource://")) {
                    val bytes = agentSession.readResource(event.resourceUri, "default",
                        sessionPrefs.getString("session_id", "android") ?: "android",
                        sessionPrefs.getString("project_id", "")?.ifBlank { null },
                        runCatching { JSONObject(event.json).optString("turn_id").ifBlank { null } }.getOrNull())
                        ?: error("The artifact is unavailable in this session")
                    contentResolver.openOutputStream(destination)?.use { it.write(bytes) }
                        ?: error("Unable to open destination document")
                } else resourceStore.copyTo(event.resourceUri, destination)
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
        lastReplyAvailable = result.ok && result.response.isNotBlank()
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
            append("\nSession: ")
            append(sessionPrefs.getString("session_id", "android"))
            append("\nProject: ")
            append(sessionPrefs.getString("project_id", "").orEmpty().ifBlank { "none" })
        }
        AlertDialog.Builder(this)
            .setTitle("Agent runtime")
            .setMessage(content)
            .setNeutralButton("Configure MCP") { _, _ -> showMcpConfiguration() }
            .setNegativeButton("HTTP model") { _, _ -> showHttpConfiguration() }
            .setPositiveButton("Session") { _, _ -> showSessionSettings() }
            .show()
    }

    private fun showSessionSettings() {
        if (turnActive) {
            Toast.makeText(this, "Wait for the current turn before switching sessions", Toast.LENGTH_SHORT).show()
            return
        }
        val session = EditText(this).apply {
            hint = "Session ID"
            setText(sessionPrefs.getString("session_id", "android"))
            setSingleLine()
        }
        val project = EditText(this).apply {
            hint = "Project ID (optional)"
            setText(sessionPrefs.getString("project_id", ""))
            setSingleLine()
        }
        val fields = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(32, 8, 32, 0)
            addView(session)
            addView(project)
        }
        AlertDialog.Builder(this).setTitle("Local session")
            .setView(fields)
            .setNeutralButton("New session") { _, _ ->
                sessionPrefs.edit().putString("session_id", "android-${UUID.randomUUID()}")
                    .putString("project_id", project.text.toString().trim()).apply()
                messages.clear()
                lastReplyAvailable = false
                messageAdapter.notifyDataSetChanged()
            }
            .setNegativeButton("Cancel", null)
            .setPositiveButton("Save") { _, _ ->
                val id = session.text.toString().trim()
                if (id.isNotEmpty()) {
                    sessionPrefs.edit().putString("session_id", id)
                        .putString("project_id", project.text.toString().trim()).apply()
                    messages.clear()
                    lastReplyAvailable = false
                    messageAdapter.notifyDataSetChanged()
                    restoreHistory()
                } else Toast.makeText(this, "Session ID is required", Toast.LENGTH_SHORT).show()
            }.show()
    }

    private fun showHttpConfiguration() {
        val old = runCatching { JSONObject(profilePrefs.getString("active", "{}") ?: "{}") }
            .getOrDefault(JSONObject())
        fun field(label: String, value: String = "", secret: Boolean = false) = EditText(this).apply {
            hint = label
            setText(value)
            setSingleLine()
            if (secret) inputType = android.text.InputType.TYPE_CLASS_TEXT or
                android.text.InputType.TYPE_TEXT_VARIATION_PASSWORD
        }
        val endpoint = field("HTTPS endpoint", old.optString("endpoint"))
        val profileName = field("Profile name", old.optString("name", "HTTP model"))
        val model = field("Model ID", old.optString("model_id"))
        val context = field("Context tokens", old.optString("context_size", "32768"))
        context.inputType = android.text.InputType.TYPE_CLASS_NUMBER
        val maxOutput = field("Maximum output tokens", old.optString("max_output_tokens", "1024"))
        maxOutput.inputType = android.text.InputType.TYPE_CLASS_NUMBER
        val token = field("API key (leave empty to retain)", secret = true)
        val tokenParameter = field("Token limit parameter", old.optString("token_limit_parameter", "max_tokens"))
        val outputFormat = field("Tool output format: json, jsonl or dsl", old.optString("tool_output_format", "json"))
        val reasoning = field("Reasoning effort (optional)", old.optString("reasoning_effort"))
        fun option(label: String, checked: Boolean) = android.widget.CheckBox(this).apply {
            text = label
            isChecked = checked
        }
        val schema = option("Supports JSON schema", old.optBoolean("supports_json_schema", true))
        val tools = option("Supports native tool calls", old.optBoolean("supports_native_tool_calls", true))
        val mapNames = option("Map tool names for provider", old.optBoolean("map_tool_names", false))
        val strict = option("Use strict JSON schema", old.optBoolean("strict_json_schema", true))
        val fields = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(32, 8, 32, 0)
            listOf(profileName, endpoint, model, context, maxOutput, token, tokenParameter, outputFormat, reasoning, schema, tools, mapNames, strict).forEach { addView(it) }
        }
        val scroll = android.widget.ScrollView(this).apply { addView(fields) }
        AlertDialog.Builder(this).setTitle("HTTP inference")
            .setView(scroll)
            .setNegativeButton("Cancel", null)
            .setNeutralButton("Profiles") { _, _ -> showHttpProfiles() }
            .setPositiveButton("Save") { _, _ ->
                val url = Uri.parse(endpoint.text.toString().trim())
                val modelId = model.text.toString().trim()
                val size = context.text.toString().toLongOrNull()
                val output = maxOutput.text.toString().toIntOrNull()
                val parameter = tokenParameter.text.toString().trim()
                val format = outputFormat.text.toString().trim()
                val name = profileName.text.toString().trim()
                if (url.scheme != "https" || url.host.isNullOrBlank() || url.userInfo != null ||
                    name.isEmpty() || name.length > 128 || modelId.isBlank() || size == null || size <= 0 || size > 1_048_576 ||
                    output == null || output !in 1..32768 || output.toLong() >= size ||
                    parameter !in setOf("max_tokens", "max_completion_tokens") || format !in setOf("json", "jsonl", "dsl")) {
                    Toast.makeText(this, "Enter a valid HTTPS endpoint, model, context and token parameter", Toast.LENGTH_LONG).show()
                    return@setPositiveButton
                }
                val sameEndpoint = old.optString("endpoint") == url.toString().trimEnd('/')
                val reference = if (sameEndpoint && old.optString("credential_ref").isNotEmpty())
                    old.optString("credential_ref") else "android-http-${UUID.randomUUID()}"
                if (token.text.isNotEmpty()) agentSession.putCredential(reference, token.text.toString())
                val hasCredential = token.text.isNotEmpty() ||
                    (sameEndpoint && old.optString("credential_ref").isNotEmpty())
                val json = JSONObject().put("name", name).put("endpoint", url.toString().trimEnd('/'))
                    .put("model_id", modelId).put("context_size", size)
                    .put("max_output_tokens", output)
                    .put("tool_output_format", format)
                    .put("credential_ref", if (hasCredential) reference else "")
                    .put("supports_json_schema", schema.isChecked).put("supports_native_tool_calls", tools.isChecked)
                    .put("map_tool_names", mapNames.isChecked)
                    .put("strict_json_schema", strict.isChecked)
                    .put("token_limit_parameter", parameter)
                    .put("reasoning_effort", reasoning.text.toString().trim()).toString()
                profilePrefs.edit().putString("profile:$name", json).putString("active", json).apply()
                prepareHttpAsync(json)
            }.show()
    }

    private fun showHttpProfiles() {
        val names = profilePrefs.all.keys.filter { it.startsWith("profile:") }
            .map { it.removePrefix("profile:") }.sorted()
        if (names.isEmpty()) {
            Toast.makeText(this, "No saved HTTP profiles", Toast.LENGTH_SHORT).show()
            return
        }
        AlertDialog.Builder(this).setTitle("HTTP profiles")
            .setItems(names.toTypedArray()) { _, index ->
                val profile = profilePrefs.getString("profile:${names[index]}", null) ?: return@setItems
                profilePrefs.edit().putString("active", profile).apply()
                prepareHttpAsync(profile)
            }.setNegativeButton("Cancel", null).show()
    }

    private fun prepareHttpAsync(profile: String) {
        isModelReady = false
        modelProgress.visibility = android.view.View.VISIBLE
        userInputEt.isEnabled = false
        userActionFab.isEnabled = false
        lifecycleScope.launch(Dispatchers.IO) {
            val ready = agentSession.configureAndPrepareHttp(profile)
            withContext(Dispatchers.Main) {
                isModelReady = ready
                if (ready) { updateModelReadyUi(); restoreHistory() } else {
                    modelProgress.visibility = android.view.View.GONE
                    runtimeStatusTv.text = "HTTP model: unavailable"
                    userActionFab.isEnabled = true
                }
                refreshRuntimeStatus()
            }
        }
    }

    private fun restoreHistory() {
        val sessionId = sessionPrefs.getString("session_id", "android") ?: "android"
        lifecycleScope.launch(Dispatchers.IO) {
            val raw = agentSession.history("default", sessionId)
            val records = runCatching { JSONArray(raw ?: "[]") }.getOrDefault(JSONArray())
            val restored = (0 until records.length()).mapNotNull { index ->
                val record = records.optJSONObject(index) ?: return@mapNotNull null
                val role = record.optString("role")
                if (role != "user" && role != "assistant") return@mapNotNull null
                Message(record.optString("id").ifBlank { UUID.randomUUID().toString() },
                    record.optString("content"), role == "user")
            }
            withContext(Dispatchers.Main) {
                if (!turnActive && sessionId == (sessionPrefs.getString("session_id", "android") ?: "android")) {
                    messages.clear()
                    messages.addAll(restored)
                    lastReplyAvailable = restored.lastOrNull()?.let { !it.isUser && it.content.isNotBlank() } == true
                    messageAdapter.notifyDataSetChanged()
                    if (messages.isNotEmpty()) messagesRv.scrollToPosition(messages.lastIndex)
                }
            }
        }
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
        if (::agentSession.isInitialized && agentSession.isConnected && !agentSession.hasRuntime()) {
            val profile = profilePrefs.getString("active", null)
            if (profile != null) prepareHttpAsync(profile)
            else pendingModelPath?.let { prepareModelAsync(it, resume = true) }
        }
    }

    private fun restoreClientState(state: Bundle?) {
        pendingModelPath = state?.getString(KEY_MODEL_PATH)
        val paths = state?.getStringArrayList(KEY_ATTACHMENT_PATHS).orEmpty()
        val uris = state?.getStringArrayList(KEY_ATTACHMENT_URIS).orEmpty()
        val mimeTypes = state?.getStringArrayList(KEY_ATTACHMENT_MIMES).orEmpty()
        paths.forEachIndexed { index, path ->
            if (java.io.File(path).isFile) attachments.add(
                AndroidImportedResource(uris.getOrNull(index).orEmpty(), path, mimeTypes.getOrNull(index)?.ifBlank { null },
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
        outState.putStringArrayList(KEY_ATTACHMENT_MIMES,
            ArrayList(attachments.map { it.mimeType.orEmpty() }))
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
        private const val KEY_ATTACHMENT_MIMES = "android_client_attachment_mimes"
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
