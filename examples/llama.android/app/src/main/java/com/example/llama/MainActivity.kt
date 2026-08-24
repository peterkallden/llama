package com.example.llama

import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.EditText
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
import com.arm.aichat.gguf.GgufMetadata
import com.arm.aichat.gguf.GgufMetadataReader
import com.google.android.material.floatingactionbutton.FloatingActionButton
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.util.UUID

/** Minimal client for the Service-owned llama-agent runtime. */
class MainActivity : AppCompatActivity() {
    private lateinit var ggufTv: TextView
    private lateinit var messagesRv: RecyclerView
    private lateinit var userInputEt: EditText
    private lateinit var userActionFab: FloatingActionButton

    private lateinit var agentSession: AgentClientSession
    private lateinit var modelManager: AndroidModelManager
    private var isModelReady = false
    private var pendingModelPath: String? = null

    private val messages = mutableListOf<Message>()
    private val messageAdapter = MessageAdapter(messages)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        onBackPressedDispatcher.addCallback {
            Log.w(TAG, "Back press does not stop the Service-owned runtime")
        }

        ggufTv = findViewById(R.id.gguf)
        messagesRv = findViewById(R.id.messages)
        messagesRv.layoutManager = LinearLayoutManager(this).apply { stackFromEnd = true }
        messagesRv.adapter = messageAdapter
        userInputEt = findViewById(R.id.user_input)
        userActionFab = findViewById(R.id.fab)
        modelManager = AndroidModelManager(this)
        agentSession = AgentClientSession(this)

        agentSession.bind {
            pendingModelPath?.let { path ->
                isModelReady = agentSession.configureModel(path)
                if (isModelReady) updateModelReadyUi()
            }
            agentSession.startPolling(lifecycleScope, ::handleAgentEvent, ::handleAgentResult)
        }

        userActionFab.setOnClickListener {
            if (isModelReady) handleUserInput() else getContent.launch(arrayOf("*/*"))
        }
    }

    private val getContent = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri -> uri?.let { handleSelectedModel(it) } }

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
        messages.add(Message(UUID.randomUUID().toString(), userMsg, true))
        messages.add(Message(UUID.randomUUID().toString(), "", false))
        messageAdapter.notifyItemRangeInserted(messages.size - 2, 2)
        messagesRv.scrollToPosition(messages.lastIndex)

        if (!agentSession.submitTurn(userMsg)) {
            messages.removeAt(messages.lastIndex)
            messages.removeAt(messages.lastIndex)
            messageAdapter.notifyDataSetChanged()
            userInputEt.isEnabled = true
            userActionFab.isEnabled = true
            Toast.makeText(this, "Unable to start agent turn", Toast.LENGTH_SHORT).show()
        }
    }

    private fun handleAgentEvent(message: String) {
        val event = runCatching { JSONObject(message).optJSONObject("event") }.getOrNull()
        val detail = event?.optString("detail").orEmpty()
        if (detail.isNotBlank()) Log.d(TAG, "Agent event: $detail")
    }

    private fun handleAgentResult(message: String) {
        val result = runCatching { JSONObject(message) }.getOrNull() ?: return
        val response = if (!result.optBoolean("ok", false)) {
            result.optString("error").ifBlank { "Agent turn failed" }
        } else result.optString("response")

        if (messages.isNotEmpty() && !messages.last().isUser) {
            messages[messages.lastIndex] = messages.last().copy(content = response)
            messageAdapter.notifyItemChanged(messages.lastIndex)
            messagesRv.scrollToPosition(messages.lastIndex)
        }
        userInputEt.isEnabled = isModelReady
        userActionFab.isEnabled = true
    }

    private fun updateModelReadyUi() {
        userInputEt.hint = "Type and send a message!"
        userInputEt.isEnabled = true
        userActionFab.setImageResource(R.drawable.outline_send_24)
        userActionFab.isEnabled = true
    }

    override fun onDestroy() {
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
