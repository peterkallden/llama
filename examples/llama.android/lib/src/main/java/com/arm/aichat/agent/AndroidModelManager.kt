package com.arm.aichat.agent

import android.content.Context
import android.net.Uri
import java.io.File

data class AndroidModel(
    val path: String,
    val sourceUri: String,
    val sizeBytes: Long,
)

/** Keeps GGUF models outside the APK and returns the path expected by AgentRuntime. */
class AndroidModelManager(private val context: Context) {
    fun importGguf(uri: Uri, name: String = "model.gguf"): AndroidModel {
        require(name.lowercase().endsWith(".gguf")) { "Android agent models must be GGUF files" }
        val root = File(context.filesDir, "agent/models").apply { mkdirs() }
        val target = File(root, safeName(name))
        val size = context.contentResolver.openInputStream(uri)?.use { input ->
            target.outputStream().use { output -> input.copyTo(output) }
            target.length()
        } ?: error("Unable to open Android model URI: $uri")
        return AndroidModel(target.absolutePath, uri.toString(), size)
    }

    fun installed(): List<File> = File(context.filesDir, "agent/models")
        .listFiles { file -> file.isFile && file.extension.equals("gguf", ignoreCase = true) }
        ?.sortedBy { it.name }
        ?: emptyList()

    private fun safeName(value: String): String = value
        .replace(Regex("[^A-Za-z0-9._-]"), "_")
        .ifBlank { "model.gguf" }
}
