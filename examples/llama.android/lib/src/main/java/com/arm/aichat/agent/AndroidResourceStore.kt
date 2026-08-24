package com.arm.aichat.agent

import android.content.Context
import android.net.Uri
import java.io.File

data class AndroidImportedResource(
    val uri: String,
    val path: String,
    val mimeType: String?,
    val sizeBytes: Long,
)

/** Imports ContentResolver data into the existing path-based agent resource seam. */
class AndroidResourceStore(private val context: Context) {
    fun import(uri: Uri, name: String = "resource-${System.currentTimeMillis()}"): AndroidImportedResource {
        val root = File(context.filesDir, "agent/resources").apply { mkdirs() }
        val target = File(root, safeName(name))
        val bytes = context.contentResolver.openInputStream(uri)?.use { input ->
            target.outputStream().use { output -> input.copyTo(output) }
            target.length()
        } ?: error("Unable to open Android resource URI: $uri")
        return AndroidImportedResource(uri.toString(), target.absolutePath,
            context.contentResolver.getType(uri), bytes)
    }

    fun delete(resource: AndroidImportedResource): Boolean =
        File(resource.path).takeIf { it.parentFile == File(context.filesDir, "agent/resources") }?.delete() == true

    private fun safeName(value: String): String = value
        .replace(Regex("[^A-Za-z0-9._-]"), "_")
        .ifBlank { "resource" }
}
